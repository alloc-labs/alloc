"""Alloc Probe — external GPU monitor via pynvml.

Monitors a training process from outside: polls GPU memory, utilization,
and power draw in a background thread. No modifications to user code.

Graceful no-op if pynvml is not installed or no GPU is available.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class StopReason(str, Enum):
    STABLE = "stable"
    TIMEOUT = "timeout"
    PROCESS_EXIT = "process_exit"
    ERROR = "error"


@dataclass
class ProbeSample:
    """Single GPU metrics sample."""

    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_util_pct: float
    power_watts: float


@dataclass
class ProbeResult:
    """Result of a Probe monitoring session."""

    peak_vram_mb: float = 0.0
    avg_gpu_util: float = 0.0
    avg_power_watts: float = 0.0
    duration_seconds: float = 0.0
    samples: list = field(default_factory=list)
    exit_code: Optional[int] = None
    error: Optional[str] = None
    probe_mode: Optional[str] = None
    steps_profiled: Optional[int] = None
    stop_reason: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_total_vram_mb: Optional[float] = None
    calibration_duration_s: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    sm_version: Optional[str] = None
    num_gpus_detected: int = 1
    process_map: Optional[list] = None
    per_gpu_peak_vram_mb: Optional[list] = None
    detected_interconnect: Optional[str] = None  # "nvlink", "pcie", or None

    @property
    def peak_vram_gb(self) -> float:
        return round(self.peak_vram_mb / 1024, 2)

    @property
    def vram_utilization_pct(self) -> Optional[float]:
        if self.gpu_total_vram_mb and self.gpu_total_vram_mb > 0:
            return round(self.peak_vram_mb / self.gpu_total_vram_mb * 100, 1)
        return None


def _try_import_pynvml():
    """Try to import pynvml. Returns module or None."""
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


def _check_stable(samples, window=20, variance_threshold=5.0):
    """Check if GPU metrics have stabilized over the last `window` samples.

    Stability = std dev of GPU util over last `window` samples < threshold.
    """
    if len(samples) < window:
        return False

    recent = samples[-window:]
    utils = [s.gpu_util_pct for s in recent]
    mean = sum(utils) / len(utils)
    variance = sum((u - mean) ** 2 for u in utils) / len(utils)
    return variance ** 0.5 < variance_threshold


def _get_child_pids(pid):
    # type: (int) -> List[int]
    """Get child PIDs of a process. Returns empty list on failure."""
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = result.stdout.strip()
        if out:
            return [int(p) for p in out.split("\n") if p.strip()]
    except Exception:
        pass
    return []


def _discover_gpu_indices(proc_pid, pynvml, fallback_index=0):
    # type: (int, ..., int) -> List[int]
    """Discover which GPUs a process (and its children) are using.

    Iterates all GPU devices and checks running compute processes.
    Falls back to [fallback_index] if discovery fails or finds nothing.
    """
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return [fallback_index]

    # Collect target PIDs: the main process + its children
    target_pids = {proc_pid}
    for child in _get_child_pids(proc_pid):
        target_pids.add(child)
        # Also check grandchildren (common with torchrun/accelerate)
        for grandchild in _get_child_pids(child):
            target_pids.add(grandchild)

    found_indices = []
    for idx in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in procs:
                if p.pid in target_pids:
                    found_indices.append(idx)
                    break
        except Exception:
            continue

    return found_indices if found_indices else [fallback_index]


def _detect_interconnect(handles, pynvml):
    # type: (list, ...) -> Optional[str]
    """Detect GPU interconnect type using NVML topology API.

    Checks topology between GPU pairs. Returns "nvlink" if any pair
    is connected via NVLink, "pcie" if all pairs use PCIe, or None
    if detection fails or only one GPU.
    """
    if len(handles) < 2:
        return None
    try:
        # Check topology between first two GPU handles
        # NVML topology levels: SINGLE(10)=NVLink, MULTIPLE(20)=NVLink multi-hop,
        # HOSTBRIDGE(30)=PCIe bridge, NODE(40)=same NUMA, SYSTEM(50)=cross-socket
        level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[0], handles[1])
        # pynvml may return an int or an enum; normalize to int
        level_val = int(level) if not isinstance(level, int) else level
        if level_val <= 20:
            return "nvlink"
        return "pcie"
    except Exception:
        return None


def probe_command(
    command,  # type: list
    *,
    poll_interval_ms=500,  # type: int
    timeout_seconds=120,  # type: int
    gpu_index=0,  # type: int
    calibrate=True,  # type: bool
):
    # type: (...) -> ProbeResult
    """Run a command and monitor GPU usage externally.

    Args:
        command: Command to run as subprocess (e.g. ["python", "train.py"])
        poll_interval_ms: How often to poll GPU metrics (default 500ms)
        timeout_seconds: Max time to monitor (default 120s)
        gpu_index: Which GPU to monitor (default 0)
        calibrate: If True (default), auto-stop when metrics stabilize.

    Returns:
        ProbeResult with peak VRAM, avg utilization, samples timeseries.
        On failure, returns a result with error set — never raises.
    """
    pynvml = _try_import_pynvml()

    # Launch the subprocess
    try:
        proc = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except Exception as e:
        return ProbeResult(
            error=f"Failed to start process: {e}",
            stop_reason=StopReason.ERROR.value,
        )

    if pynvml is None:
        # No GPU monitoring — just wait for process
        try:
            if timeout_seconds > 0:
                proc.wait(timeout=timeout_seconds)
            else:
                proc.wait()
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)
        return ProbeResult(
            exit_code=proc.returncode,
            error="pynvml not installed — install with: pip install alloc[gpu]",
            stop_reason=StopReason.ERROR.value,
        )

    # Monitor with pynvml
    samples = []  # type: list[ProbeSample]
    stop_event = threading.Event()
    ramp_up_samples = 20  # Skip first 20 samples for stability check
    stop_reason_ref = [None]  # type: list[Optional[str]]
    gpu_info_ref = [None, None]  # type: list  # [gpu_name, gpu_total_vram_mb]
    hw_info_ref = [None, None, None]  # type: list  # [driver_version, cuda_version, sm_version]
    calibration_time_ref = [None]  # type: list[Optional[float]]
    num_gpus_ref = [1]  # type: list[int]
    process_map_ref = [None]  # type: list
    per_gpu_peaks_ref = [{}]  # type: list[dict]  # {handle_idx: peak_vram_mb}
    detected_ic_ref = [None]  # type: list[Optional[str]]

    def _monitor():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Capture GPU name and total VRAM
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                gpu_info_ref[0] = name
                mem_info_init = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info_ref[1] = mem_info_init.total / (1024 * 1024)
            except Exception:
                pass

            # Capture hardware context (driver, CUDA, SM version)
            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(drv, bytes):
                    drv = drv.decode("utf-8")
                hw_info_ref[0] = drv
            except Exception:
                pass
            try:
                cuda_ver_int = pynvml.nvmlSystemGetCudaDriverVersion()
                major = cuda_ver_int // 1000
                minor = (cuda_ver_int % 1000) // 10
                hw_info_ref[1] = f"{major}.{minor}"
            except Exception:
                pass
            try:
                sm_major, sm_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                hw_info_ref[2] = f"{sm_major}.{sm_minor}"
            except Exception:
                pass

            handles = [handle]
            discovery_done = False

            while not stop_event.is_set():
                # After 5 samples, try to discover all GPUs used by the process
                if not discovery_done and len(samples) >= 5 and proc.pid:
                    try:
                        discovered = _discover_gpu_indices(proc.pid, pynvml, fallback_index=gpu_index)
                        if len(discovered) > 1:
                            handles = []
                            pmap = []
                            for idx in discovered:
                                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                                handles.append(h)
                                pmap.append({"gpu_index": idx})
                            num_gpus_ref[0] = len(handles)
                            process_map_ref[0] = pmap
                    except Exception:
                        pass
                    # Detect interconnect type between discovered GPUs
                    detected_ic_ref[0] = _detect_interconnect(handles, pynvml)
                    discovery_done = True

                # Sample from all monitored GPUs — aggregate: peak vram = max, util/power = mean
                try:
                    vram_vals = []
                    util_vals = []
                    power_vals = []
                    total_mb = 0.0
                    for h in handles:
                        mi = pynvml.nvmlDeviceGetMemoryInfo(h)
                        ut = pynvml.nvmlDeviceGetUtilizationRates(h)
                        pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        vram_vals.append(mi.used / (1024 * 1024))
                        util_vals.append(ut.gpu)
                        power_vals.append(pw)
                        total_mb = mi.total / (1024 * 1024)

                    # Track per-GPU peak VRAM for multi-GPU runs
                    if len(handles) > 1:
                        pgp = per_gpu_peaks_ref[0]
                        for gi, vm in enumerate(vram_vals):
                            pgp[gi] = max(pgp.get(gi, 0.0), vm)

                    samples.append(ProbeSample(
                        timestamp=time.time(),
                        memory_used_mb=max(vram_vals),
                        memory_total_mb=total_mb,
                        gpu_util_pct=sum(util_vals) / len(util_vals),
                        power_watts=sum(power_vals) / len(power_vals),
                    ))
                except Exception:
                    pass

                # Calibrate mode: auto-stop when stable
                if calibrate and len(samples) > ramp_up_samples:
                    from alloc.stability import check_stability, RAMP_UP_SAMPLES
                    sr = check_stability(samples, poll_interval_ms=poll_interval_ms)
                    if sr.is_stable:
                        stop_reason_ref[0] = StopReason.STABLE.value
                        calibration_time_ref[0] = time.time()
                        stop_event.set()
                        break

                stop_event.wait(poll_interval_ms / 1000.0)
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    if calibrate:
        # Calibrate mode (new default): wait for stability, process exit, or timeout
        while not stop_event.is_set() and proc.poll() is None:
            elapsed = time.time() - start_time
            if timeout_seconds > 0 and elapsed >= timeout_seconds:
                stop_reason_ref[0] = StopReason.TIMEOUT.value
                break
            stop_event.wait(0.5)

        if proc.poll() is not None and stop_reason_ref[0] is None:
            stop_reason_ref[0] = StopReason.PROCESS_EXIT.value

        # Gracefully terminate the process if still running
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    else:
        # Full mode: wait for process to complete
        try:
            if timeout_seconds > 0:
                proc.wait(timeout=timeout_seconds)
            else:
                proc.wait()
            stop_reason_ref[0] = StopReason.PROCESS_EXIT.value
        except subprocess.TimeoutExpired:
            stop_reason_ref[0] = StopReason.TIMEOUT.value
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=5)
    duration = time.time() - start_time

    # Determine probe_mode
    if calibrate:
        mode = "calibrate"
    else:
        mode = "full"

    # Compute calibration duration if we stopped due to stability
    cal_duration = None
    if calibration_time_ref[0] is not None:
        cal_duration = round(calibration_time_ref[0] - start_time, 2)

    if not samples:
        return ProbeResult(
            duration_seconds=round(duration, 2),
            exit_code=proc.returncode,
            probe_mode=mode,
            stop_reason=stop_reason_ref[0],
            gpu_name=gpu_info_ref[0],
            gpu_total_vram_mb=gpu_info_ref[1],
            driver_version=hw_info_ref[0],
            cuda_version=hw_info_ref[1],
            sm_version=hw_info_ref[2],
            num_gpus_detected=num_gpus_ref[0],
            process_map=process_map_ref[0],
            detected_interconnect=detected_ic_ref[0],
        )

    peak_vram = max(s.memory_used_mb for s in samples)
    avg_util = sum(s.gpu_util_pct for s in samples) / len(samples)
    avg_power = sum(s.power_watts for s in samples) / len(samples)

    return ProbeResult(
        peak_vram_mb=round(peak_vram, 1),
        avg_gpu_util=round(avg_util, 1),
        avg_power_watts=round(avg_power, 1),
        duration_seconds=round(duration, 2),
        samples=[
            {
                "t": round(s.timestamp - samples[0].timestamp, 2),
                "vram_mb": round(s.memory_used_mb, 1),
                "gpu_util_pct": round(s.gpu_util_pct, 1),
                "power_w": round(s.power_watts, 1),
            }
            for s in samples
        ],
        exit_code=proc.returncode,
        probe_mode=mode,
        steps_profiled=None,
        stop_reason=stop_reason_ref[0],
        gpu_name=gpu_info_ref[0],
        gpu_total_vram_mb=gpu_info_ref[1],
        calibration_duration_s=cal_duration,
        driver_version=hw_info_ref[0],
        cuda_version=hw_info_ref[1],
        sm_version=hw_info_ref[2],
        num_gpus_detected=num_gpus_ref[0],
        process_map=process_map_ref[0],
        per_gpu_peak_vram_mb=(
            [round(per_gpu_peaks_ref[0].get(i, 0), 1) for i in range(num_gpus_ref[0])]
            if len(per_gpu_peaks_ref[0]) > 1 else None
        ),
        detected_interconnect=detected_ic_ref[0],
    )
