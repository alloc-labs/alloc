"""Alloc Probe — external GPU monitor via pynvml.

Monitors a training process from outside: polls GPU memory, utilization,
and power draw in a background thread. No modifications to user code.

Graceful no-op if pynvml is not installed or no GPU is available.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynvml")
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.cuda")
del _warnings


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



def _parse_launcher_gpu_count(command):
    # type: (list) -> Optional[int]
    """Extract expected GPU count from launcher command line.

    Recognizes: torchrun, torch.distributed.launch, accelerate launch,
    deepspeed, mpirun/mpiexec.
    """
    args = [str(a) for a in command]
    cmd_str = " ".join(args)

    # torchrun --nproc_per_node=N or --nproc-per-node=N
    for i, a in enumerate(args):
        for flag in ("--nproc_per_node", "--nproc-per-node"):
            if a.startswith(flag + "="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    pass
            elif a == flag and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    pass

    # accelerate launch --num_processes=N or --num-processes=N
    for i, a in enumerate(args):
        for flag in ("--num_processes", "--num-processes"):
            if a.startswith(flag + "="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    pass
            elif a == flag and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    pass

    # deepspeed --num_gpus=N or --num-gpus=N
    for i, a in enumerate(args):
        for flag in ("--num_gpus", "--num-gpus"):
            if a.startswith(flag + "="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    pass
            elif a == flag and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    pass

    # mpirun/mpiexec -np N or -n N
    for i, a in enumerate(args):
        if a in ("-np", "-n") and i + 1 < len(args):
            if any(x in cmd_str for x in ("mpirun", "mpiexec")):
                try:
                    return int(args[i + 1])
                except ValueError:
                    pass

    return None


def _read_child_env(pid, var_name):
    # type: (int, str) -> Optional[str]
    """Read an environment variable from a child process via /proc (Linux only)."""
    try:
        env_path = "/proc/{}/environ".format(pid)
        with open(env_path, "rb") as f:
            env_data = f.read()
        for entry in env_data.split(b"\x00"):
            if entry.startswith(var_name.encode() + b"="):
                return entry.split(b"=", 1)[1].decode("utf-8")
    except Exception:
        pass
    return None


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


def _discover_gpu_indices(proc_pid, pynvml, fallback_index=0, expected_gpus=None):
    # type: (int, ..., int, Optional[int]) -> List[int]
    """Discover which GPUs a process (and its children) are using.

    Two strategies:
    1. PID-matching: walks process tree, matches PIDs against NVML compute processes.
    2. Active-GPU counting: counts GPUs with ANY compute processes. Used when PID
       matching finds fewer GPUs than expected (common for DDP launchers).

    Falls back to [fallback_index] if discovery fails or finds nothing.
    """
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return [fallback_index]

    # Respect CUDA_VISIBLE_DEVICES — only search visible GPUs
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        visible_physical = []
        for d in cvd.split(","):
            d = d.strip()
            if d:
                try:
                    idx = int(d)
                    if 0 <= idx < device_count:
                        visible_physical.append(idx)
                except ValueError:
                    # UUID-style device identifiers — try NVML UUID matching
                    try:
                        for phys_idx in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
                            uuid = pynvml.nvmlDeviceGetUUID(handle)
                            if isinstance(uuid, bytes):
                                uuid = uuid.decode("utf-8", errors="replace")
                            if d in uuid:
                                visible_physical.append(phys_idx)
                                break
                    except Exception:
                        visible_physical = list(range(device_count))
                        break
        search_indices = visible_physical if visible_physical else list(range(device_count))
    else:
        search_indices = list(range(device_count))

    # Collect target PIDs: the main process + descendants (3 levels deep).
    # torchrun uses: torchrun → elastic_agent → worker processes,
    # so we need at least 3 levels to find DDP worker GPUs.
    target_pids = {proc_pid}
    for child in _get_child_pids(proc_pid):
        target_pids.add(child)
        for grandchild in _get_child_pids(child):
            target_pids.add(grandchild)
            # Great-grandchildren: covers torchrun elastic launch wrapper
            for ggchild in _get_child_pids(grandchild):
                target_pids.add(ggchild)

    # Also try reading WORLD_SIZE from child process environments (Linux)
    child_world_size = None
    for child in _get_child_pids(proc_pid):
        ws = _read_child_env(child, "WORLD_SIZE")
        if ws:
            try:
                child_world_size = int(ws)
            except ValueError:
                pass
            break
        for grandchild in _get_child_pids(child):
            ws = _read_child_env(grandchild, "WORLD_SIZE")
            if ws:
                try:
                    child_world_size = int(ws)
                except ValueError:
                    pass
                break
            for ggchild in _get_child_pids(grandchild):
                ws = _read_child_env(ggchild, "WORLD_SIZE")
                if ws:
                    try:
                        child_world_size = int(ws)
                    except ValueError:
                        pass
                    break

    # Determine how many GPUs we expect
    effective_expected = expected_gpus
    if child_world_size is not None:
        if effective_expected is None or child_world_size > effective_expected:
            effective_expected = child_world_size

    # Strategy 1: PID-based matching
    found_indices = []
    active_indices = []  # GPUs with ANY compute processes
    for idx in search_indices:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if procs:
                active_indices.append(idx)
            for p in procs:
                if p.pid in target_pids:
                    found_indices.append(idx)
                    break
        except Exception:
            continue

    # Strategy 2: If PID matching found fewer than expected, use active GPU count.
    # This handles the common case where DDP workers are running but their PIDs
    # weren't in our process tree (e.g., process group isolation, timing issues).
    if effective_expected is not None and len(found_indices) < effective_expected:
        if len(active_indices) >= effective_expected:
            return active_indices[:effective_expected]
        # Even if active < expected, active is better than found
        if len(active_indices) > len(found_indices):
            return active_indices

    return found_indices if found_indices else [fallback_index]


def _detect_interconnect(handles, pynvml):
    # type: (list, ...) -> Optional[str]
    """Detect GPU interconnect type using NVML topology API.

    Returns "nvlink_switch", "nvlink_p2p", "nvlink" (fallback), "pcie",
    or None if detection fails or only one GPU.

    Heuristic matches callbacks.py: ≥6 active NVLink connections AND
    ≥4 GPUs → nvlink_switch (DGX/HGX), else nvlink_p2p.
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
            # NVLink detected — try to differentiate switch vs point-to-point
            try:
                active_links = 0
                for link in range(18):  # max NVLink connections
                    try:
                        state = pynvml.nvmlDeviceGetNvLinkState(handles[0], link)
                        if state:
                            active_links += 1
                    except Exception:
                        break
                # [HEURISTIC] Threshold: 6 links. DGX A100 has 12, DGX H100 has 18.
                # Point-to-point 2-GPU: 2 links. 4-GPU ring: 2-4 links.
                num_gpus = len(handles)
                if active_links >= 6 and num_gpus >= 4:
                    return "nvlink_switch"
                return "nvlink_p2p"
            except Exception:
                # Link counting failed — fall back to generic nvlink
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

    # Launch the user's training subprocess.
    # Suppress only pynvml/torch.cuda FutureWarning noise — these come from
    # Alloc's own callbacks or from torch internals, not from user code.
    # Propagates to torchrun children and most Ray workers via env inheritance.
    child_env = os.environ.copy()
    existing_pw = child_env.get("PYTHONWARNINGS", "")
    alloc_filters = (
        "ignore::FutureWarning:pynvml,"
        "ignore::DeprecationWarning:pynvml,"
        "ignore::FutureWarning:torch.cuda,"
        "ignore::DeprecationWarning:torch.cuda"
    )
    child_env["PYTHONWARNINGS"] = (
        f"{existing_pw},{alloc_filters}" if existing_pw else alloc_filters
    )
    try:
        proc = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=child_env,
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
            error="GPU monitoring unavailable — no NVIDIA driver detected",
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
            discovery_attempts = 0
            max_discovery_attempts = 3  # Retry at samples 5, 15, 30

            # Determine expected GPU count from command line + environment
            expected_gpus = 1
            launcher_gpus = _parse_launcher_gpu_count(command)
            if launcher_gpus is not None and launcher_gpus > 1:
                expected_gpus = launcher_gpus
            ws = os.environ.get("WORLD_SIZE", "").strip()
            if ws:
                try:
                    ws_int = max(1, int(ws))
                    if ws_int > expected_gpus:
                        expected_gpus = ws_int
                except ValueError:
                    pass

            while not stop_event.is_set():
                # Retry GPU discovery: at samples 5, 15, 30
                # Keep retrying if we haven't found all expected GPUs yet
                discovery_thresholds = [5, 15, 30]
                if (not discovery_done
                    and discovery_attempts < max_discovery_attempts
                    and len(samples) >= discovery_thresholds[discovery_attempts]
                    and proc.pid):
                    discovery_attempts += 1
                    try:
                        discovered = _discover_gpu_indices(
                            proc.pid, pynvml, fallback_index=gpu_index,
                            expected_gpus=expected_gpus if expected_gpus > 1 else None,
                        )
                        if len(discovered) > 1:
                            handles = []
                            pmap = []
                            for idx in discovered:
                                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                                handles.append(h)
                                pmap.append({"gpu_index": idx})
                            num_gpus_ref[0] = len(handles)
                            process_map_ref[0] = pmap
                            # Immediately sample all discovered GPUs so per_gpu_peaks
                            # is populated even if the process exits right after discovery
                            pgp = per_gpu_peaks_ref[0]
                            for gi, h in enumerate(handles):
                                try:
                                    mi = pynvml.nvmlDeviceGetMemoryInfo(h)
                                    vm = mi.used / (1024 * 1024)
                                    pgp[gi] = max(pgp.get(gi, 0.0), vm)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Detect interconnect type between discovered GPUs
                    detected_ic_ref[0] = _detect_interconnect(handles, pynvml)
                    # Stop retrying if we found expected count or exhausted attempts
                    if num_gpus_ref[0] >= expected_gpus or discovery_attempts >= max_discovery_attempts:
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

                    # Track per-GPU peak VRAM (always, even single GPU —
                    # discovery may expand handles later, and we need history from sample 0)
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

        # Gracefully terminate the process if still running (calibrate mode only)
        if proc.poll() is None:
            print("\nalloc: Calibration complete — Alloc collected enough data and is shutting down your training process.", file=sys.stderr, flush=True)
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

    # Final fallback: if NVML discovery found fewer GPUs than expected,
    # trust the command-line / environment signals. The probe may miss GPUs
    # due to DDP per-rank CVD isolation or timing races.
    launcher_count = _parse_launcher_gpu_count(command)
    if launcher_count is not None and launcher_count > num_gpus_ref[0]:
        num_gpus_ref[0] = launcher_count
    env_world = os.environ.get("WORLD_SIZE", "").strip()
    if env_world:
        try:
            ws = int(env_world)
            if ws > num_gpus_ref[0]:
                num_gpus_ref[0] = ws
        except ValueError:
            pass

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
            if num_gpus_ref[0] >= 1 and per_gpu_peaks_ref[0] else None
        ),
        detected_interconnect=detected_ic_ref[0],
    )
