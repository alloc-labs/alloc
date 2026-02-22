"""Alloc Framework Callbacks — self-contained GPU monitoring + step timing.

Each callback captures GPU metrics (VRAM, utilization, power) via a background
NVML thread AND step timing, then writes a full alloc_artifact.json.gz on
train_end. No ``alloc run`` wrapper needed.

Also writes the legacy sidecar (.alloc_callback.json) for backwards compat
with ``alloc run`` workflows.

Usage (HuggingFace):
    from alloc.callbacks import AllocCallback
    trainer = Trainer(..., callbacks=[AllocCallback()])

Usage (PyTorch Lightning):
    from alloc.callbacks import AllocLightningCallback
    trainer = Trainer(..., callbacks=[AllocLightningCallback()])
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────

_ROLLING_WINDOW = 200   # keep last N step times
_WRITE_EVERY = 50       # flush sidecar every N steps


def _compute_percentile(sorted_values, pct):
    # type: (List[float], float) -> float
    """Compute a percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _compute_timing_stats(step_times_ms):
    # type: (List[float]) -> Dict[str, float]
    """Compute p50, p90, mean, std from a list of step times (ms)."""
    if not step_times_ms:
        return {}
    sorted_vals = sorted(step_times_ms)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std = math.sqrt(variance)
    return {
        "p50": round(_compute_percentile(sorted_vals, 50), 2),
        "p90": round(_compute_percentile(sorted_vals, 90), 2),
        "mean": round(mean, 2),
        "std": round(std, 2),
    }


def _detect_distributed():
    # type: () -> tuple
    """Detect if running inside a torch.distributed process group."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return True, dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return False, 0, 1


def _write_callback_data(data):
    # type: (Dict[str, Any]) -> None
    """Write callback data to the alloc sidecar file."""
    path = os.path.join(os.getcwd(), ".alloc_callback.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _build_sidecar(
    framework,       # type: str
    step_count,      # type: int
    step_times_ms,   # type: List[float]
    batch_size,      # type: Optional[int]
    is_distributed=False,  # type: bool
    rank=0,          # type: int
    world_size=1,    # type: int
):
    # type: (...) -> Dict[str, Any]
    """Build the sidecar dict from collected timing data."""
    stats = _compute_timing_stats(step_times_ms)

    samples_per_sec = None  # type: Optional[float]
    p50 = stats.get("p50")
    if p50 and p50 > 0 and batch_size and batch_size > 0:
        samples_per_sec = round(batch_size / (p50 / 1000.0), 2)

    data = {
        "framework": framework,
        "step_count": step_count,
        "step_time_ms_p50": stats.get("p50"),
        "step_time_ms_p90": stats.get("p90"),
        "step_time_ms_mean": stats.get("mean"),
        "step_time_ms_std": stats.get("std"),
        "samples_per_sec": samples_per_sec,
        "batch_size": batch_size,
    }

    if is_distributed:
        data["is_distributed"] = True
        data["rank"] = rank
        data["world_size"] = world_size

    return data


# ── NVML Monitor ────────────────────────────────────────────────────────

_NVML_POLL_INTERVAL_S = 2.0


def _try_import_pynvml():
    """Try to import pynvml. Returns module or None."""
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


class _NvmlMonitor:
    """Background NVML sampling thread for callbacks.

    Gracefully no-ops if pynvml is not installed or NVML init fails.
    """

    def __init__(self):
        # type: () -> None
        self._pynvml = _try_import_pynvml()
        self._thread = None  # type: Optional[threading.Thread]
        self._stop_event = threading.Event()
        self._samples = []  # type: List[Dict[str, Any]]
        self._hw_context = {}  # type: Dict[str, Any]
        self._start_time = 0.0

    @property
    def available(self):
        # type: () -> bool
        return self._pynvml is not None

    def start(self):
        # type: () -> None
        """Init NVML, capture hardware context, spawn sampling daemon."""
        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlInit()
        except Exception:
            self._pynvml = None
            return

        self._start_time = time.time()

        # Capture hardware context from GPU 0
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)

            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(mem_info.total / (1024 * 1024), 1)

            self._hw_context["gpu_name"] = name
            self._hw_context["gpu_total_vram_mb"] = total_mb
        except Exception:
            pass

        try:
            drv = self._pynvml.nvmlSystemGetDriverVersion()
            if isinstance(drv, bytes):
                drv = drv.decode("utf-8")
            self._hw_context["driver_version"] = drv
        except Exception:
            pass

        try:
            cuda_ver_int = self._pynvml.nvmlSystemGetCudaDriverVersion()
            major = cuda_ver_int // 1000
            minor = (cuda_ver_int % 1000) // 10
            self._hw_context["cuda_version"] = "{}.{}".format(major, minor)
        except Exception:
            pass

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
            sm_major, sm_minor = self._pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            self._hw_context["sm_version"] = "{}.{}".format(sm_major, sm_minor)
        except Exception:
            pass

        try:
            self._hw_context["num_gpus_detected"] = self._pynvml.nvmlDeviceGetCount()
        except Exception:
            self._hw_context["num_gpus_detected"] = 1

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        # type: () -> None
        """Poll GPU metrics until stop event is set."""
        pynvml = self._pynvml
        if pynvml is None:
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop_event.is_set():
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    self._samples.append({
                        "t": round(time.time() - self._start_time, 2),
                        "vram_mb": round(mem.used / (1024 * 1024), 1),
                        "gpu_util_pct": round(float(util.gpu), 1),
                        "power_w": round(power, 1),
                    })
                except Exception:
                    pass
                self._stop_event.wait(_NVML_POLL_INTERVAL_S)
        except Exception:
            pass

    def stop(self):
        # type: () -> None
        """Stop sampling thread and shutdown NVML."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_results(self):
        # type: () -> tuple
        """Return (hw_context_dict, probe_dict) matching write_report format.

        If NVML was unavailable, returns ({}, {}).
        """
        if not self._hw_context and not self._samples:
            return {}, {}

        duration = round(time.time() - self._start_time, 2) if self._start_time else 0.0

        peak_vram = 0.0
        total_util = 0.0
        total_power = 0.0
        for s in self._samples:
            vram = s.get("vram_mb", 0.0)
            if vram > peak_vram:
                peak_vram = vram
            total_util += s.get("gpu_util_pct", 0.0)
            total_power += s.get("power_w", 0.0)

        n = len(self._samples)
        avg_util = round(total_util / n, 1) if n else 0.0
        avg_power = round(total_power / n, 1) if n else 0.0

        probe_dict = {
            "peak_vram_mb": round(peak_vram, 1),
            "avg_gpu_util": avg_util,
            "avg_power_watts": avg_power,
            "duration_seconds": duration,
            "samples": list(self._samples),
            "gpu_name": self._hw_context.get("gpu_name"),
            "gpu_total_vram_mb": self._hw_context.get("gpu_total_vram_mb"),
            "num_gpus_detected": self._hw_context.get("num_gpus_detected", 1),
            "probe_mode": "callback",
        }

        hw_context = dict(self._hw_context)
        return hw_context, probe_dict


# ── Artifact helper ─────────────────────────────────────────────────────

def _write_full_artifact(monitor, sidecar_data):
    # type: (_NvmlMonitor, Dict[str, Any]) -> None
    """Merge monitor GPU data + callback timing into a full artifact.

    Writes alloc_artifact.json.gz via artifact_writer.write_report().
    Silent no-op on any failure.
    """
    try:
        hw_context, probe_dict = monitor.get_results()

        # Merge step timing from sidecar into probe_dict
        if probe_dict:
            for key in ("step_time_ms_p50", "step_time_ms_p90", "step_time_ms_mean",
                        "step_time_ms_std", "samples_per_sec", "step_count", "batch_size",
                        "framework"):
                val = sidecar_data.get(key)
                if val is not None:
                    probe_dict[key] = val

            if sidecar_data.get("is_distributed"):
                probe_dict["is_distributed"] = True
                probe_dict["rank"] = sidecar_data.get("rank", 0)
                probe_dict["world_size"] = sidecar_data.get("world_size", 1)

        from alloc.artifact_writer import write_report
        write_report(
            probe_result=probe_dict if probe_dict else None,
            hardware_context=hw_context if hw_context else None,
        )
    except Exception:
        pass


# ── HuggingFace Callback ─────────────────────────────────────────────────

try:
    from transformers import TrainerCallback

    class AllocCallback(TrainerCallback):
        """HuggingFace Trainer callback — GPU metrics + step timing.

        Self-contained: spawns an NVML monitor thread for GPU metrics and
        writes a full alloc_artifact.json.gz on train_end. No ``alloc run``
        wrapper required.
        """

        def __init__(self):
            # type: () -> None
            self.step_count = 0          # type: int
            self._step_times_ms = []     # type: List[float]
            self._step_start = None      # type: Optional[float]
            self._batch_size = None      # type: Optional[int]
            self._last_write_step = 0    # type: int
            self._dist_checked = False   # type: bool
            self._is_distributed = False  # type: bool
            self._rank = 0               # type: int
            self._world_size = 1         # type: int
            self._monitor = _NvmlMonitor()
            self._monitor_started = False  # type: bool

        def on_step_begin(self, args, state, control, **kwargs):
            self._step_start = time.monotonic()
            if not self._monitor_started:
                self._monitor.start()
                self._monitor_started = True
            if not self._dist_checked:
                self._is_distributed, self._rank, self._world_size = _detect_distributed()
                self._dist_checked = True

        def on_step_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step

            if self._step_start is not None:
                elapsed_ms = (time.monotonic() - self._step_start) * 1000.0
                self._step_times_ms.append(elapsed_ms)
                if len(self._step_times_ms) > _ROLLING_WINDOW:
                    self._step_times_ms = self._step_times_ms[-_ROLLING_WINDOW:]
                self._step_start = None

            if self._batch_size is None:
                try:
                    bs = args.per_device_train_batch_size
                    ga = getattr(args, "gradient_accumulation_steps", 1) or 1
                    self._batch_size = bs * ga
                except Exception:
                    self._batch_size = None

            if self.step_count - self._last_write_step >= _WRITE_EVERY:
                self._flush()
                self._last_write_step = self.step_count

        def on_train_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step
            self._monitor.stop()
            sidecar = self._flush()
            _write_full_artifact(self._monitor, sidecar)

        def _flush(self):
            # type: () -> Dict[str, Any]
            data = _build_sidecar(
                framework="huggingface",
                step_count=self.step_count,
                step_times_ms=self._step_times_ms,
                batch_size=self._batch_size,
                is_distributed=self._is_distributed,
                rank=self._rank,
                world_size=self._world_size,
            )
            _write_callback_data(data)
            return data

except ImportError:
    class AllocCallback:  # type: ignore[no-redef]
        """Stub — install transformers to use AllocCallback."""

        def __init__(self):
            raise ImportError(
                "AllocCallback requires the `transformers` package. "
                "Install with: pip install transformers"
            )


# ── PyTorch Lightning Callback ───────────────────────────────────────────

try:
    from lightning.pytorch.callbacks import Callback as LightningBaseCallback

    class AllocLightningCallback(LightningBaseCallback):
        """PyTorch Lightning callback — GPU metrics + step timing.

        Self-contained: spawns an NVML monitor thread for GPU metrics and
        writes a full alloc_artifact.json.gz on train_end. No ``alloc run``
        wrapper required.
        """

        def __init__(self):
            # type: () -> None
            super().__init__()
            self.step_count = 0          # type: int
            self._step_times_ms = []     # type: List[float]
            self._step_start = None      # type: Optional[float]
            self._batch_size = None      # type: Optional[int]
            self._last_write_step = 0    # type: int
            self._dist_checked = False   # type: bool
            self._is_distributed = False  # type: bool
            self._rank = 0               # type: int
            self._world_size = 1         # type: int
            self._monitor = _NvmlMonitor()
            self._monitor_started = False  # type: bool

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self._step_start = time.monotonic()
            if not self._monitor_started:
                self._monitor.start()
                self._monitor_started = True
            if not self._dist_checked:
                self._is_distributed, self._rank, self._world_size = _detect_distributed()
                self._dist_checked = True

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.step_count = trainer.global_step

            if self._step_start is not None:
                elapsed_ms = (time.monotonic() - self._step_start) * 1000.0
                self._step_times_ms.append(elapsed_ms)
                if len(self._step_times_ms) > _ROLLING_WINDOW:
                    self._step_times_ms = self._step_times_ms[-_ROLLING_WINDOW:]
                self._step_start = None

            if self._batch_size is None:
                try:
                    if hasattr(batch, "__len__"):
                        self._batch_size = len(batch)
                    elif hasattr(batch, "shape"):
                        self._batch_size = batch.shape[0]
                    elif hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                        dm = trainer.datamodule
                        if hasattr(dm, "batch_size"):
                            self._batch_size = dm.batch_size
                except Exception:
                    self._batch_size = None

            if self.step_count - self._last_write_step >= _WRITE_EVERY:
                self._flush()
                self._last_write_step = self.step_count

        def on_train_end(self, trainer, pl_module):
            self.step_count = trainer.global_step
            self._monitor.stop()
            sidecar = self._flush()
            _write_full_artifact(self._monitor, sidecar)

        def _flush(self):
            # type: () -> Dict[str, Any]
            data = _build_sidecar(
                framework="lightning",
                step_count=self.step_count,
                step_times_ms=self._step_times_ms,
                batch_size=self._batch_size,
                is_distributed=self._is_distributed,
                rank=self._rank,
                world_size=self._world_size,
            )
            _write_callback_data(data)
            return data

except ImportError:
    class AllocLightningCallback:  # type: ignore[no-redef]
        """Stub — install lightning to use AllocLightningCallback."""

        def __init__(self):
            raise ImportError(
                "AllocLightningCallback requires the `lightning` package. "
                "Install with: pip install lightning"
            )
