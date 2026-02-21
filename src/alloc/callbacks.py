"""Alloc Framework Callbacks — capture training step timing for artifact enrichment.

Callbacks for popular ML frameworks. Write timing stats to a sidecar file
(.alloc_callback.json) so the probe can capture throughput and step latency.

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


# ── HuggingFace Callback ─────────────────────────────────────────────────

try:
    from transformers import TrainerCallback

    class AllocCallback(TrainerCallback):
        """HuggingFace Trainer callback that captures step timing for Alloc."""

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

        def on_step_begin(self, args, state, control, **kwargs):
            self._step_start = time.monotonic()
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
            self._flush()

        def _flush(self):
            # type: () -> None
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
        """PyTorch Lightning callback that captures step timing for Alloc."""

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

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self._step_start = time.monotonic()
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
            self._flush()

        def _flush(self):
            # type: () -> None
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

except ImportError:
    class AllocLightningCallback:  # type: ignore[no-redef]
        """Stub — install lightning to use AllocLightningCallback."""

        def __init__(self):
            raise ImportError(
                "AllocLightningCallback requires the `lightning` package. "
                "Install with: pip install lightning"
            )
