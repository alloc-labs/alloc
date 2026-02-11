"""Alloc Framework Callbacks — capture training step counts for artifact enrichment.

Minimal callbacks for popular ML frameworks. Write step_count to a sidecar
file so the dashboard can compute cost-per-step.

Usage (HuggingFace):
    from alloc.callbacks import AllocCallback
    trainer = Trainer(..., callbacks=[AllocCallback()])
"""

from __future__ import annotations

import json
import os
from typing import Optional


def _write_step_count(step_count, framework="unknown"):
    # type: (int, str) -> None
    """Write step count to the alloc artifact sidecar file.

    Creates a small .alloc_steps.json next to the artifact
    that gets picked up during upload.
    """
    sidecar = {
        "step_count": step_count,
        "framework": framework,
    }

    # Write to a well-known location
    steps_path = os.path.join(os.getcwd(), ".alloc_steps.json")
    try:
        with open(steps_path, "w") as f:
            json.dump(sidecar, f)
    except Exception:
        pass


try:
    from transformers import TrainerCallback

    class AllocCallback(TrainerCallback):
        """HuggingFace Trainer callback that captures step count for Alloc."""

        def __init__(self):
            # type: () -> None
            self.step_count = 0  # type: int
            self._last_write_step = 0  # type: int
            self._write_every = 10  # type: int

        def on_step_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step
            if self.step_count - self._last_write_step >= self._write_every:
                _write_step_count(self.step_count, framework="huggingface")
                self._last_write_step = self.step_count

        def on_train_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step
            _write_step_count(self.step_count, framework="huggingface")

except ImportError:
    # transformers not installed — provide a stub that raises a clear error
    class AllocCallback:  # type: ignore[no-redef]
        """Stub — install transformers to use AllocCallback."""

        def __init__(self):
            raise ImportError(
                "AllocCallback requires the `transformers` package. "
                "Install with: pip install transformers"
            )
