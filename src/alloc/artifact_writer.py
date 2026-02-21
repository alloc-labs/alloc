"""Artifact Writer — write alloc_artifact.json.gz.

Optionally uploads to W&B if wandb is active.
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from typing import Optional


def write_report(
    ghost_report: Optional[dict] = None,
    probe_result: Optional[dict] = None,
    output_path: Optional[str] = None,
    hardware_context: Optional[dict] = None,
    context: Optional[dict] = None,
) -> str:
    """Write an artifact to disk.

    Resolution order for output path:
      1. Explicit output_path parameter
      2. ALLOC_OUT env var
      3. ./alloc_artifact.json.gz

    Returns the path written to. Never raises.
    """
    try:
        resolved_path = (
            output_path
            or os.environ.get("ALLOC_OUT", "")
            or "alloc_artifact.json.gz"
        )

        report = {
            "version": "0.0.1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ghost": ghost_report,
            "probe": probe_result,
            "hardware": hardware_context,
            "context": context if context else None,
        }

        with gzip.open(resolved_path, "wt", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        _try_wandb_upload(resolved_path)
        return resolved_path
    except Exception:
        return ""


def _try_wandb_upload(path: str) -> None:
    """Upload to W&B if wandb is active. Silent no-op otherwise."""
    if not os.environ.get("WANDB_RUN_ID"):
        return
    try:
        import wandb
        if wandb.run is not None:
            artifact = wandb.Artifact("alloc-profile", type="profile")
            artifact.add_file(path)
            wandb.run.log_artifact(artifact)
    except Exception:
        pass
