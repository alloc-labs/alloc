"""Artifact loader — parse alloc_artifact.json.gz for runtime-enhanced diagnosis.

Loads the artifact created by `alloc run`, extracting GPU metrics, timing data,
and per-rank distributed information for use by Phase 2 diagnosis rules.

Never crashes. Returns None on any failure.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArtifactData:
    """Parsed runtime artifact — structured for rule consumption."""

    # Hardware (from probe)
    gpu_name: Optional[str] = None
    gpu_count: int = 1
    per_gpu_vram_total_mb: Optional[float] = None
    per_gpu_vram_used_mb: Optional[List[float]] = None
    gpu_utilization_pct: Optional[List[float]] = None
    power_draw_w: Optional[List[float]] = None
    sm_version: Optional[str] = None
    interconnect: Optional[str] = None

    # Timing (from callbacks — may be None)
    step_times_ms: Optional[List[float]] = None
    step_time_p50_ms: Optional[float] = None
    step_time_p90_ms: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    dataloader_wait_pct: Optional[float] = None

    # Per-rank data (from distributed callbacks)
    per_rank_peak_vram_mb: Optional[List[float]] = None
    per_rank_step_times_ms: Optional[List[List[float]]] = None

    # Run metadata
    exit_code: Optional[int] = None
    duration_s: Optional[float] = None
    command: Optional[str] = None
    git_sha: Optional[str] = None
    is_oom: bool = False

    # Aggregate metrics (computed from samples)
    avg_gpu_util: Optional[float] = None
    peak_vram_mb: Optional[float] = None


def load_artifact(path: str) -> Optional[ArtifactData]:
    """Load and parse alloc_artifact.json.gz. Returns None on any failure."""
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
    except Exception:
        return None

    return _parse_artifact(raw)


def find_artifact(directory: str = ".") -> Optional[str]:
    """Find most recent alloc_artifact*.json.gz in directory. Returns path or None."""
    patterns = [
        os.path.join(directory, "alloc_artifact*.json.gz"),
        os.path.join(directory, "alloc_artifact*.json"),
    ]
    candidates = []  # type: List[str]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return None

    # Sort by modification time, return newest
    return max(candidates, key=os.path.getmtime)


def _parse_artifact(raw: dict) -> ArtifactData:
    """Parse raw artifact JSON into ArtifactData."""
    data = ArtifactData()

    probe = raw.get("probe") or {}
    hardware = raw.get("hardware") or {}
    context = raw.get("context") or {}

    # Hardware
    data.gpu_name = hardware.get("gpu_name") or probe.get("gpu_name")
    data.gpu_count = hardware.get("num_gpus_detected") or 1
    data.per_gpu_vram_total_mb = _float_or_none(
        hardware.get("gpu_total_vram_mb") or probe.get("gpu_total_vram_mb")
    )
    data.sm_version = hardware.get("sm_version")
    data.interconnect = probe.get("interconnect_type")

    # Peak VRAM — from probe samples or direct field
    peak = _float_or_none(probe.get("peak_vram_mb"))
    data.peak_vram_mb = peak

    # Per-GPU VRAM: use per_rank_peak_vram_mb if available, else single peak
    per_rank = probe.get("per_rank_peak_vram_mb")
    if isinstance(per_rank, list) and per_rank:
        data.per_gpu_vram_used_mb = [float(v) for v in per_rank]
    elif peak is not None:
        data.per_gpu_vram_used_mb = [peak]

    data.per_rank_peak_vram_mb = data.per_gpu_vram_used_mb

    # GPU utilization from samples
    samples = probe.get("samples") or []
    if samples:
        utils = [s.get("gpu_util_pct") for s in samples if s.get("gpu_util_pct") is not None]
        if utils:
            data.gpu_utilization_pct = [float(u) for u in utils]
            data.avg_gpu_util = sum(data.gpu_utilization_pct) / len(data.gpu_utilization_pct)

        powers = [s.get("power_w") for s in samples if s.get("power_w") is not None]
        if powers:
            data.power_draw_w = [float(p) for p in powers]

    # Avg GPU util from probe aggregate
    if data.avg_gpu_util is None:
        data.avg_gpu_util = _float_or_none(probe.get("avg_gpu_util"))

    # Timing (from callback sidecar data merged into probe)
    data.step_time_p50_ms = _float_or_none(probe.get("step_time_ms_p50"))
    data.step_time_p90_ms = _float_or_none(probe.get("step_time_ms_p90"))
    data.throughput_samples_per_sec = _float_or_none(probe.get("samples_per_sec"))
    data.dataloader_wait_pct = _float_or_none(probe.get("dataloader_wait_pct"))

    # Run metadata
    data.exit_code = probe.get("exit_code")
    data.duration_s = _float_or_none(probe.get("duration_seconds"))
    data.command = probe.get("command")

    # Git SHA from context
    git_ctx = context.get("git") or {}
    data.git_sha = git_ctx.get("commit_sha")

    # OOM detection: exit_code != 0 AND VRAM utilization > 95%
    data.is_oom = _detect_oom(data)

    return data


def _detect_oom(data: ArtifactData) -> bool:
    """Detect probable OOM from exit code and VRAM utilization."""
    if data.exit_code is None or data.exit_code == 0:
        return False

    if data.per_gpu_vram_used_mb and data.per_gpu_vram_total_mb:
        total = data.per_gpu_vram_total_mb
        if total > 0:
            max_used = max(data.per_gpu_vram_used_mb)
            utilization = max_used / total
            if utilization > 0.95:
                return True

    return False


def _float_or_none(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
