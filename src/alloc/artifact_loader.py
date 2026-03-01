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

    # Phase timing (from callbacks with CUDA events)
    phase_forward_ms_p50: Optional[float] = None
    phase_forward_ms_p90: Optional[float] = None
    phase_backward_ms_p50: Optional[float] = None
    phase_backward_ms_p90: Optional[float] = None
    phase_optimizer_ms_p50: Optional[float] = None
    phase_optimizer_ms_p90: Optional[float] = None
    phase_dataloader_ms_p50: Optional[float] = None
    phase_dataloader_ms_p90: Optional[float] = None
    has_phase_timing: bool = False

    # Communication overhead (estimated from phase timing in distributed training)
    comm_overhead_pct: Optional[float] = None

    # Per-rank data (from distributed callbacks)
    per_rank_peak_vram_mb: Optional[List[float]] = None
    per_rank_step_times_ms: Optional[List[List[float]]] = None
    straggler_ratio: Optional[float] = None

    # Architecture metadata (from callback introspection)
    architecture_type: Optional[str] = None
    optimizer_type: Optional[str] = None
    fine_tuning_method: Optional[str] = None
    gradient_checkpointing: Optional[bool] = None
    attention_type: Optional[str] = None
    param_count: Optional[int] = None
    trainable_param_count: Optional[int] = None

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
    gpu_count = hardware.get("num_gpus_detected")
    data.gpu_count = gpu_count if gpu_count is not None and gpu_count > 0 else 1
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

    # Phase timing (from CUDA events in callbacks)
    data.phase_forward_ms_p50 = _float_or_none(probe.get("phase_forward_ms_p50"))
    data.phase_forward_ms_p90 = _float_or_none(probe.get("phase_forward_ms_p90"))
    data.phase_backward_ms_p50 = _float_or_none(probe.get("phase_backward_ms_p50"))
    data.phase_backward_ms_p90 = _float_or_none(probe.get("phase_backward_ms_p90"))
    data.phase_optimizer_ms_p50 = _float_or_none(probe.get("phase_optimizer_ms_p50"))
    data.phase_optimizer_ms_p90 = _float_or_none(probe.get("phase_optimizer_ms_p90"))
    data.phase_dataloader_ms_p50 = _float_or_none(probe.get("phase_dataloader_ms_p50"))
    data.phase_dataloader_ms_p90 = _float_or_none(probe.get("phase_dataloader_ms_p90"))
    data.has_phase_timing = bool(probe.get("has_phase_timing"))
    data.comm_overhead_pct = _float_or_none(probe.get("comm_overhead_pct"))

    # Architecture metadata (from callback introspection)
    data.architecture_type = probe.get("architecture_type")
    data.optimizer_type = probe.get("optimizer_type")
    data.fine_tuning_method = probe.get("fine_tuning_method")
    gc = probe.get("gradient_checkpointing")
    data.gradient_checkpointing = bool(gc) if gc is not None else None
    data.attention_type = probe.get("attention_type")
    data.param_count = _int_or_none(probe.get("param_count"))
    data.trainable_param_count = _int_or_none(probe.get("trainable_param_count"))

    # Raw step times (for per-rank merge / straggler detection)
    raw_times = probe.get("step_times_raw")
    if isinstance(raw_times, list) and raw_times:
        data.step_times_ms = [float(t) for t in raw_times if t is not None]

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


def _int_or_none(val) -> Optional[int]:
    """Convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def find_rank_artifacts(directory: str = ".") -> List[str]:
    """Find all alloc_artifact_rank*.json.gz files in directory."""
    pattern = os.path.join(directory, "alloc_artifact_rank*.json.gz")
    return sorted(glob.glob(pattern))


def merge_artifacts(paths: List[str]) -> Optional[ArtifactData]:
    """Load multiple rank artifacts and merge into a single ArtifactData.

    Combines per-rank VRAM peaks, step times, and computes straggler metrics.
    Returns None if no valid artifacts found.
    """
    artifacts = [load_artifact(p) for p in paths]
    artifacts = [a for a in artifacts if a is not None]
    if not artifacts:
        return None

    merged = ArtifactData()

    # Use GPU info from rank 0
    merged.gpu_name = artifacts[0].gpu_name
    merged.gpu_count = len(artifacts)
    merged.per_gpu_vram_total_mb = artifacts[0].per_gpu_vram_total_mb
    merged.sm_version = artifacts[0].sm_version
    merged.interconnect = artifacts[0].interconnect

    # Per-rank peak VRAM
    peaks = [a.peak_vram_mb for a in artifacts if a.peak_vram_mb is not None]
    if peaks:
        merged.per_rank_peak_vram_mb = peaks
        merged.peak_vram_mb = max(peaks)
        merged.per_gpu_vram_used_mb = peaks

    # Per-rank step times (raw lists from each rank's rolling window)
    rank_step_times = []
    for a in artifacts:
        if a.step_times_ms:
            rank_step_times.append(a.step_times_ms)
    if rank_step_times:
        merged.per_rank_step_times_ms = rank_step_times

    # Aggregate step timing from rank 0 (representative)
    merged.step_time_p50_ms = artifacts[0].step_time_p50_ms
    merged.step_time_p90_ms = artifacts[0].step_time_p90_ms
    merged.throughput_samples_per_sec = artifacts[0].throughput_samples_per_sec

    # Phase timing from rank 0 (representative)
    merged.has_phase_timing = artifacts[0].has_phase_timing
    merged.phase_forward_ms_p50 = artifacts[0].phase_forward_ms_p50
    merged.phase_forward_ms_p90 = artifacts[0].phase_forward_ms_p90
    merged.phase_backward_ms_p50 = artifacts[0].phase_backward_ms_p50
    merged.phase_backward_ms_p90 = artifacts[0].phase_backward_ms_p90
    merged.phase_optimizer_ms_p50 = artifacts[0].phase_optimizer_ms_p50
    merged.phase_optimizer_ms_p90 = artifacts[0].phase_optimizer_ms_p90
    merged.phase_dataloader_ms_p50 = artifacts[0].phase_dataloader_ms_p50
    merged.phase_dataloader_ms_p90 = artifacts[0].phase_dataloader_ms_p90
    merged.comm_overhead_pct = artifacts[0].comm_overhead_pct

    # Straggler detection: ratio of slowest to fastest rank by step time p50
    p50s = [a.step_time_p50_ms for a in artifacts if a.step_time_p50_ms is not None]
    if p50s and len(p50s) > 1 and min(p50s) > 0:
        merged.straggler_ratio = round(max(p50s) / min(p50s), 3)

    # Aggregate GPU util from all ranks
    all_utils = []
    for a in artifacts:
        if a.avg_gpu_util is not None:
            all_utils.append(a.avg_gpu_util)
    if all_utils:
        merged.avg_gpu_util = sum(all_utils) / len(all_utils)

    # Architecture metadata from rank 0
    merged.architecture_type = artifacts[0].architecture_type
    merged.optimizer_type = artifacts[0].optimizer_type
    merged.fine_tuning_method = artifacts[0].fine_tuning_method
    merged.gradient_checkpointing = artifacts[0].gradient_checkpointing
    merged.attention_type = artifacts[0].attention_type
    merged.param_count = artifacts[0].param_count
    merged.trainable_param_count = artifacts[0].trainable_param_count

    # Run metadata from rank 0
    merged.exit_code = artifacts[0].exit_code
    merged.duration_s = artifacts[0].duration_s
    merged.command = artifacts[0].command
    merged.git_sha = artifacts[0].git_sha
    merged.is_oom = any(a.is_oom for a in artifacts)

    return merged
