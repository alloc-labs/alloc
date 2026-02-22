"""Tests for artifact_loader.py — artifact parsing, auto-discovery, OOM detection."""

from __future__ import annotations

import gzip
import json
import os

from alloc.artifact_loader import ArtifactData, load_artifact, find_artifact, _detect_oom


def _write_artifact(path, data):
    """Write a test artifact to disk."""
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


SAMPLE_ARTIFACT = {
    "version": "0.0.1",
    "timestamp": "2026-02-21T00:00:00Z",
    "ghost": None,
    "probe": {
        "peak_vram_mb": 31200.0,
        "avg_gpu_util": 72.3,
        "duration_seconds": 24.1,
        "exit_code": 0,
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_vram_mb": 40960.0,
        "step_time_ms_p50": 148.5,
        "step_time_ms_p90": 162.3,
        "samples_per_sec": 42.3,
        "per_rank_peak_vram_mb": [31200.0, 31400.0],
        "command": "python train.py",
        "interconnect_type": "nvlink",
        "samples": [
            {"t": 0.0, "vram_mb": 28000, "gpu_util_pct": 65.0, "power_w": 250},
            {"t": 1.0, "vram_mb": 31200, "gpu_util_pct": 80.0, "power_w": 300},
        ],
    },
    "hardware": {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_vram_mb": 40960.0,
        "driver_version": "535.129.03",
        "cuda_version": "12.2",
        "sm_version": "8.0",
        "num_gpus_detected": 2,
    },
    "context": {
        "git": {"commit_sha": "abc123", "branch": "main"},
    },
}


# ---------------------------------------------------------------------------
# load_artifact — basic parsing
# ---------------------------------------------------------------------------

def test_load_artifact_gz(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data is not None
    assert data.gpu_name == "NVIDIA A100-SXM4-40GB"
    assert data.gpu_count == 2
    assert data.per_gpu_vram_total_mb == 40960.0
    assert data.peak_vram_mb == 31200.0
    assert data.sm_version == "8.0"
    assert data.interconnect == "nvlink"


def test_load_artifact_json(tmp_path):
    path = str(tmp_path / "alloc_artifact.json")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data is not None
    assert data.gpu_name == "NVIDIA A100-SXM4-40GB"


def test_load_artifact_timing(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data.step_time_p50_ms == 148.5
    assert data.step_time_p90_ms == 162.3
    assert data.throughput_samples_per_sec == 42.3


def test_load_artifact_per_rank(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data.per_gpu_vram_used_mb == [31200.0, 31400.0]
    assert data.per_rank_peak_vram_mb == [31200.0, 31400.0]


def test_load_artifact_gpu_util_from_samples(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data.gpu_utilization_pct == [65.0, 80.0]
    assert data.avg_gpu_util is not None
    assert abs(data.avg_gpu_util - 72.5) < 0.1


def test_load_artifact_run_metadata(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    data = load_artifact(path)
    assert data.exit_code == 0
    assert data.duration_s == 24.1
    assert data.command == "python train.py"
    assert data.git_sha == "abc123"
    assert data.is_oom is False


def test_load_artifact_missing_file():
    data = load_artifact("/nonexistent/path.json.gz")
    assert data is None


def test_load_artifact_corrupt_file(tmp_path):
    path = str(tmp_path / "bad.json.gz")
    with open(path, "wb") as f:
        f.write(b"not gzip data")
    data = load_artifact(path)
    assert data is None


def test_load_artifact_minimal(tmp_path):
    """Artifact with minimal probe data should still parse."""
    path = str(tmp_path / "alloc_artifact.json.gz")
    minimal = {
        "version": "0.0.1",
        "probe": {"peak_vram_mb": 5000, "exit_code": 0},
        "hardware": None,
        "context": None,
    }
    _write_artifact(path, minimal)
    data = load_artifact(path)
    assert data is not None
    assert data.peak_vram_mb == 5000.0
    assert data.gpu_name is None
    assert data.step_time_p50_ms is None


def test_load_artifact_empty_probe(tmp_path):
    """Artifact with empty probe dict should parse without error."""
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, {"version": "0.0.1", "probe": {}, "hardware": {}, "context": {}})
    data = load_artifact(path)
    assert data is not None
    assert data.peak_vram_mb is None
    assert data.gpu_count == 1


# ---------------------------------------------------------------------------
# find_artifact — auto-discovery
# ---------------------------------------------------------------------------

def test_find_artifact_in_cwd(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    _write_artifact(path, SAMPLE_ARTIFACT)
    found = find_artifact(str(tmp_path))
    assert found == path


def test_find_artifact_most_recent(tmp_path):
    import time
    old = str(tmp_path / "alloc_artifact_old.json.gz")
    _write_artifact(old, SAMPLE_ARTIFACT)
    time.sleep(0.05)
    new = str(tmp_path / "alloc_artifact_new.json.gz")
    _write_artifact(new, SAMPLE_ARTIFACT)
    found = find_artifact(str(tmp_path))
    assert found == new


def test_find_artifact_none(tmp_path):
    found = find_artifact(str(tmp_path))
    assert found is None


def test_find_artifact_json_fallback(tmp_path):
    """Plain .json artifact should also be discovered."""
    path = str(tmp_path / "alloc_artifact.json")
    _write_artifact(path, SAMPLE_ARTIFACT)
    found = find_artifact(str(tmp_path))
    assert found == path


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------

def test_oom_detected(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    oom_artifact = {
        "version": "0.0.1",
        "probe": {
            "peak_vram_mb": 39600,
            "exit_code": -9,
            "gpu_total_vram_mb": 40960,
            "per_rank_peak_vram_mb": [39600],
        },
        "hardware": {"gpu_total_vram_mb": 40960},
        "context": None,
    }
    _write_artifact(path, oom_artifact)
    data = load_artifact(path)
    assert data is not None
    assert data.is_oom is True
    assert data.exit_code == -9


def test_oom_not_detected_clean_exit(tmp_path):
    path = str(tmp_path / "alloc_artifact.json.gz")
    artifact = {
        "version": "0.0.1",
        "probe": {
            "peak_vram_mb": 39600,
            "exit_code": 0,
            "gpu_total_vram_mb": 40960,
            "per_rank_peak_vram_mb": [39600],
        },
        "hardware": {"gpu_total_vram_mb": 40960},
        "context": None,
    }
    _write_artifact(path, artifact)
    data = load_artifact(path)
    assert data is not None
    assert data.is_oom is False


def test_oom_not_detected_low_vram():
    """Non-zero exit but low VRAM — not OOM."""
    data = ArtifactData(
        exit_code=1,
        per_gpu_vram_used_mb=[20000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    assert _detect_oom(data) is False


def test_detect_oom_no_vram_data():
    """Non-zero exit but no VRAM data — can't confirm OOM."""
    data = ArtifactData(exit_code=-9)
    assert _detect_oom(data) is False
