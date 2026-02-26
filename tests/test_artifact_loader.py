"""Tests for artifact_loader.py — artifact parsing, auto-discovery, OOM detection."""

from __future__ import annotations

import gzip
import json
import os
import tempfile

import pytest

from alloc.artifact_loader import (
    ArtifactData,
    load_artifact,
    find_artifact,
    find_rank_artifacts,
    merge_artifacts,
    _detect_oom,
    _parse_artifact,
)


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


# ---------------------------------------------------------------------------
# Phase timing fields — _parse_artifact
# ---------------------------------------------------------------------------

class TestPhaseTimingFields:
    def test_phase_timing_parsed_from_probe(self):
        raw = {
            "probe": {
                "has_phase_timing": True,
                "phase_forward_ms_p50": 40.0,
                "phase_forward_ms_p90": 45.0,
                "phase_backward_ms_p50": 35.0,
                "phase_backward_ms_p90": 40.0,
                "phase_optimizer_ms_p50": 15.0,
                "phase_optimizer_ms_p90": 18.0,
                "phase_dataloader_ms_p50": 10.0,
                "phase_dataloader_ms_p90": 12.0,
            }
        }
        data = _parse_artifact(raw)
        assert data.has_phase_timing is True
        assert data.phase_forward_ms_p50 == 40.0
        assert data.phase_forward_ms_p90 == 45.0
        assert data.phase_backward_ms_p50 == 35.0
        assert data.phase_optimizer_ms_p50 == 15.0
        assert data.phase_dataloader_ms_p50 == 10.0

    def test_no_phase_timing(self):
        raw = {"probe": {"peak_vram_mb": 1000}}
        data = _parse_artifact(raw)
        assert data.has_phase_timing is False
        assert data.phase_forward_ms_p50 is None

    def test_comm_overhead_pct_parsed(self):
        raw = {"probe": {"comm_overhead_pct": 42.5, "has_phase_timing": True}}
        data = _parse_artifact(raw)
        assert data.comm_overhead_pct == 42.5

    def test_comm_overhead_pct_none_when_absent(self):
        raw = {"probe": {"peak_vram_mb": 1000}}
        data = _parse_artifact(raw)
        assert data.comm_overhead_pct is None

    def test_step_times_raw_parsed(self):
        raw = {"probe": {"step_times_raw": [100.0, 110.0, 105.0]}}
        data = _parse_artifact(raw)
        assert data.step_times_ms == [100.0, 110.0, 105.0]


# ---------------------------------------------------------------------------
# Architecture metadata fields — _parse_artifact
# ---------------------------------------------------------------------------

class TestArchitectureMetadataFields:
    def test_architecture_fields_parsed(self):
        raw = {
            "probe": {
                "architecture_type": "transformer",
                "optimizer_type": "AdamW",
                "fine_tuning_method": "lora",
                "gradient_checkpointing": True,
                "attention_type": "flash_attention_2",
                "param_count": 7000000000,
                "trainable_param_count": 4200000,
            }
        }
        data = _parse_artifact(raw)
        assert data.architecture_type == "transformer"
        assert data.optimizer_type == "AdamW"
        assert data.fine_tuning_method == "lora"
        assert data.gradient_checkpointing is True
        assert data.attention_type == "flash_attention_2"
        assert data.param_count == 7000000000
        assert data.trainable_param_count == 4200000

    def test_no_architecture_fields(self):
        raw = {"probe": {"peak_vram_mb": 1000}}
        data = _parse_artifact(raw)
        assert data.architecture_type is None
        assert data.optimizer_type is None
        assert data.fine_tuning_method is None
        assert data.gradient_checkpointing is None
        assert data.param_count is None

    def test_partial_architecture_fields(self):
        raw = {
            "probe": {
                "architecture_type": "moe",
                "optimizer_type": "SGD",
            }
        }
        data = _parse_artifact(raw)
        assert data.architecture_type == "moe"
        assert data.optimizer_type == "SGD"
        assert data.fine_tuning_method is None
        assert data.gradient_checkpointing is None

    def test_architecture_roundtrip_gz(self, tmp_path):
        """Full roundtrip: write artifact with arch fields, load, verify."""
        artifact = {
            "probe": {
                "peak_vram_mb": 30000,
                "exit_code": 0,
                "architecture_type": "transformer",
                "optimizer_type": "AdamW",
                "fine_tuning_method": "lora",
                "gradient_checkpointing": True,
                "attention_type": "sdpa",
                "param_count": 7000000000,
                "trainable_param_count": 4200000,
            },
            "hardware": {"gpu_name": "NVIDIA A100"},
            "context": {},
        }
        path = str(tmp_path / "alloc_artifact.json.gz")
        _write_artifact(path, artifact)
        data = load_artifact(path)
        assert data is not None
        assert data.architecture_type == "transformer"
        assert data.optimizer_type == "AdamW"
        assert data.fine_tuning_method == "lora"
        assert data.gradient_checkpointing is True
        assert data.attention_type == "sdpa"
        assert data.param_count == 7000000000
        assert data.trainable_param_count == 4200000


# ---------------------------------------------------------------------------
# find_rank_artifacts
# ---------------------------------------------------------------------------

class TestFindRankArtifacts:
    def test_finds_rank_files(self, tmp_path):
        # Create fake rank artifacts
        for i in range(4):
            path = tmp_path / f"alloc_artifact_rank{i}.json.gz"
            with gzip.open(str(path), "wt") as f:
                json.dump({"probe": {"rank": i}}, f)

        found = find_rank_artifacts(str(tmp_path))
        assert len(found) == 4
        assert "rank0" in found[0]
        assert "rank3" in found[3]

    def test_empty_directory(self, tmp_path):
        found = find_rank_artifacts(str(tmp_path))
        assert found == []


# ---------------------------------------------------------------------------
# merge_artifacts
# ---------------------------------------------------------------------------

class TestMergeArtifacts:
    def _write_rank_artifact(self, path, rank, peak_vram, step_p50, step_times=None):
        probe = {
            "peak_vram_mb": peak_vram,
            "avg_gpu_util": 70.0 + rank,
            "step_time_ms_p50": step_p50,
            "step_time_ms_p90": step_p50 * 1.2,
            "gpu_name": "NVIDIA A100",
            "gpu_total_vram_mb": 81920,
            "num_gpus_detected": 1,
            "rank": rank,
        }
        if step_times:
            probe["step_times_raw"] = step_times
        with gzip.open(path, "wt") as f:
            json.dump({"probe": probe, "hardware": {"gpu_name": "NVIDIA A100"}}, f)

    def test_merge_4_ranks(self, tmp_path):
        paths = []
        for i in range(4):
            p = str(tmp_path / f"rank{i}.json.gz")
            self._write_rank_artifact(p, i, 30000 + i * 1000, 100.0 + i * 5)
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged is not None
        assert merged.gpu_count == 4
        assert merged.gpu_name == "NVIDIA A100"
        assert merged.peak_vram_mb == 33000  # max of ranks
        assert len(merged.per_rank_peak_vram_mb) == 4

    def test_straggler_detection(self, tmp_path):
        paths = []
        # Rank 0-2: 100ms, Rank 3: 150ms (straggler)
        for i in range(3):
            p = str(tmp_path / f"rank{i}.json.gz")
            self._write_rank_artifact(p, i, 30000, 100.0)
            paths.append(p)
        p = str(tmp_path / "rank3.json.gz")
        self._write_rank_artifact(p, 3, 30000, 150.0)
        paths.append(p)

        merged = merge_artifacts(paths)
        assert merged is not None
        assert merged.straggler_ratio is not None
        assert merged.straggler_ratio == 1.5  # 150/100

    def test_no_straggler(self, tmp_path):
        paths = []
        for i in range(4):
            p = str(tmp_path / f"rank{i}.json.gz")
            self._write_rank_artifact(p, i, 30000, 100.0)
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged is not None
        assert merged.straggler_ratio == 1.0

    def test_merge_with_step_times(self, tmp_path):
        paths = []
        for i in range(2):
            p = str(tmp_path / f"rank{i}.json.gz")
            self._write_rank_artifact(p, i, 30000, 100.0, step_times=[95.0, 100.0, 105.0])
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged is not None
        assert merged.per_rank_step_times_ms is not None
        assert len(merged.per_rank_step_times_ms) == 2

    def test_merge_empty_list(self):
        result = merge_artifacts([])
        assert result is None

    def test_merge_invalid_files(self, tmp_path):
        p = str(tmp_path / "bad.json.gz")
        with open(p, "w") as f:
            f.write("not valid")
        result = merge_artifacts([p])
        assert result is None

    def test_merge_preserves_phase_timing(self, tmp_path):
        """Phase timing from rank 0 should be preserved in merged artifact."""
        probe = {
            "peak_vram_mb": 30000,
            "step_time_ms_p50": 100.0,
            "has_phase_timing": True,
            "phase_forward_ms_p50": 40.0,
            "phase_backward_ms_p50": 35.0,
            "gpu_name": "NVIDIA A100",
        }
        paths = []
        for i in range(2):
            p = str(tmp_path / f"rank{i}.json.gz")
            with gzip.open(p, "wt") as f:
                json.dump({"probe": {**probe, "rank": i}}, f)
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged.has_phase_timing is True
        assert merged.phase_forward_ms_p50 == 40.0
        assert merged.phase_backward_ms_p50 == 35.0

    def test_merge_preserves_comm_overhead_pct(self, tmp_path):
        """comm_overhead_pct from rank 0 should be preserved in merged artifact."""
        probe = {
            "peak_vram_mb": 30000,
            "step_time_ms_p50": 100.0,
            "has_phase_timing": True,
            "comm_overhead_pct": 33.3,
            "gpu_name": "NVIDIA A100",
        }
        paths = []
        for i in range(4):
            p = str(tmp_path / f"rank{i}.json.gz")
            with gzip.open(p, "wt") as f:
                json.dump({"probe": {**probe, "rank": i}}, f)
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged.comm_overhead_pct == 33.3

    def test_merge_preserves_architecture_metadata(self, tmp_path):
        """Architecture metadata from rank 0 should be preserved in merged artifact."""
        probe = {
            "peak_vram_mb": 30000,
            "step_time_ms_p50": 100.0,
            "gpu_name": "NVIDIA A100",
            "architecture_type": "transformer",
            "optimizer_type": "AdamW",
            "fine_tuning_method": "lora",
            "gradient_checkpointing": True,
            "attention_type": "flash_attention_2",
            "param_count": 7000000000,
            "trainable_param_count": 4200000,
        }
        paths = []
        for i in range(2):
            p = str(tmp_path / f"rank{i}.json.gz")
            with gzip.open(p, "wt") as f:
                json.dump({"probe": {**probe, "rank": i}}, f)
            paths.append(p)

        merged = merge_artifacts(paths)
        assert merged.architecture_type == "transformer"
        assert merged.optimizer_type == "AdamW"
        assert merged.fine_tuning_method == "lora"
        assert merged.gradient_checkpointing is True
        assert merged.attention_type == "flash_attention_2"
        assert merged.param_count == 7000000000
        assert merged.trainable_param_count == 4200000
