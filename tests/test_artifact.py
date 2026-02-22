"""Tests for artifact writer."""

from __future__ import annotations

import gzip
import json
import os
import tempfile

from alloc.artifact_writer import write_report


def test_artifact_has_hardware_section():
    """Artifact should contain hardware section when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_artifact.json.gz")
        hw = {
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_total_vram_mb": 81920.0,
            "driver_version": "535.129.03",
            "cuda_version": "12.2",
            "sm_version": "8.0",
            "num_gpus_detected": 2,
        }
        result_path = write_report(
            probe_result={"peak_vram_mb": 40000},
            output_path=path,
            hardware_context=hw,
        )
        assert result_path == path
        assert os.path.isfile(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["version"] == "0.0.1"
        assert data["hardware"] is not None
        assert data["hardware"]["gpu_name"] == "NVIDIA A100-SXM4-80GB"
        assert data["hardware"]["driver_version"] == "535.129.03"
        assert data["hardware"]["cuda_version"] == "12.2"
        assert data["hardware"]["sm_version"] == "8.0"
        assert data["hardware"]["num_gpus_detected"] == 2


def test_artifact_without_hardware():
    """Artifact without hardware should still work (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_artifact.json.gz")
        result_path = write_report(
            probe_result={"peak_vram_mb": 8000},
            output_path=path,
        )
        assert result_path == path

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["version"] == "0.0.1"
        assert data["hardware"] is None
        assert data["probe"]["peak_vram_mb"] == 8000


def test_artifact_version_bumped():
    """Artifact version should be 0.0.1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        write_report(output_path=path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["version"] == "0.0.1"


def test_artifact_ghost_and_probe():
    """Artifact should contain both ghost and probe when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        ghost = {"param_count_b": 7.0, "total_gb": 42.0}
        probe = {"peak_vram_mb": 35000, "avg_gpu_util": 72.5}
        write_report(
            ghost_report=ghost,
            probe_result=probe,
            output_path=path,
        )

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["ghost"]["param_count_b"] == 7.0
        assert data["probe"]["peak_vram_mb"] == 35000


def test_artifact_with_context():
    """Artifact should include context dict when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        ctx = {
            "git": {"commit_sha": "abc123", "branch": "main", "repo_name": "alloc"},
            "ray": {"job_id": "raysubmit_xyz"},
        }
        write_report(
            probe_result={"peak_vram_mb": 5000},
            output_path=path,
            context=ctx,
        )

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["context"] is not None
        assert data["context"]["git"]["branch"] == "main"
        assert data["context"]["ray"]["job_id"] == "raysubmit_xyz"


def test_artifact_without_context():
    """Artifact without context should have context=null."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        write_report(
            probe_result={"peak_vram_mb": 5000},
            output_path=path,
        )

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["context"] is None


def test_artifact_command_field():
    """Artifact probe should include the command string when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        probe = {
            "peak_vram_mb": 5000,
            "command": "python train.py --epochs 10",
        }
        write_report(
            probe_result=probe,
            output_path=path,
        )

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        assert data["probe"]["command"] == "python train.py --epochs 10"
