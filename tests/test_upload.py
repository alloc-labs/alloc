"""Tests for upload module — tier error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

import pytest

from alloc.upload import UploadLimitError, upload_artifact


def _make_artifact(tmp_path):
    """Create a minimal gzipped artifact file."""
    import gzip

    artifact = tmp_path / "test_artifact.json.gz"
    data = {"probe": {"peak_vram_mb": 1000}, "ghost": {}}
    with gzip.open(str(artifact), "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return str(artifact)


def test_upload_limit_error_raised(tmp_path):
    """429 response should raise UploadLimitError."""
    artifact_path = _make_artifact(tmp_path)

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {
        "detail": {
            "error": "upload_limit_reached",
            "tier": "free",
            "limit": 10,
            "used": 10,
            "message": "Upload limit reached (10/10 this month).",
        }
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(UploadLimitError) as exc_info:
            upload_artifact(artifact_path, "http://localhost:8000", "test-token")

    assert exc_info.value.status_code == 429


def test_upload_limit_error_detail(tmp_path):
    """UploadLimitError should carry parsed detail with tier, limit, used."""
    artifact_path = _make_artifact(tmp_path)

    detail = {
        "error": "upload_limit_reached",
        "tier": "free",
        "limit": 10,
        "used": 10,
        "message": "Upload limit reached (10/10 this month).",
    }
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"detail": detail}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(UploadLimitError) as exc_info:
            upload_artifact(artifact_path, "http://localhost:8000", "test-token")

    err = exc_info.value
    assert err.detail["tier"] == "free"
    assert err.detail["limit"] == 10
    assert err.detail["used"] == 10


def test_upload_403_raises(tmp_path):
    """403 response should also raise UploadLimitError."""
    artifact_path = _make_artifact(tmp_path)

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.json.return_value = {
        "detail": {
            "error": "ai_requires_pro",
            "tier": "free",
            "message": "AI analysis requires a Pro plan.",
        }
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(UploadLimitError) as exc_info:
            upload_artifact(artifact_path, "http://localhost:8000", "test-token")

    assert exc_info.value.status_code == 403


def test_upload_extracts_command_and_git_context(tmp_path):
    """Upload should extract command, git_sha, git_branch from artifact."""
    import gzip

    artifact = tmp_path / "ctx_artifact.json.gz"
    data = {
        "probe": {
            "peak_vram_mb": 5000,
            "command": "python train.py --epochs 10",
        },
        "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
        "ghost": {},
        "context": {
            "git": {
                "commit_sha": "abc123def456",
                "branch": "feature/new-model",
                "repo_name": "my-repo",
            }
        },
    }
    with gzip.open(str(artifact), "wt", encoding="utf-8") as f:
        json.dump(data, f)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"run_id": "run-ctx", "status": "completed"}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        upload_artifact(str(artifact), "http://localhost:8000", "test-token")

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["command"] == "python train.py --epochs 10"
    assert payload["git_sha"] == "abc123def456"
    assert payload["git_branch"] == "feature/new-model"


def test_upload_handles_missing_context(tmp_path):
    """Upload should handle missing context gracefully."""
    import gzip

    artifact = tmp_path / "no_ctx_artifact.json.gz"
    data = {"probe": {"peak_vram_mb": 5000}, "ghost": {}}
    with gzip.open(str(artifact), "wt", encoding="utf-8") as f:
        json.dump(data, f)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"run_id": "run-no-ctx", "status": "completed"}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        upload_artifact(str(artifact), "http://localhost:8000", "test-token")

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["command"] is None
    assert payload["git_sha"] is None
    assert payload["git_branch"] is None


def test_upload_enriches_payload_from_probe_and_hardware(tmp_path):
    """Upload includes GPU/timing fields used by ingest analysis."""
    import gzip

    artifact = tmp_path / "rich_artifact.json.gz"
    data = {
        "probe": {
            "peak_vram_mb": 50234.0,
            "avg_gpu_util": 71.2,
            "avg_power_watts": 281.5,
            "duration_seconds": 95.3,
            "exit_code": 0,
            "samples": [{"t": 0.0, "vram_mb": 50000, "gpu_util_pct": 70}],
            "step_count": 320,
            "step_time_ms_p50": 142.1,
            "step_time_ms_p90": 161.8,
            "samples_per_sec": 22.9,
            "dataloader_wait_pct": 11.3,
        },
        "hardware": {
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "num_gpus_detected": 4,
        },
        "ghost": {},
    }
    with gzip.open(str(artifact), "wt", encoding="utf-8") as f:
        json.dump(data, f)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"run_id": "run-123", "status": "completed"}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        resp = upload_artifact(str(artifact), "http://localhost:8000", "test-token")

    assert resp["run_id"] == "run-123"
    payload = mock_client.post.call_args.kwargs["json"]

    assert payload["gpu_type"] == "A100-80GB"
    assert payload["num_gpus"] == 4
    assert payload["step_count"] == 320
    assert payload["step_time_ms_p50"] == 142.1
    assert payload["step_time_ms_p90"] == 161.8
    assert payload["samples_per_sec"] == 22.9
    assert payload["dataloader_wait_pct"] == 11.3
