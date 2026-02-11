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
