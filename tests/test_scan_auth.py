"""Tests for scan command 401 retry + /scans/cli fallback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

from alloc.cli import app

runner = CliRunner()


def _make_resp(status_code: int, body: dict, url: str = "https://api.example.com/scans"):
    req = httpx.Request("POST", url)
    return httpx.Response(
        status_code,
        request=req,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )


def test_scan_401_refresh_retry(tmp_path: Path):
    """On 401, refresh token and retry on /scans."""
    resp_401 = _make_resp(401, {"detail": "unauthorized"})
    resp_ok = _make_resp(200, {"vram_gb": 16.0, "configs": []})

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.side_effect = [resp_401, resp_ok]

    cfg_file = tmp_path / ".alloc" / "config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"token": "old-tok", "refresh_token": "rt"}))

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with (
        patch("httpx.Client", return_value=mock_client),
        patch("alloc.cli.try_refresh_access_token", return_value="new-tok"),
    ):
        result = runner.invoke(app, ["scan", "--model", "llama-3-8b", "--json"], env=env)

    assert result.exit_code == 0
    assert mock_client.post.call_count == 2
    # Second call should use refreshed token
    second_call = mock_client.post.call_args_list[1]
    assert "Bearer new-tok" in str(second_call)


def test_scan_401_refresh_fails_fallback_public(tmp_path: Path):
    """On 401 + refresh failure, fall back to /scans/cli with warning."""
    resp_401 = _make_resp(401, {"detail": "unauthorized"})
    resp_ok = _make_resp(200, {"vram_gb": 16.0, "configs": []},
                          url="https://api.example.com/scans/cli")

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.side_effect = [resp_401, resp_ok]

    cfg_file = tmp_path / ".alloc" / "config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"token": "old-tok", "refresh_token": "rt"}))

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with (
        patch("httpx.Client", return_value=mock_client),
        patch("alloc.cli.try_refresh_access_token", return_value=None),
    ):
        result = runner.invoke(app, ["scan", "--model", "llama-3-8b", "--json"], env=env)

    assert result.exit_code == 0
    assert mock_client.post.call_count == 2
    # Second call should hit /scans/cli
    second_url = str(mock_client.post.call_args_list[1])
    assert "/scans/cli" in second_url


def test_scan_401_fallback_warns_about_dropped_features(tmp_path: Path):
    """Fallback to public scan warns user about lost org context."""
    resp_401 = _make_resp(401, {"detail": "unauthorized"})
    resp_ok = _make_resp(200, {"vram_gb": 16.0, "configs": []})

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.side_effect = [resp_401, resp_ok]

    cfg_file = tmp_path / ".alloc" / "config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"token": "old-tok", "refresh_token": "rt"}))

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with (
        patch("httpx.Client", return_value=mock_client),
        patch("alloc.cli.try_refresh_access_token", return_value=None),
    ):
        # Non-JSON mode to see the warning message
        result = runner.invoke(app, ["scan", "--model", "llama-3-8b"], env=env)

    assert result.exit_code == 0
    assert "expired" in result.output.lower() or "falling back" in result.output.lower()


def test_scan_no_token_uses_public_directly(tmp_path: Path):
    """Without a token, scan goes directly to /scans/cli."""
    resp_ok = _make_resp(200, {"vram_gb": 16.0, "configs": []})

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = resp_ok

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(app, ["scan", "--model", "llama-3-8b", "--json"], env=env)

    assert result.exit_code == 0
    assert mock_client.post.call_count == 1
    call_url = str(mock_client.post.call_args_list[0])
    assert "/scans/cli" in call_url
