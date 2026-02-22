"""Tests for CLI auth/config helpers (login/logout/whoami/upload retry)."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

from alloc.cli import app
from alloc.browser_auth import _generate_pkce_pair, _find_open_port


runner = CliRunner()


def _config_path(home: Path) -> Path:
    return home / ".alloc" / "config.json"


def test_login_token_saves_config(tmp_path: Path):
    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }
    result = runner.invoke(app, ["login", "--token", "abc123"], env=env)
    assert result.exit_code == 0

    cfg_file = _config_path(tmp_path)
    assert cfg_file.exists()
    cfg = json.loads(cfg_file.read_text())

    assert cfg["token"] == "abc123"
    assert cfg["api_url"] == "https://api.example.com"
    assert "refresh_token" not in cfg
    assert "email" not in cfg


def test_logout_clears_config(tmp_path: Path):
    cfg_file = _config_path(tmp_path)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps({"token": "t", "refresh_token": "r", "email": "e@x.com", "api_url": "x"}))

    env = {"HOME": str(tmp_path)}
    result = runner.invoke(app, ["logout"], env=env)
    assert result.exit_code == 0

    cfg = json.loads(cfg_file.read_text())
    assert "token" not in cfg
    assert "refresh_token" not in cfg
    assert "email" not in cfg
    assert cfg.get("api_url") == "x"


def test_whoami_not_logged_in_json(tmp_path: Path):
    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }
    result = runner.invoke(app, ["whoami", "--json"], env=env)
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["logged_in"] is False
    assert data["api_url"] == "https://api.example.com"


def test_whoami_logged_in_json(tmp_path: Path):
    profile_resp = MagicMock()
    profile_resp.raise_for_status.return_value = None
    profile_resp.json.return_value = {
        "email": "user@example.com",
        "user_id": "user-123",
        "onboarding_complete": True,
    }

    fleet_resp = MagicMock()
    fleet_resp.raise_for_status.return_value = None
    fleet_resp.json.return_value = {
        "objective": "best_value",
        "priority_cost": 50,
        "budget_monthly": 1000,
        "effective_budget_monthly": 800,
        "budget_cap_applied": True,
        "org_budget": {"budget_monthly_usd": 800},
        "gpus": [
            {"fleet_status": "in_fleet"},
            {"fleet_status": "explore"},
            {"fleet_status": "catalog"},
        ],
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.side_effect = [profile_resp, fleet_resp]

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
        "ALLOC_TOKEN": "t0",
    }

    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(app, ["whoami", "--json"], env=env)

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["logged_in"] is True
    assert data["token_source"] == "env"
    assert data["email"] == "user@example.com"
    assert data["fleet_count"] == 1
    assert data["explore_count"] == 1
    assert data["effective_budget_monthly_usd"] == 800
    assert data["budget_cap_applied"] is True


def test_upload_retries_after_refresh_on_401(tmp_path: Path):
    import gzip

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    # Seed config with an expired token.
    cfg_file = _config_path(tmp_path)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps({"token": "old-token", "refresh_token": "rtok"}))

    # Minimal artifact
    artifact = tmp_path / "artifact.json.gz"
    with gzip.open(str(artifact), "wt", encoding="utf-8") as f:
        json.dump({"probe": {"peak_vram_mb": 1}, "ghost": {}}, f)

    req = httpx.Request("POST", "https://api.example.com/runs/ingest")
    resp_401 = httpx.Response(401, request=req, text="unauthorized")
    err_401 = httpx.HTTPStatusError("401 unauthorized", request=req, response=resp_401)

    mock_upload = MagicMock()
    mock_upload.side_effect = [err_401, {"run_id": "run-123"}]

    with (
        patch("alloc.cli.try_refresh_access_token", return_value="new-token"),
        patch("alloc.upload.upload_artifact", mock_upload),
    ):
        result = runner.invoke(app, ["upload", str(artifact)], env=env)

    assert result.exit_code == 0
    assert mock_upload.call_count == 2
    # First call uses old token; second uses refreshed token.
    assert mock_upload.call_args_list[0].args[2] == "old-token"
    assert mock_upload.call_args_list[1].args[2] == "new-token"
    assert "Retrying" in result.output
    assert "Uploaded" in result.output


# ── PKCE helpers ──────────────────────────────────────────────────────────


def test_pkce_pair_generates_valid_s256_challenge():
    """Verify that the challenge is the S256 hash of the verifier."""
    verifier, challenge = _generate_pkce_pair()

    # Recompute expected challenge from verifier.
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    assert challenge == expected
    assert len(verifier) > 40  # tokens are ~86 chars


def test_pkce_pair_unique():
    """Each call should produce a unique pair."""
    pair_a = _generate_pkce_pair()
    pair_b = _generate_pkce_pair()
    assert pair_a[0] != pair_b[0]
    assert pair_a[1] != pair_b[1]


def test_find_open_port_returns_int():
    port = _find_open_port()
    assert isinstance(port, int)
    assert port >= 17256


def test_find_open_port_skips_busy(tmp_path: Path):
    """If a port is in use, the finder should skip it."""
    import socket

    # Bind the first port in the range.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 17256))
    try:
        port = _find_open_port(start=17256)
        # Should return a port > 17256 since 17256 is busy.
        assert port > 17256
    finally:
        s.close()


# ── Browser login flow ───────────────────────────────────────────────────


def test_browser_login_saves_tokens(tmp_path: Path):
    """--browser flag triggers browser_login and saves tokens."""
    mock_result = {
        "access_token": "at-browser-123",
        "refresh_token": "rt-browser-456",
        "email": "oauth@example.com",
    }

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with patch("alloc.browser_auth.browser_login", return_value=mock_result):
        result = runner.invoke(app, ["login", "--browser"], env=env)

    assert result.exit_code == 0, result.output
    assert "Logged in as oauth@example.com" in result.output

    cfg = json.loads((tmp_path / ".alloc" / "config.json").read_text())
    assert cfg["token"] == "at-browser-123"
    assert cfg["refresh_token"] == "rt-browser-456"
    assert cfg["email"] == "oauth@example.com"
    assert cfg["api_url"] == "https://api.example.com"


def test_browser_login_with_azure_provider(tmp_path: Path):
    mock_result = {
        "access_token": "at-azure",
        "refresh_token": "rt-azure",
        "email": "azure@example.com",
    }

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with patch("alloc.browser_auth.browser_login", return_value=mock_result) as mock_bl:
        result = runner.invoke(
            app, ["login", "--browser", "--provider", "azure"], env=env
        )

    assert result.exit_code == 0, result.output
    assert "azure@example.com" in result.output
    # Verify provider was passed through.
    mock_bl.assert_called_once()
    assert mock_bl.call_args.kwargs["provider"] == "azure"


def test_browser_login_invalid_provider(tmp_path: Path):
    env = {"HOME": str(tmp_path)}
    result = runner.invoke(
        app, ["login", "--browser", "--provider", "facebook"], env=env
    )
    assert result.exit_code != 0
    assert "Invalid --provider" in result.output


def test_browser_login_timeout(tmp_path: Path):
    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }

    with patch(
        "alloc.browser_auth.browser_login",
        side_effect=RuntimeError("Login timed out"),
    ):
        result = runner.invoke(app, ["login", "--browser"], env=env)

    assert result.exit_code != 0
    assert "Login timed out" in result.output


def test_password_failure_shows_browser_hint(tmp_path: Path):
    """When password login fails with 'Invalid login credentials', show --browser hint."""
    req = httpx.Request("POST", "https://sb.example.com/auth/v1/token")
    resp_body = json.dumps({"error_description": "Invalid login credentials"}).encode()
    resp = httpx.Response(400, request=req, content=resp_body, headers={"content-type": "application/json"})
    err = httpx.HTTPStatusError("400", request=req, response=resp)

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.side_effect = err

    env = {"HOME": str(tmp_path)}

    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(
            app,
            ["login"],
            input="test@example.com\nwrongpass\n",
            env=env,
        )

    assert result.exit_code != 0
    assert "invalid login credentials" in result.output.lower()
    assert "alloc login --browser" in result.output
