"""Tests for CLI auth/config helpers (login/logout/whoami/upload retry)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

from alloc.cli import app


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
