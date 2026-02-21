"""Tests for `alloc init --from-org` (pull org fleet/budget into .alloc.yaml)."""

from __future__ import annotations

import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from alloc.cli import app


runner = CliRunner()


def test_init_from_org_requires_login(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
    }
    result = runner.invoke(app, ["init", "--from-org", "--yes"], env=env)
    assert result.exit_code == 1
    assert ".alloc.yaml" not in {p.name for p in tmp_path.iterdir()}


def test_init_from_org_writes_alloc_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fleet_payload = {
        "objective": "fastest_within_budget",
        "priority_cost": 40,
        "budget_monthly": 10000,
        "effective_budget_monthly": 8000,
        "budget_cap_applied": True,
        "gpus": [
            {
                "gpu_id": "H100-80GB",
                "display_name": "H100-80GB",
                "fleet_status": "in_fleet",
                "cloud": "aws",
                "max_count": 8,
                "rate": 3.99,
                "rate_source": "org",
            },
            {
                "gpu_id": "A100-80GB",
                "display_name": "A100-80GB",
                "fleet_status": "explore",
                "cloud": "aws",
                "rate": 2.50,
                "rate_source": "catalog",
            },
        ],
    }

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = fleet_payload

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.return_value = resp

    env = {
        "HOME": str(tmp_path),
        "ALLOC_API_URL": "https://api.example.com",
        "ALLOC_TOKEN": "t0",
    }

    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(app, ["init", "--from-org", "--yes"], env=env)

    assert result.exit_code == 0
    cfg_path = tmp_path / ".alloc.yaml"
    assert cfg_path.exists()

    data = yaml.safe_load(cfg_path.read_text())
    assert data["objective"] == "fastest_within_budget"

    # H100 fleet entry is resolved to stable ID and includes org rate override.
    assert data["fleet"][0]["gpu"] == "nvidia-h100-sxm-80gb"
    assert data["fleet"][0]["cloud"] == "aws"
    assert data["fleet"][0]["count"] == 8
    assert data["fleet"][0]["rate"] == 3.99

    # A100 explore entry is resolved to stable ID and does not include catalog rate.
    assert data["explore"][0]["gpu"] == "nvidia-a100-sxm-80gb"
    assert data["explore"][0]["cloud"] == "aws"
    assert "rate" not in data["explore"][0]

    assert data["priority"]["cost"] == 40
    assert data["budget"]["monthly_usd"] == 8000
    # monthly/730, rounded to 4 decimals
    assert abs(float(data["budget"]["hourly_usd"]) - (8000 / 730.0)) < 1e-4

