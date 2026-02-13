"""Tests for .alloc.yaml config loading, validation, and write/read."""

from __future__ import annotations

import os
import tempfile

import pytest

from alloc.yaml_config import (
    AllocConfig,
    FleetEntry,
    load_alloc_config,
    validate_config,
    write_alloc_config,
    _parse_config,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_minimal():
    """Minimal config with just fleet."""
    cfg = _parse_config({"fleet": [{"gpu": "H100"}]})
    assert len(cfg.fleet) == 1
    assert cfg.fleet[0].gpu == "H100"
    assert cfg.priority_cost == 50


def test_parse_string_shorthand():
    """Fleet entries can be plain strings."""
    cfg = _parse_config({"fleet": ["H100", "A100"]})
    assert len(cfg.fleet) == 2
    assert cfg.fleet[0].gpu == "H100"
    assert cfg.fleet[1].gpu == "A100"


def test_parse_full_config():
    """Full config with all fields."""
    cfg = _parse_config({
        "fleet": [
            {"gpu": "H100", "cloud": "aws", "count": 8, "rate": 3.50},
            {"gpu": "A100", "cloud": "gcp"},
        ],
        "explore": [
            {"gpu": "H200"},
        ],
        "objective": "fastest_within_budget",
        "priority": {"cost": 70, "latency": 30},
        "budget": {"monthly_usd": 5000, "hourly_usd": 50},
    })
    assert len(cfg.fleet) == 2
    assert cfg.fleet[0].rate == 3.50
    assert cfg.fleet[0].count == 8
    assert len(cfg.explore) == 1
    assert cfg.explore[0].explore is True
    assert cfg.objective == "fastest_within_budget"
    assert cfg.priority_cost == 70
    assert cfg.priority_latency == 30
    assert cfg.budget_monthly == 5000
    assert cfg.budget_hourly == 50


def test_parse_empty_dict():
    """Empty dict gives defaults."""
    cfg = _parse_config({})
    assert cfg.fleet == []
    assert cfg.explore == []
    assert cfg.priority_cost == 50
    assert cfg.budget_monthly is None


def test_fleet_gpu_ids():
    cfg = AllocConfig(
        fleet=[FleetEntry(gpu="H100"), FleetEntry(gpu="A100")],
        explore=[FleetEntry(gpu="H200", explore=True)],
    )
    assert cfg.fleet_gpu_ids == ["H100", "A100"]
    assert cfg.explore_gpu_ids == ["H200"]
    assert cfg.all_gpu_ids == ["H100", "A100", "H200"]


def test_rate_overrides():
    cfg = AllocConfig(
        fleet=[FleetEntry(gpu="H100", rate=3.50), FleetEntry(gpu="A100")],
        explore=[FleetEntry(gpu="H200", rate=5.00, explore=True)],
    )
    overrides = cfg.rate_overrides
    assert overrides == {"H100": 3.50, "H200": 5.00}
    assert "A100" not in overrides


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_valid():
    cfg = AllocConfig(
        fleet=[FleetEntry(gpu="H100", count=4, rate=3.50)],
        priority_cost=70,
        budget_monthly=5000,
    )
    errors = validate_config(cfg)
    assert errors == []


def test_validate_bad_priority():
    cfg = AllocConfig(priority_cost=150)
    errors = validate_config(cfg)
    assert any("priority" in e for e in errors)


def test_validate_negative_budget():
    cfg = AllocConfig(budget_monthly=-100)
    errors = validate_config(cfg)
    assert any("budget" in e for e in errors)


def test_validate_negative_rate():
    cfg = AllocConfig(fleet=[FleetEntry(gpu="H100", rate=-1.0)])
    errors = validate_config(cfg)
    assert any("Rate" in e for e in errors)


def test_validate_zero_count():
    cfg = AllocConfig(fleet=[FleetEntry(gpu="H100", count=0)])
    errors = validate_config(cfg)
    assert any("Count" in e for e in errors)


def test_validate_empty_gpu():
    cfg = AllocConfig(fleet=[FleetEntry(gpu="")])
    errors = validate_config(cfg)
    assert any("gpu" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Write / Read round-trip
# ---------------------------------------------------------------------------

def test_write_and_read():
    """Write config to file, read it back, verify round-trip."""
    original = AllocConfig(
        fleet=[
            FleetEntry(gpu="H100", cloud="aws", count=8, rate=3.50),
            FleetEntry(gpu="A100"),
        ],
        explore=[FleetEntry(gpu="H200", explore=True)],
        priority_cost=70,
        budget_monthly=5000,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, ".alloc.yaml")
        written_path = write_alloc_config(original, path=path)
        assert written_path == path
        assert os.path.isfile(path)

        loaded = load_alloc_config(path=path)
        assert loaded is not None
        assert len(loaded.fleet) == 2
        assert loaded.fleet[0].gpu == "H100"
        assert loaded.fleet[0].rate == 3.50
        assert loaded.fleet[0].count == 8
        assert loaded.fleet[0].cloud == "aws"
        assert loaded.fleet[1].gpu == "A100"
        assert len(loaded.explore) == 1
        assert loaded.explore[0].gpu == "H200"
        assert loaded.priority_cost == 70
        assert loaded.budget_monthly == 5000


def test_load_nonexistent():
    """Loading from a nonexistent path returns None."""
    result = load_alloc_config(path="/tmp/nonexistent_alloc_config_xyz.yaml")
    assert result is None


def test_load_invalid_yaml():
    """Malformed YAML returns None."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("{{{{invalid yaml")
        path = f.name
    try:
        result = load_alloc_config(path=path)
        assert result is None
    finally:
        os.unlink(path)


def test_load_non_dict_yaml():
    """YAML that parses to a non-dict returns None."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("- just\n- a\n- list\n")
        path = f.name
    try:
        result = load_alloc_config(path=path)
        assert result is None
    finally:
        os.unlink(path)


def test_to_dict():
    """to_dict produces the expected YAML-serializable structure."""
    cfg = AllocConfig(
        fleet=[FleetEntry(gpu="H100", rate=3.50)],
        priority_cost=60,
        budget_monthly=1000,
    )
    d = cfg.to_dict()
    assert d["fleet"] == [{"gpu": "H100", "rate": 3.50}]
    assert d["priority"] == {"cost": 60, "latency": 40}
    assert d["budget"]["monthly_usd"] == 1000
