"""Tests for CLI commands."""

from typer.testing import CliRunner
from alloc.cli import app

runner = CliRunner()


def test_version():
    """alloc version should print version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "alloc v" in result.output


def test_help():
    """alloc --help should show available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ghost" in result.output
    assert "run" in result.output
    assert "scan" in result.output
    assert "version" in result.output


def test_ghost_help():
    """alloc ghost --help should show options."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "script" in result.output.lower() or "SCRIPT" in result.output


def test_scan_help():
    """alloc scan --help should show options."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--gpu" in result.output


def test_run_help_shows_full():
    """alloc run --help should show --full option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--full" in result.output


def test_run_help_hides_probe_steps():
    """alloc run --help should NOT show deprecated --probe-steps."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--probe-steps" not in result.output


# --- --json flag tests ---

def test_ghost_json_flag_in_help():
    """alloc ghost --help should show --json option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_run_json_flag_in_help():
    """alloc run --help should show --json option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_scan_json_flag_in_help():
    """alloc scan --help should show --json option."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_ghost_json_output():
    """alloc ghost --json should output valid JSON on error (unknown script)."""
    import json
    result = runner.invoke(app, ["ghost", "unknown_model.py", "--json"])
    # exits with code 1 since model can't be extracted, but should output JSON
    assert result.exit_code == 1
    output = result.output.strip()
    data = json.loads(output)
    assert "error" in data


def test_ghost_json_output_7b():
    """alloc ghost --json should output valid JSON for a known model size."""
    import json
    result = runner.invoke(app, ["ghost", "train_7b.py", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "total_gb" in data
    assert "weights_gb" in data
    assert "param_count_b" in data
    assert data["total_gb"] > 0


# --- --verbose flag tests ---

def test_ghost_verbose_flag_in_help():
    """alloc ghost --help should show --verbose option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output


def test_run_verbose_flag_in_help():
    """alloc run --help should show --verbose option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output


def test_ghost_verbose_output():
    """alloc ghost --verbose should show VRAM formula breakdown."""
    result = runner.invoke(app, ["ghost", "train_7b.py", "--verbose"])
    assert result.exit_code == 0
    assert "VRAM Formula Breakdown" in result.output or "bytes/param" in result.output


def test_ghost_verbose_and_json_mutual():
    """--json should suppress verbose output (JSON mode wins)."""
    import json
    result = runner.invoke(app, ["ghost", "train_7b.py", "--json", "--verbose"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "total_gb" in data
