"""Tests for CLI commands."""

import json as json_mod
import os
import re

from typer.testing import CliRunner
from alloc.cli import app, _read_callback_data

runner = CliRunner()

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return _ansi_re.sub("", text)


def test_version():
    """alloc version should print version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "alloc v" in result.output


def test_help():
    """alloc --help should show available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ghost" in out
    assert "run" in out
    assert "scan" in out
    assert "version" in out


def test_ghost_help():
    """alloc ghost --help should show options."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "script" in out.lower() or "SCRIPT" in out


def test_scan_help():
    """alloc scan --help should show options."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--model" in out
    assert "--gpu" in out


def test_run_help_shows_full():
    """alloc run --help should show --full option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--full" in _plain(result.output)


def test_run_help_hides_probe_steps():
    """alloc run --help should NOT show deprecated --probe-steps."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--probe-steps" not in _plain(result.output)


# --- --json flag tests ---

def test_ghost_json_flag_in_help():
    """alloc ghost --help should show --json option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_run_json_flag_in_help():
    """alloc run --help should show --json option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_scan_json_flag_in_help():
    """alloc scan --help should show --json option."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--json" in _plain(result.output)


def test_ghost_json_output():
    """alloc ghost --json should output valid JSON on error (unknown script)."""
    import json
    result = runner.invoke(app, ["ghost", "unknown_model.py", "--json"])
    # exits with code 1 since model can't be extracted, but should output JSON
    assert result.exit_code == 1
    output = result.output.strip()
    data = json.loads(output)
    assert "error" in data


def test_ghost_json_output_param_count():
    """alloc ghost --json with --param-count-b should output valid JSON."""
    import json
    result = runner.invoke(app, ["ghost", "train.py", "--json", "--param-count-b", "7.0"])
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
    assert "--verbose" in _plain(result.output)


def test_run_verbose_flag_in_help():
    """alloc run --help should show --verbose option."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in _plain(result.output)


def test_ghost_verbose_output():
    """alloc ghost --verbose should show VRAM formula breakdown."""
    result = runner.invoke(app, ["ghost", "train.py", "--verbose", "--param-count-b", "7.0"])
    assert result.exit_code == 0
    assert "VRAM Formula Breakdown" in result.output or "bytes/param" in result.output


def test_ghost_verbose_and_json_mutual():
    """--json should suppress verbose output (JSON mode wins)."""
    import json
    result = runner.invoke(app, ["ghost", "train.py", "--json", "--verbose", "--param-count-b", "7.0"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "total_gb" in data


def test_ghost_param_count_flag_in_help():
    """alloc ghost --help should show --param-count-b option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--param-count-b" in _plain(result.output)


def test_ghost_timeout_flag_in_help():
    """alloc ghost --help should show --timeout option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--timeout" in _plain(result.output)


# --- Callback sidecar reading tests ---

def test_read_callback_data_new_format(tmp_path, monkeypatch):
    """_read_callback_data reads .alloc_callback.json with timing fields."""
    monkeypatch.chdir(tmp_path)
    data = {
        "framework": "huggingface",
        "step_count": 200,
        "step_time_ms_p50": 148.5,
        "step_time_ms_p90": 152.1,
        "samples_per_sec": 42.3,
        "dataloader_wait_pct": 5.0,
    }
    (tmp_path / ".alloc_callback.json").write_text(json_mod.dumps(data))
    result = _read_callback_data()
    assert result is not None
    assert result["step_count"] == 200
    assert result["step_time_ms_p50"] == 148.5
    assert result["samples_per_sec"] == 42.3


def test_read_callback_data_legacy_file_ignored(tmp_path, monkeypatch):
    """Legacy .alloc_steps.json is ignored."""
    monkeypatch.chdir(tmp_path)
    data = {"framework": "huggingface", "step_count": 42}
    (tmp_path / ".alloc_steps.json").write_text(json_mod.dumps(data))
    result = _read_callback_data()
    assert result is None


def test_read_callback_data_prefers_new_format(tmp_path, monkeypatch):
    """When both sidecar files exist, .alloc_callback.json is preferred."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".alloc_steps.json").write_text(json_mod.dumps({"step_count": 10}))
    (tmp_path / ".alloc_callback.json").write_text(
        json_mod.dumps({"step_count": 99, "step_time_ms_p50": 100.0})
    )
    result = _read_callback_data()
    assert result["step_count"] == 99


def test_read_callback_data_returns_none(tmp_path, monkeypatch):
    """No sidecar files → None."""
    monkeypatch.chdir(tmp_path)
    assert _read_callback_data() is None


# ---------------------------------------------------------------------------
# alloc status tests
# ---------------------------------------------------------------------------


def test_status_help():
    """alloc status --help should show options."""
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--json" in out


def test_status_no_artifact(tmp_path, monkeypatch):
    """alloc status in empty dir shows no artifact message."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "No artifact found" in out
    assert "alloc v" in out


def test_status_json_no_artifact(tmp_path, monkeypatch):
    """alloc status --json in empty dir returns valid JSON with null artifact."""
    import json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["artifact"] is None
    assert data["logged_in"] is False
    assert "version" in data


def test_status_with_artifact(tmp_path, monkeypatch):
    """alloc status finds and displays artifact summary."""
    import gzip
    import json as _json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)

    artifact = {
        "probe": {
            "avg_gpu_util": 87.5,
            "peak_vram_mb": 41000,
            "exit_code": 0,
            "duration_seconds": 125.4,
            "command": "python train.py --epochs 10",
            "samples": [
                {"gpu_util_pct": 85, "power_w": 300},
                {"gpu_util_pct": 90, "power_w": 310},
            ],
        },
        "hardware": {
            "gpu_name": "A100-80GB",
            "num_gpus_detected": 2,
        },
    }
    art_path = tmp_path / "alloc_artifact.json.gz"
    with gzip.open(str(art_path), "wt", encoding="utf-8") as f:
        _json.dump(artifact, f)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "A100-80GB" in out
    assert "2x" in out
    assert "python train.py" in out


def test_status_json_with_artifact(tmp_path, monkeypatch):
    """alloc status --json with artifact returns structured data."""
    import gzip
    import json as _json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)

    artifact = {
        "probe": {
            "avg_gpu_util": 95,
            "peak_vram_mb": 70000,
            "exit_code": 0,
            "duration_seconds": 300.0,
            "samples": [],
        },
        "hardware": {
            "gpu_name": "H100-80GB",
            "num_gpus_detected": 8,
        },
    }
    art_path = tmp_path / "alloc_artifact.json.gz"
    with gzip.open(str(art_path), "wt", encoding="utf-8") as f:
        _json.dump(artifact, f)

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = _json.loads(result.output.strip())
    assert data["artifact"] is not None
    assert data["artifact"]["gpu"] == "H100-80GB"
    assert data["artifact"]["gpu_count"] == 8
    assert data["artifact"]["exit_code"] == 0


def test_status_logged_in(tmp_path, monkeypatch):
    """alloc status with token shows logged-in state."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALLOC_TOKEN", "test-token-123")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "Logged in" in out


def test_status_not_logged_in(tmp_path, monkeypatch):
    """alloc status without token shows not-logged-in state."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "Not logged in" in out
