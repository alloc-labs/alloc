"""Integration tests for the `alloc diagnose` CLI command."""

import json
import os
import re
import textwrap

from typer.testing import CliRunner
from alloc.cli import app

runner = CliRunner()

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    return _ansi_re.sub("", text)


def _write_script(tmp_path, code):
    p = tmp_path / "train.py"
    p.write_text(textwrap.dedent(code))
    return str(p)


# ---------------------------------------------------------------------------
# Basic command structure
# ---------------------------------------------------------------------------

def test_diagnose_help():
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--diff" in out
    assert "--json" in out
    assert "--verbose" in out
    assert "--severity" in out
    assert "--category" in out


def test_diagnose_missing_file():
    result = runner.invoke(app, ["diagnose", "nonexistent.py"])
    assert result.exit_code == 1
    assert "not found" in _plain(result.output).lower()


def test_diagnose_non_python():
    result = runner.invoke(app, ["diagnose", "script.sh"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Default output (Rich terminal)
# ---------------------------------------------------------------------------

def test_diagnose_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--no-config"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "DL005" in out or "finding" in out.lower()


def test_diagnose_clean_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        x = 1 + 2
    """)
    result = runner.invoke(app, ["diagnose", script, "--no-config"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "no issues" in out.lower() or "0" in out


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------

def test_diagnose_json_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "version" in data
    assert "findings" in data
    assert "summary" in data
    assert data["summary"]["total"] >= 1


def test_diagnose_json_valid_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=2)
    """)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    # Check finding schema
    if data["findings"]:
        f = data["findings"][0]
        assert "rule_id" in f
        assert "severity" in f
        assert "category" in f
        assert "title" in f
        assert "rationale" in f
        assert "confidence" in f


def test_diagnose_json_missing_file():
    result = runner.invoke(app, ["diagnose", "nope.py", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output.strip())
    assert "error" in data


# ---------------------------------------------------------------------------
# --diff output
# ---------------------------------------------------------------------------

def test_diagnose_diff_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--diff", "--no-config"])
    assert result.exit_code == 0
    out = result.output
    # Should contain diff markers or "no patches" message
    assert "---" in out or "No code patches" in out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_diagnose_severity_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--json", "--severity", "critical", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    for f in data["findings"]:
        assert f["severity"] == "critical"


def test_diagnose_category_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--json", "--category", "dataloader", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    for f in data["findings"]:
        assert f["category"] == "dataloader"


# ---------------------------------------------------------------------------
# --verbose
# ---------------------------------------------------------------------------

def test_diagnose_verbose(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["diagnose", script, "--verbose", "--no-config"])
    assert result.exit_code == 0
    out = _plain(result.output)
    # Verbose should show confidence and doc references
    assert "confidence" in out.lower() or "ref" in out.lower() or "pytorch" in out.lower()
