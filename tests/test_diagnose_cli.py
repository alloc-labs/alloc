"""Integration tests for the `alloc diagnose` CLI command."""

import gzip
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

def _write_artifact(path, data):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


SAMPLE_ARTIFACT = {
    "version": "0.0.1",
    "probe": {
        "peak_vram_mb": 35000.0,
        "avg_gpu_util": 72.3,
        "duration_seconds": 24.1,
        "exit_code": 0,
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_vram_mb": 40960.0,
        "step_time_ms_p50": 148.5,
        "step_time_ms_p90": 162.3,
        "samples_per_sec": 42.3,
        "dataloader_wait_pct": 38.0,
        "per_rank_peak_vram_mb": [35000.0],
        "command": "python train.py",
        "samples": [
            {"t": 0.0, "vram_mb": 28000, "gpu_util_pct": 65.0, "power_w": 250},
            {"t": 1.0, "vram_mb": 35000, "gpu_util_pct": 80.0, "power_w": 300},
        ],
    },
    "hardware": {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_vram_mb": 40960.0,
        "sm_version": "8.0",
        "num_gpus_detected": 1,
    },
    "context": None,
}


def test_diagnose_help():
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--diff" in out
    assert "--json" in out
    assert "--verbose" in out
    assert "--severity" in out
    assert "--category" in out
    # Phase 2 flags
    assert "--artifact" in out
    assert "--no-artifact" in out
    assert "--memory" in out
    assert "--efficiency" in out
    assert "--compare" in out


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


# ===========================================================================
# Phase 2: Runtime-enhanced diagnosis
# ===========================================================================

# ---------------------------------------------------------------------------
# Artifact auto-discovery
# ---------------------------------------------------------------------------

def test_diagnose_auto_discovers_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["version"] == "2"
    assert data["tier"] == "runtime"
    # Should have runtime-enhanced findings
    rule_ids = [f["rule_id"] for f in data["findings"]]
    assert "DL005" in rule_ids  # DataLoader issue detected
    assert "MEM001" in rule_ids  # VRAM-aware rule fires


def test_diagnose_no_artifact_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config", "--no-artifact"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["tier"] == "ast_only"


def test_diagnose_explicit_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    artifact_path = str(tmp_path / "custom_artifact.json.gz")
    _write_artifact(artifact_path, SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config", "--artifact", artifact_path])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["tier"] == "runtime"
    assert data["artifact"] == artifact_path


# ---------------------------------------------------------------------------
# --json with Phase 2 fields
# ---------------------------------------------------------------------------

def test_diagnose_json_phase2_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "tier" in data
    assert "version" in data
    assert data["version"] == "2"
    # Efficiency should be present (artifact has step timing)
    assert "efficiency" in data


# ---------------------------------------------------------------------------
# --memory
# ---------------------------------------------------------------------------

def test_diagnose_memory_with_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--memory", "--no-config"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "vram" in out.lower() or "breakdown" in out.lower()


def test_diagnose_memory_without_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        import torch
    """)
    result = runner.invoke(app, ["diagnose", script, "--memory", "--no-config", "--no-artifact"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "requires" in out.lower()


# ---------------------------------------------------------------------------
# --efficiency
# ---------------------------------------------------------------------------

def test_diagnose_efficiency_with_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4)
    """)
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), SAMPLE_ARTIFACT)
    result = runner.invoke(app, ["diagnose", script, "--efficiency", "--no-config"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "efficiency" in out.lower() or "compute" in out.lower() or "step time" in out.lower()


def test_diagnose_efficiency_without_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        import torch
    """)
    result = runner.invoke(app, ["diagnose", script, "--efficiency", "--no-config", "--no-artifact"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "requires" in out.lower()


# ---------------------------------------------------------------------------
# --compare
# ---------------------------------------------------------------------------

def test_diagnose_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        import torch
    """)
    # Current artifact (slower)
    current = dict(SAMPLE_ARTIFACT)
    current["probe"] = dict(SAMPLE_ARTIFACT["probe"])
    current["probe"]["step_time_ms_p50"] = 200.0
    current["probe"]["samples_per_sec"] = 30.0
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), current)

    # Previous artifact (faster)
    prev_path = str(tmp_path / "prev_artifact.json.gz")
    _write_artifact(prev_path, SAMPLE_ARTIFACT)

    result = runner.invoke(app, [
        "diagnose", script, "--json", "--no-config",
        "--compare", prev_path,
    ])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "comparison" in data
    assert data["comparison"]["has_regressions"] is True
    # REG001 should fire
    rule_ids = [f["rule_id"] for f in data["findings"]]
    assert "REG001" in rule_ids


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------

def test_diagnose_tier_ast_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        import torch
    """)
    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config", "--no-artifact"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["tier"] == "ast_only"


def test_diagnose_tier_runtime_distributed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _write_script(tmp_path, """
        import torch
    """)
    dist_artifact = dict(SAMPLE_ARTIFACT)
    dist_artifact["probe"] = dict(SAMPLE_ARTIFACT["probe"])
    dist_artifact["probe"]["per_rank_peak_vram_mb"] = [31000.0, 31200.0, 31100.0, 31300.0]
    dist_artifact["hardware"] = dict(SAMPLE_ARTIFACT["hardware"])
    dist_artifact["hardware"]["num_gpus_detected"] = 4
    _write_artifact(str(tmp_path / "alloc_artifact.json.gz"), dist_artifact)

    result = runner.invoke(app, ["diagnose", script, "--json", "--no-config"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["tier"] == "runtime_distributed"


# ---------------------------------------------------------------------------
# --rules flag
# ---------------------------------------------------------------------------

def test_diagnose_rules_flag():
    result = runner.invoke(app, ["diagnose", "--rules"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "DL001" in out
    assert "MEM001" in out
    assert "DIST002" in out
    assert "rules" in out.lower()


def test_diagnose_rules_json():
    result = runner.invoke(app, ["diagnose", "--rules", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "rules" in data
    assert data["total"] > 0
    ids = [r["id"] for r in data["rules"]]
    assert "DL001" in ids
    assert "REG001" in ids
    # Each rule has required fields
    for r in data["rules"]:
        assert "id" in r
        assert "title" in r
        assert "category" in r
        assert "tier" in r


def test_diagnose_no_script_no_rules():
    """Without --rules and without a script, should error."""
    result = runner.invoke(app, ["diagnose"])
    assert result.exit_code != 0
    out = _plain(result.output)
    assert "no script" in out.lower() or "usage" in out.lower()
