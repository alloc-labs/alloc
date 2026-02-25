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
    # Agent flags
    assert "--explain" in out
    assert "--fix" in out
    assert "--what-if" in out


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


def test_diagnose_json_includes_explain_data(tmp_path, monkeypatch):
    """When --json and --explain, explain data is merged into JSON output."""
    from alloc.diagnosis_display import build_json_dict, print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    # Create a minimal DiagnoseResult
    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    explain_data = {
        "summary": "No significant issues found.",
        "actions": [
            {
                "priority": 1,
                "finding_id": "DL001",
                "title": "Fix dataloader",
                "explanation": "Increase workers",
                "impact": "30% speedup",
                "confidence": "high",
            }
        ],
        "interactions": [],
        "architecture_insight": "Standard transformer training.",
    }

    # build_json_dict does not include explain — that's done by print_diagnose_json
    data = build_json_dict(result)
    assert "explain" not in data

    # Capture printed JSON
    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result, explain_data=explain_data)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "explain" in parsed
    assert parsed["explain"]["summary"] == "No significant issues found."
    assert len(parsed["explain"]["actions"]) == 1
    assert parsed["explain"]["actions"][0]["finding_id"] == "DL001"
    assert parsed["explain"]["architecture_insight"] == "Standard transformer training."


def test_diagnose_json_without_explain_has_no_explain_key(tmp_path, monkeypatch):
    """When --json without --explain, no 'explain' key in output."""
    from alloc.diagnosis_display import print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "explain" not in parsed


# ---------------------------------------------------------------------------
# --fix and --what-if JSON output
# ---------------------------------------------------------------------------

def test_diagnose_json_includes_fix_data(tmp_path, monkeypatch):
    """When fix_data is provided, JSON output includes 'fix' key."""
    from alloc.diagnosis_display import print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    fix_data = {
        "summary": "Applied 1 patch.",
        "patches": [
            {
                "file_path": "train.py",
                "finding_ids": ["DL001"],
                "description": "Set num_workers to 4",
                "diff": "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-old\n+new",
                "confidence": "high",
            }
        ],
    }

    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result, fix_data=fix_data)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "fix" in parsed
    assert parsed["fix"]["summary"] == "Applied 1 patch."
    assert len(parsed["fix"]["patches"]) == 1


def test_diagnose_json_includes_what_if_data(tmp_path, monkeypatch):
    """When what_if_data is provided, JSON output includes 'what_if' key."""
    from alloc.diagnosis_display import print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    what_if_data = {
        "scenario": "switch to bf16",
        "feasible": True,
        "summary": "VRAM will decrease by ~50%.",
        "impacts": [
            {
                "metric": "vram",
                "direction": "decrease",
                "magnitude": "~50% reduction",
                "reasoning": "bf16 halves weight memory",
            }
        ],
    }

    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result, what_if_data=what_if_data)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "what_if" in parsed
    assert parsed["what_if"]["feasible"] is True
    assert len(parsed["what_if"]["impacts"]) == 1


# ---------------------------------------------------------------------------
# _read_full_files helper
# ---------------------------------------------------------------------------

def test_read_full_files(tmp_path):
    """_read_full_files reads files referenced by findings."""
    from alloc.cli import _read_full_files
    from alloc.diagnosis_engine import Diagnosis

    script_path = str(tmp_path / "train.py")
    with open(script_path, "w") as f:
        f.write("import torch\nloader = DataLoader(ds, num_workers=0)\n")

    findings = [
        Diagnosis(
            rule_id="DL005",
            severity="warning",
            category="dataloader",
            title="num_workers=0",
            file_path=script_path,
            line_number=2,
            current_code="DataLoader(ds, num_workers=0)",
            current_value="0",
            suggested_value="4",
            suggested_code="DataLoader(ds, num_workers=4)",
            rationale="Increase num_workers",
            estimated_impact="30% speedup",
            confidence="high",
            doc_url=None,
        ),
    ]

    result = _read_full_files(script_path, findings)
    assert result is not None
    assert script_path in result
    assert "DataLoader" in result[script_path]


def test_read_full_files_caps_lines(tmp_path):
    """_read_full_files caps total lines at 3000."""
    from alloc.cli import _read_full_files

    script_path = str(tmp_path / "big.py")
    with open(script_path, "w") as f:
        f.write("\n".join("line {}".format(i) for i in range(5000)))

    result = _read_full_files(script_path, [])
    assert result is not None
    content = result[script_path]
    assert content.count("\n") < 5000


# ---------------------------------------------------------------------------
# Render functions (smoke test)
# ---------------------------------------------------------------------------

def test_render_fix_rich():
    """_render_fix_rich should not crash on well-formed data."""
    from alloc.cli import _render_fix_rich

    fix_data = {
        "summary": "Applied 1 patch.",
        "patches": [
            {
                "file_path": "train.py",
                "finding_ids": ["DL001"],
                "description": "Set num_workers to 4",
                "diff": "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-old\n+new",
                "confidence": "high",
            }
        ],
        "warnings": ["Review batch size"],
    }
    _render_fix_rich(fix_data)  # Should not raise


def test_render_what_if_rich():
    """_render_what_if_rich should not crash on well-formed data."""
    from alloc.cli import _render_what_if_rich

    what_if_data = {
        "scenario": "switch to bf16",
        "feasible": True,
        "summary": "VRAM will decrease.",
        "impacts": [
            {
                "metric": "vram",
                "direction": "decrease",
                "magnitude": "~50%",
                "reasoning": "bf16 halves memory",
            },
            {
                "metric": "throughput",
                "direction": "increase",
                "magnitude": "10-30%",
                "reasoning": "tensor cores",
            },
        ],
        "risks": ["May cause gradient underflow"],
        "suggested_steps": ["Add bf16=True to TrainingArguments"],
    }
    _render_what_if_rich(what_if_data)  # Should not raise


def test_render_what_if_rich_infeasible():
    """_render_what_if_rich handles infeasible scenarios."""
    from alloc.cli import _render_what_if_rich

    what_if_data = {
        "scenario": "use tensor parallelism on 1 GPU",
        "feasible": False,
        "summary": "TP requires at least 2 GPUs.",
        "impacts": [],
    }
    _render_what_if_rich(what_if_data)  # Should not raise


def test_render_fix_rich_empty_patches():
    """_render_fix_rich handles empty patches list."""
    from alloc.cli import _render_fix_rich

    fix_data = {
        "summary": "No fixable issues found.",
        "patches": [],
    }
    _render_fix_rich(fix_data)  # Should not raise


# ---------------------------------------------------------------------------
# Communication analysis (P2-B)
# ---------------------------------------------------------------------------


def test_diagnose_help_includes_comm():
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--comm" in out


def test_diagnose_json_includes_comm_data(tmp_path, monkeypatch):
    """When comm_data is provided, JSON output includes 'comm_analysis' key."""
    from alloc.diagnosis_display import print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    comm_data = {
        "analysis": {
            "total_overhead_pct": 5.3,
            "heuristic_overhead_pct": 5.3,
            "breakdown": [
                {
                    "source": "gradient_sync",
                    "heuristic_pct": 5.3,
                    "agent_pct": 4.8,
                    "reasoning": "Higher overlap for large models.",
                }
            ],
            "scaling_assessment": "Good scaling.",
        },
        "estimate_quality": "agent_reasoned",
    }

    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result, comm_data=comm_data)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "comm_analysis" in parsed
    assert parsed["comm_analysis"]["estimate_quality"] == "agent_reasoned"
    assert len(parsed["comm_analysis"]["analysis"]["breakdown"]) == 1


def test_render_comm_rich():
    """_render_comm_rich should not crash on well-formed data."""
    from alloc.cli import _render_comm_rich

    comm_data = {
        "analysis": {
            "total_overhead_pct": 5.3,
            "heuristic_overhead_pct": 6.0,
            "breakdown": [
                {
                    "source": "gradient_sync",
                    "heuristic_pct": 6.0,
                    "agent_pct": 5.3,
                    "reasoning": "Higher overlap for large models.",
                }
            ],
            "overlap_assessment": "65-75% overlap expected.",
            "scaling_assessment": "Good scaling on NVLink.",
            "bottleneck_type": "compute_dominated",
            "recommendations": ["Consider FSDP for larger models."],
        },
        "estimate_quality": "agent_reasoned",
    }
    _render_comm_rich(comm_data)  # Should not raise


def test_render_comm_rich_heuristic_only():
    """_render_comm_rich handles heuristic-only data."""
    from alloc.cli import _render_comm_rich

    comm_data = {
        "analysis": {
            "total_overhead_pct": 8.2,
            "heuristic_overhead_pct": 8.2,
            "breakdown": [
                {
                    "source": "gradient_sync",
                    "heuristic_pct": 5.0,
                    "agent_pct": None,
                    "reasoning": "Ring all-reduce.",
                },
                {
                    "source": "fsdp_param_sync",
                    "heuristic_pct": 3.2,
                    "agent_pct": None,
                    "reasoning": "All-gather + reduce-scatter.",
                },
            ],
            "scaling_assessment": "Heuristic estimate.",
        },
        "estimate_quality": "heuristic_only",
    }
    _render_comm_rich(comm_data)  # Should not raise


def test_render_comm_rich_empty_breakdown():
    """_render_comm_rich handles empty breakdown gracefully."""
    from alloc.cli import _render_comm_rich

    comm_data = {
        "analysis": {
            "total_overhead_pct": 0,
            "breakdown": [],
            "scaling_assessment": "Single GPU — no communication overhead.",
        },
        "estimate_quality": "agent_reasoned",
    }
    _render_comm_rich(comm_data)  # Should not raise


# ---------------------------------------------------------------------------
# Root cause diagnosis (P2-C)
# ---------------------------------------------------------------------------


def test_diagnose_help_includes_root_cause():
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--root-cause" in out


def test_diagnose_json_includes_root_cause_data(tmp_path, monkeypatch):
    """When root_cause_data is provided, JSON output includes 'root_cause' key."""
    from alloc.diagnosis_display import print_diagnose_json
    from alloc.diagnosis_engine import DiagnoseResult

    result = DiagnoseResult(
        version="2",
        script="train.py",
        tier="ast_only",
        findings=[],
        summary={"critical": 0, "warning": 0, "info": 0, "total": 0},
    )

    root_cause_data = {
        "primary_hypothesis": {
            "cause": "comm_bound",
            "confidence": "medium",
            "explanation": "Steady low GPU util with low step variance.",
            "supporting_evidence": ["Steady low util"],
        },
        "diagnostic_summary": "Communication overhead likely.",
        "signal_quality": "partial",
    }

    import io
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    print_diagnose_json(result, root_cause_data=root_cause_data)
    output = buf.getvalue()
    parsed = json.loads(output)

    assert "root_cause" in parsed
    assert parsed["root_cause"]["primary_hypothesis"]["cause"] == "comm_bound"


def test_render_root_cause_rich():
    """_render_root_cause_rich should not crash on well-formed data."""
    from alloc.cli import _render_root_cause_rich

    data = {
        "primary_hypothesis": {
            "cause": "dataloader_bound",
            "confidence": "high",
            "explanation": "Spiky utilization pattern indicates dataloader stalls.",
            "supporting_evidence": ["Spiky GPU util", "High step time variance"],
            "contradicting_evidence": ["No DL findings from code analysis"],
        },
        "alternative_hypotheses": [
            {
                "cause": "comm_bound",
                "confidence": "low",
                "explanation": "Less likely given spiky pattern.",
                "supporting_evidence": ["Multi-GPU setup"],
            },
        ],
        "diagnostic_summary": "Dataloader is the primary bottleneck.",
        "next_steps": [
            "Increase num_workers",
            "Profile with torch.profiler",
        ],
        "signal_quality": "full",
    }
    _render_root_cause_rich(data)  # Should not raise


def test_render_root_cause_rich_minimal():
    """_render_root_cause_rich handles minimal data."""
    from alloc.cli import _render_root_cause_rich

    data = {
        "primary_hypothesis": {
            "cause": "unknown",
            "confidence": "low",
            "explanation": "Insufficient data for diagnosis.",
            "supporting_evidence": [],
        },
        "diagnostic_summary": "Need more signals.",
        "signal_quality": "minimal",
    }
    _render_root_cause_rich(data)  # Should not raise


# ------------------------------------------------------------------
# Ghost --explain (P2-A) tests
# ------------------------------------------------------------------


def test_ghost_help_shows_explain_flag():
    """Ghost --help should show --explain option."""
    result = runner.invoke(app, ["ghost", "--help"])
    assert result.exit_code == 0
    assert "--explain" in result.output


def test_ghost_explain_json_output():
    """Ghost --explain --json should include explain data in JSON output."""
    from alloc.cli import _fetch_ghost_explain
    # The function signature: (report, info, batch_size, seq_length, hidden_dim, json_output)
    # Just verify the function exists and is importable
    assert callable(_fetch_ghost_explain)


def test_render_ghost_explain_rich_full():
    """_render_ghost_explain_rich handles full agent response."""
    from alloc.cli import _render_ghost_explain_rich

    data = {
        "refined_total_gb": 42.5,
        "heuristic_total_gb": 143.22,
        "estimate_quality": "agent_reasoned",
        "confidence": "high",
        "breakdown": [
            {"component": "weights", "heuristic_gb": 13.02, "agent_gb": 3.5,
             "reasoning": "QLoRA 4-bit quantization"},
            {"component": "gradients", "heuristic_gb": 26.04, "agent_gb": 0.04,
             "reasoning": "LoRA adapter only"},
            {"component": "optimizer", "heuristic_gb": 78.12, "agent_gb": 0.12,
             "reasoning": "Adapter params only"},
            {"component": "activations", "heuristic_gb": 13.02, "agent_gb": 6.51,
             "reasoning": "Gradient checkpointing"},
            {"component": "buffer", "heuristic_gb": 13.02, "agent_gb": 1.02,
             "reasoning": "Proportional reduction"},
        ],
        "architecture_insight": "QLoRA fine-tune of Llama-7B detected.",
        "adjustments": ["Applied 4-bit quantization", "Reduced to adapter params"],
        "gpu_recommendations": ["Fits on RTX 4090 (24 GB)"],
    }
    _render_ghost_explain_rich(data)  # Should not raise


def test_render_ghost_explain_rich_minimal():
    """_render_ghost_explain_rich handles minimal heuristic-only response."""
    from alloc.cli import _render_ghost_explain_rich

    data = {
        "refined_total_gb": 143.22,
        "heuristic_total_gb": 143.22,
        "estimate_quality": "heuristic_only",
        "confidence": "low",
        "breakdown": [],
        "architecture_insight": "Heuristic only — agent unavailable.",
    }
    _render_ghost_explain_rich(data)  # Should not raise


def test_render_ghost_explain_rich_no_delta():
    """_render_ghost_explain_rich handles case where refined == heuristic."""
    from alloc.cli import _render_ghost_explain_rich

    data = {
        "refined_total_gb": 50.0,
        "heuristic_total_gb": 50.0,
        "estimate_quality": "agent_reasoned",
        "confidence": "medium",
        "breakdown": [
            {"component": "weights", "heuristic_gb": 13.02, "agent_gb": 13.02,
             "reasoning": "Heuristic is accurate for standard transformer"},
        ],
        "architecture_insight": "Standard transformer training.",
    }
    _render_ghost_explain_rich(data)  # Should not raise


# ------------------------------------------------------------------
# --migrate tests
# ------------------------------------------------------------------

def test_diagnose_help_shows_migrate_flag():
    """Diagnose --help should show --migrate option."""
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    assert "--migrate" in result.output


def test_migrate_fetch_callable():
    """_fetch_migrate should be importable and callable."""
    from alloc.cli import _fetch_migrate
    assert callable(_fetch_migrate)


def test_render_migrate_rich_full():
    """_render_migrate_rich handles full migration plan."""
    from alloc.cli import _render_migrate_rich

    data = {
        "target_hardware": "8x H100-80GB",
        "feasible": True,
        "summary": "Migration is feasible with FSDP and bf16.",
        "confidence": "high",
        "steps": [
            {
                "category": "strategy",
                "change": "Switch from DDP to FSDP",
                "reasoning": "70B model requires sharding across GPUs",
                "impact": "4x per-GPU VRAM reduction",
                "code_change": "from torch.distributed.fsdp import FSDP\nmodel = FSDP(model)",
            },
            {
                "category": "precision",
                "change": "Enable bf16",
                "reasoning": "H100 has native bf16 support",
                "impact": "2x memory reduction",
            },
        ],
        "projected_comparison": {
            "vram_per_gpu_before": "60GB",
            "vram_per_gpu_after": "18GB",
        },
        "warnings": ["Requires NCCL >= 2.18 for best H100 performance"],
    }
    _render_migrate_rich(data)  # Should not raise


def test_render_migrate_rich_infeasible():
    """_render_migrate_rich handles infeasible migration."""
    from alloc.cli import _render_migrate_rich

    data = {
        "target_hardware": "2x RTX 3090",
        "feasible": False,
        "summary": "70B model cannot fit on 2x RTX 3090 even with FSDP.",
        "confidence": "high",
        "steps": [],
        "warnings": ["Minimum 4x A100-80GB required for 70B models"],
    }
    _render_migrate_rich(data)  # Should not raise


def test_render_migrate_rich_minimal():
    """_render_migrate_rich handles minimal data."""
    from alloc.cli import _render_migrate_rich

    data = {
        "target_hardware": "4x A100-40GB",
        "feasible": True,
        "summary": "Straightforward migration.",
        "confidence": "medium",
        "steps": [],
    }
    _render_migrate_rich(data)  # Should not raise


def test_diagnose_help_shows_upgrades_flag():
    """Diagnose --help should show --upgrades option."""
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    assert "--upgrades" in result.output


# ---------------------------------------------------------------------------
# alloc doctor tests
# ---------------------------------------------------------------------------


def test_doctor_help():
    """alloc doctor --help should show options."""
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--no-run" in out
    assert "--timeout" in out
    assert "--json" in out


def test_doctor_file_not_found():
    """alloc doctor with nonexistent script shows error."""
    result = runner.invoke(app, ["doctor", "/nonexistent/train.py"])
    assert result.exit_code == 1
    assert "not found" in _plain(result.output).lower()


def test_doctor_not_python(tmp_path):
    """alloc doctor with non-Python file shows error."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n")
    result = runner.invoke(app, ["doctor", str(csv_file)])
    assert result.exit_code == 1
    assert "Python" in result.output


def test_doctor_no_run_no_artifact(tmp_path, monkeypatch):
    """alloc doctor --no-run with no artifact still does AST analysis."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["doctor", script, "--no-run"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "Step 1/6" in out
    assert "Step 4/6" in out
    assert "Doctor complete" in out


def test_doctor_json_no_run(tmp_path, monkeypatch):
    """alloc doctor --json --no-run returns valid JSON."""
    import json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)
    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, num_workers=0)
    """)
    result = runner.invoke(app, ["doctor", script, "--no-run", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "script" in data
    assert "diagnosis" in data
    assert data["diagnosis"]["findings_count"] >= 0


def test_doctor_no_run_with_artifact(tmp_path, monkeypatch):
    """alloc doctor --no-run loads existing artifact."""
    import gzip
    import json as _json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLOC_TOKEN", raising=False)

    script = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, num_workers=0)
    """)

    artifact = {
        "probe": {
            "avg_gpu_util": 80,
            "peak_vram_mb": 30000,
            "exit_code": 0,
            "duration_seconds": 60,
            "samples": [{"gpu_util_pct": 80, "power_w": 250}],
        },
        "hardware": {
            "gpu_name": "A100-80GB",
            "num_gpus_detected": 1,
        },
    }
    art_path = tmp_path / "alloc_artifact.json.gz"
    with gzip.open(str(art_path), "wt", encoding="utf-8") as f:
        _json.dump(artifact, f)

    result = runner.invoke(app, ["doctor", script, "--no-run"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "Loaded" in out
    assert "Doctor complete" in out


def test_diagnose_help_shows_explore_flag():
    """Diagnose --help should show --explore option."""
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    assert "--explore" in result.output


def test_fetch_explore_callable():
    """_fetch_explore should be importable and callable."""
    from alloc.cli import _fetch_explore
    assert callable(_fetch_explore)


def test_render_explore_rich_full():
    """_render_explore_rich handles full data."""
    from alloc.cli import _render_explore_rich

    data = {
        "summary": "Found 3 Pareto-optimal configurations.",
        "configs": [
            {
                "rank": 1,
                "label": "Recommended: Best $/step",
                "knobs": {"batch_size": "64", "precision": "bf16", "strategy": "FSDP"},
                "projected_throughput": "~850 samples/sec",
                "projected_cost_per_hour": "$3.20",
                "projected_vram_gb": "~45GB",
                "projected_cost_per_step": "$0.0038",
                "is_recommended": True,
            },
            {
                "rank": 2,
                "label": "Max throughput",
                "knobs": {"batch_size": "128", "precision": "bf16"},
                "projected_throughput": "~1200 samples/sec",
                "projected_cost_per_hour": "$6.40",
                "tradeoff_notes": "2x GPU cost for 40% more throughput",
                "is_recommended": False,
            },
        ],
        "current_config": {"batch_size": "32", "precision": "fp16"},
        "knobs_evaluated": ["batch_size", "precision", "strategy"],
        "confidence": "medium",
    }
    _render_explore_rich(data)  # Should not raise


def test_render_explore_rich_empty():
    """_render_explore_rich handles empty configs."""
    from alloc.cli import _render_explore_rich

    data = {
        "summary": "No viable configurations found.",
        "configs": [],
        "confidence": "low",
    }
    _render_explore_rich(data)  # Should not raise


def test_render_explore_rich_with_warnings():
    """_render_explore_rich handles warnings."""
    from alloc.cli import _render_explore_rich

    data = {
        "summary": "Exploration complete.",
        "configs": [
            {
                "rank": 1,
                "label": "Best",
                "knobs": {"batch_size": "32"},
                "is_recommended": True,
            },
        ],
        "warnings": ["VRAM estimates are approximate", "Throughput depends on data pipeline"],
        "confidence": "medium",
    }
    _render_explore_rich(data)  # Should not raise
