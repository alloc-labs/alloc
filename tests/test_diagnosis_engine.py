"""Tests for diagnosis_engine.py — rule evaluation and result assembly."""

import textwrap

from alloc.code_analyzer import analyze_script
from alloc.diagnosis_engine import run_diagnosis, DiagnoseResult, _deduplicate
from alloc.diagnosis_rules import Diagnosis


def _analyze(tmp_path, code):
    p = tmp_path / "train.py"
    p.write_text(textwrap.dedent(code))
    return analyze_script(str(p))


HW_4GPU = {"gpu_name": "NVIDIA A100", "gpu_count": 4, "sm_version": "8.0"}


def test_run_diagnosis_basic(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=2)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    assert isinstance(result, DiagnoseResult)
    assert result.version == "1"
    assert len(result.findings) > 0
    assert result.summary["total"] > 0


def test_run_diagnosis_empty_script(tmp_path):
    findings = _analyze(tmp_path, "# nothing\n")
    result = run_diagnosis(findings)
    assert result.findings == []
    assert result.summary["total"] == 0


def test_run_diagnosis_sorting(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
        loader2 = DataLoader(ds2, num_workers=2)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    severities = [d.severity for d in result.findings]
    # Critical should come before warning/info
    if "critical" in severities and "info" in severities:
        first_crit = severities.index("critical")
        last_info = len(severities) - 1 - severities[::-1].index("info")
        assert first_crit < last_info


def test_run_diagnosis_dedup_dl001_dl005(tmp_path):
    """DL001 and DL005 on same line: only DL005 should remain."""
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    rule_ids = [d.rule_id for d in result.findings]
    assert "DL005" in rule_ids
    assert "DL001" not in rule_ids


def test_run_diagnosis_dedup_mem002_prec(tmp_path):
    """MEM002 should be suppressed when PREC rules fire."""
    findings = _analyze(tmp_path, """
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        with torch.cuda.amp.autocast():
            for batch in loader:
                loss = model(batch)
                loss.backward()
                optimizer.step()
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    rule_ids = [d.rule_id for d in result.findings]
    # PREC001 fires (deprecated autocast) -> MEM002 suppressed
    assert "PREC001" in rule_ids
    assert "MEM002" not in rule_ids


def test_summary_counts(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    assert result.summary["critical"] >= 1
    assert result.summary["total"] == len(result.findings)


def test_hardware_context_passed_through(tmp_path):
    findings = _analyze(tmp_path, "# empty\n")
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    assert result.hardware_context == HW_4GPU


def test_parse_errors_passed_through(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def broken(:\n")
    findings = analyze_script(str(p))
    result = run_diagnosis(findings)
    assert len(result.parse_errors) > 0
