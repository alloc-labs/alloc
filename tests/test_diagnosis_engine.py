"""Tests for diagnosis_engine.py — rule evaluation and result assembly."""

import textwrap

from alloc.artifact_loader import ArtifactData
from alloc.code_analyzer import analyze_script
from alloc.diagnosis_engine import (
    run_diagnosis, DiagnoseResult, _deduplicate,
    _determine_tier, _build_comparison, _build_efficiency,
    _sniff_environment,
)
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
    assert result.version == "2"
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


# ---------------------------------------------------------------------------
# Phase 2: Artifact integration
# ---------------------------------------------------------------------------

def test_run_diagnosis_with_artifact(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        per_gpu_vram_total_mb=40960.0,
        peak_vram_mb=35000.0,
        dataloader_wait_pct=38.0,
        step_time_p50_ms=148.5,
    )
    result = run_diagnosis(findings, hardware_context=HW_4GPU, artifact=artifact)
    assert result.tier == "runtime"
    # Should have runtime-enhanced findings
    rule_ids = [d.rule_id for d in result.findings]
    assert "DL005" in rule_ids
    assert "MEM001" in rule_ids


def test_run_diagnosis_without_artifact(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    assert result.tier == "ast_only"
    assert result.oom_detected is False


def test_determine_tier():
    assert _determine_tier(None) == "ast_only"
    assert _determine_tier(ArtifactData()) == "runtime"
    assert _determine_tier(ArtifactData(per_rank_peak_vram_mb=[1, 2])) == "runtime_distributed"
    assert _determine_tier(ArtifactData(per_rank_peak_vram_mb=[1])) == "runtime"


def test_build_comparison():
    cur = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        peak_vram_mb=35000.0,
        step_time_p50_ms=200.0,
        avg_gpu_util=65.0,
        throughput_samples_per_sec=30.0,
    )
    prev = ArtifactData(
        per_gpu_vram_used_mb=[31000.0],
        peak_vram_mb=31000.0,
        step_time_p50_ms=148.0,
        avg_gpu_util=72.0,
        throughput_samples_per_sec=42.0,
    )
    comp = _build_comparison(cur, prev)
    assert comp["has_regressions"] is True
    names = [m["name"] for m in comp["metrics"]]
    assert "Step time (p50)" in names
    assert "Throughput" in names


def test_build_efficiency():
    artifact = ArtifactData(
        step_time_p50_ms=100.0,
        dataloader_wait_pct=30.0,
    )
    eff = _build_efficiency(artifact)
    assert eff is not None
    assert eff["step_time_p50_ms"] == 100.0
    assert eff["data_loading_pct"] == 30.0
    assert eff["compute_pct"] == 70.0
    assert eff["bottleneck"] == "Data loading"


def test_build_efficiency_no_timing():
    artifact = ArtifactData()
    eff = _build_efficiency(artifact)
    assert eff is None


def test_dedup_dl001_dl005_keeps_dl005(tmp_path):
    """DL005 should suppress DL001 on same line (more specific)."""
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    result = run_diagnosis(findings, hardware_context=HW_4GPU)
    rule_ids = [d.rule_id for d in result.findings]
    assert "DL005" in rule_ids
    # DL001 should be suppressed by DL005 on the same line
    dl001_dl005_same_line = any(
        d.rule_id == "DL001" and d.line_number == dl005.line_number
        for d in result.findings
        for dl005 in result.findings if dl005.rule_id == "DL005"
    )
    assert not dl001_dl005_same_line


def test_oom_detection_in_result(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    artifact = ArtifactData(
        exit_code=-9,
        per_gpu_vram_used_mb=[39600.0],
        per_gpu_vram_total_mb=40960.0,
        peak_vram_mb=39600.0,
        is_oom=True,
    )
    result = run_diagnosis(findings, hardware_context=HW_4GPU, artifact=artifact)
    assert result.oom_detected is True


# ---------------------------------------------------------------------------
# Environment sniffing
# ---------------------------------------------------------------------------

def test_sniff_environment_empty(monkeypatch):
    """No training env vars set — returns empty dict."""
    for key in ["CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
                "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE"]:
        monkeypatch.delenv(key, raising=False)
    ctx = _sniff_environment()
    assert ctx == {}


def test_sniff_environment_cuda_visible(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    ctx = _sniff_environment()
    assert ctx["cuda_visible_devices"] == ["0", "1", "2", "3"]
    assert ctx["gpu_count"] == 4


def test_sniff_environment_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    ctx = _sniff_environment()
    assert ctx["world_size"] == 8
    assert ctx["rank"] == 0
    assert ctx["local_world_size"] == 4
    assert ctx["master_addr"] == "10.0.0.1"


def test_sniff_environment_nccl(monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    ctx = _sniff_environment()
    assert "nccl_config" in ctx
    assert ctx["nccl_config"]["NCCL_SOCKET_IFNAME"] == "eth0"
    assert ctx["nccl_config"]["NCCL_IB_DISABLE"] == "1"


def test_sniff_environment_omp(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    ctx = _sniff_environment()
    assert ctx["omp_num_threads"] == 4


def test_sniff_environment_alloc_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ctx = _sniff_environment()
    assert ctx["pytorch_cuda_alloc_conf"] == "expandable_segments:True"
