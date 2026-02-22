"""Tests for diagnosis_rules.py — 12 diagnosis rules."""

import textwrap

from alloc.code_analyzer import analyze_script
from alloc.diagnosis_rules import (
    rule_dl001_num_workers_low,
    rule_dl002_pin_memory,
    rule_dl003_persistent_workers,
    rule_dl004_prefetch_factor,
    rule_dl005_main_thread,
    rule_mem002_no_mixed_precision,
    rule_mem005_no_torch_compile,
    rule_prec001_deprecated_autocast,
    rule_prec002_fp16_on_ampere,
    rule_dist002_single_gpu_multi_available,
    rule_dist004_gloo_backend,
    rule_thru001_cudnn_benchmark,
)


def _analyze(tmp_path, code):
    p = tmp_path / "train.py"
    p.write_text(textwrap.dedent(code))
    return analyze_script(str(p))


HW_4GPU = {"gpu_name": "NVIDIA A100", "gpu_count": 4, "sm_version": "8.0"}
HW_1GPU = {"gpu_name": "NVIDIA A100", "gpu_count": 1, "sm_version": "8.0"}
HW_H100 = {"gpu_name": "NVIDIA H100", "gpu_count": 1, "sm_version": "9.0"}
HW_V100 = {"gpu_name": "NVIDIA V100", "gpu_count": 1, "sm_version": "7.0"}


# ---------------------------------------------------------------------------
# DL001: num_workers too low
# ---------------------------------------------------------------------------

def test_dl001_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=2)
    """)
    diags = rule_dl001_num_workers_low(findings, HW_4GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "DL001"
    assert diags[0].severity == "critical"
    assert "num_workers=8" in diags[0].suggested_value


def test_dl001_ok(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=8)
    """)
    diags = rule_dl001_num_workers_low(findings, HW_4GPU)
    assert len(diags) == 0


def test_dl001_no_hw(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=2)
    """)
    diags = rule_dl001_num_workers_low(findings, None)
    assert len(diags) == 1
    assert diags[0].confidence == "medium"


def test_dl001_skips_zero(tmp_path):
    """DL001 skips num_workers=0 (handled by DL005)."""
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    diags = rule_dl001_num_workers_low(findings, HW_4GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DL002: pin_memory
# ---------------------------------------------------------------------------

def test_dl002_fires_no_pin(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4)
    """)
    diags = rule_dl002_pin_memory(findings, HW_1GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "DL002"


def test_dl002_ok_pin_true(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4, pin_memory=True)
    """)
    diags = rule_dl002_pin_memory(findings, HW_1GPU)
    assert len(diags) == 0


def test_dl002_skips_zero_workers(tmp_path):
    """pin_memory less useful with num_workers=0."""
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    diags = rule_dl002_pin_memory(findings, HW_1GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DL003: persistent_workers
# ---------------------------------------------------------------------------

def test_dl003_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4)
    """)
    diags = rule_dl003_persistent_workers(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "DL003"


def test_dl003_ok(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4, persistent_workers=True)
    """)
    diags = rule_dl003_persistent_workers(findings)
    assert len(diags) == 0


def test_dl003_skips_zero_workers(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    diags = rule_dl003_persistent_workers(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DL004: prefetch_factor
# ---------------------------------------------------------------------------

def test_dl004_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4)
    """)
    diags = rule_dl004_prefetch_factor(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "DL004"


def test_dl004_ok(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4, prefetch_factor=2)
    """)
    diags = rule_dl004_prefetch_factor(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DL005: num_workers=0
# ---------------------------------------------------------------------------

def test_dl005_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=0)
    """)
    diags = rule_dl005_main_thread(findings, HW_4GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "DL005"
    assert diags[0].severity == "critical"


def test_dl005_skips_positive(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=4)
    """)
    diags = rule_dl005_main_thread(findings, HW_4GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# MEM002: No mixed precision
# ---------------------------------------------------------------------------

def test_mem002_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
    """)
    diags = rule_mem002_no_mixed_precision(findings, HW_1GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "MEM002"


def test_mem002_ok_with_autocast(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        with torch.amp.autocast("cuda"):
            pass
    """)
    diags = rule_mem002_no_mixed_precision(findings, HW_1GPU)
    assert len(diags) == 0


def test_mem002_no_model(tmp_path):
    findings = _analyze(tmp_path, """
        x = 1 + 2
    """)
    diags = rule_mem002_no_mixed_precision(findings, HW_1GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# MEM005: No torch.compile
# ---------------------------------------------------------------------------

def test_mem005_fires(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_mem005_no_torch_compile(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "MEM005"


def test_mem005_ok_with_compile(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = torch.compile(model)
    """)
    diags = rule_mem005_no_torch_compile(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# PREC001: Deprecated autocast API
# ---------------------------------------------------------------------------

def test_prec001_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.cuda.amp.autocast():
            pass
    """)
    diags = rule_prec001_deprecated_autocast(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "PREC001"


def test_prec001_ok_new_api(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda"):
            pass
    """)
    diags = rule_prec001_deprecated_autocast(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# PREC002: fp16 on Ampere+
# ---------------------------------------------------------------------------

def test_prec002_fires_h100(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pass
    """)
    diags = rule_prec002_fp16_on_ampere(findings, HW_H100)
    assert len(diags) == 1
    assert diags[0].rule_id == "PREC002"
    assert "bfloat16" in diags[0].suggested_code


def test_prec002_ok_bf16(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pass
    """)
    diags = rule_prec002_fp16_on_ampere(findings, HW_H100)
    assert len(diags) == 0


def test_prec002_skips_v100(tmp_path):
    """V100 is SM 7.0 — no bf16 support."""
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pass
    """)
    diags = rule_prec002_fp16_on_ampere(findings, HW_V100)
    assert len(diags) == 0


def test_prec002_no_hw(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pass
    """)
    diags = rule_prec002_fp16_on_ampere(findings, None)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DIST002: Single GPU, multi available
# ---------------------------------------------------------------------------

def test_dist002_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        model = torch.nn.Linear(10, 10)
    """)
    diags = rule_dist002_single_gpu_multi_available(findings, HW_4GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "DIST002"


def test_dist002_ok_has_ddp(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    """)
    diags = rule_dist002_single_gpu_multi_available(findings, HW_4GPU)
    assert len(diags) == 0


def test_dist002_ok_single_gpu(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
    """)
    diags = rule_dist002_single_gpu_multi_available(findings, HW_1GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DIST004: GLOO backend
# ---------------------------------------------------------------------------

def test_dist004_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch.distributed as dist
        dist.init_process_group("gloo")
    """)
    diags = rule_dist004_gloo_backend(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "DIST004"
    assert "nccl" in diags[0].suggested_code


def test_dist004_ok_nccl(tmp_path):
    findings = _analyze(tmp_path, """
        import torch.distributed as dist
        dist.init_process_group("nccl")
    """)
    diags = rule_dist004_gloo_backend(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# THRU001: cudnn.benchmark
# ---------------------------------------------------------------------------

def test_thru001_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
    """)
    diags = rule_thru001_cudnn_benchmark(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "THRU001"


def test_thru001_ok(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        torch.backends.cudnn.benchmark = True
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
    """)
    diags = rule_thru001_cudnn_benchmark(findings)
    assert len(diags) == 0


def test_thru001_no_loop(tmp_path):
    """No training loop detected — don't suggest cudnn.benchmark."""
    findings = _analyze(tmp_path, """
        import torch
        model = torch.nn.Linear(10, 10)
    """)
    diags = rule_thru001_cudnn_benchmark(findings)
    assert len(diags) == 0
