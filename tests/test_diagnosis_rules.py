"""Tests for diagnosis_rules.py — Phase 1 + Phase 2 diagnosis rules."""

import textwrap

from alloc.artifact_loader import ArtifactData
from alloc.code_analyzer import analyze_script
from alloc.diagnosis_rules import (
    rule_dl001_num_workers_low,
    rule_dl002_pin_memory,
    rule_dl003_persistent_workers,
    rule_dl004_prefetch_factor,
    rule_dl005_main_thread,
    rule_mem001_gradient_checkpointing,
    rule_mem002_no_mixed_precision,
    rule_mem003_batch_increase,
    rule_mem004_batch_too_large,
    rule_mem005_no_torch_compile,
    rule_prec001_deprecated_autocast,
    rule_prec002_fp16_on_ampere,
    rule_dist001_consider_fsdp,
    rule_dist002_single_gpu_multi_available,
    rule_dist003_gradient_accumulation,
    rule_dist004_gloo_backend,
    rule_dist005_data_parallel,
    rule_thru001_cudnn_benchmark,
    rule_thru003_reduce_grad_accum,
    rule_xs004_mixed_precision_effective,
    rule_strag001_straggler,
    rule_reg001_step_time_regression,
    rule_reg002_throughput_regression,
    rule_reg003_vram_regression,
    rule_reg004_util_regression,
    rule_upg001_deprecated_grad_scaler,
    rule_upg002_tf32_matmul,
    rule_upg003_deepspeed_single_node,
    UPGRADE_RULE_IDS,
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
# DIST005: DataParallel
# ---------------------------------------------------------------------------

def test_dist005_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch.nn as nn
        model = nn.DataParallel(model)
    """)
    diags = rule_dist005_data_parallel(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "DIST005"
    assert "DataParallel" in diags[0].title


def test_dist005_ok_ddp(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    """)
    diags = rule_dist005_data_parallel(findings)
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


# ===========================================================================
# Phase 2 Rules — Runtime-Aware
# ===========================================================================

# ---------------------------------------------------------------------------
# MEM001: Gradient checkpointing (runtime-aware)
# ---------------------------------------------------------------------------

def test_mem001_fires_high_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem001_gradient_checkpointing(findings, HW_1GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "MEM001"
    assert diags[0].severity == "warning"
    assert "85" in diags[0].current_value  # ~85% utilization


def test_mem001_ok_low_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[20000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem001_gradient_checkpointing(findings, HW_1GPU, artifact)
    assert len(diags) == 0


def test_mem001_ok_already_enabled(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.gradient_checkpointing_enable()
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem001_gradient_checkpointing(findings, HW_1GPU, artifact)
    assert len(diags) == 0


def test_mem001_no_artifact(tmp_path):
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_mem001_gradient_checkpointing(findings, HW_1GPU, None)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# MEM003: Batch size can increase
# ---------------------------------------------------------------------------

def test_mem003_fires_low_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=8, num_workers=4)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[15000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem003_batch_increase(findings, HW_1GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "MEM003"


def test_mem003_ok_high_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=32)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[30000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem003_batch_increase(findings, HW_1GPU, artifact)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# MEM004: Batch size too large
# ---------------------------------------------------------------------------

def test_mem004_fires_very_high_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=64)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[38000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem004_batch_too_large(findings, HW_1GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "MEM004"
    assert diags[0].severity == "warning"


def test_mem004_ok_moderate_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=32)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[30000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_mem004_batch_too_large(findings, HW_1GPU, artifact)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DIST001: Consider FSDP
# ---------------------------------------------------------------------------

def test_dist001_fires_ddp_high_vram(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_dist001_consider_fsdp(findings, HW_4GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "DIST001"


def test_dist001_ok_already_fsdp(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)
    """)
    artifact = ArtifactData(
        per_gpu_vram_used_mb=[35000.0],
        per_gpu_vram_total_mb=40960.0,
    )
    diags = rule_dist001_consider_fsdp(findings, HW_4GPU, artifact)
    assert len(diags) == 0


def test_dist001_ok_no_ddp(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        model = torch.nn.Linear(10, 10)
    """)
    diags = rule_dist001_consider_fsdp(findings, HW_4GPU, None)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# DIST003: Gradient accumulation
# ---------------------------------------------------------------------------

def test_dist003_fires_low_util_small_batch(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, num_workers=4)
    """)
    artifact = ArtifactData(avg_gpu_util=35.0)
    diags = rule_dist003_gradient_accumulation(findings, HW_1GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "DIST003"


def test_dist003_ok_high_util(tmp_path):
    findings = _analyze(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, num_workers=4)
    """)
    artifact = ArtifactData(avg_gpu_util=75.0)
    diags = rule_dist003_gradient_accumulation(findings, HW_1GPU, artifact)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# XS004: Mixed precision effective (positive)
# ---------------------------------------------------------------------------

def test_xs004_fires(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda"):
            pass
    """)
    artifact = ArtifactData(avg_gpu_util=90.0)
    diags = rule_xs004_mixed_precision_effective(findings, HW_1GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "XS004"
    assert diags[0].severity == "info"


def test_xs004_ok_low_util(tmp_path):
    findings = _analyze(tmp_path, """
        import torch
        with torch.amp.autocast("cuda"):
            pass
    """)
    artifact = ArtifactData(avg_gpu_util=60.0)
    diags = rule_xs004_mixed_precision_effective(findings, HW_1GPU, artifact)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# STRAG001: Straggler detection
# ---------------------------------------------------------------------------

def test_strag001_fires(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    artifact = ArtifactData(
        per_rank_step_times_ms=[
            [100.0, 102.0, 98.0],
            [101.0, 99.0, 103.0],
            [100.0, 100.0, 101.0],
            [140.0, 138.0, 142.0],  # rank 3 is >20% slower
        ],
    )
    diags = rule_strag001_straggler(findings, HW_4GPU, artifact)
    assert len(diags) == 1
    assert diags[0].rule_id == "STRAG001"
    assert "rank 3" in diags[0].title.lower()


def test_strag001_ok_uniform(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    artifact = ArtifactData(
        per_rank_step_times_ms=[
            [100.0, 102.0, 98.0],
            [101.0, 99.0, 103.0],
            [100.0, 100.0, 101.0],
            [105.0, 103.0, 107.0],  # within 20%
        ],
    )
    diags = rule_strag001_straggler(findings, HW_4GPU, artifact)
    assert len(diags) == 0


def test_strag001_single_gpu(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    artifact = ArtifactData(per_rank_step_times_ms=[[100.0, 102.0]])
    diags = rule_strag001_straggler(findings, HW_1GPU, artifact)
    assert len(diags) == 0


def test_strag001_no_step_times(tmp_path):
    """STRAG001 should not fire based on VRAM skew alone."""
    findings = _analyze(tmp_path, """import torch""")
    artifact = ArtifactData(
        per_rank_peak_vram_mb=[31000.0, 31200.0, 31100.0, 40000.0],
    )
    diags = rule_strag001_straggler(findings, HW_4GPU, artifact)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# REG001: Step time regression
# ---------------------------------------------------------------------------

def test_reg001_fires(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(step_time_p50_ms=200.0)
    previous = ArtifactData(step_time_p50_ms=148.0)
    diags = rule_reg001_step_time_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 1
    assert diags[0].rule_id == "REG001"
    assert diags[0].category == "regression"


def test_reg001_ok_no_regression(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(step_time_p50_ms=150.0)
    previous = ArtifactData(step_time_p50_ms=148.0)
    diags = rule_reg001_step_time_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 0


def test_reg001_no_compare(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(step_time_p50_ms=200.0)
    diags = rule_reg001_step_time_regression(findings, HW_1GPU, current, None)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# REG002: Throughput regression
# ---------------------------------------------------------------------------

def test_reg002_fires(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(throughput_samples_per_sec=30.0)
    previous = ArtifactData(throughput_samples_per_sec=42.0)
    diags = rule_reg002_throughput_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 1
    assert diags[0].rule_id == "REG002"


def test_reg002_ok(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(throughput_samples_per_sec=40.0)
    previous = ArtifactData(throughput_samples_per_sec=42.0)
    diags = rule_reg002_throughput_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# REG003: VRAM regression
# ---------------------------------------------------------------------------

def test_reg003_fires(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(per_gpu_vram_used_mb=[40000.0], peak_vram_mb=40000.0)
    previous = ArtifactData(per_gpu_vram_used_mb=[31000.0], peak_vram_mb=31000.0)
    diags = rule_reg003_vram_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 1
    assert diags[0].rule_id == "REG003"


# ---------------------------------------------------------------------------
# REG004: GPU utilization regression
# ---------------------------------------------------------------------------

def test_reg004_fires(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(avg_gpu_util=50.0)
    previous = ArtifactData(avg_gpu_util=80.0)
    diags = rule_reg004_util_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 1
    assert diags[0].rule_id == "REG004"


def test_reg004_ok(tmp_path):
    findings = _analyze(tmp_path, """import torch""")
    current = ArtifactData(avg_gpu_util=72.0)
    previous = ArtifactData(avg_gpu_util=80.0)
    diags = rule_reg004_util_regression(findings, HW_1GPU, current, previous)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# Framework-specific: MEM002 with HF Trainer
# ---------------------------------------------------------------------------

def test_mem002_hf_trainer_suggests_training_args(tmp_path):
    """When HF Trainer detected, MEM002 suggests TrainingArguments(bf16=True)."""
    findings = _analyze(tmp_path, """
        from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        args = TrainingArguments(output_dir="./out")
        trainer = Trainer(model=model, args=args)
    """)
    diags = rule_mem002_no_mixed_precision(findings, HW_1GPU)
    assert len(diags) == 1
    assert "TrainingArguments" in diags[0].suggested_code
    assert "bf16=True" in diags[0].suggested_value or "fp16=True" in diags[0].suggested_value


def test_mem002_hf_trainer_with_fp16_no_fire(tmp_path):
    """MEM002 should not fire when HF Trainer already has fp16."""
    findings = _analyze(tmp_path, """
        from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        args = TrainingArguments(output_dir="./out", fp16=True)
        trainer = Trainer(model=model, args=args)
    """)
    diags = rule_mem002_no_mixed_precision(findings, HW_1GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# Framework-specific: MEM005 with HF Trainer
# ---------------------------------------------------------------------------

def test_mem005_hf_trainer_suggests_training_args(tmp_path):
    """When HF Trainer detected, MEM005 suggests TrainingArguments(torch_compile=True)."""
    findings = _analyze(tmp_path, """
        from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        args = TrainingArguments(output_dir="./out")
        trainer = Trainer(model=model, args=args)
    """)
    diags = rule_mem005_no_torch_compile(findings, HW_1GPU)
    assert len(diags) == 1
    assert "TrainingArguments" in diags[0].suggested_code
    assert "torch_compile" in diags[0].suggested_value


def test_mem005_raw_pytorch_suggests_torch_compile(tmp_path):
    """Without HF Trainer, MEM005 suggests torch.compile(model)."""
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        for batch in loader:
            loss = model(batch)
            loss.backward()
    """)
    diags = rule_mem005_no_torch_compile(findings, HW_1GPU)
    assert len(diags) == 1
    assert "torch.compile" in diags[0].suggested_value


# ---------------------------------------------------------------------------
# UPG001: Deprecated GradScaler API
# ---------------------------------------------------------------------------

def test_upg001_fires_deprecated_grad_scaler(tmp_path):
    """UPG001 fires on torch.cuda.amp.GradScaler."""
    findings = _analyze(tmp_path, """
        import torch
        scaler = torch.cuda.amp.GradScaler()
    """)
    diags = rule_upg001_deprecated_grad_scaler(findings)
    assert len(diags) == 1
    assert diags[0].rule_id == "UPG001"
    assert diags[0].category == "upgrade"
    assert "torch.amp.GradScaler" in diags[0].suggested_value


def test_upg001_ok_new_api(tmp_path):
    """UPG001 does not fire on new torch.amp.GradScaler."""
    findings = _analyze(tmp_path, """
        import torch
        scaler = torch.amp.GradScaler("cuda")
    """)
    diags = rule_upg001_deprecated_grad_scaler(findings)
    assert len(diags) == 0


def test_upg001_no_grad_scaler(tmp_path):
    """UPG001 does not fire when no GradScaler is used."""
    findings = _analyze(tmp_path, """
        import torch
        model = torch.nn.Linear(10, 10)
    """)
    diags = rule_upg001_deprecated_grad_scaler(findings)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# UPG002: TF32 matmul precision not set
# ---------------------------------------------------------------------------

def test_upg002_fires_ampere_no_tf32(tmp_path):
    """UPG002 fires on Ampere+ when set_float32_matmul_precision not called."""
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_upg002_tf32_matmul(findings, HW_H100)
    assert len(diags) == 1
    assert diags[0].rule_id == "UPG002"
    assert diags[0].category == "upgrade"
    assert "set_float32_matmul_precision" in diags[0].suggested_value


def test_upg002_ok_already_set(tmp_path):
    """UPG002 does not fire when set_float32_matmul_precision is already called."""
    findings = _analyze(tmp_path, """
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_float32_matmul_precision("high")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_upg002_tf32_matmul(findings, HW_H100)
    assert len(diags) == 0


def test_upg002_skips_v100(tmp_path):
    """UPG002 does not fire on pre-Ampere GPUs."""
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_upg002_tf32_matmul(findings, HW_V100)
    assert len(diags) == 0


def test_upg002_skips_no_hw(tmp_path):
    """UPG002 does not fire when no hardware context."""
    findings = _analyze(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """)
    diags = rule_upg002_tf32_matmul(findings, None)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# UPG003: DeepSpeed on single node
# ---------------------------------------------------------------------------

def test_upg003_fires_deepspeed_single_node(tmp_path):
    """UPG003 fires on DeepSpeed with <= 8 GPUs."""
    findings = _analyze(tmp_path, """
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
    """)
    diags = rule_upg003_deepspeed_single_node(findings, HW_4GPU)
    assert len(diags) == 1
    assert diags[0].rule_id == "UPG003"
    assert diags[0].category == "upgrade"
    assert "FSDP" in diags[0].suggested_value


def test_upg003_skips_multi_node(tmp_path):
    """UPG003 does not fire when GPU count > 8 (likely multi-node)."""
    findings = _analyze(tmp_path, """
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
    """)
    hw = {"gpu_name": "H100", "gpu_count": 16, "sm_version": "9.0"}
    diags = rule_upg003_deepspeed_single_node(findings, hw)
    assert len(diags) == 0


def test_upg003_skips_no_deepspeed(tmp_path):
    """UPG003 does not fire without DeepSpeed."""
    findings = _analyze(tmp_path, """
        import torch
        model = torch.nn.Linear(10, 10)
    """)
    diags = rule_upg003_deepspeed_single_node(findings, HW_4GPU)
    assert len(diags) == 0


def test_upg003_skips_already_fsdp(tmp_path):
    """UPG003 does not fire if FSDP is already in use alongside DeepSpeed."""
    findings = _analyze(tmp_path, """
        import deepspeed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
        model = FSDP(model)
    """)
    diags = rule_upg003_deepspeed_single_node(findings, HW_4GPU)
    assert len(diags) == 0


# ---------------------------------------------------------------------------
# UPGRADE_RULE_IDS registry
# ---------------------------------------------------------------------------

def test_upgrade_rule_ids_contains_expected():
    """UPGRADE_RULE_IDS contains all expected upgrade-related rules."""
    assert "UPG001" in UPGRADE_RULE_IDS
    assert "UPG002" in UPGRADE_RULE_IDS
    assert "UPG003" in UPGRADE_RULE_IDS
    assert "PREC001" in UPGRADE_RULE_IDS
    assert "PREC002" in UPGRADE_RULE_IDS
    assert "MEM005" in UPGRADE_RULE_IDS
    assert "DIST005" in UPGRADE_RULE_IDS
    # Ensure non-upgrade rules are NOT included
    assert "DL001" not in UPGRADE_RULE_IDS
    assert "MEM001" not in UPGRADE_RULE_IDS
