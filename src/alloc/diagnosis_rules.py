"""Diagnosis rules — pure-function rules for training script analysis.

Phase 1 rules: (CodeFindings, Optional[Dict]) -> List[Diagnosis]
Phase 2 rules: (CodeFindings, Optional[Dict], Optional[ArtifactData]) -> List[Diagnosis]
Regression rules: (CodeFindings, Optional[Dict], Optional[ArtifactData], Optional[ArtifactData]) -> List[Diagnosis]

All thresholds cite public PyTorch/NVIDIA documentation. No proprietary logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from alloc.code_analyzer import CodeFindings

if TYPE_CHECKING:
    from alloc.artifact_loader import ArtifactData


@dataclass
class Diagnosis:
    rule_id: str
    severity: str            # "critical" | "warning" | "info"
    category: str            # "dataloader" | "memory" | "precision" | "distributed" | "throughput"
    title: str
    file_path: str
    line_number: int
    current_code: str
    current_value: str
    suggested_value: str
    suggested_code: str
    rationale: str
    estimated_impact: str
    confidence: str          # "high" | "medium" | "low"
    doc_url: Optional[str]
    evidence: Optional[Dict[str, str]] = None  # measured values that triggered this finding


# ---------------------------------------------------------------------------
# Helper: replace a keyword argument in a source line
# ---------------------------------------------------------------------------

def _replace_kwarg(source_line: str, name: str, old_val: str, new_val: str) -> str:
    """Replace name=old_val with name=new_val in source line."""
    pattern = rf'{name}\s*=\s*{re.escape(old_val)}\b'
    replacement = f'{name}={new_val}'
    result = re.sub(pattern, replacement, source_line, count=1)
    return result


def _insert_kwarg(source_line: str, name: str, value: str) -> str:
    """Insert name=value as a new kwarg before the closing parenthesis."""
    # Find the last ) and insert before it
    idx = source_line.rfind(")")
    if idx == -1:
        return source_line
    prefix = source_line[:idx].rstrip()
    suffix = source_line[idx:]
    if prefix.endswith(","):
        return f"{prefix} {name}={value}{suffix}"
    return f"{prefix}, {name}={value}{suffix}"


def _has_hf_trainer(findings: CodeFindings) -> bool:
    """Check if HuggingFace Trainer is detected in the code."""
    return any(d.kind == "hf_trainer" for d in findings.distributed)


# ---------------------------------------------------------------------------
# DL001: num_workers too low
# ---------------------------------------------------------------------------

def rule_dl001_num_workers_low(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """DataLoader num_workers too low.

    Fires when num_workers < recommended minimum.
    Recommended: max(4, gpu_count * 2) per PyTorch data loading tutorial.
    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    results = []
    gpu_count = (hw or {}).get("gpu_count", 1) or 1
    recommended = max(4, gpu_count * 2)

    for dl in findings.dataloaders:
        if dl.num_workers is None:
            continue
        if dl.num_workers == 0:
            continue  # Handled by DL005
        if dl.num_workers >= recommended:
            continue

        severity = "critical" if dl.num_workers <= 2 else "warning"
        confidence = "high" if hw else "medium"
        suggested_code = _replace_kwarg(
            dl.location.source_line,
            "num_workers",
            str(dl.num_workers),
            str(recommended),
        )

        results.append(Diagnosis(
            rule_id="DL001",
            severity=severity,
            category="dataloader",
            title="Dataloader workers too low",
            file_path=dl.location.file_path,
            line_number=dl.location.line_number,
            current_code=dl.location.source_line,
            current_value=f"num_workers={dl.num_workers}",
            suggested_value=f"num_workers={recommended}",
            suggested_code=suggested_code,
            rationale=f"With {gpu_count} GPU(s), recommended minimum is {recommended} workers to keep the GPU fed.",
            estimated_impact="~15-25% faster data loading",
            confidence=confidence,
            doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
            evidence={"num_workers": str(dl.num_workers), "gpu_count": str(gpu_count), "recommended": str(recommended)},
        ))

    return results


# ---------------------------------------------------------------------------
# DL002: pin_memory disabled
# ---------------------------------------------------------------------------

def rule_dl002_pin_memory(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """pin_memory not True when CUDA is available.

    Ref: https://pytorch.org/docs/stable/data.html#memory-pinning
    """
    has_cuda = hw is not None  # If we have hw context, assume CUDA
    if not has_cuda:
        # Check imports for CUDA-related patterns
        for mod in findings.imports.values():
            if "cuda" in mod.lower() or "torch" in mod.lower():
                has_cuda = True
                break

    if not has_cuda:
        return []

    results = []
    for dl in findings.dataloaders:
        if dl.pin_memory is True:
            continue
        # Skip if num_workers is 0 (explicit or default) — pin_memory less useful with main-thread loading
        if dl.num_workers is None or dl.num_workers == 0:
            continue

        suggested_code = dl.location.source_line
        if dl.pin_memory is False:
            suggested_code = _replace_kwarg(suggested_code, "pin_memory", "False", "True")
        elif dl.pin_memory is None and "pin_memory" not in dl.raw_kwargs:
            suggested_code = _insert_kwarg(suggested_code, "pin_memory", "True")

        results.append(Diagnosis(
            rule_id="DL002",
            severity="info",
            category="dataloader",
            title="pin_memory not enabled",
            file_path=dl.location.file_path,
            line_number=dl.location.line_number,
            current_code=dl.location.source_line,
            current_value="pin_memory=False" if dl.pin_memory is False else "pin_memory not set",
            suggested_value="pin_memory=True",
            suggested_code=suggested_code,
            rationale="pin_memory=True enables faster host-to-GPU data transfer via page-locked memory.",
            estimated_impact="~5-10% faster data transfer to GPU",
            confidence="medium",
            doc_url="https://pytorch.org/docs/stable/data.html#memory-pinning",
            evidence={"pin_memory": str(dl.pin_memory), "num_workers": str(dl.num_workers)},
        ))

    return results


# ---------------------------------------------------------------------------
# DL003: persistent_workers disabled
# ---------------------------------------------------------------------------

def rule_dl003_persistent_workers(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """persistent_workers not True when num_workers > 0.

    Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    results = []
    for dl in findings.dataloaders:
        if dl.num_workers is None or dl.num_workers <= 0:
            continue
        if dl.persistent_workers is True:
            continue

        suggested_code = dl.location.source_line
        if dl.persistent_workers is False:
            suggested_code = _replace_kwarg(suggested_code, "persistent_workers", "False", "True")
        elif dl.persistent_workers is None:
            suggested_code = _insert_kwarg(suggested_code, "persistent_workers", "True")

        results.append(Diagnosis(
            rule_id="DL003",
            severity="info",
            category="dataloader",
            title="persistent_workers not enabled",
            file_path=dl.location.file_path,
            line_number=dl.location.line_number,
            current_code=dl.location.source_line,
            current_value="persistent_workers=False" if dl.persistent_workers is False else "persistent_workers not set",
            suggested_value="persistent_workers=True",
            suggested_code=suggested_code,
            rationale="persistent_workers=True avoids re-creating worker processes each epoch, reducing overhead.",
            estimated_impact="~5-15% faster epoch transitions",
            confidence="medium",
            doc_url="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
            evidence={"persistent_workers": str(dl.persistent_workers), "num_workers": str(dl.num_workers)},
        ))

    return results


# ---------------------------------------------------------------------------
# DL004: prefetch_factor not set
# ---------------------------------------------------------------------------

def rule_dl004_prefetch_factor(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """prefetch_factor not set when num_workers > 0.

    Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    results = []
    for dl in findings.dataloaders:
        if dl.num_workers is None or dl.num_workers <= 0:
            continue
        if dl.prefetch_factor is not None:
            continue

        suggested_code = _insert_kwarg(dl.location.source_line, "prefetch_factor", "2")

        results.append(Diagnosis(
            rule_id="DL004",
            severity="info",
            category="dataloader",
            title="prefetch_factor not set",
            file_path=dl.location.file_path,
            line_number=dl.location.line_number,
            current_code=dl.location.source_line,
            current_value="prefetch_factor not set (default 2)",
            suggested_value="prefetch_factor=2",
            suggested_code=suggested_code,
            rationale="Explicitly setting prefetch_factor ensures predictable data prefetching behavior.",
            estimated_impact="Minimal — documents intent, prevents surprises from default changes",
            confidence="low",
            doc_url="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
            evidence={"prefetch_factor": "None", "num_workers": str(dl.num_workers)},
        ))

    return results


# ---------------------------------------------------------------------------
# DL005: num_workers=0 (main thread DataLoader)
# ---------------------------------------------------------------------------

def rule_dl005_main_thread(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """DataLoader running in main thread (num_workers=0).

    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    results = []
    gpu_count = (hw or {}).get("gpu_count", 1) or 1
    recommended = max(4, gpu_count * 2)

    for dl in findings.dataloaders:
        if dl.num_workers != 0:
            continue

        confidence = "high" if hw else "medium"
        suggested_code = _replace_kwarg(
            dl.location.source_line,
            "num_workers",
            "0",
            str(recommended),
        )

        results.append(Diagnosis(
            rule_id="DL005",
            severity="critical",
            category="dataloader",
            title="DataLoader running in main thread",
            file_path=dl.location.file_path,
            line_number=dl.location.line_number,
            current_code=dl.location.source_line,
            current_value="num_workers=0",
            suggested_value=f"num_workers={recommended}",
            suggested_code=suggested_code,
            rationale="num_workers=0 loads data in the main thread, blocking GPU computation entirely.",
            estimated_impact="~30-50% faster training with parallel data loading",
            confidence=confidence,
            doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
            evidence={"num_workers": "0", "gpu_count": str(gpu_count), "recommended": str(recommended)},
        ))

    return results


# ---------------------------------------------------------------------------
# MEM002: No mixed precision
# ---------------------------------------------------------------------------

def rule_mem002_no_mixed_precision(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """No mixed precision detected when a model is present.

    Ref: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
    """
    if not findings.models:
        return []

    # Check if any precision pattern exists
    if findings.precision:
        return []

    # Need at least a model or training loop to recommend this
    if not findings.has_training_loop and not any(m.kind == "from_pretrained" for m in findings.models):
        return []

    confidence = "medium" if hw else "low"
    gpu_name = (hw or {}).get("gpu_name", "")
    sm = (hw or {}).get("sm_version", "")

    use_bf16 = True
    if sm:
        try:
            sm_major = int(str(sm).split(".")[0])
            if sm_major < 8:
                use_bf16 = False
        except (ValueError, IndexError):
            pass

    # Framework-specific suggestion: HF Trainer uses TrainingArguments
    ev = {"sm_version": str(sm), "gpu_name": gpu_name, "bf16_supported": str(use_bf16)}

    if _has_hf_trainer(findings):
        dtype_flag = "bf16=True" if use_bf16 else "fp16=True"
        return [Diagnosis(
            rule_id="MEM002",
            severity="warning",
            category="memory",
            title="No mixed precision detected",
            file_path=findings.script_path,
            line_number=0,
            current_code="",
            current_value="No mixed precision in TrainingArguments",
            suggested_value=f"TrainingArguments({dtype_flag})",
            suggested_code=f"TrainingArguments(output_dir=..., {dtype_flag})",
            rationale="HF Trainer supports mixed precision via TrainingArguments. No manual autocast needed.",
            estimated_impact="~30-50% memory reduction, ~20-40% speedup",
            confidence=confidence,
            doc_url="https://huggingface.co/docs/transformers/perf_train_gpu_one#fp16-training",
            evidence=ev,
        )]

    dtype_suggestion = "torch.bfloat16" if use_bf16 else "torch.float16"
    return [Diagnosis(
        rule_id="MEM002",
        severity="warning",
        category="memory",
        title="No mixed precision detected",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value="No autocast or mixed precision usage",
        suggested_value=f"torch.amp.autocast('cuda', dtype={dtype_suggestion})",
        suggested_code=f"with torch.amp.autocast('cuda', dtype={dtype_suggestion}):",
        rationale="Mixed precision training reduces memory usage and increases throughput on modern GPUs.",
        estimated_impact="~30-50% memory reduction, ~20-40% speedup",
        confidence=confidence,
        doc_url="https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html",
        evidence=ev,
    )]


# ---------------------------------------------------------------------------
# MEM005: torch.compile not used
# ---------------------------------------------------------------------------

def rule_mem005_no_torch_compile(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """No torch.compile detected when a model is present.

    Ref: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    """
    if not findings.models:
        return []

    # Check if torch.compile is already used
    has_compile = any(d.kind == "torch_compile" for d in findings.distributed)
    if has_compile:
        return []

    # Framework-specific: HF Trainer supports torch_compile in TrainingArguments
    if _has_hf_trainer(findings):
        return [Diagnosis(
            rule_id="MEM005",
            severity="info",
            category="throughput",
            title="torch.compile not used",
            file_path=findings.script_path,
            line_number=0,
            current_code="",
            current_value="No torch.compile in TrainingArguments",
            suggested_value="TrainingArguments(torch_compile=True)",
            suggested_code="TrainingArguments(output_dir=..., torch_compile=True)",
            rationale="HF Trainer supports torch.compile via TrainingArguments(torch_compile=True).",
            estimated_impact="~10-30% speedup depending on model architecture",
            confidence="low",
            doc_url="https://huggingface.co/docs/transformers/perf_train_gpu_one#using-torchcompile",
            evidence={"framework": "hf_trainer"},
        )]

    return [Diagnosis(
        rule_id="MEM005",
        severity="info",
        category="throughput",
        title="torch.compile not used",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value="No torch.compile usage",
        suggested_value="model = torch.compile(model)",
        suggested_code="model = torch.compile(model)",
        rationale="torch.compile (PyTorch 2.0+) can fuse operations and optimize execution graphs.",
        estimated_impact="~10-30% speedup depending on model architecture",
        confidence="low",
        doc_url="https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    )]


# ---------------------------------------------------------------------------
# PREC001: Deprecated autocast API
# ---------------------------------------------------------------------------

def rule_prec001_deprecated_autocast(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using deprecated torch.cuda.amp.autocast API.

    Ref: https://pytorch.org/docs/stable/amp.html
    """
    results = []
    for prec in findings.precision:
        if prec.kind != "autocast" or not prec.is_deprecated_api:
            continue

        suggested_code = prec.location.source_line
        # Replace torch.cuda.amp.autocast with torch.amp.autocast("cuda")
        suggested_code = suggested_code.replace("torch.cuda.amp.autocast", 'torch.amp.autocast')
        # If no device_type arg, add "cuda"
        if 'autocast()' in suggested_code:
            suggested_code = suggested_code.replace('autocast()', 'autocast("cuda")')
        elif 'autocast(' in suggested_code and '"cuda"' not in suggested_code and "'cuda'" not in suggested_code:
            suggested_code = suggested_code.replace('autocast(', 'autocast("cuda", ', 1)

        results.append(Diagnosis(
            rule_id="PREC001",
            severity="info",
            category="precision",
            title="Deprecated autocast API",
            file_path=prec.location.file_path,
            line_number=prec.location.line_number,
            current_code=prec.location.source_line,
            current_value="torch.cuda.amp.autocast",
            suggested_value='torch.amp.autocast("cuda")',
            suggested_code=suggested_code,
            rationale="torch.cuda.amp.autocast is deprecated since PyTorch 2.4. Use torch.amp.autocast instead.",
            estimated_impact="Future-proofs code, no performance change",
            confidence="high",
            doc_url="https://pytorch.org/docs/stable/amp.html",
            evidence={"api": "torch.cuda.amp.autocast"},
        ))

    return results


# ---------------------------------------------------------------------------
# PREC002: fp16 on Ampere+ (should use bf16)
# ---------------------------------------------------------------------------

def rule_prec002_fp16_on_ampere(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using fp16 on Ampere+ hardware where bf16 is natively supported.

    SM >= 8.0 (Ampere) has native bf16 support — eliminates need for loss scaling.
    Ref: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
    """
    if not hw:
        return []

    sm = hw.get("sm_version", "")
    if not sm:
        return []

    try:
        sm_major = int(str(sm).split(".")[0])
    except (ValueError, IndexError):
        return []

    if sm_major < 8:
        return []

    results = []
    for prec in findings.precision:
        if prec.kind not in ("autocast", "half"):
            continue

        dtype = prec.dtype_str or ""
        if dtype in ("bfloat16", "bf16"):
            continue

        # Only fire if it's fp16-related
        if prec.kind == "autocast" and dtype and dtype not in ("float16", "fp16"):
            continue

        suggested_code = prec.location.source_line
        if prec.kind == "half":
            suggested_code = suggested_code.replace(".half()", ".bfloat16()")
        elif "float16" in suggested_code:
            suggested_code = suggested_code.replace("float16", "bfloat16")
        elif prec.kind == "autocast" and "dtype=" not in suggested_code:
            # autocast without explicit dtype defaults to fp16 — add bf16
            suggested_code = suggested_code.replace("autocast(", "autocast(dtype=torch.bfloat16, ", 1)

        gpu_name = hw.get("gpu_name", "GPU")

        results.append(Diagnosis(
            rule_id="PREC002",
            severity="warning",
            category="precision",
            title="Using fp16, consider bf16",
            file_path=prec.location.file_path,
            line_number=prec.location.line_number,
            current_code=prec.location.source_line,
            current_value=f"dtype: {dtype or 'float16'}",
            suggested_value="dtype: bfloat16",
            suggested_code=suggested_code,
            rationale=f"{gpu_name} supports bf16 natively — same tensor core throughput as fp16, but eliminates GradScaler overhead and avoids fp16 overflow/underflow issues.",
            estimated_impact="Eliminates GradScaler complexity, avoids numerical instability",
            confidence="high",
            doc_url="https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html",
            evidence={"current_dtype": dtype or "float16", "sm_version": str(sm), "gpu_name": gpu_name},
        ))

    return results


# ---------------------------------------------------------------------------
# DIST002: Single GPU, multi available
# ---------------------------------------------------------------------------

def rule_dist002_single_gpu_multi_available(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Training on single GPU when multiple GPUs are available.

    Ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    if not hw:
        return []

    gpu_count = hw.get("gpu_count", 1) or 1
    if gpu_count <= 1:
        return []

    # Check if any distributed pattern exists (including DeepSpeed, Accelerate, HF Trainer, DataParallel)
    has_dist = any(d.kind in ("ddp", "fsdp", "init_process_group", "deepspeed", "accelerate", "hf_trainer", "data_parallel") for d in findings.distributed)
    if has_dist:
        return []

    return [Diagnosis(
        rule_id="DIST002",
        severity="info",
        category="distributed",
        title="Single GPU training, multiple GPUs available",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"No distributed training, {gpu_count} GPUs detected",
        suggested_value="Use DistributedDataParallel or torchrun",
        suggested_code="# torchrun --nproc_per_node={} train.py".format(gpu_count),
        rationale=f"{gpu_count} GPUs detected but no distributed training patterns found.",
        estimated_impact=f"Up to ~{min(gpu_count, 8)}x throughput (actual scaling depends on model size and communication overhead)",
        confidence="medium",
        doc_url="https://pytorch.org/tutorials/intermediate/ddp_tutorial.html",
        evidence={"gpu_count": str(gpu_count)},
    )]


# ---------------------------------------------------------------------------
# DIST005: DataParallel (not DDP)
# ---------------------------------------------------------------------------

def rule_dist005_data_parallel(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using nn.DataParallel instead of DistributedDataParallel.

    DataParallel is single-process, GIL-bound, and slower than DDP.
    DDP uses one process per GPU and avoids the GIL bottleneck.

    Ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    results = []
    for dist in findings.distributed:
        if dist.kind != "data_parallel":
            continue

        suggested_code = dist.location.source_line
        for old in ("nn.DataParallel", "DataParallel"):
            if old in suggested_code:
                suggested_code = suggested_code.replace(old, "nn.parallel.DistributedDataParallel")
                break

        results.append(Diagnosis(
            rule_id="DIST005",
            severity="warning",
            category="distributed",
            title="Using DataParallel — switch to DistributedDataParallel",
            file_path=dist.location.file_path,
            line_number=dist.location.line_number,
            current_code=dist.location.source_line,
            current_value="nn.DataParallel (single-process, GIL-bound)",
            suggested_value="DistributedDataParallel (one process per GPU)",
            suggested_code=suggested_code,
            rationale="DataParallel splits batches across GPUs in a single process, bottlenecked by the GIL. DistributedDataParallel uses one process per GPU and scales better.",
            estimated_impact="~30-50% throughput improvement on multi-GPU",
            confidence="high",
            doc_url="https://pytorch.org/tutorials/intermediate/ddp_tutorial.html",
            evidence={"current_strategy": "DataParallel"},
        ))

    return results


# ---------------------------------------------------------------------------
# DIST004: GLOO backend on GPU
# ---------------------------------------------------------------------------

def rule_dist004_gloo_backend(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using GLOO backend with CUDA — NCCL is recommended for GPU training.

    Ref: https://pytorch.org/docs/stable/distributed.html#backends
    """
    results = []
    for dist in findings.distributed:
        if dist.kind != "init_process_group":
            continue
        if dist.backend != "gloo":
            continue

        suggested_code = dist.location.source_line.replace('"gloo"', '"nccl"').replace("'gloo'", "'nccl'")

        results.append(Diagnosis(
            rule_id="DIST004",
            severity="warning",
            category="distributed",
            title="GLOO backend on GPU training",
            file_path=dist.location.file_path,
            line_number=dist.location.line_number,
            current_code=dist.location.source_line,
            current_value='backend="gloo"',
            suggested_value='backend="nccl"',
            suggested_code=suggested_code,
            rationale="GLOO copies GPU tensors through host memory for collectives. NCCL uses direct GPU-to-GPU transfers (NVLink/PCIe) and is optimized for NVIDIA GPUs.",
            estimated_impact="~2-10x faster collective operations",
            confidence="high",
            doc_url="https://pytorch.org/docs/stable/distributed.html#backends",
            evidence={"backend": "gloo"},
        ))

    return results


# ---------------------------------------------------------------------------
# THRU001: cudnn.benchmark not enabled
# ---------------------------------------------------------------------------

def rule_thru001_cudnn_benchmark(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """cudnn.benchmark not enabled for training.

    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    if not findings.has_training_loop:
        return []

    has_benchmark = any(m.kind == "cudnn_benchmark" for m in findings.models)
    if has_benchmark:
        return []

    return [Diagnosis(
        rule_id="THRU001",
        severity="info",
        category="throughput",
        title="cudnn.benchmark not enabled",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value="torch.backends.cudnn.benchmark not set",
        suggested_value="torch.backends.cudnn.benchmark = True",
        suggested_code="torch.backends.cudnn.benchmark = True",
        rationale="cudnn.benchmark auto-tunes convolution algorithms for fixed-size inputs.",
        estimated_impact="~5-10% speedup for models with convolutions and fixed input sizes",
        confidence="low",
        doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
        evidence={"has_training_loop": "True"},
    )]


# ---------------------------------------------------------------------------
# MEM001: Enable gradient checkpointing (runtime-aware)
# ---------------------------------------------------------------------------

def rule_mem001_gradient_checkpointing(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Enable gradient checkpointing when VRAM > 80%.

    Ref: https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
    """
    if artifact is None or artifact.per_gpu_vram_used_mb is None:
        return []

    total_vram = artifact.per_gpu_vram_total_mb or 0
    if total_vram == 0:
        return []

    max_vram_used = max(artifact.per_gpu_vram_used_mb)
    utilization = max_vram_used / total_vram
    if utilization < 0.80:
        return []

    # Check if gradient checkpointing already enabled in AST
    for model in findings.models:
        if model.kind == "gradient_checkpointing":
            return []

    vram_pct = round(utilization * 100, 1)
    peak_gb = round(max_vram_used / 1024, 1)
    total_gb = round(total_vram / 1024, 1)
    headroom_gb = round((total_vram - max_vram_used) / 1024, 1)

    return [Diagnosis(
        rule_id="MEM001",
        severity="warning",
        category="memory",
        title="Enable gradient checkpointing",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"VRAM at {vram_pct}% ({peak_gb} GB / {total_gb} GB)",
        suggested_value="model.gradient_checkpointing_enable()",
        suggested_code="model.gradient_checkpointing_enable()",
        rationale=f"Peak VRAM {peak_gb} GB / {total_gb} GB — only {headroom_gb} GB headroom. Gradient checkpointing trades ~10-15% compute for ~40% activation memory reduction.",
        estimated_impact="~40% activation memory reduction, ~10-15% slower per step",
        confidence="high",
        doc_url="https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing",
        evidence={"vram_used_mb": str(round(max_vram_used)), "vram_total_mb": str(round(total_vram)), "vram_pct": str(vram_pct), "headroom_gb": str(headroom_gb)},
    )]


# ---------------------------------------------------------------------------
# MEM003: Batch size can increase (runtime-aware)
# ---------------------------------------------------------------------------

def rule_mem003_batch_increase(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Batch size can likely increase when VRAM utilization < 50%.

    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    if artifact is None or artifact.per_gpu_vram_used_mb is None:
        return []

    total_vram = artifact.per_gpu_vram_total_mb or 0
    if total_vram == 0:
        return []

    max_vram_used = max(artifact.per_gpu_vram_used_mb)
    utilization = max_vram_used / total_vram
    if utilization >= 0.50:
        return []

    vram_pct = round(utilization * 100, 1)
    peak_gb = round(max_vram_used / 1024, 1)
    total_gb = round(total_vram / 1024, 1)
    free_gb = round((total_vram - max_vram_used) / 1024, 1)

    # Find batch_size in DataLoader
    batch_size = None
    target_line = 0
    target_file = findings.script_path
    for dl in findings.dataloaders:
        if dl.batch_size is not None:
            batch_size = dl.batch_size
            target_line = dl.location.line_number
            target_file = dl.location.file_path
            break

    current = f"VRAM at {vram_pct}% ({peak_gb} GB / {total_gb} GB), {free_gb} GB free"
    if batch_size:
        current += f", batch_size={batch_size}"

    ev = {"vram_used_mb": str(round(max_vram_used)), "vram_total_mb": str(round(total_vram)), "vram_pct": str(vram_pct), "free_gb": str(free_gb)}
    if batch_size is not None:
        ev["batch_size"] = str(batch_size)

    return [Diagnosis(
        rule_id="MEM003",
        severity="info",
        category="memory",
        title="VRAM underutilized — batch size can likely increase",
        file_path=target_file,
        line_number=target_line,
        current_code="",
        current_value=current,
        suggested_value="Increase batch_size to use available VRAM",
        suggested_code="",
        rationale=f"Only {vram_pct}% VRAM used ({free_gb} GB free). Increasing batch size improves GPU utilization and throughput.",
        estimated_impact="~10-30% throughput improvement from better GPU utilization",
        confidence="medium",
        doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
        evidence=ev,
    )]


# ---------------------------------------------------------------------------
# MEM004: Batch size too large (runtime-aware)
# ---------------------------------------------------------------------------

def rule_mem004_batch_too_large(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Batch size may be too large when VRAM > 90% (not OOM yet).

    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    if artifact is None or artifact.per_gpu_vram_used_mb is None:
        return []

    total_vram = artifact.per_gpu_vram_total_mb or 0
    if total_vram == 0:
        return []

    max_vram_used = max(artifact.per_gpu_vram_used_mb)
    utilization = max_vram_used / total_vram
    if utilization < 0.90 or utilization > 0.99:
        return []  # >99% is OOM territory, handled by MEM001/OOM autopsy

    # Check if gradient checkpointing already enabled
    has_gc = any(m.kind == "gradient_checkpointing" for m in findings.models)

    vram_pct = round(utilization * 100, 1)
    peak_gb = round(max_vram_used / 1024, 1)
    total_gb = round(total_vram / 1024, 1)
    headroom_gb = round((total_vram - max_vram_used) / 1024, 1)

    suggestion = "Reduce batch_size or enable gradient checkpointing"
    if has_gc:
        suggestion = "Reduce batch_size — gradient checkpointing already enabled"

    return [Diagnosis(
        rule_id="MEM004",
        severity="warning",
        category="memory",
        title="VRAM near capacity — OOM risk",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"VRAM at {vram_pct}% ({peak_gb} GB / {total_gb} GB)",
        suggested_value=suggestion,
        suggested_code="",
        rationale=f"Peak VRAM {vram_pct}% with only {headroom_gb} GB headroom. Dynamic allocations or batch size variation could trigger OOM.",
        estimated_impact="Prevents OOM crashes during training",
        confidence="high",
        doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
        evidence={"vram_used_mb": str(round(max_vram_used)), "vram_total_mb": str(round(total_vram)), "vram_pct": str(vram_pct), "headroom_gb": str(headroom_gb), "has_gradient_checkpointing": str(has_gc)},
    )]


# ---------------------------------------------------------------------------
# DIST001: Consider FSDP over DDP (runtime-aware)
# ---------------------------------------------------------------------------

def rule_dist001_consider_fsdp(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Consider FSDP over DDP for large models.

    DDP replicates the full model on each GPU. For models > 1B params,
    FSDP shards model parameters across GPUs, reducing per-GPU memory.

    Ref: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
    """
    # Must have DDP detected in AST
    has_ddp = any(d.kind == "ddp" for d in findings.distributed)
    if not has_ddp:
        return []

    # Already using FSDP — no need to suggest
    has_fsdp = any(d.kind == "fsdp" for d in findings.distributed)
    if has_fsdp:
        return []

    # Check VRAM pressure from artifact
    high_vram = False
    if artifact and artifact.per_gpu_vram_used_mb and artifact.per_gpu_vram_total_mb:
        max_used = max(artifact.per_gpu_vram_used_mb)
        utilization = max_used / artifact.per_gpu_vram_total_mb
        high_vram = utilization > 0.70

    # Without runtime data, only fire if we can estimate model is large
    if not high_vram:
        # Check for large model hints (from_pretrained with known large models)
        has_large_model = False
        for model in findings.models:
            if model.kind == "from_pretrained" and model.model_name:
                name = model.model_name.lower()
                if any(s in name for s in ("70b", "13b", "65b", "40b", "34b", "mixtral")):
                    has_large_model = True
                    break
        if not has_large_model:
            return []

    ddp_finding = next(d for d in findings.distributed if d.kind == "ddp")
    rationale = "DDP replicates the full model on each GPU. FSDP shards parameters, gradients, and optimizer states across GPUs."
    if high_vram and artifact:
        vram_pct = round(max(artifact.per_gpu_vram_used_mb) / artifact.per_gpu_vram_total_mb * 100, 1)
        rationale += f" VRAM at {vram_pct}% — FSDP can reduce per-GPU memory by 2-4x."

    ev = {"current_strategy": "DDP"}
    if high_vram and artifact and artifact.per_gpu_vram_used_mb and artifact.per_gpu_vram_total_mb:
        ev["vram_pct"] = str(round(max(artifact.per_gpu_vram_used_mb) / artifact.per_gpu_vram_total_mb * 100, 1))
        ev["vram_used_mb"] = str(round(max(artifact.per_gpu_vram_used_mb)))

    return [Diagnosis(
        rule_id="DIST001",
        severity="info",
        category="distributed",
        title="Consider FSDP over DDP for memory efficiency",
        file_path=ddp_finding.location.file_path,
        line_number=ddp_finding.location.line_number,
        current_code=ddp_finding.location.source_line,
        current_value="DistributedDataParallel",
        suggested_value="FullyShardedDataParallel (FSDP)",
        suggested_code=ddp_finding.location.source_line.replace("DistributedDataParallel", "FullyShardedDataParallel"),
        rationale=rationale,
        estimated_impact="~2-4x per-GPU memory reduction, enables larger models/batches",
        confidence="medium",
        doc_url="https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html",
        evidence=ev,
    )]


# ---------------------------------------------------------------------------
# DIST003: Add gradient accumulation (runtime-aware)
# ---------------------------------------------------------------------------

def rule_dist003_gradient_accumulation(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Suggest gradient accumulation when GPU is underutilized with small batch.

    Ref: https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation
    """
    if artifact is None:
        return []

    # Need GPU utilization data
    avg_util = artifact.avg_gpu_util
    if avg_util is None or avg_util >= 60:
        return []

    # Check for small batch size in AST
    batch_size = None
    for dl in findings.dataloaders:
        if dl.batch_size is not None:
            batch_size = dl.batch_size
            break

    if batch_size is None or batch_size >= 32:
        return []  # Reasonable batch size, underutilization likely elsewhere

    return [Diagnosis(
        rule_id="DIST003",
        severity="info",
        category="distributed",
        title="Consider gradient accumulation",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"batch_size={batch_size}, GPU utilization {avg_util:.0f}%",
        suggested_value=f"Accumulate gradients over {max(2, 32 // batch_size)} steps for effective batch of ~32",
        suggested_code="",
        rationale=f"GPU utilization is {avg_util:.0f}% with batch_size={batch_size}. Gradient accumulation simulates a larger effective batch size without increasing memory.",
        estimated_impact="~15-25% training speedup from better GPU utilization",
        confidence="medium",
        doc_url="https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation",
        evidence={"batch_size": str(batch_size), "avg_gpu_util_pct": str(round(avg_util))},
    )]


# ---------------------------------------------------------------------------
# THRU003: Reduce gradient accumulation steps (runtime-aware)
# ---------------------------------------------------------------------------

def rule_thru003_reduce_grad_accum(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Reduce gradient accumulation steps when throughput is low.

    When grad_accum is detected in the AST and throughput seems low
    relative to GPU utilization, suggest reducing accumulation steps
    to decrease wall-clock time per effective step.

    Ref: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    if artifact is None:
        return []

    # Need throughput data
    sps = artifact.throughput_samples_per_sec
    avg_util = artifact.avg_gpu_util
    if sps is None or avg_util is None:
        return []

    # Only fire if GPU utilization is low (suggesting wasted cycles)
    if avg_util >= 70:
        return []

    # Check for gradient accumulation via AST-extracted field
    has_grad_accum = findings.gradient_accumulation_steps is not None

    if not has_grad_accum:
        return []

    return [Diagnosis(
        rule_id="THRU003",
        severity="info",
        category="throughput",
        title="Consider reducing gradient accumulation steps",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"Throughput {sps:.1f} samples/sec, GPU utilization {avg_util:.0f}%",
        suggested_value="Reduce gradient_accumulation_steps if VRAM allows larger batch",
        suggested_code="",
        rationale=f"Throughput is {sps:.1f} samples/sec with {avg_util:.0f}% GPU utilization. If VRAM allows, increasing batch_size and reducing accumulation steps decreases wall-clock time.",
        estimated_impact="~10-20% throughput improvement",
        confidence="low",
        doc_url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
        evidence={"throughput_sps": str(round(sps, 1)), "avg_gpu_util_pct": str(round(avg_util)), "gradient_accumulation_steps": str(findings.gradient_accumulation_steps)},
    )]


# ---------------------------------------------------------------------------
# XS004: Mixed precision confirmed effective (cross-signal, positive)
# ---------------------------------------------------------------------------

def rule_xs004_mixed_precision_effective(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Mixed precision is working effectively.

    Positive finding: AST shows autocast AND runtime GPU utilization is high.

    Ref: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
    """
    if artifact is None:
        return []

    avg_util = artifact.avg_gpu_util
    if avg_util is None or avg_util < 85:
        return []

    has_autocast = any(p.kind == "autocast" for p in findings.precision)
    if not has_autocast:
        return []

    return [Diagnosis(
        rule_id="XS004",
        severity="info",
        category="precision",
        title="Mixed precision is working effectively",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"autocast enabled, GPU utilization {avg_util:.0f}%",
        suggested_value="No change needed",
        suggested_code="",
        rationale=f"Mixed precision (autocast) is enabled and GPU utilization is {avg_util:.0f}% — the tensor cores are well utilized.",
        estimated_impact="Already optimized for this aspect",
        confidence="high",
        doc_url="https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html",
        evidence={"avg_gpu_util_pct": str(round(avg_util)), "has_autocast": "True"},
    )]


# ---------------------------------------------------------------------------
# STRAG001: Straggler detection (multi-GPU)
# ---------------------------------------------------------------------------

def rule_strag001_straggler(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Straggler GPU detected — one rank significantly slower than others.

    Fires when any rank's step time is >20% above the median.
    Step time is the correct signal for stragglers — VRAM differences
    indicate uneven sharding, which is a separate issue.
    20% is a standard distributed systems threshold for straggler detection.

    Ref: https://pytorch.org/docs/stable/distributed.html
    """
    if artifact is None:
        return []

    # Step times are the correct straggler signal (not VRAM)
    if artifact.per_rank_step_times_ms is not None and len(artifact.per_rank_step_times_ms) >= 2:
        return _check_step_time_straggler(findings, artifact)

    # No step time data available — cannot detect stragglers
    return []


def _check_step_time_straggler(
    findings: CodeFindings,
    artifact: ArtifactData,
) -> List[Diagnosis]:
    """Check for step time stragglers across ranks."""
    def _median(vals):
        """Proper median: average of two middle elements for even-length lists."""
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    rank_medians = []
    for rank_idx, times in enumerate(artifact.per_rank_step_times_ms):
        if not times:
            continue  # Skip ranks with missing data instead of aborting
        median_t = _median(times)
        rank_medians.append((rank_idx, median_t))

    if not rank_medians:
        return []

    all_medians = [m for _, m in rank_medians]
    overall_median = _median(all_medians)
    if overall_median == 0:
        return []

    stragglers = []
    for rank_idx, median_t in rank_medians:
        ratio = median_t / overall_median
        if ratio > 1.20:
            stragglers.append((rank_idx, median_t, ratio))

    if not stragglers:
        return []

    worst_rank, worst_time, worst_ratio = max(stragglers, key=lambda x: x[2])
    pct_slower = round((worst_ratio - 1.0) * 100, 0)

    ev = {
        "straggler_rank": str(worst_rank),
        "straggler_step_time_ms": str(round(worst_time, 1)),
        "median_step_time_ms": str(round(overall_median, 1)),
        "pct_slower": str(round(pct_slower)),
        "num_ranks": str(len(rank_medians)),
    }

    return [Diagnosis(
        rule_id="STRAG001",
        severity="warning",
        category="distributed",
        title=f"Straggler detected — rank {worst_rank} is {pct_slower:.0f}% slower",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"Rank {worst_rank}: {worst_time:.1f} ms/step (p50), median: {overall_median:.1f} ms/step",
        suggested_value="Investigate rank imbalance: thermal throttling, data skew, or NVLink asymmetry",
        suggested_code="",
        rationale=f"Rank {worst_rank} step time (p50) is {pct_slower:.0f}% slower than the median ({worst_time:.1f} ms vs {overall_median:.1f} ms). In synchronized training, the slowest rank determines overall throughput. Possible causes: thermal throttling, data skew, NVLink topology asymmetry.",
        estimated_impact="Resolving straggler can improve overall throughput by ~{:.0f}%".format(min(pct_slower, 30)),
        confidence="medium",
        doc_url="https://pytorch.org/docs/stable/distributed.html",
        evidence=ev,
    )]


# ---------------------------------------------------------------------------
# REG001: Step time regression
# ---------------------------------------------------------------------------

def rule_reg001_step_time_regression(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
    compare_artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Step time regression — p50 increased > 10% between runs.

    Ref: General ML training performance monitoring best practice.
    """
    if artifact is None or compare_artifact is None:
        return []

    cur_p50 = artifact.step_time_p50_ms
    prev_p50 = compare_artifact.step_time_p50_ms
    if cur_p50 is None or prev_p50 is None or prev_p50 == 0:
        return []

    pct_change = ((cur_p50 - prev_p50) / prev_p50) * 100
    if pct_change <= 10:
        return []

    return [Diagnosis(
        rule_id="REG001",
        severity="warning",
        category="regression",
        title="Step time regression detected",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"p50: {prev_p50:.1f} ms → {cur_p50:.1f} ms (+{pct_change:.0f}%)",
        suggested_value="Investigate: batch_size change, new model layers, or data pipeline change",
        suggested_code="",
        rationale=f"Step time p50 increased {pct_change:.0f}% ({prev_p50:.1f} ms → {cur_p50:.1f} ms) between runs.",
        estimated_impact=f"Training is ~{pct_change:.0f}% slower than previous run",
        confidence="high",
        doc_url=None,
        evidence={"prev_step_time_p50_ms": str(round(prev_p50, 1)), "cur_step_time_p50_ms": str(round(cur_p50, 1)), "pct_change": str(round(pct_change))},
    )]


# ---------------------------------------------------------------------------
# REG002: Throughput regression
# ---------------------------------------------------------------------------

def rule_reg002_throughput_regression(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
    compare_artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """Throughput regression — samples/sec decreased > 10%."""
    if artifact is None or compare_artifact is None:
        return []

    cur = artifact.throughput_samples_per_sec
    prev = compare_artifact.throughput_samples_per_sec
    if cur is None or prev is None or prev == 0:
        return []

    pct_change = ((cur - prev) / prev) * 100
    if pct_change >= -10:
        return []

    drop = abs(pct_change)
    return [Diagnosis(
        rule_id="REG002",
        severity="warning",
        category="regression",
        title="Throughput regression detected",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"Throughput: {prev:.1f} → {cur:.1f} samples/sec (-{drop:.0f}%)",
        suggested_value="Investigate: data pipeline, model changes, or hardware degradation",
        suggested_code="",
        rationale=f"Throughput decreased {drop:.0f}% ({prev:.1f} → {cur:.1f} samples/sec) between runs.",
        estimated_impact=f"Training is ~{drop:.0f}% slower than previous run",
        confidence="high",
        doc_url=None,
        evidence={"prev_throughput_sps": str(round(prev, 1)), "cur_throughput_sps": str(round(cur, 1)), "pct_drop": str(round(drop))},
    )]


# ---------------------------------------------------------------------------
# REG003: VRAM regression
# ---------------------------------------------------------------------------

def rule_reg003_vram_regression(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
    compare_artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """VRAM usage increased > 20% between runs."""
    if artifact is None or compare_artifact is None:
        return []

    def _peak(a):
        if a.per_gpu_vram_used_mb:
            return max(a.per_gpu_vram_used_mb)
        return a.peak_vram_mb

    cur = _peak(artifact)
    prev = _peak(compare_artifact)
    if cur is None or prev is None or prev == 0:
        return []

    pct_change = ((cur - prev) / prev) * 100
    if pct_change <= 20:
        return []

    return [Diagnosis(
        rule_id="REG003",
        severity="info",
        category="regression",
        title="VRAM usage increased",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"Peak VRAM: {prev:.0f} MB → {cur:.0f} MB (+{pct_change:.0f}%)",
        suggested_value="Check for batch_size increase, new layers, or memory leak",
        suggested_code="",
        rationale=f"Peak VRAM increased {pct_change:.0f}% ({prev:.0f} MB → {cur:.0f} MB) between runs.",
        estimated_impact="Increased OOM risk if trend continues",
        confidence="medium",
        doc_url=None,
        evidence={"prev_peak_vram_mb": str(round(prev)), "cur_peak_vram_mb": str(round(cur)), "pct_change": str(round(pct_change))},
    )]


# ---------------------------------------------------------------------------
# REG004: GPU utilization regression
# ---------------------------------------------------------------------------

def rule_reg004_util_regression(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
    compare_artifact: Optional[ArtifactData] = None,
) -> List[Diagnosis]:
    """GPU utilization decreased > 15% between runs."""
    if artifact is None or compare_artifact is None:
        return []

    cur = artifact.avg_gpu_util
    prev = compare_artifact.avg_gpu_util
    if cur is None or prev is None or prev == 0:
        return []

    # Absolute percentage point drop
    drop = prev - cur
    if drop <= 15:
        return []

    return [Diagnosis(
        rule_id="REG004",
        severity="info",
        category="regression",
        title="GPU utilization decreased",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value=f"GPU util: {prev:.0f}% → {cur:.0f}% (-{drop:.0f}pp)",
        suggested_value="Investigate: data pipeline slowdown, increased synchronization, or batch size change",
        suggested_code="",
        rationale=f"GPU utilization dropped {drop:.0f} percentage points ({prev:.0f}% → {cur:.0f}%) between runs.",
        estimated_impact=f"GPU is {drop:.0f}% less utilized than previous run",
        confidence="medium",
        doc_url=None,
        evidence={"prev_gpu_util_pct": str(round(prev)), "cur_gpu_util_pct": str(round(cur)), "drop_pp": str(round(drop))},
    )]


# ---------------------------------------------------------------------------
# UPG001: Deprecated GradScaler API
# ---------------------------------------------------------------------------

def rule_upg001_deprecated_grad_scaler(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using deprecated torch.cuda.amp.GradScaler API.

    torch.cuda.amp.GradScaler is deprecated since PyTorch 2.4.
    Use torch.amp.GradScaler("cuda") instead.

    Ref: https://pytorch.org/docs/stable/amp.html#torch.amp.GradScaler
    """
    results = []
    for prec in findings.precision:
        if prec.kind != "grad_scaler" or not prec.is_deprecated_api:
            continue

        suggested_code = prec.location.source_line
        suggested_code = suggested_code.replace(
            "torch.cuda.amp.GradScaler", "torch.amp.GradScaler"
        )
        # Add "cuda" device arg if GradScaler() has no args
        if "GradScaler()" in suggested_code:
            suggested_code = suggested_code.replace(
                'GradScaler()', 'GradScaler("cuda")'
            )

        results.append(Diagnosis(
            rule_id="UPG001",
            severity="info",
            category="upgrade",
            title="Deprecated GradScaler API",
            file_path=prec.location.file_path,
            line_number=prec.location.line_number,
            current_code=prec.location.source_line,
            current_value="torch.cuda.amp.GradScaler",
            suggested_value='torch.amp.GradScaler("cuda")',
            suggested_code=suggested_code,
            rationale="torch.cuda.amp.GradScaler is deprecated since PyTorch 2.4. Use torch.amp.GradScaler with explicit device argument.",
            estimated_impact="Future-proofs code — no performance change",
            confidence="high",
            doc_url="https://pytorch.org/docs/stable/amp.html#torch.amp.GradScaler",
            evidence={"api": "torch.cuda.amp.GradScaler", "deprecated_since": "PyTorch 2.4"},
        ))

    return results


# ---------------------------------------------------------------------------
# UPG002: tf32 matmul precision not set on Ampere+
# ---------------------------------------------------------------------------

def rule_upg002_tf32_matmul(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """TF32 matmul precision not explicitly enabled on Ampere+ hardware.

    On SM >= 8.0, torch.set_float32_matmul_precision('high') enables TF32
    for float32 matmuls, giving ~3x speedup with minimal precision loss.
    PyTorch 2.0+ defaults to 'highest' (no TF32) unless explicitly set.

    Ref: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    """
    if not hw:
        return []

    sm = hw.get("sm_version", "")
    if not sm:
        return []

    try:
        sm_major = int(str(sm).split(".")[0])
    except (ValueError, IndexError):
        return []

    if sm_major < 8:
        return []

    # Check if already set in code
    for model in findings.models:
        if model.kind == "float32_matmul_precision":
            return []

    # Check for it in distributed findings (some patterns set it there)
    # Also check raw imports/AST for the call
    # Actually, we need to check if set_float32_matmul_precision is called anywhere
    for dist in findings.distributed:
        if dist.kind == "float32_matmul_precision":
            return []

    # Only fire if there's a training loop (otherwise not relevant)
    if not findings.has_training_loop and not any(m.kind == "from_pretrained" for m in findings.models):
        return []

    # Check if using mixed precision — TF32 still helps for fp32 operations in mixed precision
    gpu_name = hw.get("gpu_name", "GPU")

    return [Diagnosis(
        rule_id="UPG002",
        severity="info",
        category="upgrade",
        title="TF32 matmul precision not set",
        file_path=findings.script_path,
        line_number=0,
        current_code="",
        current_value="Default float32 matmul precision (highest — no TF32)",
        suggested_value="torch.set_float32_matmul_precision('high')",
        suggested_code="torch.set_float32_matmul_precision('high')",
        rationale=f"{gpu_name} supports TF32 — enabling it gives ~3x float32 matmul speedup with <0.1% precision difference. Useful even with mixed precision for fp32 residual operations.",
        estimated_impact="~10-30% speedup for fp32 operations on Ampere+ GPUs",
        confidence="medium",
        doc_url="https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html",
        evidence={"sm_version": str(sm), "gpu_name": gpu_name, "feature": "TF32"},
    )]


# ---------------------------------------------------------------------------
# UPG003: DeepSpeed on single node — consider native FSDP
# ---------------------------------------------------------------------------

def rule_upg003_deepspeed_single_node(
    findings: CodeFindings,
    hw: Optional[Dict] = None,
) -> List[Diagnosis]:
    """Using DeepSpeed on single-node training where native FSDP may suffice.

    DeepSpeed adds complexity (config files, version pinning) that native
    PyTorch FSDP avoids. For single-node training, FSDP provides equivalent
    memory efficiency with fewer dependencies.

    Ref: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
    """
    has_deepspeed = any(d.kind == "deepspeed" for d in findings.distributed)
    if not has_deepspeed:
        return []

    # Already using FSDP — no need to suggest
    has_fsdp = any(d.kind == "fsdp" for d in findings.distributed)
    if has_fsdp:
        return []

    # Only fire for single-node — multi-node DeepSpeed has legitimate advantages
    gpu_count = (hw or {}).get("gpu_count", 1) or 1
    if gpu_count > 8:
        return []  # Likely multi-node, DeepSpeed may be better

    ds_finding = next(d for d in findings.distributed if d.kind == "deepspeed")

    return [Diagnosis(
        rule_id="UPG003",
        severity="info",
        category="upgrade",
        title="Consider native FSDP over DeepSpeed",
        file_path=ds_finding.location.file_path,
        line_number=ds_finding.location.line_number,
        current_code=ds_finding.location.source_line,
        current_value="DeepSpeed (external dependency)",
        suggested_value="PyTorch FSDP (native, no extra dependencies)",
        suggested_code="# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP",
        rationale="For single-node training, native PyTorch FSDP provides equivalent memory sharding without DeepSpeed's config complexity and version pinning requirements.",
        estimated_impact="Fewer dependencies, simpler config — equivalent performance for single-node",
        confidence="low",
        doc_url="https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html",
        evidence={"current_strategy": "DeepSpeed", "gpu_count": str(gpu_count)},
    )]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

# Upgrade-related rule IDs — used by --upgrades filter
UPGRADE_RULE_IDS = {
    # Dedicated upgrade rules
    "UPG001", "UPG002", "UPG003",
    # Existing rules that detect upgrade opportunities
    "PREC001",  # deprecated autocast API
    "PREC002",  # fp16 → bf16 on Ampere+
    "MEM005",   # torch.compile not used
    "DIST005",  # DataParallel → DDP
}

ALL_RULES = [
    # Phase 1 rules (AST-only, 2-arg)
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
    rule_dist005_data_parallel,
    rule_thru001_cudnn_benchmark,
    rule_upg001_deprecated_grad_scaler,
    rule_upg002_tf32_matmul,
    rule_upg003_deepspeed_single_node,
    # Phase 2 rules (runtime-aware, 3-arg)
    rule_mem001_gradient_checkpointing,
    rule_mem003_batch_increase,
    rule_mem004_batch_too_large,
    rule_dist001_consider_fsdp,
    rule_dist003_gradient_accumulation,
    rule_thru003_reduce_grad_accum,
    # Cross-signal rules (AST + runtime, 3-arg)
    rule_xs004_mixed_precision_effective,
    # Straggler detection (3-arg)
    rule_strag001_straggler,
    # Regression rules (4-arg, requires --compare)
    rule_reg001_step_time_regression,
    rule_reg002_throughput_regression,
    rule_reg003_vram_regression,
    rule_reg004_util_regression,
]
