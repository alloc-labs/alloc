"""Diagnosis rules — 12 pure-function rules for training script analysis.

Each rule: (CodeFindings, Optional[Dict]) -> List[Diagnosis]

All thresholds cite public PyTorch/NVIDIA documentation. No proprietary logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from alloc.code_analyzer import CodeFindings


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


# ---------------------------------------------------------------------------
# Helper: replace a keyword argument in a source line
# ---------------------------------------------------------------------------

def _replace_kwarg(source_line: str, name: str, old_val: str, new_val: str) -> str:
    """Replace name=old_val with name=new_val in source line."""
    pattern = rf'{name}\s*=\s*{re.escape(old_val)}'
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
        # Only fire if num_workers > 0 or not specified (default 0 makes pin_memory less useful)
        if dl.num_workers is not None and dl.num_workers == 0:
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

    dtype_suggestion = "torch.bfloat16"
    if sm:
        try:
            sm_major = int(str(sm).split(".")[0])
            if sm_major < 8:
                dtype_suggestion = "torch.float16"
        except (ValueError, IndexError):
            pass

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
            rationale=f"{gpu_name} supports bf16 natively — eliminates loss scaling overhead and avoids fp16 overflow issues.",
            estimated_impact="~5-10% speedup, eliminates GradScaler complexity",
            confidence="high",
            doc_url="https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html",
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

    # Check if any distributed pattern exists
    has_dist = any(d.kind in ("ddp", "fsdp", "init_process_group") for d in findings.distributed)
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
        estimated_impact=f"~{min(gpu_count, 8)}x throughput with linear scaling",
        confidence="medium",
        doc_url="https://pytorch.org/tutorials/intermediate/ddp_tutorial.html",
    )]


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
            rationale="GLOO backend doesn't support GPU-to-GPU communication. NCCL is optimized for NVIDIA GPUs.",
            estimated_impact="~2-10x faster collective operations",
            confidence="high",
            doc_url="https://pytorch.org/docs/stable/distributed.html#backends",
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
    )]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ALL_RULES = [
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
]
