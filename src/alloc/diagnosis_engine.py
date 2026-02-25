"""Diagnosis engine — rule evaluation, hardware detection, result assembly.

Orchestrates: CodeFindings -> rules -> dedup -> sort -> DiagnoseResult.
Supports Phase 1 (AST-only) and Phase 2 (runtime-enhanced) modes.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from alloc.artifact_loader import ArtifactData
from alloc.code_analyzer import CodeFindings, analyze_script
from alloc.diagnosis_rules import ALL_RULES, Diagnosis


@dataclass
class DiagnoseResult:
    version: str = "2"
    script: str = ""
    artifact_path: Optional[str] = None
    findings: List[Diagnosis] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    hardware_context: Optional[Dict] = None
    parse_errors: List[str] = field(default_factory=list)
    # Phase 2 fields
    tier: str = "ast_only"                  # "ast_only" | "runtime" | "runtime_distributed"
    oom_detected: bool = False
    comparison: Optional[Dict] = None       # metric deltas if --compare used
    vram_breakdown: Optional[Dict] = None   # for OOM autopsy / --memory
    efficiency: Optional[Dict] = None       # for --efficiency


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def run_diagnosis(
    findings: CodeFindings,
    *,
    hardware_context: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
    compare_artifact: Optional[ArtifactData] = None,
) -> DiagnoseResult:
    """Run all diagnosis rules against code findings.

    Args:
        findings: Output of analyze_script()
        hardware_context: Optional hardware info {gpu_name, gpu_count, sm_version, ...}
        artifact: Optional runtime artifact data (Phase 2)
        compare_artifact: Optional second artifact for cross-run comparison

    Returns:
        DiagnoseResult with sorted, deduplicated findings.
    """
    all_diags = []  # type: List[Diagnosis]

    for rule_fn in ALL_RULES:
        try:
            # Check compare first — compare rules are a superset (they also have "artifact")
            if _rule_accepts_compare(rule_fn):
                diags = rule_fn(findings, hardware_context, artifact, compare_artifact)
            elif _rule_accepts_artifact(rule_fn):
                diags = rule_fn(findings, hardware_context, artifact)
            else:
                diags = rule_fn(findings, hardware_context)
            all_diags.extend(diags)
        except Exception:
            logger.debug("Rule %s raised an exception", rule_fn.__name__, exc_info=True)
            continue

    # Deduplicate overlapping rules
    all_diags = _deduplicate(all_diags)

    # Sort: critical > warning > info, then by line number
    all_diags.sort(key=lambda d: (_SEVERITY_ORDER.get(d.severity, 99), d.line_number))

    summary = {"critical": 0, "warning": 0, "info": 0, "total": len(all_diags)}
    for d in all_diags:
        if d.severity in summary:
            summary[d.severity] += 1

    # Determine tier
    tier = _determine_tier(artifact)

    # Build comparison if compare_artifact provided
    comparison = None
    if compare_artifact is not None and artifact is not None:
        comparison = _build_comparison(artifact, compare_artifact)

    # Build VRAM breakdown if artifact available
    vram_breakdown = None
    if artifact is not None:
        vram_breakdown = _build_vram_breakdown(findings, artifact)

    # Build efficiency breakdown if artifact has timing data
    efficiency = None
    if artifact is not None and artifact.step_time_p50_ms is not None:
        efficiency = _build_efficiency(artifact)

    return DiagnoseResult(
        version="2",
        script=findings.script_path,
        artifact_path=None,
        findings=all_diags,
        summary=summary,
        hardware_context=hardware_context,
        parse_errors=findings.parse_errors,
        tier=tier,
        oom_detected=artifact.is_oom if artifact else False,
        comparison=comparison,
        vram_breakdown=vram_breakdown,
        efficiency=efficiency,
    )


def diagnose_file(
    script_path: str,
    *,
    hardware_context: Optional[Dict] = None,
    artifact: Optional[ArtifactData] = None,
) -> DiagnoseResult:
    """Convenience: analyze script + run diagnosis in one call."""
    findings = analyze_script(script_path)
    hw = hardware_context
    if hw is None:
        hw = _quick_hw_context()
    return run_diagnosis(findings, hardware_context=hw, artifact=artifact)


def _rule_accepts_artifact(rule_fn) -> bool:
    """Check if a rule function accepts an artifact parameter (3-arg form)."""
    try:
        sig = inspect.signature(rule_fn)
        params = list(sig.parameters.keys())
        return len(params) >= 3 and "artifact" in params
    except (ValueError, TypeError):
        return False


def _rule_accepts_compare(rule_fn) -> bool:
    """Check if a rule function accepts a compare_artifact parameter (4-arg form)."""
    try:
        sig = inspect.signature(rule_fn)
        params = list(sig.parameters.keys())
        return "compare_artifact" in params
    except (ValueError, TypeError):
        return False


def _determine_tier(artifact: Optional[ArtifactData]) -> str:
    """Determine analysis tier based on available data."""
    if artifact is None:
        return "ast_only"

    # Check for per-rank distributed data
    if artifact.per_rank_peak_vram_mb and len(artifact.per_rank_peak_vram_mb) > 1:
        return "runtime_distributed"

    # Has artifact but single GPU
    return "runtime"


def _build_comparison(current: ArtifactData, previous: ArtifactData) -> Dict:
    """Build metric comparison between two artifacts."""
    metrics = []  # type: List[Dict]

    def _add(name, cur_val, prev_val, unit="", higher_is_worse=True):
        if cur_val is None or prev_val is None:
            return
        if prev_val == 0:
            return
        pct_change = ((cur_val - prev_val) / abs(prev_val)) * 100
        is_regression = (pct_change > 10) if higher_is_worse else (pct_change < -10)
        metrics.append({
            "name": name,
            "before": prev_val,
            "after": cur_val,
            "unit": unit,
            "pct_change": round(pct_change, 1),
            "is_regression": is_regression,
        })

    # Peak VRAM
    cur_peak = max(current.per_gpu_vram_used_mb) if current.per_gpu_vram_used_mb else current.peak_vram_mb
    prev_peak = max(previous.per_gpu_vram_used_mb) if previous.per_gpu_vram_used_mb else previous.peak_vram_mb
    _add("Peak VRAM", cur_peak, prev_peak, "MB", higher_is_worse=True)

    # Step time
    _add("Step time (p50)", current.step_time_p50_ms, previous.step_time_p50_ms, "ms", higher_is_worse=True)

    # GPU utilization (lower is worse)
    _add("GPU utilization", current.avg_gpu_util, previous.avg_gpu_util, "%", higher_is_worse=False)

    # Throughput (lower is worse)
    _add("Throughput", current.throughput_samples_per_sec, previous.throughput_samples_per_sec, "samples/s", higher_is_worse=False)

    return {
        "metrics": metrics,
        "has_regressions": any(m["is_regression"] for m in metrics),
    }


def _build_vram_breakdown(findings: CodeFindings, artifact: ArtifactData) -> Optional[Dict]:
    """Estimate VRAM component breakdown from AST + runtime data."""
    peak = artifact.peak_vram_mb
    # Fall back to max per-GPU VRAM if peak is not available
    if peak is None and artifact.per_gpu_vram_used_mb:
        peak = max(artifact.per_gpu_vram_used_mb)
    total = artifact.per_gpu_vram_total_mb
    if peak is None:
        return None

    # Try to estimate model size from AST findings
    model_params_b = None
    dtype_bytes = 2  # default bf16/fp16

    for model in findings.models:
        if model.kind == "from_pretrained" and model.model_name:
            model_params_b = _estimate_model_params(model.model_name)
            break

    # Detect dtype from precision findings
    for prec in findings.precision:
        if prec.dtype_str in ("float32", "fp32"):
            dtype_bytes = 4
        elif prec.dtype_str in ("bfloat16", "bf16", "float16", "fp16"):
            dtype_bytes = 2

    # Detect optimizer
    optimizer_factor = 8  # Adam default: 2 states × 4 bytes each
    for opt in findings.optimizers:
        if opt.optimizer_type == "SGD":
            optimizer_factor = 4  # momentum only

    if model_params_b is not None:
        weights_gb = model_params_b * dtype_bytes
        gradients_gb = model_params_b * dtype_bytes
        optimizer_gb = model_params_b * optimizer_factor
        cuda_context_gb = 1.5
        known_gb = weights_gb + gradients_gb + optimizer_gb + cuda_context_gb
        peak_gb = peak / 1024.0
        activations_gb = max(0.0, peak_gb - known_gb)

        return {
            "weights_gb": round(weights_gb, 2),
            "gradients_gb": round(gradients_gb, 2),
            "optimizer_gb": round(optimizer_gb, 2),
            "activations_gb": round(activations_gb, 2),
            "cuda_context_gb": cuda_context_gb,
            "total_estimated_gb": round(weights_gb + gradients_gb + optimizer_gb + activations_gb + cuda_context_gb, 2),
            "hardware_capacity_gb": round(total / 1024.0, 1) if total else None,
            "headroom_gb": round((total / 1024.0) - peak_gb, 2) if total else None,
            "model_params_b": model_params_b,
            "dtype_bytes": dtype_bytes,
            "optimizer_type": findings.optimizers[0].optimizer_type if findings.optimizers else "Adam",
        }

    # Fallback: no model size estimate, just show raw peak
    peak_gb = peak / 1024.0
    return {
        "weights_gb": None,
        "gradients_gb": None,
        "optimizer_gb": None,
        "activations_gb": None,
        "cuda_context_gb": 1.5,
        "total_estimated_gb": None,
        "hardware_capacity_gb": round(total / 1024.0, 1) if total else None,
        "headroom_gb": round((total / 1024.0) - peak_gb, 2) if total else None,
        "peak_vram_gb": round(peak_gb, 2),
    }


def _build_efficiency(artifact: ArtifactData) -> Optional[Dict]:
    """Build step time efficiency decomposition from artifact timing data.

    When CUDA event phase timing is available (has_phase_timing=True), uses
    definitive phase breakdown instead of estimated dataloader_wait_pct.
    """
    p50 = artifact.step_time_p50_ms
    if p50 is None or p50 <= 0:
        return None

    # Prefer CUDA event phase timing when available
    if artifact.has_phase_timing and artifact.phase_forward_ms_p50 is not None:
        fwd = artifact.phase_forward_ms_p50 or 0.0
        bwd = artifact.phase_backward_ms_p50 or 0.0
        opt = artifact.phase_optimizer_ms_p50 or 0.0
        dl = artifact.phase_dataloader_ms_p50 or 0.0
        total = fwd + bwd + opt + dl

        if total > 0:
            fwd_pct = (fwd / total) * 100.0
            bwd_pct = (bwd / total) * 100.0
            opt_pct = (opt / total) * 100.0
            dl_pct = (dl / total) * 100.0

            # Determine bottleneck from phase data
            bottleneck = None
            bottleneck_rule = None
            if dl_pct > 30:
                bottleneck = "Data loading"
                bottleneck_rule = "DL001"
            elif bwd_pct > 50:
                bottleneck = "Backward pass (may include communication)"
            elif opt_pct > 30:
                bottleneck = "Optimizer step"

            return {
                "step_time_p50_ms": round(p50, 1),
                "source": "cuda_events",
                "forward_pct": round(fwd_pct, 1),
                "forward_ms": round(fwd, 1),
                "backward_pct": round(bwd_pct, 1),
                "backward_ms": round(bwd, 1),
                "optimizer_pct": round(opt_pct, 1),
                "optimizer_ms": round(opt, 1),
                "data_loading_pct": round(dl_pct, 1),
                "data_loading_ms": round(dl, 1),
                "bottleneck": bottleneck,
                "bottleneck_rule": bottleneck_rule,
            }

    # Fallback: estimated from dataloader_wait_pct (wall-clock based)
    dl_pct = artifact.dataloader_wait_pct or 0.0
    other_pct = 0.0
    compute_pct = max(0.0, 100.0 - dl_pct - other_pct)

    dl_ms = p50 * dl_pct / 100.0
    compute_ms = p50 * compute_pct / 100.0
    other_ms = p50 * other_pct / 100.0

    # Determine bottleneck
    bottleneck = None
    bottleneck_rule = None
    if dl_pct > 20:
        bottleneck = "Data loading"
        bottleneck_rule = "DL001"
    elif compute_pct < 50 and dl_pct < 20:
        bottleneck = "Other overhead"

    return {
        "step_time_p50_ms": round(p50, 1),
        "source": "wall_clock",
        "compute_pct": round(compute_pct, 1),
        "compute_ms": round(compute_ms, 1),
        "data_loading_pct": round(dl_pct, 1),
        "data_loading_ms": round(dl_ms, 1),
        "other_pct": round(other_pct, 1),
        "other_ms": round(other_ms, 1),
        "bottleneck": bottleneck,
        "bottleneck_rule": bottleneck_rule,
    }


def _estimate_model_params(model_name: str) -> Optional[float]:
    """Rough estimate of model parameter count in billions from HF model name.

    Uses publicly known model sizes.
    Ref: https://huggingface.co/docs/transformers/model_doc/auto
    """
    name = model_name.lower()

    # Common models — public knowledge
    estimates = {
        # LLMs
        "gpt2": 0.124,
        "gpt2-medium": 0.355,
        "gpt2-large": 0.774,
        "gpt2-xl": 1.5,
        "llama-2-7b": 7.0,
        "llama-2-13b": 13.0,
        "llama-2-70b": 70.0,
        "llama-3-8b": 8.0,
        "llama-3-70b": 70.0,
        "mistral-7b": 7.0,
        "mixtral-8x7b": 46.7,
        "bert-base": 0.110,
        "bert-large": 0.340,
        "t5-small": 0.060,
        "t5-base": 0.220,
        "t5-large": 0.770,
        "t5-3b": 3.0,
        "t5-11b": 11.0,
        # Vision models
        "resnet-50": 0.026,
        "resnet-101": 0.045,
        "resnet-152": 0.060,
        "vit-base": 0.086,
        "vit-large": 0.307,
        "clip-vit-base": 0.151,
        "clip-vit-large": 0.428,
        "dinov2-base": 0.086,
        "dinov2-large": 0.307,
        # Diffusion / generative
        "stable-diffusion": 0.865,
        # Audio
        "whisper-small": 0.244,
        "whisper-medium": 0.769,
        "whisper-large": 1.55,
    }

    for key, params in estimates.items():
        if key in name:
            return params

    # Try to extract number from model name (e.g., "model-7b", "model-13b")
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    if match:
        try:
            val = float(match.group(1))
            # Filter: model param counts are typically 0.01-2000B.
            # Numbers like 50, 101, 224 are likely layer counts or image sizes
            # (e.g., resnet-50, efficientnet-b0).
            if 0.01 <= val <= 2000:
                return val
        except ValueError:
            pass

    return None


def _deduplicate(diags: List[Diagnosis]) -> List[Diagnosis]:
    """Remove overlapping/redundant diagnoses.

    Rules:
    - MEM002 (no mixed precision) is suppressed if any PREC rule fires on same file
    - DL001 + DL005 on same line: keep DL005 (more specific)
    """
    # Track which (file, line) combos have which rules
    file_line_rules = {}  # type: Dict[tuple, List[str]]
    for d in diags:
        key = (d.file_path, d.line_number)
        file_line_rules.setdefault(key, []).append(d.rule_id)

    # Build set of files that have precision rules
    prec_files = {d.file_path for d in diags if d.rule_id.startswith("PREC")}

    result = []
    for d in diags:
        # Suppress MEM002 only if a precision rule fires in the SAME file
        if d.rule_id == "MEM002" and d.file_path in prec_files:
            continue

        # DL001 + DL005 on same line: skip DL001
        key = (d.file_path, d.line_number)
        if d.rule_id == "DL001" and "DL005" in file_line_rules.get(key, []):
            continue

        result.append(d)

    return result


def _quick_hw_context() -> Optional[Dict]:
    """One-shot NVML query + environment sniffing for hardware info.

    Returns None only if NVML fails AND no env vars are set.
    """
    ctx = _sniff_environment()

    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram_mb = mem.total / (1024 * 1024)

            sm_version = None
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                sm_version = f"{major}.{minor}"
            except Exception:
                pass

            ctx.update({
                "gpu_name": name,
                "gpu_count": device_count,
                "gpu_total_vram_mb": total_vram_mb,
                "sm_version": sm_version,
            })
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    return ctx if ctx else None


def _sniff_environment() -> Dict:
    """Read environment variables that inform diagnosis without user config.

    These provide distributed training context, GPU selection, and tuning hints.
    """
    import os

    ctx = {}  # type: Dict

    # GPU selection
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        devices = [d.strip() for d in cvd.split(",") if d.strip()]
        ctx["cuda_visible_devices"] = devices
        ctx["gpu_count"] = len(devices)

    # Distributed training context
    world_size = os.environ.get("WORLD_SIZE", "").strip()
    if world_size:
        try:
            ctx["world_size"] = int(world_size)
        except ValueError:
            pass

    rank = os.environ.get("RANK", "").strip()
    if rank:
        try:
            ctx["rank"] = int(rank)
        except ValueError:
            pass

    local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "").strip()
    if local_world_size:
        try:
            ctx["local_world_size"] = int(local_world_size)
        except ValueError:
            pass

    master_addr = os.environ.get("MASTER_ADDR", "").strip()
    if master_addr:
        ctx["master_addr"] = master_addr

    # NCCL tuning (presence indicates distributed GPU training)
    nccl_keys = ["NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_P2P_LEVEL",
                 "NCCL_NET_GDR_LEVEL", "NCCL_ALGO", "NCCL_PROTO"]
    nccl_vars = {}
    for key in nccl_keys:
        val = os.environ.get(key, "").strip()
        if val:
            nccl_vars[key] = val
    if nccl_vars:
        ctx["nccl_config"] = nccl_vars

    # CPU threading (affects data loading perf)
    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp:
        try:
            ctx["omp_num_threads"] = int(omp)
        except ValueError:
            pass

    # CUDA memory allocator tuning
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if alloc_conf:
        ctx["pytorch_cuda_alloc_conf"] = alloc_conf

    return ctx
