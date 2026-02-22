"""Diagnosis engine — rule evaluation, hardware detection, result assembly.

Orchestrates: CodeFindings -> rules -> dedup -> sort -> DiagnoseResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alloc.code_analyzer import CodeFindings, analyze_script
from alloc.diagnosis_rules import ALL_RULES, Diagnosis


@dataclass
class DiagnoseResult:
    version: str = "1"
    script: str = ""
    artifact_path: Optional[str] = None
    findings: List[Diagnosis] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    hardware_context: Optional[Dict] = None
    parse_errors: List[str] = field(default_factory=list)


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def run_diagnosis(
    findings: CodeFindings,
    *,
    hardware_context: Optional[Dict] = None,
    artifact: Optional[Dict] = None,
) -> DiagnoseResult:
    """Run all diagnosis rules against code findings.

    Args:
        findings: Output of analyze_script()
        hardware_context: Optional hardware info {gpu_name, gpu_count, sm_version, ...}
        artifact: Optional runtime artifact data (Phase 2)

    Returns:
        DiagnoseResult with sorted, deduplicated findings.
    """
    all_diags = []  # type: List[Diagnosis]

    for rule_fn in ALL_RULES:
        try:
            diags = rule_fn(findings, hardware_context)
            all_diags.extend(diags)
        except Exception:
            # Rules must never crash the engine
            continue

    # Deduplicate overlapping rules
    all_diags = _deduplicate(all_diags)

    # Sort: critical > warning > info, then by line number
    all_diags.sort(key=lambda d: (_SEVERITY_ORDER.get(d.severity, 99), d.line_number))

    summary = {"critical": 0, "warning": 0, "info": 0, "total": len(all_diags)}
    for d in all_diags:
        if d.severity in summary:
            summary[d.severity] += 1

    return DiagnoseResult(
        version="1",
        script=findings.script_path,
        findings=all_diags,
        summary=summary,
        hardware_context=hardware_context,
        parse_errors=findings.parse_errors,
    )


def diagnose_file(
    script_path: str,
    *,
    hardware_context: Optional[Dict] = None,
) -> DiagnoseResult:
    """Convenience: analyze script + run diagnosis in one call."""
    findings = analyze_script(script_path)
    hw = hardware_context
    if hw is None:
        hw = _quick_hw_context()
    return run_diagnosis(findings, hardware_context=hw)


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

    has_prec_rule = any(d.rule_id.startswith("PREC") for d in diags)

    result = []
    for d in diags:
        # Suppress MEM002 if any precision rule fires
        if d.rule_id == "MEM002" and has_prec_rule:
            continue

        # DL001 + DL005 on same line: skip DL001
        key = (d.file_path, d.line_number)
        if d.rule_id == "DL001" and "DL005" in file_line_rules.get(key, []):
            continue

        result.append(d)

    return result


def _quick_hw_context() -> Optional[Dict]:
    """One-shot NVML query for hardware info. Returns None on failure."""
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

            return {
                "gpu_name": name,
                "gpu_count": device_count,
                "gpu_total_vram_mb": total_vram_mb,
                "sm_version": sm_version,
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
