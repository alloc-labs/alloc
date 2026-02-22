"""Diagnosis display — Rich terminal, --diff, --json output for diagnose results.

Three output modes:
  1. Default: Rich terminal with colored severity badges
  2. --diff: Unified diff output, pipeable to git apply
  3. --json: Machine-readable JSON for CI/CD integration
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from alloc.diagnosis_engine import DiagnoseResult
from alloc.diagnosis_rules import Diagnosis


# ---------------------------------------------------------------------------
# Default: Rich terminal output
# ---------------------------------------------------------------------------

def print_diagnose_rich(result: DiagnoseResult, *, verbose: bool = False) -> None:
    """Print diagnosis result with Rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        _print_rich(console, result, verbose=verbose)
    except ImportError:
        _print_plain(result)


def _print_rich(console, result: DiagnoseResult, *, verbose: bool = False) -> None:
    """Rich terminal output."""
    script_name = os.path.basename(result.script) if result.script else "unknown"
    total = result.summary.get("total", 0)

    if result.parse_errors:
        for err in result.parse_errors:
            console.print(f"[yellow]Warning: {err}[/yellow]")
        if total == 0:
            console.print("[dim]No findings (script could not be fully analyzed).[/dim]")
            return

    console.print()

    if total == 0:
        console.print(f"[green]alloc diagnose[/green] — no issues found in {script_name}")
        if result.hardware_context:
            gpu = result.hardware_context.get("gpu_name", "")
            if gpu:
                console.print(f"[dim]Hardware: {gpu}[/dim]")
        console.print()
        return

    console.print(f"[green]alloc diagnose[/green] — {total} finding{'s' if total != 1 else ''} in {script_name}")

    if result.hardware_context:
        gpu = result.hardware_context.get("gpu_name", "")
        count = result.hardware_context.get("gpu_count", 1)
        sm = result.hardware_context.get("sm_version", "")
        hw_parts = []
        if gpu:
            hw_parts.append(gpu)
        if count and count > 1:
            hw_parts.append(f"x{count}")
        if sm:
            hw_parts.append(f"SM {sm}")
        if hw_parts:
            console.print(f"[dim]Hardware: {' '.join(hw_parts)}[/dim]")

    console.print()

    for diag in result.findings:
        _print_finding_rich(console, diag, verbose=verbose)

    # Summary line
    parts = []
    for sev in ("critical", "warning", "info"):
        count = result.summary.get(sev, 0)
        if count > 0:
            parts.append(f"{count} {sev}")
    console.print(f"[bold]Summary:[/bold] {', '.join(parts)}")
    console.print("[dim]Run with --diff to generate patches | --json for CI output[/dim]")
    console.print()


def _severity_badge(severity: str) -> str:
    """Rich-formatted severity badge."""
    if severity == "critical":
        return "[bold red] CRITICAL [/bold red]"
    if severity == "warning":
        return "[bold yellow] WARNING  [/bold yellow]"
    return "[bold blue]  INFO    [/bold blue]"


def _print_finding_rich(console, diag: Diagnosis, *, verbose: bool = False) -> None:
    """Print a single finding with Rich formatting."""
    badge = _severity_badge(diag.severity)
    console.print(f"{badge} {diag.rule_id} — {diag.title}")

    if diag.line_number > 0:
        file_name = os.path.basename(diag.file_path) if diag.file_path else ""
        console.print(f"  [dim]{file_name}:{diag.line_number}[/dim]   {diag.current_value} [dim]→[/dim] {diag.suggested_value}")
    else:
        console.print(f"  {diag.current_value} [dim]→[/dim] {diag.suggested_value}")

    console.print(f"  [dim]{diag.rationale}[/dim]")
    console.print(f"  [dim]Expected impact: {diag.estimated_impact}[/dim]")

    if verbose:
        console.print(f"  [dim]Confidence: {diag.confidence}[/dim]")
        if diag.doc_url:
            console.print(f"  [dim]Ref: {diag.doc_url}[/dim]")

    console.print()


def _print_plain(result: DiagnoseResult) -> None:
    """Plain-text fallback when Rich is not available."""
    script_name = os.path.basename(result.script) if result.script else "unknown"
    total = result.summary.get("total", 0)

    if total == 0:
        print(f"alloc diagnose — no issues found in {script_name}")
        return

    print(f"alloc diagnose — {total} finding{'s' if total != 1 else ''} in {script_name}")
    print()

    for diag in result.findings:
        sev = diag.severity.upper()
        print(f"  [{sev}] {diag.rule_id} — {diag.title}")
        if diag.line_number > 0:
            print(f"    {os.path.basename(diag.file_path)}:{diag.line_number}  {diag.current_value} -> {diag.suggested_value}")
        print(f"    {diag.rationale}")
        print(f"    Impact: {diag.estimated_impact}")
        print()

    parts = []
    for sev in ("critical", "warning", "info"):
        count = result.summary.get(sev, 0)
        if count > 0:
            parts.append(f"{count} {sev}")
    print(f"Summary: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# --diff mode: Unified diff output
# ---------------------------------------------------------------------------

def print_diagnose_diff(result: DiagnoseResult) -> None:
    """Print unified diff patches for all findings with code fixes."""
    patches = _build_patches(result)
    if not patches:
        print("# No code patches available.")
        return

    for patch in patches:
        print(patch)


def _build_patches(result: DiagnoseResult) -> List[str]:
    """Build unified diff strings from diagnose findings."""
    # Group findings by file
    by_file = {}  # type: Dict[str, List[Diagnosis]]
    for diag in result.findings:
        if not diag.file_path or diag.line_number <= 0:
            continue
        if not diag.current_code or not diag.suggested_code:
            continue
        if diag.current_code.strip() == diag.suggested_code.strip():
            continue
        by_file.setdefault(diag.file_path, []).append(diag)

    patches = []
    for file_path, diags in sorted(by_file.items()):
        # Sort by line number
        diags.sort(key=lambda d: d.line_number)

        for diag in diags:
            line = diag.line_number
            rel_path = os.path.basename(file_path)

            patch_lines = [
                f"--- a/{rel_path}",
                f"+++ b/{rel_path}",
                f"@@ -{line},1 +{line},1 @@",
                f"-{diag.current_code}",
                f"+{diag.suggested_code}",
            ]
            patches.append("\n".join(patch_lines))

    return patches


# ---------------------------------------------------------------------------
# --json mode: Machine-readable JSON
# ---------------------------------------------------------------------------

def print_diagnose_json(result: DiagnoseResult) -> None:
    """Print diagnosis result as JSON."""
    data = build_json_dict(result)
    print(json.dumps(data, indent=2, default=str))


def build_json_dict(result: DiagnoseResult) -> dict:
    """Build a JSON-serializable dict from DiagnoseResult."""
    findings_list = []
    for d in result.findings:
        findings_list.append({
            "rule_id": d.rule_id,
            "severity": d.severity,
            "category": d.category,
            "title": d.title,
            "file_path": d.file_path,
            "line_number": d.line_number,
            "current_value": d.current_value,
            "suggested_value": d.suggested_value,
            "rationale": d.rationale,
            "estimated_impact": d.estimated_impact,
            "confidence": d.confidence,
            "doc_url": d.doc_url,
        })

    hw = None
    if result.hardware_context:
        hw = {
            "gpu_name": result.hardware_context.get("gpu_name"),
            "gpu_count": result.hardware_context.get("gpu_count"),
            "sm_version": result.hardware_context.get("sm_version"),
        }

    return {
        "version": result.version,
        "script": result.script,
        "artifact": result.artifact_path,
        "hardware": hw,
        "findings": findings_list,
        "summary": result.summary,
    }
