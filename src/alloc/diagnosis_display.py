"""Diagnosis display — Rich terminal, --diff, --json, --memory, --efficiency output.

Output modes:
  1. Default: Rich terminal with colored severity badges
  2. --diff: Unified diff output, pipeable to git apply
  3. --json: Machine-readable JSON for CI/CD integration
  4. --memory: VRAM breakdown (estimated)
  5. --efficiency: Step time decomposition
  6. --compare: Cross-run comparison table
  7. OOM autopsy: Triggered automatically on OOM detection
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

    # Tier label
    tier_label = _tier_label(result.tier)

    if total == 0:
        console.print(f"[green]alloc diagnose[/green] — no issues found in {script_name} {tier_label}")
        if result.hardware_context:
            gpu = result.hardware_context.get("gpu_name", "")
            if gpu:
                console.print(f"[dim]Hardware: {gpu}[/dim]")
        console.print()
        return

    console.print(f"[green]alloc diagnose[/green] — {total} finding{'s' if total != 1 else ''} in {script_name} {tier_label}")

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


def _tier_label(tier: str) -> str:
    """Return display label for analysis tier."""
    if tier == "runtime_distributed":
        return "[dim](runtime-enhanced, distributed)[/dim]"
    if tier == "runtime":
        return "[dim](runtime-enhanced)[/dim]"
    return "[dim](AST-only)[/dim]"


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
            rel_path = file_path

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

    data = {
        "version": result.version,
        "script": result.script,
        "artifact": result.artifact_path,
        "tier": result.tier,
        "hardware": hw,
        "findings": findings_list,
        "summary": result.summary,
    }

    if result.oom_detected:
        data["oom_detected"] = True
        if result.vram_breakdown:
            data["vram_breakdown"] = result.vram_breakdown

    if result.comparison:
        data["comparison"] = result.comparison

    if result.efficiency:
        data["efficiency"] = result.efficiency

    return data


# ---------------------------------------------------------------------------
# --memory: VRAM breakdown
# ---------------------------------------------------------------------------

def print_diagnose_memory(result: DiagnoseResult) -> None:
    """Print VRAM breakdown from artifact data."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
    except ImportError:
        _print_memory_plain(result)
        return

    vb = result.vram_breakdown
    if not vb:
        console.print("[yellow]VRAM breakdown requires a runtime artifact.[/yellow]")
        console.print("[dim]Run: alloc run python train.py && alloc diagnose train.py --memory[/dim]")
        return

    script_name = os.path.basename(result.script) if result.script else "unknown"
    console.print()
    console.print(f"[green]alloc diagnose[/green] — VRAM breakdown for {script_name} [dim](estimated)[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Component", style="dim", no_wrap=True)
    table.add_column("Estimated", justify="right", style="bold")
    table.add_column("% of Total", justify="right")
    table.add_column("", no_wrap=True)

    total_est = vb.get("total_estimated_gb")
    capacity = vb.get("hardware_capacity_gb")

    if vb.get("weights_gb") is not None:
        components = [
            ("Model weights", vb["weights_gb"]),
            ("Gradients", vb["gradients_gb"]),
            (f"Optimizer ({vb.get('optimizer_type', 'Adam')})", vb["optimizer_gb"]),
            ("Activations (est.)", vb["activations_gb"]),
            ("CUDA context", vb["cuda_context_gb"]),
        ]

        for name, gb in components:
            pct = (gb / total_est * 100) if total_est and total_est > 0 else 0
            bar = _bar(pct, 20)
            table.add_row(name, f"{gb:.2f} GB", f"{pct:.1f}%", bar)

        table.add_row("", "", "", "")
        table.add_row("[bold]Total estimated[/bold]", f"[bold]{total_est:.2f} GB[/bold]", "", "")
    else:
        peak_gb = vb.get("peak_vram_gb", 0)
        table.add_row("[bold]Peak VRAM[/bold]", f"[bold]{peak_gb:.2f} GB[/bold]", "", "")

    if capacity:
        table.add_row("Hardware capacity", f"{capacity:.1f} GB", "", "")
        headroom = vb.get("headroom_gb", 0)
        if headroom is not None:
            label = "tight!" if headroom < 2 else ""
            table.add_row("Headroom", f"{headroom:.2f} GB", "", f"[yellow]{label}[/yellow]" if label else "")

    console.print(table)
    console.print()

    # Tips based on breakdown
    if vb.get("activations_gb") and vb["activations_gb"] > 0:
        console.print("[dim]Tip: gradient_checkpointing_enable() would reduce activations by ~60%[/dim]")
    console.print()


def _print_memory_plain(result: DiagnoseResult) -> None:
    """Plain-text VRAM breakdown."""
    vb = result.vram_breakdown
    if not vb:
        print("VRAM breakdown requires a runtime artifact.")
        return

    print("\n  VRAM breakdown (estimated)")
    print(f"  {'─' * 40}")

    if vb.get("weights_gb") is not None:
        print(f"  Model weights:      {vb['weights_gb']:>8.2f} GB")
        print(f"  Gradients:          {vb['gradients_gb']:>8.2f} GB")
        print(f"  Optimizer:          {vb['optimizer_gb']:>8.2f} GB")
        print(f"  Activations (est.): {vb['activations_gb']:>8.2f} GB")
        print(f"  CUDA context:       {vb['cuda_context_gb']:>8.2f} GB")
        print(f"  {'─' * 40}")
        print(f"  Total estimated:    {vb['total_estimated_gb']:>8.2f} GB")
    else:
        print(f"  Peak VRAM:          {vb.get('peak_vram_gb', 0):>8.2f} GB")

    capacity = vb.get("hardware_capacity_gb")
    if capacity:
        print(f"  Hardware capacity:  {capacity:>8.1f} GB")
        headroom = vb.get("headroom_gb", 0)
        if headroom is not None:
            print(f"  Headroom:           {headroom:>8.2f} GB")
    print()


def _bar(pct: float, width: int = 20) -> str:
    """Build a simple bar chart string."""
    filled = int(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "[green]" + "█" * filled + "[/green]" + "░" * (width - filled)


# ---------------------------------------------------------------------------
# --efficiency: Step time decomposition
# ---------------------------------------------------------------------------

def print_diagnose_efficiency(result: DiagnoseResult) -> None:
    """Print step time efficiency decomposition."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
    except ImportError:
        _print_efficiency_plain(result)
        return

    eff = result.efficiency
    if not eff:
        console.print("[yellow]Efficiency breakdown requires step timing data from a runtime artifact.[/yellow]")
        console.print("[dim]Run: alloc run python train.py && alloc diagnose train.py --efficiency[/dim]")
        return

    script_name = os.path.basename(result.script) if result.script else "unknown"
    console.print()
    console.print(f"[green]alloc diagnose[/green] — efficiency breakdown for {script_name} [dim](estimated)[/dim]")
    console.print()

    p50 = eff["step_time_p50_ms"]
    console.print(f"  Step time (p50): {p50:.1f} ms")
    console.print()

    # Visual bar
    compute_w = int(eff["compute_pct"] / 100 * 48)
    data_w = int(eff["data_loading_pct"] / 100 * 48)
    comm_w = int(eff["communication_pct"] / 100 * 48)
    other_w = max(0, 48 - compute_w - data_w - comm_w)

    bar = (
        "[green]" + "█" * compute_w + "[/green]"
        + "[yellow]" + "█" * data_w + "[/yellow]"
        + "[blue]" + "█" * comm_w + "[/blue]"
        + "[dim]" + "░" * other_w + "[/dim]"
    )
    console.print(f"  {bar}")
    label = f"  [green]Compute: {eff['compute_pct']:.0f}%[/green]"
    if eff["data_loading_pct"] > 0:
        label += f"  [yellow]Data: {eff['data_loading_pct']:.0f}%[/yellow]"
    if eff["communication_pct"] > 0:
        label += f"  [blue]Comm: {eff['communication_pct']:.0f}%[/blue]"
    if eff["other_pct"] > 0:
        label += f"  [dim]Other: {eff['other_pct']:.0f}%[/dim]"
    console.print(label)
    console.print()

    # Component table
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Component", style="dim", no_wrap=True)
    table.add_column("Time (est.)", justify="right", style="bold")
    table.add_column("Notes", style="dim")

    table.add_row("GPU compute", f"{eff['compute_ms']:.1f} ms", f"{eff['compute_pct']:.0f}% of step")
    if eff["data_loading_pct"] > 0:
        dl_note = f"{eff['data_loading_pct']:.0f}%"
        if eff["data_loading_pct"] > 20:
            dl_note += " — bottleneck candidate"
        table.add_row("Data loading", f"{eff['data_loading_ms']:.1f} ms", dl_note)
    if eff["communication_pct"] > 0:
        comm_note = f"{eff['communication_pct']:.0f}%"
        if eff["communication_pct"] < 15:
            comm_note += " — normal"
        table.add_row("Communication", f"{eff['communication_ms']:.1f} ms", comm_note)
    if eff["other_pct"] > 0:
        table.add_row("Other/overhead", f"{eff['other_ms']:.1f} ms", f"{eff['other_pct']:.0f}%")

    console.print(table)
    console.print()

    # Bottleneck summary
    bn = eff.get("bottleneck")
    if bn:
        console.print(f"  [bold]Bottleneck:[/bold] {bn}")
        rule = eff.get("bottleneck_rule")
        if rule:
            console.print(f"  [dim]See rule {rule} above for fix suggestions[/dim]")
    console.print()


def _print_efficiency_plain(result: DiagnoseResult) -> None:
    """Plain-text efficiency breakdown."""
    eff = result.efficiency
    if not eff:
        print("Efficiency breakdown requires step timing data.")
        return

    print(f"\n  Efficiency breakdown (estimated)")
    print(f"  Step time (p50): {eff['step_time_p50_ms']:.1f} ms")
    print(f"  {'─' * 40}")
    print(f"  GPU compute:     {eff['compute_ms']:>8.1f} ms  ({eff['compute_pct']:.0f}%)")
    if eff["data_loading_pct"] > 0:
        print(f"  Data loading:    {eff['data_loading_ms']:>8.1f} ms  ({eff['data_loading_pct']:.0f}%)")
    if eff["communication_pct"] > 0:
        print(f"  Communication:   {eff['communication_ms']:>8.1f} ms  ({eff['communication_pct']:.0f}%)")
    bn = eff.get("bottleneck")
    if bn:
        print(f"\n  Bottleneck: {bn}")
    print()


# ---------------------------------------------------------------------------
# OOM Autopsy
# ---------------------------------------------------------------------------

def print_oom_autopsy(result: DiagnoseResult) -> None:
    """Print OOM autopsy analysis."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
    except ImportError:
        _print_oom_plain(result)
        return

    script_name = os.path.basename(result.script) if result.script else "unknown"
    console.print()
    console.print(f"[bold red]alloc diagnose[/bold red] — OOM autopsy for {script_name}")
    console.print()

    vb = result.vram_breakdown
    if vb:
        capacity = vb.get("hardware_capacity_gb")
        peak_gb = vb.get("peak_vram_gb") or vb.get("total_estimated_gb")

        if capacity and peak_gb:
            pct = (peak_gb / capacity) * 100
            console.print(f"  Last recorded VRAM: {peak_gb:.1f} GB / {capacity:.1f} GB ({pct:.1f}%)")

        if vb.get("weights_gb") is not None:
            console.print()
            console.print("  [bold]Estimated VRAM breakdown:[/bold]")

            components = [
                ("Model weights", vb["weights_gb"]),
                ("Gradients", vb["gradients_gb"]),
                (f"Optimizer ({vb.get('optimizer_type', 'Adam')})", vb["optimizer_gb"]),
                ("Activations (est.)", vb["activations_gb"]),
                ("CUDA context", vb["cuda_context_gb"]),
            ]
            total = sum(gb for _, gb in components)

            for name, gb in components:
                console.print(f"    {name:<24} ~{gb:.1f} GB")
            console.print(f"    {'─' * 36}")
            console.print(f"    [bold]Total estimated          ~{total:.1f} GB[/bold]")

            if capacity and total > capacity:
                console.print(f"    [red]Exceeds {capacity:.0f} GB capacity by ~{total - capacity:.1f} GB[/red]")

    # Suggested fixes
    console.print()
    console.print("  [bold]Suggested fixes (in priority order):[/bold]")
    suggestions = [
        ("Enable gradient checkpointing", "saves ~40% activation memory"),
        ("Enable mixed precision (bf16/fp16)", "halves model + gradient memory"),
        ("Switch optimizer to 8-bit Adam", "saves ~75% optimizer memory"),
        ("Enable FSDP", "shards across GPUs"),
        ("Reduce batch_size", "reduces activation memory"),
    ]
    for i, (fix, impact) in enumerate(suggestions, 1):
        console.print(f"   {i}. {fix:<40} [dim]→ {impact}[/dim]")

    console.print()


def _print_oom_plain(result: DiagnoseResult) -> None:
    """Plain-text OOM autopsy."""
    script_name = os.path.basename(result.script) if result.script else "unknown"
    print(f"\n  OOM autopsy for {script_name}")
    print(f"  {'─' * 50}")

    vb = result.vram_breakdown
    if vb and vb.get("weights_gb") is not None:
        print(f"  Model weights:      ~{vb['weights_gb']:.1f} GB")
        print(f"  Gradients:          ~{vb['gradients_gb']:.1f} GB")
        print(f"  Optimizer:          ~{vb['optimizer_gb']:.1f} GB")
        print(f"  Activations (est.): ~{vb['activations_gb']:.1f} GB")

    print("\n  Suggested: gradient checkpointing, mixed precision, reduce batch_size")
    print()


# ---------------------------------------------------------------------------
# --compare: Cross-run comparison
# ---------------------------------------------------------------------------

def print_diagnose_comparison(result: DiagnoseResult) -> None:
    """Print cross-run comparison table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        _print_comparison_plain(result)
        return

    comp = result.comparison
    if not comp:
        console.print("[yellow]No comparison data available.[/yellow]")
        return

    script_name = os.path.basename(result.script) if result.script else "unknown"
    console.print()
    console.print(f"[green]alloc diagnose[/green] — comparison: {script_name}")
    console.print()

    metrics = comp.get("metrics", [])
    if not metrics:
        console.print("[dim]No comparable metrics found between artifacts.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Metric", style="dim", no_wrap=True)
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Change", justify="right")

    for m in metrics:
        before = _fmt_metric(m["before"], m.get("unit", ""))
        after = _fmt_metric(m["after"], m.get("unit", ""))
        pct = m["pct_change"]
        is_reg = m.get("is_regression", False)

        if pct > 0:
            change = f"+{pct:.1f}%"
            if is_reg:
                change += " [red]▲ regression[/red]"
            else:
                change += " ▲"
        elif pct < 0:
            change = f"{pct:.1f}%"
            if is_reg:
                change += " [red]▼ regression[/red]"
            else:
                change += " ▼"
        else:
            change = "—"

        table.add_row(m["name"], before, after, change)

    console.print(table)
    console.print()

    # Show regression findings
    reg_findings = [f for f in result.findings if f.category == "regression"]
    if reg_findings:
        for diag in reg_findings:
            badge = _severity_badge(diag.severity)
            console.print(f"{badge} {diag.rule_id} — {diag.title}")
            console.print(f"  [dim]{diag.rationale}[/dim]")
            console.print()


def _print_comparison_plain(result: DiagnoseResult) -> None:
    """Plain-text comparison."""
    comp = result.comparison
    if not comp:
        print("No comparison data available.")
        return

    print("\n  Cross-run comparison")
    print(f"  {'─' * 50}")

    for m in comp.get("metrics", []):
        before = _fmt_metric_plain(m["before"], m.get("unit", ""))
        after = _fmt_metric_plain(m["after"], m.get("unit", ""))
        pct = m["pct_change"]
        tag = " REGRESSION" if m.get("is_regression") else ""
        print(f"  {m['name']:<20} {before:>12} -> {after:>12}  ({pct:+.1f}%){tag}")
    print()


def _fmt_metric(value, unit: str) -> str:
    """Format a metric value with unit for Rich output."""
    if isinstance(value, float):
        if value >= 1000:
            return f"{value:,.0f} {unit}".strip()
        return f"{value:.1f} {unit}".strip()
    return f"{value} {unit}".strip()


def _fmt_metric_plain(value, unit: str) -> str:
    """Format a metric value for plain text."""
    if isinstance(value, float):
        if value >= 1000:
            return f"{value:,.0f} {unit}".strip()
        return f"{value:.1f} {unit}".strip()
    return f"{value} {unit}".strip()
