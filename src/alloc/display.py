"""Rich terminal formatting for Alloc reports."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from alloc.ghost import GhostReport
    from alloc.probe import ProbeResult


def print_ghost_report(report: GhostReport) -> None:
    """Print a GhostReport to terminal with Rich formatting."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Component", style="dim")
        table.add_column("Size", justify="right", style="bold")

        table.add_row("Model weights", f"{report.weights_gb:.2f} GB")
        table.add_row("Gradients", f"{report.gradients_gb:.2f} GB")
        table.add_row("Optimizer (Adam)", f"{report.optimizer_gb:.2f} GB")
        table.add_row("Activations (est.)", f"{report.activations_gb:.2f} GB")
        table.add_row("Buffer (10%)", f"{report.buffer_gb:.2f} GB")
        table.add_row("", "")
        table.add_row("[bold]Total VRAM[/bold]", f"[bold green]{report.total_gb:.2f} GB[/bold green]")

        header = f"Ghost Scan — {report.param_count_b:.1f}B params ({report.dtype})"
        confidence_label = _ghost_confidence_label(getattr(report, "extraction_method", None))
        console.print()
        console.print(Panel(table, title=header, border_style="green", padding=(1, 2)))
        console.print(f"  [dim]Confidence: {confidence_label}[/dim]")
        console.print()
    except ImportError:
        # Fallback without rich
        _print_ghost_plain(report)


def _print_ghost_plain(report: GhostReport) -> None:
    """Plain-text fallback when Rich is not available."""
    confidence_label = _ghost_confidence_label(getattr(report, "extraction_method", None))
    print(f"\n  Ghost Scan — {report.param_count_b:.1f}B params ({report.dtype})")
    print(f"  {'─' * 40}")
    print(f"  Model weights:      {report.weights_gb:>8.2f} GB")
    print(f"  Gradients:          {report.gradients_gb:>8.2f} GB")
    print(f"  Optimizer (Adam):   {report.optimizer_gb:>8.2f} GB")
    print(f"  Activations (est.): {report.activations_gb:>8.2f} GB")
    print(f"  Buffer (10%):       {report.buffer_gb:>8.2f} GB")
    print(f"  {'─' * 40}")
    print(f"  Total VRAM:         {report.total_gb:>8.2f} GB")
    print(f"\n  Confidence: {confidence_label}\n")


def print_probe_result(result: ProbeResult) -> None:
    """Print a ProbeResult to terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right", style="bold")

        table.add_row("Peak VRAM", f"{result.peak_vram_mb:.0f} MB ({result.peak_vram_gb:.2f} GB)")
        table.add_row("Avg GPU Utilization", f"{result.avg_gpu_util:.1f}%")
        table.add_row("Avg Power Draw", f"{result.avg_power_watts:.0f} W")
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")
        table.add_row("Samples", f"{len(result.samples)}")

        if result.exit_code is not None:
            status = "[green]success[/green]" if result.exit_code == 0 else f"[red]exit {result.exit_code}[/red]"
            table.add_row("Process", status)

        console.print()
        console.print(Panel(table, title="Alloc Probe Results", border_style="blue", padding=(1, 2)))
        console.print()
    except ImportError:
        print(f"\n  Alloc Probe Results")
        print(f"  Peak VRAM: {result.peak_vram_mb:.0f} MB")
        print(f"  Avg GPU Util: {result.avg_gpu_util:.1f}%")
        print(f"  Duration: {result.duration_seconds:.1f}s\n")


# --- Run summary display for calibrate-and-exit mode ---


def print_verdict(result, artifact_path="", step_count=None, callback_data=None, budget_context=None):
    # type: (ProbeResult, str, Optional[int], Optional[dict], Optional[dict]) -> None
    """Print a run summary panel after calibration."""
    vram_util_pct = result.vram_utilization_pct
    duration_label = _stop_reason_label(result.stop_reason, result.calibration_duration_s)

    gpu_short = result.gpu_name or "Unknown GPU"
    if gpu_short and "-" in gpu_short:
        parts = gpu_short.split("-")
        if parts[-1].upper().endswith("GB"):
            gpu_short = "-".join(parts[:-1])

    if result.exit_code is None:
        proc_status = "unknown"
    elif result.exit_code == 0:
        proc_status = "success"
    else:
        proc_status = f"exit {result.exit_code}"

    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        lines = []

        peak_gb = result.peak_vram_mb / 1024
        if result.gpu_total_vram_mb:
            total_gb = result.gpu_total_vram_mb / 1024
            lines.append(f"  Peak VRAM       {peak_gb:.1f} GB / {total_gb:.1f} GB ({gpu_short})")
        else:
            lines.append(f"  Peak VRAM       {peak_gb:.1f} GB ({gpu_short})")

        if vram_util_pct is not None:
            lines.append(f"  VRAM used       {vram_util_pct:.1f}%")

        lines.append(f"  Avg GPU util    {result.avg_gpu_util:.1f}%")
        lines.append(f"  Avg power       {result.avg_power_watts:.0f} W")
        lines.append(f"  Duration        {result.duration_seconds:.1f}s ({duration_label})")
        lines.append(f"  Samples         {len(result.samples)}")
        lines.append(f"  Process         {proc_status}")

        if callback_data:
            p50 = callback_data.get("step_time_ms_p50")
            p90 = callback_data.get("step_time_ms_p90")
            sps = callback_data.get("samples_per_sec")
            if p50 is not None and p90 is not None:
                lines.append(f"  Step time       {p50:.1f} ms (p50) / {p90:.1f} ms (p90)")
            if sps is not None:
                lines.append(f"  Throughput      {sps:.1f} samples/sec")

        if budget_context:
            cph = budget_context.get("cost_per_hour")
            budget_mo = budget_context.get("budget_monthly")
            if cph is not None and cph > 0:
                monthly_est = cph * 176
                lines.append(f"  Est. monthly    ~${monthly_est:,.0f}/mo at current rate (8h/day)")
                if budget_mo is not None and budget_mo > 0:
                    pct = (monthly_est / budget_mo) * 100
                    cap_note = ""
                    if budget_context.get("budget_cap_applied"):
                        cap_note = " (org cap applied)"
                    lines.append(f"  Budget          {pct:.0f}% of ${budget_mo:,.0f}/mo{cap_note}")

        content = "\n".join(lines)
        console.print()
        console.print(Panel(content, title="Run Summary", border_style="blue", padding=(1, 0)))

        if artifact_path:
            console.print(f"  [dim]Artifact: {artifact_path}[/dim]")
        console.print()

    except ImportError:
        _print_verdict_plain(
            result, vram_util_pct,
            gpu_short, duration_label, proc_status,
            artifact_path,
        )


def _print_verdict_plain(
    result, vram_util_pct,
    gpu_short, duration_label, proc_status,
    artifact_path,
):
    # type: (...) -> None
    """Plain-text fallback for run summary."""
    print(f"\n  Run Summary")
    print(f"  {'─' * 50}")

    peak_gb = result.peak_vram_mb / 1024
    if result.gpu_total_vram_mb:
        total_gb = result.gpu_total_vram_mb / 1024
        print(f"  Peak VRAM       {peak_gb:.1f} GB / {total_gb:.1f} GB ({gpu_short})")
    else:
        print(f"  Peak VRAM       {peak_gb:.1f} GB ({gpu_short})")

    if vram_util_pct is not None:
        print(f"  VRAM used       {vram_util_pct:.1f}%")

    print(f"  Avg GPU util    {result.avg_gpu_util:.1f}%")
    print(f"  Avg power       {result.avg_power_watts:.0f} W")
    print(f"  Duration        {result.duration_seconds:.1f}s ({duration_label})")
    print(f"  Samples         {len(result.samples)}")
    print(f"  Process         {proc_status}")

    print(f"  {'─' * 50}")
    if artifact_path:
        print(f"  Artifact: {artifact_path}")
    print()


def _stop_reason_label(stop_reason, calibration_duration_s=None):
    # type: (Optional[str], Optional[float]) -> str
    """Human-readable label for stop reason."""
    if stop_reason == "stable":
        if calibration_duration_s is not None:
            return f"auto-stopped: metrics stable at {calibration_duration_s:.1f}s"
        return "auto-stopped: metrics stable"

    if stop_reason == "timeout":
        return "reached timeout — increase --timeout for more data"

    if stop_reason == "process_exit":
        return "training process exited"

    return "unknown"


def build_verdict_dict(result, artifact_path="", step_count=None, callback_data=None, budget_context=None):
    # type: (ProbeResult, str, Optional[int], Optional[dict], Optional[dict]) -> dict
    """Build a run summary dict for JSON output."""
    vram_util_pct = result.vram_utilization_pct
    duration_label = _stop_reason_label(result.stop_reason, result.calibration_duration_s)

    if result.exit_code is None:
        proc_status = "unknown"
    elif result.exit_code == 0:
        proc_status = "success"
    else:
        proc_status = f"exit {result.exit_code}"

    d = {
        "peak_vram_mb": round(result.peak_vram_mb, 1),
        "peak_vram_gb": round(result.peak_vram_mb / 1024, 2),
        "gpu_name": result.gpu_name,
        "gpu_total_vram_mb": result.gpu_total_vram_mb,
        "vram_utilization_pct": round(vram_util_pct, 1) if vram_util_pct is not None else None,
        "avg_gpu_util": round(result.avg_gpu_util, 1),
        "avg_power_watts": round(result.avg_power_watts, 0),
        "duration_seconds": round(result.duration_seconds, 1),
        "duration_label": duration_label,
        "sample_count": len(result.samples),
        "process_status": proc_status,
        "artifact_path": artifact_path or None,
    }
    if callback_data:
        for key in ("step_time_ms_p50", "step_time_ms_p90", "samples_per_sec"):
            val = callback_data.get(key)
            if val is not None:
                d[key] = val
    if budget_context:
        cph = budget_context.get("cost_per_hour")
        budget_mo = budget_context.get("budget_monthly")
        if cph is not None and cph > 0:
            monthly_est = cph * 176
            d["budget_projection"] = {
                "cost_per_hour": cph,
                "est_monthly": round(monthly_est, 2),
                "budget_monthly": budget_mo,
                "budget_pct": round((monthly_est / budget_mo) * 100, 1) if budget_mo and budget_mo > 0 else None,
                "budget_cap_applied": budget_context.get("budget_cap_applied", False),
                "org_budget_monthly": budget_context.get("org_budget_monthly"),
            }
    return d


def print_verbose_run(result, step_count=None):
    # type: (ProbeResult, Optional[int]) -> None
    """Print detailed probe data: hardware context and sample dump."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # --- Hardware Context ---
        hw_lines = []
        if result.gpu_name:
            hw_lines.append(f"  GPU              {result.gpu_name}")
        if result.gpu_total_vram_mb:
            hw_lines.append(f"  Total VRAM       {result.gpu_total_vram_mb:.0f} MB ({result.gpu_total_vram_mb / 1024:.1f} GB)")
        if result.driver_version:
            hw_lines.append(f"  Driver           {result.driver_version}")
        if result.cuda_version:
            hw_lines.append(f"  CUDA             {result.cuda_version}")
        if result.sm_version:
            hw_lines.append(f"  SM Compute       {result.sm_version}")
        hw_lines.append(f"  GPUs detected    {result.num_gpus_detected}")
        if result.probe_mode:
            hw_lines.append(f"  Probe mode       {result.probe_mode}")
        if result.stop_reason:
            hw_lines.append(f"  Stop reason      {result.stop_reason}")
        if step_count:
            hw_lines.append(f"  Step count       {step_count} (from framework callback)")

        if hw_lines:
            console.print(Panel("\n".join(hw_lines), title="Hardware Context", border_style="cyan", padding=(1, 0)))

        # --- Probe Samples ---
        if result.samples:
            table = Table(
                show_header=True, header_style="bold cyan", box=None,
                padding=(0, 1), title="Probe Samples",
            )
            table.add_column("#", style="dim", justify="right")
            table.add_column("Time (s)", justify="right")
            table.add_column("VRAM (MB)", justify="right")
            table.add_column("GPU Util %", justify="right")
            table.add_column("Power (W)", justify="right")

            for i, s in enumerate(result.samples):
                table.add_row(
                    str(i),
                    f"{s.get('t', 0):.1f}",
                    f"{s.get('vram_mb', 0):.0f}",
                    f"{s.get('gpu_util_pct', 0):.1f}",
                    f"{s.get('power_w', 0):.0f}",
                )

            console.print()
            console.print(table)
            console.print()
    except ImportError:
        pass


def print_verbose_ghost(report):
    # type: (GhostReport) -> None
    """Print detailed VRAM formula breakdown."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        bytes_map = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2, "int8": 1}
        bpp = bytes_map.get(report.dtype, 2)

        lines = []
        lines.append(f"  Parameters      {report.param_count:,} ({report.param_count_b:.3f}B)")
        lines.append(f"  Dtype           {report.dtype} ({bpp} bytes/param)")
        lines.append("")
        lines.append(f"  Weights         {report.param_count_b:.3f}B x {bpp} bytes = {report.weights_gb:.2f} GB")
        lines.append(f"  Gradients       {report.param_count_b:.3f}B x {bpp} bytes = {report.gradients_gb:.2f} GB")
        lines.append(f"  Optimizer       {report.param_count_b:.3f}B x 12 bytes (Adam: fp32 params + momentum + variance) = {report.optimizer_gb:.2f} GB")
        lines.append(f"  Activations     batch_size x seq_len x hidden_dim x {bpp} bytes = {report.activations_gb:.2f} GB")
        subtotal = report.weights_gb + report.gradients_gb + report.optimizer_gb + report.activations_gb
        lines.append(f"  Buffer          10% x ({subtotal:.2f} GB) = {report.buffer_gb:.2f} GB")
        lines.append("")
        lines.append(f"  Total           {report.total_gb:.2f} GB")
        confidence_label = _ghost_confidence_label(getattr(report, "extraction_method", None))
        lines.append(f"  Confidence      {confidence_label}")

        console.print(Panel("\n".join(lines), title="VRAM Formula Breakdown", border_style="cyan", padding=(1, 0)))
    except ImportError:
        pass


def _ghost_confidence_label(extraction_method):
    # type: (Optional[str]) -> str
    """Return confidence label based on how model params were extracted."""
    if extraction_method == "execution":
        return "high (exact param count)"
    if extraction_method == "ast":
        return "medium (inferred from model name)"
    return "medium (static estimate)"
