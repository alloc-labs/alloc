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


# --- Verdict display for calibrate-and-exit mode ---


def print_verdict(result, artifact_path="", step_count=None, callback_data=None):
    # type: (ProbeResult, str, Optional[int], Optional[dict]) -> None
    """Print a verdict summary panel after calibration."""
    vram_util_pct = result.vram_utilization_pct
    bottleneck = _classify_bottleneck_local(
        result.peak_vram_mb,
        result.gpu_total_vram_mb,
        result.avg_gpu_util,
    )
    confidence = _compute_confidence_local(
        len(result.samples),
        result.duration_seconds,
        step_count,
        callback_data=callback_data,
    )
    recommendation = _qualitative_recommendation(bottleneck, vram_util_pct, result.avg_gpu_util)
    duration_label = _stop_reason_label(result.stop_reason, result.calibration_duration_s)

    # GPU label: shorten "NVIDIA A100-SXM4-80GB" → "NVIDIA A100-SXM4"
    gpu_short = result.gpu_name or "Unknown GPU"
    if gpu_short and "-" in gpu_short:
        parts = gpu_short.split("-")
        if parts[-1].upper().endswith("GB"):
            gpu_short = "-".join(parts[:-1])

    # Process status
    if result.exit_code is None:
        proc_status = "unknown"
    elif result.exit_code == 0:
        proc_status = "success"
    else:
        proc_status = f"exit {result.exit_code}"

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        color = _bottleneck_color(bottleneck)
        bottleneck_display = bottleneck.replace("_", " ").title()
        title = f"Verdict: {bottleneck_display} (confidence: {confidence:.2f})"

        lines = []

        # Peak VRAM line
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

        # Timing from framework callbacks
        if callback_data:
            p50 = callback_data.get("step_time_ms_p50")
            p90 = callback_data.get("step_time_ms_p90")
            sps = callback_data.get("samples_per_sec")
            dl_wait = callback_data.get("dataloader_wait_pct")
            if p50 is not None and p90 is not None:
                lines.append(f"  Step time       {p50:.1f} ms (p50) / {p90:.1f} ms (p90)")
            if sps is not None:
                lines.append(f"  Throughput      {sps:.1f} samples/sec")
            if dl_wait is not None and dl_wait > 15:
                lines.append(f"  Dataloader      ~{dl_wait:.0f}% wait (consider more workers)")

        if recommendation:
            lines.append("")
            lines.append(f"  Suggestion: {recommendation}")

        content = "\n".join(lines)
        console.print()
        console.print(Panel(content, title=title, border_style=color, padding=(1, 0)))

        if artifact_path:
            console.print(f"  [dim]Artifact: {artifact_path}[/dim]")
            console.print(f"  [dim]Next: alloc upload {artifact_path}[/dim]")
        console.print()

    except ImportError:
        _print_verdict_plain(
            result, bottleneck, confidence, vram_util_pct,
            gpu_short, duration_label, proc_status,
            recommendation, artifact_path,
        )


def _print_verdict_plain(
    result, bottleneck, confidence, vram_util_pct,
    gpu_short, duration_label, proc_status,
    recommendation, artifact_path,
):
    # type: (...) -> None
    """Plain-text fallback for verdict."""
    bottleneck_display = bottleneck.replace("_", " ").title()
    print(f"\n  Verdict: {bottleneck_display} (confidence: {confidence:.2f})")
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

    if recommendation:
        print(f"\n  Suggestion: {recommendation}")

    print(f"  {'─' * 50}")
    if artifact_path:
        print(f"  Artifact: {artifact_path}")
        print(f"  Next: alloc upload {artifact_path}")
    print()


def _classify_bottleneck_local(peak_vram_mb, gpu_total_vram_mb, avg_gpu_util):
    # type: (float, Optional[float], float) -> str
    """Classify workload bottleneck. Matches analyzer.py:116-132 thresholds."""
    compute_util = avg_gpu_util

    if gpu_total_vram_mb is None or gpu_total_vram_mb <= 0:
        # Degrade to util-only classification
        if compute_util > 80:
            return "compute_bound"
        if compute_util < 40:
            return "underutilized"
        return "unknown"

    vram_util = (peak_vram_mb / gpu_total_vram_mb) * 100

    if compute_util > 80:
        return "compute_bound"
    if compute_util < 40:
        if vram_util < 60:
            return "underutilized"
        if vram_util > 80:
            return "memory_bound"
    return "balanced"


def _compute_confidence_local(sample_count, duration_s, step_count=None, callback_data=None):
    # type: (int, float, Optional[int], Optional[dict]) -> float
    """Estimate analysis confidence. Matches analyzer.py confidence formula.

    Cap depends on signal level:
      NVML_ONLY (no callback timing) → max 0.6
      FRAMEWORK_TIMING (callback with step timing) → max 0.85
    """
    score = 0.3  # Baseline: we have probe data

    if sample_count >= 100:
        score += 0.3
    elif sample_count >= 20:
        score += 0.2
    elif sample_count >= 5:
        score += 0.1

    if duration_s >= 300:
        score += 0.2
    elif duration_s >= 60:
        score += 0.1

    if step_count and step_count > 0:
        score += 0.2

    # Determine signal level cap
    has_timing = (
        callback_data is not None
        and callback_data.get("step_time_ms_p50") is not None
    )
    cap = 0.85 if has_timing else 0.6
    return min(score, cap)


def _qualitative_recommendation(bottleneck, vram_utilization_pct, avg_gpu_util):
    # type: (str, Optional[float], float) -> Optional[str]
    """Return a human-readable suggestion based on bottleneck classification."""
    if bottleneck == "underutilized":
        if vram_utilization_pct is not None and vram_utilization_pct < 30:
            return "GPU is significantly oversized. Consider a smaller GPU."
        return "GPU is underutilized. Consider a smaller or fewer GPUs."

    if bottleneck == "memory_bound":
        return "Workload is memory-bound. Consider higher-bandwidth GPU or FSDP."

    if bottleneck == "compute_bound":
        if vram_utilization_pct is not None and vram_utilization_pct < 70:
            return "Compute-bound with VRAM headroom. Try increasing batch size."
        return None

    # balanced / unknown
    return None


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

    if stop_reason == "probe_steps":
        return "probe-steps limit reached"

    return "unknown"


def build_verdict_dict(result, artifact_path="", step_count=None, callback_data=None):
    # type: (ProbeResult, str, Optional[int], Optional[dict]) -> dict
    """Build a verdict dict with the same data as print_verdict, for JSON output."""
    vram_util_pct = result.vram_utilization_pct
    bottleneck = _classify_bottleneck_local(
        result.peak_vram_mb,
        result.gpu_total_vram_mb,
        result.avg_gpu_util,
    )
    confidence = _compute_confidence_local(
        len(result.samples),
        result.duration_seconds,
        step_count,
        callback_data=callback_data,
    )
    recommendation = _qualitative_recommendation(bottleneck, vram_util_pct, result.avg_gpu_util)
    duration_label = _stop_reason_label(result.stop_reason, result.calibration_duration_s)

    if result.exit_code is None:
        proc_status = "unknown"
    elif result.exit_code == 0:
        proc_status = "success"
    else:
        proc_status = f"exit {result.exit_code}"

    d = {
        "bottleneck": bottleneck,
        "confidence": round(confidence, 2),
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
        "recommendation": recommendation,
        "artifact_path": artifact_path or None,
    }
    # Include timing fields from callback data
    if callback_data:
        for key in ("step_time_ms_p50", "step_time_ms_p90", "samples_per_sec", "dataloader_wait_pct"):
            val = callback_data.get(key)
            if val is not None:
                d[key] = val
    return d


def print_verbose_run(result, step_count=None):
    # type: (ProbeResult, Optional[int]) -> None
    """Print detailed probe data: hardware context, sample dump, recommendation reasoning."""
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

        # --- Recommendation Reasoning ---
        vram_util_pct = result.vram_utilization_pct
        bottleneck = _classify_bottleneck_local(
            result.peak_vram_mb, result.gpu_total_vram_mb, result.avg_gpu_util,
        )
        confidence = _compute_confidence_local(
            len(result.samples), result.duration_seconds, step_count,
        )
        recommendation = _qualitative_recommendation(bottleneck, vram_util_pct, result.avg_gpu_util)

        reason_lines = []
        reason_lines.append(f"  Bottleneck: {bottleneck}")

        # Explain classification thresholds
        if result.gpu_total_vram_mb and result.gpu_total_vram_mb > 0:
            vram_util = (result.peak_vram_mb / result.gpu_total_vram_mb) * 100
            reason_lines.append(f"    VRAM util = {vram_util:.1f}%  (thresholds: <60% low, >80% high)")
        reason_lines.append(f"    GPU util  = {result.avg_gpu_util:.1f}%  (thresholds: <40% low, >80% high)")

        if bottleneck == "underutilized":
            reason_lines.append(f"    -> GPU util < 40% AND VRAM util < 60% => underutilized")
        elif bottleneck == "memory_bound":
            reason_lines.append(f"    -> GPU util < 40% AND VRAM util > 80% => memory_bound")
        elif bottleneck == "compute_bound":
            reason_lines.append(f"    -> GPU util > 80% => compute_bound")
        elif bottleneck == "balanced":
            reason_lines.append(f"    -> No extreme thresholds hit => balanced")

        reason_lines.append("")
        reason_lines.append(f"  Confidence: {confidence:.2f}")
        reason_lines.append(f"    Base score          +0.30  (have probe data)")

        sample_count = len(result.samples)
        if sample_count >= 100:
            reason_lines.append(f"    Samples ({sample_count:>4})      +0.30  (>=100)")
        elif sample_count >= 20:
            reason_lines.append(f"    Samples ({sample_count:>4})      +0.20  (>=20)")
        elif sample_count >= 5:
            reason_lines.append(f"    Samples ({sample_count:>4})      +0.10  (>=5)")
        else:
            reason_lines.append(f"    Samples ({sample_count:>4})      +0.00  (<5)")

        dur = result.duration_seconds
        if dur >= 300:
            reason_lines.append(f"    Duration ({dur:>5.0f}s)   +0.20  (>=300s)")
        elif dur >= 60:
            reason_lines.append(f"    Duration ({dur:>5.0f}s)   +0.10  (>=60s)")
        else:
            reason_lines.append(f"    Duration ({dur:>5.0f}s)   +0.00  (<60s)")

        if step_count and step_count > 0:
            reason_lines.append(f"    Step count          +0.20  (framework callback)")
        else:
            reason_lines.append(f"    Step count          +0.00  (no callback)")

        reason_lines.append(f"    Signal cap          max 0.60 (NVML_ONLY)")

        if recommendation:
            reason_lines.append("")
            reason_lines.append(f"  Recommendation: {recommendation}")

        console.print(Panel("\n".join(reason_lines), title="Recommendation Reasoning", border_style="cyan", padding=(1, 0)))

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
        return "85% (exact param count)"
    if extraction_method == "ast":
        return "75% (inferred from model name)"
    return "80% (static estimate)"


def _bottleneck_color(bottleneck):
    # type: (str) -> str
    """Rich color for bottleneck classification."""
    if bottleneck == "underutilized":
        return "yellow"
    if bottleneck == "memory_bound":
        return "red"
    if bottleneck in ("compute_bound", "balanced"):
        return "green"
    return "dim"
