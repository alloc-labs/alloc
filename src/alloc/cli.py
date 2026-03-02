"""Alloc CLI — GPU intelligence for ML training.

Commands:
    alloc ghost <script.py>    Ghost scan — static VRAM analysis without executing the model
    alloc run <command...>     Wrap training with probe monitoring, write artifact
    alloc scan --model <name>  Remote ghost scan via API — no GPU needed
    alloc diagnose <script.py> Analyze training script for performance issues
    alloc doctor <script.py>   Full training doctor — diagnose, explain, and fix
    alloc status               Show last run summary, upload state, dashboard link
    alloc login                Authenticate with Alloc dashboard
    alloc upload <artifact>    Upload an artifact to the Alloc dashboard
    alloc version              Show version
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import typer
from rich.console import Console

import json as json_mod

from alloc import __version__
from alloc.config import get_api_url, get_token, should_upload, try_refresh_access_token

app = typer.Typer(
    name="alloc",
    help="GPU intelligence for ML training. Right-size before you launch.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def ghost(
    script: str = typer.Argument(..., help="Python script to analyze (e.g. train.py)"),
    dtype: str = typer.Option("fp16", help="Data type: fp16, bf16, fp32"),
    batch_size: Optional[int] = typer.Option(None, help="Training batch size (auto-detected from model when possible)"),
    seq_length: Optional[int] = typer.Option(None, help="Sequence length (for transformer models)"),
    hidden_dim: Optional[int] = typer.Option(None, help="Hidden dimension (for transformer models)"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed VRAM formula breakdown"),
    param_count_b: Optional[float] = typer.Option(None, "--param-count-b", "-p", help="Param count in billions (skip script analysis)"),
    timeout: int = typer.Option(60, "--timeout", help="Max seconds for model extraction"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml (use catalog defaults)"),
    explain: bool = typer.Option(False, "--explain", help="AI-refined VRAM estimate (Pro tier, requires login)"),
):
    """Ghost scan — static VRAM analysis without executing the model."""
    from alloc.ghost import ghost as ghost_fn
    from alloc.display import print_ghost_report
    from alloc.model_extractor import extract_model_info

    # Load GPU context from .alloc.yaml
    gpu_context = _load_gpu_context(no_config)

    info = extract_model_info(script, timeout=timeout, param_count_b=param_count_b)

    if info is None:
        if json_output:
            _print_json({"error": f"Could not extract model from {script}"})
        else:
            console.print(f"[yellow]Could not extract model from {script}.[/yellow]")
            console.print("[dim]Supported: PyTorch nn.Module, HuggingFace AutoModel, Lightning modules.[/dim]")
            console.print(f"[dim]Tip: alloc ghost {script} --param-count-b 7.0[/dim]")
        raise typer.Exit(1)

    # Use dtype from execution if available, otherwise CLI flag
    resolved_dtype = info.dtype if info.method == "execution" else dtype

    # Use architecture info from model extraction if user didn't provide explicit values
    resolved_hidden_dim = hidden_dim if hidden_dim is not None else info.hidden_dim
    resolved_seq_length = seq_length if seq_length is not None else info.seq_length

    report = ghost_fn(
        param_count=info.param_count,
        dtype=resolved_dtype,
        batch_size=batch_size,
        seq_length=resolved_seq_length,
        hidden_dim=resolved_hidden_dim,
        activation_memory_bytes=info.activation_memory_bytes,
    )
    report.extraction_method = info.method

    # --explain: fetch AI-refined VRAM estimate
    explain_data = None
    if explain:
        explain_data = _fetch_ghost_explain(report, info, batch_size, resolved_seq_length, resolved_hidden_dim, json_output)

    if json_output:
        data = report.to_dict()
        if gpu_context:
            data["gpu_context"] = gpu_context
        if explain_data:
            data["explain"] = explain_data
        _print_json(data)
    else:
        print_ghost_report(report)
        if explain_data:
            _render_ghost_explain_rich(explain_data)
        if gpu_context and not verbose:
            _print_gpu_context_summary(gpu_context)
        if verbose:
            from alloc.display import print_verbose_ghost
            print_verbose_ghost(report)
            if gpu_context:
                _print_gpu_context_detail(gpu_context)


def _fetch_ghost_explain(report, info, batch_size, seq_length, hidden_dim, json_output):
    """Fetch AI-refined VRAM estimate from the API. Returns dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --explain. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --explain. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        # Build heuristic VRAM dict from ghost report
        heuristic_vram = {
            "weights_gb": report.weights_gb,
            "gradients_gb": report.gradients_gb,
            "optimizer_gb": report.optimizer_gb,
            "activations_gb": report.activations_gb,
            "buffer_gb": report.buffer_gb,
            "total_gb": report.total_gb,
        }

        # Build unknowns list based on what we don't know
        unknowns = []
        if batch_size is None:
            unknowns.append("batch_size not specified — activation estimate uses generic heuristic")
        if seq_length is None and hidden_dim is None:
            unknowns.append("architecture dimensions unknown — cannot use transformer-specific formula")
        if info.method == "manual":
            unknowns.append("param count from user input — model architecture unknown")

        # Build workload context from model info
        workload = {}
        if info.model_name:
            workload["model_name"] = info.model_name

        payload = {
            "heuristic_vram": heuristic_vram,
            "param_count_b": report.param_count_b,
            "dtype": report.dtype,
            "extraction_method": info.method,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "hidden_dim": hidden_dim,
            "num_layers": report.num_layers,
        }

        if info.model_name:
            payload["model_name"] = info.model_name

        if unknowns:
            payload["unknowns"] = unknowns

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/ghost-explain".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --explain"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --explain. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "Ghost explain failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]Ghost explain failed: {}[/red]".format(e))
        return None


def _render_ghost_explain_rich(explain_data):
    """Render ghost explain response as Rich terminal output."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    refined = explain_data.get("refined_total_gb", 0)
    heuristic = explain_data.get("heuristic_total_gb", 0)
    quality = explain_data.get("estimate_quality", "unknown")
    confidence = explain_data.get("confidence", "unknown")

    # Confidence color
    conf_colors = {"high": "green", "medium": "yellow", "low": "red"}
    conf_color = conf_colors.get(confidence, "white")

    # Header with delta
    if heuristic > 0 and refined != heuristic:
        delta_pct = ((refined - heuristic) / heuristic) * 100
        sign = "+" if delta_pct > 0 else ""
        console.print(
            "\n[bold cyan]AI-Refined VRAM Estimate[/bold cyan]"
            "  [dim]({})[/dim]".format(quality)
        )
        console.print(
            "  Heuristic: [dim]{:.2f} GB[/dim]  ->  "
            "Agent: [bold]{:.2f} GB[/bold]  [{}]({}{:.0f}%)[/{}]".format(
                heuristic, refined, conf_color, sign, delta_pct, conf_color,
            )
        )
    else:
        console.print(
            "\n[bold cyan]AI-Refined VRAM Estimate[/bold cyan]"
            "  [bold]{:.2f} GB[/bold]".format(refined)
        )

    console.print("  Confidence: [{}]{}[/{}]".format(conf_color, confidence, conf_color))

    # Breakdown table
    breakdown = explain_data.get("breakdown", [])
    if breakdown:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Component", style="dim")
        table.add_column("Heuristic", justify="right")
        table.add_column("Refined", justify="right", style="bold")
        table.add_column("Adjustment", style="dim")

        for item in breakdown:
            comp = item.get("component", "?")
            h_gb = item.get("heuristic_gb", 0)
            a_gb = item.get("agent_gb")
            reasoning = item.get("reasoning", "")

            h_str = "{:.2f} GB".format(h_gb)
            if a_gb is not None:
                a_str = "{:.2f} GB".format(a_gb)
                # Truncate reasoning for table display
                reason_short = reasoning[:60] + "..." if len(reasoning) > 60 else reasoning
            else:
                a_str = "[dim]—[/dim]"
                reason_short = ""

            table.add_row(comp.capitalize(), h_str, a_str, reason_short)

        console.print()
        console.print(table)

    # Architecture insight
    insight = explain_data.get("architecture_insight", "")
    if insight:
        console.print("\n  [dim]{}[/dim]".format(insight))

    # Adjustments
    adjustments = explain_data.get("adjustments", [])
    if adjustments:
        console.print("\n[bold]Adjustments:[/bold]")
        for adj in adjustments:
            console.print("  - {}".format(adj))

    # GPU recommendations
    gpu_recs = explain_data.get("gpu_recommendations", [])
    if gpu_recs:
        console.print("\n[bold]GPU Fit:[/bold]")
        for rec in gpu_recs:
            console.print("  - {}".format(rec))

    console.print()


@app.command()
def run(
    command: list[str] = typer.Argument(..., help="Command to run (e.g. python train.py)"),
    timeout: int = typer.Option(120, help="Max calibration time in seconds"),
    gpu: int = typer.Option(0, help="GPU index to monitor"),
    save: bool = typer.Option(True, help="Save artifact to disk"),
    out: Optional[str] = typer.Option(None, "--out", help="Output path for artifact"),
    upload: bool = typer.Option(False, "--upload", help="Upload artifact to Alloc dashboard after run"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Skip auto-upload even when logged in"),
    full: bool = typer.Option(False, "--full", help="Monitor full training run instead of calibrating"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show hardware context, sample dump, recommendation reasoning"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml (use catalog defaults)"),
    after: Optional[str] = typer.Option(None, "--after", help="Previous run ID to compare against (outcome tracking)"),
):
    """Run a training command with GPU monitoring."""
    from alloc.probe import probe_command
    from alloc.display import print_probe_result, print_verdict
    from alloc.artifact_writer import write_report

    # Load GPU context from .alloc.yaml
    gpu_context = _load_gpu_context(no_config)

    if not command:
        console.print("[red]No command provided.[/red]")
        console.print("Usage: alloc run python train.py")
        raise typer.Exit(1)

    # ALLOC_POLICY: "warn" or "enforce" forces full monitoring
    alloc_policy = os.environ.get("ALLOC_POLICY", "").lower().strip()
    if alloc_policy and alloc_policy not in ("warn", "enforce"):
        console.print(f"[yellow]Unknown ALLOC_POLICY='{alloc_policy}', ignoring.[/yellow]")
        alloc_policy = ""
    if alloc_policy:
        full = True

    # Determine mode
    calibrate = not full

    # Effective timeout: unlimited for --full unless user explicitly set it
    effective_timeout = timeout
    if full and timeout == 120:  # User didn't override default
        effective_timeout = 0    # Unlimited for full mode

    # Mode label
    if calibrate:
        mode_label = "Calibrate"
    elif full:
        mode_label = "Full monitoring"
    else:
        mode_label = "Calibrate"

    if not json_output:
        console.print(f"[green]alloc[/green] [dim]v{__version__}[/dim] — {mode_label}")
        console.print(f"[dim]Command: {' '.join(command)}[/dim]")
        if calibrate:
            console.print(f"[yellow]Calibrate mode: training will be stopped once metrics stabilize (timeout: {timeout}s)[/yellow]")
            console.print(f"[dim]Use --full to monitor without stopping the training process[/dim]")
        console.print()

    result = probe_command(
        command,
        timeout_seconds=effective_timeout,
        gpu_index=gpu,
        calibrate=calibrate,
    )

    if not json_output:
        if result.error and "pynvml" in result.error:
            console.print(f"[yellow]{result.error}[/yellow]")
            console.print("[dim]Process ran without GPU monitoring.[/dim]")
        elif result.error:
            console.print(f"[red]Error: {result.error}[/red]")

    # Read callback data from sidecar (written by framework callbacks)
    callback_data = _read_callback_data()
    step_count = callback_data.get("step_count") if callback_data else None

    # Discover environment context (git, container, Ray)
    from alloc.context import discover_context
    env_context = discover_context()
    topology = _infer_parallel_topology_from_env(
        num_gpus_detected=result.num_gpus_detected,
        config_interconnect=gpu_context.get("interconnect") if gpu_context else None,
        detected_interconnect=result.detected_interconnect,
    )
    objective = os.environ.get("ALLOC_OBJECTIVE", "").strip().lower() or _objective_from_context(gpu_context)
    max_budget_hourly = _max_budget_hourly_from_context(gpu_context)

    # Build artifact dict with new fields
    artifact_path = ""
    if save:
        probe_dict = {
            "peak_vram_mb": result.peak_vram_mb,
            "avg_gpu_util": result.avg_gpu_util,
            "avg_power_watts": result.avg_power_watts,
            "duration_seconds": result.duration_seconds,
            "samples": result.samples,
            "exit_code": result.exit_code,
            "probe_mode": result.probe_mode,
            "steps_profiled": result.steps_profiled,
            "stop_reason": result.stop_reason,
            "gpu_name": result.gpu_name,
            "gpu_total_vram_mb": result.gpu_total_vram_mb,
            "calibration_duration_s": result.calibration_duration_s,
            "step_count": step_count,
            "num_nodes": topology.get("num_nodes"),
            "gpus_per_node": topology.get("gpus_per_node"),
            "tp_degree": topology.get("tp_degree"),
            "pp_degree": topology.get("pp_degree"),
            "dp_degree": topology.get("dp_degree"),
            "interconnect_type": topology.get("interconnect_type"),
            "objective": objective,
            "max_budget_hourly": max_budget_hourly,
            "command": " ".join(command),
            "baseline_run_id": after,
        }
        # Merge timing fields from callback sidecar
        if callback_data:
            for key in ("step_time_ms_p50", "step_time_ms_p90", "samples_per_sec",
                         "step_time_ms_mean", "step_time_ms_std"):
                val = callback_data.get(key)
                if val is not None:
                    probe_dict[key] = val
        # Merge per-GPU peak VRAM from probe (maps to per_rank_peak_vram_mb)
        if result.per_gpu_peak_vram_mb:
            probe_dict["per_rank_peak_vram_mb"] = result.per_gpu_peak_vram_mb
        hw_context = {
            "gpu_name": result.gpu_name,
            "gpu_total_vram_mb": result.gpu_total_vram_mb,
            "driver_version": result.driver_version,
            "cuda_version": result.cuda_version,
            "sm_version": result.sm_version,
            "num_gpus_detected": result.num_gpus_detected,
        }
        artifact_path = write_report(
            probe_result=probe_dict,
            output_path=out,
            hardware_context=hw_context,
            context=env_context if env_context else None,
        )

    # Build budget context for verdict display
    budget_ctx = _build_budget_context(gpu_context, result)

    if json_output:
        from alloc.display import build_verdict_dict
        data = build_verdict_dict(result, artifact_path=artifact_path, step_count=step_count, callback_data=callback_data, budget_context=budget_ctx)
        if result.error:
            data["error"] = result.error
        _print_json(data)
    else:
        # Display verdict for all modes when we have GPU data
        if result.peak_vram_mb > 0:
            print_verdict(result, artifact_path=artifact_path, step_count=step_count, callback_data=callback_data, budget_context=budget_ctx)
            if verbose:
                from alloc.display import print_verbose_run
                print_verbose_run(result, step_count=step_count)
        elif artifact_path:
            console.print(f"[dim]Artifact saved: {artifact_path}[/dim]")

    # Upload when explicitly opted in: --upload flag or ALLOC_UPLOAD env var
    # Being logged in alone does NOT trigger upload (privacy-first)
    should_upload_artifact = not no_upload and (upload or should_upload())
    if artifact_path and should_upload_artifact:
        _try_upload(artifact_path)
    elif artifact_path and not should_upload_artifact and not no_upload:
        if not json_output:
            if get_token():
                console.print("[dim]Tip: alloc run --upload to send this artifact to your dashboard[/dim]")
            else:
                console.print("[dim]Tip: alloc login --browser to connect your dashboard[/dim]")

    if result.exit_code and result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command()
def diagnose(
    script: str = typer.Argument("", help="Python script to analyze (e.g. train.py)"),
    diff: bool = typer.Option(False, "--diff", help="Output unified diff patches"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show rule details and references"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter: critical, warning, info"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter: dataloader, memory, precision, distributed, throughput, upgrade"),
    # Phase 2 flags
    artifact: Optional[str] = typer.Option(None, "--artifact", help="Path to alloc_artifact.json.gz"),
    no_artifact: bool = typer.Option(False, "--no-artifact", help="Skip artifact auto-discovery (AST-only)"),
    memory: bool = typer.Option(False, "--memory", help="Show VRAM breakdown"),
    efficiency: bool = typer.Option(False, "--efficiency", help="Show step time decomposition"),
    compare: Optional[str] = typer.Option(None, "--compare", help="Compare with another artifact"),
    upgrades: bool = typer.Option(False, "--upgrades", help="Scan for outdated patterns and upgrade opportunities"),
    rules: bool = typer.Option(False, "--rules", help="List all available diagnosis rules"),
    explain: bool = typer.Option(False, "--explain", help="AI-powered explanation of findings (Pro tier, requires login)"),
    fix: bool = typer.Option(False, "--fix", help="AI-generated code patches for findings (Pro tier, requires login)"),
    what_if: Optional[str] = typer.Option(None, "--what-if", help='Estimate impact of a change, e.g. --what-if "switch to bf16"'),
    comm: bool = typer.Option(False, "--comm", help="AI-powered communication overhead analysis (Pro tier, requires login)"),
    root_cause: bool = typer.Option(False, "--root-cause", help="AI-powered root cause diagnosis for ambiguous bottlenecks (Pro tier, requires login)"),
    migrate: Optional[str] = typer.Option(None, "--migrate", help='Plan migration to target hardware, e.g. --migrate "8x H100-80GB"'),
    explore: bool = typer.Option(False, "--explore", help="AI-powered Pareto knob exploration (Pro tier, requires login)"),
):
    """Analyze a training script for performance issues."""
    if rules:
        _print_rules(json_output)
        raise typer.Exit(0)

    if not script:
        if json_output:
            _print_json({"error": "No script specified. Usage: alloc diagnose train.py"})
        else:
            console.print("[red]No script specified.[/red]")
            console.print("[dim]Usage: alloc diagnose train.py[/dim]")
            console.print("[dim]       alloc diagnose --rules[/dim]")
        raise typer.Exit(1)

    from alloc.code_analyzer import analyze_script as _analyze_script
    from alloc.artifact_loader import load_artifact as _load_artifact, find_artifact as _find_artifact
    from alloc.diagnosis_engine import run_diagnosis, _quick_hw_context
    from alloc.diagnosis_display import (
        print_diagnose_rich, print_diagnose_diff, print_diagnose_json,
        print_diagnose_memory, print_diagnose_efficiency,
        print_diagnose_comparison, print_oom_autopsy,
    )

    if not os.path.isfile(script):
        if json_output:
            _print_json({"error": f"File not found: {script}"})
        else:
            console.print(f"[red]File not found: {script}[/red]")
        raise typer.Exit(1)

    if not script.endswith(".py"):
        if json_output:
            _print_json({"error": f"Expected a Python file: {script}"})
        else:
            console.print(f"[yellow]Expected a Python file: {script}[/yellow]")
        raise typer.Exit(1)

    # Analyze the script
    findings = _analyze_script(script)

    # Resolve artifact (Phase 2)
    artifact_data = None
    artifact_path = None
    if not no_artifact:
        if artifact:
            artifact_path = artifact
        else:
            artifact_path = _find_artifact(".")

        if artifact_path:
            artifact_data = _load_artifact(artifact_path)
            if artifact_data is None and not json_output:
                console.print(f"[yellow]Could not parse artifact: {artifact_path}[/yellow]")

    # Load comparison artifact
    compare_data = None
    if compare:
        compare_data = _load_artifact(compare)
        if compare_data is None and not json_output:
            console.print(f"[yellow]Could not parse comparison artifact: {compare}[/yellow]")

    # Build hardware context
    hw = _quick_hw_context()

    # Merge .alloc.yaml GPU context if available
    if not no_config:
        gpu_context = _load_gpu_context(False)
        if gpu_context and hw is None:
            fleet = gpu_context.get("fleet", [])
            if fleet:
                hw = {"gpu_count": len(fleet)}
        elif gpu_context and hw is not None:
            pass  # Don't override live detection

    result = run_diagnosis(
        findings,
        hardware_context=hw,
        artifact=artifact_data,
        compare_artifact=compare_data,
    )

    # Set artifact_path on result
    if artifact_path:
        result.artifact_path = artifact_path

    # Apply filters
    if severity:
        sev = severity.strip().lower()
        result.findings = [d for d in result.findings if d.severity == sev]
        result.summary = _recount_summary(result.findings)
    if category:
        cat = category.strip().lower()
        result.findings = [d for d in result.findings if d.category == cat]
        result.summary = _recount_summary(result.findings)
    if upgrades:
        from alloc.diagnosis_rules import UPGRADE_RULE_IDS
        result.findings = [d for d in result.findings if d.rule_id in UPGRADE_RULE_IDS]
        result.summary = _recount_summary(result.findings)

    # --explain / --fix / --what-if / --comm / --root-cause / --migrate / --explore: fetch AI response before output
    explain_data = None
    fix_data = None
    what_if_data = None
    comm_data = None
    root_cause_data = None
    migrate_data = None
    explore_data = None
    if explain:
        explain_data = _fetch_explain(result, findings, artifact_data, hw, script, json_output)
    if fix:
        fix_data = _fetch_fix(result, findings, artifact_data, hw, script, json_output)
    if what_if:
        what_if_data = _fetch_what_if(result, findings, artifact_data, hw, script, what_if, json_output)
    if comm:
        comm_data = _fetch_comm(result, findings, artifact_data, hw, json_output)
    if root_cause:
        root_cause_data = _fetch_root_cause(result, findings, artifact_data, hw, script, json_output)
    if migrate:
        migrate_data = _fetch_migrate(result, findings, artifact_data, hw, script, migrate, json_output)
    if explore:
        explore_data = _fetch_explore(result, findings, artifact_data, hw, script, json_output)

    # Output routing
    if json_output:
        print_diagnose_json(result, explain_data=explain_data, fix_data=fix_data,
                            what_if_data=what_if_data, comm_data=comm_data,
                            root_cause_data=root_cause_data, migrate_data=migrate_data,
                            explore_data=explore_data)
    elif memory:
        print_diagnose_memory(result)
    elif efficiency:
        print_diagnose_efficiency(result)
    elif compare_data and result.comparison:
        print_diagnose_comparison(result)
    elif result.oom_detected:
        print_oom_autopsy(result)
        # Also show regular findings
        if result.findings:
            print_diagnose_rich(result, verbose=verbose)
    elif diff:
        print_diagnose_diff(result)
    else:
        print_diagnose_rich(result, verbose=verbose)

    # AI output: render Rich (when not --json)
    if not json_output:
        if explain and explain_data:
            _render_explain_rich(explain_data)
        if fix and fix_data:
            _render_fix_rich(fix_data)
        if what_if and what_if_data:
            _render_what_if_rich(what_if_data)
        if comm and comm_data:
            _render_comm_rich(comm_data)
        if root_cause and root_cause_data:
            _render_root_cause_rich(root_cause_data)
        if migrate and migrate_data:
            _render_migrate_rich(migrate_data)
        if explore and explore_data:
            _render_explore_rich(explore_data)


def _recount_summary(findings):
    """Recount summary after filtering."""
    summary = {"critical": 0, "warning": 0, "info": 0, "total": len(findings)}
    for d in findings:
        if d.severity in summary:
            summary[d.severity] += 1
    return summary


def _build_workload_context(findings, artifact_data, hw):
    """Build WorkloadContext dict from AST findings + artifact + hardware."""
    ctx = {}

    # Framework detection
    frameworks = set()
    for d in findings.distributed:
        if d.kind == "hf_trainer":
            frameworks.add("hf_trainer")
        elif d.kind == "deepspeed":
            frameworks.add("deepspeed")
        elif d.kind == "accelerate":
            frameworks.add("accelerate")
        elif d.kind in ("ddp", "fsdp"):
            frameworks.add(d.kind)
    if frameworks:
        ctx["framework"] = sorted(frameworks)[0]  # Primary framework

    # Strategy detection
    for d in findings.distributed:
        if d.kind in ("ddp", "fsdp"):
            ctx["strategy"] = d.kind

    # Optimizer detection
    for opt in findings.optimizers:
        ctx["optimizer_type"] = opt.optimizer_type
        break

    # Fine-tuning detection
    if hasattr(findings, "fine_tuning") and findings.fine_tuning:
        ft = findings.fine_tuning[0]
        ctx["fine_tuning_method"] = ft.method

    # Gradient checkpointing
    if hasattr(findings, "has_gradient_checkpointing"):
        ctx["gradient_checkpointing"] = findings.has_gradient_checkpointing

    # Model name
    for m in findings.models:
        if m.kind == "from_pretrained" and m.model_name:
            ctx["model_name"] = m.model_name
            break

    # Architecture type inference
    if ctx.get("model_name"):
        name = ctx["model_name"].lower()
        if any(k in name for k in ("llama", "gpt", "mistral", "qwen", "phi", "gemma", "falcon")):
            ctx["architecture_type"] = "transformer"
        elif any(k in name for k in ("resnet", "efficientnet", "mobilenet", "convnext")):
            ctx["architecture_type"] = "cnn"
        elif any(k in name for k in ("vit", "deit", "beit", "swin", "dinov2", "clip")):
            ctx["architecture_type"] = "transformer"  # Vision transformers
        elif any(k in name for k in ("stable-diffusion", "sdxl", "flux")):
            ctx["architecture_type"] = "diffusion"
        elif any(k in name for k in ("mixtral", "switch")):
            ctx["architecture_type"] = "moe"
        elif any(k in name for k in ("whisper", "wav2vec")):
            ctx["architecture_type"] = "transformer"
    if "architecture_type" not in ctx:
        ctx["architecture_type"] = "unknown"

    # From artifact
    if artifact_data:
        probe = artifact_data.get("probe", {})
        if probe.get("gpu_count"):
            ctx["num_gpus"] = probe["gpu_count"]
        if probe.get("interconnect_type"):
            ctx["interconnect_type"] = probe["interconnect_type"]

        hw_ctx = artifact_data.get("hardware_context", {})
        if hw_ctx.get("num_gpus_detected"):
            ctx["num_gpus"] = hw_ctx["num_gpus_detected"]
        if hw_ctx.get("interconnect_type"):
            ctx["interconnect_type"] = hw_ctx["interconnect_type"]

    # From hardware context
    if hw:
        if hw.get("gpu_count"):
            ctx.setdefault("num_gpus", hw["gpu_count"])

    return ctx if ctx else None


def _extract_code_snippets(script_path, findings):
    """Extract relevant code snippets around each finding's line_number.

    Returns dict of filename → code excerpt. Limits total to ~2000 lines.
    """
    snippets = {}
    total_lines = 0
    max_total = 2000
    context_lines = 5  # lines before/after each finding

    # Collect unique (file, line) pairs
    file_lines = {}
    for d in findings:
        fp = d.file_path or script_path
        if not fp or not os.path.isfile(fp):
            continue
        file_lines.setdefault(fp, set())
        if d.line_number:
            file_lines[fp].add(d.line_number)

    for fp, lines_of_interest in file_lines.items():
        if total_lines >= max_total:
            break
        try:
            with open(fp, "r") as f:
                all_lines = f.readlines()
        except Exception:
            continue

        if not lines_of_interest:
            # No specific lines — include first 50 lines
            excerpt = all_lines[:50]
            snippet = "".join(excerpt)
        else:
            # Merge ranges around each line of interest
            ranges = set()
            for ln in sorted(lines_of_interest):
                start = max(0, ln - 1 - context_lines)
                end = min(len(all_lines), ln + context_lines)
                for i in range(start, end):
                    ranges.add(i)

            sorted_ranges = sorted(ranges)
            parts = []
            prev = -2
            for idx in sorted_ranges:
                if idx > prev + 1:
                    if parts:
                        parts.append("# ...\n")
                parts.append(all_lines[idx] if idx < len(all_lines) else "")
                prev = idx
            snippet = "".join(parts)

        remaining = max_total - total_lines
        snippet_lines = snippet.split("\n")[:remaining]
        snippets[os.path.basename(fp)] = "\n".join(snippet_lines)
        total_lines += len(snippet_lines)

    return snippets if snippets else None


def _fetch_explain(result, findings, artifact_data, hw, script, json_output, quiet=False):
    """Fetch AI explanation from the API. Returns explain dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if not quiet:
            if json_output:
                import sys
                print('{"error": "Login required for --explain. Run: alloc login"}', file=sys.stderr)
            else:
                from rich.console import Console
                Console(stderr=True).print("[yellow]Login required for --explain. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "line_number": d.line_number,
                "file_path": d.file_path,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        payload = {
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        code_context = _extract_code_snippets(script, result.findings)
        if code_context:
            payload["code_context"] = code_context

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/explain".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if not quiet:
                    if json_output:
                        import sys
                        print('{"error": "Pro tier required for --explain"}', file=sys.stderr)
                    else:
                        from rich.console import Console
                        Console(stderr=True).print("[yellow]Pro tier required for --explain. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if not quiet:
            if json_output:
                import sys
                print('{{"error": "Explain request failed: {}"}}'.format(e), file=sys.stderr)
            else:
                from rich.console import Console
                Console(stderr=True).print("[red]Explain request failed: {}[/red]".format(e))
        return None


def _render_explain_rich(explain_data):
    """Render explain response as Rich terminal output."""
    from rich.console import Console
    console = Console()

    console.print("\n[bold cyan]AI Explanation[/bold cyan]")
    console.print(explain_data.get("summary", ""))

    arch_insight = explain_data.get("architecture_insight")
    if arch_insight:
        console.print("\n[dim]{}[/dim]".format(arch_insight))

    actions = explain_data.get("actions", [])
    if actions:
        console.print("\n[bold]Priority Actions:[/bold]")
        for action in actions:
            priority = action.get("priority", "?")
            title = action.get("title", "")
            explanation = action.get("explanation", "")
            impact = action.get("impact", "")
            confidence = action.get("confidence", "")
            finding_id = action.get("finding_id", "")
            label = " ({})".format(finding_id) if finding_id else ""
            console.print("  {}. [bold]{}[/bold]{}".format(priority, title, label))
            console.print("     {}".format(explanation))
            if impact:
                console.print("     [dim]Impact: {}[/dim]".format(impact))
            if confidence:
                console.print("     [dim]Confidence: {}[/dim]".format(confidence))

    interactions = explain_data.get("interactions")
    if interactions:
        console.print("\n[bold]Interactions:[/bold]")
        for ix in interactions:
            ids = ", ".join(ix.get("finding_ids", []))
            desc = ix.get("description", "")
            console.print("  [{}] {}".format(ids, desc))


def _fetch_fix(result, findings, artifact_data, hw, script, json_output, quiet=False):
    """Fetch AI-generated code patches from the API. Returns fix dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if not quiet:
            if json_output:
                import sys
                print('{"error": "Login required for --fix. Run: alloc login"}', file=sys.stderr)
            else:
                from rich.console import Console
                Console(stderr=True).print("[yellow]Login required for --fix. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()
        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "line_number": d.line_number,
                "file_path": d.file_path,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        # For --fix, send full file contents (not just snippets)
        code_context = _read_full_files(script, result.findings)

        payload = {
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
            "code_context": code_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        with httpx.Client(timeout=90) as client:
            resp = client.post(
                "{}/diagnose/fix".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if not quiet:
                    if json_output:
                        import sys
                        print('{"error": "Pro tier required for --fix"}', file=sys.stderr)
                    else:
                        from rich.console import Console
                        Console(stderr=True).print("[yellow]Pro tier required for --fix. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if not quiet:
            if json_output:
                import sys
                print('{{"error": "Fix request failed: {}"}}'.format(e), file=sys.stderr)
            else:
                from rich.console import Console
                Console(stderr=True).print("[red]Fix request failed: {}[/red]".format(e))
        return None


def _fetch_what_if(result, findings, artifact_data, hw, script, scenario, json_output):
    """Fetch what-if analysis from the API. Returns what-if dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --what-if. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --what-if. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()
        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "line_number": d.line_number,
                "file_path": d.file_path,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        code_context = _extract_code_snippets(script, result.findings)

        payload = {
            "scenario": scenario,
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
            "code_context": code_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/what-if".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --what-if"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --what-if. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "What-if request failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]What-if request failed: {}[/red]".format(e))
        return None


def _read_full_files(script, findings):
    """Read full file contents for files mentioned in findings (for --fix).

    Returns dict of {filename: content}, capped at 10 files.
    """
    import os

    files = set()
    if script and os.path.isfile(script):
        files.add(script)
    for d in findings:
        if d.file_path and os.path.isfile(d.file_path):
            files.add(d.file_path)

    code_context = {}
    total_lines = 0
    for filepath in sorted(files)[:10]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = content.split("\n")
            remaining = 3000 - total_lines
            if remaining <= 0:
                break
            if len(lines) > remaining:
                lines = lines[:remaining]
                content = "\n".join(lines)
            total_lines += len(lines)
            code_context[filepath] = content
        except Exception:
            pass

    return code_context if code_context else None


def _render_fix_rich(fix_data):
    """Render fix response as Rich terminal output with syntax-highlighted patches."""
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()

    console.print("\n[bold green]AI-Generated Fixes[/bold green]")
    console.print(fix_data.get("summary", ""))

    patches = fix_data.get("patches", [])
    if not patches:
        console.print("[dim]No patches generated.[/dim]")
        return

    for i, patch in enumerate(patches, 1):
        file_path = patch.get("file_path", "unknown")
        description = patch.get("description", "")
        diff_text = patch.get("diff", "")
        confidence = patch.get("confidence", "")
        finding_ids = patch.get("finding_ids", [])

        ids_label = " ({})".format(", ".join(finding_ids)) if finding_ids else ""
        console.print("\n[bold]Patch {}: {}[/bold]{}".format(i, file_path, ids_label))
        console.print("  {}".format(description))
        if confidence:
            console.print("  [dim]Confidence: {}[/dim]".format(confidence))

        if diff_text:
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            console.print(syntax)

    warnings = fix_data.get("warnings", [])
    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print("  - {}".format(w))

    console.print("\n[dim]Review patches carefully before applying. "
                  "Use: patch -p1 < fix.patch[/dim]")


def _render_what_if_rich(what_if_data):
    """Render what-if response as Rich terminal output."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    scenario = what_if_data.get("scenario", "")
    feasible = what_if_data.get("feasible", True)
    summary = what_if_data.get("summary", "")

    console.print("\n[bold magenta]What-If Analysis[/bold magenta]")
    console.print("[dim]Scenario: {}[/dim]".format(scenario))

    if not feasible:
        console.print("[red bold]Not feasible[/red bold]: {}".format(summary))
    else:
        console.print(summary)

    impacts = what_if_data.get("impacts", [])
    if impacts:
        table = Table(title="Estimated Impact", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Direction")
        table.add_column("Magnitude")
        table.add_column("Reasoning", max_width=50)

        direction_styles = {
            "increase": "[green]increase[/green]",
            "decrease": "[red]decrease[/red]",
            "unchanged": "[dim]unchanged[/dim]",
        }

        for imp in impacts:
            metric = imp.get("metric", "")
            direction = imp.get("direction", "unchanged")
            magnitude = imp.get("magnitude", "")
            reasoning = imp.get("reasoning", "")

            # Color-code direction based on whether it's good or bad
            styled_dir = direction_styles.get(direction, direction)
            # For VRAM and cost, decrease is good (green)
            if metric in ("vram", "cost") and direction == "decrease":
                styled_dir = "[green]decrease[/green]"
            elif metric in ("vram", "cost") and direction == "increase":
                styled_dir = "[red]increase[/red]"
            # For throughput, increase is good
            elif metric == "throughput" and direction == "increase":
                styled_dir = "[green]increase[/green]"
            elif metric == "throughput" and direction == "decrease":
                styled_dir = "[red]decrease[/red]"

            table.add_row(metric, styled_dir, magnitude or "-", reasoning)

        console.print(table)

    risks = what_if_data.get("risks", [])
    if risks:
        console.print("\n[yellow]Risks:[/yellow]")
        for r in risks:
            console.print("  - {}".format(r))

    steps = what_if_data.get("suggested_steps", [])
    if steps:
        console.print("\n[bold]Steps to implement:[/bold]")
        for i, step in enumerate(steps, 1):
            console.print("  {}. {}".format(i, step))


def _fetch_comm(result, findings, artifact_data, hw, json_output):
    """Fetch AI communication analysis from the API. Returns comm dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --comm. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --comm. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        workload = _build_workload_context(findings, artifact_data, hw)

        payload = {
            "workload": workload,
            "hardware": result.hardware_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/comm-analysis".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --comm"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --comm. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "Comm analysis failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]Comm analysis failed: {}[/red]".format(e))
        return None


def _render_comm_rich(comm_data):
    """Render communication analysis as Rich terminal output."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    analysis = comm_data.get("analysis", {})
    quality = comm_data.get("estimate_quality", "unknown")

    console.print("\n[bold cyan]Communication Overhead Analysis[/bold cyan]")

    total = analysis.get("total_overhead_pct", 0)
    heuristic = analysis.get("heuristic_overhead_pct")
    bottleneck = analysis.get("bottleneck_type")

    # Summary line
    quality_label = "[green]Agent-refined[/green]" if quality == "agent_reasoned" else "[dim]Heuristic-only[/dim]"
    console.print("Total overhead: [bold]{:.1f}%[/bold] ({})".format(total, quality_label))
    if heuristic is not None and quality == "agent_reasoned":
        delta = total - heuristic
        if abs(delta) > 0.5:
            direction = "[red]+{:.1f}%[/red]" if delta > 0 else "[green]{:.1f}%[/green]"
            console.print("  Heuristic estimate was {:.1f}% (agent adjusted by {})".format(
                heuristic, direction.format(delta)))

    if bottleneck:
        label_map = {
            "bandwidth_limited": "[red]Bandwidth Limited[/red]",
            "latency_limited": "[yellow]Latency Limited[/yellow]",
            "compute_dominated": "[green]Compute Dominated[/green]",
            "well_balanced": "[green]Well Balanced[/green]",
        }
        console.print("Bottleneck: {}".format(label_map.get(bottleneck, bottleneck)))

    # Breakdown table
    breakdown = analysis.get("breakdown", [])
    if breakdown:
        table = Table(title="Overhead Breakdown", show_header=True)
        table.add_column("Source", style="bold")
        table.add_column("Heuristic %", justify="right")
        table.add_column("Agent %", justify="right")
        table.add_column("Reasoning", max_width=50)

        for item in breakdown:
            source = item.get("source", "").replace("_", " ").title()
            h_pct = item.get("heuristic_pct")
            a_pct = item.get("agent_pct")
            reasoning = item.get("reasoning", "")

            h_str = "{:.1f}".format(h_pct) if h_pct is not None else "-"
            a_str = "{:.1f}".format(a_pct) if a_pct is not None else "-"

            table.add_row(source, h_str, a_str, reasoning)

        console.print(table)

    # Overlap assessment
    overlap = analysis.get("overlap_assessment")
    if overlap:
        console.print("\n[bold]Overlap Analysis:[/bold] {}".format(overlap))

    # Scaling assessment
    scaling = analysis.get("scaling_assessment")
    if scaling:
        console.print("[bold]Scaling:[/bold] {}".format(scaling))

    # Recommendations
    recs = analysis.get("recommendations", [])
    if recs:
        console.print("\n[bold]Recommendations:[/bold]")
        for r in recs:
            console.print("  - {}".format(r))


def _fetch_root_cause(result, findings, artifact_data, hw, script, json_output):
    """Fetch AI root cause diagnosis from the API. Returns dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --root-cause. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --root-cause. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        payload = {
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        # Derive signal patterns from artifact data
        if artifact_data:
            probe = artifact_data.get("probe", {})
            samples = probe.get("samples", [])

            # GPU utilization pattern
            if samples:
                utils = [s.get("gpu_util_pct", 0) for s in samples if s.get("gpu_util_pct") is not None]
                if utils:
                    avg_util = sum(utils) / len(utils)
                    import statistics
                    if len(utils) > 1:
                        stdev = statistics.stdev(utils)
                        cov = stdev / max(avg_util, 1)
                        if cov > 0.3:
                            payload["gpu_utilization_pattern"] = "spiky"
                        elif avg_util > 70:
                            payload["gpu_utilization_pattern"] = "steady_high"
                        else:
                            payload["gpu_utilization_pattern"] = "steady_low"

            # VRAM pattern from per-rank data
            per_rank = probe.get("per_rank_peak_vram_mb")
            if per_rank and len(per_rank) > 1:
                max_v = max(per_rank)
                min_v = min(per_rank)
                if max_v > 0 and (max_v - min_v) / max_v > 0.15:
                    payload["vram_pattern"] = "skewed"
                else:
                    payload["vram_pattern"] = "uniform"

            # Step time variance
            step_p50 = probe.get("step_time_ms_p50")
            step_p90 = probe.get("step_time_ms_p90")
            if step_p50 and step_p90 and step_p50 > 0:
                ratio = step_p90 / step_p50
                if ratio > 1.15:
                    payload["step_time_variance"] = "high"
                elif ratio > 1.05:
                    payload["step_time_variance"] = "moderate"
                else:
                    payload["step_time_variance"] = "low"

        code_context = _extract_code_snippets(script, result.findings)
        if code_context:
            payload["code_context"] = code_context

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/root-cause".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --root-cause"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --root-cause. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "Root cause analysis failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]Root cause analysis failed: {}[/red]".format(e))
        return None


def _render_root_cause_rich(data):
    """Render root cause diagnosis as Rich terminal output."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print("\n[bold cyan]Root Cause Diagnosis[/bold cyan]")

    signal_quality = data.get("signal_quality", "unknown")
    quality_map = {
        "full": "[green]Full (all signals available)[/green]",
        "partial": "[yellow]Partial (some signals missing)[/yellow]",
        "minimal": "[red]Minimal (limited data)[/red]",
    }
    console.print("Signal quality: {}".format(quality_map.get(signal_quality, signal_quality)))

    summary = data.get("diagnostic_summary", "")
    if summary:
        console.print("\n{}".format(summary))

    # Primary hypothesis
    primary = data.get("primary_hypothesis", {})
    if primary:
        cause = primary.get("cause", "unknown").replace("_", " ").title()
        confidence = primary.get("confidence", "unknown")
        explanation = primary.get("explanation", "")

        conf_style = {
            "high": "[bold green]HIGH[/bold green]",
            "medium": "[bold yellow]MEDIUM[/bold yellow]",
            "low": "[bold red]LOW[/bold red]",
        }

        console.print("\n[bold]Primary Cause:[/bold] {} (confidence: {})".format(
            cause, conf_style.get(confidence, confidence)))
        console.print("  {}".format(explanation))

        evidence = primary.get("supporting_evidence", [])
        if evidence:
            console.print("  [dim]Supporting evidence:[/dim]")
            for e in evidence:
                console.print("    + {}".format(e))

        contra = primary.get("contradicting_evidence", [])
        if contra:
            console.print("  [dim]Contradicting evidence:[/dim]")
            for c in contra:
                console.print("    - {}".format(c))

    # Alternative hypotheses
    alternatives = data.get("alternative_hypotheses", [])
    if alternatives:
        console.print("\n[bold]Alternative Hypotheses:[/bold]")
        for alt in alternatives:
            alt_cause = alt.get("cause", "unknown").replace("_", " ").title()
            alt_conf = alt.get("confidence", "unknown")
            alt_expl = alt.get("explanation", "")
            console.print("  [dim]{}[/dim] ({}) — {}".format(alt_cause, alt_conf, alt_expl))

    # Next steps
    steps = data.get("next_steps", [])
    if steps:
        console.print("\n[bold]Next Steps:[/bold]")
        for i, step in enumerate(steps, 1):
            console.print("  {}. {}".format(i, step))


def _fetch_migrate(result, findings, artifact_data, hw, script, target, json_output):
    """Fetch AI migration plan from the API. Returns dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --migrate. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --migrate. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        payload = {
            "target_hardware": target,
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        code_context = _extract_code_snippets(script, result.findings)
        if code_context:
            payload["code_context"] = code_context

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/migrate".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --migrate"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --migrate. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "Migration planning failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]Migration planning failed: {}[/red]".format(e))
        return None


def _render_migrate_rich(data):
    """Render migration plan as Rich terminal output."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    target = data.get("target_hardware", "unknown")
    feasible = data.get("feasible", True)
    confidence = data.get("confidence", "medium")
    summary = data.get("summary", "")

    # Header
    status = "[bold green]FEASIBLE[/bold green]" if feasible else "[bold red]NOT FEASIBLE[/bold red]"
    console.print("\n[bold cyan]Migration Plan[/bold cyan] — {} ({})".format(target, status))

    conf_style = {
        "high": "[green]HIGH[/green]",
        "medium": "[yellow]MEDIUM[/yellow]",
        "low": "[red]LOW[/red]",
    }
    console.print("Confidence: {}".format(conf_style.get(confidence, confidence)))

    if summary:
        console.print("\n{}".format(summary))

    # Steps table
    steps = data.get("steps", [])
    if steps:
        console.print("\n[bold]Migration Steps:[/bold]")

        cat_style = {
            "strategy": "[bold blue]Strategy[/bold blue]",
            "precision": "[bold magenta]Precision[/bold magenta]",
            "batch_size": "[bold yellow]Batch Size[/bold yellow]",
            "dataloader": "[bold green]Dataloader[/bold green]",
            "hardware_feature": "[bold cyan]Hardware[/bold cyan]",
            "config": "[bold white]Config[/bold white]",
        }

        for i, step in enumerate(steps, 1):
            cat = step.get("category", "config")
            cat_label = cat_style.get(cat, cat)
            change = step.get("change", "")
            reasoning = step.get("reasoning", "")
            impact = step.get("impact", "")
            code = step.get("code_change", "")

            console.print("\n  {}. {} {}".format(i, cat_label, change))
            console.print("     [dim]{}[/dim]".format(reasoning))
            if impact:
                console.print("     Impact: [green]{}[/green]".format(impact))
            if code:
                from rich.syntax import Syntax
                console.print("     Code:")
                console.print(Syntax(code, "python", theme="monokai", line_numbers=False, padding=1))

    # Projected comparison
    comparison = data.get("projected_comparison")
    if comparison:
        console.print("\n[bold]Projected Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in comparison.items():
            label = k.replace("_", " ").title()
            console.print("  {}: {}".format(label, v))

    # Warnings
    warnings = data.get("warnings", [])
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in warnings:
            console.print("  ! {}".format(w))


def _fetch_explore(result, findings, artifact_data, hw, script, json_output):
    """Fetch AI Pareto knob exploration from the API. Returns dict or None."""
    from alloc.config import get_token, get_api_url

    token = get_token()
    if not token:
        if json_output:
            import sys
            print('{"error": "Login required for --explore. Run: alloc login"}', file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[yellow]Login required for --explore. Run: alloc login[/yellow]")
        return None

    try:
        import httpx

        api_url = get_api_url()

        workload = _build_workload_context(findings, artifact_data, hw)

        structured_findings = []
        for d in result.findings:
            finding = {
                "rule_id": d.rule_id,
                "severity": d.severity,
                "message": d.title,
                "suggestion": d.rationale,
                "category": d.category,
            }
            if d.evidence:
                finding["evidence"] = {str(k): str(v) for k, v in d.evidence.items()}
            structured_findings.append(finding)

        payload = {
            "findings": structured_findings,
            "workload": workload,
            "hardware": result.hardware_context,
        }

        if result.efficiency:
            payload["metrics"] = result.efficiency

        code_context = _extract_code_snippets(script, result.findings)
        if code_context:
            payload["code_context"] = code_context

        # Pass budget from .alloc.yaml if available
        gpu_context = _load_gpu_context(False)
        if gpu_context and gpu_context.get("budget"):
            budget = gpu_context["budget"]
            if isinstance(budget, (int, float)) and budget > 0:
                payload["budget_hourly"] = float(budget)

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "{}/diagnose/explore".format(api_url),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(token),
                },
            )

            if resp.status_code == 403:
                if json_output:
                    import sys
                    print('{"error": "Pro tier required for --explore"}', file=sys.stderr)
                else:
                    from rich.console import Console
                    Console(stderr=True).print("[yellow]Pro tier required for --explore. Visit alloclabs.com/pricing[/yellow]")
                return None

            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        if json_output:
            import sys
            print('{{"error": "Pareto exploration failed: {}"}}'.format(e), file=sys.stderr)
        else:
            from rich.console import Console
            Console(stderr=True).print("[red]Pareto exploration failed: {}[/red]".format(e))
        return None


def _render_explore_rich(data):
    """Render Pareto exploration results as Rich terminal output."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    summary = data.get("summary", "")
    confidence = data.get("confidence", "medium")
    configs = data.get("configs", [])
    current = data.get("current_config")

    # Header
    conf_style = {
        "high": "[green]HIGH[/green]",
        "medium": "[yellow]MEDIUM[/yellow]",
        "low": "[red]LOW[/red]",
    }
    console.print("\n[bold cyan]Pareto Knob Explorer[/bold cyan]")
    console.print("Confidence: {}".format(conf_style.get(confidence, confidence)))

    if summary:
        console.print("\n{}".format(summary))

    # Current config
    if current:
        console.print("\n[bold]Current Configuration:[/bold]")
        for k, v in current.items():
            console.print("  {}: {}".format(k, v))

    # Configs table
    if configs:
        console.print("\n[bold]Pareto-Optimal Configurations:[/bold]")

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("#", style="dim")
        table.add_column("Config")
        table.add_column("Throughput")
        table.add_column("$/hr")
        table.add_column("$/step")
        table.add_column("VRAM")

        for cfg in configs:
            rank = str(cfg.get("rank", ""))
            label = cfg.get("label", "")
            if cfg.get("is_recommended"):
                label = "[bold green]{}[/bold green]".format(label)
                rank = "[bold green]{}[/bold green]".format(rank)
            throughput = cfg.get("projected_throughput", "-")
            cost_hr = cfg.get("projected_cost_per_hour", "-")
            cost_step = cfg.get("projected_cost_per_step", "-")
            vram = cfg.get("projected_vram_gb", "-")
            table.add_row(rank, label, throughput, cost_hr, cost_step, vram)

        console.print(table)

        # Show knob details for each config
        for cfg in configs:
            rank = cfg.get("rank", "?")
            label = cfg.get("label", "")
            knobs = cfg.get("knobs", {})
            notes = cfg.get("tradeoff_notes")
            if cfg.get("is_recommended"):
                console.print("\n  [bold green]#{} {}[/bold green]".format(rank, label))
            else:
                console.print("\n  [bold]#{} {}[/bold]".format(rank, label))

            for k, v in knobs.items():
                console.print("    {}: {}".format(k, v))
            if notes:
                console.print("    [dim]{}[/dim]".format(notes))

    # Knobs evaluated
    knobs_list = data.get("knobs_evaluated")
    if knobs_list:
        console.print("\n[dim]Knobs evaluated: {}[/dim]".format(", ".join(knobs_list)))

    # Warnings
    warnings = data.get("warnings", [])
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in warnings:
            console.print("  ! {}".format(w))


def _print_rules(json_output: bool) -> None:
    """List all available diagnosis rules."""
    import inspect
    from alloc.diagnosis_rules import ALL_RULES

    _category_map = {
        "dl": "dataloader",
        "mem": "memory",
        "prec": "precision",
        "dist": "distributed",
        "thru": "throughput",
        "xs": "cross-signal",
        "strag": "straggler",
        "reg": "regression",
        "upg": "upgrade",
    }

    rules_info = []
    for fn in ALL_RULES:
        name = fn.__name__  # e.g. "rule_dl001_num_workers_low"
        parts = name.replace("rule_", "", 1).split("_", 1)
        rule_id = parts[0].upper() if parts else name  # "DL001"

        title = ""
        doc = inspect.getdoc(fn)
        if doc:
            title = doc.split("\n")[0].rstrip(".")

        # Category from ID prefix
        prefix = ""
        for ch in rule_id:
            if ch.isalpha():
                prefix += ch.lower()
            else:
                break
        category = _category_map.get(prefix, prefix)

        # Data tier from argument count
        sig = inspect.signature(fn)
        nparams = len(sig.parameters)
        if nparams >= 4:
            tier = "regression"
        elif nparams >= 3:
            tier = "runtime"
        else:
            tier = "ast-only"

        rules_info.append({
            "id": rule_id,
            "title": title,
            "category": category,
            "tier": tier,
        })

    if json_output:
        _print_json({"rules": rules_info, "total": len(rules_info)})
    else:
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Rule", style="bold", no_wrap=True)
        table.add_column("Category", style="dim")
        table.add_column("Tier", style="dim")
        table.add_column("Description")

        for r in rules_info:
            table.add_row(r["id"], r["category"], r["tier"], r["title"])

        console.print()
        console.print(f"[green]alloc diagnose[/green] — {len(rules_info)} rules")
        console.print()
        console.print(table)
        console.print()
        console.print("[dim]Tiers: ast-only (no GPU needed), runtime (requires alloc_artifact), regression (requires --compare)[/dim]")


@app.command()
def scan(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. llama-3-70b)"),
    gpu: str = typer.Option("A100-80GB", "--gpu", "-g", help="Target GPU type"),
    dtype: str = typer.Option("fp16", help="Data type: fp16, bf16, fp32"),
    strategy: str = typer.Option(
        "ddp",
        help=(
            "Strategy: ddp, fsdp, tp, pp, tp+dp, pp+dp, tp+pp+dp, tp+pp+fsdp"
        ),
    ),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    objective: str = typer.Option(
        "best_value",
        help="Objective: cheapest, fastest, fastest_within_budget, best_value",
    ),
    max_budget_hourly: Optional[float] = typer.Option(
        None,
        "--max-budget-hourly",
        help="Optional max $/hr budget for this planning run",
    ),
    num_nodes: Optional[int] = typer.Option(None, "--num-nodes", help="Cluster node count"),
    gpus_per_node: Optional[int] = typer.Option(None, "--gpus-per-node", help="GPUs per node"),
    tp_degree: Optional[int] = typer.Option(None, "--tp-degree", help="Tensor parallel degree"),
    pp_degree: Optional[int] = typer.Option(None, "--pp-degree", help="Pipeline parallel degree"),
    dp_degree: Optional[int] = typer.Option(None, "--dp-degree", help="Data parallel degree"),
    interconnect: str = typer.Option(
        "unknown",
        "--interconnect",
        help="Interconnect: pcie, nvlink, infiniband, unknown",
    ),
    param_count_b: Optional[float] = typer.Option(None, "--param-count-b", "-p", help="Param count in billions (overrides model lookup)"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size (for transformer models)"),
    seq_length: Optional[int] = typer.Option(None, help="Sequence length (for transformer models)"),
    hidden_dim: Optional[int] = typer.Option(None, help="Hidden dimension (for transformer models)"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Remote ghost scan via Alloc API — no GPU needed."""
    import httpx

    # Resolve param count from model name or explicit flag
    resolved_param_count = param_count_b or _model_to_params(model)
    if resolved_param_count is None:
        if json_output:
            _print_json({"error": f"Unknown model: {model}"})
        else:
            console.print(f"[yellow]Unknown model: {model}[/yellow]")
            console.print("[dim]Use --param-count-b to specify directly.[/dim]")
        raise typer.Exit(1)

    api_url = get_api_url()
    token = get_token()

    payload = {
        "entrypoint": f"{model}.py",
        "param_count_b": resolved_param_count,
        "dtype": dtype,
        "strategy": strategy,
        "gpu_type": gpu,
        "num_gpus": num_gpus,
        "objective": objective,
        "max_budget_hourly": max_budget_hourly,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "dp_degree": dp_degree,
        "interconnect_type": interconnect,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_dim": hidden_dim,
    }

    if not json_output:
        console.print(f"[green]alloc[/green] [dim]v{__version__}[/dim] — Remote Ghost Scan")
        console.print(f"[dim]Model: {model} ({resolved_param_count}B) → {gpu} x{num_gpus}[/dim]")
        console.print()

    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        endpoint = "/scans" if token else "/scans/cli"
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{api_url}{endpoint}", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        if json_output:
            _print_json(result)
        else:
            _print_scan_result(result, gpu, strategy)
    except httpx.HTTPStatusError as e:
        if json_output:
            _print_json({"error": f"API error {e.response.status_code}"})
        elif e.response.status_code == 403:
            console.print("[yellow]AI analysis requires a Pro or Enterprise plan.[/yellow]")
            console.print("[dim]The scan still works — just without AI-powered analysis.[/dim]")
        else:
            console.print(f"[red]API error {e.response.status_code}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        if json_output:
            _print_json({"error": f"Cannot connect to {api_url}"})
        else:
            console.print(f"[red]Cannot connect to {api_url}[/red]")
            console.print("[dim]Check ALLOC_API_URL or try: alloc ghost <script.py> for local ghost scan[/dim]")
        raise typer.Exit(1)


@app.command()
def login(
    method: str = typer.Option(
        "password",
        "--method",
        help="Auth method: password (Supabase) or token (paste an access token)",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Access token (implies token login). If omitted, you'll be prompted.",
    ),
    browser: bool = typer.Option(
        False,
        "--browser",
        help="Login via browser OAuth (Google/Microsoft). Required for OAuth-only accounts.",
    ),
    provider: str = typer.Option(
        "google",
        "--provider",
        help="OAuth provider for --browser login: google or azure.",
    ),
):
    """Authenticate with Alloc dashboard."""
    import httpx
    from alloc.config import get_supabase_url, get_supabase_anon_key, load_config, save_config

    # --- Browser OAuth flow ---
    if browser:
        provider = (provider or "").strip().lower()
        if provider not in ("google", "azure"):
            console.print("[red]Invalid --provider. Use: google or azure[/red]")
            raise typer.Exit(2)

        from alloc.browser_auth import browser_login as _browser_login

        supabase_url = get_supabase_url()
        anon_key = get_supabase_anon_key()

        try:
            result = _browser_login(
                provider=provider,
                supabase_url=supabase_url,
                anon_key=anon_key,
            )
        except RuntimeError as e:
            console.print(f"[red]Login failed: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Login failed: {e}[/red]")
            raise typer.Exit(1)

        cfg = load_config()
        cfg["token"] = result["access_token"]
        cfg["refresh_token"] = result["refresh_token"]
        if result.get("email"):
            cfg["email"] = result["email"]
        cfg["api_url"] = get_api_url()
        save_config(cfg)

        who = result.get("email") or "user"
        console.print(f"[green]Logged in as {who}[/green]")
        return

    # --- Token paste flow ---
    method = (method or "").strip().lower()
    if token is not None and method == "password":
        # UX: allow `alloc login --token <...>` without requiring `--method token`.
        method = "token"
    if method not in ("password", "token"):
        console.print("[red]Invalid --method. Use: password or token[/red]")
        raise typer.Exit(2)

    if method == "token":
        access_token = token or typer.prompt("Access token", hide_input=True)
        if not access_token.strip():
            console.print("[red]Login failed: empty token.[/red]")
            raise typer.Exit(1)

        cfg = load_config()
        cfg["token"] = access_token.strip()
        cfg.pop("refresh_token", None)  # token paste mode cannot refresh automatically
        cfg.pop("email", None)
        cfg["api_url"] = get_api_url()
        save_config(cfg)
        console.print("[green]Saved token.[/green]")
        return

    # --- Password flow ---
    email = typer.prompt("Email")
    password = typer.prompt("Password", hide_input=True)

    supabase_url = get_supabase_url()
    anon_key = get_supabase_anon_key()

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{supabase_url}/auth/v1/token?grant_type=password",
                json={"email": email, "password": password},
                headers={
                    "apikey": anon_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        access_token = data.get("access_token", "")
        refresh = data.get("refresh_token", "")
        if not access_token:
            console.print("[red]Login failed: no access token received.[/red]")
            raise typer.Exit(1)

        cfg = load_config()
        cfg["token"] = access_token
        cfg["refresh_token"] = refresh
        cfg["email"] = email
        cfg["api_url"] = get_api_url()
        save_config(cfg)

        console.print(f"[green]Logged in as {email}[/green]")
    except httpx.HTTPStatusError as e:
        detail = "authentication error"
        try:
            body = e.response.json()
            desc = body.get("error_description", "")
            if desc in ("Invalid login credentials", "Email not confirmed"):
                detail = desc.lower()
        except Exception:
            pass
        console.print(f"[red]Login failed: {detail}[/red]")
        if detail == "invalid login credentials":
            console.print("[dim]Tip: signed up via Google/Microsoft? Use: alloc login --browser[/dim]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {supabase_url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def logout():
    """Log out (clear saved token from local config)."""
    from alloc.config import load_config, save_config

    cfg = load_config()
    had = bool(cfg.get("token") or cfg.get("refresh_token") or cfg.get("email"))

    cfg.pop("token", None)
    cfg.pop("refresh_token", None)
    cfg.pop("email", None)
    save_config(cfg)

    if had:
        console.print("[green]Logged out.[/green]")
    else:
        console.print("[dim]No saved session found.[/dim]")

    if os.environ.get("ALLOC_TOKEN"):
        console.print("[dim]Note: ALLOC_TOKEN is set in your environment and still overrides local config.[/dim]")


@app.command()
def whoami(
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Show current CLI auth + org context."""
    import httpx
    from alloc.config import get_api_url, get_token

    api_url = get_api_url()
    token = get_token()
    token_source = "env" if os.environ.get("ALLOC_TOKEN") else "config"

    out = {
        "api_url": api_url,
        "logged_in": bool(token),
        "token_source": token_source if token else None,
    }

    if not token:
        if json_output:
            _print_json(out)
        else:
            console.print("[yellow]Not logged in.[/yellow]")
            console.print("[dim]Run: alloc login[/dim]")
        return

    def _get(path: str) -> dict:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{api_url}{path}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    try:
        profile = _get("/profile")
        fleet = _get("/gpu-fleet")
    except httpx.HTTPStatusError as e:
        new_token = try_refresh_access_token() if e.response.status_code == 401 else None
        if new_token:
            # Retry once with refreshed token
            token = new_token
            profile = _get("/profile")
            fleet = _get("/gpu-fleet")
        else:
            if json_output:
                out["error"] = f"API error {e.response.status_code}"
                _print_json(out)
            else:
                console.print(f"[red]API error {e.response.status_code}[/red]")
                console.print("[dim]Run: alloc login[/dim]")
            raise typer.Exit(1)
    except httpx.ConnectError:
        if json_output:
            out["error"] = f"Cannot connect to {api_url}"
            _print_json(out)
        else:
            console.print(f"[red]Cannot connect to {api_url}[/red]")
        raise typer.Exit(1)

    gpus = fleet.get("gpus") or []
    fleet_count = len([g for g in gpus if g.get("fleet_status") == "in_fleet"])
    explore_count = len([g for g in gpus if g.get("fleet_status") == "explore"])

    out.update({
        "email": profile.get("email"),
        "user_id": profile.get("user_id"),
        "onboarding_complete": profile.get("onboarding_complete"),
        "objective": fleet.get("objective"),
        "priority_cost": fleet.get("priority_cost"),
        "budget_monthly_usd": fleet.get("budget_monthly"),
        "effective_budget_monthly_usd": fleet.get("effective_budget_monthly"),
        "budget_cap_applied": fleet.get("budget_cap_applied"),
        "fleet_count": fleet_count,
        "explore_count": explore_count,
        "org_budget": fleet.get("org_budget"),
    })

    if json_output:
        _print_json(out)
        return

    email = out.get("email") or "(unknown)"
    console.print(f"[green]Logged in[/green] as {email}")
    console.print(f"[dim]API: {api_url}[/dim]")
    if out.get("objective"):
        console.print(f"[dim]Objective: {out['objective']}[/dim]")
    if out.get("effective_budget_monthly_usd") is not None:
        cap = float(out["effective_budget_monthly_usd"])
        line = f"[dim]Effective budget cap: ${cap:.2f}/mo[/dim]"
        if out.get("budget_cap_applied"):
            line += " [yellow](org cap applied)[/yellow]"
        console.print(line)
    console.print(f"[dim]Fleet GPUs: {fleet_count}, Explore GPUs: {explore_count}[/dim]")
    console.print()


@app.command()
def upload(
    artifact: str = typer.Argument(..., help="Path to alloc artifact (.json.gz)"),
):
    """Upload an artifact to the Alloc dashboard."""
    if not os.path.isfile(artifact):
        console.print(f"[red]File not found: {artifact}[/red]")
        raise typer.Exit(1)

    if not artifact.endswith(".json.gz"):
        console.print("[red]Expected a .json.gz artifact file.[/red]")
        raise typer.Exit(1)

    _try_upload(artifact)


@app.command()
def init(
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive: full catalog, 50/50 priority, no budget"),
    from_org: bool = typer.Option(False, "--from-org", help="Pull fleet/budget from your org (requires alloc login)"),
):
    """Create .alloc.yaml with GPU fleet, explore, budget, and priority."""
    from alloc.yaml_config import AllocConfig, FleetEntry, write_alloc_config, validate_config
    from alloc.catalog import list_gpus

    if os.path.isfile(".alloc.yaml"):
        if not yes:
            overwrite = typer.confirm(".alloc.yaml already exists. Overwrite?", default=False)
            if not overwrite:
                raise typer.Exit(0)

    if from_org:
        import httpx

        token = get_token()
        if not token:
            console.print("[yellow]Not logged in. Run `alloc login` first.[/yellow]")
            raise typer.Exit(1)

        api_url = get_api_url()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        def _fetch() -> dict:
            with httpx.Client(timeout=15) as client:
                resp = client.get(f"{api_url}/gpu-fleet", headers=headers)
                resp.raise_for_status()
                return resp.json()

        try:
            payload = _fetch()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                new_token = try_refresh_access_token()
                if not new_token:
                    console.print("[yellow]Session expired. Run `alloc login` again.[/yellow]")
                    raise typer.Exit(1)
                headers["Authorization"] = f"Bearer {new_token}"
                payload = _fetch()
            else:
                console.print(f"[red]API error {e.response.status_code}[/red]")
                console.print(f"[dim]{e.response.text[:200]}[/dim]")
                raise typer.Exit(1)
        except httpx.ConnectError:
            console.print(f"[red]Cannot connect to {api_url}[/red]")
            raise typer.Exit(1)

        catalog = list_gpus()
        display_to_id = {g["display_name"]: g["id"] for g in catalog}

        unknown = []  # type: list[str]

        def _resolve_gpu_id(display_name: str) -> str:
            resolved = display_to_id.get(display_name)
            if resolved:
                return resolved
            unknown.append(display_name)
            return display_name

        fleet_entries = []
        explore_entries = []
        for g in payload.get("gpus") or []:
            status = (g.get("fleet_status") or "").strip().lower()
            if status not in ("in_fleet", "explore"):
                continue

            display = (g.get("display_name") or g.get("gpu_id") or "").strip()
            if not display:
                continue

            gpu_id = _resolve_gpu_id(display)

            count = None
            if g.get("max_count") is not None:
                try:
                    parsed = int(g["max_count"])
                    count = parsed if parsed > 0 else None
                except Exception:
                    count = None

            rate = None
            if g.get("rate") is not None and g.get("rate_source") in ("user", "org"):
                try:
                    parsed = float(g["rate"])
                    rate = parsed if parsed >= 0 else None
                except Exception:
                    rate = None

            entry = FleetEntry(
                gpu=gpu_id,
                cloud=g.get("cloud"),
                count=count,
                rate=rate,
                explore=(status == "explore"),
            )
            if status == "explore":
                explore_entries.append(entry)
            else:
                fleet_entries.append(entry)

        budget_monthly = payload.get("effective_budget_monthly")
        if budget_monthly is not None:
            try:
                budget_monthly = float(budget_monthly)
            except Exception:
                budget_monthly = None
        budget_hourly = None
        if budget_monthly is not None and budget_monthly > 0:
            # Match API behavior: budgets are enforced hourly using monthly/730.
            budget_hourly = round(float(budget_monthly) / 730.0, 4)

        # Preserve org ceiling so CLI can show "(org cap applied)"
        org_budget_raw = payload.get("org_budget") or {}
        org_ceiling = org_budget_raw.get("budget_monthly_usd") if isinstance(org_budget_raw, dict) else None
        if org_ceiling is not None:
            try:
                org_ceiling = float(org_ceiling)
            except Exception:
                org_ceiling = None

        # Resolve org interconnect preference
        org_interconnect = None
        raw_ic = payload.get("interconnect")
        if isinstance(raw_ic, str) and raw_ic.strip().lower() in ("pcie", "nvlink", "infiniband"):
            org_interconnect = raw_ic.strip().lower()

        config = AllocConfig(
            fleet=fleet_entries,
            explore=explore_entries,
            objective=payload.get("objective"),
            priority_cost=int(payload.get("priority_cost") or 50),
            budget_monthly=budget_monthly,
            budget_hourly=budget_hourly,
            org_budget_monthly=org_ceiling,
            interconnect=org_interconnect,
        )

        errors = validate_config(config)
        if errors:
            for err in errors:
                console.print(f"[red]{err}[/red]")
            raise typer.Exit(1)

        path = write_alloc_config(config)
        console.print(f"\n[green]Created {path}[/green] [dim](from org)[/dim]")
        console.print(f"  Fleet: {len(config.fleet)} GPUs, Explore: {len(config.explore)} GPUs")
        if config.objective:
            console.print(f"  Objective: {config.objective}")
        console.print(f"  Priority: cost={config.priority_cost}, latency={config.priority_latency}")
        if config.budget_monthly:
            console.print(f"  Budget: ${config.budget_monthly:.0f}/mo")
            if config.budget_hourly:
                console.print(f"  Budget cap: ${config.budget_hourly:.4f}/hr [dim](monthly/730)[/dim]")
        if config.interconnect:
            console.print(f"  Interconnect: {config.interconnect}")
        if payload.get("budget_cap_applied"):
            console.print("  [yellow]Note: org cap applied to effective budget[/yellow]")
        if unknown:
            uniq = sorted(set(unknown))
            console.print("  [yellow]Warning:[/yellow] some GPUs were not found in the local catalog:")
            for name in uniq[:10]:
                console.print(f"    [dim]- {name}[/dim]")
            if len(uniq) > 10:
                console.print(f"    [dim]... (+{len(uniq) - 10} more)[/dim]")
        console.print()
        return

    gpus = list_gpus()

    if yes:
        # Non-interactive: all GPUs as fleet, balanced priority
        config = AllocConfig(
            fleet=[FleetEntry(gpu=g["id"]) for g in gpus],
            priority_cost=50,
        )
    else:
        # Interactive wizard
        console.print(f"\n[green]alloc init[/green] — GPU fleet configuration\n")

        # 1. Select fleet GPUs
        console.print("[bold]Available GPUs:[/bold]")
        for i, g in enumerate(gpus):
            price_str = ""
            aws = g["pricing"].get("aws")
            if aws:
                price_str = f" ${aws:.2f}/hr"
            console.print(f"  [dim]{i + 1:>2}.[/dim] {g['display_name']:>20}  {g['vram_gb']:.0f} GB{price_str}")

        console.print()
        fleet_input = typer.prompt(
            "Fleet GPUs (comma-separated numbers, or 'all')",
            default="all",
        )

        if fleet_input.strip().lower() == "all":
            fleet_entries = [FleetEntry(gpu=g["id"]) for g in gpus]
        else:
            fleet_entries = []
            for part in fleet_input.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(gpus):
                        fleet_entries.append(FleetEntry(gpu=gpus[idx]["id"]))

        if not fleet_entries:
            console.print("[yellow]No GPUs selected, using full catalog.[/yellow]")
            fleet_entries = [FleetEntry(gpu=g["id"]) for g in gpus]

        # 2. Explore GPUs (from remaining catalog)
        fleet_ids = {e.gpu for e in fleet_entries}
        remaining = [g for g in gpus if g["id"] not in fleet_ids]
        explore_entries = []  # type: list
        if remaining:
            add_explore = typer.confirm("Add explore GPUs (evaluate GPUs you don't have)?", default=False)
            if add_explore:
                console.print("\n[bold]GPUs not in fleet:[/bold]")
                for i, g in enumerate(remaining):
                    console.print(f"  [dim]{i + 1:>2}.[/dim] {g['display_name']:>20}  {g['vram_gb']:.0f} GB")
                explore_input = typer.prompt("Explore GPUs (comma-separated numbers)", default="")
                for part in explore_input.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(remaining):
                            explore_entries.append(FleetEntry(gpu=remaining[idx]["id"], explore=True))

        # 3. Priority
        console.print("\n[bold]Priority[/bold] — how to rank GPU recommendations:")
        console.print("  1. Minimize Cost (cost=80, latency=20)")
        console.print("  2. Balanced (cost=50, latency=50)")
        console.print("  3. Minimize Time (cost=20, latency=80)")
        priority_choice = typer.prompt("Choose", default="2")
        priority_map = {"1": 80, "2": 50, "3": 20}
        priority_cost = priority_map.get(priority_choice.strip(), 50)

        # 4. Budget
        budget_monthly = None  # type: Optional[float]
        set_budget = typer.confirm("\nSet a monthly GPU budget?", default=False)
        if set_budget:
            budget_str = typer.prompt("Monthly budget (USD)", default="0")
            try:
                budget_monthly = float(budget_str)
                if budget_monthly <= 0:
                    budget_monthly = None
            except ValueError:
                budget_monthly = None

        # 5. Interconnect
        console.print("\n[bold]Interconnect[/bold] — GPU-to-GPU communication fabric:")
        console.print("  1. Unknown (auto-detect at runtime)")
        console.print("  2. PCIe")
        console.print("  3. NVLink")
        console.print("  4. InfiniBand")
        ic_choice = typer.prompt("Choose", default="1")
        ic_map = {"1": None, "2": "pcie", "3": "nvlink", "4": "infiniband"}
        interconnect_val = ic_map.get(ic_choice.strip())

        config = AllocConfig(
            fleet=fleet_entries,
            explore=explore_entries,
            priority_cost=priority_cost,
            budget_monthly=budget_monthly,
            interconnect=interconnect_val,
        )

    errors = validate_config(config)
    if errors:
        for err in errors:
            console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)

    path = write_alloc_config(config)
    fleet_count = len(config.fleet)
    explore_count = len(config.explore)
    console.print(f"\n[green]Created {path}[/green]")
    console.print(f"  Fleet: {fleet_count} GPUs, Explore: {explore_count} GPUs")
    console.print(f"  Priority: cost={config.priority_cost}, latency={config.priority_latency}")
    if config.budget_monthly:
        console.print(f"  Budget: ${config.budget_monthly:.0f}/mo")
    if config.interconnect:
        console.print(f"  Interconnect: {config.interconnect}")
    console.print()


@app.command()
def merge(
    artifacts: List[str] = typer.Argument(None, help="Artifact paths to merge (rank-tagged .json.gz files)"),
    output: str = typer.Option("alloc_artifact_merged.json.gz", "--output", "-o", help="Output path"),
    json_output: bool = typer.Option(False, "--json", help="Output merged artifact as JSON"),
):
    """Merge per-rank artifacts into a single artifact with straggler analysis."""
    from alloc.artifact_loader import find_rank_artifacts, merge_artifacts as do_merge

    # Auto-discover rank artifacts if none specified
    paths = list(artifacts) if artifacts else find_rank_artifacts()
    if not paths:
        console.print("[yellow]No rank artifacts found. Run distributed training with callbacks first.[/yellow]")
        console.print("[dim]Looking for alloc_artifact_rank*.json.gz in current directory[/dim]")
        raise typer.Exit(1)

    if not json_output:
        console.print(f"Merging {len(paths)} rank artifact(s)...")
        for p in paths:
            console.print(f"  {p}")

    merged = do_merge(paths)
    if merged is None:
        console.print("[red]Failed to merge artifacts — no valid data found.[/red]")
        raise typer.Exit(1)

    if json_output:
        import json as json_mod
        from dataclasses import asdict
        print(json_mod.dumps(asdict(merged), indent=2, default=str))
        return

    # Write merged artifact
    try:
        from alloc.artifact_writer import write_report
        probe_dict = {}
        if merged.peak_vram_mb is not None:
            probe_dict["peak_vram_mb"] = merged.peak_vram_mb
        if merged.avg_gpu_util is not None:
            probe_dict["avg_gpu_util"] = merged.avg_gpu_util
        if merged.step_time_p50_ms is not None:
            probe_dict["step_time_ms_p50"] = merged.step_time_p50_ms
        if merged.step_time_p90_ms is not None:
            probe_dict["step_time_ms_p90"] = merged.step_time_p90_ms
        if merged.per_rank_peak_vram_mb:
            probe_dict["per_rank_peak_vram_mb"] = merged.per_rank_peak_vram_mb
        if merged.per_rank_step_times_ms:
            probe_dict["per_rank_step_times_ms"] = merged.per_rank_step_times_ms
        if merged.has_phase_timing:
            probe_dict["has_phase_timing"] = True
            for attr in ("phase_forward_ms_p50", "phase_forward_ms_p90",
                        "phase_backward_ms_p50", "phase_backward_ms_p90",
                        "phase_optimizer_ms_p50", "phase_optimizer_ms_p90",
                        "phase_dataloader_ms_p50", "phase_dataloader_ms_p90"):
                val = getattr(merged, attr, None)
                if val is not None:
                    probe_dict[attr] = val
        probe_dict["gpu_name"] = merged.gpu_name
        probe_dict["num_gpus_detected"] = merged.gpu_count
        probe_dict["source"] = "merged"
        probe_dict["is_distributed"] = True
        probe_dict["world_size"] = merged.gpu_count

        hw = {}
        if merged.gpu_name:
            hw["gpu_name"] = merged.gpu_name
        if merged.per_gpu_vram_total_mb:
            hw["gpu_total_vram_mb"] = merged.per_gpu_vram_total_mb
        hw["num_gpus_detected"] = merged.gpu_count

        written = write_report(
            probe_result=probe_dict,
            hardware_context=hw if hw else None,
            output_path=output,
        )
        console.print(f"\n[green]Merged artifact written to: {written}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to write merged artifact: {e}[/red]")
        raise typer.Exit(1)

    # Display summary
    console.print(f"\n[bold]Merge Summary[/bold]")
    console.print(f"  Ranks merged: {merged.gpu_count}")
    if merged.peak_vram_mb:
        console.print(f"  Peak VRAM (max across ranks): {merged.peak_vram_mb:.0f} MB")
    if merged.per_rank_peak_vram_mb:
        console.print(f"  Per-rank peak VRAM: {', '.join(f'{v:.0f}' for v in merged.per_rank_peak_vram_mb)} MB")
    if merged.straggler_ratio is not None:
        ratio = merged.straggler_ratio
        if ratio > 1.2:
            console.print(f"  [yellow]Straggler ratio: {ratio:.2f}x (>1.2x indicates straggler)[/yellow]")
        else:
            console.print(f"  Straggler ratio: {ratio:.2f}x (healthy)")
    if merged.has_phase_timing:
        console.print(f"  Phase timing: available (CUDA events)")


@app.command()
def doctor(
    script: str = typer.Argument(..., help="Python script to analyze (e.g. train.py)"),
    timeout: int = typer.Option(60, "--timeout", help="Max calibration run time in seconds"),
    no_run: bool = typer.Option(False, "--no-run", help="Skip calibration run (use existing artifact only)"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Full training doctor — diagnose, explain, and fix in one command.

    Orchestrates: AST analysis → Ghost VRAM → calibration run → diagnosis →
    AI explanation → AI patches. Requires login (Pro tier for AI features).
    """
    from alloc.code_analyzer import analyze_script as _analyze_script
    from alloc.artifact_loader import load_artifact as _load_artifact, find_artifact as _find_artifact
    from alloc.diagnosis_engine import run_diagnosis, _quick_hw_context
    from alloc.diagnosis_display import (
        print_diagnose_rich, print_diagnose_json as _print_diagnose_json,
    )

    if not os.path.isfile(script):
        if json_output:
            _print_json({"error": "File not found: {}".format(script)})
        else:
            console.print("[red]File not found: {}[/red]".format(script))
        raise typer.Exit(1)

    if not script.endswith(".py"):
        if json_output:
            _print_json({"error": "Expected a Python file: {}".format(script)})
        else:
            console.print("[yellow]Expected a Python file: {}[/yellow]".format(script))
        raise typer.Exit(1)

    json_data = {} if json_output else None
    gpu_context = _load_gpu_context(no_config)

    # --- Step 1: AST Analysis ---
    if not json_output:
        console.print("[bold cyan]Step 1/6[/bold cyan] AST analysis...")
    findings = _analyze_script(script)
    step1_count = len(findings.models) + len(findings.dataloaders) + len(findings.optimizers) + len(findings.distributed) + len(findings.precision)
    if not json_output:
        console.print("  Found {} code patterns".format(step1_count))

    # --- Step 2: Ghost VRAM Estimation ---
    if not json_output:
        console.print("[bold cyan]Step 2/6[/bold cyan] Ghost VRAM estimation...")
    ghost_report = None
    try:
        from alloc.model_extractor import extract_model_info
        from alloc.ghost import ghost as ghost_fn

        info = extract_model_info(script, timeout=30)
        if info and info.param_count > 0:
            ghost_report = ghost_fn(param_count_b=info.param_count / 1e9 if info.param_count > 100 else info.param_count)
            if not json_output:
                console.print("  VRAM estimate: {:.1f} GB".format(ghost_report.total_gb))
        else:
            if not json_output:
                console.print("  [dim]Could not extract model — skipping VRAM estimate[/dim]")
    except Exception:
        if not json_output:
            console.print("  [dim]Ghost scan skipped (model extraction failed)[/dim]")

    # --- Step 3: Calibration Run ---
    artifact_data = None
    artifact_path = None
    if not no_run:
        if not json_output:
            console.print("[bold cyan]Step 3/6[/bold cyan] Calibration run ({}s timeout)...".format(timeout))
        try:
            from alloc.probe import probe_command
            from alloc.artifact_writer import write_report

            result = probe_command(
                ["python", script],
                timeout_seconds=timeout,
                calibrate=True,
            )

            if result.peak_vram_mb > 0:
                callback_data = _read_callback_data()
                probe_dict = {
                    "peak_vram_mb": result.peak_vram_mb,
                    "avg_gpu_util": result.avg_gpu_util,
                    "avg_power_watts": result.avg_power_watts,
                    "duration_seconds": result.duration_seconds,
                    "samples": result.samples,
                    "exit_code": result.exit_code,
                    "probe_mode": result.probe_mode,
                    "stop_reason": result.stop_reason,
                    "gpu_name": result.gpu_name,
                    "gpu_total_vram_mb": result.gpu_total_vram_mb,
                    "command": "python {}".format(script),
                }
                if callback_data:
                    for key in ("step_time_ms_p50", "step_time_ms_p90", "samples_per_sec"):
                        val = callback_data.get(key)
                        if val is not None:
                            probe_dict[key] = val
                hw_context_dict = {
                    "gpu_name": result.gpu_name,
                    "gpu_total_vram_mb": result.gpu_total_vram_mb,
                    "num_gpus_detected": result.num_gpus_detected,
                    "sm_version": result.sm_version,
                }
                artifact_path = write_report(
                    probe_result=probe_dict,
                    hardware_context=hw_context_dict,
                )
                artifact_data = _load_artifact(artifact_path)
                if not json_output:
                    console.print("  Peak VRAM: {:.0f} MB, GPU util: {:.0f}%".format(
                        result.peak_vram_mb, result.avg_gpu_util or 0))
            else:
                if not json_output:
                    console.print("  [yellow]No GPU data captured[/yellow]")

        except Exception as e:
            if not json_output:
                console.print("  [yellow]Calibration failed: {}[/yellow]".format(e))
    else:
        # Try to find existing artifact
        if not json_output:
            console.print("[bold cyan]Step 3/6[/bold cyan] Loading existing artifact...")
        artifact_path = _find_artifact(".")
        if artifact_path:
            artifact_data = _load_artifact(artifact_path)
            if not json_output and artifact_data:
                console.print("  Loaded: {}".format(artifact_path))
        else:
            if not json_output:
                console.print("  [dim]No artifact found — diagnosis will be AST-only[/dim]")

    # --- Step 4: Diagnosis (25 rules) ---
    if not json_output:
        console.print("[bold cyan]Step 4/6[/bold cyan] Running diagnosis (25 rules)...")
    hw = _quick_hw_context()
    if not no_config and gpu_context:
        fleet = gpu_context.get("fleet", [])
        if fleet and hw is None:
            hw = {"gpu_count": len(fleet)}

    diag_result = run_diagnosis(
        findings,
        hardware_context=hw,
        artifact=artifact_data,
    )
    if artifact_path:
        diag_result.artifact_path = artifact_path

    if not json_output:
        total = diag_result.summary.get("total", 0)
        crit = diag_result.summary.get("critical", 0)
        warn = diag_result.summary.get("warning", 0)
        console.print("  {} findings ({} critical, {} warning)".format(total, crit, warn))

    # Show heuristic findings
    if not json_output and diag_result.findings:
        console.print()
        print_diagnose_rich(diag_result, verbose=verbose)

    # --- Step 5: AI Explanation ---
    if not json_output:
        console.print("\n[bold cyan]Step 5/6[/bold cyan] AI explanation...")
    explain_data = None
    if diag_result.findings:
        explain_data = _fetch_explain(diag_result, findings, artifact_data, hw, script, json_output, quiet=json_output)
        if not json_output and explain_data:
            _render_explain_rich(explain_data)
        elif not json_output:
            console.print("  [dim]Skipped (login required or no findings)[/dim]")
    else:
        if not json_output:
            console.print("  [dim]No findings to explain[/dim]")

    # --- Step 6: AI Fix ---
    if not json_output:
        console.print("\n[bold cyan]Step 6/6[/bold cyan] AI code patches...")
    fix_data = None
    if diag_result.findings:
        fix_data = _fetch_fix(diag_result, findings, artifact_data, hw, script, json_output, quiet=json_output)
        if not json_output and fix_data:
            _render_fix_rich(fix_data)
        elif not json_output:
            console.print("  [dim]Skipped (login required or no findings)[/dim]")
    else:
        if not json_output:
            console.print("  [dim]No findings to fix[/dim]")

    # --- Summary ---
    if json_output:
        json_data = {
            "script": script,
            "ghost_vram_gb": ghost_report.total_gb if ghost_report else None,
            "artifact_path": artifact_path,
            "diagnosis": {
                "summary": diag_result.summary,
                "tier": diag_result.tier,
                "findings_count": len(diag_result.findings),
            },
        }
        if explain_data:
            json_data["explain"] = explain_data
        if fix_data:
            json_data["fix"] = fix_data
        _print_json(json_data)
    else:
        console.print("\n[bold green]Doctor complete.[/bold green]")
        if ghost_report:
            console.print("  Ghost VRAM: {:.1f} GB".format(ghost_report.total_gb))
        if artifact_data and artifact_data.peak_vram_mb:
            console.print("  Measured VRAM: {:.0f} MB".format(artifact_data.peak_vram_mb))
        console.print("  Findings: {}".format(len(diag_result.findings)))
        if explain_data:
            console.print("  AI explanation: included")
        if fix_data and fix_data.get("patches"):
            console.print("  AI patches: {} generated".format(len(fix_data["patches"])))
        if artifact_path:
            console.print("  Artifact: {}".format(artifact_path))


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Show last run summary, upload state, and dashboard link."""
    from alloc.artifact_loader import find_artifact, load_artifact

    token = get_token()
    api_url = get_api_url()

    out = {
        "version": __version__,
        "logged_in": bool(token),
        "api_url": api_url,
        "artifact": None,
        "dashboard_url": None,
    }

    # Find latest artifact
    artifact_path = find_artifact()
    if artifact_path:
        art = load_artifact(artifact_path)
        if art:
            art_info = {
                "path": artifact_path,
                "gpu": art.gpu_name,
                "gpu_count": art.gpu_count,
                "peak_vram_mb": round(art.peak_vram_mb, 1) if art.peak_vram_mb else None,
                "avg_gpu_util": round(art.avg_gpu_util, 1) if art.avg_gpu_util else None,
                "step_time_p50_ms": round(art.step_time_p50_ms, 1) if art.step_time_p50_ms else None,
                "throughput": round(art.throughput_samples_per_sec, 2) if art.throughput_samples_per_sec else None,
                "duration_s": round(art.duration_s, 1) if art.duration_s else None,
                "exit_code": art.exit_code,
                "is_oom": art.is_oom,
                "has_phase_timing": art.has_phase_timing,
                "command": art.command,
            }
            out["artifact"] = art_info

    # Dashboard URL (convention: alloclabs.com/dashboard/runs)
    dashboard_base = "https://www.alloclabs.com/dashboard"
    out["dashboard_url"] = f"{dashboard_base}/runs"

    if json_output:
        _print_json(out)
        return

    # Rich output
    console.print(f"[bold]alloc[/bold] v{__version__}")
    console.print()

    # Auth
    if token:
        console.print("[green]Logged in[/green]")
    else:
        console.print("[yellow]Not logged in[/yellow] — run [bold]alloc login[/bold]")

    console.print()

    # Latest artifact
    if out["artifact"]:
        a = out["artifact"]
        console.print("[bold]Last run:[/bold]")
        if a["command"]:
            console.print(f"  Command: [dim]{a['command']}[/dim]")
        console.print(f"  Artifact: [dim]{a['path']}[/dim]")
        if a["gpu"]:
            gpu_str = a["gpu"]
            if a["gpu_count"] and a["gpu_count"] > 1:
                gpu_str = f"{a['gpu_count']}x {gpu_str}"
            console.print(f"  GPU: {gpu_str}")
        if a["peak_vram_mb"]:
            console.print(f"  Peak VRAM: {a['peak_vram_mb']:.0f} MB")
        if a["avg_gpu_util"] is not None:
            console.print(f"  GPU util: {a['avg_gpu_util']:.0f}%")
        if a["step_time_p50_ms"]:
            console.print(f"  Step time (p50): {a['step_time_p50_ms']:.0f} ms")
        if a["throughput"]:
            console.print(f"  Throughput: {a['throughput']:.1f} samples/sec")
        if a["duration_s"]:
            m, s = divmod(int(a["duration_s"]), 60)
            console.print(f"  Duration: {m}m {s}s")
        if a["is_oom"]:
            console.print("  [red]OOM detected[/red]")
        elif a["exit_code"] is not None and a["exit_code"] != 0:
            console.print(f"  [yellow]Exit code: {a['exit_code']}[/yellow]")
        if a["has_phase_timing"]:
            console.print("  Phase timing: [green]available[/green]")
    else:
        console.print("[dim]No artifact found in current directory.[/dim]")
        console.print("[dim]Run: alloc run python train.py[/dim]")

    console.print()

    # Upload hint
    if out["artifact"] and not token:
        console.print("[dim]Upload to dashboard: alloc login && alloc upload " + out["artifact"]["path"] + "[/dim]")
    elif out["artifact"] and token:
        console.print(f"[dim]Upload: alloc upload {out['artifact']['path']}[/dim]")
        console.print(f"[dim]Dashboard: {out['dashboard_url']}[/dim]")
    elif token:
        console.print(f"[dim]Dashboard: {out['dashboard_url']}[/dim]")


@app.command()
def version():
    """Show alloc version."""
    console.print(f"alloc v{__version__}")


# ---------------------------------------------------------------------------
# Catalog subcommands
# ---------------------------------------------------------------------------
catalog_app = typer.Typer(
    name="catalog",
    help="Browse the GPU hardware catalog.",
    no_args_is_help=True,
)
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("list")
def catalog_list(
    sort: str = typer.Option("vram", help="Sort by: vram, cost, tflops, name"),
):
    """List all GPUs in the catalog."""
    from rich.table import Table
    from alloc.catalog import list_gpus

    gpus = list_gpus()

    if sort == "cost":
        gpus.sort(key=lambda g: next(iter(g["pricing"].values()), 999))
    elif sort == "tflops":
        gpus.sort(key=lambda g: g["bf16_tflops"], reverse=True)
    elif sort == "name":
        gpus.sort(key=lambda g: g["display_name"])
    # default: vram (already sorted)

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("GPU", style="bold", no_wrap=True)
    table.add_column("VRAM", justify="right")
    table.add_column("BF16 TFLOPS", justify="right")
    table.add_column("BW (GB/s)", justify="right")
    table.add_column("TDP", justify="right")
    table.add_column("Arch", style="dim", no_wrap=True)
    table.add_column("$/hr (AWS)", justify="right")

    for g in gpus:
        aws_price = g["pricing"].get("aws")
        price_str = f"${aws_price:.2f}" if aws_price else "\u2014"
        table.add_row(
            g["display_name"],
            f"{g['vram_gb']:.0f} GB",
            f"{g['bf16_tflops']:.0f}" if g["bf16_tflops"] else "\u2014",
            f"{g['bandwidth_gbps']:.0f}",
            f"{g['tdp_watts']}W",
            g["architecture"],
            price_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(gpus)} GPUs in catalog[/dim]")


@catalog_app.command("show")
def catalog_show(
    gpu_id: str = typer.Argument(..., help="GPU ID or alias (e.g. nvidia-h100-sxm-80gb or H100)"),
):
    """Show detailed specs for a single GPU."""
    from rich.panel import Panel
    from rich.table import Table
    from alloc.catalog import get_gpu, _ALIASES

    gpu = get_gpu(gpu_id)
    if not gpu:
        console.print(f"[red]Unknown GPU: {gpu_id}[/red]")
        console.print(f"[dim]Available aliases: {', '.join(sorted(_ALIASES.keys()))}[/dim]")
        raise typer.Exit(1)

    # Specs table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("ID", gpu["id"])
    table.add_row("Display Name", gpu["display_name"])
    table.add_row("Vendor", gpu["vendor"])
    table.add_row("Architecture", gpu["architecture"])
    table.add_row("VRAM", f"{gpu['vram_gb']:.0f} GB")
    table.add_row("Memory BW", f"{gpu['bandwidth_gbps']:.0f} GB/s")
    table.add_row("BF16 TFLOPS", f"{gpu['bf16_tflops']:.0f}" if gpu["bf16_tflops"] else "\u2014")
    table.add_row("FP16 TFLOPS", f"{gpu['fp16_tflops']:.0f}" if gpu["fp16_tflops"] else "\u2014")
    table.add_row("FP32 TFLOPS", f"{gpu['fp32_tflops']:.1f}")
    table.add_row("TF32 TFLOPS", f"{gpu['tf32_tflops']:.0f}" if gpu["tf32_tflops"] else "\u2014")
    table.add_row("TDP", f"{gpu['tdp_watts']} W")

    # Interconnect
    ic = gpu.get("interconnect")
    if ic:
        if ic.get("nvlink_gen"):
            table.add_row("NVLink", f"Gen {ic['nvlink_gen']} ({ic.get('nvlink_bw_gbps', '?')} GB/s)")
        if ic.get("pcie_gen"):
            table.add_row("PCIe", f"Gen {ic['pcie_gen']}")

    console.print(Panel(table, title=f"[bold]{gpu['display_name']}[/bold]", border_style="green"))

    # Pricing
    if gpu["pricing"]:
        console.print("\n  [bold]Pricing[/bold]")
        for cloud, price in sorted(gpu["pricing"].items()):
            console.print(f"    {cloud:>12}  ${price:.2f}/hr")
    else:
        console.print("\n  [dim]No pricing data available[/dim]")

    console.print()


def _print_json(data: dict) -> None:
    """Print a dict as formatted JSON to stdout."""
    print(json_mod.dumps(data, indent=2, default=str))


def _try_upload(artifact_path: str) -> None:
    """Attempt to upload an artifact. Prints status, never raises."""
    try:
        import httpx
        from alloc.upload import upload_artifact, UploadLimitError

        token = get_token()
        if not token:
            console.print("[yellow]Not logged in. Run `alloc login` first.[/yellow]")
            return

        api_url = get_api_url()
        console.print(f"[dim]Uploading to {api_url}...[/dim]")
        try:
            result = upload_artifact(artifact_path, api_url, token)
        except httpx.HTTPStatusError as e:
            new_token = try_refresh_access_token() if e.response.status_code == 401 else None
            if new_token:
                console.print("[dim]Session expired; refreshed token. Retrying...[/dim]")
                result = upload_artifact(artifact_path, api_url, new_token)
            else:
                raise

        run_id = result.get("run_id", "unknown")
        console.print(f"[green]Uploaded.[/green] Run ID: {run_id}")
        budget_warning = result.get("budget_warning")
        if budget_warning:
            console.print(f"[yellow]{budget_warning}[/yellow]")
    except UploadLimitError as e:
        detail = e.detail
        used = detail.get("used", "?")
        limit = detail.get("limit", "?")
        console.print(f"[yellow]Upload limit reached ({used}/{limit} this month).[/yellow]")
        console.print("[dim]Upgrade to Pro for more uploads. Artifact is saved locally.[/dim]")
        console.print(f"[dim]  {artifact_path}[/dim]")
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            console.print("[yellow]Session expired. Run `alloc login` again.[/yellow]")
        else:
            console.print(f"[yellow]Upload failed: API error {status}[/yellow]")
            console.print(f"[dim]{e.response.text[:200]}[/dim]")
        console.print(f"[dim]You can retry later: alloc upload {artifact_path}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Upload failed: {e}[/yellow]")
        console.print(f"[dim]You can retry later: alloc upload {artifact_path}[/dim]")


def _read_callback_data():
    # type: () -> Optional[dict]
    """Read callback data from .alloc_callback.json."""
    try:
        callback_path = os.path.join(os.getcwd(), ".alloc_callback.json")
        if os.path.isfile(callback_path):
            with open(callback_path, "r") as f:
                return json_mod.load(f)
    except Exception:
        pass
    return None


def _print_scan_result(result: dict, gpu: str, strategy: str) -> None:
    """Print remote scan result."""
    from rich.table import Table
    from rich.panel import Panel

    vram = result.get("vram_breakdown", {})
    verdict = result.get("strategy_verdict", {})

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Component", style="dim")
    table.add_column("Size", justify="right", style="bold")

    table.add_row("Model weights", f"{vram.get('weights_gb', 0):.2f} GB")
    table.add_row("Optimizer (Adam)", f"{vram.get('optimizer_gb', 0):.2f} GB")
    table.add_row("Activations (est.)", f"{vram.get('activations_gb', 0):.2f} GB")
    table.add_row("Buffer (10%)", f"{vram.get('buffer_gb', 0):.2f} GB")
    table.add_row("", "")
    table.add_row("[bold]Total VRAM[/bold]", f"[bold]{vram.get('total_gb', 0):.2f} GB[/bold]")

    console.print(Panel(table, title="VRAM Breakdown", border_style="green", padding=(1, 2)))

    feasible = verdict.get("feasible", False)
    status = "[green]FEASIBLE[/green]" if feasible else "[red]INFEASIBLE[/red]"
    console.print(f"  Strategy: {strategy.upper()} on {gpu} — {status}")
    if verdict.get("strategy_topology"):
        console.print(f"  [dim]Topology: {verdict['strategy_topology']}[/dim]")
    if verdict.get("objective"):
        console.print(f"  [dim]Objective: {verdict['objective']}[/dim]")
    if verdict.get("effective_max_budget_hourly") is not None:
        if verdict.get("budget_cap_applied"):
            user_cap = verdict.get("user_max_budget_hourly")
            org_cap = verdict.get("org_max_budget_hourly")
            eff_cap = float(verdict["effective_max_budget_hourly"])
            console.print("  [dim]Budget:[/dim]")
            if user_cap is not None:
                console.print(f"    [dim]Your cap:      ${float(user_cap):.4f}/hr[/dim]")
            if org_cap is not None:
                console.print(f"    [dim]Org ceiling:   ${float(org_cap):.4f}/hr[/dim]")
            console.print(f"    [dim]Effective cap: ${eff_cap:.4f}/hr (org ceiling applied)[/dim]")
        else:
            console.print(
                f"  [dim]Effective budget cap: ${float(verdict['effective_max_budget_hourly']):.4f}/hr[/dim]"
            )

    if not feasible and verdict.get("recommendation"):
        rec = verdict.get("best_recommendation") or {}
        if rec.get("strategy_topology"):
            console.print(
                f"  [yellow]Suggestion: {rec.get('strategy', '').upper()} "
                f"({rec.get('strategy_topology')})[/yellow]"
            )
        else:
            console.print(f"  [yellow]Suggestion: switch to {str(verdict['recommendation']).upper()}[/yellow]")

    if verdict.get("reason"):
        console.print(f"  [dim]{verdict['reason']}[/dim]")

    # Cost estimate if present
    cost = result.get("est_cost_per_hour")
    if cost is not None:
        console.print(f"  [dim]Est. cost: ~${cost:.2f}/hr[/dim]")

    # AI analysis if present
    ai_analysis = result.get("euler_analysis")
    if ai_analysis and ai_analysis.get("summary"):
        console.print()
        console.print(f"  [bold cyan]AI Analysis[/bold cyan]")
        console.print(f"  {ai_analysis['summary']}")
        for rec in ai_analysis.get("recommendations", []):
            console.print(f"  [dim]• {rec}[/dim]")

    console.print()


def _model_to_params(model: str) -> Optional[float]:
    """Look up model param count by name."""
    from alloc.model_registry import lookup_model_params
    return lookup_model_params(model)


# ---------------------------------------------------------------------------
# GPU context helpers
# ---------------------------------------------------------------------------

def _load_gpu_context(no_config: bool) -> Optional[dict]:
    """Load GPU context from .alloc.yaml. Returns None if disabled or not found."""
    if no_config:
        return None
    try:
        from alloc.yaml_config import load_alloc_config
        config = load_alloc_config()
        if config is None:
            return None
        return {
            "fleet": config.fleet_gpu_ids,
            "explore": config.explore_gpu_ids,
            "objective": config.objective,
            "priority_cost": config.priority_cost,
            "priority_latency": config.priority_latency,
            "budget_monthly": config.budget_monthly,
            "budget_hourly": config.budget_hourly,
            "rate_overrides": config.rate_overrides,
            "org_budget_monthly": config.org_budget_monthly,
            "interconnect": config.interconnect,
        }
    except Exception:
        return None


def _print_gpu_context_summary(ctx: dict) -> None:
    """Print a one-line GPU context summary."""
    fleet_count = len(ctx.get("fleet", []))
    explore_count = len(ctx.get("explore", []))
    parts = []
    if fleet_count:
        parts.append(f"{fleet_count} fleet")
    if explore_count:
        parts.append(f"{explore_count} explore")
    priority = f"cost={ctx.get('priority_cost', 50)}"
    parts.append(priority)
    budget = ctx.get("budget_monthly")
    if budget:
        parts.append(f"${budget:.0f}/mo")
    console.print(f"  [dim]GPU context: {', '.join(parts)} (from .alloc.yaml)[/dim]")
    console.print()


def _print_gpu_context_detail(ctx: dict) -> None:
    """Print detailed GPU context for verbose mode."""
    from rich.panel import Panel
    lines = []
    fleet = ctx.get("fleet", [])
    if fleet:
        lines.append(f"  Fleet GPUs      {', '.join(fleet)}")
    explore = ctx.get("explore", [])
    if explore:
        lines.append(f"  Explore GPUs    {', '.join(explore)}")
    lines.append(f"  Priority        cost={ctx.get('priority_cost', 50)}, latency={ctx.get('priority_latency', 50)}")
    budget_m = ctx.get("budget_monthly")
    if budget_m:
        lines.append(f"  Budget          ${budget_m:.0f}/mo")
    overrides = ctx.get("rate_overrides", {})
    if overrides:
        for gpu, rate in overrides.items():
            lines.append(f"  Rate override   {gpu}: ${rate:.2f}/hr")
    console.print(Panel("\n".join(lines), title="GPU Context (.alloc.yaml)", border_style="cyan", padding=(1, 0)))


def _infer_parallel_topology_from_env(*, num_gpus_detected: int, config_interconnect: Optional[str] = None, detected_interconnect: Optional[str] = None) -> dict:
    """Infer distributed topology hints from common launcher env vars."""

    def _get_int(name: str) -> Optional[int]:
        val = os.environ.get(name)
        if val is None:
            return None
        try:
            parsed = int(val)
            return parsed if parsed > 0 else None
        except Exception:
            return None

    world_size = _get_int("WORLD_SIZE")
    local_world = _get_int("LOCAL_WORLD_SIZE")
    nnodes = _get_int("NNODES")
    gpn = local_world or num_gpus_detected or 1

    if nnodes is None and world_size is not None and gpn > 0 and world_size % gpn == 0:
        nnodes = max(1, world_size // gpn)

    tp = _get_int("TP_SIZE") or _get_int("TENSOR_PARALLEL_SIZE")
    pp = _get_int("PP_SIZE") or _get_int("PIPELINE_PARALLEL_SIZE")
    dp = _get_int("DP_SIZE") or _get_int("DATA_PARALLEL_SIZE")

    if dp is None and world_size is not None:
        denom = (tp or 1) * (pp or 1)
        if denom > 0 and world_size % denom == 0:
            dp = max(1, world_size // denom)

    # Priority: env var > NVML probe detection > .alloc.yaml config > "unknown"
    interconnect = os.environ.get("ALLOC_INTERCONNECT", "").strip().lower()
    if not interconnect and detected_interconnect:
        interconnect = detected_interconnect
    if not interconnect and config_interconnect:
        interconnect = config_interconnect
    if not interconnect:
        interconnect = "unknown"
    if interconnect not in ("pcie", "nvlink", "nvlink_switch", "nvlink_p2p", "infiniband", "unknown"):
        interconnect = "unknown"

    return {
        "num_nodes": nnodes or 1,
        "gpus_per_node": gpn,
        "tp_degree": tp,
        "pp_degree": pp,
        "dp_degree": dp,
        "interconnect_type": interconnect,
    }


def _objective_from_context(ctx: Optional[dict]) -> str:
    """Resolve ranking objective from .alloc.yaml."""
    if not ctx:
        return "best_value"
    explicit = (ctx.get("objective") or "").strip().lower()
    if explicit in ("cheapest", "fastest", "fastest_within_budget", "best_value"):
        return explicit
    priority_cost = int(ctx.get("priority_cost", 50))
    if priority_cost >= 70:
        return "cheapest"
    if priority_cost <= 30:
        return "fastest"
    return "best_value"


def _build_budget_context(gpu_context, probe_result):
    # type: (Optional[dict], Any) -> Optional[dict]
    """Build budget context dict for verdict display from gpu_context + probe result."""
    if not gpu_context:
        return None
    budget_monthly = gpu_context.get("budget_monthly")
    rate_overrides = gpu_context.get("rate_overrides") or {}

    # Look up cost_per_hour: first try rate overrides from .alloc.yaml, then catalog
    gpu_name = getattr(probe_result, "gpu_name", None) or ""
    cost_per_hour = None

    # Check rate overrides (user's .alloc.yaml rates)
    for gpu_id, rate in rate_overrides.items():
        if gpu_id.lower() in gpu_name.lower() or gpu_name.lower() in gpu_id.lower():
            cost_per_hour = rate
            break

    # Fallback: look up from catalog
    if cost_per_hour is None and gpu_name:
        try:
            from alloc.catalog import get_default_rate
            cost_per_hour = get_default_rate(gpu_name)
        except Exception:
            pass

    if cost_per_hour is None and budget_monthly is None:
        return None

    num_gpus = getattr(probe_result, "num_gpus_detected", 1) or 1
    if cost_per_hour is not None:
        cost_per_hour = cost_per_hour * num_gpus

    org_budget = gpu_context.get("org_budget_monthly")
    budget_cap_applied = (
        budget_monthly is not None
        and org_budget is not None
        and budget_monthly >= org_budget
    )

    return {
        "cost_per_hour": cost_per_hour,
        "budget_monthly": budget_monthly,
        "org_budget_monthly": org_budget,
        "budget_cap_applied": budget_cap_applied,
    }


def _max_budget_hourly_from_context(ctx: Optional[dict]) -> Optional[float]:
    """Resolve hourly budget from .alloc.yaml context."""
    if not ctx:
        return None
    val = ctx.get("budget_hourly")
    if val is None:
        return None
    try:
        parsed = float(val)
        return parsed if parsed > 0 else None
    except Exception:
        return None
