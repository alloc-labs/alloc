"""Alloc CLI — GPU intelligence for ML training.

Commands:
    alloc ghost <script.py>    Ghost scan — static VRAM analysis without executing the model
    alloc run <command...>     Wrap training with probe monitoring, write artifact
    alloc scan --model <name>  Remote ghost scan via API — no GPU needed
    alloc diagnose <script.py> Analyze training script for performance issues
    alloc login                Authenticate with Alloc dashboard
    alloc upload <artifact>    Upload an artifact to the Alloc dashboard
    alloc version              Show version
"""

from __future__ import annotations

import os
import sys
from typing import Optional

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
    batch_size: int = typer.Option(32, help="Training batch size"),
    seq_length: int = typer.Option(2048, help="Sequence length"),
    hidden_dim: int = typer.Option(4096, help="Hidden dimension"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed VRAM formula breakdown"),
    param_count_b: Optional[float] = typer.Option(None, "--param-count-b", "-p", help="Param count in billions (skip script analysis)"),
    timeout: int = typer.Option(60, "--timeout", help="Max seconds for model extraction"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml (use catalog defaults)"),
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

    report = ghost_fn(
        param_count=info.param_count,
        dtype=resolved_dtype,
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
    )
    report.extraction_method = info.method

    if json_output:
        data = report.to_dict()
        if gpu_context:
            data["gpu_context"] = gpu_context
        _print_json(data)
    else:
        print_ghost_report(report)
        if gpu_context and not verbose:
            _print_gpu_context_summary(gpu_context)
        if verbose:
            from alloc.display import print_verbose_ghost
            print_verbose_ghost(report)
            if gpu_context:
                _print_gpu_context_detail(gpu_context)


@app.command()
def run(
    command: list[str] = typer.Argument(..., help="Command to run (e.g. python train.py)"),
    timeout: int = typer.Option(120, help="Max calibration time in seconds"),
    gpu: int = typer.Option(0, help="GPU index to monitor"),
    save: bool = typer.Option(True, help="Save artifact to disk"),
    out: Optional[str] = typer.Option(None, "--out", help="Output path for artifact"),
    upload: bool = typer.Option(False, "--upload", help="Upload artifact to Alloc dashboard after run (automatic when logged in)"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Skip auto-upload even when logged in"),
    full: bool = typer.Option(False, "--full", help="Monitor full training run instead of calibrating"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show hardware context, sample dump, recommendation reasoning"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml (use catalog defaults)"),
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
            console.print(f"[dim]Auto-stop when metrics stabilize (timeout: {timeout}s)[/dim]")
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

    # Auto-upload when logged in, --upload flag, or ALLOC_UPLOAD env var
    # --no-upload overrides everything
    should_auto = not no_upload and (upload or should_upload() or bool(get_token()))
    if artifact_path and should_auto:
        _try_upload(artifact_path)
    elif artifact_path and not should_auto and not no_upload and not get_token():
        if not json_output:
            console.print("[dim]Tip: alloc login --browser to auto-upload runs to your dashboard[/dim]")

    if result.exit_code and result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command()
def diagnose(
    script: str = typer.Argument(..., help="Python script to analyze (e.g. train.py)"),
    diff: bool = typer.Option(False, "--diff", help="Output unified diff patches"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show rule details and references"),
    no_config: bool = typer.Option(False, "--no-config", help="Skip .alloc.yaml"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter: critical, warning, info"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter: dataloader, memory, precision, distributed, throughput"),
):
    """Analyze a training script for performance issues."""
    from alloc.code_analyzer import analyze_script as _analyze_script
    from alloc.diagnosis_engine import run_diagnosis, _quick_hw_context
    from alloc.diagnosis_display import print_diagnose_rich, print_diagnose_diff, print_diagnose_json

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

    # Build hardware context
    hw = _quick_hw_context()

    # Merge .alloc.yaml GPU context if available
    if not no_config:
        gpu_context = _load_gpu_context(False)
        if gpu_context and hw is None:
            # Use config hints if no live hardware
            ic = gpu_context.get("interconnect")
            fleet = gpu_context.get("fleet", [])
            if fleet:
                hw = {"gpu_count": len(fleet)}
        elif gpu_context and hw is not None:
            # Supplement live hardware with config
            fleet = gpu_context.get("fleet", [])
            if fleet and hw.get("gpu_count", 1) <= 1:
                pass  # Don't override live detection

    result = run_diagnosis(findings, hardware_context=hw)

    # Apply filters
    if severity:
        sev = severity.strip().lower()
        result.findings = [d for d in result.findings if d.severity == sev]
        result.summary = _recount_summary(result.findings)
    if category:
        cat = category.strip().lower()
        result.findings = [d for d in result.findings if d.category == cat]
        result.summary = _recount_summary(result.findings)

    # Output
    if json_output:
        print_diagnose_json(result)
    elif diff:
        print_diagnose_diff(result)
    else:
        print_diagnose_rich(result, verbose=verbose)


def _recount_summary(findings):
    """Recount summary after filtering."""
    summary = {"critical": 0, "warning": 0, "info": 0, "total": len(findings)}
    for d in findings:
        if d.severity in summary:
            summary[d.severity] += 1
    return summary


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
    batch_size: int = typer.Option(32, help="Batch size"),
    seq_length: int = typer.Option(2048, help="Sequence length"),
    hidden_dim: int = typer.Option(4096, help="Hidden dimension"),
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
    if interconnect not in ("pcie", "nvlink", "infiniband", "unknown"):
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
