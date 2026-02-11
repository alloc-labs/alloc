"""Alloc CLI — GPU intelligence for ML training.

Commands:
    alloc ghost <script.py>    Ghost scan — static VRAM analysis without executing the model
    alloc run <command...>     Wrap training with probe monitoring, write artifact
    alloc scan --model <name>  Remote ghost scan via API — no GPU needed
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
from alloc.config import get_api_url, get_token, should_upload

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
):
    """Ghost scan — static VRAM analysis without executing the model."""
    from alloc.ghost import ghost as ghost_fn
    from alloc.display import print_ghost_report

    # Try to extract param count from the script by importing it
    param_count = _extract_param_count(script)

    if param_count is None:
        if json_output:
            _print_json({"error": f"Could not extract model from {script}"})
        else:
            console.print(f"[yellow]Could not extract model from {script}.[/yellow]")
            console.print("[dim]Tip: Use 'alloc scan --param-count-b 7.0' for direct param count input.[/dim]")
        raise typer.Exit(1)

    report = ghost_fn(
        param_count=param_count,
        dtype=dtype,
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
    )

    if json_output:
        _print_json(report.to_dict())
    else:
        print_ghost_report(report)
        if verbose:
            from alloc.display import print_verbose_ghost
            print_verbose_ghost(report)


@app.command()
def run(
    command: list[str] = typer.Argument(..., help="Command to run (e.g. python train.py)"),
    timeout: int = typer.Option(120, help="Max calibration time in seconds"),
    gpu: int = typer.Option(0, help="GPU index to monitor"),
    save: bool = typer.Option(True, help="Save artifact to disk"),
    out: Optional[str] = typer.Option(None, "--out", help="Output path for artifact"),
    upload: bool = typer.Option(False, "--upload", help="Upload artifact to Alloc dashboard after run"),
    full: bool = typer.Option(False, "--full", help="Monitor full training run instead of calibrating"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show hardware context, sample dump, recommendation reasoning"),
    probe_steps: Optional[int] = typer.Option(None, "--probe-steps", hidden=True, help="[Deprecated] Use default calibration instead"),
):
    """Run a training command with GPU monitoring."""
    from alloc.probe import probe_command
    from alloc.display import print_probe_result, print_verdict
    from alloc.artifact_writer import write_report

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

    # Deprecation warnings
    if probe_steps is not None and full:
        console.print("[yellow]--probe-steps ignored when --full is set.[/yellow]")
        probe_steps = None
    elif probe_steps is not None:
        console.print("[yellow]--probe-steps is deprecated. Default calibration mode auto-stops when metrics stabilize.[/yellow]")

    # Determine mode
    calibrate = not full and probe_steps is None

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
        mode_label = "Short-run probe (deprecated)"

    if not json_output:
        console.print(f"[green]alloc[/green] [dim]v{__version__}[/dim] — {mode_label}")
        console.print(f"[dim]Command: {' '.join(command)}[/dim]")
        if calibrate:
            console.print(f"[dim]Auto-stop when metrics stabilize (timeout: {timeout}s)[/dim]")
        elif probe_steps:
            console.print(f"[dim]Profiling for ~{probe_steps} samples (auto-stop when stable)[/dim]")
        console.print()

    result = probe_command(
        command,
        timeout_seconds=effective_timeout,
        gpu_index=gpu,
        probe_steps=probe_steps,
        calibrate=calibrate,
    )

    if not json_output:
        if result.error and "pynvml" in result.error:
            console.print(f"[yellow]{result.error}[/yellow]")
            console.print("[dim]Process ran without GPU monitoring.[/dim]")
        elif result.error:
            console.print(f"[red]Error: {result.error}[/red]")

        if probe_steps and result.steps_profiled:
            console.print(f"[dim]Profiled {result.steps_profiled} samples in {result.duration_seconds:.1f}s[/dim]")

    # Read step count from sidecar (written by framework callbacks)
    step_count = _read_step_count()

    # Discover environment context (git, container, Ray)
    from alloc.context import discover_context
    env_context = discover_context()

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
        }
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

    if json_output:
        from alloc.display import build_verdict_dict
        data = build_verdict_dict(result, artifact_path=artifact_path, step_count=step_count)
        if result.error:
            data["error"] = result.error
        _print_json(data)
    else:
        # Display verdict for all modes when we have GPU data
        if result.peak_vram_mb > 0:
            print_verdict(result, artifact_path=artifact_path, step_count=step_count)
            if verbose:
                from alloc.display import print_verbose_run
                print_verbose_run(result, step_count=step_count)
        elif artifact_path:
            console.print(f"[dim]Artifact saved: {artifact_path}[/dim]")

    # Next action hint
    if artifact_path and not (upload or should_upload()):
        pass  # print_verdict already shows "Next: alloc upload ..."

    # Upload if --upload flag or ALLOC_UPLOAD env var
    if artifact_path and (upload or should_upload()):
        _try_upload(artifact_path)

    if result.exit_code and result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command()
def scan(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. llama-3-70b)"),
    gpu: str = typer.Option("A100-80GB", "--gpu", "-g", help="Target GPU type"),
    dtype: str = typer.Option("fp16", help="Data type: fp16, bf16, fp32"),
    strategy: str = typer.Option("ddp", help="Strategy: ddp, fsdp, deepspeed"),
    num_gpus: int = typer.Option(4, help="Number of GPUs"),
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

        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{api_url}/scans/cli", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        if json_output:
            _print_json(result)
        else:
            _print_scan_result(result, gpu, strategy)
    except httpx.HTTPStatusError as e:
        if json_output:
            _print_json({"error": f"API error {e.response.status_code}", "detail": e.response.text[:200]})
        elif e.response.status_code == 403:
            console.print("[yellow]AI analysis requires a Pro or Enterprise plan.[/yellow]")
            console.print("[dim]The scan still works — just without Euler AI analysis.[/dim]")
        else:
            console.print(f"[red]API error {e.response.status_code}[/red]")
            console.print(f"[dim]{e.response.text[:200]}[/dim]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        if json_output:
            _print_json({"error": f"Cannot connect to {api_url}"})
        else:
            console.print(f"[red]Cannot connect to {api_url}[/red]")
            console.print("[dim]Check ALLOC_API_URL or try: alloc ghost <script.py> for local ghost scan[/dim]")
        raise typer.Exit(1)


@app.command()
def login():
    """Authenticate with Alloc dashboard."""
    import httpx
    from alloc.config import get_supabase_url, get_supabase_anon_key, load_config, save_config

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

        token = data.get("access_token", "")
        refresh = data.get("refresh_token", "")
        if not token:
            console.print("[red]Login failed: no access token received.[/red]")
            raise typer.Exit(1)

        cfg = load_config()
        cfg["token"] = token
        cfg["refresh_token"] = refresh
        cfg["email"] = email
        cfg["api_url"] = get_api_url()
        save_config(cfg)

        console.print(f"[green]Logged in as {email}[/green]")
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("error_description", e.response.text[:200])
        except Exception:
            detail = e.response.text[:200]
        console.print(f"[red]Login failed: {detail}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {supabase_url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise typer.Exit(1)


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
        from alloc.upload import upload_artifact, UploadLimitError

        token = get_token()
        if not token:
            console.print("[yellow]Not logged in. Run `alloc login` first.[/yellow]")
            return

        api_url = get_api_url()
        console.print(f"[dim]Uploading to {api_url}...[/dim]")
        result = upload_artifact(artifact_path, api_url, token)
        run_id = result.get("run_id", "unknown")
        console.print(f"[green]Uploaded.[/green] Run ID: {run_id}")
    except UploadLimitError as e:
        detail = e.detail
        used = detail.get("used", "?")
        limit = detail.get("limit", "?")
        console.print(f"[yellow]Upload limit reached ({used}/{limit} this month).[/yellow]")
        console.print("[dim]Upgrade to Pro for more uploads. Artifact is saved locally.[/dim]")
        console.print(f"[dim]  {artifact_path}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Upload failed: {e}[/yellow]")
        console.print(f"[dim]You can retry later: alloc upload {artifact_path}[/dim]")


def _read_step_count():
    # type: () -> Optional[int]
    """Read step count from sidecar file written by framework callbacks."""
    try:
        steps_path = os.path.join(os.getcwd(), ".alloc_steps.json")
        if os.path.isfile(steps_path):
            import json as json_mod
            with open(steps_path, "r") as f:
                data = json_mod.load(f)
            return data.get("step_count")
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

    if not feasible and verdict.get("recommendation"):
        rec = verdict["recommendation"]
        console.print(f"  [yellow]Suggestion: switch to {rec.upper()}[/yellow]")

    if verdict.get("reason"):
        console.print(f"  [dim]{verdict['reason']}[/dim]")

    # Cost estimate if present
    cost = result.get("est_cost_per_hour")
    if cost is not None:
        console.print(f"  [dim]Est. cost: ~${cost:.2f}/hr[/dim]")

    # Euler analysis if present
    euler = result.get("euler_analysis")
    if euler and euler.get("summary"):
        console.print()
        console.print(f"  [bold cyan]Euler Analysis[/bold cyan]")
        console.print(f"  {euler['summary']}")
        for rec in euler.get("recommendations", []):
            console.print(f"  [dim]• {rec}[/dim]")

    console.print()


def _extract_param_count(script: str) -> Optional[int]:
    """Try to extract param count from a Python script. Returns None if can't."""
    # For now, don't execute the script — just check common model names in filename
    import os
    basename = os.path.basename(script).lower()

    # Common model size patterns
    patterns = {
        "70b": int(70e9), "65b": int(65e9), "40b": int(40e9),
        "33b": int(33e9), "30b": int(30e9), "13b": int(13e9),
        "8b": int(8e9), "7b": int(7e9), "3b": int(3e9),
        "1.5b": int(1.5e9), "1b": int(1e9),
        "350m": int(350e6), "125m": int(125e6),
    }
    for pattern, count in patterns.items():
        if pattern in basename:
            return count

    return None


# Well-known model names → param count in billions
_MODEL_PARAMS = {
    "llama-3-70b": 70.0,
    "llama-3-8b": 8.03,
    "llama-2-70b": 70.0,
    "llama-2-13b": 13.0,
    "llama-2-7b": 7.0,
    "mistral-7b": 7.24,
    "mixtral-8x7b": 46.7,
    "gpt2": 0.124,
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.5,
    "bert-base": 0.110,
    "bert-large": 0.340,
    "t5-small": 0.060,
    "t5-base": 0.220,
    "t5-large": 0.770,
    "t5-xl": 3.0,
    "t5-xxl": 11.0,
    "falcon-7b": 7.0,
    "falcon-40b": 40.0,
    "phi-2": 2.78,
    "gemma-2b": 2.51,
    "gemma-7b": 8.54,
    "qwen-7b": 7.72,
    "qwen-14b": 14.2,
    "qwen-72b": 72.7,
    "deepseek-7b": 6.9,
    "deepseek-67b": 67.0,
    "vit-base": 0.086,
    "vit-large": 0.307,
    "whisper-small": 0.244,
    "whisper-medium": 0.769,
    "whisper-large": 1.55,
}


def _model_to_params(model: str) -> Optional[float]:
    """Look up model param count by name."""
    normalized = model.lower().strip()
    return _MODEL_PARAMS.get(normalized)
