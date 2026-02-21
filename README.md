# alloc (by [Alloc Labs](https://www.alloclabs.com))

Engineer-first training pipeline diagnosis: profile short runs, detect bottlenecks across data loading and comms, and pick the right GPU config and parallelism strategy.

[![Website](https://img.shields.io/badge/alloclabs.com-website-22c55e)](https://www.alloclabs.com)
[![PyPI](https://img.shields.io/pypi/v/alloc)](https://pypi.org/project/alloc/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> Built by [Alloc Labs](https://www.alloclabs.com): holistic training pipeline visibility — bottleneck diagnosis, strategy selection, and GPU right-sizing — so you stop wasting compute.

## What Alloc Does

Most ML teams waste spend because resource decisions are guesswork and feedback arrives too late. Alloc gives you a progressive workflow:

- **Pre-flight**: estimate VRAM fit and rank feasible configs by objective (`alloc scan`, `alloc ghost`)
- **Calibration run**: measure peak VRAM + utilization (and optionally step timing) from a short run (`alloc run`)
- **Run history**: auto-uploads to dashboard when logged in for team visibility and budget-aware proposals

Alloc is launcher-first. It works with `python`, `torchrun`, `accelerate`, and cluster entrypoints (Slurm, Ray, Kubernetes) because it does not require framework-specific wrappers for baseline value.

## Who This Is For

- **Solo engineers** who want a fast sanity check before burning GPU time
- **ML teams** who need repeatable right-sizing and bottleneck visibility
- **Platform/infra leads** who want budget-aware controls without rewriting training code

## Why It Is Low Friction

- **No code changes required** for baseline value (`alloc run`)
- **Optional deeper integration** via callbacks when you want richer timing signals
- **Local-first artifacts** so users still get value without cloud connectivity
- **Progressive adoption** from local CLI to team workflows and governance

## Install

```bash
pip install alloc
```

GPU monitoring (`pynvml`) is included. On machines without NVIDIA drivers, `alloc run` still executes your command — it just skips GPU metrics.

`alloc` does not depend on torch. If you want `alloc ghost train.py` to infer param counts from a script, torch must be installed in that environment, otherwise use `--param-count-b`.

## Commands

### `alloc scan`: Remote Ghost Scan (no GPU needed)

```bash
alloc scan --model llama-3-70b --gpu A100-80GB
alloc scan --model mistral-7b --gpu A10G --strategy fsdp --num-gpus 4
alloc scan --param-count-b 13.0 --gpu H100-80GB --dtype bf16

# Objective + budget constraints
alloc scan --model llama-3-70b --gpu H100-80GB --objective fastest_within_budget --max-budget-hourly 12

# Topology hints (optional, improves planner quality)
alloc scan --param-count-b 70 --gpu H100-80GB --num-gpus 64 --num-nodes 8 --gpus-per-node 8 --interconnect infiniband
```

### `alloc ghost`: Local VRAM estimation

```bash
alloc ghost train.py --dtype bf16 --batch-size 32
alloc ghost train.py --param-count-b 7.0   # manual override
```

Analyzes your training script to discover model parameters and computes a VRAM breakdown. Uses a three-method fallback: (1) `--param-count-b` manual override, (2) subprocess execution to find `nn.Module` classes and count parameters, (3) AST parsing for `from_pretrained()` calls.

### `alloc run`: Training with GPU monitoring

```bash
alloc run python train.py                # calibrate and exit (default)
alloc run --full python train.py         # monitor full training run
alloc run torchrun --nproc_per_node=4 train.py
alloc run -- python train.py --epochs 10
```

Wraps your command, monitors GPU memory/utilization/power via `pynvml`, and writes an artifact.

**Default: calibrate-and-exit.** Auto-stops when GPU metrics stabilize, prints a verdict with bottleneck classification and a top recommendation, then exits. Use `--timeout N` to adjust max calibration time (default 120s). Use `--full` to monitor the entire run.

**Multi-GPU:** Automatically discovers all GPUs used by the process tree (works with `torchrun`, `accelerate launch`, etc.).

**Hardware context:** Captures driver version, CUDA version, and SM compute capability from NVML.

### `alloc login`: Authenticate with dashboard

```bash
alloc login --browser            # opens browser for Google/Microsoft sign-in (recommended)
alloc login                      # email + password (Supabase password accounts)
alloc login --token <TOKEN>      # paste an access token directly
```

Once logged in, `alloc run` auto-uploads artifacts to the dashboard. Use `--no-upload` to skip for a specific run.

### `alloc whoami`: Show current auth + org context

```bash
alloc whoami
alloc whoami --json
```

Prints the current identity (when logged in), plus objective, effective budget cap, and fleet counts.

### `alloc logout`: Clear local session

```bash
alloc logout
```

Clears saved `token`/`refresh_token` from `~/.alloc/config.json`.

### `alloc upload`: Upload artifact to dashboard

```bash
alloc upload alloc_artifact.json.gz
```

Uploads a previously saved `.json.gz` artifact to the dashboard via `POST /runs/ingest`. Requires authentication (`alloc login` first).

If your session token has expired and a `refresh_token` is available (password login flow), `alloc upload` refreshes once and retries automatically.

### `alloc catalog`: Browse GPU hardware catalog

```bash
alloc catalog list                           # list all 13 GPUs (sorted by VRAM)
alloc catalog list --sort cost               # sort by $/hr
alloc catalog list --sort tflops             # sort by BF16 TFLOPS
alloc catalog show H100                      # detailed specs for H100
alloc catalog show nvidia-a100-sxm-80gb      # lookup by stable ID
```

Offline reference for GPU specs, interconnect details, and cloud pricing. Supports aliases (H100, A100, T4) and stable IDs.

### `alloc init`: Configure GPU fleet and budget

```bash
alloc init                     # interactive wizard
alloc init --yes               # non-interactive defaults (full catalog, 50/50 priority)
alloc init --from-org --yes    # pull fleet/budget/objective from your org (requires alloc login)
```

Creates a `.alloc.yaml` file in the current directory with your GPU fleet, explore list, budget, and priority weights. When present, `ghost`, `run`, and `scan` automatically use fleet context for recommendations. Use `--no-config` on any command to skip it.

### `alloc version`

```bash
alloc version
```

## Python API

```python
import alloc

# Static VRAM analysis (never crashes your training)
report = alloc.ghost(model)
print(report.total_gb)  # e.g., 115.42

# Or from param count (no torch needed)
report = alloc.ghost(param_count_b=7.0, dtype="bf16")
```

## Framework Callbacks

Optional callbacks for deeper profiling. Captures step-level timing, throughput, and dataloader wait estimates.

```python
# HuggingFace Transformers
from alloc import HuggingFaceCallback
trainer = Trainer(..., callbacks=[HuggingFaceCallback()])

# PyTorch Lightning
from alloc import LightningCallback
trainer = Trainer(..., callbacks=[LightningCallback()])
```

Callbacks write a `.alloc_callback.json` sidecar with step time (p50/p90), samples/sec, and estimated dataloader wait %. This unlocks higher confidence analysis and dataloader bottleneck detection.

## Configuration

Alloc works with zero config. You can optionally configure it with environment variables and/or a `.alloc.yaml` in your repo.

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOC_API_URL` | `https://alloc-production-ffc2.up.railway.app` | API endpoint for remote scans |
| `ALLOC_TOKEN` | (empty) | Auth token for API calls |
| `ALLOC_UPLOAD` | `true` when logged in | Upload results to dashboard (use `--no-upload` to skip) |
| `ALLOC_OUT` | `alloc_artifact.json.gz` | Artifact output path |
| `ALLOC_GPU_COUNT_CANDIDATES` | (empty) | Override GPU-count candidates for ranking (comma-separated ints) |

## Architecture

| Module | Purpose |
|--------|---------|
| `ghost.py` | VRAM estimation from parameter count. Computes weights + gradients + optimizer + activations + buffer breakdown. |
| `model_extractor.py` | Three-method model discovery: subprocess execution (`nn.Module` finder), AST parsing (`from_pretrained`), manual override. |
| `probe.py` | External GPU monitoring via `pynvml`. Process-tree aware multi-GPU discovery. Captures hardware context (driver, CUDA, SM version). |
| `stability.py` | Multi-signal stability detection for calibrate-and-exit (VRAM plateau + util std dev + power std dev). |
| `catalog/` | Bundled GPU hardware catalog (13 GPUs) with specs and pricing. Powers `alloc catalog` commands. |
| `context.py` | Context autodiscovery: git (SHA, branch, repo), container (Docker/Podman), Ray (job ID, cluster). |
| `artifact_writer.py` | Artifact Writer: writes `alloc_artifact.json.gz` with probe, ghost, hardware, and context sections. |
| `cli.py` | Typer CLI with `ghost`, `run`, `scan`, `login`, `upload`, `init`, `catalog`, `version` commands. |
| `yaml_config.py` | `.alloc.yaml` parser: fleet, explore, priority, budget. Loaded automatically by `ghost`, `run`, `scan`. |
| `callbacks.py` | Framework callbacks: HuggingFace `TrainerCallback` and Lightning `Callback` with step timing (p50/p90), throughput, and dataloader wait estimation. |
| `upload.py` | Artifact uploader: POSTs `.json.gz` to `POST /runs/ingest`. |
| `display.py` | Rich terminal formatting for reports. |
| `config.py` | Env-var-only configuration (API URL, Supabase URL, token storage). |

## Design Principles

1. **Zero config**: `alloc run python train.py` works out of the box
2. **No monkey-patching**: External monitoring only; deeper signals are opt-in
3. **Never crash user's training**: All Alloc failures are caught and training continues
4. **Progressive disclosure**: Individual use first, team governance later

## Telemetry Levels

Alloc intentionally starts non-invasive and adds richer signals only when you opt in.

- **NVML (today)**: peak VRAM, GPU utilization, power draw, basic hardware context (driver/CUDA/SM), multi-GPU discovery from the process tree.
- **Framework timing (today, opt-in)**: step time p50/p90, samples/sec, estimated dataloader wait percentage via HF/Lightning callbacks.
- **Distributed timing (planned, opt-in)**: per-rank timing skew, communication overhead, stronger interconnect-aware recommendations.
