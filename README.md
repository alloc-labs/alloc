# alloc

[![PyPI version](https://img.shields.io/pypi/v/alloc.svg)](https://pypi.org/project/alloc/)
[![Python](https://img.shields.io/pypi/pyversions/alloc.svg)](https://pypi.org/project/alloc/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/alloc-labs/alloc/actions/workflows/ci.yml/badge.svg)](https://github.com/alloc-labs/alloc/actions/workflows/ci.yml)

GPU intelligence for ML training. Estimate VRAM, detect bottlenecks, and right-size hardware before you launch.

```bash
pip install alloc
```

## What it does

Alloc wraps your training command, monitors GPU usage for ~30 seconds, and tells you if you're wasting money.

```bash
$ alloc run python train.py

  Peak VRAM    18.2 GB / 80.0 GB (22.8%)
  GPU Util     34% avg
  Bottleneck   underutilized
  Confidence   0.58

  Recommendation: Downsize from A100-80GB to A10G-24GB
  Potential savings: ~62% ($1.12/hr)

  Artifact saved to ./alloc_artifact.json.gz
```

No code changes. No config files. Works with any framework.

## Commands

### `alloc run` -- Monitor training

```bash
alloc run python train.py                # calibrate and exit (default)
alloc run --full python train.py         # monitor full training run
alloc run torchrun --nproc_per_node=4 train.py  # multi-GPU
alloc run -- python train.py --epochs 10        # use -- to separate flags
```

Auto-stops when GPU metrics stabilize (~30-60s). Discovers all GPUs used by the process tree. Captures hardware context (driver, CUDA, SM version). Writes `alloc_artifact.json.gz` with everything.

Use `--json` for machine-readable output. Use `--verbose` for detailed breakdown and reasoning.

### `alloc ghost` -- Static VRAM estimation

```bash
alloc ghost train_7b.py
alloc ghost train_7b.py --dtype bf16 --batch-size 32
alloc ghost train_7b.py --json
alloc ghost train_7b.py --verbose    # show formula breakdown
```

Estimates VRAM from model parameters without running anything. No GPU needed.

### `alloc scan` -- Remote analysis

```bash
alloc scan --model llama-3-70b --gpu A100-80GB
alloc scan --model mistral-7b --gpu A10G --strategy fsdp --num-gpus 4
alloc scan --param-count-b 13.0 --gpu H100-80GB --dtype bf16 --json
```

Runs VRAM + cost analysis on the Alloc API. No GPU required.

### `alloc catalog` -- GPU hardware reference

```bash
alloc catalog list                  # all 13 GPUs sorted by VRAM
alloc catalog list --sort cost      # sort by $/hr
alloc catalog show H100             # detailed specs
```

Offline GPU specs, interconnect details, and cloud pricing.

### Other commands

```bash
alloc login                         # authenticate with dashboard
alloc upload alloc_artifact.json.gz # upload artifact to dashboard
alloc version                       # print version
```

## Python API

```python
import alloc

# Static VRAM estimation (no GPU needed)
report = alloc.ghost(model)
print(report.total_gb)

# Or from param count (no torch needed)
report = alloc.ghost(param_count_b=7.0, dtype="bf16")
```

## How it works

Alloc launches your training command as a subprocess and monitors GPU memory, utilization, and power draw externally via [NVML](https://developer.nvidia.com/management-library-nvml). When metrics stabilize, it stops, classifies the bottleneck, and writes a compressed artifact.

No monkey-patching. No framework hooks (unless you opt in). Your training code runs unmodified.

| Module | Purpose |
|--------|---------|
| `ghost.py` | Static VRAM estimation via parameter walking or pure math |
| `probe.py` | External GPU monitoring via pynvml with multi-GPU discovery |
| `stability.py` | Multi-signal stability detection for auto-stop |
| `artifact_writer.py` | Writes compressed JSON artifact with probe + hardware data |
| `callbacks.py` | Optional framework callbacks (HuggingFace TrainerCallback) |
| `catalog/` | Bundled GPU hardware catalog (13 GPUs) with specs and pricing |
| `context.py` | Auto-discovers git, container, and Ray metadata |
| `display.py` | Rich terminal output formatting |
| `cli.py` | Typer CLI entry point |
| `config.py` | Environment variable configuration |

## Configuration

All config is via environment variables. Zero config files required for local use.

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOC_API_URL` | `https://alloc-production-ffc2.up.railway.app` | API endpoint |
| `ALLOC_TOKEN` | -- | Auth token (from `alloc login`) |
| `ALLOC_UPLOAD` | `false` | Auto-upload artifacts |

## Alloc Cloud

The CLI works standalone for local profiling. [Alloc Cloud](https://www.alloclabs.com) adds:

- Dashboard with run history and trend analysis
- Bottleneck classification and right-sizing recommendations
- Cost projections across 80+ GPU configurations
- Auto-proposals when savings exceed 25%
- Team management and budget controls (coming soon)

## Requirements

- Python 3.8+
- `pynvml` for GPU monitoring (`pip install alloc[gpu]`)
- Works on Linux and macOS (GPU features require NVIDIA drivers)

## License

[Apache 2.0](LICENSE)
