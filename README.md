<div align="center">

# alloc

**GPU intelligence for ML training.**
**Estimate VRAM, detect bottlenecks, right-size hardware — before you launch.**

[![PyPI version](https://img.shields.io/pypi/v/alloc.svg?style=flat-square&color=00e5a0)](https://pypi.org/project/alloc/)
[![Python](https://img.shields.io/pypi/pyversions/alloc.svg?style=flat-square)](https://pypi.org/project/alloc/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![CI](https://github.com/alloc-labs/alloc/actions/workflows/ci.yml/badge.svg)](https://github.com/alloc-labs/alloc/actions/workflows/ci.yml)

[Documentation](https://www.alloclabs.com/docs) · [Blog](https://www.alloclabs.com/blog) · [Try the Demo](https://www.alloclabs.com/demo) · [Alloc Cloud](https://www.alloclabs.com)

</div>

---

```bash
pip install alloc
```

## What it does

Alloc wraps your training command, monitors GPU usage for ~30 seconds, and tells you if you're wasting money.

```
$ alloc run python train.py

 ┌──────────────────────────────────────────────────┐
 │                  ALLOC VERDICT                    │
 ├──────────────────────────────────────────────────┤
 │                                                   │
 │  Peak VRAM    18.2 GB / 80.0 GB  (22.8%)        │
 │  GPU Util     34% avg                             │
 │  Bottleneck   ■ underutilized                     │
 │  Confidence   0.58                                │
 │                                                   │
 │  ▸ Downsize from A100-80GB to A10G-24GB          │
 │    Potential savings: ~62% ($1.12/hr)             │
 │                                                   │
 │  Artifact → ./alloc_artifact.json.gz             │
 └──────────────────────────────────────────────────┘
```

No code changes. No config files. Works with any framework.

> **Why Alloc?** Most ML teams over-provision GPUs because they don't know what they actually need.
> Alloc tells you in 30 seconds. Read more: [We Wasted $12k on GPUs](https://www.alloclabs.com/blog/gpu-cost-waste)

---

## Quick Start

```bash
# Install
pip install alloc

# Profile a training run (~30s, auto-stops)
alloc run python train.py

# Static VRAM estimation (no GPU needed)
alloc ghost train_7b.py

# Remote analysis (no GPU needed)
alloc scan --model llama-3-70b --gpu A100-80GB

# Upload results to dashboard
alloc login
alloc upload alloc_artifact.json.gz
```

> Full walkthrough: [Get Started in 60 Seconds](https://www.alloclabs.com/docs/quickstart)

---

## Commands

### `alloc run` — Profile training

```bash
alloc run python train.py                        # calibrate and exit (default)
alloc run --full python train.py                 # monitor full training run
alloc run torchrun --nproc_per_node=4 train.py   # multi-GPU (auto-discovered)
alloc run -- python train.py --epochs 10         # use -- to separate flags
```

Auto-stops when GPU metrics stabilize (~30-60s). Discovers all GPUs used by the process tree. Captures hardware context (driver, CUDA, SM version). Writes `alloc_artifact.json.gz`.

| Flag | Description |
|------|-------------|
| `--full` | Monitor the entire training run instead of auto-stopping |
| `--timeout N` | Max calibration time in seconds (default: 120) |
| `--json` | Machine-readable JSON output |
| `--verbose` | Detailed breakdown with probe samples and reasoning |
| `--upload` | Best-effort upload to [Alloc Cloud](https://www.alloclabs.com) on completion |
| `--out PATH` | Override artifact output path |

### `alloc ghost` — Static VRAM estimation

```bash
alloc ghost train_7b.py                          # estimate from script
alloc ghost train_7b.py --dtype bf16 --batch-size 32
alloc ghost train_7b.py --verbose                # show VRAM formula breakdown
```

Estimates VRAM from model parameters without running anything. No GPU needed.

> How it works: [Ghost Scan — Estimate VRAM Before You Launch](https://www.alloclabs.com/docs/ghost-scan)

### `alloc scan` — Remote analysis

```bash
alloc scan --model llama-3-70b --gpu A100-80GB
alloc scan --model mistral-7b --gpu A10G --strategy fsdp --num-gpus 4
alloc scan --param-count-b 13.0 --gpu H100-80GB --dtype bf16 --json
```

Runs VRAM + cost analysis on the Alloc API. No GPU required, no auth required.

> Not sure which GPU to pick? Read: [A100 vs H100 vs L40S](https://www.alloclabs.com/blog/a100-vs-h100-vs-l40s)

### `alloc catalog` — GPU reference

```bash
alloc catalog list                    # all 13 GPUs sorted by VRAM
alloc catalog list --sort cost        # sort by $/hr
alloc catalog list --sort tflops      # sort by BF16 TFLOPS
alloc catalog show H100               # detailed specs for H100
```

Offline GPU specs, interconnect details, and cloud pricing across AWS, GCP, Azure, Lambda, and CoreWeave.

<details>
<summary><b>Example: <code>alloc catalog list</code></b></summary>

```
 GPU               VRAM    BF16 TFLOPS    $/hr (on-demand)
 ─────────────────────────────────────────────────────────
 T4-16GB           16 GB        65          $0.53
 A10G-24GB         24 GB       125          $1.10
 L4-24GB           24 GB       121          $0.81
 RTX-4090-24GB     24 GB       165          $0.74
 V100-32GB         32 GB        28          $0.74
 A100-40GB         40 GB       312          $1.84
 L40S-48GB         48 GB       362          $1.84
 A100-80GB         80 GB       312          $2.21
 H100-80GB         80 GB       990          $3.09
 H200-141GB       141 GB       990          $3.75
 ...
```

</details>

### Other commands

```bash
alloc login                            # authenticate with Alloc Cloud
alloc upload alloc_artifact.json.gz    # upload artifact to dashboard
alloc version                          # print version
```

---

## Python API

```python
import alloc

# Static VRAM estimation (no GPU needed)
report = alloc.ghost(model)
print(f"{report.total_gb:.1f} GB")   # e.g., 115.4 GB

# Or from param count (no torch needed)
report = alloc.ghost(param_count_b=7.0, dtype="bf16")
print(report.weights_gb)             # weight memory
print(report.optimizer_gb)           # optimizer state memory
print(report.total_gb)               # total estimated VRAM

# HuggingFace integration
from alloc import HuggingFaceCallback
trainer = Trainer(..., callbacks=[HuggingFaceCallback()])
```

---

## How it works

```
 alloc run python train.py
      │
      ├─→ subprocess.Popen(["python", "train.py"])
      │        (your code runs unmodified)
      │
      └─→ probe thread (pynvml)
           ├── polls every 500ms: VRAM, GPU util, power
           ├── discovers all GPUs via process tree
           ├── captures hardware context (driver, CUDA, SM)
           └── auto-stops when metrics stabilize
                    │
                    ▼
           alloc_artifact.json.gz
```

No monkey-patching. No framework hooks (unless you [opt in](src/alloc/callbacks.py)). Your training code runs unmodified.

> Learn more: [GPU Right-Sizing — Stop Over-Provisioning](https://www.alloclabs.com/docs/right-sizing)

<details>
<summary><b>Architecture</b></summary>

| Module | Purpose |
|--------|---------|
| [`ghost.py`](src/alloc/ghost.py) | Static VRAM estimation via parameter walking or pure math |
| [`probe.py`](src/alloc/probe.py) | External GPU monitoring via pynvml with multi-GPU discovery |
| [`stability.py`](src/alloc/stability.py) | Multi-signal stability detection for auto-stop |
| [`artifact_writer.py`](src/alloc/artifact_writer.py) | Writes compressed JSON artifact with probe + hardware data |
| [`callbacks.py`](src/alloc/callbacks.py) | Framework callbacks (HuggingFace TrainerCallback) |
| [`catalog/`](src/alloc/catalog/) | Bundled GPU hardware catalog (13 GPUs) with specs and pricing |
| [`context.py`](src/alloc/context.py) | Auto-discovers git, container, and Ray metadata |
| [`display.py`](src/alloc/display.py) | Rich terminal output formatting |
| [`cli.py`](src/alloc/cli.py) | Typer CLI entry point |
| [`config.py`](src/alloc/config.py) | Environment variable configuration |

</details>

---

## Configuration

All config is via environment variables. Zero config files required for local use.

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOC_API_URL` | `https://alloc-production-ffc2.up.railway.app` | API endpoint |
| `ALLOC_TOKEN` | — | Auth token (from `alloc login`) |
| `ALLOC_UPLOAD` | `false` | Auto-upload artifacts to [Alloc Cloud](https://www.alloclabs.com) |

---

## Alloc Cloud

The CLI works standalone for local profiling. [**Alloc Cloud**](https://www.alloclabs.com) adds a dashboard and intelligence layer:

| Feature | CLI (free) | Cloud |
|---------|:---:|:---:|
| VRAM estimation | **Yes** | **Yes** |
| GPU profiling + bottleneck detection | **Yes** | **Yes** |
| Artifact generation | **Yes** | **Yes** |
| GPU hardware catalog | **Yes** | **Yes** |
| Run history and trends | — | **Yes** |
| Cost projections across 80+ configs | — | **Yes** |
| Auto-proposals (>25% savings) | — | **Yes** |
| Team management and budgets | — | Coming soon |

[**Sign up free**](https://www.alloclabs.com/signup) · [Read the docs](https://www.alloclabs.com/docs)

---

## Requirements

- Python 3.8+
- `pynvml` for GPU monitoring: `pip install alloc[gpu]`
- Works on Linux and macOS (GPU features require NVIDIA drivers)

## Contributing

Bug reports and feature requests welcome via [GitHub Issues](https://github.com/alloc-labs/alloc/issues). See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE) — Alloc Labs, 2026
