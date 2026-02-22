# alloc

**Find and fix training bottlenecks. Zero code changes.**

[![PyPI](https://img.shields.io/pypi/v/alloc)](https://pypi.org/project/alloc/)
[![Python](https://img.shields.io/pypi/pyversions/alloc)](https://pypi.org/project/alloc/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

```bash
pip install alloc
alloc run python train.py
```

```
alloc v0.0.2 — Calibrate

 Run Summary
  Peak VRAM       31.2 GB / 40.0 GB (A100)
  VRAM used       78.0%
  Avg GPU util    72.3%
  Avg power       287 W
  Duration        24.1s (auto-stopped: metrics stable at 18.2s)
  Step time       148.5 ms (p50) / 152.1 ms (p90)
  Throughput      42.3 samples/sec

  Artifact: alloc_artifact.json.gz
```

That's it. No decorators, no config files, no code changes. Alloc wraps your command, profiles GPU usage, and tells you what's wrong.

---

## What you get

**`alloc diagnose`** reads your training script and tells you exactly what to change:

```bash
alloc diagnose train.py
```
```
alloc diagnose — 3 findings in train.py

 CRITICAL  DL005 — DataLoader running in main thread
   train.py:47   num_workers=0 → num_workers=8
   num_workers=0 loads data in the main thread, blocking GPU computation entirely.
   Expected impact: ~30-50% faster training with parallel data loading

 WARNING   PREC002 — Using fp16, consider bf16
   train.py:56   dtype: float16 → dtype: bfloat16
   H100 supports bf16 natively — eliminates loss scaling overhead.
   Expected impact: ~5-10% speedup, eliminates GradScaler complexity

 INFO      THRU001 — cudnn.benchmark not enabled
   Add: torch.backends.cudnn.benchmark = True
   Expected impact: ~5-10% speedup for fixed-size inputs

Summary: 1 critical, 1 warning, 1 info
Run with --diff to generate patches | --json for CI output
```

**`alloc ghost`** estimates VRAM before you launch:

```bash
alloc ghost train.py --dtype bf16
```
```
 Ghost Scan — 7.0B params (bf16)

  Model weights       13.04 GB
  Gradients           13.04 GB
  Optimizer (Adam)    78.23 GB
  Activations (est.)   0.50 GB
  Buffer (10%)        10.48 GB

  Total VRAM         115.28 GB
```

**`alloc scan`** ranks GPU configs without a GPU:

```bash
alloc scan --model llama-3-70b --gpu H100-80GB --num-gpus 8
```

---

## Works with everything

Alloc wraps your launch command. No framework-specific setup required.

```bash
alloc run python train.py
alloc run torchrun --nproc_per_node=4 train.py
alloc run accelerate launch train.py
alloc run srun python train.py           # Slurm
alloc run ray job submit -- python train.py
```

Multi-GPU detection is automatic (discovers all GPUs in the process tree).

---

## Deeper signals (optional)

Add a one-line callback for step-level timing:

```python
# HuggingFace
from alloc import HuggingFaceCallback
trainer = Trainer(..., callbacks=[HuggingFaceCallback()])

# Lightning
from alloc import LightningCallback
trainer = Trainer(..., callbacks=[LightningCallback()])
```

This unlocks step time p50/p90, throughput, and dataloader bottleneck detection.

---

## All commands

| Command | What it does |
|---------|-------------|
| `alloc run <cmd>` | Profile a training run (auto-stops when stable) |
| `alloc diagnose <script>` | AST analysis with specific fix suggestions |
| `alloc ghost <script>` | Estimate VRAM before launching |
| `alloc scan --model <name>` | Rank GPU configs remotely (no GPU needed) |
| `alloc catalog list` | Browse 13 GPUs with specs and pricing |
| `alloc init` | Configure GPU fleet and budget (`.alloc.yaml`) |
| `alloc login` | Authenticate for dashboard + auto-upload |

Every command supports `--json` for CI/CD integration.

---

## Dashboard

Log in to get team visibility, budget tracking, and optimization proposals:

```bash
alloc login --browser
alloc run python train.py    # auto-uploads when logged in
```

Dashboard at [alloclabs.com](https://www.alloclabs.com)

---

## Design principles

1. **Zero config** — `alloc run python train.py` works out of the box
2. **Never crash training** — all Alloc failures are caught silently
3. **No monkey-patching** — external monitoring only, deeper signals opt-in
4. **Local-first** — works in air-gapped environments, no internet required

---

## Links

- [Website](https://www.alloclabs.com)
- [Documentation](https://www.alloclabs.com/docs)
- [PyPI](https://pypi.org/project/alloc/)
