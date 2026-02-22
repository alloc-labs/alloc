"""Tests for code_analyzer.py — AST-based ML pattern extraction."""

import os
import textwrap

from alloc.code_analyzer import analyze_script, CodeFindings


def _write_script(tmp_path, code):
    """Write a Python script to tmp_path and return its path."""
    p = tmp_path / "train.py"
    p.write_text(textwrap.dedent(code))
    return str(p)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_nonexistent_file(tmp_path):
    findings = analyze_script(str(tmp_path / "nope.py"))
    assert len(findings.parse_errors) == 1
    assert "not found" in findings.parse_errors[0].lower()


def test_syntax_error(tmp_path):
    p = _write_script(tmp_path, "def broken(:\n")
    findings = analyze_script(p)
    assert len(findings.parse_errors) == 1
    assert "syntax" in findings.parse_errors[0].lower()


def test_empty_script(tmp_path):
    p = _write_script(tmp_path, "# empty\n")
    findings = analyze_script(p)
    assert findings.dataloaders == []
    assert findings.optimizers == []
    assert findings.parse_errors == []


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

def test_imports_basic(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        from torch.utils.data import DataLoader
        from torch import nn as nn_mod
    """)
    findings = analyze_script(p)
    assert findings.imports["torch"] == "torch"
    assert findings.imports["DataLoader"] == "torch.utils.data.DataLoader"
    assert findings.imports["nn_mod"] == "torch.nn"


# ---------------------------------------------------------------------------
# DataLoader detection
# ---------------------------------------------------------------------------

def test_dataloader_simple(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, num_workers=2)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    dl = findings.dataloaders[0]
    assert dl.batch_size == 32
    assert dl.num_workers == 2
    assert dl.pin_memory is None
    assert dl.persistent_workers is None


def test_dataloader_all_kwargs(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    dl = findings.dataloaders[0]
    assert dl.batch_size == 64
    assert dl.num_workers == 8
    assert dl.pin_memory is True
    assert dl.persistent_workers is True
    assert dl.prefetch_factor == 4


def test_dataloader_no_kwargs(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    dl = findings.dataloaders[0]
    assert dl.num_workers is None
    assert dl.batch_size is None


def test_dataloader_via_module(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils import data
        loader = data.DataLoader(ds, num_workers=4)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    assert findings.dataloaders[0].num_workers == 4


def test_dataloader_via_torch(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        loader = torch.utils.data.DataLoader(ds, num_workers=0)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    assert findings.dataloaders[0].num_workers == 0


def test_dataloader_dynamic_workers(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, num_workers=config.workers)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    assert findings.dataloaders[0].num_workers is None
    assert "num_workers" in findings.dataloaders[0].raw_kwargs


def test_multiple_dataloaders(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, num_workers=2)
        val_loader = DataLoader(val_ds, num_workers=0)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 2


# ---------------------------------------------------------------------------
# Optimizer detection
# ---------------------------------------------------------------------------

def test_optimizer_adam(tmp_path):
    p = _write_script(tmp_path, """
        from torch.optim import Adam
        opt = Adam(model.parameters(), lr=3e-4, weight_decay=0.01)
    """)
    findings = analyze_script(p)
    assert len(findings.optimizers) == 1
    opt = findings.optimizers[0]
    assert opt.optimizer_type == "Adam"
    assert opt.learning_rate == 3e-4
    assert opt.weight_decay == 0.01


def test_optimizer_adamw_via_torch(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        opt = torch.optim.AdamW(params, lr=1e-5)
    """)
    findings = analyze_script(p)
    assert len(findings.optimizers) == 1
    assert findings.optimizers[0].optimizer_type == "AdamW"


def test_optimizer_sgd(tmp_path):
    p = _write_script(tmp_path, """
        from torch.optim import SGD
        opt = SGD(model.parameters(), lr=0.1)
    """)
    findings = analyze_script(p)
    assert len(findings.optimizers) == 1
    assert findings.optimizers[0].optimizer_type == "SGD"


# ---------------------------------------------------------------------------
# Precision detection
# ---------------------------------------------------------------------------

def test_autocast_new_api(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pass
    """)
    findings = analyze_script(p)
    assert len(findings.precision) == 1
    prec = findings.precision[0]
    assert prec.kind == "autocast"
    assert prec.is_deprecated_api is False


def test_autocast_deprecated(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        with torch.cuda.amp.autocast():
            pass
    """)
    findings = analyze_script(p)
    assert len(findings.precision) >= 1
    autocast = [p for p in findings.precision if p.kind == "autocast"]
    assert len(autocast) == 1
    assert autocast[0].is_deprecated_api is True


def test_half_call(tmp_path):
    p = _write_script(tmp_path, """
        model = model.half()
    """)
    findings = analyze_script(p)
    half = [p for p in findings.precision if p.kind == "half"]
    assert len(half) == 1
    assert half[0].dtype_str == "float16"


def test_grad_scaler(tmp_path):
    p = _write_script(tmp_path, """
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    """)
    findings = analyze_script(p)
    gs = [p for p in findings.precision if p.kind == "grad_scaler"]
    assert len(gs) == 1


# ---------------------------------------------------------------------------
# Distributed detection
# ---------------------------------------------------------------------------

def test_ddp_detection(tmp_path):
    p = _write_script(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    """)
    findings = analyze_script(p)
    dist = [d for d in findings.distributed if d.kind == "ddp"]
    assert len(dist) == 1


def test_init_process_group(tmp_path):
    p = _write_script(tmp_path, """
        import torch.distributed as dist
        dist.init_process_group("nccl")
    """)
    findings = analyze_script(p)
    ipg = [d for d in findings.distributed if d.kind == "init_process_group"]
    assert len(ipg) == 1
    assert ipg[0].backend == "nccl"


def test_init_process_group_backend_kwarg(tmp_path):
    p = _write_script(tmp_path, """
        import torch.distributed as dist
        dist.init_process_group(backend="gloo")
    """)
    findings = analyze_script(p)
    ipg = [d for d in findings.distributed if d.kind == "init_process_group"]
    assert len(ipg) == 1
    assert ipg[0].backend == "gloo"


def test_torch_compile(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        model = torch.compile(model)
    """)
    findings = analyze_script(p)
    tc = [d for d in findings.distributed if d.kind == "torch_compile"]
    assert len(tc) == 1


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def test_from_pretrained(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    """)
    findings = analyze_script(p)
    fp = [m for m in findings.models if m.kind == "from_pretrained"]
    assert len(fp) == 1
    assert fp[0].model_name == "meta-llama/Llama-2-7b-hf"


def test_gradient_checkpointing(tmp_path):
    p = _write_script(tmp_path, """
        model.gradient_checkpointing_enable()
    """)
    findings = analyze_script(p)
    gc = [m for m in findings.models if m.kind == "gradient_checkpointing"]
    assert len(gc) == 1


def test_cudnn_benchmark(tmp_path):
    p = _write_script(tmp_path, """
        import torch
        torch.backends.cudnn.benchmark = True
    """)
    findings = analyze_script(p)
    cb = [m for m in findings.models if m.kind == "cudnn_benchmark"]
    assert len(cb) == 1


# ---------------------------------------------------------------------------
# Training loop detection
# ---------------------------------------------------------------------------

def test_training_loop_backward_step(tmp_path):
    p = _write_script(tmp_path, """
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
    """)
    findings = analyze_script(p)
    assert findings.has_training_loop is True


def test_training_loop_for_train(tmp_path):
    p = _write_script(tmp_path, """
        model.train()
        for batch in loader:
            output = model(batch)
    """)
    findings = analyze_script(p)
    assert findings.has_training_loop is True


def test_no_training_loop(tmp_path):
    p = _write_script(tmp_path, """
        model = load_model()
        result = model.predict(data)
    """)
    findings = analyze_script(p)
    assert findings.has_training_loop is False


# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------

def test_source_location_line_number(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        x = 1
        loader = DataLoader(dataset, num_workers=2)
    """)
    findings = analyze_script(p)
    assert len(findings.dataloaders) == 1
    loc = findings.dataloaders[0].location
    assert loc.line_number > 0
    assert "DataLoader" in loc.source_line


# ---------------------------------------------------------------------------
# Edge case: raw_kwargs tracking
# ---------------------------------------------------------------------------

def test_raw_kwargs(tmp_path):
    p = _write_script(tmp_path, """
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, num_workers=2, pin_memory=True)
    """)
    findings = analyze_script(p)
    raw = findings.dataloaders[0].raw_kwargs
    assert "num_workers" in raw
    assert "pin_memory" in raw


# ---------------------------------------------------------------------------
# DeepSpeed detection
# ---------------------------------------------------------------------------

def test_data_parallel_detected(tmp_path):
    p = _write_script(tmp_path, """
        import torch.nn as nn
        model = nn.DataParallel(model)
    """)
    findings = analyze_script(p)
    dp = [d for d in findings.distributed if d.kind == "data_parallel"]
    assert len(dp) == 1


def test_data_parallel_not_confused_with_ddp(tmp_path):
    """DataParallel detection should not trigger on DistributedDataParallel."""
    p = _write_script(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model)
    """)
    findings = analyze_script(p)
    dp = [d for d in findings.distributed if d.kind == "data_parallel"]
    assert len(dp) == 0
    ddp = [d for d in findings.distributed if d.kind == "ddp"]
    assert len(ddp) == 1


def test_deepspeed_initialize(tmp_path):
    p = _write_script(tmp_path, """
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
    """)
    findings = analyze_script(p)
    ds = [d for d in findings.distributed if d.kind == "deepspeed"]
    assert len(ds) == 1


def test_deepspeed_via_training_args(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./out", deepspeed="ds_config.json")
    """)
    findings = analyze_script(p)
    ds = [d for d in findings.distributed if d.kind == "deepspeed"]
    assert len(ds) == 1


# ---------------------------------------------------------------------------
# HuggingFace Accelerate detection
# ---------------------------------------------------------------------------

def test_accelerate_accelerator(tmp_path):
    p = _write_script(tmp_path, """
        from accelerate import Accelerator
        accelerator = Accelerator()
    """)
    findings = analyze_script(p)
    acc = [d for d in findings.distributed if d.kind == "accelerate"]
    assert len(acc) == 1


# ---------------------------------------------------------------------------
# HuggingFace Trainer / TrainingArguments detection
# ---------------------------------------------------------------------------

def test_hf_trainer_detected(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import Trainer, TrainingArguments
        args = TrainingArguments(output_dir="./out")
        trainer = Trainer(model=model, args=args)
    """)
    findings = analyze_script(p)
    hf = [d for d in findings.distributed if d.kind == "hf_trainer"]
    assert len(hf) == 1
    assert findings.has_training_loop is True


def test_training_args_fp16(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./out", fp16=True)
    """)
    findings = analyze_script(p)
    prec = [pr for pr in findings.precision if pr.dtype_str == "float16"]
    assert len(prec) == 1


def test_training_args_bf16(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./out", bf16=True)
    """)
    findings = analyze_script(p)
    prec = [pr for pr in findings.precision if pr.dtype_str == "bfloat16"]
    assert len(prec) == 1


def test_training_args_grad_accum(tmp_path):
    p = _write_script(tmp_path, """
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./out", gradient_accumulation_steps=4)
    """)
    findings = analyze_script(p)
    assert findings.gradient_accumulation_steps == 4


def test_training_args_no_precision(tmp_path):
    """TrainingArguments without fp16/bf16 should not add precision findings."""
    p = _write_script(tmp_path, """
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir="./out", learning_rate=5e-5)
    """)
    findings = analyze_script(p)
    assert len(findings.precision) == 0
    assert findings.gradient_accumulation_steps is None


# ---------------------------------------------------------------------------
# Multi-file import following
# ---------------------------------------------------------------------------

def test_follow_local_import_model(tmp_path):
    """Follow local import to find model in separate file."""
    # model.py with from_pretrained
    model_file = tmp_path / "model.py"
    model_file.write_text(textwrap.dedent("""
        from transformers import AutoModelForCausalLM
        def get_model():
            return AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    """))

    # train.py imports from model.py
    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from model import get_model
        from torch.utils.data import DataLoader
        model = get_model()
        loader = DataLoader(dataset, num_workers=2)
    """))

    findings = analyze_script(str(train_file))
    # Should find the model from the imported file
    fp = [m for m in findings.models if m.kind == "from_pretrained"]
    assert len(fp) == 1
    assert fp[0].model_name == "meta-llama/Llama-2-7b-hf"
    # DataLoader from main script should still be found
    assert len(findings.dataloaders) == 1


def test_follow_local_import_optimizer(tmp_path):
    """Follow local import to find optimizer in separate file."""
    optim_file = tmp_path / "optim_setup.py"
    optim_file.write_text(textwrap.dedent("""
        from torch.optim import AdamW
        def make_optimizer(model):
            return AdamW(model.parameters(), lr=1e-4)
    """))

    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from optim_setup import make_optimizer
        import torch
        model = torch.nn.Linear(10, 10)
    """))

    findings = analyze_script(str(train_file))
    assert len(findings.optimizers) == 1
    assert findings.optimizers[0].optimizer_type == "AdamW"


def test_follow_import_skips_pip_packages(tmp_path):
    """Should not follow imports from pip packages (dotted modules)."""
    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    """))

    findings = analyze_script(str(train_file))
    # Should find the model from direct code, but NOT try to follow 'transformers'
    assert len(findings.models) == 1
    assert len(findings.parse_errors) == 0


def test_follow_import_missing_file(tmp_path):
    """Should silently skip imports where the file doesn't exist."""
    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from nonexistent_module import something
        import torch
    """))

    findings = analyze_script(str(train_file))
    assert len(findings.parse_errors) == 0  # Should not error


def test_follow_import_syntax_error(tmp_path):
    """Should silently skip imported files with syntax errors."""
    bad_file = tmp_path / "bad_module.py"
    bad_file.write_text("def broken(:\n")

    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from bad_module import something
        import torch
    """))

    findings = analyze_script(str(train_file))
    assert len(findings.parse_errors) == 0  # Main script is fine


def test_follow_import_distributed(tmp_path):
    """Follow local import to find distributed setup."""
    dist_file = tmp_path / "distributed.py"
    dist_file.write_text(textwrap.dedent("""
        import torch.distributed as dist
        def setup():
            dist.init_process_group("nccl")
    """))

    train_file = tmp_path / "train.py"
    train_file.write_text(textwrap.dedent("""
        from distributed import setup
        setup()
    """))

    findings = analyze_script(str(train_file))
    ipg = [d for d in findings.distributed if d.kind == "init_process_group"]
    assert len(ipg) == 1
    assert ipg[0].backend == "nccl"
