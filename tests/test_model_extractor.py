"""Tests for model extraction (subprocess + AST fallback)."""

from __future__ import annotations

import os
import tempfile

import pytest

from alloc.model_extractor import extract_model_info


# ---------------------------------------------------------------------------
# Helpers (must be defined before skipif decorators reference them)
# ---------------------------------------------------------------------------

def _has_torch():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _write_script(code: str) -> str:
    """Write code to a temp .py file, return path."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="alloc_test_")
    os.write(fd, code.encode("utf-8"))
    os.close(fd)
    return path


_needs_torch = pytest.mark.skipif(not _has_torch(), reason="torch not installed")


# ---------------------------------------------------------------------------
# Tests: no-torch-required
# ---------------------------------------------------------------------------

def test_manual_override():
    """--param-count-b skips execution, file doesn't need to exist."""
    info = extract_model_info("nonexistent.py", param_count_b=7.0)
    assert info is not None
    assert info.param_count == int(7e9)
    assert info.method == "manual"
    assert info.dtype == "float16"


def test_nonexistent_script():
    """Missing file without manual override returns None."""
    info = extract_model_info("this_file_does_not_exist.py")
    assert info is None


def test_no_model_found():
    """Script with no nn.Module returns None."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("x = 42\n")
        f.flush()
        path = f.name
    try:
        info = extract_model_info(path, timeout=10)
        assert info is None
    finally:
        os.unlink(path)


def test_syntax_error():
    """Script with syntax error returns None."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("def broken(\n")
        f.flush()
        path = f.name
    try:
        info = extract_model_info(path, timeout=10)
        assert info is None
    finally:
        os.unlink(path)


def test_timeout():
    """Hanging script is killed, returns None."""
    script = _write_script("""
import time
time.sleep(300)
""")
    try:
        info = extract_model_info(script, timeout=2)
        assert info is None
    finally:
        os.unlink(script)


# ---------------------------------------------------------------------------
# Tests: require torch
# ---------------------------------------------------------------------------

@_needs_torch
def test_simple_nn_module():
    """Module-level model = MyModel() found via execution."""
    script = _write_script("""
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

model = MyModel()
""")
    try:
        info = extract_model_info(script, timeout=15)
        assert info is not None
        assert info.method == "execution"
        assert info.param_count > 0
        assert info.model_name == "model"
    finally:
        os.unlink(script)


@_needs_torch
def test_class_only_no_instance():
    """Class defined, no module-level instance — instantiation fallback finds it."""
    script = _write_script("""
import torch.nn as nn

class OnlyAClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 50)
    def forward(self, x):
        return self.layer(x)
""")
    try:
        info = extract_model_info(script, timeout=15)
        assert info is not None
        assert info.method == "execution"
        assert info.param_count > 0
        assert info.model_name == "OnlyAClass"
    finally:
        os.unlink(script)


@_needs_torch
def test_if_name_main_guard():
    """Model inside __main__ guard — class instantiation fallback finds it."""
    script = _write_script("""
import torch.nn as nn

class GuardedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = GuardedModel()
    print("training...")
""")
    try:
        info = extract_model_info(script, timeout=15)
        assert info is not None
        assert info.method == "execution"
        assert info.param_count > 0
    finally:
        os.unlink(script)


@_needs_torch
def test_sys_exit_in_script():
    """sys.exit() in script doesn't crash extractor."""
    script = _write_script("""
import sys
import torch.nn as nn

class ExitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)
    def forward(self, x):
        return self.fc(x)

model = ExitModel()
sys.exit(0)
""")
    try:
        info = extract_model_info(script, timeout=15)
        assert info is not None
        assert info.method == "execution"
        assert info.param_count > 0
    finally:
        os.unlink(script)


@_needs_torch
def test_multiple_models_picks_largest():
    """Two models → picks larger one."""
    script = _write_script("""
import torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 100)
    def forward(self, x):
        return self.fc2(self.fc1(x))

small = SmallModel()
big = BigModel()
""")
    try:
        info = extract_model_info(script, timeout=15)
        assert info is not None
        assert info.method == "execution"
        # BigModel: 1000*500 + 500 + 500*100 + 100 = 550,600 params
        # SmallModel: 10*5 + 5 = 55 params
        assert info.param_count > 500000
        assert info.model_name == "big"
    finally:
        os.unlink(script)
