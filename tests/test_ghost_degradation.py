"""Tests for ghost structured degradation on distributed scripts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from alloc.cli import app
from alloc.model_extractor import ModelInfo, extract_model_info

runner = CliRunner()


def test_distributed_error_returns_structured_modelinfo():
    """When extractor subprocess fails with a distributed keyword, return structured ModelInfo."""
    sidecar_data = json.dumps({
        "status": "error",
        "error": "RuntimeError: torch.distributed.init_process_group requires MASTER_ADDR",
    })

    def _fake_subprocess_run(*args, **kwargs):
        # Write the sidecar file
        sidecar_path = args[0][3]  # [python, -m, alloc.extractor_runner, sidecar_path, script_path]
        with open(sidecar_path, "w") as f:
            f.write(sidecar_data)

    # Create a dummy script
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="alloc_test_dist_")
    os.write(fd, b"import torch\ntorch.distributed.init_process_group('nccl')\n")
    os.close(fd)

    try:
        with patch("subprocess.run", side_effect=_fake_subprocess_run):
            info = extract_model_info(script_path)

        assert info is not None
        assert info.extraction_error == "distributed_entrypoint"
        assert "distributed runtime" in info.extraction_detail
        assert info.param_count == 0
    finally:
        os.unlink(script_path)


def test_distributed_error_nccl_keyword():
    """NCCL errors should be caught as distributed failures."""
    sidecar_data = json.dumps({
        "status": "error",
        "error": "NCCL error: unhandled system error",
    })

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="alloc_test_nccl_")
    os.write(fd, b"pass\n")
    os.close(fd)

    try:
        def _fake_run(*args, **kwargs):
            sidecar_path = args[0][3]
            with open(sidecar_path, "w") as f:
                f.write(sidecar_data)

        with patch("subprocess.run", side_effect=_fake_run):
            info = extract_model_info(script_path)

        assert info is not None
        assert info.extraction_error == "distributed_entrypoint"
    finally:
        os.unlink(script_path)


def test_non_distributed_error_returns_none():
    """Non-distributed errors should still return None (fall through to AST)."""
    sidecar_data = json.dumps({
        "status": "error",
        "error": "ImportError: No module named 'custom_lib'",
    })

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="alloc_test_other_")
    # Script with no from_pretrained so AST also returns None
    os.write(fd, b"import custom_lib\n")
    os.close(fd)

    try:
        def _fake_run(*args, **kwargs):
            sidecar_path = args[0][3]
            with open(sidecar_path, "w") as f:
                f.write(sidecar_data)

        with patch("subprocess.run", side_effect=_fake_run):
            info = extract_model_info(script_path)

        # Should be None because error is not distributed and AST won't find a model either
        assert info is None
    finally:
        os.unlink(script_path)


def test_ghost_cli_distributed_error_json(tmp_path: Path):
    """ghost --json shows structured error for distributed scripts."""
    script_path = tmp_path / "train_ddp.py"
    script_path.write_text("import torch\ntorch.distributed.init_process_group('nccl')\n")

    dist_info = ModelInfo(
        param_count=0,
        dtype="float16",
        model_name=None,
        method="execution",
        extraction_error="distributed_entrypoint",
        extraction_detail="Script requires a distributed runtime (e.g. torchrun). Run ghost on the model definition file instead of the launcher script.",
    )

    with patch("alloc.model_extractor.extract_model_info", return_value=dist_info):
        result = runner.invoke(app, ["ghost", str(script_path), "--json"])

    assert result.exit_code != 0
    data = json.loads(result.output)
    assert data["error"] == "distributed_entrypoint"
    assert data["supported"] is False
    assert "distributed runtime" in data["detail"]


def test_ghost_cli_distributed_error_human(tmp_path: Path):
    """ghost shows human-readable message with tip for distributed scripts."""
    script_path = tmp_path / "train_ddp.py"
    script_path.write_text("import torch\n")

    dist_info = ModelInfo(
        param_count=0,
        dtype="float16",
        model_name=None,
        method="execution",
        extraction_error="distributed_entrypoint",
        extraction_detail="Script requires a distributed runtime (e.g. torchrun). Run ghost on the model definition file instead of the launcher script.",
    )

    with patch("alloc.model_extractor.extract_model_info", return_value=dist_info):
        result = runner.invoke(app, ["ghost", str(script_path)])

    assert result.exit_code != 0
    assert "distributed runtime" in result.output
    assert "model.py" in result.output  # tip about pointing to model file
