"""Tests for alloc ray submit."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch, call

import pytest
from typer.testing import CliRunner

from alloc.ray_submit import _build_entrypoint, _build_runtime_env, _submit_via_shell


runner = CliRunner()


# ---------------------------------------------------------------------------
# _build_runtime_env tests
# ---------------------------------------------------------------------------

def test_build_runtime_env_injects_alloc():
    """alloc[gpu] should appear in pip list."""
    env = _build_runtime_env()
    assert "alloc[gpu]" in env["pip"]


def test_build_runtime_env_no_duplicate():
    """If user already has alloc in pip, don't add again."""
    env = _build_runtime_env(extra_pip=["alloc[gpu]", "wandb"])
    alloc_count = sum(1 for p in env["pip"] if p.startswith("alloc"))
    assert alloc_count == 1


def test_build_runtime_env_no_duplicate_alloc_plain():
    """If user has plain 'alloc' in pip, don't add alloc[gpu]."""
    env = _build_runtime_env(extra_pip=["alloc", "wandb"])
    alloc_count = sum(1 for p in env["pip"] if p.startswith("alloc"))
    assert alloc_count == 1


@patch("alloc.ray_submit.get_token", return_value="test-token-123")
def test_build_runtime_env_token_passthrough(mock_token):
    """ALLOC_TOKEN should be passed in env_vars."""
    env = _build_runtime_env()
    assert env["env_vars"]["ALLOC_TOKEN"] == "test-token-123"


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_no_token(mock_token):
    """When no token, ALLOC_TOKEN should not be in env_vars."""
    env = _build_runtime_env()
    assert "ALLOC_TOKEN" not in env.get("env_vars", {})


@patch("alloc.ray_submit.get_token", return_value="tok")
def test_build_runtime_env_upload_flag(mock_token):
    """ALLOC_UPLOAD=true when upload=True."""
    env = _build_runtime_env(upload=True)
    assert env["env_vars"]["ALLOC_UPLOAD"] == "true"


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_no_upload_flag(mock_token):
    """ALLOC_UPLOAD should not be set when upload=False."""
    env = _build_runtime_env(upload=False)
    assert env.get("env_vars", {}).get("ALLOC_UPLOAD") is None


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_extra_pip(mock_token):
    """User's --pip packages should be in pip list."""
    env = _build_runtime_env(extra_pip=["wandb", "datasets"])
    assert "wandb" in env["pip"]
    assert "datasets" in env["pip"]
    assert "alloc[gpu]" in env["pip"]


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_extra_env(mock_token):
    """User's --env vars should be in env_vars."""
    env = _build_runtime_env(extra_env={"WANDB_API_KEY": "abc123"})
    assert env["env_vars"]["WANDB_API_KEY"] == "abc123"


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_working_dir(mock_token):
    """working_dir should be set when provided."""
    env = _build_runtime_env(working_dir="/tmp/my-project")
    assert env["working_dir"] == "/tmp/my-project"


@patch("alloc.ray_submit.get_token", return_value="")
def test_build_runtime_env_no_working_dir(mock_token):
    """working_dir should not be set when not provided."""
    env = _build_runtime_env()
    assert "working_dir" not in env


# ---------------------------------------------------------------------------
# _build_entrypoint tests
# ---------------------------------------------------------------------------

def test_build_entrypoint_basic():
    """Basic entrypoint should be 'alloc run -- python train.py'."""
    ep = _build_entrypoint(["python", "train.py"])
    assert ep == "alloc run -- python train.py"


def test_build_entrypoint_full():
    """--full flag should be passed through."""
    ep = _build_entrypoint(["python", "train.py"], full=True)
    assert "--full" in ep
    assert ep == "alloc run --full -- python train.py"


def test_build_entrypoint_upload():
    """--upload flag should be passed through."""
    ep = _build_entrypoint(["python", "train.py"], upload=True)
    assert "--upload" in ep


def test_build_entrypoint_no_config():
    """--no-config flag should be passed through."""
    ep = _build_entrypoint(["python", "train.py"], no_config=True)
    assert "--no-config" in ep


def test_build_entrypoint_custom_timeout():
    """Non-default timeout should be passed through."""
    ep = _build_entrypoint(["python", "train.py"], timeout=60)
    assert "--timeout 60" in ep


def test_build_entrypoint_default_timeout_omitted():
    """Default timeout (120) should not appear in entrypoint."""
    ep = _build_entrypoint(["python", "train.py"], timeout=120)
    assert "--timeout" not in ep


def test_build_entrypoint_all_flags():
    """All flags together."""
    ep = _build_entrypoint(
        ["python", "train.py"],
        full=True,
        timeout=60,
        upload=True,
        no_config=True,
    )
    assert "alloc run --full --timeout 60 --upload --no-config -- python train.py" == ep


# ---------------------------------------------------------------------------
# _submit_via_shell tests
# ---------------------------------------------------------------------------

@patch("alloc.ray_submit.subprocess.Popen")
def test_submit_via_shell_constructs_command(mock_popen):
    """Shell fallback should construct correct ray CLI args."""
    # Mock the Popen context
    mock_proc = MagicMock()
    mock_proc.stdout = iter([
        "Job 'raysubmit_abc123' submitted successfully\n",
        "Training complete\n",
    ])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0
    mock_popen.return_value = mock_proc

    result = _submit_via_shell(
        address="http://cluster:8265",
        entrypoint="alloc run -- python train.py",
        runtime_env={"pip": ["alloc[gpu]"]},
    )

    # Verify Popen was called with correct args
    args = mock_popen.call_args[0][0]
    assert args[0] == "ray"
    assert args[1] == "job"
    assert args[2] == "submit"
    assert "--address" in args
    assert "http://cluster:8265" in args
    assert "--runtime-env-json" in args
    assert result["job_id"] == "raysubmit_abc123"
    assert result["status"] == "SUCCEEDED"
    assert result["exit_code"] == 0


@patch("alloc.ray_submit.subprocess.Popen")
def test_submit_via_shell_no_wait(mock_popen):
    """--no-wait should be passed to ray CLI."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Submitted raysubmit_xyz\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0
    mock_popen.return_value = mock_proc

    _submit_via_shell(
        address="http://localhost:8265",
        entrypoint="alloc run -- python train.py",
        runtime_env={"pip": ["alloc[gpu]"]},
        no_wait=True,
    )

    args = mock_popen.call_args[0][0]
    assert "--no-wait" in args


@patch("alloc.ray_submit.subprocess.Popen", side_effect=FileNotFoundError)
def test_submit_via_shell_ray_not_found(mock_popen):
    """When ray CLI is not found, return error."""
    result = _submit_via_shell(
        address="http://localhost:8265",
        entrypoint="alloc run -- python train.py",
        runtime_env={"pip": ["alloc[gpu]"]},
    )
    assert result["status"] == "ERROR"
    assert result["exit_code"] == 1
    assert "ray CLI not found" in result["error"]


# ---------------------------------------------------------------------------
# _submit_via_api tests
# ---------------------------------------------------------------------------

def test_submit_via_api_calls_client():
    """API path should call submit_job on JobSubmissionClient."""
    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.submit_job.return_value = "raysubmit_api123"
    mock_client.tail_job_logs.return_value = iter(["log line\n"])
    mock_client.get_job_status.return_value = "SUCCEEDED"

    # Patch the import inside the function
    with patch.dict("sys.modules", {"ray": MagicMock(), "ray.job_submission": MagicMock(JobSubmissionClient=mock_client_cls)}):
        from alloc.ray_submit import _submit_via_api
        result = _submit_via_api(
            address="http://cluster:8265",
            entrypoint="alloc run -- python train.py",
            runtime_env={"pip": ["alloc[gpu]"]},
        )

    mock_client_cls.assert_called_once_with("http://cluster:8265")
    mock_client.submit_job.assert_called_once_with(
        entrypoint="alloc run -- python train.py",
        runtime_env={"pip": ["alloc[gpu]"]},
    )
    assert result["job_id"] == "raysubmit_api123"
    assert result["status"] == "SUCCEEDED"


def test_submit_via_api_no_wait():
    """No-wait mode should return immediately after submit."""
    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.submit_job.return_value = "raysubmit_nw456"

    with patch.dict("sys.modules", {"ray": MagicMock(), "ray.job_submission": MagicMock(JobSubmissionClient=mock_client_cls)}):
        from alloc.ray_submit import _submit_via_api
        result = _submit_via_api(
            address="http://cluster:8265",
            entrypoint="alloc run -- python train.py",
            runtime_env={"pip": ["alloc[gpu]"]},
            no_wait=True,
        )

    assert result["status"] == "SUBMITTED"
    mock_client.tail_job_logs.assert_not_called()


# ---------------------------------------------------------------------------
# submit_ray_job integration tests (API vs shell fallback)
# ---------------------------------------------------------------------------

def test_submit_no_ray_falls_back_to_shell():
    """When ray is not importable, submit_ray_job should use shell fallback."""
    import importlib
    # Make ray import fail
    with patch.dict("sys.modules", {"ray": None, "ray.job_submission": None}):
        with patch("alloc.ray_submit._submit_via_shell") as mock_shell:
            mock_shell.return_value = {"job_id": "raysubmit_shell", "status": "SUCCEEDED", "exit_code": 0}
            with patch("alloc.ray_submit.get_token", return_value=""):
                from alloc.ray_submit import submit_ray_job
                result = submit_ray_job(command=["python", "train.py"])
    assert mock_shell.called
    assert result["job_id"] == "raysubmit_shell"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_ray_submit_help():
    """alloc ray submit --help should show options."""
    from alloc.cli import app
    result = runner.invoke(app, ["ray", "submit", "--help"])
    assert result.exit_code == 0
    assert "--address" in result.output
    assert "--upload" in result.output
    assert "--full" in result.output
    assert "--no-wait" in result.output
    assert "--pip" in result.output
    assert "--env" in result.output


def test_ray_help():
    """alloc ray --help should show submit subcommand."""
    from alloc.cli import app
    result = runner.invoke(app, ["ray", "--help"])
    assert result.exit_code == 0
    assert "submit" in result.output
