"""Alloc Ray integration — submit jobs to Ray clusters with GPU monitoring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from alloc.config import get_api_url, get_token


def submit_ray_job(
    command: List[str],
    address: str = "http://127.0.0.1:8265",
    working_dir: Optional[str] = None,
    upload: bool = False,
    full: bool = False,
    timeout: int = 120,
    no_config: bool = False,
    no_wait: bool = False,
    extra_pip: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Submit a training job to a Ray cluster wrapped with alloc run.

    Returns dict with keys: job_id, status, exit_code.
    """
    entrypoint = _build_entrypoint(command, full=full, timeout=timeout, upload=upload, no_config=no_config)
    runtime_env = _build_runtime_env(
        extra_pip=extra_pip,
        extra_env=extra_env,
        upload=upload,
        working_dir=working_dir,
    )

    # Try Python API first, fall back to shell
    try:
        from ray.job_submission import JobSubmissionClient  # type: ignore[import]
        return _submit_via_api(
            address=address,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            no_wait=no_wait,
        )
    except ImportError:
        return _submit_via_shell(
            address=address,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            no_wait=no_wait,
        )


def _build_entrypoint(
    command: List[str],
    full: bool = False,
    timeout: int = 120,
    upload: bool = False,
    no_config: bool = False,
) -> str:
    """Build the alloc run ... entrypoint string for the Ray worker."""
    parts = ["alloc", "run"]
    if full:
        parts.append("--full")
    if timeout != 120:
        parts.extend(["--timeout", str(timeout)])
    if upload:
        parts.append("--upload")
    if no_config:
        parts.append("--no-config")
    parts.append("--")
    parts.extend(command)
    return " ".join(parts)


def _build_runtime_env(
    extra_pip: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    upload: bool = False,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct Ray runtime_env dict with alloc injected."""
    runtime_env = {}  # type: Dict[str, Any]

    # Pip dependencies — inject alloc[gpu]
    pip_deps = list(extra_pip) if extra_pip else []
    # Don't duplicate if user already has alloc in their deps
    has_alloc = any(dep.startswith("alloc") for dep in pip_deps)
    if not has_alloc:
        pip_deps.insert(0, "alloc[gpu]")
    runtime_env["pip"] = pip_deps

    # Environment variables
    env_vars = dict(extra_env) if extra_env else {}

    # Pass auth token so upload works on the worker
    token = get_token()
    if token:
        env_vars.setdefault("ALLOC_TOKEN", token)

    # Pass upload flag
    if upload:
        env_vars.setdefault("ALLOC_UPLOAD", "true")

    # Pass API URL if non-default
    api_url = get_api_url()
    if api_url and "ALLOC_API_URL" not in env_vars:
        env_vars["ALLOC_API_URL"] = api_url

    if env_vars:
        runtime_env["env_vars"] = env_vars

    # Working directory
    if working_dir:
        runtime_env["working_dir"] = working_dir

    return runtime_env


def _submit_via_api(
    address: str,
    entrypoint: str,
    runtime_env: Dict[str, Any],
    no_wait: bool = False,
) -> Dict[str, Any]:
    """Submit via Ray Python API (JobSubmissionClient)."""
    from ray.job_submission import JobSubmissionClient  # type: ignore[import]

    client = JobSubmissionClient(address)
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)

    if no_wait:
        return {"job_id": job_id, "status": "SUBMITTED", "exit_code": 0}

    # Stream logs
    try:
        for line in client.tail_job_logs(job_id):
            print(line, end="")
    except Exception:
        pass

    # Poll for final status
    status = _poll_job_status(client, job_id)
    exit_code = 0 if status == "SUCCEEDED" else 1
    return {"job_id": job_id, "status": status, "exit_code": exit_code}


def _poll_job_status(client: Any, job_id: str, timeout: int = 600) -> str:
    """Poll Ray job status until terminal state or timeout."""
    terminal = {"SUCCEEDED", "FAILED", "STOPPED"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = str(client.get_job_status(job_id))
            # Ray returns JobStatus enum — normalize to string
            status_str = status.replace("JobStatus.", "")
            if status_str in terminal:
                return status_str
        except Exception:
            pass
        time.sleep(2)
    return "TIMEOUT"


def _submit_via_shell(
    address: str,
    entrypoint: str,
    runtime_env: Dict[str, Any],
    no_wait: bool = False,
) -> Dict[str, Any]:
    """Submit via `ray job submit` CLI (shell fallback)."""
    runtime_env_json = json.dumps(runtime_env)

    cmd = [
        "ray", "job", "submit",
        "--address", address,
        "--runtime-env-json", runtime_env_json,
    ]
    if no_wait:
        cmd.append("--no-wait")
    cmd.append("--")
    cmd.extend(entrypoint.split())

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return {
            "job_id": None,
            "status": "ERROR",
            "exit_code": 1,
            "error": "ray CLI not found. Install with: pip install alloc[ray]",
        }

    job_id = None
    lines = []
    for line in proc.stdout:  # type: ignore[union-attr]
        print(line, end="")
        lines.append(line)
        # Parse job ID from Ray output
        if "submitted successfully" in line.lower() or "raysubmit_" in line:
            for word in line.split():
                clean = word.strip("'\"")
                if clean.startswith("raysubmit_"):
                    job_id = clean
                    break

    proc.wait()
    exit_code = proc.returncode or 0
    status = "SUCCEEDED" if exit_code == 0 else "FAILED"
    return {"job_id": job_id, "status": status, "exit_code": exit_code}
