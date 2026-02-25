"""Artifact upload — POST extracted fields to /runs/ingest as JSON."""

from __future__ import annotations

import gzip
import json
from typing import Optional


class UploadLimitError(Exception):
    """Raised when the server rejects an upload due to tier limits."""

    def __init__(self, status_code: int, detail: dict):
        self.status_code = status_code
        self.detail = detail
        msg = detail.get("message", f"Upload rejected (HTTP {status_code})")
        super().__init__(msg)


def _normalize_gpu_type(raw_name: object) -> Optional[str]:
    """Best-effort normalization from NVML GPU names to catalog IDs."""
    if raw_name is None:
        return None

    name = str(raw_name).strip()
    if not name:
        return None

    upper = name.upper()
    if "H100" in upper and "NVL" in upper:
        return "H100-NVL"
    if "H200" in upper:
        return "H200"
    if "H100" in upper:
        return "H100-80GB"
    if "A100" in upper and "40" in upper:
        return "A100-40GB"
    if "A100" in upper:
        return "A100-80GB"
    if "A10G" in upper:
        return "A10G"
    if "L40S" in upper:
        return "L40S"
    if "L4" in upper:
        return "L4"
    if "T4" in upper:
        return "T4"
    if "V100" in upper:
        return "V100-32GB"
    if "4090" in upper:
        return "RTX-4090"
    if "3090" in upper:
        return "RTX-3090"
    return None


def _to_positive_int(value: object, default: int = 1) -> int:
    """Parse a positive integer with a safe fallback."""
    try:
        parsed = int(value)  # type: ignore[arg-type]
        return parsed if parsed > 0 else default
    except Exception:
        return default


def upload_artifact(artifact_path: str, api_url: str, token: str) -> dict:
    """Upload a .json.gz artifact to POST /runs/ingest.

    Reads the artifact, extracts summary fields, and sends JSON.
    Returns response dict with run_id and status.
    Raises UploadLimitError on 402/403/429, other errors via raise_for_status.
    """
    import httpx

    with gzip.open(artifact_path, "rt", encoding="utf-8") as f:
        report = json.load(f)

    probe = report.get("probe") or {}
    ghost = report.get("ghost") or {}
    hardware = report.get("hardware") or {}
    context = report.get("context") or {}
    git_ctx = context.get("git") or {}

    gpu_type = (
        probe.get("gpu_type")
        or _normalize_gpu_type(probe.get("gpu_name"))
        or _normalize_gpu_type(hardware.get("gpu_name"))
    )
    num_gpus = _to_positive_int(
        probe.get("num_gpus") or hardware.get("num_gpus_detected"),
        default=1,
    )

    payload = {
        "model_name": probe.get("model_name"),
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "strategy": probe.get("strategy"),
        "tp_degree": probe.get("tp_degree"),
        "pp_degree": probe.get("pp_degree"),
        "dp_degree": probe.get("dp_degree"),
        "num_nodes": probe.get("num_nodes"),
        "gpus_per_node": probe.get("gpus_per_node"),
        "interconnect_type": probe.get("interconnect_type"),
        "objective": probe.get("objective"),
        "max_budget_hourly": probe.get("max_budget_hourly"),
        "peak_vram_mb": probe.get("peak_vram_mb"),
        "avg_gpu_util": probe.get("avg_gpu_util"),
        "avg_power_watts": probe.get("avg_power_watts"),
        "duration_s": probe.get("duration_seconds"),
        "exit_code": probe.get("exit_code"),
        "probe_samples": probe.get("samples"),
        "step_count": probe.get("step_count"),
        "step_time_ms_p50": probe.get("step_time_ms_p50"),
        "step_time_ms_p90": probe.get("step_time_ms_p90"),
        "samples_per_sec": probe.get("samples_per_sec"),
        "dataloader_wait_pct": probe.get("dataloader_wait_pct"),
        "comm_overhead_pct": probe.get("comm_overhead_pct"),
        "per_rank_peak_vram_mb": probe.get("per_rank_peak_vram_mb"),
        # Architecture fields: probe (callbacks) takes priority over ghost defaults
        "batch_size": probe.get("batch_size") or (ghost.get("batch_size") if ghost else None),
        "seq_length": ghost.get("seq_length") if ghost else None,
        "hidden_dim": ghost.get("hidden_dim") if ghost else None,
        "num_layers": ghost.get("num_layers") if ghost else None,
        "ghost_report": ghost if ghost else None,
        "source": probe.get("source") or "cli",
        "command": probe.get("command"),
        "git_sha": git_ctx.get("commit_sha"),
        "git_branch": git_ctx.get("branch"),
        # Phase timing (from CUDA events in callbacks)
        "phase_forward_ms_p50": probe.get("phase_forward_ms_p50"),
        "phase_forward_ms_p90": probe.get("phase_forward_ms_p90"),
        "phase_backward_ms_p50": probe.get("phase_backward_ms_p50"),
        "phase_backward_ms_p90": probe.get("phase_backward_ms_p90"),
        "phase_optimizer_ms_p50": probe.get("phase_optimizer_ms_p50"),
        "phase_optimizer_ms_p90": probe.get("phase_optimizer_ms_p90"),
        "phase_dataloader_ms_p50": probe.get("phase_dataloader_ms_p50"),
        "phase_dataloader_ms_p90": probe.get("phase_dataloader_ms_p90"),
        "has_phase_timing": probe.get("has_phase_timing"),
        # Architecture metadata (from callback introspection)
        "architecture_type": probe.get("architecture_type"),
        "optimizer_type": probe.get("optimizer_type"),
        "fine_tuning_method": probe.get("fine_tuning_method"),
        "gradient_checkpointing": probe.get("gradient_checkpointing"),
        "attention_type": probe.get("attention_type"),
        "param_count": probe.get("param_count"),
        "trainable_param_count": probe.get("trainable_param_count"),
        # Outcome tracking: link to previous run
        "baseline_run_id": probe.get("baseline_run_id"),
    }

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{api_url}/runs/ingest",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )

        if resp.status_code in (402, 403, 429):
            try:
                detail = resp.json().get("detail", {})
            except Exception:
                detail = {"message": resp.text[:200]}
            raise UploadLimitError(resp.status_code, detail)

        resp.raise_for_status()
        return resp.json()
