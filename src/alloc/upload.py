"""Artifact upload — POST extracted fields to /runs/ingest as JSON."""

from __future__ import annotations

import gzip
import json


class UploadLimitError(Exception):
    """Raised when the server rejects an upload due to tier limits."""

    def __init__(self, status_code: int, detail: dict):
        self.status_code = status_code
        self.detail = detail
        msg = detail.get("message", f"Upload rejected (HTTP {status_code})")
        super().__init__(msg)


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

    payload = {
        "peak_vram_mb": probe.get("peak_vram_mb"),
        "avg_gpu_util": probe.get("avg_gpu_util"),
        "avg_power_watts": probe.get("avg_power_watts"),
        "duration_s": probe.get("duration_seconds"),
        "exit_code": probe.get("exit_code"),
        "probe_samples": probe.get("samples"),
        "ghost_report": ghost if ghost else None,
        "source": "cli",
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
