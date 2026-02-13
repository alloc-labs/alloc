"""Alloc CLI configuration — persistent config file + env var overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

def _config_dir() -> Path:
    # Compute dynamically so tests and containerized runs can override HOME.
    return Path.home() / ".alloc"


def _config_file() -> Path:
    return _config_dir() / "config.json"

_DEFAULT_API_URL = "https://alloc-production-ffc2.up.railway.app"
_DEFAULT_SUPABASE_URL = "https://stysqykttruzpcnzxshp.supabase.co"
_DEFAULT_SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0eXNxeWt0dHJ1enBjbnp4c2hwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA1ODgzOTQsImV4cCI6MjA4NjE2NDM5NH0.cHOBh5ei90Vj359TesKJ5GMZlyLoWFkMoNYs-HrKAtw"


def load_config() -> dict:
    """Read ~/.alloc/config.json or return empty dict."""
    try:
        cfg_file = _config_file()
        if cfg_file.exists():
            return json.loads(cfg_file.read_text())
    except Exception:
        pass
    return {}


def save_config(data: dict) -> None:
    """Write data to ~/.alloc/config.json. Creates dir if needed."""
    try:
        cfg_dir = _config_dir()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(cfg_dir, 0o700)
        cfg_file = _config_file()
        cfg_file.write_text(json.dumps(data, indent=2) + "\n")
        os.chmod(cfg_file, 0o600)
    except Exception:
        pass


def get_token() -> str:
    """Auth token. Env var takes precedence over config file."""
    env = os.environ.get("ALLOC_TOKEN", "")
    if env:
        return env
    return load_config().get("token", "")


def get_api_url() -> str:
    """API URL. Env var > config file > default."""
    env = os.environ.get("ALLOC_API_URL", "")
    if env:
        return env
    return load_config().get("api_url", _DEFAULT_API_URL)


def get_supabase_url() -> str:
    """Supabase URL. Env var > default."""
    return os.environ.get("ALLOC_SUPABASE_URL", _DEFAULT_SUPABASE_URL)


def get_supabase_anon_key() -> str:
    """Supabase anon key. Env var > default."""
    return os.environ.get("ALLOC_SUPABASE_ANON_KEY", _DEFAULT_SUPABASE_ANON_KEY)


def should_upload() -> bool:
    """Whether to upload results to the Alloc dashboard."""
    return os.environ.get("ALLOC_UPLOAD", "").lower() in ("1", "true", "yes")


def try_refresh_access_token() -> Optional[str]:
    """Attempt to refresh the saved access token using refresh_token.

    Returns the new access token on success, otherwise None.

    Notes:
    - If ALLOC_TOKEN is set, this function returns None (env tokens can't be updated).
    - This calls Supabase directly, so it requires ALLOC_SUPABASE_URL and
      ALLOC_SUPABASE_ANON_KEY (or defaults) to be correct.
    """
    if os.environ.get("ALLOC_TOKEN"):
        return None

    cfg = load_config()
    refresh_token = (cfg.get("refresh_token") or "").strip()
    if not refresh_token:
        return None

    # Local import to keep config module import-time side effects minimal.
    import httpx

    supabase_url = get_supabase_url()
    anon_key = get_supabase_anon_key()

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{supabase_url}/auth/v1/token?grant_type=refresh_token",
                json={"refresh_token": refresh_token},
                headers={
                    "apikey": anon_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    access_token = (data.get("access_token") or "").strip()
    if not access_token:
        return None

    cfg["token"] = access_token
    cfg["refresh_token"] = (data.get("refresh_token") or refresh_token).strip()
    save_config(cfg)
    return access_token
