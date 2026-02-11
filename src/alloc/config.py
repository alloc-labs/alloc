"""Alloc CLI configuration — persistent config file + env var overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_CONFIG_DIR = Path.home() / ".alloc"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULT_API_URL = "https://alloc-production-ffc2.up.railway.app"
_DEFAULT_SUPABASE_URL = "https://stysqykttruzpcnzxshp.supabase.co"
_DEFAULT_SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0eXNxeWt0dHJ1enBjbnp4c2hwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA1ODgzOTQsImV4cCI6MjA4NjE2NDM5NH0.cHOBh5ei90Vj359TesKJ5GMZlyLoWFkMoNYs-HrKAtw"


def load_config() -> dict:
    """Read ~/.alloc/config.json or return empty dict."""
    try:
        if _CONFIG_FILE.exists():
            return json.loads(_CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}


def save_config(data: dict) -> None:
    """Write data to ~/.alloc/config.json. Creates dir if needed."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")
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
