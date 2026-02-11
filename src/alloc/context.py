"""Context autodiscovery — capture environment metadata into the artifact.

All discovery functions are fail-safe: they return None on any error.
The artifact gets a `context` dict with optional fields. Never fails,
never slows down the CLI.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional


def discover_context() -> dict:
    """Discover all available environment context.

    Returns a dict with optional keys: git, container, ray.
    Only includes sections where data was found.
    """
    ctx = {}  # type: dict

    git = _discover_git()
    if git:
        ctx["git"] = git

    container = _discover_container()
    if container:
        ctx["container"] = container

    ray = _discover_ray()
    if ray:
        ctx["ray"] = ray

    return ctx


def _discover_git() -> Optional[dict]:
    """Discover git context: commit SHA, branch, repo name."""
    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        git = {}  # type: dict

        # Commit SHA
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            git["commit_sha"] = sha.stdout.strip()

        # Branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0 and branch.stdout.strip():
            git["branch"] = branch.stdout.strip()

        # Repo name (from remote origin URL or top-level dir name)
        remote = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            git["repo_url"] = remote.stdout.strip()
            git["repo_name"] = _repo_name_from_url(remote.stdout.strip())
        else:
            # Fallback: use the top-level directory name
            toplevel = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
            )
            if toplevel.returncode == 0 and toplevel.stdout.strip():
                git["repo_name"] = os.path.basename(toplevel.stdout.strip())

        return git if git else None
    except Exception:
        return None


def _repo_name_from_url(url: str) -> str:
    """Extract repo name from a git remote URL.

    Handles:
      git@github.com:org/repo.git → repo
      https://github.com/org/repo.git → repo
      https://github.com/org/repo → repo
    """
    # Strip trailing .git
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    # Take the last path component
    return name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _discover_container() -> Optional[dict]:
    """Discover container context: container ID, image name."""
    try:
        container = {}  # type: dict

        # Check for /.dockerenv (Docker) or /run/.containerenv (Podman)
        in_docker = os.path.isfile("/.dockerenv")
        in_podman = os.path.isfile("/run/.containerenv")

        if not in_docker and not in_podman:
            # Also check cgroup for container evidence
            if not _check_cgroup_for_container():
                return None

        # Container ID from /proc/self/cgroup or hostname
        cid = _read_container_id()
        if cid:
            container["container_id"] = cid

        # Image name from env var (commonly set by orchestrators)
        image = os.environ.get("CONTAINER_IMAGE") or os.environ.get("DOCKER_IMAGE")
        if image:
            container["image"] = image

        container["runtime"] = "podman" if in_podman else "docker"

        return container if container else None
    except Exception:
        return None


def _check_cgroup_for_container() -> bool:
    """Check /proc/self/cgroup for container evidence."""
    try:
        with open("/proc/self/cgroup", "r") as f:
            content = f.read()
        return "docker" in content or "containerd" in content or "kubepods" in content
    except Exception:
        return False


def _read_container_id() -> Optional[str]:
    """Read container ID from /proc/self/cgroup or hostname."""
    # Try /proc/self/cgroup first
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split("/")
                for part in reversed(parts):
                    # Container IDs are 64-char hex strings
                    cleaned = part.split("-")[-1].rstrip(".scope")
                    if len(cleaned) == 64 and all(c in "0123456789abcdef" for c in cleaned):
                        return cleaned[:12]  # Short form
    except Exception:
        pass

    # Fallback: hostname inside a container is often the short container ID
    try:
        hostname = os.environ.get("HOSTNAME", "")
        if len(hostname) == 12 and all(c in "0123456789abcdef" for c in hostname):
            return hostname
    except Exception:
        pass

    return None


def _discover_ray() -> Optional[dict]:
    """Discover Ray context from environment variables."""
    try:
        ray = {}  # type: dict

        job_id = os.environ.get("RAY_JOB_ID")
        if job_id:
            ray["job_id"] = job_id

        cluster = os.environ.get("RAY_CLUSTER_NAME") or os.environ.get("RAY_ADDRESS")
        if cluster:
            ray["cluster"] = cluster

        node_id = os.environ.get("RAY_NODE_ID")
        if node_id:
            ray["node_id"] = node_id

        return ray if ray else None
    except Exception:
        return None
