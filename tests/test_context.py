"""Tests for context autodiscovery."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from alloc.context import (
    discover_context,
    _discover_git,
    _discover_container,
    _discover_ray,
    _repo_name_from_url,
)


class TestRepoNameFromUrl:
    def test_ssh_url(self):
        assert _repo_name_from_url("git@github.com:alloc-labs/alloc.git") == "alloc"

    def test_https_url(self):
        assert _repo_name_from_url("https://github.com/alloc-labs/alloc.git") == "alloc"

    def test_https_no_git_suffix(self):
        assert _repo_name_from_url("https://github.com/alloc-labs/alloc") == "alloc"

    def test_trailing_slash(self):
        assert _repo_name_from_url("https://github.com/alloc-labs/alloc/") == "alloc"


class TestDiscoverGit:
    def test_returns_dict_in_git_repo(self):
        """Should return git context when inside a git repo."""
        # We're running tests inside the alloc monorepo, so git context should be available
        result = _discover_git()
        assert result is not None
        assert "commit_sha" in result
        assert len(result["commit_sha"]) == 40  # Full SHA
        assert "branch" in result

    def test_returns_none_outside_git(self, tmp_path, monkeypatch):
        """Should return None when not in a git repo."""
        monkeypatch.chdir(tmp_path)
        result = _discover_git()
        assert result is None


class TestDiscoverRay:
    def test_returns_none_without_env(self):
        """Should return None when no Ray env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            result = _discover_ray()
            assert result is None

    def test_returns_job_id(self):
        """Should capture RAY_JOB_ID when set."""
        with patch.dict(os.environ, {"RAY_JOB_ID": "raysubmit_abc123"}):
            result = _discover_ray()
            assert result is not None
            assert result["job_id"] == "raysubmit_abc123"

    def test_returns_cluster_from_address(self):
        """Should capture cluster from RAY_ADDRESS."""
        with patch.dict(os.environ, {"RAY_JOB_ID": "j1", "RAY_ADDRESS": "ray://10.0.0.1:10001"}):
            result = _discover_ray()
            assert result["cluster"] == "ray://10.0.0.1:10001"

    def test_returns_cluster_name_over_address(self):
        """RAY_CLUSTER_NAME should take precedence over RAY_ADDRESS."""
        with patch.dict(os.environ, {
            "RAY_JOB_ID": "j1",
            "RAY_CLUSTER_NAME": "my-cluster",
            "RAY_ADDRESS": "ray://10.0.0.1:10001",
        }):
            result = _discover_ray()
            assert result["cluster"] == "my-cluster"


class TestDiscoverContainer:
    def test_returns_none_on_host(self):
        """Should return None when not in a container (no /.dockerenv, no cgroup evidence)."""
        # On macOS (where tests run), there's no /.dockerenv or /proc/self/cgroup
        result = _discover_container()
        assert result is None

    @patch("alloc.context.os.path.isfile")
    def test_detects_docker(self, mock_isfile):
        """Should detect Docker container."""
        def isfile_side_effect(path):
            return path == "/.dockerenv"
        mock_isfile.side_effect = isfile_side_effect

        with patch.dict(os.environ, {"HOSTNAME": "abc123def456"}):
            result = _discover_container()
            assert result is not None
            assert result["runtime"] == "docker"
            assert result["container_id"] == "abc123def456"

    @patch("alloc.context.os.path.isfile")
    def test_detects_podman(self, mock_isfile):
        """Should detect Podman container."""
        def isfile_side_effect(path):
            return path == "/run/.containerenv"
        mock_isfile.side_effect = isfile_side_effect

        result = _discover_container()
        assert result is not None
        assert result["runtime"] == "podman"


class TestDiscoverContext:
    def test_returns_dict(self):
        """discover_context() should always return a dict."""
        result = discover_context()
        assert isinstance(result, dict)

    def test_includes_git_in_repo(self):
        """Should include git context when in a git repo."""
        result = discover_context()
        assert "git" in result

    def test_no_ray_without_env(self):
        """Should not include ray context without env vars."""
        with patch.dict(os.environ, {}, clear=True):
            # Still includes git since we're in a repo
            result = discover_context()
            assert "ray" not in result

    def test_never_raises(self):
        """discover_context() should never raise, even if subprocess fails."""
        with patch("alloc.context.subprocess.run", side_effect=Exception("boom")):
            result = discover_context()
            assert isinstance(result, dict)
