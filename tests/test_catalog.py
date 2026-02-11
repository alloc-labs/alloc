"""Tests for GPU catalog CLI module."""

from __future__ import annotations

from alloc.catalog import list_gpus, get_gpu, _ALIASES


class TestListGpus:
    def test_returns_list(self):
        gpus = list_gpus()
        assert isinstance(gpus, list)
        assert len(gpus) > 0

    def test_has_13_gpus(self):
        gpus = list_gpus()
        assert len(gpus) == 13

    def test_sorted_by_vram_desc(self):
        gpus = list_gpus()
        vrams = [g["vram_gb"] for g in gpus]
        assert vrams == sorted(vrams, reverse=True)

    def test_gpu_has_required_fields(self):
        gpus = list_gpus()
        for g in gpus:
            assert "id" in g
            assert "display_name" in g
            assert "vram_gb" in g
            assert "architecture" in g
            assert "pricing" in g

    def test_h100_has_pricing(self):
        gpus = list_gpus()
        h100 = [g for g in gpus if g["display_name"] == "H100-80GB"]
        assert len(h100) == 1
        assert "aws" in h100[0]["pricing"]


class TestGetGpu:
    def test_by_stable_id(self):
        gpu = get_gpu("nvidia-h100-sxm-80gb")
        assert gpu is not None
        assert gpu["display_name"] == "H100-80GB"
        assert gpu["vram_gb"] == 80

    def test_by_alias(self):
        gpu = get_gpu("H100")
        assert gpu is not None
        assert gpu["display_name"] == "H100-80GB"

    def test_by_display_name_alias(self):
        gpu = get_gpu("A100-80GB")
        assert gpu is not None
        assert gpu["vram_gb"] == 80

    def test_unknown_returns_none(self):
        gpu = get_gpu("nvidia-nonexistent")
        assert gpu is None

    def test_has_interconnect(self):
        gpu = get_gpu("nvidia-h100-sxm-80gb")
        assert gpu is not None
        assert gpu["interconnect"] is not None
        assert gpu["interconnect"]["nvlink_gen"] == 4

    def test_has_pricing(self):
        gpu = get_gpu("nvidia-a100-sxm-80gb")
        assert gpu is not None
        assert "aws" in gpu["pricing"]
        assert gpu["pricing"]["aws"] > 0

    def test_t4_no_bf16(self):
        gpu = get_gpu("T4")
        assert gpu is not None
        assert gpu["bf16_tflops"] == 0


class TestAliases:
    def test_all_aliases_resolve(self):
        """Every alias should resolve to a valid GPU."""
        for alias, stable_id in _ALIASES.items():
            gpu = get_gpu(stable_id)
            assert gpu is not None, f"Alias {alias} -> {stable_id} not found"
