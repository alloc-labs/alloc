"""Tests for strategy inference from topology degrees (P0-B)."""

from __future__ import annotations

import os
from unittest.mock import patch

from alloc.cli import _infer_parallel_topology_from_env


class TestStrategyInference:
    """Strategy should be inferred from TP/PP/DP degrees when present."""

    def _topo(self, env=None, num_gpus=4):
        env = env or {}
        with patch.dict(os.environ, env, clear=False):
            return _infer_parallel_topology_from_env(
                num_gpus_detected=num_gpus,
            )

    def test_no_degrees_strategy_none(self):
        """When no degree env vars set, strategy should be None."""
        result = self._topo({})
        assert result["strategy"] is None

    def test_dp_only_is_ddp(self):
        """WORLD_SIZE=4 with no TP/PP → dp inferred → strategy=ddp."""
        result = self._topo({"WORLD_SIZE": "4"})
        assert result["strategy"] == "ddp"
        assert result["dp_degree"] == 4

    def test_tp_only(self):
        """TP_SIZE=4 alone → strategy=tp."""
        result = self._topo({"TP_SIZE": "4"})
        assert result["strategy"] == "tp"

    def test_pp_only(self):
        """PP_SIZE=4 alone → strategy=pp."""
        result = self._topo({"PP_SIZE": "4"})
        assert result["strategy"] == "pp"

    def test_tp_dp(self):
        """TP_SIZE=2 with DP_SIZE=2 → strategy=tp+dp."""
        result = self._topo({"TP_SIZE": "2", "DP_SIZE": "2"})
        assert result["strategy"] == "tp+dp"

    def test_pp_dp(self):
        """PP_SIZE=2 with DP_SIZE=2 → strategy=pp+dp."""
        result = self._topo({"PP_SIZE": "2", "DP_SIZE": "2"})
        assert result["strategy"] == "pp+dp"

    def test_tp_pp_dp(self):
        """All three degrees → strategy=tp+pp+dp."""
        result = self._topo({"TP_SIZE": "2", "PP_SIZE": "2", "DP_SIZE": "2"})
        assert result["strategy"] == "tp+pp+dp"

    def test_tp_pp_no_dp(self):
        """TP+PP without explicit DP → strategy=tp+pp+dp."""
        result = self._topo({"TP_SIZE": "2", "PP_SIZE": "2"})
        assert result["strategy"] == "tp+pp+dp"

    def test_tp_size_1_not_counted(self):
        """TP_SIZE=1 should not count as tensor parallelism."""
        result = self._topo({"TP_SIZE": "1", "DP_SIZE": "4"})
        assert result["strategy"] == "ddp"

    def test_pp_size_1_not_counted(self):
        """PP_SIZE=1 should not count as pipeline parallelism."""
        result = self._topo({"PP_SIZE": "1", "DP_SIZE": "4"})
        assert result["strategy"] == "ddp"

    def test_dp_inferred_from_world_size(self):
        """DP inferred from WORLD_SIZE / (TP * PP) → strategy includes dp."""
        result = self._topo({"WORLD_SIZE": "8", "TP_SIZE": "2"})
        assert result["dp_degree"] == 4
        assert result["strategy"] == "tp+dp"


class TestProcessMapInProbeDictAssembly:
    """process_map should reach probe_dict from ProbeResult."""

    def test_process_map_present_in_topology_return(self):
        """Topology dict now includes strategy field."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            topo = _infer_parallel_topology_from_env(num_gpus_detected=4)
        assert "strategy" in topo
        assert topo["strategy"] == "ddp"
