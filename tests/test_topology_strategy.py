"""Tests for strategy inference from topology degrees (P0-B)."""

from __future__ import annotations

import os
from unittest.mock import patch

from alloc.cli import _infer_parallel_topology_from_env


class TestStrategyInference:
    """Strategy should be inferred from TP/PP/DP degrees when present."""

    def _topo(self, env=None, num_gpus=4, **kwargs):
        env = env or {}
        with patch.dict(os.environ, env, clear=False):
            return _infer_parallel_topology_from_env(
                num_gpus_detected=num_gpus,
                **kwargs,
            )

    def test_no_degrees_multi_gpu_strategy_none(self):
        """When no degree env vars and no AST hint, strategy stays None."""
        result = self._topo({}, num_gpus=4)
        assert result["strategy"] is None
        assert result["strategy_detection_method"] is None

    def test_single_gpu_no_degrees_strategy_none(self):
        """Single GPU with no degrees → strategy stays None."""
        result = self._topo({}, num_gpus=1)
        assert result["strategy"] is None

    def test_world_size_only_strategy_none(self):
        """WORLD_SIZE=4 with no explicit DP_SIZE → strategy=None (ambiguous)."""
        result = self._topo({"WORLD_SIZE": "4"})
        assert result["strategy"] is None
        assert result["dp_degree"] == 4  # degree still derived for topology

    def test_explicit_dp_size_is_ddp(self):
        """Explicit DP_SIZE=4 → strategy=ddp."""
        result = self._topo({"DP_SIZE": "4"})
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


class TestStrategyOverride:
    """--strategy override takes highest precedence."""

    def _topo(self, env=None, num_gpus=4, **kwargs):
        env = env or {}
        with patch.dict(os.environ, env, clear=False):
            return _infer_parallel_topology_from_env(
                num_gpus_detected=num_gpus,
                **kwargs,
            )

    def test_override_beats_env_degrees(self):
        """Explicit --strategy overrides env var inference."""
        result = self._topo({"WORLD_SIZE": "4"}, strategy_override="fsdp")
        assert result["strategy"] == "fsdp"
        assert result["strategy_detection_method"] == "user_override"

    def test_override_beats_ast_hint(self):
        """Explicit --strategy overrides AST hint."""
        result = self._topo({}, num_gpus=2, strategy_override="fsdp", ast_strategy_hint="ddp")
        assert result["strategy"] == "fsdp"
        assert result["strategy_detection_method"] == "user_override"

    def test_ast_hint_used_when_multi_gpu(self):
        """AST hint used when no env degrees and multi-GPU."""
        result = self._topo({}, num_gpus=2, ast_strategy_hint="fsdp")
        assert result["strategy"] == "fsdp"
        assert result["strategy_detection_method"] == "ast_analysis"
        assert result["dp_degree"] == 2

    def test_ast_hint_ignored_when_single_gpu(self):
        """AST hint ignored for single GPU — no distributed strategy applies."""
        result = self._topo({}, num_gpus=1, ast_strategy_hint="fsdp")
        assert result["strategy"] is None

    def test_env_degrees_beat_ast_hint(self):
        """Env var TP/PP degrees take precedence over AST hint."""
        result = self._topo({"TP_SIZE": "2", "DP_SIZE": "2"}, ast_strategy_hint="fsdp")
        assert result["strategy"] == "tp+dp"
        assert result["strategy_detection_method"] == "env_degrees"

    def test_world_size_plus_ast_fsdp_returns_fsdp(self):
        """WORLD_SIZE=2 + ast_hint='fsdp' → strategy='fsdp' (real torchrun FSDP case)."""
        result = self._topo({"WORLD_SIZE": "2"}, num_gpus=2, ast_strategy_hint="fsdp")
        assert result["strategy"] == "fsdp"
        assert result["strategy_detection_method"] == "ast_analysis"
        assert result["dp_degree"] == 2

    def test_world_size_plus_ast_deepspeed_returns_deepspeed(self):
        """WORLD_SIZE=4 + ast_hint='deepspeed' → strategy='deepspeed'."""
        result = self._topo({"WORLD_SIZE": "4"}, num_gpus=4, ast_strategy_hint="deepspeed")
        assert result["strategy"] == "deepspeed"
        assert result["strategy_detection_method"] == "ast_analysis"

    def test_world_size_only_no_hint_stays_none(self):
        """WORLD_SIZE=2 with no AST hint → strategy=None (ambiguous)."""
        result = self._topo({"WORLD_SIZE": "2"}, num_gpus=2)
        assert result["strategy"] is None
        assert result["dp_degree"] == 2

    def test_unknown_multi_gpu_stays_none(self):
        """Multi-GPU with no hint and no env vars → strategy=None, not ddp."""
        result = self._topo({}, num_gpus=4)
        assert result["strategy"] is None
        assert result["strategy_detection_method"] is None

    def test_strategy_detection_method_in_result(self):
        """strategy_detection_method is always present in result."""
        result = self._topo({"DP_SIZE": "4"})
        assert "strategy_detection_method" in result
        assert result["strategy_detection_method"] == "env_degrees"


class TestDetectStrategyHint:
    """code_analyzer.detect_strategy_hint returns correct strategy from AST."""

    def test_fsdp_script(self, tmp_path):
        script = tmp_path / "train_fsdp.py"
        script.write_text(
            "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n"
            "model = FSDP(model)\n"
        )
        from alloc.code_analyzer import detect_strategy_hint
        assert detect_strategy_hint(str(script)) == "fsdp"

    def test_ddp_script(self, tmp_path):
        script = tmp_path / "train_ddp.py"
        script.write_text(
            "from torch.nn.parallel import DistributedDataParallel as DDP\n"
            "model = DDP(model)\n"
        )
        from alloc.code_analyzer import detect_strategy_hint
        assert detect_strategy_hint(str(script)) == "ddp"

    def test_fsdp_beats_ddp_when_both_present(self, tmp_path):
        script = tmp_path / "train_both.py"
        script.write_text(
            "from torch.nn.parallel import DistributedDataParallel as DDP\n"
            "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n"
            "model = FSDP(model)\n"
        )
        from alloc.code_analyzer import detect_strategy_hint
        assert detect_strategy_hint(str(script)) == "fsdp"

    def test_no_distributed_returns_none(self, tmp_path):
        script = tmp_path / "train_simple.py"
        script.write_text("import torch\nmodel = torch.nn.Linear(10, 10)\n")
        from alloc.code_analyzer import detect_strategy_hint
        assert detect_strategy_hint(str(script)) is None

    def test_nonexistent_file_returns_none(self):
        from alloc.code_analyzer import detect_strategy_hint
        assert detect_strategy_hint("/nonexistent/train.py") is None


class TestProcessMapInProbeDictAssembly:
    """process_map should reach probe_dict from ProbeResult."""

    def test_process_map_present_in_topology_return(self):
        """Topology dict now includes strategy field."""
        with patch.dict(os.environ, {"DP_SIZE": "4"}, clear=False):
            topo = _infer_parallel_topology_from_env(num_gpus_detected=4)
        assert "strategy" in topo
        assert topo["strategy"] == "ddp"
