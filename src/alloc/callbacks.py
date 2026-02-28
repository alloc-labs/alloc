"""Alloc Framework Callbacks — self-contained GPU monitoring + step timing.

Each callback captures GPU metrics (VRAM, utilization, power) via a background
NVML thread AND step timing, then writes a full alloc_artifact.json.gz on
train_end. No ``alloc run`` wrapper needed.

Phase timing via CUDA events (P1.5): when CUDA is available, records
forward/backward/optimizer/dataloader phase durations at framework hook
boundaries. Falls back to wall-clock only when CUDA is unavailable.

Also writes the legacy sidecar (.alloc_callback.json) for backwards compat
with ``alloc run`` workflows.

Usage (HuggingFace):
    from alloc.callbacks import AllocCallback
    trainer = Trainer(..., callbacks=[AllocCallback()])

Usage (PyTorch Lightning):
    from alloc.callbacks import AllocLightningCallback
    trainer = Trainer(..., callbacks=[AllocLightningCallback()])
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────

_ROLLING_WINDOW = 200   # keep last N step times
_WRITE_EVERY = 50       # flush sidecar every N steps
_WARMUP_STEPS = 5       # skip first N steps for JIT/compilation noise


def _compute_percentile(sorted_values, pct):
    # type: (List[float], float) -> float
    """Compute a percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _compute_timing_stats(step_times_ms):
    # type: (List[float]) -> Dict[str, float]
    """Compute p50, p90, mean, std from a list of step times (ms)."""
    if not step_times_ms:
        return {}
    sorted_vals = sorted(step_times_ms)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std = math.sqrt(variance)
    return {
        "p50": round(_compute_percentile(sorted_vals, 50), 2),
        "p90": round(_compute_percentile(sorted_vals, 90), 2),
        "mean": round(mean, 2),
        "std": round(std, 2),
    }


def _compute_phase_stats(phase_times_ms):
    # type: (List[float]) -> Dict[str, float]
    """Compute p50, p90 for a phase time list (ms)."""
    if not phase_times_ms:
        return {}
    sorted_vals = sorted(phase_times_ms)
    return {
        "p50": round(_compute_percentile(sorted_vals, 50), 2),
        "p90": round(_compute_percentile(sorted_vals, 90), 2),
    }


def _try_cuda_available():
    # type: () -> bool
    """Check if torch.cuda is available. Cached after first call."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _create_cuda_event():
    """Create a CUDA event for timing. Returns None if CUDA unavailable."""
    try:
        import torch
        return torch.cuda.Event(enable_timing=True)
    except Exception:
        return None


def _detect_distributed():
    # type: () -> tuple
    """Detect if running inside a torch.distributed process group."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return True, dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return False, 0, 1


def _detect_architecture(model, optimizer=None, training_args=None):
    # type: (Any, Any, Any) -> Dict[str, Any]
    """Extract architecture metadata from model/optimizer/training_args.

    Returns dict with: architecture_type, optimizer_type, fine_tuning_method,
    gradient_checkpointing, model_type, attention_type. All values Optional.
    Safe to call on any model — never raises.
    """
    info = {}  # type: Dict[str, Any]

    if model is None:
        return info

    try:
        # --- Model type / architecture ---
        config = getattr(model, "config", None)
        if config is not None:
            model_type = getattr(config, "model_type", None)
            if model_type:
                info["model_type"] = model_type
            # Architecture classification
            arch_classes = {
                "transformer": {"llama", "gpt2", "gpt_neo", "gpt_neox", "gptj",
                                "mistral", "qwen2", "phi", "gemma", "falcon",
                                "bert", "roberta", "t5", "bart", "mbart",
                                "whisper", "wav2vec2", "vit", "deit", "beit",
                                "swin", "clip", "dinov2"},
                "moe": {"mixtral", "switch_transformers"},
                "diffusion": {"unet_2d_condition"},
            }
            for arch, types in arch_classes.items():
                if model_type and model_type.lower() in types:
                    info["architecture_type"] = arch
                    break
            if "architecture_type" not in info:
                # Heuristic: if model has attention layers, it's likely a transformer
                if hasattr(config, "num_attention_heads") or hasattr(config, "num_heads"):
                    info["architecture_type"] = "transformer"

        # --- Gradient checkpointing ---
        gc = getattr(model, "is_gradient_checkpointing", None)
        if gc is not None:
            info["gradient_checkpointing"] = bool(gc)
        elif config is not None and getattr(config, "gradient_checkpointing", None):
            info["gradient_checkpointing"] = True
        # HF TrainingArguments
        if training_args is not None and getattr(training_args, "gradient_checkpointing", None):
            info["gradient_checkpointing"] = True

        # --- Fine-tuning method ---
        # Detect PEFT/LoRA by checking for common PEFT wrapper attributes
        if hasattr(model, "peft_config"):
            peft_cfg = model.peft_config
            if isinstance(peft_cfg, dict):
                # peft_config is a dict of adapter_name -> config
                for adapter_cfg in peft_cfg.values():
                    peft_type = getattr(adapter_cfg, "peft_type", None)
                    if peft_type:
                        peft_type_str = str(peft_type).lower()
                        if "lora" in peft_type_str:
                            info["fine_tuning_method"] = "lora"
                        elif "ia3" in peft_type_str:
                            info["fine_tuning_method"] = "ia3"
                        elif "prefix" in peft_type_str:
                            info["fine_tuning_method"] = "prefix_tuning"
                        else:
                            info["fine_tuning_method"] = "peft_generic"
                    break
            elif hasattr(peft_cfg, "peft_type"):
                info["fine_tuning_method"] = str(peft_cfg.peft_type).lower()

        # Detect frozen param ratio as sign of fine-tuning
        if "fine_tuning_method" not in info:
            try:
                total = 0
                trainable = 0
                for p in model.parameters():
                    total += p.numel()
                    if p.requires_grad:
                        trainable += p.numel()
                if total > 0 and trainable > 0:
                    ratio = trainable / total
                    info["param_count"] = total
                    info["trainable_param_count"] = trainable
                    if ratio < 0.1:
                        info["fine_tuning_method"] = "adapter"  # generic adapter/frozen
            except Exception:
                pass

        # --- Attention type ---
        try:
            attn_impl = getattr(config, "_attn_implementation", None) if config else None
            if attn_impl:
                info["attention_type"] = str(attn_impl)
            elif config and getattr(config, "attn_implementation", None):
                info["attention_type"] = str(config.attn_implementation)
        except Exception:
            pass

    except Exception:
        pass

    # --- Optimizer type ---
    if optimizer is not None:
        try:
            opt_class = type(optimizer).__name__
            info["optimizer_type"] = opt_class
        except Exception:
            pass

    return info


def _write_callback_data(data):
    # type: (Dict[str, Any]) -> None
    """Write callback data to the alloc sidecar file."""
    path = os.path.join(os.getcwd(), ".alloc_callback.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _build_sidecar(
    framework,       # type: str
    step_count,      # type: int
    step_times_ms,   # type: List[float]
    batch_size,      # type: Optional[int]
    is_distributed=False,  # type: bool
    rank=0,          # type: int
    world_size=1,    # type: int
    phase_stats=None,  # type: Optional[Dict[str, Dict[str, float]]]
    architecture_info=None,  # type: Optional[Dict[str, Any]]
):
    # type: (...) -> Dict[str, Any]
    """Build the sidecar dict from collected timing data.

    Args:
        phase_stats: Optional dict with keys "forward", "backward", "optimizer",
                     "dataloader", each containing {"p50": float, "p90": float}.
        architecture_info: Optional dict from _detect_architecture() with
                          architecture_type, optimizer_type, etc.
    """
    stats = _compute_timing_stats(step_times_ms)

    samples_per_sec = None  # type: Optional[float]
    p50 = stats.get("p50")
    if p50 and p50 > 0 and batch_size and batch_size > 0:
        samples_per_sec = round(batch_size / (p50 / 1000.0), 2)

    data = {
        "framework": framework,
        "step_count": step_count,
        "step_time_ms_p50": stats.get("p50"),
        "step_time_ms_p90": stats.get("p90"),
        "step_time_ms_mean": stats.get("mean"),
        "step_time_ms_std": stats.get("std"),
        "samples_per_sec": samples_per_sec,
        "batch_size": batch_size,
    }

    if is_distributed:
        data["is_distributed"] = True
        data["rank"] = rank
        data["world_size"] = world_size

    # Phase timing from CUDA events
    if phase_stats:
        # has_phase_timing = True only when we have CUDA-event-based phases
        # (forward/backward), not just wall-clock dataloader timing
        has_cuda_phases = bool(phase_stats.get("forward") or phase_stats.get("backward"))
        data["has_phase_timing"] = has_cuda_phases
        for phase_name in ("forward", "backward", "optimizer", "dataloader"):
            ps = phase_stats.get(phase_name, {})
            if ps:
                data["phase_{}_ms_p50".format(phase_name)] = ps.get("p50")
                data["phase_{}_ms_p90".format(phase_name)] = ps.get("p90")

    # Communication overhead estimate (distributed + phase timing)
    # In distributed training, backward phase includes gradient all-reduce.
    # Overhead ≈ (backward - forward) / backward × 100.
    if is_distributed and phase_stats:
        fwd = (phase_stats.get("forward") or {}).get("p50")
        bwd = (phase_stats.get("backward") or {}).get("p50")
        if fwd is not None and bwd is not None and bwd > fwd > 0:
            data["comm_overhead_pct"] = round((bwd - fwd) / bwd * 100, 1)

    # Architecture metadata from model/optimizer introspection
    if architecture_info:
        for key in ("architecture_type", "optimizer_type", "fine_tuning_method",
                     "gradient_checkpointing", "model_type", "attention_type",
                     "param_count", "trainable_param_count"):
            val = architecture_info.get(key)
            if val is not None:
                data[key] = val

    return data


# ── NVML Monitor ────────────────────────────────────────────────────────

_NVML_POLL_INTERVAL_S = 2.0


def _try_import_pynvml():
    """Try to import pynvml. Returns module or None."""
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


class _NvmlMonitor:
    """Background NVML sampling thread for callbacks.

    Monitors ALL visible GPUs (respects CUDA_VISIBLE_DEVICES via NVML).
    Produces per-GPU peak VRAM for straggler detection.
    Thread-safe: uses a lock for _samples access.
    Gracefully no-ops if pynvml is not installed or NVML init fails.
    """

    def __init__(self):
        # type: () -> None
        self._pynvml = _try_import_pynvml()
        self._thread = None  # type: Optional[threading.Thread]
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples = []  # type: List[Dict[str, Any]]
        self._hw_context = {}  # type: Dict[str, Any]
        self._start_time = 0.0
        self._stop_time = None  # type: Optional[float]
        self._gpu_count = 0  # actual GPUs being monitored
        self._gpu_handles = []  # type: List[Any]
        self._per_gpu_peak_vram_mb = []  # type: List[float]
        self._stopped = False  # type: bool

    @property
    def available(self):
        # type: () -> bool
        return self._pynvml is not None

    def start(self):
        # type: () -> None
        """Init NVML, capture hardware context, spawn sampling daemon."""
        if self._pynvml is None:
            return
        if self._stopped:
            return  # already ran and stopped — no restart
        try:
            self._pynvml.nvmlInit()
        except Exception:
            self._pynvml = None
            return

        self._start_time = time.time()

        # Discover all GPUs
        try:
            self._gpu_count = self._pynvml.nvmlDeviceGetCount()
        except Exception:
            self._gpu_count = 1

        # Get handles for all GPUs
        self._gpu_handles = []
        for i in range(self._gpu_count):
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                self._gpu_handles.append(handle)
            except Exception:
                pass

        if not self._gpu_handles:
            return

        self._per_gpu_peak_vram_mb = [0.0] * len(self._gpu_handles)

        # Capture hardware context from GPU 0
        try:
            handle = self._gpu_handles[0]
            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(mem_info.total / (1024 * 1024), 1)
            self._hw_context["gpu_name"] = name
            self._hw_context["gpu_total_vram_mb"] = total_mb
        except Exception:
            pass

        try:
            drv = self._pynvml.nvmlSystemGetDriverVersion()
            if isinstance(drv, bytes):
                drv = drv.decode("utf-8")
            self._hw_context["driver_version"] = drv
        except Exception:
            pass

        try:
            cuda_ver_int = self._pynvml.nvmlSystemGetCudaDriverVersion()
            major = cuda_ver_int // 1000
            minor = (cuda_ver_int % 1000) // 10
            self._hw_context["cuda_version"] = "{}.{}".format(major, minor)
        except Exception:
            pass

        try:
            handle = self._gpu_handles[0]
            sm_major, sm_minor = self._pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            self._hw_context["sm_version"] = "{}.{}".format(sm_major, sm_minor)
        except Exception:
            pass

        self._hw_context["num_gpus_detected"] = self._gpu_count

        # Detect NVLink interconnect and topology (NVSwitch vs point-to-point)
        try:
            handle = self._gpu_handles[0]
            active_links = 0
            is_nvswitch = False
            for link in range(18):  # max NVLink connections
                try:
                    state = self._pynvml.nvmlDeviceGetNvLinkState(handle, link)
                    if state:
                        active_links += 1
                        # Try to detect if remote end is NVSwitch vs GPU
                        try:
                            remote_info = self._pynvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(handle, link)
                            # NVSwitch has device_id 0x22a3 (Ampere) or 0x26b1 (Hopper)
                            # More reliably: NVSwitch full-mesh has 6+ active links per GPU
                            # while point-to-point typically has 1-2 per peer
                        except Exception:
                            pass
                except Exception:
                    break

            if active_links > 0:
                # Heuristic: NVSwitch-based systems (DGX, HGX) have 6+ active
                # NVLink connections per GPU to the switch fabric. Point-to-point
                # NVLink (e.g., workstation GPUs) typically has 1-4 links per peer.
                # With 4+ GPUs and 6+ links, it's almost certainly NVSwitch.
                # [HEURISTIC] Threshold: 6 links. DGX A100 has 12, DGX H100 has 18.
                # Point-to-point 2-GPU: 2 links. 4-GPU ring: 2-4 links.
                if active_links >= 6 and self._gpu_count >= 4:
                    is_nvswitch = True

                if is_nvswitch:
                    self._hw_context["interconnect_type"] = "nvlink_switch"
                else:
                    self._hw_context["interconnect_type"] = "nvlink_p2p"
            else:
                self._hw_context["interconnect_type"] = "pcie"

            self._hw_context["nvlink_active_links"] = active_links
        except Exception:
            pass

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        # type: () -> None
        """Poll GPU metrics for ALL GPUs until stop event is set."""
        pynvml = self._pynvml
        if pynvml is None or not self._gpu_handles:
            return
        try:
            while not self._stop_event.is_set():
                t = round(time.time() - self._start_time, 2)
                total_vram = 0.0
                total_util = 0.0
                total_power = 0.0
                gpu_count = len(self._gpu_handles)
                per_gpu_vram = []  # type: List[float]

                for idx, handle in enumerate(self._gpu_handles):
                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_mb = round(mem.used / (1024 * 1024), 1)
                        total_vram += vram_mb
                        per_gpu_vram.append(vram_mb)
                    except Exception:
                        vram_mb = 0.0
                        per_gpu_vram.append(0.0)

                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_util += float(util.gpu)
                    except Exception:
                        pass

                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        total_power += power
                    except Exception:
                        pass

                sample = {
                    "t": t,
                    "vram_mb": round(total_vram / gpu_count, 1) if gpu_count else 0.0,
                    "gpu_util_pct": round(total_util / gpu_count, 1) if gpu_count else 0.0,
                    "power_w": round(total_power / gpu_count, 1) if gpu_count else 0.0,
                }
                with self._lock:
                    self._samples.append(sample)
                    # Track per-GPU peak VRAM under lock (read by get_results)
                    for idx, v in enumerate(per_gpu_vram):
                        if idx < len(self._per_gpu_peak_vram_mb) and v > self._per_gpu_peak_vram_mb[idx]:
                            self._per_gpu_peak_vram_mb[idx] = v

                self._stop_event.wait(_NVML_POLL_INTERVAL_S)
        except Exception:
            pass

    def stop(self):
        # type: () -> None
        """Stop sampling thread and shutdown NVML."""
        if self._stopped:
            return
        self._stopped = True
        self._stop_time = time.time()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_results(self):
        # type: () -> tuple
        """Return (hw_context_dict, probe_dict) matching write_report format.

        If NVML was unavailable, returns ({}, {}).
        """
        with self._lock:
            samples_copy = list(self._samples)
            peak_vram_copy = list(self._per_gpu_peak_vram_mb)

        if not self._hw_context and not samples_copy:
            return {}, {}

        if self._stop_time and self._start_time:
            duration = round(self._stop_time - self._start_time, 2)
        elif self._start_time:
            duration = round(time.time() - self._start_time, 2)
        else:
            duration = 0.0

        # Use per-GPU peak tracking for peak_vram_mb (max across all GPUs),
        # matching probe.py convention. The per-sample vram_mb is averaged
        # across GPUs, so max(sample.vram_mb) would underreport on skewed
        # multi-GPU workloads (e.g., PP where rank 0 has the embedding layer).
        peak_vram = 0.0
        if peak_vram_copy and any(v > 0 for v in peak_vram_copy):
            peak_vram = max(peak_vram_copy)
        else:
            for s in samples_copy:
                vram = s.get("vram_mb", 0.0)
                if vram > peak_vram:
                    peak_vram = vram

        total_util = 0.0
        total_power = 0.0
        for s in samples_copy:
            total_util += s.get("gpu_util_pct", 0.0)
            total_power += s.get("power_w", 0.0)

        n = len(samples_copy)
        avg_util = round(total_util / n, 1) if n else 0.0
        avg_power = round(total_power / n, 1) if n else 0.0

        probe_dict = {
            "peak_vram_mb": round(peak_vram, 1),
            "avg_gpu_util": avg_util,
            "avg_power_watts": avg_power,
            "duration_seconds": duration,
            "samples": samples_copy,
            "gpu_name": self._hw_context.get("gpu_name"),
            "gpu_total_vram_mb": self._hw_context.get("gpu_total_vram_mb"),
            "num_gpus_detected": self._hw_context.get("num_gpus_detected", 1),
            "probe_mode": "callback",
        }

        # Per-GPU peak VRAM — critical for straggler detection
        if peak_vram_copy and any(v > 0 for v in peak_vram_copy):
            probe_dict["per_rank_peak_vram_mb"] = [
                round(v, 1) for v in peak_vram_copy
            ]

        # Interconnect type if detected
        if self._hw_context.get("interconnect_type"):
            probe_dict["interconnect_type"] = self._hw_context["interconnect_type"]

        hw_context = dict(self._hw_context)
        return hw_context, probe_dict


# ── Artifact helper ─────────────────────────────────────────────────────

def _write_full_artifact(monitor, sidecar_data, step_times_raw=None):
    # type: (_NvmlMonitor, Dict[str, Any], Optional[List[float]]) -> None
    """Merge monitor GPU data + callback timing into a full artifact.

    Writes alloc_artifact.json.gz (or rank-tagged variant) via
    artifact_writer.write_report(). Silent no-op on any failure.

    For distributed training, non-rank-0 processes write
    alloc_artifact_rank{N}.json.gz instead of the default name.
    """
    try:
        hw_context, probe_dict = monitor.get_results()

        # Use probe_dict if it has data, otherwise create from sidecar timing
        if probe_dict is None:
            probe_dict = {}

        # Always merge step timing from sidecar into probe_dict
        merge_keys = [
            "step_time_ms_p50", "step_time_ms_p90", "step_time_ms_mean",
            "step_time_ms_std", "samples_per_sec", "step_count", "batch_size",
            "framework", "has_phase_timing",
            "phase_forward_ms_p50", "phase_forward_ms_p90",
            "phase_backward_ms_p50", "phase_backward_ms_p90",
            "phase_optimizer_ms_p50", "phase_optimizer_ms_p90",
            "phase_dataloader_ms_p50", "phase_dataloader_ms_p90",
            # Architecture metadata (P3-A)
            "architecture_type", "optimizer_type", "fine_tuning_method",
            "gradient_checkpointing", "model_type", "attention_type",
            "param_count", "trainable_param_count",
            # Distributed signals (P3.5-C)
            "comm_overhead_pct",
        ]
        for key in merge_keys:
            val = sidecar_data.get(key)
            if val is not None:
                probe_dict[key] = val

        # Set source to "callback" for standalone callback artifacts
        probe_dict["source"] = "callback"

        # Include raw step times for per-rank merge/straggler detection
        if step_times_raw:
            probe_dict["step_times_raw"] = step_times_raw[-_ROLLING_WINDOW:]

        # Determine output filename based on rank
        output_path = None
        if sidecar_data.get("is_distributed"):
            probe_dict["is_distributed"] = True
            rank = sidecar_data.get("rank", 0)
            probe_dict["rank"] = rank
            probe_dict["world_size"] = sidecar_data.get("world_size", 1)
            if rank > 0:
                output_path = "alloc_artifact_rank{}.json.gz".format(rank)

        from alloc.artifact_writer import write_report
        write_report(
            probe_result=probe_dict if probe_dict else None,
            hardware_context=hw_context if hw_context else None,
            output_path=output_path,
        )
    except Exception:
        pass


# ── HuggingFace Callback ─────────────────────────────────────────────────

try:
    from transformers import TrainerCallback

    class AllocCallback(TrainerCallback):
        """HuggingFace Trainer callback — GPU metrics + step timing + phase decomposition.

        Self-contained: spawns an NVML monitor thread for GPU metrics and
        writes a full alloc_artifact.json.gz on train_end. No ``alloc run``
        wrapper required.

        Phase timing (P1.5): when CUDA is available, records CUDA events at
        framework hook boundaries to decompose step time into forward, backward,
        optimizer, and dataloader phases.
        """

        def __init__(self):
            # type: () -> None
            self.step_count = 0          # type: int
            self._step_times_ms = []     # type: List[float]
            self._step_start = None      # type: Optional[float]
            self._batch_size = None      # type: Optional[int]
            self._last_write_step = 0    # type: int
            self._dist_checked = False   # type: bool
            self._is_distributed = False  # type: bool
            self._rank = 0               # type: int
            self._world_size = 1         # type: int
            self._monitor = _NvmlMonitor()
            self._monitor_started = False  # type: bool
            # Phase timing (CUDA events)
            self._cuda_available = None  # type: Optional[bool]
            self._evt_step_start = None  # type: Any
            self._evt_backward_start = None  # type: Any
            self._evt_optimizer_start = None  # type: Any
            self._evt_step_end = None  # type: Any
            self._phase_forward_ms = []   # type: List[float]
            self._phase_backward_ms = []  # type: List[float]
            self._phase_optimizer_ms = []  # type: List[float]
            self._phase_dataloader_ms = []  # type: List[float]
            self._prev_step_end_wall = None  # type: Optional[float]
            # Architecture detection (P3-A)
            self._arch_info = None  # type: Optional[Dict[str, Any]]
            self._arch_detected = False  # type: bool

        def _detect_arch_once(self, **kwargs):
            # type: (...) -> None
            """Detect architecture info once from model/optimizer/args."""
            if self._arch_detected:
                return
            self._arch_detected = True
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            args = kwargs.get("args")  # HF TrainingArguments
            if model is not None:
                self._arch_info = _detect_architecture(model, optimizer=optimizer, training_args=args)

        def _init_cuda_events(self):
            # type: () -> None
            """Lazily initialize CUDA events on first use."""
            if self._cuda_available is not None:
                return
            self._cuda_available = _try_cuda_available()
            if self._cuda_available:
                self._evt_step_start = _create_cuda_event()
                self._evt_backward_start = _create_cuda_event()
                self._evt_optimizer_start = _create_cuda_event()
                self._evt_step_end = _create_cuda_event()
                if not all([self._evt_step_start, self._evt_backward_start,
                           self._evt_optimizer_start, self._evt_step_end]):
                    self._cuda_available = False

        def on_step_begin(self, args, state, control, **kwargs):
            # Architecture detection (once, on first step when model is available)
            self._detect_arch_once(**kwargs)

            # Dataloader time = wall gap between prev step end and this step begin
            now = time.monotonic()
            if self._prev_step_end_wall is not None:
                dl_ms = (now - self._prev_step_end_wall) * 1000.0
                self._phase_dataloader_ms.append(dl_ms)
                if len(self._phase_dataloader_ms) > _ROLLING_WINDOW:
                    self._phase_dataloader_ms = self._phase_dataloader_ms[-_ROLLING_WINDOW:]

            self._step_start = now
            if not self._monitor_started:
                self._monitor.start()
                self._monitor_started = True
            if not self._dist_checked:
                self._is_distributed, self._rank, self._world_size = _detect_distributed()
                self._dist_checked = True

            # Record CUDA event at step start (after dataloader)
            self._init_cuda_events()
            if self._cuda_available and self._evt_step_start is not None:
                try:
                    self._evt_step_start.record()
                except Exception:
                    pass

        def on_backward(self, args, state, control, **kwargs):
            """Called at the start of backward pass — marks end of forward."""
            if self._cuda_available and self._evt_backward_start is not None:
                try:
                    self._evt_backward_start.record()
                except Exception:
                    pass

        def on_before_optimizer_step(self, args, state, control, **kwargs):
            """Called before optimizer.step() — marks end of backward."""
            if self._cuda_available and self._evt_optimizer_start is not None:
                try:
                    self._evt_optimizer_start.record()
                except Exception:
                    pass

        def on_step_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step

            # Record CUDA event at step end
            if self._cuda_available and self._evt_step_end is not None:
                try:
                    self._evt_step_end.record()
                except Exception:
                    pass

            if self._step_start is not None:
                elapsed_ms = (time.monotonic() - self._step_start) * 1000.0
                self._step_times_ms.append(elapsed_ms)
                if len(self._step_times_ms) > _ROLLING_WINDOW:
                    self._step_times_ms = self._step_times_ms[-_ROLLING_WINDOW:]
                self._step_start = None

            self._prev_step_end_wall = time.monotonic()

            if self._batch_size is None:
                try:
                    bs = args.per_device_train_batch_size
                    ga = getattr(args, "gradient_accumulation_steps", 1) or 1
                    self._batch_size = bs * ga
                except Exception:
                    self._batch_size = None

            # Collect phase durations from CUDA events (non-blocking query)
            if (self._cuda_available and self.step_count > _WARMUP_STEPS
                    and self.step_count % _WRITE_EVERY == 0):
                self._sync_phase_events()

            if self.step_count - self._last_write_step >= _WRITE_EVERY:
                self._flush()
                self._last_write_step = self.step_count

        def _sync_phase_events(self, force=False):
            # type: (bool) -> None
            """Collect phase durations from CUDA events.

            Uses non-blocking query() to avoid stalling the training loop.
            After 50 steps of GPU work, events are nearly always complete.
            On train_end, force=True uses synchronize() to flush final data.
            """
            try:
                if not all([self._evt_step_start, self._evt_backward_start,
                           self._evt_optimizer_start, self._evt_step_end]):
                    return
                if force:
                    self._evt_step_end.synchronize()
                elif not self._evt_step_end.query():
                    return  # Events not yet complete — skip, no stall
                fwd = self._evt_step_start.elapsed_time(self._evt_backward_start)
                bwd = self._evt_backward_start.elapsed_time(self._evt_optimizer_start)
                opt = self._evt_optimizer_start.elapsed_time(self._evt_step_end)
                if fwd > 0:
                    self._phase_forward_ms.append(fwd)
                    if len(self._phase_forward_ms) > _ROLLING_WINDOW:
                        self._phase_forward_ms = self._phase_forward_ms[-_ROLLING_WINDOW:]
                if bwd > 0:
                    self._phase_backward_ms.append(bwd)
                    if len(self._phase_backward_ms) > _ROLLING_WINDOW:
                        self._phase_backward_ms = self._phase_backward_ms[-_ROLLING_WINDOW:]
                if opt > 0:
                    self._phase_optimizer_ms.append(opt)
                    if len(self._phase_optimizer_ms) > _ROLLING_WINDOW:
                        self._phase_optimizer_ms = self._phase_optimizer_ms[-_ROLLING_WINDOW:]
            except Exception:
                pass

        def on_train_end(self, args, state, control, **kwargs):
            self.step_count = state.global_step
            if self._cuda_available:
                self._sync_phase_events(force=True)
            self._monitor.stop()
            sidecar = self._flush()
            _write_full_artifact(self._monitor, sidecar, step_times_raw=self._step_times_ms)

        def _get_phase_stats(self):
            # type: () -> Optional[Dict[str, Dict[str, float]]]
            """Compute phase timing stats from rolling windows."""
            if not any([self._phase_forward_ms, self._phase_backward_ms,
                       self._phase_optimizer_ms, self._phase_dataloader_ms]):
                return None
            stats = {}
            if self._phase_forward_ms:
                stats["forward"] = _compute_phase_stats(self._phase_forward_ms)
            if self._phase_backward_ms:
                stats["backward"] = _compute_phase_stats(self._phase_backward_ms)
            if self._phase_optimizer_ms:
                stats["optimizer"] = _compute_phase_stats(self._phase_optimizer_ms)
            if self._phase_dataloader_ms:
                stats["dataloader"] = _compute_phase_stats(self._phase_dataloader_ms)
            return stats if stats else None

        def _flush(self):
            # type: () -> Dict[str, Any]
            data = _build_sidecar(
                framework="huggingface",
                step_count=self.step_count,
                step_times_ms=self._step_times_ms,
                batch_size=self._batch_size,
                is_distributed=self._is_distributed,
                rank=self._rank,
                world_size=self._world_size,
                phase_stats=self._get_phase_stats(),
                architecture_info=self._arch_info,
            )
            _write_callback_data(data)
            return data

except ImportError:
    class AllocCallback:  # type: ignore[no-redef]
        """Stub — install transformers to use AllocCallback."""

        def __init__(self):
            raise ImportError(
                "AllocCallback requires the `transformers` package. "
                "Install with: pip install transformers"
            )


# ── PyTorch Lightning Callback ───────────────────────────────────────────

try:
    from lightning.pytorch.callbacks import Callback as LightningBaseCallback

    class AllocLightningCallback(LightningBaseCallback):
        """PyTorch Lightning callback — GPU metrics + step timing + phase decomposition.

        Self-contained: spawns an NVML monitor thread for GPU metrics and
        writes a full alloc_artifact.json.gz on train_end. No ``alloc run``
        wrapper required.

        Phase timing (P1.5): when CUDA is available, records CUDA events at
        framework hook boundaries to decompose step time into forward, backward,
        optimizer, and dataloader phases.
        """

        def __init__(self):
            # type: () -> None
            super().__init__()
            self.step_count = 0          # type: int
            self._step_times_ms = []     # type: List[float]
            self._step_start = None      # type: Optional[float]
            self._batch_size = None      # type: Optional[int]
            self._last_write_step = 0    # type: int
            self._dist_checked = False   # type: bool
            self._is_distributed = False  # type: bool
            self._rank = 0               # type: int
            self._world_size = 1         # type: int
            self._monitor = _NvmlMonitor()
            self._monitor_started = False  # type: bool
            # Phase timing (CUDA events)
            self._cuda_available = None  # type: Optional[bool]
            self._evt_step_start = None  # type: Any
            self._evt_backward_start = None  # type: Any
            self._evt_optimizer_start = None  # type: Any
            self._evt_step_end = None  # type: Any
            self._phase_forward_ms = []   # type: List[float]
            self._phase_backward_ms = []  # type: List[float]
            self._phase_optimizer_ms = []  # type: List[float]
            self._phase_dataloader_ms = []  # type: List[float]
            self._prev_step_end_wall = None  # type: Optional[float]
            # Architecture detection (P3-A)
            self._arch_info = None  # type: Optional[Dict[str, Any]]
            self._arch_detected = False  # type: bool

        def _init_cuda_events(self):
            # type: () -> None
            """Lazily initialize CUDA events on first use."""
            if self._cuda_available is not None:
                return
            self._cuda_available = _try_cuda_available()
            if self._cuda_available:
                self._evt_step_start = _create_cuda_event()
                self._evt_backward_start = _create_cuda_event()
                self._evt_optimizer_start = _create_cuda_event()
                self._evt_step_end = _create_cuda_event()
                if not all([self._evt_step_start, self._evt_backward_start,
                           self._evt_optimizer_start, self._evt_step_end]):
                    self._cuda_available = False

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # Architecture detection (once, on first step)
            if not self._arch_detected:
                self._arch_detected = True
                optimizer = None
                if hasattr(trainer, "optimizers") and trainer.optimizers:
                    optimizer = trainer.optimizers[0]
                self._arch_info = _detect_architecture(pl_module, optimizer=optimizer)

            # Dataloader time = wall gap between prev step end and this step begin
            now = time.monotonic()
            if self._prev_step_end_wall is not None:
                dl_ms = (now - self._prev_step_end_wall) * 1000.0
                self._phase_dataloader_ms.append(dl_ms)
                if len(self._phase_dataloader_ms) > _ROLLING_WINDOW:
                    self._phase_dataloader_ms = self._phase_dataloader_ms[-_ROLLING_WINDOW:]

            self._step_start = now
            if not self._monitor_started:
                self._monitor.start()
                self._monitor_started = True
            if not self._dist_checked:
                self._is_distributed, self._rank, self._world_size = _detect_distributed()
                self._dist_checked = True

            # Record CUDA event at step start (after dataloader)
            self._init_cuda_events()
            if self._cuda_available and self._evt_step_start is not None:
                try:
                    self._evt_step_start.record()
                except Exception:
                    pass

        def on_before_backward(self, trainer, pl_module, loss):
            """Called before backward pass — marks end of forward."""
            if self._cuda_available and self._evt_backward_start is not None:
                try:
                    self._evt_backward_start.record()
                except Exception:
                    pass

        def on_before_optimizer_step(self, trainer, optimizer, optimizer_idx=0):
            """Called before optimizer.step() — marks end of backward."""
            if self._cuda_available and self._evt_optimizer_start is not None:
                try:
                    self._evt_optimizer_start.record()
                except Exception:
                    pass

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.step_count = trainer.global_step

            # Record CUDA event at step end
            if self._cuda_available and self._evt_step_end is not None:
                try:
                    self._evt_step_end.record()
                except Exception:
                    pass

            if self._step_start is not None:
                elapsed_ms = (time.monotonic() - self._step_start) * 1000.0
                self._step_times_ms.append(elapsed_ms)
                if len(self._step_times_ms) > _ROLLING_WINDOW:
                    self._step_times_ms = self._step_times_ms[-_ROLLING_WINDOW:]
                self._step_start = None

            self._prev_step_end_wall = time.monotonic()

            if self._batch_size is None:
                try:
                    # batch can be a tensor, dict, list, or tuple
                    if hasattr(batch, "shape"):
                        # Tensor — first dim is batch size
                        self._batch_size = batch.shape[0]
                    elif isinstance(batch, dict):
                        # Dict of tensors — get batch size from first value
                        first_val = next(iter(batch.values()), None)
                        if first_val is not None and hasattr(first_val, "shape"):
                            self._batch_size = first_val.shape[0]
                    elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                        # List/tuple of tensors — get batch size from first element
                        first_elem = batch[0]
                        if hasattr(first_elem, "shape"):
                            self._batch_size = first_elem.shape[0]
                    if self._batch_size is None and hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                        dm = trainer.datamodule
                        if hasattr(dm, "batch_size"):
                            self._batch_size = dm.batch_size
                except Exception:
                    self._batch_size = None

            # Collect phase durations from CUDA events (non-blocking query)
            if (self._cuda_available and self.step_count > _WARMUP_STEPS
                    and self.step_count % _WRITE_EVERY == 0):
                self._sync_phase_events()

            if self.step_count - self._last_write_step >= _WRITE_EVERY:
                self._flush()
                self._last_write_step = self.step_count

        def _sync_phase_events(self, force=False):
            # type: (bool) -> None
            """Collect phase durations from CUDA events.

            Uses non-blocking query() to avoid stalling the training loop.
            After 50 steps of GPU work, events are nearly always complete.
            On train_end, force=True uses synchronize() to flush final data.
            """
            try:
                if not all([self._evt_step_start, self._evt_backward_start,
                           self._evt_optimizer_start, self._evt_step_end]):
                    return
                if force:
                    self._evt_step_end.synchronize()
                elif not self._evt_step_end.query():
                    return  # Events not yet complete — skip, no stall
                fwd = self._evt_step_start.elapsed_time(self._evt_backward_start)
                bwd = self._evt_backward_start.elapsed_time(self._evt_optimizer_start)
                opt = self._evt_optimizer_start.elapsed_time(self._evt_step_end)
                if fwd > 0:
                    self._phase_forward_ms.append(fwd)
                    if len(self._phase_forward_ms) > _ROLLING_WINDOW:
                        self._phase_forward_ms = self._phase_forward_ms[-_ROLLING_WINDOW:]
                if bwd > 0:
                    self._phase_backward_ms.append(bwd)
                    if len(self._phase_backward_ms) > _ROLLING_WINDOW:
                        self._phase_backward_ms = self._phase_backward_ms[-_ROLLING_WINDOW:]
                if opt > 0:
                    self._phase_optimizer_ms.append(opt)
                    if len(self._phase_optimizer_ms) > _ROLLING_WINDOW:
                        self._phase_optimizer_ms = self._phase_optimizer_ms[-_ROLLING_WINDOW:]
            except Exception:
                pass

        def on_train_end(self, trainer, pl_module):
            self.step_count = trainer.global_step
            if self._cuda_available:
                self._sync_phase_events(force=True)
            self._monitor.stop()
            sidecar = self._flush()
            _write_full_artifact(self._monitor, sidecar, step_times_raw=self._step_times_ms)

        def _get_phase_stats(self):
            # type: () -> Optional[Dict[str, Dict[str, float]]]
            """Compute phase timing stats from rolling windows."""
            if not any([self._phase_forward_ms, self._phase_backward_ms,
                       self._phase_optimizer_ms, self._phase_dataloader_ms]):
                return None
            stats = {}
            if self._phase_forward_ms:
                stats["forward"] = _compute_phase_stats(self._phase_forward_ms)
            if self._phase_backward_ms:
                stats["backward"] = _compute_phase_stats(self._phase_backward_ms)
            if self._phase_optimizer_ms:
                stats["optimizer"] = _compute_phase_stats(self._phase_optimizer_ms)
            if self._phase_dataloader_ms:
                stats["dataloader"] = _compute_phase_stats(self._phase_dataloader_ms)
            return stats if stats else None

        def _flush(self):
            # type: () -> Dict[str, Any]
            data = _build_sidecar(
                framework="lightning",
                step_count=self.step_count,
                step_times_ms=self._step_times_ms,
                batch_size=self._batch_size,
                is_distributed=self._is_distributed,
                rank=self._rank,
                world_size=self._world_size,
                phase_stats=self._get_phase_stats(),
                architecture_info=self._arch_info,
            )
            _write_callback_data(data)
            return data

except ImportError:
    class AllocLightningCallback:  # type: ignore[no-redef]
        """Stub — install lightning to use AllocLightningCallback."""

        def __init__(self):
            raise ImportError(
                "AllocLightningCallback requires the `lightning` package. "
                "Install with: pip install lightning"
            )
