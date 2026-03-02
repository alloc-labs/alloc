""".alloc.yaml config — GPU fleet, explore, budget, priority.

Searches for .alloc.yaml in cwd, then parents, then ~/.alloc/preferences.yaml.
Never crashes. Returns None or defaults on missing/invalid config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


CONFIG_FILENAME = ".alloc.yaml"
GLOBAL_PREFS_PATH = Path.home() / ".alloc" / "preferences.yaml"

_ALLOWED_OBJECTIVES = {
    "cheapest",
    "fastest",
    "fastest_within_budget",
    "best_value",
}

_ALLOWED_INTERCONNECTS = {
    "pcie",
    "nvlink",
    "nvlink_switch",
    "nvlink_p2p",
    "infiniband",
    "unknown",
}


@dataclass
class FleetEntry:
    """A GPU in the user's fleet or explore list."""

    gpu: str                        # GPU ID or alias (e.g. "H100", "nvidia-h100-sxm-80gb")
    cloud: Optional[str] = None     # "aws", "gcp", "azure", "lambda", etc.
    count: Optional[int] = None     # Max GPUs available
    rate: Optional[float] = None    # Custom $/hr override
    explore: bool = False           # True = "I don't have this but want to evaluate"


@dataclass
class AllocConfig:
    """Parsed .alloc.yaml content."""

    fleet: List[FleetEntry] = field(default_factory=list)
    explore: List[FleetEntry] = field(default_factory=list)
    objective: Optional[str] = None  # cheapest | fastest | fastest_within_budget | best_value
    priority_cost: int = 50         # 0-100, latency = 100 - cost
    budget_monthly: Optional[float] = None  # Monthly budget in USD
    budget_hourly: Optional[float] = None   # Hourly budget cap
    org_budget_monthly: Optional[float] = None  # Org ceiling (from --from-org sync)
    interconnect: Optional[str] = None  # pcie | nvlink | infiniband | unknown

    @property
    def priority_latency(self) -> int:
        return 100 - self.priority_cost

    @property
    def fleet_gpu_ids(self) -> List[str]:
        """GPU IDs from fleet entries."""
        return [e.gpu for e in self.fleet]

    @property
    def explore_gpu_ids(self) -> List[str]:
        """GPU IDs from explore entries."""
        return [e.gpu for e in self.explore]

    @property
    def all_gpu_ids(self) -> List[str]:
        """All GPU IDs (fleet + explore)."""
        return self.fleet_gpu_ids + self.explore_gpu_ids

    @property
    def rate_overrides(self) -> Dict[str, float]:
        """GPU ID → custom $/hr for entries with rate set."""
        overrides = {}
        for e in self.fleet + self.explore:
            if e.rate is not None:
                overrides[e.gpu] = e.rate
        return overrides

    def to_dict(self) -> dict:
        """Serialize to dict suitable for YAML output."""
        d = {}  # type: dict

        if self.objective is not None:
            d["objective"] = self.objective

        if self.fleet:
            d["fleet"] = [_entry_to_dict(e) for e in self.fleet]

        if self.explore:
            d["explore"] = [_entry_to_dict(e) for e in self.explore]

        d["priority"] = {
            "cost": self.priority_cost,
            "latency": self.priority_latency,
        }

        if self.budget_monthly is not None:
            d.setdefault("budget", {})["monthly_usd"] = self.budget_monthly
        if self.budget_hourly is not None:
            d.setdefault("budget", {})["hourly_usd"] = self.budget_hourly
        if self.org_budget_monthly is not None:
            d.setdefault("budget", {})["org_ceiling_usd"] = self.org_budget_monthly

        if self.interconnect is not None:
            d["interconnect"] = self.interconnect

        return d


def _entry_to_dict(e: FleetEntry) -> dict:
    """Serialize a FleetEntry, omitting None/default fields."""
    d = {"gpu": e.gpu}  # type: dict
    if e.cloud is not None:
        d["cloud"] = e.cloud
    if e.count is not None:
        d["count"] = e.count
    if e.rate is not None:
        d["rate"] = e.rate
    return d


def load_alloc_config(path: Optional[str] = None) -> Optional[AllocConfig]:
    """Load and parse .alloc.yaml.

    Search order:
      1. Explicit path (if provided)
      2. .alloc.yaml in cwd
      3. .alloc.yaml in parent directories (up to filesystem root)
      4. ~/.alloc/preferences.yaml

    Returns None if no config file found or parse fails.
    """
    config_path = _find_config(path)
    if config_path is None:
        return None

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    return _parse_config(raw)


def validate_config(config: AllocConfig) -> List[str]:
    """Validate an AllocConfig. Returns list of error strings (empty = valid)."""
    errors = []

    if config.objective is not None and config.objective not in _ALLOWED_OBJECTIVES:
        errors.append(
            f"objective must be one of {sorted(_ALLOWED_OBJECTIVES)}, got {config.objective}"
        )

    if config.interconnect is not None and config.interconnect not in _ALLOWED_INTERCONNECTS:
        errors.append(
            f"interconnect must be one of {sorted(_ALLOWED_INTERCONNECTS)}, got {config.interconnect}"
        )

    if config.priority_cost < 0 or config.priority_cost > 100:
        errors.append(f"priority.cost must be 0-100, got {config.priority_cost}")

    if config.budget_monthly is not None and config.budget_monthly < 0:
        errors.append(f"budget.monthly_usd must be >= 0, got {config.budget_monthly}")

    if config.budget_hourly is not None and config.budget_hourly < 0:
        errors.append(f"budget.hourly_usd must be >= 0, got {config.budget_hourly}")

    for entry in config.fleet + config.explore:
        if not entry.gpu:
            errors.append("Fleet/explore entry missing 'gpu' field")
        if entry.rate is not None and entry.rate < 0:
            errors.append(f"Rate for {entry.gpu} must be >= 0, got {entry.rate}")
        if entry.count is not None and entry.count < 1:
            errors.append(f"Count for {entry.gpu} must be >= 1, got {entry.count}")

    return errors


def write_alloc_config(config: AllocConfig, path: Optional[str] = None) -> str:
    """Write config to YAML file. Returns the path written to."""
    out_path = path or os.path.join(os.getcwd(), CONFIG_FILENAME)

    data = config.to_dict()

    with open(out_path, "w") as f:
        f.write("# Alloc GPU configuration\n")
        f.write("# Docs: https://alloclabs.com/docs/right-sizing\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return out_path


def _find_config(explicit_path: Optional[str] = None) -> Optional[str]:
    """Find .alloc.yaml by searching cwd → parents → global prefs."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        return None

    # Walk from cwd upward
    current = Path.cwd()
    for _ in range(50):  # Safety limit
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return str(candidate)
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Global preferences
    if GLOBAL_PREFS_PATH.is_file():
        return str(GLOBAL_PREFS_PATH)

    return None


def _parse_config(raw: dict) -> AllocConfig:
    """Parse raw YAML dict into AllocConfig."""
    objective = raw.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        objective = None

    fleet = []
    for item in raw.get("fleet", []):
        if isinstance(item, str):
            fleet.append(FleetEntry(gpu=item))
        elif isinstance(item, dict):
            fleet.append(FleetEntry(
                gpu=item.get("gpu", ""),
                cloud=item.get("cloud"),
                count=item.get("count"),
                rate=item.get("rate"),
                explore=False,
            ))

    explore = []
    for item in raw.get("explore", []):
        if isinstance(item, str):
            explore.append(FleetEntry(gpu=item, explore=True))
        elif isinstance(item, dict):
            explore.append(FleetEntry(
                gpu=item.get("gpu", ""),
                cloud=item.get("cloud"),
                count=item.get("count"),
                rate=item.get("rate"),
                explore=True,
            ))

    priority = raw.get("priority", {})
    priority_cost = priority.get("cost", 50) if isinstance(priority, dict) else 50

    budget = raw.get("budget", {})
    budget_monthly = budget.get("monthly_usd") if isinstance(budget, dict) else None
    budget_hourly = budget.get("hourly_usd") if isinstance(budget, dict) else None
    org_budget_monthly = budget.get("org_ceiling_usd") if isinstance(budget, dict) else None

    interconnect = raw.get("interconnect")
    if isinstance(interconnect, str) and interconnect.strip():
        interconnect = interconnect.strip().lower()
        if interconnect not in _ALLOWED_INTERCONNECTS:
            interconnect = None
    else:
        interconnect = None

    return AllocConfig(
        fleet=fleet,
        explore=explore,
        objective=objective,
        priority_cost=priority_cost,
        budget_monthly=budget_monthly,
        budget_hourly=budget_hourly,
        org_budget_monthly=org_budget_monthly,
        interconnect=interconnect,
    )
