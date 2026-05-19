"""Central configuration for dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    """Top-level build parameters."""

    data_root: Path = Path("data")
    output_root: Path = Path("data") / "dataset_build_260505"
    target_sample_rate_hz: float = 500_000.0
    train_ratio: float = 0.7
    random_seed: int = 260505
    window_duration_s: float = 0.02
    t_min_s: float = -0.0080
    t_max_s: float = 0.0050
    t_step_s: float = 0.0001
    w_min: float = 0.5
    w_max: float = 1.5
    flow_group_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "F130": 10,
            "F130A": 20,
            "F130C": 10,
        }
    )
    noise_datasets: tuple[str, ...] = ("F130", "F130A", "F130C")


DEFAULT_BUILD_CONFIG = BuildConfig()
