"""Shared types for dataset build operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawSignalRecord:
    """A normalized waveform record used by the build pipeline."""

    dataset_name: str
    path: Path
    signal: np.ndarray
    sample_rate: float
    starttime: str
    arrival_time: str
    timestamp: float | None
    sample_type: str
    method: str | None = None
    arrival_source: str = "labeled"
    original_sample_rate: float | None = None
    original_npts: int | None = None
    data_info: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_name(self) -> str:
        return self.path.name

    @property
    def sample_stem(self) -> str:
        return self.path.stem

    @property
    def duration_s(self) -> float:
        return float(len(self.signal) / self.sample_rate) if self.sample_rate > 0 else 0.0

    @property
    def arrival_offset_s(self) -> float:
        from .windowing import arrival_offset_seconds

        return arrival_offset_seconds(self)


@dataclass(frozen=True)
class SplitPlan:
    """Split plan for one dataset family."""

    dataset_name: str
    train_ratio: float
    random_seed: int
    group_size: int | None = None


@dataclass(frozen=True)
class SplitAssignment:
    """One source record assigned to train or test."""

    source_path: Path
    dataset_name: str
    split: str
    group_id: str
    sample_rate_hz: float
    timestamp_text: str
    method: str | None = None


@dataclass(frozen=True)
class DatasetManifestRow:
    """Row written to a split manifest csv."""

    source_file_name: str
    source_file_path: str
    dataset_source: str
    group_id: str
    split: str
    sample_rate_hz: float
    timestamp_text: str
    method: str | None
    random_seed: int


@dataclass(frozen=True)
class MixedSampleSpec:
    """Parameters used to create one mixed broken-wire sample."""

    dataset_name: str
    split: str
    noise_dataset: str
    source_noise: RawSignalRecord
    source_bk: RawSignalRecord
    w1: float
    w2: float
    t_s: float
    suffix: str


@dataclass(frozen=True)
class BuildLogRow:
    """One row of the sample-level build log."""

    record_id: str
    dataset_name: str
    split: str
    sample_class: str
    output_file_name: str
    output_file_path: str
    source_noise_file_name: str
    source_noise_file_path: str
    source_bk_file_name: str
    source_bk_file_path: str
    source_noise_dataset: str
    source_bk_method: str | None
    source_noise_sample_rate_hz: float
    source_bk_sample_rate_hz: float
    target_sample_rate_hz: float
    source_noise_duration_s: float
    source_bk_duration_s: float
    output_duration_s: float
    arrival_offset_s: float
    window_start_offset_t_s: float
    window_start_absolute_offset_s: float
    window_start_index: int
    window_end_index: int
    w1: float
    w2: float
    noise_rms: float
    bk_rms: float
    mixed_rms: float
    resample_noise_action: str
    resample_bk_action: str
    random_seed: int
    build_time: str
    builder_version: str
    status: str
    error_message: str = ""
