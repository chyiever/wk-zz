"""Write NPZ samples and related files."""

from __future__ import annotations

import random
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .types import BuildLogRow, MixedSampleSpec, RawSignalRecord
from .windowing import format_starttime_text, window_to_datetime


def build_short_suffix(rng: np.random.Generator) -> str:
    """Create a short collision-resistant suffix."""
    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789")
    return "".join(rng.choice(alphabet, size=2, replace=True).tolist())


def format_weight(value: float) -> str:
    return f"{float(value):.2f}"


def format_t_value(value: float) -> str:
    return f"{float(value):+.4f}".replace("+", "")


_SAMPLE_RATE_STEM_RE = re.compile(r"-(\d+)K-", re.IGNORECASE)


def normalize_bk_stem_for_output(source_bk: RawSignalRecord, target_sample_rate_khz: int = 500) -> str:
    """Normalize the broken-wire stem so mixed outputs use a consistent sample-rate token."""
    stem = source_bk.sample_stem
    return _SAMPLE_RATE_STEM_RE.sub(f"-{int(target_sample_rate_khz)}K-", stem, count=1)


def make_output_file_name(source_bk: RawSignalRecord, noise_dataset: str, w1: float, w2: float, t_s: float, suffix: str) -> str:
    """Build the output file name."""
    return (
        f"{normalize_bk_stem_for_output(source_bk)}-mix-{noise_dataset}-"
        f"{format_weight(w1)}_{format_weight(w2)}_{format_t_value(t_s)}-{suffix}.npz"
    )


def build_data_info(
    spec: MixedSampleSpec,
    output_starttime: str,
    output_timestamp: float,
    noise_rms: float,
    bk_rms: float,
    mixed_rms: float,
    resample_noise_action: str,
    resample_bk_action: str,
    output_file_name: str,
    random_seed: int,
    target_sample_rate_hz: float,
    window_start_offset_t_s: float,
) -> dict[str, Any]:
    """Build the data_info dictionary stored inside the NPZ file."""
    return {
        "sample_type": "BK14",
        "dataset_name": spec.dataset_name,
        "build_mode": "noise_plus_broken_wire",
        "source_noise_file": spec.source_noise.sample_name,
        "source_bk_file": spec.source_bk.sample_name,
        "source_noise_dataset": spec.noise_dataset,
        "source_bk_method": spec.source_bk.method,
        "target_sample_rate_hz": target_sample_rate_hz,
        "window_duration_s": 0.02,
        "window_start_offset_t_s": window_start_offset_t_s,
        "w1": spec.w1,
        "w2": spec.w2,
        "noise_rms": noise_rms,
        "bk_rms": bk_rms,
        "mixed_rms": mixed_rms,
        "resample_noise_action": resample_noise_action,
        "resample_bk_action": resample_bk_action,
        "random_seed": random_seed,
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "builder_version": "2026-05-05",
        "output_file_name": output_file_name,
        "output_starttime": output_starttime,
        "output_timestamp": output_timestamp,
    }


def write_npz(
    output_path: Path,
    signal_values: np.ndarray,
    sample_rate_hz: float,
    starttime: str,
    arrival_time: str,
    timestamp: float,
    sample_type: str,
    data_info: dict[str, Any],
) -> None:
    """Persist a sample in the repository's NPZ format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        phase_data=np.asarray(signal_values, dtype=float),
        sample_rate=np.asarray(float(sample_rate_hz)),
        comm_count=np.asarray(int(len(signal_values))),
        npts=np.asarray(int(len(signal_values))),
        timestamp=np.asarray(float(timestamp)),
        starttime=np.asarray(starttime),
        arrival_time=np.asarray(arrival_time),
        type=np.asarray(sample_type),
        data_info=np.asarray(data_info, dtype=object),
    )


def copy_raw_sample(source_path: Path, target_path: Path) -> None:
    """Copy a raw source sample into the output tree."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def log_row_to_dict(row: BuildLogRow) -> dict[str, Any]:
    return row.__dict__.copy()
