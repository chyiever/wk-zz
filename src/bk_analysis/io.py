"""Input/output helpers for BK14 signal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal


BROKEN_WIRE_METHODS = ("corrosion", "cut")


@dataclass(frozen=True)
class SignalRecord:
    """Container for a single labeled waveform."""

    path: Path
    method: str
    signal: np.ndarray
    sample_rate: float
    starttime: str
    arrival_time: str
    sample_type: str
    arrival_source: str
    original_sample_rate: float
    resample_action: str

    @property
    def arrival_offset_seconds(self) -> float:
        """Return the arrival time offset relative to the waveform start."""
        if self.arrival_source == "estimated":
            return self.arrival_index / self.sample_rate
        start_dt = datetime.strptime(self.starttime, "%Y%m%dT%H%M%S.%f")
        arrival_dt = datetime.strptime(self.arrival_time, "%Y%m%d%H%M%S.%f")
        return (arrival_dt - start_dt).total_seconds()

    @property
    def arrival_index(self) -> int:
        """Return the nearest sample index for the labeled arrival time."""
        if self.arrival_source == "estimated":
            return estimate_arrival_index(self.signal, self.sample_rate)
        index = int(round(self.arrival_offset_seconds * self.sample_rate))
        return int(np.clip(index, 0, len(self.signal) - 1))


def estimate_arrival_index(signal_values: np.ndarray, sample_rate: float) -> int:
    """Estimate the event onset from a smoothed energy envelope."""
    envelope = np.abs(signal.hilbert(signal_values))
    energy = envelope ** 2
    smooth_samples = max(1, int(round(sample_rate * 0.0005)))
    kernel = np.ones(smooth_samples, dtype=float) / smooth_samples
    smoothed_energy = np.convolve(energy, kernel, mode="same")

    baseline_length = max(10, min(len(smoothed_energy) // 5, int(round(sample_rate * 0.01))))
    baseline = smoothed_energy[:baseline_length]
    threshold = float(np.mean(baseline) + 6.0 * np.std(baseline))
    candidate_indices = np.flatnonzero(smoothed_energy >= threshold)
    if len(candidate_indices) == 0:
        return int(np.argmax(smoothed_energy))
    return int(candidate_indices[0])


def load_signal_record(path: Path, method: str) -> SignalRecord:
    """Load a single ``npz`` waveform into a normalized record."""
    with np.load(path, allow_pickle=True) as data:
        signal_values = np.asarray(data["phase_data"], dtype=float)
        sample_rate = float(data["sample_rate"])
        arrival_time = str(data["arrival_time"].tolist())
        arrival_source = "labeled"
        if arrival_time in {"None", "", "nan"}:
            arrival_source = "estimated"
            arrival_time = ""
        return SignalRecord(
            path=path,
            method=method,
            signal=signal_values,
            sample_rate=sample_rate,
            starttime=str(data["starttime"].tolist()),
            arrival_time=arrival_time,
            sample_type=str(data["type"].tolist()),
            arrival_source=arrival_source,
            original_sample_rate=sample_rate,
            resample_action="none",
        )


def normalize_record_sample_rate(record: SignalRecord, target_sample_rate_hz: float = 500_000.0) -> SignalRecord:
    """
    Normalize one record to a target sample rate for feature extraction.

    Rules:
    - If source rate > target, use polyphase downsampling.
    - If source rate < target, use linear interpolation.
    - If equal, keep unchanged.
    """
    source_rate = float(record.sample_rate)
    target_rate = float(target_sample_rate_hz)
    if source_rate <= 0.0 or target_rate <= 0.0:
        raise ValueError("Sample rate must be positive.")

    if np.isclose(source_rate, target_rate):
        return SignalRecord(
            path=record.path,
            method=record.method,
            signal=record.signal.copy(),
            sample_rate=target_rate,
            starttime=record.starttime,
            arrival_time=record.arrival_time,
            sample_type=record.sample_type,
            arrival_source=record.arrival_source,
            original_sample_rate=record.original_sample_rate,
            resample_action="none",
        )

    if source_rate > target_rate:
        # Remove linear trend before polyphase downsampling.
        # This reduces DC leakage and edge transients on 500K BK14 records.
        signal_values = signal.detrend(record.signal, type="linear") if len(record.signal) else record.signal.copy()
        ratio = Fraction(target_rate / source_rate).limit_denominator(2000)
        signal_resampled = signal.resample_poly(signal_values, up=ratio.numerator, down=ratio.denominator, padtype="line")
        action = "downsampled"
    else:
        target_n_samples = max(1, int(round(len(record.signal) * target_rate / source_rate)))
        source_time = np.arange(len(record.signal), dtype=float) / source_rate
        target_time = np.arange(target_n_samples, dtype=float) / target_rate
        if len(source_time) > 0:
            target_time = np.clip(target_time, source_time[0], source_time[-1])
        signal_resampled = np.interp(target_time, source_time, record.signal)
        action = "interpolated"

    return SignalRecord(
        path=record.path,
        method=record.method,
        signal=np.asarray(signal_resampled, dtype=float),
        sample_rate=target_rate,
        starttime=record.starttime,
        arrival_time=record.arrival_time,
        sample_type=record.sample_type,
        arrival_source=record.arrival_source,
        original_sample_rate=record.original_sample_rate,
        resample_action=action,
    )


def iter_generation_method_files(data_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield all BK14 files grouped by generation method."""
    method_root = data_dir / "BK14-0.06S(Generation method)"
    for method in BROKEN_WIRE_METHODS:
        for path in sorted((method_root / method).glob("*.npz")):
            yield method, path


def load_generation_method_records(data_dir: Path) -> list[SignalRecord]:
    """Load all corrosion and cut samples."""
    return [
        load_signal_record(path=path, method=method)
        for method, path in iter_generation_method_files(data_dir)
    ]


def build_record_index(records: list[SignalRecord]) -> pd.DataFrame:
    """Build a compact table that describes all loaded records."""
    rows = []
    for record in records:
        rows.append(
            {
                "method": record.method,
                "path": str(record.path),
                "sample_rate_hz": record.sample_rate,
                "original_sample_rate_hz": record.original_sample_rate,
                "n_samples": len(record.signal),
                "duration_s": len(record.signal) / record.sample_rate,
                "arrival_offset_s": record.arrival_offset_seconds,
                "arrival_index": record.arrival_index,
                "arrival_source": record.arrival_source,
                "resample_action": record.resample_action,
            }
        )
    return pd.DataFrame(rows)
