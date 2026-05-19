"""Windowing and timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from bk_analysis.io import estimate_arrival_index

from .types import RawSignalRecord


def parse_starttime_text(starttime: str) -> datetime:
    """Parse the common BK/F30 start-time string format."""
    starttime = str(starttime)
    for fmt in ("%Y%m%dT%H%M%S.%f", "%Y%m%d%H%M%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(starttime, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported time format: {starttime}")


def format_starttime_text(value: datetime) -> str:
    """Format a datetime into the BK-style text representation."""
    return value.strftime("%Y%m%dT%H%M%S.%f")[:-3]


def arrival_offset_seconds(record: RawSignalRecord) -> float:
    """Return the arrival offset relative to the waveform start."""
    if record.arrival_source == "estimated":
        if record.signal.size == 0 or record.sample_rate <= 0:
            return 0.0
        return float(estimate_arrival_index(record.signal, record.sample_rate) / record.sample_rate)
    if not record.arrival_time or record.arrival_time in {"None", "nan"}:
        return 0.0
    start_dt = parse_starttime_text(record.starttime)
    arrival_dt = parse_starttime_text(record.arrival_time)
    return float((arrival_dt - start_dt).total_seconds())


def build_t_candidates(t_min_s: float, t_max_s: float, step_s: float) -> np.ndarray:
    """Return a discrete candidate grid for the window shift t."""
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    count = int(round((t_max_s - t_min_s) / step_s))
    values = t_min_s + np.arange(count + 1, dtype=float) * step_s
    return np.round(values, 10)


def pick_window_start(
    arrival_offset_s: float,
    window_duration_s: float,
    signal_duration_s: float,
    sample_rate_hz: float,
    rng: np.random.Generator,
    t_candidates: np.ndarray,
    max_tries: int = 200,
) -> tuple[float, float, int]:
    """Sample a valid window start offset and return the absolute offset, selected t, and sample index."""
    if window_duration_s <= 0:
        raise ValueError("window_duration_s must be positive")
    for _ in range(max_tries):
        t_s = float(rng.choice(t_candidates))
        start_offset_s = float(arrival_offset_s + t_s)
        end_offset_s = start_offset_s + window_duration_s
        if start_offset_s >= 0.0 and end_offset_s <= signal_duration_s:
            window_start_index = int(round(start_offset_s * sample_rate_hz))
            return start_offset_s, t_s, window_start_index
    raise RuntimeError("failed to sample a valid window start")


def window_to_datetime(starttime: str, start_offset_s: float) -> datetime:
    """Convert a window offset to an absolute datetime."""
    return parse_starttime_text(starttime) + timedelta(seconds=float(start_offset_s))
