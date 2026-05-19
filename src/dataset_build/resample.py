"""Sample-rate normalization helpers."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


def resample_to_rate(values: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> tuple[np.ndarray, str]:
    """Resample a signal to the target rate and return the resampled signal plus the action label."""
    values = np.asarray(values, dtype=float)
    source_rate_hz = float(source_rate_hz)
    target_rate_hz = float(target_rate_hz)
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    if np.isclose(source_rate_hz, target_rate_hz):
        return values.copy(), "none"
    if source_rate_hz > target_rate_hz:
        # Remove linear trend before polyphase downsampling.
        # This reduces DC leakage and edge transients for 500K BK14 records.
        values = signal.detrend(values, type="linear") if values.size else values.copy()
        ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(2000)
        resampled = signal.resample_poly(values, up=ratio.numerator, down=ratio.denominator, padtype="line")
        return np.asarray(resampled, dtype=float), "downsampled"
    target_n = max(1, int(round(len(values) * target_rate_hz / source_rate_hz)))
    source_time = np.arange(len(values), dtype=float) / source_rate_hz
    target_time = np.arange(target_n, dtype=float) / target_rate_hz
    if len(source_time) > 0:
        target_time = np.clip(target_time, source_time[0], source_time[-1])
    return np.interp(target_time, source_time, values), "interpolated"
