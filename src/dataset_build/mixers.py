"""Signal mixing and quality metrics."""

from __future__ import annotations

import numpy as np
from scipy import signal


def center_signal(values: np.ndarray) -> np.ndarray:
    """Remove the mean from a signal."""
    values = np.asarray(values, dtype=float)
    return values - float(np.mean(values)) if values.size else values.copy()


def rms(values: np.ndarray) -> float:
    """Compute the root-mean-square."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def bandpass_for_quality(
    values: np.ndarray,
    sample_rate_hz: float,
    low_cutoff_hz: float = 5_000.0,
    high_cutoff_hz: float = 100_000.0,
) -> np.ndarray:
    """Apply linear-detrend + temporary band-pass for quality-control statistics only."""
    values = np.asarray(values, dtype=float)
    nyquist = 0.5 * float(sample_rate_hz)
    if values.size == 0:
        return values.copy()
    detrended = signal.detrend(values, type="linear")
    low = float(low_cutoff_hz)
    high = float(high_cutoff_hz)
    if low <= 0:
        low = 1.0
    if high <= low:
        high = nyquist * 0.999
    high = min(high, nyquist * 0.999)
    if low >= high:
        return detrended
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, detrended)


def highpass_for_quality(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 5_000.0) -> np.ndarray:
    """Backward-compatible alias: maps to band-pass with a near-Nyquist upper bound."""
    nyquist = 0.5 * float(sample_rate_hz)
    return bandpass_for_quality(
        values=values,
        sample_rate_hz=sample_rate_hz,
        low_cutoff_hz=cutoff_hz,
        high_cutoff_hz=nyquist * 0.999,
    )


def mix_signals(noise: np.ndarray, broken_wire: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """Linearly mix two equal-length signals."""
    noise = np.asarray(noise, dtype=float)
    broken_wire = np.asarray(broken_wire, dtype=float)
    if noise.shape != broken_wire.shape:
        raise ValueError("signals must have the same shape")
    return np.asarray(w1 * noise + w2 * broken_wire, dtype=float)


def compute_quality_rms(
    noise: np.ndarray,
    broken_wire: np.ndarray,
    mixed: np.ndarray,
    sample_rate_hz: float,
) -> tuple[float, float, float]:
    """Compute QC RMS values using a temporary band-pass filter."""
    noise_hp = bandpass_for_quality(noise, sample_rate_hz)
    broken_wire_hp = bandpass_for_quality(broken_wire, sample_rate_hz)
    mixed_hp = bandpass_for_quality(mixed, sample_rate_hz)
    return rms(noise_hp), rms(broken_wire_hp), rms(mixed_hp)
