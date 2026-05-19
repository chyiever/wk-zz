"""Feature extraction for broken-wire waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .io import SignalRecord


DEFAULT_BANDS_HZ = (
    (1_000, 100_000),
    (1_000, 5_000),
    (5_000, 10_000),
    (10_000, 50_000),
    (50_000, 100_000),
    (20_000, 30_000),
)


@dataclass(frozen=True)
class TimeFeatureResult:
    """Time-domain measurements for one waveform."""

    duration_s: float
    start_energy_0p01s: float
    end_index: int
    band_peak_to_peak: dict[tuple[int, int], float]


@dataclass(frozen=True)
class SpectralFeatureResult:
    """Spectral measurements for one waveform."""

    frequency_hz: np.ndarray
    psd: np.ndarray
    cumulative_energy_ratio: np.ndarray


def _moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Smooth a vector with a centered moving average."""
    if window_samples <= 1:
        return values
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(values, kernel, mode="same")


def compute_smoothed_energy(signal_values: np.ndarray, sample_rate: float, smooth_ms: float = 0.5) -> np.ndarray:
    """Return a short-time energy proxy based on the analytic-signal envelope."""
    envelope = np.abs(signal.hilbert(signal_values))
    energy = envelope ** 2
    window_samples = max(1, int(round(sample_rate * smooth_ms / 1_000.0)))
    return _moving_average(energy, window_samples)


def find_decay_end_index(
    signal_values: np.ndarray,
    sample_rate: float,
    arrival_index: int,
    smooth_ms: float = 0.5,
) -> int:
    """
    Find the end sample where the post-arrival energy decays to peak/e.

    The search is constrained to samples after the labeled arrival time.
    """
    post_arrival_energy = compute_smoothed_energy(signal_values[arrival_index:], sample_rate, smooth_ms=smooth_ms)
    if len(post_arrival_energy) == 0:
        return arrival_index
    peak_relative_index = int(np.argmax(post_arrival_energy))
    peak_energy = float(post_arrival_energy[peak_relative_index])
    threshold = peak_energy / np.e
    tail = post_arrival_energy[peak_relative_index:]
    below_threshold = np.flatnonzero(tail <= threshold)
    if len(below_threshold) == 0:
        return len(signal_values) - 1
    return arrival_index + peak_relative_index + int(below_threshold[0])


def _band_filter(
    signal_values: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter for a target band."""
    nyquist = 0.5 * sample_rate
    if low_hz < 0 or high_hz <= 0 or low_hz >= high_hz:
        raise ValueError(f"Invalid band [{low_hz}, {high_hz}] Hz.")

    low = max(0.0, low_hz)
    high = min(high_hz, nyquist * 0.999)
    if low >= high:
        raise ValueError(f"Band [{low_hz}, {high_hz}] Hz is outside Nyquist {nyquist} Hz.")

    if low <= 0.0:
        wn = high / nyquist
        sos = signal.butter(N=4, Wn=wn, btype="lowpass", output="sos")
    elif high_hz >= nyquist:
        wn = low / nyquist
        sos = signal.butter(N=4, Wn=wn, btype="highpass", output="sos")
    else:
        wn = [low / nyquist, high / nyquist]
        sos = signal.butter(N=4, Wn=wn, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, signal_values)


def compute_band_peak_to_peak(
    signal_values: np.ndarray,
    sample_rate: float,
    bands_hz: tuple[tuple[int, int], ...] = DEFAULT_BANDS_HZ,
    highpass_cutoff_hz: float = 1_000.0,
) -> dict[tuple[int, int], float]:
    """Measure peak-to-peak value after global high-pass + per-band filtering."""
    results: dict[tuple[int, int], float] = {}
    high_passed = _band_filter(
        signal_values,
        sample_rate,
        low_hz=highpass_cutoff_hz,
        high_hz=0.5 * sample_rate,
    )
    for low_hz, high_hz in bands_hz:
        filtered = _band_filter(high_passed, sample_rate, low_hz=low_hz, high_hz=high_hz)
        results[(low_hz, high_hz)] = float(np.ptp(filtered))
    return results


def compute_time_features(
    record: SignalRecord,
    bands_hz: tuple[tuple[int, int], ...] = DEFAULT_BANDS_HZ,
    ptp_highpass_cutoff_hz: float = 1_000.0,
) -> TimeFeatureResult:
    """Extract the requested time-domain features from one record."""
    arrival_index = record.arrival_index
    end_index = find_decay_end_index(record.signal, record.sample_rate, arrival_index)
    duration_s = (end_index - arrival_index) / record.sample_rate

    start_window_samples = max(1, int(round(0.01 * record.sample_rate)))
    start_window = record.signal[arrival_index : min(len(record.signal), arrival_index + start_window_samples)]
    start_energy_0p01s = float(np.sum(start_window ** 2))

    band_peak_to_peak = compute_band_peak_to_peak(
        signal_values=record.signal,
        sample_rate=record.sample_rate,
        bands_hz=bands_hz,
        highpass_cutoff_hz=ptp_highpass_cutoff_hz,
    )
    return TimeFeatureResult(
        duration_s=duration_s,
        start_energy_0p01s=start_energy_0p01s,
        end_index=end_index,
        band_peak_to_peak=band_peak_to_peak,
    )


def compute_psd_segment(
    signal_values: np.ndarray,
    sample_rate: float,
    start_index: int,
    segment_duration_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD over a fixed window that starts at the arrival time."""
    segment_samples = max(8, int(round(segment_duration_s * sample_rate)))
    segment = signal_values[start_index : min(len(signal_values), start_index + segment_samples)]
    if len(segment) < segment_samples:
        segment = np.pad(segment, (0, segment_samples - len(segment)))

    nperseg = min(2048, len(segment))
    noverlap = nperseg // 2
    frequency_hz, psd = signal.welch(
        segment,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return frequency_hz, psd


def compute_spectral_features(
    record: SignalRecord,
    psd_segment_duration_s: float = 0.02,
) -> SpectralFeatureResult:
    """Extract PSD and cumulative-energy curves for one record."""
    frequency_hz, psd = compute_psd_segment(
        signal_values=record.signal,
        sample_rate=record.sample_rate,
        start_index=record.arrival_index,
        segment_duration_s=psd_segment_duration_s,
    )
    cumulative_energy = np.cumsum(psd)
    cumulative_energy_ratio = cumulative_energy / cumulative_energy[-1]
    return SpectralFeatureResult(
        frequency_hz=frequency_hz,
        psd=psd,
        cumulative_energy_ratio=cumulative_energy_ratio,
    )
