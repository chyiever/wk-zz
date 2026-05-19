"""Core analysis routines for F30 flow-noise samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal


@dataclass(frozen=True)
class WaveformRecord:
    """One waveform record."""

    sample_id: str
    sample_name: str
    path: Path
    signal: np.ndarray
    sample_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisParams:
    """Central parameters for waveform analysis."""

    highpass_hz: float = 1_000.0
    analysis_band_hz: tuple[float, float] = (4_000.0, 40_000.0)
    envelope_smooth_ms: float = 0.20
    envelope_quantiles: tuple[float, float] = (0.05, 0.95)
    stft_window_ms: float = 0.64
    stft_overlap: float = 0.875
    stft_nfft: int | None = None
    f0_search_hz: tuple[float, float] = (4_000.0, 12_000.0)
    h2_search_hz: tuple[float, float] = (8_000.0, 24_000.0)
    h3_search_hz: tuple[float, float] = (12_000.0, 36_000.0)
    ridge_jump_penalty_hz: float = 2_000.0
    ridge_relative_tolerance: float = 0.08
    harmonic_presence_ratio_h2: float = 0.08
    harmonic_presence_ratio_h3: float = 0.05
    f0_harmonic_weights: tuple[float, float, float] = (1.0, 1.35, 0.55)
    ridge_smooth_frames: int = 9
    peak_min_distance_ms: float = 1.5
    peak_prominence_hz: float = 1_000.0
    peak_height_ratio: float = 0.90
    eps: float = 1e-12


@dataclass(frozen=True)
class HarmonicTrack:
    """Tracked harmonic ridge and validity mask."""

    multiplier: int
    ridge_hz: np.ndarray
    ridge_index: np.ndarray
    ridge_power: np.ndarray
    valid_mask: np.ndarray
    total_duration_ms: float
    longest_duration_ms: float
    presence_ratio: float
    energy_ratio_to_f0: float
    rel_error_median: float


@dataclass(frozen=True)
class SampleAnalysis:
    """Reusable outputs for one analyzed sample."""

    record: WaveformRecord
    params: AnalysisParams
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    envelope: np.ndarray
    time_s: np.ndarray
    onset_index: int
    offset_index: int
    peak_index: int
    stft_freqs_hz: np.ndarray
    stft_times_s: np.ndarray
    stft_complex: np.ndarray
    stft_power: np.ndarray
    f0_ridge_hz: np.ndarray
    f0_ridge_index: np.ndarray
    f0_ridge_power: np.ndarray
    f0_ridge_smooth_hz: np.ndarray
    h2_track: HarmonicTrack
    h3_track: HarmonicTrack
    f0_peak_indices: np.ndarray
    quadratic_coeffs: np.ndarray
    quadratic_fit_hz: np.ndarray
    feature_row: dict[str, float]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = int(max(1, window))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _butter_filter(values: np.ndarray, sample_rate: float, band_hz: tuple[float, float], btype: str) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    if btype == "highpass":
        normalized = min(max(band_hz[0] / nyquist, 1e-6), 0.999)
        sos = signal.butter(4, normalized, btype="highpass", output="sos")
    else:
        low = min(max(band_hz[0] / nyquist, 1e-6), 0.999)
        high = min(max(band_hz[1] / nyquist, low + 1e-6), 0.999)
        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, values)


def _compute_stft(values: np.ndarray, sample_rate: float, window_ms: float, overlap: float, nfft: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(32, int(round(sample_rate * window_ms / 1_000.0)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap)))
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(nperseg)))
    freqs_hz, times_s, complex_spec = signal.stft(
        values,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(complex_spec) ** 2
    return freqs_hz, times_s, complex_spec, power


def _estimate_event_support(envelope: np.ndarray, quantiles: tuple[float, float], eps: float) -> tuple[int, int]:
    energy = np.maximum(envelope, 0.0) ** 2
    cumulative = np.cumsum(energy)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    if total <= eps:
        return 0, max(0, len(envelope) - 1)
    low_idx = int(np.searchsorted(cumulative, quantiles[0] * total))
    high_idx = int(np.searchsorted(cumulative, quantiles[1] * total))
    high_idx = min(max(high_idx, low_idx + 1), len(envelope) - 1)
    return low_idx, high_idx


def _dynamic_path_from_score(freq_subset_hz: np.ndarray, score: np.ndarray, jump_penalty_hz: float) -> np.ndarray:
    n_freq, n_time = score.shape
    dp = np.empty_like(score)
    back = np.zeros_like(score, dtype=int)
    dp[:, 0] = score[:, 0]
    transition_penalty = np.abs(freq_subset_hz[:, None] - freq_subset_hz[None, :]) / max(jump_penalty_hz, 1.0)
    for frame_idx in range(1, n_time):
        candidate = dp[:, frame_idx - 1][None, :] - transition_penalty
        back[:, frame_idx] = np.argmax(candidate, axis=1)
        dp[:, frame_idx] = score[:, frame_idx] + candidate[np.arange(n_freq), back[:, frame_idx]]

    ridge_local = np.zeros(n_time, dtype=int)
    ridge_local[-1] = int(np.argmax(dp[:, -1]))
    for frame_idx in range(n_time - 1, 0, -1):
        ridge_local[frame_idx - 1] = back[ridge_local[frame_idx], frame_idx]
    return ridge_local


def _nearest_bin_indices(freq_axis_hz: np.ndarray, target_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(freq_axis_hz, target_hz)
    idx = np.clip(idx, 0, len(freq_axis_hz) - 1)
    left = np.clip(idx - 1, 0, len(freq_axis_hz) - 1)
    right = idx
    choose_right = np.abs(freq_axis_hz[right] - target_hz) <= np.abs(freq_axis_hz[left] - target_hz)
    nearest = np.where(choose_right, right, left)
    valid = (target_hz >= freq_axis_hz[0]) & (target_hz <= freq_axis_hz[-1])
    return nearest.astype(int), valid.astype(bool)


def _estimate_f0_ridge_harmonic_sum(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    params: AnalysisParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (freqs_hz >= params.f0_search_hz[0]) & (freqs_hz <= params.f0_search_hz[1])
    freq_subset = freqs_hz[mask]
    if freq_subset.size == 0 or power.shape[1] == 0:
        return np.zeros(power.shape[1], dtype=float), np.zeros(power.shape[1], dtype=int), np.zeros(power.shape[1], dtype=float)

    power_subset = power[mask, :]
    harmonic_score = np.zeros_like(power_subset)
    for mult, weight in enumerate(params.f0_harmonic_weights, start=1):
        target_hz = mult * freq_subset
        harmonic_idx, valid = _nearest_bin_indices(freqs_hz, target_hz)
        harmonic_power = power[harmonic_idx, :]
        harmonic_power = np.where(valid[:, None], harmonic_power, params.eps)
        harmonic_score += weight * np.log(harmonic_power + params.eps)

    ridge_local = _dynamic_path_from_score(freq_subset, harmonic_score, params.ridge_jump_penalty_hz)
    ridge_index = np.flatnonzero(mask)[ridge_local]
    ridge_hz = freqs_hz[ridge_index]
    ridge_power = power[ridge_index, np.arange(power.shape[1])]
    return ridge_hz, ridge_index, ridge_power


def _dynamic_ridge(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    search_hz: tuple[float, float],
    jump_penalty_hz: float,
    prior_hz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (freqs_hz >= search_hz[0]) & (freqs_hz <= search_hz[1])
    freq_subset = freqs_hz[mask]
    power_subset = power[mask, :]
    n_freq, n_time = power_subset.shape
    if n_freq == 0 or n_time == 0:
        return np.zeros(power.shape[1], dtype=float), np.zeros(power.shape[1], dtype=int), np.zeros(power.shape[1], dtype=float)

    score = np.log(power_subset + 1e-12)
    if prior_hz is not None:
        score = score - np.abs(freq_subset[:, None] - prior_hz[None, :]) / max(jump_penalty_hz, 1.0)

    ridge_local = _dynamic_path_from_score(freq_subset, score, jump_penalty_hz)

    ridge_index = np.flatnonzero(mask)[ridge_local]
    ridge_hz = freqs_hz[ridge_index]
    ridge_power = power[ridge_index, np.arange(n_time)]
    return ridge_hz, ridge_index, ridge_power


def _smooth_track(track_hz: np.ndarray, window: int) -> np.ndarray:
    if track_hz.size == 0:
        return track_hz.copy()
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    return _moving_average(track_hz, window)


def _duration_from_mask(mask: np.ndarray, times_s: np.ndarray) -> tuple[float, float]:
    if mask.size == 0 or times_s.size == 0:
        return 0.0, 0.0
    if len(times_s) == 1:
        frame_dt = 0.0
    else:
        frame_dt = float(np.median(np.diff(times_s)))
    total_duration_ms = float(np.sum(mask) * frame_dt * 1_000.0)
    longest = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_duration_ms = float(longest * frame_dt * 1_000.0)
    return total_duration_ms, longest_duration_ms


def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _track_harmonic(
    multiplier: int,
    params: AnalysisParams,
    freqs_hz: np.ndarray,
    power: np.ndarray,
    f0_ridge_hz: np.ndarray,
    f0_ridge_power: np.ndarray,
    search_hz: tuple[float, float],
    presence_ratio: float,
    times_s: np.ndarray,
) -> HarmonicTrack:
    prior_hz = multiplier * f0_ridge_hz if f0_ridge_hz.size else None
    ridge_hz, ridge_index, ridge_power = _dynamic_ridge(freqs_hz, power, search_hz, params.ridge_jump_penalty_hz, prior_hz=prior_hz)
    if ridge_hz.size == 0:
        zeros = np.zeros_like(f0_ridge_hz, dtype=bool)
        return HarmonicTrack(multiplier, ridge_hz, ridge_index, ridge_power, zeros, 0.0, 0.0, 0.0, 0.0, np.nan)

    bin_hz = freqs_hz[1] - freqs_hz[0] if len(freqs_hz) > 1 else 1.0
    target_hz = multiplier * f0_ridge_hz
    tolerance_hz = np.maximum(2.0 * bin_hz, params.ridge_relative_tolerance * np.maximum(target_hz, params.eps))
    rel_power = ridge_power / (f0_ridge_power + params.eps)
    valid_mask = (np.abs(ridge_hz - target_hz) <= tolerance_hz) & (rel_power >= presence_ratio)
    total_duration_ms, longest_duration_ms = _duration_from_mask(valid_mask, times_s)
    presence_ratio_value = float(np.mean(valid_mask)) if valid_mask.size else 0.0
    energy_ratio = float(np.sum(ridge_power) / (np.sum(f0_ridge_power) + params.eps)) if ridge_power.size else 0.0
    rel_error = np.abs(ridge_hz - target_hz) / (target_hz + params.eps)
    rel_error_median = float(np.median(rel_error[valid_mask])) if np.any(valid_mask) else np.nan
    return HarmonicTrack(
        multiplier=multiplier,
        ridge_hz=ridge_hz,
        ridge_index=ridge_index,
        ridge_power=ridge_power,
        valid_mask=valid_mask,
        total_duration_ms=total_duration_ms,
        longest_duration_ms=longest_duration_ms,
        presence_ratio=presence_ratio_value,
        energy_ratio_to_f0=energy_ratio,
        rel_error_median=rel_error_median,
    )


def _find_f0_peaks(track_hz: np.ndarray, times_s: np.ndarray, params: AnalysisParams) -> np.ndarray:
    if track_hz.size < 3:
        return np.zeros(0, dtype=int)
    if len(times_s) > 1:
        frame_dt_s = float(np.median(np.diff(times_s)))
    else:
        frame_dt_s = 1.0
    min_distance_frames = max(1, int(round((params.peak_min_distance_ms / 1_000.0) / max(frame_dt_s, params.eps))))
    peaks, _ = signal.find_peaks(track_hz, distance=min_distance_frames, prominence=params.peak_prominence_hz)
    if peaks.size == 0:
        strongest = int(np.argmax(track_hz))
        return np.asarray([strongest], dtype=int)
    peak_heights = track_hz[peaks]
    significant_mask = peak_heights >= (params.peak_height_ratio * np.max(track_hz))
    filtered_peaks = peaks[significant_mask]
    if filtered_peaks.size == 0:
        strongest = int(peaks[np.argmax(peak_heights)])
        return np.asarray([strongest], dtype=int)
    return filtered_peaks.astype(int)


def _quadratic_fit(times_s: np.ndarray, ridge_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times_s.size < 3 or ridge_hz.size < 3:
        return np.zeros(3, dtype=float), np.full_like(ridge_hz, np.nan, dtype=float)
    coeffs = np.polyfit(times_s, ridge_hz, 2)
    fit = np.polyval(coeffs, times_s)
    return coeffs, fit


def extract_feature_row(
    record: WaveformRecord,
    params: AnalysisParams,
    time_s: np.ndarray,
    filtered_signal: np.ndarray,
    envelope: np.ndarray,
    onset_index: int,
    offset_index: int,
    peak_index: int,
    stft_freqs_hz: np.ndarray,
    stft_times_s: np.ndarray,
    stft_power: np.ndarray,
    f0_ridge_hz: np.ndarray,
    f0_ridge_smooth_hz: np.ndarray,
    f0_ridge_power: np.ndarray,
    h2_track: HarmonicTrack,
    h3_track: HarmonicTrack,
    f0_peak_indices: np.ndarray,
    quadratic_coeffs: np.ndarray,
    quadratic_fit_hz: np.ndarray,
) -> dict[str, float]:
    eps = params.eps
    support_samples = max(offset_index - onset_index, 1)
    support_duration_ms = float(support_samples / record.sample_rate * 1_000.0)
    rise_ms = float(max(peak_index - onset_index, 0) / record.sample_rate * 1_000.0)
    decay_ms = float(max(offset_index - peak_index, 0) / record.sample_rate * 1_000.0)
    peak_pos_ratio = float((peak_index - onset_index) / support_samples)
    decay_rise_ratio = float(decay_ms / max(rise_ms, eps))
    envelope_symmetry = _safe_corrcoef(envelope, envelope[::-1])
    envelope_energy = envelope ** 2
    before_energy = float(np.sum(envelope_energy[onset_index : peak_index + 1]))
    after_energy = float(np.sum(envelope_energy[peak_index : offset_index + 1]))
    bulge_threshold = 0.6 * float(np.max(envelope) + eps)
    bulge_coverage = float(np.mean(envelope[onset_index : offset_index + 1] >= bulge_threshold))

    track_peak_index = int(np.argmax(f0_ridge_smooth_hz)) if f0_ridge_smooth_hz.size else 0
    track_peak_time_ms = float(stft_times_s[track_peak_index] * 1_000.0) if stft_times_s.size else np.nan
    track_peak_hz = float(f0_ridge_smooth_hz[track_peak_index]) if f0_ridge_smooth_hz.size else np.nan
    f0_start_hz = float(f0_ridge_smooth_hz[0]) if f0_ridge_smooth_hz.size else np.nan
    f0_end_hz = float(f0_ridge_smooth_hz[-1]) if f0_ridge_smooth_hz.size else np.nan
    f0_span_hz = float(np.max(f0_ridge_smooth_hz) - np.min(f0_ridge_smooth_hz)) if f0_ridge_smooth_hz.size else np.nan
    dfdt_hz_s = np.gradient(f0_ridge_smooth_hz, stft_times_s + eps) if f0_ridge_smooth_hz.size > 1 else np.zeros_like(f0_ridge_smooth_hz)
    up_ratio = float(np.mean(dfdt_hz_s > params.ridge_jump_penalty_hz)) if dfdt_hz_s.size else 0.0
    down_ratio = float(np.mean(dfdt_hz_s < -params.ridge_jump_penalty_hz)) if dfdt_hz_s.size else 0.0
    sign_change_count = int(np.sum(np.diff(np.sign(dfdt_hz_s)) != 0)) if dfdt_hz_s.size > 1 else 0

    arch_r2 = np.nan
    arch_vertex_ms = np.nan
    arch_score = 0.0
    if quadratic_fit_hz.size and np.all(np.isfinite(quadratic_fit_hz)) and f0_ridge_smooth_hz.size:
        ss_res = float(np.sum((f0_ridge_smooth_hz - quadratic_fit_hz) ** 2))
        ss_tot = float(np.sum((f0_ridge_smooth_hz - np.mean(f0_ridge_smooth_hz)) ** 2))
        arch_r2 = float(1.0 - ss_res / (ss_tot + eps))
        if abs(quadratic_coeffs[0]) > eps:
            vertex_t = -quadratic_coeffs[1] / (2.0 * quadratic_coeffs[0])
            arch_vertex_ms = float(vertex_t * 1_000.0)
            if quadratic_coeffs[0] < 0.0 and stft_times_s.size:
                if 0.15 * stft_times_s[-1] <= vertex_t <= 0.85 * stft_times_s[-1]:
                    arch_score = arch_r2

    spectrum_mean = np.mean(stft_power, axis=1) if stft_power.size else np.zeros(0, dtype=float)
    dominant_band_energy = float(np.sum(stft_power[(stft_freqs_hz >= params.analysis_band_hz[0]) & (stft_freqs_hz <= params.analysis_band_hz[1]), :]))
    total_tf_energy = float(np.sum(stft_power))

    return {
        "sample_id": record.sample_id,
        "sample_name": record.sample_name,
        "sample_path": str(record.path),
        "sample_rate_hz": float(record.sample_rate),
        "signal_duration_ms": float(len(record.signal) / record.sample_rate * 1_000.0),
        "raw_ptp": float(np.ptp(record.signal)),
        "filtered_rms": float(np.sqrt(np.mean(filtered_signal ** 2))),
        "filtered_ptp": float(np.ptp(filtered_signal)),
        "envelope_peak": float(np.max(envelope)),
        "event_support_ms": support_duration_ms,
        "rise_ms": rise_ms,
        "decay_ms": decay_ms,
        "peak_pos_ratio": peak_pos_ratio,
        "decay_rise_ratio": decay_rise_ratio,
        "envelope_symmetry": envelope_symmetry,
        "envelope_energy_ratio_before_after": float(before_energy / (after_energy + eps)),
        "bulge_coverage": bulge_coverage,
        "h2_duration_ms": h2_track.total_duration_ms,
        "h2_longest_duration_ms": h2_track.longest_duration_ms,
        "h3_duration_ms": h3_track.total_duration_ms,
        "h3_longest_duration_ms": h3_track.longest_duration_ms,
        "f0_start_khz": f0_start_hz / 1_000.0 if np.isfinite(f0_start_hz) else np.nan,
        "f0_end_khz": f0_end_hz / 1_000.0 if np.isfinite(f0_end_hz) else np.nan,
        "f0_peak_khz": track_peak_hz / 1_000.0 if np.isfinite(track_peak_hz) else np.nan,
        "f0_peak_time_ms": track_peak_time_ms,
        "f0_mean_khz": float(np.mean(f0_ridge_smooth_hz) / 1_000.0) if f0_ridge_smooth_hz.size else np.nan,
        "f0_span_khz": f0_span_hz / 1_000.0 if np.isfinite(f0_span_hz) else np.nan,
        "f0_peak_count": float(len(f0_peak_indices)),
        "f0_curve_up_ratio": up_ratio,
        "f0_curve_down_ratio": down_ratio,
        "f0_curve_turn_count": float(sign_change_count),
        "f0_arch_r2": arch_r2,
        "f0_arch_score": arch_score,
        "f0_arch_vertex_ms": arch_vertex_ms,
        "h2_presence_ratio": h2_track.presence_ratio,
        "h2_energy_ratio_to_f0": h2_track.energy_ratio_to_f0,
        "h2_rel_error_median": h2_track.rel_error_median,
        "h3_presence_ratio": h3_track.presence_ratio,
        "h3_energy_ratio_to_f0": h3_track.energy_ratio_to_f0,
        "h3_rel_error_median": h3_track.rel_error_median,
        "ridge_band_energy_ratio": float(dominant_band_energy / (total_tf_energy + eps)),
        "stft_spectral_peak_khz": float(stft_freqs_hz[int(np.argmax(spectrum_mean))] / 1_000.0) if spectrum_mean.size else np.nan,
    }


def analyze_sample(record: WaveformRecord, params: AnalysisParams | None = None) -> SampleAnalysis:
    """Analyze one waveform sample."""
    params = params or AnalysisParams()
    raw_signal = np.asarray(record.signal, dtype=float)
    detrended = signal.detrend(raw_signal - np.mean(raw_signal))
    highpassed = _butter_filter(detrended, record.sample_rate, (params.highpass_hz, 0.0), "highpass")
    filtered_signal = _butter_filter(highpassed, record.sample_rate, params.analysis_band_hz, "bandpass")
    envelope = np.abs(signal.hilbert(filtered_signal))
    smooth_window = max(1, int(round(record.sample_rate * params.envelope_smooth_ms / 1_000.0)))
    envelope = _moving_average(envelope, smooth_window)
    time_s = np.arange(len(raw_signal), dtype=float) / record.sample_rate
    onset_index, offset_index = _estimate_event_support(envelope, params.envelope_quantiles, params.eps)
    peak_index = int(np.argmax(envelope)) if envelope.size else 0

    stft_freqs_hz, stft_times_s, stft_complex, stft_power = _compute_stft(
        filtered_signal,
        record.sample_rate,
        params.stft_window_ms,
        params.stft_overlap,
        params.stft_nfft,
    )
    f0_ridge_hz, f0_ridge_index, f0_ridge_power = _estimate_f0_ridge_harmonic_sum(
        stft_freqs_hz,
        stft_power,
        params,
    )
    f0_ridge_smooth_hz = _smooth_track(f0_ridge_hz, params.ridge_smooth_frames)
    h2_track = _track_harmonic(
        2,
        params,
        stft_freqs_hz,
        stft_power,
        f0_ridge_smooth_hz,
        f0_ridge_power,
        params.h2_search_hz,
        params.harmonic_presence_ratio_h2,
        stft_times_s,
    )
    h3_track = _track_harmonic(
        3,
        params,
        stft_freqs_hz,
        stft_power,
        f0_ridge_smooth_hz,
        f0_ridge_power,
        params.h3_search_hz,
        params.harmonic_presence_ratio_h3,
        stft_times_s,
    )
    f0_peak_indices = _find_f0_peaks(f0_ridge_smooth_hz, stft_times_s, params)
    quadratic_coeffs, quadratic_fit_hz = _quadratic_fit(stft_times_s, f0_ridge_smooth_hz)

    feature_row = extract_feature_row(
        record=record,
        params=params,
        time_s=time_s,
        filtered_signal=filtered_signal,
        envelope=envelope,
        onset_index=onset_index,
        offset_index=offset_index,
        peak_index=peak_index,
        stft_freqs_hz=stft_freqs_hz,
        stft_times_s=stft_times_s,
        stft_power=stft_power,
        f0_ridge_hz=f0_ridge_hz,
        f0_ridge_smooth_hz=f0_ridge_smooth_hz,
        f0_ridge_power=f0_ridge_power,
        h2_track=h2_track,
        h3_track=h3_track,
        f0_peak_indices=f0_peak_indices,
        quadratic_coeffs=quadratic_coeffs,
        quadratic_fit_hz=quadratic_fit_hz,
    )

    return SampleAnalysis(
        record=record,
        params=params,
        raw_signal=raw_signal,
        filtered_signal=filtered_signal,
        envelope=envelope,
        time_s=time_s,
        onset_index=onset_index,
        offset_index=offset_index,
        peak_index=peak_index,
        stft_freqs_hz=stft_freqs_hz,
        stft_times_s=stft_times_s,
        stft_complex=stft_complex,
        stft_power=stft_power,
        f0_ridge_hz=f0_ridge_hz,
        f0_ridge_index=f0_ridge_index,
        f0_ridge_power=f0_ridge_power,
        f0_ridge_smooth_hz=f0_ridge_smooth_hz,
        h2_track=h2_track,
        h3_track=h3_track,
        f0_peak_indices=f0_peak_indices,
        quadratic_coeffs=quadratic_coeffs,
        quadratic_fit_hz=quadratic_fit_hz,
        feature_row=feature_row,
    )


def analyze_directory(root_dir: str | Path, params: AnalysisParams | None = None) -> pd.DataFrame:
    """Analyze all NPZ files in one directory and return a feature table."""
    from .io_utils import iter_npz_files, load_waveform_record

    params = params or AnalysisParams()
    rows: list[dict[str, float]] = []
    for path in iter_npz_files(root_dir):
        analysis = analyze_sample(load_waveform_record(path), params=params)
        rows.append(analysis.feature_row)
    return pd.DataFrame(rows)



