"""Core data structures and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureRecord:
    """One waveform sample for feature computation."""

    sample_id: str
    sample_name: str
    sample_type: str
    sample_type_code: int
    path: Path
    signal: np.ndarray
    sample_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureResult:
    """Feature vector for one record."""

    sample_id: str
    sample_name: str
    sample_type: str
    sample_type_code: int
    features: dict[str, float]


@dataclass(frozen=True)
class FeatureParams:
    """
    Central feature parameter set.

    Notes for users:
    - `highpass_hz`: preprocessing high-pass cutoff. Default 1000 Hz. Recommended 500-5000 Hz.
    - `main_band_hz`: main analysis band for envelope/STFT/ridge/residual. Default (5000, 60000).
    - `low_band_hz`, `mid_band_hz`, `high1_band_hz`, `high2_band_hz`, `harmonic_band_hz`:
      reusable sub-bands for energy ratios and residual analysis.
    - `normalize_robust`: median/MAD normalization switch. Disable when absolute amplitude matters.
    - `envelope_smooth_ms`: envelope smoothing window. Default 0.2 ms. Recommended 0.1-0.5 ms.
    - `stft_window_ms`, `stft_overlap`, `stft_nfft`: standard STFT parameters.
    - `short_stft_window_ms`, `short_stft_overlap`: transient-sensitive STFT parameters.
    - `ridge_main_search_hz`, `ridge_h2_search_hz`, `ridge_jump_penalty_hz`:
      ridge extraction search region and smoothness controls.
    - `ridge_relative_bandwidth`: harmonic mask relative bandwidth. Default 0.08.
    - `ridge_valid_energy_ratio`: ridge frame validity gate: valid iff frame band energy >
      this ratio times the window's median frame energy. Default 0.2.
    - `ridge_fixed_band_hz`: fixed half-bandwidth (Hz) around ridges for `R_h`. Default 1000.
    - `bulge_threshold_ratio`, `bulge_min_distance_ms`: bulge counting parameters.
    - `residual_abnormal_threshold`: threshold for abnormal frame counting.
    - `local_window_ms`, `asymmetry_window_ms`: local statistics window controls.
    - `wavelet_name`, `wavelet_level`: wavelet packet parameters.
    - `damped_freqs_hz`, `damped_decay_ms`: damped-atom search grid for `C_damp` and `Delta_J`.
    - `n_jobs`: batch parallel worker count.
    """

    highpass_hz: float = 1_000.0
    main_band_hz: tuple[float, float] = (5_000.0, 100_000.0)
    low_band_hz: tuple[float, float] = (5_000.0, 15_000.0)
    mid_band_hz: tuple[float, float] = (15_000.0, 30_000.0)
    high1_band_hz: tuple[float, float] = (20_000.0, 40_000.0)
    high2_band_hz: tuple[float, float] = (20_000.0, 100_000.0)
    harmonic_band_hz: tuple[float, float] = (25_000.0, 100_000.0)
    normalize_robust: bool = True
    envelope_smooth_ms: float = 0.2
    onset_quantile: float = 0.05
    offset_quantile: float = 0.95
    stft_window_ms: float = 0.64
    stft_overlap: float = 0.875
    stft_nfft: int | None = None
    short_stft_window_ms: float = 0.32
    short_stft_overlap: float = 0.75
    short_stft_nfft: int | None = None
    ridge_main_search_hz: tuple[float, float] = (5_000.0, 30_000.0)
    ridge_h2_search_hz: tuple[float, float] = (10_000.0, 100_000.0)
    ridge_jump_penalty_hz: float = 2_000.0
    ridge_relative_bandwidth: float = 0.08
    ridge_valid_energy_ratio: float = 0.2
    ridge_fixed_band_hz: float = 1_000.0
    bulge_threshold_ratio: float = 0.6
    bulge_min_distance_ms: float = 0.2
    residual_abnormal_threshold: float = 0.55
    local_window_ms: float = 0.4
    asymmetry_window_ms: float = 0.6
    wavelet_name: str = "db4"
    wavelet_level: int = 0
    renyi_alpha: float = 2.0
    damped_freqs_hz: tuple[float, ...] = (8_000.0, 12_000.0, 16_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0, 45_000.0)
    damped_decay_ms: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.2, 2.0)
    eps: float = 1e-12
    n_jobs: int = 1


@dataclass
class FeatureContext:
    """All reusable intermediate quantities for one signal."""

    record: FeatureRecord
    params: FeatureParams
    raw_signal: np.ndarray
    preprocessed_signal: np.ndarray
    main_signal: np.ndarray
    high_signal: np.ndarray
    high2_signal: np.ndarray
    envelope: np.ndarray
    envelope_fit: np.ndarray
    envelope_peak_index: int
    onset_index: int
    offset_index: int
    stft_freqs: np.ndarray
    stft_times: np.ndarray
    stft_complex: np.ndarray
    stft_power: np.ndarray
    stft_backend: str
    short_freqs: np.ndarray
    short_times: np.ndarray
    short_power: np.ndarray
    ridge_f1: np.ndarray
    ridge_f2: np.ndarray
    ridge_idx_f1: np.ndarray
    ridge_idx_f2: np.ndarray
    ridge_mask: np.ndarray
    harmonic_energy: float
    total_energy_tf: float
    residual_power: np.ndarray
    reconstructed_signal: np.ndarray
    residual_signal: np.ndarray
    wavelet_node_energies: dict[str, float]
    wavelet_sample_rate: float
    requested_features: frozenset[str] | None = None
