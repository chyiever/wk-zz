"""Feature formula parameter hub.

This file is intended for manual adjustment by users.
Each parameter block includes:
- feature name(s)
- core function family
- physical meaning
- default value
- recommended tuning range
- tuning rationale
"""

from __future__ import annotations

from .base import FeatureParams


# Feature name: preprocessing high-pass / common pre-filter.
# Function family: `preprocess_signal` inside `signal_ops.build_context`.
# Physical meaning: suppresses baseline drift and very-low-frequency interference before all downstream features.
# Default: 1000 Hz.
# Recommended range: 500-5000 Hz.
# Tuning rationale: increase when slow trend/noise dominates; decrease when low-frequency event content must be retained.
PREPROCESS_HIGHPASS_HZ = 1_000.0

# Feature name: main analysis band.
# Function family: envelope, STFT, ridge, residual reconstruction.
# Physical meaning: the core event band assumed to carry the useful waveform structure.
# Default: (5 kHz, 60 kHz).
# Recommended range: low cutoff 3-10 kHz, high cutoff min(40-80 kHz, 0.8 Nyquist).
# Tuning rationale: widen when the sensor chain supports more bandwidth; narrow when out-of-band noise is strong.
MAIN_BAND_HZ = (5_000.0, 60_000.0)

# Feature names: R_hl, beta_H, R_res_hl, R_high_res, T_half_high.
# Function family: band-energy and high-frequency decay features.
# Physical meaning: reusable low/mid/high bands for high-low ratio and residual high-frequency diagnostics.
# Default bands: low=(5,15)kHz, mid=(15,30)kHz, high1=(20,40)kHz, high2=(20,60)kHz, harmonic-high=(25,60)kHz.
# Recommended range: choose contiguous sub-bands that match the actual front-end passband.
LOW_BAND_HZ = (5_000.0, 15_000.0)
MID_BAND_HZ = (15_000.0, 30_000.0)
HIGH1_BAND_HZ = (20_000.0, 40_000.0)
HIGH2_BAND_HZ = (20_000.0, 60_000.0)
HARMONIC_BAND_HZ = (25_000.0, 60_000.0)

# Feature names: most amplitude-insensitive features.
# Function family: robust normalization.
# Physical meaning: median/MAD normalization weakens sensitivity to gain, propagation distance and outliers.
# Default: True.
# Recommended options: True/False.
# Tuning rationale: disable only when absolute amplitude is itself an important feature.
NORMALIZE_ROBUST = True

# Feature names: r_p, A_env, S_env, Sk_env, R_td, R_fb, C_bulge, N_bulge, epsilon_env, eta_bw.
# Function family: envelope extraction and bulge fitting.
# Physical meaning: envelope smoothing controls bulge shape stability.
# Default: 0.2 ms.
# Recommended range: 0.1-0.5 ms.
# Tuning rationale: increase in noisy data, reduce if the event is very short and sharp.
ENVELOPE_SMOOTH_MS = 0.2

# Feature names: onset/offset dependent features such as r_p, A_env, eta_bw.
# Function family: cumulative-energy boundary detection.
# Physical meaning: low/high quantiles define effective event support.
# Default: 5% and 95%.
# Recommended range: 1-10% and 90-99%.
# Tuning rationale: wider support improves robustness, tighter support emphasizes the dominant transient body.
ONSET_QUANTILE = 0.05
OFFSET_QUANTILE = 0.95

# Feature names: SC_mean, k_sc, R_hl_mean, k_hl, beta_H, H_tf, H_alpha, rho_r, G_gap, R2_ridge, S_arch, H2_ratio.
# Function family: standard STFT.
# Physical meaning: frequency-resolution-oriented time-frequency representation.
# Default: 0.64 ms window, 87.5% overlap, nfft auto.
# Recommended range: 0.5-0.8 ms, overlap 0.75-0.875.
# Tuning rationale: longer windows improve ridge stability; shorter windows improve temporal localization.
STFT_WINDOW_MS = 0.64
STFT_OVERLAP = 0.875
STFT_NFFT = None

# Feature names: F_peak, SK_max and other transient-sensitive measurements.
# Function family: short-window STFT.
# Physical meaning: time-resolution-oriented spectrum for short transient detection.
# Default: 0.32 ms window, 75% overlap, nfft auto.
# Recommended range: 0.2-0.4 ms, overlap 0.75-0.875.
SHORT_STFT_WINDOW_MS = 0.32
SHORT_STFT_OVERLAP = 0.75
SHORT_STFT_NFFT = None

# Feature names: rho_r, G_gap, R2_ridge, S_arch, H2_ratio, epsilon_2x, C_h, rho_up, rho_down, N_turn, Delta_f_span, C_f.
# Function family: ridge extraction and harmonic mask construction.
# Physical meaning: main-ridge and second-harmonic ridge search region plus temporal smoothness.
# Defaults: main search=(5,30)kHz, second harmonic=(10,60)kHz, jump penalty=2000 Hz, relative bandwidth=0.08.
# Recommended ranges: search bands follow sensor bandwidth; jump penalty 500-5000 Hz; relative bandwidth 0.03-0.12.
RIDGE_MAIN_SEARCH_HZ = (5_000.0, 30_000.0)
RIDGE_H2_SEARCH_HZ = (10_000.0, 60_000.0)
RIDGE_JUMP_PENALTY_HZ = 2_000.0
RIDGE_RELATIVE_BANDWIDTH = 0.08

# Feature names: rho_r, H2_ratio, epsilon_2x, C_h, rho_up, rho_down.
# Function family: ridge frame validity gate.
# Physical meaning: a time frame is "valid" when its main-band energy exceeds this ratio times the
# window median frame energy; invalid frames are excluded from ridge-continuity statistics.
# Default: 0.2. Recommended range: 0.05-0.5.
RIDGE_VALID_ENERGY_RATIO = 0.2

# Feature name: R_h.
# Function family: fixed-neighborhood ridge band integration.
# Physical meaning: half-bandwidth (Hz) of the fixed ± band around f1/f2 used to distinguish R_h from
# the relative-bandwidth R_2_1. Default: 1000 Hz. Recommended range: 500-2000 Hz.
RIDGE_FIXED_BAND_HZ = 1_000.0

# Feature names: SNR_band_db, E_excess, high_observable and harmonic observability gates.
# The first fraction of each window is used as a conservative local background reference when
# an external pre-event baseline is unavailable.  A feature is observable above 3 dB by default.
BACKGROUND_FRACTION = 0.20
OBSERVABLE_SNR_DB = 3.0

# Feature names: C_bulge, N_bulge.
# Function family: local bulge counting.
# Physical meaning: the threshold ratio defines what portion of the envelope is considered the main bulge.
# Default: threshold ratio 0.6, minimum distance 0.2 ms.
# Recommended ranges: threshold 0.4-0.8, distance 0.1-0.5 ms.
# Tuning rationale: lower threshold counts more secondary bulges; larger minimum distance merges nearby bulges.
BULGE_THRESHOLD_RATIO = 0.6
BULGE_MIN_DISTANCE_MS = 0.2

# Feature name: N_abn.
# Function family: residual abnormal-frame counting.
# Physical meaning: frame-level residual-energy ratio threshold.
# Default: 0.55.
# Recommended range: 0.4-0.8.
# Tuning rationale: lower values are more sensitive, higher values reduce false abnormal counts.
RESIDUAL_ABNORMAL_THRESHOLD = 0.55

# Feature names: K_res_max, eta_asym.
# Function family: local residual window statistics.
# Physical meaning: local window length for short-time statistics around the trigger.
# Default: local=0.4 ms, asymmetry=0.6 ms.
# Recommended range: 0.2-1.0 ms.
LOCAL_WINDOW_MS = 0.4
ASYMMETRY_WINDOW_MS = 0.6

# Feature names: R_wp_*, H_wp, I_burst, D_WPT.
# Function family: wavelet packet analysis.
# Physical meaning: controls wavelet basis and decomposition depth.
# Default: db4, level=4.
# Recommended range: level 3-5.
# Tuning rationale: higher level yields finer bands but requires adequate sample rate and event duration.
WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

# Feature name: H_alpha.
# Function family: Renyi entropy.
# Physical meaning: entropy order emphasizing concentration.
# Default: 2.0.
# Recommended range: 1.5-4.0.
# Tuning rationale: larger alpha emphasizes dominant concentrated energy more strongly.
RENYI_ALPHA = 2.0

# Feature names: C_damp, alpha_hat, Q_MP, Delta_J, eta_dict.
# Function family: damped atom matching and model competition.
# Physical meaning: search grid for candidate damped sinusoids.
# Defaults: frequencies 8-45 kHz, decay constants 0.1-2.0 ms.
# Recommended range: frequency grid should cover suspected damped oscillation band; decay 0.05-3.0 ms.
DAMPED_FREQS_HZ = (8_000.0, 12_000.0, 16_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0, 45_000.0)
DAMPED_DECAY_MS = (0.1, 0.2, 0.4, 0.8, 1.2, 2.0)

# Generic numerical stability constant.
EPS = 1e-12

# Batch parallelism. Use 1 in restricted environments; use larger values on unconstrained local machines.
N_JOBS = 1


DEFAULT_FEATURE_PARAMS = FeatureParams(
    highpass_hz=PREPROCESS_HIGHPASS_HZ,
    main_band_hz=MAIN_BAND_HZ,
    low_band_hz=LOW_BAND_HZ,
    mid_band_hz=MID_BAND_HZ,
    high1_band_hz=HIGH1_BAND_HZ,
    high2_band_hz=HIGH2_BAND_HZ,
    harmonic_band_hz=HARMONIC_BAND_HZ,
    normalize_robust=NORMALIZE_ROBUST,
    envelope_smooth_ms=ENVELOPE_SMOOTH_MS,
    onset_quantile=ONSET_QUANTILE,
    offset_quantile=OFFSET_QUANTILE,
    stft_window_ms=STFT_WINDOW_MS,
    stft_overlap=STFT_OVERLAP,
    stft_nfft=STFT_NFFT,
    short_stft_window_ms=SHORT_STFT_WINDOW_MS,
    short_stft_overlap=SHORT_STFT_OVERLAP,
    short_stft_nfft=SHORT_STFT_NFFT,
    ridge_main_search_hz=RIDGE_MAIN_SEARCH_HZ,
    ridge_h2_search_hz=RIDGE_H2_SEARCH_HZ,
    ridge_jump_penalty_hz=RIDGE_JUMP_PENALTY_HZ,
    ridge_relative_bandwidth=RIDGE_RELATIVE_BANDWIDTH,
    ridge_valid_energy_ratio=RIDGE_VALID_ENERGY_RATIO,
    ridge_fixed_band_hz=RIDGE_FIXED_BAND_HZ,
    background_fraction=BACKGROUND_FRACTION,
    observable_snr_db=OBSERVABLE_SNR_DB,
    bulge_threshold_ratio=BULGE_THRESHOLD_RATIO,
    bulge_min_distance_ms=BULGE_MIN_DISTANCE_MS,
    residual_abnormal_threshold=RESIDUAL_ABNORMAL_THRESHOLD,
    local_window_ms=LOCAL_WINDOW_MS,
    asymmetry_window_ms=ASYMMETRY_WINDOW_MS,
    wavelet_name=WAVELET_NAME,
    wavelet_level=WAVELET_LEVEL,
    renyi_alpha=RENYI_ALPHA,
    damped_freqs_hz=DAMPED_FREQS_HZ,
    damped_decay_ms=DAMPED_DECAY_MS,
    eps=EPS,
    n_jobs=N_JOBS,
)


def build_feature_params(**overrides) -> FeatureParams:
    """Build a parameter set with explicit overrides."""
    return FeatureParams(**{**DEFAULT_FEATURE_PARAMS.__dict__, **overrides})
