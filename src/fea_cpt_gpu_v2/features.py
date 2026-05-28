"""Grouped feature calculators covering all features in 特征汇总.md."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal, stats

from .base import FeatureContext, FeatureResult
from .gpu_backend import rank2_hankel_quality
from .signal_ops import safe_divide, smooth_envelope


def _line_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _band_mask(freqs: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    return (freqs >= band_hz[0]) & (freqs <= band_hz[1])


def _framewise_energy(power: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if power.size == 0:
        return np.zeros(0, dtype=float)
    return np.sum(power[mask, :], axis=0)


def _spectral_centroid(freqs: np.ndarray, power: np.ndarray, eps: float) -> np.ndarray:
    numerator = np.sum(freqs[:, None] * power, axis=0)
    denominator = np.sum(power, axis=0) + eps
    return numerator / denominator


def _renyi_entropy(prob: np.ndarray, alpha: float, eps: float) -> float:
    if abs(alpha - 1.0) < 1e-8:
        return float(-np.sum(prob * np.log(prob + eps)))
    return float(np.log(np.sum(prob ** alpha) + eps) / (1.0 - alpha))


def _tkeo(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.zeros_like(values)
    output = np.zeros_like(values)
    output[1:-1] = values[1:-1] ** 2 - values[:-2] * values[2:]
    return output


def _crest_factor(values: np.ndarray, eps: float) -> float:
    return float(np.max(np.abs(values)) / (math.sqrt(np.mean(values ** 2)) + eps))


def _local_kurtosis_max(values: np.ndarray, window: int) -> float:
    if len(values) < window or window < 4:
        return float(stats.kurtosis(values, fisher=False, bias=False)) if len(values) >= 4 else 0.0
    views = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    kurtosis_values = stats.kurtosis(views, axis=1, fisher=False, bias=False)
    max_value = float(np.nanmax(kurtosis_values)) if kurtosis_values.size else -np.inf
    return max_value if np.isfinite(max_value) else 0.0


def _spectral_kurtosis(power: np.ndarray, eps: float) -> np.ndarray:
    mean_power = np.mean(power, axis=1, keepdims=True)
    var_power = np.var(power, axis=1, keepdims=True) + eps
    normalized = (power - mean_power) / np.sqrt(var_power)
    return np.mean(normalized ** 4, axis=1)


def _damped_atom_match(values: np.ndarray, sample_rate: float, freqs_hz: tuple[float, ...], decays_ms: tuple[float, ...], eps: float) -> tuple[float, np.ndarray]:
    time_axis = np.arange(len(values), dtype=float) / sample_rate
    values_norm = np.linalg.norm(values) + eps
    best_score = 0.0
    best_atom = np.zeros_like(values)
    for freq_hz in freqs_hz:
        for decay_ms in decays_ms:
            tau = decay_ms / 1_000.0
            atom = np.exp(-time_axis / max(tau, eps)) * np.cos(2.0 * np.pi * freq_hz * time_axis)
            atom_norm = np.linalg.norm(atom) + eps
            score = float(abs(np.dot(values, atom)) / (values_norm * atom_norm))
            if score > best_score:
                best_score = score
                best_atom = atom / atom_norm
    return best_score, best_atom


def compute_all_features(context: FeatureContext) -> FeatureResult:
    """Compute the full feature set for one context."""
    params = context.params
    fs = context.record.sample_rate
    eps = params.eps
    signal_values = context.main_signal
    residual = context.residual_signal
    envelope = context.envelope
    n = len(signal_values)
    time_axis = np.arange(n, dtype=float) / fs
    peak_idx = context.envelope_peak_index
    onset_idx = context.onset_index
    offset_idx = context.offset_index
    on_t = onset_idx / fs
    off_t = offset_idx / fs
    peak_t = peak_idx / fs
    duration = max(off_t - on_t, eps)
    energy_env = envelope ** 2
    before_energy = float(np.sum(energy_env[onset_idx:peak_idx + 1]))
    after_energy = float(np.sum(energy_env[peak_idx:offset_idx + 1]))

    features: dict[str, float] = {}

    features["r_p"] = float((peak_t - on_t) / duration)
    features["C_E"] = float(np.sum(time_axis * energy_env) / (np.sum(energy_env) + eps))
    features["A_env"] = float(((off_t - peak_t) - (peak_t - on_t)) / duration)
    features["S_env"] = float(np.corrcoef(envelope, envelope[::-1])[0, 1]) if len(envelope) > 3 else 0.0
    features["Sk_env"] = float(stats.skew(envelope, bias=False)) if len(envelope) > 2 else 0.0
    rise_time = max((peak_idx - onset_idx) / fs, eps)
    decay_time = max((offset_idx - peak_idx) / fs, eps)
    features["R_td"] = float(decay_time / rise_time)
    features["R_fb"] = float(before_energy / (after_energy + eps))
    bulge_threshold = params.bulge_threshold_ratio * float(np.max(envelope) + eps)
    bulge_mask = envelope >= bulge_threshold
    features["C_bulge"] = float(np.mean(bulge_mask))
    min_distance = max(1, int(round(fs * params.bulge_min_distance_ms / 1_000.0)))
    peaks, _ = signal.find_peaks(envelope, height=bulge_threshold, distance=min_distance)
    features["N_bulge"] = float(len(peaks))
    features["epsilon_env"] = float(np.linalg.norm(envelope - context.envelope_fit) / (np.linalg.norm(envelope) + eps))
    cum = np.cumsum(energy_env)
    low_idx = int(np.searchsorted(cum, 0.05 * cum[-1])) if cum.size and cum[-1] > 0 else 0
    high_idx = int(np.searchsorted(cum, 0.95 * cum[-1])) if cum.size and cum[-1] > 0 else len(envelope) - 1
    eta_num = (high_idx - peak_idx) + 1
    eta_den = (peak_idx - low_idx) + 1
    features["eta_bw"] = float(eta_num / max(1, eta_den))

    sc = _spectral_centroid(context.stft_freqs, context.stft_power, eps)
    low_energy = _framewise_energy(context.stft_power, _band_mask(context.stft_freqs, params.low_band_hz))
    high_energy = _framewise_energy(context.stft_power, _band_mask(context.stft_freqs, params.high2_band_hz))
    features["SC_mean"] = float(np.mean(sc)) if sc.size else 0.0
    features["k_sc"] = _line_slope(context.stft_times, sc) if sc.size else 0.0
    r_hl = safe_divide(high_energy, low_energy, eps)
    features["R_hl_mean"] = float(np.mean(r_hl)) if np.size(r_hl) else 0.0
    features["k_hl"] = _line_slope(context.stft_times, np.asarray(r_hl)) if np.size(r_hl) else 0.0
    features["beta_H"] = _line_slope(context.stft_times, np.log(high_energy + eps)) if high_energy.size else 0.0
    prob_tf = context.stft_power / (np.sum(context.stft_power) + eps)
    features["H_tf"] = float(-np.sum(prob_tf * np.log(prob_tf + eps)))
    power_spectrum = np.mean(context.stft_power, axis=1) if context.stft_power.size else np.zeros(0)
    features["SF"] = float(np.exp(np.mean(np.log(power_spectrum + eps))) / (np.mean(power_spectrum) + eps)) if power_spectrum.size else 0.0
    features["H_alpha"] = _renyi_entropy(prob_tf, params.renyi_alpha, eps) if prob_tf.size else 0.0

    active = context.ridge_f1 > 0.0
    ridge_valid = float(np.mean(active)) if active.size else 0.0
    features["rho_r"] = ridge_valid
    gaps = np.logical_and(active[:-1], np.abs(np.diff(context.ridge_f1)) < eps) if len(active) > 1 else np.zeros(0, dtype=bool)
    features["G_gap"] = float(np.mean(gaps)) if gaps.size else 0.0
    if np.sum(active) >= 3:
        coeffs = np.polyfit(context.stft_times[active], context.ridge_f1[active], 2)
        fit = np.polyval(coeffs, context.stft_times[active])
        features["R2_ridge"] = _r2_score(context.ridge_f1[active], fit)
        vertex = -coeffs[1] / (2.0 * coeffs[0] + eps)
        features["S_arch"] = float(features["R2_ridge"] * (1.0 if coeffs[0] < 0.0 and 0.2 * context.stft_times[-1] <= vertex <= 0.8 * context.stft_times[-1] else 0.0))
    else:
        features["R2_ridge"] = 0.0
        features["S_arch"] = 0.0
    diff_h2 = np.abs(context.ridge_f2 - 2.0 * context.ridge_f1)
    bin_hz = context.stft_freqs[1] - context.stft_freqs[0] if len(context.stft_freqs) > 1 else 1.0
    tol = np.maximum(2.0 * bin_hz, params.ridge_relative_bandwidth * np.maximum(context.ridge_f1, eps))
    harmonic_match = diff_h2 < tol
    features["H2_ratio"] = float(np.mean(harmonic_match[active])) if np.any(active) else 0.0
    e1 = np.sum(context.stft_power[context.ridge_idx_f1, np.arange(len(context.ridge_idx_f1))]) if len(context.ridge_idx_f1) else 0.0
    e2 = np.sum(context.stft_power[context.ridge_idx_f2, np.arange(len(context.ridge_idx_f2))]) if len(context.ridge_idx_f2) else 0.0
    features["R_2_1"] = float(e2 / (e1 + eps))
    harmonic_stack = 0.0
    if len(context.ridge_f1):
        for multiplier in (1, 2, 3):
            target = multiplier * context.ridge_f1
            for t, freq in enumerate(target):
                idx = int(np.argmin(np.abs(context.stft_freqs - freq))) if len(context.stft_freqs) else 0
                harmonic_stack += float(context.stft_power[idx, t])
    features["H_stack"] = float(harmonic_stack / (context.total_energy_tf + eps))
    features["R_h"] = float(e2 / (e1 + eps))
    features["epsilon_2x"] = float(np.median(diff_h2[active] / (context.ridge_f1[active] + eps))) if np.any(active) else 0.0
    features["C_h"] = float(np.mean(diff_h2[active])) if np.any(active) else 0.0
    features["R_harm"] = float(context.harmonic_energy / (context.total_energy_tf + eps))
    features["Ridge_coh"] = features["R_harm"]
    dfdt = np.gradient(context.ridge_f1, context.stft_times + eps) if len(context.ridge_f1) > 1 else np.zeros_like(context.ridge_f1)
    slope_threshold = params.ridge_jump_penalty_hz
    features["rho_up"] = float(np.mean(dfdt > slope_threshold)) if dfdt.size else 0.0
    features["rho_down"] = float(np.mean(dfdt < -slope_threshold)) if dfdt.size else 0.0
    sign_changes = np.diff(np.sign(dfdt))
    features["N_turn"] = float(np.sum(sign_changes != 0)) if sign_changes.size else 0.0
    features["Delta_f_span"] = float(np.max(context.ridge_f1) - np.min(context.ridge_f1)) if len(context.ridge_f1) else 0.0
    curvature = np.gradient(np.gradient(context.ridge_f1, context.stft_times + eps), context.stft_times + eps) if len(context.ridge_f1) > 2 else np.zeros_like(context.ridge_f1)
    features["C_f"] = float(np.mean(np.abs(curvature) / (np.mean(np.abs(context.ridge_f1)) + eps))) if curvature.size else 0.0

    features["E_harm"] = float(context.harmonic_energy)
    features["E_res"] = float(np.sum(context.residual_power))
    features["rho_res"] = float(features["E_res"] / (context.total_energy_tf + eps))
    high_res_energy = float(np.sum(context.residual_power[_band_mask(context.stft_freqs, params.harmonic_band_hz), :]))
    features["R_high_res"] = float(high_res_energy / (context.total_energy_tf + eps))
    features["epsilon_rec"] = float(np.sum((signal_values - context.reconstructed_signal) ** 2) / (np.sum(signal_values ** 2) + eps))
    residual_ratio_per_frame = np.sum(context.residual_power, axis=0) / (np.sum(context.stft_power, axis=0) + eps) if context.residual_power.size else np.zeros(0)
    features["N_abn"] = float(np.sum(residual_ratio_per_frame > params.residual_abnormal_threshold))

    spectral_flux = np.maximum(np.diff(np.sum(context.short_power, axis=0), prepend=0.0), 0.0)
    features["F_peak"] = float(np.max(spectral_flux)) if spectral_flux.size else 0.0
    tkeo_values = _tkeo(signal_values)
    tkeo_res = _tkeo(residual)
    features["R_tkeo"] = float(np.max(tkeo_values) / (np.mean(tkeo_values) + eps)) if tkeo_values.size else 0.0
    features["R_res_tkeo"] = float(np.max(tkeo_res) / (np.mean(tkeo_res) + eps)) if tkeo_res.size else 0.0
    window = max(8, int(round(fs * params.local_window_ms / 1_000.0)))
    features["K_loc"] = float(stats.kurtosis(signal_values, fisher=False, bias=False)) if len(signal_values) >= 4 else 0.0
    features["K_res_max"] = _local_kurtosis_max(residual, window)
    sk = _spectral_kurtosis(context.short_power, eps) if context.short_power.size else np.zeros(0)
    features["SK_max"] = float(np.max(sk)) if sk.size else 0.0
    features["CF_res"] = _crest_factor(residual, eps)
    high_env = smooth_envelope(context.high2_signal, fs, params.envelope_smooth_ms)
    high_peak = int(np.argmax(high_env)) if high_env.size else 0
    half_level = 0.5 * float(np.max(high_env)) if high_env.size else 0.0
    below = np.flatnonzero(high_env[high_peak:] <= half_level) if high_env.size else np.zeros(0, dtype=int)
    features["T_half_high"] = float((below[0] / fs) if below.size else 0.0)

    sc_res = _spectral_centroid(context.stft_freqs, context.residual_power, eps)
    low_res = _framewise_energy(context.residual_power, _band_mask(context.stft_freqs, params.low_band_hz))
    high_res = _framewise_energy(context.residual_power, _band_mask(context.stft_freqs, params.high2_band_hz))
    features["SC_res_mean"] = float(np.mean(sc_res)) if sc_res.size else 0.0
    features["k_res_sc"] = _line_slope(context.stft_times, sc_res) if sc_res.size else 0.0
    r_res_hl = safe_divide(high_res, low_res, eps)
    features["R_res_hl_mean"] = float(np.mean(r_res_hl)) if np.size(r_res_hl) else 0.0
    features["k_res_hl"] = _line_slope(context.stft_times, np.asarray(r_res_hl)) if np.size(r_res_hl) else 0.0
    residual_env = smooth_envelope(residual, fs, params.envelope_smooth_ms)
    features["S_res_env"] = float(np.corrcoef(residual_env, residual_env[::-1])[0, 1]) if len(residual_env) > 3 else 0.0
    asym_window = max(1, int(round(fs * params.asymmetry_window_ms / 1_000.0)))
    start = max(0, peak_idx - asym_window)
    stop = min(len(residual), peak_idx + asym_window)
    before = float(np.sum(residual[start:peak_idx] ** 2))
    after = float(np.sum(residual[peak_idx:stop] ** 2))
    features["eta_asym"] = float(after / (before + eps))

    total_wp = float(sum(context.wavelet_node_energies.values())) + eps
    sorted_nodes = sorted(context.wavelet_node_energies.items())
    for node_name, energy in sorted_nodes:
        features[f"R_wp_{node_name}"] = float(energy / total_wp)
    wp_prob = np.asarray([energy / total_wp for _, energy in sorted_nodes], dtype=float)
    features["H_wp"] = float(-np.sum(wp_prob * np.log(wp_prob + eps))) if wp_prob.size else 0.0
    features["I_burst"] = float(np.max(wp_prob) / (np.median(wp_prob) + eps)) if wp_prob.size else 0.0
    split = max(1, len(wp_prob) // 2)
    features["D_WPT"] = float(np.sum(wp_prob[split:]) - np.sum(wp_prob[:split])) if wp_prob.size else 0.0

    c_damp, best_atom = _damped_atom_match(residual, fs, params.damped_freqs_hz, params.damped_decay_ms, eps)
    features["C_damp"] = c_damp
    residual_env = np.maximum(residual_env, eps)
    features["alpha_hat"] = float(-_line_slope(time_axis, np.log(residual_env)))
    features["Q_MP"] = rank2_hankel_quality(residual, eps=eps, max_cols=128)
    projection = float(np.dot(residual, best_atom))
    damage_component = projection * best_atom
    baseline_error = float(np.sum((signal_values - context.reconstructed_signal) ** 2))
    enhanced_error = float(np.sum((signal_values - context.reconstructed_signal - damage_component) ** 2))
    features["Delta_J"] = float((baseline_error - enhanced_error) / (np.sum(signal_values ** 2) + eps))
    harm_coeff = float(np.linalg.norm(context.reconstructed_signal, ord=1))
    dmg_coeff = float(np.linalg.norm(damage_component, ord=1))
    features["eta_dict"] = float(dmg_coeff / (harm_coeff + dmg_coeff + eps))

    return FeatureResult(
        sample_id=context.record.sample_id,
        sample_name=context.record.sample_name,
        sample_type=context.record.sample_type,
        sample_type_code=context.record.sample_type_code,
        features=features,
    )
