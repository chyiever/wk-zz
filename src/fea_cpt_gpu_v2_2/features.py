"""Grouped feature calculators covering all features in 特征汇总.md.

修订记录（2026-09，依据 docs/特征一致性核对报告.md §3.2-A 修改）：
- T_half_high：显式按"峰后衰减时长"起算，峰后从未跌破 50% 时返回 NaN 而非 0.0。
- rho_r / H2_ratio / epsilon_2x / C_h / rho_up / rho_down / N_turn / Delta_f_span / C_f：
  引入帧有效性判据（主带帧能量 > 阈值×帧能量中位数），无效帧不再参与统计，消除 rho_r 恒为 1 的退化。
- G_gap：判据由"|df1|<1e-12 卡同一频点"改为"脊线丢失帧 或 |df1|>2 倍频率分辨率"。
- N_turn：剔除零斜率样本后再统计方向变化，消除 +→0→- 重复计数。
- beta_H / alpha_hat：由全窗拟合改为"峰值→事件终点"的峰后窗拟合，避免上升段污染衰减斜率。
- F_peak：按频点先做正增量（半波整流）再求和；首帧通量置 0。
- SK_max：改用经典谱峭度估计器（Antoni：<|X|^4>/<|X|^2>^2 - 2），对高斯过程趋于 0。
- R_2_1 / H_stack：脊线单 bin 改为 ±相对带宽 带内积分。
- R_h：改为 f1/f2 固定 ±1 kHz 邻域带积分，与 R_2_1（相对带宽）形成互补而非重复。
- D_WPT：高低频节点集合改按"子带中心频率 > 主带几何中点"划分，而非按节点序对半分。
"""

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
    """Classic spectral-kurtosis estimator (Antoni): <|X|^4>/<|X|^2>^2 - 2 per frequency bin.

    输入为 STFT 功率谱（=|X|^2），因此 <|X|^4> = mean(P^2)。高斯过程理论上趋近 0。
    """
    if power.ndim != 2 or power.shape[1] < 2:
        return np.zeros(0, dtype=float)
    mean_p = np.mean(power, axis=1)
    mean_p2 = np.mean(power ** 2, axis=1)
    return mean_p2 / (mean_p ** 2 + eps) - 2.0


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


def _ridge_active_frames(context: FeatureContext) -> np.ndarray:
    """帧有效性判据：主带帧能量须大于阈值×全窗帧能量中位数，且脊线存在。

    用于修复 DP 脊线"只要有数据必返回正频率"导致 rho_r 恒为 1 的退化。
    """
    f1 = np.asarray(context.ridge_f1, dtype=float)
    n = f1.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    frame_energy = np.sum(context.stft_power, axis=0) if context.stft_power.size else np.zeros(n, dtype=float)
    frame_energy = np.asarray(frame_energy, dtype=float)
    if frame_energy.size != n:
        return f1 > 0.0
    if frame_energy.size:
        median_energy = float(np.median(frame_energy))
        floor = context.params.ridge_valid_energy_ratio * median_energy
    else:
        floor = 0.0
    return (f1 > 0.0) & (frame_energy > floor)


def _ridge_band_energy(
    power: np.ndarray,
    freqs: np.ndarray,
    ridge_f: np.ndarray,
    half_width: float | np.ndarray,
    skip_clipped: bool = True,
) -> float:
    """沿脊线在其 ±half_width 邻域内做带内能量积分。

    skip_clipped=True 时跳过邻域超出 STFT 频率轴范围的帧，避免边缘部分积分偏置。
    """
    if power.ndim != 2 or len(freqs) == 0 or len(ridge_f) == 0:
        return 0.0
    n_frames = power.shape[1]
    if n_frames == 0:
        return 0.0
    hw = np.broadcast_to(np.asarray(half_width, dtype=float), (n_frames,))
    f_lo = float(freqs[0])
    f_hi = float(freqs[-1])
    total = 0.0
    for t in range(n_frames):
        f = float(ridge_f[t])
        if not np.isfinite(f) or f <= 0.0:
            continue
        if skip_clipped and (f - hw[t] < f_lo or f + hw[t] > f_hi):
            continue
        idx = np.abs(freqs - f) <= hw[t]
        total += float(np.sum(power[idx, t]))
    return total


def _node_center_hz(path: str, fs: float) -> float:
    """小波包子带中心频率：按 path 的 a/d 逐层对半划分 0..fs/2 频率轴。"""
    lo, hi = 0.0, fs / 2.0
    for ch in path:
        mid = 0.5 * (lo + hi)
        if ch == "d":
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    after_energy = float(np.sum(energy_env[peak_idx + 1:offset_idx + 1]))

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

    # beta_H：峰值→事件终点 的峰后窗拟合 log(高频能量)，避免上升段污染衰减斜率
    post_mask = (context.stft_times >= peak_t) & (context.stft_times <= off_t) if context.stft_times.size else np.zeros(0, dtype=bool)
    if high_energy.size and np.sum(post_mask) >= 2:
        features["beta_H"] = _line_slope(context.stft_times[post_mask], np.log(high_energy[post_mask] + eps))
    else:
        features["beta_H"] = _line_slope(context.stft_times, np.log(high_energy + eps)) if high_energy.size else 0.0

    prob_tf = context.stft_power / (np.sum(context.stft_power) + eps)
    features["H_tf"] = float(-np.sum(prob_tf * np.log(prob_tf + eps)))
    power_spectrum = np.mean(context.stft_power, axis=1) if context.stft_power.size else np.zeros(0)
    features["SF"] = float(np.exp(np.mean(np.log(power_spectrum + eps))) / (np.mean(power_spectrum) + eps)) if power_spectrum.size else 0.0
    features["H_alpha"] = _renyi_entropy(prob_tf, params.renyi_alpha, eps) if prob_tf.size else 0.0

    # 帧有效性（主带帧能量门限），rho_r 与后续脊线统计共用
    active = _ridge_active_frames(context)
    ridge_valid = float(np.mean(active)) if active.size else 0.0
    features["rho_r"] = ridge_valid

    # G_gap：间隙 = 脊线丢失帧 或 有效脊线跳变超过 2 倍频率分辨率；分母为总帧数
    bin_hz = context.stft_freqs[1] - context.stft_freqs[0] if len(context.stft_freqs) > 1 else 1.0
    if len(active) > 1:
        jump = np.zeros(len(active), dtype=bool)
        jump[1:] = active[1:] & active[:-1] & (np.abs(np.diff(context.ridge_f1)) > 2.0 * bin_hz)
        gaps = (~active) | jump
        features["G_gap"] = float(np.mean(gaps))
    else:
        features["G_gap"] = 0.0

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
    tol = np.maximum(2.0 * bin_hz, params.ridge_relative_bandwidth * np.maximum(context.ridge_f1, eps))
    harmonic_match = diff_h2 < tol
    features["H2_ratio"] = float(np.mean(harmonic_match[active])) if np.any(active) else 0.0

    # R_2_1：f1/f2 脊线 ±相对带宽 带内能量积分
    if len(context.ridge_f1) and context.stft_power.size:
        rel_hw1 = np.maximum(2.0 * bin_hz, params.ridge_relative_bandwidth * np.maximum(context.ridge_f1, 0.0))
        rel_hw2 = np.maximum(2.0 * bin_hz, params.ridge_relative_bandwidth * np.maximum(context.ridge_f2, 0.0))
        e1 = _ridge_band_energy(context.stft_power, context.stft_freqs, context.ridge_f1, rel_hw1)
        e2 = _ridge_band_energy(context.stft_power, context.stft_freqs, context.ridge_f2, rel_hw2)
    else:
        e1, e2 = 0.0, 0.0
    features["R_2_1"] = float(e2 / (e1 + eps))

    # H_stack：1×/2×/3×f1 谐波位置 ±相对带宽 带内积分
    harmonic_stack = 0.0
    if len(context.ridge_f1) and context.stft_power.size:
        for multiplier in (1, 2, 3):
            target = multiplier * context.ridge_f1
            hw = np.maximum(2.0 * bin_hz, params.ridge_relative_bandwidth * np.maximum(target, 0.0))
            harmonic_stack += _ridge_band_energy(context.stft_power, context.stft_freqs, target, hw)
    features["H_stack"] = float(harmonic_stack / (context.total_energy_tf + eps))

    # R_h：f1/f2 脊线固定 ±1 kHz 邻域带积分，与 R_2_1（相对带宽）互补
    if len(context.ridge_f1) and context.stft_power.size:
        fixed_hw = params.ridge_fixed_band_hz
        eh1 = _ridge_band_energy(context.stft_power, context.stft_freqs, context.ridge_f1, fixed_hw)
        eh2 = _ridge_band_energy(context.stft_power, context.stft_freqs, context.ridge_f2, fixed_hw)
    else:
        eh1, eh2 = 0.0, 0.0
    features["R_h"] = float(eh2 / (eh1 + eps))

    features["epsilon_2x"] = float(np.median(diff_h2[active] / (context.ridge_f1[active] + eps))) if np.any(active) else 0.0
    features["C_h"] = float(np.mean(diff_h2[active])) if np.any(active) else 0.0
    features["R_harm"] = float(context.harmonic_energy / (context.total_energy_tf + eps))
    features["Ridge_coh"] = features["R_harm"]  # 兼容别名列：与 R_harm 同值，已停用（不参与模型选择）

    # rho_up/rho_down/N_turn：仅统计有效帧，且先剔除零斜率样本再数方向变化
    if len(context.ridge_f1) > 1 and context.stft_times.size > 1 and np.any(active):
        pair_valid = active[:-1] & active[1:]
        df = np.diff(context.ridge_f1)
        dt = np.diff(context.stft_times)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = df / np.maximum(dt, eps)
        slopes = slopes[pair_valid]
        slopes = slopes[np.isfinite(slopes)]
    else:
        slopes = np.zeros(0, dtype=float)
    slope_threshold = params.ridge_jump_penalty_hz
    features["rho_up"] = float(np.mean(slopes > slope_threshold)) if slopes.size else 0.0
    features["rho_down"] = float(np.mean(slopes < -slope_threshold)) if slopes.size else 0.0
    nonzero_slopes = slopes[slopes != 0.0]
    sign_changes = np.diff(np.sign(nonzero_slopes))
    features["N_turn"] = float(np.sum(sign_changes != 0)) if sign_changes.size else 0.0

    ridge_valid_f1 = context.ridge_f1[active]
    features["Delta_f_span"] = float(np.max(ridge_valid_f1) - np.min(ridge_valid_f1)) if ridge_valid_f1.size else 0.0
    curvature = np.gradient(np.gradient(context.ridge_f1, context.stft_times + eps), context.stft_times + eps) if len(context.ridge_f1) > 2 else np.zeros_like(context.ridge_f1)
    curvature_valid = curvature[active]
    features["C_f"] = float(np.mean(np.abs(curvature_valid) / (np.mean(np.abs(ridge_valid_f1)) + eps))) if curvature_valid.size and ridge_valid_f1.size else 0.0

    features["E_harm"] = float(context.harmonic_energy)
    features["E_res"] = float(np.sum(context.residual_power))
    features["rho_res"] = float(features["E_res"] / (context.total_energy_tf + eps))
    high_res_energy = float(np.sum(context.residual_power[_band_mask(context.stft_freqs, params.harmonic_band_hz), :]))
    features["R_high_res"] = float(high_res_energy / (context.total_energy_tf + eps))
    features["epsilon_rec"] = float(np.sum((signal_values - context.reconstructed_signal) ** 2) / (np.sum(signal_values ** 2) + eps))
    residual_ratio_per_frame = np.sum(context.residual_power, axis=0) / (np.sum(context.stft_power, axis=0) + eps) if context.residual_power.size else np.zeros(0)
    features["N_abn"] = float(np.sum(residual_ratio_per_frame > params.residual_abnormal_threshold))

    # F_peak：逐频点功率正增量（半波整流）后求和；prepend 用首帧自身使首帧通量=0
    if context.short_power.size:
        flux_diff = np.diff(context.short_power, axis=1, prepend=context.short_power[:, :1])
        spectral_flux = np.maximum(flux_diff, 0.0).sum(axis=0)
    else:
        spectral_flux = np.zeros(0, dtype=float)
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

    # T_half_high：高频包络峰后首次跌破 50% 的"衰减时长"（峰值起算）；从未跌破返回 NaN
    high_env = smooth_envelope(context.high2_signal, fs, params.envelope_smooth_ms)
    high_peak = int(np.argmax(high_env)) if high_env.size else 0
    half_level = 0.5 * float(np.max(high_env)) if high_env.size else 0.0
    below = np.flatnonzero(high_env[high_peak:] <= half_level) if high_env.size else np.zeros(0, dtype=int)
    if below.size:
        features["T_half_high"] = float(below[0] / fs)
    else:
        features["T_half_high"] = float("nan")

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

    # D_WPT：按"子带中心频率 > 主带几何中点"划分 HF/LF 节点集合
    if context.wavelet_node_energies:
        main_lo, main_hi = params.main_band_hz
        geo_mid = math.sqrt(max(main_lo, 1e-9) * max(main_hi, 1e-9))
        hf_energy = 0.0
        for node_name, energy in context.wavelet_node_energies.items():
            if _node_center_hz(node_name, fs) > geo_mid:
                hf_energy += energy
        lf_energy = max(total_wp - hf_energy, 0.0)
        features["D_WPT"] = float((hf_energy - lf_energy) / total_wp)
    else:
        features["D_WPT"] = 0.0

    c_damp, best_atom = _damped_atom_match(residual, fs, params.damped_freqs_hz, params.damped_decay_ms, eps)
    features["C_damp"] = c_damp
    residual_env = np.maximum(residual_env, eps)
    # alpha_hat：残差包络对数在 峰值→事件终点 的峰后窗拟合（衰减常数），避免上升段污染
    seg_end = max(peak_idx + 1, min(offset_idx, len(residual_env) - 1))
    if seg_end - peak_idx >= 2:
        features["alpha_hat"] = float(-_line_slope(time_axis[peak_idx:seg_end + 1], np.log(residual_env[peak_idx:seg_end + 1])))
    else:
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
