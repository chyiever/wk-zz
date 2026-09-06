"""Shared preprocessing and signal-analysis utilities (v2.2: GPU STFT + batched fix).

v2.2 优化：修复 batched STFT 输出维度注释和 n_frames 计算。
torch.stft 批量输入 (N, L) 返回 (N, n_freq, n_time)，非 (n_freq, n_time, N)。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pywt
from scipy import signal

from .base import FeatureContext, FeatureParams, FeatureRecord

# GPU STFT 后端（惰性初始化）
_torch_available = False
_torch = None
_device = None


def _init_gpu_stft():
    """惰性初始化 torch GPU 后端。"""
    global _torch_available, _torch, _device
    if _torch is not None:
        return _torch_available
    flag = os.getenv("FEA_CPT_USE_GPU", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        _torch_available = False
        _torch = None
        _device = None
        return _torch_available
    try:
        import torch
        _torch = torch
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            _torch_available = True
        else:
            _device = torch.device('cpu')
            _torch_available = False
    except ImportError:
        _torch_available = False
        _torch = None
    return _torch_available


_init_gpu_stft()


def safe_divide(numerator: float | np.ndarray, denominator: float | np.ndarray, eps: float) -> float | np.ndarray:
    """Numerically safe division."""
    return np.asarray(numerator) / (np.asarray(denominator) + eps)


def robust_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    """Median/MAD normalization."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad + eps
    return (values - median) / scale


def butter_filter(values: np.ndarray, sample_rate: float, band_hz: tuple[float, float], order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter."""
    low_hz, high_hz = band_hz
    nyq = 0.5 * sample_rate
    low = max(0.0, low_hz) / nyq
    high = min(high_hz, nyq * 0.999) / nyq
    if low <= 0.0:
        sos = signal.butter(order, high, btype="lowpass", output="sos")
    elif high >= 0.999:
        sos = signal.butter(order, low, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, values)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average."""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def smooth_envelope(values: np.ndarray, sample_rate: float, smooth_ms: float) -> np.ndarray:
    """Hilbert-envelope plus moving average smoothing."""
    envelope = np.abs(signal.hilbert(values))
    window = max(1, int(round(sample_rate * smooth_ms / 1_000.0)))
    return moving_average(envelope, window)


def estimate_bounds_from_energy(envelope: np.ndarray, q_low: float, q_high: float) -> tuple[int, int]:
    """Estimate onset/offset from cumulative envelope energy."""
    energy = np.maximum(envelope, 0.0) ** 2
    cumulative = np.cumsum(energy)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    if total <= 0.0:
        return 0, max(0, len(envelope) - 1)
    low_idx = int(np.searchsorted(cumulative, q_low * total))
    high_idx = int(np.searchsorted(cumulative, q_high * total))
    return low_idx, min(max(low_idx + 1, high_idx), len(envelope) - 1)


def fit_bulge_envelope(envelope: np.ndarray, peak_index: int, eps: float) -> np.ndarray:
    """Piecewise Gaussian bulge fit used by envelope mismatch features."""
    n = len(envelope)
    x = np.arange(n, dtype=float)
    weights = np.maximum(envelope, eps)
    left_var = np.average((x[: peak_index + 1] - peak_index) ** 2, weights=weights[: peak_index + 1])
    right_var = np.average((x[peak_index:] - peak_index) ** 2, weights=weights[peak_index:])
    left_sigma = max(math.sqrt(max(left_var, eps)), 1.0)
    right_sigma = max(math.sqrt(max(right_var, eps)), 1.0)
    fit = np.empty_like(envelope, dtype=float)
    fit[:peak_index] = envelope[peak_index] * np.exp(-0.5 * ((x[:peak_index] - peak_index) / left_sigma) ** 2)
    fit[peak_index:] = envelope[peak_index] * np.exp(-0.5 * ((x[peak_index:] - peak_index) / right_sigma) ** 2)
    return fit


def _stft_gpu(values: np.ndarray, sample_rate: float, window_ms: float, overlap: float, nfft: int | None, batched: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated STFT using torch.stft.

    将信号转到 GPU，执行 batched STFT，结果转回 numpy。
    """
    import torch

    nperseg = max(16, int(round(sample_rate * window_ms / 1_000.0)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap)))
    if nfft is None:
        nfft = int(2 ** math.ceil(math.log2(nperseg)))

    # 构建 Hann 窗
    window_tensor = torch.hann_window(nperseg, periodic=True, device=_device)

    if batched:
        # values shape: (n_windows, n_samples_per_window)
        # 转为 GPU tensor
        x = torch.from_numpy(values.astype(np.float32)).to(_device)
        
        # torch.stft 批量输入 (N, L) 返回 (N, n_freq, n_time)
        spec = torch.stft(
            x,
            n_fft=nfft,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            window=window_tensor,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # 转回 CPU numpy: (n_windows, n_freq, n_time)
        spec_cpu = spec.cpu().numpy()
        power = np.abs(spec_cpu) ** 2

        # 频率轴（不变）
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        # 时间轴（不变）
        hop_length = nperseg - noverlap
        n_frames = spec_cpu.shape[-1]
        times = (np.arange(n_frames) * hop_length + nperseg // 2) / sample_rate
        
        return freqs, times, spec_cpu, power
    else:
        # 单信号路径（保持原逻辑）
        # 信号转 GPU tensor
        x = torch.from_numpy(values.astype(np.float32)).to(_device)

        # torch.stft 返回复数频谱 (n_freq, n_time)
        spec = torch.stft(
            x,
            n_fft=nfft,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            window=window_tensor,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # 转回 CPU numpy
        spec_cpu = spec.cpu().numpy()
        power = np.abs(spec_cpu) ** 2

        # 频率轴
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        # 时间轴
        hop_length = nperseg - noverlap
        n_frames = spec_cpu.shape[-1]
        times = (np.arange(n_frames) * hop_length + nperseg // 2) / sample_rate

        return freqs, times, spec_cpu, power


def compute_stft_power(values: np.ndarray, sample_rate: float, window_ms: float, overlap: float, nfft: int | None, batched: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """STFT power spectrum.

    v2: 优先使用 GPU (torch.stft)，回退到 scipy。
    """
    if _torch_available:
        return _stft_gpu(values, sample_rate, window_ms, overlap, nfft, batched=batched)
    # Fallback: scipy CPU path
    nperseg = max(16, int(round(sample_rate * window_ms / 1_000.0)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap)))
    if nfft is None:
        nfft = int(2 ** math.ceil(math.log2(nperseg)))
    freqs, times, complex_spec = signal.stft(
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
    return freqs, times, complex_spec, power


def _dynamic_programming_ridge(freqs: np.ndarray, power: np.ndarray, search_hz: tuple[float, float], jump_penalty_hz: float, prior_hz: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract a smooth ridge using dynamic programming."""
    mask = (freqs >= search_hz[0]) & (freqs <= search_hz[1])
    freq_subset = freqs[mask]
    power_subset = power[mask]
    n_freq, n_time = power_subset.shape
    if n_freq == 0 or n_time == 0:
        return np.zeros(power.shape[1], dtype=float), np.zeros(power.shape[1], dtype=int)

    score = np.log(power_subset + 1e-12)
    if prior_hz is not None:
        prior_penalty = -np.abs(freq_subset[:, None] - prior_hz[None, :]) / max(jump_penalty_hz, 1.0)
        score = score + prior_penalty

    dp = np.empty_like(score)
    back = np.zeros_like(score, dtype=int)
    dp[:, 0] = score[:, 0]
    freq_gap = np.abs(freq_subset[:, None] - freq_subset[None, :]) / max(jump_penalty_hz, 1.0)
    for t in range(1, n_time):
        candidate = dp[:, t - 1][None, :] - freq_gap
        back[:, t] = np.argmax(candidate, axis=1)
        dp[:, t] = score[:, t] + candidate[np.arange(n_freq), back[:, t]]

    ridge_local = np.zeros(n_time, dtype=int)
    ridge_local[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n_time - 1, 0, -1):
        ridge_local[t - 1] = back[ridge_local[t], t]
    ridge_global = np.flatnonzero(mask)[ridge_local]
    return freqs[ridge_global], ridge_global


def build_ridge_mask(freqs: np.ndarray, ridge_f1: np.ndarray, ridge_f2: np.ndarray, relative_bandwidth: float) -> np.ndarray:
    """Build harmonic mask around primary and second-harmonic ridges."""
    mask = np.zeros((len(freqs), len(ridge_f1)), dtype=np.float32)
    if len(freqs) == 0:
        return mask
    bin_hz = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for t in range(len(ridge_f1)):
        for freq in (ridge_f1[t], ridge_f2[t]):
            if freq <= 0.0:
                continue
            bandwidth = max(2.0 * bin_hz, relative_bandwidth * freq)
            idx = np.abs(freqs - freq) <= bandwidth
            mask[idx, t] = 1.0
    return mask


def reconstruct_from_mask(complex_spec: np.ndarray, mask: np.ndarray, sample_rate: float, window_ms: float, overlap: float) -> np.ndarray:
    """Inverse-STFT reconstruction from a time-frequency mask."""
    nperseg = max(16, int(round(sample_rate * window_ms / 1_000.0)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap)))
    _, reconstructed = signal.istft(
        complex_spec * mask,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    return np.asarray(reconstructed, dtype=float)


def compute_wavelet_node_energies(values: np.ndarray, wavelet_name: str, level: int) -> dict[str, float]:
    """Wavelet packet terminal node energies."""
    packet = pywt.WaveletPacket(data=values, wavelet=wavelet_name, mode="symmetric", maxlevel=level)
    nodes = packet.get_level(level, order="freq")
    return {node.path: float(np.sum(np.asarray(node.data, dtype=float) ** 2)) for node in nodes}


def build_context(record: FeatureRecord, params: FeatureParams) -> FeatureContext:
    """Compute all reusable intermediates for one record."""
    raw = np.asarray(record.signal, dtype=float)
    detrended = signal.detrend(raw - np.mean(raw))
    highpassed = butter_filter(detrended, record.sample_rate, (params.highpass_hz, record.sample_rate * 0.49))
    preprocessed = robust_normalize(highpassed, params.eps) if params.normalize_robust else highpassed
    main_signal = butter_filter(preprocessed, record.sample_rate, params.main_band_hz)
    high_signal = butter_filter(preprocessed, record.sample_rate, params.high1_band_hz)
    high2_signal = butter_filter(preprocessed, record.sample_rate, params.high2_band_hz)
    envelope = smooth_envelope(main_signal, record.sample_rate, params.envelope_smooth_ms)
    peak_index = int(np.argmax(envelope)) if envelope.size else 0
    onset_index, offset_index = estimate_bounds_from_energy(envelope, params.onset_quantile, params.offset_quantile)
    envelope_fit = fit_bulge_envelope(envelope, peak_index, params.eps)

    stft_freqs, stft_times, stft_complex, stft_power = compute_stft_power(main_signal, record.sample_rate, params.stft_window_ms, params.stft_overlap, params.stft_nfft)
    short_freqs, short_times, _, short_power = compute_stft_power(main_signal, record.sample_rate, params.short_stft_window_ms, params.short_stft_overlap, params.short_stft_nfft)

    ridge_f1, ridge_idx_f1 = _dynamic_programming_ridge(stft_freqs, stft_power, params.ridge_main_search_hz, params.ridge_jump_penalty_hz)
    ridge_f2, ridge_idx_f2 = _dynamic_programming_ridge(stft_freqs, stft_power, params.ridge_h2_search_hz, params.ridge_jump_penalty_hz, prior_hz=2.0 * ridge_f1 if ridge_f1.size else None)
    ridge_mask = build_ridge_mask(stft_freqs, ridge_f1, ridge_f2, params.ridge_relative_bandwidth)
    residual_power = stft_power * (1.0 - ridge_mask)
    total_energy_tf = float(np.sum(stft_power))
    harmonic_energy = float(np.sum(stft_power * ridge_mask))
    reconstructed = reconstruct_from_mask(stft_complex, ridge_mask, record.sample_rate, params.stft_window_ms, params.stft_overlap)
    if reconstructed.size < main_signal.size:
        reconstructed = np.pad(reconstructed, (0, main_signal.size - reconstructed.size))
    reconstructed = reconstructed[: main_signal.size]
    residual_signal = main_signal - reconstructed
    node_energies = compute_wavelet_node_energies(main_signal, params.wavelet_name, params.wavelet_level)

    return FeatureContext(
        record=record,
        params=params,
        raw_signal=raw,
        preprocessed_signal=preprocessed,
        main_signal=main_signal,
        high_signal=high_signal,
        high2_signal=high2_signal,
        envelope=envelope,
        envelope_fit=envelope_fit,
        envelope_peak_index=peak_index,
        onset_index=onset_index,
        offset_index=offset_index,
        stft_freqs=stft_freqs,
        stft_times=stft_times,
        stft_complex=stft_complex,
        stft_power=stft_power,
        short_freqs=short_freqs,
        short_times=short_times,
        short_power=short_power,
        ridge_f1=ridge_f1,
        ridge_f2=ridge_f2,
        ridge_idx_f1=ridge_idx_f1,
        ridge_idx_f2=ridge_idx_f2,
        ridge_mask=ridge_mask,
        harmonic_energy=harmonic_energy,
        total_energy_tf=total_energy_tf,
        residual_power=residual_power,
        reconstructed_signal=reconstructed,
        residual_signal=residual_signal,
        wavelet_node_energies=node_energies,
    )
