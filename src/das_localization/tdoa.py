"""TDOA 估计模块：参考通道选择、互相关、GCC-PHAT、PSR筛选。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DelayResult:
    delays: np.ndarray
    psr: np.ndarray
    peak: np.ndarray
    valid_mask: np.ndarray
    method: str
    ref_idx: int


def compute_snr_proxy(x: np.ndarray, q: float = 0.2) -> float:
    """用能量分位近似 SNR：峰值能量 / 背景能量。"""
    x = np.asarray(x, dtype=np.float64)
    p = x**2
    hi = np.quantile(p, 1.0 - q)
    lo = np.quantile(p, q)
    return float((hi + 1e-12) / (lo + 1e-12))


def select_reference_channel(data_fc: np.ndarray, mode: str = "best_snr", fixed_idx: int = 0) -> int:
    """选择参考通道，输入 shape=(frames, channels)。"""
    channels = data_fc.shape[1]
    if mode == "fixed":
        if not 0 <= fixed_idx < channels:
            raise ValueError("fixed_idx 超出通道范围")
        return int(fixed_idx)

    snr_list = [compute_snr_proxy(data_fc[:, c]) for c in range(channels)]
    return int(np.argmax(snr_list))


def _parabolic_subsample(cc: np.ndarray, idx: int) -> float:
    """抛物线插值，返回亚采样偏移量。"""
    if idx <= 0 or idx >= len(cc) - 1:
        return 0.0
    y1, y2, y3 = cc[idx - 1], cc[idx], cc[idx + 1]
    denom = 2.0 * (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-12:
        return 0.0
    return float((y1 - y3) / denom)


def _compute_psr(cc: np.ndarray, peak_idx: int, guard: int = 3) -> float:
    """计算 Peak-to-Sidelobe Ratio。"""
    abs_cc = np.abs(cc)
    peak_val = abs_cc[peak_idx]
    mask = np.ones_like(abs_cc, dtype=bool)
    left = max(0, peak_idx - guard)
    right = min(abs_cc.size, peak_idx + guard + 1)
    mask[left:right] = False
    sidelobes = abs_cc[mask]
    if sidelobes.size == 0:
        return 0.0
    return float((peak_val + 1e-12) / (np.mean(sidelobes) + 1e-12))


def _xcorr(sig: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return signal.correlate(sig, ref, mode="full", method="fft")


def _gcc_phat(sig: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = len(sig) + len(ref)
    spec_sig = np.fft.rfft(sig, n=n)
    spec_ref = np.fft.rfft(ref, n=n)
    cross = spec_sig * np.conj(spec_ref)
    cross /= np.maximum(np.abs(cross), eps)
    cc = np.fft.irfft(cross, n=n)
    cc = np.concatenate((cc[-(n // 2):], cc[: n // 2 + (n % 2)]))
    return cc


def estimate_delays(
    data_fc: np.ndarray,
    fs: float,
    ref_idx: int,
    method: str = "gcc_phat",
    max_tau_sec: float | None = None,
    psr_th: float = 6.0,
) -> DelayResult:
    """估计所有通道相对参考通道的 TDOA。"""
    frames, channels = data_fc.shape
    ref = data_fc[:, ref_idx]
    delays = np.zeros(channels, dtype=np.float64)
    psr_arr = np.zeros(channels, dtype=np.float64)
    peak_arr = np.zeros(channels, dtype=np.float64)
    valid = np.ones(channels, dtype=bool)

    for c in range(channels):
        sig = data_fc[:, c]
        if method == "gcc_phat":
            cc = _gcc_phat(sig, ref)
            lags = np.arange(-len(cc) // 2, len(cc) - len(cc) // 2)
        elif method in {"ncc", "fft_xcorr"}:
            cc = _xcorr(sig, ref)
            lags = np.arange(-(frames - 1), frames)
            if method == "ncc":
                denom = (np.linalg.norm(sig) * np.linalg.norm(ref) + 1e-12)
                cc = cc / denom
        else:
            raise ValueError(f"未知 method: {method}")

        if max_tau_sec is not None:
            max_lag = int(max_tau_sec * fs)
            m = (lags >= -max_lag) & (lags <= max_lag)
            cc = cc[m]
            lags = lags[m]

        pk = int(np.argmax(np.abs(cc)))
        delta = _parabolic_subsample(np.abs(cc), pk)
        lag = lags[pk] + delta
        delays[c] = lag / fs
        peak_arr[c] = float(np.abs(cc[pk]))
        psr_arr[c] = _compute_psr(cc, pk)

    delays[ref_idx] = 0.0
    valid = psr_arr >= psr_th
    valid[ref_idx] = True
    return DelayResult(delays=delays, psr=psr_arr, peak=peak_arr, valid_mask=valid, method=method, ref_idx=ref_idx)


def apply_physical_tau_window(delays: np.ndarray, max_tau_sec: float) -> np.ndarray:
    """物理时延窗筛选。"""
    return np.abs(delays) <= max_tau_sec
