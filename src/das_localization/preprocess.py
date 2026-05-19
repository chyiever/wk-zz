"""预处理模块：去均值、滤波、归一化。"""

from __future__ import annotations

import numpy as np
from scipy import signal


def remove_dc(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """按通道去均值。默认数据 shape=(frames, channels)，沿时间轴去均值。"""
    data = np.asarray(data, dtype=np.float64)
    return data - np.mean(data, axis=axis, keepdims=True)


def highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float = 100.0,
    order: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """零相位高通滤波。"""
    nyquist = fs / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError("高通截止频率必须位于 (0, Nyquist)")
    b, a = signal.butter(order, cutoff_hz / nyquist, btype="high")
    return signal.filtfilt(b, a, np.asarray(data, dtype=np.float64), axis=axis)


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    """零相位带通滤波。"""
    nyquist = fs / 2.0
    if not 0 < lowcut_hz < highcut_hz < nyquist:
        raise ValueError("带通频率必须满足 0 < low < high < Nyquist")
    b, a = signal.butter(order, [lowcut_hz / nyquist, highcut_hz / nyquist], btype="band")
    return signal.filtfilt(b, a, np.asarray(data, dtype=np.float64), axis=axis)


def normalize_channels(data: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """按通道标准化，降低增益差异。"""
    data = np.asarray(data, dtype=np.float64)
    std = np.std(data, axis=axis, keepdims=True)
    return data / np.maximum(std, eps)
