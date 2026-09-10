"""滑窗批量特征提取模块（v2.2：持久化进程池 + 共享内存 + GPU集中化STFT + 文件级流水线）。

用于对连续真实数据（npz/tdms）进行滑窗全量特征提取。
支持多文件夹批量处理、200kHz/500kHz 统一采样率、窗级并行。

v2.2 优化（对比 v2.1）：
1. 持久化进程池：ProcessPoolExecutor 全生命周期复用，不再每 batch 重建
2. 共享内存：signal_pre 存入 multiprocessing.shared_memory，worker 零拷贝读取
3. GPU STFT 集中化批处理：主进程 batched GPU STFT → 共享内存 → worker 纯 CPU
4. 文件级流水线：ThreadPoolExecutor 后台预加载下一文件，I/O 与计算重叠
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import sys
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from math import gcd
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from tqdm.auto import tqdm

from .base import FeatureContext, FeatureRecord
from .features import FEATURE_FAMILIES, compute_all_features, feature_names_for_families
from .gpu_backend import gpu_backend_info
from .params import DEFAULT_FEATURE_PARAMS
from .signal_ops import active_stft_backend, build_context, butter_filter, compute_stft_power, inverse_stft, robust_normalize, infer_wavelet_level


def _default_feature_families(
    band_name: str,
    band_hz: tuple[float, float],
    harmonic_band_name: str,
) -> tuple[str, ...]:
    """Assign physically applicable feature families instead of an 80-feature Cartesian product."""
    low, high = band_hz
    if band_name == harmonic_band_name:
        return tuple(FEATURE_FAMILIES)
    if high <= 1_500.0:
        # A 0.64 ms STFT at 1 MHz cannot resolve 100 Hz-1 kHz.  Keep waveform/background
        # summaries here; low-frequency spectral features require the planned long-window group.
        return ("classic", "time", "background")
    if high <= 15_000.0:
        return ("classic", "time", "spectral", "entropy", "cepstral", "ridge", "background")
    if low >= 30_000.0:
        return ("classic", "time", "spectral", "entropy", "cepstral", "residual", "background")
    return ("classic", "time", "spectral", "entropy", "cepstral", "ridge", "residual", "background")


def build_feature_request_map(
    params_map: dict[str, Any],
    families_by_band: dict[str, tuple[str, ...]] | None = None,
    harmonic_band_name: str | None = None,
) -> tuple[dict[str, frozenset[str]], str]:
    """Build per-band feature allowlists and choose one explicit cross-frequency harmonic context."""
    if not params_map:
        raise ValueError("params_map is empty")
    if harmonic_band_name is None:
        harmonic_band_name = max(
            params_map,
            key=lambda name: params_map[name].main_band_hz[1] - params_map[name].main_band_hz[0],
        )
    if harmonic_band_name not in params_map:
        raise ValueError(f"harmonic_band_name not found: {harmonic_band_name}")
    requests: dict[str, frozenset[str]] = {}
    for name, params in params_map.items():
        families = (
            families_by_band[name]
            if families_by_band is not None and name in families_by_band
            else _default_feature_families(name, params.main_band_hz, harmonic_band_name)
        )
        if name != harmonic_band_name and "harmonic" in families:
            raise ValueError(
                f"harmonic family may only be emitted from {harmonic_band_name!r}; got {name!r}"
            )
        requests[name] = feature_names_for_families(families)
    return requests, harmonic_band_name


FEATURE_SCHEMA_VERSION = "pccp-v6-band100k-no-snr-gate-wpt-auto-20260910"
ALLOWED_NAN_BASE_FEATURES = frozenset({
    "T_half_high", "alpha_hat", "beta_H",
})


def expected_output_features(
    feature_requests: dict[str, frozenset[str]],
    wavelet_level: int | dict[str, int],
) -> frozenset[str]:
    """Return the exact required feature schema, including dynamic WPT node columns."""
    expected: set[str] = set()
    for band_name, requested in feature_requests.items():
        level = wavelet_level.get(band_name, 4) if isinstance(wavelet_level, dict) else wavelet_level
        all_node_paths = tuple("".join(chars) for chars in product("ad", repeat=level))
        expected.update(f"{band_name}__{name}" for name in requested)
        if "H_wp" in requested:
            expected.update(f"{band_name}__R_wp_{path}" for path in all_node_paths)
    return frozenset(expected)


def validate_completed_file(
    rows_features: list[dict[str, object]],
    rows_log: list[dict[str, object]],
    windows: list[tuple[int, int, int, int, int]],
    feature_requests: dict[str, frozenset[str]],
    wavelet_level: int | dict[str, int],
) -> list[str]:
    """Strict success gate used before CSV output and processed-log updates."""
    errors: list[str] = []
    expected_ids = {int(window[0]) for window in windows}
    actual_ids = {int(row.get("window_id", -1)) for row in rows_features}
    if len(rows_features) != len(windows) or actual_ids != expected_ids:
        errors.append(f"window_count_or_ids expected={len(windows)} actual={len(rows_features)}")
    if len(rows_log) != len(windows):
        errors.append(f"log_count expected={len(windows)} actual={len(rows_log)}")
    worker_errors = [str(row.get("missing_selected_features", "")).strip() for row in rows_log]
    worker_errors = [value for value in worker_errors if value]
    if worker_errors:
        errors.append(f"worker_errors={len(worker_errors)} first={worker_errors[0][:160]}")

    expected = expected_output_features(feature_requests, wavelet_level)
    for row in rows_features:
        missing = expected - set(row)
        if missing:
            errors.append(f"window={row.get('window_id')} missing_features={len(missing)} first={sorted(missing)[:3]}")
            break
        for column in expected:
            value = float(row[column])
            base = column.split("__", 1)[-1]
            if np.isinf(value) or (np.isnan(value) and base not in ALLOWED_NAN_BASE_FEATURES):
                errors.append(f"window={row.get('window_id')} invalid_value={column}:{value}")
                break
        if errors:
            break
    return errors

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None  # type: ignore[assignment]

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def _auto_detect_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 2)


def _bind_numa() -> bool:
    if psutil is None:
        return False
    try:
        p = psutil.Process()
        all_cores = list(range(os.cpu_count() or 4))
        current = p.cpu_affinity()
        if current != all_cores:
            p.cpu_affinity(all_cores)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 共享内存工具
# ---------------------------------------------------------------------------

class _SharedArrayPack:
    """将多个 numpy 数组打包到一块连续共享内存中。"""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self._meta: dict[str, dict[str, Any]] = {}
        offset = 0
        for name, arr in arrays.items():
            aligned_offset = (offset + 63) & ~63
            self._meta[name] = {
                'shape': tuple(arr.shape),
                'dtype': str(arr.dtype),
                'offset': aligned_offset,
                'nbytes': arr.nbytes,
            }
            offset = aligned_offset + arr.nbytes

        total_size = offset
        self._shm = SharedMemory(create=True, size=total_size)

        for name, arr in arrays.items():
            m = self._meta[name]
            dst = np.ndarray(
                arr.shape, dtype=arr.dtype,
                buffer=self._shm.buf[m['offset']:m['offset'] + m['nbytes']],
            )
            dst[:] = arr

    def get_info(self) -> dict:
        return {
            'shm_name': self._shm.name,
            'meta': self._meta,
        }

    def cleanup(self):
        self._shm.close()
        self._shm.unlink()


def _attach_shared(info: dict) -> tuple[dict[str, np.ndarray], SharedMemory]:
    """附加到共享内存并重建 numpy 数组视图（worker 端调用）。"""
    shm = SharedMemory(name=info['shm_name'])
    arrays: dict[str, np.ndarray] = {}
    for name, m in info['meta'].items():
        arrays[name] = np.ndarray(
            m['shape'], dtype=np.dtype(m['dtype']),
            buffer=shm.buf[m['offset']:m['offset'] + m['nbytes']],
        )
    return arrays, shm


# ---------------------------------------------------------------------------
# Worker 局部缓存：同一文件的共享内存只附加一次
# ---------------------------------------------------------------------------

_worker_shm_cache: dict[str, tuple[dict[str, np.ndarray], SharedMemory]] = {}


def _worker_get_arrays(shm_info: dict) -> tuple[dict[str, np.ndarray], SharedMemory]:
    key = shm_info['shm_name']
    if key in _worker_shm_cache:
        return _worker_shm_cache[key]
    # 清理旧的缓存条目
    for old_key, (_, old_shm) in list(_worker_shm_cache.items()):
        try:
            old_shm.close()
        except Exception:
            pass
        del _worker_shm_cache[old_key]
    arrays, shm = _attach_shared(shm_info)
    _worker_shm_cache[key] = (arrays, shm)
    return arrays, shm


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _scalar_text(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip()
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, 'tolist') and not isinstance(value, str):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar_text(value[0])
    return str(value).strip()


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        if arr.size == 1:
            return float(arr.reshape(()).item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _first_property(props: dict[str, object], names: tuple[str, ...]) -> object | None:
    normalized = {str(k).lower(): v for k, v in props.items()}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _infer_sample_rate_from_filename(path: Path) -> float | None:
    stem = path.stem
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(k|m)?\s*hz(?![a-zA-Z])', stem, flags=re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or '').lower()
        if unit == 'k':
            return val * 1_000.0
        if unit == 'm':
            return val * 1_000_000.0
        return val
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*([kKmM])(?![a-zA-Z])', stem)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'k':
            return val * 1_000.0
        if unit == 'm':
            return val * 1_000_000.0
    return None


def _load_npz_source(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        signal_values = np.asarray(data['phase_data'], dtype=float)
        sample_rate = float(np.asarray(data['sample_rate']).item())
        starttime_raw = _scalar_text(data.get('starttime', '')) if 'starttime' in data else ''
        arrival_time_raw = _scalar_text(data.get('arrival_time', '')) if 'arrival_time' in data else ''
        sample_type = _scalar_text(data.get('type', path.parent.name)) if 'type' in data else path.parent.name
    return {
        'source_format': 'npz',
        'signal_values': signal_values,
        'sample_rate': sample_rate,
        'starttime_raw': starttime_raw,
        'arrival_time_raw': arrival_time_raw,
        'sample_type': sample_type,
        'source_group_name': '',
        'source_channel_name': '',
        'source_detail': '',
    }


def _select_tdms_channel(tdms_file):
    if TdmsFile is None:
        raise ImportError('nptdms is required for .tdms files. Install with: pip install nptdms')
    for g in tdms_file.groups():
        for c in g.channels():
            if str(c.name).lower() in {'phase_data', 'signal', 'data', 'values', 'channel0', 'ch0'}:
                return g, c
    best = None
    best_len = -1
    for g in tdms_file.groups():
        for c in g.channels():
            try:
                arr = np.asarray(c[:])
                if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                    continue
            except Exception:
                continue
            if arr.size > best_len:
                best_len = arr.size
                best = (g, c)
    if best is None:
        raise ValueError('No usable numeric channel found in TDMS')
    return best


def _load_tdms_source(path: Path, fallback_sample_rate: float | None = None, channel_name: str | None = None) -> dict[str, object]:
    td = TdmsFile.read(path)
    
    # 如果指定了通道名称，尝试查找指定通道
    if channel_name is not None:
        target_channel = None
        target_group = None
        for g in td.groups():
            for c in g.channels():
                if str(c.name).lower() == channel_name.lower():
                    target_channel = c
                    target_group = g
                    break
            if target_channel is not None:
                break
        
        if target_channel is not None:
            g, c = target_group, target_channel
        else:
            # 如果指定通道不存在，回退到默认选择逻辑
            g, c = _select_tdms_channel(td)
    else:
        g, c = _select_tdms_channel(td)
    
    signal_values = np.asarray(c[:], dtype=float)
    props = {}
    props.update(getattr(td, 'properties', {}) or {})
    props.update(getattr(g, 'properties', {}) or {})
    props.update(getattr(c, 'properties', {}) or {})
    sample_rate = _coerce_float(_first_property(props, ('sample_rate', 'sample_rate_hz', 'sampling_rate', 'sampling_rate_hz')))
    if sample_rate is None:
        wf_inc = _coerce_float(_first_property(props, ('wf_increment',)))
        if wf_inc and wf_inc > 0:
            sample_rate = 1.0 / wf_inc
    if sample_rate is None or sample_rate <= 0:
        sample_rate = _infer_sample_rate_from_filename(path)
    if (sample_rate is None or sample_rate <= 0) and fallback_sample_rate is not None:
        sample_rate = float(fallback_sample_rate)
    if sample_rate is None or sample_rate <= 0:
        raise ValueError(f'Cannot infer sample rate from TDMS file: {path}')
    starttime_raw = _scalar_text(_first_property(props, ('starttime', 'start_time', 'wf_start_time')))
    arrival_time_raw = _scalar_text(_first_property(props, ('arrival_time', 'arrivaltime')))
    sample_type = _scalar_text(_first_property(props, ('type', 'sample_type'))) or path.parent.name
    return {
        'source_format': 'tdms',
        'signal_values': signal_values,
        'sample_rate': float(sample_rate),
        'starttime_raw': starttime_raw,
        'arrival_time_raw': arrival_time_raw,
        'sample_type': sample_type,
        'source_group_name': str(g.name),
        'source_channel_name': str(c.name),
        'source_detail': f'{g.name}/{c.name}',
    }


def load_source_file(path: Path, tdms_fallback_sample_rate: float | None = None, tdms_channel_name: str | None = None) -> dict[str, object]:
    suf = path.suffix.lower()
    if suf == '.npz':
        return _load_npz_source(path)
    if suf == '.tdms':
        return _load_tdms_source(path, tdms_fallback_sample_rate, tdms_channel_name)
    raise ValueError(f'Unsupported file type: {path.suffix}')


def parse_starttime(starttime_raw: str) -> datetime | None:
    if not starttime_raw:
        return None
    fmts = [
        '%Y%m%dT%H%M%S.%f', '%Y%m%dT%H%M%S',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(starttime_raw, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(starttime_raw.replace('Z', '+00:00'))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 升采样
# ---------------------------------------------------------------------------

def upsample_to_target(signal: np.ndarray, sample_rate: float, target_rate: float = 500_000.0) -> tuple[np.ndarray, float]:
    if resample_poly is None:
        raise ImportError('scipy is required for upsampling. Install with: pip install scipy')
    if sample_rate >= target_rate:
        return np.asarray(signal, dtype=float), float(sample_rate)
    g = gcd(int(round(target_rate)), int(round(sample_rate)))
    up = int(round(target_rate)) // g
    down = int(round(sample_rate)) // g
    if up == down:
        return np.asarray(signal, dtype=float), float(sample_rate)
    resampled = resample_poly(signal, up, down)
    return np.asarray(resampled, dtype=float), float(target_rate)


# ---------------------------------------------------------------------------
# 滑窗
# ---------------------------------------------------------------------------

def list_window_ranges(
    n_samples: int,
    sample_rate: float,
    window_duration_s: float,
    overlap: float,
) -> list[tuple[int, int, int, int, int]]:
    win = int(round(window_duration_s * sample_rate))
    if win <= 0:
        raise ValueError('window_samples must be positive')
    if n_samples < win:
        return []
    step = max(1, int(round(win * (1.0 - overlap))))
    out = []
    idx = 0
    wid = 0
    while idx + win <= n_samples:
        out.append((wid, idx, idx + win, win, step))
        idx += step
        wid += 1
    return out


# ---------------------------------------------------------------------------
# 频带参数构建
# ---------------------------------------------------------------------------

def _safe_band(low: float, high: float, nyq: float) -> tuple[float, float]:
    low = max(1.0, min(low, nyq * 0.98))
    high = max(low + 1.0, min(high, nyq * 0.995))
    return (float(low), float(high))


def build_params_for_band(band: tuple[float, float], sample_rate: float) -> Any:
    low, high = band
    nyq = sample_rate / 2.0
    low, high = _safe_band(low, high, nyq)
    span = max(high - low, 10.0)
    low_band = _safe_band(low, low + 0.30 * span, nyq)
    mid_band = _safe_band(low + 0.30 * span, low + 0.60 * span, nyq)
    high1_band = _safe_band(low + 0.50 * span, low + 0.80 * span, nyq)
    high2_band = _safe_band(low + 0.60 * span, high, nyq)
    harmonic_band = _safe_band(low + 0.50 * span, high, nyq)
    # Observable harmonic contract for the known 100 Hz-60 kHz PCCP range: f1 is meaningful up
    # to 30 kHz and f2 is searched in the same full context up to 60 kHz.
    ridge_main = _safe_band(low, min(low + 0.65 * span, 50_000.0, high), nyq)
    ridge_h2 = _safe_band(max(100.0, low * 2.0), min(high, 100_000.0, nyq * 0.995), nyq)
    effective_rate = min(float(sample_rate), max(4_000.0, 3.0 * high))
    wpt_level = infer_wavelet_level(effective_rate)
    return replace(
        DEFAULT_FEATURE_PARAMS,
        # The outer whole-file preprocessing already removes drift.  Keep the per-band setting
        # below the requested passband instead of deleting the known 100 Hz-1 kHz content.
        highpass_hz=100.0,
        main_band_hz=(low, high),
        low_band_hz=low_band,
        mid_band_hz=mid_band,
        high1_band_hz=high1_band,
        high2_band_hz=high2_band,
        harmonic_band_hz=harmonic_band,
        ridge_main_search_hz=ridge_main,
        ridge_h2_search_hz=ridge_h2,
        wavelet_level=wpt_level,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# 单窗口特征计算（与 v2.1 相同，保持特征一致性）
# ---------------------------------------------------------------------------

def compute_shared_stft(
    window_signal: np.ndarray,
    sample_rate: float,
    bands: list[tuple[str, tuple[float, float]]],
    params_map: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build safe per-band STFT contexts for one window.

    This compatibility helper intentionally does not crop one unnormalised wide-band STFT.  Each
    spectrum is computed from the same canonical band signal used by the time-domain features.
    The production batch path performs the same operation for many windows/bands at once.
    """
    detrended = sp_signal.detrend(np.asarray(window_signal, dtype=float) - np.mean(window_signal))
    reference_params = next(iter(params_map.values()))
    highpassed = butter_filter(detrended, sample_rate, (reference_params.highpass_hz, sample_rate * 0.49))
    preprocessed = robust_normalize(highpassed, reference_params.eps) if reference_params.normalize_robust else highpassed
    backend = "torch_cpu" if active_stft_backend() == "torch" else "scipy"
    shared: dict[str, dict[str, Any]] = {}
    for band_name, params in params_map.items():
        main_signal = butter_filter(preprocessed, sample_rate, params.main_band_hz)
        stft_freqs, stft_times, stft_complex, stft_power = compute_stft_power(
            main_signal, sample_rate, params.stft_window_ms, params.stft_overlap, params.stft_nfft,
        )
        short_freqs, short_times, _, short_power = compute_stft_power(
            main_signal, sample_rate, params.short_stft_window_ms, params.short_stft_overlap, params.short_stft_nfft,
        )
        low_hz, high_hz = params.main_band_hz
        mask = (stft_freqs >= low_hz) & (stft_freqs <= high_hz)
        mask_short = (short_freqs >= low_hz) & (short_freqs <= high_hz)
        band_complex = np.asarray(stft_complex).copy()
        band_power = np.asarray(stft_power, dtype=np.float64).copy()
        band_complex[~mask, :] = 0.0
        band_power[~mask, :] = 0.0
        band_short_power = np.asarray(short_power, dtype=np.float64).copy()
        band_short_power[~mask_short, :] = 0.0

        shared[band_name] = {
            'main_signal': main_signal,
            'preprocessed_signal': preprocessed,
            'stft_backend': backend,
            'stft_freqs': stft_freqs,
            'stft_times': stft_times,
            'stft_complex': band_complex,
            'stft_power': band_power,
            'short_freqs': short_freqs,
            'short_times': short_times,
            'short_power': band_short_power,
        }

    return shared


def compute_all_features_for_window(
    window_signal: np.ndarray,
    sample_rate: float,
    params_map: dict[str, Any],
    shared_stft: dict[str, dict[str, Any]] | None = None,
    feature_requests: dict[str, frozenset[str]] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if feature_requests is None:
        feature_requests, _ = build_feature_request_map(params_map)

    if shared_stft is not None:
        from .signal_ops import (
            _dynamic_programming_ridge,
            build_ridge_mask,
            reconstruct_from_mask,
            compute_wavelet_node_energies,
            robust_normalize,
            smooth_envelope,
            estimate_bounds_from_energy,
            fit_bulge_envelope,
        )
        from scipy import signal as sp_signal

        for band_name, params in params_map.items():
            s = shared_stft[band_name]
            preprocessed = np.asarray(s['preprocessed_signal'], dtype=float)
            main_signal = np.asarray(s['main_signal'], dtype=float)
            envelope = smooth_envelope(main_signal, sample_rate, params.envelope_smooth_ms)
            peak_index = int(np.argmax(envelope)) if envelope.size else 0
            onset_index, offset_index = estimate_bounds_from_energy(envelope, params.onset_quantile, params.offset_quantile)
            envelope_fit = fit_bulge_envelope(envelope, peak_index, params.eps)

            ridge_f1, ridge_idx_f1 = _dynamic_programming_ridge(
                s['stft_freqs'], s['stft_power'],
                params.ridge_main_search_hz, params.ridge_jump_penalty_hz,
            )
            request = feature_requests.get(band_name, frozenset())
            if request & FEATURE_FAMILIES['harmonic']:
                ridge_f2, ridge_idx_f2 = _dynamic_programming_ridge(
                    s['stft_freqs'], s['stft_power'],
                    params.ridge_h2_search_hz, params.ridge_jump_penalty_hz,
                    prior_hz=2.0 * ridge_f1 if ridge_f1.size else None,
                )
            else:
                ridge_f2 = np.zeros_like(ridge_f1)
                ridge_idx_f2 = np.zeros_like(ridge_idx_f1)

            ridge_mask_full = build_ridge_mask(s['stft_freqs'], ridge_f1, ridge_f2, params.ridge_relative_bandwidth)
            harmonic_complex = s['stft_complex'] * ridge_mask_full
            residual_complex = s['stft_complex'] * (1.0 - ridge_mask_full)
            total_energy_tf = float(np.sum(s['stft_power'], dtype=np.float64))
            harmonic_energy = float(np.sum(np.abs(np.asarray(harmonic_complex, dtype=np.complex128)) ** 2, dtype=np.float64))
            residual_power_band = np.abs(np.asarray(residual_complex, dtype=np.complex128)) ** 2

            reconstructed = inverse_stft(
                harmonic_complex, sample_rate, params.stft_window_ms, params.stft_overlap,
                backend=s['stft_backend'], length=main_signal.size,
            )
            residual_signal = inverse_stft(
                residual_complex, sample_rate, params.stft_window_ms, params.stft_overlap,
                backend=s['stft_backend'], length=main_signal.size,
            )
            if request & FEATURE_FAMILIES['wavelet']:
                node_energies, wavelet_sample_rate = compute_wavelet_node_energies(
                    main_signal, params.wavelet_name, params.wavelet_level,
                    sample_rate, params.main_band_hz, return_sample_rate=True,
                )
            else:
                node_energies, wavelet_sample_rate = {}, sample_rate

            high_signal = butter_filter(preprocessed, sample_rate, params.high1_band_hz)
            high2_signal = butter_filter(preprocessed, sample_rate, params.high2_band_hz)

            rec = FeatureRecord(
                sample_id='w', sample_name='w', sample_type='raw',
                sample_type_code=0, path=Path('.'),
                signal=np.asarray(window_signal, dtype=float),
                sample_rate=float(sample_rate), metadata={},
            )

            context = FeatureContext(
                record=rec, params=params,
                raw_signal=np.asarray(window_signal, dtype=float),
                preprocessed_signal=preprocessed,
                main_signal=main_signal,
                high_signal=high_signal,
                high2_signal=high2_signal,
                envelope=envelope,
                envelope_fit=envelope_fit,
                envelope_peak_index=peak_index,
                onset_index=onset_index,
                offset_index=offset_index,
                stft_freqs=s['stft_freqs'],
                stft_times=s['stft_times'],
                stft_complex=s['stft_complex'],
                stft_power=s['stft_power'],
                stft_backend=s['stft_backend'],
                short_freqs=s['short_freqs'],
                short_times=s['short_times'],
                short_power=s['short_power'],
                ridge_f1=ridge_f1,
                ridge_f2=ridge_f2,
                ridge_idx_f1=ridge_idx_f1,
                ridge_idx_f2=ridge_idx_f2,
                ridge_mask=ridge_mask_full,
                harmonic_energy=harmonic_energy,
                total_energy_tf=total_energy_tf,
                residual_power=residual_power_band,
                reconstructed_signal=reconstructed,
                residual_signal=residual_signal,
                wavelet_node_energies=node_energies,
                wavelet_sample_rate=wavelet_sample_rate,
                requested_features=feature_requests.get(band_name),
            )

            result = compute_all_features(context)
            for k, v in result.features.items():
                out[f'{band_name}__{k}'] = float(v)
    else:
        rec = FeatureRecord(
            sample_id='w', sample_name='w', sample_type='raw',
            sample_type_code=0, path=Path('.'),
            signal=np.asarray(window_signal, dtype=float),
            sample_rate=float(sample_rate), metadata={},
        )
        for band_name, params in params_map.items():
            context = build_context(rec, params)
            context.requested_features = feature_requests.get(band_name)
            result = compute_all_features(context)
            for k, v in result.features.items():
                out[f'{band_name}__{k}'] = float(v)

    return out


# ---------------------------------------------------------------------------
# v2.2 Worker 函数：从共享内存读取数据，纯 CPU 特征计算
# ---------------------------------------------------------------------------

def _worker_process_window(
    sig_shm_info: dict,
    stft_shm_info: dict | None,
    win_id: int,
    i0: int,
    i1: int,
    win_len: int,
    step_len: int,
    stft_chunk_idx: int,
    sample_rate: float,
    params_map: dict[str, Any],
    base_meta: dict[str, object],
    start_dt: datetime | None,
    bands: list[tuple[str, tuple[float, float]]],
    enable_shared_stft: bool,
    feature_requests: dict[str, frozenset[str]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """v2.2 Worker：从共享内存读取信号和 STFT 结果，纯 CPU 特征计算。"""
    try:
        # 从共享内存读取信号
        sig_arrays, _ = _worker_get_arrays(sig_shm_info)
        win_signal = sig_arrays['signal_pre'][i0:i1].copy()

        if enable_shared_stft and stft_shm_info is not None:
            # 从共享内存读取 STFT 结果
            stft_arrays, _ = _worker_get_arrays(stft_shm_info)
            stft_freqs = stft_arrays['stft_freqs']
            stft_times = stft_arrays['stft_times']
            stft_complex_win = stft_arrays['stft_complex'][:, stft_chunk_idx]
            stft_power_win = stft_arrays['stft_power'][:, stft_chunk_idx]
            short_freqs = stft_arrays['short_freqs']
            short_times = stft_arrays['short_times']
            short_power_win = stft_arrays['short_power'][:, stft_chunk_idx]
            main_signals_win = stft_arrays['main_signals'][:, stft_chunk_idx]
            preprocessed_win = stft_arrays['preprocessed_signals'][stft_chunk_idx]
            backend = 'torch_cpu' if int(stft_arrays['stft_backend_code'][0]) == 1 else 'scipy'

            # 按频带切片 STFT 结果（与 compute_shared_stft 逻辑一致）
            shared_stft: dict[str, dict[str, Any]] = {}
            for band_index, (band_name, params) in enumerate(params_map.items()):
                shared_stft[band_name] = {
                    'main_signal': main_signals_win[band_index],
                    'preprocessed_signal': preprocessed_win,
                    'stft_backend': backend,
                    'stft_freqs': stft_freqs,
                    'stft_times': stft_times,
                    'stft_complex': stft_complex_win[band_index],
                    'stft_power': stft_power_win[band_index],
                    'short_freqs': short_freqs,
                    'short_times': short_times,
                    'short_power': short_power_win[band_index],
                }

            fvals = compute_all_features_for_window(
                win_signal, sample_rate, params_map, shared_stft=shared_stft,
                feature_requests=feature_requests,
            )
        else:
            # Reference path: use the same full-file band signals as the batch path, but compute
            # one window at a time.  This is the correctness oracle for batch/worker comparisons.
            reference_params = next(iter(params_map.values()))
            scale = 1.4826 * float(np.median(np.abs(win_signal - np.median(win_signal)))) + reference_params.eps
            preprocessed_win = win_signal / scale if reference_params.normalize_robust else win_signal
            per_band: dict[str, dict[str, Any]] = {}
            backend = 'torch_cpu' if active_stft_backend() == 'torch' else 'scipy'
            for band_index, (band_name, params) in enumerate(params_map.items()):
                main_signal = sig_arrays[f'band_{band_index}'][i0:i1].copy()
                if params.normalize_robust:
                    main_signal = main_signal / scale
                freqs, times, complex_spec, power = compute_stft_power(
                    main_signal, sample_rate, params.stft_window_ms, params.stft_overlap, params.stft_nfft,
                )
                short_freqs, short_times, _, short_power = compute_stft_power(
                    main_signal, sample_rate, params.short_stft_window_ms, params.short_stft_overlap, params.short_stft_nfft,
                )
                mask = (freqs >= params.main_band_hz[0]) & (freqs <= params.main_band_hz[1])
                short_mask = (short_freqs >= params.main_band_hz[0]) & (short_freqs <= params.main_band_hz[1])
                complex_spec = np.asarray(complex_spec).copy()
                power = np.asarray(power, dtype=np.float64).copy()
                short_power = np.asarray(short_power, dtype=np.float64).copy()
                complex_spec[~mask] = 0.0
                power[~mask] = 0.0
                short_power[~short_mask] = 0.0
                per_band[band_name] = {
                    'main_signal': main_signal,
                    'preprocessed_signal': preprocessed_win,
                    'stft_backend': backend,
                    'stft_freqs': freqs, 'stft_times': times,
                    'stft_complex': complex_spec, 'stft_power': power,
                    'short_freqs': short_freqs, 'short_times': short_times,
                    'short_power': short_power,
                }
            fvals = compute_all_features_for_window(
                win_signal, sample_rate, params_map, shared_stft=per_band,
                feature_requests=feature_requests,
            )

        missing = ''
    except Exception as e:
        fvals = {}
        missing = str(e)

    base = dict(base_meta)
    base['window_id'] = int(win_id)
    base['window_start_index'] = int(i0)
    base['window_end_index'] = int(i1)
    base['window_length_samples'] = int(win_len)
    base['window_step_samples'] = int(step_len)
    base['window_duration_s'] = float(win_len / sample_rate)
    base['window_start_offset_s'] = float(i0 / sample_rate)

    if start_dt is not None:
        base['window_start_datetime'] = (start_dt + timedelta(seconds=float(i0 / sample_rate))).strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        base['window_start_datetime'] = ''

    feat_row = dict(base)
    feat_row.update({k: float(v) for k, v in fvals.items()})

    log_row = dict(base)
    log_row['missing_selected_features'] = missing
    return feat_row, log_row


# ---------------------------------------------------------------------------
# v2.2 主进程：集中化 GPU STFT 批处理
# ---------------------------------------------------------------------------

def _compute_batched_stft(
    signal_pre: np.ndarray,
    band_signals: dict[str, np.ndarray],
    windows: list[tuple[int, int, int, int, int]],
    sample_rate: float,
    params_map: dict[str, Any],
    stft_batch_size: int = 200,
) -> list[tuple[_SharedArrayPack, int, int]]:
    """主进程：对窗口信号做批量 GPU STFT，结果存入共享内存。

    按 stft_batch_size 分块处理，避免 GPU 内存溢出。

    Returns:
        [(stft_pack, chunk_start, chunk_end), ...]
    """
    results: list[tuple[_SharedArrayPack, int, int]] = []

    for chunk_start in range(0, len(windows), stft_batch_size):
        chunk_end = min(chunk_start + stft_batch_size, len(windows))
        pack = _compute_one_stft_chunk(
            signal_pre, band_signals, windows, sample_rate, params_map, chunk_start, chunk_end,
        )
        results.append((pack, chunk_start, chunk_end))

    return results


def _compute_one_stft_chunk(
    signal_pre: np.ndarray,
    band_signals: dict[str, np.ndarray],
    windows: list[tuple[int, int, int, int, int]],
    sample_rate: float,
    params_map: dict[str, Any],
    chunk_start: int,
    chunk_end: int,
) -> _SharedArrayPack:
    """Compute one STFT chunk and store it in shared memory."""
    if not params_map:
        raise ValueError('params_map is empty')
    reference_params = next(iter(params_map.values()))

    chunk_windows = windows[chunk_start:chunk_end]
    win_len = chunk_windows[0][3]
    band_names = list(params_map)
    n_bands = len(band_names)
    n_windows = len(chunk_windows)
    main_signals = np.empty((n_bands, n_windows, win_len), dtype=np.float64)
    preprocessed_signals = np.empty((n_windows, win_len), dtype=np.float64)
    for row_idx, (_, i0, i1, _, _) in enumerate(chunk_windows):
        reference_window = np.asarray(signal_pre[i0:i1], dtype=np.float64)
        scale = 1.4826 * float(np.median(np.abs(reference_window - np.median(reference_window)))) + reference_params.eps
        preprocessed_signals[row_idx] = reference_window / scale if reference_params.normalize_robust else reference_window
        for band_index, band_name in enumerate(band_names):
            band_window = np.asarray(band_signals[band_name][i0:i1], dtype=np.float64)
            main_signals[band_index, row_idx] = band_window / scale if params_map[band_name].normalize_robust else band_window

    # All current bands share one STFT configuration.  Flatten (band, window) for one batched GPU
    # call, then restore the two axes.  This shares execution without sharing a mathematically
    # different wide-band spectrum.
    flat_signals = main_signals.reshape(n_bands * n_windows, win_len)

    stft_freqs, stft_times, stft_complex, stft_power = compute_stft_power(
        flat_signals,
        sample_rate,
        reference_params.stft_window_ms,
        reference_params.stft_overlap,
        reference_params.stft_nfft,
        batched=True,
    )
    short_freqs, short_times, _, short_power = compute_stft_power(
        flat_signals,
        sample_rate,
        reference_params.short_stft_window_ms,
        reference_params.short_stft_overlap,
        reference_params.short_stft_nfft,
        batched=True,
    )
    stft_complex = np.asarray(stft_complex).reshape(n_bands, n_windows, stft_complex.shape[-2], stft_complex.shape[-1])
    stft_power = np.asarray(stft_power, dtype=np.float64).reshape(n_bands, n_windows, stft_power.shape[-2], stft_power.shape[-1])
    short_power = np.asarray(short_power, dtype=np.float64).reshape(n_bands, n_windows, short_power.shape[-2], short_power.shape[-1])
    for band_index, band_name in enumerate(band_names):
        params = params_map[band_name]
        mask = (stft_freqs >= params.main_band_hz[0]) & (stft_freqs <= params.main_band_hz[1])
        short_mask = (short_freqs >= params.main_band_hz[0]) & (short_freqs <= params.main_band_hz[1])
        stft_complex[band_index, :, ~mask, :] = 0.0
        stft_power[band_index, :, ~mask, :] = 0.0
        short_power[band_index, :, ~short_mask, :] = 0.0

    return _SharedArrayPack({
        'stft_freqs': stft_freqs.astype(np.float64),
        'stft_times': stft_times.astype(np.float64),
        'stft_complex': stft_complex,
        'stft_power': stft_power,
        'short_freqs': short_freqs.astype(np.float64),
        'short_times': short_times.astype(np.float64),
        'short_power': short_power,
        'main_signals': main_signals,
        'preprocessed_signals': preprocessed_signals,
        'stft_backend_code': np.asarray([1 if active_stft_backend() == 'torch' else 0], dtype=np.int8),
    })


def build_canonical_band_signals(
    signal_pre: np.ndarray,
    sample_rate: float,
    params_map: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Filter each full source once so every window/backend uses the same band signal."""
    return {
        name: np.asarray(butter_filter(signal_pre, sample_rate, params.main_band_hz), dtype=np.float64)
        for name, params in params_map.items()
    }


def _is_recoverable_stft_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        'out of memory' in text
        or ('cuda' in text and 'memory' in text)
        or 'winerror 1455' in text
        or 'page file' in text
        or '\u9875\u9762\u6587\u4ef6\u592a\u5c0f' in text
        or '页面文件太小' in text
    )


def _clear_transient_memory() -> None:
    try:
        from .signal_ops import clear_gpu_cache
        clear_gpu_cache()
    except Exception:
        pass


def _iter_adaptive_stft_chunks(
    signal_pre: np.ndarray,
    band_signals: dict[str, np.ndarray],
    windows: list[tuple[int, int, int, int, int]],
    sample_rate: float,
    params_map: dict[str, Any],
    stft_batch_size: int,
    min_batch_size: int = 10,
):
    """Yield one shared-memory STFT chunk at a time, shrinking batches on memory pressure."""
    batch_size = max(1, int(stft_batch_size))
    min_batch_size = max(1, int(min_batch_size))
    chunk_start = 0
    while chunk_start < len(windows):
        chunk_end = min(chunk_start + batch_size, len(windows))
        try:
            pack = _compute_one_stft_chunk(
                signal_pre,
                band_signals,
                windows,
                sample_rate,
                params_map,
                chunk_start,
                chunk_end,
            )
        except (RuntimeError, OSError) as exc:
            if batch_size <= min_batch_size or not _is_recoverable_stft_memory_error(exc):
                raise
            batch_size = max(min_batch_size, batch_size // 2)
            _clear_transient_memory()
            continue
        yield pack, chunk_start, chunk_end
        chunk_start = chunk_end


# ---------------------------------------------------------------------------
# 配置与单文件处理
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowConfig:
    bands: list[tuple[str, tuple[float, float]]] = field(default_factory=lambda: [
        ('b_100_100k', (100.0, 100_000.0)),
        ('b_1k_100k', (1_000.0, 100_000.0)),
    ])
    preproc_band: tuple[float, float] = (100.0, 105_000.0)
    window_duration_s: float = 0.02
    window_overlap: float = 0.50
    target_sample_rate: float = 500_000.0
    tdms_fallback_sample_rate: float | None = None
    tdms_channel_name: str | None = None  # 指定TDMS通道名称，None表示使用默认选择逻辑
    window_workers: int | None = None
    window_batch_size: int = 2048
    enable_numa_binding: bool = True
    enable_shared_stft: bool = True
    stft_batch_size: int = 200
    feature_families_by_band: dict[str, tuple[str, ...]] | None = None
    harmonic_band_name: str | None = None


def _process_one_window(
    win_tuple: tuple[int, int, int, int, int],
    signal_pre: np.ndarray,
    sample_rate: float,
    params_map: dict[str, Any],
    base_meta: dict[str, object],
    start_dt: datetime | None,
    enable_shared_stft: bool = False,
    bands: list[tuple[str, tuple[float, float]]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """v2.1 兼容接口：单窗口处理（用于 process_source_file 回退路径）。"""
    win_id, i0, i1, win_len, step_len = win_tuple
    win_signal = signal_pre[i0:i1]

    try:
        shared_stft = None
        if enable_shared_stft and bands is not None:
            shared_stft = compute_shared_stft(win_signal, sample_rate, bands, params_map)
        fvals = compute_all_features_for_window(win_signal, sample_rate, params_map, shared_stft=shared_stft)
        missing = ''
    except Exception as e:
        fvals = {}
        missing = str(e)

    base = dict(base_meta)
    base['window_id'] = int(win_id)
    base['window_start_index'] = int(i0)
    base['window_end_index'] = int(i1)
    base['window_length_samples'] = int(win_len)
    base['window_step_samples'] = int(step_len)
    base['window_duration_s'] = float(win_len / sample_rate)
    base['window_start_offset_s'] = float(i0 / sample_rate)
    if start_dt is not None:
        base['window_start_datetime'] = (start_dt + timedelta(seconds=float(i0 / sample_rate))).strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        base['window_start_datetime'] = ''

    feat_row = dict(base)
    feat_row.update({k: float(v) for k, v in fvals.items()})

    log_row = dict(base)
    log_row['missing_selected_features'] = missing
    return feat_row, log_row


def process_source_file(
    file_path: Path,
    config: SlidingWindowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """处理单个源文件：加载 -> 升采样 -> 预处理 -> GPU批量STFT -> 并行特征计算。"""
    src = load_source_file(file_path, config.tdms_fallback_sample_rate, config.tdms_channel_name)
    raw_signal = np.asarray(src['signal_values'], dtype=float)
    sample_rate = float(src['sample_rate'])
    signal_up, effective_rate = upsample_to_target(raw_signal, sample_rate, config.target_sample_rate)
    centered = signal_up - float(np.mean(signal_up))
    signal_pre = butter_filter(centered, sample_rate=effective_rate, band_hz=config.preproc_band, order=4)
    params_map = {name: build_params_for_band(band, effective_rate) for name, band in config.bands}
    band_signals = build_canonical_band_signals(signal_pre, effective_rate, params_map)
    feature_requests, harmonic_band_name = build_feature_request_map(
        params_map, config.feature_families_by_band, config.harmonic_band_name,
    )

    starttime_raw = str(src['starttime_raw'])
    n_samples = len(signal_pre)
    duration_s = n_samples / effective_rate if effective_rate > 0 else float('nan')
    start_dt = parse_starttime(starttime_raw)

    base_meta: dict[str, object] = {
        'source_file_name': file_path.name,
        'source_file_path': str(file_path),
        'source_format': str(src['source_format']),
        'source_group_name': str(src.get('source_group_name', '')),
        'source_channel_name': str(src.get('source_channel_name', '')),
        'source_detail': str(src.get('source_detail', '')),
        'sample_rate_hz': float(effective_rate),
        'original_sample_rate_hz': float(sample_rate),
        'source_n_samples': int(n_samples),
        'source_duration_s': float(duration_s),
        'starttime_raw': starttime_raw,
        'arrival_time_raw': str(src['arrival_time_raw']),
        'sample_type': str(src['sample_type']),
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'stft_backend': active_stft_backend(),
        'power_dtype': 'float64',
        'harmonic_context_band': harmonic_band_name,
    }

    windows = list_window_ranges(n_samples, effective_rate, config.window_duration_s, config.window_overlap)
    if not windows:
        return pd.DataFrame(), pd.DataFrame()

    rows_features: list[dict[str, object]] = []
    rows_log: list[dict[str, object]] = []

    max_workers = config.window_workers if config.window_workers is not None else _auto_detect_workers()
    max_workers = max(1, int(max_workers))

    # 创建 signal_pre 共享内存
    sig_arrays = {'signal_pre': signal_pre}
    sig_arrays.update({f'band_{i}': band_signals[name] for i, name in enumerate(params_map)})
    sig_pack = _SharedArrayPack(sig_arrays)
    sig_info = sig_pack.get_info()

    try:
        if config.enable_shared_stft:
            # 主进程批量 GPU STFT
            stft_chunks = _iter_adaptive_stft_chunks(
                signal_pre,
                band_signals,
                windows,
                effective_rate,
                params_map,
                stft_batch_size=config.stft_batch_size,
            )

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for stft_pack, chunk_start, chunk_end in stft_chunks:
                    try:
                        futures = []
                        stft_info = stft_pack.get_info()
                        for idx_in_chunk, (win_id, i0, i1, win_len, step_len) in enumerate(
                            windows[chunk_start:chunk_end]
                        ):
                            f = ex.submit(
                                _worker_process_window,
                                sig_info, stft_info,
                                win_id, i0, i1, win_len, step_len, idx_in_chunk,
                                effective_rate, params_map, base_meta, start_dt,
                                config.bands, config.enable_shared_stft, feature_requests,
                            )
                            futures.append(f)

                        for fut in as_completed(futures):
                            feat_row, log_row = fut.result()
                            rows_features.append(feat_row)
                            rows_log.append(log_row)
                    finally:
                        stft_pack.cleanup()
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for b0 in range(0, len(windows), config.window_batch_size):
                    chunk = windows[b0:b0 + config.window_batch_size]
                    for w in chunk:
                        f = ex.submit(
                            _worker_process_window,
                            sig_info, None,
                            w[0], w[1], w[2], w[3], w[4], 0,
                            effective_rate, params_map, base_meta, start_dt,
                            config.bands, False, feature_requests,
                        )
                        futures.append(f)

                for fut in as_completed(futures):
                    feat_row, log_row = fut.result()
                    rows_features.append(feat_row)
                    rows_log.append(log_row)
    finally:
        sig_pack.cleanup()

    rows_features.sort(key=lambda x: int(x['window_id']))
    rows_log.sort(key=lambda x: int(x['window_id']))

    return pd.DataFrame(rows_features), pd.DataFrame(rows_log)


# ---------------------------------------------------------------------------
# 文件预加载（用于流水线）
# ---------------------------------------------------------------------------

def _preload_file(
    file_path: Path,
    config: SlidingWindowConfig,
) -> dict[str, Any] | None:
    """加载并预处理文件，返回计算所需的数据。用于后台线程预加载。"""
    try:
        src = load_source_file(file_path, config.tdms_fallback_sample_rate, config.tdms_channel_name)
        raw_signal = np.asarray(src['signal_values'], dtype=float)
        sample_rate = float(src['sample_rate'])
        signal_up, effective_rate = upsample_to_target(raw_signal, sample_rate, config.target_sample_rate)
        centered = signal_up - float(np.mean(signal_up))
        signal_pre = butter_filter(centered, sample_rate=effective_rate, band_hz=config.preproc_band, order=4)
        params_map = {name: build_params_for_band(band, effective_rate) for name, band in config.bands}
        band_signals = build_canonical_band_signals(signal_pre, effective_rate, params_map)
        feature_requests, harmonic_band_name = build_feature_request_map(
            params_map, config.feature_families_by_band, config.harmonic_band_name,
        )

        starttime_raw = str(src['starttime_raw'])
        n_samples = len(signal_pre)
        duration_s = n_samples / effective_rate if effective_rate > 0 else float('nan')
        start_dt = parse_starttime(starttime_raw)

        base_meta: dict[str, object] = {
            'source_file_name': file_path.name,
            'source_file_path': str(file_path),
            'source_format': str(src['source_format']),
            'source_group_name': str(src.get('source_group_name', '')),
            'source_channel_name': str(src.get('source_channel_name', '')),
            'source_detail': str(src.get('source_detail', '')),
            'sample_rate_hz': float(effective_rate),
            'original_sample_rate_hz': float(sample_rate),
            'source_n_samples': int(n_samples),
            'source_duration_s': float(duration_s),
            'starttime_raw': starttime_raw,
            'arrival_time_raw': str(src['arrival_time_raw']),
            'sample_type': str(src['sample_type']),
            'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'stft_backend': active_stft_backend(),
            'power_dtype': 'float64',
            'harmonic_context_band': harmonic_band_name,
        }

        windows = list_window_ranges(n_samples, effective_rate, config.window_duration_s, config.window_overlap)

        return {
            'file_path': file_path,
            'signal_pre': signal_pre,
            'effective_rate': effective_rate,
            'params_map': params_map,
            'band_signals': band_signals,
            'feature_requests': feature_requests,
            'harmonic_band_name': harmonic_band_name,
            'base_meta': base_meta,
            'start_dt': start_dt,
            'windows': windows,
        }
    except Exception as e:
        logging.getLogger('fea_cpt_gpu_v2_2').warning('预加载文件失败 %s: %s', file_path, e)
        return None


# ---------------------------------------------------------------------------
# v2.2 批量处理主流程
# ---------------------------------------------------------------------------

def _format_duration_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{minutes}m{secs:02d}s'
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f'{hours}h{minutes:02d}m'


def build_sliding_window_dataset(
    source_paths: Sequence[Path],
    config: SlidingWindowConfig | None = None,
    output_dir: Path | None = None,
    processed_list_path: Path | None = None,
    npz_per_csv: int = 100,
    show_progress: bool = True,
) -> dict[str, Any]:
    """v2.2 批量处理：持久化进程池 + 共享内存 + GPU 集中化 STFT + 文件级流水线。"""
    config = config or SlidingWindowConfig()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if config.enable_numa_binding:
        bound = _bind_numa()
        if bound:
            logging.getLogger('fea_cpt_gpu_v2_2').info('NUMA 绑定成功')

    processed_set: set[str] = set()
    if processed_list_path is not None and processed_list_path.exists():
        processed_set = {
            ln.strip()
            for ln in processed_list_path.read_text(encoding='utf-8').splitlines()
            if ln.strip()
        }

    total_files = len(source_paths)
    stats = {'processed': 0, 'skipped': 0, 'windows': 0, 'failed': 0}
    file_counter = 0
    progress_started_at = time.time()

    def _update_progress_bar(pbar: tqdm) -> None:
        if not show_progress:
            return
        done = stats['processed'] + stats['skipped'] + stats['failed']
        elapsed = time.time() - progress_started_at
        avg = elapsed / done if done > 0 else 0.0
        remaining = max(total_files - done, 0)
        eta = avg * remaining if done > 0 else 0.0
        total_est = avg * total_files if done > 0 else 0.0
        pbar.set_postfix_str(
            '已处理={processed} 跳过={skipped} 失败={failed} '
            '平均={avg}/文件 预计总用时={total} 剩余={eta}'.format(
                processed=stats['processed'],
                skipped=stats['skipped'],
                failed=stats['failed'],
                avg=_format_duration_seconds(avg),
                total=_format_duration_seconds(total_est),
                eta=_format_duration_seconds(eta),
            )
        )

    max_workers = config.window_workers if config.window_workers is not None else _auto_detect_workers()
    max_workers = max(1, int(max_workers))

    logger = logging.getLogger('fea_cpt_gpu_v2_2')

    # v2.2 核心：持久化进程池
    executor = ProcessPoolExecutor(max_workers=max_workers)
    # v2.2 核心：预加载线程池（文件级流水线）
    preload_executor = ThreadPoolExecutor(max_workers=1)

    try:
        # 预加载第一个需要处理的文件
        first_unprocessed = None
        for fp in source_paths:
            if str(fp) not in processed_set:
                first_unprocessed = fp
                break

        next_file_future = None
        if first_unprocessed is not None:
            next_file_future = preload_executor.submit(_preload_file, first_unprocessed, config)

        pbar = tqdm(
            source_paths,
            total=total_files,
            desc='处理全部文件',
            unit='file',
            disable=not show_progress,
        )
        for fp in pbar:
            fp_str = str(fp)
            if fp_str in processed_set:
                stats['skipped'] += 1
                _update_progress_bar(pbar)
                continue

            file_counter += 1
            chunk_index = (file_counter - 1) // npz_per_csv + 1

            # 等待当前文件预加载完成
            file_data = None
            if next_file_future is not None:
                try:
                    file_data = next_file_future.result()
                except Exception as e:
                    logger.warning('预加载文件失败 %s: %s', fp, e)
                    file_data = None
                next_file_future = None

            # 如果预加载的文件不是当前文件（可能因为跳过了），则直接加载
            if file_data is None or file_data.get('file_path') != fp:
                file_data = _preload_file(fp, config)

            # 启动下一个文件的预加载（流水线）
            next_fp = None
            found_current = False
            for candidate in source_paths:
                if found_current and str(candidate) not in processed_set:
                    next_fp = candidate
                    break
                if candidate == fp:
                    found_current = True
            if next_fp is not None:
                next_file_future = preload_executor.submit(_preload_file, next_fp, config)

            if file_data is None:
                stats['failed'] += 1
                if output_dir is not None:
                    log_path = output_dir / 'failed_samples.log'
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(f'{datetime.now().isoformat()} | {fp.name} | preload failed\n')
                _update_progress_bar(pbar)
                continue

            signal_pre = file_data['signal_pre']
            effective_rate = file_data['effective_rate']
            params_map = file_data['params_map']
            band_signals = file_data['band_signals']
            feature_requests = file_data['feature_requests']
            base_meta = file_data['base_meta']
            start_dt = file_data['start_dt']
            windows = file_data['windows']

            if not windows:
                stats['skipped'] += 1
                _update_progress_bar(pbar)
                continue

            # 创建 signal_pre 共享内存
            sig_arrays = {'signal_pre': signal_pre}
            sig_arrays.update({f'band_{i}': band_signals[name] for i, name in enumerate(params_map)})
            sig_pack = _SharedArrayPack(sig_arrays)
            sig_info = sig_pack.get_info()

            rows_features: list[dict[str, object]] = []
            rows_log: list[dict[str, object]] = []

            try:
                if config.enable_shared_stft:
                    # 主进程批量 GPU STFT
                    stft_chunks = _iter_adaptive_stft_chunks(
                        signal_pre,
                        band_signals,
                        windows,
                        effective_rate,
                        params_map,
                        stft_batch_size=config.stft_batch_size,
                    )

                    # Double-buffer the producer: while CPU workers consume chunk N,
                    # compute chunk N+1 on the GPU.  The numerical path is unchanged;
                    # only stage scheduling is overlapped.
                    stft_iter = iter(stft_chunks)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stft-producer") as stft_producer:
                        pending_stft = stft_producer.submit(next, stft_iter, None)
                        while True:
                            current = pending_stft.result()
                            if current is None:
                                break
                            pending_stft = stft_producer.submit(next, stft_iter, None)
                            stft_pack, chunk_start, chunk_end = current
                            try:
                                futures = []
                                stft_info = stft_pack.get_info()
                                for idx_in_chunk, (win_id, i0, i1, win_len, step_len) in enumerate(
                                    windows[chunk_start:chunk_end]
                                ):
                                    f = executor.submit(
                                        _worker_process_window,
                                        sig_info, stft_info,
                                        win_id, i0, i1, win_len, step_len, idx_in_chunk,
                                        effective_rate, params_map, base_meta, start_dt,
                                        config.bands, config.enable_shared_stft, feature_requests,
                                    )
                                    futures.append(f)

                                for fut in as_completed(futures):
                                    try:
                                        feat_row, log_row = fut.result()
                                        rows_features.append(feat_row)
                                        rows_log.append(log_row)
                                    except Exception as e:
                                        stats['failed'] += 1
                                        logger.warning('window compute failed: %s', e)
                            finally:
                                stft_pack.cleanup()
                else:
                    futures = []
                    for b0 in range(0, len(windows), config.window_batch_size):
                        chunk = windows[b0:b0 + config.window_batch_size]
                        for w in chunk:
                            f = executor.submit(
                                _worker_process_window,
                                sig_info, None,
                                w[0], w[1], w[2], w[3], w[4], 0,
                                effective_rate, params_map, base_meta, start_dt,
                                config.bands, False, feature_requests,
                            )
                            futures.append(f)

                    for fut in as_completed(futures):
                        try:
                            feat_row, log_row = fut.result()
                            rows_features.append(feat_row)
                            rows_log.append(log_row)
                        except Exception as e:
                            stats['failed'] += 1
                            logger.warning('窗口计算失败: %s', e)
            finally:
                sig_pack.cleanup()

            rows_features.sort(key=lambda x: int(x['window_id']))
            rows_log.sort(key=lambda x: int(x['window_id']))

            completion_errors = validate_completed_file(
                rows_features,
                rows_log,
                windows,
                feature_requests,
                {name: params.wavelet_level for name, params in params_map.items()},
            )
            if completion_errors:
                stats['failed'] += 1
                logger.error('文件完整性校验失败 %s: %s', fp, '; '.join(completion_errors))
                if output_dir is not None:
                    retry_path = output_dir / 'retry_samples.log'
                    with retry_path.open('a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()} | {fp} | {'; '.join(completion_errors)}\n")
                _update_progress_bar(pbar)
                continue

            df_features = pd.DataFrame(rows_features)
            df_log = pd.DataFrame(rows_log)

            if df_features.empty:
                stats['skipped'] += 1
                _update_progress_bar(pbar)
                continue

            if output_dir is not None:
                feature_csv = output_dir / f'features_part_{chunk_index:04d}.csv'
                log_csv = output_dir / f'log_part_{chunk_index:04d}.csv'
                feature_header = not feature_csv.exists() or feature_csv.stat().st_size == 0
                log_header = not log_csv.exists() or log_csv.stat().st_size == 0
                df_features.to_csv(feature_csv, mode='a', header=feature_header, index=False, encoding='utf-8-sig')
                df_log.to_csv(log_csv, mode='a', header=log_header, index=False, encoding='utf-8-sig')

                if processed_list_path is not None:
                    with processed_list_path.open('a', encoding='utf-8') as f:
                        f.write(fp_str + '\n')

            processed_set.add(fp_str)
            stats['processed'] += 1
            stats['windows'] += len(df_features)
            _update_progress_bar(pbar)

    finally:
        executor.shutdown(wait=True)
        preload_executor.shutdown(wait=True)

    return stats


# ---------------------------------------------------------------------------
# 文件发现与降采样
# ---------------------------------------------------------------------------

def discover_source_files(root_dirs: Sequence[str | Path], max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for root in root_dirs:
        root = Path(root)
        if root.is_file() and root.suffix.lower() in {'.npz', '.tdms'}:
            files.append(root)
        elif root.is_dir():
            files.extend(
                p for p in root.rglob('*')
                if p.is_file() and p.suffix.lower() in {'.npz', '.tdms'}
            )
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def downsample_source_files(
    files: list[Path],
    ratio: float = 0.6,
    seed: int = 42,
) -> list[Path]:
    if ratio >= 1.0 or ratio <= 0.0:
        return list(files)
    rng = random.Random(seed)
    folder_groups: dict[Path, list[Path]] = {}
    for f in files:
        folder_groups.setdefault(f.parent, []).append(f)
    sampled: list[Path] = []
    for folder, group in folder_groups.items():
        n = max(1, math.ceil(len(group) * ratio))
        chosen = rng.sample(group, min(n, len(group)))
        sampled.extend(chosen)
    sampled.sort()
    return sampled
