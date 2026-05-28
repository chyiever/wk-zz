"""滑窗批量特征提取模块。

用于对连续真实数据（npz/tdms）进行滑窗全量特征提取。
支持多文件夹批量处理、200kHz/500kHz 统一采样率、窗级并行。
"""

from __future__ import annotations

import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from math import gcd
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .base import FeatureRecord
from .features import compute_all_features
from .gpu_backend import gpu_backend_info
from .params import DEFAULT_FEATURE_PARAMS
from .signal_ops import build_context, butter_filter

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None  # type: ignore[assignment]

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _scalar_text(value: object) -> str:
    """将 np 标量/bytes/单元素列表转为字符串。"""
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
    """尝试将任意值转为 float。"""
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
    """从属性字典中按优先级查找第一个匹配键。"""
    normalized = {str(k).lower(): v for k, v in props.items()}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _infer_sample_rate_from_filename(path: Path) -> float | None:
    """从文件名推断采样率（支持 500K, 200kHz, 0.5MHz 等格式）。"""
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
    """从 npz 文件加载信号数据。"""
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
    """自动选择 TDMS 文件中的信号通道。"""
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


def _load_tdms_source(path: Path, fallback_sample_rate: float | None = None) -> dict[str, object]:
    """从 tdms 文件加载信号数据。"""
    td = TdmsFile.read(path)
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


def load_source_file(path: Path, tdms_fallback_sample_rate: float | None = None) -> dict[str, object]:
    """根据文件后缀加载 npz 或 tdms 源文件。"""
    suf = path.suffix.lower()
    if suf == '.npz':
        return _load_npz_source(path)
    if suf == '.tdms':
        return _load_tdms_source(path, tdms_fallback_sample_rate)
    raise ValueError(f'Unsupported file type: {path.suffix}')


def parse_starttime(starttime_raw: str) -> datetime | None:
    """解析 starttime 字符串为 datetime 对象。"""
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
    """将信号升采样到目标采样率。

    仅当 sample_rate < target_rate 时执行升采样。
    使用 scipy.signal.resample_poly（多相滤波器组，高效且数值稳定）。

    Returns:
        (resampled_signal, effective_sample_rate)
    """
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
    """计算滑窗范围。

    Args:
        n_samples: 信号总样本数
        sample_rate: 采样率 (Hz)
        window_duration_s: 窗口时长 (秒)
        overlap: 重叠比例 (0~1)

    Returns:
        list of (window_id, start_idx, end_idx, window_length, step_length)
    """
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
    """将频带限制在 Nyquist 范围内。"""
    low = max(1.0, min(low, nyq * 0.98))
    high = max(low + 1.0, min(high, nyq * 0.995))
    return (float(low), float(high))


def build_params_for_band(band: tuple[float, float], sample_rate: float) -> Any:
    """为指定频带构建 FeatureParams。"""
    low, high = band
    nyq = sample_rate / 2.0
    low, high = _safe_band(low, high, nyq)
    span = max(high - low, 10.0)

    low_band = _safe_band(low, low + 0.30 * span, nyq)
    mid_band = _safe_band(low + 0.30 * span, low + 0.60 * span, nyq)
    high1_band = _safe_band(low + 0.50 * span, low + 0.80 * span, nyq)
    high2_band = _safe_band(low + 0.60 * span, high, nyq)
    harmonic_band = _safe_band(low + 0.50 * span, high, nyq)
    ridge_main = _safe_band(low, low + 0.65 * span, nyq)
    ridge_h2 = _safe_band(max(low * 2.0, low + 0.20 * span), min(high * 2.0, nyq * 0.995), nyq)

    return replace(
        DEFAULT_FEATURE_PARAMS,
        highpass_hz=1_000.0,
        main_band_hz=(low, high),
        low_band_hz=low_band,
        mid_band_hz=mid_band,
        high1_band_hz=high1_band,
        high2_band_hz=high2_band,
        harmonic_band_hz=harmonic_band,
        ridge_main_search_hz=ridge_main,
        ridge_h2_search_hz=ridge_h2,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# 单窗口特征计算
# ---------------------------------------------------------------------------

def compute_all_features_for_window(
    window_signal: np.ndarray,
    sample_rate: float,
    params_map: dict[str, Any],
) -> dict[str, float]:
    """对单个窗口计算所有频带的全量特征。

    Args:
        window_signal: 窗口信号（已预处理）
        sample_rate: 采样率 (Hz)
        params_map: {band_name: FeatureParams} 频带参数字典

    Returns:
        {f"{band_name}__{feature_name}": value} 特征字典
    """
    rec = FeatureRecord(
        sample_id='w',
        sample_name='w',
        sample_type='raw',
        sample_type_code=0,
        path=Path('.'),
        signal=np.asarray(window_signal, dtype=float),
        sample_rate=float(sample_rate),
        metadata={},
    )

    out: dict[str, float] = {}
    for band_name, params in params_map.items():
        context = build_context(rec, params)
        result = compute_all_features(context)
        for k, v in result.features.items():
            out[f'{band_name}__{k}'] = float(v)
    return out


# ---------------------------------------------------------------------------
# 单文件处理
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowConfig:
    """滑窗处理配置。"""
    bands: list[tuple[str, tuple[float, float]]] = field(default_factory=lambda: [
        ('b_1k_100k', (1_000.0, 100_000.0)),
        ('b_1k_10k', (1_000.0, 10_000.0)),
    ])
    preproc_band: tuple[float, float] = (1_000.0, 95_000.0)
    window_duration_s: float = 0.02
    window_overlap: float = 0.50
    target_sample_rate: float = 500_000.0
    tdms_fallback_sample_rate: float | None = None
    window_workers: int = 8
    window_batch_size: int = 256


def _process_one_window(
    win_tuple: tuple[int, int, int, int, int],
    signal_pre: np.ndarray,
    sample_rate: float,
    params_map: dict[str, Any],
    base_meta: dict[str, object],
    start_dt: datetime | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """处理单个窗口，返回 (特征行, 日志行)。"""
    win_id, i0, i1, win_len, step_len = win_tuple
    win_signal = signal_pre[i0:i1]

    try:
        fvals = compute_all_features_for_window(win_signal, sample_rate, params_map)
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
    """处理单个源文件：加载 -> 升采样 -> 预处理 -> 滑窗 -> 特征计算。

    Returns:
        (df_features, df_log) 两个 DataFrame
    """
    src = load_source_file(file_path, config.tdms_fallback_sample_rate)
    raw_signal = np.asarray(src['signal_values'], dtype=float)
    sample_rate = float(src['sample_rate'])

    # 升采样到统一采样率
    signal_up, effective_rate = upsample_to_target(raw_signal, sample_rate, config.target_sample_rate)

    # 整条信号预处理一次
    centered = signal_up - float(np.mean(signal_up))
    signal_pre = butter_filter(centered, sample_rate=effective_rate, band_hz=config.preproc_band, order=4)

    # 构建频带参数缓存
    params_map = {name: build_params_for_band(band, effective_rate) for name, band in config.bands}

    # 元信息
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
    }

    # 生成窗口
    windows = list_window_ranges(n_samples, effective_rate, config.window_duration_s, config.window_overlap)
    if not windows:
        return pd.DataFrame(), pd.DataFrame()

    rows_features: list[dict[str, object]] = []
    rows_log: list[dict[str, object]] = []

    max_workers = max(1, int(config.window_workers))
    for b0 in range(0, len(windows), config.window_batch_size):
        chunk = windows[b0:b0 + config.window_batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_process_one_window, w, signal_pre, effective_rate, params_map, base_meta, start_dt)
                for w in chunk
            ]
            for fut in as_completed(futures):
                feat_row, log_row = fut.result()
                rows_features.append(feat_row)
                rows_log.append(log_row)

    # 按 window_id 排序
    rows_features.sort(key=lambda x: int(x['window_id']))
    rows_log.sort(key=lambda x: int(x['window_id']))

    df_features = pd.DataFrame(rows_features)
    df_log = pd.DataFrame(rows_log)
    return df_features, df_log


# ---------------------------------------------------------------------------
# 批量处理主流程
# ---------------------------------------------------------------------------

def build_sliding_window_dataset(
    source_paths: Sequence[Path],
    config: SlidingWindowConfig | None = None,
    output_dir: Path | None = None,
    processed_list_path: Path | None = None,
    npz_per_csv: int = 100,
    show_progress: bool = True,
) -> dict[str, Any]:
    """批量处理多个源文件，输出分块 CSV。

    Args:
        source_paths: 源文件路径列表
        config: 滑窗配置
        output_dir: 输出目录
        processed_list_path: 已处理文件记录路径
        npz_per_csv: 每个 CSV 包含的文件数
        show_progress: 是否显示进度条

    Returns:
        统计信息字典 {processed, skipped, windows, failed}
    """
    config = config or SlidingWindowConfig()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 加载已处理记录
    processed_set: set[str] = set()
    if processed_list_path is not None and processed_list_path.exists():
        processed_set = {
            ln.strip()
            for ln in processed_list_path.read_text(encoding='utf-8').splitlines()
            if ln.strip()
        }

    stats = {'processed': 0, 'skipped': 0, 'windows': 0, 'failed': 0}
    file_counter = 0

    for fp in tqdm(source_paths, desc='处理文件', disable=not show_progress):
        fp_str = str(fp)
        if fp_str in processed_set:
            stats['skipped'] += 1
            continue

        file_counter += 1
        chunk_index = (file_counter - 1) // npz_per_csv + 1

        try:
            df_features, df_log = process_source_file(fp, config)
        except Exception as e:
            stats['failed'] += 1
            if output_dir is not None:
                log_path = output_dir / 'failed_samples.log'
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'{datetime.now().isoformat()} | {fp.name} | {type(e).__name__}: {e}\n')
            continue

        if df_features.empty:
            stats['skipped'] += 1
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

    return stats


def discover_source_files(root_dirs: Sequence[str | Path], max_files: int | None = None) -> list[Path]:
    """从多个根目录发现所有 npz/tdms 文件。"""
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
