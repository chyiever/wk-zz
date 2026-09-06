from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import numpy as np

from fea_cpt_gpu_v2_2.sliding_window import (
    build_params_for_band,
    compute_all_features_for_window,
    compute_shared_stft,
)


@lru_cache(maxsize=32)
def _get_params_map(
    sample_rate: float,
    bands_key: tuple[tuple[str, tuple[float, float]], ...],
) -> dict[str, Any]:
    """按采样率和频带配置缓存特征参数，避免每个窗口重复构造。"""
    return {
        name: build_params_for_band(tuple(band), float(sample_rate))
        for name, band in bands_key
    }


def parse_npz_ts(s: str | bytes) -> datetime:
    """解析 npz 头文件中的时间字符串。"""
    s = str(s).strip()
    if "T" in s:
        return datetime.strptime(s, "%Y%m%dT%H%M%S.%f")
    return datetime.strptime(s, "%Y%m%d%H%M%S.%f")


def load_npz_channel(path: str | Path, channel_index: int = 0) -> dict[str, object]:
    """读取单个 npz 文件，返回指定通道波形和必要头信息。"""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        phase = np.asarray(data["phase_data"], dtype=float)
        if phase.ndim == 2:
            signal = phase[:, channel_index]
        else:
            signal = phase
        sample_rate = float(np.asarray(data["sample_rate"]).item())
        starttime_raw = str(data["starttime"])
        arrival_time_raw = str(data["arrival_time"])
        sample_type = str(data["type"]) if "type" in data else path.parent.name
        channel_names = list(data["channel_names"]) if "channel_names" in data else []
    return {
        "source_format": "npz",
        "signal": signal,
        "sample_rate": sample_rate,
        "starttime_raw": starttime_raw,
        "arrival_time_raw": arrival_time_raw,
        "sample_type": sample_type,
        "channel_names": channel_names,
        "channel_index": channel_index,
    }


def get_cut_windows(
    signal_len: int,
    sample_rate: float,
    arrival_offset_s: float,
    modes: list[tuple[str, float, float]],
) -> list[tuple[str, int, int]]:
    """以 arrival_time 为 0 点，将毫秒窗口转换为样本索引窗口。"""
    result = []
    for name, t0_ms, t1_ms in modes:
        i0 = int(round((arrival_offset_s + t0_ms / 1000.0) * sample_rate))
        i1 = int(round((arrival_offset_s + t1_ms / 1000.0) * sample_rate))
        i0 = max(0, i0)
        i1 = min(signal_len, i1)
        result.append((name, i0, i1))
    return result


def compute_features_for_window(
    seg_signal: np.ndarray,
    sample_rate: float,
    bands: list[tuple[str, tuple[float, float]]],
    enable_shared_stft: bool = False,
) -> dict[str, float]:
    """对单个截取窗口计算全部频带特征。"""
    bands_key = tuple((name, tuple(band)) for name, band in bands)
    params_map = _get_params_map(float(sample_rate), bands_key)
    shared_stft = None
    if enable_shared_stft:
        shared_stft = compute_shared_stft(seg_signal, sample_rate, bands, params_map)
    return compute_all_features_for_window(
        seg_signal,
        sample_rate,
        params_map,
        shared_stft=shared_stft,
    )


def process_event_record(
    record: dict[str, object],
    bands: list[tuple[str, tuple[float, float]]],
    cut_modes: list[tuple[str, float, float]],
    channel_index: int = 0,
    enable_shared_stft: bool = False,
) -> tuple[list[dict[str, object]], tuple[str, str, str] | None]:
    """处理一个源数据文件，返回该文件所有截取窗口的特征行。"""
    fp = Path(record["path"])
    label = str(record["label"])
    try:
        src = load_npz_channel(fp, channel_index=channel_index)
        signal = src["signal"]
        fs = float(src["sample_rate"])
        arrival_offset_s = (
            parse_npz_ts(src["arrival_time_raw"])
            - parse_npz_ts(src["starttime_raw"])
        ).total_seconds()
        cut_windows = get_cut_windows(len(signal), fs, arrival_offset_s, cut_modes)

        rows: list[dict[str, object]] = []
        for name, i0, i1 in cut_windows:
            seg = signal[i0:i1]
            feats = compute_features_for_window(
                seg,
                fs,
                bands,
                enable_shared_stft=enable_shared_stft,
            )
            row: dict[str, object] = {
                "source_file_name": fp.name,
                "source_file_path": str(fp),
                "label": label,
                "source_file_index_in_label": int(record["file_index_in_label"]),
                "output_part": int(record["output_part"]),
                "window_mode": name,
                "window_start_ms": (i0 / fs - arrival_offset_s) * 1000.0,
                "window_end_ms": (i1 / fs - arrival_offset_s) * 1000.0,
                "window_n_samples": int(i1 - i0),
                "sample_rate_hz": fs,
                "arrival_time": src["arrival_time_raw"],
                "starttime": src["starttime_raw"],
                "channel_index": channel_index,
            }
            row.update(feats)
            rows.append(row)
        return rows, None
    except Exception as e:
        return [], (label, fp.name, str(e))


def process_event_record_timed(
    record: dict[str, object],
    bands: list[tuple[str, tuple[float, float]]],
    cut_modes: list[tuple[str, float, float]],
    channel_index: int = 0,
    enable_shared_stft: bool = False,
) -> tuple[list[dict[str, object]], tuple[str, str, str] | None, float]:
    """处理一个源数据文件，并返回该文件处理耗时。"""
    start = time.time()
    rows, err = process_event_record(
        record,
        bands,
        cut_modes,
        channel_index=channel_index,
        enable_shared_stft=enable_shared_stft,
    )
    return rows, err, time.time() - start
