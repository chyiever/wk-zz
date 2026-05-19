"""I/O 工具：读取 eDAS 相位 bin 并转换为弧度。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PHASE_RAD_SCALE = np.pi / 32767.0
_POINTS_PATTERN = re.compile(r"-(\d+)pt-")
_HZ_PATTERN = re.compile(r"-(\d+)Hz-")


def infer_points_per_frame(file_path: str | Path) -> int | None:
    """从文件名解析每帧点数，如 0032pt。"""
    match = _POINTS_PATTERN.search(Path(file_path).name)
    return int(match.group(1)) if match else None


def infer_sample_rate_hz(file_path: str | Path) -> float | None:
    """从文件名解析采样率，如 2000Hz。"""
    match = _HZ_PATTERN.search(Path(file_path).name)
    return float(match.group(1)) if match else None


def read_phase_bin_file(file_path: str | Path, points_per_frame: int | None = None) -> np.ndarray:
    """读取单个 bin 文件，返回 shape=(frames, points) 的相位弧度数据。"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    if points_per_frame is None:
        points_per_frame = infer_points_per_frame(path)
    if points_per_frame is None or points_per_frame <= 0:
        raise ValueError("points_per_frame 无法解析，请手动指定正整数")

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError("输入 bin 文件为空")
    if raw.size % points_per_frame != 0:
        raise ValueError(
            f"数据长度 {raw.size} 不能整除 points_per_frame={points_per_frame}"
        )

    frames = raw.size // points_per_frame
    data = raw.reshape(frames, points_per_frame).astype(np.float64) * PHASE_RAD_SCALE
    return data
