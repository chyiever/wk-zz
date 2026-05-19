"""批处理模块：切窗、批量定位、日志落盘。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import compute_error_metrics, results_to_dataframe
from .localization import LocalizeConfig, localize_single_event


def sliding_windows(data_fc: np.ndarray, win_size: int, step: int) -> list[np.ndarray]:
    """按时间轴切窗，输入 shape=(frames, channels)。"""
    n = data_fc.shape[0]
    if win_size <= 0 or step <= 0:
        raise ValueError("win_size 与 step 必须为正整数")
    windows = []
    if n <= win_size:
        # 当总长度短于窗口时，回退为单窗口整段处理，避免批处理为空。
        return [data_fc]
    for start in range(0, max(1, n - win_size + 1), step):
        end = start + win_size
        if end > n:
            break
        windows.append(data_fc[start:end, :])
    return windows


def run_batch(
    data_fc: np.ndarray,
    cfg: LocalizeConfig,
    output_root: str | Path,
    win_size: int,
    step: int,
    gt_x: float | None = None,
) -> tuple[list[dict], dict]:
    """批量运行定位并保存指标。"""
    output_root = Path(output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    windows = sliding_windows(data_fc, win_size=win_size, step=step)
    results = [localize_single_event(w, cfg) for w in windows]

    df = results_to_dataframe(results)
    df.to_csv(output_root / "metrics" / "batch_results.csv", index=False, encoding="utf-8-sig")

    metrics = compute_error_metrics(results, gt_x=gt_x)
    pd.DataFrame([metrics]).to_csv(output_root / "metrics" / "batch_summary.csv", index=False, encoding="utf-8-sig")

    return results, metrics
