"""评估模块：指标统计与消融汇总。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_error_metrics(results: list[dict], gt_x: float | None = None) -> dict:
    """计算总体指标；若有 gt_x 则计算定位误差。"""
    n = len(results)
    if n == 0:
        return {"count": 0}

    status = [r.get("status", "rejected") for r in results]
    rmse = np.array([float(r.get("rmse_tdoa", np.inf)) for r in results], dtype=np.float64)
    conf = np.array([float(r.get("confidence", 0.0)) for r in results], dtype=np.float64)

    out = {
        "count": n,
        "ok_rate": float(np.mean(np.array(status) == "ok")),
        "reject_rate": float(np.mean(np.array(status) == "rejected")),
        "mean_rmse_tdoa": float(np.nanmean(rmse[np.isfinite(rmse)])) if np.isfinite(rmse).any() else np.inf,
        "mean_confidence": float(np.nanmean(conf)),
    }

    if gt_x is not None:
        x_pred = np.array([np.nan if r.get("source_x") is None else float(r.get("source_x")) for r in results])
        mask = np.isfinite(x_pred)
        if np.any(mask):
            out["mae_x"] = float(np.mean(np.abs(x_pred[mask] - gt_x)))
            out["rmse_x"] = float(np.sqrt(np.mean((x_pred[mask] - gt_x) ** 2)))
    return out


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """将结果列表转换为 DataFrame，便于保存与可视化。"""
    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "idx": i,
                "source_x": r.get("source_x"),
                "source_z": r.get("source_z"),
                "valid_channels": r.get("valid_channels"),
                "rmse_tdoa": r.get("rmse_tdoa"),
                "confidence": r.get("confidence"),
                "status": r.get("status"),
                "reason": r.get("reason"),
                "fit_model": r.get("fit_model"),
                "ref_idx": r.get("ref_idx"),
            }
        )
    return pd.DataFrame(rows)
