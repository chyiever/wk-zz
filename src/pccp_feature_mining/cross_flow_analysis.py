"""跨流速一致性与工况敏感性分析。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .feature_schema import impute_with_median, numeric_feature_frame


def _pair_auc(frame: pd.DataFrame, x: pd.DataFrame, feature: str, pos_label: str, neg_label: str) -> tuple[float, float]:
    labels = frame["source_label"].astype(str)
    mask = labels.isin([pos_label, neg_label])
    y = labels.loc[mask].eq(pos_label).astype(int).to_numpy()
    values = x.loc[mask, feature].to_numpy(dtype=float)
    if len(np.unique(y)) < 2 or np.nanstd(values) == 0:
        return np.nan, np.nan
    auc = float(roc_auc_score(y, values))
    return auc, max(auc, 1.0 - auc)


def _median_shift(frame: pd.DataFrame, x: pd.DataFrame, feature: str, label_a: str, label_b: str) -> float:
    labels = frame["source_label"].astype(str)
    a = x.loc[labels.eq(label_a), feature].to_numpy(dtype=float)
    b = x.loc[labels.eq(label_b), feature].to_numpy(dtype=float)
    both = np.concatenate([a, b])
    scale = np.nanpercentile(both, 75) - np.nanpercentile(both, 25)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(both)
    if not np.isfinite(scale) or scale <= 0:
        return 0.0
    return float(abs(np.nanmedian(a) - np.nanmedian(b)) / scale)


def evaluate_cross_flow_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """比较v0与v0.5下BK相对流噪的稳定判别能力。"""

    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        auc00, sep00 = _pair_auc(frame, x, feature, "BK00", "FL00")
        auc05, sep05 = _pair_auc(frame, x, feature, "BK05", "FL05")
        direction_same = int(pd.notna(auc00) and pd.notna(auc05) and ((auc00 >= 0.5) == (auc05 >= 0.5)))
        sep00_lift = max(float(sep00) - 0.5, 0.0) if pd.notna(sep00) else 0.0
        sep05_lift = max(float(sep05) - 0.5, 0.0) if pd.notna(sep05) else 0.0
        flow_sensitivity = _median_shift(frame, x, feature, "BK00", "BK05")
        fl_sensitivity = _median_shift(frame, x, feature, "FL00", "FL05")
        consistency = min(sep00_lift, sep05_lift) / max(max(sep00_lift, sep05_lift), 1e-12)
        cross_score = direction_same * (0.55 * (sep00_lift + sep05_lift) + 0.45 * consistency) / (
            1.0 + 0.5 * flow_sensitivity + 0.25 * fl_sensitivity
        )
        rows.append(
            {
                "feature": feature,
                "auc_BK00_vs_FL00": auc00,
                "auc_BK05_vs_FL05": auc05,
                "auc_lift_BK00_vs_FL00": sep00_lift,
                "auc_lift_BK05_vs_FL05": sep05_lift,
                "direction_same": direction_same,
                "cross_condition_consistency": consistency,
                "bk00_bk05_median_shift_norm": flow_sensitivity,
                "fl00_fl05_median_shift_norm": fl_sensitivity,
                "flow_sensitivity_penalty": flow_sensitivity + 0.5 * fl_sensitivity,
                "cross_flow_score": cross_score,
            }
        )
    return pd.DataFrame(rows).sort_values("cross_flow_score", ascending=False)
