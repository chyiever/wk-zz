"""单特征判别能力评估。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from .feature_schema import impute_with_median, numeric_feature_frame


COMPARISONS = {
    "BK_NONBK": (("BK00", "BK05"), ("FL00", "FL05", "QJ00", "QJ05")),
    "BK_QJ": (("BK00", "BK05"), ("QJ00", "QJ05")),
    "BK00_BK05": (("BK05",), ("BK00",)),
    "FL00_FL05": (("FL05",), ("FL00",)),
    "QJ00_QJ05": (("QJ05",), ("QJ00",)),
}


def _safe_auc(y: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """返回有方向AUC和无方向分离度AUC。"""

    if len(np.unique(y)) < 2 or np.nanstd(values) == 0:
        return np.nan, np.nan
    try:
        auc = float(roc_auc_score(y, values))
    except ValueError:
        return np.nan, np.nan
    return auc, max(auc, 1.0 - auc)


def _cliff_delta(pos: np.ndarray, neg: np.ndarray) -> float:
    """使用Mann-Whitney U统计量计算Cliff's delta。"""

    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    try:
        u_stat = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    except ValueError:
        return np.nan
    return float((2.0 * u_stat) / (len(pos) * len(neg)) - 1.0)


def _mutual_information(values: np.ndarray, y: np.ndarray, random_state: int) -> float:
    if len(np.unique(y)) < 2 or np.nanstd(values) == 0:
        return 0.0
    try:
        return float(mutual_info_classif(values.reshape(-1, 1), y, random_state=random_state, discrete_features=False)[0])
    except Exception:
        return 0.0


def evaluate_feature_discrimination(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    random_state: int = 42,
    comparisons: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> pd.DataFrame:
    """对每个特征计算多组二分类判别指标。"""

    comparisons = COMPARISONS if comparisons is None else comparisons
    x_all = impute_with_median(numeric_feature_frame(frame, feature_columns))
    rows: list[dict[str, object]] = []

    for comp_name, (positive_labels, negative_labels) in comparisons.items():
        labels = frame["source_label"].astype(str)
        mask = labels.isin(positive_labels + negative_labels)
        if not bool(mask.any()):
            continue
        y = labels.loc[mask].isin(positive_labels).astype(int).to_numpy()
        x = x_all.loc[mask, :]
        for feature in feature_columns:
            values = x[feature].to_numpy(dtype=float)
            pos = values[y == 1]
            neg = values[y == 0]
            auc_signed, auc_abs = _safe_auc(y, values)
            pooled_iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
            pooled_iqr = pooled_iqr if pooled_iqr > 0 else np.nanstd(values)
            median_diff = float(np.nanmedian(pos) - np.nanmedian(neg)) if len(pos) and len(neg) else np.nan
            w_dist = float(wasserstein_distance(pos, neg)) if len(pos) and len(neg) else np.nan
            norm_w = float(w_dist / pooled_iqr) if pooled_iqr and pooled_iqr > 0 else np.nan
            cliff = _cliff_delta(pos, neg)
            mi = _mutual_information(values, y, random_state=random_state)
            rows.append(
                {
                    "comparison": comp_name,
                    "feature": feature,
                    "positive_labels": ",".join(positive_labels),
                    "negative_labels": ",".join(negative_labels),
                    "n_positive": int((y == 1).sum()),
                    "n_negative": int((y == 0).sum()),
                    "auc_signed": auc_signed,
                    "auc_abs": auc_abs,
                    "auc_lift": auc_abs - 0.5 if pd.notna(auc_abs) else np.nan,
                    "wasserstein": w_dist,
                    "wasserstein_norm": norm_w,
                    "cliff_delta": cliff,
                    "abs_cliff_delta": abs(cliff) if pd.notna(cliff) else np.nan,
                    "mutual_info": mi,
                    "median_positive": float(np.nanmedian(pos)) if len(pos) else np.nan,
                    "median_negative": float(np.nanmedian(neg)) if len(neg) else np.nan,
                    "median_diff": median_diff,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["discrimination_score"] = (
        out["auc_lift"].fillna(0).clip(lower=0) * 2.0 * 0.55
        + out["abs_cliff_delta"].fillna(0).clip(upper=1.0) * 0.30
        + (out["mutual_info"].fillna(0) / (1.0 + out["mutual_info"].fillna(0))) * 0.15
    )
    return out.sort_values(["comparison", "discrimination_score"], ascending=[True, False])
