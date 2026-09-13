"""单特征判别能力评估。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wasserstein_distance
from sklearn.metrics import mutual_info_score, roc_auc_score

from .feature_schema import impute_with_median, numeric_feature_frame


COMPARISONS = {
    "BK00_NONBK00": (("BK00",), ("FL00", "QJ00")),
    "BK00_QJ00": (("BK00",), ("QJ00",)),
    "BK05_QJ05": (("BK05",), ("QJ05",)),
    "BK05_NONBK05": (("BK05",), ("FL05", "QJ05")),
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


def _rank_auc_cliff(y: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(np.unique(y)) < 2 or np.nanstd(values) == 0:
        return np.nan, np.nan, np.nan

    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < 3:
        return np.nan, np.nan, np.nan
    finite_y = y[finite_mask]
    finite_values = values[finite_mask]
    n_pos = int(np.sum(finite_y == 1))
    n_neg = int(np.sum(finite_y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan

    ranks = rankdata(finite_values, method="average")
    pos_rank_sum = float(np.sum(ranks[finite_y == 1]))
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    auc_signed = float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))
    auc_abs = max(auc_signed, 1.0 - auc_signed)
    cliff = 2.0 * auc_signed - 1.0
    return auc_signed, auc_abs, float(cliff)


def _mutual_information(values: np.ndarray, y: np.ndarray, random_state: int) -> float:
    del random_state
    if len(np.unique(y)) < 2 or np.nanstd(values) == 0:
        return 0.0
    try:
        finite_mask = np.isfinite(values)
        if int(finite_mask.sum()) < 3:
            return 0.0
        finite_values = values[finite_mask]
        finite_y = y[finite_mask]
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        value_range = value_max - value_min
        if not np.isfinite(value_range) or value_range <= 0:
            return 0.0

        n_bins = min(16, max(2, int(np.sqrt(finite_values.size))))
        scaled = (finite_values - value_min) / value_range
        bin_codes = np.minimum((scaled * n_bins).astype(int), n_bins - 1)
        return float(mutual_info_score(finite_y, bin_codes))
    except Exception:
        return 0.0


def _metric_record(values: np.ndarray, y: np.ndarray, random_state: int) -> dict[str, float]:
    pos = values[y == 1]
    neg = values[y == 0]
    auc_signed, auc_abs, cliff = _rank_auc_cliff(y, values)
    pooled_iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
    pooled_iqr = pooled_iqr if pooled_iqr > 0 else np.nanstd(values)
    median_diff = float(np.nanmedian(pos) - np.nanmedian(neg)) if len(pos) and len(neg) else np.nan
    w_dist = float(wasserstein_distance(pos, neg)) if len(pos) and len(neg) else np.nan
    norm_w = float(w_dist / pooled_iqr) if pooled_iqr and pooled_iqr > 0 else np.nan
    mi = _mutual_information(values, y, random_state=random_state)
    return {
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


def _mutual_information_columns(values: np.ndarray, y: np.ndarray, max_bins: int = 16) -> np.ndarray:
    if len(np.unique(y)) < 2 or values.shape[0] < 3:
        return np.zeros(values.shape[1], dtype=float)

    y = np.asarray(y, dtype=int)
    n_rows, n_features = values.shape
    n_bins = min(int(max_bins), max(2, int(np.sqrt(n_rows))))
    value_min = np.nanmin(values, axis=0)
    value_max = np.nanmax(values, axis=0)
    value_range = value_max - value_min
    valid = np.isfinite(value_range) & (value_range > 0)
    out = np.zeros(n_features, dtype=float)
    if not bool(valid.any()):
        return out

    scaled = np.zeros_like(values[:, valid], dtype=float)
    scaled[:, :] = (values[:, valid] - value_min[valid]) / value_range[valid]
    codes = np.minimum(np.floor(scaled * n_bins).astype(np.int16), n_bins - 1)

    pos_mask = y == 1
    neg_mask = y == 0
    p_pos = float(pos_mask.mean())
    p_neg = float(neg_mask.mean())
    if p_pos <= 0 or p_neg <= 0:
        return out

    mi = np.zeros(codes.shape[1], dtype=float)
    for bin_id in range(n_bins):
        in_bin = codes == bin_id
        count_pos = in_bin[pos_mask, :].sum(axis=0).astype(float)
        count_neg = in_bin[neg_mask, :].sum(axis=0).astype(float)
        count_total = count_pos + count_neg
        px = count_total / n_rows

        pos_cell = count_pos > 0
        if bool(pos_cell.any()):
            pxy = count_pos[pos_cell] / n_rows
            mi[pos_cell] += pxy * np.log(pxy / (px[pos_cell] * p_pos))

        neg_cell = count_neg > 0
        if bool(neg_cell.any()):
            pxy = count_neg[neg_cell] / n_rows
            mi[neg_cell] += pxy * np.log(pxy / (px[neg_cell] * p_neg))

    out[valid] = mi
    return out


def _metric_arrays(values: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=int)
    pos_mask = y == 1
    neg_mask = y == 0
    pos = values[pos_mask, :]
    neg = values[neg_mask, :]
    n_pos = int(pos.shape[0])
    n_neg = int(neg.shape[0])
    n_features = int(values.shape[1])

    auc_signed = np.full(n_features, np.nan, dtype=float)
    auc_abs = np.full(n_features, np.nan, dtype=float)
    cliff = np.full(n_features, np.nan, dtype=float)
    if n_pos > 0 and n_neg > 0:
        non_constant = np.nanstd(values, axis=0) > 0
        ranks = rankdata(values, method="average", axis=0)
        pos_rank_sum = np.sum(ranks[pos_mask, :], axis=0)
        u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
        signed = np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0)
        auc_signed[non_constant] = signed[non_constant]
        auc_abs[non_constant] = np.maximum(signed[non_constant], 1.0 - signed[non_constant])
        cliff[non_constant] = 2.0 * signed[non_constant] - 1.0

    pooled_iqr = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    pooled_std = np.nanstd(values, axis=0)
    scale = np.where(pooled_iqr > 0, pooled_iqr, pooled_std)
    median_positive = np.nanmedian(pos, axis=0) if n_pos else np.full(n_features, np.nan)
    median_negative = np.nanmedian(neg, axis=0) if n_neg else np.full(n_features, np.nan)
    median_diff = median_positive - median_negative

    if n_pos and n_neg and n_pos == n_neg:
        wasserstein = np.nanmean(np.abs(np.sort(pos, axis=0) - np.sort(neg, axis=0)), axis=0)
    elif n_pos and n_neg:
        wasserstein = np.asarray(
            [float(wasserstein_distance(pos[:, idx], neg[:, idx])) for idx in range(n_features)],
            dtype=float,
        )
    else:
        wasserstein = np.full(n_features, np.nan, dtype=float)
    wasserstein_norm = np.divide(
        wasserstein,
        scale,
        out=np.full(n_features, np.nan, dtype=float),
        where=np.isfinite(scale) & (scale > 0),
    )

    mutual_info = _mutual_information_columns(values, y)
    return {
        "auc_signed": auc_signed,
        "auc_abs": auc_abs,
        "auc_lift": auc_abs - 0.5,
        "wasserstein": wasserstein,
        "wasserstein_norm": wasserstein_norm,
        "cliff_delta": cliff,
        "abs_cliff_delta": np.abs(cliff),
        "mutual_info": mutual_info,
        "median_positive": median_positive,
        "median_negative": median_negative,
        "median_diff": median_diff,
    }


def _array_nanmean(records: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    values = np.stack([record[key] for record in records], axis=0)
    finite = np.isfinite(values)
    sums = np.nansum(values, axis=0)
    counts = finite.sum(axis=0)
    return np.divide(sums, counts, out=np.full(values.shape[1], np.nan, dtype=float), where=counts > 0)


def _array_nanstd(records: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    values = np.stack([record[key] for record in records], axis=0)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = _array_nanmean(records, key)
    centered = np.where(finite, values - means, 0.0)
    var = np.divide(
        np.sum(centered * centered, axis=0),
        counts,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    return np.sqrt(var)


def _nanmean(records: list[dict[str, float]], key: str) -> float:
    values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _nanstd(records: list[dict[str, float]], key: str) -> float:
    values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
    return float(np.nanstd(values)) if np.isfinite(values).any() else np.nan


def evaluate_feature_discrimination(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    random_state: int = 42,
    comparisons: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    balance_repeats: int = 1,
    balance_min_ratio: float = 1.2,
    progress: bool = False,
) -> pd.DataFrame:
    """对每个特征计算多组二分类判别指标。

    当 balance_repeats > 1 且正负样本数量不一致时，每轮把多数类无放回
    下采样到少数类数量，再对各轮指标取均值，降低类别数量不均衡带来的估计偏差。
    """

    comparisons = COMPARISONS if comparisons is None else comparisons
    x_all = impute_with_median(numeric_feature_frame(frame, feature_columns))
    rows: list[dict[str, object]] = []

    for comp_name, (positive_labels, negative_labels) in comparisons.items():
        comp_start = time.perf_counter()
        labels = frame["source_label"].astype(str)
        mask = labels.isin(positive_labels + negative_labels)
        if not bool(mask.any()):
            continue
        y_all = labels.loc[mask].isin(positive_labels).astype(int).to_numpy()
        x = x_all.loc[mask, :].reset_index(drop=True)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        n_pos_total = int(len(pos_idx))
        n_neg_total = int(len(neg_idx))
        n_per_class = int(min(n_pos_total, n_neg_total))
        n_max_class = int(max(n_pos_total, n_neg_total))
        imbalance_ratio = float(n_max_class / n_per_class) if n_per_class > 0 else np.inf
        use_balanced_repeats = bool(
            balance_repeats > 1
            and n_pos_total != n_neg_total
            and n_per_class > 0
            and imbalance_ratio >= float(balance_min_ratio)
        )
        rng = np.random.default_rng(random_state)
        if use_balanced_repeats:
            sample_indices = []
            for _ in range(balance_repeats):
                pos_sample = pos_idx if n_pos_total == n_per_class else rng.choice(pos_idx, size=n_per_class, replace=False)
                neg_sample = neg_idx if n_neg_total == n_per_class else rng.choice(neg_idx, size=n_per_class, replace=False)
                sample_indices.append(np.concatenate([pos_sample, neg_sample]))
            sample_policy = "balanced_downsample_mean"
            n_positive = n_per_class
            n_negative = n_per_class
            repeat_count = int(balance_repeats)
        else:
            sample_indices = [np.arange(len(y_all))]
            sample_policy = "full_sample"
            n_positive = n_pos_total
            n_negative = n_neg_total
            repeat_count = 1
        all_values = x.loc[:, list(feature_columns)].to_numpy(dtype=float)
        repeat_records = [
            _metric_arrays(all_values[sample_idx, :], y_all[sample_idx])
            for sample_idx in sample_indices
        ]
        metric_mean = {
            key: _array_nanmean(repeat_records, key)
            for key in (
                "auc_signed",
                "auc_abs",
                "auc_lift",
                "wasserstein",
                "wasserstein_norm",
                "cliff_delta",
                "abs_cliff_delta",
                "mutual_info",
                "median_positive",
                "median_negative",
                "median_diff",
            )
        }
        metric_std = {
            key: _array_nanstd(repeat_records, key)
            for key in ("auc_abs", "wasserstein_norm", "abs_cliff_delta", "mutual_info")
        }
        for feature_idx, feature in enumerate(feature_columns):
            rows.append(
                {
                    "comparison": comp_name,
                    "feature": feature,
                    "positive_labels": ",".join(positive_labels),
                    "negative_labels": ",".join(negative_labels),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "n_positive_total": n_pos_total,
                    "n_negative_total": n_neg_total,
                    "balance_repeats": repeat_count,
                    "sample_policy": sample_policy,
                    "auc_signed": float(metric_mean["auc_signed"][feature_idx]),
                    "auc_abs": float(metric_mean["auc_abs"][feature_idx]),
                    "auc_abs_std": float(metric_std["auc_abs"][feature_idx]),
                    "auc_lift": float(metric_mean["auc_lift"][feature_idx]),
                    "wasserstein": float(metric_mean["wasserstein"][feature_idx]),
                    "wasserstein_norm": float(metric_mean["wasserstein_norm"][feature_idx]),
                    "wasserstein_norm_std": float(metric_std["wasserstein_norm"][feature_idx]),
                    "cliff_delta": float(metric_mean["cliff_delta"][feature_idx]),
                    "abs_cliff_delta": float(metric_mean["abs_cliff_delta"][feature_idx]),
                    "abs_cliff_delta_std": float(metric_std["abs_cliff_delta"][feature_idx]),
                    "mutual_info": float(metric_mean["mutual_info"][feature_idx]),
                    "mutual_info_std": float(metric_std["mutual_info"][feature_idx]),
                    "median_positive": float(metric_mean["median_positive"][feature_idx]),
                    "median_negative": float(metric_mean["median_negative"][feature_idx]),
                    "median_diff": float(metric_mean["median_diff"][feature_idx]),
                }
            )
        if progress:
            elapsed = time.perf_counter() - comp_start
            print(
                f"{comp_name}: {len(feature_columns)} features, "
                f"{repeat_count} repeat(s), {sample_policy}, {elapsed:.1f}s"
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
