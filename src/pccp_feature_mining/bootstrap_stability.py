"""Bootstrap平衡抽样稳定性分析。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .feature_schema import impute_with_median, numeric_feature_frame


def _sample_other_indices(frame: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> np.ndarray:
    """按采样组抽取Other样本，QJ保持事件完整，FL按窗口独立。"""

    other = frame[frame["target_label"].eq("Other")]
    group_to_indices = [idx.to_numpy() for _, idx in other.groupby("sampling_group_id", sort=False).groups.items()]
    if not group_to_indices:
        return np.array([], dtype=int)

    order = rng.permutation(len(group_to_indices))
    picked: list[int] = []
    for pos in order:
        picked.extend(group_to_indices[int(pos)].tolist())
        if len(picked) >= target_rows:
            break
    if len(picked) < target_rows:
        extra_groups = rng.choice(len(group_to_indices), size=len(group_to_indices), replace=True)
        for pos in extra_groups:
            picked.extend(group_to_indices[int(pos)].tolist())
            if len(picked) >= target_rows:
                break
    return np.asarray(picked[:target_rows], dtype=int)


def _auc_scores(x: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """计算一轮bootstrap内所有特征的无方向AUC分离度。"""

    scores: dict[str, float] = {}
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        if np.nanstd(values) == 0 or len(np.unique(y)) < 2:
            scores[col] = 0.0
            continue
        try:
            auc = float(roc_auc_score(y, values))
        except ValueError:
            auc = 0.5
        scores[col] = max(auc, 1.0 - auc) - 0.5
    return pd.Series(scores).sort_values(ascending=False)


def run_bootstrap_stability(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    rounds: int = 200,
    top_ks: tuple[int, ...] = (10, 20, 30),
    random_state: int = 42,
) -> pd.DataFrame:
    """重复“全部BK + 等量Other”抽样，统计Top-K出现频率。"""

    rng = np.random.default_rng(random_state)
    x_all = impute_with_median(numeric_feature_frame(frame, feature_columns))
    bk_indices = frame.index[frame["target_label"].eq("BK")].to_numpy()
    if bk_indices.size == 0:
        raise ValueError("bootstrap稳定性分析需要至少一个BK样本")

    rank_sum = pd.Series(0.0, index=feature_columns)
    rank_sq_sum = pd.Series(0.0, index=feature_columns)
    score_sum = pd.Series(0.0, index=feature_columns)
    top_counts = {k: pd.Series(0, index=feature_columns, dtype=int) for k in top_ks}

    for _ in range(rounds):
        other_indices = _sample_other_indices(frame, int(bk_indices.size), rng)
        sample_indices = np.concatenate([bk_indices, other_indices])
        y = frame.loc[sample_indices, "target_label"].eq("BK").astype(int).to_numpy()
        scores = _auc_scores(x_all.loc[sample_indices, :], y)
        ranks = pd.Series(np.arange(1, len(scores) + 1), index=scores.index).reindex(feature_columns).fillna(len(scores))
        rank_sum += ranks
        rank_sq_sum += ranks.pow(2)
        score_sum += scores.reindex(feature_columns).fillna(0.0)
        for k in top_ks:
            top_counts[k].loc[scores.head(k).index] += 1

    mean_rank = rank_sum / rounds
    rank_std = (rank_sq_sum / rounds - mean_rank.pow(2)).clip(lower=0).pow(0.5)
    rows = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "mean_rank": mean_rank.values,
            "rank_std": rank_std.values,
            "mean_auc_lift": (score_sum / rounds).values,
        }
    )
    for k in top_ks:
        rows[f"top{k}_frequency"] = (top_counts[k] / rounds).values
    if "top20_frequency" not in rows.columns:
        nearest = min(top_ks, key=lambda k: abs(k - 20))
        rows["top20_frequency"] = rows[f"top{nearest}_frequency"]
    max_rank = max(len(feature_columns), 1)
    rows["rank_stability_score"] = (1.0 - (rows["mean_rank"] - 1.0) / max_rank).clip(0.0, 1.0)
    return rows.sort_values(["top20_frequency", "mean_auc_lift", "mean_rank"], ascending=[False, False, True])
