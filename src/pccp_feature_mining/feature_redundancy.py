"""特征冗余分析与相关簇构建。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import impute_with_median, numeric_feature_frame


def compute_correlation_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    method: str = "spearman",
    max_rows: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """计算特征相关矩阵；大表先抽样以控制耗时。"""

    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    if len(x) > max_rows:
        x = x.sample(n=max_rows, random_state=random_state)
    return x.corr(method=method).fillna(0.0)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame:
    """输出高相关特征对。"""

    values = corr.to_numpy()
    cols = list(corr.columns)
    rows: list[dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(values[i, j])
            if abs(c) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": c, "abs_correlation": abs(c)})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False) if rows else pd.DataFrame(
        columns=["feature_a", "feature_b", "correlation", "abs_correlation"]
    )


def build_correlation_clusters(
    corr: pd.DataFrame,
    relevance: pd.Series,
    threshold: float = 0.92,
) -> pd.DataFrame:
    """按相关阈值构建贪心簇，优先保留判别分高的代表特征。"""

    ordered = list(relevance.sort_values(ascending=False).index)
    assigned: dict[str, int] = {}
    representatives: dict[int, str] = {}
    cluster_id = 0

    for feature in ordered:
        if feature in assigned:
            continue
        cluster_id += 1
        assigned[feature] = cluster_id
        representatives[cluster_id] = feature
        related = corr.index[corr.loc[feature].abs() >= threshold].tolist()
        for other in related:
            assigned.setdefault(other, cluster_id)

    rows: list[dict[str, object]] = []
    for feature in corr.columns:
        cid = assigned.get(feature)
        rep = representatives.get(cid, feature)
        corr_to_rep = float(corr.loc[feature, rep]) if feature in corr.index and rep in corr.columns else np.nan
        max_corr_to_better = 0.0
        better = relevance[relevance > relevance.get(feature, -np.inf)].index.intersection(corr.columns)
        if len(better):
            max_corr_to_better = float(corr.loc[feature, better].abs().max())
        rows.append(
            {
                "feature": feature,
                "cluster_id": cid,
                "cluster_representative": rep,
                "is_representative": feature == rep,
                "corr_to_representative": corr_to_rep,
                "max_abs_corr_to_better_feature": max_corr_to_better,
                "redundancy_penalty": max_corr_to_better,
                "relevance_score": float(relevance.get(feature, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id", "is_representative", "relevance_score"], ascending=[True, False, False])
