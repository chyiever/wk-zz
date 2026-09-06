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


def compare_correlation_methods(
    pearson_pairs: pd.DataFrame,
    spearman_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """对比两种相关方法识别出的高相关特征对。"""

    def _normalize(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["feature_min", "feature_max", value_name])
        out = frame.copy()
        out["feature_min"] = out[["feature_a", "feature_b"]].min(axis=1)
        out["feature_max"] = out[["feature_a", "feature_b"]].max(axis=1)
        return out[["feature_min", "feature_max", value_name]]

    pearson = _normalize(pearson_pairs.rename(columns={"correlation": "pearson_correlation"}), "pearson_correlation")
    spearman = _normalize(spearman_pairs.rename(columns={"correlation": "spearman_correlation"}), "spearman_correlation")
    merged = pearson.merge(spearman, on=["feature_min", "feature_max"], how="outer")
    merged = merged.rename(columns={"feature_min": "feature_a", "feature_max": "feature_b"})
    merged["abs_pearson"] = merged["pearson_correlation"].abs()
    merged["abs_spearman"] = merged["spearman_correlation"].abs()
    merged["method_hit"] = np.select(
        [
            merged["pearson_correlation"].notna() & merged["spearman_correlation"].notna(),
            merged["pearson_correlation"].notna(),
            merged["spearman_correlation"].notna(),
        ],
        ["Pearson+Spearman", "Pearson_only", "Spearman_only"],
        default="none",
    )
    return merged.sort_values(["method_hit", "abs_spearman", "abs_pearson"], ascending=[True, False, False])


def build_redundancy_recommendations(
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
    relevance: pd.Series,
    threshold: float = 0.92,
) -> pd.DataFrame:
    """合并Pearson/Spearman高相关图，输出每个冗余组的保留建议。"""

    features = list(pearson_corr.columns.intersection(spearman_corr.columns))
    adjacency = {feature: set() for feature in features}
    for corr in (pearson_corr, spearman_corr):
        for i, a in enumerate(features):
            related = corr.index[corr.loc[a, features].abs() >= threshold].tolist()
            for b in related:
                if a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    group_id = 0
    for feature in features:
        if feature in seen or not adjacency[feature]:
            continue
        group_id += 1
        stack = [feature]
        group: list[str] = []
        seen.add(feature)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        ordered = sorted(group, key=lambda x: float(relevance.get(x, 0.0)), reverse=True)
        keep = ordered[0]
        drop = ordered[1:]
        sub_p = pearson_corr.loc[ordered, ordered].abs().to_numpy()
        sub_s = spearman_corr.loc[ordered, ordered].abs().to_numpy()
        tri = np.triu_indices(len(ordered), k=1)
        p_values = sub_p[tri] if len(ordered) > 1 else np.array([])
        s_values = sub_s[tri] if len(ordered) > 1 else np.array([])
        rows.append(
            {
                "redundancy_group_id": group_id,
                "feature_count": len(ordered),
                "features_in_group": ";".join(ordered),
                "recommended_keep": keep,
                "recommended_drop": ";".join(drop),
                "max_abs_pearson": float(np.nanmax(p_values)) if p_values.size else np.nan,
                "max_abs_spearman": float(np.nanmax(s_values)) if s_values.size else np.nan,
                "pearson_high_pair_count": int(np.sum(p_values >= threshold)) if p_values.size else 0,
                "spearman_high_pair_count": int(np.sum(s_values >= threshold)) if s_values.size else 0,
                "recommendation_reason": "保留组内BK_NONBK判别分最高的特征，其余视为高相关冗余候选",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "redundancy_group_id",
                "feature_count",
                "features_in_group",
                "recommended_keep",
                "recommended_drop",
                "max_abs_pearson",
                "max_abs_spearman",
                "pearson_high_pair_count",
                "spearman_high_pair_count",
                "recommendation_reason",
            ]
        )
    return pd.DataFrame(rows).sort_values(["feature_count", "max_abs_spearman"], ascending=[False, False])


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
        # 非代表特征即使与代表特征相关性完全相同、判别分并列，也应受到
        # 冗余惩罚，否则最终排序会保留大量物理含义重复的特征。
        max_corr_to_better = abs(corr_to_rep) if feature != rep and pd.notna(corr_to_rep) else 0.0
        better = relevance[relevance > relevance.get(feature, -np.inf)].index.intersection(corr.columns)
        if len(better):
            max_corr_to_better = max(max_corr_to_better, float(corr.loc[feature, better].abs().max()))
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
