"""mRMR和最终特征评分。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_relevance_series(discrimination: pd.DataFrame, comparison: str = "BK_NONBK") -> pd.Series:
    """从单特征判别表中提取某一比较任务的相关性分数。"""

    subset = discrimination[discrimination["comparison"].eq(comparison)]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.set_index("feature")["discrimination_score"].astype(float).sort_values(ascending=False)


def run_mrmr_ranking(
    relevance: pd.Series,
    corr: pd.DataFrame,
    top_n: int = 80,
    redundancy_weight: float = 0.5,
) -> pd.DataFrame:
    """基于相关矩阵实现轻量mRMR贪心排序。"""

    candidates = [f for f in relevance.index if f in corr.columns]
    selected: list[str] = []
    rows: list[dict[str, object]] = []

    while candidates and len(selected) < min(top_n, len(candidates) + len(selected)):
        best_feature = None
        best_score = -np.inf
        best_redundancy = 0.0
        for feature in candidates:
            rel = float(relevance.get(feature, 0.0))
            if selected:
                redundancy = float(corr.loc[feature, selected].abs().mean())
            else:
                redundancy = 0.0
            score = rel - redundancy_weight * redundancy
            if score > best_score:
                best_score = score
                best_feature = feature
                best_redundancy = redundancy
        if best_feature is None:
            break
        selected.append(best_feature)
        candidates.remove(best_feature)
        rows.append(
            {
                "mrmr_rank": len(selected),
                "feature": best_feature,
                "relevance_score": float(relevance.get(best_feature, 0.0)),
                "mean_abs_corr_to_selected": best_redundancy,
                "mrmr_score": best_score,
            }
        )
    return pd.DataFrame(rows)


def _score_map(frame: pd.DataFrame, key: str, value: str) -> pd.Series:
    if frame.empty or key not in frame.columns or value not in frame.columns:
        return pd.Series(dtype=float)
    return frame.set_index(key)[value].astype(float)


def build_final_ranking(
    feature_columns: list[str] | tuple[str, ...],
    discrimination: pd.DataFrame,
    stability: pd.DataFrame,
    cross_flow: pd.DataFrame,
    redundancy: pd.DataFrame,
    mrmr_rank: pd.DataFrame,
) -> pd.DataFrame:
    """融合判别、稳定性、跨流速一致性和冗余得到最终等级。"""

    disc_pivot = discrimination.pivot_table(
        index="feature",
        columns="comparison",
        values="discrimination_score",
        aggfunc="max",
    )
    stable_top20 = _score_map(stability, "feature", "top20_frequency")
    stable_rank = _score_map(stability, "feature", "rank_stability_score")
    cross_score = _score_map(cross_flow, "feature", "cross_flow_score")
    flow_sensitivity = _score_map(cross_flow, "feature", "flow_sensitivity_penalty")
    redundancy_penalty = _score_map(redundancy, "feature", "redundancy_penalty")
    mrmr = _score_map(mrmr_rank, "feature", "mrmr_rank")

    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        bk_nonbk = float(disc_pivot.loc[feature, "BK_NONBK"]) if feature in disc_pivot.index and "BK_NONBK" in disc_pivot.columns else 0.0
        bk_qj = float(disc_pivot.loc[feature, "BK_QJ"]) if feature in disc_pivot.index and "BK_QJ" in disc_pivot.columns else 0.0
        stability_score = 0.6 * float(stable_top20.get(feature, 0.0)) + 0.4 * float(stable_rank.get(feature, 0.0))
        cf_score = float(cross_score.get(feature, 0.0))
        flow_penalty = float(flow_sensitivity.get(feature, 0.0))
        red_penalty = float(redundancy_penalty.get(feature, 0.0))
        raw_score = (0.42 * bk_nonbk + 0.23 * bk_qj + 0.20 * stability_score + 0.15 * cf_score) / (
            1.0 + 0.65 * flow_penalty + 0.45 * red_penalty
        )
        rows.append(
            {
                "feature": feature,
                "final_score": raw_score,
                "bk_nonbk_score": bk_nonbk,
                "bk_qj_score": bk_qj,
                "bootstrap_stability_score": stability_score,
                "cross_flow_score": cf_score,
                "flow_sensitivity_penalty": flow_penalty,
                "redundancy_penalty": red_penalty,
                "mrmr_rank": int(mrmr.get(feature)) if pd.notna(mrmr.get(feature, np.nan)) else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)
    q75 = float(out["final_score"].quantile(0.75)) if len(out) else 0.0
    q45 = float(out["final_score"].quantile(0.45)) if len(out) else 0.0
    out["feature_grade"] = "C"
    out.loc[(out["final_score"] >= q45) | (out["final_rank"] <= 80), "feature_grade"] = "B"
    out.loc[(out["final_score"] >= q75) & (out["redundancy_penalty"] < 0.92) & (out["bootstrap_stability_score"] > 0), "feature_grade"] = "A"
    return out
