"""多特征组合搜索：mRMR前缀、近似ReliefF和顺序前向选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_schema import impute_with_median, numeric_feature_frame


def build_binary_frame(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    positive_labels: tuple[str, ...],
    negative_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    """按标签抽取二分类矩阵。"""

    labels = frame["source_label"].astype(str)
    mask = labels.isin(positive_labels + negative_labels)
    y = labels.loc[mask].isin(positive_labels).astype(int).to_numpy()
    x = impute_with_median(numeric_feature_frame(frame.loc[mask], feature_columns))
    return x, y


def lda_auc_score(x: pd.DataFrame, y: np.ndarray, cv_splits: int = 5, random_state: int = 42) -> tuple[float, float]:
    """用LDA交叉验证估计组合特征的线性可分性。"""

    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return 0.5, 0.0
    counts = np.bincount(y)
    n_splits = int(min(cv_splits, counts.min())) if counts.size > 1 else 0
    if n_splits < 2:
        return 0.5, 0.0
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    seps: list[float] = []
    for train_idx, test_idx in cv.split(x, y):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        model.fit(x.iloc[train_idx], y[train_idx])
        score = model.decision_function(x.iloc[test_idx])
        auc = float(roc_auc_score(y[test_idx], score))
        aucs.append(max(auc, 1.0 - auc))
        pos = score[y[test_idx] == 1]
        neg = score[y[test_idx] == 0]
        pooled = np.sqrt((np.var(pos) + np.var(neg)) / 2.0) + 1e-12
        seps.append(float(abs(np.mean(pos) - np.mean(neg)) / pooled))
    return float(np.mean(aucs)), float(np.mean(seps))


def evaluate_mrmr_prefixes(
    frame: pd.DataFrame,
    mrmr_rank: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    positive_labels: tuple[str, ...] = ("BK00", "BK05"),
    negative_labels: tuple[str, ...] = ("FL00", "FL05", "QJ00", "QJ05"),
    counts: tuple[int, ...] = (5, 10, 20, 30, 50),
    random_state: int = 42,
) -> pd.DataFrame:
    """评估mRMR排序前K个特征组成的候选集合。"""

    ranked = [f for f in mrmr_rank["feature"].tolist() if f in feature_columns]
    x_all, y = build_binary_frame(frame, feature_columns, positive_labels, negative_labels)
    rows: list[dict[str, object]] = []
    for k in counts:
        selected = ranked[: min(k, len(ranked))]
        if not selected:
            continue
        auc, sep = lda_auc_score(x_all[selected], y, random_state=random_state)
        rows.append(
            {
                "method": "mRMR_prefix",
                "comparison": "BK_NONBK",
                "feature_count": len(selected),
                "cv_auc_abs": auc,
                "lda_separation": sep,
                "selected_features": ";".join(selected),
            }
        )
    return pd.DataFrame(rows)


def relief_like_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    positive_labels: tuple[str, ...] = ("BK00", "BK05"),
    negative_labels: tuple[str, ...] = ("FL00", "FL05", "QJ00", "QJ05"),
    max_rows: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    """近似ReliefF：特征能拉近同类、拉远异类则得分更高。"""

    x, y = build_binary_frame(frame, feature_columns, positive_labels, negative_labels)
    if len(x) > max_rows:
        rng = np.random.default_rng(random_state)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_pos = min(len(pos_idx), max_rows // 2)
        n_neg = min(len(neg_idx), max_rows - n_pos)
        keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), rng.choice(neg_idx, n_neg, replace=False)])
        x = x.iloc[keep].reset_index(drop=True)
        y = y[keep]
    scaled = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
    scores = pd.Series(0.0, index=scaled.columns)
    for cls in [0, 1]:
        own = scaled.iloc[y == cls]
        other = scaled.iloc[y != cls]
        if len(own) < 2 or other.empty:
            continue
        hit_nn = NearestNeighbors(n_neighbors=2).fit(own)
        miss_nn = NearestNeighbors(n_neighbors=1).fit(other)
        _, hit_idx = hit_nn.kneighbors(own)
        _, miss_idx = miss_nn.kneighbors(own)
        hit = own.iloc[hit_idx[:, 1]].reset_index(drop=True)
        miss = other.iloc[miss_idx[:, 0]].reset_index(drop=True)
        base = own.reset_index(drop=True)
        scores += (base - miss).abs().mean(axis=0) - (base - hit).abs().mean(axis=0)
    out = pd.DataFrame({"feature": scores.index, "relief_score": scores.values})
    return out.sort_values("relief_score", ascending=False)


def sequential_forward_search(
    frame: pd.DataFrame,
    candidate_features: list[str],
    feature_columns: list[str] | tuple[str, ...],
    positive_labels: tuple[str, ...] = ("BK00", "BK05"),
    negative_labels: tuple[str, ...] = ("FL00", "FL05", "QJ00", "QJ05"),
    max_selected: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """顺序前向选择：每轮加入使LDA交叉验证AUC提升最大的特征。"""

    x_all, y = build_binary_frame(frame, feature_columns, positive_labels, negative_labels)
    remaining = [f for f in candidate_features if f in x_all.columns]
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    best_auc = 0.5

    while remaining and len(selected) < max_selected:
        best_feature = None
        best_sep = 0.0
        round_auc = -np.inf
        for feature in remaining:
            trial = selected + [feature]
            auc, sep = lda_auc_score(x_all[trial], y, random_state=random_state)
            if auc > round_auc:
                round_auc = auc
                best_sep = sep
                best_feature = feature
        if best_feature is None:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        best_auc = max(best_auc, float(round_auc))
        rows.append(
            {
                "method": "SFS_LDA",
                "comparison": "BK_NONBK",
                "step": len(selected),
                "added_feature": best_feature,
                "feature_count": len(selected),
                "cv_auc_abs": float(round_auc),
                "best_auc_so_far": best_auc,
                "lda_separation": best_sep,
                "selected_features": ";".join(selected),
            }
        )
    return pd.DataFrame(rows)
