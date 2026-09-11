"""特征挖掘后的分类验证：逻辑回归、线性SVM和RBF-SVM。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .event_parser import source_stem
from .feature_schema import numeric_feature_frame


CLASSIFICATION_COMPARISONS = {
    "BK00_vs_NONBK00": (("BK00",), ("FL00", "QJ00")),
    "BK00_vs_QJ00": (("BK00",), ("QJ00",)),
    "BK05_vs_QJ05": (("BK05",), ("QJ05",)),
    "BK05_vs_NONBK05": (("BK05",), ("FL05", "QJ05")),
    "BK_vs_NONBK": (("BK00", "BK05"), ("FL00", "FL05", "QJ00", "QJ05")),
}


def _classification_group_id(frame: pd.DataFrame) -> pd.Series:
    """分类验证采用更严格的源文件分组，避免同源窗口泄漏。"""

    if "source_file_name" in frame.columns:
        src = frame["source_file_name"].map(source_stem)
    elif "source_file_path" in frame.columns:
        src = frame["source_file_path"].map(source_stem)
    else:
        src = pd.Series([f"row_{i}" for i in frame.index], index=frame.index)
    return frame["source_label"].astype(str) + "::" + src.astype(str)


def _split_by_label_groups(
    subset: pd.DataFrame,
    train_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """每个来源标签内部按源文件组划分训练/测试。"""

    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []
    groups = _classification_group_id(subset)
    for _, label_frame in subset.groupby("source_label", sort=True):
        label_groups = groups.loc[label_frame.index].drop_duplicates().to_numpy()
        rng.shuffle(label_groups)
        if len(label_groups) <= 1:
            n_train = 1
        else:
            n_train = min(max(1, int(round(len(label_groups) * train_ratio))), len(label_groups) - 1)
        train_groups = set(label_groups[:n_train])
        is_train = groups.loc[label_frame.index].isin(train_groups)
        train_idx.extend(label_frame.index[is_train].tolist())
        test_idx.extend(label_frame.index[~is_train].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _sample_train_balance(
    frame: pd.DataFrame,
    y: pd.Series,
    max_train_rows_per_class: int | None,
    random_state: int,
) -> pd.Index:
    """限制训练集多数类规模，避免SVM被海量Other拖慢。"""

    if max_train_rows_per_class is None:
        return frame.index
    rng = np.random.default_rng(random_state)
    keep: list[int] = []
    for cls in sorted(y.unique()):
        idx = frame.index[y.eq(cls)].to_numpy()
        if len(idx) > max_train_rows_per_class:
            idx = rng.choice(idx, size=max_train_rows_per_class, replace=False)
        keep.extend(idx.tolist())
    return pd.Index(keep)


def _feature_order_for_comparison(
    comparison: str,
    feature_discrimination: pd.DataFrame,
    final_ranking: pd.DataFrame,
    comparison_feature_order: dict[str, str] | None = None,
    feature_order_source: str = "discrimination",
) -> list[str]:
    """优先使用对应比较任务的单特征分数，缺失时退回最终总排名。"""

    if feature_order_source not in {"discrimination", "final_ranking"}:
        raise ValueError("feature_order_source must be 'discrimination' or 'final_ranking'")
    if feature_order_source == "final_ranking":
        score_col = "final_score" if "final_score" in final_ranking.columns else "condition_final_score"
        if score_col in final_ranking.columns:
            return final_ranking.sort_values(score_col, ascending=False)["feature"].tolist()
        return final_ranking["feature"].tolist() if "feature" in final_ranking.columns else []

    mapping = {
        "BK00_vs_NONBK00": "BK00_NONBK00",
        "BK00_vs_QJ00": "BK00_QJ00",
        "BK05_vs_QJ05": "BK05_QJ05",
        "BK05_vs_NONBK05": "BK05_NONBK05",
        "BK_vs_NONBK": "BK_NONBK",
    }
    if comparison_feature_order:
        mapping.update(comparison_feature_order)
    key = mapping.get(comparison)
    subset = feature_discrimination[feature_discrimination["comparison"].eq(key)] if key else pd.DataFrame()
    if not subset.empty:
        return subset.sort_values("discrimination_score", ascending=False)["feature"].tolist()
    return final_ranking.sort_values("final_score", ascending=False)["feature"].tolist()


def _build_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "svm_linear": SVC(kernel="linear", class_weight="balanced", random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=random_state),
    }


def run_classification_tests(
    frame: pd.DataFrame,
    feature_discrimination: pd.DataFrame,
    final_ranking: pd.DataFrame,
    output_dir: Path | None = None,
    feature_counts: tuple[int, ...] = (5, 10, 20, 30, 50),
    train_ratio: float = 0.7,
    max_train_rows_per_class: int | None = 3000,
    random_state: int = 42,
    comparisons: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    comparison_feature_order: dict[str, str] | None = None,
    feature_order_source: str = "discrimination",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对五个关键任务做分类验证，并输出推荐特征数量。"""

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    models = _build_models(random_state)
    active_comparisons = CLASSIFICATION_COMPARISONS if comparisons is None else comparisons

    for comparison, (pos_labels, neg_labels) in active_comparisons.items():
        labels = frame["source_label"].astype(str)
        subset = frame.loc[labels.isin(pos_labels + neg_labels)].copy()
        if subset.empty:
            continue
        y_all = subset["source_label"].isin(pos_labels).astype(int)
        train_idx, test_idx = _split_by_label_groups(subset, train_ratio=train_ratio, random_state=random_state)
        train_frame = subset.loc[train_idx]
        test_frame = subset.loc[test_idx]
        y_train_full = y_all.loc[train_idx]
        y_test = y_all.loc[test_idx]
        sampled_train_idx = _sample_train_balance(train_frame, y_train_full, max_train_rows_per_class, random_state)
        train_frame = train_frame.loc[sampled_train_idx]
        y_train = y_all.loc[sampled_train_idx]

        ordered_features = _feature_order_for_comparison(
            comparison,
            feature_discrimination,
            final_ranking,
            comparison_feature_order=comparison_feature_order,
            feature_order_source=feature_order_source,
        )
        for k in feature_counts:
            selected = [f for f in ordered_features[:k] if f in frame.columns]
            if not selected or y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            x_train = numeric_feature_frame(train_frame, selected)
            x_test = numeric_feature_frame(test_frame, selected)
            for model_name, estimator in models.items():
                pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
                pipe.fit(x_train, y_train)
                pred = pipe.predict(x_test)
                if hasattr(pipe, "decision_function"):
                    score = pipe.decision_function(x_test)
                else:
                    score = pipe.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, score))
                tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
                row = {
                    "comparison": comparison,
                    "model": model_name,
                    "feature_count": len(selected),
                    "train_rows": int(len(train_frame)),
                    "test_rows": int(len(test_frame)),
                    "train_positive": int(y_train.sum()),
                    "train_negative": int((y_train == 0).sum()),
                    "test_positive": int(y_test.sum()),
                    "test_negative": int((y_test == 0).sum()),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "auc": auc,
                    "f1_positive": float(f1_score(y_test, pred, zero_division=0)),
                    "recall_positive": float(tp / (tp + fn)) if tp + fn else 0.0,
                    "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
                    "feature_order_source": feature_order_source,
                    "selected_features": ";".join(selected),
                }
                rows.append(row)
                if output_dir is not None:
                    model_dir = output_dir / "models" / comparison / model_name / f"top_{len(selected)}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(pipe, model_dir / "pipeline.pkl")

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()
    results["selection_score"] = 0.45 * results["balanced_accuracy"] + 0.35 * results["auc"] + 0.20 * results["recall_positive"]
    for comparison, group in results.groupby("comparison", sort=True):
        best = group.sort_values(["selection_score", "feature_count"], ascending=[False, True]).iloc[0]
        feature_rows.append(
            {
                "comparison": comparison,
                "recommended_feature_count": int(best["feature_count"]),
                "recommended_model": best["model"],
                "selection_score": float(best["selection_score"]),
                "balanced_accuracy": float(best["balanced_accuracy"]),
                "auc": float(best["auc"]),
                "recall_positive": float(best["recall_positive"]),
                "specificity": float(best["specificity"]),
                "feature_order_source": best["feature_order_source"],
                "recommended_features": best["selected_features"],
                "split_note": "按source_file_name分组，且在每个来源标签内划分训练/测试；训练集多数类最多采样到max_train_rows_per_class。",
            }
        )
    recommendations = pd.DataFrame(feature_rows).sort_values("comparison")
    return results.sort_values(["comparison", "selection_score"], ascending=[True, False]), recommendations
