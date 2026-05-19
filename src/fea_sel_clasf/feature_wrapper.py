"""Stage-B wrapper feature selection via RFECV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

from .types import LoadedFeatureTable, RFECVResult


def _build_cv(y: np.ndarray, random_state: int, n_splits: int = 5) -> StratifiedKFold:
    counts = np.bincount(y.astype(int))
    min_class = int(counts[counts > 0].min()) if counts.size and np.any(counts > 0) else 2
    splits = max(2, min(n_splits, min_class))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def run_rfecv_selection(
    table: LoadedFeatureTable,
    train_mask: pd.Series,
    candidate_features: list[str] | tuple[str, ...],
    random_state: int = 42,
    min_features_to_select: int = 5,
    scoring: str = "f1",
) -> RFECVResult:
    """Run RFECV on the filtered feature set and return the surviving features."""
    frame = table.frame.loc[train_mask].copy()
    y = frame[table.label_column].to_numpy(dtype=int)
    x = frame.loc[:, list(candidate_features)].apply(pd.to_numeric, errors="coerce")
    if not x.empty:
        x = x.copy()
        x.loc[:, x.isna().all()] = 0.0
    if len(candidate_features) == 0:
        raise ValueError("candidate_features is empty")
    if len(candidate_features) == 1:
        summary = pd.DataFrame(
            {
                "feature": list(candidate_features),
                "rfecv_support": [True],
                "rfecv_ranking": [1],
                "selected_final": [True],
            }
        )
        return RFECVResult(
            summary=summary,
            selected_features=tuple(candidate_features),
            estimator_name="LinearSVC",
            cv_scoring=str(scoring),
            cv_best_mean_score=None,
            cv_best_std_score=None,
            cv_best_n_features=1,
            cv_curve=None,
        )
    cv = _build_cv(y, random_state=random_state)

    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual=False,
                    random_state=random_state,
                    max_iter=10_000,
                ),
            ),
        ]
    )
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring=scoring,
        importance_getter="named_steps.classifier.coef_",
        min_features_to_select=min(min_features_to_select, max(1, len(candidate_features))),
        n_jobs=-1,
    )
    selector.fit(x, y)

    # RFECV CV curve (n_features -> mean/std test score) for reporting/debugging.
    cv_curve = None
    cv_best_mean = None
    cv_best_std = None
    cv_best_n_features = None
    if hasattr(selector, "cv_results_") and isinstance(getattr(selector, "cv_results_"), dict):
        # Newer scikit-learn versions.
        cv_results = getattr(selector, "cv_results_", {})
        cv_df = pd.DataFrame(
            {
                key: value
                for key, value in cv_results.items()
                if np.asarray(value).ndim == 1
            }
        )
        if "mean_test_score" in cv_df.columns:
            best_idx = int(cv_df["mean_test_score"].to_numpy().argmax())
            cv_best_mean = float(cv_df.loc[best_idx, "mean_test_score"])
            if "std_test_score" in cv_df.columns:
                cv_best_std = float(cv_df.loc[best_idx, "std_test_score"])
            if "n_features" in cv_df.columns:
                cv_best_n_features = int(cv_df.loc[best_idx, "n_features"])
        cv_curve_cols = [c for c in ["n_features", "mean_test_score", "std_test_score"] if c in cv_df.columns]
        if cv_curve_cols:
            cv_curve = cv_df.loc[:, cv_curve_cols].copy().sort_values(
                ["n_features"] if "n_features" in cv_curve_cols else cv_curve_cols
            )
    elif hasattr(selector, "grid_scores_"):
        # Backwards-compatible path for older scikit-learn versions.
        scores = np.asarray(getattr(selector, "grid_scores_"), dtype=float)
        if scores.size:
            best_idx = int(np.nanargmax(scores))
            cv_best_mean = float(scores[best_idx])
            # std is not available via grid_scores_ in old sklearn.
            cv_best_std = None
            # With step=1, RFECV evaluates feature counts: p, p-1, ..., min_features_to_select.
            p = int(x.shape[1])
            cv_best_n_features = int(p - best_idx)
            n_features_seq = list(range(p, min_features_to_select - 1, -1))
            n_features_seq = n_features_seq[: scores.size]
            cv_curve = pd.DataFrame({"n_features": n_features_seq, "mean_test_score": scores})

    selected = tuple(x.columns[selector.support_].tolist())
    summary = pd.DataFrame(
        {
            "feature": x.columns,
            "rfecv_support": selector.support_,
            "rfecv_ranking": selector.ranking_,
            "selected_final": selector.support_,
        }
    )
    summary = summary.sort_values(["selected_final", "rfecv_ranking", "feature"], ascending=[False, True, True]).reset_index(drop=True)
    return RFECVResult(
        summary=summary,
        selected_features=selected,
        estimator_name="LinearSVC",
        cv_scoring=str(scoring),
        cv_best_mean_score=cv_best_mean,
        cv_best_std_score=cv_best_std,
        cv_best_n_features=cv_best_n_features if cv_best_n_features is not None else int(getattr(selector, "n_features_", len(selected))),
        cv_curve=cv_curve,
    )
