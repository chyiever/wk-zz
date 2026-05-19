"""Candidate classifiers and model search helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .types import LoadedFeatureTable, ModelSearchResult


@dataclass(frozen=True)
class CandidateModelSpec:
    """One classifier family and its search grid."""

    name: str
    builder: Callable[[int], object]
    param_grid: dict[str, list[object]]


def _build_cv(y: np.ndarray, random_state: int, n_splits: int = 5) -> StratifiedKFold:
    counts = np.bincount(y.astype(int))
    min_class = int(counts[counts > 0].min()) if counts.size and np.any(counts > 0) else 2
    splits = max(2, min(n_splits, min_class))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def build_candidate_models(random_state: int = 42, run_mode: str = "cpu") -> list[CandidateModelSpec]:
    """Return the candidate classifier pool required by the plan."""
    mode = str(run_mode).strip().lower()
    if mode not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported run_mode={run_mode!r}, expected 'cpu' or 'gpu'")

    models: list[CandidateModelSpec] = [
        CandidateModelSpec(
            name="logistic_regression",
            builder=lambda seed: LogisticRegression(
                class_weight="balanced",
                max_iter=4_000,
                solver="liblinear",
                random_state=seed,
            ),
            param_grid={"classifier__C": [0.5, 1.0, 2.0, 4.0, 8.0]},
        ),
        CandidateModelSpec(
            name="rbf_svm",
            builder=lambda seed: SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
            param_grid={"classifier__C": [0.5, 1.0, 2.0, 4.0, 8.0], "classifier__gamma": ["scale", 1e-3, 3e-3, 1e-2]},
        ),
        CandidateModelSpec(
            name="random_forest",
            builder=lambda seed: RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=seed,
                n_estimators=400,
                n_jobs=1,
            ),
            param_grid={"classifier__max_depth": [None, 8, 16], "classifier__min_samples_leaf": [1, 2]},
        ),
    ]

    try:  # pragma: no cover - optional dependency
        from xgboost import XGBClassifier

        xgb_kwargs = {
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": 1.0,
            "n_jobs": 1,
        }
        if mode == "gpu":
            xgb_kwargs.update({"tree_method": "hist", "device": "cuda"})

        models.append(
            CandidateModelSpec(
                name=f"xgboost_{mode}",
                builder=lambda seed: XGBClassifier(**{**xgb_kwargs, "random_state": seed}),
                param_grid={"classifier__max_depth": [3, 4, 5], "classifier__learning_rate": [0.03, 0.05, 0.1]},
            )
        )
    except Exception:
        pass
    return models


def _build_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _selection_key(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["mean_recall_pos"]),
        float(row["mean_balanced_accuracy"]),
        -float(row["std_recall_pos"]),
        -float(row["std_balanced_accuracy"]),
    )


def search_best_model(
    table: LoadedFeatureTable,
    train_mask: pd.Series,
    selected_features: list[str] | tuple[str, ...],
    random_state: int = 42,
    run_mode: str = "cpu",
    scoring: str | None = None,
) -> ModelSearchResult:
    """
    Search the candidate model pool on the training-domain subset.

    The search key follows the executable plan:
    1. maximize cross-validated positive recall,
    2. then balanced accuracy,
    3. then prefer lower variance across folds.
    """
    del scoring  # The selector uses explicit metrics and a custom ranking rule.
    frame = table.frame.loc[train_mask].copy()
    y = frame[table.label_column].to_numpy(dtype=int)
    x = frame.loc[:, list(selected_features)].apply(pd.to_numeric, errors="coerce")
    if not x.empty:
        x = x.copy()
        x.loc[:, x.isna().all()] = 0.0
    cv = _build_cv(y, random_state=random_state)

    records: list[dict[str, object]] = []
    best_pipeline: Pipeline | None = None
    best_name = ""
    best_params: dict[str, object] = {}
    best_key: tuple[float, float, float, float] | None = None

    for spec in build_candidate_models(random_state=random_state, run_mode=run_mode):
        base_pipeline = _build_pipeline(spec.builder(random_state))
        for params in ParameterGrid(spec.param_grid):
            pipeline = clone(base_pipeline).set_params(**params)
            scores = cross_validate(
                pipeline,
                x,
                y,
                cv=cv,
                scoring={
                    "recall_pos": make_scorer(recall_score, zero_division=0),
                    "balanced_accuracy": "balanced_accuracy",
                    "precision_pos": make_scorer(precision_score, zero_division=0),
                    "f1_pos": make_scorer(f1_score, zero_division=0),
                    "roc_auc": "roc_auc",
                },
                return_train_score=False,
                n_jobs=1,
            )
            row = {
                "model_name": spec.name,
                "params": params,
                "mean_recall_pos": float(np.mean(scores["test_recall_pos"])),
                "std_recall_pos": float(np.std(scores["test_recall_pos"], ddof=1) if len(scores["test_recall_pos"]) > 1 else 0.0),
                "mean_balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
                "std_balanced_accuracy": float(np.std(scores["test_balanced_accuracy"], ddof=1) if len(scores["test_balanced_accuracy"]) > 1 else 0.0),
                "mean_precision_pos": float(np.mean(scores["test_precision_pos"])),
                "mean_f1_pos": float(np.mean(scores["test_f1_pos"])),
                "mean_auroc": float(np.mean(scores["test_roc_auc"])),
            }
            records.append(row)
            key = _selection_key(pd.Series(row))
            if best_key is None or key > best_key:
                best_key = key
                best_pipeline = clone(pipeline)
                best_name = spec.name
                best_params = params

    cv_results = pd.DataFrame(records).sort_values(
        ["mean_recall_pos", "mean_balanced_accuracy", "std_recall_pos", "std_balanced_accuracy"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    if best_pipeline is None:
        raise RuntimeError("model search did not produce any candidate")

    best_pipeline.fit(x, y)
    oof_prob = cross_val_predict(best_pipeline, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    fitted_imputer = best_pipeline.named_steps["imputer"]
    fitted_scaler = best_pipeline.named_steps["scaler"]
    fitted_classifier = best_pipeline.named_steps["classifier"]
    return ModelSearchResult(
        model_name=best_name,
        params=best_params,
        cv_results=cv_results,
        pipeline=best_pipeline,
        oof_probability=pd.Series(oof_prob, index=frame.index, name="oof_probability"),
        oof_label=pd.Series(y, index=frame.index, name="label"),
        imputer=fitted_imputer,
        scaler=fitted_scaler,
        classifier=fitted_classifier,
        selected_features=tuple(selected_features),
    )
