"""Shared data structures for the cross-condition classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedFeatureTable:
    """A loaded feature table plus the inferred feature column list."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    meta_columns: tuple[str, ...]
    label_column: str
    split_column: str
    domain_column: str


@dataclass(frozen=True)
class ExperimentSpec:
    """One cross-condition experiment definition."""

    name: str
    train_domains: tuple[str, ...]
    test_domains: tuple[str, ...]


@dataclass(frozen=True)
class StageAResult:
    """Output of the first-stage feature filter."""

    scores: pd.DataFrame
    selected_features: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class RFECVResult:
    """Output of the wrapper feature selection stage."""

    summary: pd.DataFrame
    selected_features: tuple[str, ...]
    estimator_name: str

    # Optional RFECV cross-validation diagnostics (does not affect selection logic).
    cv_scoring: str | None = None
    cv_best_mean_score: float | None = None
    cv_best_std_score: float | None = None
    cv_best_n_features: int | None = None
    cv_curve: pd.DataFrame | None = None


@dataclass(frozen=True)
class ModelSearchResult:
    """Best classifier and the associated cross-validation summary."""

    model_name: str
    params: dict[str, object]
    cv_results: pd.DataFrame
    pipeline: object
    oof_probability: pd.Series
    oof_label: pd.Series
    imputer: object
    scaler: object
    classifier: object
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdResult:
    """Decision threshold derived from out-of-fold probabilities."""

    threshold: float
    recall_pos: float
    fpr: float
    balanced_accuracy: float
    method: str


@dataclass(frozen=True)
class ExperimentResult:
    """One finished experiment bundle."""

    spec: ExperimentSpec
    output_dir: Path
    stage_a: StageAResult
    rfecv: RFECVResult
    model: ModelSearchResult
    threshold: ThresholdResult
    summary: pd.DataFrame
