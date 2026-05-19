"""Evaluation helpers for binary classification outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the standard metrics requested by the plan."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall_pos = recall_score(y_true, y_pred, zero_division=0)
    precision_pos = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {
        "n_samples": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_pos": float(recall_pos),
        "precision_pos": float(precision_pos),
        "specificity_neg": float(specificity),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def confusion_matrix_frame(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray, threshold: float) -> pd.DataFrame:
    """Return a 2x2 confusion matrix as a dataframe."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def write_metrics_json(path: str | Path, metrics: dict[str, float | int | str]) -> None:
    """Persist metrics as a UTF-8 JSON file."""
    path = Path(path)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
