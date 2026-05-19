"""Decision threshold selection from out-of-fold probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from .types import ThresholdResult


def choose_threshold_from_oof(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    target_recall: float = 0.95,
) -> ThresholdResult:
    """Pick the smallest-FPR threshold that still satisfies the recall target."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    feasible = np.where(tpr >= target_recall)[0]
    if feasible.size:
        candidates = pd.DataFrame(
            {
                "threshold": thresholds[feasible],
                "fpr": fpr[feasible],
                "tpr": tpr[feasible],
            }
        ).sort_values(["fpr", "threshold"], ascending=[True, False]).reset_index(drop=True)
        row = candidates.iloc[0]
        threshold = float(row["threshold"])
        method = "min_fpr_at_target_recall"
    else:
        # Fallback: maximize balanced accuracy on the OOF curve.
        ba = (tpr + (1.0 - fpr)) / 2.0
        best = int(np.argmax(ba))
        threshold = float(thresholds[best])
        method = "max_balanced_accuracy_fallback"

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn + 1e-12)
    fpr_value = fp / (fp + tn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    return ThresholdResult(
        threshold=threshold,
        recall_pos=float(recall),
        fpr=float(fpr_value),
        balanced_accuracy=float(balanced_accuracy),
        method=method,
    )
