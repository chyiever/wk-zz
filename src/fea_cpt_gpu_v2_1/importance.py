"""Feature-importance ranking methods commonly used in feature engineering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance


def rank_feature_importance(X: pd.DataFrame, y: Iterable, task: str = "classification", random_state: int = 42, top_k: int | None = 20) -> dict[str, pd.DataFrame]:
    """Return 3 common feature-importance rankings."""
    y = np.asarray(list(y))
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")

    if task == "classification":
        mi = mutual_info_classif(X, y, random_state=random_state)
        model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        mi = mutual_info_regression(X, y, random_state=random_state)
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)

    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=random_state, n_jobs=-1)

    outputs = {
        "mutual_info": pd.DataFrame({"feature": X.columns, "score": mi}).sort_values("score", ascending=False),
        "random_forest": pd.DataFrame({"feature": X.columns, "score": model.feature_importances_}).sort_values("score", ascending=False),
        "permutation": pd.DataFrame({"feature": X.columns, "score": perm.importances_mean}).sort_values("score", ascending=False),
    }
    if top_k is not None:
        outputs = {name: frame.head(top_k).reset_index(drop=True) for name, frame in outputs.items()}
    return outputs
