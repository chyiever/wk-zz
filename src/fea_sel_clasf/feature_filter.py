"""Stage-A feature filtering and ranking."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from .types import LoadedFeatureTable, StageAResult


def _numeric_feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")


def _pairwise_wasserstein_stability(values: pd.DataFrame, domains: pd.Series) -> pd.Series:
    """Higher is better: 1 means domain distributions are identical."""
    unique_domains = [domain for domain in sorted(domains.dropna().unique()) if str(domain).strip()]
    if len(unique_domains) < 2:
        return pd.Series(1.0, index=values.columns)

    scores: dict[str, float] = {}
    domain_masks = {domain: (domains == domain).to_numpy() for domain in unique_domains}
    for feature in values.columns:
        per_domain = []
        for domain in unique_domains:
            samples = values.loc[domain_masks[domain], feature].dropna().to_numpy(dtype=float)
            if samples.size:
                per_domain.append((domain, samples))
        if len(per_domain) < 2:
            scores[feature] = 1.0
            continue

        distances = []
        pooled = np.concatenate([array for _, array in per_domain])
        scale = float(np.nanstd(pooled)) + 1e-12
        for (_, arr_a), (_, arr_b) in combinations(per_domain, 2):
            distances.append(wasserstein_distance(arr_a, arr_b) / scale)
        mean_distance = float(np.mean(distances)) if distances else 0.0
        scores[feature] = float(1.0 / (1.0 + mean_distance))
    return pd.Series(scores)


def run_stage_a_filter(
    table: LoadedFeatureTable,
    train_mask: pd.Series,
    missing_threshold: float = 0.2,
    low_variance_threshold: float = 1e-8,
    corr_threshold: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Stage-A scores for all features on the training rows only.

    This function performs the Stage-A *scoring* once and returns a score table
    for **all** features (one row per feature). It does *not* depend on Top-K.

    Use :func:`slice_stage_a_scores` to derive a specific Top-K selection
    (and a :class:`~src.fea_sel_clasf.types.StageAResult`) without recomputing
    mutual information / stability.
    """
    frame = table.frame.loc[train_mask].copy()
    feature_columns = list(table.feature_columns)
    x_raw = _numeric_feature_matrix(frame, tuple(feature_columns))
    if not x_raw.empty:
        x_raw = x_raw.copy()
        x_raw.loc[:, x_raw.isna().all()] = 0.0
    y = frame[table.label_column].to_numpy(dtype=int)

    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(imputer.fit_transform(x_raw), columns=feature_columns, index=frame.index)

    missing_rate = x_raw.isna().mean(axis=0)
    variance = x_raw.var(axis=0, ddof=0).fillna(0.0)
    corr_with_label: dict[str, float] = {}
    for feature in feature_columns:
        series = x_imputed[feature].to_numpy(dtype=float)
        if series.size == 0 or np.allclose(series, series[0]):
            corr_with_label[feature] = 0.0
            continue
        corr = np.corrcoef(series, y)[0, 1]
        corr_with_label[feature] = float(abs(corr)) if np.isfinite(corr) else 0.0

    valid_features = [
        feature
        for feature in feature_columns
        if float(missing_rate[feature]) <= missing_threshold and float(variance[feature]) > low_variance_threshold
    ]

    if valid_features:
        mi_values = mutual_info_classif(
            x_imputed[valid_features].to_numpy(dtype=float),
            y,
            random_state=random_state,
        )
        stability = _pairwise_wasserstein_stability(x_imputed[valid_features], frame[table.domain_column])
        mi_series = pd.Series(mi_values, index=valid_features, dtype=float)
        mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-12)
        score_series = 0.7 * mi_norm + 0.3 * stability.reindex(valid_features).fillna(0.0)
    else:
        mi_series = pd.Series(dtype=float)
        stability = pd.Series(dtype=float)
        mi_norm = pd.Series(dtype=float)
        score_series = pd.Series(dtype=float)

    score_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_rate": [float(missing_rate[col]) for col in feature_columns],
            "variance": [float(variance[col]) for col in feature_columns],
            "abs_corr_label": [float(corr_with_label.get(col, 0.0)) for col in feature_columns],
            "mutual_info": [float(mi_series.get(col, np.nan)) for col in feature_columns],
            "mi_norm": [float(mi_norm.get(col, np.nan)) for col in feature_columns],
            "stability": [float(stability.get(col, np.nan)) for col in feature_columns],
            "score": [float(score_series.get(col, np.nan)) for col in feature_columns],
        }
    )
    score_frame["keep_missing"] = score_frame["missing_rate"] <= missing_threshold
    score_frame["keep_variance"] = score_frame["variance"] > low_variance_threshold
    score_frame["eligible"] = score_frame["keep_missing"] & score_frame["keep_variance"] & score_frame["score"].notna()

    eligible = score_frame.loc[score_frame["eligible"]].sort_values(
        ["score", "abs_corr_label", "stability"],
        ascending=[False, False, False],
    )
    eligible_features = eligible["feature"].tolist()

    if eligible_features:
        corr_matrix = x_imputed[eligible_features].corr().abs().fillna(0.0)
        kept: list[str] = []
        for feature in eligible_features:
            if all(float(corr_matrix.loc[feature, other]) <= corr_threshold for other in kept):
                kept.append(feature)
    else:
        kept = []

    score_frame["keep_corr"] = score_frame["feature"].isin(kept)
    score_frame["stage_a_rank"] = np.nan
    ranked = score_frame.loc[score_frame["keep_corr"]].sort_values(
        ["score", "abs_corr_label", "stability"],
        ascending=[False, False, False],
    ).copy()
    ranked["stage_a_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    rank_map = ranked.set_index("feature")["stage_a_rank"]
    score_frame["stage_a_rank"] = score_frame["feature"].map(rank_map)

    # Selection markers are Top-K dependent. Initialize them in a neutral state.
    score_frame["selected_stage_a"] = False
    score_frame["selected_stage_a_rank"] = score_frame["stage_a_rank"]

    # Keep ordering stable and human-friendly.
    return score_frame.sort_values(["keep_corr", "score"], ascending=[False, False]).reset_index(drop=True)


def slice_stage_a_scores(scores: pd.DataFrame, top_k: int) -> StageAResult:
    """Select the Top-K Stage-A features from a precomputed score table.

    Parameters
    ----------
    scores:
        Output from :func:`run_stage_a_filter`.
    top_k:
        Number of top-ranked (after correlation pruning) features to keep.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    required = {"feature", "keep_corr", "stage_a_rank", "score", "abs_corr_label", "stability"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing required columns: {sorted(missing)}")

    view = scores.copy()
    ranked = (
        view.loc[view["keep_corr"] & view["stage_a_rank"].notna()]
        .sort_values(["stage_a_rank"], ascending=[True])
    )
    selected = ranked.head(top_k)
    selected_features = tuple(selected["feature"].tolist())

    view["selected_stage_a"] = view["feature"].isin(selected_features)
    view["selected_stage_a_rank"] = view["stage_a_rank"]

    return StageAResult(
        scores=view.sort_values(["selected_stage_a", "score"], ascending=[False, False]).reset_index(drop=True),
        selected_features=selected_features,
        top_k=int(top_k),
    )
