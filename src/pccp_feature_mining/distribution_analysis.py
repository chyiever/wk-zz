"""六类特征分布分析：箱线图、KDE、PCA和可选UMAP。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from .feature_schema import impute_with_median, numeric_feature_frame
from .visualization import AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, _configure_font


def select_distribution_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    top_n: int = 12,
) -> pd.DataFrame:
    """选择六类之间差异较明显的特征用于分布图展示。

    这里使用“类均值方差 / 类内方差”的朴素F比值，不参与最终评分，只用于
    Notebook02 的可视化筛查。
    """

    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    rows: list[dict[str, object]] = []
    labels = frame["source_label"].astype(str)
    for feature in feature_columns:
        values = x[feature]
        group_means = values.groupby(labels).mean()
        between = float(group_means.var(ddof=0))
        within = float(values.groupby(labels).var(ddof=0).mean())
        score = between / (within + 1e-12)
        rows.append({"feature": feature, "six_class_f_ratio": score, "between_var": between, "within_var": within})
    return pd.DataFrame(rows).sort_values("six_class_f_ratio", ascending=False).head(top_n)


def run_pca_projection(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    max_rows: int = 8000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """标准化全部特征后做PCA二维投影。"""

    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    meta = frame[["source_label", "target_label", "signal_family", "flow_condition", "event_id"]].copy()
    if len(x) > max_rows:
        sampled = x.sample(n=max_rows, random_state=random_state).index
        x = x.loc[sampled]
        meta = meta.loc[sampled]
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(scaled)
    projection = meta.reset_index(drop=True)
    projection["PC1"] = emb[:, 0]
    projection["PC2"] = emb[:, 1]
    explained = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return projection, explained


def run_umap_projection(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    max_rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """可选UMAP投影；环境未安装umap-learn时返回空表。"""

    try:
        import umap  # type: ignore
    except Exception:
        return pd.DataFrame(
            columns=["source_label", "target_label", "signal_family", "flow_condition", "event_id", "UMAP1", "UMAP2"]
        )

    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    meta = frame[["source_label", "target_label", "signal_family", "flow_condition", "event_id"]].copy()
    if len(x) > max_rows:
        sampled = x.sample(n=max_rows, random_state=random_state).index
        x = x.loc[sampled]
        meta = meta.loc[sampled]
    scaled = RobustScaler().fit_transform(x)
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=random_state).fit_transform(scaled)
    projection = meta.reset_index(drop=True)
    projection["UMAP1"] = emb[:, 0]
    projection["UMAP2"] = emb[:, 1]
    return projection


def compute_kde_overlap_summary(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    label_col: str = "source_label",
    grid_size: int = 256,
    max_samples_per_group: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """用KDE重叠面积汇总多组分布差异。

    对每个特征和每一对标签，先用 Gaussian KDE 估计概率密度，再计算
    overlap = integral(min(p_i(x), p_j(x)) dx)。overlap 越小、1-overlap 越大，
    两组分布差异越明显。
    """

    if not features or label_col not in frame.columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "kde_pair_count",
                "kde_overlap_mean",
                "kde_overlap_min",
                "kde_overlap_max",
                "kde_separation_mean",
                "kde_separation_max",
            ]
        )

    rng = np.random.default_rng(random_state)
    x = impute_with_median(numeric_feature_frame(frame, features))
    labels = frame[label_col].astype(str).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for feature in features:
        values = pd.to_numeric(x[feature].reset_index(drop=True), errors="coerce")
        group_values: dict[str, np.ndarray] = {}
        for label in sorted(labels.dropna().unique()):
            arr = values.loc[labels.eq(label)].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > max_samples_per_group:
                arr = rng.choice(arr, size=max_samples_per_group, replace=False)
            if len(arr) >= 2 and float(np.nanstd(arr)) > 0:
                group_values[label] = arr

        if len(group_values) < 2:
            rows.append(
                {
                    "feature": feature,
                    "kde_pair_count": 0,
                    "kde_overlap_mean": np.nan,
                    "kde_overlap_min": np.nan,
                    "kde_overlap_max": np.nan,
                    "kde_separation_mean": np.nan,
                    "kde_separation_max": np.nan,
                }
            )
            continue

        pooled = np.concatenate(list(group_values.values()))
        lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        grid = np.linspace(float(lo), float(hi), grid_size)

        densities: dict[str, np.ndarray] = {}
        for label, arr in group_values.items():
            try:
                density = gaussian_kde(arr, bw_method="scott")(grid)
            except Exception:
                hist, edges = np.histogram(arr, bins=min(64, max(8, len(arr) // 10)), range=(lo, hi), density=True)
                centers = (edges[:-1] + edges[1:]) / 2.0
                density = np.interp(grid, centers, hist, left=0.0, right=0.0)
            area = float(np.trapz(density, grid))
            densities[label] = density / area if area > 0 else density

        overlaps: list[float] = []
        density_items = list(densities.items())
        for i in range(len(density_items)):
            for j in range(i + 1, len(density_items)):
                overlap = float(np.trapz(np.minimum(density_items[i][1], density_items[j][1]), grid))
                overlaps.append(float(np.clip(overlap, 0.0, 1.0)))

        overlap_arr = np.asarray(overlaps, dtype=float)
        rows.append(
            {
                "feature": feature,
                "kde_pair_count": int(len(overlaps)),
                "kde_overlap_mean": float(np.nanmean(overlap_arr)),
                "kde_overlap_min": float(np.nanmin(overlap_arr)),
                "kde_overlap_max": float(np.nanmax(overlap_arr)),
                "kde_separation_mean": float(1.0 - np.nanmean(overlap_arr)),
                "kde_separation_max": float(1.0 - np.nanmin(overlap_arr)),
            }
        )

    return pd.DataFrame(rows).sort_values(["kde_separation_mean", "kde_separation_max"], ascending=[False, False])


def select_bk_0_30ms_samples(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one 0-30 ms feature row per BK event and report the selection audit.

    Each physical BK event is expanded into six overlapping windows during feature
    extraction.  Sample-size analyses must not count those correlated rows as six
    independent observations, so this selector keeps only the canonical ``0_30``
    window and verifies that an event does not contribute more than one row.
    """

    required = {"source_label", "event_id", "window_mode", "window_start_ms", "window_end_ms"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"BK 0-30 ms selection requires columns: {', '.join(missing)}")

    labels = frame["source_label"].astype(str)
    bk = frame.loc[labels.str.startswith("BK")].copy()
    if bk.empty:
        raise ValueError("No BK rows were found for the 0-30 ms sample analysis")

    starts = pd.to_numeric(bk["window_start_ms"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(bk["window_end_ms"], errors="coerce").to_numpy(dtype=float)
    target_mask = (
        bk["window_mode"].astype(str).str.strip().eq("0_30").to_numpy()
        & np.isclose(starts, 0.0, rtol=0.0, atol=1e-6)
        & np.isclose(ends, 30.0, rtol=0.0, atol=1e-6)
    )
    selected = bk.loc[target_mask].copy()

    audit_rows: list[dict[str, object]] = []
    for label, group in bk.groupby("source_label", sort=True):
        group_target = selected.loc[selected["source_label"].astype(str).eq(str(label))]
        target_counts = group_target.groupby("event_id", dropna=False).size()
        all_events = group["event_id"].nunique(dropna=False)
        audit_rows.append(
            {
                "source_label": str(label),
                "all_derived_window_rows": int(len(group)),
                "all_bk_events": int(all_events),
                "selected_0_30ms_rows": int(len(group_target)),
                "selected_bk_events": int(group_target["event_id"].nunique(dropna=False)),
                "excluded_derived_window_rows": int(len(group) - len(group_target)),
                "events_missing_0_30ms_window": int(all_events - group_target["event_id"].nunique(dropna=False)),
                "duplicate_0_30ms_rows": int(target_counts.sub(1).clip(lower=0).sum()),
            }
        )

    audit = pd.DataFrame(audit_rows)
    duplicate_mask = selected.duplicated(subset=["source_label", "event_id"], keep=False)
    if bool(duplicate_mask.any()):
        duplicate_events = selected.loc[duplicate_mask, "event_id"].astype(str).nunique()
        raise ValueError(f"Found duplicate 0-30 ms rows for {duplicate_events} BK event(s)")
    if selected.empty:
        raise ValueError("No BK rows matched window_mode='0_30' with 0-30 ms boundaries")
    return selected, audit


def _default_sample_sizes(total_rows: int, min_samples: int = 5) -> list[int]:
    """Build readable sample-size checkpoints up to the available rows."""

    if total_rows <= 0:
        return []
    base = [
        min_samples,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        80,
        100,
        150,
        200,
        300,
        500,
        800,
        1000,
        1500,
        2000,
        3000,
        5000,
        8000,
        10000,
    ]
    sizes = sorted({int(n) for n in base if min_samples <= int(n) <= total_rows})
    if total_rows < min_samples:
        sizes = [total_rows]
    elif total_rows not in sizes:
        sizes.append(int(total_rows))
    return sizes


def build_sample_size_grid(
    total_rows: int,
    min_samples: int = 5,
    dense_until: int = 20,
    growth: float = 1.3,
    max_points: int = 32,
) -> list[int]:
    """Design a resolution-matched sample-size grid for stability curves.

    Bootstrap RSE of a statistic decays roughly like ``n ** -0.5``, so grid
    points should be spaced so that the expected RSE changes by a similar
    relative amount between neighbours:

    * while ``n <= dense_until`` every integer is kept (one extra sample still
      moves ``1/sqrt(n)`` by a visible amount, and the "consecutive points"
      stability rule needs adjacent integers there);
    * above that, sizes grow geometrically with ratio ``growth`` (>= 1.05);
    * the full sample size is always the last point, and geometric points
      within one growth step of it are pruned as redundant;
    * the geometric region is additionally coarsened if it would exceed
      ``max_points`` grid points in total.
    """

    if total_rows < 2:
        return []
    growth = max(float(growth), 1.05)
    start = max(2, min(int(min_samples), int(total_rows)))
    dense_until = min(max(int(dense_until), start), int(total_rows))

    dense = list(range(start, dense_until + 1))
    if int(total_rows) <= dense_until:
        return dense

    last_dense = dense[-1] if dense else start - 1
    room = max(int(max_points) - len(dense) - 1, 1)
    ratio = max(growth, (int(total_rows) / max(last_dense, 1)) ** (1.0 / room))

    geometric: list[int] = []
    current = float(last_dense)
    while True:
        current = math.ceil(current * ratio)
        if current >= int(total_rows):
            break
        geometric.append(int(current))
    geometric = [n for n in geometric if n < int(total_rows) / ratio]

    sizes = sorted(set(dense + geometric + [int(total_rows)]))
    return sizes


def _first_stable_size(
    values: list[float],
    sizes: list[int],
    threshold: float,
    consecutive_points: int,
) -> int | float:
    """Return the first sample size where enough consecutive curve points satisfy a threshold."""

    if not values or not sizes:
        return np.nan
    ok = np.asarray([np.isfinite(v) and v <= threshold for v in values], dtype=bool)
    need = max(int(consecutive_points), 1)
    for start in range(0, len(ok) - need + 1):
        if bool(ok[start : start + need].all()):
            return int(sizes[start])
    return np.nan


def estimate_bootstrap_statistic_stability(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    label_col: str = "source_label",
    sample_sizes: list[int] | tuple[int, ...] | None = None,
    sample_sizes_by_label: dict[str, list[int] | tuple[int, ...]] | None = None,
    repeats: int = 200,
    min_samples: int = 5,
    rse_threshold: float = 0.10,
    consecutive_points: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap five statistics per feature and summarize their relative SE.

    For every source label and sample size, the function draws ``repeats`` samples
    with replacement.  Each replicate estimates the mean, standard deviation and
    Q10/Q50/Q90 of every feature.  For statistic ``theta``, relative standard error
    is ``std(theta_boot) / abs(mean(theta_boot))``.  A scale-aware numerical floor
    is used only when a statistic is zero or extremely close to zero; these cases
    remain visible through the large RSE and ``denominator_was_floored`` flag.

    Returns ``detail`` (one row per label/size/feature/statistic), ``curve`` (median,
    P90 and maximum RSE at each label/size), and ``summary`` (the first sample size
    whose P90 RSE stays below the threshold for the requested consecutive points).
    """

    if label_col not in frame.columns:
        raise KeyError(f"{label_col!r} not found in frame")
    if not feature_columns:
        detail_columns = [
            label_col, "total_rows", "sample_size", "feature", "statistic",
            "bootstrap_estimate", "bootstrap_se", "rse", "denominator_was_floored",
        ]
        curve_columns = [
            label_col, "total_rows", "sample_size", "median_rse", "p90_rse",
            "max_rse", "feature_statistic_count", "repeats_used", "is_full_sample_size",
        ]
        summary_columns = [
            label_col, "total_rows", "recommended_stable_sample_size", "stability_status",
            "rse_threshold", "consecutive_points", "final_median_rse", "final_p90_rse",
            "final_max_rse",
        ]
        return (
            pd.DataFrame(columns=detail_columns),
            pd.DataFrame(columns=curve_columns),
            pd.DataFrame(columns=summary_columns),
        )

    labels = frame[label_col].astype(str)
    statistic_names = np.asarray(["mean", "std", "q10", "q50", "q90"], dtype=object)
    detail_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for label in sorted(labels.dropna().unique()):
        label_mask = labels.eq(label).to_numpy()
        group_frame = impute_with_median(numeric_feature_frame(frame.loc[label_mask, :], feature_columns))
        group = group_frame.to_numpy(dtype=float)
        total_rows = int(group.shape[0])
        if total_rows == 0:
            continue

        label_sample_sizes = None if sample_sizes_by_label is None else sample_sizes_by_label.get(str(label))
        configured_sizes = label_sample_sizes if label_sample_sizes is not None else sample_sizes
        sizes = list(configured_sizes) if configured_sizes is not None else build_sample_size_grid(total_rows, min_samples)
        sizes = sorted({int(n) for n in sizes if 2 <= int(n) <= total_rows})
        if not sizes:
            continue
        if total_rows not in sizes:
            sizes.append(total_rows)

        seed_payload = f"bootstrap-statistics\0{int(random_state)}\0{label}".encode("utf-8")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little"))
        curve_for_label: list[dict[str, object]] = []

        feature_scale = np.nanpercentile(np.abs(group), 90, axis=0)
        feature_scale = np.where(np.isfinite(feature_scale), feature_scale, 0.0)

        for size in sizes:
            current_repeats = max(int(repeats), 2)
            boot = np.empty((current_repeats, 5, group.shape[1]), dtype=float)
            for rep in range(current_repeats):
                sampled = group[rng.choice(total_rows, size=size, replace=True), :]
                boot[rep, 0, :] = np.mean(sampled, axis=0)
                boot[rep, 1, :] = np.std(sampled, axis=0, ddof=1)
                boot[rep, 2:5, :] = np.quantile(sampled, [0.10, 0.50, 0.90], axis=0)

            estimates = np.mean(boot, axis=0)
            bootstrap_se = np.std(boot, axis=0, ddof=1)
            denominator_floor = np.maximum(feature_scale * 1e-12, np.finfo(float).eps)
            denominator = np.maximum(np.abs(estimates), denominator_floor[None, :])
            denominator_was_floored = np.abs(estimates) < denominator_floor[None, :]
            rse = bootstrap_se / denominator

            finite_rse = rse[np.isfinite(rse)]
            curve_row = {
                label_col: str(label),
                "total_rows": total_rows,
                "sample_size": int(size),
                "median_rse": float(np.median(finite_rse)) if finite_rse.size else np.nan,
                "p90_rse": float(np.percentile(finite_rse, 90)) if finite_rse.size else np.nan,
                "max_rse": float(np.max(finite_rse)) if finite_rse.size else np.nan,
                "feature_statistic_count": int(finite_rse.size),
                "repeats_used": current_repeats,
                "is_full_sample_size": bool(size >= total_rows),
            }
            curve_rows.append(curve_row)
            curve_for_label.append(curve_row)

            for statistic_index, statistic_name in enumerate(statistic_names):
                for feature_index, feature in enumerate(feature_columns):
                    detail_rows.append(
                        {
                            label_col: str(label),
                            "total_rows": total_rows,
                            "sample_size": int(size),
                            "feature": str(feature),
                            "statistic": str(statistic_name),
                            "bootstrap_estimate": float(estimates[statistic_index, feature_index]),
                            "bootstrap_se": float(bootstrap_se[statistic_index, feature_index]),
                            "rse": float(rse[statistic_index, feature_index]),
                            "denominator_was_floored": bool(
                                denominator_was_floored[statistic_index, feature_index]
                            ),
                        }
                    )

        p90_values = [float(row["p90_rse"]) for row in curve_for_label]
        stable_size = _first_stable_size(p90_values, sizes, rse_threshold, consecutive_points)
        final = curve_for_label[-1]
        summary_rows.append(
            {
                label_col: str(label),
                "total_rows": total_rows,
                "recommended_stable_sample_size": int(stable_size) if np.isfinite(stable_size) else np.nan,
                "stability_status": "stable" if np.isfinite(stable_size) else "not_stable_with_current_rows",
                "rse_threshold": float(rse_threshold),
                "consecutive_points": int(max(consecutive_points, 1)),
                "min_sample_size_checked": int(min(sizes)),
                "max_sample_size_checked": int(max(sizes)),
                "final_median_rse": float(final["median_rse"]),
                "final_p90_rse": float(final["p90_rse"]),
                "final_max_rse": float(final["max_rse"]),
            }
        )

    detail = pd.DataFrame(detail_rows)
    curve = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["stability_status", "recommended_stable_sample_size", label_col])
    return detail, curve, summary


def estimate_feature_distribution_mmd(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    label_col: str = "source_label",
    sample_sizes: list[int] | tuple[int, ...] | None = None,
    repeats: int = 100,
    min_samples: int = 5,
    rff_components: int = 512,
    mmd_threshold: float = 0.05,
    consecutive_points: int = 2,
    max_rows: int = 10000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate convergence of the joint feature distribution with Gaussian MMD.

    The complete feature matrix is median-imputed and robustly scaled.  To keep the
    high-dimensional calculation bounded, Gaussian-RBF MMD is approximated with
    random Fourier features (RFF): ``MMD = ||mean(phi(X_n))-mean(phi(X_ref))||_2``.
    The kernel bandwidth uses the median pairwise-distance heuristic.  If ``label_col``
    exists, each source label receives equal weight: ``sample_size`` means rows per
    label and each replicate draws that many rows from every label.  This prevents a
    large FL/QJ class from hiding a small BK class.  The fixed reference is the mean
    of the full empirical class embeddings, so the maximum sample size is not forced
    to produce zero MMD.
    """

    if not feature_columns:
        return (
            pd.DataFrame(columns=[
                "sample_size", "mmd_mean", "mmd_std", "mmd_p90", "repeats_used",
                "reference_rows", "rff_components", "rbf_gamma", "is_full_sample_size",
            ]),
            pd.DataFrame(columns=[
                "recommended_stable_sample_size", "stability_status", "mmd_threshold",
                "consecutive_points", "reference_rows", "rff_components", "rbf_gamma",
            ]),
        )

    x_frame = impute_with_median(numeric_feature_frame(frame, feature_columns))
    x = x_frame.to_numpy(dtype=float)
    if x.shape[0] < 2:
        raise ValueError("MMD convergence analysis requires at least two rows")

    rng = np.random.default_rng(random_state)
    total_rows = int(x.shape[0])
    analysis_rows = min(total_rows, max(int(max_rows), 2))
    labels = frame[label_col].astype(str).to_numpy() if label_col in frame.columns else None
    if analysis_rows < total_rows:
        if labels is None:
            analysis_index = rng.choice(total_rows, size=analysis_rows, replace=False)
        else:
            unique_labels = sorted(pd.unique(labels))
            rows_per_label = max(2, analysis_rows // max(len(unique_labels), 1))
            selected_parts = []
            for label in unique_labels:
                candidates = np.flatnonzero(labels == label)
                take = min(len(candidates), rows_per_label)
                selected_parts.append(rng.choice(candidates, size=take, replace=False))
            analysis_index = np.sort(np.concatenate(selected_parts))
        x = x[analysis_index, :]
        if labels is not None:
            labels = labels[analysis_index]
    x = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    bandwidth_rows = min(len(x), 512)
    bandwidth_index = rng.choice(len(x), size=bandwidth_rows, replace=False)
    bandwidth_sample = x[bandwidth_index, :]
    squared_norm = np.sum(np.square(bandwidth_sample), axis=1)
    squared_distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * bandwidth_sample @ bandwidth_sample.T
    upper = squared_distances[np.triu_indices(bandwidth_rows, k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0.0)]
    median_squared_distance = float(np.median(upper)) if upper.size else 1.0
    gamma = 1.0 / max(2.0 * median_squared_distance, np.finfo(float).eps)

    components = max(int(rff_components), 32)
    weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(x.shape[1], components))
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=components)
    phi = np.sqrt(2.0 / components) * np.cos(x @ weights + offsets)
    if labels is None:
        group_indices = {"all": np.arange(len(phi), dtype=int)}
    else:
        group_indices = {
            str(label): np.flatnonzero(labels == label)
            for label in sorted(pd.unique(labels))
            if np.any(labels == label)
        }
    reference_embedding = np.mean(
        np.stack([np.mean(phi[index, :], axis=0) for index in group_indices.values()], axis=0),
        axis=0,
    )

    max_size_per_label = min(len(index) for index in group_indices.values())
    configured_sizes = (
        list(sample_sizes)
        if sample_sizes is not None
        else build_sample_size_grid(max_size_per_label, min_samples)
    )
    sizes = sorted({int(n) for n in configured_sizes if 2 <= int(n) <= max_size_per_label})
    if not sizes:
        sizes = [max_size_per_label]
    elif max_size_per_label not in sizes:
        sizes.append(max_size_per_label)

    rows: list[dict[str, object]] = []
    for size in sizes:
        current_repeats = max(int(repeats), 2)
        mmd_values = np.empty(current_repeats, dtype=float)
        for rep in range(current_repeats):
            sampled_embeddings = [
                np.mean(phi[rng.choice(index, size=size, replace=True), :], axis=0)
                for index in group_indices.values()
            ]
            delta = np.mean(np.stack(sampled_embeddings, axis=0), axis=0) - reference_embedding
            mmd_values[rep] = float(np.linalg.norm(delta))
        rows.append(
            {
                "sample_size": int(size),
                "sample_size_per_label": int(size),
                "bootstrap_rows_total": int(size * len(group_indices)),
                "mmd_mean": float(np.mean(mmd_values)),
                "mmd_std": float(np.std(mmd_values, ddof=1)),
                "mmd_p90": float(np.percentile(mmd_values, 90)),
                "repeats_used": current_repeats,
                "reference_rows": int(len(x)),
                "source_label_count": int(len(group_indices)),
                "original_total_rows": total_rows,
                "rff_components": components,
                "rbf_gamma": gamma,
                "is_full_sample_size": bool(size >= max_size_per_label),
            }
        )

    curve = pd.DataFrame(rows)
    stable_size = _first_stable_size(curve["mmd_p90"].tolist(), sizes, mmd_threshold, consecutive_points)
    summary = pd.DataFrame(
        [
            {
                "recommended_stable_sample_size": int(stable_size) if np.isfinite(stable_size) else np.nan,
                "stability_status": "converged" if np.isfinite(stable_size) else "not_converged_with_current_rows",
                "mmd_threshold": float(mmd_threshold),
                "consecutive_points": int(max(consecutive_points, 1)),
                "reference_rows": int(len(x)),
                "original_total_rows": total_rows,
                "source_label_count": int(len(group_indices)),
                "max_sample_size_per_label": int(max_size_per_label),
                "rff_components": components,
                "rbf_gamma": gamma,
                "final_mmd_mean": float(curve.iloc[-1]["mmd_mean"]),
                "final_mmd_p90": float(curve.iloc[-1]["mmd_p90"]),
            }
        ]
    )
    return curve, summary


def summarize_bootstrap_rse_by_statistic(
    detail: pd.DataFrame,
    label_col: str = "source_label",
) -> pd.DataFrame:
    """Aggregate feature-level RSE separately for mean/std/Q10/Q50/Q90."""

    required = {label_col, "total_rows", "sample_size", "statistic", "rse"}
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise KeyError(f"Bootstrap RSE detail is missing columns: {', '.join(missing)}")
    if detail.empty:
        return pd.DataFrame(
            columns=[
                label_col,
                "total_rows",
                "sample_size",
                "statistic",
                "median_rse",
                "p90_rse",
                "max_rse",
                "feature_count",
                "floored_denominator_count",
            ]
        )

    work = detail.copy()
    work["rse"] = pd.to_numeric(work["rse"], errors="coerce")
    work = work.loc[np.isfinite(work["rse"].to_numpy(dtype=float))].copy()
    group_columns = [label_col, "total_rows", "sample_size", "statistic"]
    summary = (
        work.groupby(group_columns, sort=True, dropna=False)["rse"]
        .agg(
            median_rse="median",
            p90_rse=lambda values: float(np.percentile(values, 90)),
            max_rse="max",
            feature_count="size",
        )
        .reset_index()
    )
    if "denominator_was_floored" in work.columns:
        floored = (
            work.assign(_floored=work["denominator_was_floored"].fillna(False).astype(bool).astype(int))
            .groupby(group_columns, sort=True, dropna=False)["_floored"]
            .sum()
            .rename("floored_denominator_count")
            .reset_index()
        )
        summary = summary.merge(floored, on=group_columns, how="left")
    else:
        summary["floored_denominator_count"] = 0
    summary["floored_denominator_fraction"] = (
        summary["floored_denominator_count"] / summary["feature_count"].clip(lower=1)
    )
    return summary


def plot_bootstrap_rse_curves(
    curve: pd.DataFrame,
    output_path: Path,
    statistic_curve: pd.DataFrame | None = None,
    threshold: float | None = None,
    label_col: str = "source_label",
) -> None:
    """Plot median and P90 RSE for all five statistics plus their aggregate.

    Each statistic-specific ordinate summarizes feature-level RSE values for one
    source label and sample size.  Dashed curves show the median (typical feature),
    while solid curves show P90 (90% feature coverage).  The sixth panel aggregates
    all five statistics and all features; its P90 still drives the stability decision.
    """

    if curve.empty or label_col not in curve.columns:
        return
    if statistic_curve is None or statistic_curve.empty:
        statistic_curve = pd.DataFrame()

    from matplotlib.ticker import PercentFormatter

    _configure_font()
    statistic_specs = [
        ("mean", "均值 mean：中心水平"),
        ("std", "标准差 std：离散程度"),
        ("q10", "Q10：分布下尾"),
        ("q50", "Q50：中位典型水平"),
        ("q90", "Q90：分布上尾"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6), sharex=False, sharey=False)
    flat_axes = axes.ravel()

    labels = sorted(curve[label_col].dropna().astype(str).unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    label_colors = {label: color_cycle[index % len(color_cycle)] for index, label in enumerate(labels)}

    def _plot_two_summaries(ax: plt.Axes, data: pd.DataFrame) -> None:
        for label, group in data.groupby(label_col):
            label_text = str(label)
            ordered = group.sort_values("sample_size")
            for column, linestyle, marker in (
                ("median_rse", "--", None),
                ("p90_rse", "-", "o"),
            ):
                values = pd.to_numeric(ordered[column], errors="coerce")
                valid = values.gt(0) & np.isfinite(values)
                ax.plot(
                    ordered.loc[valid, "sample_size"],
                    values.loc[valid],
                    color=label_colors.get(label_text),
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.35,
                    markersize=3.0,
                )

    for ax, (statistic, title) in zip(flat_axes[:5], statistic_specs):
        selected = statistic_curve.loc[statistic_curve.get("statistic", pd.Series(dtype=str)).eq(statistic)]
        _plot_two_summaries(ax, selected)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    aggregate_ax = flat_axes[5]
    _plot_two_summaries(aggregate_ax, curve)
    aggregate_ax.set_title("总体：全部特征 × 五种统计量", fontsize=TITLE_FONT_SIZE)

    for ax in flat_axes:
        if threshold is not None and float(threshold) > 0:
            ax.axhline(
                float(threshold), color="tab:red", linestyle="--", linewidth=1.15,
                label=f"阈值 {float(threshold):.0%}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("每类独立样本量 n", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("跨特征 RSE 汇总（无量纲）", fontsize=AXIS_FONT_SIZE)
        ax.grid(True, which="both", alpha=0.22)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
    from matplotlib.lines import Line2D

    source_handles = [
        Line2D([0], [0], color=label_colors[label], linewidth=1.6, label=label)
        for label in labels
    ]
    metric_handles = [
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.6, label="median RSE（典型特征）"),
        Line2D([0], [0], color="0.25", linestyle="-", marker="o", markersize=3.2,
               linewidth=1.6, label="P90 RSE（90%特征覆盖）"),
    ]
    if threshold is not None and float(threshold) > 0:
        metric_handles.append(
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.15,
                   label=f"P90 判稳阈值 {float(threshold):.0%}")
        )
    fig.legend(
        handles=source_handles, title="来源类别", fontsize=TICK_FONT_SIZE,
        title_fontsize=AXIS_FONT_SIZE, ncol=max(len(source_handles), 1),
        loc="upper center", bbox_to_anchor=(0.5, 0.955),
    )
    fig.legend(
        handles=metric_handles, fontsize=TICK_FONT_SIZE, ncol=len(metric_handles),
        loc="upper center", bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle("Bootstrap 统计特征稳定性：median 与 P90 RSE", fontsize=TITLE_FONT_SIZE, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_convergence_curve(curve: pd.DataFrame, output_path: Path, threshold: float | None = None) -> None:
    """Plot mean and P90 Gaussian-MMD convergence curves."""

    if curve.empty:
        return
    _configure_font()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ordered = curve.sort_values("sample_size")
    ax.plot(ordered["sample_size"], ordered["mmd_mean"], marker="o", label="MMD均值")
    ax.plot(ordered["sample_size"], ordered["mmd_p90"], marker="s", label="MMD P90")
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:red", linestyle="--", linewidth=1.2, label="收敛阈值")
    ax.set_xscale("log")
    ax.set_xlabel("样本量 n", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Gaussian-RBF MMD（RFF近似）", fontsize=AXIS_FONT_SIZE)
    ax.set_title("整体特征空间分布收敛", fontsize=TITLE_FONT_SIZE)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=TICK_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def estimate_source_feature_stability(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    label_col: str = "source_label",
    sample_sizes: list[int] | tuple[int, ...] | None = None,
    sample_sizes_by_label: dict[str, list[int] | tuple[int, ...]] | None = None,
    repeats: int = 100,
    min_samples: int = 5,
    pairwise_threshold: float = 0.05,
    reference_threshold: float | None = None,
    consecutive_points: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate when each source label has enough samples for stable feature means.

    Each source label is processed independently. Missing-value imputation, feature
    scaling and the random stream are all label-local, so adding, removing or
    reordering other labels cannot change a label's curve.

    At each sample size, two independent bootstrap mean vectors are compared with
    the label-local relative RMS gap::

        sqrt(mean(((mean_a - mean_b) / within_label_feature_rms) ** 2))

    The feature RMS is ``sqrt(mean(x ** 2))`` over the current label. Unlike z-score
    scaling, this keeps each feature's within-label variation relative to its typical
    signal magnitude, so labels are not forced onto the same theoretical curve. The
    metric is invariant to feature-unit rescaling and to the presence of other labels.
    Both the mean and P90 of repeated bootstrap gaps are reported; the conservative
    P90 drives the stability decision.

    ``sample_sizes_by_label`` overrides the shared/default grid for selected labels,
    allowing small groups to use every integer while large groups keep sparse points.
    """

    if label_col not in frame.columns:
        raise KeyError(f"{label_col!r} not found in frame")
    if reference_threshold is not None:
        pairwise_threshold = float(reference_threshold)
    if not feature_columns:
        empty_curve_cols = [
            label_col,
            "total_rows",
            "sample_size",
            "paired_bootstrap_relative_rms_gap_mean",
            "paired_bootstrap_relative_rms_gap_p90",
            "repeats_used",
            "is_full_sample_size",
        ]
        empty_summary_cols = [
            label_col,
            "total_rows",
            "stable_sample_size_pairwise_bootstrap",
            "recommended_stable_sample_size",
            "stability_status",
            "pairwise_threshold",
        ]
        return pd.DataFrame(columns=empty_curve_cols), pd.DataFrame(columns=empty_summary_cols)

    labels = frame[label_col].astype(str)
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for label in sorted(labels.dropna().unique()):
        label_mask = labels.eq(label).to_numpy()
        group_frame = impute_with_median(numeric_feature_frame(frame.loc[label_mask, :], feature_columns))
        raw_group = group_frame.to_numpy(dtype=float)
        feature_rms = np.sqrt(np.mean(np.square(raw_group), axis=0))
        group = np.divide(
            raw_group,
            feature_rms,
            out=np.zeros_like(raw_group),
            where=np.isfinite(feature_rms) & (feature_rms > 0.0),
        )
        total_rows = int(group.shape[0])
        if total_rows == 0:
            continue

        seed_payload = f"{int(random_state)}\0{label}".encode("utf-8")
        label_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little")
        rng = np.random.default_rng(label_seed)

        label_sample_sizes = None if sample_sizes_by_label is None else sample_sizes_by_label.get(str(label))
        configured_sizes = label_sample_sizes if label_sample_sizes is not None else sample_sizes
        sizes = list(configured_sizes) if configured_sizes is not None else _default_sample_sizes(total_rows, min_samples)
        sizes = sorted({int(n) for n in sizes if 1 <= int(n) <= total_rows})
        if not sizes:
            continue
        if total_rows not in sizes:
            sizes.append(total_rows)

        pairwise_mean_gaps: dict[int, float] = {}
        pairwise_p90_gaps: dict[int, float] = {}
        repeats_used: dict[int, int] = {}

        for size in sizes:
            current_repeats = max(int(repeats), 1)
            pairwise_gaps = np.empty(current_repeats, dtype=float)
            for rep in range(current_repeats):
                paired_a = rng.choice(total_rows, size=size, replace=True)
                paired_b = rng.choice(total_rows, size=size, replace=True)
                mean_a = group[paired_a, :].mean(axis=0)
                mean_b = group[paired_b, :].mean(axis=0)
                pairwise_gaps[rep] = float(np.sqrt(np.mean(np.square(mean_a - mean_b))))
            pairwise_mean_gaps[size] = float(np.mean(pairwise_gaps))
            pairwise_p90_gaps[size] = float(np.percentile(pairwise_gaps, 90))
            repeats_used[size] = int(current_repeats)

        for size in sizes:
            curve_rows.append(
                {
                    label_col: label,
                    "total_rows": total_rows,
                    "sample_size": int(size),
                    "paired_bootstrap_relative_rms_gap_mean": pairwise_mean_gaps[size],
                    "paired_bootstrap_relative_rms_gap_p90": pairwise_p90_gaps[size],
                    "repeats_used": repeats_used[size],
                    "is_full_sample_size": bool(size >= total_rows),
                }
            )

        pairwise_values = [pairwise_p90_gaps[size] for size in sizes]
        stable_pairwise = _first_stable_size(pairwise_values, sizes, pairwise_threshold, consecutive_points)
        recommended = int(stable_pairwise) if np.isfinite(stable_pairwise) else np.nan
        status = "stable" if np.isfinite(stable_pairwise) else "not_stable_with_current_rows"

        summary_rows.append(
            {
                label_col: label,
                "total_rows": total_rows,
                "stable_sample_size_pairwise_bootstrap": stable_pairwise,
                "recommended_stable_sample_size": recommended,
                "stability_status": status,
                "pairwise_threshold": float(pairwise_threshold),
                "consecutive_points": int(max(consecutive_points, 1)),
                "min_sample_size_checked": int(min(sizes)),
                "max_sample_size_checked": int(max(sizes)),
                "final_paired_bootstrap_relative_rms_gap_p90": pairwise_p90_gaps[sizes[-1]],
            }
        )

    curve = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["stability_status", "recommended_stable_sample_size", label_col])
    return curve, summary


def plot_source_feature_stability_curves(
    curve: pd.DataFrame,
    output_path: Path,
    label_col: str = "source_label",
) -> None:
    """Plot label-local paired-bootstrap stability curves by source label."""

    if curve.empty or label_col not in curve.columns:
        return
    _configure_font()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=False)
    metrics = [
        ("paired_bootstrap_relative_rms_gap_mean", "双Bootstrap相对均值差异"),
        ("paired_bootstrap_relative_rms_gap_p90", "双Bootstrap相对均值差异P90"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for label, grp in curve.groupby(label_col):
            valid = grp.dropna(subset=[metric]).sort_values("sample_size")
            if valid.empty:
                continue
            ax.plot(valid["sample_size"], valid[metric], marker="o", linewidth=1.4, markersize=3.5, label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("样本量 n", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("类内相对RMS误差", fontsize=AXIS_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
    axes[0].legend(title="来源类别", fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
    fig.suptitle("各来源类别独立计算的特征均值稳定性", fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_kde_by_label(
    frame: pd.DataFrame,
    features: list[str],
    output_path: Path,
    col_wrap: int = 2,
    label_col: str = "source_label",
) -> None:
    """绘制六类KDE曲线，观察单个特征分布重叠程度。

    col_wrap 控制每行子图数量：如需 4列2行排布的8个特征，传 col_wrap=4。
    """

    if not features:
        return
    if label_col not in frame.columns:
        return
    _configure_font()
    x = impute_with_median(numeric_feature_frame(frame, features))
    plot_df = pd.concat([frame[[label_col]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    long_df = plot_df.melt(id_vars=label_col, var_name="feature", value_name="value")
    g = sns.FacetGrid(long_df, col="feature", col_wrap=col_wrap, hue=label_col, sharex=False, sharey=False, height=2.7)
    g.map_dataframe(sns.kdeplot, x="value", fill=False, common_norm=False, warn_singular=False)
    legend_title = "来源标签" if label_col == "source_label" else "比较标签"
    g.add_legend(title=legend_title, fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
    g.set_axis_labels("特征值", "密度", fontsize=AXIS_FONT_SIZE)
    g.set_titles("{col_name}", fontsize=TITLE_FONT_SIZE)
    g.fig.suptitle("六类特征KDE分布", y=1.02, fontsize=TITLE_FONT_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def plot_projection(projection: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    """绘制PCA/UMAP二维散点图。

    颜色按信号家族（signal_family: BK/FL/QJ）区分，同一家族的
    BK00/BK05 等使用相同颜色；点型按流速工况（flow_condition）
    区分，v0 为圆点、v0.5 为叉号；图例逐条标注完整来源标签
    （如 BK00/BK05/FL00/FL05/QJ00/QJ05），颜色表达家族、
    点型表达流速。
    """

    if projection.empty:
        return
    _configure_font()
    data = projection.copy()
    if "signal_family" not in data.columns:
        data["signal_family"] = data["source_label"].astype(str).str.extract(r"^([A-Z]+)")[0].fillna("OTHER")
    data["flow_condition"] = data.get("flow_condition", pd.Series("v0", index=data.index)).astype(str)

    families = sorted(f_ for f_ in data["signal_family"].dropna().unique())
    palette = {fam: color for fam, color in zip(families, sns.color_palette("tab10", n_colors=max(len(families), 3)))}
    markers = {"v0": "o", "v0.5": "X", "unknown": "o"}

    plt.figure(figsize=(7.5, 5.8))
    for (fam, flow), grp in data.groupby(["signal_family", "flow_condition"]):
        marker = markers.get(flow, "o")
        plt.scatter(
            grp[x_col],
            grp[y_col],
            color=palette.get(fam, "#333333"),
            marker=marker,
            s=14,
            alpha=0.72,
            linewidth=0,
        )
    handle_rows = []
    for (fam, flow), grp in data.groupby(["signal_family", "flow_condition"]):
        if "source_label" in grp.columns:
            label_parts = grp["source_label"].dropna().astype(str).unique()
            label = "/".join(sorted(label_parts)) if len(label_parts) else str(fam)
        else:
            label = str(fam)
        handle_rows.append((label, fam, flow))
    handle_rows.sort(key=lambda item: item[0])
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers.get(flow, "o"),
            color="w",
            markerfacecolor=palette.get(fam, "#333333"),
            markersize=6,
            label=label,
        )
        for label, fam, flow in handle_rows
    ]
    plt.legend(handles=handles, title="来源标签", fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
    plt.xlabel(x_col, fontsize=AXIS_FONT_SIZE)
    plt.ylabel(y_col, fontsize=AXIS_FONT_SIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
