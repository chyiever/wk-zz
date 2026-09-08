"""六类特征分布分析：箱线图、KDE、PCA和可选UMAP。"""

from __future__ import annotations

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


def estimate_source_feature_stability(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    label_col: str = "source_label",
    sample_sizes: list[int] | tuple[int, ...] | None = None,
    repeats: int = 100,
    min_samples: int = 5,
    adjacent_threshold: float = 0.02,
    pairwise_threshold: float = 0.05,
    reference_threshold: float | None = None,
    consecutive_points: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate when each source label has enough samples for stable feature means.

    Two independent estimates are reported:
    1. adjacent mean drift: relative L2 change between the averaged mean vector at n_i
       and n_(i-1);
    2. paired bootstrap agreement: relative L2 gap between two independent bootstrap
       mean vectors at the same sample size.
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
            "previous_sample_size",
            "adjacent_mean_rel_l2_change",
            "paired_bootstrap_rel_l2_gap_mean",
            "paired_bootstrap_rel_l2_gap_p90",
            "repeats_used",
            "is_full_sample_size",
        ]
        empty_summary_cols = [
            label_col,
            "total_rows",
            "stable_sample_size_adjacent_mean",
            "stable_sample_size_pairwise_bootstrap",
            "recommended_stable_sample_size",
            "stability_status",
            "adjacent_threshold",
            "pairwise_threshold",
        ]
        return pd.DataFrame(columns=empty_curve_cols), pd.DataFrame(columns=empty_summary_cols)

    rng = np.random.default_rng(random_state)
    x = impute_with_median(numeric_feature_frame(frame, feature_columns))
    scaled = pd.DataFrame(StandardScaler().fit_transform(x), columns=list(feature_columns), index=frame.index)
    labels = frame[label_col].astype(str)
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    eps = 1e-12

    for label in sorted(labels.dropna().unique()):
        idx = labels[labels.eq(label)].index
        group = scaled.loc[idx, :].to_numpy(dtype=float)
        total_rows = int(group.shape[0])
        if total_rows == 0:
            continue

        sizes = list(sample_sizes) if sample_sizes is not None else _default_sample_sizes(total_rows, min_samples)
        sizes = sorted({int(n) for n in sizes if 1 <= int(n) <= total_rows})
        if not sizes:
            continue
        if total_rows not in sizes:
            sizes.append(total_rows)

        mean_vectors: dict[int, np.ndarray] = {}
        pairwise_mean_gaps: dict[int, float] = {}
        pairwise_p90_gaps: dict[int, float] = {}
        repeats_used: dict[int, int] = {}

        for size in sizes:
            current_repeats = max(int(repeats), 1)
            sample_means = np.empty((current_repeats, group.shape[1]), dtype=float)
            pairwise_gaps = np.empty(current_repeats, dtype=float)
            for rep in range(current_repeats):
                replace_single = size >= total_rows
                picked = rng.choice(total_rows, size=size, replace=replace_single)
                sample_means[rep, :] = group[picked, :].mean(axis=0)

                paired_a = rng.choice(total_rows, size=size, replace=True)
                paired_b = rng.choice(total_rows, size=size, replace=True)
                mean_a = group[paired_a, :].mean(axis=0)
                mean_b = group[paired_b, :].mean(axis=0)
                scale = 0.5 * (float(np.linalg.norm(mean_a)) + float(np.linalg.norm(mean_b))) + eps
                pairwise_gaps[rep] = float(np.linalg.norm(mean_a - mean_b) / scale)
            mean_vectors[size] = sample_means.mean(axis=0)
            pairwise_mean_gaps[size] = float(np.mean(pairwise_gaps))
            pairwise_p90_gaps[size] = float(np.percentile(pairwise_gaps, 90))
            repeats_used[size] = int(current_repeats)

        adjacent_changes: list[float] = []
        previous_size: int | None = None
        for size in sizes:
            if previous_size is None:
                adjacent_change = np.nan
            else:
                previous = mean_vectors[previous_size]
                current = mean_vectors[size]
                adjacent_change = float(np.linalg.norm(current - previous) / (np.linalg.norm(previous) + eps))
            adjacent_changes.append(adjacent_change)
            curve_rows.append(
                {
                    label_col: label,
                    "total_rows": total_rows,
                    "sample_size": int(size),
                    "previous_sample_size": previous_size,
                    "adjacent_mean_rel_l2_change": adjacent_change,
                    "paired_bootstrap_rel_l2_gap_mean": pairwise_mean_gaps[size],
                    "paired_bootstrap_rel_l2_gap_p90": pairwise_p90_gaps[size],
                    "repeats_used": repeats_used[size],
                    "is_full_sample_size": bool(size >= total_rows),
                }
            )
            previous_size = int(size)

        pairwise_values = [pairwise_p90_gaps[size] for size in sizes]
        stable_adjacent = _first_stable_size(adjacent_changes, sizes, adjacent_threshold, consecutive_points)
        stable_pairwise = _first_stable_size(pairwise_values, sizes, pairwise_threshold, consecutive_points)
        stable_candidates = [v for v in [stable_adjacent, stable_pairwise] if np.isfinite(v)]
        recommended = int(max(stable_candidates)) if stable_candidates else np.nan
        if np.isfinite(stable_adjacent) and np.isfinite(stable_pairwise):
            status = "two_methods_stable"
        elif stable_candidates:
            status = "one_method_stable"
        else:
            status = "not_stable_with_current_rows"

        summary_rows.append(
            {
                label_col: label,
                "total_rows": total_rows,
                "stable_sample_size_adjacent_mean": stable_adjacent,
                "stable_sample_size_pairwise_bootstrap": stable_pairwise,
                "recommended_stable_sample_size": recommended,
                "stability_status": status,
                "adjacent_threshold": float(adjacent_threshold),
                "pairwise_threshold": float(pairwise_threshold),
                "consecutive_points": int(max(consecutive_points, 1)),
                "min_sample_size_checked": int(min(sizes)),
                "max_sample_size_checked": int(max(sizes)),
                "final_adjacent_mean_rel_l2_change": adjacent_changes[-1],
                "final_paired_bootstrap_rel_l2_gap_p90": pairwise_p90_gaps[sizes[-1]],
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
    """Plot adjacent-mean drift and paired-bootstrap agreement curves by source label."""

    if curve.empty or label_col not in curve.columns:
        return
    _configure_font()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=False)
    metrics = [
        ("adjacent_mean_rel_l2_change", "相邻均值向量相对变化"),
        ("paired_bootstrap_rel_l2_gap_p90", "双Bootstrap均值差异P90"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for label, grp in curve.groupby(label_col):
            valid = grp.dropna(subset=[metric]).sort_values("sample_size")
            if valid.empty:
                continue
            ax.plot(valid["sample_size"], valid[metric], marker="o", linewidth=1.4, markersize=3.5, label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("样本量 n", fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel("相对L2误差", fontsize=AXIS_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
    axes[0].legend(title="来源类别", fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
    fig.suptitle("各来源类别特征均值稳定性估计", fontsize=TITLE_FONT_SIZE)
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
