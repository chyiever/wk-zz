"""六类特征分布分析：箱线图、KDE、PCA和可选UMAP。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_feature_kde_by_label(frame: pd.DataFrame, features: list[str], output_path: Path) -> None:
    """绘制六类KDE曲线，观察单个特征分布重叠程度。"""

    if not features:
        return
    _configure_font()
    x = impute_with_median(numeric_feature_frame(frame, features))
    plot_df = pd.concat([frame[["source_label"]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    long_df = plot_df.melt(id_vars="source_label", var_name="feature", value_name="value")
    g = sns.FacetGrid(long_df, col="feature", col_wrap=2, hue="source_label", sharex=False, sharey=False, height=2.7)
    g.map_dataframe(sns.kdeplot, x="value", fill=False, common_norm=False, warn_singular=False)
    g.add_legend(title="来源标签", fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
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
    区分，v0 为圆点、v0.5 为叉号；图例仅展示信号家族颜色，
    不标注流速工况。
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
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[fam], markersize=6, label=fam)
        for fam in families
    ]
    plt.legend(handles=handles, title="信号家族", fontsize=TICK_FONT_SIZE, title_fontsize=AXIS_FONT_SIZE)
    plt.xlabel(x_col, fontsize=AXIS_FONT_SIZE)
    plt.ylabel(y_col, fontsize=AXIS_FONT_SIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
