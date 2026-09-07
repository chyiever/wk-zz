"""特征挖掘结果可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .feature_schema import impute_with_median, numeric_feature_frame


AXIS_FONT_SIZE = 10
TICK_FONT_SIZE = 9
TITLE_FONT_SIZE = 12


def _configure_font() -> None:
    """统一设置字体、字号等视觉风格，保证各图观感一致。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelsize"] = AXIS_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
    plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
    plt.rcParams["legend.fontsize"] = TICK_FONT_SIZE


def plot_top_feature_boxplots(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    max_features: int = 12,
    log_scale: bool = False,
) -> None:
    """绘制Top特征在六类标签下的箱线图。

    log_scale=True 时对 y 轴使用对数刻度，适合特征值跨越多个数量级的情形
    （如能量比、方差类特征）；对数刻度下的箱线图基于对数变换后的分位数。
    """

    if not feature_columns:
        return
    _configure_font()
    cols = feature_columns[:max_features]
    x = impute_with_median(numeric_feature_frame(frame, cols))
    plot_df = pd.concat([frame[["source_label"]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    long_df = plot_df.melt(id_vars="source_label", var_name="feature", value_name="value")
    height = max(4.0, 2.0 * len(cols))
    if log_scale:
        long_df = long_df[long_df["value"] > 0].copy()
    g = sns.catplot(
        data=long_df,
        x="source_label",
        y="value",
        col="feature",
        col_wrap=2,
        kind="box",
        sharey=False,
        height=2.8,
        aspect=1.35,
        fliersize=1.2,
        log_scale=log_scale,
    )
    g.fig.set_size_inches(11.0, height)
    g.set_axis_labels("来源标签", "特征值", fontsize=AXIS_FONT_SIZE)
    g.set_titles("{col_name}", fontsize=TITLE_FONT_SIZE)
    g.fig.suptitle("Top特征六类分布箱线图", y=1.02, fontsize=TITLE_FONT_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)


def plot_final_ranking(final_ranking: pd.DataFrame, output_path: Path, top_n: int = 30) -> None:
    """绘制最终评分Top-N条形图。"""

    if final_ranking.empty:
        return
    _configure_font()
    top = final_ranking.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.28 * len(top))))
    plt.barh(top["feature"], top["final_score"], color="#2f6f8f")
    plt.xlabel("最终评分")
    plt.title("PCCP断丝候选特征最终排序")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
