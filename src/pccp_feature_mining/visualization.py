"""特征挖掘结果可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .feature_schema import impute_with_median, numeric_feature_frame


def _configure_font() -> None:
    """优先使用Windows中文字体；缺失时仍可保存图片。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_top_feature_boxplots(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    max_features: int = 12,
) -> None:
    """绘制Top特征在六类标签下的箱线图。"""

    if not feature_columns:
        return
    _configure_font()
    cols = feature_columns[:max_features]
    x = impute_with_median(numeric_feature_frame(frame, cols))
    plot_df = pd.concat([frame[["source_label"]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    long_df = plot_df.melt(id_vars="source_label", var_name="feature", value_name="value")
    height = max(4.0, 2.0 * len(cols))
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
    )
    g.fig.set_size_inches(11.0, height)
    g.set_axis_labels("来源标签", "特征值")
    g.set_titles("{col_name}")
    g.fig.suptitle("Top特征六类分布箱线图", y=1.02)
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
