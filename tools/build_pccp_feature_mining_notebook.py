"""生成 PCCP 断丝特征挖掘 notebook。

使用脚本生成 notebook 可以避免手工编辑大段 JSON，并保证中文以 UTF-8 写入。
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb")


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def build_notebook() -> dict[str, object]:
    cells: list[dict[str, object]] = []

    cells.append(
        md(
            """
# PCCP断丝信号特征挖掘

本 notebook 对应开发方案中的 Notebook01-Notebook08，并新增 Notebook09 分类测试。输入是六类已经计算好的特征向量，不重新计算原始波形特征；目标是从数百个候选特征中筛出能够稳定表征 PCCP 断丝响应的少量核心特征。全流程核心代码位于 `src/pccp_feature_mining`，notebook 负责组织实验、展示结果、解释指标含义并给出结论。
"""
        )
    )

    cells.append(
        md(
            """
## Notebook00 环境初始化与显示工具

本节只做三件事：定位项目根目录、导入显示工具、定义统一的表格/图片展示函数。后续每个表都用 `show_table()` 输出“表号 + 标题 + 表格”，每张结果图都用 `show_image()` 在 notebook 内直接显示。
"""
        )
    )
    cells.append(
        code(
            """
from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import Image, Markdown, display


def find_project_root(start: Path) -> Path:
    \"\"\"向上查找.git，保证从notebooks目录启动时也能定位项目根目录。\"\"\"
    p = start.resolve()
    for candidate in [p] + list(p.parents):
        if (candidate / '.git').exists():
            return candidate
    return p


PROJECT_ROOT = find_project_root(Path.cwd())
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pccp_feature_mining.feature_schema import add_feature_meaning_columns, describe_feature_list

RUN_DIR = PROJECT_ROOT / 'outputs/PCCP_feature_mining_notebook/manual_run'
RUN_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NO = 0
FIGURE_NO = 0


def show_table(title: str, frame: pd.DataFrame, rows: int | None = None) -> pd.DataFrame:
    \"\"\"显示带编号标题的表格，并自动补充特征中文含义列。\"\"\"
    global TABLE_NO
    TABLE_NO += 1
    out = add_feature_meaning_columns(frame)
    if rows is not None:
        out = out.head(rows)
    display(Markdown(f'**表 {TABLE_NO} {title}**'))
    display(out)
    return out


def show_image(title: str, path: Path) -> None:
    \"\"\"显示带编号标题的结果图。\"\"\"
    global FIGURE_NO
    FIGURE_NO += 1
    display(Markdown(f'**图 {FIGURE_NO} {title}**'))
    display(Image(filename=str(path)))


print('项目根目录:', PROJECT_ROOT)
print('默认输出目录:', RUN_DIR)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook01 数据读取与质量检查

本节实现“数据读取与质量检查”。算法上先递归读取六类输入目录下的 `features*.csv`，使用显式目录标签覆盖 CSV 内部可能不一致的标签，再统一生成 `source_label`、`target_label`、`signal_family`、`flow_condition`、`event_id` 和 `sampling_group_id`。

本版新增行级有效特征检测：旧逻辑只检查 CSV 表头列数和是否为空表，因此类似 `features_20260905_173711_part_0001.csv` 这种“661列、17066行，但 6568 行没有任何有效频带特征值”的文件不会被跳过。现在读取时会计算每行非空有效特征数，并剔除低于 `CONFIG.min_valid_feature_values_per_row` 的行；这些行不会进入 Notebook02-Notebook09 的分布、判别、冗余、组合、稳定性、跨流速或分类分析。

可调参数：`min_feature_csv_columns` 控制文件级表头列数门槛；`min_valid_feature_values_per_row` 控制行级有效特征数门槛。若后续特征总数变化，建议设置为“预期有效特征数的 10%-20%”或至少 50，本 notebook 当前使用 100。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.config import FeatureInput, MiningConfig
from pccp_feature_mining.data_loader import load_feature_dataset
from pccp_feature_mining.quality_control import build_dataset_summary, build_feature_quality, feature_list_frame
from pccp_feature_mining.report_generator import write_csv, write_markdown_summary

FEATURE_INPUTS = [
    FeatureInput('FL00', PROJECT_ROOT / r'outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0-flow_features'),
    FeatureInput('FL05', PROJECT_ROOT / r'outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0.5-flow_features'),
    FeatureInput('BK00', PROJECT_ROOT / r'outputs/DATA09_multi_label_single_event_features/v0-bk_features_BK00'),
    FeatureInput('QJ00', PROJECT_ROOT / r'outputs/DATA09_multi_label_single_event_features/v0-qj_features_QJ00'),
    FeatureInput('BK05', PROJECT_ROOT / r'outputs/DATA09_multi_label_single_event_features/v05-bk_features_BK05'),
    FeatureInput('QJ05', PROJECT_ROOT / r'outputs/DATA09_multi_label_single_event_features/v05-qj_features_QJ05'),
]

CONFIG = MiningConfig(
    output_dir=PROJECT_ROOT / 'outputs/PCCP_feature_mining_notebook',
    feature_inputs=tuple(FEATURE_INPUTS),
    min_feature_csv_columns=100,
    min_valid_feature_values_per_row=100,
    random_state=42,
    correlation_threshold=0.92,
    bootstrap_rounds=200,
    bootstrap_top_ks=(10, 20, 30),
    mrmr_top_n=80,
    max_correlation_rows=6000,
    distribution_top_n=12,
    projection_max_rows=8000,
    combination_feature_counts=(5, 10, 20, 30, 50),
    sfs_candidate_count=30,
    sfs_max_selected=15,
    classification_feature_counts=(5, 10, 20, 30, 50),
    classification_max_train_rows_per_class=3000,
    max_rows_per_label=None,  # 快速调试可设为1000；正式分析保持None。
)

config_table = pd.DataFrame([
    {'参数': 'min_feature_csv_columns', '当前值': CONFIG.min_feature_csv_columns, '含义': '文件级最低列数；列数低于该值的CSV直接跳过'},
    {'参数': 'min_valid_feature_values_per_row', '当前值': CONFIG.min_valid_feature_values_per_row, '含义': '行级最低有效特征数；低于该值的样本行不进入后续分析'},
    {'参数': 'bootstrap_rounds', '当前值': CONFIG.bootstrap_rounds, '含义': 'Bootstrap重复抽样次数'},
    {'参数': 'bootstrap_top_ks', '当前值': str(CONFIG.bootstrap_top_ks), '含义': '统计Top-K出现频率的K值'},
])
show_table('Notebook01 关键质量控制参数', config_table)
"""
        )
    )
    cells.append(
        code(
            """
dataset = load_feature_dataset(
    feature_inputs=CONFIG.feature_inputs,
    min_feature_csv_columns=CONFIG.min_feature_csv_columns,
    min_valid_feature_values_per_row=CONFIG.min_valid_feature_values_per_row,
    max_rows_per_label=CONFIG.max_rows_per_label,
    random_state=CONFIG.random_state,
)
df = dataset.frame
feature_cols = list(dataset.feature_columns)

dataset_summary = build_dataset_summary(df)
feature_quality = build_feature_quality(df, feature_cols)
feature_list = feature_list_frame(feature_cols)
load_issues = dataset.load_report[
    (dataset.load_report['status'] != 'ok') | (dataset.load_report['dropped_low_valid_feature_rows'] > 0)
].copy()

write_csv(dataset.load_report, RUN_DIR / 'load_report.csv')
write_csv(dataset_summary, RUN_DIR / 'dataset_summary.csv')
write_csv(feature_list, RUN_DIR / 'feature_list.csv')
write_csv(feature_quality, RUN_DIR / 'quality_report.csv')

print(f'样本行数: {len(df):,}, 特征数: {len(feature_cols):,}')
show_table('六类样本读取汇总', dataset_summary)
show_table('有效特征清单样例', feature_list, rows=30)
show_table('特征质量报告Top20', feature_quality, rows=20)
show_table('读取异常与低有效特征行审计', load_issues)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook02 六类特征分布分析

“六类特征分布”不是指只有六个特征，而是指在六个来源标签 `BK00/BK05/FL00/FL05/QJ00/QJ05` 上比较候选特征的分布。CSV 中有数百个频带特征，本节先对全部可用特征计算 `six_class_f_ratio`，再选出最能体现六类差异的前 `CONFIG.distribution_top_n` 个特征画图展示。

计算口径：`six_class_f_ratio = 类间均值方差 / 类内方差均值`。该值越大，说明该特征在六类之间的均值差异越明显，更适合用于分布展示。箱线图观察中位数和四分位距；KDE 观察分布重叠程度；PCA/UMAP 用二维投影观察整体特征空间的类间结构。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.distribution_analysis import (
    estimate_source_feature_stability,
    plot_source_feature_stability_curves,
    select_distribution_features,
    run_pca_projection,
    run_umap_projection,
    plot_feature_kde_by_label,
    plot_projection,
    select_bk_0_30ms_samples,
)
from pccp_feature_mining.visualization import plot_final_ranking, plot_top_feature_boxplots

six_class_distribution_features = select_distribution_features(df, feature_cols, top_n=CONFIG.distribution_top_n)
pca_projection, pca_explained = run_pca_projection(df, feature_cols, max_rows=CONFIG.projection_max_rows, random_state=CONFIG.random_state)
umap_projection = run_umap_projection(df, feature_cols, max_rows=min(CONFIG.projection_max_rows, 5000), random_state=CONFIG.random_state)

write_csv(add_feature_meaning_columns(six_class_distribution_features), RUN_DIR / 'six_class_distribution_features.csv')
write_csv(pca_projection, RUN_DIR / 'six_class_pca_projection.csv')
write_csv(pca_explained, RUN_DIR / 'six_class_pca_explained_variance.csv')
write_csv(umap_projection, RUN_DIR / 'six_class_umap_projection.csv')

boxplot_path = RUN_DIR / 'plots/six_class_feature_boxplots.png'
kde_path = RUN_DIR / 'plots/six_class_feature_kde.png'
pca_path = RUN_DIR / 'plots/six_class_pca.png'
umap_path = RUN_DIR / 'plots/six_class_umap.png'

plot_top_feature_boxplots(df, six_class_distribution_features['feature'].head(12).tolist(), boxplot_path, max_features=12)
plot_feature_kde_by_label(df, six_class_distribution_features['feature'].head(8).tolist(), kde_path)
plot_projection(pca_projection, 'PC1', 'PC2', pca_path, '六类特征PCA投影')
if not umap_projection.empty:
    plot_projection(umap_projection, 'UMAP1', 'UMAP2', umap_path, '六类特征UMAP投影')

show_table('六类分布差异最大的展示特征', six_class_distribution_features)
show_table('PCA解释方差', pca_explained)
show_image('六类特征箱线图', boxplot_path)
show_image('六类特征KDE分布图', kde_path)
show_image('六类特征PCA投影图', pca_path)
if not umap_projection.empty:
    show_image('六类特征UMAP投影图', umap_path)
"""
        )
    )

    cells.append(
        md(
            """
### 2.7.1 全部派生窗口的来源类别特征均值稳定性

对 `BK00/BK05/FL00/FL05/QJ00/QJ05` 的全部有效特征行估计均值稳定性。六类分别进行类内中位数插补、类内标准差归一化和独立 Bootstrap，任一类别的曲线不受其他类别是否加入影响。误差为两组 Bootstrap 均值向量的类内标准化 RMS 差异；该无量纲数值可比较各类的相对抽样稳定性，但不表示原始信号幅值差异。BK 的每个物理事件会派生六个相互重叠的窗口，因此这里的 BK 行数不能解释为独立事件数；独立事件口径见 2.7.2。
"""
        )
    )
    cells.append(
        code(
            """
sample_stability_curve, sample_stability_summary = estimate_source_feature_stability(
    df,
    feature_cols,
    repeats=CONFIG.sample_stability_repeats,
    min_samples=CONFIG.sample_stability_min_samples,
    pairwise_threshold=CONFIG.sample_stability_pairwise_threshold,
    consecutive_points=CONFIG.sample_stability_consecutive_points,
    random_state=CONFIG.random_state,
)
write_csv(sample_stability_curve, RUN_DIR / 'sample_stability_curve.csv')
write_csv(sample_stability_summary, RUN_DIR / 'sample_stability_summary.csv')
sample_stability_plot_path = RUN_DIR / 'plots/sample_stability_curves.png'
plot_source_feature_stability_curves(sample_stability_curve, sample_stability_plot_path)

sample_stability_parameter_table = pd.DataFrame([
    {'参数': 'sample_stability_repeats', '当前值': CONFIG.sample_stability_repeats, '含义': '每个样本量重复抽样次数'},
    {'参数': 'sample_stability_min_samples', '当前值': CONFIG.sample_stability_min_samples, '含义': '自动样本量网格的最小起点'},
    {'参数': 'sample_stability_pairwise_threshold', '当前值': CONFIG.sample_stability_pairwise_threshold, '含义': '双Bootstrap类内标准化RMS差异P90稳定阈值'},
    {'参数': 'sample_stability_consecutive_points', '当前值': CONFIG.sample_stability_consecutive_points, '含义': '连续满足阈值的样本量点数'},
])
show_table('2.7.1 全部派生窗口样本量稳定性估计参数', sample_stability_parameter_table)
show_table('2.7.1 各来源类别建议稳定样本量', sample_stability_summary)
show_table('2.7.1 全部派生窗口样本量稳定性曲线明细', sample_stability_curve, rows=60)
show_image('2.7.1 各来源类别特征均值稳定性曲线', sample_stability_plot_path)
"""
        )
    )

    cells.append(
        md(
            """
### 2.7.2 BK 断丝事件的 0-30 ms 单窗口样本稳定性

一个物理 BK 事件会派生六个重叠窗口。本小节只保留 `window_mode='0_30'` 且实际边界为 0-30 ms 的窗口，保证每个 `event_id` 最多贡献一个样本，再分别测试 `BK00` 和 `BK05` 的均值稳定性。图中同时加入 FL00、FL05、QJ00、QJ05；六类均独立进行插补、尺度估计和 Bootstrap，加入其他类别不会改变任一类别的误差曲线。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.distribution_analysis import (
    estimate_source_feature_stability,
    plot_source_feature_stability_curves,
    select_bk_0_30ms_samples,
)

bk_0_30ms_df, bk_0_30ms_selection_audit = select_bk_0_30ms_samples(df)
bk_0_30ms_sample_sizes_by_label = {
    str(label): list(range(min(max(int(CONFIG.sample_stability_min_samples), 1), len(group)), len(group) + 1))
    for label, group in bk_0_30ms_df.groupby('source_label', sort=True)
}
non_bk_df = df.loc[~df['source_label'].astype(str).str.startswith('BK')]
bk_0_30ms_comparison_df = pd.concat([non_bk_df, bk_0_30ms_df], axis=0)
bk_0_30ms_comparison_curve, bk_0_30ms_comparison_summary = estimate_source_feature_stability(
    bk_0_30ms_comparison_df,
    feature_cols,
    sample_sizes_by_label=bk_0_30ms_sample_sizes_by_label,
    repeats=CONFIG.sample_stability_repeats,
    min_samples=CONFIG.sample_stability_min_samples,
    pairwise_threshold=CONFIG.sample_stability_pairwise_threshold,
    consecutive_points=CONFIG.sample_stability_consecutive_points,
    random_state=CONFIG.random_state,
)
bk_0_30ms_stability_curve = bk_0_30ms_comparison_curve.loc[
    bk_0_30ms_comparison_curve['source_label'].astype(str).str.startswith('BK')
].copy()
bk_0_30ms_stability_summary = bk_0_30ms_comparison_summary.loc[
    bk_0_30ms_comparison_summary['source_label'].astype(str).str.startswith('BK')
].copy()
write_csv(bk_0_30ms_selection_audit, RUN_DIR / 'bk_0_30ms_selection_audit.csv')
write_csv(bk_0_30ms_stability_curve, RUN_DIR / 'bk_0_30ms_sample_stability_curve.csv')
write_csv(bk_0_30ms_stability_summary, RUN_DIR / 'bk_0_30ms_sample_stability_summary.csv')
write_csv(bk_0_30ms_comparison_curve, RUN_DIR / 'bk_0_30ms_with_nonbk_sample_stability_curve.csv')
write_csv(bk_0_30ms_comparison_summary, RUN_DIR / 'bk_0_30ms_with_nonbk_sample_stability_summary.csv')
bk_0_30ms_stability_plot_path = RUN_DIR / 'plots/bk_0_30ms_with_nonbk_sample_stability_curves.png'
plot_source_feature_stability_curves(bk_0_30ms_comparison_curve, bk_0_30ms_stability_plot_path)

show_table('2.7.2 BK 0-30 ms窗口筛选审计', bk_0_30ms_selection_audit)
show_table('2.7.2 BK 0-30 ms单窗口建议稳定样本量', bk_0_30ms_stability_summary)
show_table('2.7.2 BK 0-30 ms单窗口样本量稳定性曲线明细', bk_0_30ms_stability_curve, rows=60)
show_image('2.7.2 BK 0-30 ms单窗口校正后的六类特征均值稳定性曲线', bk_0_30ms_stability_plot_path)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook03 单特征判别能力分析

本节对每个单特征计算二分类判别能力。AUC 衡量阈值从低到高扫描时的排序分离能力，`auc_abs` 越接近 1 越好，0.5 表示随机；`auc_lift = auc_abs - 0.5`。Wasserstein distance 衡量两类分布整体位移，归一化后越大越分离；Cliff delta 衡量两类样本两两比较的优势概率；Mutual Information 衡量非线性依赖。综合分 `discrimination_score` 越大，说明单特征越值得进入候选集。

本章按五个二级比较任务输出结果：3.1 `BK00 vs NONBK00`，3.2 `BK00 vs QJ00`，3.3 `BK05 vs QJ05`，3.4 `BK05 vs NONBK05`，3.5 `BK vs NONBK`。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.feature_discrimination import evaluate_feature_discrimination

feature_discrimination = evaluate_feature_discrimination(df, feature_cols, random_state=CONFIG.random_state)
write_csv(add_feature_meaning_columns(feature_discrimination), RUN_DIR / 'feature_discrimination.csv')

metric_explain = pd.DataFrame([
    {'列名': 'auc_abs', '含义': '无方向AUC，越接近1越好，0.5表示随机', '公式': 'max(AUC, 1-AUC)'},
    {'列名': 'auc_lift', '含义': 'AUC相对随机分类的提升，越大越好', '公式': 'auc_abs - 0.5'},
    {'列名': 'wasserstein_norm', '含义': '按IQR/标准差归一化的分布距离，越大越好', '公式': 'W(pos, neg) / scale'},
    {'列名': 'abs_cliff_delta', '含义': '两类两两比较优势强度，0到1，越大越好', '公式': '|P(pos>neg)-P(pos<neg)|'},
    {'列名': 'mutual_info', '含义': '特征与类别的信息量，越大越好', '公式': 'MI(feature, label)'},
    {'列名': 'discrimination_score', '含义': '单特征综合判别分，越大越好', '公式': '0.45*auc_lift + 0.25*W + 0.2*Cliff + 0.1*MI_norm'},
])
show_table('单特征判别指标含义', metric_explain)
"""
        )
    )

    comparison_sections = [
        ("3.1 BK00 vs NONBK00", "BK00_NONBK00", "BK00 与同流速非断丝样本 `FL00+QJ00` 比较，回答零流速下断丝是否能从背景和敲击中分离。"),
        ("3.2 BK00 vs QJ00", "BK00_QJ00", "只比较零流速断丝与零流速敲击，重点检查抗敲击干扰的断丝特异性。"),
        ("3.3 BK05 vs QJ05", "BK05_QJ05", "只比较 v0.5 流速断丝与 v0.5 流速敲击，重点检查有流速背景下的抗敲击能力。"),
        ("3.4 BK05 vs NONBK05", "BK05_NONBK05", "BK05 与同流速非断丝样本 `FL05+QJ05` 比较，回答 v0.5 流速下断丝是否可分。"),
        ("3.5 BK vs NONBK", "BK_NONBK", "合并两种流速后比较全部断丝与全部非断丝，是最终特征排序的主任务。"),
    ]
    for heading, key, desc in comparison_sections:
        cells.append(md(f"### {heading}\n\n{desc}"))
        cells.append(
            code(
                f"""
cols = ['comparison', 'feature', 'n_positive', 'n_negative', 'auc_abs', 'wasserstein_norm', 'abs_cliff_delta', 'mutual_info', 'discrimination_score']
show_table('{heading} 单特征Top20', feature_discrimination[feature_discrimination['comparison'].eq('{key}')][cols], rows=20)
"""
            )
        )

    cells.append(
        md(
            """
## Notebook04 特征冗余分析

高相关特征携带的信息高度重复，直接一起进入模型会增加解释成本，也可能放大同一物理量的权重。本章分三部分：4.1 使用 Pearson 相关分析线性冗余；4.2 使用 Spearman 相关分析单调冗余；4.3 对比两种方法，形成“哪些2个或3个/更多特征极高相关、建议保留哪1个”的冗余推荐表。

阈值由 `CONFIG.correlation_threshold` 控制，当前为 0.92。若希望更严格去重可提高到 0.95 或 0.98；若希望更早发现潜在重复表达，可降到 0.85-0.90。
"""
        )
    )
    cells.append(md("### 4.1 使用 Pearson 相关\n\nPearson 相关系数度量线性关系：$r=\\frac{cov(x,y)}{\\sigma_x\\sigma_y}$。它适合发现成比例变化或近似线性重复的特征。"))
    cells.append(
        code(
            """
from pccp_feature_mining.feature_selection import build_relevance_series, run_mrmr_ranking, build_final_ranking
from pccp_feature_mining.feature_redundancy import (
    build_correlation_clusters,
    build_redundancy_recommendations,
    compare_correlation_methods,
    compute_correlation_matrix,
    high_correlation_pairs,
)

relevance = build_relevance_series(feature_discrimination, comparison='BK_NONBK')
pearson_corr = compute_correlation_matrix(
    df,
    feature_cols,
    method='pearson',
    max_rows=CONFIG.max_correlation_rows,
    random_state=CONFIG.random_state,
)
pearson_corr.to_csv(RUN_DIR / 'pearson_correlation_matrix.csv', encoding='utf-8-sig')
pearson_high_pairs = high_correlation_pairs(pearson_corr, threshold=CONFIG.correlation_threshold)
write_csv(add_feature_meaning_columns(pearson_high_pairs), RUN_DIR / 'pearson_high_correlation_pairs.csv')
show_table('Pearson高相关特征对Top30', pearson_high_pairs, rows=30)
"""
        )
    )
    cells.append(md("### 4.2 使用 Spearman 相关\n\nSpearman 相关系数先把数值转换为秩，再计算秩的 Pearson 相关，偏重单调关系。它对尺度变化和非线性单调变换更稳健。本流程后续 mRMR 和最终冗余惩罚使用 Spearman 相关矩阵。"))
    cells.append(
        code(
            """
spearman_corr = compute_correlation_matrix(
    df,
    feature_cols,
    method='spearman',
    max_rows=CONFIG.max_correlation_rows,
    random_state=CONFIG.random_state,
)
spearman_corr.to_csv(RUN_DIR / 'spearman_correlation_matrix.csv', encoding='utf-8-sig')
spearman_high_pairs = high_correlation_pairs(spearman_corr, threshold=CONFIG.correlation_threshold)
corr = spearman_corr
high_pairs = spearman_high_pairs
correlation_cluster = build_correlation_clusters(spearman_corr, relevance, threshold=CONFIG.correlation_threshold)
write_csv(add_feature_meaning_columns(spearman_high_pairs), RUN_DIR / 'spearman_high_correlation_pairs.csv')
write_csv(add_feature_meaning_columns(high_pairs), RUN_DIR / 'high_correlation_pairs.csv')
write_csv(add_feature_meaning_columns(correlation_cluster), RUN_DIR / 'correlation_cluster.csv')
show_table('Spearman高相关特征对Top30', spearman_high_pairs, rows=30)
show_table('Spearman相关簇Top30', correlation_cluster, rows=30)
"""
        )
    )
    cells.append(md("### 4.3 Pearson 与 Spearman 结果对比\n\n本节把两种方法命中的高相关特征对合并，再按连通分量形成冗余组。组内推荐保留 `BK vs NONBK` 判别分最高的1个特征，其余特征作为冗余删除候选；如果一个组包含3个或更多特征，说明这些特征共同表达同一类信息的风险更高。"))
    cells.append(
        code(
            """
correlation_method_comparison = compare_correlation_methods(pearson_high_pairs, spearman_high_pairs)
redundancy_recommendations = build_redundancy_recommendations(
    pearson_corr,
    spearman_corr,
    relevance=relevance,
    threshold=CONFIG.correlation_threshold,
)
write_csv(add_feature_meaning_columns(correlation_method_comparison), RUN_DIR / 'correlation_method_comparison.csv')
write_csv(add_feature_meaning_columns(redundancy_recommendations), RUN_DIR / 'redundancy_recommendations.csv')
show_table('Pearson与Spearman高相关特征对对比Top40', correlation_method_comparison, rows=40)
show_table('高相关冗余组与保留建议Top40', redundancy_recommendations, rows=40)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook05 多特征组合搜索

本章分成四个小节，用不同方法构造特征组合并对比。组合评价统一用 LDA 交叉验证：对训练折拟合标准化 + LDA，在验证折计算无方向 AUC 和 LDA 投影分离度。`cv_auc_abs` 越大越好；`lda_separation = |mean(score_pos)-mean(score_neg)| / pooled_std`，越大表示投影后两类分得越开。
"""
        )
    )
    cells.append(md("### 5.1 mRMR前缀组合\n\nmRMR 按“高相关性、低冗余”选特征。每一步最大化 `mrmr_score = relevance_score - redundancy_weight * mean_abs_corr_to_selected`，再测试前 K 个特征的组合效果。"))
    cells.append(
        code(
            """
from pccp_feature_mining.combination_search import evaluate_mrmr_prefixes, relief_like_ranking, sequential_forward_search

mrmr_rank = run_mrmr_ranking(
    relevance=relevance,
    corr=corr,
    top_n=CONFIG.mrmr_top_n,
    redundancy_weight=CONFIG.mrmr_redundancy_weight,
)
mrmr_prefix = evaluate_mrmr_prefixes(
    df,
    mrmr_rank,
    feature_cols,
    counts=CONFIG.combination_feature_counts,
    random_state=CONFIG.random_state,
)
write_csv(add_feature_meaning_columns(mrmr_rank), RUN_DIR / 'mrmr_rank.csv')
write_csv(add_feature_meaning_columns(mrmr_prefix), RUN_DIR / 'feature_combination_mrmr_prefix.csv')
show_table('mRMR特征排序Top30', mrmr_rank, rows=30)
show_table('mRMR前缀组合评价', mrmr_prefix)
"""
        )
    )
    cells.append(md("### 5.2 近似 ReliefF 排序\n\nReliefF 思路是在局部邻域中比较最近同类与最近异类。若某个特征让同类更接近、异类更远离，该特征得分更高。当前实现为工程近似版，用于提供与 mRMR 不同的候选视角。"))
    cells.append(
        code(
            """
relieff_rank = relief_like_ranking(df, feature_cols, random_state=CONFIG.random_state)
write_csv(add_feature_meaning_columns(relieff_rank), RUN_DIR / 'relieff_rank.csv')
show_table('近似ReliefF特征排序Top30', relieff_rank, rows=30)
"""
        )
    )
    cells.append(md("### 5.3 顺序前向选择 SFS\n\nSFS 从 mRMR 和 ReliefF 的前若干候选中搜索。每轮尝试把一个候选特征加入当前组合，选择让 LDA 交叉验证 AUC 最大的特征，直到达到 `CONFIG.sfs_max_selected` 或没有候选。"))
    cells.append(
        code(
            """
sfs_candidates = list(dict.fromkeys(mrmr_rank['feature'].head(CONFIG.sfs_candidate_count).tolist() + relieff_rank['feature'].head(CONFIG.sfs_candidate_count).tolist()))
sfs_selection_path = sequential_forward_search(
    df,
    sfs_candidates,
    feature_cols,
    max_selected=CONFIG.sfs_max_selected,
    random_state=CONFIG.random_state,
)
write_csv(add_feature_meaning_columns(sfs_selection_path), RUN_DIR / 'sfs_selection_path.csv')
show_table('SFS逐步选择路径', sfs_selection_path)
"""
        )
    )
    cells.append(md("### 5.4 多方法结果对比\n\n本节把 mRMR 前缀组合与 SFS 路径合并成统一对比表。优先关注 `cv_auc_abs` 高、`lda_separation` 高且特征数量较少的组合。"))
    cells.append(
        code(
            """
if sfs_selection_path.empty:
    feature_combination_search = mrmr_prefix.copy()
else:
    feature_combination_search = pd.concat([
        mrmr_prefix,
        sfs_selection_path.rename(columns={'step': 'sfs_step'})[['method', 'comparison', 'feature_count', 'cv_auc_abs', 'lda_separation', 'selected_features']]
    ], ignore_index=True, sort=False)

write_csv(add_feature_meaning_columns(feature_combination_search), RUN_DIR / 'feature_combination_search.csv')
show_table('多特征组合搜索方法对比Top20', feature_combination_search.sort_values('cv_auc_abs', ascending=False), rows=20)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook06 特征稳定性分析

本节实现 Bootstrap 稳定性分析。原理是重复构造多个平衡子样本，在每个子样本上重新计算单特征 AUC 排名；如果某个特征在很多轮抽样中都进入 Top10/Top20/Top30，说明它不是由某次样本偶然性产生的候选。

当前参数：抽样 `CONFIG.bootstrap_rounds=200` 次；每轮保留全部 BK 样本，BK 数量为 `df[target_label == 'BK']` 的行数；每轮从 Other 中抽取同样数量的样本，即每轮样本量约为 `2 * BK样本数`；统计 `CONFIG.bootstrap_top_ks=(10, 20, 30)` 的出现频率。Other 抽样按 `sampling_group_id` 执行：QJ 尽量保持事件完整，FL 窗口按独立窗口处理。

参数调整方法：快速调试可把 `bootstrap_rounds` 设为 20-50；正式报告建议 200-1000。若 BK 样本继续增加，Other 每轮抽样仍建议与 BK 等量；若希望更保守，可把 `top_ks` 扩展为 `(10, 20, 30, 50)`。指标解释：`mean_rank` 越小越好，`rank_std` 越小越稳定，`top10_frequency/top20_frequency/top30_frequency` 越高越稳定。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.bootstrap_stability import run_bootstrap_stability

bootstrap_parameter_table = pd.DataFrame([
    {'参数': 'rounds', '当前值': CONFIG.bootstrap_rounds, '计算/含义': '重复抽样轮数'},
    {'参数': 'BK rows per round', '当前值': int(df['target_label'].eq('BK').sum()), '计算/含义': '每轮保留全部BK样本'},
    {'参数': 'Other rows per round', '当前值': int(df['target_label'].eq('BK').sum()), '计算/含义': '每轮从Other抽取与BK等量的样本'},
    {'参数': 'top_ks', '当前值': str(CONFIG.bootstrap_top_ks), '计算/含义': '统计进入Top-K集合的频率'},
    {'参数': 'random_state', '当前值': CONFIG.random_state, '计算/含义': '随机数种子，保证结果可复现'},
])
show_table('Bootstrap稳定性分析参数', bootstrap_parameter_table)

feature_stability = run_bootstrap_stability(
    df,
    feature_cols,
    rounds=CONFIG.bootstrap_rounds,
    top_ks=CONFIG.bootstrap_top_ks,
    random_state=CONFIG.random_state,
)
write_csv(add_feature_meaning_columns(feature_stability), RUN_DIR / 'feature_stability.csv')
show_table('Bootstrap稳定特征Top40', feature_stability, rows=40)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook07 跨流速一致性分析

本节比较 `BK00 vs FL00` 和 `BK05 vs FL05`，寻找在 v0 与 v0.5 流速下都能稳定区分断丝与流噪背景的特征。表中列举的是按 `cross_flow_score` 排序后的前若干特征，因为这些特征同时满足“两个流速下都有判别力、判别方向一致、受流速本身影响较小”。

主要公式：`auc_lift = max(AUC, 1-AUC) - 0.5`；`cross_condition_consistency = min(lift00, lift05) / max(lift00, lift05)`；`median_shift_norm = |median(a)-median(b)| / scale`，其中 `scale` 优先用两组数据合并后的 IQR；`cross_flow_score = direction_same * (0.55*(lift00+lift05)+0.45*consistency) / (1+0.5*bk_shift+0.25*fl_shift)`。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.cross_flow_analysis import evaluate_cross_flow_features

cross_flow_metric_explain = pd.DataFrame([
    {'列名': 'auc_BK00_vs_FL00', '含义': 'v0流速下BK相对FL的原始AUC', '方向': '偏离0.5越大越好', '计算方式': 'roc_auc_score(BK00=1, FL00=0)'},
    {'列名': 'auc_BK05_vs_FL05', '含义': 'v0.5流速下BK相对FL的原始AUC', '方向': '偏离0.5越大越好', '计算方式': 'roc_auc_score(BK05=1, FL05=0)'},
    {'列名': 'auc_lift_BK00_vs_FL00', '含义': 'v0无方向AUC提升', '方向': '越大越好', '计算方式': 'max(AUC, 1-AUC)-0.5'},
    {'列名': 'auc_lift_BK05_vs_FL05', '含义': 'v0.5无方向AUC提升', '方向': '越大越好', '计算方式': 'max(AUC, 1-AUC)-0.5'},
    {'列名': 'direction_same', '含义': '两个流速下BK方向是否一致', '方向': '1优于0', '计算方式': '(auc00>=0.5)==(auc05>=0.5)'},
    {'列名': 'cross_condition_consistency', '含义': '两个流速下判别强度是否接近', '方向': '越大越好', '计算方式': 'min(lift00,lift05)/max(lift00,lift05)'},
    {'列名': 'bk00_bk05_median_shift_norm', '含义': 'BK特征值受流速变化影响的归一化中位数位移', '方向': '越小越好', '计算方式': '|median(BK00)-median(BK05)|/scale'},
    {'列名': 'fl00_fl05_median_shift_norm', '含义': 'FL背景特征值受流速变化影响的归一化中位数位移', '方向': '越小越好', '计算方式': '|median(FL00)-median(FL05)|/scale'},
    {'列名': 'flow_sensitivity_penalty', '含义': '流速敏感性惩罚项', '方向': '越小越好', '计算方式': 'bk_shift + 0.5*fl_shift'},
    {'列名': 'cross_flow_score', '含义': '跨流速稳定断丝判别综合分', '方向': '越大越好', '计算方式': '方向一致性、双流速AUC提升、一致性和流速惩罚的综合'},
])
show_table('跨流速一致性指标含义与公式', cross_flow_metric_explain)

cross_flow_feature = evaluate_cross_flow_features(df, feature_cols)
write_csv(add_feature_meaning_columns(cross_flow_feature), RUN_DIR / 'cross_flow_feature.csv')
show_table('跨流速一致性特征Top40', cross_flow_feature, rows=40)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook08 最终特征评价与结论

本节融合判别能力、抗 QJ 干扰能力、Bootstrap 稳定性、跨流速一致性和冗余惩罚，得到最终特征排序。最终分数越大越推荐；`feature_grade=A` 表示稳定断丝核心候选特征，`B` 表示候选特征，`C` 表示工况相关、冗余较高或稳定性不足的特征。

字段解释：`bk_nonbk_score` 越大越能区分断丝与其他；`bk_qj_score` 越大越能区分断丝与敲击干扰；`bootstrap_stability_score` 越大越稳定；`cross_flow_score` 越大越跨流速稳定；`flow_sensitivity_penalty` 和 `redundancy_penalty` 越大越不利。
"""
        )
    )
    cells.append(
        code(
            """
final_feature_ranking = build_final_ranking(
    feature_columns=feature_cols,
    discrimination=feature_discrimination,
    stability=feature_stability,
    cross_flow=cross_flow_feature,
    redundancy=correlation_cluster,
    mrmr_rank=mrmr_rank,
)
write_csv(add_feature_meaning_columns(final_feature_ranking), RUN_DIR / 'final_feature_ranking.csv')

final_ranking_path = RUN_DIR / 'plots/final_feature_ranking_top30.png'
top_boxplot_path = RUN_DIR / 'plots/top_feature_boxplots.png'
plot_final_ranking(final_feature_ranking, final_ranking_path, top_n=30)
plot_top_feature_boxplots(
    df,
    final_feature_ranking.head(CONFIG.max_plot_features)['feature'].tolist(),
    top_boxplot_path,
    max_features=12,
)

show_table('最终特征排序Top60', final_feature_ranking, rows=60)
show_image('最终特征Top30排序图', final_ranking_path)
show_image('最终Top特征六类箱线图', top_boxplot_path)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook09 特征挖掘后的分类测试

本节用于验证挖掘出的特征是否能支持实际分类。模型包括逻辑回归、线性 SVM 和 RBF-SVM。数据集划分采用比挖掘阶段更严格的规则：在每个来源标签内部按 `source_file_name` 分组划分训练/测试，同一源文件不会同时进入训练集和测试集；训练集多数类最多采样到 `classification_max_train_rows_per_class`，测试集保持分组划分后的真实分布。

结果解释：`balanced_accuracy` 越大越好，适合不均衡数据；`auc` 越大越好，衡量排序能力；`recall_positive` 越大表示断丝漏检越少；`specificity` 越大表示 Other/QJ 误报越少。推荐特征数量优先选择综合分最高且特征数较少的组合。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.classification_test import run_classification_tests

classification_results, comparison_feature_recommendations = run_classification_tests(
    df,
    feature_discrimination=feature_discrimination,
    final_ranking=final_feature_ranking,
    output_dir=RUN_DIR / 'classification',
    feature_counts=CONFIG.classification_feature_counts,
    max_train_rows_per_class=CONFIG.classification_max_train_rows_per_class,
    random_state=CONFIG.random_state,
)
write_csv(add_feature_meaning_columns(classification_results), RUN_DIR / 'classification_test_results.csv')
write_csv(add_feature_meaning_columns(comparison_feature_recommendations), RUN_DIR / 'comparison_feature_recommendations.csv')
write_markdown_summary(RUN_DIR / 'summary_report.md', dataset_summary, final_feature_ranking, dataset.load_report)

show_table('分类测试结果Top50', classification_results, rows=50)
show_table('五个比较任务的推荐特征组合', comparison_feature_recommendations)
"""
        )
    )
    cells.append(
        code(
            """
conclusion_rows = []
for _, row in comparison_feature_recommendations.iterrows():
    feats = str(row['recommended_features']).split(';')
    conclusion_rows.append({
        '比较任务': row['comparison'],
        '建议特征数量': int(row['recommended_feature_count']),
        '建议模型': row['recommended_model'],
        'balanced_accuracy越大越好': row['balanced_accuracy'],
        'auc越大越好': row['auc'],
        '断丝召回率越大越好': row['recall_positive'],
        'Other/QJ特异度越大越好': row['specificity'],
        '具体特征': '；'.join(feats),
        '具体特征中文含义': describe_feature_list(feats),
    })
conclusion_table = pd.DataFrame(conclusion_rows)
write_csv(conclusion_table, RUN_DIR / 'classification_conclusion_table.csv')
show_table('分类验证结论汇总', conclusion_table)
"""
        )
    )

    cells.append(
        md(
            """
## Notebook10 一键运行入口

本节提供命令式入口，适合不逐节查看中间结果时直接生成全部产物。它会自动创建带时间戳的输出目录，并生成方案约定的核心 CSV、分布图、组合搜索结果、最终排序和分类测试结果。正式分析建议保持 `bootstrap_rounds=200` 或更高；快速调试可设置 `max_rows_per_label` 和较小的 `bootstrap_rounds`。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.run_all import run_pccp_feature_mining

# summary = run_pccp_feature_mining(CONFIG)
# summary
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
