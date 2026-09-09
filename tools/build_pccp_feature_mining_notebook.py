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
    estimate_bootstrap_statistic_stability,
    estimate_feature_distribution_mmd,
    plot_bootstrap_rse_curves,
    plot_mmd_convergence_curve,
    summarize_bootstrap_rse_by_statistic,
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
### 2.7.1 统计特征稳定性：Bootstrap RSE

BK 每个物理事件仅保留 0-30 ms 窗口，FL/QJ 使用原始样本。对每个来源类别和样本量重复 Bootstrap；每个特征计算 mean、std、Q10、Q50、Q90，再以 `RSE = Bootstrap标准误 / |Bootstrap估计均值|` 计算相对标准误。汇总全部“特征 × 统计量”的 median RSE、P90 RSE 与最大 RSE；默认 P90 RSE 连续两个点不高于 10% 时判定稳定。

图 19 分成六个面板：mean、std、Q10、Q50、Q90 各一个面板，最后一个面板汇总全部“特征 × 五种统计量”。每个面板同时绘制 median RSE（虚线，代表典型特征）和 P90 RSE（实线圆点，代表 90% 特征覆盖）；来源类别由颜色区分。纵轴是无量纲相对抽样不确定性，不是特征幅值或能量。mean 表示中心水平稳定性，std 表示独立事件间离散程度稳定性，Q10/Q90 分别表示下尾/上尾覆盖稳定性，Q50 表示典型事件水平稳定性。

RSE 是标准误与估计量绝对值的比值，没有 100% 上限。RSE 大于 100% 表示 Bootstrap 标准误已经大于统计量自身绝对值，即相对尺度上极不稳定；本数据主要由均值/分位数接近零造成分母效应，以及 `k_hl/k_sc/k_res_*` 等比值型、重尾特征在重采样中剧烈变化造成。此时应结合绝对标准误、Bootstrap 区间、近零分母标记和原始分布判断，不能把数值直接解释成“误差概率超过 100%”。

10% 是本项目偏严格的工程目标，不是 Bootstrap 理论给出的统一常数。近似正态时 95% 相对置信半宽约为 `1.96 × RSE`，所以 RSE=10% 对应约 ±19.6%；若要求 95% 半宽不超过估计值的 10%，应使用约 5.1%。建议 ≤5% 视为高精度、5%-10% 视为工程稳定、10%-20% 仅作探索、≥30% 作为强不稳定警示；接近零的统计量应改看绝对标准误或置信区间。依据包括 Efron (1979, DOI: 10.1214/aos/1176344552)、CDC/NCHS RSE 定义与近零警告、Statistics Canada 按具体产品采用不同 CV 界限，以及 Jonsson & Nyberg (2022, DOI: 10.1002/psp4.12790) 对尾部分位数精度和 Bootstrap 样本数的讨论。
"""
        )
    )
    cells.append(
        code(
            """
RSE_STABILITY_THRESHOLD = 0.10
STABILITY_CONSECUTIVE_POINTS = 2
bk_0_30ms_df, bk_0_30ms_selection_audit = select_bk_0_30ms_samples(df)
non_bk_df = df.loc[~df['source_label'].astype(str).str.startswith('BK')]
sample_sufficiency_df = pd.concat([non_bk_df, bk_0_30ms_df], axis=0, ignore_index=False)
stat_sample_sizes_by_label = {
    str(label): list(range(min(max(int(CONFIG.sample_stability_min_samples), 2), len(group)), len(group) + 1))
    for label, group in bk_0_30ms_df.groupby('source_label', sort=True)
}
bootstrap_statistic_detail, bootstrap_statistic_curve, bootstrap_statistic_summary = estimate_bootstrap_statistic_stability(
    sample_sufficiency_df,
    feature_cols,
    sample_sizes_by_label=stat_sample_sizes_by_label,
    repeats=max(int(CONFIG.sample_stability_repeats), 200),
    min_samples=CONFIG.sample_stability_min_samples,
    rse_threshold=RSE_STABILITY_THRESHOLD,
    consecutive_points=STABILITY_CONSECUTIVE_POINTS,
    random_state=CONFIG.random_state,
)
write_csv(bk_0_30ms_selection_audit, RUN_DIR / 'sample_sufficiency_bk_0_30ms_audit.csv')
write_csv(bootstrap_statistic_detail, RUN_DIR / 'bootstrap_statistic_stability_detail.csv')
write_csv(bootstrap_statistic_curve, RUN_DIR / 'bootstrap_statistic_stability_curve.csv')
write_csv(bootstrap_statistic_summary, RUN_DIR / 'bootstrap_statistic_stability_summary.csv')
bootstrap_statistic_by_measure = summarize_bootstrap_rse_by_statistic(bootstrap_statistic_detail)
write_csv(bootstrap_statistic_by_measure, RUN_DIR / 'bootstrap_statistic_stability_by_statistic.csv')
bootstrap_rse_plot_path = RUN_DIR / 'plots/bootstrap_statistic_rse_curves.png'
plot_bootstrap_rse_curves(
    bootstrap_statistic_curve,
    bootstrap_rse_plot_path,
    statistic_curve=bootstrap_statistic_by_measure,
    threshold=RSE_STABILITY_THRESHOLD,
)

sample_stability_parameter_table = pd.DataFrame([
    {'参数': 'Bootstrap repeats', '当前值': max(int(CONFIG.sample_stability_repeats), 200), '含义': '每个类别、每个样本量重复次数'},
    {'参数': 'statistics', '当前值': 'mean, std, Q10, Q50, Q90', '含义': '每个特征计算的统计量'},
    {'参数': 'RSE threshold', '当前值': RSE_STABILITY_THRESHOLD, '含义': 'P90 RSE稳定阈值'},
    {'参数': 'consecutive points', '当前值': STABILITY_CONSECUTIVE_POINTS, '含义': '连续满足阈值的点数'},
])
show_table('2.7.1 Bootstrap统计稳定性参数', sample_stability_parameter_table)
show_table('2.7.1 各来源类别统计稳定性结论', bootstrap_statistic_summary)
show_table('2.7.1 Median/P90/最大RSE曲线明细', bootstrap_statistic_curve, rows=60)
show_table('2.7.1 当前最大样本量的高RSE特征统计量', bootstrap_statistic_detail.loc[bootstrap_statistic_detail['sample_size'].eq(bootstrap_statistic_detail['total_rows'])].sort_values('rse', ascending=False), rows=50)
show_image('2.7.1 Bootstrap统计特征稳定性RSE曲线', bootstrap_rse_plot_path)
"""
        )
    )

    cells.append(
        md(
            """
### 2.7.2 特征分布稳定性：MMD

用 Gaussian-RBF MMD 比较每个样本量的高维联合特征分布与全量参考分布。数百维特征先做稳健标准化，再用随机傅里叶特征近似核均值。六类等权，`sample_size_per_label` 表示每类抽取数；MMD P90 连续两个点不高于 0.05 时判定整体特征空间收敛。
"""
        )
    )
    cells.append(
        code(
            """
MMD_STABILITY_THRESHOLD = 0.05
MMD_RFF_COMPONENTS = 512
mmd_max_size_per_label = int(sample_sufficiency_df.groupby('source_label').size().min())
mmd_start = min(max(int(CONFIG.sample_stability_min_samples), 2), mmd_max_size_per_label)
mmd_sample_sizes = list(range(mmd_start, mmd_max_size_per_label + 1)) if mmd_max_size_per_label <= 100 else None
mmd_convergence_curve, mmd_convergence_summary = estimate_feature_distribution_mmd(
    sample_sufficiency_df,
    feature_cols,
    sample_sizes=mmd_sample_sizes,
    repeats=max(int(CONFIG.sample_stability_repeats), 100),
    min_samples=CONFIG.sample_stability_min_samples,
    rff_components=MMD_RFF_COMPONENTS,
    mmd_threshold=MMD_STABILITY_THRESHOLD,
    consecutive_points=STABILITY_CONSECUTIVE_POINTS,
    max_rows=10000,
    random_state=CONFIG.random_state,
)
write_csv(mmd_convergence_curve, RUN_DIR / 'feature_distribution_mmd_curve.csv')
write_csv(mmd_convergence_summary, RUN_DIR / 'feature_distribution_mmd_summary.csv')
mmd_plot_path = RUN_DIR / 'plots/feature_distribution_mmd_convergence.png'
plot_mmd_convergence_curve(mmd_convergence_curve, mmd_plot_path, threshold=MMD_STABILITY_THRESHOLD)
show_table('2.7.2 整体特征空间MMD收敛结论', mmd_convergence_summary)
show_table('2.7.2 各样本量MMD曲线明细', mmd_convergence_curve, rows=100)
show_image('2.7.2 整体特征空间MMD收敛曲线', mmd_plot_path)
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
