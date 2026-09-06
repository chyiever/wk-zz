"""生成PCCP断丝特征挖掘notebook。

使用脚本生成 notebook 可以避免手工编辑大段 JSON，也能保证中文以 UTF-8
写入，避免 PowerShell 命令行编码导致中文变成问号。
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

本 notebook 严格对应开发方案中的 Notebook01-Notebook08，并新增 Notebook09 分类测试。输入是六类已经计算好的特征向量，不重新计算原始波形特征；目标是从数百个候选特征中筛出能够稳定表征 PCCP 断丝响应的少量核心特征。全流程核心代码位于 `src/pccp_feature_mining`，notebook 负责组织实验、展示结果、解释指标含义并给出结论。
"""
        )
    )

    cells.append(
        md(
            """
## Notebook00 运行结果与算法逻辑审查

本节先读取当前 notebook 或一键流程已经生成的结果文件，检查是否存在算法逻辑问题。重点检查四类风险：数据读取是否漏类或跳过异常 CSV、BK 与 Other 是否严重不均衡、最终排序是否保留完全重复特征、方案要求的核心输出是否齐全。若发现问题，本 notebook 后续章节会用修正后的代码重新生成对应结果。
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

RUN_DIR = PROJECT_ROOT / 'outputs/PCCP_feature_mining_notebook/manual_run'
print('项目根目录:', PROJECT_ROOT)
print('默认输出目录:', RUN_DIR)
"""
        )
    )
    cells.append(
        code(
            """
required_outputs = [
    'dataset_summary.csv',
    'feature_list.csv',
    'quality_report.csv',
    'feature_discrimination.csv',
    'feature_stability.csv',
    'cross_flow_feature.csv',
    'final_feature_ranking.csv',
]
logic_checks = []

if not RUN_DIR.exists():
    logic_checks.append({'检查项': '已有输出目录', '状态': '未发现', '说明': '后续章节运行后会生成。'})
else:
    missing = [name for name in required_outputs if not (RUN_DIR / name).exists()]
    logic_checks.append({'检查项': '方案核心输出文件', '状态': '通过' if not missing else '缺失', '说明': ';'.join(missing) if missing else '核心CSV齐全。'})

    ds_path = RUN_DIR / 'dataset_summary.csv'
    if ds_path.exists():
        ds = pd.read_csv(ds_path, encoding='utf-8-sig')
        labels = set(ds['source_label'].astype(str))
        expected = {'FL00', 'FL05', 'BK00', 'QJ00', 'BK05', 'QJ05'}
        logic_checks.append({'检查项': '六类标签读取', '状态': '通过' if labels == expected else '需检查', '说明': f'读取标签={sorted(labels)}'})
        bk_rows = ds.loc[ds['target_label'].eq('BK'), 'rows'].sum()
        other_rows = ds.loc[ds['target_label'].eq('Other'), 'rows'].sum()
        ratio = other_rows / max(bk_rows, 1)
        logic_checks.append({'检查项': '类别不均衡', '状态': '提示', '说明': f'BK={bk_rows}, Other={other_rows}, Other/BK={ratio:.1f}；后续使用Bootstrap平衡评估和class_weight分类器。'})

    load_path = RUN_DIR / 'load_report.csv'
    if load_path.exists():
        load_report = pd.read_csv(load_path, encoding='utf-8-sig')
        skipped = load_report[load_report['status'].ne('ok')]
        logic_checks.append({'检查项': '异常CSV跳过', '状态': '提示' if len(skipped) else '通过', '说明': f'跳过/失败文件数={len(skipped)}；列数不足的异常分片不会进入分析。'})

    high_path = RUN_DIR / 'high_correlation_pairs.csv'
    final_path = RUN_DIR / 'final_feature_ranking.csv'
    if high_path.exists() and final_path.exists():
        high_pairs = pd.read_csv(high_path, encoding='utf-8-sig')
        final = pd.read_csv(final_path, encoding='utf-8-sig')
        top = set(final.head(30)['feature'])
        dup_top = high_pairs[(high_pairs['abs_correlation'] >= 0.999) & high_pairs['feature_a'].isin(top) & high_pairs['feature_b'].isin(top)]
        logic_checks.append({'检查项': 'Top特征完全相关重复', '状态': '需修正' if len(dup_top) else '通过', '说明': f'Top30内完全相关对数={len(dup_top)}；新版冗余惩罚已对非代表特征加重惩罚。'})

pd.DataFrame(logic_checks)
"""
        )
    )
    cells.append(
        md(
            """
### 0.1 逻辑审查结论

已有输出显示六类样本均能读取，`FL05` 存在一个表头列数不足的异常分片并被跳过，这是正确行为。主要逻辑短板是旧版 notebook 缺少六类分布图、组合搜索、分类验证和比较任务级结论；此外完全相关重复特征可能同时进入 Top 排名，新版 `feature_redundancy.py` 已将非代表重复特征纳入冗余惩罚。
"""
        )
    )

    cells.append(
        md(
            """
## Notebook01 数据读取与质量检查

本节实现方案中的“数据读取与质量检查”。算法上先递归读取六类输入目录下的 `features*.csv`，使用显式目录标签覆盖 CSV 内部可能不一致的标签，再统一生成 `source_label`、`target_label`、`signal_family`、`flow_condition`、`event_id` 和 `sampling_group_id`。质量检查关注缺失率、非有限值、尺度范围、IQR 异常值比例和特征列识别结果。

输出解释：`dataset_summary.csv` 中 `rows` 是窗口样本数，`source_files` 是物理源文件数，`sampling_groups` 是后续平衡采样使用的组数；`quality_report.csv` 中 `missing_rate` 越小越好，`unique_values` 太小说明特征近似恒定，`outlier_rate_iqr3` 越高越需要人工检查。
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
RUN_DIR.mkdir(parents=True, exist_ok=True)
"""
        )
    )
    cells.append(
        code(
            """
dataset = load_feature_dataset(
    feature_inputs=CONFIG.feature_inputs,
    min_feature_csv_columns=CONFIG.min_feature_csv_columns,
    max_rows_per_label=CONFIG.max_rows_per_label,
    random_state=CONFIG.random_state,
)
df = dataset.frame
feature_cols = list(dataset.feature_columns)

dataset_summary = build_dataset_summary(df)
feature_quality = build_feature_quality(df, feature_cols)
feature_list = feature_list_frame(feature_cols)

write_csv(dataset.load_report, RUN_DIR / 'load_report.csv')
write_csv(dataset_summary, RUN_DIR / 'dataset_summary.csv')
write_csv(feature_list, RUN_DIR / 'feature_list.csv')
write_csv(feature_quality, RUN_DIR / 'quality_report.csv')

print(f'样本行数: {len(df):,}, 特征数: {len(feature_cols):,}')
display(dataset_summary)
display(feature_quality.head(20))
display(dataset.load_report[dataset.load_report['status'] != 'ok'])
"""
        )
    )

    cells.append(
        md(
            """
## Notebook02 六类特征分布分析

本节补齐方案中的六类特征分布分析。箱线图用于观察中位数、四分位距和异常点；KDE 用于观察分布重叠程度，重叠越小通常越有利于分类；PCA 将全部标准化特征投影到二维，观察六类样本是否天然分离；UMAP 若环境安装 `umap-learn` 则作为非线性投影补充，否则输出空表并跳过图片。

输出解释：`six_class_distribution_features.csv` 的 `six_class_f_ratio` 表示类间均值差异相对类内波动的比例，越大说明该特征越适合做分布展示；`six_class_pca_explained_variance.csv` 中解释方差越高，二维 PCA 图越能代表原始特征空间。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.distribution_analysis import (
    select_distribution_features,
    run_pca_projection,
    run_umap_projection,
    plot_feature_kde_by_label,
    plot_projection,
)
from pccp_feature_mining.visualization import plot_final_ranking, plot_top_feature_boxplots

six_class_distribution_features = select_distribution_features(df, feature_cols, top_n=CONFIG.distribution_top_n)
pca_projection, pca_explained = run_pca_projection(df, feature_cols, max_rows=CONFIG.projection_max_rows, random_state=CONFIG.random_state)
umap_projection = run_umap_projection(df, feature_cols, max_rows=min(CONFIG.projection_max_rows, 5000), random_state=CONFIG.random_state)

write_csv(six_class_distribution_features, RUN_DIR / 'six_class_distribution_features.csv')
write_csv(pca_projection, RUN_DIR / 'six_class_pca_projection.csv')
write_csv(pca_explained, RUN_DIR / 'six_class_pca_explained_variance.csv')
write_csv(umap_projection, RUN_DIR / 'six_class_umap_projection.csv')

plot_top_feature_boxplots(df, six_class_distribution_features['feature'].head(12).tolist(), RUN_DIR / 'plots/six_class_feature_boxplots.png', max_features=12)
plot_feature_kde_by_label(df, six_class_distribution_features['feature'].head(8).tolist(), RUN_DIR / 'plots/six_class_feature_kde.png')
plot_projection(pca_projection, 'PC1', 'PC2', RUN_DIR / 'plots/six_class_pca.png', '六类特征PCA投影')
if not umap_projection.empty:
    plot_projection(umap_projection, 'UMAP1', 'UMAP2', RUN_DIR / 'plots/six_class_umap.png', '六类特征UMAP投影')

display(six_class_distribution_features)
display(pca_explained)
print('分布图输出:', RUN_DIR / 'plots')
"""
        )
    )

    cells.append(
        md(
            """
## Notebook03 单特征判别能力分析

本节对每个单特征计算二分类判别能力。AUC 衡量阈值从低到高扫描时的排序分离能力，`auc_abs` 越接近 1 越好，0.5 表示随机；Wasserstein distance 衡量两类分布整体位移，归一化后越大越分离；Cliff delta 衡量两类样本两两比较的优势概率，绝对值越大越好；Mutual Information 衡量非线性依赖，越大越好。综合分 `discrimination_score` 越大，说明单特征越值得进入候选集。

本节按方案和实际需求拆成 5 个重点比较：3.1 `BK00 vs NONBK00`，3.2 `BK00 vs QJ00`，3.3 `BK05 vs QJ05`，3.4 `BK05 vs NONBK05`，3.5 `BK vs NONBK`。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.feature_discrimination import evaluate_feature_discrimination

feature_discrimination = evaluate_feature_discrimination(df, feature_cols, random_state=CONFIG.random_state)
write_csv(feature_discrimination, RUN_DIR / 'feature_discrimination.csv')

metric_explain = pd.DataFrame([
    {'列名': 'auc_abs', '含义': '无方向AUC，越接近1越好，0.5表示随机'},
    {'列名': 'auc_lift', '含义': 'auc_abs - 0.5，越大越好'},
    {'列名': 'wasserstein_norm', '含义': '按IQR/标准差归一化的分布距离，越大越好'},
    {'列名': 'abs_cliff_delta', '含义': '两类两两比较优势强度，0到1，越大越好'},
    {'列名': 'mutual_info', '含义': '特征与类别的信息量，越大越好'},
    {'列名': 'discrimination_score', '含义': '单特征综合判别分，越大越好'},
])
display(metric_explain)
"""
        )
    )
    cells.append(
        code(
            """
for comparison in ['BK00_NONBK00', 'BK00_QJ00', 'BK05_QJ05', 'BK05_NONBK05', 'BK_NONBK', 'BK_QJ']:
    print('\\n', '=' * 20, comparison, '=' * 20)
    cols = ['comparison', 'feature', 'n_positive', 'n_negative', 'auc_abs', 'wasserstein_norm', 'abs_cliff_delta', 'mutual_info', 'discrimination_score']
    display(feature_discrimination[feature_discrimination['comparison'].eq(comparison)][cols].head(20))
"""
        )
    )

    cells.append(
        md(
            """
## Notebook04 特征冗余分析

本节分析特征冗余。Pearson 相关偏重线性关系，Spearman 相关偏重单调关系；本流程默认使用 Spearman，因为大量特征的数值尺度和分布形状不同。高相关特征对说明两个特征携带的信息高度重复，相关簇用于保留每组中判别分最高的代表特征，并对非代表重复特征增加惩罚。

输出解释：`high_correlation_pairs.csv` 中 `abs_correlation` 越接近 1 冗余越强；`correlation_cluster.csv` 中 `is_representative=True` 表示该簇推荐代表，`redundancy_penalty` 越高表示越应避免与更优特征同时进入最终组合。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.feature_selection import build_relevance_series, run_mrmr_ranking, build_final_ranking
from pccp_feature_mining.feature_redundancy import compute_correlation_matrix, high_correlation_pairs, build_correlation_clusters

relevance = build_relevance_series(feature_discrimination, comparison='BK_NONBK')
corr = compute_correlation_matrix(
    df,
    feature_cols,
    method='spearman',
    max_rows=CONFIG.max_correlation_rows,
    random_state=CONFIG.random_state,
)
corr.to_csv(RUN_DIR / 'spearman_correlation_matrix.csv', encoding='utf-8-sig')

high_pairs = high_correlation_pairs(corr, threshold=CONFIG.correlation_threshold)
correlation_cluster = build_correlation_clusters(corr, relevance, threshold=CONFIG.correlation_threshold)
write_csv(high_pairs, RUN_DIR / 'high_correlation_pairs.csv')
write_csv(correlation_cluster, RUN_DIR / 'correlation_cluster.csv')

display(high_pairs.head(30))
display(correlation_cluster.head(30))
"""
        )
    )

    cells.append(
        md(
            """
## Notebook05 多特征组合搜索

本节补齐方案中的多特征组合搜索。mRMR 先选择与目标类别相关性强、同时与已选特征冗余低的特征；近似 ReliefF 通过最近同类和最近异类样本比较，奖励“拉近同类、拉远异类”的特征；Sequential Forward Selection 每轮把候选特征逐一加入当前组合，用 LDA 交叉验证 AUC 和类间分离度选择提升最大的特征。

输出解释：`mrmr_score` 越大越优；`relief_score` 越大表示局部邻域区分能力越强；`cv_auc_abs` 越大表示组合在交叉验证中分离越稳定；`lda_separation` 越大表示 LDA 投影后两类均值相对类内方差分离越明显。
"""
        )
    )
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
write_csv(mrmr_rank, RUN_DIR / 'mrmr_rank.csv')

mrmr_prefix = evaluate_mrmr_prefixes(
    df,
    mrmr_rank,
    feature_cols,
    counts=CONFIG.combination_feature_counts,
    random_state=CONFIG.random_state,
)
relieff_rank = relief_like_ranking(df, feature_cols, random_state=CONFIG.random_state)
sfs_candidates = list(dict.fromkeys(mrmr_rank['feature'].head(CONFIG.sfs_candidate_count).tolist() + relieff_rank['feature'].head(CONFIG.sfs_candidate_count).tolist()))
sfs_selection_path = sequential_forward_search(
    df,
    sfs_candidates,
    feature_cols,
    max_selected=CONFIG.sfs_max_selected,
    random_state=CONFIG.random_state,
)
if sfs_selection_path.empty:
    feature_combination_search = mrmr_prefix.copy()
else:
    feature_combination_search = pd.concat([
        mrmr_prefix,
        sfs_selection_path.rename(columns={'step': 'sfs_step'})[['method', 'comparison', 'feature_count', 'cv_auc_abs', 'lda_separation', 'selected_features']]
    ], ignore_index=True, sort=False)

write_csv(mrmr_prefix, RUN_DIR / 'feature_combination_mrmr_prefix.csv')
write_csv(relieff_rank, RUN_DIR / 'relieff_rank.csv')
write_csv(sfs_selection_path, RUN_DIR / 'sfs_selection_path.csv')
write_csv(feature_combination_search, RUN_DIR / 'feature_combination_search.csv')

display(mrmr_rank.head(30))
display(relieff_rank.head(30))
display(sfs_selection_path)
display(feature_combination_search.sort_values('cv_auc_abs', ascending=False).head(20))
"""
        )
    )

    cells.append(
        md(
            """
## Notebook06 特征稳定性分析

本节实现方案中的 Bootstrap 稳定性分析。每轮保留全部 BK 样本，再从 Other 中抽取与 BK 规模接近的样本，重复评估单特征 AUC 排名。这样可以降低 FL 背景样本数量远大于 BK 时对统计结果的影响。`mean_rank` 越小越好，`rank_std` 越小越稳定，`top10_frequency/top20_frequency/top30_frequency` 越高说明特征更稳定地出现在候选集合中。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.bootstrap_stability import run_bootstrap_stability

feature_stability = run_bootstrap_stability(
    df,
    feature_cols,
    rounds=CONFIG.bootstrap_rounds,
    top_ks=CONFIG.bootstrap_top_ks,
    random_state=CONFIG.random_state,
)
write_csv(feature_stability, RUN_DIR / 'feature_stability.csv')
display(feature_stability.head(40))
"""
        )
    )

    cells.append(
        md(
            """
## Notebook07 跨流速一致性分析

本节比较 `BK00 vs FL00` 和 `BK05 vs FL05`，寻找在 v0 与 v0.5 流速下都能稳定区分断丝与流噪背景的特征。`direction_same=1` 表示两个流速下特征方向一致；`cross_condition_consistency` 越高说明两个流速下判别强度越接近；`flow_sensitivity_penalty` 越高说明特征本身受流速变化影响越大；`cross_flow_score` 越大越适合作为跨流速稳定特征。
"""
        )
    )
    cells.append(
        code(
            """
from pccp_feature_mining.cross_flow_analysis import evaluate_cross_flow_features

cross_flow_feature = evaluate_cross_flow_features(df, feature_cols)
write_csv(cross_flow_feature, RUN_DIR / 'cross_flow_feature.csv')
display(cross_flow_feature.head(40))
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
write_csv(final_feature_ranking, RUN_DIR / 'final_feature_ranking.csv')

plot_final_ranking(final_feature_ranking, RUN_DIR / 'plots/final_feature_ranking_top30.png', top_n=30)
plot_top_feature_boxplots(
    df,
    final_feature_ranking.head(CONFIG.max_plot_features)['feature'].tolist(),
    RUN_DIR / 'plots/top_feature_boxplots.png',
    max_features=12,
)

display(final_feature_ranking.head(60))
print('最终排序输出:', RUN_DIR / 'final_feature_ranking.csv')
"""
        )
    )

    cells.append(
        md(
            """
## Notebook09 特征挖掘后的分类测试

本节新增分类测试，用于验证挖掘出的特征是否能支持实际分类。模型包括逻辑回归、线性 SVM 和 RBF-SVM。数据集划分采用比挖掘阶段更严格的规则：在每个来源标签内部按 `source_file_name` 分组划分训练/测试，同一源文件不会同时进入训练集和测试集；训练集多数类最多采样到 `classification_max_train_rows_per_class`，测试集保持分组划分后的真实分布。

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
write_csv(classification_results, RUN_DIR / 'classification_test_results.csv')
write_csv(comparison_feature_recommendations, RUN_DIR / 'comparison_feature_recommendations.csv')
write_markdown_summary(RUN_DIR / 'summary_report.md', dataset_summary, final_feature_ranking, dataset.load_report)

display(classification_results.head(50))
display(comparison_feature_recommendations)
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
    })
conclusion_table = pd.DataFrame(conclusion_rows)
write_csv(conclusion_table, RUN_DIR / 'classification_conclusion_table.csv')
display(conclusion_table)
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
