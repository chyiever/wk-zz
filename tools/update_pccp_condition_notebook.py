"""重建 PCCP 特征挖掘 notebook 第 04-10 节。

该脚本只替换 04 节之后的单元格，并保留 00-03 节已有内容与输出。
使用脚本生成 notebook 可以避免手工编辑大段 JSON 时破坏结构。
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


def build_new_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []

    cells.extend(
        [
            md(
                """
## 04 静水工况（0 m/s）断丝特征分析与测试

本节只研究静水工况下的断丝识别：正类为 `BK00`，负类为同一流速下的 `FL00+QJ00`，即 `BK00 vs non-BK00`。这样先排除流速变化带来的复杂性，从最简单工况检验“断丝与其他信号”的差异。

分析顺序按“从简单到复杂”组织：先用 Pearson/Spearman 找冗余，再做多特征组合搜索，然后用 Bootstrap 检查稳定性，最后把筛出的特征放入分类器验证。
"""
            ),
            md(
                """
### 4.0 分工况分析工具

本单元定义 04、05、06 三章共用的函数。每个工况都复用同一套代码，避免因为复制粘贴导致指标口径不一致。

- 冗余分析：用 Pearson 相关 $r$ 检查线性重复，用 Spearman 相关 $\\rho$ 检查单调重复；若 $|r|$ 或 $|\\rho|\\ge0.92$，认为两个特征提供高度重复的信息。
- 组合搜索：先用 mRMR 得到“高判别、低冗余”的候选序列，再用 ReliefF 从局部邻域角度补充排序，最后用 SFS 直接围绕 LDA 交叉验证 AUC 贪心搜索组合。
- 稳定性：用 Bootstrap 多次构造“全量断丝 + 等量非断丝”平衡样本，统计特征进入 Top-K 的频率。
- 分类测试：用逻辑回归、线性 SVM、RBF-SVM 比较不同特征数量下的真实分类表现。
"""
            ),
            code(
                """
sync_output_counters(min_table_no=24, min_figure_no=20)
import numpy as np
from pccp_feature_mining.feature_selection import build_relevance_series, run_mrmr_ranking
from pccp_feature_mining.feature_redundancy import (
    build_correlation_clusters,
    build_redundancy_recommendations,
    compare_correlation_methods,
    compute_correlation_matrix,
    high_correlation_pairs,
)
from pccp_feature_mining.combination_search import evaluate_mrmr_prefixes, relief_like_ranking, sequential_forward_search
from pccp_feature_mining.bootstrap_stability import run_bootstrap_stability
from pccp_feature_mining.classification_test import run_classification_tests
from pccp_feature_mining.cross_flow_analysis import evaluate_cross_flow_features

CONDITION_TASKS = {
    'v0': {
        'chapter': '04',
        'label': '静水工况（0 m/s）',
        'short': 'v0',
        'comparison': 'BK00_NONBK00',
        'classification': 'BK00_vs_NONBK00',
        'positive_labels': ('BK00',),
        'negative_labels': ('FL00', 'QJ00'),
    },
    'v05': {
        'chapter': '05',
        'label': '0.5 m/s工况',
        'short': 'v05',
        'comparison': 'BK05_NONBK05',
        'classification': 'BK05_vs_NONBK05',
        'positive_labels': ('BK05',),
        'negative_labels': ('FL05', 'QJ05'),
    },
    'all': {
        'chapter': '06',
        'label': '多工况合并',
        'short': 'all_flow',
        'comparison': 'BK_NONBK',
        'classification': 'BK_vs_NONBK',
        'positive_labels': ('BK00', 'BK05'),
        'negative_labels': ('FL00', 'FL05', 'QJ00', 'QJ05'),
    },
}
condition_results = {}


def _task(key: str) -> dict:
    return CONDITION_TASKS[key]


def _result(key: str) -> dict:
    return condition_results.setdefault(key, {'task': _task(key)})


def _task_frame(task: dict) -> pd.DataFrame:
    labels = task['positive_labels'] + task['negative_labels']
    return df[df['source_label'].astype(str).isin(labels)].copy()


def _task_path(task: dict, suffix: str) -> Path:
    return RUN_DIR / f\"{task['short']}_{suffix}\"


def _feature_list_text(features: list[str], max_items: int = 12) -> str:
    return '；'.join(features[:max_items])


def _build_condition_final_ranking(task: dict, stability: pd.DataFrame, redundancy: pd.DataFrame, mrmr_rank: pd.DataFrame) -> pd.DataFrame:
    \"\"\"融合单特征判别、稳定性、mRMR和冗余惩罚，得到当前工况的最终排序。\"\"\"
    disc = feature_discrimination[feature_discrimination['comparison'].eq(task['comparison'])]
    disc_score = disc.set_index('feature')['discrimination_score'].astype(float) if not disc.empty else pd.Series(dtype=float)
    stable_top20 = stability.set_index('feature')['top20_frequency'].astype(float) if not stability.empty and 'top20_frequency' in stability.columns else pd.Series(dtype=float)
    stable_rank = stability.set_index('feature')['rank_stability_score'].astype(float) if not stability.empty and 'rank_stability_score' in stability.columns else pd.Series(dtype=float)
    red_penalty = redundancy.set_index('feature')['redundancy_penalty'].astype(float) if not redundancy.empty and 'redundancy_penalty' in redundancy.columns else pd.Series(dtype=float)
    mrmr_series = mrmr_rank.set_index('feature')['mrmr_rank'].astype(float) if not mrmr_rank.empty and 'mrmr_rank' in mrmr_rank.columns else pd.Series(dtype=float)
    max_mrmr_rank = max(float(mrmr_series.max()) if len(mrmr_series) else 1.0, 1.0)

    rows = []
    for feature in feature_cols:
        d_score = float(disc_score.get(feature, 0.0))
        stability_score = 0.6 * float(stable_top20.get(feature, 0.0)) + 0.4 * float(stable_rank.get(feature, 0.0))
        m_rank = mrmr_series.get(feature, np.nan)
        mrmr_score = 0.0 if pd.isna(m_rank) else 1.0 - (float(m_rank) - 1.0) / max_mrmr_rank
        penalty = float(red_penalty.get(feature, 0.0))
        final_score = (0.55 * d_score + 0.25 * stability_score + 0.20 * mrmr_score) / (1.0 + 0.45 * penalty)
        rows.append({
            'condition': task['label'],
            'comparison': task['comparison'],
            'feature': feature,
            'condition_final_score': final_score,
            'discrimination_score': d_score,
            'bootstrap_stability_score': stability_score,
            'mrmr_score_norm': mrmr_score,
            'redundancy_penalty': penalty,
            'mrmr_rank': int(m_rank) if pd.notna(m_rank) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values('condition_final_score', ascending=False).reset_index(drop=True)
    out['condition_rank'] = np.arange(1, len(out) + 1)
    q75 = float(out['condition_final_score'].quantile(0.75)) if len(out) else 0.0
    q45 = float(out['condition_final_score'].quantile(0.45)) if len(out) else 0.0
    out['feature_grade'] = 'C'
    out.loc[(out['condition_final_score'] >= q45) | (out['condition_rank'] <= 80), 'feature_grade'] = 'B'
    out.loc[(out['condition_final_score'] >= q75) & (out['redundancy_penalty'] < CONFIG.correlation_threshold), 'feature_grade'] = 'A'
    return out


def run_condition_pearson(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    task_df = _task_frame(task)
    relevance = build_relevance_series(feature_discrimination, comparison=task['comparison'])
    pearson_corr = compute_correlation_matrix(task_df, feature_cols, method='pearson', max_rows=CONFIG.max_correlation_rows, random_state=CONFIG.random_state)
    pearson_high_pairs = high_correlation_pairs(pearson_corr, threshold=CONFIG.correlation_threshold)
    result.update({'frame': task_df, 'relevance': relevance, 'pearson_corr': pearson_corr, 'pearson_high_pairs': pearson_high_pairs})
    pearson_corr.to_csv(_task_path(task, 'pearson_correlation_matrix.csv'), encoding='utf-8-sig')
    write_csv(add_feature_meaning_columns(pearson_high_pairs), _task_path(task, 'pearson_high_correlation_pairs.csv'))
    return show_table(f\"{task['label']} Pearson高相关特征对\", pearson_high_pairs, rows=10)


def run_condition_spearman(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    task_df = result.get('frame', _task_frame(task))
    relevance = result.get('relevance', build_relevance_series(feature_discrimination, comparison=task['comparison']))
    spearman_corr = compute_correlation_matrix(task_df, feature_cols, method='spearman', max_rows=CONFIG.max_correlation_rows, random_state=CONFIG.random_state)
    spearman_high_pairs = high_correlation_pairs(spearman_corr, threshold=CONFIG.correlation_threshold)
    correlation_cluster = build_correlation_clusters(spearman_corr, relevance, threshold=CONFIG.correlation_threshold)
    result.update({'spearman_corr': spearman_corr, 'spearman_high_pairs': spearman_high_pairs, 'correlation_cluster': correlation_cluster})
    spearman_corr.to_csv(_task_path(task, 'spearman_correlation_matrix.csv'), encoding='utf-8-sig')
    write_csv(add_feature_meaning_columns(spearman_high_pairs), _task_path(task, 'spearman_high_correlation_pairs.csv'))
    write_csv(add_feature_meaning_columns(correlation_cluster), _task_path(task, 'correlation_cluster.csv'))
    show_table(f\"{task['label']} Spearman高相关特征对\", spearman_high_pairs, rows=10)
    return show_table(f\"{task['label']} Spearman相关簇\", correlation_cluster, rows=10)


def run_condition_redundancy_summary(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    method_comparison = compare_correlation_methods(result['pearson_high_pairs'], result['spearman_high_pairs'])
    recommendations = build_redundancy_recommendations(result['pearson_corr'], result['spearman_corr'], result['relevance'], threshold=CONFIG.correlation_threshold)
    result.update({'correlation_method_comparison': method_comparison, 'redundancy_recommendations': recommendations})
    write_csv(add_feature_meaning_columns(method_comparison), _task_path(task, 'correlation_method_comparison.csv'))
    write_csv(add_feature_meaning_columns(recommendations), _task_path(task, 'redundancy_recommendations.csv'))
    show_table(f\"{task['label']} Pearson与Spearman高相关对比\", method_comparison, rows=10)
    return show_table(f\"{task['label']} 高相关冗余组与保留建议\", recommendations, rows=10)


def run_condition_combinations(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    mrmr_rank = run_mrmr_ranking(result['relevance'], result['spearman_corr'], top_n=CONFIG.mrmr_top_n, redundancy_weight=CONFIG.mrmr_redundancy_weight)
    mrmr_prefix = evaluate_mrmr_prefixes(df, mrmr_rank, feature_cols, positive_labels=task['positive_labels'], negative_labels=task['negative_labels'], counts=CONFIG.combination_feature_counts, random_state=CONFIG.random_state)
    relieff_rank = relief_like_ranking(df, feature_cols, positive_labels=task['positive_labels'], negative_labels=task['negative_labels'], random_state=CONFIG.random_state)
    sfs_candidates = list(dict.fromkeys(mrmr_rank['feature'].head(CONFIG.sfs_candidate_count).tolist() + relieff_rank['feature'].head(CONFIG.sfs_candidate_count).tolist()))
    sfs_selection_path = sequential_forward_search(df, sfs_candidates, feature_cols, positive_labels=task['positive_labels'], negative_labels=task['negative_labels'], max_selected=CONFIG.sfs_max_selected, random_state=CONFIG.random_state)
    for frame in (mrmr_prefix, sfs_selection_path):
        if not frame.empty and 'comparison' in frame.columns:
            frame['comparison'] = task['comparison']
    if sfs_selection_path.empty:
        combination_search = mrmr_prefix.copy()
    else:
        combination_search = pd.concat([
            mrmr_prefix,
            sfs_selection_path.rename(columns={'step': 'sfs_step'})[['method', 'comparison', 'feature_count', 'cv_auc_abs', 'lda_separation', 'selected_features']],
        ], ignore_index=True, sort=False)
    result.update({'mrmr_rank': mrmr_rank, 'mrmr_prefix': mrmr_prefix, 'relieff_rank': relieff_rank, 'sfs_selection_path': sfs_selection_path, 'feature_combination_search': combination_search})
    write_csv(add_feature_meaning_columns(mrmr_rank), _task_path(task, 'mrmr_rank.csv'))
    write_csv(add_feature_meaning_columns(mrmr_prefix), _task_path(task, 'feature_combination_mrmr_prefix.csv'))
    write_csv(add_feature_meaning_columns(relieff_rank), _task_path(task, 'relieff_rank.csv'))
    write_csv(add_feature_meaning_columns(sfs_selection_path), _task_path(task, 'sfs_selection_path.csv'))
    write_csv(add_feature_meaning_columns(combination_search), _task_path(task, 'feature_combination_search.csv'))
    show_table(f\"{task['label']} mRMR特征排序\", mrmr_rank, rows=10)
    show_table(f\"{task['label']} 近似ReliefF特征排序\", relieff_rank, rows=10)
    show_table(f\"{task['label']} SFS逐步选择路径\", sfs_selection_path, rows=10)
    return show_table(f\"{task['label']} 多特征组合搜索对比\", combination_search.sort_values('cv_auc_abs', ascending=False), rows=10)


def run_condition_stability(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    stability = run_bootstrap_stability(result.get('frame', _task_frame(task)), feature_cols, rounds=CONFIG.bootstrap_rounds, top_ks=CONFIG.bootstrap_top_ks, random_state=CONFIG.random_state)
    result['feature_stability'] = stability
    write_csv(add_feature_meaning_columns(stability), _task_path(task, 'feature_stability.csv'))
    return show_table(f\"{task['label']} Bootstrap稳定特征\", stability, rows=10)


def run_condition_final(key: str) -> pd.DataFrame:
    task = _task(key)
    result = _result(key)
    final_ranking = _build_condition_final_ranking(task, result['feature_stability'], result['correlation_cluster'], result['mrmr_rank'])
    result['final_ranking'] = final_ranking
    write_csv(add_feature_meaning_columns(final_ranking), _task_path(task, 'final_feature_ranking.csv'))
    ranking_path = RUN_DIR / 'plots' / f\"{task['short']}_final_feature_ranking_top30.png\"
    boxplot_path = RUN_DIR / 'plots' / f\"{task['short']}_top_feature_boxplots.png\"
    plot_final_ranking(final_ranking.rename(columns={'condition_final_score': 'final_score'}), ranking_path, top_n=30)
    plot_top_feature_boxplots(result.get('frame', _task_frame(task)), final_ranking.head(12)['feature'].tolist(), boxplot_path, max_features=12)
    show_table(f\"{task['label']} 最终特征选择结论\", final_ranking, rows=10)
    show_image(f\"{task['label']} 最终特征Top30排序图\", ranking_path)
    show_image(f\"{task['label']} Top特征箱线图\", boxplot_path)
    return final_ranking


def run_condition_classification(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = _task(key)
    result = _result(key)
    final_for_classifier = result['final_ranking'].rename(columns={'condition_final_score': 'final_score'}).copy()
    classification_results, recommendations = run_classification_tests(
        df,
        feature_discrimination=feature_discrimination,
        final_ranking=final_for_classifier,
        output_dir=RUN_DIR / 'classification' / task['short'],
        feature_counts=CONFIG.classification_feature_counts,
        max_train_rows_per_class=CONFIG.classification_max_train_rows_per_class,
        random_state=CONFIG.random_state,
        comparisons={task['classification']: (task['positive_labels'], task['negative_labels'])},
        comparison_feature_order={task['classification']: task['comparison']},
    )
    result.update({'classification_results': classification_results, 'classification_recommendations': recommendations})
    write_csv(add_feature_meaning_columns(classification_results), _task_path(task, 'classification_test_results.csv'))
    write_csv(add_feature_meaning_columns(recommendations), _task_path(task, 'classification_recommendations.csv'))
    show_table(f\"{task['label']} 分类测试结果\", classification_results, rows=10)
    return classification_results, show_table(f\"{task['label']} 分类推荐特征组合\", recommendations, rows=10)
"""
            ),
        ]
    )

    def add_condition_section(key: str, chapter: str, title: str, comparison_text: str) -> None:
        if key == "v0":
            section_title = "### 4.1 静水工况下的冗余分析"
            prefix = "静水工况"
        elif key == "v05":
            cells.append(md(f"## 05 {title}\n\n本节分析 0.5 m/s 流速下的断丝识别：正类为 `BK05`，负类为 `FL05+QJ05`。其结构与第 4 节一致，便于比较两个单一流速工况下的特征差异。"))
            section_title = "### 5.1 0.5 m/s工况下的冗余分析"
            prefix = "0.5 m/s工况"
        else:
            cells.append(md(f"## 06 {title}\n\n本节把两种流速合并为二分类任务：正类为 `BK00+BK05`，负类为 `FL00+FL05+QJ00+QJ05`。它更接近实际部署问题：模型需要同时面对不同流速下的断丝与非断丝信号。"))
            section_title = "### 6.1 多工况冗余分析"
            prefix = "多工况"
        cells.extend(
            [
                md(
                    f"""
{section_title}

冗余分析的目标不是直接判断特征好坏，而是找到“表达同一信息”的特征组。后续最终特征选择会优先保留组内判别分更高、稳定性更好的代表特征。
"""
                ),
                md(
                    f"""
#### {chapter}.1.1 Pearson线性冗余分析

Pearson 相关系数度量两个特征的线性相关：
$$r=\\frac{{\\sum_i(x_i-\\bar{{x}})(y_i-\\bar{{y}})}}{{\\sqrt{{\\sum_i(x_i-\\bar{{x}})^2}}\\sqrt{{\\sum_i(y_i-\\bar{{y}})^2}}}},\\qquad r\\in[-1,1]$$
当 $|r|\\ge \\mathrm{{correlation\\_threshold}}$ 时，本流程认为两者存在线性冗余。
"""
                ),
                code(f"run_condition_pearson('{key}')"),
                md(
                    f"""
#### {chapter}.1.2 Spearman单调冗余分析

Spearman 相关先把数值转换成秩，再计算 Pearson 相关：
$$\\rho=1-\\frac{{6\\sum_i d_i^2}}{{n(n^2-1)}}$$
其中 $d_i$ 是两个特征在第 $i$ 个样本上的秩差。它对非线性但单调的冗余关系更敏感，因此本流程用 Spearman 相关簇作为最终冗余惩罚依据。
"""
                ),
                code(f"run_condition_spearman('{key}')"),
                md(
                    f"""
#### {chapter}.1.3 Pearson与Spearman冗余建议

将两种相关方法命中的高相关特征对合并成连通组，并在每组中保留当前工况单特征判别分最高的代表特征：
$$f^*=\\arg\\max_{{f\\in G}}D_{{{comparison_text}}}(f)$$
其余特征作为冗余删除候选，而不是立即永久删除；若后续分类测试显示某些冗余特征有组合增益，可以重新纳入。
"""
                ),
                code(f"run_condition_redundancy_summary('{key}')"),
                md(
                    f"""
### {chapter}.2 {prefix}下的多特征组合搜索

本节目标为 `{comparison_text}`。mRMR 每一步选择兼顾相关性和低冗余的特征：
$$\\mathrm{{score}}(f)=D(f)-\\lambda\\frac{{1}}{{|S|}}\\sum_{{g\\in S}}|\\rho(f,g)|$$
ReliefF 从近邻结构判断特征能否拉开异类、压近同类；SFS 则每轮加入使 LDA 交叉验证 AUC 最大的特征：
$$f_t^*=\\arg\\max_{{f\\in C_t}}\\mathrm{{AUC}}_{{LDA}}(S_t\\cup\\{{f\\}})$$
"""
                ),
                code(f"run_condition_combinations('{key}')"),
                md(
                    f"""
### {chapter}.3 {prefix}下的特征稳定性分析

Bootstrap 稳定性用于回答：某个特征是否只是在某一次抽样中偶然靠前。Top-K 频率定义为：
$$\\mathrm{{freq}}_K(f)=\\frac{{1}}{{B}}\\sum_{{b=1}}^B\\mathbf{{1}}[\\mathrm{{rank}}_b(f)\\le K]$$
频率越高，表示该特征在当前工况下越稳定。
"""
                ),
                code(f"run_condition_stability('{key}')"),
                md(
                    f"""
### {chapter}.4 {prefix}下的最终特征选择结论

最终分数融合单特征判别、Bootstrap 稳定性、mRMR 排名和冗余惩罚：
$$S=\\frac{{0.55D+0.25S_{{stab}}+0.20S_{{mRMR}}}}{{1+0.45P_{{red}}}}$$
其中 $D$ 为当前工况判别分，$S_{{stab}}$ 为稳定性分，$S_{{mRMR}}$ 为归一化 mRMR 排名分，$P_{{red}}$ 为与更优特征的最大相关冗余惩罚。
"""
                ),
                code(f"{key}_final_feature_ranking = run_condition_final('{key}')"),
                md(
                    f"""
### {chapter}.5 {prefix}下的分类测试

本节只测试 `{comparison_text}`。训练/测试按源文件分组划分，避免同一源文件窗口同时出现在训练集和测试集造成信息泄漏。评价指标包括 balanced accuracy、AUC、断丝召回率和非断丝特异度。
"""
                ),
                code(f"{key}_classification_results, {key}_classification_recommendations = run_condition_classification('{key}')"),
            ]
        )

    add_condition_section("v0", "4", "静水工况（0 m/s）断丝特征分析与测试", "BK00_NONBK00")
    add_condition_section("v05", "5", "0.5 m/s工况断丝特征分析与测试", "BK05_NONBK05")
    add_condition_section("all", "6", "多工况断丝特征分析与测试", "BK_NONBK")

    cells.extend(
        [
            md(
                """
## 07 跨流速特征一致性分析

本节对比 4.4、5.4、6.4 的最终特征，以及 4.5、5.5、6.5 的分类结果。目标是区分三类特征：单工况有效、跨工况一致、分类有效但解释不稳。
"""
            ),
            code(
                """
sync_output_counters(min_table_no=80, min_figure_no=40)
# 7.1 传统跨流速指标：观察BK相对FL背景的方向一致性和流速敏感性。
cross_flow_feature = evaluate_cross_flow_features(df, feature_cols)
write_csv(add_feature_meaning_columns(cross_flow_feature), RUN_DIR / 'cross_flow_feature.csv')
show_table('7.1 BK相对FL背景的跨流速一致性特征', cross_flow_feature, rows=10)


def _top_features_from_result(key: str, n: int = 20) -> list[str]:
    return condition_results[key]['final_ranking'].head(n)['feature'].tolist()


v0_top20 = _top_features_from_result('v0', 20)
v05_top20 = _top_features_from_result('v05', 20)
all_top20 = _top_features_from_result('all', 20)
common_three = sorted(set(v0_top20) & set(v05_top20) & set(all_top20))
common_single_flow = sorted(set(v0_top20) & set(v05_top20))

cross_condition_overlap = pd.DataFrame([
    {'对比对象': '4.4静水Top20 ∩ 5.4_0.5m/s Top20', '重叠数量': len(common_single_flow), '重叠特征': '；'.join(common_single_flow), '中文含义': describe_feature_list(common_single_flow)},
    {'对比对象': '4.4静水Top20 ∩ 5.4_0.5m/s Top20 ∩ 6.4多工况Top20', '重叠数量': len(common_three), '重叠特征': '；'.join(common_three), '中文含义': describe_feature_list(common_three)},
    {'对比对象': '仅静水Top20', '重叠数量': len(sorted(set(v0_top20) - set(v05_top20))), '重叠特征': '；'.join(sorted(set(v0_top20) - set(v05_top20))), '中文含义': describe_feature_list(sorted(set(v0_top20) - set(v05_top20)))},
    {'对比对象': '仅0.5m/s Top20', '重叠数量': len(sorted(set(v05_top20) - set(v0_top20))), '重叠特征': '；'.join(sorted(set(v05_top20) - set(v0_top20))), '中文含义': describe_feature_list(sorted(set(v05_top20) - set(v0_top20)))},
])
write_csv(cross_condition_overlap, RUN_DIR / 'cross_condition_feature_overlap.csv')
show_table('7.2 三类工况最终特征Top20重叠对比', cross_condition_overlap, rows=10)

classification_recommendation_summary = pd.concat([
    condition_results['v0']['classification_recommendations'].assign(condition='静水工况（0 m/s）'),
    condition_results['v05']['classification_recommendations'].assign(condition='0.5 m/s工况'),
    condition_results['all']['classification_recommendations'].assign(condition='多工况合并'),
], ignore_index=True, sort=False)
write_csv(add_feature_meaning_columns(classification_recommendation_summary), RUN_DIR / 'cross_condition_classification_recommendations.csv')
show_table('7.3 三类工况分类推荐结果对比', classification_recommendation_summary, rows=10)
"""
            ),
            md(
                """
## 08 三类工况推荐特征包汇总

本节把 04、05、06 三章的最终排序合并为可直接归档的总表。科研写作时，建议优先报告“三工况共同靠前”的特征，再补充单工况专用特征。
"""
            ),
            code(
                """
sync_output_counters(min_table_no=83, min_figure_no=40)
condition_final_top = pd.concat([
    condition_results['v0']['final_ranking'].head(30),
    condition_results['v05']['final_ranking'].head(30),
    condition_results['all']['final_ranking'].head(30),
], ignore_index=True, sort=False)
write_csv(add_feature_meaning_columns(condition_final_top), RUN_DIR / 'condition_final_top_features.csv')
show_table('三类工况Top30最终特征汇总', condition_final_top, rows=10)

recommended_feature_package = pd.DataFrame([
    {'推荐层级': '跨流速通用优先', '选择依据': '同时出现在4.4、5.4、6.4的Top20中', '特征': '；'.join(common_three), '中文含义': describe_feature_list(common_three)},
    {'推荐层级': '单流速模型补充', '选择依据': '分别参考4.4和5.4中靠前但未共同出现的特征', '特征': '静水：' + _feature_list_text(v0_top20) + '；0.5m/s：' + _feature_list_text(v05_top20), '中文含义': describe_feature_list(list(dict.fromkeys(v0_top20[:12] + v05_top20[:12])))},
    {'推荐层级': '多工况分类候选', '选择依据': '参考6.4最终排序和6.5分类推荐组合', '特征': _feature_list_text(all_top20), '中文含义': describe_feature_list(all_top20[:12])},
])
write_csv(recommended_feature_package, RUN_DIR / 'recommended_feature_package.csv')
show_table('推荐特征包汇总', recommended_feature_package, rows=10)
"""
            ),
            md(
                """
## 09 复现实验与结果审计

本节记录本 notebook 的关键输出位置和展示约束。所有完整表格都会写入 `RUN_DIR`，notebook 页面中每次最多展示 10 行，避免长表格干扰阅读。
"""
            ),
            code(
                """
sync_output_counters(min_table_no=85, min_figure_no=40)
notebook_audit = pd.DataFrame([
    {'审计项': '输出目录', '结果': str(RUN_DIR), '说明': '完整CSV、图像和分类模型均存储在该目录下'},
    {'审计项': '表格展示行数', '结果': '最多10行', '说明': 'show_table默认head(10)，显式rows也控制在10以内'},
    {'审计项': '工况任务', '结果': 'v0, v05, all_flow', '说明': '分别对应第4、5、6节'},
    {'审计项': '分类划分', '结果': '按source_file_name分组', '说明': '降低同源窗口泄漏风险'},
    {'审计项': '随机种子', '结果': CONFIG.random_state, '说明': 'Bootstrap、抽样、分类划分保持可复现'},
])
write_csv(notebook_audit, RUN_DIR / 'notebook_audit.csv')
show_table('复现实验与结果审计', notebook_audit, rows=10)
"""
            ),
            md(
                """
## 10 一键运行入口

如果不想逐节执行，可以在前面 00-03 完成后，取消下方函数调用的注释。函数会按 04、05、06、07、08、09 的顺序运行三类工况分析，并返回推荐特征包和分类推荐结果。
"""
            ),
            code(
                """
def run_condition_feature_mining_notebook() -> dict[str, pd.DataFrame]:
    \"\"\"按本notebook第04-09节顺序一键运行分工况特征分析。\"\"\"
    for key in ('v0', 'v05', 'all'):
        run_condition_pearson(key)
        run_condition_spearman(key)
        run_condition_redundancy_summary(key)
        run_condition_combinations(key)
        run_condition_stability(key)
        run_condition_final(key)
        run_condition_classification(key)
    return {
        'v0_final': condition_results['v0']['final_ranking'],
        'v05_final': condition_results['v05']['final_ranking'],
        'all_final': condition_results['all']['final_ranking'],
        'v0_classification': condition_results['v0']['classification_recommendations'],
        'v05_classification': condition_results['v05']['classification_recommendations'],
        'all_classification': condition_results['all']['classification_recommendations'],
    }

# summary = run_condition_feature_mining_notebook()
# summary
"""
            ),
        ]
    )
    return cells


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            src = src.replace("rows=100", "rows=10")
            src = src.replace("rows=50", "rows=10")
            src = src.replace("rows=30", "rows=10")
            src = src.replace("rows=20", "rows=10")
            cell["source"] = src.splitlines(True)
    nb["cells"] = nb["cells"][:36] + build_new_cells()
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK_PATH} with {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
