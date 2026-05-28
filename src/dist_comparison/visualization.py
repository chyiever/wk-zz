"""特征分布对比可视化函数"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_coverage_comparison(coverages: dict, feature_names: list,
                             save_path: str = None) -> plt.Figure:
    """
    绘制训练集在各特征上对真实数据分位数的覆盖率对比。

    参数:
        coverages: dict {group_name: {feat: coverage_dict}}
        feature_names: 特征短名列表
        save_path: 保存路径
    """
    n_groups = len(coverages)
    n_feats = len(feature_names)
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    q_labels = ['P1', 'P5', 'P25', 'P50', 'P75', 'P95', 'P99']

    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 5), sharey=True)
    if n_feats == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    group_names = list(coverages.keys())

    for f_idx, feat in enumerate(feature_names):
        ax = axes[f_idx]
        x = np.arange(len(quantiles))
        width = 0.8 / n_groups

        for g_idx, g_name in enumerate(group_names):
            cov_lower = [coverages[g_name][feat][q]['coverage_lower'] for q in quantiles]
            cov_upper = [coverages[g_name][feat][q]['coverage_upper'] for q in quantiles]
            ax.bar(x + g_idx * width, cov_lower, width, label=f'{g_name} (下尾)',
                   color=colors[g_idx], alpha=0.7)

        ax.set_xticks(x + width * (n_groups - 1) / 2)
        ax.set_xticklabels(q_labels, fontsize=10)
        ax.set_title(feat, fontsize=12)
        ax.set_ylabel('覆盖率', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('训练集对真实数据各分位数的覆盖率对比', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_distribution_overlap(x_ref: np.ndarray, x_test_noise: np.ndarray,
                              x_test_break: np.ndarray,
                              feature_names: list,
                              save_path: str = None) -> plt.Figure:
    """
    绘制真实数据与训练集（噪声/断丝）的分布重叠区域。

    参数:
        x_ref: 真实数据，2D array (n_samples, n_features)
        x_test_noise: 训练集噪声，2D array
        x_test_break: 训练集断丝，2D array
        feature_names: 特征短名列表
        save_path: 保存路径
    """
    n_feats = len(feature_names)
    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4.5))
    if n_feats == 1:
        axes = [axes]

    for idx, feat in enumerate(feature_names):
        ax = axes[idx]

        v_ref = x_ref[:, idx]
        v_noise = x_test_noise[:, idx]
        v_break = x_test_break[:, idx]

        # 共享 x 轴范围
        x_min = min(v_ref.min(), v_noise.min(), v_break.min())
        x_max = max(v_ref.max(), v_noise.max(), v_break.max())
        x_range = max(x_max - x_min, 1e-10)
        xs = np.linspace(x_min - 0.02 * x_range, x_max + 0.02 * x_range, 500)

        # KDE
        for v, color, label, ls in [
            (v_ref, '#1f77b4', '真实数据(噪声参考)', '-'),
            (v_noise, '#ff7f0e', '训练集: 噪声', '--'),
            (v_break, '#d62728', '训练集: 断丝', ':'),
        ]:
            if v.size > 1:
                kde = gaussian_kde(v)
                ys = kde(xs)
                ax.plot(xs, ys, color=color, linewidth=2, linestyle=ls, label=label)

                # 填充重叠区域
                ax.fill_between(xs, 0, ys, color=color, alpha=0.15)

        ax.set_title(feat, fontsize=11)
        ax.set_xlabel('特征值', fontsize=9)
        ax.set_ylabel('概率密度', fontsize=9)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('真实数据与训练集特征分布重叠对比', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ood_detection(ood_results: dict, feature_names: list,
                       save_path: str = None) -> plt.Figure:
    """
    绘制 OOD 检测结果：训练集超出真实数据范围的样本比例。

    参数:
        ood_results: dict {group_name: {feat: ood_dict}}
        feature_names: 特征短名列表
        save_path: 保存路径
    """
    n_groups = len(ood_results)
    n_feats = len(feature_names)
    group_names = list(ood_results.keys())

    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 5), sharey=True)
    if n_feats == 1:
        axes = [axes]

    colors = ['#ff7f0e', '#2ca02c']

    for f_idx, feat in enumerate(feature_names):
        ax = axes[f_idx]
        x = np.arange(n_groups)
        width = 0.35

        for g_idx, g_name in enumerate(group_names):
            ood = ood_results[g_name][feat]
            below = ood['ood_below_pct'] * 100
            above = ood['ood_above_pct'] * 100
            ax.bar(x[g_idx] - width / 2, below, width, label=f'{g_name}: 低于P1',
                   color=colors[g_idx], alpha=0.7)
            ax.bar(x[g_idx] + width / 2, above, width, label=f'{g_name}: 高于P99',
                   color=colors[g_idx], alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([g.replace('训练集+测试集: ', '') for g in group_names], fontsize=10)
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel('OOD 样本比例 (%)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('训练集 OOD 样本比例（超出真实数据 P1~P99 范围）', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_representativeness_summary(mmd_results: dict, wass_results: dict,
                                    coverage_results: dict, ood_results: dict,
                                    feature_names: list,
                                    save_path: str = None) -> plt.Figure:
    """
    绘制训练集代表性综合评估图。

    参数:
        mmd_results: dict {group_name: {feat: mmd_value}}
        wass_results: dict {group_name: {feat: wass_value}}
        coverage_results: dict {group_name: {feat: coverage_dict}}
        ood_results: dict {group_name: {feat: ood_dict}}
        feature_names: 特征短名列表
        save_path: 保存路径
    """
    n_feats = len(feature_names)
    group_names = list(mmd_results.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. MMD 对比
    ax = axes[0, 0]
    x = np.arange(n_feats)
    width = 0.8 / n_groups
    colors = ['#ff7f0e', '#2ca02c']
    for g_idx, g_name in enumerate(group_names):
        vals = [mmd_results[g_name][f] for f in feature_names]
        ax.bar(x + g_idx * width, vals, width, label=g_name.replace('训练集+测试集: ', ''),
               color=colors[g_idx], alpha=0.8)
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('MMD²', fontsize=10)
    ax.set_title('MMD 距离（越小表示分布越接近真实数据）', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. Wasserstein 距离对比
    ax = axes[0, 1]
    for g_idx, g_name in enumerate(group_names):
        vals = [wass_results[g_name][f] for f in feature_names]
        ax.bar(x + g_idx * width, vals, width, label=g_name.replace('训练集+测试集: ', ''),
               color=colors[g_idx], alpha=0.8)
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Wasserstein 距离', fontsize=10)
    ax.set_title('Wasserstein 距离（越小表示分布越接近真实数据）', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 3. 整体覆盖率对比
    ax = axes[1, 0]
    for g_idx, g_name in enumerate(group_names):
        vals = [coverage_results[g_name][f]['overall']['coverage_main'] * 100 for f in feature_names]
        ax.bar(x + g_idx * width, vals, width, label=g_name.replace('训练集+测试集: ', ''),
               color=colors[g_idx], alpha=0.8)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% 基准线')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('覆盖率 (%)', fontsize=10)
    ax.set_title('训练集落在真实数据 P2.5~P97.5 范围内的比例', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. OOD 比例对比
    ax = axes[1, 1]
    for g_idx, g_name in enumerate(group_names):
        vals = [ood_results[g_name][f]['ood_total_pct'] * 100 for f in feature_names]
        ax.bar(x + g_idx * width, vals, width, label=g_name.replace('训练集+测试集: ', ''),
               color=colors[g_idx], alpha=0.8)
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% 警戒线')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('OOD 比例 (%)', fontsize=10)
    ax.set_title('训练集超出真实数据 P1~P99 范围的样本比例', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('训练集代表性综合评估（以真实数据为参考分布）', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
