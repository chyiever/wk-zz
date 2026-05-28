"""特征分布对比度量函数：MMD、Wasserstein距离、覆盖率、OOD比例"""

import numpy as np
from scipy.stats import kstest
from scipy.spatial.distance import cdist


def compute_mmd(x_ref: np.ndarray, x_test: np.ndarray,
                n_subsample: int = 5000, sigma: float = None) -> float:
    """
    计算 Maximum Mean Discrepancy (MMD) 使用 RBF 核。

    参数:
        x_ref: 参考分布样本（真实数据），1D array
        x_test: 测试分布样本（训练集），1D array
        n_subsample: 每组的最大采样数（MMD 复杂度 O(n^2)）
        sigma: RBF 核带宽，None 时自动使用 median heuristic

    返回:
        MMD^2 统计量
    """
    rng = np.random.default_rng(42)
    n = min(len(x_ref), len(x_test), n_subsample)
    x = rng.choice(x_ref, n, replace=False)
    y = rng.choice(x_test, n, replace=False)

    # Median heuristic for sigma
    if sigma is None:
        diffs = np.concatenate([
            np.abs(np.diff(np.sort(x[:500]))),
            np.abs(np.diff(np.sort(y[:500]))),
        ])
        sigma = np.median(diffs) if len(diffs) > 0 else 1.0
        sigma = max(sigma, 1e-10)

    # Compute pairwise distances
    xx = cdist(x.reshape(-1, 1), x.reshape(-1, 1), metric='euclidean')
    yy = cdist(y.reshape(-1, 1), y.reshape(-1, 1), metric='euclidean')
    xy = cdist(x.reshape(-1, 1), y.reshape(-1, 1), metric='euclidean')

    # RBF kernel
    k_xx = np.exp(-xx ** 2 / (2 * sigma ** 2))
    k_yy = np.exp(-yy ** 2 / (2 * sigma ** 2))
    k_xy = np.exp(-xy ** 2 / (2 * sigma ** 2))

    mmd_sq = (k_xx.sum() - np.diag(k_xx).sum()) / (n * (n - 1)) \
           + (k_yy.sum() - np.diag(k_yy).sum()) / (n * (n - 1)) \
           - 2 * k_xy.mean()

    return max(float(mmd_sq), 0.0)


def compute_wasserstein(x_ref: np.ndarray, x_test: np.ndarray) -> float:
    """
    计算一维 Wasserstein (Earth Mover's) 距离。

    参数:
        x_ref: 参考分布样本（真实数据），1D array
        x_test: 测试分布样本（训练集），1D array

    返回:
        Wasserstein-1 距离
    """
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(x_ref, x_test))


def compute_coverage(x_ref: np.ndarray, x_test: np.ndarray,
                     quantiles: list = None) -> dict:
    """
    计算训练集样本在真实数据各分位数范围内的覆盖率。

    参数:
        x_ref: 参考分布样本（真实数据），1D array
        x_test: 测试分布样本（训练集），1D array
        quantiles: 要计算的分位点列表，默认 [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    返回:
        dict: 每个分位点 {q: (ref_value, coverage_ratio)}
        覆盖率 = 训练集样本落在 [ref_min, ref_q] 或 [ref_q, ref_max] 内的比例
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    ref_min = float(np.min(x_ref))
    ref_max = float(np.max(x_ref))
    ref_values = np.percentile(x_ref, [q * 100 for q in quantiles])

    coverage = {}
    for q, ref_val in zip(quantiles, ref_values):
        # 覆盖率：训练集样本落在 [ref_min, ref_val] 区间内的比例（下尾）
        mask_lower = (x_test >= ref_min) & (x_test <= ref_val)
        cov_lower = float(np.mean(mask_lower))

        # 覆盖率：训练集样本落在 [ref_val, ref_max] 区间内的比例（上尾）
        mask_upper = (x_test >= ref_val) & (x_test <= ref_max)
        cov_upper = float(np.mean(mask_upper))

        coverage[q] = {
            "ref_value": float(ref_val),
            "coverage_lower": cov_lower,
            "coverage_upper": cov_upper,
        }

    # 整体覆盖率：训练集样本落在真实数据 [P2.5, P97.5] 范围内的比例
    ref_p025, ref_p975 = np.percentile(x_ref, [2.5, 97.5])
    mask_main = (x_test >= ref_p025) & (x_test <= ref_p975)
    coverage["overall"] = {
        "ref_p025": float(ref_p025),
        "ref_p975": float(ref_p975),
        "coverage_main": float(np.mean(mask_main)),
    }

    return coverage


def compute_ood_ratio(x_ref: np.ndarray, x_test: np.ndarray,
                      threshold_percentile: float = 99.0) -> dict:
    """
    计算训练集中超出真实数据分布范围的样本比例（OOD 检测）。

    参数:
        x_ref: 参考分布样本（真实数据），1D array
        x_test: 测试分布样本（训练集），1D array
        threshold_percentile: 真实数据的阈值百分位，默认 99

    返回:
        dict: 包含 OOD 比例、超出方向等统计信息
    """
    ref_min = float(np.min(x_ref))
    ref_max = float(np.max(x_ref))
    ref_p01, ref_p99 = np.percentile(x_ref, [1, threshold_percentile])

    # OOD：训练集样本超出真实数据 [P1, P99] 范围的比例
    ood_below = float(np.mean(x_test < ref_p01))
    ood_above = float(np.mean(x_test > ref_p99))
    ood_total = ood_below + ood_above

    # OOD 样本的极端程度
    ood_samples = x_test[(x_test < ref_p01) | (x_test > ref_p99)]
    if len(ood_samples) > 0:
        max_ood_below = float(np.min(x_test[x_test < ref_p01])) if np.any(x_test < ref_p01) else None
        max_ood_above = float(np.max(x_test[x_test > ref_p99])) if np.any(x_test > ref_p99) else None
    else:
        max_ood_below = None
        max_ood_above = None

    return {
        "ref_min": ref_min,
        "ref_max": ref_max,
        "ref_p01": ref_p01,
        "ref_p99": ref_p99,
        "ood_below_pct": ood_below,
        "ood_above_pct": ood_above,
        "ood_total_pct": ood_total,
        "max_ood_below": max_ood_below,
        "max_ood_above": max_ood_above,
        "n_ood": len(ood_samples),
        "n_test": len(x_test),
    }
