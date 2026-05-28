"""训练集代表性评估：基于真实数据参考分布的特征分布对比分析"""

from dist_comparison.metrics import (
    compute_mmd,
    compute_wasserstein,
    compute_coverage,
    compute_ood_ratio,
)

__all__ = [
    "compute_mmd",
    "compute_wasserstein",
    "compute_coverage",
    "compute_ood_ratio",
]
