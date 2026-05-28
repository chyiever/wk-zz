"""训练集代表性评估：基于真实数据参考分布的特征分布对比分析"""

from dist_comparison.metrics import (
    compute_mmd,
    compute_wasserstein,
    compute_coverage,
    compute_ood_ratio,
)
from dist_comparison.visualization import (
    plot_coverage_comparison,
    plot_distribution_overlap,
    plot_ood_detection,
    plot_representativeness_summary,
)

__all__ = [
    "compute_mmd",
    "compute_wasserstein",
    "compute_coverage",
    "compute_ood_ratio",
    "plot_coverage_comparison",
    "plot_distribution_overlap",
    "plot_ood_detection",
    "plot_representativeness_summary",
]
