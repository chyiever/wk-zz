"""Reusable signal feature computation package."""

from .base import FeatureContext, FeatureParams, FeatureRecord, FeatureResult
from .gpu_backend import gpu_backend_info
from .importance import rank_feature_importance
from .pipeline import build_feature_dataset, compute_feature_table_for_paths, validate_feature_table

__all__ = [
    "FeatureContext",
    "FeatureParams",
    "FeatureRecord",
    "FeatureResult",
    "build_feature_dataset",
    "compute_feature_table_for_paths",
    "gpu_backend_info",
    "rank_feature_importance",
    "validate_feature_table",
]
