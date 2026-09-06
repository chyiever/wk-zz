"""Reusable signal feature computation package (v2.2: Persistent Executor + SharedMemory + Centralized GPU STFT + Pipeline)."""

from .base import FeatureContext, FeatureParams, FeatureRecord, FeatureResult
from .gpu_backend import gpu_backend_info
from .importance import rank_feature_importance
from .pipeline import build_feature_dataset, compute_feature_table_for_paths, validate_feature_table
from .sliding_window import (
    SlidingWindowConfig,
    build_sliding_window_dataset,
    compute_all_features_for_window,
    compute_shared_stft,
    discover_source_files,
    downsample_source_files,
    list_window_ranges,
    load_source_file,
    process_source_file,
    upsample_to_target,
)

__all__ = [
    "FeatureContext",
    "FeatureParams",
    "FeatureRecord",
    "FeatureResult",
    "SlidingWindowConfig",
    "build_feature_dataset",
    "build_sliding_window_dataset",
    "compute_all_features_for_window",
    "compute_shared_stft",
    "compute_feature_table_for_paths",
    "discover_source_files",
    "downsample_source_files",
    "gpu_backend_info",
    "list_window_ranges",
    "load_source_file",
    "process_source_file",
    "rank_feature_importance",
    "upsample_to_target",
    "validate_feature_table",
]
