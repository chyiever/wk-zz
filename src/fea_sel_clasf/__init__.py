"""Cross-condition broken-wire feature selection and classification package."""

from .data_io import load_feature_table
from .evaluator import evaluate_binary_predictions
from .feature_filter import run_stage_a_filter, slice_stage_a_scores
from .feature_wrapper import run_rfecv_selection
from .model_zoo import search_best_model
from .runner import run_cross_condition_experiments
from .split_protocol import build_experiment_specs
from .thresholding import choose_threshold_from_oof

__all__ = [
    "build_experiment_specs",
    "choose_threshold_from_oof",
    "evaluate_binary_predictions",
    "load_feature_table",
    "run_cross_condition_experiments",
    "run_rfecv_selection",
    "run_stage_a_filter",
    "slice_stage_a_scores",
    "search_best_model",
]
