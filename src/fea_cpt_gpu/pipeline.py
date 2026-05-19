"""Batch pipeline for reusable feature computation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .base import FeatureParams
from .features import compute_all_features
from .io_utils import iter_npz_files, load_feature_record
from .logging_utils import setup_logger
from .signal_ops import build_context


def _compute_one(path: Path, params: FeatureParams, log_dir: Path | None) -> dict[str, float | int | str]:
    logger = setup_logger("fea_cpt.compute", log_dir=log_dir)
    logger.info("Start feature computation for %s", path.name)
    record = load_feature_record(path)
    context = build_context(record, params)
    result = compute_all_features(context)
    row: dict[str, float | int | str] = {
        "sample_id": result.sample_id,
        "sample_name": result.sample_name,
        "sample_type": result.sample_type,
        "sample_type_code": result.sample_type_code,
        "path": str(path),
    }
    row.update(result.features)
    logger.info("Finished feature computation for %s with %d features", path.name, len(result.features))
    return row


def compute_feature_table_for_paths(paths: Sequence[str | Path], params: FeatureParams | None = None, show_progress: bool = True, log_dir: str | Path | None = None) -> pd.DataFrame:
    """Compute the full feature table for an explicit file list."""
    params = params or FeatureParams()
    path_list = [Path(path) for path in paths]
    logger = setup_logger("fea_cpt.pipeline", log_dir=log_dir)
    logger.info("Batch computation started for %d files", len(path_list))
    iterator = tqdm(path_list, desc="计算特征", disable=not show_progress)
    if params.n_jobs == 1:
        rows = [
            _compute_one(path, params, Path(log_dir) if log_dir is not None else None)
            for path in iterator
        ]
    else:
        rows = Parallel(n_jobs=params.n_jobs, prefer='threads')(
            delayed(_compute_one)(path, params, Path(log_dir) if log_dir is not None else None)
            for path in iterator
        )
    frame = pd.DataFrame(rows).sort_values(["sample_type_code", "sample_id"]).reset_index(drop=True)
    logger.info("Batch computation finished, resulting table shape=%s", frame.shape)
    return frame


def build_feature_dataset(data_root: str | Path, params: FeatureParams | None = None, output_dir: str | Path | None = None, show_progress: bool = True) -> pd.DataFrame:
    """Compute all features for all npz files under a root directory and optionally save outputs."""
    params = params or FeatureParams()
    data_root = Path(data_root)
    paths = iter_npz_files(data_root)
    output_path = Path(output_dir) if output_dir is not None else None
    log_dir = output_path / "logs" if output_path is not None else None
    frame = compute_feature_table_for_paths(paths, params=params, show_progress=show_progress, log_dir=log_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path / "feature_table.csv", index=False, encoding="utf-8-sig")
        feature_columns = [col for col in frame.columns if col not in {"sample_id", "sample_name", "sample_type", "sample_type_code", "path"}]
        pd.DataFrame(
            {
                "feature_name": feature_columns,
                "feature_index": np.arange(len(feature_columns), dtype=int),
            }
        ).to_csv(output_path / "feature_vector_mapping.csv", index=False, encoding="utf-8-sig")
    return frame


def validate_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Quick numerical range validation for computed features."""
    meta_cols = {"sample_id", "sample_name", "sample_type", "sample_type_code", "path"}
    feature_cols = [col for col in frame.columns if col not in meta_cols]
    rows = []
    for column in feature_cols:
        series = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(series)
        values = series[finite]
        rows.append(
            {
                "feature_name": column,
                "non_null_count": int(values.shape[0]),
                "min": float(values.min()) if not values.empty else np.nan,
                "max": float(values.max()) if not values.empty else np.nan,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "std": float(values.std()) if not values.empty else np.nan,
                "abs_max_log10": float(np.log10(max(np.max(np.abs(values)), 1e-12))) if not values.empty else np.nan,
                "has_nan": bool(series.isna().any()),
                "has_inf": bool((~np.isfinite(series.fillna(0.0))).any()),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
