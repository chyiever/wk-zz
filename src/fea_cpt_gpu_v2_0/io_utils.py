"""Input helpers for reusable feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import FeatureRecord


def infer_sample_type_code(sample_type: str) -> int:
    """Map sample type text to a stable integer code."""
    normalized = sample_type.strip().upper()
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(normalized)) % 10_000


def load_feature_record(path: str | Path) -> FeatureRecord:
    """Load one npz sample into a generic record."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        signal_values = np.asarray(data["phase_data"], dtype=float)
        sample_rate = float(data["sample_rate"])
        sample_type = str(data.get("type", path.parent.name).tolist())
        sample_name = path.name
        sample_id = path.stem
        metadata = {
            "starttime": str(data.get("starttime", "").tolist()),
            "arrival_time": str(data.get("arrival_time", "").tolist()),
            "timestamp": float(data.get("timestamp", np.nan).tolist()) if "timestamp" in data else np.nan,
        }
    return FeatureRecord(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_type=sample_type,
        sample_type_code=infer_sample_type_code(sample_type),
        path=path,
        signal=signal_values,
        sample_rate=sample_rate,
        metadata=metadata,
    )


def iter_npz_files(root_dir: str | Path) -> list[Path]:
    """Return all npz files under a directory tree."""
    root_dir = Path(root_dir)
    return sorted(root_dir.rglob("*.npz"))
