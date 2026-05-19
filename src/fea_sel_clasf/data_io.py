"""Input loading and normalization helpers for feature tables."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .types import LoadedFeatureTable


DEFAULT_META_COLUMNS = (
    "sample_id",
    "label",
    "sample_name",
    "sample_rate",
    "folder_name",
    "timestamp",
    "path",
    "sample_type",
    "sample_type_code",
    "split",
    "domain",
    "base_domain",
)

_TRAILING_SPLIT_RE = re.compile(r"_(tr|te|train|test)$", re.IGNORECASE)


def _normalize_label_value(value: object) -> int:
    """Map common binary label encodings to 0/1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("label column contains missing values")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "pos", "positive", "broken_wire", "brokenwire", "bk", "defect", "fault", "断丝"}:
        return 1
    if text in {"0", "neg", "negative", "noise", "normal", "flow_noise", "流噪声"}:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"unsupported label value: {value!r}") from exc


def _normalize_text(value: object) -> str:
    """Return a stable uppercase domain token from a folder or file name."""
    text = str(value).strip()
    text = Path(text).stem
    text = text.replace("\\", "/").split("/")[-1]
    text = _TRAILING_SPLIT_RE.sub("", text)
    text = re.sub(r"^BK14_", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^BK14-", "", text, flags=re.IGNORECASE)
    return text.upper()


def infer_split_from_row(row: pd.Series) -> str:
    """Infer train/test split from explicit columns or folder names."""
    if "split" in row and pd.notna(row["split"]):
        text = str(row["split"]).strip().lower()
        if text in {"train", "tr", "training"}:
            return "train"
        if text in {"test", "te", "testing"}:
            return "test"
    folder_value = row.get("folder_name", row.get("sample_name", ""))
    folder_text = str(folder_value).strip().lower()
    if folder_text.endswith(("_tr", "_train")):
        return "train"
    if folder_text.endswith(("_te", "_test")):
        return "test"
    return "unknown"


def infer_base_domain(row: pd.Series) -> str:
    """Infer the paired physical domain from a row."""
    folder_value = row.get("folder_name", row.get("sample_name", ""))
    folder_text = _normalize_text(folder_value)
    folder_text = _TRAILING_SPLIT_RE.sub("", folder_text)
    return folder_text


def load_feature_table(csv_path: str | Path, label_column: str = "label") -> LoadedFeatureTable:
    """
    Load a feature table exported from the feature-extraction stage.

    The loader keeps all columns, but it also infers a canonical ``base_domain``
    and ``split`` column so that cross-condition experiments can be grouped
    reliably even when the input CSV only stores ``folder_name``.
    """
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if label_column not in frame.columns:
        raise ValueError(f"missing required label column: {label_column}")

    normalized = frame.copy()
    normalized[label_column] = normalized[label_column].map(_normalize_label_value).astype(int)
    normalized["split"] = normalized.apply(infer_split_from_row, axis=1)
    normalized["base_domain"] = normalized.apply(infer_base_domain, axis=1)

    meta_columns = tuple(col for col in DEFAULT_META_COLUMNS if col in normalized.columns or col in {"split", "base_domain"})
    feature_columns = tuple(
        col
        for col in normalized.columns
        if col not in set(DEFAULT_META_COLUMNS) and col not in {"split", "base_domain"}
    )
    if not feature_columns:
        raise ValueError("no feature columns found in input table")

    return LoadedFeatureTable(
        frame=normalized,
        feature_columns=feature_columns,
        meta_columns=meta_columns,
        label_column=label_column,
        split_column="split",
        domain_column="base_domain",
    )
