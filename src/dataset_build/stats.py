"""Dataset summary helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def summarize_directory(root: Path) -> pd.DataFrame:
    """Summarize file counts for one output tree."""
    rows = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = len(list(subdir.rglob("*.npz")))
        rows.append({"dataset_name": subdir.name, "sample_count": count})
    return pd.DataFrame(rows)


def summarize_log(log_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main log by dataset and split."""
    if log_df.empty:
        return pd.DataFrame(columns=["dataset_name", "split", "sample_count"])
    return (
        log_df.groupby(["dataset_name", "split"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values(["dataset_name", "split"])
        .reset_index(drop=True)
    )

