"""Manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .types import DatasetManifestRow


def build_manifest_dataframe(rows: list[DatasetManifestRow]) -> pd.DataFrame:
    """Convert manifest rows into a dataframe."""
    return pd.DataFrame([row.__dict__ for row in rows])


def write_manifest(rows: list[DatasetManifestRow], output_path: Path) -> pd.DataFrame:
    """Persist a manifest dataframe."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_manifest_dataframe(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

