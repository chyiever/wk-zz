"""Logging helpers for dataset building."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import BuildLogRow, DatasetManifestRow


def build_logger(log_file: Path) -> logging.Logger:
    """Create a file-backed logger."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dataset_build::{log_file}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def write_log_csv(rows: list[BuildLogRow], output_path: Path) -> pd.DataFrame:
    """Write the main build log."""
    df = pd.DataFrame([row.__dict__ for row in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def write_manifest_csv(rows: list[DatasetManifestRow], output_path: Path) -> pd.DataFrame:
    """Write a split manifest."""
    df = pd.DataFrame([row.__dict__ for row in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

