"""Train/test splitting helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import DatasetManifestRow, RawSignalRecord, SplitAssignment


def split_bk14_records(records: list[RawSignalRecord], train_ratio: float, seed: int) -> tuple[list[RawSignalRecord], list[RawSignalRecord]]:
    """Split BK14 samples stratified by generation method."""
    rng = np.random.default_rng(seed)
    train_records: list[RawSignalRecord] = []
    test_records: list[RawSignalRecord] = []
    by_method: dict[str, list[RawSignalRecord]] = defaultdict(list)
    for record in records:
        by_method[str(record.method or "unknown")].append(record)
    for method, method_records in by_method.items():
        indices = np.arange(len(method_records))
        rng.shuffle(indices)
        n_train = int(round(len(method_records) * train_ratio))
        n_train = min(max(n_train, 1), len(method_records) - 1)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        train_records.extend([method_records[i] for i in train_idx])
        test_records.extend([method_records[i] for i in test_idx])
    train_records.sort(key=lambda record: record.path.name)
    test_records.sort(key=lambda record: record.path.name)
    return train_records, test_records


def _make_flow_groups(records: list[RawSignalRecord], group_size: int) -> list[list[RawSignalRecord]]:
    sorted_records = sorted(
        records,
        key=lambda record: (float(record.sample_rate), str(record.starttime), record.path.name),
    )
    groups: list[list[RawSignalRecord]] = []
    for start in range(0, len(sorted_records), group_size):
        groups.append(sorted_records[start : start + group_size])
    return groups


def split_flow_records(
    records: list[RawSignalRecord],
    dataset_name: str,
    train_ratio: float,
    seed: int,
    group_size: int,
) -> tuple[list[RawSignalRecord], list[RawSignalRecord], list[SplitAssignment], list[DatasetManifestRow]]:
    """Split flow-noise records by grouped time windows."""
    rng = np.random.default_rng(seed)
    train_records: list[RawSignalRecord] = []
    test_records: list[RawSignalRecord] = []
    assignments: list[SplitAssignment] = []
    manifest_rows: list[DatasetManifestRow] = []

    rate_groups: dict[float, list[RawSignalRecord]] = defaultdict(list)
    for record in records:
        rate_groups[float(record.sample_rate)].append(record)

    for rate in sorted(rate_groups):
        groups = _make_flow_groups(rate_groups[rate], group_size=group_size)
        order = np.arange(len(groups))
        rng.shuffle(order)
        n_train_groups = int(round(len(groups) * train_ratio))
        n_train_groups = min(max(n_train_groups, 1 if len(groups) > 1 else len(groups)), len(groups))
        train_group_ids = set(order[:n_train_groups].tolist())
        for group_index, group in enumerate(groups):
            split = "train" if group_index in train_group_ids else "test"
            target_bucket = train_records if split == "train" else test_records
            target_bucket.extend(group)
            group_id = f"{dataset_name}_sr{int(rate/1000):03d}_g{group_index:03d}"
            for record in group:
                assignments.append(
                    SplitAssignment(
                        source_path=record.path,
                        dataset_name=dataset_name,
                        split=split,
                        group_id=group_id,
                        sample_rate_hz=float(record.sample_rate),
                        timestamp_text=str(record.starttime),
                        method=record.method,
                    )
                )
                manifest_rows.append(
                    DatasetManifestRow(
                        source_file_name=record.path.name,
                        source_file_path=str(record.path),
                        dataset_source=dataset_name,
                        group_id=group_id,
                        split=split,
                        sample_rate_hz=float(record.sample_rate),
                        timestamp_text=str(record.starttime),
                        method=record.method,
                        random_seed=seed,
                    )
                )

    train_records.sort(key=lambda record: record.path.name)
    test_records.sort(key=lambda record: record.path.name)
    return train_records, test_records, assignments, manifest_rows

