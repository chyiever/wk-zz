"""Load BK14 and flow-noise NPZ samples into common records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from bk_analysis.io import iter_generation_method_files, load_signal_record as load_bk14_signal_record
from f30_fea.io_utils import load_waveform_record

from .resample import resample_to_rate
from .types import RawSignalRecord
from .windowing import parse_starttime_text

_FLOW_FILENAME_RE = re.compile(r"^(?P<prefix>.+?)-FIP-(?P<rate>\d+)K-(?P<stamp>\d{8}T\d{6}\.\d+)\.npz$", re.IGNORECASE)


def _extract_timestamp_text(path: Path) -> str:
    match = _FLOW_FILENAME_RE.match(path.name)
    if match:
        return match.group("stamp")
    return path.stem


def _load_npz_data_info(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "data_info" in data:
            info = data["data_info"].item()
            return dict(info) if isinstance(info, dict) else {"data_info": info}
    return {}


def load_bk14_records(data_root: str | Path) -> list[RawSignalRecord]:
    """Load the 24 BK14 broken-wire samples."""
    data_root = Path(data_root)
    records = []
    for method, path in iter_generation_method_files(data_root):
        bk_record = load_bk14_signal_record(path=path, method=method)
        timestamp_value = None
        try:
            timestamp_value = parse_starttime_text(bk_record.starttime).timestamp()
        except ValueError:
            timestamp_value = None
        records.append(
            RawSignalRecord(
                dataset_name="BK14",
                path=path,
                signal=np.asarray(bk_record.signal, dtype=float),
                sample_rate=float(bk_record.sample_rate),
                starttime=str(bk_record.starttime),
                arrival_time=str(bk_record.arrival_time),
                timestamp=timestamp_value,
                sample_type=str(bk_record.sample_type),
                method=str(bk_record.method),
                arrival_source=str(bk_record.arrival_source),
                original_sample_rate=float(bk_record.original_sample_rate),
                original_npts=int(len(bk_record.signal)),
                data_info=_load_npz_data_info(path),
            )
        )
    return records


def load_flow_records(data_root: str | Path, dataset_name: str) -> list[RawSignalRecord]:
    """Load one flow-noise dataset into common records."""
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name
    records: list[RawSignalRecord] = []
    for path in sorted(dataset_dir.glob("*.npz")):
        waveform = load_waveform_record(path)
        metadata = dict(waveform.metadata)
        starttime = str(metadata.get("starttime", path.stem))
        arrival_time = str(metadata.get("arrival_time", ""))
        timestamp = float(metadata["timestamp"]) if "timestamp" in metadata else None
        records.append(
            RawSignalRecord(
                dataset_name=dataset_name,
                path=path,
                signal=np.asarray(waveform.signal, dtype=float),
                sample_rate=float(waveform.sample_rate),
                starttime=starttime,
                arrival_time=arrival_time,
                timestamp=timestamp,
                sample_type=dataset_name,
                method=None,
                arrival_source="estimated",
                original_sample_rate=float(waveform.sample_rate),
                original_npts=int(len(waveform.signal)),
                data_info=_load_npz_data_info(path),
            )
        )
    return records


def normalize_record(record: RawSignalRecord, target_sample_rate_hz: float) -> tuple[RawSignalRecord, str]:
    """Normalize a record to the target sample rate."""
    resampled_signal, action = resample_to_rate(record.signal, record.sample_rate, target_sample_rate_hz)
    return (
        RawSignalRecord(
            dataset_name=record.dataset_name,
            path=record.path,
            signal=resampled_signal,
            sample_rate=float(target_sample_rate_hz),
            starttime=record.starttime,
            arrival_time=record.arrival_time,
            timestamp=record.timestamp,
            sample_type=record.sample_type,
            method=record.method,
            arrival_source=record.arrival_source,
            original_sample_rate=record.original_sample_rate or record.sample_rate,
            original_npts=record.original_npts or len(record.signal),
            data_info=dict(record.data_info),
        ),
        action,
    )


def load_dataset_records(data_root: str | Path, dataset_name: str) -> list[RawSignalRecord]:
    """Load one dataset by name."""
    if dataset_name == "BK14":
        return load_bk14_records(data_root)
    return load_flow_records(data_root, dataset_name)
