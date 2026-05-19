"""IO helpers for F30 waveform analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import WaveformRecord


def load_waveform_record(path: str | Path) -> WaveformRecord:
    """Load one NPZ waveform sample."""
    sample_path = Path(path)
    with np.load(sample_path, allow_pickle=True) as data:
        signal_values = np.asarray(data["phase_data"], dtype=float)
        sample_rate = float(np.asarray(data["sample_rate"]).item())
        metadata = {key: str(np.asarray(data[key]).item()) for key in ("starttime", "arrival_time") if key in data}
        if "timestamp" in data:
            metadata["timestamp"] = float(np.asarray(data["timestamp"]).item())
    return WaveformRecord(
        sample_id=sample_path.stem,
        sample_name=sample_path.name,
        path=sample_path,
        signal=signal_values,
        sample_rate=sample_rate,
        metadata=metadata,
    )


def iter_npz_files(root_dir: str | Path) -> list[Path]:
    """Return sorted NPZ files under a directory."""
    root = Path(root_dir)
    return sorted(root.rglob("*.npz"))
