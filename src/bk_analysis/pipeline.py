"""High-level analysis pipeline for BK14 generation-method comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import DEFAULT_BANDS_HZ, compute_spectral_features, compute_time_features
from .io import build_record_index, load_generation_method_records


@dataclass(frozen=True)
class CurveSummary:
    """Percentile summary for a family of aligned curves."""

    x: np.ndarray
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    mean: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the summary as plain arrays for notebook-side plotting or export."""
        return {
            "x": self.x,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Bundle of computed tables and arrays produced by the analysis pipeline."""

    record_index: pd.DataFrame
    time_features: pd.DataFrame
    method_summary: pd.DataFrame
    spectral_records: pd.DataFrame
    psd_summaries: dict[str, CurveSummary]
    energy_summaries: dict[str, CurveSummary]


def _build_common_frequency_grid(records: pd.DataFrame, n_points: int = 400) -> np.ndarray:
    """Use the minimum Nyquist frequency to define a shared comparison grid."""
    max_frequency_hz = 0.5 * float(records["sample_rate_hz"].min())
    return np.linspace(0.0, max_frequency_hz, n_points)


def _align_curves(frequencies: list[np.ndarray], values: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Interpolate a list of curves onto the same x-axis."""
    aligned = []
    for frequency_hz, curve in zip(frequencies, values):
        aligned.append(np.interp(grid, frequency_hz, curve))
    return np.vstack(aligned)


def _summarize_curve_family(curves: np.ndarray, x_grid: np.ndarray) -> CurveSummary:
    """Create median and percentile bands for an empirical curve family."""
    return CurveSummary(
        x=x_grid,
        median=np.median(curves, axis=0),
        p10=np.percentile(curves, 10, axis=0),
        p90=np.percentile(curves, 90, axis=0),
        mean=np.mean(curves, axis=0),
    )


def _build_method_summary(
    time_features: pd.DataFrame,
    time_feature_bands_hz: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    """Create descriptive statistics for each generation method."""
    summary_frames = []
    value_columns = ["duration_s", "start_energy_0p01s"] + [f"ptp_band_{low}_{high}_hz" for low, high in time_feature_bands_hz]
    for method, group in time_features.groupby("method"):
        stats = group[value_columns].agg(["mean", "median", "std", "min", "max"]).T
        stats.insert(0, "method", method)
        stats.insert(1, "feature", stats.index)
        summary_frames.append(stats.reset_index(drop=True))
    return pd.concat(summary_frames, ignore_index=True)


def summarize_curves_to_dataframes(summaries: dict[str, CurveSummary], x_name: str, prefix: str) -> dict[str, pd.DataFrame]:
    """Convert curve summaries into export-friendly data frames."""
    output: dict[str, pd.DataFrame] = {}
    for method, summary in summaries.items():
        output[method] = pd.DataFrame(
            {
                x_name: summary.x,
                f"median_{prefix}": summary.median,
                f"p10_{prefix}": summary.p10,
                f"p90_{prefix}": summary.p90,
                f"mean_{prefix}": summary.mean,
            }
        )
    return output


def run_generation_method_analysis(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    time_feature_bands_hz: tuple[tuple[int, int], ...] = DEFAULT_BANDS_HZ,
    ptp_highpass_cutoff_hz: float = 1_000.0,
    psd_segment_duration_s: float = 0.02,
) -> AnalysisResult:
    """
    Run the full comparison workflow for corrosion vs cut broken-wire signals.

    Features are computed on the original sample rate of each record.
    Time-domain band features use ``time_feature_bands_hz``.
    Peak-to-peak computation first applies a high-pass filter at ``ptp_highpass_cutoff_hz``.
    PSD is computed in a fixed post-arrival window of ``psd_segment_duration_s`` seconds.
    When ``output_dir`` is provided, computed tables are persisted there.
    """
    data_dir = Path(data_dir)
    records = load_generation_method_records(data_dir)
    record_index = build_record_index(records)

    time_rows = []
    spectral_rows: list[dict[str, object]] = []
    for record in records:
        time_result = compute_time_features(
            record,
            bands_hz=time_feature_bands_hz,
            ptp_highpass_cutoff_hz=ptp_highpass_cutoff_hz,
        )
        spectral_result = compute_spectral_features(
            record,
            psd_segment_duration_s=psd_segment_duration_s,
        )

        row = {
            "method": record.method,
            "path": str(record.path),
            "sample_rate_hz": record.sample_rate,
            "arrival_offset_s": record.arrival_offset_seconds,
            "duration_s": time_result.duration_s,
            "start_energy_0p01s": time_result.start_energy_0p01s,
            "end_index": time_result.end_index,
        }
        for (low_hz, high_hz), value in time_result.band_peak_to_peak.items():
            row[f"ptp_band_{low_hz}_{high_hz}_hz"] = value
        time_rows.append(row)

        spectral_rows.append(
            {
                "method": record.method,
                "path": str(record.path),
                "frequency_hz": spectral_result.frequency_hz,
                "psd": spectral_result.psd,
                "cumulative_energy_ratio": spectral_result.cumulative_energy_ratio,
            }
        )

    time_features = pd.DataFrame(time_rows)
    spectral_records = pd.DataFrame(spectral_rows)
    method_summary = _build_method_summary(time_features, time_feature_bands_hz=time_feature_bands_hz)

    common_frequency_hz = _build_common_frequency_grid(record_index)
    psd_summaries: dict[str, CurveSummary] = {}
    energy_summaries: dict[str, CurveSummary] = {}
    for method, group in spectral_records.groupby("method"):
        psd_curves = _align_curves(group["frequency_hz"].tolist(), group["psd"].tolist(), common_frequency_hz)
        energy_curves = _align_curves(
            group["frequency_hz"].tolist(),
            group["cumulative_energy_ratio"].tolist(),
            common_frequency_hz,
        )
        psd_summaries[method] = _summarize_curve_family(psd_curves, common_frequency_hz)
        energy_summaries[method] = _summarize_curve_family(energy_curves, common_frequency_hz)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        record_index.to_csv(output_path / "record_index.csv", index=False)
        time_features.to_csv(output_path / "time_features.csv", index=False)
        method_summary.to_csv(output_path / "method_summary.csv", index=False)

        psd_summary_frames = summarize_curves_to_dataframes(psd_summaries, x_name="frequency_hz", prefix="psd")
        for method, frame in psd_summary_frames.items():
            frame.to_csv(output_path / f"psd_summary_{method}.csv", index=False)

        energy_summary_frames = summarize_curves_to_dataframes(energy_summaries, x_name="frequency_hz", prefix="ratio")
        for method, frame in energy_summary_frames.items():
            frame.to_csv(output_path / f"cumulative_energy_summary_{method}.csv", index=False)

    return AnalysisResult(
        record_index=record_index,
        time_features=time_features,
        method_summary=method_summary,
        spectral_records=spectral_records,
        psd_summaries=psd_summaries,
        energy_summaries=energy_summaries,
    )
