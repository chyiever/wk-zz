"""Dataset build pipeline."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BuildConfig, DEFAULT_BUILD_CONFIG
from .loaders import load_dataset_records, normalize_record
from .logging_utils import build_logger, write_log_csv
from .manifests import write_manifest
from .mixers import center_signal, compute_quality_rms, mix_signals
from .splitters import split_bk14_records, split_flow_records
from .types import BuildLogRow, DatasetManifestRow, MixedSampleSpec, RawSignalRecord
from .windowing import build_t_candidates, format_starttime_text, window_to_datetime
from .writers import (
    build_data_info,
    build_short_suffix,
    copy_raw_sample,
    make_output_file_name,
    write_npz,
)


@dataclass(frozen=True)
class BuildResult:
    """Summary of one completed build run."""

    config: BuildConfig
    output_root: Path
    log_df: pd.DataFrame
    manifest_dfs: dict[str, pd.DataFrame]
    summary_df: pd.DataFrame
    report_path: Path


def _ensure_root_structure(output_root: Path) -> dict[str, Path]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    manifests_dir = output_root / "manifests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return {"logs": logs_dir, "manifests": manifests_dir}


def _copy_selected_records(records: list[RawSignalRecord], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        copy_raw_sample(record.path, target_dir / record.path.name)


def _normalize_record(record: RawSignalRecord, target_sample_rate_hz: float) -> tuple[RawSignalRecord, str]:
    normalized, action = normalize_record(record, target_sample_rate_hz)
    return normalized, action


def _build_mixed_sample(
    spec: MixedSampleSpec,
    target_dir: Path,
    config: BuildConfig,
    rng: np.random.Generator,
) -> BuildLogRow:
    target_sample_rate_hz = float(config.target_sample_rate_hz)
    t_candidates = build_t_candidates(config.t_min_s, config.t_max_s, config.t_step_s)
    target_npts = int(round(config.window_duration_s * target_sample_rate_hz))

    noise_norm, noise_action = _normalize_record(spec.source_noise, target_sample_rate_hz)
    bk_norm, bk_action = _normalize_record(spec.source_bk, target_sample_rate_hz)
    arrival_offset_s = float(bk_norm.arrival_offset_s)
    actual_t_s = float(spec.t_s)
    start_offset_s = float(arrival_offset_s + actual_t_s)
    signal_duration_s = float(len(bk_norm.signal) / target_sample_rate_hz)
    if start_offset_s < 0 or start_offset_s + config.window_duration_s > signal_duration_s:
        start_offset_s, actual_t_s, start_idx = pick_window_start(
            arrival_offset_s=arrival_offset_s,
            window_duration_s=config.window_duration_s,
            signal_duration_s=signal_duration_s,
            sample_rate_hz=target_sample_rate_hz,
            rng=rng,
            t_candidates=t_candidates,
        )
    else:
        start_idx = int(round(start_offset_s * target_sample_rate_hz))
    end_idx = start_idx + target_npts
    if end_idx > len(bk_norm.signal):
        raise ValueError("window extends beyond broken-wire sample")

    output_file_name = make_output_file_name(spec.source_bk, spec.noise_dataset, spec.w1, spec.w2, actual_t_s, spec.suffix)
    output_path = target_dir / output_file_name

    bk_window = np.asarray(bk_norm.signal[start_idx:end_idx], dtype=float)
    noise_window = np.asarray(noise_norm.signal[:target_npts], dtype=float)
    if len(noise_window) != len(bk_window):
        raise ValueError("window length mismatch")

    noise_centered = center_signal(noise_window)
    bk_centered = center_signal(bk_window)
    mixed_signal = mix_signals(noise_centered, bk_centered, spec.w1, spec.w2)
    noise_rms, bk_rms, mixed_rms = compute_quality_rms(noise_centered, bk_centered, mixed_signal, target_sample_rate_hz)

    output_datetime = window_to_datetime(bk_norm.starttime, start_offset_s)
    output_starttime = format_starttime_text(output_datetime)
    output_timestamp = float(output_datetime.timestamp())
    data_info = build_data_info(
        spec=spec,
        output_starttime=output_starttime,
        output_timestamp=output_timestamp,
        noise_rms=noise_rms,
        bk_rms=bk_rms,
        mixed_rms=mixed_rms,
        resample_noise_action=noise_action,
        resample_bk_action=bk_action,
        output_file_name=output_file_name,
        random_seed=int(config.random_seed),
        target_sample_rate_hz=target_sample_rate_hz,
        window_start_offset_t_s=actual_t_s,
    )
    write_npz(
        output_path=output_path,
        signal_values=mixed_signal,
        sample_rate_hz=target_sample_rate_hz,
        starttime=output_starttime,
        arrival_time=bk_norm.arrival_time,
        timestamp=output_timestamp,
        sample_type="BK14",
        data_info=data_info,
    )
    return BuildLogRow(
        record_id=output_file_name,
        dataset_name=spec.dataset_name,
        split=spec.split,
        sample_class="broken_wire",
        output_file_name=output_file_name,
        output_file_path=str(output_path),
        source_noise_file_name=spec.source_noise.sample_name,
        source_noise_file_path=str(spec.source_noise.path),
        source_bk_file_name=spec.source_bk.sample_name,
        source_bk_file_path=str(spec.source_bk.path),
        source_noise_dataset=spec.noise_dataset,
        source_bk_method=str(spec.source_bk.method or ""),
        source_noise_sample_rate_hz=float(noise_norm.sample_rate),
        source_bk_sample_rate_hz=float(bk_norm.sample_rate),
        target_sample_rate_hz=target_sample_rate_hz,
        source_noise_duration_s=float(len(noise_norm.signal) / target_sample_rate_hz),
        source_bk_duration_s=float(len(bk_norm.signal) / target_sample_rate_hz),
        output_duration_s=float(len(mixed_signal) / target_sample_rate_hz),
        arrival_offset_s=arrival_offset_s,
        window_start_offset_t_s=float(actual_t_s),
        window_start_absolute_offset_s=start_offset_s,
        window_start_index=start_idx,
        window_end_index=end_idx,
        w1=float(spec.w1),
        w2=float(spec.w2),
        noise_rms=float(noise_rms),
        bk_rms=float(bk_rms),
        mixed_rms=float(mixed_rms),
        resample_noise_action=noise_action,
        resample_bk_action=bk_action,
        random_seed=int(config.random_seed),
        build_time=datetime.now().isoformat(timespec="seconds"),
        builder_version="2026-05-05",
        status="success",
        error_message="",
    )


def _build_mixed_dataset(
    noise_dataset: str,
    split: str,
    noise_records: list[RawSignalRecord],
    bk_records: list[RawSignalRecord],
    config: BuildConfig,
    rng: np.random.Generator,
) -> tuple[list[BuildLogRow], pd.DataFrame]:
    dataset_name = f"BK14_{noise_dataset.lower()}_{split}"
    target_dir = config.output_root / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BuildLogRow] = []
    for _ in range(len(noise_records)):
        source_noise = noise_records[int(rng.integers(0, len(noise_records)))]
        source_bk = bk_records[int(rng.integers(0, len(bk_records)))]
        t_s = float(rng.choice(build_t_candidates(config.t_min_s, config.t_max_s, config.t_step_s)))
        w1 = float(np.round(rng.uniform(config.w_min, config.w_max), 2))
        w2 = float(np.round(rng.uniform(config.w_min, config.w_max), 2))
        suffix = build_short_suffix(rng)
        spec = MixedSampleSpec(
            dataset_name=dataset_name,
            split=split,
            noise_dataset=noise_dataset,
            source_noise=source_noise,
            source_bk=source_bk,
            w1=w1,
            w2=w2,
            t_s=t_s,
            suffix=suffix,
        )
        try:
            rows.append(_build_mixed_sample(spec, target_dir, config, rng))
        except Exception as exc:  # pragma: no cover - best effort logging
            output_file_name = make_output_file_name(source_bk, noise_dataset, w1, w2, t_s, suffix)
            rows.append(
                BuildLogRow(
                    record_id=output_file_name,
                    dataset_name=dataset_name,
                    split=split,
                    sample_class="broken_wire",
                    output_file_name=output_file_name,
                    output_file_path=str(target_dir / output_file_name),
                    source_noise_file_name=source_noise.sample_name,
                    source_noise_file_path=str(source_noise.path),
                    source_bk_file_name=source_bk.sample_name,
                    source_bk_file_path=str(source_bk.path),
                    source_noise_dataset=noise_dataset,
                    source_bk_method=str(source_bk.method or ""),
                    source_noise_sample_rate_hz=float(source_noise.sample_rate),
                    source_bk_sample_rate_hz=float(source_bk.sample_rate),
                    target_sample_rate_hz=float(config.target_sample_rate_hz),
                    source_noise_duration_s=float(source_noise.duration_s),
                    source_bk_duration_s=float(source_bk.duration_s),
                    output_duration_s=float(config.window_duration_s),
                    arrival_offset_s=float(source_bk.arrival_offset_s),
                    window_start_offset_t_s=float(t_s),
                    window_start_absolute_offset_s=float(source_bk.arrival_offset_s + t_s),
                    window_start_index=-1,
                    window_end_index=-1,
                    w1=float(w1),
                    w2=float(w2),
                    noise_rms=float("nan"),
                    bk_rms=float("nan"),
                    mixed_rms=float("nan"),
                    resample_noise_action="error",
                    resample_bk_action="error",
                    random_seed=int(config.random_seed),
                    build_time=datetime.now().isoformat(timespec="seconds"),
                    builder_version="2026-05-05",
                    status="failed",
                    error_message=str(exc),
                )
            )
    return rows, pd.DataFrame([row.__dict__ for row in rows])


def _write_dataset_report(
    config: BuildConfig,
    log_df: pd.DataFrame,
    manifest_dfs: dict[str, pd.DataFrame],
    output_root: Path,
) -> Path:
    report_path = output_root / "logs" / "dataset_build.md"
    summary_df = log_df.groupby(["dataset_name", "split"], dropna=False).size().reset_index(name="sample_count").sort_values(["dataset_name", "split"])
    lines = [
        "# dataset_build summary",
        "",
        f"- build_time: {datetime.now().isoformat(timespec='seconds')}",
        f"- data_root: {config.data_root}",
        f"- output_root: {output_root}",
        f"- target_sample_rate_hz: {config.target_sample_rate_hz}",
        f"- train_ratio: {config.train_ratio}",
        f"- random_seed: {config.random_seed}",
        "",
        "## sample counts",
        "",
        summary_df.to_string(index=False) if not summary_df.empty else "_empty_",
        "",
        "## manifests",
        "",
    ]
    for name, frame in manifest_dfs.items():
        lines.append(f"- {name}: {len(frame)} rows")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_single_noise_domain(
    dataset_name: str,
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    show_progress: bool = True,
) -> BuildResult:
    """Build one flow-noise domain and its paired mixed datasets."""
    rng = np.random.default_rng(config.random_seed)
    output_dirs = _ensure_root_structure(config.output_root)
    logger = build_logger(output_dirs["logs"] / "dataset_build_runtime.log")
    logger.info("start build for %s", dataset_name)

    bk_records = load_dataset_records(config.data_root, "BK14")
    bk_train, bk_test = split_bk14_records(bk_records, config.train_ratio, config.random_seed)
    _copy_selected_records(bk_train, config.output_root / "BK14_tr")
    _copy_selected_records(bk_test, config.output_root / "BK14_te")

    flow_records = load_dataset_records(config.data_root, dataset_name)
    group_size = config.flow_group_sizes[dataset_name]
    flow_train, flow_test, _, flow_manifest_rows = split_flow_records(
        flow_records,
        dataset_name=dataset_name,
        train_ratio=config.train_ratio,
        seed=config.random_seed,
        group_size=group_size,
    )
    _copy_selected_records(flow_train, config.output_root / f"{dataset_name}_tr")
    _copy_selected_records(flow_test, config.output_root / f"{dataset_name}_te")

    bk_train_paths = {record.path for record in bk_train}
    manifests = {
        "bk14_split_manifest": write_manifest(
            [
                DatasetManifestRow(
                    source_file_name=record.sample_name,
                    source_file_path=str(record.path),
                    dataset_source="BK14",
                    group_id=f"BK14_{record.method}",
                    split="train" if record.path in bk_train_paths else "test",
                    sample_rate_hz=float(record.sample_rate),
                    timestamp_text=str(record.starttime),
                    method=record.method,
                    random_seed=config.random_seed,
                )
                for record in bk_train + bk_test
            ],
            output_dirs["manifests"] / "bk14_split_manifest.csv",
        ),
        f"{dataset_name.lower()}_split_manifest": write_manifest(
            flow_manifest_rows,
            output_dirs["manifests"] / f"{dataset_name.lower()}_split_manifest.csv",
        ),
    }

    log_rows: list[BuildLogRow] = []
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm([("train", flow_train, bk_train), ("test", flow_test, bk_test)], desc=f"build {dataset_name}")
        except Exception:
            iterator = [("train", flow_train, bk_train), ("test", flow_test, bk_test)]
    else:
        iterator = [("train", flow_train, bk_train), ("test", flow_test, bk_test)]

    for split, noise_records, bk_split_records in iterator:
        mixed_rows, _ = _build_mixed_dataset(dataset_name, split, noise_records, bk_split_records, config, rng)
        log_rows.extend(mixed_rows)

    log_df = write_log_csv(log_rows, output_dirs["logs"] / "dataset_build_log.csv")
    report_path = _write_dataset_report(config, log_df, manifests, config.output_root)
    summary_df = log_df.groupby(["dataset_name", "split"], dropna=False).size().reset_index(name="sample_count").sort_values(["dataset_name", "split"]).reset_index(drop=True)
    logger.info("build finished for %s", dataset_name)
    return BuildResult(
        config=config,
        output_root=config.output_root,
        log_df=log_df,
        manifest_dfs=manifests,
        summary_df=summary_df,
        report_path=report_path,
    )


def build_all_datasets(config: BuildConfig = DEFAULT_BUILD_CONFIG, show_progress: bool = True) -> BuildResult:
    """Build all datasets described in the design document."""
    rng = np.random.default_rng(config.random_seed)
    output_dirs = _ensure_root_structure(config.output_root)
    logger = build_logger(output_dirs["logs"] / "dataset_build_runtime.log")
    logger.info("start full build")

    bk_records = load_dataset_records(config.data_root, "BK14")
    bk_train, bk_test = split_bk14_records(bk_records, config.train_ratio, config.random_seed)
    _copy_selected_records(bk_train, config.output_root / "BK14_tr")
    _copy_selected_records(bk_test, config.output_root / "BK14_te")

    bk_train_paths = {record.path for record in bk_train}
    bk_manifest_rows: list[DatasetManifestRow] = [
        DatasetManifestRow(
            source_file_name=record.sample_name,
            source_file_path=str(record.path),
            dataset_source="BK14",
            group_id=f"BK14_{record.method}",
            split="train" if record.path in bk_train_paths else "test",
            sample_rate_hz=float(record.sample_rate),
            timestamp_text=str(record.starttime),
            method=record.method,
            random_seed=config.random_seed,
        )
        for record in bk_train + bk_test
    ]

    flow_manifests: dict[str, pd.DataFrame] = {}
    flow_splits: dict[str, tuple[list[RawSignalRecord], list[RawSignalRecord]]] = {}
    for dataset_name in config.noise_datasets:
        flow_records = load_dataset_records(config.data_root, dataset_name)
        group_size = config.flow_group_sizes[dataset_name]
        flow_train, flow_test, _, flow_manifest_rows = split_flow_records(
            flow_records,
            dataset_name=dataset_name,
            train_ratio=config.train_ratio,
            seed=config.random_seed,
            group_size=group_size,
        )
        flow_splits[dataset_name] = (flow_train, flow_test)
        flow_manifests[f"{dataset_name.lower()}_split_manifest"] = write_manifest(
            flow_manifest_rows,
            output_dirs["manifests"] / f"{dataset_name.lower()}_split_manifest.csv",
        )
        _copy_selected_records(flow_train, config.output_root / f"{dataset_name}_tr")
        _copy_selected_records(flow_test, config.output_root / f"{dataset_name}_te")

    manifest_dfs: dict[str, pd.DataFrame] = {
        "bk14_split_manifest": write_manifest(bk_manifest_rows, output_dirs["manifests"] / "bk14_split_manifest.csv"),
        **flow_manifests,
    }

    log_rows: list[BuildLogRow] = []
    tasks = []
    for dataset_name in config.noise_datasets:
        train_noise, test_noise = flow_splits[dataset_name]
        tasks.append((dataset_name, "train", train_noise, bk_train))
        tasks.append((dataset_name, "test", test_noise, bk_test))

    iterator = tasks
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(tasks, desc="build datasets")
        except Exception:
            iterator = tasks

    for dataset_name, split, noise_records, bk_split_records in iterator:
        mixed_rows, _ = _build_mixed_dataset(dataset_name, split, noise_records, bk_split_records, config, rng)
        log_rows.extend(mixed_rows)

    log_df = write_log_csv(log_rows, output_dirs["logs"] / "dataset_build_log.csv")
    report_path = _write_dataset_report(config, log_df, manifest_dfs, config.output_root)
    summary_df = log_df.groupby(["dataset_name", "split"], dropna=False).size().reset_index(name="sample_count").sort_values(["dataset_name", "split"]).reset_index(drop=True)
    logger.info("full build finished")
    return BuildResult(
        config=config,
        output_root=config.output_root,
        log_df=log_df,
        manifest_dfs=manifest_dfs,
        summary_df=summary_df,
        report_path=report_path,
    )
