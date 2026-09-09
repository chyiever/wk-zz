from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("FEA_CPT_USE_GPU", "1")

from fea_cpt_gpu_v2_2.sliding_window import (  # noqa: E402
    SlidingWindowConfig,
    build_params_for_band,
    build_sliding_window_dataset,
    compute_all_features_for_window,
    compute_shared_stft,
    discover_source_files,
    load_source_file,
    upsample_to_target,
)
from fea_cpt_gpu_v2_2.signal_ops import butter_filter  # noqa: E402


BANDS = [
    ("b_1k_100k", (1_000.0, 100_000.0)),
    ("b_1k_10k", (1_000.0, 10_000.0)),
    ("b_10k_20k", (10_000.0, 20_000.0)),
    ("b_20k_30k", (20_000.0, 30_000.0)),
    ("b_30k_40k", (30_000.0, 40_000.0)),
    ("b_40k_60k", (40_000.0, 60_000.0)),
    ("b_10k_50k", (10_000.0, 50_000.0)),
    ("b_1k_50k", (1_000.0, 50_000.0)),
]

META_COLS = {
    "source_file_name",
    "source_file_path",
    "source_format",
    "source_group_name",
    "source_channel_name",
    "source_detail",
    "sample_rate_hz",
    "original_sample_rate_hz",
    "source_n_samples",
    "source_duration_s",
    "starttime_raw",
    "arrival_time_raw",
    "sample_type",
    "label",
    "window_id",
    "window_mode",
    "window_start_index",
    "window_end_index",
    "window_length_samples",
    "window_step_samples",
    "window_duration_s",
    "window_start_offset_s",
    "window_start_datetime",
    "window_start_ms",
    "window_end_ms",
    "window_n_samples",
    "split",
}


def parse_npz_ts(value: str) -> datetime:
    text = str(value).strip()
    if "T" in text:
        return datetime.strptime(text, "%Y%m%dT%H%M%S.%f")
    return datetime.strptime(text, "%Y%m%d%H%M%S.%f")


def event_windows(n_samples: int, fs: float, arrival_offset_s: float) -> list[tuple[str, int, int, float, float]]:
    modes = [
        ("m10_20", -10.0, 20.0),
        ("p00_30", 0.0, 30.0),
        ("p05_35", 5.0, 35.0),
    ]
    out = []
    for name, start_ms, end_ms in modes:
        start = int(round((arrival_offset_s + start_ms / 1000.0) * fs))
        end = int(round((arrival_offset_s + end_ms / 1000.0) * fs))
        start = max(0, min(start, n_samples))
        end = max(start, min(end, n_samples))
        if end - start == int(round(0.030 * fs)):
            out.append((name, start, end, start_ms, end_ms))
    return out


def extract_event_features(
    data_root: Path,
    label: str,
    output_csv: Path,
    channel_index: int = 0,
    max_files: int | None = None,
) -> pd.DataFrame:
    files = sorted(data_root.glob("*.npz"))
    if max_files is not None:
        files = files[:max_files]
    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for fp in files:
        try:
            src = load_source_file(fp, None, None)
            raw = np.asarray(src["signal_values"], dtype=float)
            if raw.ndim == 2:
                raw = raw[:, channel_index]
            fs0 = float(src["sample_rate"])
            sig, fs = upsample_to_target(raw, fs0, 1_000_000.0)
            sig = butter_filter(sig - float(np.mean(sig)), sample_rate=fs, band_hz=(1_000.0, 400_000.0), order=4)
            params_map = {name: build_params_for_band(band, fs) for name, band in BANDS}
            arrival_offset_s = (parse_npz_ts(str(src["arrival_time_raw"])) - parse_npz_ts(str(src["starttime_raw"]))).total_seconds()
            for wid, (mode, i0, i1, start_ms, end_ms) in enumerate(event_windows(len(sig), fs, arrival_offset_s)):
                segment = sig[i0:i1]
                shared = compute_shared_stft(segment, fs, BANDS, params_map)
                feats = compute_all_features_for_window(segment, fs, params_map, shared_stft=shared)
                row: dict[str, object] = {
                    "source_file_name": fp.name,
                    "source_file_path": str(fp),
                    "source_format": "npz",
                    "source_group_name": "",
                    "source_channel_name": f"phase_data[{channel_index}]",
                    "source_detail": f"{fp.parent.name}/phase_data[{channel_index}]",
                    "sample_rate_hz": fs,
                    "original_sample_rate_hz": fs0,
                    "source_n_samples": len(sig),
                    "source_duration_s": len(sig) / fs,
                    "starttime_raw": str(src["starttime_raw"]),
                    "arrival_time_raw": str(src["arrival_time_raw"]),
                    "sample_type": label,
                    "label": label,
                    "window_id": wid,
                    "window_mode": mode,
                    "window_start_index": i0,
                    "window_end_index": i1,
                    "window_length_samples": i1 - i0,
                    "window_step_samples": i1 - i0,
                    "window_duration_s": (i1 - i0) / fs,
                    "window_start_offset_s": i0 / fs,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "window_n_samples": i1 - i0,
                }
                row.update(feats)
                rows.append(row)
        except Exception as exc:
            failed.append({"source_file_name": fp.name, "error": repr(exc)})
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if failed:
        pd.DataFrame(failed).to_csv(output_csv.with_name(output_csv.stem + "_failed.csv"), index=False, encoding="utf-8-sig")
    return df


def extract_flow_features(
    data_root: Path,
    output_dir: Path,
    max_files: int | None,
    max_windows_per_file: int | None = None,
    run_timestamp: str | None = None,
) -> pd.DataFrame:
    files = discover_source_files([data_root], max_files=max_files)
    config = SlidingWindowConfig(
        bands=BANDS,
        preproc_band=(1_000.0, 95_000.0),
        window_duration_s=0.030,
        window_overlap=0.0,
        target_sample_rate=1_000_000.0,
        tdms_fallback_sample_rate=1_000_000.0,
        tdms_channel_name="Untitled",
        window_workers=1,
        enable_shared_stft=True,
        stft_batch_size=80,
    )
    if output_dir.exists():
        for old in output_dir.glob("features_part_*.csv"):
            old.unlink()
        for old in output_dir.glob("log_part_*.csv"):
            old.unlink()
        processed = output_dir / "processed_source_files.txt"
        if processed.exists():
            processed.unlink()
    if max_windows_per_file is None:
        processed_name = f"processed_source_files_{run_timestamp}.txt" if run_timestamp else "processed_source_files.txt"
        build_sliding_window_dataset(files, config, output_dir, output_dir / processed_name, npz_per_csv=4)
        if run_timestamp:
            for csv_path in sorted(output_dir.glob("features_part_*.csv")):
                csv_path.rename(csv_path.with_name(f"features_{run_timestamp}_{csv_path.name.removeprefix('features_')}"))
            for csv_path in sorted(output_dir.glob("log_part_*.csv")):
                csv_path.rename(csv_path.with_name(f"log_{run_timestamp}_{csv_path.name.removeprefix('log_')}"))
        pattern = f"features_{run_timestamp}_part_*.csv" if run_timestamp else "features_part_*.csv"
        chunks = sorted(output_dir.glob(pattern))
        df = pd.concat((pd.read_csv(p) for p in chunks), ignore_index=True) if chunks else pd.DataFrame()
    else:
        from fea_cpt_gpu_v2_2.sliding_window import list_window_ranges

        rows = []
        for fp in files:
            src = load_source_file(fp, config.tdms_fallback_sample_rate, config.tdms_channel_name)
            raw = np.asarray(src["signal_values"], dtype=float)
            fs0 = float(src["sample_rate"])
            sig, fs = upsample_to_target(raw, fs0, config.target_sample_rate)
            sig = butter_filter(sig - float(np.mean(sig)), sample_rate=fs, band_hz=config.preproc_band, order=4)
            params_map = {name: build_params_for_band(band, fs) for name, band in config.bands}
            windows = list_window_ranges(len(sig), fs, config.window_duration_s, config.window_overlap)[:max_windows_per_file]
            for wid, i0, i1, win_len, step_len in windows:
                segment = sig[i0:i1]
                shared = compute_shared_stft(segment, fs, BANDS, params_map)
                feats = compute_all_features_for_window(segment, fs, params_map, shared_stft=shared)
                row: dict[str, object] = {
                    "source_file_name": fp.name,
                    "source_file_path": str(fp),
                    "source_format": str(src["source_format"]),
                    "source_group_name": str(src.get("source_group_name", "")),
                    "source_channel_name": str(src.get("source_channel_name", "")),
                    "source_detail": str(src.get("source_detail", "")),
                    "sample_rate_hz": fs,
                    "original_sample_rate_hz": fs0,
                    "source_n_samples": len(sig),
                    "source_duration_s": len(sig) / fs,
                    "starttime_raw": str(src["starttime_raw"]),
                    "arrival_time_raw": str(src["arrival_time_raw"]),
                    "sample_type": "flow",
                    "label": "flow",
                    "window_id": wid,
                    "window_start_index": i0,
                    "window_end_index": i1,
                    "window_length_samples": win_len,
                    "window_step_samples": step_len,
                    "window_duration_s": win_len / fs,
                    "window_start_offset_s": i0 / fs,
                    "window_start_datetime": "",
                }
                row.update(feats)
                rows.append(row)
        df = pd.DataFrame(rows)
        all_name = f"features_all_{run_timestamp}.csv" if run_timestamp else "features_all.csv"
        df.to_csv(output_dir / all_name, index=False, encoding="utf-8-sig")
    if not df.empty:
        df["label"] = "flow"
        df["sample_type"] = "flow"
        all_name = f"features_all_{run_timestamp}.csv" if run_timestamp else "features_all.csv"
        df.to_csv(output_dir / all_name, index=False, encoding="utf-8-sig")
    return df


def split_by_source(df: pd.DataFrame, train_ratio: float = 0.6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for label, g in df.groupby("label", sort=True):
        sources = np.array(sorted(g["source_file_name"].astype(str).unique()))
        rng.shuffle(sources)
        n_train = max(1, int(round(len(sources) * train_ratio)))
        if len(sources) > 1:
            n_train = min(n_train, len(sources) - 1)
        train_sources = set(sources[:n_train])
        part = g.copy()
        part["split"] = np.where(part["source_file_name"].astype(str).isin(train_sources), "rank_train", "test")
        out.append(part)
    return pd.concat(out, ignore_index=True)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    good = []
    for c in cols:
        values = pd.to_numeric(df[c], errors="coerce")
        if values.notna().sum() > 0 and values.nunique(dropna=True) > 1:
            good.append(c)
    return good


def run_importance(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    top_n: int = 20,
    run_timestamp: str | None = None,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    df = split_by_source(df, seed=seed)
    feature_cols = numeric_feature_columns(df)
    train = df[df["split"] == "rank_train"].copy()
    test = df[df["split"] == "test"].copy()
    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train["label"].astype(str)
    y_test = test["label"].astype(str)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rf = model.named_steps["randomforestclassifier"]
    impurity = pd.Series(rf.feature_importances_, index=feature_cols, name="rf_importance")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "rf_importance": impurity.reindex(feature_cols).values,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values(["perm_importance_mean", "rf_importance"], ascending=False)
    suffix = f"_{run_timestamp}" if run_timestamp else ""
    importance.to_csv(output_dir / f"feature_importance{suffix}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(output_dir / f"data09_feature_dataset_with_split{suffix}.csv", index=False, encoding="utf-8-sig")

    labels = sorted(df["label"].astype(str).unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    summary = {
        "rows_total": int(len(df)),
        "rows_rank_train": int(len(train)),
        "rows_test": int(len(test)),
        "features_used": int(len(feature_cols)),
        "labels": labels,
        "source_overlap": sorted(set(train["source_file_name"]) & set(test["source_file_name"])),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": report,
    }
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top = importance.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.32 * len(top))))
    plt.barh(top["feature"], top["perm_importance_mean"], xerr=top["perm_importance_std"], color="#2f6f8f")
    plt.xlabel("Permutation importance on held-out 40%")
    plt.title("DATA09 feature importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_top20{suffix}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Hold-out confusion matrix, accuracy={summary['test_accuracy']:.3f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix{suffix}.png", dpi=180)
    plt.close()

    dist_features = importance.head(8)["feature"].tolist()
    long_df = df[["label", "split"] + dist_features].melt(id_vars=["label", "split"], var_name="feature", value_name="value")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    g = sns.FacetGrid(long_df, col="feature", col_wrap=2, hue="label", sharex=False, sharey=False, height=3.0)
    g.map_dataframe(sns.kdeplot, x="value", fill=False, common_norm=False)
    g.add_legend()
    g.fig.suptitle("Top feature distributions by class", y=1.02)
    g.tight_layout()
    g.savefig(output_dir / f"top_feature_distributions{suffix}.png", dpi=180)
    plt.close(g.fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-flow-files", type=int, default=None)
    parser.add_argument("--max-flow-windows-per-file", type=int, default=None)
    parser.add_argument("--max-event-files-per-class", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for this run. Defaults to outputs/DATA09_feature_importance_30ms.",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Timestamp suffix added to output filenames, for example 20260904_153000.",
    )
    args = parser.parse_args()
    t0 = time.time()
    data_root = WORKSPACE / "DATA09"
    out = args.output_dir if args.output_dir is not None else WORKSPACE / "outputs" / "DATA09_feature_importance_30ms"
    features_out = out / "features"
    suffix = f"_{args.run_timestamp}" if args.run_timestamp else ""
    bk = extract_event_features(
        data_root / "v0-bk",
        "bk",
        features_out / f"DATA09_v0-bk_features_30ms{suffix}.csv",
        max_files=args.max_event_files_per_class,
    )
    qj = extract_event_features(
        data_root / "v0-qj",
        "qj",
        features_out / f"DATA09_v0-qj_features_30ms{suffix}.csv",
        max_files=args.max_event_files_per_class,
    )
    flow = extract_flow_features(
        data_root / "v0-flow",
        features_out / f"DATA09_v0-flow_features_30ms{suffix}",
        args.max_flow_files,
        args.max_flow_windows_per_file,
        args.run_timestamp,
    )
    df = pd.concat([flow, bk, qj], ignore_index=True, sort=False)
    df.to_csv(out / f"data09_all_features_30ms{suffix}.csv", index=False, encoding="utf-8-sig")
    summary = run_importance(df, out / "analysis", run_timestamp=args.run_timestamp)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
