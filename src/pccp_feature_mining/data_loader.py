"""六类PCCP特征表读取与标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import FeatureInput, binary_label, flow_condition, signal_family
from .event_parser import add_event_columns
from .feature_schema import infer_feature_columns


@dataclass(frozen=True)
class LoadedDataset:
    """已标准化的数据集对象。"""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    load_report: pd.DataFrame


def discover_feature_csvs(input_dir: Path) -> list[Path]:
    """递归查找特征CSV，排除日志和质量报告。"""

    return sorted(
        p
        for p in Path(input_dir).rglob("features*.csv")
        if p.is_file() and not p.name.lower().startswith("log_")
    )


def _read_one_csv(csv_path: Path, label: str, min_columns: int) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """读取单个CSV，并返回读取报告。"""

    report: dict[str, object] = {
        "label": label,
        "path": str(csv_path),
        "status": "ok",
        "rows": 0,
        "columns": 0,
        "message": "",
    }
    try:
        header = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
        report["columns"] = int(len(header.columns))
        if len(header.columns) < min_columns:
            report["status"] = "skipped"
            report["message"] = f"表头列数{len(header.columns)}小于阈值{min_columns}"
            return None, report

        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        report["rows"] = int(len(df))
        report["columns"] = int(len(df.columns))
        if df.empty:
            report["status"] = "skipped"
            report["message"] = "空表"
            return None, report

        df = df.copy()
        df["source_label"] = str(label)
        df["target_label"] = binary_label(label)
        df["signal_family"] = signal_family(label)
        df["flow_condition"] = flow_condition(label)
        return df, report
    except Exception as exc:  # pragma: no cover - 真实数据异常兜底
        report["status"] = "failed"
        report["message"] = repr(exc)
        return None, report


def load_feature_dataset(
    feature_inputs: list[FeatureInput] | tuple[FeatureInput, ...],
    min_feature_csv_columns: int = 100,
    max_rows_per_label: int | None = None,
    random_state: int = 42,
) -> LoadedDataset:
    """读取六类特征目录，统一标签和事件分组。"""

    frames: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []

    for item in feature_inputs:
        csvs = discover_feature_csvs(Path(item.path))
        if not csvs:
            reports.append(
                {
                    "label": item.label,
                    "path": str(item.path),
                    "status": "failed",
                    "rows": 0,
                    "columns": 0,
                    "message": "未找到features*.csv",
                }
            )
            continue
        label_frames: list[pd.DataFrame] = []
        for csv_path in csvs:
            df, report = _read_one_csv(csv_path, item.label, min_feature_csv_columns)
            reports.append(report)
            if df is not None:
                label_frames.append(df)
        if not label_frames:
            continue
        label_df = pd.concat(label_frames, ignore_index=True, sort=False)
        if max_rows_per_label is not None and len(label_df) > max_rows_per_label:
            label_df = label_df.sample(n=max_rows_per_label, random_state=random_state).sort_index()
        frames.append(label_df)

    if not frames:
        raise ValueError("没有成功读取任何特征表，请检查输入目录和CSV格式")

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = add_event_columns(frame)
    feature_columns = tuple(infer_feature_columns(frame))
    if not feature_columns:
        raise ValueError("未识别到可用数值特征列")

    return LoadedDataset(
        frame=frame,
        feature_columns=feature_columns,
        load_report=pd.DataFrame(reports),
    )
