"""特征列识别与清洗规则。"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


META_COLUMNS = {
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
    "source_label",
    "target_label",
    "signal_family",
    "flow_condition",
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
    "source_file_index_in_label",
    "output_part",
    "arrival_time",
    "starttime",
    "channel_index",
    "event_id",
    "sampling_group_id",
    "row_uid",
    "split",
}

FEATURE_NAME_RE = re.compile(r"^b_[0-9a-zA-Z_]+__")


def infer_feature_columns(frame: pd.DataFrame, extra_meta: Iterable[str] = ()) -> list[str]:
    """识别可用于挖掘的数值特征列。

    优先使用形如 ``b_1k_100k__C_E`` 的频带特征列；若输入表将来扩展为非
    频带命名，也允许所有非元数据、可转成数值且非恒定的列进入分析。
    """

    meta = META_COLUMNS | set(extra_meta)
    candidates: list[str] = []
    for col in frame.columns:
        if col in meta:
            continue
        if FEATURE_NAME_RE.match(str(col)) or "__" in str(col):
            candidates.append(str(col))
        elif pd.api.types.is_numeric_dtype(frame[col]):
            candidates.append(str(col))

    good: list[str] = []
    for col in candidates:
        values = pd.to_numeric(frame[col], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.size >= 3 and finite.nunique(dropna=True) > 1:
            good.append(col)
    return good


def numeric_feature_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """将特征矩阵统一为有限浮点数，非有限值暂保留为NaN供后续插补。"""

    x = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).astype(float)


def impute_with_median(x: pd.DataFrame) -> pd.DataFrame:
    """按列中位数插补缺失值；全缺失列用0兜底。"""

    med = x.median(axis=0, skipna=True).fillna(0.0)
    return x.fillna(med)
