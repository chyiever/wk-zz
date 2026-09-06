"""物理事件分组与采样单元解析。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import signal_family


def source_stem(value: object) -> str:
    """从文件名或路径中提取稳定的源文件主名。"""

    text = str(value or "").strip()
    if not text:
        return "unknown_source"
    return Path(text.replace("\\", "/")).stem


def add_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """补充事件组和采样组列。

    BK/QJ来自少量物理事件，同一源文件的多个窗口必须作为同一事件处理；
    FL为连续流噪背景，方案要求可按窗口独立抽样，因此每行单独成组。
    """

    out = frame.copy()
    if "source_file_name" in out.columns:
        src = out["source_file_name"].map(source_stem)
    elif "source_file_path" in out.columns:
        src = out["source_file_path"].map(source_stem)
    else:
        src = pd.Series([f"row_{idx}" for idx in out.index], index=out.index)

    out["event_id"] = out["source_label"].astype(str) + "::" + src.astype(str)
    row_uid = out["source_label"].astype(str) + "::row_" + out.index.astype(str)
    out["row_uid"] = row_uid

    family = out["source_label"].map(signal_family)
    out["sampling_group_id"] = out["event_id"]
    out.loc[family.eq("FL"), "sampling_group_id"] = row_uid[family.eq("FL")]
    return out
