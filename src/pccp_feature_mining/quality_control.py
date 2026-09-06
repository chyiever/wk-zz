"""数据质量检查与摘要输出。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import numeric_feature_frame


def build_dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """按六类标签统计样本、源文件和事件数量。"""

    rows: list[dict[str, object]] = []
    for label, group in frame.groupby("source_label", sort=True):
        rows.append(
            {
                "source_label": label,
                "target_label": group["target_label"].iloc[0],
                "signal_family": group["signal_family"].iloc[0],
                "flow_condition": group["flow_condition"].iloc[0],
                "rows": int(len(group)),
                "source_files": int(group["event_id"].nunique()),
                "sampling_groups": int(group["sampling_group_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_quality(frame: pd.DataFrame, feature_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """计算每个特征的缺失、非有限、尺度和粗略异常值比例。"""

    x = numeric_feature_frame(frame, feature_columns)
    rows: list[dict[str, object]] = []
    n = max(len(x), 1)
    for col in x.columns:
        s = x[col]
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        q1 = finite.quantile(0.25) if len(finite) else np.nan
        q3 = finite.quantile(0.75) if len(finite) else np.nan
        iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
        if pd.notna(iqr) and iqr > 0:
            outlier_rate = float(((finite < q1 - 3.0 * iqr) | (finite > q3 + 3.0 * iqr)).mean())
        else:
            outlier_rate = 0.0
        rows.append(
            {
                "feature": col,
                "non_missing": int(finite.size),
                "missing_rate": float(1.0 - finite.size / n),
                "unique_values": int(finite.nunique(dropna=True)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=0)) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "iqr": float(iqr) if pd.notna(iqr) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "outlier_rate_iqr3": outlier_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "feature"], ascending=[True, True])


def feature_list_frame(feature_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """输出特征清单，拆分频带和基础特征名，方便人工审查。"""

    rows = []
    for col in feature_columns:
        if "__" in col:
            band, name = col.split("__", 1)
        else:
            band, name = "", col
        rows.append({"feature": col, "band": band, "base_feature": name})
    return pd.DataFrame(rows)
