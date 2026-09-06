"""PCCP断丝特征挖掘配置。

本模块只存放路径、标签和实验参数，避免 notebook 中散落硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FeatureInput:
    """一个特征输入目录及其显式标签。"""

    label: str
    path: Path


@dataclass(frozen=True)
class MiningConfig:
    """特征挖掘流程的主要可调参数。"""

    output_dir: Path = Path("outputs/PCCP_feature_mining")
    min_feature_csv_columns: int = 100
    min_valid_feature_values_per_row: int = 100
    random_state: int = 42
    correlation_threshold: float = 0.92
    bootstrap_rounds: int = 200
    bootstrap_top_ks: tuple[int, ...] = (10, 20, 30)
    mrmr_top_n: int = 80
    mrmr_redundancy_weight: float = 0.5
    max_correlation_rows: int = 6000
    max_plot_features: int = 24
    distribution_top_n: int = 12
    projection_max_rows: int = 8000
    combination_feature_counts: tuple[int, ...] = (5, 10, 20, 30, 50)
    sfs_candidate_count: int = 30
    sfs_max_selected: int = 15
    classification_feature_counts: tuple[int, ...] = (5, 10, 20, 30, 50)
    classification_max_train_rows_per_class: int | None = 3000
    # 仅用于快速调试。正式运行保持 None，使用全部数据。
    max_rows_per_label: int | None = None
    feature_inputs: tuple[FeatureInput, ...] = field(default_factory=tuple)


def default_feature_inputs(workspace: Path | None = None) -> tuple[FeatureInput, ...]:
    """返回本次DATA09六类特征表的默认输入。"""

    root = Path.cwd() if workspace is None else Path(workspace)
    return (
        FeatureInput(
            "FL00",
            root / "outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0-flow_features",
        ),
        FeatureInput(
            "FL05",
            root / "outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0.5-flow_features",
        ),
        FeatureInput("BK00", root / "outputs/DATA09_multi_label_single_event_features/v0-bk_features_BK00"),
        FeatureInput("QJ00", root / "outputs/DATA09_multi_label_single_event_features/v0-qj_features_QJ00"),
        FeatureInput("BK05", root / "outputs/DATA09_multi_label_single_event_features/v05-bk_features_BK05"),
        FeatureInput("QJ05", root / "outputs/DATA09_multi_label_single_event_features/v05-qj_features_QJ05"),
    )


def binary_label(label: str) -> str:
    """六类来源标签到二分类目标的映射。"""

    return "BK" if str(label).upper().startswith("BK") else "Other"


def flow_condition(label: str) -> str:
    """从标签中解析流速工况，00代表v0，05代表v0.5。"""

    text = str(label).upper()
    if text.endswith("00"):
        return "v0"
    if text.endswith("05"):
        return "v0.5"
    return "unknown"


def signal_family(label: str) -> str:
    """从标签中解析信号家族：BK、FL、QJ。"""

    text = str(label).upper()
    if text.startswith("BK"):
        return "BK"
    if text.startswith("FL"):
        return "FL"
    if text.startswith("QJ"):
        return "QJ"
    return "OTHER"
