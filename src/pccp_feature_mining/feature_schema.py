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

BASE_FEATURE_MEANINGS = {
    "r_p": "包络峰位比，描述主峰在分析窗内出现的相对位置",
    "C_E": "能量质心位置，描述能量在时间轴上的前后偏置",
    "A_env": "包络不对称度，描述包络上升与衰减形状差异",
    "S_env": "包络尖锐度，描述包络峰值相对背景的突出程度",
    "Sk_env": "包络偏度，描述包络幅值分布的不对称性",
    "R_td": "时域动态范围比，描述峰值与背景幅值差异",
    "R_fb": "前后能量比，描述事件前段与后段能量相对强度",
    "C_bulge": "包络鼓包集中度，描述局部鼓包结构强弱",
    "N_bulge": "包络鼓包数量，描述显著局部鼓包个数",
    "epsilon_env": "包络拟合残差，描述包络形状偏离基准模型的程度",
    "eta_bw": "包络带宽占比，描述有效能量时间宽度",
    "SC_mean": "谱质心均值，描述频谱能量中心",
    "k_sc": "谱质心变化斜率，描述频谱中心随时间迁移趋势",
    "R_hl_mean": "高低频能量比均值，描述高频能量相对低频能量的平均占比",
    "k_hl": "高低频能量比变化斜率，描述高频占比随时间变化趋势",
    "beta_H": "高频能量衰减斜率，描述高频成分衰减快慢",
    "H_tf": "时频熵，描述时频能量分布复杂度",
    "SF": "谱平坦度，描述频谱接近噪声或窄带结构的程度",
    "H_alpha": "谱幂律斜率，描述频谱随频率衰减的形态",
    "rho_r": "脊线能量占比，描述主频率脊线集中程度",
    "G_gap": "脊线间隔稳定性，描述主脊线与次脊线分离程度",
    "R2_ridge": "脊线拟合优度，描述主脊线轨迹可解释性",
    "S_arch": "脊线拱形分数，描述频率轨迹是否呈拱形变化",
    "H2_ratio": "二倍频能量比，描述二倍频结构强度",
    "R_2_1": "二倍频/基频比，描述谐波相对基频强度",
    "H_stack": "谐波栈能量比，描述多阶谐波组织程度",
    "R_h": "谐波能量占比，描述谐波结构总体强度",
    "epsilon_2x": "二倍频一致性误差，描述二倍频轨迹与基频倍频关系偏差",
    "C_f": "频率曲率或中心频率相关量，描述主频率轨迹形态",
    "Delta_f_span": "频率跨度，描述主要能量覆盖的频率范围",
    "E_harm": "谐波能量，描述谐波成分绝对强度",
    "E_res": "残差能量，描述模型未解释的剩余能量",
    "R_res_hl_mean": "残差高低频能量比均值，描述残差中高频异常占比",
    "R_high_res": "高频残差占比，描述高频未解释成分强度",
    "R_res_tkeo": "残差TKEO峰值比，描述残差冲击瞬态尖锐程度",
    "F_peak": "残差或频谱峰值强度，描述局部峰值突出程度",
    "R_wp_daaa": "小波包节点能量占比 daaa，描述对应子带能量比例",
    "R_wp_daad": "小波包节点能量占比 daad，描述对应子带能量比例",
    "R_wp_dada": "小波包节点能量占比 dada，描述对应子带能量比例",
    "R_wp_dadd": "小波包节点能量占比 dadd，描述对应子带能量比例",
    "R_wp_ddaa": "小波包节点能量占比 ddaa，描述对应子带能量比例",
    "R_wp_ddad": "小波包节点能量占比 ddad，描述对应子带能量比例",
    "R_wp_ddda": "小波包节点能量占比 ddda，描述对应子带能量比例",
    "R_wp_dddd": "小波包节点能量占比 dddd，描述对应子带能量比例",
    "H_wp": "小波包能量熵，描述多分辨率能量分布复杂度",
    "I_burst": "小波包突发指数，描述局部突发能量强度",
    "D_WPT": "高频-低频小波包能差，描述高频异常偏置",
    "C_damp": "阻尼原子匹配度，描述阻尼振荡成分强弱",
    "alpha_hat": "估计阻尼系数，描述衰减速度",
    "Q_MP": "矩阵铅笔拟合优度，描述阻尼正弦模型解释能力",
    "Delta_J": "加入损伤原子后的误差下降率，描述损伤分量解释增益",
    "eta_dict": "字典损伤系数比，描述损伤字典成分占比",
}


def split_feature_name(feature: str) -> tuple[str, str]:
    """拆分频带前缀和基础特征名。"""

    text = str(feature)
    if "__" not in text:
        return "", text
    band, base = text.split("__", 1)
    return band, base


def band_chinese_label(band: str) -> str:
    """将 b_10k_50k 转为 10-50 kHz 频带。"""

    if not band:
        return "全局或未标注频带"
    nums = re.findall(r"(\d+)k", band)
    if len(nums) >= 2:
        return f"{nums[0]}-{nums[1]} kHz频带"
    if len(nums) == 1:
        return f"{nums[0]} kHz相关频带"
    return f"{band}频带"


def feature_chinese_meaning(feature: str) -> str:
    """生成特征中文含义，优先使用基础特征字典并补充频带信息。"""

    band, base = split_feature_name(str(feature))
    meaning = BASE_FEATURE_MEANINGS.get(base)
    if meaning is None and base.startswith("R_wp_"):
        meaning = f"小波包节点能量占比 {base.removeprefix('R_wp_')}，描述对应子带能量比例"
    if meaning is None:
        meaning = f"{base}，暂未在特征字典中补充详细物理解释"
    return f"{band_chinese_label(band)}，{meaning}"


def describe_feature_list(features: str | Iterable[str], sep: str = "；") -> str:
    """把一个或多个特征名转换为中文含义列表。"""

    if isinstance(features, str):
        items = [item.strip() for item in re.split(r"[;；]", features) if item.strip()]
    else:
        items = [str(item).strip() for item in features if str(item).strip()]
    return sep.join(f"{item}: {feature_chinese_meaning(item)}" for item in items)


def add_feature_meaning_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """给包含特征名的结果表追加中文含义列。"""

    out = frame.copy()
    if "feature" in out.columns and "特征中文含义" not in out.columns:
        out["特征中文含义"] = out["feature"].map(feature_chinese_meaning)
    if "feature_a" in out.columns and "feature_a中文含义" not in out.columns:
        out["feature_a中文含义"] = out["feature_a"].map(feature_chinese_meaning)
    if "feature_b" in out.columns and "feature_b中文含义" not in out.columns:
        out["feature_b中文含义"] = out["feature_b"].map(feature_chinese_meaning)
    if "cluster_representative" in out.columns and "代表特征中文含义" not in out.columns:
        out["代表特征中文含义"] = out["cluster_representative"].map(feature_chinese_meaning)
    if "recommended_keep" in out.columns and "建议保留特征中文含义" not in out.columns:
        out["建议保留特征中文含义"] = out["recommended_keep"].map(feature_chinese_meaning)
    if "added_feature" in out.columns and "新增特征中文含义" not in out.columns:
        out["新增特征中文含义"] = out["added_feature"].map(feature_chinese_meaning)
    if "selected_features" in out.columns and "所选特征中文含义" not in out.columns:
        out["所选特征中文含义"] = out["selected_features"].map(describe_feature_list)
    if "recommended_features" in out.columns and "推荐特征中文含义" not in out.columns:
        out["推荐特征中文含义"] = out["recommended_features"].map(describe_feature_list)
    if "features_in_group" in out.columns and "特征组中文含义" not in out.columns:
        out["特征组中文含义"] = out["features_in_group"].map(describe_feature_list)
    if "recommended_drop" in out.columns and "建议删除特征中文含义" not in out.columns:
        out["建议删除特征中文含义"] = out["recommended_drop"].map(describe_feature_list)
    return out


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
