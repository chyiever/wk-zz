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
    "feature_schema_version",
    "stft_backend",
    "power_dtype",
    "harmonic_context_band",
}

FEATURE_NAME_RE = re.compile(r"^b_[0-9a-zA-Z_]+__")

BASE_FEATURE_MEANINGS = {
    "mean": "当前规范带通信号的算术均值（去均值后应接近零）",
    "variance": "当前规范带通信号的方差，描述幅值波动能量",
    "rms": "当前规范带通信号的均方根幅值",
    "skewness": "当前规范带通信号幅值分布的偏度",
    "kurtosis": "当前规范带通信号幅值分布的 Pearson 峭度",
    "waveform_factor": "波形因子，RMS 与平均绝对幅值之比",
    "crest_factor": "峰值因子，峰值绝对幅值与 RMS 之比",
    "impulse_factor": "脉冲因子，峰值绝对幅值与平均绝对幅值之比",
    "clearance_factor": "裕度因子，峰值绝对幅值与平均平方根幅值平方之比",
    "permutation_entropy": "归一化排列熵，描述相邻采样排序模式复杂度",
    "MPE_scale2": "尺度 2 多尺度排列熵",
    "MPE_scale3": "尺度 3 多尺度排列熵",
    "singular_spectrum_entropy": "奇异谱熵，描述 Hankel 奇异值能量分布复杂度",
    "spectral_spread": "频谱延展度，频率相对谱质心的加权标准差",
    "power_spectral_entropy": "功率谱熵，描述平均功率谱的频率分布复杂度",
    "energy_entropy": "分段能量熵，描述时间分解片段的能量分散程度",
    **{f"MFCC_{index:02d}": f"第 {index} 个梅尔频率倒谱系数" for index in range(1, 14)},
    "r_p": "包络峰位比，描述主峰在分析窗内出现的相对位置",
    "C_E": "能量质心位置，描述能量在时间轴上的前后偏置",
    "A_env": "包络不对称度，描述包络上升与衰减形状差异",
    "S_env": "包络对称度，包络与其反折信号的相关系数，描述包络左右对称程度",
    "Sk_env": "包络偏度，描述包络幅值分布的不对称性",
    "R_td": "上升/衰减时间比，衰减时间相对上升时间的比值，描述快起慢衰特征",
    "R_fb": "前后能量比，描述事件前段与后段能量相对强度",
    "C_bulge": "包络鼓包集中度，描述局部鼓包结构强弱",
    "N_bulge": "包络鼓包数量，描述显著局部鼓包个数",
    "epsilon_env": "包络拟合残差，描述包络形状偏离基准模型的程度",
    "eta_bw": "包络左右宽度比，5%–95% 累积能量区间围绕峰值的前后宽度比（时域量，与频率带宽无关）",
    "SC_mean": "谱质心均值，描述频谱能量中心",
    "k_sc": "谱质心变化斜率，描述频谱中心随时间迁移趋势",
    "R_hl_mean": "高低频能量比均值，描述高频能量相对低频能量的平均占比",
    "k_hl": "高低频能量比变化斜率，描述高频占比随时间变化趋势",
    "beta_H": "高频能量衰减斜率，描述高频成分衰减快慢",
    "H_tf": "时频熵，描述时频能量分布复杂度",
    "SF": "谱平坦度，描述频谱接近噪声或窄带结构的程度",
    "H_alpha": "Renyi 熵，时频能量分布的 α 阶集中度度量",
    "rho_r": "脊线有效占比，主脊线有效帧占全部帧的比例，描述事件占据的帧比例（有效帧判据：主带帧能量>0.2×帧能量中位数）",
    "G_gap": "脊线间隙比，脊线丢失或跳变超过 2 倍频率分辨率的帧占比，描述间断/跳变程度",
    "R2_ridge": "脊线拟合优度，描述主脊线轨迹可解释性",
    "S_arch": "脊线拱形分数，描述频率轨迹是否呈拱形变化",
    "H2_ratio": "二倍频一致性比，满足 |f2-2f1|<Δf 的有效帧占比，描述 2:1 谐波关系出现频率",
    "H2_observable": "二倍频可观测标志，事件频带SNR达到门限且二倍频脊线能量有效时为1",
    "R_2_1": "二倍频/基频比，f2/f1 脊线 ±相对带宽(±0.08f₁) 邻域能量之比，描述谐波相对基频强度",
    "H_stack": "谐波栈能量比，描述多阶谐波组织程度",
    "R_h": "谐波能量比，f1/f2 脊线固定 ±1 kHz 邻域能量之比，描述二倍频相对基频强度（与 R_2_1 互补）",
    "epsilon_2x": "二倍频一致性误差，描述二倍频轨迹与基频倍频关系偏差",
    "C_f": "归一化频率曲率，主脊线二阶导数均值除以平均主频，描述主频率轨迹弯折程度",
    "Delta_f_span": "频率跨度，描述主要能量覆盖的频率范围",
    "E_harm": "谐波能量，描述谐波成分绝对强度",
    "E_res": "残差能量，描述模型未解释的剩余能量",
    "R_res_hl_mean": "残差高低频能量比均值，描述残差中高频异常占比",
    "R_high_res": "高频残差占比，描述高频未解释成分强度",
    "R_res_tkeo": "残差TKEO峰值比，描述残差冲击瞬态尖锐程度",
    "F_peak": "谱通量峰值，逐频点正增量（半波整流）求和后的帧间突变最大值，描述瞬态起始强度",
    "R_wp_daaa": "小波包节点能量占比 daaa，描述对应子带能量比例",
    "R_wp_daad": "小波包节点能量占比 daad，描述对应子带能量比例",
    "R_wp_dada": "小波包节点能量占比 dada，描述对应子带能量比例",
    "R_wp_dadd": "小波包节点能量占比 dadd，描述对应子带能量比例",
    "R_wp_ddaa": "小波包节点能量占比 ddaa，描述对应子带能量比例",
    "R_wp_ddad": "小波包节点能量占比 ddad，描述对应子带能量比例",
    "R_wp_ddda": "小波包节点能量占比 ddda，描述对应子带能量比例",
    "R_wp_dddd": "小波包节点能量占比 dddd，描述对应子带能量比例",
    "H_wp": "小波包能量熵，描述多分辨率能量分布复杂度",
    "I_burst": "小波包子带能量集中度，峰值子带能量与中位子带能量之比，描述能量在子带间的集中程度",
    "D_WPT": "高频-低频小波包能差，子带中心频率高于主带几何中点的节点能量和与低频之差，描述高频异常偏置",
    "C_damp": "阻尼原子匹配度，描述阻尼振荡成分强弱",
    "alpha_hat": "估计阻尼系数，峰后衰减段残差包络对数线性拟合斜率的相反数，描述衰减速度",
    "Q_MP": "Hankel 低秩重建质量，残差 Hankel 矩阵秩 2 SVD 重建质量，描述残差能否用单个阻尼振荡解释",
    "Delta_J": "加入损伤原子后的误差下降率，描述损伤分量解释增益",
    "eta_dict": "损伤分量波形占比，损伤分量与谐波重构波形 L1 范数之比，描述损伤分量占主导程度（波形口径，非稀疏系数比）",
    "C_h": "谐波一致性水平，主脊线与二倍频脊线间距绝对值的均值，即 mean(|f2-2f1|)，越小说明谐波关系越严格",
    "R_harm": "谐波能量占比，谐波分量能量与总能量的比值，描述谐波结构总体强度",
    "Ridge_coh": "谐波能量占比（R_harm 同值别名，已停用），描述脊线能量集中程度",
    "rho_up": "脊线上升率，主脊线频率上升斜率超过阈值的帧占比，描述频率上扫的活跃程度",
    "rho_down": "脊线下降率，主脊线频率下降斜率低于 -阈值的帧占比，描述频率下扫的活跃程度",
    "N_turn": "脊线转向次数，剔除零斜率后主脊线频率斜率变号的次数，描述频率轨迹的曲折程度",
    "rho_res": "残差能量占比，残差能量与总能量的比值，描述未能被谐波模型解释的能量比例",
    "N_abn": "异常帧数，残差能量占比超过阈值的帧数，描述残差爆发突变的时帧数量",
    "R_tkeo": "原信号TKEO峰值比，TKEO输出峰值与均值的比值，描述冲击瞬态尖锐程度",
    "K_loc": "原信号峭度，四阶标准矩，描述幅值分布的重尾程度",
    "K_res_max": "残差局部峭度最大值，滑动窗内残差峭度的最大值，描述残差中最强的局部重尾冲击",
    "SK_max": "谱峭度峰值，各频点经典谱峭度(<|X|^4>/<|X|^2>^2-2)的最大值，描述瞬态主导频带",
    "CF_res": "残差峰值因子，残差峰值与RMS之比，描述残差冲击峰值相对能量水平的突出程度",
    "T_half_high": "高频半衰期，高频包络从峰值衰减到一半所需时长，描述高频成分衰减快慢（峰后未跌破 50% 时返回 NaN）",
    "SC_res_mean": "残差谱质心均值，残差频谱能量的平均中心频率，描述残差能量集中频段",
    "k_res_sc": "残差谱质心变化斜率，残差谱质心随时间线性回归斜率，描述残差主频随时间迁移趋势",
    "k_res_hl": "残差高低频能量比变化斜率，残差中高频能量占比随时间的变化速率",
    "S_res_env": "残差包络对称度，残差包络与其反折信号的相关系数，描述残差包络形状的对称性",
    "eta_asym": "残差不对称比，峰值后残差能量与峰值前残差能量之比，描述残差包络的上升/衰减不对称性",
    "epsilon_rec": "整体重构归一化误差，原始信号与谐波重构信号之差的能量占原始能量的比例",
    "SNR_band_db": "当前频带事件段相对局部背景段的能量信噪比（dB）",
    "E_excess": "当前频带扣除局部背景均值后的非负超额能量",
    "SNR_high_db": "当前频带内部高频子带事件段相对局部背景段的信噪比（dB）",
    "high_observable": "高频子带可观测标志，高频SNR达到门限且事件能量有效时为1",
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
