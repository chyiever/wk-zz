import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/DATA09_v0-qj_sample_label_normalization.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """# DATA09 v0-qj 样本标签规范化

本 notebook 用于把 `DATA09/v0-qj` 目录中的 `QJ` 样本标签统一规范化为 `QJ00`。

迁移时通常只需要改配置区的 `DATA_DIR`、`OLD_LABEL`、`NEW_LABEL`。其余单元按顺序执行即可。

规范化范围：

- 文件名：只修改以 `QJ-` 开头的 `.npz` 文件，例如 `QJ-FIP-...npz` -> `QJ00-FIP-...npz`。
- 文件内头信息：同步修改 `.npz` 顶层字段 `type` 以及 `data_info["sample_type"]`。
- 已经是 `QJ00` 的文件会被复核，但不会重复改名。

设计原则：

- 先预演，再执行。`DRY_RUN = True` 时只打印计划，不写文件。
- 标签匹配使用边界约束，避免把 `QJ05`、`QJ001` 这类不同标签误改。
- 写 `.npz` 时使用同目录临时文件替换，减少中途失败造成半写入文件的风险。
- 执行后重新扫描复核，确认不再存在 `QJ` 残留。"""
    ),
    code(
        """from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Any

import numpy as np


# =========================
# 配置区：迁移到其他批次时优先修改这里
# =========================

# 当前 notebook 有两种常见启动方式：
# 1. 在项目根目录 E:/codes/ZZ-BK 启动 Jupyter，此时相对路径 DATA09/v0-qj 可直接找到数据；
# 2. 在 notebooks/ 目录内打开 notebook，此时相对路径会被解析到 notebooks/DATA09/v0-qj，导致扫不到文件。
# 因此下面用 find_project_root() 向上查找包含 DATA09 的项目根目录，减少迁移时的路径错误。
DATA_DIR = Path(\"DATA09/v0-qj\")

# OLD_LABEL 是历史不规范标签，NEW_LABEL 是目标标准标签。
# 本次任务只允许 QJ -> QJ00；其他标签不受影响。
OLD_LABEL = \"QJ\"
NEW_LABEL = \"QJ00\"

# 第一次迁移建议保持 True，检查输出计划无误后再改成 False 执行。
# 当前仓库开发时已用同一逻辑实际执行过规范化；再次运行会显示无待处理项。
DRY_RUN = True

# 若目标文件已存在，默认停止，避免覆盖已有样本。
# 只有在明确确认两个文件代表同一样本且需要覆盖时，才应改为 True。
ALLOW_OVERWRITE = False


def find_project_root(start: Path | None = None) -> Path:
    \"\"\"从当前工作目录向上查找项目根目录；找不到时回退到当前工作目录。\"\"\"
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / \"DATA09\").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR
DATA_DIR = DATA_DIR.resolve()
print(f\"DATA_DIR = {DATA_DIR}\")
print(f\"DRY_RUN = {DRY_RUN}\")
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f\"数据目录不存在: {DATA_DIR}\\n\"
        \"请检查 DATA_DIR，或确认 Jupyter 当前工作目录是否位于项目目录 E:/codes/ZZ-BK 内。\"
    )
"""
    ),
    code(
        """# =========================
# 工具函数：标签识别与 .npz 头信息读写
# =========================


def _normalize_scalar_label(value: Any, old_label: str, new_label: str) -> tuple[Any, bool]:
    \"\"\"只把完整标签 old_label 替换为 new_label，避免误伤 QJ05 / QJ001。\"\"\"
    if isinstance(value, bytes):
        text = value.decode(\"utf-8\")
        return (new_label.encode(\"utf-8\"), True) if text == old_label else (value, False)
    if isinstance(value, str):
        return (new_label, True) if value == old_label else (value, False)
    return value, False


def _normalize_data_info(data_info: Any, old_label: str, new_label: str) -> tuple[Any, bool]:
    \"\"\"规范化 data_info 字典中的 sample_type，并保持其他元数据不变。\"\"\"
    if not isinstance(data_info, dict):
        return data_info, False

    changed = False
    new_info = dict(data_info)
    if \"sample_type\" in new_info:
        new_value, item_changed = _normalize_scalar_label(new_info[\"sample_type\"], old_label, new_label)
        new_info[\"sample_type\"] = new_value
        changed = changed or item_changed
    return new_info, changed


def load_npz_payload(path: Path) -> dict[str, Any]:
    \"\"\"完整读取 .npz 内容，关闭句柄后再写回，避免 Windows 下文件占用。\"\"\"
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def normalize_npz_header(path: Path, old_label: str, new_label: str, dry_run: bool = True) -> dict[str, bool]:
    \"\"\"同步规范化 .npz 顶层 type 和 data_info.sample_type。\"\"\"
    payload = load_npz_payload(path)
    changed_type = False
    changed_data_info = False

    if \"type\" in payload:
        arr = payload[\"type\"]
        if getattr(arr, \"shape\", None) == ():
            new_value, changed_type = _normalize_scalar_label(arr.item(), old_label, new_label)
            if changed_type:
                payload[\"type\"] = np.array(new_value)

    if \"data_info\" in payload:
        arr = payload[\"data_info\"]
        if getattr(arr, \"shape\", None) == ():
            new_info, changed_data_info = _normalize_data_info(arr.item(), old_label, new_label)
            if changed_data_info:
                payload[\"data_info\"] = np.array(new_info, dtype=object)

    if (changed_type or changed_data_info) and not dry_run:
        # 写入同目录临时文件，然后 os.replace 原子替换原文件。
        # 这样即使中途失败，原始 .npz 通常仍保留在原路径。
        fd, tmp_name = tempfile.mkstemp(prefix=path.stem + \".\", suffix=\".tmp.npz\", dir=path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(tmp_path, **payload)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return {\"type\": changed_type, \"data_info\": changed_data_info}
"""
    ),
    code(
        """# =========================
# 扫描目录并生成改名计划
# =========================


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path


def build_rename_plan(data_dir: Path, old_label: str, new_label: str) -> list[RenamePlan]:
    \"\"\"仅改以 OLD_LABEL + '-' 开头的文件名，不改已经规范化或其他标签的文件。\"\"\"
    plans: list[RenamePlan] = []
    prefix = old_label + \"-\"
    new_prefix = new_label + \"-\"
    for path in sorted(data_dir.glob(\"*.npz\")):
        if path.name.startswith(prefix):
            target = path.with_name(new_prefix + path.name[len(prefix):])
            plans.append(RenamePlan(path, target))
    return plans


rename_plans = build_rename_plan(DATA_DIR, OLD_LABEL, NEW_LABEL)
all_npz_files = sorted(DATA_DIR.glob(\"*.npz\"))

print(f\".npz 文件总数: {len(all_npz_files)}\")
print(f\"需要改名的文件数: {len(rename_plans)}\")
for plan in rename_plans[:10]:
    print(f\"  {plan.source.name} -> {plan.target.name}\")
if len(rename_plans) > 10:
    print(f\"  ... 其余 {len(rename_plans) - 10} 个省略\")
"""
    ),
    code(
        """# =========================
# 预执行检查：目标重名检测
# =========================


conflicts = [plan for plan in rename_plans if plan.target.exists() and plan.target != plan.source]
if conflicts and not ALLOW_OVERWRITE:
    print(\"发现目标文件已存在，已停止。请人工确认后再决定是否允许覆盖：\")
    for plan in conflicts[:20]:
        print(f\"  {plan.source.name} -> {plan.target.name}\")
    raise FileExistsError(f\"目标重名数量: {len(conflicts)}\")

print(\"重名检查通过。\")"""
    ),
    code(
        """# =========================
# 执行规范化
# =========================


renamed_count = 0
header_type_count = 0
header_data_info_count = 0

# 先处理所有文件内头信息，包括已经叫 QJ00 但头信息仍可能是 QJ 的文件。
# 这样即使某些文件不需要改名，也能被统一复核和修正。
for path in all_npz_files:
    result = normalize_npz_header(path, OLD_LABEL, NEW_LABEL, dry_run=DRY_RUN)
    header_type_count += int(result[\"type\"])
    header_data_info_count += int(result[\"data_info\"])

# 再改文件名。先改内容再改名，日志更容易按原始问题定位。
for plan in rename_plans:
    if DRY_RUN:
        continue
    if plan.target.exists() and ALLOW_OVERWRITE:
        plan.target.unlink()
    plan.source.rename(plan.target)
    renamed_count += 1

print(\"执行结果：\")
print(f\"  文件名改名: {renamed_count if not DRY_RUN else 0} / {len(rename_plans)}\")
print(f\"  type 字段待改/已改: {header_type_count}\")
print(f\"  data_info.sample_type 待改/已改: {header_data_info_count}\")
print(\"提示：DRY_RUN=True 时上面的字段数量表示待修改数量，不会写盘。\")"""
    ),
    code(
        """# =========================
# 复核：确认文件名和头信息均已统一
# =========================


def inspect_label_state(data_dir: Path, old_label: str, new_label: str) -> dict[str, Any]:
    \"\"\"返回当前目录的标签状态摘要；迁移后应 old_* 全部为 0。\"\"\"
    files = sorted(data_dir.glob(\"*.npz\"))
    old_name_files = [p.name for p in files if p.name.startswith(old_label + \"-\")]
    new_name_files = [p.name for p in files if p.name.startswith(new_label + \"-\")]
    old_type_files: list[str] = []
    old_sample_type_files: list[str] = []
    other_sample_types: dict[str, int] = {}

    for path in files:
        payload = load_npz_payload(path)
        type_value = payload.get(\"type\")
        if type_value is not None and getattr(type_value, \"shape\", None) == ():
            if type_value.item() == old_label:
                old_type_files.append(path.name)

        data_info = payload.get(\"data_info\")
        if data_info is not None and getattr(data_info, \"shape\", None) == ():
            item = data_info.item()
            if isinstance(item, dict):
                sample_type = item.get(\"sample_type\")
                other_sample_types[str(sample_type)] = other_sample_types.get(str(sample_type), 0) + 1
                if sample_type == old_label:
                    old_sample_type_files.append(path.name)

    return {
        \"total_npz\": len(files),
        \"new_name_count\": len(new_name_files),
        \"old_name_count\": len(old_name_files),
        \"old_type_count\": len(old_type_files),
        \"old_sample_type_count\": len(old_sample_type_files),
        \"sample_type_distribution\": other_sample_types,
        \"old_name_examples\": old_name_files[:10],
        \"old_type_examples\": old_type_files[:10],
        \"old_sample_type_examples\": old_sample_type_files[:10],
    }


state = inspect_label_state(DATA_DIR, OLD_LABEL, NEW_LABEL)
state"""
    ),
    md(
        """## 头文件完整性检查

本节用于批量检查所有 `.npz` 的头文件/元数据字段是否完整、是否彼此一致。

重点检查项：

- `starttime`：片段开始时间，通常来自文件名中的采集时间。
- `arrival_time`：初至时间/到时，用于后续按到时截取样本。
- `sample_rate`：采样率，当前批次应为 1 MHz。
- `npts`、`comm_count`、`phase_data.shape[0]`：样本点数应一致。
- `channel_count`、`channel_names`、`phase_data.shape[1]`：通道数量应一致。
- `data_info.duration_seconds`：时长应约等于 `npts / sample_rate`。
- `type` 与 `data_info.sample_type`：标签应一致。

检查只读数据，不会修改文件。异常明细会保存到 `outputs/DATA09_v0-qj_header_audit.csv`。"""
    ),
    code(
        """# =========================
# 头文件完整性检查：字段存在性、字段一致性、时间与时长合理性
# =========================

from datetime import datetime
import math
import pandas as pd


# 这些字段是当前 DATA09 断丝样本后续处理依赖的最小头文件集合。
# 如果迁移到其他数据格式，可以先调整 REQUIRED_HEADER_KEYS，再复用下面的审计函数。
REQUIRED_HEADER_KEYS = [
    \"phase_data\",
    \"channels\",
    \"channel_names\",
    \"channel_count\",
    \"sample_rate\",
    \"comm_count\",
    \"npts\",
    \"timestamp\",
    \"starttime\",
    \"arrival_time\",
    \"type\",
    \"data_info\",
]

# 当前批次采样率来自文件名的 1000K 和头文件 sample_rate，预期为 1_000_000 Hz。
# 若迁移到 500K 或 200K 批次，应同步修改 EXPECTED_SAMPLE_RATE。
EXPECTED_SAMPLE_RATE = 1_000_000.0

# 浮点计算和导出软件可能带来极小误差，因此时长一致性不做严格相等。
DURATION_TOLERANCE_SECONDS = 1e-6


def scalar_or_none(payload: dict[str, Any], key: str) -> Any:
    \"\"\"读取 .npz 标量字段；缺失或非标量时返回 None，避免审计中断。\"\"\"
    arr = payload.get(key)
    if arr is None or getattr(arr, \"shape\", None) != ():
        return None
    return arr.item()


def parse_data09_time(value: Any) -> datetime | None:
    \"\"\"解析 DATA09 中常见的 starttime/arrival_time 字符串格式。\"\"\"
    if value is None:
        return None
    text = str(value)
    for fmt in (\"%Y%m%dT%H%M%S.%f\", \"%Y%m%d%H%M%S.%f\", \"%Y%m%dT%H%M%S\", \"%Y%m%d%H%M%S\"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def audit_one_npz_header(path: Path) -> dict[str, Any]:
    \"\"\"审计单个 .npz 文件的头文件信息，并把问题汇总到 issues 字段。\"\"\"
    payload = load_npz_payload(path)
    issues: list[str] = []

    missing_keys = [key for key in REQUIRED_HEADER_KEYS if key not in payload]
    if missing_keys:
        issues.append(\"missing_keys=\" + \",\".join(missing_keys))

    phase_shape = getattr(payload.get(\"phase_data\"), \"shape\", None)
    channels_shape = getattr(payload.get(\"channels\"), \"shape\", None)
    channel_names = payload.get(\"channel_names\")
    channel_names_list = channel_names.tolist() if channel_names is not None else []

    sample_rate = scalar_or_none(payload, \"sample_rate\")
    npts = scalar_or_none(payload, \"npts\")
    comm_count = scalar_or_none(payload, \"comm_count\")
    channel_count = scalar_or_none(payload, \"channel_count\")
    type_value = scalar_or_none(payload, \"type\")
    starttime_value = scalar_or_none(payload, \"starttime\")
    arrival_time_value = scalar_or_none(payload, \"arrival_time\")

    data_info = scalar_or_none(payload, \"data_info\")
    if not isinstance(data_info, dict):
        data_info = {}
        issues.append(\"data_info_not_dict\")

    info_npts = data_info.get(\"npts\")
    info_length = data_info.get(\"length\")
    info_duration = data_info.get(\"duration_seconds\")
    info_sample_type = data_info.get(\"sample_type\")
    info_starttime = data_info.get(\"starttime\")
    info_arrival_time = data_info.get(\"arrival_time\")
    info_channel_count = data_info.get(\"channel_count\")

    # 点数一致性：后续滑窗和按到时截取都依赖 npts 与实际数组长度一致。
    phase_rows = phase_shape[0] if phase_shape and len(phase_shape) >= 1 else None
    phase_cols = phase_shape[1] if phase_shape and len(phase_shape) >= 2 else 1
    if npts is not None and phase_rows is not None and int(npts) != int(phase_rows):
        issues.append(\"npts_mismatch_phase_rows\")
    if comm_count is not None and npts is not None and int(comm_count) != int(npts):
        issues.append(\"comm_count_mismatch_npts\")
    if info_npts is not None and npts is not None and int(info_npts) != int(npts):
        issues.append(\"data_info_npts_mismatch\")
    if info_length is not None and npts is not None and int(info_length) != int(npts):
        issues.append(\"data_info_length_mismatch\")

    # 通道一致性：当前样本是双通道，channel_count、channel_names 和 phase_data 列数应互相吻合。
    if channel_count is not None and phase_cols is not None and int(channel_count) != int(phase_cols):
        issues.append(\"channel_count_mismatch_phase_cols\")
    if channel_count is not None and channel_names_list and int(channel_count) != len(channel_names_list):
        issues.append(\"channel_count_mismatch_channel_names\")
    if info_channel_count is not None and channel_count is not None and int(info_channel_count) != int(channel_count):
        issues.append(\"data_info_channel_count_mismatch\")
    if channels_shape is not None and phase_shape is not None and tuple(channels_shape) != tuple(phase_shape):
        issues.append(\"channels_shape_mismatch_phase_data\")

    # 标签一致性：文件名、顶层 type、data_info.sample_type 应指向同一类样本。
    filename_label = path.name.split(\"-\", 1)[0]
    if type_value is not None and str(type_value) != filename_label:
        issues.append(\"type_mismatch_filename_label\")
    if info_sample_type is not None and type_value is not None and str(info_sample_type) != str(type_value):
        issues.append(\"data_info_sample_type_mismatch_type\")

    # 采样率与时长一致性：duration_seconds 应约等于 npts / sample_rate。
    expected_duration = None
    if sample_rate is not None and npts is not None and float(sample_rate) > 0:
        expected_duration = float(npts) / float(sample_rate)
        if not math.isclose(float(sample_rate), EXPECTED_SAMPLE_RATE, rel_tol=0.0, abs_tol=1e-6):
            issues.append(\"unexpected_sample_rate\")
        if info_duration is not None and not math.isclose(float(info_duration), expected_duration, rel_tol=0.0, abs_tol=DURATION_TOLERANCE_SECONDS):
            issues.append(\"duration_mismatch_npts_over_sample_rate\")

    # 时间字段：starttime 与 arrival_time 必须能解析，且初至时间应落在片段覆盖区间内。
    start_dt = parse_data09_time(starttime_value)
    arrival_dt = parse_data09_time(arrival_time_value)
    if start_dt is None:
        issues.append(\"starttime_parse_failed\")
    if arrival_dt is None:
        issues.append(\"arrival_time_parse_failed\")
    arrival_offset_seconds = None
    if start_dt is not None and arrival_dt is not None:
        arrival_offset_seconds = (arrival_dt - start_dt).total_seconds()
        if expected_duration is not None and not (0 <= arrival_offset_seconds <= expected_duration):
            issues.append(\"arrival_time_outside_segment\")

    if info_starttime is not None and starttime_value is not None and str(info_starttime) != str(starttime_value):
        issues.append(\"data_info_starttime_mismatch\")
    if info_arrival_time is not None and arrival_time_value is not None and str(info_arrival_time) != str(arrival_time_value):
        issues.append(\"data_info_arrival_time_mismatch\")

    return {
        \"file_name\": path.name,
        \"file_size_bytes\": path.stat().st_size,
        \"missing_keys\": \",\".join(missing_keys),
        \"type\": type_value,
        \"data_info_sample_type\": info_sample_type,
        \"sample_rate\": sample_rate,
        \"npts\": npts,
        \"comm_count\": comm_count,
        \"phase_shape\": str(phase_shape),
        \"channels_shape\": str(channels_shape),
        \"channel_count\": channel_count,
        \"channel_names\": \"|\".join(map(str, channel_names_list)),
        \"starttime\": starttime_value,
        \"arrival_time\": arrival_time_value,
        \"arrival_offset_seconds\": arrival_offset_seconds,
        \"duration_seconds\": info_duration,
        \"expected_duration_seconds\": expected_duration,
        \"issues\": \";\".join(issues),
        \"issue_count\": len(issues),
    }


audit_records = [audit_one_npz_header(path) for path in sorted(DATA_DIR.glob(\"*.npz\"))]
header_audit_df = pd.DataFrame(audit_records)
audit_output_path = Path(\"outputs/DATA09_v0-qj_header_audit.csv\")
audit_output_path.parent.mkdir(parents=True, exist_ok=True)
header_audit_df.to_csv(audit_output_path, index=False, encoding=\"utf-8-sig\")

print(f\"审计文件数: {len(header_audit_df)}\")
if header_audit_df.empty:
    raise FileNotFoundError(
        f\"未在 {DATA_DIR} 下找到 .npz 文件。\\n\"
        \"请检查 DATA_DIR 是否指向真实数据目录；如果 notebook 从 notebooks/ 目录启动，建议运行配置单元后确认打印出的 DATA_DIR。\"
    )
print(f\"存在异常的文件数: {(header_audit_df['issue_count'] > 0).sum()}\")
print(f\"审计明细已保存: {audit_output_path.resolve()}\")

summary_columns = [
    \"sample_rate\",
    \"npts\",
    \"channel_count\",
    \"duration_seconds\",
    \"arrival_offset_seconds\",
    \"issue_count\",
]
display(header_audit_df[summary_columns].describe(include=\"all\"))
display(header_audit_df.loc[header_audit_df[\"issue_count\"] > 0].head(20))
"""
    ),
    md(
        """## 迁移注意事项

- 如果迁移到 `v05-qj` 等目录，优先确认目标标签是否应为 `QJ05`，不要直接套用 `QJ00`。
- 若 `.npz` 内部字段名变化，重点检查 `type` 与 `data_info.sample_type` 是否仍存在；不存在时可扩展 `normalize_npz_header()`。
- 如需处理 `.csv`、`.json` 或其他文本头文件，应新增独立函数，不要把二进制 `.npz` 当文本替换。
- 大批量执行前建议先备份数据目录，或在 Git/LFS/外部数据版本系统中记录迁移前状态。"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK_PATH.resolve())
