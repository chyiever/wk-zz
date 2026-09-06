"""结果文件写出与简要报告生成。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """统一CSV编码，便于Excel直接打开中文字段。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _table_text(frame: pd.DataFrame) -> str:
    """生成无额外依赖的表格文本，避免强制安装tabulate。"""

    if frame.empty:
        return "（空表）"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_markdown_summary(
    output_path: Path,
    dataset_summary: pd.DataFrame,
    final_ranking: pd.DataFrame,
    load_report: pd.DataFrame,
) -> None:
    """生成给人工复核用的轻量Markdown摘要。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_a = final_ranking[final_ranking["feature_grade"].eq("A")].head(30)
    skipped = load_report[load_report["status"].ne("ok")]
    lines = [
        "# PCCP断丝特征挖掘结果摘要",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 总特征数：{len(final_ranking)}",
        f"- A级特征数：{int(final_ranking['feature_grade'].eq('A').sum())}",
        f"- B级特征数：{int(final_ranking['feature_grade'].eq('B').sum())}",
        f"- C级特征数：{int(final_ranking['feature_grade'].eq('C').sum())}",
        "",
        "## 数据摘要",
        "",
        _table_text(dataset_summary),
        "",
        "## Top A级特征",
        "",
        _table_text(
            top_a[
                [
                    "final_rank",
                    "feature",
                    "final_score",
                    "bk_nonbk_score",
                    "bk_qj_score",
                    "bootstrap_stability_score",
                    "cross_flow_score",
                ]
            ]
        ),
    ]
    if not skipped.empty:
        lines.extend(["", "## 跳过或失败的输入文件", "", _table_text(skipped)])
    output_path.write_text("\n".join(lines), encoding="utf-8")
