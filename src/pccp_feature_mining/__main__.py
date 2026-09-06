"""命令行入口：python -m pccp_feature_mining。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MiningConfig, default_feature_inputs
from .run_all import run_pccp_feature_mining


def main() -> None:
    parser = argparse.ArgumentParser(description="PCCP断丝特征挖掘一键流程")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/PCCP_feature_mining"))
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--max-rows-per-label", type=int, default=None, help="快速调试用；正式运行不设置")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = MiningConfig(
        output_dir=args.output_dir,
        bootstrap_rounds=args.bootstrap_rounds,
        max_rows_per_label=args.max_rows_per_label,
        random_state=args.random_state,
        feature_inputs=default_feature_inputs(),
    )
    summary = run_pccp_feature_mining(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
