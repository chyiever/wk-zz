"""Validate feature computation on one real sample."""

from __future__ import annotations

from pathlib import Path

from .params import DEFAULT_FEATURE_PARAMS
from .pipeline import compute_feature_table_for_paths, validate_feature_table


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    sample_path = sorted((workspace / "data").rglob("*.npz"))[0]
    output_dir = workspace / "outputs" / "fea_cpt_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = compute_feature_table_for_paths(
        [sample_path],
        params=DEFAULT_FEATURE_PARAMS,
        show_progress=False,
        log_dir=output_dir / "logs",
    )
    summary = validate_feature_table(frame)
    frame.to_csv(output_dir / "one_sample_feature_table.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "one_sample_feature_validation.csv", index=False, encoding="utf-8-sig")
    print(sample_path)
    print(frame.iloc[:, :12].to_string(index=False))
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
