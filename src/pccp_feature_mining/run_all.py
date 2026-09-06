"""PCCP断丝特征挖掘一键流程。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .bootstrap_stability import run_bootstrap_stability
from .config import MiningConfig, default_feature_inputs
from .cross_flow_analysis import evaluate_cross_flow_features
from .data_loader import load_feature_dataset
from .feature_discrimination import evaluate_feature_discrimination
from .feature_redundancy import build_correlation_clusters, compute_correlation_matrix, high_correlation_pairs
from .feature_selection import build_final_ranking, build_relevance_series, run_mrmr_ranking
from .quality_control import build_dataset_summary, build_feature_quality, feature_list_frame
from .report_generator import write_csv, write_markdown_summary
from .visualization import plot_final_ranking, plot_top_feature_boxplots


def run_pccp_feature_mining(config: MiningConfig | None = None) -> dict[str, object]:
    """运行完整挖掘流程并写出方案要求的核心结果文件。"""

    cfg = MiningConfig(feature_inputs=default_feature_inputs()) if config is None else config
    feature_inputs = cfg.feature_inputs or default_feature_inputs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_feature_dataset(
        feature_inputs=feature_inputs,
        min_feature_csv_columns=cfg.min_feature_csv_columns,
        max_rows_per_label=cfg.max_rows_per_label,
        random_state=cfg.random_state,
    )
    frame = dataset.frame
    feature_columns = list(dataset.feature_columns)

    dataset_summary = build_dataset_summary(frame)
    feature_quality = build_feature_quality(frame, feature_columns)
    feature_list = feature_list_frame(feature_columns)
    write_csv(dataset.load_report, output_dir / "load_report.csv")
    write_csv(dataset_summary, output_dir / "dataset_summary.csv")
    write_csv(feature_list, output_dir / "feature_list.csv")
    write_csv(feature_quality, output_dir / "quality_report.csv")

    discrimination = evaluate_feature_discrimination(frame, feature_columns, random_state=cfg.random_state)
    write_csv(discrimination, output_dir / "feature_discrimination.csv")

    relevance = build_relevance_series(discrimination, comparison="BK_NONBK")
    corr = compute_correlation_matrix(
        frame,
        feature_columns,
        method="spearman",
        max_rows=cfg.max_correlation_rows,
        random_state=cfg.random_state,
    )
    corr.to_csv(output_dir / "spearman_correlation_matrix.csv", encoding="utf-8-sig")
    high_pairs = high_correlation_pairs(corr, threshold=cfg.correlation_threshold)
    write_csv(high_pairs, output_dir / "high_correlation_pairs.csv")
    clusters = build_correlation_clusters(corr, relevance=relevance, threshold=cfg.correlation_threshold)
    write_csv(clusters, output_dir / "correlation_cluster.csv")

    mrmr_rank = run_mrmr_ranking(
        relevance=relevance,
        corr=corr,
        top_n=cfg.mrmr_top_n,
        redundancy_weight=cfg.mrmr_redundancy_weight,
    )
    write_csv(mrmr_rank, output_dir / "mrmr_rank.csv")

    stability = run_bootstrap_stability(
        frame,
        feature_columns,
        rounds=cfg.bootstrap_rounds,
        top_ks=cfg.bootstrap_top_ks,
        random_state=cfg.random_state,
    )
    write_csv(stability, output_dir / "feature_stability.csv")

    cross_flow = evaluate_cross_flow_features(frame, feature_columns)
    write_csv(cross_flow, output_dir / "cross_flow_feature.csv")

    final_ranking = build_final_ranking(
        feature_columns=feature_columns,
        discrimination=discrimination,
        stability=stability,
        cross_flow=cross_flow,
        redundancy=clusters,
        mrmr_rank=mrmr_rank,
    )
    write_csv(final_ranking, output_dir / "final_feature_ranking.csv")

    plot_top_feature_boxplots(
        frame,
        final_ranking.head(cfg.max_plot_features)["feature"].tolist(),
        output_dir / "plots/top_feature_boxplots.png",
        max_features=min(12, cfg.max_plot_features),
    )
    plot_final_ranking(final_ranking, output_dir / "plots/final_feature_ranking_top30.png", top_n=30)
    write_markdown_summary(output_dir / "summary_report.md", dataset_summary, final_ranking, dataset.load_report)

    return {
        "output_dir": str(output_dir.resolve()),
        "rows": int(len(frame)),
        "features": int(len(feature_columns)),
        "labels": sorted(frame["source_label"].astype(str).unique().tolist()),
        "top_features": final_ranking.head(20)["feature"].tolist(),
    }
