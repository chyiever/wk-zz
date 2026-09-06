"""PCCP断丝特征挖掘一键流程。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .bootstrap_stability import run_bootstrap_stability
from .classification_test import run_classification_tests
from .combination_search import evaluate_mrmr_prefixes, relief_like_ranking, sequential_forward_search
from .config import MiningConfig, default_feature_inputs
from .cross_flow_analysis import evaluate_cross_flow_features
from .data_loader import load_feature_dataset
from .distribution_analysis import (
    plot_feature_kde_by_label,
    plot_projection,
    run_pca_projection,
    run_umap_projection,
    select_distribution_features,
)
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

    distribution_features = select_distribution_features(frame, feature_columns, top_n=cfg.distribution_top_n)
    write_csv(distribution_features, output_dir / "six_class_distribution_features.csv")
    pca_projection, pca_explained = run_pca_projection(
        frame,
        feature_columns,
        max_rows=cfg.projection_max_rows,
        random_state=cfg.random_state,
    )
    write_csv(pca_projection, output_dir / "six_class_pca_projection.csv")
    write_csv(pca_explained, output_dir / "six_class_pca_explained_variance.csv")
    umap_projection = run_umap_projection(
        frame,
        feature_columns,
        max_rows=min(cfg.projection_max_rows, 5000),
        random_state=cfg.random_state,
    )
    write_csv(umap_projection, output_dir / "six_class_umap_projection.csv")
    plot_top_feature_boxplots(
        frame,
        distribution_features["feature"].head(12).tolist(),
        output_dir / "plots/six_class_feature_boxplots.png",
        max_features=12,
    )
    plot_feature_kde_by_label(
        frame,
        distribution_features["feature"].head(8).tolist(),
        output_dir / "plots/six_class_feature_kde.png",
    )
    plot_projection(pca_projection, "PC1", "PC2", output_dir / "plots/six_class_pca.png", "六类特征PCA投影")
    if not umap_projection.empty:
        plot_projection(umap_projection, "UMAP1", "UMAP2", output_dir / "plots/six_class_umap.png", "六类特征UMAP投影")

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

    mrmr_prefix = evaluate_mrmr_prefixes(
        frame,
        mrmr_rank,
        feature_columns,
        counts=cfg.combination_feature_counts,
        random_state=cfg.random_state,
    )
    write_csv(mrmr_prefix, output_dir / "feature_combination_mrmr_prefix.csv")
    relief_rank = relief_like_ranking(frame, feature_columns, random_state=cfg.random_state)
    write_csv(relief_rank, output_dir / "relieff_rank.csv")
    sfs_candidates = list(dict.fromkeys(mrmr_rank["feature"].head(cfg.sfs_candidate_count).tolist() + relief_rank["feature"].head(cfg.sfs_candidate_count).tolist()))
    sfs_path = sequential_forward_search(
        frame,
        sfs_candidates,
        feature_columns,
        max_selected=cfg.sfs_max_selected,
        random_state=cfg.random_state,
    )
    write_csv(sfs_path, output_dir / "sfs_selection_path.csv")
    combination_summary = mrmr_prefix.copy()
    if not sfs_path.empty:
        combination_summary = pd.concat(
            [
                combination_summary,
                sfs_path.rename(columns={"step": "sfs_step"})[
                    ["method", "comparison", "feature_count", "cv_auc_abs", "lda_separation", "selected_features"]
                ],
            ],
            ignore_index=True,
            sort=False,
        )
    write_csv(combination_summary, output_dir / "feature_combination_search.csv")

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

    classification_results, feature_recommendations = run_classification_tests(
        frame,
        feature_discrimination=discrimination,
        final_ranking=final_ranking,
        output_dir=output_dir / "classification",
        feature_counts=cfg.classification_feature_counts,
        max_train_rows_per_class=cfg.classification_max_train_rows_per_class,
        random_state=cfg.random_state,
    )
    write_csv(classification_results, output_dir / "classification_test_results.csv")
    write_csv(feature_recommendations, output_dir / "comparison_feature_recommendations.csv")

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
