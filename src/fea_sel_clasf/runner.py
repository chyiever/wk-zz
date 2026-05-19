"""End-to-end cross-condition experiment runner."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_io import load_feature_table
from .evaluator import confusion_matrix_frame, evaluate_binary_predictions, write_metrics_json
from .feature_filter import run_stage_a_filter, slice_stage_a_scores
from .feature_wrapper import run_rfecv_selection
from .model_zoo import search_best_model
from .split_protocol import build_experiment_specs
from .thresholding import choose_threshold_from_oof


def _train_mask(frame: pd.DataFrame, domains: tuple[str, ...]) -> pd.Series:
    return (frame["split"] == "train") & frame["base_domain"].isin(domains)


def _test_mask(frame: pd.DataFrame, domain: str) -> pd.Series:
    return (frame["split"] == "test") & (frame["base_domain"] == domain)


def _safe_json(value: object) -> object:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _write_pickle(path: Path, obj: object) -> None:
    joblib.dump(obj, path)


def run_cross_condition_experiments(
    input_csv: str | Path,
    output_root: str | Path,
    random_state: int = 42,
    run_mode: str = "cpu",
    top_k_candidates: tuple[int, ...] = (60, 80, 120),
    target_recall: float = 0.95,
) -> pd.DataFrame:
    """
    Run all seven cross-condition experiments and write the required artifacts.

    The function assumes the input CSV contains both train and test splits. The
    split is inferred from ``split`` or ``folder_name``; the base physical
    domain is inferred from ``folder_name`` or ``sample_name``.
    """
    table = load_feature_table(input_csv)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    manifest = {
        "input_csv": str(Path(input_csv).resolve()),
        "output_root": str(output_root.resolve()),
        "random_state": int(random_state),
        "run_mode": str(run_mode),
        "target_recall": float(target_recall),
        "top_k_candidates": list(top_k_candidates),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiments": [],
    }

    for spec in build_experiment_specs():
        exp_dir = output_root / spec.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        train_mask = _train_mask(table.frame, spec.train_domains)
        if not bool(train_mask.any()):
            raise ValueError(f"no training rows found for {spec.name}")

        stage_a_rows: list[pd.DataFrame] = []
        rfecv_rows: list[pd.DataFrame] = []
        model_candidates: list[dict[str, object]] = []

        best_bundle: dict[str, object] | None = None

        stage_a_scores = run_stage_a_filter(
            table=table,
            train_mask=train_mask,
            random_state=random_state,
        )

        for top_k in top_k_candidates:
            stage_a = slice_stage_a_scores(stage_a_scores, top_k=top_k)
            stage_a_df = stage_a.scores.copy()
            stage_a_df["experiment"] = spec.name
            stage_a_df["top_k"] = top_k
            stage_a_rows.append(stage_a_df)

            rfecv = run_rfecv_selection(
                table=table,
                train_mask=train_mask,
                candidate_features=stage_a.selected_features,
                random_state=random_state,
            )
            rfecv_df = rfecv.summary.copy()
            rfecv_df["experiment"] = spec.name
            rfecv_df["top_k"] = top_k
            rfecv_rows.append(rfecv_df)

            model = search_best_model(
                table=table,
                train_mask=train_mask,
                selected_features=rfecv.selected_features,
                random_state=random_state,
                run_mode=run_mode,
            )
            threshold = choose_threshold_from_oof(model.oof_label, model.oof_probability, target_recall=target_recall)

            candidate_bundle = {
                "top_k": top_k,
                "stage_a": stage_a,
                "rfecv": rfecv,
                "model": model,
                "threshold": threshold,
                "selection_key": (
                    float(model.cv_results.iloc[0]["mean_recall_pos"]),
                    float(model.cv_results.iloc[0]["mean_balanced_accuracy"]),
                    -float(model.cv_results.iloc[0]["std_recall_pos"]),
                    -float(model.cv_results.iloc[0]["std_balanced_accuracy"]),
                ),
            }
            model_candidates.append(
                {
                    "experiment": spec.name,
                    "top_k": top_k,
                    "model_name": model.model_name,
                    "threshold": threshold.threshold,
                    "threshold_method": threshold.method,
                    "cv_mean_recall_pos": float(model.cv_results.iloc[0]["mean_recall_pos"]),
                    "cv_mean_balanced_accuracy": float(model.cv_results.iloc[0]["mean_balanced_accuracy"]),
                    "cv_std_recall_pos": float(model.cv_results.iloc[0]["std_recall_pos"]),
                    "cv_std_balanced_accuracy": float(model.cv_results.iloc[0]["std_balanced_accuracy"]),
                    "selected_feature_count": len(rfecv.selected_features),
                    "run_mode": str(run_mode),
                }
            )

            if best_bundle is None or candidate_bundle["selection_key"] > best_bundle["selection_key"]:
                best_bundle = candidate_bundle

        if best_bundle is None:
            raise RuntimeError(f"no candidate bundle built for {spec.name}")

        selected_stage_a = best_bundle["stage_a"]
        selected_rfecv = best_bundle["rfecv"]
        selected_model = best_bundle["model"]
        selected_threshold = best_bundle["threshold"]
        threshold_payload = {
            "threshold": selected_threshold.threshold,
            "recall_pos": selected_threshold.recall_pos,
            "fpr": selected_threshold.fpr,
            "balanced_accuracy": selected_threshold.balanced_accuracy,
            "method": selected_threshold.method,
            "target_recall": target_recall,
        }

        # Persist the winning artifacts.
        selected_stage_a.scores.to_csv(exp_dir / "feature_scores.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": list(selected_rfecv.selected_features)}).to_csv(
            exp_dir / "selected_features.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(model_candidates).to_csv(exp_dir / "model_candidates.csv", index=False, encoding="utf-8-sig")
        selected_rfecv.summary.to_csv(exp_dir / "rfecv_summary.csv", index=False, encoding="utf-8-sig")

        _write_pickle(exp_dir / "imputer.pkl", selected_model.imputer)
        _write_pickle(exp_dir / "scaler.pkl", selected_model.scaler)
        _write_pickle(exp_dir / "classifier.pkl", selected_model.classifier)

        write_metrics_json(exp_dir / "threshold.json", threshold_payload)

        exp_summary = []
        for domain in spec.test_domains:
            test_mask = _test_mask(table.frame, domain)
            if not bool(test_mask.any()):
                continue
            test_frame = table.frame.loc[test_mask].copy()
            x_test = test_frame.loc[:, list(selected_rfecv.selected_features)].apply(pd.to_numeric, errors="coerce")
            y_test = test_frame[table.label_column].to_numpy(dtype=int)
            prob = selected_model.pipeline.predict_proba(x_test)[:, 1]
            metrics = evaluate_binary_predictions(y_test, prob, selected_threshold.threshold)
            metrics_payload = dict(metrics)
            metrics_payload.update(
                {
                    "experiment": spec.name,
                    "test_domain": domain,
                    "train_domains": list(spec.train_domains),
                    "model_name": selected_model.model_name,
                    "top_k": int(best_bundle["top_k"]),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "selected_feature_count": len(selected_rfecv.selected_features),
                    "random_state": int(random_state),
                    "run_mode": str(run_mode),
                }
            )
            write_metrics_json(exp_dir / f"metrics_{domain}.json", metrics_payload)
            confusion_matrix_frame(y_test, prob, selected_threshold.threshold).to_csv(
                exp_dir / f"confusion_matrix_{domain}.csv",
                encoding="utf-8-sig",
            )
            exp_summary.append(metrics_payload)
            summary_rows.append(metrics_payload)

        summary_df = pd.DataFrame(exp_summary)
        if not summary_df.empty:
            summary_df.to_csv(exp_dir / "summary.csv", index=False, encoding="utf-8-sig")

        run_manifest = {
            "experiment": spec.name,
            "train_domains": list(spec.train_domains),
            "test_domains": list(spec.test_domains),
            "selected_top_k": int(best_bundle["top_k"]),
            "model_name": selected_model.model_name,
            "model_params": {k: _safe_json(v) for k, v in selected_model.params.items()},
            "threshold": threshold_payload,
            "selected_features": list(selected_rfecv.selected_features),
            "n_train_rows": int(train_mask.sum()),
            "n_test_rows": {domain: int(_test_mask(table.frame, domain).sum()) for domain in spec.test_domains},
            "random_state": int(random_state),
            "run_mode": str(run_mode),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (exp_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest["experiments"].append(run_manifest)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "summary_cross_condition.csv", index=False, encoding="utf-8-sig")
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
