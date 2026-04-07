import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import (
    calibration_summary,
    compute_binary_metrics,
    compute_ece,
    save_json,
)
from protocol_utils import load_csv_rows, sha256_file, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Belge v1.3 uyumlu track-level evaluation contract.")
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregation-primary", type=str, default="mean_probability")
    parser.add_argument("--aggregation-secondary", type=str, default="majority_vote")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def learn_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold, best_f1


def learn_temperature(y_true, y_prob):
    clipped = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))

    candidate_temperatures = np.linspace(0.5, 5.0, 91)
    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in candidate_temperatures:
        calibrated_prob = 1.0 / (1.0 + np.exp(-(logits / temperature)))
        nll = -np.mean(
            y_true * np.log(np.clip(calibrated_prob, 1e-6, 1.0)) +
            (1 - y_true) * np.log(np.clip(1 - calibrated_prob, 1e-6, 1.0))
        )
        if nll < best_nll:
            best_nll = float(nll)
            best_temperature = float(temperature)

    return best_temperature


def apply_temperature(y_prob, temperature):
    clipped = np.clip(y_prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated = 1.0 / (1.0 + np.exp(-(logits / temperature)))
    return calibrated.astype(np.float32)


def enrich_predictions(predictions_path: Path, split_name: str):
    df = pd.read_csv(predictions_path)
    if "prob_fake" not in df.columns:
        raise ValueError(f"prob_fake kolonu bulunamadi: {predictions_path}")

    df = df.copy()
    df["split"] = split_name
    if "pred" not in df.columns:
        df["pred"] = (df["prob_fake"] >= 0.5).astype(np.int64)
    return df


def build_data_summary(global_manifest_rows, split_manifest_rows):
    split_lookup = {row["stem_id"]: row["split"] for row in split_manifest_rows}
    summary_rows = []
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in global_manifest_rows if split_lookup[row["stem_id"]] == split_name]
        by_family = {}
        for row in split_rows:
            family = "real" if row["origin_type"] == "real" else row["derived_family"]
            if family not in by_family:
                by_family[family] = {
                    "split": split_name,
                    "family": family,
                    "track_count": 0,
                    "sample_count": 0,
                    "patch_count": 0,
                }
            by_family[family]["track_count"] += 1
            by_family[family]["sample_count"] += 1
            by_family[family]["patch_count"] += int(row["patch_count"])
        summary_rows.extend(by_family.values())
    return pd.DataFrame(summary_rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_manifest_rows = load_csv_rows(args.global_manifest)
    split_manifest_rows = load_csv_rows(args.split_manifest)

    val_df = enrich_predictions(args.val_predictions, "val")
    test_df = enrich_predictions(args.test_predictions, "test")

    val_true = val_df["label"].to_numpy(dtype=np.int64)
    val_prob = val_df["prob_fake"].to_numpy(dtype=np.float32)
    test_true = test_df["label"].to_numpy(dtype=np.int64)
    test_prob = test_df["prob_fake"].to_numpy(dtype=np.float32)

    best_threshold, val_best_f1 = learn_best_threshold(val_true, val_prob)
    temperature = learn_temperature(val_true, val_prob)

    val_prob_cal = apply_temperature(val_prob, temperature)
    test_prob_cal = apply_temperature(test_prob, temperature)

    val_metrics_at_05 = compute_binary_metrics(val_true, val_prob_cal, threshold=0.5)
    val_metrics_best = compute_binary_metrics(val_true, val_prob_cal, threshold=best_threshold)
    test_metrics_at_05 = compute_binary_metrics(test_true, test_prob_cal, threshold=0.5)
    test_metrics_best = compute_binary_metrics(test_true, test_prob_cal, threshold=best_threshold)

    val_calibration = calibration_summary(val_true, val_prob, val_prob_cal)
    test_calibration = calibration_summary(test_true, test_prob, test_prob_cal)

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["prob_fake_calibrated"] = val_prob_cal
    test_df["prob_fake_calibrated"] = test_prob_cal
    val_df["pred_best_threshold"] = (val_prob_cal >= best_threshold).astype(np.int64)
    test_df["pred_best_threshold"] = (test_prob_cal >= best_threshold).astype(np.int64)

    metrics_summary = {
        "config": {
            "aggregation_primary": args.aggregation_primary,
            "aggregation_secondary": args.aggregation_secondary,
            "seed": args.seed,
            "global_manifest_checksum": sha256_file(args.global_manifest),
            "split_manifest_checksum": sha256_file(args.split_manifest),
        },
        "thresholds": {
            "best_threshold": best_threshold,
            "temperature": temperature,
            "val_best_f1": val_best_f1,
        },
        "validation": {
            "metrics_at_0_5": val_metrics_at_05,
            "metrics_at_best_threshold": val_metrics_best,
            "calibration": val_calibration,
        },
        "test": {
            "metrics_at_0_5": test_metrics_at_05,
            "metrics_at_best_threshold": test_metrics_best,
            "calibration": test_calibration,
        },
    }

    data_summary_df = build_data_summary(global_manifest_rows, split_manifest_rows)

    save_json(metrics_summary, args.output_dir / "metrics_summary.json")
    save_json(
        {
            "validation": val_calibration,
            "test": test_calibration,
        },
        args.output_dir / "calibration.json",
    )
    val_df.to_csv(args.output_dir / "predictions_track_val.csv", index=False)
    test_df.to_csv(args.output_dir / "predictions_track_test.csv", index=False)
    data_summary_df.to_csv(args.output_dir / "data_summary.csv", index=False)

    confusion_rows = [
        {
            "split": "test",
            "threshold_type": "0.5",
            "tn": test_metrics_at_05["tn"],
            "fp": test_metrics_at_05["fp"],
            "fn": test_metrics_at_05["fn"],
            "tp": test_metrics_at_05["tp"],
        },
        {
            "split": "test",
            "threshold_type": "best_threshold",
            "tn": test_metrics_best["tn"],
            "fp": test_metrics_best["fp"],
            "fn": test_metrics_best["fn"],
            "tp": test_metrics_best["tp"],
        },
    ]
    write_csv_rows(
        args.output_dir / "confusion_matrix.csv",
        ["split", "threshold_type", "tn", "fp", "fn", "tp"],
        confusion_rows,
    )

    print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
