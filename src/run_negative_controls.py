import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol_utils import load_csv_rows, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Belge v1.3 uyumlu negatif kontrol ve sanity check ozeti.")
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_split_overlap_report(split_manifest_rows):
    split_to_stems = {}
    for row in split_manifest_rows:
        split_to_stems.setdefault(row["split"], set()).add(row["stem_id"])

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    rows = []
    for left, right in pairs:
        overlap = sorted(split_to_stems.get(left, set()) & split_to_stems.get(right, set()))
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "overlap_count": len(overlap),
                "example_overlap": overlap[:10],
            }
        )
    return rows


def build_manifest_integrity_report(global_manifest_rows):
    df = pd.DataFrame(global_manifest_rows)
    duplicate_sample_ids = int(df["sample_id"].duplicated().sum())
    missing_spec_paths = int((~df["spectrogram_path"].map(lambda p: Path(p).exists())).sum())
    invalid_origin_rows = int((~df["origin_type"].isin(["real", "derived"])).sum())
    invalid_patch_counts = int((df["patch_count"].astype(int) <= 0).sum())

    return {
        "duplicate_sample_ids": duplicate_sample_ids,
        "missing_spectrogram_paths": missing_spec_paths,
        "invalid_origin_rows": invalid_origin_rows,
        "non_positive_patch_count_rows": invalid_patch_counts,
        "row_count": int(len(df)),
    }


def build_same_source_report(global_manifest_rows, test_predictions_path: Path | None):
    if test_predictions_path is None or not test_predictions_path.exists():
        return {
            "available": False,
            "reason": "test predictions bulunamadi",
        }

    pred_df = pd.read_csv(test_predictions_path)
    manifest_df = pd.DataFrame(global_manifest_rows)[["stem_id", "derived_family", "origin_type"]]
    pred_df["track_id"] = pred_df["track_id"].astype(str)
    manifest_df["stem_id"] = manifest_df["stem_id"].astype(str)
    merged = pred_df.merge(
        manifest_df,
        left_on=["track_id"],
        right_on=["stem_id"],
        how="left",
    )
    merged["same_source_flag"] = merged["origin_type"].eq("real")

    real_rows = merged.loc[merged["label"] == 0]
    if real_rows.empty:
        return {
            "available": False,
            "reason": "real label satiri bulunamadi",
        }

    false_positive_rate = float(np.mean(real_rows["pred_best_threshold"])) if "pred_best_threshold" in real_rows.columns else float(np.mean(real_rows["pred"]))

    return {
        "available": True,
        "real_track_count": int(len(real_rows)),
        "false_positive_rate": false_positive_rate,
        "mean_score": float(real_rows["prob_fake_calibrated"].mean()) if "prob_fake_calibrated" in real_rows.columns else float(real_rows["prob_fake"].mean()),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_manifest_rows = load_csv_rows(args.global_manifest)
    split_manifest_rows = load_csv_rows(args.split_manifest)

    overlap_report = build_split_overlap_report(split_manifest_rows)
    integrity_report = build_manifest_integrity_report(global_manifest_rows)
    same_source_report = build_same_source_report(global_manifest_rows, args.test_predictions)

    payload = {
        "split_overlap": overlap_report,
        "manifest_integrity": integrity_report,
        "same_source_control": same_source_report,
    }

    write_json(args.output_dir / "negative_controls.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
