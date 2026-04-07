import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(metric_fn, *args, default=np.nan, **kwargs):
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        return default


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(_safe_metric(roc_auc_score, y_true, y_prob)),
        "auprc": float(_safe_metric(average_precision_score, y_true, y_prob)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
    }
    return metrics


def aggregate_to_clip_level(prediction_rows):
    crop_df = pd.DataFrame(prediction_rows)
    clip_df = (
        crop_df.groupby("clip_id", as_index=False)
        .agg(
            track_id=("track_id", "first"),
            family=("family", "first"),
            label=("label", "first"),
            prob_fake=("prob_fake", "mean"),
            n_crops=("clip_id", "size"),
        )
    )
    return crop_df, clip_df


def per_family_metrics(clip_df: pd.DataFrame, threshold: float = 0.5):
    rows = []
    for family, family_df in clip_df.groupby("family"):
        rows.append(
            {
                "family": family,
                **compute_binary_metrics(
                    family_df["label"].to_numpy(),
                    family_df["prob_fake"].to_numpy(),
                    threshold=threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_ece(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if not np.any(mask):
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "avg_confidence": np.nan,
                    "avg_accuracy": np.nan,
                }
            )
            continue

        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        weight = float(np.mean(mask))
        ece += abs(confidence - accuracy) * weight
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(np.sum(mask)),
                "avg_confidence": confidence,
                "avg_accuracy": accuracy,
            }
        )

    return float(ece), pd.DataFrame(rows)


def calibration_summary(y_true, y_prob_before, y_prob_after):
    ece_before, bins_before = compute_ece(y_true, y_prob_before)
    ece_after, bins_after = compute_ece(y_true, y_prob_after)

    clipped_before = np.clip(np.asarray(y_prob_before, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    clipped_after = np.clip(np.asarray(y_prob_after, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y_true = np.asarray(y_true, dtype=np.float64)

    nll_before = float(
        -np.mean(y_true * np.log(clipped_before) + (1.0 - y_true) * np.log(1.0 - clipped_before))
    )
    nll_after = float(
        -np.mean(y_true * np.log(clipped_after) + (1.0 - y_true) * np.log(1.0 - clipped_after))
    )

    return {
        "ece_before": ece_before,
        "ece_after": ece_after,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "bins_before": bins_before.to_dict(orient="records"),
        "bins_after": bins_after.to_dict(orient="records"),
    }


def save_json(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
