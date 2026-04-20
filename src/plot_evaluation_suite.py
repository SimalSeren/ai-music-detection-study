import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.manifold import TSNE


def plot_embeddings(embeddings_path: Path, output_dir: Path, prefix: str):
    if not embeddings_path.exists():
        return
    
    data = np.load(embeddings_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    families = data["families"]
    
    if len(features) > 2000:
        # Subsample if too large for t-SNE
        idx = np.random.choice(len(features), 2000, replace=False)
        features = features[idx]
        families = families[idx]
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    unique_families = list(dict.fromkeys(families.tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, family in enumerate(unique_families):
        mask = families == family
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=str(family),
            color=cmap(idx % 10),
            s=24,
            alpha=0.7,
        )
    plt.title(f"{prefix.upper()} t-SNE Embeddings (Subsampled)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_tsne_embeddings.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Tek deney klasoru icin standart evaluation gorselleri uretir.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_roc_pr_curves(df: pd.DataFrame, output_dir: Path, prefix: str):
    y_true = df["label"].to_numpy()
    y_score = df["prob_fake"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=plt.gca())
    plt.title(f"{prefix.upper()} ROC Curve")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=plt.gca())
    plt.title(f"{prefix.upper()} PR Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_roc_pr.png", dpi=200)
    plt.close()


def save_score_histogram(df: pd.DataFrame, output_dir: Path, prefix: str):
    plt.figure(figsize=(8, 5))
    real_scores = df.loc[df["label"] == 0, "prob_fake"].to_numpy()
    fake_scores = df.loc[df["label"] == 1, "prob_fake"].to_numpy()

    bins = np.linspace(0.0, 1.0, 30)
    plt.hist(real_scores, bins=bins, alpha=0.6, label="Real", density=True)
    plt.hist(fake_scores, bins=bins, alpha=0.6, label="Fake", density=True)
    plt.xlabel("Predicted Fake Probability")
    plt.ylabel("Density")
    plt.title(f"{prefix.upper()} Score Histogram")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_score_hist.png", dpi=200)
    plt.close()


def save_family_bar_chart(family_df: pd.DataFrame, output_dir: Path, prefix: str):
    metric_df = family_df.sort_values(by="f1", ascending=False)
    x = np.arange(len(metric_df))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, metric_df["f1"], width=width, label="F1")
    plt.bar(x + width / 2, metric_df["balanced_accuracy"], width=width, label="Balanced Acc")
    plt.xticks(x, metric_df["family"], rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"{prefix.upper()} Family-wise Performance")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_family_bars.png", dpi=200)
    plt.close()


def save_confusion_heatmap(metrics_payload, output_dir: Path, prefix: str):
    cm = np.array(
        [
            [metrics_payload["tn"], metrics_payload["fp"]],
            [metrics_payload["fn"], metrics_payload["tp"]],
        ]
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred Real", "Pred Fake"])
    plt.yticks([0, 1], ["True Real", "True Fake"])
    plt.title(f"{prefix.upper()} Confusion Matrix")

    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()


def save_threshold_curve(df: pd.DataFrame, output_dir: Path, prefix: str):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []

    y_true = df["label"].to_numpy(dtype=np.int64)
    y_prob = df["prob_fake"].to_numpy(dtype=np.float32)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        balanced_accuracy = 0.5 * ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)))
        rows.append(
            {
                "threshold": threshold,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(output_dir / f"{prefix}_threshold_sweep.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="Balanced Acc")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title(f"{prefix.upper()} Threshold Sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_threshold_sweep.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    logs_dir = args.experiment_dir / "logs"
    metrics_dir = args.experiment_dir / "metrics"
    figures_dir = args.experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    clip_predictions = pd.read_csv(logs_dir / f"{args.split_name}_clip_predictions.csv")
    family_metrics = pd.read_csv(logs_dir / f"{args.split_name}_family_metrics.csv")
    clip_metrics = load_json(metrics_dir / f"{args.split_name}_clip_metrics.json")

    save_roc_pr_curves(clip_predictions, figures_dir, args.split_name)
    save_score_histogram(clip_predictions, figures_dir, args.split_name)
    save_family_bar_chart(family_metrics, figures_dir, args.split_name)
    save_confusion_heatmap(clip_metrics, figures_dir, args.split_name)
    save_threshold_curve(clip_predictions, figures_dir, args.split_name)
    
    embeddings_file = logs_dir / f"{args.split_name}_embeddings.npz"
    if embeddings_file.exists():
        plot_embeddings(embeddings_file, figures_dir, args.split_name)


if __name__ == "__main__":
    main()
