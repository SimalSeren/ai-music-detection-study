import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cropped_dataset import CroppedSpectrogramDataset
from metrics_utils import (
    aggregate_to_clip_level,
    compute_binary_metrics,
    per_family_metrics,
    save_json,
)
from model_factory import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Crop-based baseline training with clip-level evaluation.")
    parser.add_argument("--train-split", type=str, default="data/splits_labeled/train.txt")
    parser.add_argument("--val-split", type=str, default="data/splits_labeled/val.txt")
    parser.add_argument("--test-split", type=str, default="data/splits_labeled/test.txt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-width", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="simplecnn")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def ensure_dirs(output_root: Path, model_dir: Path):
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, loader, device, criterion, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    prediction_rows = []
    all_features = []

    with torch.no_grad():
        for crops, labels, clip_ids, track_ids, families, paths, starts in loader:
            crops = crops.to(device)
            labels = labels.to(device)

            if hasattr(model, "forward_features"):
                features = model.forward_features(crops)
                if isinstance(features, torch.Tensor) and features.dim() > 2:
                    features = torch.flatten(features, 1)
                logits = model.classifier(features) if hasattr(model, "classifier") else model(crops)
            else:
                logits = model(crops)
                features = logits  # Fallback

            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            total_loss += loss.item()
            all_features.append(features.cpu().numpy())

            for idx in range(len(labels)):
                prediction_rows.append(
                    {
                        "clip_id": clip_ids[idx],
                        "track_id": track_ids[idx],
                        "family": families[idx],
                        "label": int(labels[idx].item()),
                        "prob_fake": float(probs[idx].item()),
                        "pred": int(preds[idx].item()),
                        "path": paths[idx],
                        "start": int(starts[idx]),
                    }
                )

    import numpy as np
    features_np = np.concatenate(all_features, axis=0) if all_features else np.array([])
    crop_df, clip_df = aggregate_to_clip_level(prediction_rows)
    crop_metrics = compute_binary_metrics(crop_df["label"], crop_df["prob_fake"], threshold=threshold)
    clip_metrics = compute_binary_metrics(clip_df["label"], clip_df["prob_fake"], threshold=threshold)
    family_df = per_family_metrics(clip_df, threshold=threshold)

    return {
        "avg_loss": total_loss / max(len(loader), 1),
        "crop_df": crop_df,
        "clip_df": clip_df,
        "family_df": family_df,
        "crop_metrics": crop_metrics,
        "clip_metrics": clip_metrics,
        "features": features_np,
        "labels": np.array([r["label"] for r in prediction_rows]),
        "families": np.array([r["family"] for r in prediction_rows])
    }


def plot_training_curves(history_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_crop_acc"], label="Train Crop Acc")
    plt.plot(history_df["epoch"], history_df["val_clip_f1"], label="Val Clip F1")
    plt.plot(history_df["epoch"], history_df["val_clip_bal_acc"], label="Val Clip Bal Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training vs Clip-Level Validation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_eval_bundle(prefix: str, results: dict, output_root: Path):
    metrics_dir = output_root / "metrics"
    logs_dir = output_root / "logs"

    save_json(results["crop_metrics"], metrics_dir / f"{prefix}_crop_metrics.json")
    save_json(results["clip_metrics"], metrics_dir / f"{prefix}_clip_metrics.json")
    results["crop_df"].to_csv(logs_dir / f"{prefix}_crop_predictions.csv", index=False)
    results["clip_df"].to_csv(logs_dir / f"{prefix}_clip_predictions.csv", index=False)
    results["family_df"].to_csv(logs_dir / f"{prefix}_family_metrics.csv", index=False)
    
    if "features" in results and len(results["features"]) > 0:
        import numpy as np
        np.savez_compressed(
            logs_dir / f"{prefix}_embeddings.npz", 
            features=results["features"], 
            labels=results["labels"],
            families=results["families"]
        )


def main():
    args = parse_args()
    ensure_dirs(args.output_root, args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = CroppedSpectrogramDataset(
        args.train_split,
        crop_width=args.crop_width,
        stride=args.stride,
        augment=True, # Genellemeyi artırmak için eğitim anında maskeleme
    )
    val_dataset = CroppedSpectrogramDataset(
        args.val_split,
        crop_width=args.crop_width,
        stride=args.stride,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = None
    if Path(args.test_split).exists():
        test_dataset = CroppedSpectrogramDataset(
            args.test_split,
            crop_width=args.crop_width,
            stride=args.stride,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = create_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_rows = []
    best_val_metric = float("-inf")
    best_model_path = args.model_dir / f"best_{args.model}_cropped.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for crops, labels, *_ in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = running_loss / max(len(train_loader), 1)
        train_crop_acc = correct / max(total, 1)
        val_results = evaluate_model(model, val_loader, device, criterion)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_crop_acc": train_crop_acc,
            "val_loss": val_results["avg_loss"],
            "val_crop_acc": val_results["crop_metrics"]["accuracy"],
            "val_crop_f1": val_results["crop_metrics"]["f1"],
            "val_clip_acc": val_results["clip_metrics"]["accuracy"],
            "val_clip_bal_acc": val_results["clip_metrics"]["balanced_accuracy"],
            "val_clip_f1": val_results["clip_metrics"]["f1"],
            "val_clip_auroc": val_results["clip_metrics"]["auroc"],
            "val_clip_auprc": val_results["clip_metrics"]["auprc"],
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Crop Acc: {train_crop_acc:.4f} | "
            f"Val Clip F1: {history_row['val_clip_f1']:.4f} | "
            f"Val Clip Bal Acc: {history_row['val_clip_bal_acc']:.4f}"
        )

        if history_row["val_clip_bal_acc"] > best_val_metric:
            best_val_metric = history_row["val_clip_bal_acc"]
            torch.save(model.state_dict(), best_model_path)
            save_eval_bundle("val_best", val_results, args.output_root)
            print(f"Best model saved: {best_model_path}")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(args.output_root / "logs" / "training_history.csv", index=False)
    plot_training_curves(history_df, args.output_root / "figures" / "training_curves.png")

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    if test_loader is not None:
        test_results = evaluate_model(model, test_loader, device, criterion)
        save_eval_bundle("test", test_results, args.output_root)
        print("Test clip metrics:", test_results["clip_metrics"])


if __name__ == "__main__":
    main()
