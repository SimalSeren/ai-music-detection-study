import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attention_mil import AttentionMILClassifier
from clip_dataset import ClipSpectrogramDataset, clip_collate_fn
from metrics_utils import compute_binary_metrics, per_family_metrics, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Clip-level attention MIL training.")
    parser.add_argument("--train-split", type=str, default="data/splits_labeled/train.txt")
    parser.add_argument("--val-split", type=str, default="data/splits_labeled/val.txt")
    parser.add_argument("--test-split", type=str, default="data/splits_labeled/test.txt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-width", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=Path("results_attention"))
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
    rows = []

    with torch.no_grad():
        for batch in loader:
            crops = batch["crops"].to(device)
            crop_mask = batch["crop_mask"].to(device)
            labels = batch["label"].to(device)

            logits, attn_weights, _ = model(crops, crop_mask)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            total_loss += loss.item()

            for idx in range(labels.size(0)):
                valid_weights = attn_weights[idx][crop_mask[idx]].detach().cpu().numpy()
                rows.append(
                    {
                        "clip_id": batch["clip_id"][idx],
                        "track_id": batch["track_id"][idx],
                        "family": batch["family"][idx],
                        "label": int(labels[idx].item()),
                        "prob_fake": float(probs[idx].item()),
                        "pred": int(preds[idx].item()),
                        "path": batch["path"][idx],
                        "n_crops": int(crop_mask[idx].sum().item()),
                        "attention_entropy": float(
                            -(valid_weights * np.log(np.clip(valid_weights.astype(np.float64), 1e-12, None))).sum()
                        ) if len(valid_weights) > 0 else 0.0,
                    }
                )

    clip_df = pd.DataFrame(rows)
    clip_metrics = compute_binary_metrics(clip_df["label"], clip_df["prob_fake"], threshold=threshold)
    family_df = per_family_metrics(clip_df, threshold=threshold)

    return {
        "avg_loss": total_loss / max(len(loader), 1),
        "clip_df": clip_df,
        "family_df": family_df,
        "clip_metrics": clip_metrics,
    }


def plot_training_curves(history_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Attention MIL Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_clip_acc"], label="Train Clip Acc")
    plt.plot(history_df["epoch"], history_df["val_clip_f1"], label="Val Clip F1")
    plt.plot(history_df["epoch"], history_df["val_clip_bal_acc"], label="Val Clip Bal Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Attention MIL Validation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_eval_bundle(prefix: str, results: dict, output_root: Path):
    metrics_dir = output_root / "metrics"
    logs_dir = output_root / "logs"

    save_json(results["clip_metrics"], metrics_dir / f"{prefix}_clip_metrics.json")
    results["clip_df"].to_csv(logs_dir / f"{prefix}_clip_predictions.csv", index=False)
    results["family_df"].to_csv(logs_dir / f"{prefix}_family_metrics.csv", index=False)


def main():
    args = parse_args()
    ensure_dirs(args.output_root, args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = ClipSpectrogramDataset(
        args.train_split,
        crop_width=args.crop_width,
        stride=args.stride,
    )
    val_dataset = ClipSpectrogramDataset(
        args.val_split,
        crop_width=args.crop_width,
        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=clip_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=clip_collate_fn,
    )

    test_loader = None
    if Path(args.test_split).exists():
        test_dataset = ClipSpectrogramDataset(
            args.test_split,
            crop_width=args.crop_width,
            stride=args.stride,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=clip_collate_fn,
        )

    model = AttentionMILClassifier(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_rows = []
    best_val_metric = float("-inf")
    best_model_path = args.model_dir / "best_clip_attention.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            crops = batch["crops"].to(device)
            crop_mask = batch["crop_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(crops, crop_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = running_loss / max(len(train_loader), 1)
        train_clip_acc = correct / max(total, 1)
        val_results = evaluate_model(model, val_loader, device, criterion)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_clip_acc": train_clip_acc,
            "val_loss": val_results["avg_loss"],
            "val_clip_acc": val_results["clip_metrics"]["accuracy"],
            "val_clip_bal_acc": val_results["clip_metrics"]["balanced_accuracy"],
            "val_clip_f1": val_results["clip_metrics"]["f1"],
            "val_clip_auroc": val_results["clip_metrics"]["auroc"],
            "val_clip_auprc": val_results["clip_metrics"]["auprc"],
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Clip Acc: {train_clip_acc:.4f} | "
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
