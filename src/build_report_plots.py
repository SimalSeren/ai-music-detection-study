import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Rapor/sunum icin ozet benchmark gorselleri uretir.")
    parser.add_argument("--reports-dir", type=Path, default=Path("results_reports_today"))
    return parser.parse_args()


def ensure_output_dir(reports_dir: Path):
    output_dir = reports_dir / "summary_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_model_comparison(benchmark_df: pd.DataFrame, output_dir: Path):
    df = benchmark_df.copy()
    df["label"] = df["model"] + " | " + df["experiment_dir"].str.replace("results_", "", regex=False)

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, df["clip_f1"], width=width, label="F1")
    plt.bar(x, df["clip_balanced_accuracy"], width=width, label="Balanced Acc")
    plt.bar(x + width, df["clip_auroc"], width=width, label="AUROC")
    plt.xticks(x, df["label"], rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison on Available In-Distribution Pilots")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_bars.png", dpi=220)
    plt.close()


def plot_same_mixed_ood(benchmark_df: pd.DataFrame, ood_df: pd.DataFrame, output_dir: Path):
    rows = []

    same_family = benchmark_df.loc[benchmark_df["experiment_dir"].str.contains("report100", case=False, na=False)]
    if not same_family.empty:
        rows.append(("Same-family ResNet", float(same_family.iloc[0]["clip_balanced_accuracy"])))

    mixed_resnet = benchmark_df.loc[benchmark_df["experiment_dir"].str.contains("resnet_mixed", case=False, na=False)]
    if not mixed_resnet.empty:
        rows.append(("Mixed-family ResNet", float(mixed_resnet.iloc[0]["clip_balanced_accuracy"])))

    mixed_artifact = benchmark_df.loc[benchmark_df["experiment_dir"].str.contains("artifactnet_mixed", case=False, na=False)]
    if not mixed_artifact.empty:
        rows.append(("Mixed-family ArtifactNet", float(mixed_artifact.iloc[0]["clip_balanced_accuracy"])))

    for _, row in ood_df.iterrows():
        rows.append((f"OOD {row['model']} -> {row['held_out_family']}", float(row["held_out_balanced_accuracy"])))

    plot_df = pd.DataFrame(rows, columns=["setting", "balanced_accuracy"])

    plt.figure(figsize=(12, 6))
    colors = ["#2a6f97" if "OOD" not in s else "#bb3e03" for s in plot_df["setting"]]
    plt.bar(plot_df["setting"], plot_df["balanced_accuracy"], color=colors)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Balanced Accuracy")
    plt.title("Same-family vs Mixed-family vs OOD Generalization")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "same_mixed_ood_balanced_accuracy.png", dpi=220)
    plt.close()


def plot_ood_heatmap(ood_matrix_df: pd.DataFrame, output_dir: Path):
    df = ood_matrix_df.set_index("model")
    values = df.to_numpy(dtype=float)

    plt.figure(figsize=(8, 4.5))
    plt.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Held-out Balanced Accuracy")
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=20, ha="right")
    plt.yticks(np.arange(len(df.index)), df.index)
    plt.title("OOD Benchmark Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if pd.isna(value) else f"{value:.2f}"
            plt.text(j, i, text, ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_dir / "ood_heatmap_balanced_accuracy.png", dpi=220)
    plt.close()


def plot_ood_per_family(ood_df: pd.DataFrame, output_dir: Path):
    pivot = ood_df.pivot(index="held_out_family", columns="model", values="held_out_balanced_accuracy")
    pivot = pivot.sort_index()

    x = np.arange(len(pivot.index))
    width = 0.35 if len(pivot.columns) > 1 else 0.5

    plt.figure(figsize=(9, 5))
    for idx, model in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        plt.bar(x + offset, pivot[model].fillna(0.0), width=width, label=model)

    plt.xticks(x, pivot.index, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Held-out Balanced Accuracy")
    plt.title("OOD Performance by Held-out Fake Family")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_family_comparison.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.reports_dir)

    benchmark_df = pd.read_csv(args.reports_dir / "benchmark_summary.csv")
    ood_df = pd.read_csv(args.reports_dir / "ood_benchmark_summary.csv")
    ood_matrix_df = pd.read_csv(args.reports_dir / "ood_benchmark_matrix_balanced_accuracy.csv")

    plot_model_comparison(benchmark_df, output_dir)
    plot_same_mixed_ood(benchmark_df, ood_df, output_dir)
    plot_ood_heatmap(ood_matrix_df, output_dir)
    plot_ood_per_family(ood_df, output_dir)

    print(f"Ozet rapor gorselleri kaydedildi: {output_dir}")


if __name__ == "__main__":
    main()
