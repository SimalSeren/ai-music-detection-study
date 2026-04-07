import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Held-out family deneylerinden OOD benchmark matrisi olusturur.")
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        required=True,
        help="results_ood/<model>/<family> klasorleri",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results_reports"))
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_model_and_family(experiment_dir: Path):
    name = experiment_dir.name.lower()

    if "resnet" in name:
        model = "resnet"
    elif "artifact" in name:
        model = "artifactnet"
    elif "attention" in name:
        model = "attention_mil"
    else:
        model = experiment_dir.parent.name

    if "quantize" in name:
        family = "quantize_8bit"
    elif "resample" in name:
        family = "resample_8k"
    else:
        family = experiment_dir.name

    return model, family


def build_rows(experiment_dir: Path):
    metrics_dir = experiment_dir / "metrics"
    logs_dir = experiment_dir / "logs"

    model, held_out_family = infer_model_and_family(experiment_dir)
    clip_metrics = load_json(metrics_dir / "test_clip_metrics.json")
    family_df = pd.read_csv(logs_dir / "test_family_metrics.csv")
    target_family_df = family_df.loc[family_df["family"] == held_out_family]

    if target_family_df.empty:
        raise ValueError(
            f"Held-out family sonucu bulunamadi: {held_out_family} in {logs_dir / 'test_family_metrics.csv'}"
        )

    target_row = target_family_df.iloc[0]
    real_row = family_df.loc[family_df["family"] == "real"].iloc[0]

    return {
        "experiment_dir": experiment_dir.as_posix(),
        "model": model,
        "held_out_family": held_out_family,
        "clip_accuracy": clip_metrics["accuracy"],
        "clip_balanced_accuracy": clip_metrics["balanced_accuracy"],
        "clip_f1": clip_metrics["f1"],
        "clip_auroc": clip_metrics["auroc"],
        "held_out_f1": target_row["f1"],
        "held_out_balanced_accuracy": target_row["balanced_accuracy"],
        "held_out_auroc": target_row["auroc"],
        "real_f1": real_row["f1"],
        "real_balanced_accuracy": real_row["balanced_accuracy"],
    }


def build_markdown_table(df: pd.DataFrame):
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_matrix(df: pd.DataFrame, metric: str):
    matrix_df = (
        df.pivot(index="model", columns="held_out_family", values=metric)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return matrix_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_rows(Path(experiment_dir)) for experiment_dir in args.experiment_dirs]
    summary_df = pd.DataFrame(rows).sort_values(
        by=["held_out_balanced_accuracy", "held_out_f1"],
        ascending=False,
    )

    summary_csv = args.output_dir / "ood_benchmark_summary.csv"
    summary_md = args.output_dir / "ood_benchmark_summary.md"
    matrix_csv = args.output_dir / "ood_benchmark_matrix_balanced_accuracy.csv"
    matrix_md = args.output_dir / "ood_benchmark_matrix_balanced_accuracy.md"

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_table(summary_df))

    matrix_df = build_matrix(summary_df, "held_out_balanced_accuracy")
    matrix_df.to_csv(matrix_csv)
    with open(matrix_md, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_table(matrix_df.reset_index()))

    print(f"OOD summary kaydedildi: {summary_csv}")
    print(f"OOD matrix kaydedildi: {matrix_csv}")


if __name__ == "__main__":
    main()
