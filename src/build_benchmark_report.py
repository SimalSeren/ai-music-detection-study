import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Deney ciktilarini benchmark tablosuna donusturur.")
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        required=True,
        help="Her biri metrics/logs alt klasorlerine sahip deney cikti klasorleri.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results_reports"))
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_model_name(experiment_dir: Path):
    name = experiment_dir.name.lower()
    if "attention" in name:
        return "attention_mil"
    if "artifact" in name:
        return "artifactnet"
    if "transformer" in name:
        return "transformer"
    if "resnet" in name:
        return "resnet"
    if "simple" in name or "baseline" in name:
        return "simplecnn"
    return experiment_dir.name


def infer_training_mode(experiment_dir: Path):
    name = experiment_dir.name.lower()
    if "attention" in name:
        return "clip_level"
    return "crop_level"


def build_summary_row(experiment_dir: Path):
    metrics_dir = experiment_dir / "metrics"
    logs_dir = experiment_dir / "logs"

    test_metrics = load_json(metrics_dir / "test_clip_metrics.json")
    history_path = logs_dir / "training_history.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

    best_epoch = int(history_df.loc[history_df["val_clip_bal_acc"].idxmax(), "epoch"]) if not history_df.empty else None
    family_metrics_path = logs_dir / "test_family_metrics.csv"
    family_df = pd.read_csv(family_metrics_path) if family_metrics_path.exists() else pd.DataFrame()

    row = {
        "experiment_dir": experiment_dir.as_posix(),
        "model": infer_model_name(experiment_dir),
        "training_mode": infer_training_mode(experiment_dir),
        "best_epoch": best_epoch,
        "clip_accuracy": test_metrics.get("accuracy"),
        "clip_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "clip_precision": test_metrics.get("precision"),
        "clip_recall": test_metrics.get("recall"),
        "clip_f1": test_metrics.get("f1"),
        "clip_auroc": test_metrics.get("auroc"),
        "clip_auprc": test_metrics.get("auprc"),
        "tp": test_metrics.get("tp"),
        "tn": test_metrics.get("tn"),
        "fp": test_metrics.get("fp"),
        "fn": test_metrics.get("fn"),
    }

    if not family_df.empty:
        for _, family_row in family_df.iterrows():
            family = str(family_row["family"])
            row[f"{family}_f1"] = family_row["f1"]
            row[f"{family}_bal_acc"] = family_row["balanced_accuracy"]
            row[f"{family}_auroc"] = family_row["auroc"]

    return row


def build_markdown_table(df: pd.DataFrame):
    priority_columns = [
        "model",
        "training_mode",
        "best_epoch",
        "clip_accuracy",
        "clip_balanced_accuracy",
        "clip_f1",
        "clip_auroc",
        "clip_auprc",
    ]
    extra_columns = [column for column in df.columns if column not in priority_columns + ["experiment_dir"]]
    ordered_columns = ["experiment_dir"] + priority_columns + sorted(extra_columns)
    df = df[ordered_columns]
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_summary_row(Path(experiment_dir)) for experiment_dir in args.experiment_dirs]
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(by=["clip_balanced_accuracy", "clip_f1"], ascending=False)

    summary_csv = args.output_dir / "benchmark_summary.csv"
    summary_md = args.output_dir / "benchmark_summary.md"

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_table(summary_df))

    print(f"CSV kaydedildi: {summary_csv}")
    print(f"Markdown kaydedildi: {summary_md}")


if __name__ == "__main__":
    main()
