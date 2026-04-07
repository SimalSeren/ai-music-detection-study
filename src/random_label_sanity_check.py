import argparse
import json
import random
from pathlib import Path

import pandas as pd

from protocol_utils import load_csv_rows, write_csv_rows, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Random-label sanity check split uretecisi.")
    parser.add_argument("--input-split", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    rows = []
    label_counts_before = {0: 0, 1: 0}
    label_counts_after = {0: 0, 1: 0}

    with open(args.input_split, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            path, label, stem_id, family = line.split("|")
            label = int(label)
            label_counts_before[label] += 1
            random_label = rng.randint(0, 1)
            label_counts_after[random_label] += 1
            rows.append(
                {
                    "path": path,
                    "label": random_label,
                    "stem_id": stem_id,
                    "family": family,
                }
            )

    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_split, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f'{row["path"]}|{row["label"]}|{row["stem_id"]}|{row["family"]}\n')

    write_json(
        args.output_split.with_suffix(".json"),
        {
            "seed": args.seed,
            "label_counts_before": label_counts_before,
            "label_counts_after": label_counts_after,
            "n_rows": len(rows),
        },
    )

    print(json.dumps(
        {
            "output_split": args.output_split.as_posix(),
            "label_counts_before": label_counts_before,
            "label_counts_after": label_counts_after,
            "n_rows": len(rows),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
