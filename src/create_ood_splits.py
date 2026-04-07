import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Leave-one-family-out OOD split dosyalari olusturur.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/splits_labeled/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits_labeled/ood"),
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_split(split_path: Path, rows):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f'{row["path"]}|{row["label"]}|{row["track_id"]}|{row["family"]}\n'
            )


def main():
    args = parse_args()
    rows = load_manifest(args.manifest_path)

    fake_families = sorted({row["family"] for row in rows if row["family"] != "real"})
    if not fake_families:
        raise ValueError("Manifest icinde fake family bulunamadi.")

    summary_rows = []

    for held_out_family in fake_families:
        family_dir = args.output_dir / held_out_family

        train_rows = [
            row
            for row in rows
            if row["split"] == "train" and row["family"] != held_out_family
        ]
        val_rows = [
            row
            for row in rows
            if row["split"] == "val" and row["family"] != held_out_family
        ]
        test_rows = [
            row
            for row in rows
            if row["split"] == "test" and row["family"] in {"real", held_out_family}
        ]

        write_split(family_dir / "train.txt", train_rows)
        write_split(family_dir / "val.txt", val_rows)
        write_split(family_dir / "test.txt", test_rows)

        family_summary = defaultdict(int)
        for row in train_rows:
            family_summary[f"train_{row['family']}"] += 1
        for row in val_rows:
            family_summary[f"val_{row['family']}"] += 1
        for row in test_rows:
            family_summary[f"test_{row['family']}"] += 1

        summary_rows.append(
            {
                "held_out_family": held_out_family,
                "n_train_rows": len(train_rows),
                "n_val_rows": len(val_rows),
                "n_test_rows": len(test_rows),
                **dict(sorted(family_summary.items())),
            }
        )

        print(
            f"Held-out family: {held_out_family} | "
            f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    summary_path = args.output_dir / "ood_split_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"OOD split ozeti kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()
