import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol_utils import (
    GLOBAL_MANIFEST_FIELDS,
    SPLIT_MANIFEST_FIELDS,
    sha256_file,
    sha256_json,
    write_csv_rows,
)


def discover_real_files(real_dir: Path):
    real_files = sorted(real_dir.glob("*.npy"))
    return {path.stem: path for path in real_files}


def discover_fake_families(fake_dir: Path):
    fake_families = {}

    direct_files = sorted(fake_dir.glob("*.npy"))
    if direct_files:
        fake_families["griffinlim"] = {path.stem: path for path in direct_files}

    for subdir in sorted(path for path in fake_dir.iterdir() if path.is_dir()):
        npy_files = sorted(subdir.glob("*.npy"))
        if npy_files:
            fake_families[subdir.name] = {path.stem: path for path in npy_files}

    if not fake_families:
        raise FileNotFoundError(f"Fake spectrogram bulunamadi: {fake_dir}")

    return fake_families


def split_track_ids(track_ids, train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    track_ids = list(track_ids)
    rng.shuffle(track_ids)

    n = len(track_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": track_ids[:train_end],
        "val": track_ids[train_end:val_end],
        "test": track_ids[val_end:],
    }


def build_global_manifest(real_map, fake_family_maps, pipeline_id: str, sr: int, duration_sec: int):
    rows = []
    for track_id, real_path in sorted(real_map.items()):
        real_spec = np.load(real_path, mmap_mode="r")
        rows.append(
            {
                "sample_id": f"{track_id}:real",
                "stem_id": track_id,
                "source_dataset": "fma_small",
                "origin_type": "real",
                "transform_group": "none",
                "derived_family": "real",
                "audio_path": "",
                "spectrogram_path": real_path.as_posix(),
                "sr": sr,
                "duration_sec": duration_sec,
                "frame_count": int(real_spec.shape[1]),
                "patch_count": 0,
                "pipeline_id": pipeline_id,
            }
        )

        for family_name, family_map in sorted(fake_family_maps.items()):
            fake_path = family_map.get(track_id)
            if fake_path is None:
                continue

            fake_spec = np.load(fake_path, mmap_mode="r")
            rows.append(
                {
                    "sample_id": f"{track_id}:{family_name}",
                    "stem_id": track_id,
                    "source_dataset": "fma_small",
                    "origin_type": "derived",
                    "transform_group": infer_transform_group(family_name),
                    "derived_family": family_name,
                    "audio_path": "",
                    "spectrogram_path": fake_path.as_posix(),
                    "sr": sr,
                    "duration_sec": duration_sec,
                    "frame_count": int(fake_spec.shape[1]),
                    "patch_count": 0,
                    "pipeline_id": pipeline_id,
                }
            )

    return rows


def infer_transform_group(family_name: str) -> str:
    family_name = family_name.lower()
    if "griffinlim" in family_name:
        return "reconstruction"
    if "resample" in family_name:
        return "resample_roundtrip"
    if "noise" in family_name:
        return "noise_injection"
    if "quant" in family_name:
        return "codec_roundtrip"
    return "generator"


def build_split_manifest(stem_ids, split_ids, split_seed: int, split_version: str, checksum: str):
    split_lookup = {}
    for split_name, items in split_ids.items():
        for stem_id in items:
            split_lookup[stem_id] = split_name

    rows = []
    for stem_id in stem_ids:
        rows.append(
            {
                "stem_id": stem_id,
                "split": split_lookup[stem_id],
                "split_seed": split_seed,
                "split_version": split_version,
                "global_manifest_checksum": checksum,
            }
        )
    return rows


def backfill_patch_counts(global_manifest_rows, crop_width: int, stride: int):
    for row in global_manifest_rows:
        frame_count = int(row["frame_count"])
        if frame_count < crop_width:
            patch_count = 0
        else:
            patch_count = ((frame_count - crop_width) // stride) + 1
        row["patch_count"] = patch_count
    return global_manifest_rows


def write_protocol_outputs(global_manifest_rows, split_manifest_rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    global_manifest_path = output_dir / "global_manifest.csv"
    split_manifest_path = output_dir / "split_manifest.csv"
    manifest_path = output_dir / "manifest.csv"

    write_csv_rows(global_manifest_path, GLOBAL_MANIFEST_FIELDS, global_manifest_rows)
    write_csv_rows(split_manifest_path, SPLIT_MANIFEST_FIELDS, split_manifest_rows)

    split_lookup = {row["stem_id"]: row["split"] for row in split_manifest_rows}
    labeled_rows = []
    for row in global_manifest_rows:
        split_name = split_lookup[row["stem_id"]]
        label = 0 if row["origin_type"] == "real" else 1
        family = "real" if label == 0 else row["derived_family"]
        labeled_rows.append(
            {
                "split": split_name,
                "track_id": row["stem_id"],
                "label": label,
                "family": family,
                "path": row["spectrogram_path"],
            }
        )

    write_csv_rows(manifest_path, ["split", "track_id", "label", "family", "path"], labeled_rows)

    grouped_rows = defaultdict(list)
    for row in labeled_rows:
        grouped_rows[row["split"]].append(row)

    for split_name, split_rows in grouped_rows.items():
        split_path = output_dir / f"{split_name}.txt"
        with open(split_path, "w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(
                    f'{row["path"]}|{row["label"]}|{row["track_id"]}|{row["family"]}\n'
                )

    test_family_dir = output_dir / "test_by_family"
    test_family_dir.mkdir(parents=True, exist_ok=True)
    for family_name in sorted({row["family"] for row in labeled_rows if row["split"] == "test"}):
        family_rows = [
            row
            for row in labeled_rows
            if row["split"] == "test" and row["family"] in {"real", family_name}
        ]
        family_path = test_family_dir / f"{family_name}.txt"
        with open(family_path, "w", encoding="utf-8") as handle:
            for row in family_rows:
                handle.write(
                    f'{row["path"]}|{row["label"]}|{row["track_id"]}|{row["family"]}\n'
                )

    return global_manifest_path, split_manifest_path, manifest_path


def build_summary(global_manifest_rows, split_manifest_rows):
    split_lookup = {row["stem_id"]: row["split"] for row in split_manifest_rows}
    summary = {
        "total_rows": len(global_manifest_rows),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in global_manifest_rows if split_lookup[row["stem_id"]] == split_name]
        split_track_ids = sorted({row["stem_id"] for row in split_rows})
        family_counts = defaultdict(int)
        for row in split_rows:
            family = "real" if row["origin_type"] == "real" else row["derived_family"]
            family_counts[family] += 1

        summary["splits"][split_name] = {
            "n_rows": len(split_rows),
            "n_tracks": len(split_track_ids),
            "family_counts": dict(sorted(family_counts.items())),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Leakage'siz, track-level labeled split olusturur.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/processed/fma_specs"))
    parser.add_argument("--fake-dir", type=Path, default=Path("data/processed/fake_specs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits_labeled"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-version", type=str, default="v1.3")
    parser.add_argument("--pipeline-id", type=str, default="fma16k10s_stftdb_v1")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration-sec", type=int, default=10)
    parser.add_argument("--crop-width", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    real_map = discover_real_files(args.real_dir)
    fake_family_maps = discover_fake_families(args.fake_dir)

    common_track_ids = set(real_map.keys())
    for family_map in fake_family_maps.values():
        common_track_ids &= set(family_map.keys())

    if not common_track_ids:
        raise ValueError("Real ve fake family'ler arasinda ortak track bulunamadi.")

    split_ids = split_track_ids(
        track_ids=sorted(common_track_ids),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    global_manifest_rows = build_global_manifest(
        real_map,
        fake_family_maps,
        pipeline_id=args.pipeline_id,
        sr=args.sr,
        duration_sec=args.duration_sec,
    )
    global_manifest_rows = backfill_patch_counts(
        global_manifest_rows,
        crop_width=args.crop_width,
        stride=args.stride,
    )
    global_checksum = sha256_json(global_manifest_rows)
    split_manifest_rows = build_split_manifest(
        stem_ids=sorted(common_track_ids),
        split_ids=split_ids,
        split_seed=args.seed,
        split_version=args.split_version,
        checksum=global_checksum,
    )
    global_manifest_path, split_manifest_path, manifest_path = write_protocol_outputs(
        global_manifest_rows,
        split_manifest_rows,
        args.output_dir,
    )

    summary = build_summary(global_manifest_rows, split_manifest_rows)
    summary["global_manifest_checksum"] = global_checksum
    summary["global_manifest_file_checksum"] = sha256_file(global_manifest_path)
    summary["split_manifest_file_checksum"] = sha256_file(split_manifest_path)
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Global manifest kaydedildi: {global_manifest_path}")
    print(f"Split manifest kaydedildi: {split_manifest_path}")
    print(f"Labeled manifest kaydedildi: {manifest_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
