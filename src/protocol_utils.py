import csv
import hashlib
import json
from pathlib import Path


GLOBAL_MANIFEST_FIELDS = [
    "sample_id",
    "stem_id",
    "source_dataset",
    "origin_type",
    "transform_group",
    "derived_family",
    "audio_path",
    "spectrogram_path",
    "sr",
    "duration_sec",
    "frame_count",
    "patch_count",
    "pipeline_id",
]

SPLIT_MANIFEST_FIELDS = [
    "stem_id",
    "split",
    "split_seed",
    "split_version",
    "global_manifest_checksum",
]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(payload) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
