from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ClipSpectrogramDataset(Dataset):
    def __init__(self, file_list: str, crop_width: int = 64, stride: int = 32):
        self.file_list = Path(file_list)
        self.crop_width = crop_width
        self.stride = stride
        self.samples = []

        if not self.file_list.exists():
            raise FileNotFoundError(f"Split dosyasi bulunamadi: {self.file_list}")

        with open(self.file_list, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                raise ValueError(f"Gecersiz split satiri: {line}")

            path = Path(parts[0])
            label = int(parts[1])
            track_id = parts[2] if len(parts) >= 3 else path.stem
            family = parts[3] if len(parts) >= 4 else ("real" if label == 0 else "fake")
            clip_id = f"{track_id}|{family}|{label}"

            spec = np.load(path, mmap_mode="r")
            total_width = int(spec.shape[1])
            if total_width < crop_width:
                raise ValueError(
                    f"Crop width ({crop_width}) spectrogram genisliginden buyuk: "
                    f"{path} shape={spec.shape}"
                )

            starts = list(range(0, total_width - crop_width + 1, stride))
            self.samples.append(
                {
                    "path": path,
                    "label": label,
                    "track_id": track_id,
                    "family": family,
                    "clip_id": clip_id,
                    "starts": starts,
                }
            )

        if not self.samples:
            raise ValueError("Hic clip ornegi olusmadi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec = np.load(sample["path"]).astype(np.float32)

        crops = []
        for start in sample["starts"]:
            crop = spec[:, start : start + self.crop_width]
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)

        crops = np.stack(crops, axis=0)

        return {
            "crops": torch.tensor(crops, dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "clip_id": sample["clip_id"],
            "track_id": sample["track_id"],
            "family": sample["family"],
            "path": str(sample["path"]),
            "n_crops": crops.shape[0],
        }


def clip_collate_fn(batch):
    max_crops = max(item["n_crops"] for item in batch)
    crop_shape = batch[0]["crops"].shape[1:]

    padded_crops = []
    crop_masks = []
    labels = []
    clip_ids = []
    track_ids = []
    families = []
    paths = []

    for item in batch:
        n_crops = item["n_crops"]
        pad_count = max_crops - n_crops

        if pad_count > 0:
            padding = torch.zeros((pad_count, *crop_shape), dtype=item["crops"].dtype)
            crops = torch.cat([item["crops"], padding], dim=0)
        else:
            crops = item["crops"]

        mask = torch.zeros(max_crops, dtype=torch.bool)
        mask[:n_crops] = True

        padded_crops.append(crops)
        crop_masks.append(mask)
        labels.append(item["label"])
        clip_ids.append(item["clip_id"])
        track_ids.append(item["track_id"])
        families.append(item["family"])
        paths.append(item["path"])

    return {
        "crops": torch.stack(padded_crops, dim=0),
        "crop_mask": torch.stack(crop_masks, dim=0),
        "label": torch.stack(labels, dim=0),
        "clip_id": clip_ids,
        "track_id": track_ids,
        "family": families,
        "path": paths,
    }
