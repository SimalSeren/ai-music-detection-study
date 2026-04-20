from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _apply_spec_augment(crop: np.ndarray, max_freq_mask: int = 24, max_time_mask: int = 8) -> np.ndarray:
    augmented = crop.copy()
    freq_bins, time_bins = augmented.shape

    if freq_bins > 1:
        freq_width = np.random.randint(0, min(max_freq_mask, freq_bins) + 1)
        if freq_width > 0:
            freq_start = np.random.randint(0, freq_bins - freq_width + 1)
            augmented[freq_start:freq_start + freq_width, :] = augmented.min()

    if time_bins > 1:
        time_width = np.random.randint(0, min(max_time_mask, time_bins) + 1)
        if time_width > 0:
            time_start = np.random.randint(0, time_bins - time_width + 1)
            augmented[:, time_start:time_start + time_width] = augmented.min()

    return augmented


class CroppedSpectrogramDataset(Dataset):
    def __init__(
        self,
        file_list: str,
        crop_width: int = 64,
        stride: int = 32,
        augment: bool = False,
    ):
        self.file_list = Path(file_list)
        self.crop_width = crop_width
        self.stride = stride
        self.augment = augment
        self.samples = []

        with open(self.file_list, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        for line in lines:
            parts = line.split("|")
            if len(parts) == 2:
                path_str, label_str = parts
                track_id = Path(path_str).stem
                family = "real" if int(label_str) == 0 else "fake"
            elif len(parts) == 4:
                path_str, label_str, track_id, family = parts
            else:
                raise ValueError(f"Gecersiz split satiri: {line}")

            path = Path(path_str)
            label = int(label_str)

            spec = np.load(path, mmap_mode="r")
            total_width = int(spec.shape[1])
            clip_id = f"{track_id}|{family}|{label}"

            for start in range(0, total_width - crop_width + 1, stride):
                self.samples.append((path, label, clip_id, track_id, family, start))

        if not self.samples:
            raise ValueError("Hic crop ornegi olusmadi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, clip_id, track_id, family, start = self.samples[idx]

        spec = np.load(path).astype(np.float32)
        crop = spec[:, start:start + self.crop_width]
        if self.augment:
            crop = _apply_spec_augment(crop)

        crop = np.expand_dims(crop, axis=0)
        crop = torch.tensor(crop, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return crop, label, clip_id, track_id, family, str(path), start
