from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CroppedSpectrogramDataset(Dataset):
    def __init__(self, file_list: str, crop_width: int = 64, stride: int = 32):
        self.file_list = Path(file_list)
        self.crop_width = crop_width
        self.stride = stride
        self.samples = []

        with open(self.file_list, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            path_str, label_str = line.split("|")
            path = Path(path_str)
            label = int(label_str)

            spec = np.load(path)
            total_width = spec.shape[1]

            for start in range(0, total_width - crop_width + 1, stride):
                self.samples.append((path, label, start))

        if len(self.samples) == 0:
            raise ValueError("Hiç crop örneği oluşmadı.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, start = self.samples[idx]

        spec = np.load(path).astype(np.float32)
        crop = spec[:, start:start + self.crop_width]

        crop = np.expand_dims(crop, axis=0)  # (1, H, W)
        crop = torch.tensor(crop, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return crop, label, str(path), start