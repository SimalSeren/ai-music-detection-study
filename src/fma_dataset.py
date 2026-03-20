from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class FMASpectrogramDataset(Dataset):
    def __init__(self, file_list: str):
        self.file_list = Path(file_list)
        self.samples = []

        with open(self.file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_str, label_str = line.split("|")
                self.samples.append((Path(path_str), int(label_str)))

        if len(self.samples) == 0:
            raise ValueError(f"Liste boş: {self.file_list}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = np.load(path).astype(np.float32)

        spec = np.expand_dims(spec, axis=0)
        spec = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return spec, label, str(path)