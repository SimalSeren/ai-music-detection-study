from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class FMASpectrogramDataset(Dataset):
    def __init__(self, spec_dir: str):
        self.spec_dir = Path(spec_dir)
        self.files = sorted(self.spec_dir.glob("*.npy"))

        if len(self.files) == 0:
            raise ValueError(f"Hiç .npy dosyası bulunamadı: {self.spec_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        spec = np.load(path).astype(np.float32)

        # kanal boyutu ekle
        spec = np.expand_dims(spec, axis=0)

        spec = torch.tensor(spec, dtype=torch.float32)

        # şimdilik dummy label
        label = torch.tensor(0, dtype=torch.long)

        return spec, label, str(path)