from torch.utils.data import DataLoader
from cropped_dataset import CroppedSpectrogramDataset

dataset = CroppedSpectrogramDataset(
    "data/splits_labeled/train.txt",
    crop_width=64,
    stride=32
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

x, y, paths, starts = next(iter(loader))

print("Batch shape:", x.shape)
print("Labels shape:", y.shape)
print("İlk path:", paths[0])
print("İlk start:", starts[0])