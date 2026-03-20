from torch.utils.data import DataLoader
from fma_dataset import FMASpectrogramDataset

dataset = FMASpectrogramDataset("data/splits/train.txt")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y, paths = next(iter(loader))

print("Batch tensor shape:", x.shape)
print("Batch labels shape:", y.shape)
print("İlk path:", paths[0])