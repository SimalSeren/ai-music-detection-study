from torch.utils.data import DataLoader
from fma_dataset import FMASpectrogramDataset

dataset = FMASpectrogramDataset("data/processed/fma_specs")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

batch = next(iter(loader))
x, y, paths = batch

print("Batch tensor shape:", x.shape)
print("Batch labels shape:", y.shape)
print("İlk path:", paths[0])