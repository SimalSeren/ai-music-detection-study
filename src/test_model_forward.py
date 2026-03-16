from torch.utils.data import DataLoader
from fma_dataset import FMASpectrogramDataset
from simple_cnn import SimpleCNN

dataset = FMASpectrogramDataset("data/processed/fma_specs")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y, paths = next(iter(loader))

model = SimpleCNN()
out = model(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)