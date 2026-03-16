from fma_dataset import FMASpectrogramDataset

dataset = FMASpectrogramDataset("data/processed/fma_specs")

print("Dataset boyutu:", len(dataset))

x, y, path = dataset[0]
print("İlk örnek path:", path)
print("Tensor shape:", x.shape)
print("Label:", y)
print("Min:", x.min().item(), "Max:", x.max().item())