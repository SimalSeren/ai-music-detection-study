from cropped_dataset import CroppedSpectrogramDataset

dataset = CroppedSpectrogramDataset(
    "data/splits_labeled/train.txt",
    crop_width=64,
    stride=32
)

print("Toplam crop örneği:", len(dataset))

x, y, path, start = dataset[0]
print("İlk dosya:", path)
print("Başlangıç frame:", start)
print("Crop shape:", x.shape)
print("Label:", y)