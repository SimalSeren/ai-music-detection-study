from pathlib import Path
import random

random.seed(42)

real_files = sorted(Path("data/processed/fma_specs").glob("*.npy"))
fake_files = sorted(Path("data/processed/fake_specs").glob("*.npy"))

# ortak isimlere göre eşleştirme
real_map = {p.stem: p for p in real_files}
fake_map = {p.stem: p for p in fake_files}

common_keys = sorted(set(real_map.keys()) & set(fake_map.keys()))

samples = []
for key in common_keys:
    samples.append((real_map[key], 0))
    samples.append((fake_map[key], 1))

random.shuffle(samples)

n = len(samples)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_samples = samples[:train_end]
val_samples = samples[train_end:val_end]
test_samples = samples[val_end:]

out_dir = Path("data/splits_labeled")
out_dir.mkdir(parents=True, exist_ok=True)

def save_split(split, filename):
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        for path, label in split:
            f.write(f"{str(path).replace('\\', '/')}|{label}\n")

save_split(train_samples, "train.txt")
save_split(val_samples, "val.txt")
save_split(test_samples, "test.txt")

print("Ortak örnek sayısı:", len(common_keys))
print("Toplam labeled sample:", len(samples))
print("Train:", len(train_samples))
print("Val:", len(val_samples))
print("Test:", len(test_samples))