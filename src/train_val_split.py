from pathlib import Path
import random

random.seed(42)

spec_dir = Path("data/processed/fma_specs")
files = sorted(spec_dir.glob("*.npy"))

random.shuffle(files)

n = len(files)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_files = files[:train_end]
val_files = files[train_end:val_end]
test_files = files[val_end:]

out_dir = Path("data/splits")
out_dir.mkdir(parents=True, exist_ok=True)

def save_list(paths, filename):
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p).replace("\\", "/") + "\n")

save_list(train_files, "train.txt")
save_list(val_files, "val.txt")
save_list(test_files, "test.txt")

print("Toplam:", n)
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))