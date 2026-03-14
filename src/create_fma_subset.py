from pathlib import Path
import random

random.seed(42)

root = Path("data/fma_small")
files = sorted(root.rglob("*.mp3"))

subset_size = 100
subset = random.sample(files, subset_size)

output_file = Path("data/fma_subset_100.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for path in subset:
        f.write(str(path).replace("\\", "/") + "\n")

print(f"Toplam dosya: {len(files)}")
print(f"Subset boyutu: {len(subset)}")
print(f"Liste kaydedildi: {output_file}")