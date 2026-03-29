from pathlib import Path
import random

random.seed(42)

root = Path("data/fma_small")
files = sorted(root.rglob("*.mp3"))

subset_sizes = [100, 500, 1000]

for subset_size in subset_sizes:
    if len(files) < subset_size:
        print(f"Uyarı: {subset_size} istenecek dosya yok. Atlanıyor.")
        continue
    subset = random.sample(files, subset_size)
    output_file = Path(f"data/fma_subset_{subset_size}.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for path in subset:
            f.write(str(path).replace("\\", "/") + "\n")
    print(f"Subset boyutu {subset_size} listesi kaydedildi: {output_file}")
