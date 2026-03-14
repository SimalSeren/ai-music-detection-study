from pathlib import Path
import numpy as np

spec_dir = Path("data/processed/fma_specs")
files = sorted(spec_dir.glob("*.npy"))

print("Toplam spectrogram:", len(files))

for f in files[:5]:
    arr = np.load(f)
    print(f.name, arr.shape, arr.min(), arr.max())