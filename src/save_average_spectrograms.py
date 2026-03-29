from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

real_dir = Path("data/processed/fma_specs")
fake_dir = Path("data/processed/fake_specs")

real_files = sorted(real_dir.glob("*.npy"))
fake_files = sorted(fake_dir.glob("*.npy"))

# ortak isimler
real_map = {p.stem: p for p in real_files}
fake_map = {p.stem: p for p in fake_files}
common = sorted(set(real_map.keys()) & set(fake_map.keys()))

real_stack = []
fake_stack = []

for name in common:
    real_stack.append(np.load(real_map[name]))
    fake_stack.append(np.load(fake_map[name]))

real_mean = np.mean(np.stack(real_stack), axis=0)
fake_mean = np.mean(np.stack(fake_stack), axis=0)

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(real_mean, aspect="auto", origin="lower")
plt.title("Ortalama Gerçek Spectrogram")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(fake_mean, aspect="auto", origin="lower")
plt.title("Ortalama Sahte Spectrogram")
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / "average_real_vs_fake.png", dpi=200)
plt.close()

print("Ortalama spectrogram görseli kaydedildi.")