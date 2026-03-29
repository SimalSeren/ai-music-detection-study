from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

real_dir = Path("data/processed/fma_specs")
fake_dir = Path("data/processed/fake_specs")

common = sorted(set(p.stem for p in real_dir.glob("*.npy")) & set(p.stem for p in fake_dir.glob("*.npy")))

if not common:
    raise ValueError("Ortak real/fake spectrogram bulunamadı.")

output_dir = Path("results/figures/multiple_examples")
output_dir.mkdir(parents=True, exist_ok=True)

# İlk 5 örnek
selected = common[:5]

for name in selected:
    real = np.load(real_dir / f"{name}.npy")
    fake = np.load(fake_dir / f"{name}.npy")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(real, aspect="auto", origin="lower")
    plt.title(f"Gerçek - {name}")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(fake, aspect="auto", origin="lower")
    plt.title(f"Sahte (Griffin-Lim) - {name}")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_{name}.png", dpi=200)
    plt.close()

print("İlk 5 örnek için karşılaştırma görselleri kaydedildi.")