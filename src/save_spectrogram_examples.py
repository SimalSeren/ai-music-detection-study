from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Gerçek ve fake spectrogram klasörleri
real_dir = Path("data/processed/fma_specs")
fake_dir = Path("data/processed/fake_specs")

# Ortak dosya isimlerini bul
common = sorted(set(p.stem for p in real_dir.glob("*.npy")) & set(p.stem for p in fake_dir.glob("*.npy")))

if not common:
    raise ValueError("Ortak real/fake spectrogram bulunamadı.")

# İlk ortak örneği seç
name = common[0]

real = np.load(real_dir / f"{name}.npy")
fake = np.load(fake_dir / f"{name}.npy")

# Çıktı klasörü
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Real spectrogram kaydet
plt.figure(figsize=(10, 4))
plt.imshow(real, aspect="auto", origin="lower")
plt.title(f"Gerçek Spectrogram - {name}")
plt.colorbar()
plt.tight_layout()
plt.savefig(output_dir / f"real_{name}.png", dpi=200)
plt.close()

# Fake spectrogram kaydet
plt.figure(figsize=(10, 4))
plt.imshow(fake, aspect="auto", origin="lower")
plt.title(f"Sahte Spectrogram (Griffin-Lim) - {name}")
plt.colorbar()
plt.tight_layout()
plt.savefig(output_dir / f"fake_{name}.png", dpi=200)
plt.close()

# Karşılaştırmalı tek figür olarak kaydet
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

print("Spectrogram görselleri kaydedildi:")
print(output_dir / f"real_{name}.png")
print(output_dir / f"fake_{name}.png")
print(output_dir / f"comparison_{name}.png")