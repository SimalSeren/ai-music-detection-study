from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Örnek ses dosyası yolu
audio_path = Path("data/fma_small/000/000002.mp3")

# Dosya var mı kontrol et
if not audio_path.exists():
    raise FileNotFoundError(f"Dosya bulunamadı: {audio_path}")

# Ses yükle
y, sr = librosa.load(audio_path, sr=None, mono=True)

print("Dosya:", audio_path)
print("Sample rate:", sr)
print("Süre (sn):", len(y) / sr)
print("Waveform shape:", y.shape)

# STFT -> genlik spektrogramı
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

# dB ölçeğine çevir
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Görselleştir
plt.figure(figsize=(12, 5))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Amplitude Spectrogram (dB)")
plt.tight_layout()
plt.show()