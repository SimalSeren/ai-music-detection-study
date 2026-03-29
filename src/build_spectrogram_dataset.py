from pathlib import Path
import librosa
import numpy as np

input_list = Path("data/fma_subset_1000.txt")
output_dir = Path("data/processed/fma_specs")
output_dir.mkdir(parents=True, exist_ok=True)

with open(input_list, "r", encoding="utf-8") as f:
    paths = [Path(line.strip()) for line in f if line.strip()]

for i, audio_path in enumerate(paths):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # İlk 10 saniyeyi al
        max_len = 10 * sr
        y = y[:max_len]

        # Kısa ise pad et
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))

        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        out_name = audio_path.stem + ".npy"
        out_path = output_dir / out_name
        np.save(out_path, S_db)

        if i < 5:
            print(f"Kaydedildi: {out_path} | shape={S_db.shape}")

    except Exception as e:
        print(f"Hata: {audio_path} -> {e}")

print("Bitti.")