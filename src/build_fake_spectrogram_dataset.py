from pathlib import Path
import librosa
import numpy as np

input_dir = Path("data/reconstructed/griffinlim")
output_dir = Path("data/processed/fake_specs")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.wav"))

for i, audio_path in enumerate(files):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        max_len = 10 * sr
        y = y[:max_len]
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))

        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        out_path = output_dir / f"{audio_path.stem}.npy"
        np.save(out_path, S_db)

        if i < 5:
            print(f"Kaydedildi: {out_path} | shape={S_db.shape}")

    except Exception as e:
        print(f"Hata: {audio_path} -> {e}")

print("Fake spectrogram üretimi tamamlandı.")