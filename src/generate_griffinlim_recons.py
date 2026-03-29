from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

input_list = Path("data/fma_subset_500.txt")
output_dir = Path("data/reconstructed/griffinlim")
output_dir.mkdir(parents=True, exist_ok=True)

target_sr = 16000
target_len_sec = 10

with open(input_list, "r", encoding="utf-8") as f:
    paths = [Path(line.strip()) for line in f if line.strip()]

for i, audio_path in enumerate(paths):
    try:
        # yükle
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # sabit uzunluk
        max_len = target_len_sec * target_sr
        y = y[:max_len]
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))

        # mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=128,
            power=2.0
        )

        # mel → audio (Griffin-Lim tabanlı yaklaşık reconstruction)
        y_recon = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=target_sr,
            n_fft=1024,
            hop_length=320,
            n_iter=32
        )

        out_path = output_dir / f"{audio_path.stem}.wav"
        sf.write(out_path, y_recon, target_sr)

        if i < 5:
            print(f"Kaydedildi: {out_path}")

    except Exception as e:
        print(f"Hata: {audio_path} -> {e}")

print("Griffin-Lim reconstruction tamamlandı.")