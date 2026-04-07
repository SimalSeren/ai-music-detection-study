import argparse
from pathlib import Path

import numpy as np

from audio_utils import load_audio_ffmpeg
from spectrogram_utils import compute_log_spectrogram


def parse_args():
    parser = argparse.ArgumentParser(description="Gercek seslerden spectrogram dataset olusturur.")
    parser.add_argument("--input-list", type=Path, default=Path("data/fma_subset_1000.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/fma_specs"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    return parser.parse_args()


def load_paths(input_list: Path):
    with open(input_list, "r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_len = args.clip_seconds * args.sample_rate
    audio_paths = load_paths(args.input_list)

    for index, audio_path in enumerate(audio_paths, start=1):
        try:
            y = load_audio_ffmpeg(audio_path, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds)

            spec_db = compute_log_spectrogram(
                y,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )

            output_path = args.output_dir / f"{audio_path.stem}.npy"
            np.save(output_path, spec_db)

            if index <= 5:
                print(f"Kaydedildi: {output_path} | shape={spec_db.shape}")
        except Exception as exc:
            print(f"Hata: {audio_path} -> {exc}")

    print("Gercek spectrogram dataset tamamlandi.")


if __name__ == "__main__":
    main()
