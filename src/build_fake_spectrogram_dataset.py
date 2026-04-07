import argparse
from pathlib import Path

import numpy as np

from audio_utils import load_audio_ffmpeg
from spectrogram_utils import compute_log_spectrogram


AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac")


def parse_args():
    parser = argparse.ArgumentParser(description="Fake audio family'lerinden spectrogram dataset olusturur.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/reconstructed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/fake_specs"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    return parser.parse_args()


def discover_family_dirs(input_dir: Path):
    direct_files = []
    for pattern in AUDIO_EXTENSIONS:
        direct_files.extend(sorted(input_dir.glob(pattern)))

    if direct_files:
        return {"griffinlim": input_dir}

    family_dirs = {}
    for subdir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        audio_files = []
        for pattern in AUDIO_EXTENSIONS:
            audio_files.extend(sorted(subdir.glob(pattern)))
        if audio_files:
            family_dirs[subdir.name] = subdir

    if not family_dirs:
        raise FileNotFoundError(f"Fake audio bulunamadi: {input_dir}")

    return family_dirs


def list_audio_files(directory: Path):
    files = []
    for pattern in AUDIO_EXTENSIONS:
        files.extend(sorted(directory.glob(pattern)))
    return files


def main():
    args = parse_args()
    family_dirs = discover_family_dirs(args.input_dir)

    max_len = args.clip_seconds * args.sample_rate

    for family_name, family_dir in family_dirs.items():
        family_output_dir = args.output_dir / family_name
        family_output_dir.mkdir(parents=True, exist_ok=True)

        for index, audio_path in enumerate(list_audio_files(family_dir), start=1):
            try:
                y = load_audio_ffmpeg(audio_path, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds)

                spec_db = compute_log_spectrogram(
                    y,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )

                output_path = family_output_dir / f"{audio_path.stem}.npy"
                np.save(output_path, spec_db)

                if index <= 3:
                    print(f"[{family_name}] Kaydedildi: {output_path} | shape={spec_db.shape}")
            except Exception as exc:
                print(f"[{family_name}] Hata: {audio_path} -> {exc}")

    print("Fake spectrogram dataset tamamlandi.")


if __name__ == "__main__":
    main()
