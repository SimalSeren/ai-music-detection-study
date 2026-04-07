import argparse
from pathlib import Path

import librosa
import soundfile as sf

from audio_utils import load_audio_ffmpeg


def parse_args():
    parser = argparse.ArgumentParser(description="Griffin-Lim tabanli reconstruction family uretir.")
    parser.add_argument("--input-list", type=Path, default=Path("data/fma_subset_1000.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("data/reconstructed"))
    parser.add_argument("--family-name", type=str, default="griffinlim_mel32")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--n-iter", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_paths(input_list: Path):
    with open(input_list, "r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def main():
    args = parse_args()
    output_dir = args.output_root / args.family_name
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = load_paths(args.input_list)
    if args.limit > 0:
        audio_paths = audio_paths[args.offset : args.offset + args.limit]
    elif args.offset > 0:
        audio_paths = audio_paths[args.offset:]
    for index, audio_path in enumerate(audio_paths, start=1):
        try:
            y = load_audio_ffmpeg(audio_path, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds)

            stft = librosa.stft(
                y=y,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )
            magnitude = abs(stft)

            y_recon = librosa.griffinlim(
                magnitude,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_iter=args.n_iter,
            )

            out_path = output_dir / f"{audio_path.stem}.wav"
            sf.write(out_path, y_recon, args.sample_rate)

            if index <= 5:
                print(f"[{args.family_name}] Kaydedildi: {out_path}")
            elif index % 10 == 0:
                print(f"[{args.family_name}] Ilerleme: {index}/{len(audio_paths)}")
        except Exception as exc:
            print(f"[{args.family_name}] Hata: {audio_path} -> {exc}")

    print(f"{args.family_name} reconstruction tamamlandi.")


if __name__ == "__main__":
    main()
