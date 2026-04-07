import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

from audio_utils import load_audio_ffmpeg


def parse_args():
    parser = argparse.ArgumentParser(description="Codec gerektirmeyen degradation family'leri uretir.")
    parser.add_argument("--input-list", type=Path, default=Path("data/fma_subset_1000.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("data/reconstructed"))
    parser.add_argument(
        "--families",
        nargs="+",
        default=["resample_8k", "quantize_8bit", "smoothed_noise", "mp3_16k", "hard_clipping"],
        help="Uretilecek family listesi.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_paths(input_list: Path):
    with open(input_list, "r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]

import pydub
import io

def degrade_audio(y: np.ndarray, sr: int, family: str):
    if family == "resample_8k":
        degraded = resample_poly(y, up=1, down=2).astype(np.float32)
        degraded = resample_poly(degraded, up=2, down=1).astype(np.float32)
        return degraded
    if family == "quantize_8bit":
        clipped = np.clip(y, -1.0, 1.0)
        levels = 2**8 - 1
        quantized = np.round(((clipped + 1.0) * 0.5) * levels) / levels
        return (quantized * 2.0) - 1.0
    if family == "smoothed_noise":
        noise = np.random.normal(0.0, 0.003, size=y.shape).astype(np.float32)
        smoothed = lfilter([1.0, -0.65], [1.0], y + noise).astype(np.float32)
        return np.tanh(smoothed)
    if family == "hard_clipping":
        return np.clip(y * 1.5, -0.8, 0.8).astype(np.float32)
    if family == "mp3_16k":
        # Convert numpy array to 16-bit PCM for pydub
        y_int16 = np.int16(y * 32767)
        audio_segment = pydub.AudioSegment(
            y_int16.tobytes(), 
            frame_rate=sr,
            sample_width=2, 
            channels=1
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3", bitrate="16k")
        buffer.seek(0)
        degraded_segment = pydub.AudioSegment.from_mp3(buffer)
        samples = np.array(degraded_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
        # return same length as original
        if len(samples) > len(y): samples = samples[:len(y)]
        elif len(samples) < len(y): samples = np.pad(samples, (0, len(y) - len(samples)))
        return samples
def main():
    args = parse_args()
    audio_paths = load_paths(args.input_list)
    if args.limit > 0:
        audio_paths = audio_paths[args.offset : args.offset + args.limit]
    elif args.offset > 0:
        audio_paths = audio_paths[args.offset:]
    max_len = args.clip_seconds * args.sample_rate

    for family in args.families:
        family_dir = args.output_root / family
        family_dir.mkdir(parents=True, exist_ok=True)

        for index, audio_path in enumerate(audio_paths, start=1):
            try:
                y = load_audio_ffmpeg(audio_path, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds)

                y_deg = degrade_audio(y, args.sample_rate, family)
                y_deg = y_deg[:max_len]
                if len(y_deg) < max_len:
                    y_deg = np.pad(y_deg, (0, max_len - len(y_deg)))

                out_path = family_dir / f"{audio_path.stem}.wav"
                sf.write(out_path, y_deg, args.sample_rate)

                if index <= 5:
                    print(f"[{family}] Kaydedildi: {out_path}")
                elif index % 25 == 0:
                    print(f"[{family}] Ilerleme: {index}/{len(audio_paths)}")
            except Exception as exc:
                print(f"[{family}] Hata: {audio_path} -> {exc}")

    print("Degradation family uretimi tamamlandi.")


if __name__ == "__main__":
    main()
