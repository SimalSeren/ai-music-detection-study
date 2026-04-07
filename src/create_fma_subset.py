import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Mevcut FMA klasorunden yalnizca var olan dosyalarla subset listesi uretir.")
    parser.add_argument("--root", type=Path, default=Path("data/fma_small"))
    parser.add_argument("--output-file", type=Path, default=Path("data/fma_subset_available.txt"))
    parser.add_argument("--subset-size", type=int, default=0, help="0 ise tum mevcut dosyalari yazar.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(args.root.rglob("*.mp3"))
    if not files:
        raise ValueError(f"MP3 bulunamadi: {args.root}")

    if args.subset_size and args.subset_size < len(files):
        if args.offset > 0:
            files = files[args.offset : args.offset + args.subset_size]
        else:
            rng = random.Random(args.seed)
            files = sorted(rng.sample(files, args.subset_size))
    elif args.offset > 0:
        files = files[args.offset:]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(path.as_posix() + "\n")

    print(f"Subset kaydedildi: {args.output_file}")
    print(f"Toplam dosya: {len(files)}")


if __name__ == "__main__":
    main()
