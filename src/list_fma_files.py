from pathlib import Path

print("script başladı")

root = Path("data/fma_small")
print("root:", root)
print("var mı:", root.exists())

files = sorted(root.rglob("*.mp3"))

print("Toplam mp3 sayısı:", len(files))
print("İlk 10 dosya:")
for f in files[:10]:
    print(f)