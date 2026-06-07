"""Zip data/v10_dataset/ -> data/v10_dataset.zip (ZIP_STORED; images already compressed).
Usage:  python scripts/v10_zip.py
"""
import pathlib
import time
import zipfile

src = pathlib.Path("data/v10_dataset")
out = pathlib.Path("data/v10_dataset.zip")
files = sorted(src.iterdir())
t = time.time()
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for p in files:
        z.write(p, p.name)
mb = out.stat().st_size / 1e6
print(f"zipped {len(files)} files -> {out} ({mb:.0f} MB) in {time.time() - t:.0f}s")
