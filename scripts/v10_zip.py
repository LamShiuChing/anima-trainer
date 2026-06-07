"""Zip the v10 training data -> data/v10_dataset.zip (ZIP_STORED; images already compressed).

Bundles two top-level folders so the Vast fetch can place them as separate diffusion-pipe dirs:
  dataset/  <- data/v10_dataset      (the main curated+captioned set)
  char/     <- data/v10_char         (the oversampled trigger-word character set, if present)
Usage:  python scripts/v10_zip.py
"""
import pathlib
import time
import zipfile

OUT = pathlib.Path("data/v10_dataset.zip")
SPECS = [("dataset", pathlib.Path("data/v10_dataset"))]
_char = pathlib.Path("data/v10_char")
if _char.is_dir() and any(_char.iterdir()):
    SPECS.append(("char", _char))

t = time.time()
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for prefix, d in SPECS:
        for p in sorted(d.iterdir()):
            if p.is_file():
                z.write(p, f"{prefix}/{p.name}")
                n += 1
mb = OUT.stat().st_size / 1e6
folders = ", ".join(f"{prefix} ({sum(1 for _ in d.iterdir())})" for prefix, d in SPECS)
print(f"zipped {n} files [{folders}] -> {OUT} ({mb:.0f} MB) in {time.time() - t:.0f}s")
