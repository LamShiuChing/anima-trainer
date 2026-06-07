"""Zip the v10 training data for upload (ZIP_STORED; images already compressed).

Emits TWO zips (matching scripts/vast_fetch_v10.sh's two-ID fetch):
  data/v10_dataset.zip  <- data/v10_dataset   (main curated+captioned set)
  data/v10_char.zip     <- data/v10_char       (oversampled trigger-word character set, if present)
Upload both to Drive; pass the dataset ID then the char ID to vast_fetch_v10.sh.
Usage:  python scripts/v10_zip.py
"""
import pathlib
import time
import zipfile


def zip_dir(src, out):
    files = [p for p in sorted(src.iterdir()) if p.is_file()]
    t = time.time()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.name)        # flat
    mb = out.stat().st_size / 1e6
    print(f"zipped {len(files)} files -> {out} ({mb:.0f} MB) in {time.time() - t:.0f}s")


zip_dir(pathlib.Path("data/v10_dataset"), pathlib.Path("data/v10_dataset.zip"))
_char = pathlib.Path("data/v10_char")
if _char.is_dir() and any(_char.iterdir()):
    zip_dir(_char, pathlib.Path("data/v10_char.zip"))
