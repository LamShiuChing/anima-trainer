"""v8 curation: data/v8_raw/{detail,anchor,bg} -> data/v8_clean + data/v8_manifest.csv.

Replaces stage 1 for the v8 run. Gates each sourced image:
  1. >=1536 px short side          (real detail; <1536 upscales soft)
  2. Laplacian sharpness >= 100    (reject soft/upscaled even if dimensions are large)
  3. phash dedup (hamming 8)       (kill near-duplicate reposts)
  4. AR-crop to 0.66-1.5           (Anima DiT pos-emb 120-patch cap; wide/tall else crash at 1536)
Writes a flat data/v8_clean/ + a manifest with a `bucket` column and width/height/blur_var/phash so the
existing stage 3 (caption) and stage 4 (build) consume it unchanged. Prints the 60/35/5 split + a warning.

Reuses stage-1 helpers (blur_variance, phash, image_size, dedup, is_corrupt, common.*).
Pure helpers (ar_crop_box, ratio_ok) stay import-safe (stdlib only) so tests can import this module.
"""
import sys
from pathlib import Path

RAW = Path("data/v8_raw")
CLEAN = Path("data/v8_clean")
MANIFEST = "data/v8_manifest.csv"
BUCKETS = ["detail", "anchor", "bg"]
MIN_SHORT = 1536
BLUR_MIN = 100.0
HAMMING = 8
MIN_AR, MAX_AR = 0.66, 1.5
TARGET = {"detail": 0.60, "anchor": 0.35, "bg": 0.05}


def ar_crop_box(w, h, min_ar=MIN_AR, max_ar=MAX_AR):
    """Center-crop (left,top,right,bottom) so w/h lands in [min_ar,max_ar]. No-op if already in range.
    Trims the LONG side only -> the short side (and thus min-dimension >=1536) is preserved."""
    ar = w / h
    if min_ar <= ar <= max_ar:
        return (0, 0, w, h)
    if ar > max_ar:                       # too wide -> trim width
        new_w = round(max_ar * h)
        off = (w - new_w) // 2
        return (off, 0, off + new_w, h)
    new_h = round(w / min_ar)             # too tall -> trim height
    off = (h - new_h) // 2
    return (0, off, w, off + new_h)


def ratio_ok(counts, target=TARGET, tol=0.10):
    """True iff every bucket's share is within `tol` of its target. Returns (ok, message)."""
    total = sum(counts.values()) or 1
    for k, frac in target.items():
        actual = counts.get(k, 0) / total
        if abs(actual - frac) > tol:
            return False, f"bucket '{k}' at {actual:.0%} (target {frac:.0%})"
    return True, "ratios within tolerance"


def _load_stage1():
    """Import 01_ingest_clean.py by path (leading digit -> not a normal module name)."""
    import importlib.util
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))     # so stage1's `import common` resolves
    spec = importlib.util.spec_from_file_location("ingest1", here / "01_ingest_clean.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import shutil
    from PIL import Image
    s1 = _load_stage1()
    common = s1.common

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    rows, counts = [], {b: 0 for b in BUCKETS}
    for bucket in BUCKETS:
        src = RAW / bucket
        if not src.is_dir():
            continue
        gated = []
        for p in common.iter_images(src):
            if s1.is_corrupt(p):
                continue
            w, h = s1.image_size(p)
            if min(w, h) < MIN_SHORT:
                continue
            if s1.blur_variance(p) < BLUR_MIN:
                continue
            gated.append(p)
        keep, _drop = s1.dedup(gated, HAMMING)
        for p in keep:
            w, h = s1.image_size(p)
            box = ar_crop_box(w, h)
            im = Image.open(p).convert("RGB")
            if box != (0, 0, w, h):
                im = im.crop(box)
            dest = CLEAN / f"{bucket}_{p.stem}.jpg"
            im.save(dest, quality=95)
            cw, ch = im.size
            rows.append({
                "path": str(dest).replace("\\", "/"), "bucket": bucket,
                "width": str(cw), "height": str(ch),
                "phash": str(s1.phash(p)), "blur_var": f"{s1.blur_variance(p):.1f}",
                "dropped": "False", "drop_reason": "",
            })
            counts[bucket] += 1

    common.write_manifest(MANIFEST, rows)
    ok, msg = ratio_ok(counts)
    print(f"v8 curate: kept {sum(counts.values())} {counts} -> {CLEAN}")
    print(f"ratio check: {'OK' if ok else 'WARN'} - {msg}")
    print(f"manifest -> {MANIFEST}  (next: python src/03_caption.py)")


if __name__ == "__main__":
    main()
