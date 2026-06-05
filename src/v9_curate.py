"""v9 curation: whole pool (data/raw + data/v8_raw + data/v9_x + data/v9_nsfw) -> data/v9_clean
+ data/v9_manifest.csv. Per-image gates (no source exclusion):
  1. >=1536 px short side          (real detail; <1536 upscales soft)
  2. overall Laplacian >= 100      (reject soft/upscaled/compressed)
  3. grid-patch background-sharp   (NEW: drop bokeh/soft-bg; the v9 fix)
  4. phash dedup (hamming 8)       (GLOBAL across sources; keep highest-res per group)
  5. AR-crop to 0.66-1.5           (Anima DiT pos-emb 120-patch cap; else crash at 1536)
Underage block is NOT here -- it stays in stage 3 (03_caption.py WD14 block_tags), which runs
before training. Output manifest columns (path, source, width, height, phash, blur_var, bg_metric,
dropped, drop_reason) are consumed unchanged by stage 3 (caption) + stage 4 (build).

Parallel: ProcessPoolExecutor(24) + tqdm for the gate + crop passes (user has 24 cores).
Heavy libs (cv2/numpy/PIL/imagehash/tqdm/common) are imported INSIDE functions so this module
imports with stdlib only -> tests can import the pure helpers without those deps installed.
"""
from pathlib import Path

# --- sources + gate constants ---
# data/v9_x is a manual STAGING area for X pulls -> hand-pick keepers into data/raw; NOT curated directly.
SOURCES = ["data/raw", "data/v8_raw", "data/v9_nsfw"]
CLEAN = Path("data/v9_clean")
MANIFEST = "data/v9_manifest.csv"
MIN_SHORT = 1536
BLUR_MIN = 100.0
HAMMING = 8
MIN_AR, MAX_AR = 0.66, 1.5
WORKERS = 24

# --- background-sharpness gate (grid-patch, "approach C") ---
GRID_N = 4                  # 4x4 = 16 tiles
BG_TILE_T = 100.0           # a tile is "sharp" if its Laplacian variance >= this
BG_MIN_SHARP_FRAC = 0.5     # keep image if >= this fraction of tiles are sharp


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


def passes_bg_sharpness(tile_vars, tile_t=BG_TILE_T, min_frac=BG_MIN_SHARP_FRAC):
    """USER TUNING POINT -- this rule defines "good background" for the model.

    Default rule: keep the image if at least `min_frac` of grid tiles have Laplacian variance
    >= `tile_t`. Bokeh has a bimodal sharpness map (sharp subject blob + soft everything-else)
    -> few sharp tiles -> fails. Deep-focus is uniformly sharp -> most tiles sharp -> passes.

    Alternatives to try (see spec; calibrate in Task 7's calibration step):
      - median(tile_vars) >= tile_t
      - min(tile_vars) >= some floor   (penalizes ANY large soft region, e.g. one blurred corner)
      - 25th-percentile tile >= floor
    Tune `tile_t` / `min_frac` (or swap the rule) after the calibration printout.
    """
    if not tile_vars:
        return False
    sharp = sum(1 for v in tile_vars if v >= tile_t)
    return (sharp / len(tile_vars)) >= min_frac


def grid_laplacian_vars(gray, n=GRID_N):
    """Split a 2D grayscale array into n*n tiles -> Laplacian variance per tile (row-major)."""
    import cv2
    h, w = gray.shape[:2]
    ys = [int(round(i * h / n)) for i in range(n + 1)]
    xs = [int(round(j * w / n)) for j in range(n + 1)]
    out = []
    for i in range(n):
        for j in range(n):
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()) if tile.size else 0.0)
    return out


def _gate_one(task):
    """Worker (spawn-safe, top-level): (path_str, source) -> dict | None.
    Heavy libs imported inside so module import stays light. None = failed a gate / unreadable."""
    path_str, source = task
    import cv2
    import numpy as np
    import imagehash
    from PIL import Image
    try:
        gray = cv2.imdecode(np.fromfile(path_str, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:                       # cv2 can't decode (e.g. some webp) -> skip
            return None
        h, w = gray.shape[:2]
        if min(w, h) < MIN_SHORT:              # gate 1: resolution
            return None
        overall = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if overall < BLUR_MIN:                 # gate 2: overall sharpness (also filters compression)
            return None
        tile_vars = grid_laplacian_vars(gray, GRID_N)
        if not passes_bg_sharpness(tile_vars):  # gate 3: background sharpness (the v9 fix)
            return None
        with Image.open(path_str) as im:        # compute phash; Hamming dedup runs in main (gate 4)
            ph = imagehash.phash(im.convert("RGB"))
        bg_metric = sum(1 for v in tile_vars if v >= BG_TILE_T) / len(tile_vars)
        return {"path": path_str, "source": source, "w": w, "h": h, "px": w * h,
                "blur": overall, "bg_metric": bg_metric, "phash": ph}
    except Exception:
        return None


def _dedup_local(items, threshold):
    """Greedy near-dup grouping on precomputed phashes (no re-open). Keep highest-res per group.
    items: list of dicts with 'phash' (supports `-` => hamming) + 'px' (pixel count)."""
    keep, used = [], set()
    for i in range(len(items)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(items)):
            if j not in used and (items[i]["phash"] - items[j]["phash"]) <= threshold:
                group.append(j)
        used.update(group)
        keep.append(items[max(group, key=lambda k: items[k]["px"])])
    return keep


def _crop_save_one(item):
    """Worker (spawn-safe, top-level): AR-crop + save to CLEAN -> manifest row dict | None."""
    from PIL import Image
    p, w, h = item["path"], item["w"], item["h"]
    try:
        im = Image.open(p).convert("RGB")
        box = ar_crop_box(w, h)
        if box != (0, 0, w, h):
            im = im.crop(box)
        stem = Path(p).stem
        dest = CLEAN / f"{item['source']}_{stem}.jpg"
        k = 1
        while dest.exists():                          # exact-name collision guard (cross-source)
            dest = CLEAN / f"{item['source']}_{stem}_{k}.jpg"
            k += 1
        im.save(dest, quality=95)
        cw, ch = im.size
        return {"path": str(dest).replace("\\", "/"), "source": item["source"],
                "width": str(cw), "height": str(ch), "phash": str(item["phash"]),
                "blur_var": f"{item['blur']:.1f}", "bg_metric": f"{item['bg_metric']:.3f}",
                "dropped": "False", "drop_reason": ""}
    except Exception as e:
        print(f"  skip (save failed) {p}: {e}")
        return None


def main():
    import shutil
    import concurrent.futures as cf
    from tqdm import tqdm
    import common

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src in SOURCES:
        d = Path(src)
        if not d.is_dir():
            print(f"  (skip missing source {src})")
            continue
        for p in common.iter_images(d):
            tasks.append((str(p), d.name))
    print(f"gating {len(tasks)} images from {len(SOURCES)} sources with {WORKERS} workers...")

    survivors = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for res in tqdm(ex.map(_gate_one, tasks, chunksize=8), total=len(tasks), desc="gate"):
            if res is not None:
                survivors.append(res)
    print(f"  passed gates: {len(survivors)}/{len(tasks)}")

    kept = _dedup_local(survivors, HAMMING)
    print(f"  after global dedup: {len(kept)}")

    rows = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for row in tqdm(ex.map(_crop_save_one, kept, chunksize=8), total=len(kept), desc="crop"):
            if row is not None:
                rows.append(row)

    common.write_manifest(MANIFEST, rows)
    dist = {}
    for r in rows:
        dist[r["source"]] = dist.get(r["source"], 0) + 1
    print(f"v9 curate: wrote {len(rows)} pairs {dist} -> {CLEAN}")
    print(f"manifest -> {MANIFEST}  (next: python src/03_caption.py)")


if __name__ == "__main__":
    main()
