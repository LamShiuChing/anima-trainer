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
SOURCES = ["data/raw", "data/v8_raw", "data/v9_x", "data/v9_nsfw"]
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
