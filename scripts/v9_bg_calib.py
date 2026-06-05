"""v9 background-sharpness gate CALIBRATION helper.

Point it at a folder of sample images (mix known-bokeh + known-deep-focus). It prints, per image,
whether the gate PASSES + the sharp-tile fraction + the per-tile Laplacian variances, so you can
tune BG_TILE_T / BG_MIN_SHARP_FRAC (or the rule) in src/v9_curate.py BEFORE running the full curate.

Usage:
  python scripts/v9_bg_calib.py path/to/sample_folder
  python scripts/v9_bg_calib.py path/to/sample_folder --tile-t 120 --min-frac 0.6   # try overrides

Read the output: a deep-focus image should have most tiles >> tile_t; a bokeh image should have a
few high (subject) tiles and many low (blurred bg) tiles. Set tile_t between those two clusters and
min_frac so deep-focus PASSES and bokeh FAILS. Then edit the constants in src/v9_curate.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import v9_curate as v  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Calibrate the v9 grid background-sharpness gate.")
    ap.add_argument("folder", help="folder of sample images (mix bokeh + deep-focus)")
    ap.add_argument("--tile-t", type=float, default=v.BG_TILE_T, help=f"sharp-tile threshold (default {v.BG_TILE_T})")
    ap.add_argument("--min-frac", type=float, default=v.BG_MIN_SHARP_FRAC, help=f"min sharp fraction (default {v.BG_MIN_SHARP_FRAC})")
    args = ap.parse_args()

    import cv2
    import numpy as np

    folder = Path(args.folder)
    paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not paths:
        print(f"no images under {folder}")
        return
    print(f"tile_t={args.tile_t}  min_frac={args.min_frac}  (grid {v.GRID_N}x{v.GRID_N})\n")
    for p in paths:
        gray = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ?? unreadable        {p.name}")
            continue
        tv = v.grid_laplacian_vars(gray, v.GRID_N)
        frac = sum(1 for x in tv if x >= args.tile_t) / len(tv)
        ok = v.passes_bg_sharpness(tv, tile_t=args.tile_t, min_frac=args.min_frac)
        tiles = " ".join(f"{int(x):>5}" for x in tv)
        print(f"  {'PASS' if ok else 'DROP'}  frac={frac:.2f}  {p.name}\n        tiles: {tiles}")


if __name__ == "__main__":
    main()
