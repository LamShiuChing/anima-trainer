"""v10 curation: data/raw -> data/v10_clean + data/v10_manifest.csv.

Photoreal render-style restart. Measures REAL high-frequency detail (not nominal pixels) so
'big but upscaled/soft/compressed' files drop while genuinely-sharp photos survive. Gates:
  1. corrupt/unreadable drop
  2. min(w,h) >= 1280              (nominal floor; <1280 upscales too hard at 1536)
  3. phash dedup (hamming 8)       (keep highest-px per near-dup group)
  4. scale-aware sharpness         (Laplacian var on a fixed 512px-long analysis copy)
  5. FFT high-freq energy ratio    (true-detail signal; kills upscaled/soft 'fake big')
  6. JPEG quality + blockiness     (kills heavily compressed)
  7. AR-crop to 0.66-1.5           (Anima DiT pos-emb 120-patch / 1920px cap)
Underage backstop runs in src/v10_caption.py (GPU taggers load there).
Emits ALL metrics to the manifest so thresholds tune from the real distribution.
`python src/v10_curate.py --calibrate 200` samples N images, prints percentiles, writes nothing.

Pure helpers (ar_crop_box, analysis_resize_dims, fft_highfreq_ratio, blockiness,
jpeg_quality_estimate) are stdlib/numpy-only and import-safe for tests.
"""
import sys
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/v10_clean")
MANIFEST = "data/v10_manifest.csv"

MIN_SHORT = 1280
HAMMING = 8
MIN_AR, MAX_AR = 0.66, 1.5
ANALYSIS_LONG = 512          # sharpness measured on a 512px-long copy (scale-aware, comparable)
FFT_CUTOFF = 0.25           # high-freq band starts at 0.25 * Nyquist

# --- thresholds (CALIBRATED 2026-06-06 on a 120-img sample of data/raw; gentle ~p10/p90 cuts) ---
SHARP_MIN = 190.0           # ~p10 sharp (drop softest ~10%); raise to 345 (p25) for a stricter/smaller set
FFT_MIN = 0.52             # ~p5-p10 fft_hf (drop softest/upscaled tail; metric range is narrow 0.41-0.76)
JPEG_Q_MIN = 70            # ~p10; NOT 85 — raw is social-media JPEG (median q83), 85 would drop ~60%
BLOCK_MAX = 1.6            # ~p90-p95 blockiness (drop blockiest/most-compressed tail)

# Standard JPEG luminance quantization base table (ITU-T T.81 Annex K.1)
STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


def ar_crop_box(w, h, min_ar=MIN_AR, max_ar=MAX_AR):
    """Center-crop (left,top,right,bottom) so w/h lands in [min_ar,max_ar]. No-op if in range.
    Trims the LONG side only -> short side (>=1280 floor) preserved."""
    ar = w / h
    if min_ar <= ar <= max_ar:
        return (0, 0, w, h)
    if ar > max_ar:
        new_w = round(max_ar * h)
        off = (w - new_w) // 2
        return (off, 0, off + new_w, h)
    new_h = round(w / min_ar)
    off = (h - new_h) // 2
    return (0, off, w, off + new_h)


def analysis_resize_dims(w, h, target_long=ANALYSIS_LONG):
    """Dims for a downscaled analysis copy whose LONG side == target_long. Never upscales."""
    long_side = max(w, h)
    if long_side <= target_long:
        return (w, h)
    s = target_long / long_side
    return (max(1, round(w * s)), max(1, round(h * s)))


def fft_highfreq_ratio(gray, cutoff=FFT_CUTOFF):
    """Fraction of 2D-FFT magnitude energy at radius > cutoff*Nyquist. Upscaled/soft -> low ratio."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float64)
    g = g - g.mean()
    if not np.any(g):
        return 0.0
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = g.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)   # 0 at center, ~1 at Nyquist
    total = mag.sum()
    if total <= 0:
        return 0.0
    return float(mag[r > cutoff].sum() / total)


def blockiness(gray):
    """Mean |gradient| on 8px block boundaries minus off-boundary. >0 => JPEG block artifacts."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float64)
    dh = np.abs(np.diff(g, axis=1))
    cols = np.arange(dh.shape[1])
    on = dh[:, (cols % 8) == 7]
    off = dh[:, (cols % 8) != 7]
    if on.size == 0 or off.size == 0:
        return 0.0
    return float(on.mean() - off.mean())


def jpeg_quality_estimate(qtable):
    """Approx libjpeg quality (1..100) from a luminance quant table (list of 64 ints).
    None/empty (e.g. PNG) -> 100. Inverts libjpeg's quality->scale mapping via avg ratio vs STD_LUMA."""
    if not qtable:
        return 100
    ratios = [q / b for q, b in zip(qtable, STD_LUMA) if b]
    if not ratios:
        return 100
    scale = 100.0 * sum(ratios) / len(ratios)      # libjpeg: qtable ~= base * scale / 100
    if scale <= 0:
        return 100
    q = (200.0 - scale) / 2.0 if scale < 100 else 5000.0 / scale
    return int(max(1, min(100, round(q))))


def _to_gray(path):
    """Read an image as a float64 grayscale numpy array (unicode-path safe via np.fromfile)."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cv2 could not decode")
    return img


def _analysis_small(gray):
    """Downscale to a fixed 512px-long float64 copy -> scale-comparable AND cheap for
    sharpness + FFT (a full-res 2D FFT on a 6k image is seconds/image; on the 512 copy it is ms)."""
    import cv2
    import numpy as np
    h, w = gray.shape
    nw, nh = analysis_resize_dims(w, h)
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA) if (nw, nh) != (w, h) else gray
    return small.astype(np.float64)


def scale_aware_sharpness(gray):
    """Laplacian variance on the fixed-size analysis copy -> comparable across resolutions."""
    import cv2
    return float(cv2.Laplacian(_analysis_small(gray), cv2.CV_64F).var())


def jpeg_qtable_luma(path):
    """Luminance quant table (list of 64) for a JPEG, else None (PNG/etc)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            q = getattr(im, "quantization", None)
        return list(q[0]) if q and 0 in q else None
    except Exception:
        return None


def metrics_for(path, w=None, h=None):
    """All quality metrics for one image. Sharpness + FFT run on a fixed 512px analysis copy
    (scale-comparable + fast); blockiness on FULL-res (the 8px JPEG block grid must stay intact).
    w/h are accepted for call-site convenience but recomputed from the decoded array."""
    import cv2
    gray = _to_gray(path)
    small = _analysis_small(gray)
    return {
        "sharp": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        "fft_hf": fft_highfreq_ratio(small),
        "blockiness": blockiness(gray),
        "jpeg_q": jpeg_quality_estimate(jpeg_qtable_luma(path)),
    }


def quality_reject(m):
    """Return a drop_reason string if any quality gate fails, else ''."""
    if m["sharp"] < SHARP_MIN:
        return f"soft(sharp={m['sharp']:.0f})"
    if m["fft_hf"] < FFT_MIN:
        return f"upscaled(fft={m['fft_hf']:.3f})"
    if m["jpeg_q"] < JPEG_Q_MIN:
        return f"compressed(q={m['jpeg_q']})"
    if m["blockiness"] > BLOCK_MAX:
        return f"blocky({m['blockiness']:.1f})"
    return ""


def _dedup_local(items, threshold):
    """Greedy near-dup grouping on precomputed phashes; keep highest-px per group."""
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


def _load_stage1():
    import importlib.util
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("ingest1", here / "01_ingest_clean.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gather(s1):
    """Floor + readability + phash, returning survivor items (pre-dedup, pre-quality). Shows progress."""
    from tqdm import tqdm
    common = s1.common
    paths = list(common.iter_images(RAW))      # cheap: just paths (no decode) -> gives tqdm a total
    items, skipped = [], 0
    for p in tqdm(paths, desc="scan floor+phash", unit="img", dynamic_ncols=True):
        try:
            if s1.is_corrupt(p):
                continue
            w, h = s1.image_size(p)
            if min(w, h) < MIN_SHORT:
                continue
            ph = s1.phash(p)
        except Exception as e:
            tqdm.write(f"  skip (unreadable) {p.name}: {e}")
            skipped += 1
            continue
        items.append({"path": p, "phash": ph, "w": w, "h": h, "px": w * h})
    print(f"scan: {len(items)} survivors >= {MIN_SHORT}px of {len(paths)} raw files "
          f"({skipped} unreadable)", flush=True)
    return items, skipped


def calibrate(n):
    """Sample ~n readable >=floor images DIRECTLY (no full phash scan / no dedup) and print metric
    percentiles to pick thresholds. Writes nothing. Fast: strides across the pool + early-stops at n."""
    import numpy as np
    from tqdm import tqdm
    s1 = _load_stage1()
    common = s1.common
    paths = list(common.iter_images(RAW))               # cheap: just paths (no decode)
    step = max(1, len(paths) // (n * 3))                 # spread across pool; ~3x candidates (floor cuts ~half)
    candidates = paths[::step]
    print(f"calibrate: {len(paths)} raw files; probing every {step}th (~{len(candidates)}) "
          f"to collect {n} metric samples...", flush=True)
    cols = {"sharp": [], "fft_hf": [], "blockiness": [], "jpeg_q": []}
    bar = tqdm(candidates, desc="calibrate", unit="img", dynamic_ncols=True)
    for p in bar:
        if len(cols["sharp"]) >= n:
            break
        try:
            if s1.is_corrupt(p):
                continue
            w, h = s1.image_size(p)
            if min(w, h) < MIN_SHORT:
                continue
            m = metrics_for(p)
        except Exception as e:
            tqdm.write(f"  skip {p.name}: {e}")
            continue
        for k in cols:
            cols[k].append(m[k])
        bar.set_postfix(kept=len(cols["sharp"]))
    bar.close()
    n_ok = len(cols["sharp"])
    if n_ok == 0:
        print(f"no readable images >= {MIN_SHORT}px in the sample — check data/raw", flush=True)
        return
    print(f"\ncalibration on {n_ok} sampled images:")
    for k, vals in cols.items():
        a = np.array(vals, dtype=float)
        pcts = {p: float(np.percentile(a, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
        print(f"  {k:11s} min={a.min():.3f} " + " ".join(f"p{p}={pcts[p]:.3f}" for p in pcts) + f" max={a.max():.3f}")
    print("\nSet SHARP_MIN / FFT_MIN / JPEG_Q_MIN / BLOCK_MAX in v10_curate.py from these, then run without --calibrate.")


def main():
    import shutil
    from PIL import Image
    if len(sys.argv) >= 3 and sys.argv[1] == "--calibrate":
        calibrate(int(sys.argv[2]))
        return

    s1 = _load_stage1()
    common = s1.common
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    items, skipped = _gather(s1)
    kept_items = _dedup_local(items, HAMMING)
    print(f"dedup: {len(kept_items)} kept of {len(items)} (removed {len(items) - len(kept_items)} near-dups). "
          f"Computing quality metrics...", flush=True)

    rows, kept, dropped = [], 0, 0
    for it in tqdm(kept_items, desc="metrics+crop", unit="img", dynamic_ncols=True):
        p, w, h = it["path"], it["w"], it["h"]
        try:
            m = metrics_for(p, w, h)
        except Exception as e:
            tqdm.write(f"  skip (metric fail) {p.name}: {e}")
            skipped += 1
            continue
        reason = quality_reject(m)
        if reason:
            rows.append({"path": str(p).replace("\\", "/"), "width": str(w), "height": str(h),
                         "phash": str(it["phash"]), "sharp": f"{m['sharp']:.1f}",
                         "fft_hf": f"{m['fft_hf']:.4f}", "jpeg_q": str(m["jpeg_q"]),
                         "blockiness": f"{m['blockiness']:.2f}", "dropped": "True", "drop_reason": reason})
            dropped += 1
            continue
        try:
            im = Image.open(p).convert("RGB")
            box = ar_crop_box(w, h)
            if box != (0, 0, w, h):
                im = im.crop(box)
            dest = CLEAN / f"{p.stem}.jpg"
            im.save(dest, quality=95)
        except Exception as e:
            tqdm.write(f"  skip (save failed) {p.name}: {e}")
            skipped += 1
            continue
        cw, ch = im.size
        rows.append({"path": str(dest).replace("\\", "/"), "width": str(cw), "height": str(ch),
                     "phash": str(it["phash"]), "sharp": f"{m['sharp']:.1f}",
                     "fft_hf": f"{m['fft_hf']:.4f}", "jpeg_q": str(m["jpeg_q"]),
                     "blockiness": f"{m['blockiness']:.2f}", "dropped": "False", "drop_reason": ""})
        kept += 1

    common.write_manifest(MANIFEST, rows)
    print(f"v10 curate: kept {kept}, quality-dropped {dropped}, skipped {skipped} -> {CLEAN}")
    print(f"manifest -> {MANIFEST}  (next: python src/v10_caption.py)")


if __name__ == "__main__":
    main()
