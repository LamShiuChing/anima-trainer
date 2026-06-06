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

# --- thresholds (CALIBRATED in a later task; placeholders until then) ---
SHARP_MIN = 0.0             # scale-aware Laplacian variance floor      (set from --calibrate)
FFT_MIN = 0.0              # FFT high-freq energy ratio floor          (set from --calibrate)
JPEG_Q_MIN = 0             # drop JPEGs below this estimated quality    (set from --calibrate; e.g. 85)
BLOCK_MAX = 1e9            # drop above this blockiness                 (set from --calibrate)

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
