"""Stage 1: dedup, drop tiny/blurry/corrupt, OCR-flag meme/screenshot text -> data/clean/ + manifest."""
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

import common

LOG = common.setup_logging()


def is_corrupt(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def is_too_small(path, min_size):
    w, h = image_size(path)
    return min(w, h) < min_size


def blur_variance(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def hamming(h1, h2):
    return h1 - h2


def dedup(paths, hamming_threshold):
    """Greedy near-dup grouping; within a group keep the highest-resolution image."""
    hashes = {p: phash(p) for p in paths}
    keep, drop, used = [], [], set()
    for p in paths:
        if p in used:
            continue
        group = [p]
        for q in paths:
            if q is p or q in used:
                continue
            if hamming(hashes[p], hashes[q]) <= hamming_threshold:
                group.append(q)
        for g in group:
            used.add(g)
        best = max(group, key=lambda x: image_size(x)[0] * image_size(x)[1])
        keep.append(best)
        drop.extend([g for g in group if g is not best])
    return keep, drop


def ocr_text_area_ratio(path, engine):
    """Fraction of image area covered by detected text boxes (0..1)."""
    result, _ = engine(str(path))
    if not result:
        return 0.0
    with Image.open(path) as im:
        area = im.size[0] * im.size[1]
    text_area = 0.0
    for box, _text, _conf in result:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        text_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return min(text_area / area, 1.0)


def main():
    cfg = common.load_config()
    ing = cfg["ingest"]
    raw = Path(cfg["paths"]["raw"])
    clean = Path(cfg["paths"]["clean"])
    clean.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    all_imgs = list(common.iter_images(raw))
    LOG.info("Stage 1: %d raw images", len(all_imgs))

    survivors, rows = [], []
    for p in all_imgs:
        reason = ""
        if is_corrupt(p):
            reason = "corrupt"
        elif is_too_small(p, ing["min_size"]):
            reason = "too_small"
        elif blur_variance(p) < ing["blur_var_threshold"]:
            reason = "blurry"
        if reason:
            rows.append({"path": str(p), "dropped": "True", "drop_reason": reason})
        else:
            survivors.append(p)

    keep, dup_drop = dedup(survivors, ing["phash_hamming_threshold"])
    for p in dup_drop:
        rows.append({"path": str(p), "dropped": "True", "drop_reason": "duplicate"})

    for p in keep:
        w, h = image_size(p)
        ratio = ocr_text_area_ratio(p, ocr)
        flagged = ratio > ing["ocr_text_area_ratio_flag"]
        dest = clean / p.name
        shutil.copy2(p, dest)
        rows.append({
            "path": str(dest), "width": str(w), "height": str(h),
            "phash": str(phash(p)), "blur_var": f"{blur_variance(p):.1f}",
            "ocr_ratio": f"{ratio:.3f}", "dropped": "False",
            "drop_reason": "text_overlay_flag" if flagged else "",
        })

    common.write_manifest(cfg["paths"]["manifest"], rows)
    LOG.info("Stage 1 done: kept %d, dropped %d -> %s", len(keep), len(rows) - len(keep), clean)


if __name__ == "__main__":
    main()
