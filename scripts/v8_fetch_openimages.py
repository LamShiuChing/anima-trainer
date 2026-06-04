"""v8 sourcing — pull Open Images V7 via FiftyOne, bucket into data/v8_raw/{detail,anchor,bg}.

What it does:
  - detail classes (hands/phones/footwear/clothing) -> bbox-CROP the object (padded), saved
    ONLY if the crop clears --min-short (so the detail is genuinely high-res, not an upscale).
  - anchor: full image kept when a person-class is present and min(w,h) >= --min-short.
  - bg: full image kept when NO person is present, a background class is present, >= --min-short.

HONEST LIMITATION: Open Images source images skew <=1024 px, so most detail bbox crops will NOT
clear 1536 -> expect FEW detail crops here. This script is for VARIETY / anchor / bg. The real
high-frequency detail fuel = your manual Unsplash/Pixabay downloads in data/v8_raw/detail/.

Run:
  pip install fiftyone
  python scripts/v8_fetch_openimages.py --max-samples 2000
  python scripts/v8_fetch_openimages.py --max-samples 500 --split validation   # more, different split
"""
import argparse
import sys
from pathlib import Path

# Open Images V7 detection label -> our detail subtype
DETAIL_CLASSES = {
    "Human hand": "hand", "Human foot": "foot", "Footwear": "foot",
    "Sandal": "foot", "High heels": "foot",
    "Mobile phone": "phone", "Telephone": "phone",
    "Jeans": "fabric", "Dress": "fabric", "Shirt": "fabric", "Suit": "fabric",
    "Trousers": "fabric", "Sweater": "fabric", "Jacket": "fabric",
}
ANCHOR_CLASSES = ["Person", "Woman", "Man", "Girl", "Boy"]
BG_CLASSES = [
    "Couch", "Bed", "Kitchen & dining room table", "Houseplant", "Coffee table",
    "Curtain", "Bookcase", "Stairs", "Fireplace", "Sink", "Window",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Pull Open Images into data/v8_raw buckets.")
    ap.add_argument("--out", default="data/v8_raw", help="output root (has detail/anchor/bg)")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--max-samples", type=int, default=2000, help="OI samples to scan")
    ap.add_argument("--min-short", type=int, default=1536, help="min short side (px) to keep")
    ap.add_argument("--pad", type=float, default=0.15, help="bbox pad fraction for detail crops")
    return ap.parse_args()


def _require_fiftyone():
    try:
        import fiftyone as fo  # noqa: F401
        import fiftyone.zoo as foz  # noqa: F401
        return fo, foz
    except ImportError:
        sys.stderr.write("FiftyOne not installed. Run:  pip install fiftyone\n")
        sys.exit(1)


def crop_box(det, w, h, pad):
    """FiftyOne bbox [x,y,bw,bh] (normalized, top-left origin) -> padded pixel (l,t,r,b)."""
    x, y, bw, bh = det.bounding_box
    l, t = x * w, y * h
    r, b = (x + bw) * w, (y + bh) * h
    pw, ph = (r - l) * pad, (b - t) * pad
    l = max(0, int(l - pw)); t = max(0, int(t - ph))
    r = min(w, int(r + pw)); b = min(h, int(b + ph))
    return l, t, r, b


def main():
    args = parse_args()
    fo, foz = _require_fiftyone()
    from PIL import Image

    out = Path(args.out)
    for b in ("detail", "anchor", "bg"):
        (out / b).mkdir(parents=True, exist_ok=True)

    classes = list(DETAIL_CLASSES) + ANCHOR_CLASSES + BG_CLASSES
    print(f"Loading Open Images V7 [{args.split}] up to {args.max_samples} samples "
          f"matching {len(classes)} classes...")
    ds = foz.load_zoo_dataset(
        "open-images-v7", split=args.split, label_types=["detections"],
        classes=classes, max_samples=args.max_samples, only_matching=True,
    )
    ds.compute_metadata()

    counts = {"detail": 0, "anchor": 0, "bg": 0}
    detail_seen = 0
    for s in ds:
        w, h = s.metadata.width, s.metadata.height
        if not w or not h:
            continue
        dets = s.ground_truth.detections if s.ground_truth else []
        labels = {d.label for d in dets}
        has_person = any(c in labels for c in ANCHOR_CLASSES)
        big_enough = min(w, h) >= args.min_short

        # detail bbox crops (only the ones that clear the floor)
        for d in dets:
            if d.label not in DETAIL_CLASSES:
                continue
            detail_seen += 1
            l, t, r, b = crop_box(d, w, h, args.pad)
            if min(r - l, b - t) < args.min_short:
                continue
            sub = DETAIL_CLASSES[d.label]
            try:
                im = Image.open(s.filepath).convert("RGB").crop((l, t, r, b))
            except Exception:
                continue
            im.save(out / "detail" / f"oi_{sub}_{Path(s.filepath).stem}_{counts['detail']}.jpg", quality=95)
            counts["detail"] += 1

        # anchor: full person image
        if big_enough and has_person:
            try:
                Image.open(s.filepath).convert("RGB").save(
                    out / "anchor" / f"oi_{Path(s.filepath).stem}.jpg", quality=95)
                counts["anchor"] += 1
            except Exception:
                pass
        # bg: full no-person scene
        elif big_enough and (labels & set(BG_CLASSES)):
            try:
                Image.open(s.filepath).convert("RGB").save(
                    out / "bg" / f"oi_{Path(s.filepath).stem}.jpg", quality=95)
                counts["bg"] += 1
            except Exception:
                pass

    print(f"\nDone. Wrote {counts} -> {out}/")
    print(f"detail bboxes seen: {detail_seen}, kept (>= {args.min_short}px): {counts['detail']}")
    if counts["detail"] < 20:
        print("NOTE: few/no >=1536 detail crops (Open Images skews <=1024). "
              "Get high-res detail close-ups from Unsplash/Pixabay -> data/v8_raw/detail/.")


if __name__ == "__main__":
    main()
