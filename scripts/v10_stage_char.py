"""Stage the trigger-word character set (data/r) into data/v10_char/ for v10 training.

 - swaps the trigger word 'reiko' -> 'rrr' (whole word) in the SOURCE .txt (data/r) and the staged copy
 - AR-crops each image to 0.66-1.5 (Anima DiT pos-emb cap; tall/wide else crash at 1536)
 - saves png + .txt pairs into data/v10_char/

Source = data/r (png + txt pairs). Run once:  python scripts/v10_stage_char.py
"""
import importlib.util
import pathlib
import re

from PIL import Image

SRC = pathlib.Path("data/r")
DST = pathlib.Path("data/v10_char")
OLD, NEW = "reiko", "rrr"

_spec = importlib.util.spec_from_file_location("v10_curate", "src/v10_curate.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    pairs = cropped = missing = 0
    for img in sorted(SRC.glob("*.png")):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            print(f"  no caption for {img.name}"); missing += 1; continue
        new_cap = re.sub(rf"\b{OLD}\b", NEW, txt.read_text(encoding="utf-8"))
        txt.write_text(new_cap, encoding="utf-8")                 # swap trigger in the source too
        im = Image.open(img).convert("RGB")
        w, h = im.size
        box = v10.ar_crop_box(w, h, 0.67, 1.49)   # inset from 0.66/1.5 so rounding never dips out of range
        if box != (0, 0, w, h):
            im = im.crop(box); cropped += 1
        im.save(DST / (img.stem + ".png"))
        (DST / (img.stem + ".txt")).write_text(new_cap, encoding="utf-8")
        pairs += 1
    print(f"staged {pairs} pairs -> {DST} (AR-cropped {cropped}, missing-caption {missing})")

    bad = [p.name for p in DST.glob("*.png")
           if not (0.66 <= (lambda s: s[0] / s[1])(Image.open(p).size) <= 1.5)]
    residual = sum(len(re.findall(rf"\b{OLD}\b", p.read_text(encoding="utf-8")))
                   for p in DST.glob("*.txt"))
    print(f"out-of-AR after crop: {len(bad)}   |   '{OLD}' residual in staged txt: {residual}")


if __name__ == "__main__":
    main()
