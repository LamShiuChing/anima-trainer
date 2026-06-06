"""Stage 3 (v10): RAM++ real-photo tags + Falconsai rating + base quality-token prefix -> caption.

No anime/booru tagger for captions, no Gemini, no realism anchor. WD14 runs ONLY as the
adults-only underage drop gate (its tags are discarded). Caption shape:
  '<quality_prefix>, <rating>, <ram++ tags>'
e.g. 'masterpiece, best quality, score_7, rating:general, woman, kitchen, window'
Writes `caption` into data/v10_manifest.csv; underage rows are marked dropped=True.
"""
import os
os.environ.setdefault("USE_TF", "0")     # torch-only (transformers + numpy 2.x)
os.environ.setdefault("USE_FLAX", "0")

from pathlib import Path


# ---- pure logic (unit-tested) ----

def assemble_caption(quality_prefix, rating, ram_tags):
    """quality_prefix + rating + de-duplicated, stripped RAM++ tags, comma-joined. Empties dropped."""
    seen, tags = set(), []
    for t in ram_tags:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)
    parts = [quality_prefix.strip(), rating.strip(), *tags]
    return ", ".join(p for p in parts if p)


def underage_hit(tags_csv, block_terms):
    """Intersection of comma-tokenized tags with the underage block set (adults-only boundary)."""
    tagset = {t.strip().lower() for t in tags_csv.split(",")}
    return set(block_terms) & tagset


import torch
from PIL import Image
from tqdm import tqdm

import common

LOG = common.setup_logging()


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):
        if key.upper() in up:
            return label_map[key]
    return default_tag


class NSFWTagger:
    """Falconsai rating -> booru rating tag (rating:general / rating:explicit)."""
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        name = cfg["caption"]["nsfw_model"]
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
        self.label_map = cfg["caption"]["nsfw_label_map"]
        self.default = cfg["caption"]["nsfw_default_tag"]

    @torch.no_grad()
    def tag(self, path):
        image = Image.open(path).convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        return map_safety(self.model.config.id2label[idx], self.label_map, self.default)


class UnderageGate:
    """WD14 booru tagger used ONLY for the adults-only underage hard-block (tags discarded)."""
    def __init__(self, cfg):
        from imgutils.tagging import get_wd14_tags
        self._tag = get_wd14_tags
        c = cfg["caption"]
        self.model_name = c.get("wd_model_name", "EVA02_Large")
        self.threshold = c.get("wd_general_threshold", 0.25)

    def tags_csv(self, path):
        _rating, general, _chars = self._tag(
            path, model_name=self.model_name, general_threshold=self.threshold,
            no_underline=True, drop_overlap=True)
        return ", ".join(general.keys())


class RAMTagger:
    """RAM++ (Recognize Anything Plus) real-photo tagger. Pipe-delimited english tags -> list."""
    def __init__(self, cfg, device="cuda"):
        from ram.models import ram_plus
        from ram import get_transform, inference_ram
        c = cfg["caption"]["ram"]
        self.size = c.get("image_size", 384)
        self.transform = get_transform(image_size=self.size)
        self.model = ram_plus(pretrained=c["checkpoint"], image_size=self.size, vit="swin_l").eval().to(device)
        self._infer = inference_ram
        self.device = device

    @torch.no_grad()
    def tags(self, path):
        img = self.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        res = self._infer(img, self.model)        # res[0] = "tag1 | tag2 | ..."
        return [t.strip() for t in res[0].split("|") if t.strip()]


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    quality_prefix = cap_cfg["ram"]["quality_prefix"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3 (v10): captioning up to %d images (RAM++ tags + Falconsai rating + underage gate)", len(kept))

    block_terms = {t.strip().lower() for t in cap_cfg.get("block_tags", [])}
    nsfw = NSFWTagger(cfg)
    gate = UnderageGate(cfg)
    ram = RAMTagger(cfg)

    updates, blocked, captioned, skipped = {}, 0, 0, 0
    for r in tqdm(kept, desc="caption", unit="img", dynamic_ncols=True):
        if not Path(r["path"]).exists():        # spot-review: file culled from data/v10_clean -> skip
            skipped += 1
            continue
        hit = underage_hit(gate.tags_csv(r["path"]), block_terms)
        if hit:
            updates[r["path"]] = {"dropped": "True", "drop_reason": "underage_flag:" + ",".join(sorted(hit))}
            blocked += 1
            continue
        rating = nsfw.tag(r["path"])
        tags = ram.tags(r["path"])
        caption = assemble_caption(quality_prefix, rating, tags)
        updates[r["path"]] = {"rating_tag": rating, "caption": caption}
        captioned += 1

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 (v10) done. captioned=%d blocked(underage)=%d skipped(missing)=%d",
             captioned, blocked, skipped)


if __name__ == "__main__":
    main()
