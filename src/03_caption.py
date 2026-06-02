"""Stage 3 (v5): WD14 tags + local safety tag + Gemini structured style/NL -> assembled caption.

Per survivor: WD14 booru tags (also the adults-only underage hard-block, on comma tokens),
local NSFW safety tag (Falconsai), and one Gemini call for the enum-locked style vocab + NL.
Caption assembly + Gemini live in gemini_caption.py. Rows whose file no longer exists are
skipped (enables manual spot-review by deleting files from data/clean before this stage).
"""
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import common
import gemini_caption as gcap

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):
        if key.upper() in up:
            return label_map[key]
    return default_tag


def underage_hit(wd14_tags, block_terms):
    """Intersection of comma-tokenized WD14 tags with the underage block set (adults-only boundary)."""
    tagset = {t.strip().lower() for t in wd14_tags.split(",")}
    return set(block_terms) & tagset


# ---- model wrappers (smoke-tested) ----

class NSFWTagger:
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


class WDTaggerCaptioner:
    """WD14 v3 booru tagger (uncensored, ~single forward pass). Comma-separated content tags."""
    def __init__(self, cfg):
        from imgutils.tagging import get_wd14_tags
        self._tag = get_wd14_tags
        c = cfg["caption"]
        self.model_name = c.get("wd_model_name", "SwinV2_v3")
        self.threshold = c.get("wd_general_threshold", 0.35)

    def caption(self, path):
        _rating, general, _chars = self._tag(
            path, model_name=self.model_name, general_threshold=self.threshold,
            no_underline=True, drop_overlap=True)
        return ", ".join(general.keys())


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3 (v5): captioning up to %d images (WD14 + safety + Gemini)", len(kept))

    block_terms = {t.strip().lower() for t in cap_cfg.get("block_tags", [])}
    nsfw = NSFWTagger(cfg)
    wd = WDTaggerCaptioner(cfg)
    cache_file = cap_cfg["gemini"]["cache_file"]
    gem = gcap.GeminiCaptioner(cfg, cache=gcap.load_cache(cache_file))

    updates, blocked, skipped = {}, 0, 0
    try:
        for idx, r in enumerate(tqdm(kept, desc="caption", unit="img", dynamic_ncols=True)):
            if not Path(r["path"]).exists():       # spot-review: file culled -> skip
                skipped += 1
                continue
            tags = wd.caption(r["path"])
            hit = underage_hit(tags, block_terms)
            if hit:
                updates[r["path"]] = {"dropped": "True", "drop_reason": "underage_flag:" + ",".join(sorted(hit))}
                blocked += 1
                continue
            safety = nsfw.tag(r["path"])
            parts = gem.caption(r["path"], tags)
            caption = gcap.assemble_caption(parts, safety, tags)
            updates[r["path"]] = {"safety_tag": safety, "quality_tag": parts["quality_level"], "caption": caption}
            if idx < 3:
                tqdm.write(f"[sample {idx}] {caption}")
    finally:
        gcap.save_cache(cache_file, gem.cache)     # persist even on interrupt (resumable)

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done. blocked(underage)=%d skipped(missing-file)=%d", blocked, skipped)


if __name__ == "__main__":
    main()
