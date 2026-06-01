"""Stage 3: NSFW safety tag + quality tag -> tag-only caption "<quality>, <safety>". Augments manifest.

v2: tag-only (no JoyCaption NL, no trigger word). Domain-shift finetune bakes realism in broadly;
minimal captions are intentional. Quality words steer at inference; safety tag separates SFW/NSFW.
"""
import torch
from PIL import Image

import common

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def quality_tag_for(bucket, quality_tag_map):
    return quality_tag_map[bucket]


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):  # longest-first: "UNSAFE" before "SAFE"
        if key.upper() in up:
            return label_map[key]
    return default_tag


def assemble_caption(quality_tag, safety_tag):
    return f"{quality_tag}, {safety_tag}"


# ---- model wrapper (smoke-tested) ----

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
        model_label = self.model.config.id2label[idx]
        return map_safety(model_label, self.label_map, self.default)


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3: tagging %d images (tag-only captions)", len(kept))

    nsfw = NSFWTagger(cfg)
    updates = {}
    for r in kept:
        bucket = r.get("bucket")
        if not bucket:
            raise RuntimeError(f"No bucket for {r['path']} - run stage 2 (02_quality_score) first.")
        qtag = quality_tag_for(bucket, cap_cfg["quality_tag_map"])
        stag = nsfw.tag(r["path"])
        caption = assemble_caption(qtag, stag)
        updates[r["path"]] = {"safety_tag": stag, "quality_tag": qtag, "caption": caption}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done.")


if __name__ == "__main__":
    main()
