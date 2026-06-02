"""Stage 3 (v5): WD14 tags + local safety tag + Gemini structured style/NL -> assembled caption.

Per survivor: WD14 booru tags (also the adults-only underage hard-block, on comma tokens),
local NSFW safety tag (Falconsai), and one Gemini call for the enum-locked style vocab + NL.
Caption assembly + Gemini live in gemini_caption.py. Rows whose file no longer exists are
skipped (enables manual spot-review by deleting files from data/clean before this stage).
"""
import os
os.environ.setdefault("USE_TF", "0")    # torch-only: stop transformers importing TensorFlow (breaks under numpy 2.x)
os.environ.setdefault("USE_FLAX", "0")

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
    try:
        from dotenv import load_dotenv
        load_dotenv()                  # load GEMINI_API_KEY from .env into the environment
    except ImportError:
        pass                           # ok if not using a .env file (env var set another way)
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

    # Pass 1 (serial, local GPU): WD14 tags + underage hard-block + NSFW safety tag.
    worklist = []   # (row, tags, safety) for survivors
    for r in tqdm(kept, desc="tag+safety", unit="img", dynamic_ncols=True):
        if not Path(r["path"]).exists():           # spot-review: file culled -> skip
            skipped += 1
            continue
        tags = wd.caption(r["path"])
        hit = underage_hit(tags, block_terms)
        if hit:
            updates[r["path"]] = {"dropped": "True", "drop_reason": "underage_flag:" + ",".join(sorted(hit))}
            blocked += 1
            continue
        worklist.append((r, tags, nsfw.tag(r["path"])))

    # Pre-flight: one real Gemini call so a systematic error (auth/SDK/config) aborts LOUDLY here,
    # before the concurrent pass silently falls back to tags-only for every image.
    if worklist:
        r0, tags0, safety0 = worklist[0]
        try:
            raw0 = gem._default_generate(r0["path"], tags0)
        except Exception as e:
            raise RuntimeError(f"Gemini pre-flight call failed: {e!r}. Fix before the full run "
                               f"(no captions were written).") from e
        LOG.info("Gemini pre-flight OK. Sample caption:\n  %s",
                 gcap.assemble_caption(gcap.coerce_response(raw0), safety0, tags0)[:250])

    # Pass 2 (concurrent): one Gemini call per survivor, thread-pooled (size = caption.gemini.concurrency).
    tags_by_path = {r["path"]: tags for (r, tags, _s) in worklist}
    safety_by_path = {r["path"]: safety for (r, _t, safety) in worklist}
    gemini_ok = [0]
    bar = tqdm(total=len(worklist), desc="gemini", unit="img", dynamic_ncols=True)

    def on_result(path, parts):
        bar.update(1)
        if parts["quality_level"]:
            gemini_ok[0] += 1
        caption = gcap.assemble_caption(parts, safety_by_path[path], tags_by_path[path])
        updates[path] = {"safety_tag": safety_by_path[path], "quality_tag": parts["quality_level"], "caption": caption}

    try:
        gem.caption_many([(r["path"], tags) for (r, tags, _s) in worklist], on_result=on_result)
    finally:
        bar.close()
        gcap.save_cache(cache_file, gem.cache)     # persist even on interrupt (resumable)

    # Outside the try: cache (saved in finally) prevents re-billing on resume; manifest is updated only
    # on a clean finish so rows are never left partially written.
    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done. captioned=%d gemini_ok=%d blocked(underage)=%d skipped(missing)=%d",
             len(worklist), gemini_ok[0], blocked, skipped)


if __name__ == "__main__":
    main()
