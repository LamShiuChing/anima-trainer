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
