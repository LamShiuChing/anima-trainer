# Anima Realism v5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-tool the prep pipeline so v5 trains Anima at 1024 from base DiT on a smaller, technically-clean dataset with rich enum-locked captions (WD14 tags + local safety + Gemini structured style/NL).

**Architecture:** Curation gates are cheap + local (phash dedup, sub-1024 drop, blur_var sharpness drop — all via existing stage-1 flags). CLIP aesthetic scoring (stage 2) is deleted; Gemini emits the quality tag directly. A new `src/gemini_caption.py` does one structured API call per survivor returning an enum-locked style vocab + NL description, with `BLOCK_NONE` safety settings and a tags-only fallback on refusal. Captions assemble to `realistic photo, <quality>, <capture_style>, <lighting>, <condition>, <safety>, <wd14 tags>[, watermark], <nl>`.

**Tech Stack:** Python 3.10, pytest, PyYAML, diffusion-pipe (Vast), `google-genai` SDK, `dghs-imgutils` (WD14), `transformers` (Falconsai NSFW), `python-dotenv`.

**Spec:** `docs/superpowers/specs/2026-06-02-anima-realism-v5-design.md`

**Conventions (match existing code):**
- Numbered stage modules are loaded in tests via `from conftest import load_stage` → `load_stage("03_caption.py")`.
- Pure logic functions are unit-tested; model/API wrappers are smoke-tested only (network deps).
- Manifest is a CSV of string values; `common.read_manifest/write_manifest/augment_manifest` handle IO.
- Run all tests with: `python -m pytest tests/ -v` (run from project root `D:/anima training`).

---

### Task 1: v5 config + dependencies

**Files:**
- Modify: `config/pipeline.yaml`
- Modify: `requirements.txt`
- Create: `.env.example`
- Test: `tests/test_config_v5.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_v5.py
import common  # conftest puts src/ on sys.path


def test_v5_config_values():
    cfg = common.load_config()  # the real config/pipeline.yaml

    # curation: drop sub-1024 + raise dedup aggressiveness at ingest
    assert cfg["ingest"]["min_size"] == 1024
    assert cfg["ingest"]["drop_small"] is True
    assert cfg["ingest"]["phash_hamming_threshold"] == 8

    # CLIP aesthetic stage removed entirely
    assert "quality" not in cfg

    # Gemini captioner block
    g = cfg["caption"]["gemini"]
    assert g["model"] == "gemini-2.5-flash-lite"
    assert g["safety_block_none"] is True

    # dataset: 1024, blur backstop, no aesthetic bucket filter
    assert cfg["dataset"]["resolutions"] == [1024]
    assert cfg["dataset"]["min_resolution"] == 1024
    assert "min_blur_var" in cfg["dataset"]
    assert "buckets_to_keep" not in cfg["dataset"]

    # finetune: from base (empty init_from), v5 project, CFG dropout 0.10
    f = cfg["finetune"]
    assert f["project_name"] == "anima_realism_ft_v5"
    assert f["init_from"] == ""
    assert f["epochs"] == 20
    assert f["caption_dropout_percent"] == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_v5.py -v`
Expected: FAIL (current config still has `quality:`, `min_size: 512`, no `caption.gemini`, etc.)

- [ ] **Step 3: Edit `config/pipeline.yaml`**

In `ingest:` set:
```yaml
  min_size: 1024                 # v5: drop sub-1024 at ingest (training res; no wasted captions)
  blur_var_threshold: 100.0      # v5: TUNE from the blur_var distribution before the real run
  phash_hamming_threshold: 8     # v5: kill social-media reposts harder
  drop_small: true               # v5: enforce the 1024 floor
  drop_blurry: false             # v5: set true AFTER tuning blur_var_threshold (see RUNBOOK)
  run_ocr_flag: false
```

Delete the entire `quality:` block (CLIP aesthetic — removed in v5).

Replace the `caption:` block body with (keep `nsfw_model`/`nsfw_label_map`/`nsfw_default_tag`/`block_tags`/`wd_*`):
```yaml
caption:                         # v5: WD14 tags + local safety + Gemini structured style/NL
  nsfw_model: Falconsai/nsfw_image_detection
  nsfw_label_map:
    NORMAL: safe
    NSFW: explicit
  nsfw_default_tag: safe
  block_tags: [loli, shota, toddlercon, child, baby, infant, toddler, aged_down]
  wd_model_name: SwinV2_v3
  wd_general_threshold: 0.35
  gemini:
    model: gemini-2.5-flash-lite     # cheapest vision model w/ free tier (2.0-flash-lite deprecated 2026-06-01)
    safety_block_none: true          # BLOCK_NONE on the 4 adjustable categories; child-safety always on (Google core)
    max_output_tokens: 256
    max_retries: 4
    cache_file: data/gemini_cache.json
```

In `dataset:` remove `buckets_to_keep`; set:
```yaml
  min_resolution: 1024
  resolutions: [1024]
  min_blur_var: 0.0              # backstop only (stage 1 does the real blur drop); raise if needed
```

In `finetune:` set:
```yaml
  project_name: anima_realism_ft_v5
  init_from: ""                  # v5: from BASE DiT (no warm-start)
  epochs: 20
  caption_dropout_percent: 0.10
```

- [ ] **Step 4: Edit `requirements.txt`** — append:

```
google-genai>=1.0       # Gemini structured captioning (v5)
python-dotenv>=1.0      # load GEMINI_API_KEY from .env
```

- [ ] **Step 5: Create `.env.example`**

```
# Copy to .env (gitignored) and fill in. Use a THROWAWAY/project key — explicit-content
# captioning with BLOCK_NONE "may be subject to review" per Google.
GEMINI_API_KEY=your_key_here
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_config_v5.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config/pipeline.yaml requirements.txt .env.example tests/test_config_v5.py
git commit -m "config(v5): 1024 from-base, drop CLIP, add Gemini captioner block"
```

---

### Task 2: Gemini caption module — vocab, schema, assembly (pure logic)

**Files:**
- Create: `src/gemini_caption.py`
- Test: `tests/test_gemini_caption.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemini_caption.py
from conftest import load_stage

gc = load_stage("gemini_caption.py")


def test_vocab_sizes():
    assert gc.VOCAB["quality_level"] == ["masterpiece, best quality", "high quality", "low quality"]
    assert "amateur snapshot" in gc.VOCAB["capture_style"]
    assert "direct on-camera flash" in gc.VOCAB["lighting"]
    assert "sharp focus" in gc.VOCAB["condition"]


def test_build_schema_has_enums_and_required():
    s = gc.build_schema()
    assert s["properties"]["capture_style"]["enum"] == gc.VOCAB["capture_style"]
    assert s["properties"]["lighting"]["items"]["enum"] == gc.VOCAB["lighting"]
    assert set(s["required"]) == {"quality_level", "capture_style", "has_watermark", "description"}


def test_coerce_filters_out_of_vocab_and_clamps_arrays():
    raw = {
        "quality_level": "high quality",
        "capture_style": "NOT_A_STYLE",                       # invalid -> ""
        "lighting": ["direct on-camera flash", "golden hour", "low light"],  # 3 -> clamp to 2
        "condition": ["sharp focus", "bogus"],                # filter bogus
        "has_watermark": True,
        "description": "  a person.\n ",
    }
    out = gc.coerce_response(raw)
    assert out["quality_level"] == "high quality"
    assert out["capture_style"] == ""
    assert out["lighting"] == ["direct on-camera flash", "golden hour"]
    assert out["condition"] == ["sharp focus"]
    assert out["has_watermark"] is True
    assert out["nl"] == "a person"                            # cleaned, trailing period stripped


def test_coerce_empty_is_all_blank():
    out = gc.coerce_response({})
    assert out == {"quality_level": "", "capture_style": "", "lighting": [],
                   "condition": [], "has_watermark": False, "nl": ""}


def test_assemble_full_caption():
    parts = {"quality_level": "low quality", "capture_style": "amateur snapshot",
             "lighting": ["direct on-camera flash"], "condition": ["grainy / high ISO"],
             "has_watermark": False, "nl": "a woman in a kitchen holding a mug"}
    out = gc.assemble_caption(parts, safety_tag="safe", wd14_tags="woman, kitchen, mug")
    assert out == ("realistic photo, low quality, amateur snapshot, direct on-camera flash, "
                   "grainy / high ISO, safe, woman, kitchen, mug, a woman in a kitchen holding a mug")


def test_assemble_appends_watermark_token():
    parts = {"quality_level": "high quality", "capture_style": "", "lighting": [],
             "condition": [], "has_watermark": True, "nl": ""}
    out = gc.assemble_caption(parts, safety_tag="explicit", wd14_tags="logo, text")
    assert out == "realistic photo, high quality, explicit, logo, text, watermark"


def test_assemble_fallback_tags_only_when_nl_empty():
    parts = {"quality_level": "", "capture_style": "", "lighting": [],
             "condition": [], "has_watermark": False, "nl": ""}
    out = gc.assemble_caption(parts, safety_tag="explicit", wd14_tags="nude, bed")
    assert out == "realistic photo, explicit, nude, bed"   # graceful: anchor + safety + tags only


def test_build_prompt_includes_tags_and_no_anchor_instruction():
    p = gc.build_prompt("woman, kitchen")
    assert "woman, kitchen" in p
    assert "JSON" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemini_caption.py -v`
Expected: FAIL with "No module named ... gemini_caption" / attribute errors

- [ ] **Step 3: Write `src/gemini_caption.py` (pure-logic portion)**

```python
"""v5 Gemini captioner: enum-locked style vocab + NL description, BLOCK_NONE, tags-only fallback.

Caption format (anchor prepended here, NOT by Gemini):
  realistic photo, <quality>, <capture_style>, <lighting...>, <condition...>, <safety>, <wd14 tags>[, watermark][, <nl>]
"""
import re

ANCHOR = "realistic photo"

VOCAB = {
    "quality_level": ["masterpiece, best quality", "high quality", "low quality"],
    "capture_style": ["amateur snapshot", "casual phone photo", "semi-professional",
                      "professional photograph", "studio portrait"],
    "lighting": ["direct on-camera flash", "natural daylight", "golden hour",
                 "overcast flat light", "indoor artificial light", "low light",
                 "soft window light", "studio lighting"],
    "condition": ["sharp focus", "soft focus", "grainy / high ISO", "motion blur",
                  "compressed / low-res", "overexposed", "underexposed"],
}


def clean_nl(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.rstrip(".").strip()


def build_schema(vocab=VOCAB):
    return {
        "type": "object",
        "properties": {
            "quality_level": {"type": "string", "enum": vocab["quality_level"]},
            "capture_style": {"type": "string", "enum": vocab["capture_style"]},
            "lighting": {"type": "array", "items": {"type": "string", "enum": vocab["lighting"]}},
            "condition": {"type": "array", "items": {"type": "string", "enum": vocab["condition"]}},
            "has_watermark": {"type": "boolean"},
            "description": {"type": "string"},
        },
        "required": ["quality_level", "capture_style", "has_watermark", "description"],
    }


def build_prompt(wd14_tags):
    return (
        "You are labeling a photo to train an image model. Return JSON only, matching the schema. "
        "Write 'description' as one factual sentence covering subject, appearance, clothing or lack "
        "of it, pose, expression, setting, lighting, and camera framing. Do NOT mention image "
        "quality, resolution, or begin with a label. Choose the enum values that best match the "
        "image. These content tags are accurate context: " + (wd14_tags or "")
    )


def coerce_response(raw, vocab=VOCAB):
    """Validate a raw Gemini dict against the vocab. Out-of-vocab -> dropped; arrays clamped to 2."""
    raw = raw or {}
    q = raw.get("quality_level")
    c = raw.get("capture_style")
    return {
        "quality_level": q if q in vocab["quality_level"] else "",
        "capture_style": c if c in vocab["capture_style"] else "",
        "lighting": [x for x in (raw.get("lighting") or []) if x in vocab["lighting"]][:2],
        "condition": [x for x in (raw.get("condition") or []) if x in vocab["condition"]][:2],
        "has_watermark": bool(raw.get("has_watermark")),
        "nl": clean_nl(raw.get("description")),
    }


def assemble_caption(parts, safety_tag, wd14_tags):
    """parts = coerce_response output. Empty pieces are omitted; anchor always leads."""
    p = [ANCHOR]
    if parts["quality_level"]:
        p.append(parts["quality_level"])
    if parts["capture_style"]:
        p.append(parts["capture_style"])
    p += parts["lighting"]
    p += parts["condition"]
    p.append(safety_tag)
    if wd14_tags:
        p.append(wd14_tags)
    if parts["has_watermark"]:
        p.append("watermark")
    if parts["nl"]:
        p.append(parts["nl"])
    return ", ".join(x for x in p if x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gemini_caption.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gemini_caption.py tests/test_gemini_caption.py
git commit -m "feat(v5): gemini_caption vocab/schema/assembly (pure logic, TDD)"
```

---

### Task 3: GeminiCaptioner class — injectable generate, resumable cache, fallback

**Files:**
- Modify: `src/gemini_caption.py` (append class + cache helpers)
- Test: `tests/test_gemini_caption.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
# --- append to tests/test_gemini_caption.py ---

def _cfg(tmp_path):
    return {"caption": {"gemini": {"model": "gemini-2.5-flash-lite", "safety_block_none": True,
                                   "max_output_tokens": 256, "max_retries": 2,
                                   "cache_file": str(tmp_path / "cache.json")}}}


def test_captioner_calls_generate_and_coerces(tmp_path):
    def fake_generate(path, tags):
        return {"quality_level": "high quality", "capture_style": "amateur snapshot",
                "lighting": [], "condition": [], "has_watermark": False, "description": "a dog"}
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=fake_generate)
    out = cap.caption("x.jpg", "dog")
    assert out["quality_level"] == "high quality"
    assert out["nl"] == "a dog"


def test_captioner_cache_hit_skips_generate(tmp_path):
    def boom(path, tags):
        raise AssertionError("generate must not be called on a cache hit")
    seeded = {"x.jpg": {"quality_level": "low quality", "capture_style": "", "lighting": [],
                        "condition": [], "has_watermark": False, "nl": "cached"}}
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=boom, cache=seeded)
    assert cap.caption("x.jpg", "dog")["nl"] == "cached"


def test_captioner_refusal_returns_blank_for_fallback(tmp_path):
    def refuse(path, tags):
        raise RuntimeError("blocked")           # API refusal / error
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=refuse)
    out = cap.caption("x.jpg", "nude, bed")
    assert out == {"quality_level": "", "capture_style": "", "lighting": [],
                   "condition": [], "has_watermark": False, "nl": ""}  # -> assemble = tags only


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    gc.save_cache(path, {"a.jpg": {"nl": "x"}})
    assert gc.load_cache(path) == {"a.jpg": {"nl": "x"}}
    assert gc.load_cache(tmp_path / "missing.json") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemini_caption.py -v`
Expected: FAIL (`GeminiCaptioner`, `load_cache`, `save_cache` undefined)

- [ ] **Step 3: Append to `src/gemini_caption.py`**

```python
import json
from pathlib import Path


def load_cache(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


class GeminiCaptioner:
    """Captions one image -> coerced style dict. `generate(path, tags) -> raw dict` is injectable
    (tests pass a fake). On any generate error/refusal the image falls back to a blank dict, which
    assemble_caption renders as anchor + safety + tags only."""
    def __init__(self, cfg, generate=None, cache=None):
        g = cfg["caption"]["gemini"]
        self.model = g["model"]
        self.max_output_tokens = g.get("max_output_tokens", 256)
        self.max_retries = g.get("max_retries", 4)
        self.block_none = g.get("safety_block_none", True)
        self.schema = build_schema()
        self.cache = cache if cache is not None else {}
        self._generate = generate or self._default_generate

    def caption(self, path, wd14_tags):
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        try:
            raw = self._generate(path, wd14_tags)
        except Exception:
            raw = {}                       # refusal/rate-limit-exhausted/error -> tags-only fallback
        result = coerce_response(raw)
        self.cache[key] = result
        return result

    def _default_generate(self, path, wd14_tags):
        """Real Gemini call. Not unit-tested (network). Returns a parsed JSON dict or {}."""
        from google import genai
        from google.genai import types
        client = genai.Client()            # reads GEMINI_API_KEY
        none = types.HarmBlockThreshold.BLOCK_NONE
        safety = [types.SafetySetting(category=c, threshold=none) for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS,
        )] if self.block_none else None
        img = types.Part.from_bytes(data=Path(path).read_bytes(),
                                    mime_type="image/jpeg")
        cfg = types.GenerateContentConfig(
            safety_settings=safety,
            response_mime_type="application/json",
            response_schema=self.schema,
            max_output_tokens=self.max_output_tokens,
        )
        last = None
        for _ in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=[img, build_prompt(wd14_tags)], config=cfg)
                return json.loads(resp.text) if resp.text else {}
            except Exception as e:           # rate-limit/5xx -> retry; final raises -> caption() fallback
                last = e
        raise last
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gemini_caption.py -v`
Expected: PASS (12 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/gemini_caption.py tests/test_gemini_caption.py
git commit -m "feat(v5): GeminiCaptioner with injectable generate, resumable cache, fallback"
```

---

### Task 4: Stage 4 curate — blur_var backstop, drop bucket filter

**Files:**
- Modify: `src/04_build_dataset.py:15-29` (the `curate` function) and `main()` call
- Test: `tests/test_04_build_dataset.py` (replace the two `curate` tests)

- [ ] **Step 1: Replace the two curate tests**

Delete `test_curate_keeps_all_three_buckets` and `test_curate_min_resolution_and_quality`; add:

```python
def test_curate_drops_dropped_rows_only_by_default():
    rows = [
        {"path": "a.jpg", "dropped": "False"},
        {"path": "b.jpg", "dropped": "True"},
    ]
    kept = stage.curate(rows)
    assert {r["path"] for r in kept} == {"a.jpg"}


def test_curate_min_resolution_and_blur():
    rows = [
        {"path": "ok.jpg",    "dropped": "False", "width": "1024", "height": "1300", "blur_var": "250.0"},
        {"path": "small.jpg", "dropped": "False", "width": "800",  "height": "1300", "blur_var": "250.0"},  # <1024
        {"path": "soft.jpg",  "dropped": "False", "width": "1200", "height": "1200", "blur_var": "40.0"},   # <min_blur
        {"path": "nosize.jpg","dropped": "False", "blur_var": "250.0"},                                     # missing size
    ]
    kept = stage.curate(rows, min_resolution=1024, min_blur_var=100.0)
    assert {r["path"] for r in kept} == {"ok.jpg"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: FAIL (`curate()` still requires `buckets_to_keep`)

- [ ] **Step 3: Rewrite `curate` in `src/04_build_dataset.py`**

```python
def curate(rows, min_resolution=0, min_blur_var=0.0):
    """Keep non-dropped rows. v5: no aesthetic-bucket filter (all buckets kept, tagged).
    Optional technical gates read sizes/blur recorded by stage 1 (no image re-read)."""
    out = []
    for r in rows:
        if r.get("dropped") != "False":
            continue
        if min_resolution:
            try:
                if min(int(r["width"]), int(r["height"])) < min_resolution:
                    continue
            except (KeyError, ValueError):
                continue  # no size on record -> exclude from a resolution-filtered run
        if min_blur_var:
            try:
                if float(r["blur_var"]) < min_blur_var:
                    continue
            except (KeyError, ValueError):
                continue
        out.append(r)
    return out
```

Update `main()` — replace the `curate(...)` call and its log line:

```python
    kept = curate(rows, ds.get("min_resolution", 0), ds.get("min_blur_var", 0.0))
    LOG.info("Stage 4: curated %d images (min_resolution=%s, min_blur_var=%s)",
             len(kept), ds.get("min_resolution", 0), ds.get("min_blur_var", 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/04_build_dataset.py tests/test_04_build_dataset.py
git commit -m "feat(v5): stage4 curate by technical gates (res+blur), drop aesthetic bucket filter"
```

---

### Task 5: Stage 3 rewrite — WD14 tags + local safety + Gemini, underage block on tags

**Files:**
- Modify: `src/03_caption.py` (rewrite `main()`; add `underage_hit`; keep `map_safety` + `NSFWTagger` + `WDTaggerCaptioner`; remove `quality_tag_for`, `clean_nl`, `assemble_caption`, `JoyCaptioner`, `ToriiGateCaptioner`, `build_captioner`)
- Test: `tests/test_03_caption.py` (keep `map_safety` tests; replace the rest)

- [ ] **Step 1: Replace tests in `tests/test_03_caption.py`**

```python
from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"


def test_underage_hit_fires_on_wd14_comma_tags():
    block = {"child", "baby", "toddler"}
    assert stage.underage_hit("1girl, child, indoor", block) == {"child"}
    assert stage.underage_hit("woman, kitchen, mug", block) == set()


def test_underage_hit_needs_comma_tokens_not_nl_substring():
    # regression: this is WHY WD14 tags (not Gemini NL) must drive the block.
    block = {"child"}
    assert stage.underage_hit("a child sitting on a chair", block) == set()  # NL won't match -> would leak
    assert stage.underage_hit("child, chair", block) == {"child"}            # comma tags match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: FAIL (`underage_hit` undefined)

- [ ] **Step 3: Rewrite `src/03_caption.py`**

Replace the whole file with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/03_caption.py tests/test_03_caption.py
git commit -m "feat(v5): stage3 = WD14 tags + safety + Gemini; underage block on comma tags; skip culled files"
```

---

### Task 6: Delete CLIP stage + dead helper, full-suite green

**Files:**
- Delete: `src/02_quality_score.py`, `tests/test_02_quality_score.py`
- Modify: `src/common.py` (remove `ensure_aesthetic_weights`)

- [ ] **Step 1: Delete the CLIP stage + its test**

```bash
git rm src/02_quality_score.py tests/test_02_quality_score.py
```

- [ ] **Step 2: Remove `ensure_aesthetic_weights` from `src/common.py`**

Delete the entire `ensure_aesthetic_weights` function (the last function in the file, lines ~77-92) and its trailing blank line. Leave the rest untouched.

- [ ] **Step 3: Verify nothing else references the removed symbols**

Run: `git grep -n "ensure_aesthetic_weights\|02_quality_score\|AestheticScorer\|score_to_bucket"`
Expected: no matches (empty output). If any appear, fix them before continuing.

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS, all tests (config_v5, gemini_caption ×12, 01, 03, 04, 05, common). No `test_02`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(v5): delete CLIP aesthetic stage + ensure_aesthetic_weights (Gemini emits quality)"
```

---

### Task 7: RUNBOOK — v5 build order (blur tuning, Gemini caption, train)

**Files:**
- Modify: `RUNBOOK.md` (add a v5 section)

- [ ] **Step 1: Append a `## v5 run` section to `RUNBOOK.md`**

Document the exact build order (no code to test — this is operator docs):

```markdown
## v5 run (1024, from base, Gemini captions)

0. FREE pre-check: generate from v3-epoch4 at 1024 (matched res) before any GPU spend.
1. `.env`: copy `.env.example` -> `.env`, set GEMINI_API_KEY (throwaway/project key).
2. Stage 1 ingest (drop_small=true, min_size=1024, drop_blurry=false): records blur_var, drops <1024 + dups.
3. Tune sharpness: inspect the blur_var column distribution of kept rows; pick a threshold at the soft
   tail. Set ingest.blur_var_threshold + ingest.drop_blurry=true, re-run stage 1.
4. Spot-review: delete obviously-bad survivors from data/clean (stage 3 skips missing files).
5. Stage 3 caption: WD14 + safety + Gemini (resumable via data/gemini_cache.json). Sanity-check the
   first ~50 captions; adjust the Gemini prompt rubric if quality buckets skew.
6. Stage 4 build (min_resolution=1024, optional min_blur_var backstop) + Stage 5 anima.toml
   (init_from="" => base DiT, epochs=20). Confirm post-gate image count is sufficient.
7. Upload data/dataset + tomls to Vast; train; preview each epoch in ComfyUI; stop at best.
```

- [ ] **Step 2: Commit**

```bash
git add RUNBOOK.md
git commit -m "docs(v5): RUNBOOK build order — blur tuning, Gemini caption, from-base train"
```

---

## Self-Review

**Spec coverage:**
- 1024 from-base → Task 1 (config `init_from:""`, `resolutions:[1024]`); stage 5 already honors empty `init_from`. ✓
- Curate by technical defects only (dedup→8, drop <1024, blur gate), keep all buckets → Task 1 (ingest flags) + Task 4 (curate). ✓
- Delete CLIP stage → Task 6. ✓
- WD14 tags + local safety + Gemini enum vocab + NL, BLOCK_NONE + fallback → Tasks 2,3,5. ✓
- Watermark flag → tag (not drop) → Task 2 (`assemble` appends `watermark`), schema `has_watermark`. ✓
- Anchor `realistic photo,` leads every caption → Task 2 (`ANCHOR`). ✓
- Underage block survives the captioner switch → Task 5 (`underage_hit` on WD14 comma tags + regression test). ✓
- Resumable Gemini cache, rate-limit retry → Task 3. ✓
- caption_dropout 0.10, project v5, epochs 20 → Task 1. ✓
- `.env` for key (gitignored) → Task 1 (`.env.example`); `.gitignore` already updated. ✓
- Build-time blur tuning + first-50 calibration → Task 7 RUNBOOK. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; `blur_var_threshold: 100.0` is explicitly an operator-tuned value with the tuning procedure in Task 7 (not a code placeholder). ✓

**Type consistency:** `coerce_response` returns keys `{quality_level, capture_style, lighting, condition, has_watermark, nl}`; `assemble_caption` and `GeminiCaptioner.caption` consume exactly those; stage 3 reads `parts["quality_level"]`/`parts["nl"]`. `curate(rows, min_resolution, min_blur_var)` signature matches its Task 4 call and tests. `underage_hit(wd14_tags, block_terms)` matches tests. ✓
