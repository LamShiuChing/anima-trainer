"""Stage 3b (v10): structured Gemini caption — photographic enums + real-photo tags + NL paragraph.

The SOLE caption-text source for v10 (RAM++ dropped: its flat keyword bag was noisy and mislabeled).
ONE Gemini call per image returns structured JSON with three layers:
  1. enum photographic/style properties (the v7 rubric, real-photo subset),
  2. a real-photo TAG list (concise comma keywords for inference triggers),
  3. a detailed NL paragraph (subject / face / body / clothing / pose / setting).
Assembled with the base quality tokens + the rating already in the manifest:

  masterpiece, best quality, score_7, <rating>, <enum props...>, <tags...>[, watermark], <NL paragraph>

Reuses src/v10_caption.py's underage gate + rating (manifest already has rating_tag + dropped flags).
Model = gemini-3-flash-preview (BLOCK_NONE -> captions NSFW; thinking disabled, else thinking tokens
truncate the JSON). Thread pool + resumable cache (path -> raw JSON). Idempotent: rebuilds the caption
from cache every run. Prompt + structure adapted from the user's lenstag-ai app + the v7 enum rubric.

Run:  python src/v10_caption_gemini.py    (after src/v10_caption.py set rating_tag + underage drops)
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
import re
import threading
import time
from pathlib import Path

import common

LOG = common.setup_logging()

# v7 enum rubric, real-photo subset. `quality` is handled by the fixed prefix; `rating` stays here
# (4-level booru ladder, richer than Falconsai's binary -> falls back to the manifest rating_tag).
VOCAB = {
    "shot_type": ["extreme close-up", "close-up", "portrait", "upper body", "cowboy shot",
                  "full body", "wide shot"],
    "view": ["front view", "three-quarter view", "profile view", "back view",
             "looking over shoulder", "looking at viewer", "looking away"],
    "camera_angle": ["eye level", "from above", "from below", "overhead", "dutch angle"],
    "capture_style": ["amateur snapshot", "casual phone photo", "social media selfie",
                      "candid photo", "semi-professional", "professional photograph",
                      "editorial photography", "studio portrait"],
    "lighting": ["direct flash", "natural daylight", "golden hour", "blue hour",
                 "overcast flat light", "indoor artificial light", "low light",
                 "soft window light", "studio lighting", "backlit", "rim light",
                 "neon lighting", "harsh sunlight", "ring light", "candlelight"],
    "condition": ["sharp focus", "soft focus", "grainy / high ISO", "motion blur",
                  "overexposed", "underexposed", "lens flare", "chromatic aberration",
                  "vignette", "red-eye"],
    "color_grade": ["natural color", "warm tones", "cool tones", "muted", "vibrant",
                    "high contrast", "film grain", "film look", "black and white",
                    "sepia", "faded", "teal and orange"],
    "camera_lens": ["phone camera", "compact camera", "DSLR", "85mm bokeh", "50mm",
                    "35mm", "wide-angle", "fisheye", "macro", "film camera"],
    "depth_of_field": ["shallow depth of field", "deep focus"],
    "expression": ["neutral expression", "smile", "laughing", "serious", "seductive",
                   "surprised", "crying", "pout", "open mouth"],
    "body_type": ["slim", "average build", "athletic", "curvy", "plus-size", "muscular", "petite"],
    "breast_size": ["flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts"],
    "ethnicity": ["east asian", "southeast asian", "south asian", "white", "black",
                  "hispanic", "middle eastern", "mixed"],
    "skin_tone": ["fair skin", "light skin", "olive skin", "tan skin", "brown skin", "dark skin"],
    "setting_type": ["bedroom", "living room", "kitchen", "bathroom", "studio", "office",
                     "city street", "nature", "beach", "pool", "cafe", "restaurant", "bar",
                     "gym", "car", "party"],
    "rating": ["safe", "suggestive", "explicit"],     # simple, prefix-free (no "rating:" / no booru ladder)
}
SINGLE_SLOTS = ("shot_type", "view", "camera_angle", "capture_style", "color_grade", "camera_lens",
                "depth_of_field", "expression", "body_type", "breast_size", "ethnicity", "skin_tone",
                "setting_type", "rating")
ARRAY_SLOTS = ("lighting", "condition")
ARRAY_MAX = 2
TAGS_MAX = 30
# caption assembly order for the enum props (quality prefix + rating lead; tags + NL trail)
ENUM_ORDER = ("shot_type", "view", "camera_angle", "capture_style", "lighting", "condition",
              "color_grade", "camera_lens", "depth_of_field", "expression", "body_type",
              "breast_size", "ethnicity", "skin_tone", "setting_type")
# rating REQUIRED -> Gemini always emits it (no Falconsai fallback needed in the Gemini-only design)
REQUIRED = ["shot_type", "capture_style", "rating", "tags", "description"]

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


# ---- pure logic (unit-tested) ----

def mime_for(path):
    return MIME.get(Path(path).suffix.lower(), "image/jpeg")


def clean_nl(text):
    return re.sub(r"\s+", " ", text or "").strip().rstrip(".").strip()


def clean_tag(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower()).strip(", ")


def build_schema(vocab=VOCAB):
    props = {}
    for k in SINGLE_SLOTS:
        props[k] = {"type": "string", "enum": vocab[k]}
    for k in ARRAY_SLOTS:
        props[k] = {"type": "array", "items": {"type": "string", "enum": vocab[k]}}
    props["tags"] = {"type": "array", "items": {"type": "string"}}
    props["has_watermark"] = {"type": "boolean"}
    props["description"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(REQUIRED)}


def build_prompt():
    return (
        "You are labeling a REAL photograph to train a photorealistic image model. Return JSON only, "
        "matching the schema. For each enum field choose the single value that best fits (omit an "
        "optional field if it does not clearly apply). 'tags': 8-30 concise lowercase real-world "
        "keywords (subjects, clothing, objects, accessories, setting, attributes) — NOT a sentence. "
        "'description': a detailed factual paragraph (50-100 words) in this order: the subject (how "
        "many people, that they are adults, apparent gender); the face (shape, eyes, lips, nose, "
        "brows, skin, hair color/length/style, makeup); the visible body (build, torso, chest, "
        "midriff, arms, legs as visible); clothing with materials/colors and accessories (bags, "
        "phones, jewelry, glasses, shoes); pose; then setting and notable background objects/detail. "
        "Do NOT mention image quality, resolution, camera, or lighting in the description (those are "
        "separate enum fields). Do NOT begin with a label or preamble."
    )


def coerce(raw, vocab=VOCAB):
    """Validate a raw Gemini dict: enums dropped if out-of-vocab, arrays clamped, tags cleaned/deduped."""
    raw = raw or {}
    out = {}
    for k in SINGLE_SLOTS:
        v = raw.get(k)
        out[k] = v if v in vocab[k] else ""
    for k in ARRAY_SLOTS:
        out[k] = [x for x in (raw.get(k) or []) if x in vocab[k]][:ARRAY_MAX]
    seen, tags = set(), []
    for t in (raw.get("tags") or []):
        t = clean_tag(t)
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    out["tags"] = tags[:TAGS_MAX]
    out["has_watermark"] = bool(raw.get("has_watermark"))
    out["nl"] = clean_nl(raw.get("description"))
    return out


def assemble(parts, quality_prefix, fallback_rating="safe"):
    """Full caption: quality prefix, rating, enum props, tags, [watermark], NL paragraph."""
    p = [quality_prefix.strip(), parts.get("rating") or fallback_rating]
    for k in ENUM_ORDER:
        if k in ARRAY_SLOTS:
            p += parts.get(k, [])
        elif parts.get(k):
            p.append(parts[k])
    p += parts.get("tags", [])
    if parts.get("has_watermark"):
        p.append("watermark")
    if parts.get("nl"):
        p.append(parts["nl"])
    return ", ".join(x for x in p if x)


# ---- Gemini wrapper (not unit-tested: network) ----

class GeminiCaptioner:
    def __init__(self, cfg):
        c = cfg["caption"]["nl"]
        self.model = c.get("model", "gemini-3-flash-preview")
        self.concurrency = int(c.get("concurrency", 8))
        self.max_output_tokens = int(c.get("max_output_tokens", 800))
        self.block_none = bool(c.get("block_none", True))
        self.max_retries = 4
        self.schema = build_schema()
        self.prompt = build_prompt()
        self._client = None
        self._lock = threading.Lock()

    def _client_lazy(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from google import genai
                    self._client = genai.Client()      # reads GEMINI_API_KEY
        return self._client

    def caption_raw(self, path):
        """One structured Gemini call -> raw dict (coerce() validates). {} on empty/refusal."""
        from google.genai import types
        client = self._client_lazy()
        none = types.HarmBlockThreshold.BLOCK_NONE
        safety = [types.SafetySetting(category=cat, threshold=none) for cat in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )] if self.block_none else None
        img = types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime_for(path))
        kwargs = dict(
            safety_settings=safety,
            response_mime_type="application/json",
            response_schema=self.schema,
            temperature=0.7,
            max_output_tokens=self.max_output_tokens,
        )
        if hasattr(types, "ThinkingConfig"):            # Gemini 3 thinking eats the budget -> disable
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        cfg = types.GenerateContentConfig(**kwargs)
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=[img, self.prompt], config=cfg)
                return json.loads(resp.text) if resp.text else {}
            except Exception as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        raise last


# ---- cache ----

def load_cache(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path, cache):
    Path(path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    cfg = common.load_config()
    nl_cfg = cfg["caption"]["nl"]
    quality_prefix = nl_cfg["quality_prefix"]
    cache_file = nl_cfg["cache_file"]
    manifest = cfg["paths"]["manifest"]
    rows = common.read_manifest(manifest)
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3b (v10): structured Gemini caption for %d images (model=%s concurrency=%s)",
             len(kept), nl_cfg.get("model"), nl_cfg.get("concurrency"))

    cache = load_cache(cache_file)
    gem = GeminiCaptioner(cfg)

    todo = [r for r in kept if r["path"] not in cache and Path(r["path"]).exists()]
    if todo:                            # pre-flight: fail loud on bad key/model/schema before the pool
        try:
            sample = gem.caption_raw(todo[0]["path"])
        except Exception as e:
            raise RuntimeError(f"Gemini pre-flight failed: {e!r}. Fix key/model before the full run.") from e
        cache[todo[0]["path"]] = sample
        LOG.info("pre-flight OK. sample caption:\n  %s",
                 assemble(coerce(sample), quality_prefix)[:300])

    ok = empty = 0
    bar = tqdm(total=len(kept), desc="gemini-caption", unit="img", dynamic_ncols=True)

    def work(r):
        path = r["path"]
        if path in cache:
            return path, cache[path]
        if not Path(path).exists():
            return path, {}
        try:
            return path, gem.caption_raw(path)
        except Exception as e:
            tqdm.write(f"  caption fail {Path(path).name}: {e}")
            return path, None           # None -> don't cache; retried next run

    try:
        with ThreadPoolExecutor(max_workers=max(1, gem.concurrency)) as ex:
            for i, (path, raw) in enumerate(ex.map(work, kept), 1):
                bar.update(1)
                if raw is None:
                    continue
                cache[path] = raw
                if raw:
                    ok += 1
                else:
                    empty += 1
                if i % 100 == 0:
                    save_cache(cache_file, cache)
    finally:
        bar.close()
        save_cache(cache_file, cache)

    for r in kept:                      # rebuild captions from cached JSON (idempotent)
        parts = coerce(cache.get(r["path"], {}))
        r["caption"] = assemble(parts, quality_prefix)
    common.write_manifest(manifest, rows)
    LOG.info("Stage 3b done. ok=%d empty/refused=%d (of %d). captions rebuilt -> %s. "
             "Next: python src/04_build_dataset.py", ok, empty, len(kept), manifest)


if __name__ == "__main__":
    main()
