"""v7 Gemini captioner: expanded enum-locked controlled vocab + rich NL, BLOCK_NONE, tags-only fallback.

No 'realistic photo' anchor (v7: 100% photo data -> a token on every image carries no signal).
Caption assembled locally from: enum slots (Gemini) + resolution (derived from pixel size) + booru
rating (Gemini, Falconsai fallback) + WD14 tags + watermark flag + rich NL.

Assembled order:
  <shot_type>, <view>, <camera_angle>, <quality>, <resolution>, <capture_style>, <lighting..>,
  <condition..>, <color_grade>, <camera_lens>, <depth_of_field>, <expression>, <body_type>,
  <breast_size>, <ethnicity>, <skin_tone>, <setting_type>, <rating>, <wd14 tags>[, watermark], <nl>
"""
import json
import re
import threading
import time
from pathlib import Path

VOCAB = {
    "shot_type": ["extreme close-up", "close-up", "portrait", "upper body", "cowboy shot",
                  "full body", "wide shot"],
    "view": ["front view", "three-quarter view", "profile view", "back view",
             "looking over shoulder", "looking at viewer", "looking away"],
    "camera_angle": ["eye level", "from above", "from below", "overhead", "dutch angle"],
    "quality": ["masterpiece", "best quality", "high quality", "normal quality",
                "low quality", "worst quality"],
    "capture_style": ["amateur snapshot", "casual phone photo", "social media selfie",
                      "candid photo", "semi-professional", "professional photograph",
                      "editorial photography", "studio portrait"],
    "lighting": ["direct flash", "natural daylight", "golden hour", "blue hour",
                 "overcast flat light", "indoor artificial light", "low light",
                 "soft window light", "studio lighting", "backlit", "rim light",
                 "neon lighting", "harsh sunlight", "ring light", "candlelight"],
    "condition": ["sharp focus", "soft focus", "grainy / high ISO", "motion blur",
                  "compressed / low-res", "overexposed", "underexposed", "lens flare",
                  "chromatic aberration", "vignette", "jpeg artifacts", "red-eye"],
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
    "rating": ["rating:general", "rating:sensitive", "rating:questionable", "rating:explicit"],
}

# single-pick string enums (Gemini returns one; coerce drops out-of-vocab -> "")
SINGLE_SLOTS = ("shot_type", "view", "camera_angle", "quality", "capture_style",
                "color_grade", "camera_lens", "depth_of_field", "expression",
                "body_type", "breast_size", "ethnicity", "skin_tone", "setting_type", "rating")
# multi-pick array enums (kept up to ARRAY_MAX)
ARRAY_SLOTS = ("lighting", "condition")
ARRAY_MAX = 2
# Gemini must always emit these
REQUIRED = ["shot_type", "quality", "capture_style", "rating", "has_watermark", "description"]
# caption assembly order. "_resolution"/"_rating" are handled specially in assemble_caption.
_ORDER = ("shot_type", "view", "camera_angle", "quality", "_resolution", "capture_style",
          "lighting", "condition", "color_grade", "camera_lens", "depth_of_field",
          "expression", "body_type", "breast_size", "ethnicity", "skin_tone",
          "setting_type", "_rating")


def clean_nl(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.rstrip(".").strip()


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".gif": "image/gif"}


def mime_for(path):
    """Gemini needs the correct image mime type; data/clean keeps original formats (jpg/png/webp)."""
    return _MIME.get(Path(path).suffix.lower(), "image/jpeg")


def resolution_tag(width, height):
    """Booru-style resolution token from pixel size (derived locally, NOT from Gemini).
    absurdres >= 2048 short side; highres >= 1024; '' below (v7 keeps down to 768)."""
    try:
        short = min(int(width), int(height))
    except (TypeError, ValueError):
        return ""
    if short >= 2048:
        return "absurdres"
    if short >= 1024:
        return "highres"
    return ""


def build_schema(vocab=VOCAB):
    props = {}
    for k in SINGLE_SLOTS:
        props[k] = {"type": "string", "enum": vocab[k]}
    for k in ARRAY_SLOTS:
        props[k] = {"type": "array", "items": {"type": "string", "enum": vocab[k]}}
    props["has_watermark"] = {"type": "boolean"}
    props["description"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(REQUIRED)}


def build_prompt(wd14_tags):
    return (
        "You are labeling a real photograph to train an image model. Return JSON only, matching the "
        "schema. Choose the enum value that best fits each field; omit an optional field if it does "
        "not clearly apply. Write 'description' as a detailed, factual account in this order: the "
        "subject (how many people, that they are adults, apparent gender); the face (face shape, eyes, "
        "lips, nose, brows, skin, hair color/length/style, makeup); then the visible body (build, "
        "torso, chest, midriff, arms, legs, as visible); then clothing with materials, colors and "
        "accessories (bags, phones, jewelry, glasses, shoes); then pose; then the setting and notable "
        "background objects and fine detail. Do NOT mention image quality, resolution, camera, or "
        "lighting in the description (those are separate fields), and do NOT begin with a label. "
        "These content tags are accurate context: " + (wd14_tags or "")
    )


def coerce_response(raw, vocab=VOCAB):
    """Validate a raw Gemini dict against the vocab. Out-of-vocab -> dropped; arrays clamped to ARRAY_MAX."""
    raw = raw or {}
    out = {}
    for k in SINGLE_SLOTS:
        v = raw.get(k)
        out[k] = v if v in vocab[k] else ""
    for k in ARRAY_SLOTS:
        out[k] = [x for x in (raw.get(k) or []) if x in vocab[k]][:ARRAY_MAX]
    out["has_watermark"] = bool(raw.get("has_watermark"))
    out["nl"] = clean_nl(raw.get("description"))
    return out


def assemble_caption(parts, wd14_tags, resolution="", fallback_rating="rating:general"):
    """parts = coerce_response output. Empty pieces omitted; no anchor (v7).
    rating = Gemini's if present else fallback_rating (Falconsai-derived); resolution derived locally."""
    p = []
    for k in _ORDER:
        if k == "_resolution":
            if resolution:
                p.append(resolution)
        elif k == "_rating":
            p.append(parts.get("rating") or fallback_rating)
        elif k in ARRAY_SLOTS:
            p += parts.get(k, [])
        elif parts.get(k):
            p.append(parts[k])
    if wd14_tags:
        p.append(wd14_tags)
    if parts.get("has_watermark"):
        p.append("watermark")
    if parts.get("nl"):
        p.append(parts["nl"])
    return ", ".join(x for x in p if x)


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
    assemble_caption renders as tags + fallback rating only."""
    def __init__(self, cfg, generate=None, cache=None):
        g = cfg["caption"]["gemini"]
        self.model = g["model"]
        self.max_output_tokens = g.get("max_output_tokens", 450)
        self.max_retries = g.get("max_retries", 4)
        self.block_none = g.get("safety_block_none", True)
        self.schema = build_schema()
        self.concurrency = g.get("concurrency", 10)
        self.cache = cache if cache is not None else {}
        self._lock = threading.Lock()      # guards self.cache during concurrent caption_many
        self._generate = generate or self._default_generate

    def caption(self, path, wd14_tags):
        key = str(path)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        try:
            raw = self._generate(path, wd14_tags)
            errored = False
        except Exception:
            raw, errored = {}, True        # network/API/config error -> blank now, but DO NOT cache (retry next run)
        result = coerce_response(raw)
        if not errored:
            with self._lock:
                self.cache[key] = result   # cache successes incl. legit empty refusals; never cache errors
        return result

    def caption_many(self, items, on_result=None):
        """items: iterable of (path, wd14_tags). Captions concurrently (thread pool size=self.concurrency).
        Returns {path: parts}. on_result(path, parts) runs in the calling thread as each finishes
        (use for progress + early-abort). Cache-aware and thread-safe."""
        from concurrent.futures import ThreadPoolExecutor
        items = list(items)
        results = {}

        def work(it):
            path, tags = it
            return path, self.caption(path, tags)

        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            for path, parts in ex.map(work, items):
                results[path] = parts
                if on_result is not None:
                    on_result(path, parts)
        return results

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
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )] if self.block_none else None
        img = types.Part.from_bytes(data=Path(path).read_bytes(),
                                    mime_type=mime_for(path))
        cfg = types.GenerateContentConfig(
            safety_settings=safety,
            response_mime_type="application/json",
            response_schema=self.schema,
            max_output_tokens=self.max_output_tokens,
        )
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=[img, build_prompt(wd14_tags)], config=cfg)
                return json.loads(resp.text) if resp.text else {}
            except Exception as e:           # rate-limit/5xx -> backoff + retry; final raises -> caption() fallback
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))   # exp backoff capped 30s (self-throttles 429 under concurrency)
        raise last
