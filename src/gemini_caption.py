"""v5 Gemini captioner: enum-locked style vocab + NL description, BLOCK_NONE, tags-only fallback.

Caption format (anchor prepended here, NOT by Gemini):
  realistic photo, <quality>, <capture_style>, <lighting...>, <condition...>, <safety>, <wd14 tags>[, watermark][, <nl>]
"""
import json
import re
from pathlib import Path

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


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".gif": "image/gif"}


def mime_for(path):
    """Gemini needs the correct image mime type; data/clean keeps original formats (jpg/png/webp)."""
    return _MIME.get(Path(path).suffix.lower(), "image/jpeg")


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
                                    mime_type=mime_for(path))
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
