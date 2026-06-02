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
