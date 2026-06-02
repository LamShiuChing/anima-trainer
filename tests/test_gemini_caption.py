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


def test_mime_for_by_extension():
    assert gc.mime_for("a.jpg") == "image/jpeg"
    assert gc.mime_for("a.JPEG") == "image/jpeg"
    assert gc.mime_for("b.png") == "image/png"
    assert gc.mime_for("c.webp") == "image/webp"
    assert gc.mime_for("d.unknown") == "image/jpeg"


def test_captioner_does_not_cache_on_error(tmp_path):
    calls = {"n": 0}
    def flaky(path, tags):
        calls["n"] += 1
        raise RuntimeError("transient")
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=flaky)
    cap.caption("x.jpg", "t")      # error -> blank, must NOT be cached
    cap.caption("x.jpg", "t")      # so this retries instead of a cache hit
    assert calls["n"] == 2


def test_caption_many_runs_all_and_caches(tmp_path):
    def fake(path, tags):
        return {"quality_level": "high quality", "capture_style": "amateur snapshot",
                "lighting": [], "condition": [], "has_watermark": False, "description": path}
    cfg = _cfg(tmp_path)
    cfg["caption"]["gemini"]["concurrency"] = 4
    cap = gc.GeminiCaptioner(cfg, generate=fake)
    got = []
    res = cap.caption_many([("a.jpg", "t1"), ("b.jpg", "t2"), ("c.jpg", "t3")],
                           on_result=lambda p, parts: got.append(p))
    assert set(res) == {"a.jpg", "b.jpg", "c.jpg"}
    assert res["b.jpg"]["nl"] == "b.jpg"
    assert set(got) == {"a.jpg", "b.jpg", "c.jpg"}        # on_result fired per image
    assert set(cap.cache) == {"a.jpg", "b.jpg", "c.jpg"}  # all cached
