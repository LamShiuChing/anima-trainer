# tests/test_gemini_caption.py  (v7: expanded enum vocab, no anchor, Gemini rating + fallback)
from conftest import load_stage

gc = load_stage("gemini_caption.py")


def test_vocab_has_v7_slots_and_booru_ladders():
    assert gc.VOCAB["quality"] == ["masterpiece", "best quality", "high quality",
                                   "normal quality", "low quality", "worst quality"]
    assert gc.VOCAB["rating"] == ["rating:general", "rating:sensitive",
                                  "rating:questionable", "rating:explicit"]
    assert gc.VOCAB["breast_size"][0] == "flat chest"
    for slot in ("shot_type", "view", "camera_angle", "capture_style", "lighting", "condition",
                 "color_grade", "camera_lens", "depth_of_field", "expression", "body_type",
                 "ethnicity", "skin_tone", "setting_type"):
        assert slot in gc.VOCAB and len(gc.VOCAB[slot]) >= 2
    assert "full body" in gc.VOCAB["shot_type"]
    assert "profile view" in gc.VOCAB["view"]


def test_no_anchor_constant():
    assert not hasattr(gc, "ANCHOR")


def test_slot_partitions_are_consistent():
    # every single/array slot must exist in VOCAB; the two sets must not overlap
    assert set(gc.SINGLE_SLOTS).isdisjoint(gc.ARRAY_SLOTS)
    for k in gc.SINGLE_SLOTS + gc.ARRAY_SLOTS:
        assert k in gc.VOCAB
    assert set(gc.ARRAY_SLOTS) == {"lighting", "condition"}


def test_build_schema_enums_and_required():
    s = gc.build_schema()
    assert s["properties"]["capture_style"]["enum"] == gc.VOCAB["capture_style"]
    assert s["properties"]["lighting"]["items"]["enum"] == gc.VOCAB["lighting"]
    assert s["properties"]["rating"]["enum"] == gc.VOCAB["rating"]
    assert set(s["required"]) == {"shot_type", "quality", "capture_style", "rating",
                                  "has_watermark", "description"}


def test_resolution_tag_thresholds():
    assert gc.resolution_tag(3000, 2048) == "absurdres"
    assert gc.resolution_tag(2048, 4000) == "absurdres"
    assert gc.resolution_tag(1024, 1500) == "highres"
    assert gc.resolution_tag(1536, 1024) == "highres"
    assert gc.resolution_tag(800, 1200) == ""          # short side < 1024
    assert gc.resolution_tag(None, "x") == ""           # bad input -> safe


def test_coerce_filters_and_clamps():
    raw = {
        "shot_type": "full body",
        "view": "NOPE",                                   # invalid -> ""
        "quality": "best quality",
        "capture_style": "amateur snapshot",
        "lighting": ["direct flash", "golden hour", "low light"],   # 3 -> clamp 2
        "condition": ["sharp focus", "bogus"],
        "rating": "rating:explicit",
        "breast_size": "large breasts",
        "has_watermark": True,
        "description": "  a person.\n ",
    }
    out = gc.coerce_response(raw)
    assert out["shot_type"] == "full body"
    assert out["view"] == ""
    assert out["quality"] == "best quality"
    assert out["lighting"] == ["direct flash", "golden hour"]
    assert out["condition"] == ["sharp focus"]
    assert out["rating"] == "rating:explicit"
    assert out["breast_size"] == "large breasts"
    assert out["has_watermark"] is True
    assert out["nl"] == "a person"


def test_coerce_empty_is_all_blank():
    out = gc.coerce_response({})
    for k in gc.SINGLE_SLOTS:
        assert out[k] == ""
    for k in gc.ARRAY_SLOTS:
        assert out[k] == []
    assert out["has_watermark"] is False
    assert out["nl"] == ""


def test_assemble_full_caption_order_no_anchor():
    parts = gc.coerce_response({
        "shot_type": "full body", "view": "front view", "quality": "high quality",
        "capture_style": "casual phone photo", "lighting": ["natural daylight"],
        "condition": ["sharp focus"], "expression": "smile", "body_type": "slim",
        "ethnicity": "east asian", "skin_tone": "fair skin", "setting_type": "city street",
        "rating": "rating:general", "has_watermark": False,
        "description": "a woman holding an iced coffee",
    })
    out = gc.assemble_caption(parts, wd14_tags="1girl, hoodie, coffee",
                              resolution="highres", fallback_rating="rating:general")
    assert out == ("full body, front view, high quality, highres, casual phone photo, "
                   "natural daylight, sharp focus, smile, slim, east asian, fair skin, "
                   "city street, rating:general, 1girl, hoodie, coffee, "
                   "a woman holding an iced coffee")


def test_assemble_uses_gemini_rating_over_fallback():
    parts = gc.coerce_response({"shot_type": "close-up", "quality": "best quality",
                                "capture_style": "studio portrait", "rating": "rating:explicit",
                                "description": "x"})
    out = gc.assemble_caption(parts, wd14_tags="nude", resolution="", fallback_rating="rating:general")
    assert "rating:explicit" in out and "rating:general" not in out


def test_assemble_falls_back_to_falconsai_rating_when_gemini_blank():
    parts = gc.coerce_response({})                       # Gemini refusal -> blank
    out = gc.assemble_caption(parts, wd14_tags="nude, bed", resolution="",
                              fallback_rating="rating:explicit")
    assert out == "rating:explicit, nude, bed"          # graceful: fallback rating + tags only


def test_assemble_appends_watermark_token():
    parts = gc.coerce_response({"shot_type": "portrait", "quality": "high quality",
                                "capture_style": "professional photograph",
                                "rating": "rating:general", "has_watermark": True,
                                "description": ""})
    out = gc.assemble_caption(parts, wd14_tags="logo, text", resolution="")
    assert out.endswith("logo, text, watermark")
    assert out.startswith("portrait, high quality, professional photograph, rating:general")


def test_build_prompt_demands_detail_no_anchor():
    p = gc.build_prompt("woman, kitchen")
    assert "woman, kitchen" in p
    assert "JSON" in p
    assert "realistic photo" not in p.lower()           # no anchor instruction
    assert "background" in p.lower()                     # demands scene detail


# ---- GeminiCaptioner (injected fake generate) ----

def _cfg(tmp_path):
    return {"caption": {"gemini": {"model": "gemini-2.5-flash-lite", "safety_block_none": True,
                                   "max_output_tokens": 450, "max_retries": 2,
                                   "cache_file": str(tmp_path / "cache.json")}}}


def test_captioner_calls_generate_and_coerces(tmp_path):
    def fake_generate(path, tags):
        return {"shot_type": "full body", "quality": "high quality",
                "capture_style": "amateur snapshot", "rating": "rating:general",
                "has_watermark": False, "description": "a dog"}
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=fake_generate)
    out = cap.caption("x.jpg", "dog")
    assert out["quality"] == "high quality"
    assert out["rating"] == "rating:general"
    assert out["nl"] == "a dog"


def test_captioner_refusal_returns_blank_for_fallback(tmp_path):
    def refuse(path, tags):
        raise RuntimeError("blocked")
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=refuse)
    assert cap.caption("x.jpg", "nude, bed") == gc.coerce_response({})


def test_captioner_cache_hit_skips_generate(tmp_path):
    def boom(path, tags):
        raise AssertionError("generate must not be called on a cache hit")
    seeded = {"x.jpg": gc.coerce_response({"quality": "low quality", "description": "cached"})}
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=boom, cache=seeded)
    assert cap.caption("x.jpg", "dog")["nl"] == "cached"


def test_captioner_does_not_cache_on_error(tmp_path):
    calls = {"n": 0}
    def flaky(path, tags):
        calls["n"] += 1
        raise RuntimeError("transient")
    cap = gc.GeminiCaptioner(_cfg(tmp_path), generate=flaky)
    cap.caption("x.jpg", "t")
    cap.caption("x.jpg", "t")
    assert calls["n"] == 2


def test_caption_many_runs_all_and_caches(tmp_path):
    def fake(path, tags):
        return {"shot_type": "portrait", "quality": "high quality",
                "capture_style": "amateur snapshot", "rating": "rating:general",
                "has_watermark": False, "description": path}
    cfg = _cfg(tmp_path)
    cfg["caption"]["gemini"]["concurrency"] = 4
    cap = gc.GeminiCaptioner(cfg, generate=fake)
    got = []
    res = cap.caption_many([("a.jpg", "t1"), ("b.jpg", "t2"), ("c.jpg", "t3")],
                           on_result=lambda p, parts: got.append(p))
    assert set(res) == {"a.jpg", "b.jpg", "c.jpg"}
    assert res["b.jpg"]["nl"] == "b.jpg"
    assert set(got) == {"a.jpg", "b.jpg", "c.jpg"}
    assert set(cap.cache) == {"a.jpg", "b.jpg", "c.jpg"}


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    gc.save_cache(path, {"a.jpg": {"nl": "x"}})
    assert gc.load_cache(path) == {"a.jpg": {"nl": "x"}}
    assert gc.load_cache(tmp_path / "missing.json") == {}


def test_mime_for_by_extension():
    assert gc.mime_for("a.jpg") == "image/jpeg"
    assert gc.mime_for("b.PNG") == "image/png"
    assert gc.mime_for("c.webp") == "image/webp"
    assert gc.mime_for("d.unknown") == "image/jpeg"
