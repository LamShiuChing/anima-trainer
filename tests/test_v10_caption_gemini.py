"""Unit tests for v10 structured-Gemini caption pure helpers (no network)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v10_caption_gemini", pathlib.Path(__file__).resolve().parents[1] / "src" / "v10_caption_gemini.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def test_clean_tag_lowercases_and_trims():
    assert g.clean_tag("  Polka Dot ,") == "polka dot"


def test_clean_nl_collapses_and_strips_period():
    assert g.clean_nl("A  woman\n walks.") == "A woman walks"


def test_coerce_drops_out_of_vocab_enums():
    raw = {"shot_type": "full body", "view": "NOT_A_VIEW", "rating": "explicit"}
    out = g.coerce(raw)
    assert out["shot_type"] == "full body"
    assert out["view"] == ""                 # out-of-vocab dropped
    assert out["rating"] == "explicit"


def test_coerce_clamps_arrays_and_dedups_tags():
    raw = {"lighting": ["natural daylight", "golden hour", "low light"],   # > ARRAY_MAX
           "condition": ["sharp focus", "bogus"],
           "tags": [" Woman ", "woman", "DRESS", "dress", "road"]}
    out = g.coerce(raw)
    assert out["lighting"] == ["natural daylight", "golden hour"]          # clamped to 2
    assert out["condition"] == ["sharp focus"]                            # bogus dropped
    assert out["tags"] == ["woman", "dress", "road"]                      # lowercased + deduped


def test_assemble_full_order():
    parts = {"quality": "masterpiece", "shot_type": "full body", "view": "front view", "rating": "safe",
             "lighting": ["natural daylight"], "condition": [], "setting_type": "city street",
             "tags": ["woman", "dress"], "has_watermark": False,
             "nl": "A woman in a dress on a street"}
    out = g.assemble(parts)
    assert out == ("masterpiece, safe, full body, front view, "
                   "natural daylight, city street, woman, dress, A woman in a dress on a street")


def test_assemble_quality_and_rating_fallback_and_watermark():
    parts = g.coerce({"shot_type": "portrait", "tags": ["man"], "has_watermark": True,
                      "description": "A man."})
    out = g.assemble(parts, fallback_quality="best quality", fallback_rating="safe")
    # no quality/rating from model -> fallbacks lead; watermark appended before NL
    assert out.startswith("best quality, safe, portrait, ")
    assert ", man, watermark, A man" in out


def test_coerce_validates_quality():
    assert g.coerce({"quality": "masterpiece"})["quality"] == "masterpiece"
    assert g.coerce({"quality": "amazing"})["quality"] == ""        # out-of-vocab dropped


def test_build_schema_requires_quality_and_rating():
    schema = g.build_schema()
    assert "quality" in schema["required"] and "rating" in schema["required"]
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["quality"]["enum"][0] == "masterpiece"
