"""Unit tests for v10 caption pure helpers (no model load)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v10_caption", pathlib.Path(__file__).resolve().parents[1] / "src" / "v10_caption.py")
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def test_assemble_basic():
    out = cap.assemble_caption("masterpiece, best quality, score_7", "rating:general",
                               ["woman", "kitchen", "window"])
    assert out == "masterpiece, best quality, score_7, rating:general, woman, kitchen, window"


def test_assemble_skips_empty_rating_and_tags():
    out = cap.assemble_caption("masterpiece, best quality, score_7", "", [])
    assert out == "masterpiece, best quality, score_7"


def test_assemble_dedups_and_strips_tags():
    out = cap.assemble_caption("masterpiece, best quality, score_7", "rating:explicit",
                               [" woman ", "woman", "smile"])
    assert out == "masterpiece, best quality, score_7, rating:explicit, woman, smile"


def test_underage_hit_matches():
    hit = cap.underage_hit("woman, child, kitchen", {"child", "loli"})
    assert hit == {"child"}


def test_underage_hit_clean():
    assert cap.underage_hit("woman, kitchen, window", {"child", "loli"}) == set()
