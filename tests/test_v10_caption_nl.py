"""Unit tests for v10 NL-caption pure helpers (no network)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v10_caption_nl", pathlib.Path(__file__).resolve().parents[1] / "src" / "v10_caption_nl.py")
nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nl)


def test_normalize_collapses_whitespace_and_newlines():
    assert nl.normalize_nl("  A  woman\n  walking.\n") == "A woman walking."


def test_normalize_none_is_empty():
    assert nl.normalize_nl(None) == ""


def test_assemble_appends_nl():
    out = nl.assemble_full("masterpiece, best quality, score_7, rating:general, woman, road",
                           "A woman walks along a city road in daylight.")
    assert out == ("masterpiece, best quality, score_7, rating:general, woman, road, "
                   "A woman walks along a city road in daylight.")


def test_assemble_empty_nl_returns_tags_only():
    base = "masterpiece, best quality, score_7, rating:general, woman, road"
    assert nl.assemble_full(base, "") == base
    assert nl.assemble_full(base, None) == base


def test_assemble_strips_trailing_comma_on_base():
    out = nl.assemble_full("a, b, c,", "Hello world.")
    assert out == "a, b, c, Hello world."


def test_assemble_idempotent_base_unchanged():
    # rebuilding from the same caption_tags + nl yields the same result (no drift)
    base = "q, rating:general, woman"
    once = nl.assemble_full(base, "Desc.")
    twice = nl.assemble_full(base, "Desc.")
    assert once == twice == "q, rating:general, woman, Desc."
