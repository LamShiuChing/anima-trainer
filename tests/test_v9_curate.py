"""Unit tests for v9 curation pure helpers (AR-crop, background-sharpness rule, dedup)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v9_curate", pathlib.Path(__file__).resolve().parents[1] / "src" / "v9_curate.py")
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)


# --- AR-crop (carried from v8; pos-emb 0.66-1.5 cap) ---
def test_ar_crop_wide_to_max():
    assert v9.ar_crop_box(2000, 1000, 0.66, 1.5) == (250, 0, 1750, 1000)


def test_ar_crop_in_range_noop():
    assert v9.ar_crop_box(1600, 1600, 0.66, 1.5) == (0, 0, 1600, 1600)


def test_ar_crop_tall_to_min():
    assert v9.ar_crop_box(1000, 2000, 0.66, 1.5) == (0, 242, 1000, 1757)


def test_ar_crop_preserves_short_side():
    for w, h in [(4000, 1500), (1536, 4000), (1600, 1536)]:
        l, t, r, b = v9.ar_crop_box(w, h, 0.66, 1.5)
        assert min(r - l, b - t) >= min(w, h)


# --- background-sharpness rule (the v9 centerpiece) ---
def test_bg_uniform_sharp_passes():
    # deep-focus: all 16 tiles sharp -> keep
    assert v9.passes_bg_sharpness([200.0] * 16) is True


def test_bg_bimodal_bokeh_fails():
    # bokeh: 4 sharp subject tiles, 12 soft background tiles -> drop
    assert v9.passes_bg_sharpness([300.0] * 4 + [10.0] * 12) is False


def test_bg_empty_fails():
    assert v9.passes_bg_sharpness([]) is False


def test_bg_half_sharp_passes_at_default_fraction():
    # exactly half sharp -> passes at default min_frac 0.5
    assert v9.passes_bg_sharpness([200.0] * 8 + [10.0] * 8) is True
