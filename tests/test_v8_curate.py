"""Unit tests for v8 curation pure helpers (AR-crop math + ratio report)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v8_curate", pathlib.Path(__file__).resolve().parents[1] / "src" / "v8_curate.py")
v8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v8)


def test_ar_crop_wide_to_max():
    # 2000x1000 (AR 2.0) -> trim width to AR 1.5 -> 1500 wide, centered
    assert v8.ar_crop_box(2000, 1000, 0.66, 1.5) == (250, 0, 1750, 1000)


def test_ar_crop_in_range_noop():
    assert v8.ar_crop_box(1600, 1600, 0.66, 1.5) == (0, 0, 1600, 1600)


def test_ar_crop_tall_to_min():
    # 1000x2000 (AR 0.5) -> trim height to AR 0.66 -> height round(1000/0.66)=1515, centered
    assert v8.ar_crop_box(1000, 2000, 0.66, 1.5) == (0, 242, 1000, 1757)


def test_ar_crop_preserves_short_side():
    # cropping must never shrink min-dimension below the >=1536 gate
    for w, h in [(4000, 1500), (1536, 4000), (1600, 1536)]:
        l, t, r, b = v8.ar_crop_box(w, h, 0.66, 1.5)
        assert min(r - l, b - t) >= min(w, h)  # short side preserved


def test_ratio_ok_pass():
    ok, _ = v8.ratio_ok({"detail": 60, "anchor": 35, "bg": 5})
    assert ok is True


def test_ratio_ok_warns_when_off():
    ok, msg = v8.ratio_ok({"detail": 10, "anchor": 80, "bg": 10})
    assert ok is False and "detail" in msg
