"""Unit tests for v10 curation pure helpers (no image IO)."""
import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "v10_curate", pathlib.Path(__file__).resolve().parents[1] / "src" / "v10_curate.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)


def test_ar_crop_wide_to_max():
    assert v10.ar_crop_box(2000, 1000, 0.66, 1.5) == (250, 0, 1750, 1000)


def test_ar_crop_in_range_noop():
    assert v10.ar_crop_box(1600, 1600, 0.66, 1.5) == (0, 0, 1600, 1600)


def test_ar_crop_tall_to_min():
    assert v10.ar_crop_box(1000, 2000, 0.66, 1.5) == (0, 242, 1000, 1757)


def test_analysis_resize_downscales_long_side():
    assert v10.analysis_resize_dims(2000, 1000, 512) == (512, 256)
    assert v10.analysis_resize_dims(1000, 2000, 512) == (256, 512)


def test_analysis_resize_no_upscale():
    # already smaller than target -> unchanged (never upscale the analysis copy)
    assert v10.analysis_resize_dims(300, 200, 512) == (300, 200)


def test_fft_highfreq_ratio_flat_is_zero():
    flat = np.zeros((256, 256), dtype=np.float64)
    assert v10.fft_highfreq_ratio(flat) == 0.0


def test_fft_highfreq_ratio_noise_is_high():
    rng = np.random.RandomState(0)
    noise = rng.rand(256, 256)
    # white noise has broad spectrum -> substantial high-freq energy
    assert v10.fft_highfreq_ratio(noise, cutoff=0.25) > 0.3


def test_blockiness_detects_8px_steps():
    smooth = np.tile(np.linspace(0, 255, 64), (64, 1))
    blocky = smooth.copy()
    blocky[:, ::8] += 40                      # inject discontinuities on 8px column boundaries
    assert v10.blockiness(blocky) > v10.blockiness(smooth)


def test_jpeg_quality_standard_table_is_about_50():
    q = v10.jpeg_quality_estimate(v10.STD_LUMA)
    assert 45 <= q <= 55


def test_jpeg_quality_double_table_is_lower():
    doubled = [2 * x for x in v10.STD_LUMA]
    assert v10.jpeg_quality_estimate(doubled) < 40


def test_jpeg_quality_none_table_is_max():
    assert v10.jpeg_quality_estimate(None) == 100
