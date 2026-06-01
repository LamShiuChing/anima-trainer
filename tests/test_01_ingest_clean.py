import numpy as np
from PIL import Image

from conftest import load_stage

stage = load_stage("01_ingest_clean.py")


def test_too_small_detected(make_image):
    small = make_image("s.jpg", size=(300, 900))   # min dim 300 < 512
    big = make_image("b.jpg", size=(800, 800))
    assert stage.is_too_small(small, min_size=512) is True
    assert stage.is_too_small(big, min_size=512) is False


def test_blur_variance_ranks_noise_above_flat(make_image):
    flat = make_image("flat.jpg", size=(600, 600), color=(128, 128, 128))
    noisy = make_image("noisy.jpg", size=(600, 600), noise=True)
    assert stage.blur_variance(flat) < stage.blur_variance(noisy)


def test_corrupt_file_detected(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    assert stage.is_corrupt(bad) is True


def test_phash_near_duplicates_group(make_image):
    a = make_image("a.jpg", size=(512, 512), color=(10, 20, 30))
    a2 = make_image("a2.jpg", size=(512, 512), color=(11, 21, 31))  # nearly identical
    far = make_image("far.jpg", size=(512, 512), noise=True)
    ha, ha2, hf = (stage.phash(a), stage.phash(a2), stage.phash(far))
    assert stage.hamming(ha, ha2) <= 6
    assert stage.hamming(ha, hf) > 6


def test_dedup_keeps_one_per_group_highest_resolution(make_image):
    big = make_image("big.jpg", size=(1024, 1024), color=(10, 20, 30))
    small = make_image("small.jpg", size=(512, 512), color=(11, 21, 31))  # near-dup, lower res
    keep, drop = stage.dedup([big, small], hamming_threshold=6)
    assert big in keep
    assert small in drop


def test_drop_reason_respects_flags():
    # corrupt always drops; small/blurry only drop when their flag is on
    assert stage.drop_reason(corrupt=True, too_small=True, blurry=True,
                             drop_small=False, drop_blurry=False) == "corrupt"
    assert stage.drop_reason(corrupt=False, too_small=True, blurry=False,
                             drop_small=False, drop_blurry=False) == ""
    assert stage.drop_reason(corrupt=False, too_small=True, blurry=False,
                             drop_small=True, drop_blurry=False) == "too_small"
    assert stage.drop_reason(corrupt=False, too_small=False, blurry=True,
                             drop_small=False, drop_blurry=True) == "blurry"
