try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from conftest import load_stage

stage = load_stage("04_build_dataset.py")


def test_curate_keeps_all_three_buckets():
    rows = [
        {"path": "a.jpg", "dropped": "False", "bucket": "good", "caption": "c1"},
        {"path": "b.jpg", "dropped": "False", "bucket": "medium", "caption": "c2"},
        {"path": "c.jpg", "dropped": "False", "bucket": "bad", "caption": "c3"},
        {"path": "d.jpg", "dropped": "True", "bucket": "good", "caption": "c4"},
    ]
    kept = stage.curate(rows, buckets_to_keep=["good", "medium", "bad"])
    assert {r["path"] for r in kept} == {"a.jpg", "b.jpg", "c.jpg"}  # only the dropped one excluded


def test_curate_min_resolution_and_quality():
    rows = [
        {"path": "big_good.jpg", "dropped": "False", "bucket": "good", "width": "1024", "height": "800"},
        {"path": "small_good.jpg", "dropped": "False", "bucket": "good", "width": "640", "height": "900"},   # min 640 < 768
        {"path": "big_bad.jpg", "dropped": "False", "bucket": "bad", "width": "1200", "height": "1200"},     # bad bucket
        {"path": "big_medium.jpg", "dropped": "False", "bucket": "medium", "width": "768", "height": "1000"},
        {"path": "nosize.jpg", "dropped": "False", "bucket": "good"},                                        # no size -> excluded
    ]
    kept = stage.curate(rows, buckets_to_keep=["good", "medium"], min_resolution=768)
    assert {r["path"] for r in kept} == {"big_good.jpg", "big_medium.jpg"}


def test_dataset_toml_diffusion_pipe_schema(tmp_path):
    out = tmp_path / "dataset.toml"
    stage.write_dataset_toml(
        out, image_dir="/workspace/anima/data/dataset",
        resolutions=[512], min_ar=0.5, max_ar=2.0, num_ar_buckets=7, num_repeats=1,
    )
    data = tomllib.loads(out.read_text(encoding="utf-8"))
    assert data["resolutions"] == [512]
    assert data["enable_ar_bucket"] is True
    assert data["min_ar"] == 0.5
    assert data["max_ar"] == 2.0
    assert data["num_ar_buckets"] == 7
    assert data["frame_buckets"] == [1]                       # image-only training
    d0 = data["directory"][0]
    assert d0["path"] == "/workspace/anima/data/dataset"
    assert d0["num_repeats"] == 1


def test_sidecar_written(tmp_path, make_image):
    img = make_image("x.jpg")
    dest_dir = tmp_path / "dataset"
    dest_dir.mkdir()
    stage.write_pair(img, "masterpiece, best quality, safe", dest_dir)
    assert (dest_dir / "x.jpg").exists()
    assert (dest_dir / "x.txt").read_text(encoding="utf-8") == "masterpiece, best quality, safe"


def test_webp_converted_to_jpg(tmp_path, make_image):
    img = make_image("y.webp")            # diffusion-pipe rejects webp -> stage 4 must convert
    dest_dir = tmp_path / "dataset"
    dest_dir.mkdir()
    stage.write_pair(img, "high quality, safe", dest_dir)
    assert (dest_dir / "y.jpg").exists()
    assert not (dest_dir / "y.webp").exists()
    assert (dest_dir / "y.txt").read_text(encoding="utf-8") == "high quality, safe"
