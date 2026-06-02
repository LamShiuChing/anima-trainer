try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from conftest import load_stage

stage = load_stage("04_build_dataset.py")


def test_curate_drops_dropped_rows_only_by_default():
    rows = [
        {"path": "a.jpg", "dropped": "False", "caption": "realistic photo, safe, x"},
        {"path": "b.jpg", "dropped": "True", "caption": "realistic photo, safe, y"},
    ]
    kept = stage.curate(rows)
    assert {r["path"] for r in kept} == {"a.jpg"}


def test_curate_requires_caption():
    rows = [
        {"path": "has.jpg", "dropped": "False", "caption": "realistic photo, safe, x"},
        {"path": "none.jpg", "dropped": "False"},                 # stage-3 skipped -> no caption
        {"path": "empty.jpg", "dropped": "False", "caption": ""},  # empty caption
    ]
    kept = stage.curate(rows)
    assert {r["path"] for r in kept} == {"has.jpg"}   # only the captioned row is buildable


def test_curate_min_resolution_and_blur():
    cap = "realistic photo, safe, x"
    rows = [
        {"path": "ok.jpg",    "dropped": "False", "caption": cap, "width": "1024", "height": "1300", "blur_var": "250.0"},
        {"path": "small.jpg", "dropped": "False", "caption": cap, "width": "800",  "height": "1300", "blur_var": "250.0"},  # <1024
        {"path": "soft.jpg",  "dropped": "False", "caption": cap, "width": "1200", "height": "1200", "blur_var": "40.0"},   # <min_blur
        {"path": "nosize.jpg","dropped": "False", "caption": cap, "blur_var": "250.0"},                                     # missing size
    ]
    kept = stage.curate(rows, min_resolution=1024, min_blur_var=100.0)
    assert {r["path"] for r in kept} == {"ok.jpg"}


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
