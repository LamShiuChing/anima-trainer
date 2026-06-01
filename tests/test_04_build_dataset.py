try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport
from pathlib import Path

from conftest import load_stage

stage = load_stage("04_build_dataset.py")


def test_curate_keeps_only_configured_buckets():
    rows = [
        {"path": "a.jpg", "dropped": "False", "bucket": "good", "caption": "c1"},
        {"path": "b.jpg", "dropped": "False", "bucket": "medium", "caption": "c2"},
        {"path": "c.jpg", "dropped": "False", "bucket": "bad", "caption": "c3"},
        {"path": "d.jpg", "dropped": "True", "bucket": "good", "caption": "c4"},
    ]
    kept = stage.curate(rows, buckets_to_keep=["good", "medium"])
    assert {r["path"] for r in kept} == {"a.jpg", "b.jpg"}


def test_dataset_toml_is_valid_and_has_subset(tmp_path):
    out = tmp_path / "dataset.toml"
    stage.write_dataset_toml(out, image_dir="data/dataset", resolution=768, num_repeats=5, caption_dropout_rate=0.1)
    data = tomllib.loads(out.read_text(encoding="utf-8"))
    assert data["general"]["resolution"] == 768
    assert data["general"]["enable_bucket"] is True
    sub = data["datasets"][0]["subsets"][0]
    assert sub["num_repeats"] == 5
    assert sub["image_dir"] == "data/dataset"
    assert sub["caption_extension"] == ".txt"


def test_sidecar_written(tmp_path, make_image):
    img = make_image("x.jpg")
    dest_dir = tmp_path / "dataset"
    dest_dir.mkdir()
    stage.write_pair(img, "masterpiece, best quality, safe, realistic photo, a cat", dest_dir)
    assert (dest_dir / "x.jpg").exists()
    assert (dest_dir / "x.txt").read_text(encoding="utf-8") == "masterpiece, best quality, safe, realistic photo, a cat"
