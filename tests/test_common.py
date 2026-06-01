import csv
from pathlib import Path

import pytest

from conftest import load_stage  # noqa: F401  (ensures src on path)
import common


def test_load_config_reads_yaml(tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text("paths:\n  raw: data/raw\ningest:\n  min_size: 512\n", encoding="utf-8")
    cfg = common.load_config(cfg_file)
    assert cfg["ingest"]["min_size"] == 512
    assert cfg["paths"]["raw"] == "data/raw"


def test_manifest_roundtrip_and_augment(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"path": "a.jpg", "width": "800", "height": "600", "dropped": "False", "drop_reason": ""},
        {"path": "b.jpg", "width": "512", "height": "512", "dropped": "True", "drop_reason": "blurry"},
    ]
    common.write_manifest(manifest, rows)
    back = common.read_manifest(manifest)
    assert back[0]["path"] == "a.jpg"
    assert back[1]["drop_reason"] == "blurry"

    # augment: add a column keyed by path, others untouched
    common.augment_manifest(manifest, {"a.jpg": {"bucket": "good"}, "b.jpg": {"bucket": "bad"}})
    aug = common.read_manifest(manifest)
    by_path = {r["path"]: r for r in aug}
    assert by_path["a.jpg"]["bucket"] == "good"
    assert by_path["b.jpg"]["drop_reason"] == "blurry"  # preserved
