"""Shared utilities: config, logging, image iteration, manifest IO."""
import csv
import logging
import sys
from pathlib import Path

import yaml

LOG = logging.getLogger("anima")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def setup_logging():
    if not LOG.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(h)
        LOG.setLevel(logging.INFO)
    return LOG


def load_config(path=None):
    # Default is project-relative (common.py lives in src/), so it works regardless of CWD.
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_images(directory):
    directory = Path(directory)
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def read_manifest(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # union of keys, stable order: first-seen
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def augment_manifest(path, updates_by_path):
    """updates_by_path: {row_path: {col: value, ...}}. Adds/overwrites columns, preserves the rest."""
    rows = read_manifest(path)
    matched = set()
    for r in rows:
        upd = updates_by_path.get(r["path"])
        if upd:
            r.update({k: str(v) for k, v in upd.items()})
            matched.add(r["path"])
    for key in updates_by_path:
        if key not in matched:
            LOG.warning("augment_manifest: no manifest row matched path %r (update dropped)", key)
    write_manifest(path, rows)


def ensure_aesthetic_weights(cfg):
    """Download the aesthetic MLP .pth once into models/aesthetic/ if absent. Returns the local Path."""
    import urllib.request

    dest = Path(cfg["quality"]["aesthetic_weights_file"])
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["quality"]["aesthetic_weights_url"]
    LOG.info("Downloading aesthetic weights: %s", url)
    # Download to a temp sibling then atomically replace, so an interrupted download
    # never leaves a truncated file that a later run would wrongly skip.
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest
