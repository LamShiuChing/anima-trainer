"""Stage 4: curate good+medium, copy to flat data/dataset/, write img.txt sidecars, emit dataset TOML."""
import shutil
from pathlib import Path

import common

LOG = common.setup_logging()


def curate(rows, buckets_to_keep):
    return [r for r in rows if r.get("dropped") == "False" and r.get("bucket") in buckets_to_keep]


def write_pair(img_path, caption, dest_dir):
    img_path = Path(img_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest_dir / img_path.name)
    (dest_dir / (img_path.stem + ".txt")).write_text(caption, encoding="utf-8")


def write_dataset_toml(out_path, image_dir, resolution, num_repeats, caption_dropout_rate):
    # image_dir uses forward slashes (sd-scripts accepts them on Windows; avoids TOML backslash-escaping).
    image_dir = str(image_dir).replace("\\", "/")
    toml = f"""[general]
resolution = {resolution}
enable_bucket = true
bucket_no_upscale = false
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096

[[datasets]]
resolution = {resolution}

  [[datasets.subsets]]
  num_repeats = {num_repeats}
  image_dir = "{image_dir}"
  caption_extension = ".txt"
  caption_dropout_rate = {caption_dropout_rate}
"""
    Path(out_path).write_text(toml, encoding="utf-8")


def main():
    cfg = common.load_config()
    ds = cfg["dataset"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = curate(rows, ds["buckets_to_keep"])
    dest = Path(cfg["paths"]["dataset"])
    if dest.exists():
        shutil.rmtree(dest)  # rebuild cleanly (idempotent)
    LOG.info("Stage 4: curated %d images -> %s", len(kept), dest)

    for r in kept:
        write_pair(r["path"], r["caption"], dest)

    toml_path = Path(cfg["paths"]["outputs"]) / f"{cfg['train']['project_name']}_dataset_config.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_toml(toml_path, dest, ds["resolution"], ds["num_repeats"], ds["caption_dropout_rate"])
    LOG.info("Stage 4 done. Dataset TOML -> %s", toml_path)


if __name__ == "__main__":
    main()
