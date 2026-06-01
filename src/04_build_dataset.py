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


def write_dataset_toml(out_path, image_dir, resolutions, min_ar, max_ar, num_ar_buckets, num_repeats):
    """diffusion-pipe dataset config. frame_buckets=[1] => image-only.
    diffusion-pipe resizes each image to the target AREA (upscaling smaller ones);
    no per-image no-upscale flag exists, hence the low default resolution in pipeline.yaml."""
    image_dir = str(image_dir).replace("\\", "/")
    res_list = ", ".join(str(r) for r in resolutions)
    toml = f"""# diffusion-pipe dataset config (Anima full finetune, images only)
resolutions = [{res_list}]
enable_ar_bucket = true
min_ar = {min_ar}
max_ar = {max_ar}
num_ar_buckets = {num_ar_buckets}
frame_buckets = [1]

[[directory]]
path = '{image_dir}'
num_repeats = {num_repeats}
"""
    Path(out_path).write_text(toml, encoding="utf-8")


def main():
    cfg = common.load_config()
    ds = cfg["dataset"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = curate(rows, ds["buckets_to_keep"])
    dest = Path(cfg["paths"]["dataset"])
    # Safety: dest is wiped below; never let a misconfig point it at the raw/clean inputs.
    for protected in ("raw", "clean"):
        if dest.resolve() == Path(cfg["paths"][protected]).resolve():
            raise RuntimeError(f"paths.dataset must not equal paths.{protected} (would delete inputs).")
    if dest.exists():
        shutil.rmtree(dest)  # rebuild cleanly (idempotent)
    LOG.info("Stage 4: curated %d images -> %s", len(kept), dest)

    for r in kept:
        write_pair(r["path"], r["caption"], dest)

    fcfg = cfg["finetune"]
    base = fcfg["base_dir"].rstrip("/")
    vast_dataset_dir = f"{base}/data/dataset"   # where the dataset lives ON Vast
    toml_path = Path(cfg["paths"]["outputs"]) / f"{fcfg['project_name']}_dataset_config.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_toml(
        toml_path, image_dir=vast_dataset_dir,
        resolutions=ds["resolutions"], min_ar=ds["min_ar"], max_ar=ds["max_ar"],
        num_ar_buckets=ds["num_ar_buckets"], num_repeats=ds["num_repeats"],
    )
    LOG.info("Stage 4 done. dataset.toml -> %s (image_dir=%s)", toml_path, vast_dataset_dir)


if __name__ == "__main__":
    main()
