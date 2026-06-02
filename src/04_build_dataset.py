"""Stage 4: curate kept buckets, copy to flat data/dataset/, write img.txt sidecars, emit dataset TOML."""
import shutil
from pathlib import Path

from PIL import Image

import common

LOG = common.setup_logging()

# diffusion-pipe treats webp/gif/bmp as video (or rejects them) -> convert those to jpg on copy.
PASSTHROUGH_EXTS = {".jpg", ".jpeg", ".png"}


def curate(rows, min_resolution=0, min_blur_var=0.0):
    """Keep non-dropped rows. v5: no aesthetic-bucket filter (all buckets kept, tagged).
    Optional technical gates read sizes/blur recorded by stage 1 (no image re-read)."""
    out = []
    for r in rows:
        if r.get("dropped") != "False":
            continue
        if not r.get("caption"):
            continue  # uncaptioned (stage 3 skipped/failed the row) -> nothing to build
        if min_resolution:
            try:
                if min(int(r["width"]), int(r["height"])) < min_resolution:
                    continue
            except (KeyError, ValueError):
                continue  # no size on record -> exclude from a resolution-filtered run
        if min_blur_var:
            try:
                if float(r["blur_var"]) < min_blur_var:
                    continue
            except (KeyError, ValueError):
                continue
        out.append(r)
    return out


def write_pair(img_path, caption, dest_dir):
    img_path = Path(img_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if img_path.suffix.lower() in PASSTHROUGH_EXTS:
        shutil.copy2(img_path, dest_dir / img_path.name)
    else:
        # convert webp/gif/bmp -> jpg (diffusion-pipe can't ingest them); keep the stem so the .txt matches
        Image.open(img_path).convert("RGB").save(dest_dir / (img_path.stem + ".jpg"), quality=95)
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
    kept = curate(rows, ds.get("min_resolution", 0), ds.get("min_blur_var", 0.0))
    LOG.info("Stage 4: curated %d images (min_resolution=%s, min_blur_var=%s)",
             len(kept), ds.get("min_resolution", 0), ds.get("min_blur_var", 0.0))
    dest = Path(cfg["paths"]["dataset"])
    # Safety: dest is wiped below; never let a misconfig point it at the raw/clean inputs.
    for protected in ("raw", "clean"):
        if dest.resolve() == Path(cfg["paths"][protected]).resolve():
            raise RuntimeError(f"paths.dataset must not equal paths.{protected} (would delete inputs).")
    if dest.exists():
        shutil.rmtree(dest)  # rebuild cleanly (idempotent)
    LOG.info("Stage 4: curated %d images -> %s", len(kept), dest)

    written = 0
    for r in kept:
        if not Path(r["path"]).exists():
            LOG.warning("Stage 4: source missing, skipping %s", r["path"])
            continue
        write_pair(r["path"], r["caption"], dest)
        written += 1
    LOG.info("Stage 4: wrote %d image+caption pairs", written)

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
