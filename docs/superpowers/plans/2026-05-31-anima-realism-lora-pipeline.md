# Anima Realism LoRA — Phase 1 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 6-stage local Windows pipeline that turns `data/raw/` (~3000 social-media photos) into a trained realism domain-shift LoRA `.safetensors` for the Anima diffusion model.

**Architecture:** Six idempotent stages. A single `config/pipeline.yaml` holds every path + threshold. A shared `src/common.py` owns config loading, logging, image iteration, and the `manifest.csv` contract (the spine that each stage augments). Stages 1–5 are Python (pipeline venv, GPU for CLIP/JoyCaption/NSFW). Stage 6 is PowerShell (clones + sets up the separate trainer venv, downloads Anima models, runs `accelerate launch` headless with fixed-seed sample previews). Each stage splits **pure decision logic** (unit-tested with synthetic inputs) from **GPU model calls** (smoke-tested on real images).

**Tech Stack:** Python 3.12, PyTorch cu128, `transformers` (CLIP ViT-L/14, JoyCaption beta-one LLaVA @ 4-bit nf4, MichalMlodawski FocalNet NSFW), `imagehash`, `opencv-python`, `rapidocr-onnxruntime`, `pyyaml`, `pytest`. Trainer: `gazingstars123/Anima-Standalone-Trainer` (sd-scripts `anima_train_network.py`, `networks.lora_anima`).

**Locked build-time decisions (from spec §12):**
- NSFW safety tag: `MichalMlodawski/nsfw-image-detection-large` (3-class SAFE/QUESTIONABLE/UNSAFE → safe/sensitive/explicit).
- Captioner: `fancyfeast/llama-joycaption-beta-one-hf-llava` loaded **4-bit nf4** via bitsandbytes (bf16 needs ~17GB > 16GB VRAM; 4-bit ≈ 6–8GB).
- Aesthetic: `christophschuhmann/improved-aesthetic-predictor` (`sac+logos+ava1-l14-linearMSE.pth`) on CLIP ViT-L/14 embeddings.
- Stage 6 auto-clones the trainer and runs `setup_env.bat` if absent (idempotent).
- **Trainer repo name corrected:** spec/CLAUDE.md say `gazingstars` — the real repo is **`gazingstars123/Anima-Standalone-Trainer`**.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `config/pipeline.yaml` | Single source of all paths + thresholds. |
| `src/common.py` | Config load, logger, `iter_images`, manifest read/write/augment, `ensure_aesthetic_weights`. |
| `src/01_ingest_clean.py` | Size/blur/corrupt filters, phash near-dup dedup, OCR text-area flag → `data/clean/` + manifest rows. |
| `src/02_quality_score.py` | CLIP+MLP aesthetic score → good/medium/bad bucket. Augments manifest. |
| `src/03_caption.py` | JoyCaption NL + NSFW safety tag + quality tag → assembled caption. Augments manifest. |
| `src/04_build_dataset.py` | Curate (keep good+medium), copy to flat `data/dataset/`, write `img.txt` sidecars, emit dataset TOML. |
| `src/05_make_train_config.py` | Emit training TOML + `sample_prompts.txt`. |
| `scripts/download_models.ps1` | Pull Anima DiT/TE/VAE from HF into `models/`. |
| `scripts/06_train.ps1` | Clone+setup trainer (idempotent), then `accelerate launch` with samples. |
| `tests/conftest.py` | `load_stage()` importlib loader + synthetic-image fixtures. |
| `tests/test_*.py` | One per stage + common; pure-logic asserts + skipped GPU smoke tests. |
| `requirements.txt` | Pipeline venv deps (torch installed separately, documented). |
| `.gitignore` | Ignore `data/`, `outputs/`, `models/`, `trainer/`, venvs. |
| `README.md` | Run order + OOM fallback. |

**Two venvs (never mixed):** pipeline venv (`requirements.txt`, this plan's stages 1–5) vs trainer venv (`setup_env.bat`, PyTorch 2.7 cu128, stage 6). Keep separate to avoid dependency clashes.

**Manifest contract (`data/manifest.csv`):** columns appended stage-by-stage.
- Stage 1 writes: `path, width, height, phash, blur_var, ocr_ratio, dropped, drop_reason`
- Stage 2 augments: `aesthetic_score, bucket`
- Stage 3 augments: `safety_tag, quality_tag, caption`
- Stage 4 reads it, no writes back.

Re-running a stage recomputes only its own columns (idempotent).

---

## Task 1: Scaffolding — config, deps, common module, test harness

**Files:**
- Create: `config/pipeline.yaml`, `requirements.txt`, `.gitignore`, `src/common.py`, `tests/conftest.py`, `tests/test_common.py`, `pytest.ini`

- [ ] **Step 1: Write `config/pipeline.yaml`**

```yaml
# Single source of truth for all paths + thresholds. All paths are project-root-relative.
paths:
  raw: data/raw
  clean: data/clean
  dataset: data/dataset
  outputs: outputs
  manifest: data/manifest.csv
  models_dir: models
  trainer_dir: trainer

ingest:
  min_size: 512                  # drop if min(width,height) < this
  blur_var_threshold: 100.0      # drop if cv2 Laplacian variance < this (blurry)
  phash_hamming_threshold: 6     # <= this Hamming distance => near-duplicate
  ocr_text_area_ratio_flag: 0.10 # > this fraction of area covered by text => flag meme/screenshot

quality:
  clip_model: openai/clip-vit-large-patch14
  aesthetic_weights_url: https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth
  aesthetic_weights_file: models/aesthetic/sac+logos+ava1-l14-linearMSE.pth
  bucket_good_min: 6.0           # score >= => good
  bucket_medium_min: 5.0         # score >= => medium ; else bad

caption:
  joycaption_model: fancyfeast/llama-joycaption-beta-one-hf-llava
  joycaption_prompt: "Write a descriptive caption for this image in a formal tone. Do not comment on image quality, resolution, or aesthetic rating."
  max_new_tokens: 300
  nsfw_model: MichalMlodawski/nsfw-image-detection-large
  nsfw_label_map:                # substring of model label (upper) -> anima safety tag
    SAFE: safe
    QUESTIONABLE: sensitive
    UNSAFE: explicit
  nsfw_default_tag: safe         # fallback if no label substring matches
  trigger: "realistic photo"
  quality_tag_map:
    good: "masterpiece, best quality"
    medium: "high quality"
    bad: "low quality"

dataset:
  buckets_to_keep: [good, medium]  # phase-1 curation; "bad" dropped
  num_repeats: 5
  resolution: 768
  caption_dropout_rate: 0.1

train:
  project_name: anima_realism_v1
  network_dim: 32
  network_alpha: 32
  learning_rate: 1.0e-4
  max_train_epochs: 10
  resolution: 768
  seed: 42

models:                          # Anima training assets pulled by download_models.ps1
  hf_repo: circlestone-labs/Anima
  dit: split_files/diffusion_models/anima-base-v1.0.safetensors
  te:  split_files/text_encoders/qwen_3_06b_base.safetensors
  vae: split_files/vae/qwen_image_vae.safetensors

trainer:
  repo: https://github.com/gazingstars123/Anima-Standalone-Trainer
```

- [ ] **Step 2: Write `requirements.txt`**

```text
# Pipeline venv ONLY (separate from the trainer's setup_env.bat venv).
# Install torch FIRST, separately, with CUDA 12.8 wheels:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Then: pip install -r requirements.txt
pillow>=10.0
numpy>=1.26
pyyaml>=6.0
imagehash>=4.3
opencv-python>=4.9
rapidocr-onnxruntime>=1.3
transformers>=4.45
accelerate>=0.34
bitsandbytes>=0.44       # 4-bit JoyCaption; Windows wheels supported
sentencepiece>=0.2
huggingface_hub>=0.25
tqdm>=4.66
pytest>=8.0
tomli>=2.0 ; python_version < "3.11"   # tomllib backport (system Python is 3.10)
```

- [ ] **Step 3: Write `.gitignore`**

```text
data/
outputs/
models/
trainer/
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    gpu: integration test that loads a real model and needs CUDA
```

- [ ] **Step 5: Write the failing test `tests/test_common.py`**

```python
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
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so stages can `import common` and tests can `import common`


def load_stage(filename):
    """Import a numbered stage module (e.g. '01_ingest_clean.py') that cannot be imported normally."""
    path = SRC / filename
    name = filename[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def make_image(tmp_path):
    """Factory: write a synthetic JPEG/PNG and return its Path."""
    def _make(name="img.jpg", size=(800, 600), color=(120, 80, 40), noise=False):
        w, h = size
        if noise:
            arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
            img = Image.fromarray(arr)
        else:
            img = Image.new("RGB", size, color)
        p = tmp_path / name
        img.save(p)
        return p
    return _make
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `python -m pytest tests/test_common.py -v`
Expected: FAIL — `module 'common' has no attribute 'load_config'` (file not yet written).

- [ ] **Step 8: Write `src/common.py`**

```python
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


def load_config(path="config/pipeline.yaml"):
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
    for r in rows:
        upd = updates_by_path.get(r["path"])
        if upd:
            r.update({k: str(v) for k, v in upd.items()})
    write_manifest(path, rows)


def ensure_aesthetic_weights(cfg):
    """Download the aesthetic MLP .pth once into models/aesthetic/ if absent. Returns the local Path."""
    from huggingface_hub import hf_hub_download  # noqa: F401  (kept import local; not used here)
    import urllib.request

    dest = Path(cfg["quality"]["aesthetic_weights_file"])
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["quality"]["aesthetic_weights_url"]
    LOG.info("Downloading aesthetic weights: %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `python -m pytest tests/test_common.py -v`
Expected: PASS (2 passed).

- [ ] **Step 10: Commit**

```bash
git add config/pipeline.yaml requirements.txt .gitignore pytest.ini src/common.py tests/conftest.py tests/test_common.py
git commit -m "feat: scaffold pipeline config, common module, test harness"
```

---

## Task 2: Stage 1 — ingest & clean

**Files:**
- Create: `src/01_ingest_clean.py`, `tests/test_01_ingest_clean.py`

- [ ] **Step 1: Write the failing test `tests/test_01_ingest_clean.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_01_ingest_clean.py -v`
Expected: FAIL — `No module named '01_ingest_clean'` content / `AttributeError`.

- [ ] **Step 3: Write `src/01_ingest_clean.py`**

```python
"""Stage 1: dedup, drop tiny/blurry/corrupt, OCR-flag meme/screenshot text -> data/clean/ + manifest."""
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

import common

LOG = common.setup_logging()


def is_corrupt(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def is_too_small(path, min_size):
    w, h = image_size(path)
    return min(w, h) < min_size


def blur_variance(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def hamming(h1, h2):
    return h1 - h2


def dedup(paths, hamming_threshold):
    """Greedy near-dup grouping; within a group keep the highest-resolution image."""
    hashes = {p: phash(p) for p in paths}
    keep, drop, used = [], [], set()
    for p in paths:
        if p in used:
            continue
        group = [p]
        for q in paths:
            if q is p or q in used:
                continue
            if hamming(hashes[p], hashes[q]) <= hamming_threshold:
                group.append(q)
        for g in group:
            used.add(g)
        best = max(group, key=lambda x: image_size(x)[0] * image_size(x)[1])
        keep.append(best)
        drop.extend([g for g in group if g is not best])
    return keep, drop


def ocr_text_area_ratio(path, engine):
    """Fraction of image area covered by detected text boxes (0..1)."""
    result, _ = engine(str(path))
    if not result:
        return 0.0
    with Image.open(path) as im:
        area = im.size[0] * im.size[1]
    text_area = 0.0
    for box, _text, _conf in result:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        text_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return min(text_area / area, 1.0)


def main():
    cfg = common.load_config()
    ing = cfg["ingest"]
    raw = Path(cfg["paths"]["raw"])
    clean = Path(cfg["paths"]["clean"])
    clean.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    all_imgs = list(common.iter_images(raw))
    LOG.info("Stage 1: %d raw images", len(all_imgs))

    survivors, rows = [], []
    for p in all_imgs:
        reason = ""
        if is_corrupt(p):
            reason = "corrupt"
        elif is_too_small(p, ing["min_size"]):
            reason = "too_small"
        elif blur_variance(p) < ing["blur_var_threshold"]:
            reason = "blurry"
        if reason:
            rows.append({"path": str(p), "dropped": "True", "drop_reason": reason})
        else:
            survivors.append(p)

    keep, dup_drop = dedup(survivors, ing["phash_hamming_threshold"])
    for p in dup_drop:
        rows.append({"path": str(p), "dropped": "True", "drop_reason": "duplicate"})

    for p in keep:
        w, h = image_size(p)
        ratio = ocr_text_area_ratio(p, ocr)
        flagged = ratio > ing["ocr_text_area_ratio_flag"]
        dest = clean / p.name
        shutil.copy2(p, dest)
        rows.append({
            "path": str(dest), "width": str(w), "height": str(h),
            "phash": str(phash(p)), "blur_var": f"{blur_variance(p):.1f}",
            "ocr_ratio": f"{ratio:.3f}", "dropped": "False",
            "drop_reason": "text_overlay_flag" if flagged else "",
        })

    common.write_manifest(cfg["paths"]["manifest"], rows)
    LOG.info("Stage 1 done: kept %d, dropped %d -> %s", len(keep), len(rows) - len(keep), clean)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_01_ingest_clean.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/01_ingest_clean.py tests/test_01_ingest_clean.py
git commit -m "feat: stage 1 ingest/clean (dedup, blur/size/corrupt filter, OCR flag)"
```

---

## Task 3: Stage 2 — aesthetic quality scoring

**Files:**
- Create: `src/02_quality_score.py`, `tests/test_02_quality_score.py`

- [ ] **Step 1: Write the failing test `tests/test_02_quality_score.py`**

```python
from conftest import load_stage

stage = load_stage("02_quality_score.py")


def test_score_to_bucket_thresholds():
    assert stage.score_to_bucket(7.2, good_min=6.0, medium_min=5.0) == "good"
    assert stage.score_to_bucket(6.0, good_min=6.0, medium_min=5.0) == "good"
    assert stage.score_to_bucket(5.4, good_min=6.0, medium_min=5.0) == "medium"
    assert stage.score_to_bucket(5.0, good_min=6.0, medium_min=5.0) == "medium"
    assert stage.score_to_bucket(3.1, good_min=6.0, medium_min=5.0) == "bad"


def test_mlp_shape():
    import torch
    mlp = stage.AestheticMLP(768)
    out = mlp(torch.zeros(1, 768))
    assert out.shape == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_02_quality_score.py -v`
Expected: FAIL — `AttributeError: ... 'score_to_bucket'`.

- [ ] **Step 3: Write `src/02_quality_score.py`**

```python
"""Stage 2: CLIP+MLP aesthetic score -> good/medium/bad bucket. Augments manifest."""
import torch
import torch.nn as nn
from PIL import Image

import common

LOG = common.setup_logging()


class AestheticMLP(nn.Module):
    """Architecture from christophschuhmann/improved-aesthetic-predictor (CLIP ViT-L/14, 768-dim)."""
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def score_to_bucket(score, good_min, medium_min):
    if score >= good_min:
        return "good"
    if score >= medium_min:
        return "medium"
    return "bad"


class AestheticScorer:
    def __init__(self, cfg, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        name = cfg["quality"]["clip_model"]
        self.clip = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.mlp = AestheticMLP(768).to(device).eval()
        weights = common.ensure_aesthetic_weights(cfg)
        # weights_only=True: .pth is a downloaded state_dict (tensors only); blocks arbitrary-code unpickling.
        self.mlp.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    @torch.no_grad()
    def score(self, path):
        img = Image.open(path).convert("RGB")
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        feats = self.clip.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # L2-normalize (predictor was trained on normalized CLIP feats)
        return float(self.mlp(feats).item())


def main():
    cfg = common.load_config()
    q = cfg["quality"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 2: scoring %d kept images", len(kept))

    scorer = AestheticScorer(cfg)
    updates = {}
    for r in kept:
        s = scorer.score(r["path"])
        updates[r["path"]] = {"aesthetic_score": f"{s:.3f}", "bucket": score_to_bucket(s, q["bucket_good_min"], q["bucket_medium_min"])}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 2 done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_02_quality_score.py -v`
Expected: PASS (2 passed). (`test_mlp_shape` needs torch CPU only — no CUDA required.)

- [ ] **Step 5: Commit**

```bash
git add src/02_quality_score.py tests/test_02_quality_score.py
git commit -m "feat: stage 2 aesthetic scoring (CLIP+MLP -> good/medium/bad buckets)"
```

---

## Task 4: Stage 3 — captioning (JoyCaption + NSFW + quality tag)

**Files:**
- Create: `src/03_caption.py`, `tests/test_03_caption.py`

- [ ] **Step 1: Write the failing test `tests/test_03_caption.py`**

```python
from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_assemble_caption_order_and_commas():
    out = stage.assemble_caption(
        quality_tag="masterpiece, best quality",
        safety_tag="safe",
        trigger="realistic photo",
        nl="a woman on a park bench at golden hour, 35mm",
    )
    assert out == "masterpiece, best quality, safe, realistic photo, a woman on a park bench at golden hour, 35mm"


def test_quality_tag_from_bucket():
    qmap = {"good": "masterpiece, best quality", "medium": "high quality", "bad": "low quality"}
    assert stage.quality_tag_for("good", qmap) == "masterpiece, best quality"
    assert stage.quality_tag_for("bad", qmap) == "low quality"


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"  # fallback


def test_nl_cleanup_strips_newlines_and_trailing_period():
    assert stage.clean_nl("  A photo of a cat.\nSecond line.  ") == "A photo of a cat. Second line"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Write `src/03_caption.py`**

```python
"""Stage 3: JoyCaption NL (4-bit) + NSFW safety tag + quality tag -> assembled caption. Augments manifest."""
import re

import torch
from PIL import Image

import common

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def quality_tag_for(bucket, quality_tag_map):
    return quality_tag_map[bucket]


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key, tag in label_map.items():
        if key.upper() in up:
            return tag
    return default_tag


def clean_nl(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".").strip()


def assemble_caption(quality_tag, safety_tag, trigger, nl):
    return f"{quality_tag}, {safety_tag}, {trigger}, {nl}"


# ---- model wrappers (smoke-tested) ----

class JoyCaptioner:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        name = cfg["caption"]["joycaption_model"]
        self.prompt = cfg["caption"]["joycaption_prompt"]
        self.max_new_tokens = cfg["caption"]["max_new_tokens"]
        self.device = device
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            name, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map=device,
        ).eval()

    @torch.no_grad()
    def caption(self, path):
        image = Image.open(path).convert("RGB")
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": self.prompt},
        ]
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)[0]
        ids = ids[inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return clean_nl(text)


class NSFWTagger:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        name = cfg["caption"]["nsfw_model"]
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
        self.label_map = cfg["caption"]["nsfw_label_map"]
        self.default = cfg["caption"]["nsfw_default_tag"]

    @torch.no_grad()
    def tag(self, path):
        image = Image.open(path).convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        model_label = self.model.config.id2label[idx]
        return map_safety(model_label, self.label_map, self.default)


def main():
    cfg = common.load_config()
    cap_cfg = cfg["caption"]
    rows = common.read_manifest(cfg["paths"]["manifest"])
    kept = [r for r in rows if r.get("dropped") == "False"]
    LOG.info("Stage 3: captioning %d images", len(kept))

    joy = JoyCaptioner(cfg)
    nsfw = NSFWTagger(cfg)
    updates = {}
    for r in kept:
        qtag = quality_tag_for(r["bucket"], cap_cfg["quality_tag_map"])
        stag = nsfw.tag(r["path"])
        nl = joy.caption(r["path"])
        caption = assemble_caption(qtag, stag, cap_cfg["trigger"], nl)
        updates[r["path"]] = {"safety_tag": stag, "quality_tag": qtag, "caption": caption}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: One-time smoke check of NSFW label names (run after deps installed)**

Run:
```bash
python -c "from transformers import AutoModelForImageClassification as M; print(M.from_pretrained('MichalMlodawski/nsfw-image-detection-large').config.id2label)"
```
Expected: a dict like `{0: 'SAFE', 1: 'QUESTIONABLE', 2: 'UNSAFE'}`. If the exact strings differ, update `caption.nsfw_label_map` keys in `pipeline.yaml` so each maps by substring. No code change needed (matching is substring + case-insensitive).

- [ ] **Step 6: Commit**

```bash
git add src/03_caption.py tests/test_03_caption.py
git commit -m "feat: stage 3 captioning (JoyCaption 4bit + 3-class NSFW + quality tag)"
```

---

## Task 5: Stage 4 — build dataset (curate, sidecars, dataset TOML)

**Files:**
- Create: `src/04_build_dataset.py`, `tests/test_04_build_dataset.py`

- [ ] **Step 1: Write the failing test `tests/test_04_build_dataset.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Write `src/04_build_dataset.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: PASS (3 passed). (`tomllib` is stdlib in Python 3.11+.)

- [ ] **Step 5: Commit**

```bash
git add src/04_build_dataset.py tests/test_04_build_dataset.py
git commit -m "feat: stage 4 build dataset (curate, sidecars, dataset TOML)"
```

---

## Task 6: Stage 5 — emit training TOML + sample prompts

**Files:**
- Create: `src/05_make_train_config.py`, `tests/test_05_make_train_config.py`

- [ ] **Step 1: Write the failing test `tests/test_05_make_train_config.py`**

```python
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport
from pathlib import Path

from conftest import load_stage

stage = load_stage("05_make_train_config.py")


def test_training_toml_valid_and_key_fields(tmp_path):
    out = tmp_path / "train.toml"
    stage.write_training_toml(
        out,
        dit="models/anima-base-v1.0.safetensors",
        te="models/qwen_3_06b_base.safetensors",
        vae="models/qwen_image_vae.safetensors",
        output_dir="outputs",
        output_name="anima_realism_v1",
        sample_prompts="outputs/sample_prompts.txt",
        dim=32, alpha=32, lr=1e-4, epochs=10, seed=42,
    )
    d = tomllib.loads(out.read_text(encoding="utf-8"))
    assert d["network_module"] == "networks.lora_anima"
    assert d["network_train_unet_only"] is True   # freeze Qwen3 TE
    assert d["cache_latents"] is True
    assert d["cache_text_encoder_outputs"] is True
    assert d["network_dim"] == 32
    assert d["learning_rate"] == 1e-4
    assert d["mixed_precision"] == "bf16"
    assert d["sample_every_n_epochs"] == 1


def test_sample_prompts_demonstrate_quality_and_safety_steering():
    prompts = stage.build_sample_prompts(trigger="realistic photo", seed=42)
    text = "\n".join(prompts)
    assert "realistic photo" in text
    assert "masterpiece, best quality" in text
    assert "low quality" in text       # success criterion #4: quality axis
    assert "--seed 42" in text          # fixed seed for comparable previews
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_05_make_train_config.py -v`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Write `src/05_make_train_config.py`**

```python
"""Stage 5: emit training TOML (sd-scripts/anima schema) + fixed-seed sample_prompts.txt."""
from pathlib import Path

import common

LOG = common.setup_logging()


def build_sample_prompts(trigger, seed):
    """Fixed-seed previews proving the trigger + quality/safety steering (success criteria #3,#4)."""
    common_tail = f"--w 768 --h 768 --d {seed} --s 24 --l 5.0"
    return [
        f"masterpiece, best quality, safe, {trigger}, a woman sitting on a park bench at golden hour, 35mm {common_tail}",
        f"masterpiece, best quality, safe, {trigger}, a city street in the rain at night, neon reflections {common_tail}",
        f"low quality, safe, {trigger}, a woman sitting on a park bench at golden hour, 35mm {common_tail}",
        f"high quality, safe, {trigger}, portrait of an old man, natural window light {common_tail}",
    ]


def write_training_toml(out_path, dit, te, vae, output_dir, output_name, sample_prompts,
                        dim, alpha, lr, epochs, seed):
    fwd = lambda p: str(p).replace("\\", "/")
    toml = f"""pretrained_model_name_or_path = "{fwd(dit)}"
qwen3 = "{fwd(te)}"
vae = "{fwd(vae)}"
network_module = "networks.lora_anima"
network_dim = {dim}
network_alpha = {alpha}
network_train_unet_only = true
learning_rate = {lr}
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1", "betas=[0.9, 0.99]"]
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 1
lr_warmup_steps = 100
max_train_epochs = {epochs}
train_batch_size = 1
gradient_accumulation_steps = 1
max_grad_norm = 1.0
seed = {seed}
timestep_sampling = "sigmoid"
discrete_flow_shift = 1.0
qwen3_max_token_length = 512
t5_max_token_length = 512
mixed_precision = "bf16"
gradient_checkpointing = true
cache_latents = true
cache_text_encoder_outputs = true
vae_chunk_size = 64
vae_disable_cache = true
output_dir = "{fwd(output_dir)}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "bf16"
save_every_n_epochs = 1
save_last_n_epochs = 4
shuffle_caption = false
caption_extension = ".txt"
noise_offset = 0.03
multires_noise_discount = 0.3
sample_prompts = "{fwd(sample_prompts)}"
sample_every_n_epochs = 1
sample_at_first = true
sample_sampler = "euler_a"
"""
    Path(out_path).write_text(toml, encoding="utf-8")


def main():
    cfg = common.load_config()
    t = cfg["train"]
    m = cfg["models"]
    models_dir = Path(cfg["paths"]["models_dir"])
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)

    sample_path = out / "sample_prompts.txt"
    sample_path.write_text("\n".join(build_sample_prompts(cfg["caption"]["trigger"], t["seed"])) + "\n", encoding="utf-8")

    toml_path = out / f"{t['project_name']}_training_config.toml"
    write_training_toml(
        toml_path,
        dit=models_dir / Path(m["dit"]).name,
        te=models_dir / Path(m["te"]).name,
        vae=models_dir / Path(m["vae"]).name,
        output_dir=cfg["paths"]["outputs"],
        output_name=t["project_name"],
        sample_prompts=sample_path,
        dim=t["network_dim"], alpha=t["network_alpha"], lr=t["learning_rate"],
        epochs=t["max_train_epochs"], seed=t["seed"],
    )
    LOG.info("Stage 5 done. Training TOML -> %s ; samples -> %s", toml_path, sample_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_05_make_train_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/05_make_train_config.py tests/test_05_make_train_config.py
git commit -m "feat: stage 5 emit training TOML + fixed-seed sample prompts"
```

---

## Task 7: Model download script (Anima DiT/TE/VAE)

**Files:**
- Create: `scripts/download_models.ps1`

- [ ] **Step 1: Write `scripts/download_models.ps1`**

```powershell
# Downloads the 3 Anima training assets into models/ (idempotent: skips existing files).
# Reads repo + relative paths from config/pipeline.yaml's `models:` block (kept inline here for a no-dep script).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$repo = "circlestone-labs/Anima"
$prefix = "https://huggingface.co/$repo/resolve/main"
$files = @(
    "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "split_files/vae/qwen_image_vae.safetensors"
)

foreach ($f in $files) {
    $name = Split-Path $f -Leaf
    $dest = Join-Path $modelsDir $name
    if (Test-Path $dest) {
        Write-Host "exists, skip: $name"
        continue
    }
    $url = "$prefix/$f"
    Write-Host "downloading: $name"
    # -L follows redirects to the HF CDN; resumes via -C - if interrupted.
    curl.exe -L -C - -o $dest $url
}
Write-Host "Models ready in $modelsDir"
```

- [ ] **Step 2: Verify the script parses (no download)**

Run: `powershell -NoProfile -Command "& { . .\scripts\download_models.ps1 -WhatIf } 2>&1 | Select-Object -First 1"` — if `-WhatIf` is unsupported, instead run a syntax-only check:
Run: `powershell -NoProfile -Command "[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\scripts\download_models.ps1), [ref]$null, [ref]$null); 'OK'"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_models.ps1
git commit -m "feat: Anima model download script (idempotent, resumable)"
```

---

## Task 8: Stage 6 — train launcher (clone+setup trainer, accelerate launch)

**Files:**
- Create: `scripts/06_train.ps1`

- [ ] **Step 1: Write `scripts/06_train.ps1`**

```powershell
# Stage 6: idempotently provision the trainer, then run headless LoRA training with sample previews.
# Prereq: run stages 1-5 (Python) and scripts/download_models.ps1 first.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$trainerDir = Join-Path $root "trainer"
$project = "anima_realism_v1"
$outputs = Join-Path $root "outputs"
$trainToml = Join-Path $outputs "${project}_training_config.toml"
$dataToml  = Join-Path $outputs "${project}_dataset_config.toml"

foreach ($t in @($trainToml, $dataToml)) {
    if (-not (Test-Path $t)) { throw "Missing $t  - run stages 4 & 5 first." }
}

# 1. Clone the trainer if absent.
if (-not (Test-Path $trainerDir)) {
    Write-Host "Cloning Anima-Standalone-Trainer..."
    git clone https://github.com/gazingstars123/Anima-Standalone-Trainer $trainerDir
}

# 2. Run setup_env.bat once (marker file => idempotent; installs PyTorch 2.7 cu128 venv).
$marker = Join-Path $trainerDir ".env_ready"
if (-not (Test-Path $marker)) {
    Write-Host "Running setup_env.bat (first-time, ~10-15 min)..."
    Push-Location $trainerDir
    cmd /c setup_env.bat
    Pop-Location
    New-Item -ItemType File -Path $marker | Out-Null
}

# 3. Confirm the Anima LoRA network module exists.
if (-not (Test-Path (Join-Path $trainerDir "networks\lora_anima.py"))) {
    throw "networks\lora_anima.py not found in trainer checkout - aborting."
}

# 4. Launch training headless. Use the trainer venv's python via accelerate.
Push-Location $trainerDir
$venvActivate = Join-Path $trainerDir "venv\Scripts\activate.ps1"
if (Test-Path $venvActivate) { . $venvActivate }
accelerate launch --num_cpu_threads_per_process 1 `
  "$trainerDir\anima_train_network.py" `
  --config_file "$trainToml" `
  --dataset_config "$dataToml"
Pop-Location

Write-Host "Training launched. LoRA + sample previews -> $outputs"
```

- [ ] **Step 2: Verify the script parses (no run)**

Run: `powershell -NoProfile -Command "[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path .\scripts\06_train.ps1), [ref]$null, [ref]$null); 'OK'"`
Expected: `OK`.

- [ ] **Step 3: Confirm the trainer's venv path after first real setup**

After a real `setup_env.bat` run, confirm the venv activate path the script assumes (`trainer\venv\Scripts\activate.ps1`). If `setup_env.bat` names the venv differently (e.g. `.venv` or `env`), update the `$venvActivate` line. This is the one path the upstream README does not pin — verify against the actual checkout, do not guess.

- [ ] **Step 4: Commit**

```bash
git add scripts/06_train.ps1
git commit -m "feat: stage 6 train launcher (idempotent trainer setup + accelerate launch)"
```

---

## Task 9: README + full test sweep

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Anima Realism LoRA — Phase 1 Pipeline

Turns `data/raw/` photos into a realism domain-shift LoRA for the Anima diffusion model.
Local, Windows, RTX 4080 (16GB). Full design: `docs/superpowers/specs/2026-05-31-anima-realism-lora-design.md`.

## Setup (two separate venvs)

**Pipeline venv** (stages 1-5):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
The trainer venv is created automatically by stage 6 (`setup_env.bat`) — do not mix the two.

## Run order
```powershell
# Pipeline venv active:
python src/01_ingest_clean.py
python src/02_quality_score.py
python src/03_caption.py
python src/04_build_dataset.py
python src/05_make_train_config.py
# Models + train (uses curl + the trainer venv it provisions):
.\scripts\download_models.ps1
.\scripts\06_train.ps1
```
Every stage is idempotent and reads the previous stage's output. Config lives in `config/pipeline.yaml`.

## OOM fallback
If training OOMs at the documented settings, edit `config/pipeline.yaml` `train:` → `network_dim: 8`,
`network_alpha: 8`, `resolution: 512`, then re-run `python src/05_make_train_config.py` and `.\scripts\06_train.ps1`.
(Cached latents pin resolution — changing res forces a re-cache, which the trainer does automatically.)

## Tests
```powershell
python -m pytest -v          # pure-logic suite, no GPU needed
```
````

- [ ] **Step 2: Run the full pure-logic test suite**

Run: `python -m pytest -v`
Expected: all tests PASS (common + stages 1–5). GPU model wrappers are exercised by the live run, not unit tests.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README run order + OOM fallback"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §2 cleaning/dedup → Task 2; §2 NSFW safety-tag-not-filter → Task 4 (`map_safety`, kept in manifest, never dropped on NSFW); §2 curation good+medium → Task 5 (`curate`); §5 caption format → Task 4 (`assemble_caption`, order asserted); §6 all 6 stages → Tasks 2–8; §7 training+dataset TOML → Tasks 5,6 (schema reproduced + `tomllib`-validated); §8 16GB recipe → Task 6 TOML asserts `cache_latents`/`cache_text_encoder_outputs`/bf16; §8 OOM fallback → README; §10 success #3/#4 sample steering → Task 6 `build_sample_prompts`; §12 open items → all resolved (locked decisions in header).
- **Placeholder scan:** no TBD/"handle edge cases"/"similar to". All code complete.
- **Type consistency:** manifest columns flow stage1→stage4 consistently (`dropped`, `bucket`, `caption`); `assemble_caption`/`score_to_bucket`/`curate`/`write_dataset_toml`/`write_training_toml` signatures match their call sites and tests.
- **Known build-time verifications flagged inline (cannot be pre-asserted):** NSFW `id2label` exact strings (Task 4 Step 5), trainer venv activate path (Task 8 Step 3).
````
