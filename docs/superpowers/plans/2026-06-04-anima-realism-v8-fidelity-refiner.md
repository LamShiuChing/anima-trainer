# Anima Realism v8 — Fidelity Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an all-clean ~600–900 image dataset (60% detail close-ups / 35% candid anchor / 5% backgrounds) and the train/eval configs for a full-finetune fidelity-refiner pass warm-started from `V7_epoch17.safetensors`, to erase the learned compression look and sharpen fingers/clothes/phones/toes/backgrounds while keeping the amateur aesthetic.

**Architecture:** Reuse the existing v5/v7 local-prep pipeline (`src/`, `config/pipeline.yaml`, `scripts/`). New work is upstream **sourcing + curation** (external clean images, AR-crop, 100%-zoom sharpness gate, bucket ratio) feeding the existing captioning (stage 3) → build (stage 4) → Vast train flow. Compression-erase comes from warm-starting ep17 and training only on clean data; amateur preservation comes from honest split-axis captioning + the candid anchor.

**Tech Stack:** Python 3.10 (global env, `USE_TF=0`), FiftyOne (Open Images), PIL/OpenCV, imagehash, WD14 EVA02 + Gemini (existing), diffusion-pipe on Vast (RTX 6000/A100), deepspeed.

**Spec:** `docs/superpowers/specs/2026-06-04-anima-realism-v8-fidelity-refiner-design.md`

---

## File structure

| Path | Responsibility | New/Modify |
|---|---|---|
| `data/v8_raw/{detail,anchor,bg}/` + `README.md` | Drop zone for sourced images, bucketed | Created (Task 0 ✅) |
| `scripts/v8_fetch_openimages.py` | FiftyOne Open-Images puller → bbox-crop detail / full anchor+bg, size-gated, bucket-sorted | New (Task 1) |
| `src/v8_curate.py` | Unify `data/v8_raw/*` → ≥1536 + sharpness + dedup + AR-crop 0.66–1.5 → `data/v8_clean/` + `data/v8_manifest.csv` (bucket column) + ratio report | New (Task 2) |
| `tests/test_v8_curate.py` | Unit tests for AR-crop, size gate, ratio report | New (Task 2) |
| `config/pipeline.yaml` | Add `v8` overrides (paths + finetune block) | Modify (Task 3) |
| `outputs/anima_realism_ft_v8_dataset_config.toml` | Emitted by stage 4 | Generated (Task 4) |
| `outputs/anima_realism_ft_v8_train_config.toml` | Train toml (clone v7 + diffs) | New (Task 5) |
| `scripts/run_v8_train.sh`, `scripts/vast_fetch_v8.sh` | Vast launch + fetch (clone v7) | New (Task 6) |
| `docs/superpowers/specs/2026-06-04-v8-eval-prompts.md` | Frozen eval prompts + seeds + CFG | New (Task 7) |

**Reused as-is:** `src/01_ingest_clean.py` (helpers imported by curate), `src/03_caption.py` + `src/gemini_caption.py` (captioning), `src/04_build_dataset.py` (build), `scripts/vast_setup.sh`.

---

## Task 0: Sourcing folders ✅ DONE

`data/v8_raw/{detail,anchor,bg}/` + `README.md` created. User drops curated Unsplash/Pixabay/Pexels images into the matching bucket. Rules: 100%-zoom sharp, ≥1536 short side, all-clean, amateur (anchor), feet curated hard.

---

## Task 1: FiftyOne Open-Images sourcing script

**Files:**
- Create: `scripts/v8_fetch_openimages.py`

Pulls Open Images V7 by target classes, bbox-crops detail classes (only when the crop is ≥ the short-side floor), keeps full images for anchor/bg when ≥ floor, sorts into `data/v8_raw/{detail,anchor,bg}/`. **Honest limitation baked into logs:** Open Images source images skew ≤1024 px, so few bbox crops clear 1536 — this script is for *variety / anchor / bg*; the high-res detail fuel comes from manual Unsplash/Pixabay.

- [ ] **Step 1: Write the script** (full code in the repo file; key interface below)

```python
# scripts/v8_fetch_openimages.py  — run: python scripts/v8_fetch_openimages.py --max-samples 2000
DETAIL_CLASSES = {  # Open Images detection label -> our detail subtype
    "Human hand": "hand", "Human foot": "foot", "Footwear": "foot",
    "Sandal": "foot", "High heels": "foot", "Mobile phone": "phone",
    "Telephone": "phone", "Jeans": "fabric", "Dress": "fabric",
    "Shirt": "fabric", "Suit": "fabric", "Trousers": "fabric",
    "Sweater": "fabric", "Jacket": "fabric",
}
ANCHOR_CLASSES = ["Person", "Woman", "Man", "Girl", "Boy"]
BG_CLASSES = ["Couch", "Bed", "Kitchen & dining room table", "Houseplant",
              "Coffee table", "Curtain", "Bookcase", "Stairs", "Fireplace"]
# load_zoo_dataset("open-images-v7", split, label_types=["detections"],
#   classes=ALL, max_samples, only_matching=True) -> compute_metadata()
# per sample: detail = bbox-crop (pad 0.15) saved IFF min(crop_w,crop_h) >= min_short;
#   anchor = full image IFF a person-class present and min(w,h) >= min_short;
#   bg = full image IFF NO person-class present, a bg-class present, min(w,h) >= min_short.
# Summary prints per-bucket counts + the ">=1536 detail crops are rare" caveat.
```

- [ ] **Step 2: Verify it imports / errors helpfully without FiftyOne**

Run: `python -c "import scripts.v8_fetch_openimages"` (after `pip install fiftyone`)
Expected: no import error; running without fiftyone prints `pip install fiftyone` hint and exits 1.

- [ ] **Step 3: Smoke run, small sample**

Run: `python scripts/v8_fetch_openimages.py --max-samples 50 --min-short 1536`
Expected: downloads a small OI subset, prints bucket counts, writes any qualifying images to `data/v8_raw/{detail,anchor,bg}/`. Detail count may be 0 (expected — see caveat).

- [ ] **Step 4: Commit**

```bash
git add scripts/v8_fetch_openimages.py
git commit -m "feat(v8): FiftyOne Open-Images sourcing -> bucketed v8_raw"
```

---

## Task 2: v8 curation script (the real gate)

**Files:**
- Create: `src/v8_curate.py`
- Test: `tests/test_v8_curate.py`

Reads `data/v8_raw/{detail,anchor,bg}/`, applies: (1) ≥1536 short side, (2) Laplacian sharpness ≥ threshold, (3) phash dedup, (4) AR-crop to 0.66–1.5 (center-crop the out-of-range ones), writes flat `data/v8_clean/` + `data/v8_manifest.csv` carrying a `bucket` column + width/height/blur_var/phash (so stage 4 reuses them), and prints the actual 60/35/5 split with a warning if off. Reuses `src/01_ingest_clean.py` helpers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v8_curate.py
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("v8_curate", pathlib.Path("src/v8_curate.py"))
v8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)

def test_ar_crop_wide_to_max():
    # 2000x1000 (AR 2.0) -> crop to AR 1.5 -> 1500x1000, centered
    box = v8.ar_crop_box(2000, 1000, min_ar=0.66, max_ar=1.5)
    assert box == (250, 0, 1750, 1000)  # left,top,right,bottom

def test_ar_crop_in_range_noop():
    assert v8.ar_crop_box(1600, 1600, 0.66, 1.5) == (0, 0, 1600, 1600)

def test_ar_crop_tall_to_min():
    # 1000x2000 (AR 0.5) -> crop to AR 0.66 -> width 1000, height 1000/0.66=1515
    box = v8.ar_crop_box(1000, 2000, 0.66, 1.5)
    assert box == (0, 242, 1000, 1757)

def test_ratio_report_warns_when_off():
    counts = {"detail": 10, "anchor": 80, "bg": 10}  # detail way under 60%
    ok, msg = v8.ratio_ok(counts, target={"detail": .60, "anchor": .35, "bg": .05}, tol=.10)
    assert ok is False and "detail" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v8_curate.py -v`
Expected: FAIL (module/functions not defined)

- [ ] **Step 3: Implement `src/v8_curate.py`**

```python
"""v8 curation: data/v8_raw/{detail,anchor,bg} -> data/v8_clean + v8_manifest.csv.
Gates: >=1536 short side, Laplacian sharpness, phash dedup, AR-crop to 0.66-1.5.
Reuses stage-1 helpers (blur_variance, phash, image_size, dedup)."""
import os, shutil, importlib.util
from pathlib import Path
from PIL import Image

# import stage-1 helpers (filename starts with a digit -> import by path)
_s1 = importlib.util.spec_from_file_location("ingest1", Path(__file__).with_name("01_ingest_clean.py"))
ingest1 = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(ingest1)

RAW = Path("data/v8_raw"); CLEAN = Path("data/v8_clean"); MANIFEST = "data/v8_manifest.csv"
BUCKETS = ["detail", "anchor", "bg"]
MIN_SHORT = 1536
BLUR_MIN = 100.0
HAMMING = 8
TARGET = {"detail": 0.60, "anchor": 0.35, "bg": 0.05}

def ar_crop_box(w, h, min_ar, max_ar):
    """Return (left,top,right,bottom) center-crop so that w/h in [min_ar,max_ar]. No-op if already in range."""
    ar = w / h
    if min_ar <= ar <= max_ar:
        return (0, 0, w, h)
    if ar > max_ar:                      # too wide -> trim width
        new_w = round(max_ar * h)
        off = (w - new_w) // 2
        return (off, 0, off + new_w, h)
    new_h = round(w / min_ar)            # too tall -> trim height
    off = (h - new_h) // 2
    return (0, off, w, off + new_h)

def ratio_ok(counts, target=TARGET, tol=0.10):
    total = sum(counts.values()) or 1
    for k, frac in target.items():
        actual = counts.get(k, 0) / total
        if abs(actual - frac) > tol:
            return False, f"bucket '{k}' at {actual:.0%} (target {frac:.0%})"
    return True, "ratios within tolerance"

def main():
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {b: 0 for b in BUCKETS}
    for bucket in BUCKETS:
        src = RAW / bucket
        if not src.is_dir():
            continue
        imgs = list(ingest1.common.iter_images(src))
        # size + sharpness gate
        gated = []
        for p in imgs:
            if ingest1.is_corrupt(p):
                continue
            w, h = ingest1.image_size(p)
            if min(w, h) < MIN_SHORT:
                continue
            if ingest1.blur_variance(p) < BLUR_MIN:
                continue
            gated.append(p)
        keep, _ = ingest1.dedup(gated, HAMMING)
        for p in keep:
            w, h = ingest1.image_size(p)
            box = ar_crop_box(w, h, 0.66, 1.5)
            dest = CLEAN / f"{bucket}_{p.stem}.jpg"
            im = Image.open(p).convert("RGB")
            if box != (0, 0, w, h):
                im = im.crop(box)
            im.save(dest, quality=95)
            cw, ch = im.size
            rows.append({"path": str(dest), "bucket": bucket,
                         "width": str(cw), "height": str(ch),
                         "phash": str(ingest1.phash(p)),
                         "blur_var": f"{ingest1.blur_variance(p):.1f}",
                         "dropped": "False", "drop_reason": ""})
            counts[bucket] += 1
    ingest1.common.write_manifest(MANIFEST, rows)
    ok, msg = ratio_ok(counts)
    print(f"v8 curate: kept {sum(counts.values())} {counts} -> {CLEAN}")
    print(f"ratio check: {'OK' if ok else 'WARN'} - {msg}")

if __name__ == "__main__":
    os.environ["USE_TF"] = "0"
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v8_curate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/v8_curate.py tests/test_v8_curate.py
git commit -m "feat(v8): curation - size/sharpness/dedup/AR-crop + ratio report"
```

---

## Task 3: Wire config for v8 (captioning + build reuse existing stages)

**Files:**
- Modify: `config/pipeline.yaml`

The existing stage 3 (caption) and stage 4 (build) read `paths.clean` / `paths.dataset` / `paths.manifest` and `finetune.project_name`. Point them at v8 for the run. Captions land in `data/v8_manifest.csv` (stage 3 appends a `caption` column).

- [ ] **Step 1: Edit `config/pipeline.yaml`** — set for the v8 run:

```yaml
paths:
  clean: data/v8_clean         # was data/clean
  dataset: data/v8_dataset     # was data/dataset
  manifest: data/v8_manifest.csv  # was data/manifest.csv
finetune:
  project_name: anima_realism_ft_v8   # stage 4 names dataset.toml by this
dataset:
  min_resolution: 1536         # all v8 data is >=1536; safe to raise (was 1024)
```

(Leave `dataset.resolutions=[1536]`, `min_ar=0.66`, `max_ar=1.5` — already correct for v8.)

- [ ] **Step 2: Delete the stale Gemini cache** (v7-shape → would crash stage 3)

Run: `del data\gemini_cache.json` (PowerShell: `Remove-Item data/gemini_cache.json -ErrorAction SilentlyContinue`)
Expected: file removed (or already absent).

- [ ] **Step 3: Commit config**

```bash
git add config/pipeline.yaml
git commit -m "config(v8): point clean/dataset/manifest + project_name at v8"
```

---

## Task 4: Caption + build the v8 dataset (run existing stages)

**Files:** (no new code — runs `src/03_caption.py`, `src/04_build_dataset.py`)

- [ ] **Step 1: Run curation** → `data/v8_clean` + manifest

Run: `python src/v8_curate.py`
Expected: prints kept counts + ratio OK/WARN. If WARN, add/remove source images and re-run.

- [ ] **Step 2: Caption** (WD14 EVA02 + Gemini; needs `.env` GEMINI key)

Run: `python src/03_caption.py`
Expected: per-image WD14 tags + Gemini NL written into `data/v8_manifest.csv` `caption` column; underage hard-block backstop active. Watch the refusal/fallback log.

- [ ] **Step 3: Sanity-check a few captions** (esp. detail crops + bg)

Run: open `data/v8_manifest.csv`; confirm detail crops have WD14 tags like `hand, holding phone, barefoot, denim`; confirm anchor rows carry `amateur snapshot`/`casual phone photo` + high-fidelity quality tags. Fix the captioner only if clearly broken.

- [ ] **Step 4: Build** → `data/v8_dataset` + dataset toml

Run: `python src/04_build_dataset.py`
Expected: `data/v8_dataset/` filled with `img.jpg`+`img.txt` pairs; emits `outputs/anima_realism_ft_v8_dataset_config.toml` (resolutions=[1536], min_ar=0.66, max_ar=1.5).

- [ ] **Step 5: Commit the emitted dataset toml**

```bash
git add outputs/anima_realism_ft_v8_dataset_config.toml
git commit -m "feat(v8): emit dataset.toml (1536, AR 0.66-1.5)"
```

---

## Task 5: v8 train config

**Files:**
- Create: `outputs/anima_realism_ft_v8_train_config.toml` (clone `..._v7_train_config.toml`)

- [ ] **Step 1: Copy the v7 train toml and apply diffs**

Diffs vs v7 train toml:
- `transformer_path` → `/workspace/anima/models/anima_v7_epoch17.safetensors` *(warm-start ep17; was the v6 keeper)*
- `lr` → `4e-06` *(refiner; lower than validated 8e-6)*
- `epochs` → `10` *(ceiling; save-every + pick-best)*
- `output_dir` → `/workspace/anima/outputs/anima_realism_ft_v8`
- Keep: `adamw_optimi`, `warmup_steps=100`, `save_every_n_epochs=1`, `llm_adapter_lr=0`, `caption_dropout_percent=0.10`, `shuffle_tags=false`, `tag_dropout_percent=0`, `activation_checkpointing=true`.
- OOM fallbacks present but commented: `[model] qwen_nf4=true`, `[optimizer] adamw8bit`.

- [ ] **Step 2: Verify toml parses**

Run: `python -c "import tomllib; tomllib.load(open('outputs/anima_realism_ft_v8_train_config.toml','rb')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add outputs/anima_realism_ft_v8_train_config.toml
git commit -m "feat(v8): train toml - warm-start ep17, lr 4e-6, 10ep"
```

---

## Task 6: Vast scripts

**Files:**
- Create: `scripts/run_v8_train.sh` (clone `run_v7_train.sh`), `scripts/vast_fetch_v8.sh` (clone `vast_fetch_v7.sh`)

- [ ] **Step 1: Clone + edit `run_v8_train.sh`** — copy the v8 tomls into place, `nohup deepspeed --num_gpus=1 train.py --deepspeed --config <v8 train toml>`, log → `/workspace/train_v8.log`. Warm-start guard: assert `anima_v7_epoch17.safetensors` exists before launch.

- [ ] **Step 2: Clone + edit `vast_fetch_v8.sh`** — gdown the v8 `dataset.zip` + `V7_epoch17.safetensors` (IDs as args), size/count checks (mirror v7).

- [ ] **Step 3: Verify shell syntax**

Run: `bash -n scripts/run_v8_train.sh && bash -n scripts/vast_fetch_v8.sh && echo ok`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v8_train.sh scripts/vast_fetch_v8.sh
git commit -m "feat(v8): Vast run + fetch scripts (ep17 warm-start guard)"
```

---

## Task 7: Frozen eval prompts

**Files:**
- Create: `docs/superpowers/specs/2026-06-04-v8-eval-prompts.md`

- [ ] **Step 1: Write the eval doc** — fixed prompts + fixed seed list + inference settings:
  - **(a) detail:** "amateur snapshot, best quality, highres, sharp, close-up, a hand holding a smartphone, detailed fingers"; a denim/knit fabric close-up; bare feet / footwear close-up; generic hands.
  - **(b) whole-person amateur:** casual phone-photo framings (drift check vs stock).
  - **(c) background/scene.**
  - Each with fixed seeds (same list every epoch). Negative: `jpeg artifacts, compressed, blurry, low quality`.
  - Settings: CFG 3–4.5, optional RescaleCFG ~0.7, sampler euler/dpmpp_2m + simple/beta, steps 20–30, VAE `qwen_image_vae.safetensors`.
  - Run on **ep17 baseline + every saved v8 epoch**; pick best; watch drift (b) + overfit (variety collapse).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-04-v8-eval-prompts.md
git commit -m "docs(v8): frozen eval prompts + seeds + low-CFG settings"
```

---

## Execution order & gates

1. Task 0 ✅ → user fills `data/v8_raw/*` (manual Unsplash/Pixabay) + Task 1 (FiftyOne) for variety.
2. **Gate:** enough images that Task 2's ratio report is ~60/35/5. If not, source more.
3. Task 2 → 3 → 4 (curate → caption → build). **Gate:** spot-check captions.
4. Tasks 5–7 (configs/scripts/eval) can run in parallel with sourcing.
5. Vast: `vast_setup.sh` → `vast_fetch_v8.sh` → `run_v8_train.sh` → eval every epoch vs ep17 → pick best → **download + verify ~4.18 GB single-file + full `train_v8.log` BEFORE destroy.**

## Self-review notes
- **Spec coverage:** sourcing(T0/1), all-clean+≥1536+sharpness+AR-crop(T2), split-axis captioning(T3/4 reuse v7 captioner), warm-start ep17 + lr 4e-6 + 10ep(T5), Vast + pre-destroy(T6 + order), eval/low-CFG/drift(T7). Covered.
- **Open Images resolution caveat** is documented (T1) so detail fuel is not silently assumed from a ≤1024 source.
- **AR-crop** is a real function with tests (T2) — the mechanical crash-gate is enforced, not just toml-bucketed.
