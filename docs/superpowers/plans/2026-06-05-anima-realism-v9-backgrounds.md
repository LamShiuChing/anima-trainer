# Anima Realism v9 — Background Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v9 code/config artifacts — a whole-pool curation script with a NEW grid-patch background-sharpness gate (24-core + tqdm), an X-API sourcing script, and the v9 train/dataset/Vast configs — to fix blurry selfie/portrait backgrounds in the Anima photoreal finetune.

**Architecture:** v9 re-curates the entire historical image pool (`data/raw` + `data/v8_raw` + new `data/v9_x` + manual `data/v9_nsfw`) through per-image gates (≥1536 short side, overall Laplacian ≥100, **NEW background-sharpness gate**, phash dedup, AR-crop 0.66–1.5) into `data/v9_clean/` + `data/v9_manifest.csv`. The existing stage-3 captioner (WD14 + Gemini, incl. the underage block) and stage-4 dataset builder consume it unchanged. Training warm-starts `v8_epoch10` at lr 6e-6 for 20 epochs (save-every, pick-best) on Vast.

**Tech Stack:** Python 3 (stdlib + `cv2`, `numpy`, `imagehash`, `PIL`, `tqdm`), `concurrent.futures.ProcessPoolExecutor`, pytest, diffusion-pipe (deepspeed) on Vast.ai, X API v2 (pay-per-use), Gemini + WD14 EVA02 captioner (existing).

**Spec:** `docs/superpowers/specs/2026-06-05-anima-realism-v9-backgrounds-design.md`

---

## File Structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `src/v9_curate.py` | Whole-pool curation: gates (incl. grid bg-sharpness) + dedup + AR-crop → `data/v9_clean/` + manifest. Parallel (24 workers) + tqdm. | Create |
| `tests/test_v9_curate.py` | Unit tests for pure helpers: AR-crop, `passes_bg_sharpness`, `_dedup_local` (+ guarded cv2 grid test). | Create |
| `scripts/v9_fetch_x.py` | X API v2 sourcing (timeline/search) → `data/v9_x/`, resolution pre-filter, cost guard. | Create |
| `outputs/anima_realism_ft_v9_train_config.toml` | diffusion-pipe train config: warm-start ep10, lr 6e-6, 20 ep. | Create |
| `outputs/anima_realism_ft_v9_dataset_config.toml` | diffusion-pipe dataset config (1536, AR 0.66–1.5). | Create |
| `scripts/run_v9_train.sh` | Vast launch: copy tomls, guard inputs, `nohup deepspeed`. | Create |
| `scripts/vast_fetch_v9.sh` | Vast fetch: gdown dataset.zip + ep10 ckpt, size/count checks. | Create |
| `config/pipeline.yaml` | Repoint manifest/dataset/project_name → v9. | Modify (lines 5, 7, 49) |
| `docs/superpowers/specs/2026-06-05-v9-eval-prompts.md` | Frozen eval prompts (background + amateur-drift + NSFW canaries). | Create |

**Decisions locked from the spec:**
- **Underage block stays in stage 3** (`03_caption.py` already runs WD14 `block_tags`). The curate worker does NOT run WD14 (too heavy × 24 workers). The block runs before training regardless.
- **Dedup is GLOBAL** across all sources (kills cross-source reposts), not per-source like v8.
- **`passes_bg_sharpness` ships a working default rule**, clearly marked as the user's tuning point.

---

## Task 1: Pure helpers — `ar_crop_box` + `passes_bg_sharpness`

These are stdlib-only and import-safe (no cv2/PIL at module top), so tests import the module without the heavy deps installed — same pattern as `src/v8_curate.py`.

**Files:**
- Create: `src/v9_curate.py` (module skeleton + the two pure helpers)
- Test: `tests/test_v9_curate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v9_curate.py`:

```python
"""Unit tests for v9 curation pure helpers (AR-crop, background-sharpness rule, dedup)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "v9_curate", pathlib.Path(__file__).resolve().parents[1] / "src" / "v9_curate.py")
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)


# --- AR-crop (carried from v8; pos-emb 0.66-1.5 cap) ---
def test_ar_crop_wide_to_max():
    assert v9.ar_crop_box(2000, 1000, 0.66, 1.5) == (250, 0, 1750, 1000)


def test_ar_crop_in_range_noop():
    assert v9.ar_crop_box(1600, 1600, 0.66, 1.5) == (0, 0, 1600, 1600)


def test_ar_crop_tall_to_min():
    assert v9.ar_crop_box(1000, 2000, 0.66, 1.5) == (0, 242, 1000, 1757)


def test_ar_crop_preserves_short_side():
    for w, h in [(4000, 1500), (1536, 4000), (1600, 1536)]:
        l, t, r, b = v9.ar_crop_box(w, h, 0.66, 1.5)
        assert min(r - l, b - t) >= min(w, h)


# --- background-sharpness rule (the v9 centerpiece) ---
def test_bg_uniform_sharp_passes():
    # deep-focus: all 16 tiles sharp -> keep
    assert v9.passes_bg_sharpness([200.0] * 16) is True


def test_bg_bimodal_bokeh_fails():
    # bokeh: 4 sharp subject tiles, 12 soft background tiles -> drop
    assert v9.passes_bg_sharpness([300.0] * 4 + [10.0] * 12) is False


def test_bg_empty_fails():
    assert v9.passes_bg_sharpness([]) is False


def test_bg_half_sharp_passes_at_default_fraction():
    # exactly half sharp -> passes at default min_frac 0.5
    assert v9.passes_bg_sharpness([200.0] * 8 + [10.0] * 8) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v9_curate.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (`src/v9_curate.py` doesn't exist yet).

- [ ] **Step 3: Write the module skeleton + the two pure helpers**

Create `src/v9_curate.py`:

```python
"""v9 curation: whole pool (data/raw + data/v8_raw + data/v9_x + data/v9_nsfw) -> data/v9_clean
+ data/v9_manifest.csv. Per-image gates (no source exclusion):
  1. >=1536 px short side          (real detail; <1536 upscales soft)
  2. overall Laplacian >= 100      (reject soft/upscaled/compressed)
  3. grid-patch background-sharp   (NEW: drop bokeh/soft-bg; the v9 fix)
  4. phash dedup (hamming 8)       (GLOBAL across sources; keep highest-res per group)
  5. AR-crop to 0.66-1.5           (Anima DiT pos-emb 120-patch cap; else crash at 1536)
Underage block is NOT here -- it stays in stage 3 (03_caption.py WD14 block_tags), which runs
before training. Output manifest columns (path, source, width, height, phash, blur_var, bg_metric,
dropped, drop_reason) are consumed unchanged by stage 3 (caption) + stage 4 (build).

Parallel: ProcessPoolExecutor(24) + tqdm for the gate + crop passes (user has 24 cores).
Heavy libs (cv2/numpy/PIL/imagehash/tqdm/common) are imported INSIDE functions so this module
imports with stdlib only -> tests can import the pure helpers without those deps installed.
"""
from pathlib import Path

# --- sources + gate constants ---
SOURCES = ["data/raw", "data/v8_raw", "data/v9_x", "data/v9_nsfw"]
CLEAN = Path("data/v9_clean")
MANIFEST = "data/v9_manifest.csv"
MIN_SHORT = 1536
BLUR_MIN = 100.0
HAMMING = 8
MIN_AR, MAX_AR = 0.66, 1.5
WORKERS = 24

# --- background-sharpness gate (grid-patch, "approach C") ---
GRID_N = 4                  # 4x4 = 16 tiles
BG_TILE_T = 100.0           # a tile is "sharp" if its Laplacian variance >= this
BG_MIN_SHARP_FRAC = 0.5     # keep image if >= this fraction of tiles are sharp


def ar_crop_box(w, h, min_ar=MIN_AR, max_ar=MAX_AR):
    """Center-crop (left,top,right,bottom) so w/h lands in [min_ar,max_ar]. No-op if already in range.
    Trims the LONG side only -> the short side (and thus min-dimension >=1536) is preserved."""
    ar = w / h
    if min_ar <= ar <= max_ar:
        return (0, 0, w, h)
    if ar > max_ar:                       # too wide -> trim width
        new_w = round(max_ar * h)
        off = (w - new_w) // 2
        return (off, 0, off + new_w, h)
    new_h = round(w / min_ar)             # too tall -> trim height
    off = (h - new_h) // 2
    return (0, off, w, off + new_h)


def passes_bg_sharpness(tile_vars, tile_t=BG_TILE_T, min_frac=BG_MIN_SHARP_FRAC):
    """USER TUNING POINT -- this rule defines "good background" for the model.

    Default rule: keep the image if at least `min_frac` of grid tiles have Laplacian variance
    >= `tile_t`. Bokeh has a bimodal sharpness map (sharp subject blob + soft everything-else)
    -> few sharp tiles -> fails. Deep-focus is uniformly sharp -> most tiles sharp -> passes.

    Alternatives to try (see spec; calibrate in Task 7's calibration step):
      - median(tile_vars) >= tile_t
      - min(tile_vars) >= some floor   (penalizes ANY large soft region, e.g. one blurred corner)
      - 25th-percentile tile >= floor
    Tune `tile_t` / `min_frac` (or swap the rule) after the calibration printout.
    """
    if not tile_vars:
        return False
    sharp = sum(1 for v in tile_vars if v >= tile_t)
    return (sharp / len(tile_vars)) >= min_frac
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v9_curate.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/v9_curate.py tests/test_v9_curate.py
git commit -m "feat(v9): curate skeleton + pure helpers (ar_crop_box, passes_bg_sharpness)"
```

---

## Task 2: Grid Laplacian + the gate worker

Adds the cv2/numpy compute (`grid_laplacian_vars`) and the spawn-safe gate worker (`_gate_one`). The cv2 test is guarded with `importorskip` so it's skipped in envs without OpenCV (same convention as `tests/test_01_ingest_clean.py`).

**Files:**
- Modify: `src/v9_curate.py` (add `grid_laplacian_vars`, `_gate_one`)
- Test: `tests/test_v9_curate.py` (add a guarded grid test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v9_curate.py`:

```python
def test_grid_laplacian_vars_shape_and_bimodal():
    np = __import__("pytest").importorskip("numpy")
    __import__("pytest").importorskip("cv2")
    # left half = sharp checkerboard, right half = flat gray -> right tiles ~0 variance
    img = np.full((400, 400), 128, dtype=np.uint8)
    img[:, :200][::2, ::2] = 255
    img[:, :200][1::2, 1::2] = 0
    vars_ = v9.grid_laplacian_vars(img, n=4)
    assert len(vars_) == 16
    left = [vars_[r * 4 + c] for r in range(4) for c in range(2)]    # cols 0-1
    right = [vars_[r * 4 + c] for r in range(4) for c in range(2, 4)]  # cols 2-3
    assert min(left) > max(right)   # sharp side strictly sharper than flat side
```

- [ ] **Step 2: Run test to verify it fails (or skips without cv2)**

Run: `python -m pytest tests/test_v9_curate.py::test_grid_laplacian_vars_shape_and_bimodal -v`
Expected: FAIL with `AttributeError: module 'v9_curate' has no attribute 'grid_laplacian_vars'` (if cv2+numpy installed), or SKIP (if not installed).

- [ ] **Step 3: Add `grid_laplacian_vars` + `_gate_one`**

Add to `src/v9_curate.py` (after `passes_bg_sharpness`):

```python
def grid_laplacian_vars(gray, n=GRID_N):
    """Split a 2D grayscale array into n*n tiles -> Laplacian variance per tile (row-major)."""
    import cv2
    h, w = gray.shape[:2]
    ys = [int(round(i * h / n)) for i in range(n + 1)]
    xs = [int(round(j * w / n)) for j in range(n + 1)]
    out = []
    for i in range(n):
        for j in range(n):
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()) if tile.size else 0.0)
    return out


def _gate_one(task):
    """Worker (spawn-safe, top-level): (path_str, source) -> dict | None.
    Heavy libs imported inside so module import stays light. None = failed a gate / unreadable."""
    path_str, source = task
    import cv2
    import numpy as np
    import imagehash
    from PIL import Image
    try:
        gray = cv2.imdecode(np.fromfile(path_str, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:                       # cv2 can't decode (e.g. some webp) -> skip
            return None
        h, w = gray.shape[:2]
        if min(w, h) < MIN_SHORT:              # gate 1: resolution
            return None
        overall = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if overall < BLUR_MIN:                 # gate 2: overall sharpness (also filters compression)
            return None
        tile_vars = grid_laplacian_vars(gray, GRID_N)
        if not passes_bg_sharpness(tile_vars):  # gate 3: background sharpness (the v9 fix)
            return None
        with Image.open(path_str) as im:        # phash for dedup (gate 4, done in parent)
            ph = imagehash.phash(im.convert("RGB"))
        bg_metric = sum(1 for v in tile_vars if v >= BG_TILE_T) / len(tile_vars)
        return {"path": path_str, "source": source, "w": w, "h": h, "px": w * h,
                "blur": overall, "bg_metric": bg_metric, "phash": ph}
    except Exception:
        return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_v9_curate.py -v`
Expected: PASS (9 tests, or 8 PASS + 1 SKIP if cv2/numpy absent).

- [ ] **Step 5: Commit**

```bash
git add src/v9_curate.py tests/test_v9_curate.py
git commit -m "feat(v9): grid Laplacian + spawn-safe gate worker"
```

---

## Task 3: Dedup + crop-save worker + `main()` orchestration

**Files:**
- Modify: `src/v9_curate.py` (add `_dedup_local`, `_crop_save_one`, `main`)
- Test: `tests/test_v9_curate.py` (add dedup test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v9_curate.py`:

```python
class _PH:
    """Stub phash supporting hamming subtraction, so dedup is testable without imagehash."""
    def __init__(self, v):
        self.v = v
    def __sub__(self, other):
        return abs(self.v - other.v)


def test_dedup_keeps_highest_res_per_group():
    items = [
        {"phash": _PH(0), "px": 100},     # near-dup of next (hamming 1 <= 8); lower res -> dropped
        {"phash": _PH(1), "px": 400},     # near-dup; higher res -> KEEP
        {"phash": _PH(100), "px": 50},    # distinct (hamming 99 > 8) -> KEEP
    ]
    kept = v9._dedup_local(items, 8)
    assert sorted(it["px"] for it in kept) == [50, 400]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v9_curate.py::test_dedup_keeps_highest_res_per_group -v`
Expected: FAIL — `module 'v9_curate' has no attribute '_dedup_local'`.

- [ ] **Step 3: Add `_dedup_local`, `_crop_save_one`, `main`**

Add to `src/v9_curate.py`:

```python
def _dedup_local(items, threshold):
    """Greedy near-dup grouping on precomputed phashes (no re-open). Keep highest-res per group.
    items: list of dicts with 'phash' (supports `-` => hamming) + 'px' (pixel count)."""
    keep, used = [], set()
    for i in range(len(items)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(items)):
            if j not in used and (items[i]["phash"] - items[j]["phash"]) <= threshold:
                group.append(j)
        used.update(group)
        keep.append(items[max(group, key=lambda k: items[k]["px"])])
    return keep


def _crop_save_one(item):
    """Worker (spawn-safe, top-level): AR-crop + save to CLEAN -> manifest row dict | None."""
    from PIL import Image
    p, w, h = item["path"], item["w"], item["h"]
    try:
        im = Image.open(p).convert("RGB")
        box = ar_crop_box(w, h)
        if box != (0, 0, w, h):
            im = im.crop(box)
        stem = Path(p).stem
        dest = CLEAN / f"{item['source']}_{stem}.jpg"
        k = 1
        while dest.exists():                          # exact-name collision guard (cross-source)
            dest = CLEAN / f"{item['source']}_{stem}_{k}.jpg"
            k += 1
        im.save(dest, quality=95)
        cw, ch = im.size
        return {"path": str(dest).replace("\\", "/"), "source": item["source"],
                "width": str(cw), "height": str(ch), "phash": str(item["phash"]),
                "blur_var": f"{item['blur']:.1f}", "bg_metric": f"{item['bg_metric']:.3f}",
                "dropped": "False", "drop_reason": ""}
    except Exception as e:
        print(f"  skip (save failed) {p}: {e}")
        return None


def main():
    import shutil
    import concurrent.futures as cf
    from tqdm import tqdm
    import common

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src in SOURCES:
        d = Path(src)
        if not d.is_dir():
            print(f"  (skip missing source {src})")
            continue
        for p in common.iter_images(d):
            tasks.append((str(p), d.name))
    print(f"gating {len(tasks)} images from {len(SOURCES)} sources with {WORKERS} workers...")

    survivors = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for res in tqdm(ex.map(_gate_one, tasks, chunksize=8), total=len(tasks), desc="gate"):
            if res is not None:
                survivors.append(res)
    print(f"  passed gates: {len(survivors)}/{len(tasks)}")

    kept = _dedup_local(survivors, HAMMING)
    print(f"  after global dedup: {len(kept)}")

    rows = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for row in tqdm(ex.map(_crop_save_one, kept, chunksize=8), total=len(kept), desc="crop"):
            if row is not None:
                rows.append(row)

    common.write_manifest(MANIFEST, rows)
    dist = {}
    for r in rows:
        dist[r["source"]] = dist.get(r["source"], 0) + 1
    print(f"v9 curate: wrote {len(rows)} pairs {dist} -> {CLEAN}")
    print(f"manifest -> {MANIFEST}  (next: python src/03_caption.py)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v9_curate.py -v`
Expected: PASS (10 tests, or 9 PASS + 1 SKIP without cv2).

- [ ] **Step 5: Commit**

```bash
git add src/v9_curate.py tests/test_v9_curate.py
git commit -m "feat(v9): global dedup + parallel crop-save + main orchestration"
```

---

## Task 4: Repoint `config/pipeline.yaml` to v9

**Files:**
- Modify: `config/pipeline.yaml` (3 lines)

- [ ] **Step 1: Edit the three values**

In `config/pipeline.yaml`:
- Line 5: `dataset: data/v8_dataset` → `dataset: data/v9_dataset       # v9: re-curated whole pool`
- Line 7: `manifest: data/v8_manifest.csv` → `manifest: data/v9_manifest.csv # v9: written by src/v9_curate.py, consumed by stages 3+4`
- Line 49: `project_name: anima_realism_ft_v8 ...` → `project_name: anima_realism_ft_v9   # v9: background fix. stage4 names dataset.toml by this`

Leave `dataset.min_resolution: 1024` (harmless backstop; v9_curate already gates ≥1536) and `resolutions: [1536]` unchanged.

- [ ] **Step 2: Verify the config still loads**

Run: `python -c "import sys; sys.path.insert(0,'src'); import common; c=common.load_config(); print(c['paths']['manifest'], c['paths']['dataset'], c['finetune']['project_name'])"`
Expected: `data/v9_manifest.csv data/v9_dataset anima_realism_ft_v9`

- [ ] **Step 3: Commit**

```bash
git add config/pipeline.yaml
git commit -m "chore(v9): repoint pipeline.yaml manifest/dataset/project_name to v9"
```

---

## Task 5: Train + dataset TOMLs (warm-start ep10, lr 6e-6, 20 ep)

**Files:**
- Create: `outputs/anima_realism_ft_v9_train_config.toml`
- Create: `outputs/anima_realism_ft_v9_dataset_config.toml`

- [ ] **Step 1: Write the train config**

Create `outputs/anima_realism_ft_v9_train_config.toml`:

```toml
# diffusion-pipe — Anima FULL finetune. v9 = BACKGROUND FIX + style epochs.
# WARM-START from the V8 keeper (v8_epoch10.safetensors): keeps ep10's realism/detail/lighting,
# trains on a BIGGER, sharp-BACKGROUND-gated whole-pool dataset to fix blurry selfie/portrait
# backgrounds (per-concept blur learning) while keeping the amateur aesthetic + NSFW.
# lr 6e-6 (hotter than v8's 4e-6 = headroom; cooler than v7's 8e-6). epochs 20 (ep10 was still
# climbing); save-every -> pick best on the frozen eval set (loss is blind here).
# Spec: docs/superpowers/specs/2026-06-05-anima-realism-v9-backgrounds-design.md
output_dir = '/workspace/anima/outputs/anima_realism_ft_v9'
dataset = '/workspace/anima/outputs/anima_realism_ft_v9_dataset_config.toml'

epochs = 20
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
gradient_clipping = 1.0
steps_per_print = 100
warmup_steps = 100
activation_checkpointing = true

# Eval disabled (no eval set in-trainer). Preview epoch checkpoints in ComfyUI with the frozen eval prompts.
eval_before_first_step = false
eval_every_n_epochs = 100000

save_every_n_epochs = 1
checkpoint_every_n_minutes = 999999
save_dtype = 'bfloat16'

[model]
type = 'anima'
# WARM-START: upload the V8 keeper here as anima_v8_epoch10.safetensors (NOT base, NOT v7).
transformer_path = '/workspace/anima/models/anima_v8_epoch10.safetensors'
vae_path = '/workspace/anima/models/qwen_image_vae.safetensors'
qwen_path = '/workspace/anima/models/Qwen3-0.6B-Base'
dtype = 'bfloat16'
llm_adapter_lr = 0
cache_text_embeddings = false
shuffle_tags = false          # captions carry one NL sentence -> shuffling commas would shred it
tag_delimiter = ', '
shuffle_keep_first_n = 1
tag_dropout_percent = 0
caption_dropout_percent = 0.1
caption_mode = 'tags'
timestep_sample_method = 'logit_normal'
# ⚠️ 1536 VRAM: same as v7/v8 (ran fine on the 96GB Blackwell). If OOM on a smaller card:
#    (1) uncomment qwen_nf4 below, and/or (2) switch [optimizer] to adamw8bit (needs bitsandbytes).
# qwen_nf4 = true

# NOTE: no [adapter] block => full finetune (not LoRA).

[optimizer]
type = 'adamw_optimi'         # same as v7/v8; switch to adamw8bit only if 1536 OOMs the card
lr = 6e-06                    # v9: hotter than v8's 4e-6 (headroom; no drift seen at 4e-6). Watch amateur drift.
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8

[monitoring]
enable_wandb = false
```

- [ ] **Step 2: Write the dataset config**

Create `outputs/anima_realism_ft_v9_dataset_config.toml` (identical shape to v8; stage 4 will regenerate the same content on build — committing it now lets the Vast scripts work before a local build):

```toml
# diffusion-pipe dataset config (Anima full finetune, images only)
resolutions = [1536]
enable_ar_bucket = true
min_ar = 0.66
max_ar = 1.5
num_ar_buckets = 7
frame_buckets = [1]

[[directory]]
path = '/workspace/anima/data/dataset'
num_repeats = 1
```

- [ ] **Step 3: Verify TOMLs parse**

Run: `python -c "import tomllib; [print(k, 'OK') for k in ['train','dataset'] for _ in [tomllib.load(open(f'outputs/anima_realism_ft_v9_{k}_config.toml','rb'))]]"`
Expected: `train OK` then `dataset OK` (no exception).

- [ ] **Step 4: Commit**

```bash
git add outputs/anima_realism_ft_v9_train_config.toml outputs/anima_realism_ft_v9_dataset_config.toml
git commit -m "feat(v9): train+dataset tomls (warm-start ep10, lr 6e-6, 20ep, 1536 AR0.66-1.5)"
```

---

## Task 6: Vast scripts — `run_v9_train.sh` + `vast_fetch_v9.sh`

**Files:**
- Create: `scripts/run_v9_train.sh`
- Create: `scripts/vast_fetch_v9.sh`

- [ ] **Step 1: Write the launch script**

Create `scripts/run_v9_train.sh`:

```bash
#!/usr/bin/env bash
# v9 launch on Vast = 1536 full-finetune BACKGROUND FIX, WARM-START from the V8 keeper (epoch10),
# lr 6e-6, epochs 20 (save-every, pick best). Requires: warm-start ckpt
# (models/anima_v8_epoch10.safetensors), VAE + Qwen3 dir (vast_setup.sh), v9 dataset (data/dataset,
# uploaded via vast_fetch_v9.sh), and the two v9 tomls (this repo).
# Log -> /workspace/train_v9.log (download from Jupyter after run for loss-trend analysis).
set -euo pipefail
BASE=/workspace/anima

mkdir -p "$BASE/outputs"
cp "$BASE/repo/outputs/anima_realism_ft_v9_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v9_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima_v8_epoch10.safetensors" || { echo "MISSING warm-start ckpt -> upload v8_epoch10.safetensors to models/anima_v8_epoch10.safetensors"; exit 1; }
test -f "$BASE/models/qwen_image_vae.safetensors"  || { echo "MISSING VAE -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> run scripts/vast_fetch_v9.sh first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v9_train_config.toml" \
  > /workspace/train_v9.log 2>&1 &
echo "STARTED v9 pid $!  --  watch: tail -f /workspace/train_v9.log"
```

- [ ] **Step 2: Write the fetch script**

Create `scripts/vast_fetch_v9.sh`:

```bash
#!/usr/bin/env bash
# Fetch v9 inputs onto a fresh Vast instance: dataset zip + the V8 warm-start checkpoint (epoch10).
# IDs are passed as ARGS (never committed -- repo is public, Drive links are private).
# Usage: bash scripts/vast_fetch_v9.sh <DATASET_ZIP_GDRIVE_ID> <EP10_GDRIVE_ID>
set -euo pipefail
BASE="${ANIMA_BASE:-/workspace/anima}"
CKPT_BYTES=4182218360   # exact size of an Anima DiT epoch (bfloat16 single-file); guards gdown truncation

[ $# -eq 2 ] || { echo "usage: $0 <DATASET_ZIP_ID> <EP10_ID>"; exit 1; }
DATASET_ID="$1"; CKPT_ID="$2"

mkdir -p "$BASE/data" "$BASE/models"
pip install -q gdown

# 1) warm-start checkpoint -> the exact path run_v9_train.sh guards
gdown "$CKPT_ID" -O "$BASE/models/anima_v8_epoch10.safetensors"
SZ=$(stat -c%s "$BASE/models/anima_v8_epoch10.safetensors")
if [ "$SZ" != "$CKPT_BYTES" ]; then
  echo "CKPT SIZE WRONG: got $SZ, expect $CKPT_BYTES (gdown likely returned the virus-scan HTML page)."
  echo "Retry: gdown --fuzzy 'https://drive.google.com/uc?id=$CKPT_ID' -O $BASE/models/anima_v8_epoch10.safetensors"
  exit 1
fi
echo "ckpt size OK ($SZ)"

# 2) dataset zip -> flatten to data/dataset (find wherever the .txt sidecars land)
gdown "$DATASET_ID" -O "$BASE/data/dataset.zip"
rm -rf "$BASE/data/_stage" "$BASE/data/dataset"
mkdir -p "$BASE/data/_stage" "$BASE/data/dataset"
unzip -q -o "$BASE/data/dataset.zip" -d "$BASE/data/_stage"
TXT=$(find "$BASE/data/_stage" -name '*.txt' -print -quit)
[ -n "$TXT" ] || { echo "NO .txt sidecars in zip -> wrong archive?"; exit 1; }
SRC=$(dirname "$TXT")
mv "$SRC"/* "$BASE/data/dataset"/
rm -rf "$BASE/data/_stage"

CNT=$(ls "$BASE/data/dataset" | wc -l)
IMG=$(find "$BASE/data/dataset" -type f ! -name '*.txt' | wc -l)
TXTN=$(find "$BASE/data/dataset" -name '*.txt' | wc -l)
echo "dataset: $CNT files ($IMG images + $TXTN captions)"
[ "$IMG" -gt 0 ] && [ "$TXTN" -gt 0 ] || { echo "MISSING images or captions"; exit 1; }
echo "fetch OK -> now: bash $BASE/repo/scripts/run_v9_train.sh"
```

- [ ] **Step 3: Verify scripts are syntactically valid**

Run: `bash -n scripts/run_v9_train.sh && bash -n scripts/vast_fetch_v9.sh && echo "both OK"`
Expected: `both OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v9_train.sh scripts/vast_fetch_v9.sh
git commit -m "feat(v9): Vast run + fetch scripts (warm-start ep10, train_v9.log)"
```

---

## Task 7: X sourcing script — `scripts/v9_fetch_x.py`

X API v2 pay-per-use. Timeline mode (recommended, cheap) + search mode. Resolution pre-filter from metadata; downloads `name=orig` from the CDN (free). Cost guard via `--max-reads` + running `$` print.

**Files:**
- Create: `scripts/v9_fetch_x.py`

- [ ] **Step 1: Write the script**

Create `scripts/v9_fetch_x.py`:

```python
"""v9 X (Twitter) sourcing -- download ORIGINAL-resolution photos from target accounts' timelines
(or a search query) into data/v9_x/ via the X API v2 (pay-per-use).

WHY: X has amateur deep-focus selfies/mirror-selfies that Pexels lacks (right content), but every
upload is re-encoded to JPEG (diseased encoding). The v9 curate gates (>=1536 + Laplacian + grid
background-sharpness) are the safety net -> expect ~10% yield. Targeting good accounts beats search.

Setup (one time):
  1. X developer account + project/app, enable pay-per-use, and SET A SPENDING LIMIT in the console.
     Pricing (verified 2026-06-05): post read $0.005, media read $0.005. See
     https://docs.x.com/x-api/getting-started/pricing
  2. App-only Bearer Token -> add to .env in project root (same file as GEMINI_API_KEY):
        X_BEARER_TOKEN=AAAA...
  3. stdlib urllib only; dotenv optional.

Run:
  # timeline mode (recommended) -- one or more handles (no @), comma-separated
  python scripts/v9_fetch_x.py --handles someacct,another --max-reads 2000
  # search mode -- recent (last 7 days) by default, or --full-archive (back to 2006)
  python scripts/v9_fetch_x.py --query "mirror selfie full body" --max-reads 1000 --full-archive

Notes:
  - Filters min(width,height) >= --min-short (default 1536) from API metadata BEFORE downloading.
  - Downloads name=orig (largest, ~<=4096). Skips media_keys already on disk (safe to re-run).
  - --max-reads caps posts fetched (cost guard). The script prints a running $ estimate.
  - ToS: training-data use of X content is a gray area; NSFW of real people = consent/legal. Your call.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.x.com/2"
PRICE_READ = 0.005  # USD per post read AND per media resource (pay-per-use, 2026-06-05)


def get_token():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    tok = (os.environ.get("X_BEARER_TOKEN") or "").strip().strip('"').strip("'").strip()
    if not tok:
        sys.stderr.write("X_BEARER_TOKEN not set. Add it to .env as X_BEARER_TOKEN=...\n")
        sys.exit(1)
    return tok


def api_get(url, token):
    """GET with bearer auth; retry on 429/5xx with exponential backoff."""
    for attempt in range(6):
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "anima-v9/1.0 (research dataset tool)",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                wait = 2 ** attempt * 5
                sys.stderr.write(f"  HTTP {e.code}; backoff {wait}s (attempt {attempt + 1})\n")
                time.sleep(wait)
                continue
            sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}\n")
            raise
    raise RuntimeError("api_get: exhausted retries")


def resolve_user(handle, token):
    data = api_get(f"{API}/users/by/username/{urllib.parse.quote(handle)}", token)
    uid = data.get("data", {}).get("id")
    if not uid:
        raise RuntimeError(f"could not resolve handle '{handle}': {data}")
    return uid


def _media_params():
    return ("expansions=attachments.media_keys"
            "&media.fields=url,width,height,type&max_results=100")


def iter_pages(base_url, token, max_reads, reads_state):
    """Yield each page's includes.media list, paginating until exhausted or max_reads hit.
    reads_state = mutable [posts_read] counter (each returned post = one read)."""
    token_param = None
    while reads_state[0] < max_reads:
        url = base_url
        if token_param:
            url += f"&pagination_token={token_param}" if "users/" in base_url else f"&next_token={token_param}"
        data = api_get(url, token)
        posts = data.get("data", []) or []
        reads_state[0] += len(posts)
        media = data.get("includes", {}).get("media", []) or []
        yield media
        meta = data.get("meta", {})
        token_param = meta.get("next_token")
        if not token_param or not posts:
            return


def save_media(media_list, out, min_short, saved_state, reads_state):
    for m in media_list:
        if m.get("type") != "photo":
            continue
        w, h = m.get("width", 0), m.get("height", 0)
        if min(w, h) < min_short:
            continue
        key = m.get("media_key")
        url = m.get("url")
        if not key or not url:
            continue
        dest = out / f"x_{key}.jpg"
        if dest.exists():
            continue
        orig = url + ("&" if "?" in url else "?") + "name=orig"
        try:
            req = urllib.request.Request(orig, headers={"User-Agent": "anima-v9/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            reads_state[1] += 1  # media read billed
            saved_state[0] += 1
        except Exception as e:
            sys.stderr.write(f"  download failed {key}: {e!r}\n")


def main():
    ap = argparse.ArgumentParser(description="Download high-res X photos into data/v9_x/.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--handles", help="comma-separated account handles (no @) for timeline mode")
    g.add_argument("--query", help="search query (recent 7d, or --full-archive)")
    ap.add_argument("--full-archive", action="store_true", help="use full-archive search (back to 2006)")
    ap.add_argument("--max-reads", type=int, default=2000, help="cap posts fetched (cost guard)")
    ap.add_argument("--min-short", type=int, default=1536, help="min short side (px) to keep")
    ap.add_argument("--out", default="data/v9_x")
    args = ap.parse_args()

    token = get_token()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved_state = [0]            # images saved
    reads_state = [0, 0]         # [posts_read, media_read]

    if args.handles:
        for handle in [h.strip().lstrip("@") for h in args.handles.split(",") if h.strip()]:
            if reads_state[0] >= args.max_reads:
                break
            print(f"timeline @{handle} ...")
            uid = resolve_user(handle, token)
            base = f"{API}/users/{uid}/tweets?exclude=replies,retweets&{_media_params()}"
            for media in iter_pages(base, token, args.max_reads, reads_state):
                save_media(media, out, args.min_short, saved_state, reads_state)
                print(f"  saved={saved_state[0]} posts_read={reads_state[0]} "
                      f"~${reads_state[0]*PRICE_READ + reads_state[1]*PRICE_READ:.2f}")
    else:
        endpoint = "tweets/search/all" if args.full_archive else "tweets/search/recent"
        base = f"{API}/{endpoint}?query={urllib.parse.quote(args.query)}&{_media_params()}"
        for media in iter_pages(base, token, args.max_reads, reads_state):
            save_media(media, out, args.min_short, saved_state, reads_state)
            print(f"  saved={saved_state[0]} posts_read={reads_state[0]} "
                  f"~${reads_state[0]*PRICE_READ + reads_state[1]*PRICE_READ:.2f}")

    cost = reads_state[0] * PRICE_READ + reads_state[1] * PRICE_READ
    print(f"\nDone. Saved {saved_state[0]} images (>= {args.min_short}px) -> {out}/")
    print(f"Reads: {reads_state[0]} posts + {reads_state[1]} media  ~= ${cost:.2f} (estimate)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses + the CLI works**

Run: `python scripts/v9_fetch_x.py --help`
Expected: argparse usage printed (mutually-exclusive `--handles`/`--query`, `--max-reads`, etc.), no import error.

- [ ] **Step 3: Commit**

```bash
git add scripts/v9_fetch_x.py
git commit -m "feat(v9): X API v2 sourcing script (timeline/search, res pre-filter, cost guard)"
```

---

## Task 8: Frozen eval-prompts doc

**Files:**
- Create: `docs/superpowers/specs/2026-06-05-v9-eval-prompts.md`

- [ ] **Step 1: Write the eval-prompts doc**

Create `docs/superpowers/specs/2026-06-05-v9-eval-prompts.md`:

```markdown
# Anima Realism v9 — frozen eval prompts

> Run on EVERY saved epoch + the **`v8_epoch10` baseline** (apples-to-apples: did v9 beat ep10?).
> Loss is BLIND (flat-noise across v5-v8). Judge ONLY by eval images. Fixed seeds across all epochs.

## Inference settings (Anima flow-matching DiT)
- **CFG 3.0-4.5** (high CFG oversaturates — that's a CFG signature, not undertraining), optional RescaleCFG ~0.7.
- Sampler `euler` or `dpmpp_2m` + `simple`/`beta`, steps 20-30.
- VAE = `qwen_image_vae.safetensors`. Generate at ~1536 area (1536x1536 / 1344x1728 / 1856x1280; AR 0.66-1.5, dims ÷64).
- **Fidelity/sharp-bg dial:** prompt append `amateur snapshot, best quality, highres, sharp` +
  negative `jpeg artifacts, compressed, blurry, low quality, bokeh, blurred background`.

## (a) BACKGROUND canaries — the core v9 target
1. `amateur snapshot, a woman taking a selfie in a detailed living room, bookshelf and window and plants behind her, best quality, highres, sharp`
2. `casual phone photo, mirror selfie of a person in a messy bedroom, clothes and posters on the wall in focus, sharp`
3. `amateur snapshot, full body portrait of a person on a busy city street, shops and signs and people in the background, deep focus, sharp`
4. `casual phone photo, a person standing in a kitchen, cabinets and appliances and counter clutter visible and sharp`
Judge: is the BACKGROUND sharp + coherent (not blurry/slopish/melted)? This is the pass/fail for v9.

## (b) AMATEUR-DRIFT canary — primary regression risk at lr 6e-6
5. `amateur snapshot, casual photo of a person in a backyard, natural daylight`
6. `casual phone photo, candid of a person sitting on a couch at home`
Judge: still amateur/candid, or drifted toward polished studio/stock? Drift -> stop / pick earlier epoch.

## (c) NSFW capability check
7. (user-supplied explicit prompt in the v9 vocab) — confirm explicit capability survived + its background improved too.

## Stop signals
- Amateur -> stock drift = stop / lower LR / pick earlier epoch.
- Backgrounds sharpen but composition variety collapses = overfit -> pick the prior epoch.
- Pick the BEST epoch regardless of number. Download it (verified ~4.18 GB) BEFORE destroying the instance.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-05-v9-eval-prompts.md
git commit -m "docs(v9): frozen eval prompts (background + amateur-drift + nsfw canaries)"
```

---

## Task 9: Full test sweep + push

- [ ] **Step 1: Run the v9 tests**

Run: `python -m pytest tests/test_v9_curate.py -v`
Expected: PASS (10 tests, or 9 PASS + 1 SKIP if cv2/numpy absent).

- [ ] **Step 2: Run the full suite (sanity — no regressions)**

Run: `python -m pytest tests/ -v` (per CLAUDE.md, `tests/test_01_ingest_clean.py` may error if `imagehash`/`cv2` absent in the active env — that's pre-existing, not a v9 regression).
Expected: v9 tests PASS; no NEW failures introduced by v9 changes.

- [ ] **Step 3: Push the branch**

```bash
git push origin v5-build
```

Expected: branch pushed; all v9 commits on `origin/v5-build`.

---

## Operational runbook (user-in-the-loop; AFTER the code tasks above)

These steps need API keys, GPU rental, and manual NSFW — run with the user, not as autonomous code tasks.

1. **(Optional) Source X data:** set `X_BEARER_TOKEN` in `.env` + a spending limit in the X console.
   `python scripts/v9_fetch_x.py --handles acct1,acct2,... --max-reads 2000` → `data/v9_x/`.
   Targeting 10-20 good amateur-photo accounts ≈ ~$100.
2. **Manual NSFW:** drop high-res, deep-focus, legal-adult images into `data/v9_nsfw/`.
3. **Curate:** `python src/v9_curate.py` → watch tqdm; expect ~800-1500 kept. Eyeball `data/v9_clean/` for
   any bokeh that slipped through.
4. **⚠️ Calibrate the bg gate (centerpiece):** if too many bokeh pass OR too many good shots dropped, add a
   one-off print of `tile_vars` for a few known-bokeh vs known-deep-focus files (temporary scriptlet calling
   `v9_curate.grid_laplacian_vars`), then tune `BG_TILE_T` / `BG_MIN_SHARP_FRAC` or swap the `passes_bg_sharpness`
   rule (median / min-tile floor — see the function docstring). Re-run curate. This is the make-or-break tuning.
5. **Caption:** `python src/03_caption.py` (reuses the v7 captioner + `data/gemini_cache.json` incrementally;
   runs the WD14 underage block). **Do NOT delete the cache** — v9 is v7-shape.
6. **Build:** `python src/04_build_dataset.py` → `data/v9_dataset/` + regenerates
   `outputs/anima_realism_ft_v9_dataset_config.toml`. Commit the regenerated toml if it changed.
7. **Zip + upload:** zip `data/v9_dataset/` → Drive (get ID). Upload local
   `…\ComfyUI\models\diffusion_models\v8_epoch10.safetensors` → Drive (get ID).
8. **Vast:** rent a 80-96 GB card → `scripts/vast_setup.sh` → `bash scripts/vast_fetch_v9.sh <DATASET_ID> <EP10_ID>`
   → `bash scripts/run_v9_train.sh` → `tail -f /workspace/train_v9.log`. Verify torch sees the GPU first
   (`python -c "import torch; print(torch.cuda.get_device_name(0))"`); `nvidia-smi -l 5` during latent caching.
9. **Eval:** run `docs/superpowers/specs/2026-06-05-v9-eval-prompts.md` on every saved epoch + ep10 baseline.
10. **⚠️ Pre-destroy checklist:** best epoch on local disk (verified ~4.18 GB single-file DiT) + full
    `train_v9.log` downloaded + optional neighbor epoch. THEN **destroy** (not stop). ~$1.3/hr — destroy the
    moment train+download finish.

---

## Self-Review

**Spec coverage:** Goal/diagnosis → Task 1-3 (gates) + Tasks 5-6 (warm-start ep10, lr 6e-6, 20ep). Three-axis
principle + grid bg-sharpness gate → Task 1-2 + calibration (runbook 4). Whole-pool sources → Task 3 `SOURCES`.
Dedup global → Task 3. AR cap → Task 1/3. Multicore+tqdm → Task 3. Captioner reuse + keep cache → runbook 5.
Underage block in stage 3 → noted in Task 3 docstring + runbook 5. Config repoint → Task 4. Tomls → Task 5.
Vast scripts → Task 6. X sourcing (pay-per-use, timeline, pre-filter, cost guard) → Task 7. Eval canaries → Task 8.
Safety/legal-adults → Task 7 docstring + runbook 2. All spec sections covered.

**Placeholder scan:** No TBD/TODO. `passes_bg_sharpness` ships a complete working default (not a placeholder) with
the tuning alternatives in its docstring + a dedicated calibration step. The single user-supplied NSFW eval prompt
(Task 8 item 7) is intentionally user-private, not a code gap.

**Type consistency:** `_gate_one` returns dicts with keys `path/source/w/h/px/blur/bg_metric/phash`; `_dedup_local`
consumes `phash`+`px`; `_crop_save_one` consumes `path/w/h/source/phash/blur/bg_metric` and emits manifest keys
`path/source/width/height/phash/blur_var/bg_metric/dropped/drop_reason` — matching stage-4 `curate()` expectations
(`dropped`=="False", `width`/`height`, `caption` added later by stage 3). `passes_bg_sharpness`/`grid_laplacian_vars`/
`ar_crop_box` signatures consistent across tasks. Constants (`MIN_SHORT`, `BLUR_MIN`, `HAMMING`, `GRID_N`, `BG_TILE_T`,
`BG_MIN_SHARP_FRAC`, `WORKERS`, `SOURCES`, `CLEAN`, `MANIFEST`) defined once in Task 1, used throughout.
```
