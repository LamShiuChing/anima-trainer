# Anima Realism Finetune v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Phase-1 local LoRA pipeline into a cloud full-finetune pipeline: tag-only captions (no JoyCaption, no trigger), tag-don't-filter ingest, and diffusion-pipe `anima.toml`/`dataset.toml` config emission — runnable end-to-end on a rented Vast.ai Linux GPU.

**Architecture:** Keep the existing stage/manifest design. Stage 2 (CLIP aesthetic scoring) is unchanged. Stage 1 gates its drop-heuristics behind config flags (only corrupt-drop + phash dedup run). Stage 3 drops JoyCaption and assembles `<quality>, <safety>` captions. Stages 4/5 emit diffusion-pipe TOML instead of kohya TOML. New bash scripts + `RUNBOOK.md` drive the run on Vast.

**Tech Stack:** Python 3.10 pipeline (pillow, transformers, CLIP, imagehash), diffusion-pipe (DeepSpeed) on Linux, Vast.ai, pytest + tomli backport.

**Reference (verified live 2026-06-01):**
- diffusion-pipe Anima fork: `bluvoll/diffusion-pipe` (`examples/anima.toml`, `examples/dataset.toml`), upstream PR [tdrussell/diffusion-pipe#505](https://github.com/tdrussell/diffusion-pipe/pull/505).
- Anima loader (`models/anima.py`): `qwen_path` accepts a single `.safetensors` file OR an HF dir; `vae_path` loaded by `WanVAE` (Qwen-Image VAE is Wan-derived); `transformer_path` = `anima-base-v1.0.safetensors`; `llm_adapter_lr = 0` → `requires_grad_(False)` on the LLM adapter.
- Full finetune = **omit the `[adapter]` block**. Tag-only = `caption_mode = 'tags'`. `cache_text_embeddings = false` required for tag dropout/shuffle.
- VRAM (authors): ~31 GB @512px, ~33 GB @768px w/ activation checkpointing → a single ≥48 GB GPU suffices.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `config/pipeline.yaml` | Modify | Add ingest flags (off), diffusion-pipe dataset params, `finetune:` block; `buckets_to_keep` → all three. |
| `src/01_ingest_clean.py` | Modify | Gate small/blur/OCR behind flags; default run = corrupt-drop + phash dedup only. Guard `rapidocr` import. |
| `src/02_quality_score.py` | Unchanged | CLIP aesthetic → bucket (tags everything, drops nothing). |
| `src/03_caption.py` | Modify | Remove `JoyCaptioner` + `clean_nl`; `assemble_caption(quality, safety)` → `"<quality>, <safety>"`. |
| `src/04_build_dataset.py` | Modify | Keep all buckets; emit diffusion-pipe `dataset.toml` (resolutions/AR-buckets/`[[directory]]`). |
| `src/05_make_train_config.py` | Modify | Emit diffusion-pipe `anima.toml` (full finetune, `type='anima'`, `llm_adapter_lr=0`). Drop sd-scripts sample-prompts. |
| `scripts/vast_setup.sh` | Create | One-shot on Vast: clone fork, install deps, `wget` Anima models. |
| `scripts/run_prep.sh` | Create | Run stages 01→05 on Vast (pipeline venv). |
| `RUNBOOK.md` | Create | User-facing step-by-step: Vast → upload → setup → prep → train → retrieve. |
| `requirements.txt` | Modify | Remove `bitsandbytes` (JoyCaption-only). Keep transformers/CLIP/NSFW deps. |
| `tests/test_03_caption.py` | Modify | New `assemble_caption` signature; drop `clean_nl` test. |
| `tests/test_04_build_dataset.py` | Modify | diffusion-pipe `dataset.toml` assertions; `buckets_to_keep` all. |
| `tests/test_05_make_train_config.py` | Modify | `anima.toml` assertions; drop sample-prompts test. |

Stages 1 & 2 keep all their existing pure functions (tests stay green); only `main()`/flags change.

---

### Task 1: Config — pipeline.yaml v2

**Files:**
- Modify: `config/pipeline.yaml`

- [ ] **Step 1: Edit `ingest:` block** — add three flags (after `ocr_text_area_ratio_flag`):

```yaml
ingest:
  min_size: 512                  # kept for reference; not enforced when drop_small=false
  blur_var_threshold: 100.0      # kept for reference; not enforced when drop_blurry=false
  phash_hamming_threshold: 6     # <= this Hamming distance => near-duplicate (ALWAYS enforced)
  ocr_text_area_ratio_flag: 0.10 # only used when run_ocr_flag=true
  drop_small: false              # v2: tag-don't-filter; diffusion-pipe AR-buckets handle small images
  drop_blurry: false             # v2: keep blurry, they just won't score 'good'
  run_ocr_flag: false            # v2: skip OCR entirely (we keep everything anyway)
```

- [ ] **Step 2: Edit `dataset:` block** — keep all buckets, add diffusion-pipe params:

```yaml
dataset:
  buckets_to_keep: [good, medium, bad]  # v2: keep all; mixed quality becomes a tag, not a discard
  # diffusion-pipe dataset.toml params:
  resolutions: [512]             # Anima native res; run 1. diffusion-pipe resizes to this AREA (upscales smaller imgs), so 512 minimizes upscaling on social data
  min_ar: 0.5                    # widest portrait bucket (1:2)
  max_ar: 2.0                    # widest landscape bucket (2:1)
  num_ar_buckets: 7
  num_repeats: 1                 # lots of data; no duplication needed
```

(Leave `caption_dropout_rate` line removed — it now lives in `finetune.caption_dropout_percent`.)

- [ ] **Step 3: Edit `caption:` block** — drop JoyCaption keys, drop trigger:

```yaml
caption:
  nsfw_model: MichalMlodawski/nsfw-image-detection-large
  nsfw_label_map:                # substring of model label (upper) -> anima safety tag
    SAFE: safe
    QUESTIONABLE: sensitive
    UNSAFE: explicit
  nsfw_default_tag: safe
  quality_tag_map:
    good: "masterpiece, best quality"
    medium: "high quality"
    bad: "low quality"
```

- [ ] **Step 4: Replace `train:` block with `finetune:` block** (diffusion-pipe full finetune):

```yaml
finetune:                        # diffusion-pipe anima.toml (FULL finetune, not LoRA)
  base_dir: /workspace/anima     # Vast working dir (RUNBOOK sets this up)
  project_name: anima_realism_ft_v1
  epochs: 10
  lr: 8.0e-6                     # authors' full-finetune rec (8e-6 or lower); low to avoid catastrophic forgetting
  optimizer: AdamW8bitKahan      # 8-bit optimizer for lower VRAM
  warmup_steps: 100
  save_every_n_epochs: 1
  checkpoint_every_n_minutes: 30
  llm_adapter_lr: 0              # freeze Qwen3 LLM adapter (domain shift; preserve text understanding)
  tag_dropout_percent: 0.10
  caption_dropout_percent: 0.05  # CFG: % steps trained with empty caption
  dit_file: anima-base-v1.0.safetensors
  vae_file: qwen_image_vae.safetensors
  qwen_file: qwen_3_06b_base.safetensors
```

Keep `paths:` and `models:` blocks as-is.

- [ ] **Step 5: Verify YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('config/pipeline.yaml',encoding='utf-8'))"`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add config/pipeline.yaml
git commit -m "config: v2 finetune — tag-don't-filter ingest, diffusion-pipe params, finetune block"
```

---

### Task 2: Stage 1 — gate drop-heuristics behind flags

**Files:**
- Modify: `src/01_ingest_clean.py:86-131` (the `main()` and the rapidocr import)
- Test: `tests/test_01_ingest_clean.py` (existing function tests stay; add a flag-logic test)

- [ ] **Step 1: Write the failing test** (append to `tests/test_01_ingest_clean.py`):

```python
def test_drop_reason_respects_flags():
    # pure helper: given flags, decide a drop reason for a candidate
    assert stage.drop_reason(corrupt=True, too_small=True, blurry=True,
                             drop_small=False, drop_blurry=False) == "corrupt"
    assert stage.drop_reason(corrupt=False, too_small=True, blurry=False,
                             drop_small=False, drop_blurry=False) == ""
    assert stage.drop_reason(corrupt=False, too_small=True, blurry=False,
                             drop_small=True, drop_blurry=False) == "too_small"
    assert stage.drop_reason(corrupt=False, too_small=False, blurry=True,
                             drop_small=False, drop_blurry=True) == "blurry"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_01_ingest_clean.py::test_drop_reason_respects_flags -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'drop_reason'`.

- [ ] **Step 3: Add the `drop_reason` pure helper** (insert after `is_too_small`, around line 31):

```python
def drop_reason(corrupt, too_small, blurry, drop_small, blurry_flag=None, drop_blurry=False):
    """Decide the drop reason given heuristic results + which heuristics are enabled.
    corrupt always drops (would crash the trainer). small/blurry only drop if their flag is on."""
    if corrupt:
        return "corrupt"
    if drop_small and too_small:
        return "too_small"
    if drop_blurry and blurry:
        return "blurry"
    return ""
```

(Note: the test calls it with kwargs `corrupt, too_small, blurry, drop_small, drop_blurry`; match that signature — drop the unused `blurry_flag`.)

Final signature:

```python
def drop_reason(corrupt, too_small, blurry, drop_small, drop_blurry):
    if corrupt:
        return "corrupt"
    if drop_small and too_small:
        return "too_small"
    if drop_blurry and blurry:
        return "blurry"
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_01_ingest_clean.py::test_drop_reason_respects_flags -v`
Expected: PASS.

- [ ] **Step 5: Rewrite `main()` to use flags + guard OCR import** (replace lines 86-131):

```python
def main():
    cfg = common.load_config()
    ing = cfg["ingest"]
    raw = Path(cfg["paths"]["raw"])
    clean = Path(cfg["paths"]["clean"])
    clean.mkdir(parents=True, exist_ok=True)

    drop_small = ing.get("drop_small", False)
    drop_blurry = ing.get("drop_blurry", False)
    run_ocr = ing.get("run_ocr_flag", False)

    ocr = None
    if run_ocr:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

    all_imgs = list(common.iter_images(raw))
    LOG.info("Stage 1: %d raw images (drop_small=%s drop_blurry=%s run_ocr=%s)",
             len(all_imgs), drop_small, drop_blurry, run_ocr)

    survivors, rows = [], []
    for p in all_imgs:
        corrupt = is_corrupt(p)
        too_small = (not corrupt) and is_too_small(p, ing["min_size"])
        blurry = (not corrupt) and (blur_variance(p) < ing["blur_var_threshold"])
        reason = drop_reason(corrupt, too_small, blurry, drop_small, drop_blurry)
        if reason:
            rows.append({"path": str(p), "dropped": "True", "drop_reason": reason})
        else:
            survivors.append(p)

    keep, dup_drop = dedup(survivors, ing["phash_hamming_threshold"])
    for p in dup_drop:
        rows.append({"path": str(p), "dropped": "True", "drop_reason": "duplicate"})

    for p in keep:
        w, h = image_size(p)
        ratio = ocr_text_area_ratio(p, ocr) if run_ocr else 0.0
        flagged = run_ocr and ratio > ing["ocr_text_area_ratio_flag"]
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
```

- [ ] **Step 6: Run the full stage-1 test file**

Run: `python -m pytest tests/test_01_ingest_clean.py -v`
Expected: all PASS (old function tests + new flag test).

- [ ] **Step 7: Commit**

```bash
git add src/01_ingest_clean.py tests/test_01_ingest_clean.py
git commit -m "feat(stage1): tag-don't-filter — gate small/blur/OCR behind flags, keep corrupt-drop + dedup"
```

---

### Task 3: Stage 3 — tag-only captions (remove JoyCaption)

**Files:**
- Modify: `src/03_caption.py` (remove `JoyCaptioner`, `clean_nl`; change `assemble_caption`; simplify `main()`)
- Test: `tests/test_03_caption.py`

- [ ] **Step 1: Rewrite the caption tests** (replace `tests/test_03_caption.py` entirely):

```python
from conftest import load_stage

stage = load_stage("03_caption.py")

LABEL_MAP = {"SAFE": "safe", "QUESTIONABLE": "sensitive", "UNSAFE": "explicit"}


def test_assemble_caption_quality_then_safety():
    out = stage.assemble_caption(quality_tag="masterpiece, best quality", safety_tag="safe")
    assert out == "masterpiece, best quality, safe"


def test_assemble_caption_low_quality_explicit():
    out = stage.assemble_caption(quality_tag="low quality", safety_tag="explicit")
    assert out == "low quality, explicit"


def test_quality_tag_from_bucket():
    qmap = {"good": "masterpiece, best quality", "medium": "high quality", "bad": "low quality"}
    assert stage.quality_tag_for("good", qmap) == "masterpiece, best quality"
    assert stage.quality_tag_for("bad", qmap) == "low quality"


def test_map_nsfw_label_substring():
    assert stage.map_safety("SAFE", LABEL_MAP, "safe") == "safe"
    assert stage.map_safety("QUESTIONABLE_CONTENT", LABEL_MAP, "safe") == "sensitive"
    assert stage.map_safety("LABEL_UNSAFE", LABEL_MAP, "safe") == "explicit"
    assert stage.map_safety("weird_unknown", LABEL_MAP, "safe") == "safe"  # fallback
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: FAIL — `assemble_caption` still requires `trigger`/`nl` args.

- [ ] **Step 3: Rewrite `src/03_caption.py`** (full replacement):

```python
"""Stage 3: NSFW safety tag + quality tag -> tag-only caption "<quality>, <safety>". Augments manifest.

v2: tag-only (no JoyCaption NL, no trigger word). Domain-shift finetune bakes realism in broadly;
minimal captions are intentional. Quality words steer at inference; safety tag separates SFW/NSFW.
"""
import torch
from PIL import Image

import common

LOG = common.setup_logging()


# ---- pure logic (unit-tested) ----

def quality_tag_for(bucket, quality_tag_map):
    return quality_tag_map[bucket]


def map_safety(model_label, label_map, default_tag):
    up = model_label.upper()
    for key in sorted(label_map, key=len, reverse=True):  # longest-first: "UNSAFE" before "SAFE"
        if key.upper() in up:
            return label_map[key]
    return default_tag


def assemble_caption(quality_tag, safety_tag):
    return f"{quality_tag}, {safety_tag}"


# ---- model wrapper (smoke-tested) ----

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
    LOG.info("Stage 3: tagging %d images (tag-only captions)", len(kept))

    nsfw = NSFWTagger(cfg)
    updates = {}
    for r in kept:
        bucket = r.get("bucket")
        if not bucket:
            raise RuntimeError(f"No bucket for {r['path']} - run stage 2 (02_quality_score) first.")
        qtag = quality_tag_for(bucket, cap_cfg["quality_tag_map"])
        stag = nsfw.tag(r["path"])
        caption = assemble_caption(qtag, stag)
        updates[r["path"]] = {"safety_tag": stag, "quality_tag": qtag, "caption": caption}

    common.augment_manifest(cfg["paths"]["manifest"], updates)
    LOG.info("Stage 3 done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_03_caption.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/03_caption.py tests/test_03_caption.py
git commit -m "feat(stage3): tag-only captions — remove JoyCaption, drop trigger/NL"
```

---

### Task 4: Stage 4 — diffusion-pipe dataset.toml + keep all buckets

**Files:**
- Modify: `src/04_build_dataset.py` (replace `write_dataset_toml`, update `main()` to read new `dataset` keys)
- Test: `tests/test_04_build_dataset.py`

- [ ] **Step 1: Rewrite the tests** (replace `tests/test_04_build_dataset.py` entirely):

```python
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from conftest import load_stage

stage = load_stage("04_build_dataset.py")


def test_curate_keeps_all_three_buckets():
    rows = [
        {"path": "a.jpg", "dropped": "False", "bucket": "good", "caption": "c1"},
        {"path": "b.jpg", "dropped": "False", "bucket": "medium", "caption": "c2"},
        {"path": "c.jpg", "dropped": "False", "bucket": "bad", "caption": "c3"},
        {"path": "d.jpg", "dropped": "True", "bucket": "good", "caption": "c4"},
    ]
    kept = stage.curate(rows, buckets_to_keep=["good", "medium", "bad"])
    assert {r["path"] for r in kept} == {"a.jpg", "b.jpg", "c.jpg"}  # only the dropped one excluded


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: FAIL — `write_dataset_toml` has the old kohya signature/schema.

- [ ] **Step 3: Replace `write_dataset_toml` and update `main()`** in `src/04_build_dataset.py`:

Replace the `write_dataset_toml` function (lines 22-42) with:

```python
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
```

Replace the body of `main()` from the `toml_path = ...` line (lines 62-65) with:

```python
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
```

(The `dest` wipe/copy logic above it is unchanged. `dest` is the LOCAL build dir from `paths.dataset`; the emitted toml points at the VAST path `base_dir/data/dataset` where you upload it.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_04_build_dataset.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/04_build_dataset.py tests/test_04_build_dataset.py
git commit -m "feat(stage4): emit diffusion-pipe dataset.toml, keep all quality buckets"
```

---

### Task 5: Stage 5 — diffusion-pipe anima.toml (full finetune)

**Files:**
- Modify: `src/05_make_train_config.py` (full replacement)
- Test: `tests/test_05_make_train_config.py`

- [ ] **Step 1: Rewrite the tests** (replace `tests/test_05_make_train_config.py` entirely):

```python
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

from conftest import load_stage

stage = load_stage("05_make_train_config.py")


def test_anima_toml_valid_and_full_finetune(tmp_path):
    out = tmp_path / "anima.toml"
    stage.write_anima_toml(
        out,
        base_dir="/workspace/anima",
        project_name="anima_realism_ft_v1",
        dit_file="anima-base-v1.0.safetensors",
        vae_file="qwen_image_vae.safetensors",
        qwen_file="qwen_3_06b_base.safetensors",
        epochs=10, lr=8.0e-6, optimizer="AdamW8bitKahan", warmup_steps=100,
        save_every_n_epochs=1, checkpoint_every_n_minutes=30,
        llm_adapter_lr=0, tag_dropout_percent=0.10, caption_dropout_percent=0.05,
    )
    d = tomllib.loads(out.read_text(encoding="utf-8"))
    # full finetune => NO [adapter] block
    assert "adapter" not in d
    assert d["model"]["type"] == "anima"
    assert d["model"]["transformer_path"] == "/workspace/anima/models/anima-base-v1.0.safetensors"
    assert d["model"]["qwen_path"] == "/workspace/anima/models/qwen_3_06b_base.safetensors"
    assert d["model"]["vae_path"] == "/workspace/anima/models/qwen_image_vae.safetensors"
    assert d["model"]["llm_adapter_lr"] == 0          # freeze Qwen3 adapter
    assert d["model"]["caption_mode"] == "tags"        # tag-only
    assert d["model"]["cache_text_embeddings"] is False
    assert d["epochs"] == 10
    assert d["activation_checkpointing"] is True
    assert d["optimizer"]["type"] == "AdamW8bitKahan"
    assert d["optimizer"]["lr"] == 8.0e-6
    assert d["dataset"].endswith("anima_realism_ft_v1_dataset_config.toml")


def test_anima_toml_paths_use_forward_slashes(tmp_path):
    out = tmp_path / "anima.toml"
    stage.write_anima_toml(
        out, base_dir="/workspace/anima", project_name="p",
        dit_file="a.safetensors", vae_file="v.safetensors", qwen_file="q.safetensors",
        epochs=1, lr=8e-6, optimizer="AdamW8bitKahan", warmup_steps=1,
        save_every_n_epochs=1, checkpoint_every_n_minutes=30,
        llm_adapter_lr=0, tag_dropout_percent=0.1, caption_dropout_percent=0.05,
    )
    text = out.read_text(encoding="utf-8")
    assert "\\\\" not in text  # no backslashes in any path
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_05_make_train_config.py -v`
Expected: FAIL — `write_anima_toml` does not exist.

- [ ] **Step 3: Rewrite `src/05_make_train_config.py`** (full replacement):

```python
"""Stage 5: emit diffusion-pipe anima.toml (FULL finetune) for the Anima model.

Full finetune = NO [adapter] block. Tag-only captions (caption_mode='tags').
llm_adapter_lr=0 freezes the Qwen3 LLM adapter. Paths are absolute Vast (Linux) paths.
"""
from pathlib import Path

import common

LOG = common.setup_logging()


def write_anima_toml(out_path, base_dir, project_name, dit_file, vae_file, qwen_file,
                     epochs, lr, optimizer, warmup_steps, save_every_n_epochs,
                     checkpoint_every_n_minutes, llm_adapter_lr,
                     tag_dropout_percent, caption_dropout_percent):
    base = base_dir.rstrip("/")
    out_dir = f"{base}/outputs/{project_name}"
    dataset_toml = f"{base}/outputs/{project_name}_dataset_config.toml"
    models = f"{base}/models"
    toml = f"""# diffusion-pipe — Anima FULL finetune (realism domain shift). Generated by stage 5.
output_dir = '{out_dir}'
dataset = '{dataset_toml}'

epochs = {epochs}
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
gradient_clipping = 1.0
warmup_steps = {warmup_steps}
activation_checkpointing = true

# Eval disabled for run 1 (no eval set). Preview epoch checkpoints in ComfyUI instead.
eval_before_first_step = false
eval_every_n_epochs = 100000

save_every_n_epochs = {save_every_n_epochs}
checkpoint_every_n_minutes = {checkpoint_every_n_minutes}
save_dtype = 'bfloat16'

[model]
type = 'anima'
transformer_path = '{models}/{dit_file}'
vae_path = '{models}/{vae_file}'
qwen_path = '{models}/{qwen_file}'
dtype = 'bfloat16'
llm_adapter_lr = {llm_adapter_lr}
cache_text_embeddings = false
shuffle_tags = true
tag_delimiter = ', '
shuffle_keep_first_n = 1
tag_dropout_percent = {tag_dropout_percent}
caption_dropout_percent = {caption_dropout_percent}
caption_mode = 'tags'
timestep_sample_method = 'logit_normal'

# NOTE: no [adapter] block => full finetune (not LoRA).

[optimizer]
type = '{optimizer}'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = 0.01
stabilize = false

[monitoring]
enable_wandb = false
"""
    Path(out_path).write_text(toml, encoding="utf-8")


def main():
    cfg = common.load_config()
    f = cfg["finetune"]
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    toml_path = out / f"{f['project_name']}_train_config.toml"
    write_anima_toml(
        toml_path,
        base_dir=f["base_dir"], project_name=f["project_name"],
        dit_file=f["dit_file"], vae_file=f["vae_file"], qwen_file=f["qwen_file"],
        epochs=f["epochs"], lr=f["lr"], optimizer=f["optimizer"], warmup_steps=f["warmup_steps"],
        save_every_n_epochs=f["save_every_n_epochs"],
        checkpoint_every_n_minutes=f["checkpoint_every_n_minutes"],
        llm_adapter_lr=f["llm_adapter_lr"],
        tag_dropout_percent=f["tag_dropout_percent"],
        caption_dropout_percent=f["caption_dropout_percent"],
    )
    LOG.info("Stage 5 done. anima.toml -> %s", toml_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_05_make_train_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/05_make_train_config.py tests/test_05_make_train_config.py
git commit -m "feat(stage5): emit diffusion-pipe anima.toml for full finetune"
```

---

### Task 6: requirements.txt — drop JoyCaption-only dep

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Remove the `bitsandbytes` line** (JoyCaption 4-bit only — no longer used). Also drop the JoyCaption comment on the transformers line; leave transformers (CLIP + NSFW classifier need it). Replace these two lines:

```
transformers>=4.45,<5    # CLIP aesthetic + NSFW classifier (kept <5 for stability)
sentencepiece>=0.2
```

(remove the `bitsandbytes>=0.44 ...` line entirely.)

- [ ] **Step 2: Verify file still lists pillow/transformers/imagehash/opencv/pyyaml/tomli**

Run: `python -c "print(open('requirements.txt',encoding='utf-8').read())"`
Expected: no `bitsandbytes`; transformers/CLIP deps present.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: drop bitsandbytes (was JoyCaption-only)"
```

---

### Task 7: Vast setup script

**Files:**
- Create: `scripts/vast_setup.sh`

- [ ] **Step 1: Write `scripts/vast_setup.sh`**

```bash
#!/usr/bin/env bash
# One-shot setup on a fresh Vast.ai Linux instance for Anima full finetune.
# Usage: bash scripts/vast_setup.sh
set -euo pipefail

BASE="${ANIMA_BASE:-/workspace/anima}"
DP_DIR="$BASE/diffusion-pipe"
MODELS="$BASE/models"
HF="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"

mkdir -p "$MODELS" "$BASE/data" "$BASE/outputs"

# 1) diffusion-pipe (Anima fork) + submodules
if [ ! -d "$DP_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/bluvoll/diffusion-pipe "$DP_DIR"
fi
cd "$DP_DIR"
git submodule update --init --recursive

# 2) Python deps (instance image already has CUDA torch). deepspeed + diffusion-pipe reqs.
pip install --upgrade pip
pip install deepspeed
pip install -r requirements.txt

# 3) Anima models (~5.6 GB)
wget -c -O "$MODELS/anima-base-v1.0.safetensors" "$HF/diffusion_models/anima-base-v1.0.safetensors"
wget -c -O "$MODELS/qwen_3_06b_base.safetensors"  "$HF/text_encoders/qwen_3_06b_base.safetensors"
wget -c -O "$MODELS/qwen_image_vae.safetensors"   "$HF/vae/qwen_image_vae.safetensors"

echo "Setup done. Models in $MODELS ; diffusion-pipe in $DP_DIR"
```

- [ ] **Step 2: Syntax-check**

Run: `bash -n scripts/vast_setup.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/vast_setup.sh
git commit -m "feat: Vast setup script (clone diffusion-pipe fork + download Anima models)"
```

---

### Task 8: Prep runner script

**Files:**
- Create: `scripts/run_prep.sh`

- [ ] **Step 1: Write `scripts/run_prep.sh`**

```bash
#!/usr/bin/env bash
# Run the prep pipeline (stages 1-5) on Vast. Produces data/dataset + the two TOMLs.
# Assumes a pipeline venv with CUDA torch + requirements.txt installed, and data/raw populated.
# Usage: bash scripts/run_prep.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

python src/01_ingest_clean.py
python src/02_quality_score.py
python src/03_caption.py
python src/04_build_dataset.py
python src/05_make_train_config.py

echo "Prep done. Review outputs/*_train_config.toml and outputs/*_dataset_config.toml,"
echo "then copy data/dataset to \$ANIMA_BASE/data/dataset before launching training."
```

- [ ] **Step 2: Syntax-check**

Run: `bash -n scripts/run_prep.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_prep.sh
git commit -m "feat: prep runner script (stages 1-5)"
```

---

### Task 9: RUNBOOK.md

**Files:**
- Create: `RUNBOOK.md`

- [ ] **Step 1: Write `RUNBOOK.md`** (exact content below):

````markdown
# RUNBOOK — Anima Realism Full Finetune on Vast.ai

End-to-end first run. All prep + training happen on one rented Linux GPU.

## 0. Prerequisites (do locally, before renting)

- A folder of your photos (any sizes/formats). Zip it: `dataset_raw.zip`.
- A Vast.ai account with credit (~$10). A 512px run is ~2–3 h on one 48 GB GPU (~$1.5–2).
- Host `dataset_raw.zip` somewhere with a direct download link (Google Drive direct link,
  Dropbox `?dl=1`, S3, or `huggingface-cli upload` to a private dataset repo). This is the
  simplest way to get it onto an ephemeral instance. (Alternative: `scp` after the instance is up.)

## 1. Rent a GPU on Vast.ai

1. vast.ai → **Search**.
2. Filters: **GPU RAM ≥ 48 GB** (A6000 / L40S / A100-40 or 80). Disk **≥ 80 GB**. Reliability high.
3. **Template:** choose a PyTorch CUDA image, e.g. `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
   (or a Vast "PyTorch" recommended template). On-demand (not interruptible) for the first run.
4. Rent → open the instance's **Jupyter** or **SSH**.

## 2. Get this repo + your data onto the instance

```bash
export ANIMA_BASE=/workspace/anima
mkdir -p $ANIMA_BASE && cd $ANIMA_BASE
git clone <YOUR_REPO_URL> repo        # this project (the pipeline + scripts)
cd repo

# your photos:
mkdir -p data/raw
cd data/raw
wget -O dataset_raw.zip "<YOUR_DIRECT_ZIP_LINK>"
unzip -q dataset_raw.zip && rm dataset_raw.zip
# flatten if the zip made a subfolder:
find . -mindepth 2 -type f -exec mv -t . {} + 2>/dev/null || true
cd "$ANIMA_BASE/repo"
ls data/raw | head        # sanity: your images are here
```

## 3. Install diffusion-pipe + download Anima models

```bash
bash scripts/vast_setup.sh        # clones fork, pip installs, wgets the 3 model files (~5.6 GB)
```

## 4. Install the prep pipeline deps (separate from training)

```bash
python -m venv $ANIMA_BASE/prepvenv
source $ANIMA_BASE/prepvenv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 5. Run prep (dedup → score → tag → build dataset + configs)

```bash
bash scripts/run_prep.sh
```

This writes:
- `data/dataset/` — flat folder of kept images + `.txt` tag captions.
- `outputs/anima_realism_ft_v1_dataset_config.toml`
- `outputs/anima_realism_ft_v1_train_config.toml`

**Sanity check a few captions** (should look like `masterpiece, best quality, safe`):
```bash
head -c 200 $(ls data/dataset/*.txt | head -1); echo
```

Move the dataset where the config expects it, and copy the configs into diffusion-pipe:
```bash
mkdir -p $ANIMA_BASE/data $ANIMA_BASE/outputs
cp -r data/dataset $ANIMA_BASE/data/dataset
cp outputs/anima_realism_ft_v1_*.toml $ANIMA_BASE/outputs/
```

## 6. Launch the finetune

```bash
deactivate 2>/dev/null || true        # leave prepvenv; use the instance's training torch
cd $ANIMA_BASE/diffusion-pipe
deepspeed --num_gpus=1 train.py --deepspeed \
  --config $ANIMA_BASE/outputs/anima_realism_ft_v1_train_config.toml
```

Watch the loss. Checkpoints save every epoch to
`$ANIMA_BASE/outputs/anima_realism_ft_v1/` and every 30 min.

## 7. Preview a checkpoint

diffusion-pipe in-training image eval is OFF for run 1. To preview: load an epoch checkpoint
in **ComfyUI** with the Anima workflow (DiT + `qwen_3_06b_base` TE + `qwen_image_vae`) and prompt
`masterpiece, best quality, safe`. Compare against the base model to see the realism shift.

## 8. Retrieve your model before stopping the instance

```bash
# from your local machine:
scp -P <SSH_PORT> root@<INSTANCE_IP>:$ANIMA_BASE/outputs/anima_realism_ft_v1/'*.safetensors' .
# or push to HuggingFace from the instance:
#   huggingface-cli login && huggingface-cli upload <you>/anima-realism outputs/anima_realism_ft_v1
```

**Then DESTROY the instance** (Vast bills while it exists, even stopped, for storage).

## Troubleshooting

- **CUDA OOM:** lower nothing first — confirm `activation_checkpointing = true` in the toml.
  Then in `outputs/..._train_config.toml` set `[model] qwen_nf4 = true`, or switch `[optimizer] type`
  to `CAME`. Last resort: drop `resolutions = [512]` to `[448]` in the dataset toml.
- **"qwen_path" load error:** confirm `models/qwen_3_06b_base.safetensors` downloaded fully
  (re-run `scripts/vast_setup.sh`; `wget -c` resumes). If the loader insists on an HF dir,
  `huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir models/Qwen3-0.6B` and set
  `qwen_path` to that folder.
- **NSFW tags all "safe":** the classifier's `id2label` strings may differ from the substrings in
  `config/pipeline.yaml` (`caption.nsfw_label_map`). Print them once:
  `python -c "from transformers import AutoModelForImageClassification as M; print(M.from_pretrained('MichalMlodawski/nsfw-image-detection-large').config.id2label)"`
  and adjust the map substrings.
- **Instance interrupted mid-train:** re-rent, re-run setup, and resume with diffusion-pipe's
  `--resume_from_checkpoint` pointing at the last saved global step dir.
````

- [ ] **Step 2: Commit**

```bash
git add RUNBOOK.md
git commit -m "docs: RUNBOOK for Vast full-finetune first run"
```

---

### Task 10: Full suite + retire dead artifacts

**Files:**
- Delete: `scripts/06_train.ps1`, `scripts/download_models.ps1` (Windows-local LoRA path, superseded)
- Modify: `README.md` (point to RUNBOOK + spec v2) — optional if time-boxed

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest -v`
Expected: all PASS (stages 1-5 + common). No reference to removed `JoyCaptioner`/`build_sample_prompts`.

- [ ] **Step 2: Remove the superseded Windows trainer scripts**

```bash
git rm scripts/06_train.ps1 scripts/download_models.ps1
```

(Vast/Linux replaces them; `scripts/vast_setup.sh` + `RUNBOOK.md` are the new path.)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: retire Windows-local LoRA scripts (superseded by Vast runbook)"
```

---

## Self-Review

**Spec coverage:**
- Full finetune via diffusion-pipe → Tasks 5, 7, 9. ✅
- Vast infra → Tasks 7, 9. ✅
- Tag-only captions, no trigger → Task 3. ✅
- CLIP-score → native quality words → stage 2 unchanged + Task 1 (`quality_tag_map`). ✅
- Tag-don't-filter (corrupt+dedup only) → Task 2. ✅
- 768→512 resolution correction (diffusion-pipe upscales-to-area) → Task 1 (`resolutions=[512]`) + Task 4 + RUNBOOK troubleshooting. ✅
- Safety tagging kept (hard boundary) → Task 3 (`NSFWTagger`). ✅
- `llm_adapter_lr=0` freeze → Task 5. ✅
- Model path resolution (qwen single-file ok) → Task 5 + RUNBOOK troubleshooting. ✅

**Placeholder scan:** none — every code/config/script step shows full content.

**Type/signature consistency:**
- `assemble_caption(quality_tag, safety_tag)` — defined Task 3, used Task 3 `main()`. ✅
- `write_dataset_toml(out_path, image_dir, resolutions, min_ar, max_ar, num_ar_buckets, num_repeats)` — Task 4 def + test + `main()` call match. ✅
- `write_anima_toml(...)` — Task 5 def signature matches test calls and `main()` call (same kwarg names). ✅
- `drop_reason(corrupt, too_small, blurry, drop_small, drop_blurry)` — Task 2 final signature matches test + `main()`. ✅
- Config keys (`finetune.*`, `dataset.resolutions/min_ar/...`, `ingest.drop_*`) introduced in Task 1, consumed in Tasks 2/4/5. ✅

**Open item carried to runbook (not a blocker):** NSFW `id2label` exact strings — verified at first live run (troubleshooting note + one-liner provided).
