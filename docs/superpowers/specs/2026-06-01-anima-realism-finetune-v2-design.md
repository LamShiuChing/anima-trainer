# Anima Realism **Finetune** (v2, cloud) — design

> Supersedes the Phase-1 LoRA design (`2026-05-31-anima-realism-lora-design.md`) for the actual first run.
> Durable record — survives folder/drive moves. See also `CLAUDE.md`.

## Goal

Train a **realism domain-shift** of the **Anima** diffusion model (anime base → photoreal output)
via **full finetune** (not LoRA), on a **rented cloud GPU (Vast.ai)**.

## What changed from v1 (and why)

| v1 (Phase 1) | v2 (this run) | Reason |
|---|---|---|
| LoRA (dim 32) | **Full finetune** | Anime→photoreal is a domain shift; low-rank deltas can't relocate the output distribution. |
| Local RTX 4080 (16 GB) | **Vast.ai, single ~48 GB GPU** | Full finetune needs ~33 GB VRAM @ 768px — cannot fit 16 GB, regardless of speed. |
| JoyCaption NL captions | **Tag-only captions** (no VLM) | Minimal captions bake realism in broadly; VLM was the slow stage. User goal = speed + style baking. |
| No trigger word | (still has trigger in v1) → **no trigger** | Full finetune shifts the whole model realistic; a trigger is optional and user opted out. |
| Aesthetic-score *dropping* (good/medium kept, bad dropped) | **Tag everything, drop nothing** for quality | "Use whatever I upload." Mixed quality becomes a `low quality` tag, not a discard. |
| 6 stages, kohya/sd-scripts + gazingstars trainer | **4 stages + diffusion-pipe** | diffusion-pipe is the framework with Anima full-finetune support. |

## Anima model facts (unchanged)

- DiT **2B**, base = NVIDIA **Cosmos-Predict2-2B-Text2Image**. Native train res **512**.
- Text encoder = **Qwen3-0.6B** (LLM, not CLIP). VAE = **Qwen-Image VAE**.
- Files (HF `circlestone-labs/Anima`, prefix `resolve/main/split_files/`):
  DiT `diffusion_models/anima-base-v1.0.safetensors` (4.18 GB);
  TE `text_encoders/qwen_3_06b_base.safetensors` (1.19 GB);
  VAE `vae/qwen_image_vae.safetensors` (254 MB).
- License: CircleStone Labs Non-Commercial.

## Trainer — diffusion-pipe

- Framework: **`diffusion-pipe`** (tdrussell). Anima/Cosmos-Predict2 support added via
  bluvoll/duongve — [tdrussell/diffusion-pipe PR #505](https://github.com/tdrussell/diffusion-pipe/pull/505)
  (ref fork: `bluvoll/diffusion-pipe`). **Confirm exact repo/branch at build time.**
- Invocation (full finetune): `deepspeed --num_gpus=1 train.py --deepspeed --config anima.toml`.
- Authors' tested numbers: **~31 GB @ 512px, ~33 GB @ 1024px** with gradient checkpointing (2×RTX 4090).
- Configs (`anima.toml`, `dataset.toml`) ship with the repo as commented templates.

### Hyperparameters (full finetune, 16GB-irrelevant — cloud)

- LR **8e-6** (lower than LoRA's 1e-4; full finetune is sensitive — too high → catastrophic forgetting).
- batch 1, gradient_accumulation as VRAM allows, **AdamW8bit**, **bf16**, **gradient_checkpointing on**.
- `llm_adapter_lr = 0` → **freeze the Qwen3 adapter** (finetune equivalent of v1's `network_train_unet_only`).
- **Resolution 768, `bucket_no_upscale` on**, aspect-ratio bucketing on (handles wild aspect ratios; small
  social images train at native bucket, never upscaled → no soft/upscaled texture baked into "realism").
- `num_repeats = 1`, `epochs ≈ 10` (lots of data; finetune doesn't need high repeats). Tune after run 1.
- Save checkpoint **every epoch**, keep last few (Vast instances can be interrupted).
- Caption dropout ~0.1 (some steps see empty caption → unconditional realism baking).

## Where it runs — **all on Vast**

Prep + train both run on the rented Linux GPU. Rationale: prep (CLIP score + NSFW tag) is cheap in
GPU-time (~minutes, <$1), and running on the clean Linux Docker image sidesteps untested Windows
venv issues (bitsandbytes CUDA wheel on Win, etc.). **What the user uploads = just the raw photos.**
Models are pulled on-instance via `wget` from HF.

## Pipeline — 4 stages (Python, run on Vast via bash runbook)

The existing `src/0X_*.py` Python is cross-platform; only the `.ps1` launchers are Windows-specific.
A new bash runbook drives the `.py` stages on Linux. Changes from v1 scripts:

1. **S1 `01_ingest_clean.py`** — keep **corrupt/undecodable drop** + **near-dup phash dedup** only.
   Disable blur-drop and OCR meme-drop (config flags → off). → `data/clean/`
2. **S2 `02_quality_score.py`** — CLIP aesthetic → good/medium/bad bucket. **Tag only, drop nothing.**
3. **S3 `03_caption.py`** — **remove JoyCaption.** Assemble tag-only caption:
   `<quality words>, <safety tag>`. Quality words from S2 bucket
   (`masterpiece, best quality` / `high quality` / `low quality`). Safety tag from fast NSFW classifier
   (`safe` / `sensitive` / `explicit`). **No NL, no trigger word.** Write `img.txt` sidecars.
4. **S4 `04_build_dataset.py` + new config emitter** — flat dataset dir (img + matching `.txt`),
   emit diffusion-pipe **`dataset.toml`** + **`anima.toml`**. (v1 stages 05/06 + `.ps1` launchers retired.)

## Caption format (final)

```
masterpiece, best quality, safe
```
One quality phrase + one safety tag. No trigger, no NL. ~9 combos (3 quality × 3 safety).
Low caption diversity is acceptable: this is a domain shift, not concept learning. The quality phrase
is a steering knob at inference; safety tag separates SFW/NSFW.

## Safety boundary (unchanged, hard)

Safety-**tag**, never filter. **Legal adult content only** — real adults, consensual; no minors /
non-consensual. NSFW classifier = `MichalMlodawski/nsfw-image-detection-large` (fast image classifier).

## Outputs

Full-finetuned `.safetensors` per epoch → keep last few. Fixed-seed sample previews each epoch to
watch realism emerge. Final checkpoint downloaded off Vast before instance teardown.

## Deliverables of the build

- Adapted `src/01..04` (+ config flags in `config/pipeline.yaml`).
- diffusion-pipe `anima.toml` + `dataset.toml` templates.
- **`RUNBOOK.md`** — exact step-by-step: Vast account/credit → pick GPU → launch instance →
  upload photos → clone diffusion-pipe (Anima fork) → download Anima models → run S1–S4 prep →
  launch finetune → monitor → retrieve checkpoint.
- Updated unit tests for changed caption assembly + filtering.

## Open items (resolve at build time)

- diffusion-pipe Anima fork: exact repo + branch/commit that has PR #505 merged (confirm live).
- NSFW model `id2label` exact strings → adjust `caption.nsfw_label_map` substrings if needed.
- Vast data persistence: ephemeral disk vs attached volume vs host-a-zip-and-wget. Pick in runbook.
- Epoch/repeat count: 10/1 is a starting guess; tune after first run's preview quality.
- `anima.toml` exact key names (diffusion-pipe schema differs from kohya TOML) — read from repo template.
