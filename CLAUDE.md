# CLAUDE.md — Anima Realism LoRA project

> Portable project memory. Lives in the project folder so it survives moving to another drive.
> Full design lives in `docs/superpowers/specs/2026-05-31-anima-realism-lora-design.md`.

## Goal

Train a **realism domain-shift LoRA** for the **Anima** diffusion model (anime base) so it outputs
realistic photographs. Community realism finetunes of Anima already exist on Civitai, so it works —
it's a domain shift fought against the anime prior.

- **Phase 1 (current):** prove pipeline + data + captions with a LoRA, on **local RTX 4080 (16GB), Windows 11**.
- **Phase 2 (future, separate spec):** full finetune on all ~3000 images, on RunPod.

## Status (2026-05-31)

**Pipeline BUILT and merged to `master`** (subagent-driven exec, 13 task commits, **19 unit tests passing**).
All 6 stages + `common.py` + `download_models.ps1` + `06_train.ps1` + README in place. Build-time open items
(§12) resolved: NSFW = `MichalMlodawski/nsfw-image-detection-large` (3-class), captioner = JoyCaption **4-bit nf4**
(bf16's ~17GB won't fit 16GB), aesthetic = `improved-aesthetic-predictor` (CLIP-L/14 + MLP), stage 6 auto-clones+setups trainer.

**Built offline with CPU torch + pure-logic tests only.** NOT yet run live. **Next step: live run on the 4080** —
`python -m venv .venv` then `pip install torch torchvision --index-url .../cu128` + `pip install -r requirements.txt`,
then stages 1→5, `download_models.ps1`, `06_train.ps1`. Two checks still deferred to first live run (flagged in
plan): NSFW model `id2label` exact strings (→ adjust `caption.nsfw_label_map` if needed) and the trainer venv
activate path inside `06_train.ps1` (setup_env.bat may name the venv differently).

## Dataset

- ~3000 photos from social media (Reddit, X, Threads) → expect JPEG artifacts, watermarks, text
  overlays, screenshots/memes, heavy near-duplicate reposts, wild aspect ratios, mixed quality.
- **NSFW present.** Handled by **safety-tagging, never filtering**. Hard boundary: **legal adult
  content only** (real adults, consensual; no minors / non-consensual).
- Phase-1 LoRA curates to **best ~500–800** (good+medium buckets; drop "bad"). Style/domain LoRA
  sweet spot ≈ 500. Full 3000 → Phase 2.

## Anima model facts

- DiT (Diffusion Transformer) **2B**, base = NVIDIA **Cosmos-Predict2-2B-Text2Image** (photoreal-capable).
- **Text encoder = Qwen3-0.6B** (`qwen_3_06b_base.safetensors`) — an LLM, not CLIP.
- **VAE = Qwen-Image VAE** (`qwen_image_vae.safetensors`).
- DiT weights = `anima-base-v1.0.safetensors`.
- Anime/illustration model; **not natively photoreal** (`❌ Photorealism` on model card).
- License: CircleStone Labs Non-Commercial.
- **Don't train the LLM adapter** — for LoRA this = `network_train_unet_only = true` (freezes Qwen3 TE).

### Model download URLs (HF `circlestone-labs/Anima`, prefix `resolve/main/`)
| Part | File | Size | Path |
|------|------|------|------|
| DiT | `anima-base-v1.0.safetensors` | 4.18 GB | `split_files/diffusion_models/anima-base-v1.0.safetensors` |
| TE | `qwen_3_06b_base.safetensors` | 1.19 GB | `split_files/text_encoders/qwen_3_06b_base.safetensors` |
| VAE | `qwen_image_vae.safetensors` | 254 MB | `split_files/vae/qwen_image_vae.safetensors` |

## Trainer

- **Local backend:** [gazingstars123/Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)
  (Windows `setup_env.bat`, sd-scripts based, ships `anima_train_network.py`). Run **headless** — skip its Web UI.
- **Config reference:** notebook `Copy of ANIMA_Trainer_v5.ipynb`
  (repo `citronlegacy/citron-colab-anima-lora-trainer`) — gives the exact TOML schema + invocation.
- **Invocation:**
  `accelerate launch anima_train_network.py --config_file <train.toml> --dataset_config <data.toml>`
  with `network_module = networks.lora_anima`.
- Notebook's `<1000 steps` rule is a **Colab disconnect limit — does NOT apply locally.**

## Caption format

```
<quality tags>, <safety tag>, realistic photo, <natural-language description>
```
e.g. `masterpiece, best quality, safe, realistic photo, a woman on a park bench at golden hour, 35mm`

- Quality tags from aesthetic-score bucket; safety tag from NSFW classifier; `realistic photo` = trigger.
- **Captioner = JoyCaption** (NSFW-capable, runs on 4080). Florence-2 / Qwen2.5-VL rejected (censored).
- Captioner ≠ text encoder: Qwen3 TE encodes whatever text is written; no benefit to "matching" captioner to TE.

## Pipeline — 6 local stages (each one script, idempotent, reads prior stage's dir)

1. `src/01_ingest_clean.py` — phash dedup, drop tiny/blurry/corrupt, OCR-flag meme/screenshot text → `data/clean/`
2. `src/02_quality_score.py` — CLIP aesthetic score → good/medium/bad buckets (this is how mixed quality becomes useful: bad → `low quality` tag, not discarded)
3. `src/03_caption.py` — JoyCaption NL + NSFW safety tag + quality tag → assembled caption
4. `src/04_build_dataset.py` — copy curated subset, write `img.txt` sidecars (flat folder), emit dataset TOML
5. `src/05_make_train_config.py` — emit training TOML
6. `scripts/06_train.ps1` — setup Standalone-Trainer, download models, `accelerate launch`, fixed-seed sample previews

Config: all paths + thresholds in `config/pipeline.yaml`. Two venvs: pipeline (cleaning/captioning) vs trainer (`setup_env.bat`, PyTorch 2.7 cu128).

## Key hyperparameters (16GB recipe)

- Fit-16GB: `cache_latents=true` + `cache_text_encoder_outputs=true` (precompute+offload VAE/TE) + `gradient_checkpointing` + `bf16` + `AdamW8bit` + batch 1.
- Start: **dim/alpha 32, res 768, lr 1e-4, repeats 5, epochs 10**. OOM fallback: **dim 8, res 512**.
- lr 1e-4 = notebook proven default (dim~20); 2e-5 = conservative model-card value.
- Cached latents pin resolution (no random-crop aug) → re-cache if res changes.

## Open items (resolve at build time)

- JoyCaption Windows install (~8–12GB weights; confirm bitsandbytes CUDA wheel on Win).
- Aesthetic predictor pick (lightest that runs on 4080).
- NSFW/safety classifier pick (photo-domain).
- Confirm `networks.lora_anima` present after `setup_env.bat`.

## Caveats after folder move

- Paths in the spec/scripts assume project root; update if drive letter changes.
- The `~/.claude/projects/.../memory/` store is keyed to the **old** path and won't auto-load from the
  new location — **this CLAUDE.md is the durable record.**
