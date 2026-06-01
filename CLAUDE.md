# CLAUDE.md — Anima Realism LoRA project

> Portable project memory. Lives in the project folder so it survives moving to another drive.
> Full design lives in `docs/superpowers/specs/2026-05-31-anima-realism-lora-design.md`.

## Goal

Train a **realism domain-shift LoRA** for the **Anima** diffusion model (anime base) so it outputs
realistic photographs. Community realism finetunes of Anima already exist on Civitai, so it works —
it's a domain shift fought against the anime prior.

- **~~Phase 1: LoRA on local 4080~~ — SUPERSEDED.** See Status below: pivoted to full finetune on Vast.ai.
- **Current:** full finetune on all uploaded photos via **diffusion-pipe** on a **rented ~48GB cloud GPU (Vast.ai)**.

## Status (2026-06-01) — PIVOTED to v2: full finetune on Vast

**Major pivot (user decision):** LoRA→**full finetune**, local 4080→**rented cloud GPU (Vast.ai)**,
JoyCaption NL→**tag-only captions**, drop-filter→**tag-don't-filter**, trigger word→**none**.
Reason: anime→photoreal is a domain shift LoRA can't fully relocate; full finetune needs ~33GB VRAM
(won't fit 16GB regardless of speed). **v2 spec:** `docs/superpowers/specs/2026-06-01-anima-realism-finetune-v2-design.md`;
**plan:** `docs/superpowers/plans/2026-06-01-anima-realism-finetune-v2.md`; **runbook:** `RUNBOOK.md`.

**v2 BUILT + merged to `master`** (plan-driven, 10 tasks, **20 unit tests passing**). Pipeline slimmed
4-stage: S1 ingest (corrupt-drop + phash dedup only; small/blur/OCR gated off via `ingest.drop_*` flags),
S2 CLIP aesthetic score (tags all, drops none), S3 NSFW safety tag + quality words → caption `"<quality>, <safety>"`
(no JoyCaption, no trigger), S4+S5 emit **diffusion-pipe** `dataset.toml` + `anima.toml` (NOT kohya).
Trainer is now **diffusion-pipe** (`bluvoll/diffusion-pipe`, full-finetune support via tdrussell PR #505),
launched `deepspeed --num_gpus=1 train.py --deepspeed --config anima.toml`. Old `06_train.ps1` +
`download_models.ps1` deleted (superseded by `scripts/vast_setup.sh` + `scripts/run_prep.sh` + RUNBOOK).
Anima loader confirmed: `qwen_path` accepts the single `qwen_3_06b_base.safetensors`; `vae_path` via WanVAE
class (Qwen-Image VAE is Wan-derived); `llm_adapter_lr=0` freezes the Qwen3 adapter.

**Still NOT run live.** Configs emit + parse correctly (verified). First live run = follow `RUNBOOK.md` on Vast.
Deferred to first live run: NSFW `id2label` exact strings (→ adjust `caption.nsfw_label_map`); diffusion-pipe
in-train image eval is OFF (preview epoch checkpoints in ComfyUI); confirm bluvoll fork branch still has Anima.
**Resolution = 512 for run 1** (diffusion-pipe resizes to target AREA / upscales smaller imgs — no no-upscale flag —
so 512≈Anima native minimizes upscaling on mixed social data; bump to 768 once a 512 run validates).

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
