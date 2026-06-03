# CLAUDE.md — Anima Realism finetune project

> Portable project memory. Lives in the project folder so it survives moving to another drive.
> Current design: `docs/superpowers/specs/2026-06-02-anima-realism-v5-design.md` (v5).
> History: `2026-05-31-...lora-design.md` (LoRA, abandoned), `2026-06-01-...v2-design.md` (v2 tag-only).

## Goal

Make the **Anima** diffusion model (anime base; Qwen3-0.6B TE + Qwen-Image VAE) output **realistic photos** —
a domain shift fought against the anime prior. Community realism finetunes of Anima exist on Civitai, so it works.

- **Current = v5:** full finetune (not LoRA) **from base DiT at 1024**, on a rented **Vast.ai A100-80GB**,
  via **diffusion-pipe**, on ~1942 Gemini-captioned sharp photos. (LoRA on local 4080 abandoned: 16GB can't fit a
  full finetune, and the anime→photo shift is too large for LoRA.)

## Status (2026-06-02) — v5 TRAINING LIVE on Vast (A100-80GB)

**Current model = v5.** Pipeline rebuilt + validated end-to-end (subagent-driven, all tests green); training
running on a rented Vast A100. v2/v3/v4 superseded. Spec: `docs/superpowers/specs/2026-06-02-anima-realism-v5-design.md`;
plan: `docs/superpowers/plans/2026-06-02-anima-realism-v5.md`. Branch **`v5-build`** (pushed to origin, **NOT merged to master**).

**Why v5 / what changed vs v2-v4 (tag-only captions → blurry, undertrained):**
- **Train at 1024 from BASE DiT** (`finetune.init_from=""`), 20 epochs, save_every_epoch, lr 8e-6, Qwen3 frozen
  (`llm_adapter_lr=0`), optimizer `adamw_optimi`. ~50GB VRAM.
- **Curate by TECHNICAL defects only, not aesthetics:** phash dedup (hamming **8**) + drop `min(w,h)<1024` +
  drop `blur_var<100` (Laplacian sharpness gate). **Keep all aesthetic buckets** (tagged, not dropped).
  Design spine: *aesthetic-bad ≠ blurry* — gate on focus/res, tag the rest.
- **CLIP aesthetic stage (S2) DELETED** — Gemini emits the quality tag.
- **Captions = WD14 tags + local NSFW safety + Gemini structured output (enum-locked style vocab + NL).**
  The Qwen3 LLM TE was starved by v2 tag-only captions → mushy/blur; rich NL + controlled style tokens fix it.

**Curation funnel (this run):** 5949 raw → dedup+`<1024` → 3279 → +`blur≥100` → 1957 → −13 underage(WD14) −2 missing
→ **1942 captioned @1024**. Dataset = `data/dataset/` (3884 files, 1.63 GB), uploaded to Vast via GDrive+gdown.

**Captioning (LOCAL prep on 4080 + Gemini API; not on the rented GPU):**
- **Gemini `gemini-2.5-flash-lite`** (cheapest vision, free tier), structured `response_schema` with ENUMS,
  `safety_settings=BLOCK_NONE` on the 4 adjustable cats. **Enum name is `HARM_CATEGORY_DANGEROUS_CONTENT`** (NOT
  `_DANGEROUS` — the docs were wrong; this bug caused a 100% silent-blank run). Concurrency 12 (thread pool + exp
  backoff). Resumable cache `data/gemini_cache.json` (caches successes + legit refusals; **never errors**).
  Key in `.env` (gitignored, throwaway). Logic in `src/gemini_caption.py`.
- **Refusal rate observed: safe 3%, explicit 43%** (Gemini declines hardcore → tags-only fallback, accepted).
  79% full Gemini captions overall.
- Local models: **WD14 SwinV2_v3** (dghs-imgutils — tags + underage block) + **Falconsai/nsfw_image_detection**
  (safety tag). **No JoyCaption** (8B too slow on 4080).

**Hard safety boundary (unchanged):** legal adults only. WD14 `block_tags` (loli/shota/child/...) hard-DROP;
Gemini core child-safety is always-on (non-disableable). 13 blocked this run.

**Live gotchas hit + fixed (read before debugging a re-run):**
- Global Python + **numpy 2.x** → transformers auto-imports TensorFlow → `_ARRAY_API not found` crash.
  Fix: `os.environ["USE_TF"]="0"` at top of `src/03_caption.py` (torch-only).
- Gemini 100% blank: wrong enum raised every call, swallowed by `except`. Added **pre-flight probe** (aborts
  loudly) + **cache-only-on-success**.
- **`shuffle_tags`/`tag_dropout` would SHRED the NL sentence** (splits on its commas) → set `shuffle_tags=false`,
  `tag_dropout_percent=0`. Hybrid tag+NL caption must be used verbatim. (`caption_dropout=0.1` kept = CFG.)
- Stage 1 doesn't wipe `data/clean` → re-runs accumulate orphans (delete files not in manifest, or wipe before re-run).
  Stage 4 curate now **requires a caption** (skips uncaptioned/missing rows that else crash the copy).
- **Vast Jupyter terminal wraps lines >~95 chars + hangs on pasted heredocs.** Use `git fetch` + short scripts
  (`scripts/run_v5_train.sh`), never long pastes. The two tomls were force-added to `v5-build` so they `curl`/checkout.

**v5 RESULT — trained to epoch 20 (lr 8e-6). SUCCESS as a photoreal base:** overall realism + lighting good,
close-up faces good. **Weak: small faces (medium/full-body shots), hands/feet, background detail.** The **small-detail** weaknesses
(faces-in-wide-shots, hands/feet, bg) are **resolution-bound** (1024 under-resolves small-in-frame) + subject-focused
data → fix at **INFERENCE** (ADetailer/FaceDetailer + HandDetailer + hires-fix), NOT by more epochs. **BUT epoch 20 is
the best AND the overall look was still improving at 20 (undertrained, NOT overcooked)** → lr 8e-6 was too gentle to
converge in 20 epochs; more epochs and/or higher LR (**reinforces v6**) push overall realism further. **Keeper = epoch
20** → DOWNLOAD → destroy instance (v3/v4 were lost by never downloading).

## Next — v6 plan (user decisions 2026-06-03; execute next session)

Goal: push **fine detail** (background, hands, phones/objects) + test whether a higher LR strengthens the look.
- **Train res 1536** (up from 1024). ⚠️ **VRAM**: 1536 full-finetune likely >80 GB → may OOM on A100-80GB; resolve first
  (`[model] qwen_nf4=true`, an H100, or check actual usage).
- **Dataset `min_resolution: 768`** (down from v5's 1024) — keep MORE images; user accepts diffusion-pipe upscaling
  768→1536 (soft-upscale trade-off, for more data/detail/variety). Keep the blur sharpness gate.
- **lr higher** (pre-staged 1.5e-5; "higher LR" per user — confirm 1.5–2e-5), **epochs 18–20** (v5 was still climbing at
  20 → don't cap low; save-every-epoch + pick best). From base for a clean test; consider warm-start from **v5-epoch20**
  to save 1536 compute (decide next session). NOTE: v5 still improving at 20 also means a cheap alternative to v6 =
  just **extend v5** (warm-start ep20, +N epochs at 8e-6) — but v6's higher LR converges faster + tests the hypothesis.
- **Captions more detailed** — edit the Gemini prompt (`src/gemini_caption.py` `build_prompt`) to describe background,
  objects, accessories (phones etc.), finer detail; bump `max_output_tokens` (~256→400).
- **NSFW handling per user:** send ALL images to Gemini, **no pre-routing** (already true in v5) — let the API response
  decide what it captions (BLOCK_NONE + tags-only fallback, already true). NEW: let **Gemini emit the safety tag** (add
  to schema) instead of pre-determining. **KEEP the WD14 underage hard-block — legal boundary, non-negotiable.**
- A pre-staged v6 (`anima_realism_ft_v6_*` tomls + `scripts/run_v6_train.sh`; lr 1.5e-5 / 15ep / 1024 / from-base) is on
  `v5-build` — **update it to 1536 + min_res 768 + richer caption prompt + Gemini-safety-tag** before launching.

**Carried-over tradeoff:** v5's `blur≥100` gate biased toward sharp/professional shots → few `amateur snapshot` images →
that control token is weakly trained. v6's `min_res 768` (more data) partly helps; loosen blur or tune rubric if amateur
realism is wanted.

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

## Caption format (v5) — enum-locked controlled vocab

```
realistic photo, <quality_level>, <capture_style>, <lighting..>, <condition..>, <safety>, <wd14 tags>[, watermark], <NL description>
```
e.g. `realistic photo, masterpiece, best quality, amateur snapshot, direct on-camera flash, grainy / high ISO, safe, 1girl, kitchen, a woman leaning on a counter holding a mug`

- Anchor `realistic photo` always leads — the inference handle to pull output off the anime prior.
- **Controlled vocab** (Gemini MUST pick from these enums → consistent = reliable inference triggers):
  - `quality_level`: `masterpiece, best quality` | `high quality` | `low quality`
  - `capture_style`: `amateur snapshot` | `casual phone photo` | `semi-professional` | `professional photograph` | `studio portrait`
  - `lighting` (0–2): `direct on-camera flash` | `natural daylight` | `golden hour` | `overcast flat light` | `indoor artificial light` | `low light` | `soft window light` | `studio lighting`
  - `condition` (0–2): `sharp focus` | `soft focus` | `grainy / high ISO` | `motion blur` | `compressed / low-res` | `overexposed` | `underexposed`
- `safety` (safe/explicit) from Falconsai. `watermark` token appended when Gemini flags it → **negative-prompt** at inference. NL = free Gemini text.
- All defined in `src/gemini_caption.py` (`VOCAB`, `build_schema`, `build_prompt`, `coerce_response`, `assemble_caption`).
- Captioner ≠ text encoder: Qwen3 TE encodes whatever text is written; no benefit to "matching" captioner to TE.

## Pipeline — v5 (prep runs LOCALLY on 4080; Gemini via API; S2 deleted)

1. `src/01_ingest_clean.py` — phash dedup (hamming 8) + drop corrupt/`<1024`/`blur_var<thr`; records width/height/blur_var.
   Knobs: `ingest.drop_small`+`min_size`, `ingest.drop_blurry`+`blur_var_threshold` (tune from distribution), `phash_hamming_threshold`.
2. ~~`src/02_quality_score.py`~~ — **DELETED** (CLIP aesthetic; Gemini emits quality now).
3. `src/03_caption.py` — WD14 tags (+ underage block) + Falconsai safety + Gemini structured enum/NL → caption.
   Two-pass: serial local tag/safety → **concurrent** Gemini (`caption_many`). Gemini logic in `src/gemini_caption.py`.
4. `src/04_build_dataset.py` — `curate()` (require caption + 1024 + blur backstop), copy to flat `data/dataset/` +
   `.txt` sidecars, emit diffusion-pipe `dataset.toml`.
5. `src/05_make_train_config.py` — emit `anima.toml` (from base, `shuffle_tags=false`, no `[adapter]` = full finetune).

**Vast launch:** `scripts/vast_setup.sh` (clone `bluvoll/diffusion-pipe` + download 3 Anima models) → upload `data/dataset`
→ `scripts/run_v5_train.sh` (copies tomls into place + `nohup deepspeed --num_gpus=1 train.py --deepspeed --config anima.toml`).
Watch `tail -f /workspace/train.log`; epoch checkpoints in `outputs/anima_realism_ft_v5/<ts>/epoch*/`.

Config: all paths + thresholds in `config/pipeline.yaml`. Tests: `python -m pytest tests/ -v`
(exclude `tests/test_01_ingest_clean.py` if `imagehash`/`cv2` not installed in the active env).

## Key training hyperparameters (v5, A100-80GB)

- 1024 res, from base DiT, 20 epochs, `save_every_n_epochs=1`, lr **8e-6**, `adamw_optimi`, `activation_checkpointing=true`,
  `llm_adapter_lr=0` (freeze Qwen3), `caption_dropout=0.1`, `tag_dropout=0`, `shuffle_tags=false`. ~50 GB VRAM.
- OOM fallback: add `[model] qwen_nf4=true`, or drop resolution. (40GB cards OOM at 1024.)
- diffusion-pipe pre-caches latents (one VAE pass, AR-bucketed) before epoch 1; `cache_text_embeddings=false` (TE frozen).

## Caveats after folder move

- Paths in the spec/scripts assume project root; update if drive letter changes.
- The `~/.claude/projects/.../memory/` store is keyed to the **old** path and won't auto-load from the
  new location — **this CLAUDE.md is the durable record.**
