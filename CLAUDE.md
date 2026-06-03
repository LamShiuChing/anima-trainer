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

## Next — v6 = extend-v5 convergence probe (REFRAMED 2026-06-03; configs DONE, ready to launch)

**v6 reframed.** User goal = **push overall realism** (not fine detail). The old bundled plan (1536 + min_res
768 + richer captions + safety-tag, all at once) was CUT — it confounds variables, risks OOM, and forces a
costly recapture. Fine detail (small faces/hands/bg) is resolution-bound → handled on a separate **inference
track** (ADetailer/FaceDetailer + HandDetailer + hires-fix), not by retraining.

**The probe answers ONE question cheaply: was v5's ceiling undertraining or LR?** We KNOW v5 was undertrained
(monotonic climb, no plateau). We have ZERO evidence LR was too low (never saw a plateau) — that was an
assumption. Dataset ruled out for overall realism (v5 makes good realism). So change ONE variable:
- **Warm-start v5 epoch20, continue at the SAME lr 8e-6, +5 epochs (cum ~25), everything else byte-identical to v5.**
- Decision rule: **still climbing → undertraining; pick best epoch / extend.** **Plateau → 8e-6 ceiling → escalate
  to v6b** (fresh-from-base, lr 1.5–2e-5, 18–20 ep). The probe's curve shape is the answer.
- v5 checkpoints saved locally (single-file DiT, 4.18 GB) in ComfyUI `models/diffusion_models/`:
  `epoch10/12/15/20.safetensors` + base `anima_baseV10.safetensors`. **epoch20 = warm-start source.**

**DONE on `v5-build` (commit 63f4ab7, pushed):** `outputs/anima_realism_ft_v6_train_config.toml` (transformer_path
→ `anima_v5_epoch20.safetensors`, lr 8e-6, epochs 5), `scripts/run_v6_train.sh` (warm-start guard, log →
`/workspace/train_v6.log`), frozen eval set `docs/superpowers/specs/2026-06-03-v6-eval-prompts.md`.
Spec: `docs/superpowers/specs/2026-06-03-anima-realism-v6-design.md`. Plan (runbook): `docs/superpowers/plans/2026-06-03-anima-realism-v6.md`.

**To launch (fresh A100, Vast runbook = plan Task 6):** `git clone -b v5-build .../LamShiuChing/anima-trainer repo`
→ `vast_setup.sh` → upload `v5_dataset.tgz` (gdown) + **upload epoch20 → `models/anima_v5_epoch20.safetensors`
(gdown)** → `run_v6_train.sh` → `tail -f /workspace/train_v6.log`. **Download `train_v6.log` (loss trend, hand to
Claude) + best epoch BEFORE destroying.** Then eval v5-ep20 + v6 ep1..5 with the frozen prompt set.

### v6b (deferred — only if probe plateaus, OR a future fine-detail run)
- Higher LR: fresh-from-base, lr 1.5–2e-5, 18–20 ep, same data/captions. Reuse frozen eval set.
- Fine-detail bundle (separate experiment): **1536** (⚠️ OOM >80 GB → `qwen_nf4`/H100/measure), **min_res 768**
  (more data, accept 768→1536 upscale), **richer captions** (`src/gemini_caption.py` `build_prompt` describe
  bg/objects/accessories; `max_output_tokens` ~256→400), **Gemini-emitted safety tag** (add to schema). **KEEP
  WD14 underage hard-block — legal, non-negotiable.**

**Carried-over tradeoff:** v5's `blur≥100` gate biased toward sharp/pro shots → `amateur snapshot` token weakly
trained. The probe doesn't fix this (same data); a future `min_res 768`/looser-blur run would.

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
