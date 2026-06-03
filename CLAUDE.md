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

## v6 = extend-v5 convergence probe — RESULT: UNDERTRAINING (2026-06-03)

**PROBE RESULT:** continued v5-ep20 @ 8e-6; **cum-epoch 25 much better, still climbing, NOT overcooked.**
→ **UNDERTRAINING confirmed; lr 8e-6 was fine** (NOT the limiter). The original v5 weakness = too few steps,
not too-low LR. **No v6b higher-LR escalation needed** (no plateau). User: could go cum-epoch 35–40.
**Consequence for V7:** lr 8e-6 is now a VALIDATED hyperparameter; the lever is **more epochs + save-every +
pick best**. V7 warm-starts from the v6 keeper so the epochs go toward 1536/new-captions, not re-learning realism.

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

**VRAM — RTX 6000 Pro 96 GB (Blackwell):** v5 used ~50 GB at 1024 → ample headroom, so keep v5's fp32
`adamw_optimi` → probe is BYTE-IDENTICAL to v5 (no 8-bit confound). No bitsandbytes. ⚠️ **Blackwell sm_120**
needs recent CUDA (~12.8+) + torch (~2.7+) — verify the Vast image sees the GPU
(`python -c "import torch; print(torch.cuda.get_device_name(0))"`) before launch; upgrade torch if old.
96 GB also makes a future **1536 v6b** fit on one card (no nf4/offload). (History: an interim 40 GB plan used
`adamw8bit`; reverted when the 96 GB card appeared.)

**DONE on `v5-build` (pushed):** `outputs/anima_realism_ft_v6_train_config.toml` (transformer_path
→ `anima_v5_epoch20.safetensors`, lr 8e-6, epochs 5, optimizer `adamw_optimi` = v5), `scripts/run_v6_train.sh`
(warm-start guard, log → `/workspace/train_v6.log`), `scripts/vast_fetch_v6.sh` (gdown dataset.zip + ckpt, IDs
as args, size/count checks), frozen eval set `docs/superpowers/specs/2026-06-03-v6-eval-prompts.md`.
Spec: `docs/superpowers/specs/2026-06-03-anima-realism-v6-design.md`. Plan (runbook): `docs/superpowers/plans/2026-06-03-anima-realism-v6.md`.

**To launch (fresh A100, Vast runbook = plan Task 6):** `git clone -b v5-build .../LamShiuChing/anima-trainer repo`
→ `vast_setup.sh` → upload `v5_dataset.tgz` (gdown) + **upload epoch20 → `models/anima_v5_epoch20.safetensors`
(gdown)** → `run_v6_train.sh` → `tail -f /workspace/train_v6.log`. **Download `train_v6.log` (loss trend, hand to
Claude) + best epoch BEFORE destroying.** Then eval v5-ep20 + v6 ep1..5 with the frozen prompt set.

### v6b higher-LR escalation — NOT triggered
Probe never plateaued (lr 8e-6 was fine), so the fresh-from-base higher-LR arm is **dropped**. The fine-detail
bundle it carried (1536, richer captions) is now folded into **V7** (below), done properly.

**Carried-over tradeoff:** v5's `blur≥100` gate biased toward sharp/pro shots → `amateur snapshot` token weakly
trained. The probe doesn't fix this (same data); a future `min_res 768`/looser-blur run would.

### V7 captioning overhaul (decisions 2026-06-03; enum vocab under review)
Goal: richer + more controllable captions (caption == inference prompt). Pairs with V7 = **1536 train + higher LR
+ more/originals data**. Decisions:
- **DROP the `realistic photo` anchor** — 100% photo data ⇒ a token on every image carries no signal.
  ⚠️ only safe if V7 **warm-starts from the v6 keeper** (from-base would lose the anti-anime switch); recommend warm-start.
- **Expand enums** (controllability). Discipline: **populate-or-dead** (~50–100+ imgs/token) + prefer **booru-native
  vocab** (Anima base has priors → strong, cheap triggers). New slots: `shot_type`, `camera_angle`, `camera_lens`,
  `depth_of_field`, `color_grade`, `expression`; `quality` → booru ladder (masterpiece…worst quality); `resolution`
  (absurdres/highres/lowres) **auto-derived from pixel size in stage 1, not Gemini**.
- **Division of labor:** enums = photographic/style layer; **content (person/hair/clothes/accessories/setting) =
  WD14 booru tags + rich NL** (too open to enumerate).
- **Rating via Gemini** (booru ladder `rating:general/sensitive/questionable/explicit`) replaces binary safe/explicit.
- **NSFW adult: no local block** (already all→Gemini, BLOCK_NONE, refuse→WD14-tags-only fallback). Gemini emits rating.
- **WD14 more detailed:** swap SwinV2_v3 → **EVA02-Large v3**, lower `general_threshold` ~0.25–0.3. Key for the ~43%
  explicit images Gemini refuses (booru anatomical tags = the NSFW caption richness). Tradeoff: more noise tags.
- **Gemini NL richer:** describe background/objects/materials/accessories; `max_output_tokens` 256→~450.
- 🚫 **WD14 underage hard-block KEPT — non-negotiable.** User states dataset is all-adult; the block is then a
  **no-op** (drops nothing) and exists purely as a backstop against a single mislabeled/slipped file. Zero cost to
  keep, catastrophic+illegal risk if removed. Not a quality setting.
- **IMPLEMENTED on `v5-build`** (2026-06-03): `src/gemini_caption.py` rewritten (new `VOCAB` 18 slots,
  `SINGLE_SLOTS`/`ARRAY_SLOTS`, `resolution_tag`, anchor removed, rating+fallback), `src/03_caption.py` wired
  (derived resolution + Falconsai fallback rating), `config/pipeline.yaml` caption→v7 (EVA02_Large @0.25,
  max_output_tokens 450, nsfw map→rating). Tests green (40 passed). Spec + full vocab + **user prompt guide**:
  `docs/superpowers/specs/2026-06-03-anima-realism-v7-captioning-design.md`.
- ⚠️ Captions change only on the **NEXT dataset rebuild (V7)**; the running v6/v5 model still uses the OLD v5
  caption format (see "## Caption format (v5)" below) — don't prompt the v6 model with V7-only tokens.

### V7 training config (created 2026-06-03)
- **Dataset prep:** new raw ~6200 → stages 1/3/4 LOCALLY. Floor **1024** (`ingest.min_size`), train res **1536**
  (`dataset.resolutions=[1536]`; curate `min_resolution` stays 1024 = the floor), blur gate strict (100).
  `project_name=anima_realism_ft_v7` → stage 4 emits `outputs/anima_realism_ft_v7_dataset_config.toml`.
  **MUST delete `data/gemini_cache.json`** before stage 3 (old cache = v5 shape → crash). EVA02 downloads first run.
- **Train:** `outputs/anima_realism_ft_v7_train_config.toml` + `scripts/run_v7_train.sh`. **WARM-START from the v6
  keeper** (`models/anima_v6_keeper.safetensors` — upload the best v6 epoch), **lr 8e-6 (VALIDATED), epochs 40,
  save-every, pick best.** Log → `/workspace/train_v7.log`.
- ⚠️ **1536 VRAM**: ~2.25× the pixels of 1024 (v5 ~50 GB @1024) → 1536 full-FT may exceed 80 GB. OOM fallbacks
  (in the toml): `[model] qwen_nf4=true` and/or `[optimizer] adamw8bit` (needs bitsandbytes), and/or the 96 GB card.
  Measure peak with `nvidia-smi -l 5` during latent caching. 40 epochs @1536 = heavy compute — save-every lets you stop early.
- **Upscaling principle:** diffusion-pipe resizes all to 1536; sources <1536 **upscale = soft** (no new detail),
  ≥1536 **downscale = crisp**. 1024 floor = mild upscale for 1024–1536; the real detail fuel is ≥1536 originals.

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
