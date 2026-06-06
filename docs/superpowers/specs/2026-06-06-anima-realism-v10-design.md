# Anima Realism v10 — photoreal render-style finetune (preserve base concepts)

**Date:** 2026-06-06
**Branch:** `v5-build`
**Supersedes direction of:** v9 (background-enrichment plan). v10 is a clean restart with a different philosophy.

## 1. Goal & philosophy shift

Make Anima output **photorealistic** images while **preserving the base model's concept/character
knowledge and visual diversity** (prompt a concept the base knows → get it rendered as a photo).

This is a **render-style finetune**, not a domain wipe. Every prior run (v5–v9) tried to *erase* the
anime prior by training hard. v10 inverts that: warm-start the base, gentle-train on **only
genuinely-sharp photos**, and **stop before concepts erode** (pick-best on a concept-retention eval).

**Output is 100% photoreal, 0% anime.** "Preserve concepts" means keep the base's broad visual
knowledge (poses, objects, scenes, compositions — what the HF authors call "existing diversity"),
not the anime *render style*.

### What the Anima HF repo confirms (read 2026-06-06)

- Base = NVIDIA **Cosmos-Predict2-2B-Text2Image**. TE = Qwen3-0.6B. VAE = Qwen-Image VAE.
- Trained on millions of anime + ~800k non-anime **artistic** images (LAION-POP, DeviantArt),
  **filtered to EXCLUDE photos.** → The base has essentially **zero photographic content**;
  photoreal is a genuine hard shift. This justifies a long run (50 epochs) even at a gentle LR.
- Authors' finetune guidance: **"keep LLM adapter LR at zero"** (we do: `llm_adapter_lr=0`) and the
  model needs a **"light touch" due to existing diversity** (matches gentle LR + pick-best; that
  diversity is exactly what we want to keep).
- Base-native prompt format (booru/score):
  `masterpiece, best quality, score_7, safe, [character/series/artist] [general tags]`
  neg: `worst quality, low quality, score_1, score_2, score_3, artist name`.
- Author-confirmed inference: res 512–1536, steps 30–50, **CFG 4–5**, samplers
  `er_sde` / `euler_a` / `dpmpp_2m_sde_gpu`, scheduler `beta57`. Matches our prior low-CFG finding.

## 2. Decisions (locked with user)

| Decision | Choice | Why |
|---|---|---|
| Output | 100% photoreal, 0% anime | User goal |
| Preserve | base concept/diversity knowledge | The base's power |
| Warm-start | **base** `anima-base-v1.0.safetensors` | Max concept knowledge; v8 ep10 already drifted |
| Data source | `data/raw/` (6396 readable) | "any photo from raw" |
| Resolution floor | **1280** short side (~2699 candidates pre-gate) | User; <1280 upscales too hard at 1536 |
| Train resolution | **1536** | User; full detail |
| Captions | **`masterpiece, best quality, score_7, [rating]` + RAM++ tags + Florence-2 short caption** | Real-photo tagger (not booru anime); quality tokens reuse base priors; NL feeds the LLM TE |
| Anchor | **none** ("realistic photo" anchor dropped) | 100% photo output; concept safety comes from warm-start + pick-best, not an anchor |
| LR | **6e-6** | Gentle ("light touch"); pick-best is the real preservation lever |
| Epochs | **50**, `save_every_n_epochs=5` (10 ckpts ≈ 42 GB) | Undertraining seen 4× at low LR; 50ep = project experience |
| Eval | photoreal set **+ concept-retention set** | Catch concept drift, pick the epoch before erosion |
| Optimizer | `adamw_optimi` fp32, full-FT (not LoRA), freeze Qwen3 (`llm_adapter_lr=0`) | Validated v5; erasing arty-render needs base-weight movement |

## 3. Curation — the new core (`src/v10_curate.py`)

**Problem:** detail ≠ pixels. A 6000×6000 50 MB file can be a soft source upscaled (zero real
high-frequency detail); a sharp 1300px JPEG can be genuinely crisp. The existing v8 gate (raw
Laplacian ≥100 + dimension floor) is **scale-blind** — a large upscale clears it. v10 needs gates
that measure **actual high-frequency content**, independent of nominal size.

Pipeline over `data/raw/`, in order:

1. **Corrupt / unreadable drop** (reuse stage-1 `is_corrupt`).
2. **Nominal floor:** min(w,h) ≥ **1280**.
3. **phash dedup** (hamming 8), keep highest-px per near-dup group (reuse stage-1 `phash` + greedy group).
4. **Scale-aware sharpness:** Laplacian variance computed on a **fixed-size center crop / fixed
   downscale** so a 6k and a 1.3k image are judged on the same scale. Threshold tuned from the
   distribution (start ~ the v5 value, re-tune from emitted metrics). Kills "big but soft".
5. **Upscale detector (the fake-big killer):**
   - *Round-trip residual:* downscale 2× then upscale back; compute SSIM vs original. Real-detail
     photos lose a lot (low SSIM); already-upscaled ones barely change (high SSIM → flag).
   - *FFT radial spectrum:* estimate the **true detail resolution** from where the radial power
     spectrum collapses to the noise floor; drop where true-res ≪ nominal-res.
   - Both cheap, local; combine (either-flags → drop, with tunable thresholds).
6. **Compression gate:** read JPEG **quantization table** (`PIL Image.quantization`) → estimate
   quality, drop Q < ~85; plus an **8×8 blockiness** metric for DCT block artifacts.
7. **Underage hard-block backstop (NON-NEGOTIABLE):** run a minor-detector purely as a **drop
   gate** (discard the detector's tags; we caption with RAM++). Kept for legal/safety even though
   we no longer use a booru tagger for captions. (Implementation: reuse the existing WD14 underage
   `block_tags` path as a safety-only filter, or an equivalent.)
8. **AR-crop to 0.66–1.5** (Anima DiT pos-emb 120-patch / 1920px cap; wider/taller crashes at 1536).
   Center-crop the long side only so short side (≥1280) is preserved. Reuse `ar_crop_box`.

**Output:** flat `data/v10_clean/` + `data/v10_manifest.csv`. The manifest records **all metrics**
(width, height, blur_var, ssim_residual, est_true_res, jpeg_q, blockiness, phash, drop_reason) so
thresholds can be tuned from the real distribution (as the v5 blur threshold was). **No buckets**
(general photoreal — the v8 detail/anchor/bg bucketing was for a per-concept de-compression
strategy that no longer applies).

**Reuse:** stage-1 helpers (`blur_variance`, `phash`, `image_size`, `is_corrupt`, `common.*`) and
v8's `ar_crop_box` / dedup. Pure helpers stay stdlib-only and import-safe for tests.

## 4. Captions (`src/v10_caption.py`)

Per image, assemble:

```
masterpiece, best quality, score_7, <rating>, <RAM++ tags>, <Florence-2 short caption>
```

- **Quality tokens** `masterpiece, best quality, score_7` — base-native, strong learned priors;
  since we curate to only-sharp photos, this binds "best quality" → photoreal-sharp = a free
  inference quality dial. Not anime/2D vocab.
- **`<rating>`** — `safe` / `explicit` from the existing local NSFW detector (Falconsai), booru-style.
- **RAM++ tags** (Recognize Anything Plus) — real-world tag vocabulary (`woman, kitchen, window,
  sunlight, mug, smile`). Local, no API. The photo-domain equivalent of WD14.
- **Florence-2 short caption** — one NL sentence (MIT, tiny, fast, local). Feeds the Qwen3 **LLM**
  text encoder properly; mitigates the tag-only mushiness that hurt v2/v3/v4.
- `shuffle_tags=false`, `tag_dropout=0` (the NL sentence has commas — shuffling would shred it).
- `caption_dropout=0.1` kept (CFG).

**No Gemini, no WD14/EVA02 for captions, no realism anchor.**

**Known tradeoff (flagged, accepted):** RAM++/Florence-2 are weaker on explicit NSFW anatomical
detail than the old EVA02 booru tags. If NSFW fidelity disappoints after eval, add a fallback in a
later iteration; not in v10 scope.

**Env:** local on the 4080, global Python, `USE_TF=0` at top (numpy-2/TF auto-import crash, per
prior runs). RAM++ and Florence-2 both download on first run.

## 5. Build (`src/v10_build_dataset.py`)

Copy captioned `data/v10_clean/` → flat `data/v10_dataset/` + `.txt` sidecars; emit the
diffusion-pipe `outputs/anima_realism_ft_v10_dataset_config.toml`. Require a caption + the 1280
floor backstop. (Mirror stage 4 with v10 paths; can reuse `04_build_dataset.py` logic.)

## 6. Train config (`outputs/anima_realism_ft_v10_train_config.toml`)

- `transformer_path` → **base** `anima-base-v1.0.safetensors` (warm-start from base).
- `lr = 6e-6`, `epochs = 50`, `save_every_n_epochs = 5`.
- `resolutions = [1536]`, AR limits **0.66–1.5** baked into the dataset toml.
- `optimizer = adamw_optimi` (fp32), `activation_checkpointing = true`, `llm_adapter_lr = 0`
  (freeze Qwen3), `caption_dropout = 0.1`, `shuffle_tags = false`, `tag_dropout_percent = 0`.
- `project_name = anima_realism_ft_v10`.
- `cache_text_embeddings = false` (TE frozen).
- **OOM fallbacks (commented):** `[model] qwen_nf4 = true`, `[optimizer] adamw8bit`.

## 7. Host / VRAM

1536 full-FT from base ≈ 2.25× the pixels of v5's 1024 (~50 GB) → likely **> 80 GB**. Rent a
**96 GB card** (RTX 6000 Pro Blackwell, as v7) or A100-80 with the OOM fallbacks. Blackwell sm_120
needs CUDA 12.8+/torch 2.7+ — verify `torch.cuda.get_device_name(0)` before launch. Measure peak
with `nvidia-smi -l 5` during latent caching.

**Compute/cost:** ~40 min/epoch at 1536 (v7 rate) → 50 ep ≈ **~33 hr** (~$45–65 at $1.3–2/hr); if
~12 min/ep (v8 rate), ~10 hr. Measure epoch 1, extrapolate. **Destroy the instant training +
download finish.** Disk: 10 ckpts × 4.18 GB ≈ 42 GB.

## 8. Eval (every saved epoch, ckpts ep5..ep50)

- **Photoreal set** — realism/quality climb (frozen prompts, like prior runs).
- **Concept-retention set** — frozen prompts of concepts/characters the base knows, written in the
  **base-native format** (`masterpiece, best quality, score_7, safe, [concept] [general tags]`).
  Watch for degradation → **pick the epoch where photoreal is strong but concept rendering still
  holds.** This is the safeguard that replaces the dropped anchor.
- **Inference defaults** (from HF + prior findings): res ≤1536 (AR 0.66–1.5, dims ÷64), steps
  30–50, **CFG 4–5**, samplers `er_sde`/`euler_a`/`dpmpp_2m_sde_gpu`, scheduler `beta57`,
  VAE = `qwen_image_vae.safetensors`.

## 9. Deliverables

- `src/v10_curate.py` + `tests/test_v10_curate.py`
- `src/v10_caption.py` (RAM++ + Florence-2 + Falconsai rating)
- `src/v10_build_dataset.py`
- `outputs/anima_realism_ft_v10_{train,dataset}_config.toml`
- `scripts/run_v10_train.sh`, `scripts/vast_fetch_v10.sh`
- Eval prompt docs: `docs/superpowers/specs/2026-06-06-v10-eval-prompts.md`
  (photoreal + concept-retention sets)
- This spec.

## 10. Risks / open items

- **6e-6 from a zero-photo base may undershoot photoreal in 50 ep** (light touch vs hard shift).
  Mitigation: save-every-5 + pick-best; if ep50 still climbing and not photoreal enough, extend or
  bump LR next run (v5 reached good realism from base at 8e-6/20ep, so 6e-6/50ep should arrive).
- **Concept erosion without an anchor** — pick-best on the concept-retention eval is the only guard.
  If concepts erode too early relative to photoreal arriving, reconsider re-introducing a quality/
  task token as a soft anchor.
- **RAM++/Florence-2 NSFW weakness** — accepted for v10.
- **Curate thresholds** (sharpness / SSIM-residual / true-res / jpeg-Q / blockiness) need tuning
  from the emitted metric distribution before the full run — budget a calibration pass on a sample.
