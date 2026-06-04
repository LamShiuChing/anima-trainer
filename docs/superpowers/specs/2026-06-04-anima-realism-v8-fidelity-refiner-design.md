# Anima Realism v8 — design (fidelity refiner pass)

> Date: 2026-06-04 · Branch: `v5-build` · Supersedes the "V7-HD 4k-originals" sketch in `CLAUDE.md`
> Predecessor: V7 (keeper = `epoch18`, full-FT 1536, warm-start from v6 keeper, lr 8e-6)

## Goal

ep18 is a good photoreal base, but it was trained on **compressed social-media images** and so learned
to **reproduce compression** (JPEG blocking, soft high-frequency, mushy fingers/fabric/edges). v8 is a
**fidelity refiner**: warm-start ep18 and train on a small, **all-clean** set to

1. **erase the learned compression look** (move weights toward clean pixels), and
2. **add high-frequency detail** to fingers, clothes/fabric, held objects (phones), toes/footwear, and
   backgrounds,

**while preserving the amateur / casual aesthetic.** This is NOT a realism run (ep18 already has realism)
and NOT a small-faces-in-wide-shots run (that stays on the inference track — see Out of Scope).

## Core principle — fidelity ⟂ aesthetic

Fidelity (clean pixels) and aesthetic (amateur vs studio) are **independent dials**. The failure mode to
avoid: if "clean / sharp" only ever co-occurs with polished studio shots in the training data, the model
**fuses clean → studio** and the amateur look drifts to stock. Prevention = **honest split-axis
captioning**: label the two axes separately so the model sees *clean* and *candid* co-occur.

The V7 captioner already carries both axes:
- **Fidelity axis:** booru quality ladder (`masterpiece` / `best quality` … `worst quality`) +
  `resolution` (`absurdres` / `highres` / `lowres`, auto-derived from pixel size in stage 1).
- **Aesthetic axis:** `capture_style` enum (`amateur snapshot`, `casual phone photo`, … `studio portrait`).

At inference this becomes a controllable dial: prompt `amateur snapshot, best quality, highres, sharp` +
negative `jpeg artifacts, compressed, blurry, low quality`. The negative works because ep18 already
associates those tokens with compression; v8 simply stops reinforcing them and pulls the base clean.

## Diagnosis — why a refiner, not more epochs (what we KNOW vs ASSUME)

- **KNOW (measured):** V7 climbed monotonically ep5 → ep18 ("much better"); no plateau observed. Some
  ep18 outputs are soft/low-fidelity. Training sources were largely <1536 **upscaled-soft** (no real
  high-frequency content).
- **ML certainty:** a diffusion model **cannot learn detail absent from its data.** Upscaled-soft sources
  contain no high-frequency texture → crisp fingernails / fabric weave / pores are **unlearnable from
  that set at any epoch count.** More epochs ⇒ more *confident reproduction of soft*. So the
  **sharpness / micro-detail** deficit is **data-capped, not undertraining.**
- **Split the symptom:** *structure* (5 fingers, correct topology) can still improve with epochs;
  *sharpness + the compression look* is data-bound. v8's targets ("clear" fingers/clothes/toes) are
  mostly the data-bound half.
- **User call (2026-06-04):** ep18 "not that bad," but it generates compressed quality because it trained
  on compressed images → needs a refiner pass for high fidelity while keeping amateur. v8 executes that.

(Cheap confirmation tests — local close-up-vs-full-body gen, and a +epochs resume probe — were considered
and deemed unnecessary: the upscale-soft data ceiling is decisive for the sharpness half, and the user
confirmed the compression-look diagnosis directly.)

## Approach — warm-start ep18, full-finetune, all-clean refiner

**Training method = full-finetune** (not LoRA). Two jobs favor it: (1) *erasing* compression means moving
the base weights off the artifact distribution — full-FT overwrites; a LoRA only adds a low-rank delta on
top of still-compressed ep18 weights, so residual compression leaks at low adapter weight. (2) Full-FT
output is a **single-file DiT**, matching the v5→v6→v7 warm-start lineage (usable as the next warm-start;
no merge step). The old "LoRA too weak" note in `CLAUDE.md` was about the large anime→photo shift from
base, not a small fidelity refine — but full-FT still wins on the erase-compression half. Drift risk is
controlled by **low LR + the candid anchor + save-every + pick-best**.

## Dataset — ~600–900 images, ALL clean / high-fidelity, ZERO compressed

Three buckets (**60 / 35 / 5**):

| Bucket | Share | Content | Source |
|---|---|---|---|
| **Detail close-ups** | 60% | hands, hand+phone, clothing/fabric texture, footwear/feet, held objects — **target detail LARGE in frame + tack-sharp** | external CC0/research (Open Images via FiftyOne, Unsplash research dataset) |
| **Candid anchor** | 35% | clean-but-casual **whole-person** shots (phone-style candids shot on good cameras) — holds amateur look + identity variety + anti-drift | external (Unsplash candids) |
| **Backgrounds** | 5% | interiors / scenes for environment fidelity | Open Images (FiftyOne), Poly Haven **photographic renders only** |

**Why pixel-area matters:** a full-body person shot at 1536 still under-resolves a hand (~40 px → mush).
To teach clear fingers the hand must occupy meaningful pixel area and be sharp → the 60% bucket is
deliberately detail-prominent, not whole-person.

**Why all-clean (no compressed anchor):** the user's own originals are the compressed source (the disease).
Using clean external candids as the anchor keeps the **fidelity axis uniformly clean** so compression is
erased hardest; the amateur look is held by the candids' content + `amateur snapshot` tags, not by
re-feeding compressed pixels.

**Curation — hard gates:**
- **Curate at 100% zoom.** Reject upscaled, re-compressed, or soft files. Stated resolution is NOT enough.
- **≥1536 px on the short side** (real detail fuel; ≥1536 downscales crisp, <1536 upscales soft).
- **Crop every image to AR 0.66–1.5** (see hard limit below). Landscapes/architecture skew wide and will
  crash otherwise.
- **Feet caveat:** external foot/toe close-ups skew fetish-stock (weird angles, oily/studio). Curate hard
  or weight toes lower; do not let them drag the set toward a studio look.
- **Poly Haven caveat:** use photographic renders, NOT raw tileable texture maps / HDRIs (teach
  tiling/flatness).

### ⚠️ Hard limit — Anima DiT pos-emb cap (carried from V7)
Pos-emb caps at **120 patches = 1920 px/side** (VAE/8 × patch/2 = ÷16). At 1536 train res, **AR must
stay 0.66–1.5**; AR 0.5/2.0 → 2176 px side → 136 patches → `AssertionError` crash (hit + fixed in V7).
Baked into the dataset toml (`min_ar=0.66`, `max_ar=1.5`) and enforced at crop time.

## Captioning — existing V7 pipeline (no code changes expected)

- WD14 **EVA02-Large v3 @ general_threshold ~0.25** + Gemini structured (18-slot enum vocab + rich NL).
- **Content/detail triggers = WD14 booru tags** (`hand`, `holding phone`, `barefoot`, `denim`, `knit`,
  `sweater`, …) — Anima base has priors for these → strong, cheap inference handles.
- Detail crops: person-centric enums (`expression`, `body`, `ethnicity`, `skin`) will be mostly empty —
  acceptable; WD14 + NL describe the crop (e.g. "close-up, hands holding a smartphone, detailed fingers").
- Fidelity/aesthetic axes captioned honestly per the Core Principle above.
- ⚠️ **Delete `data/gemini_cache.json` before captioning** (v7-shape cache from the V7 run → crash on
  reuse). EVA02 weights already downloaded from V7.
- Verify a handful of non-person / close-up captions read sanely before captioning the full set.

## Training config — `outputs/anima_realism_ft_v8_train_config.toml`

Diff vs the V7 train toml:
- `transformer_path` → the **ep18 keeper** single-file DiT (warm-start source).
- `lr` = **4e-6** *(starting value; lower than the validated 8e-6 → refine without disturbing learned
  concepts. Tune from eval — too low + ~700 imgs + ≤10 ep may barely move detail.)*
- `epochs` = **10** (ceiling; pick best, stop early via save-every).
- `[optimizer] type` = `adamw_optimi` (fp32) — unchanged from V7.
- `save_every_n_epochs = 1`, `llm_adapter_lr = 0` (freeze Qwen3), `caption_dropout_percent = 0.1`,
  `shuffle_tags = false`, `tag_dropout_percent = 0`, `activation_checkpointing = true` — unchanged.
- `output_dir` → `…/anima_realism_ft_v8`.

### `outputs/anima_realism_ft_v8_dataset_config.toml`
- `resolutions = [1536]`, `min_ar = 0.66`, `max_ar = 1.5`, dataset path → the v8 clean set.

### Warm-start caveat (carried from v6/v7)
diffusion-pipe loads **weights only**, not Adam moments → first ~100 steps re-warm a fresh optimizer on
adapted weights. `warmup_steps = 100` covers it. Expect a possible small epoch-1 dip that recovers —
judge the trend, not epoch 1.

### VRAM
1536 full-FT ran fine on V7's RTX 6000 Pro (Blackwell, 96 GB). OOM fallbacks (in the toml): `[model]
qwen_nf4 = true` and/or `[optimizer] adamw8bit` (needs bitsandbytes). Measure peak with `nvidia-smi -l 5`
during latent caching. **Blackwell sm_120 compat:** verify the Vast image's torch sees the GPU
(`python -c "import torch; print(torch.cuda.get_device_name(0))"`) before launch; upgrade torch if old.

## Eval protocol — the ONLY quality signal (loss is blind)

Flow-matching loss is blind to perceptual quality (flat-noise 0.067–0.124 across all of V7; re-confirmed
twice). Judge **only by eval images.**

- **Frozen prompt set + fixed seeds**, run on **every saved epoch + the ep18 baseline** (apples-to-apples:
  did we actually move past ep18?). Prompts must cover:
  - **(a) detail:** hand + phone close-up, fabric/clothing close-up, feet/footwear, generic hands.
  - **(b) whole-person amateur:** to catch aesthetic drift toward stock (the primary regression risk).
  - **(c) background / scene:** environment fidelity.
- **Inference settings:** **CFG 3–4.5** (Anima flow-matching DiT oversaturates at high CFG), optional
  RescaleCFG ~0.7, sampler `euler` / `dpmpp_2m` + `simple` / `beta`, steps 20–30, VAE
  `qwen_image_vae.safetensors`.
- **Fidelity dial:** prompt `amateur snapshot, best quality, highres, sharp` + negative
  `jpeg artifacts, compressed, blurry, low quality`.
- **Stop signals:** (1) person/aesthetic regression toward stock = drift → stop / lower LR; (2) sharper
  textures but composition variety collapsing = overfit → pick the prior epoch. Expected lift ep3–6.
- Download `train_v8.log` for loss-trend **corroboration only** (still-falling supports "still learning").

## Vast execution

Fresh, cheaper single-GPU host is fine (1536 full-FT fits one 80–96 GB card).

1. `scripts/vast_setup.sh` — clone diffusion-pipe + download the 3 Anima base models.
2. Upload the **v8 clean dataset** (gdown) + the **ep18 keeper** (gdown) → `…/models/anima_v7_epoch18.safetensors`.
3. `scripts/vast_fetch_v8.sh` (gdown dataset + ep18, IDs as args, size/count checks).
4. `scripts/run_v8_train.sh` — copies tomls into place + `nohup deepspeed --num_gpus=1 train.py …`,
   log → `/workspace/train_v8.log`. Watch `tail -f`.
5. Checkpoints in `outputs/anima_realism_ft_v8/<ts>/epoch{1..10}/`.

### ⚠️ Pre-destroy checklist (the v3/v4-loss lesson)
Before destroying ANY instance: (a) best epoch on local disk, **verified ~4.18 GB single-file DiT** (not
truncated); (b) FULL `train_v8.log` downloaded; (c) optional neighbor epoch as fallback. THEN **destroy**
(not stop — stop still bills storage).

### Vast terminal gotcha (carried over)
Jupyter wraps lines >~95 chars + hangs on pasted heredocs → use `git fetch` + short scripts +
`VAR=val` split lines, never long pastes. Tomls force-added to `v5-build` so they checkout/curl.

## Files to produce

- `outputs/anima_realism_ft_v8_train_config.toml`, `outputs/anima_realism_ft_v8_dataset_config.toml`
- `scripts/run_v8_train.sh`, `scripts/vast_fetch_v8.sh`
- A **dataset-build / curation step** for external sourcing + AR-crop + 100%-zoom gate (FiftyOne pull for
  Open Images; curated Unsplash; Poly Haven backgrounds). New script(s) under `src/` or `scripts/`.
- Frozen eval-prompt doc (alongside this spec).
- This spec.

## Out of scope (separate tracks)

- **Small faces in wide shots** → inference track (ADetailer / FaceDetailer + hires-fix). Not data-fixable
  at 1536.
- **Hands/feet structure at inference** → HandDetailer / ADetailer remains the cheap lever for residual
  cases; v8 raises the floor, inference cleans the tail.
- **Retraining realism from scratch** → not needed; ep18 has realism. v8 only refines fidelity/detail.

## Risks

- **Aesthetic drift to stock** — external clean data skews polished. Mitigations: honest split-axis
  captioning, 35% candid anchor, low LR, per-epoch whole-person eval. **Primary risk; watch eval (b).**
- **Feet sourcing quality** — fetish-stock skew. Curate hard or down-weight toes.
- **AR-crop missed** → `AssertionError` crash at 1536. Mechanical gate must run on every image.
- **LR mis-set** — 4e-6 is a guess; too low barely moves detail, too high drifts. Tune from eval.
- **Overfit on a small set** (~700 imgs) → texture sharpens, variety collapses. Save-every + pick-best.
- **Checkpoint loss** — the recurring failure mode. Pre-destroy checklist is mandatory.

## Success criteria

- Eval shows, vs the ep18 baseline on fixed prompts+seeds: **cleaner pixels (compression look reduced) +
  sharper fingers/clothes/phones/toes/background**, with the **amateur aesthetic intact** (no stock drift).
- A chosen best epoch downloaded + verified locally before instance teardown.
- Inference fidelity dial confirmed working (amateur + high-quality prompt, compression in negative).
