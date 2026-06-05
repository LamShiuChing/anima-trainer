# Anima Realism v9 — design (background fix + style epochs)

> Date: 2026-06-05 · Branch: `v5-build` · Predecessor: V8 (keeper = `v8_epoch10.safetensors`, full-FT 1536,
> warm-start from `V7_epoch17.safetensors`, lr 4e-6, 10 ep — "great, still climbing")

> **Warm-start source (on disk):**
> `C:\Users\erede\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\v8_epoch10.safetensors`
> This file is the v9 warm-start source AND the eval baseline ("did v9 beat ep10?").

## Goal

v8 ep10 is a strong photoreal amateur base, but **backgrounds in selfies / mirror-selfies / portraits render
blurry, incoherent, or "slopish"** while the person renders well. v9 fixes the backgrounds and squeezes the
remaining style/lighting climb, **keeping the amateur aesthetic + NSFW capability.**

**Scope decision (2026-06-05): anime knowledge is explicitly DROPPED.** Goal = a pure photoreal amateur-style
generator with NSFW. This is *not* an anime→real hybrid; preserving the Anima base's anime priors is a
non-goal. (See "Strategic note" — the project's earlier "fight the anime prior" framing already eroded ~70%
of anime; v9 stops pretending otherwise and optimizes only for photoreal output.)

## Diagnosis — why backgrounds are blurry (the root cause)

**Per-concept blur learning** — the same mechanism CLAUDE.md documents for compression ("compression is learned
PER CONCEPT, not globally"), applied to focus:

- The model makes blurry *selfie/mirror-selfie/portrait* backgrounds because its **training data for those
  concepts had blurry (bokeh) backgrounds.** It learned "selfie ⇒ soft background." It has never been shown
  enough selfies/portraits with *sharp* backgrounds to believe otherwise.
- **Selective, not global, forgetting.** The person still renders well (people = the well-represented concept);
  only background/scene rendering rotted. Full-FT moves all weights; with zero scene/sharp-background signal in
  the trainset, the background-rendering pathways drifted to mush. v8 compounded this — **405 images for 10
  epochs** is a tiny, narrow set → steep over-specialization toward "what those 405 look like" (bokeh-bg
  portraits + decontextualized detail crops + a starved 9-image bg bucket).

**The fix is therefore DATA, not epochs** (CLAUDE.md, confirmed): more epochs on the v8 bg-poor set cannot add
background detail. v9 must (a) show the model many selfies/portraits **with sharp, deep-focus backgrounds**, and
(b) be **bigger and more diverse** than 405 to undo the over-specialization.

### Why generic backgrounds are NOT the fix
A landscape/room photo teaches "landscape ⇒ sharp," not "selfie background ⇒ sharp." Per the per-concept
principle, the de-compression must happen **on the use-case concept itself** — selfies/mirror-selfies/portraits
whose whole frame is in focus. Generic environment shots are a weak secondary; the primary fuel is sharp-bg
versions of the user's actual compositions.

### The DoF physics works FOR us
Amateur phone snapshots are **naturally deep-focus** (tiny sensor + wide lens ⇒ whole frame sharp). Bokeh comes
from pro cameras / portrait-mode (fast lens, big sensor). So "amateur snapshot + sharp background" is the *same
physical image* — fixing backgrounds and reinforcing the `amateur snapshot` token are the **same data**.
Corollary: **pro stock (Pexels) is the wrong source** — it skews bokeh + studio, which both fights the amateur
token (CLAUDE.md: v8's blur gate biased pro → amateur weakly trained) and reintroduces the disease.

## Core principle — three independent quality axes

A single sharpness number hides the problem. There are **three orthogonal axes**, and v9's disease lives on
axes 2–3 while the existing gate only controls axis 1-adjacent:

1. **Resolution** — pixel dimensions (`≥1536` controls this).
2. **Background sharpness** — bokeh (bimodal sharpness map: sharp subject blob + soft everything-else) vs
   deep-focus (uniformly sharp). **Nothing in the current pipeline measures this.** ← the v9 centerpiece.
3. **Compression / artifacts** — JPEG mush. Mostly handled for free by a strict overall Laplacian gate
   (compressed/soft images fail it).

A resolution-only filter (the naive "just keep >1536px from the old data") re-imports the disease in HD: a
3000 px portrait-mode shot scores great on axis 1, fails axis 2. The decisive new gate is **axis 2**.

## Approach — bigger, sharp-background dataset; warm-start ep10; same full-FT playbook

Training method = **full-finetune** (unchanged lineage: single-file DiT, warm-startable, matches v5→v8).
Warm-start `v8_epoch10` keeps its texture/lighting/detail wins; the *dataset* is the variable that fixes
backgrounds. Drift controlled by save-every + pick-best.

## Dataset — whole pool, per-image gates, sharp-background filter

**Use the WHOLE historical pool, gated per-image — no source exclusion.** Per-image gates recover the good
amateur+diverse+sharp-bg shots that v8 threw away wholesale by excluding sources.

| Source dir | Have now | Character |
|---|---|---|
| `data/raw` | 6213 (1489 ≥1536 short) | v7 social-media sources — amateur, diverse scenes, compression-prone |
| `data/v8_raw` | 943 (833 ≥1536 short) | Pexels — clean but pro/bokeh/narrow |
| `data/v9_x/` | (new) | X timeline pulls — amateur deep-focus selfies; JPEG-recompressed |
| `data/v9_nsfw/` | (new, manual) | user's manual NSFW downloads |

Pre-gate high-res survivors: ~2322 from the two existing pools; after the sharpness + bg-sharp + dedup gates,
expect **~800–1500 clean sharp-bg pairs** (+ whatever X/NSFW add). This kills the v8 narrowness (405 → ~1k+),
which is itself the anti-over-specialization fix.

**v9 drops v8's bucket ratio-targeting (60/35/5).** Backgrounds are now fixed by the bg-sharpness gate, not by a
"bg bucket." Keep a `source` label in the manifest for stats only; do not gate on ratio.

### Gates (every image must pass all)
1. **≥1536 px short side** — fidelity floor (`<1536` upscales soft).
2. **Overall Laplacian sharpness ≥100** — rejects soft/upscaled/compressed (doubles as the axis-3 filter).
3. **Background-sharpness gate (NEW, grid-patch / "approach C")** — see next section. The axis-2 fix.
4. **phash dedup** (hamming 8) — keep the highest-resolution image per near-dup group.
5. **AR-crop to 0.66–1.5** — Anima DiT pos-emb cap (see hard limit); trims long side only, short side ≥1536 preserved.
6. **Underage hard-block** — WD14 block tags, non-negotiable, legal-adults-only backstop (see Safety).

### Background-sharpness gate — grid-patch (approach C)
**Why grid-patch (not center-vs-edge or segmentation):** robust to subject position (mirror/full-body subjects
span the frame, breaking center=subject assumptions), no ML dependency, and it directly tests the
bimodal-vs-uniform signal that distinguishes bokeh from deep-focus.

**Algorithm:**
1. Tile the (grayscale) image into an `N×N` grid (e.g. `N=4` → 16 tiles).
2. Compute Laplacian variance per tile → an `N×N` array of per-region sharpness.
3. **Decision rule = `passes_bg_sharpness(tile_vars) -> bool`** — *user-authored* (learning-mode contribution).
   This rule literally defines "good background" for the model. Candidate rules to choose among:
   - fraction of tiles above a threshold `T` ≥ some ratio (e.g. ≥60% of tiles ≥ `T`), or
   - median tile variance ≥ `T`, or
   - the *minimum* tile (or 25th-percentile tile) ≥ a floor (penalizes any large soft region).
   The implementer (v9 builder) leaves this function stubbed with `tile_vars` computed; the user fills the body.

**Tunability:** `N`, `T`, and the rule are config/constants. Calibrate on a handful of known bokeh vs known
deep-focus images before the full run (the spec's build step includes a calibration check).

### ⚠️ Hard limit — Anima DiT pos-emb cap (carried from V7/V8)
Pos-emb caps at **120 patches = 1920 px/side** (VAE/8 × patch/2 = ÷16). At 1536 train res, **AR must stay
0.66–1.5**; AR 0.5/2.0 → 2176 px side → 136 patches → `AssertionError`. Enforced at crop time + in the dataset toml.

## Curation script — `src/v9_curate.py` (evolved from `v8_curate.py`)

Reuses stage-1 helpers (`blur_variance`, `phash`, `image_size`, `is_corrupt`, `common.*`) and the v8 pure
helpers (`ar_crop_box`, `_dedup_local`). Pure helpers stay stdlib-only / import-safe for tests.

**Multicore + progress bar (user has 24 cores / 32 logical):**
- **`ProcessPoolExecutor(max_workers=24)`** for the per-image gate pass — processes, not threads: the
  decode + Laplacian + grid-bg + phash mix is GIL-bound (cv2 releases the GIL, pure-Python phash does not), so
  true parallelism needs processes. **Pass file paths to workers, not decoded images** (each worker opens its
  own file → no giant pickled bitmaps across the process boundary).
- **`tqdm`** progress bar over `concurrent.futures.as_completed(...)`.

**Flow:**
1. **Parallel gate pass** (24 workers, tqdm): each image → `{path, source, w, h, blur_var, bg_metric, phash}`
   or `None` if it fails any gate / is unreadable / underage. (Underage check: reuse the WD14 block; if running
   WD14 here is too heavy per-worker, defer the block to stage 3 captioning where WD14 already runs — but keep
   a note that the block MUST run somewhere before training.)
2. **Serial dedup** (`_dedup_local`, keep highest-res per group) — O(n²) on ~2.3k survivors (~2.7M cheap hamming
   ops) → fast; not worth parallelizing.
3. **Parallel crop+copy** (tqdm): AR-crop 0.66–1.5 → save to `data/v9_clean/` (flat, `{source}_{stem}.jpg`,
   quality 95) + write `data/v9_manifest.csv` (columns: `path, source, width, height, phash, blur_var,
   bg_metric, dropped, drop_reason`).

**Tests — `tests/test_v9_curate.py`:** AR-crop math (carried from v8) + the grid bg-sharpness gate on synthetic
fixtures (a uniformly-sharp array passes; a bimodal sharp-center/soft-edges array fails) + dedup keeps-highest-res.

## X sourcing — `scripts/v9_fetch_x.py` (optional but first-class)

Mirror of `v8_fetch_pexels.py`. X has the *right content* (amateur deep-focus selfies/mirror-selfies that
Pexels lacks) but *diseased encoding* (every upload re-encoded to JPEG, typically ≤2048 `large`) → expect
~10% yield through the gates. The gates are the safety net.

**X API reality (verified 2026-06-05, docs.x.com):**
- **Pay-per-use is the only option for new developers** (Free tier discontinued; new Basic/Pro closed;
  Enterprise ~$42k). Credit-based, no subscription, spending limit settable in the dev console.
- **Posts read = $0.005 each**, Media read = $0.005, User read = $0.010, **Owned reads = $0.001** (your own
  account data). Resources deduplicated within a 24h UTC window (re-pull same post same day = free).
- **Timeline `GET /2/users/{id}/tweets`** — max 100/request, paginate via `pagination_token` (classic ~3200-post
  per-account cap), `exclude=replies,retweets` → only the account's own original media. Bearer token (app-only).
- **Full-archive search** (back to 2006) is available on pay-per-use (500/request); recent search = 7 days, 100/request.
- **Media metadata pre-filter:** `expansions=attachments.media_keys&media.fields=url,width,height,type` returns
  pixel `width`/`height` → drop `<1536` *before* downloading. Largest image = append `?format=jpg&name=orig`
  (size aliases: thumb/small 680/medium 1200/large 2048/**orig** ~≤4096). Image file download from
  `pbs.twimg.com` is **free** (CDN, not metered).

**Strategy:** **timeline >> search.** Search firehose wastes $0.005/read on junk; a hand-picked amateur-photo
account's timeline with `exclude=replies,retweets` is near-100% real-photo hit-rate (~1000 posts ≈ $5–10).
Breadth = MORE accounts (3200 cap is per-account), not deeper history. Targeting 10–20 good accounts → a few
hundred targeted images for ~$100.

**Script design:**
- Auth: `X_BEARER_TOKEN` in `.env` (same file as `GEMINI_API_KEY` / `PEXELS_API_KEY`).
- **Primary = timeline mode:** `--handles a,b,c` → resolve each via `GET /2/users/by/username/:handle` → id →
  paginate `GET /2/users/:id/tweets` (`exclude=replies,retweets`, `max_results=100`, expansions + media.fields)
  → for each `type==photo` with `min(w,h)≥1536` → download `name=orig` → `data/v9_x/`.
- **Optional = search mode:** `--query` over recent/full-archive.
- **Cost guard:** `--max-reads` budget cap + running `$` estimate printed; skip already-downloaded `media_key`;
  429 backoff. Print running quota/spend like the Pexels script prints `quota_left`.

**⚠️ Flags (user's call, recorded not enforced):** X Developer Agreement restricts off-platform storage of
content → training-data use is a ToS gray area. NSFW of real people carries consent/legal weight beyond the
underage block — source responsibly.

## Captioning — existing V7 pipeline (no code changes)

- `src/03_caption.py` + `src/gemini_caption.py` (V7 vocab) + WD14 **EVA02-Large v3 @ ~0.25** + Gemini
  structured (18-slot enums + rich NL). Rating via Gemini; NSFW refusals → WD14-tags fallback (carries the
  explicit anatomical detail). This already handles the NSFW slice.
- **Keep `data/gemini_cache.json`** — v9 uses the SAME v7 captioner, so the cache is compatible and incremental
  (only new images get captioned). (Contrast v7→v8 boundary, which required no delete because both are v7-shape;
  v9 is also v7-shape. Do NOT delete.)
- The manifest `source` label is organizational only; the captioner reads images + writes `.txt`, agnostic to source.

## Config — `config/pipeline.yaml` repoint

- `paths.manifest` → `data/v9_manifest.csv`
- `paths.dataset` → `data/v9_dataset`
- `finetune.project_name` → `anima_realism_ft_v9`
- min short side stays `1536`. Stage 4 (`04_build_dataset.py`) consumes `data/v9_manifest.csv` unchanged and
  emits `outputs/anima_realism_ft_v9_dataset_config.toml`.

## Training config — `outputs/anima_realism_ft_v9_train_config.toml`

Diff vs the v8 train toml:
- `transformer_path` → the **`v8_epoch10` keeper** single-file DiT (warm-start source).
- `lr` = **6e-6** (0.000006) — hotter than v8's 4e-6 (CLAUDE.md notes headroom: zero drift seen at 4e-6), cooler
  than v7's 8e-6. The bigger/cleaner dataset can take it; faster convergence on more data.
- `epochs` = **20** (ceiling; pick best, stop early via save-every). ep10 was still climbing → go longer.
- `[optimizer] type` = `adamw_optimi` (fp32) — unchanged.
- `save_every_n_epochs = 1`, `llm_adapter_lr = 0` (freeze Qwen3), `caption_dropout_percent = 0.1`,
  `shuffle_tags = false`, `tag_dropout_percent = 0`, `activation_checkpointing = true`, `warmup_steps = 100`
  (weights-only warm-start re-warms a fresh optimizer) — unchanged.
- `output_dir` → `…/anima_realism_ft_v9`.

### `outputs/anima_realism_ft_v9_dataset_config.toml`
- `resolutions = [1536]`, `min_ar = 0.66`, `max_ar = 1.5`, dataset path → the v9 clean set.

### VRAM / Blackwell
1536 full-FT fit on the 80–96 GB cards used for v7/v8. OOM fallbacks in the toml: `[model] qwen_nf4 = true`
and/or `[optimizer] adamw8bit` (needs bitsandbytes). Verify torch sees the GPU
(`python -c "import torch; print(torch.cuda.get_device_name(0))"`) before launch; upgrade torch if old.
Note: bigger dataset + 20 ep ⇒ longer wall-clock per epoch than v8's ~12 min (more images/epoch).

## Eval protocol — eval images are the ONLY signal (loss is blind)

Flow-matching loss is blind to perceptual quality (flat-noise across all of v5–v8, re-confirmed 4×). Judge
**only by eval images**, on **every saved epoch + the `v8_epoch10` baseline** (apples-to-apples: did v9 beat
ep10?).

- Reuse the v8 frozen prompt set + fixed seeds, and **ADD:**
  - **(a) background canaries** — the core v9 target: `selfie in a detailed living room (bookshelf, window,
    plants)`, `mirror selfie in a messy bedroom`, `portrait on a busy city street`, `kitchen background`. Watch
    whether the background renders sharp + coherent vs blurry/slopish.
  - **(b) amateur-drift canary** — whole-person amateur snapshot, to catch aesthetic drift toward stock at the
    hotter 6e-6 (the primary regression risk).
  - **(c) NSFW capability check** — confirm the explicit capability survived (and its backgrounds improved too).
- **Inference settings:** CFG **3–4.5** (Anima flow-matching DiT oversaturates at high CFG), optional RescaleCFG
  ~0.7, sampler `euler`/`dpmpp_2m` + `simple`/`beta`, steps 20–30, VAE `qwen_image_vae.safetensors`.
- **Dial:** prompt `amateur snapshot, best quality, highres, sharp` + negative
  `jpeg artifacts, compressed, blurry, low quality, bokeh, blurred background`.
- **Stop signals:** (1) amateur→stock drift = stop / lower LR; (2) backgrounds sharpen but composition variety
  collapses = overfit → pick the prior epoch. Pick the best epoch regardless of number.

## Vast execution

1. Upload `v8_epoch10.safetensors` → Drive (get a new file ID for `vast_fetch_v9.sh`). Local source:
   `…\ComfyUI\models\diffusion_models\v8_epoch10.safetensors`.
2. Build `data/v9_dataset.zip`, upload → Drive (get ID).
3. `scripts/vast_setup.sh` — clone diffusion-pipe + the 3 Anima base models (unchanged).
4. `scripts/vast_fetch_v9.sh` — gdown dataset + ep10 (IDs as short `VAR=...` lines; size/count checks).
5. `scripts/run_v9_train.sh` — copy tomls into place + `nohup deepspeed --num_gpus=1 train.py …`, log →
   `/workspace/train_v9.log`. Watch `tail -f`.
6. Checkpoints in `outputs/anima_realism_ft_v9/<ts>/epoch{1..20}/`.

### ⚠️ Pre-destroy checklist (the v3/v4-loss lesson)
Before destroying ANY instance: (a) best epoch on local disk, **verified ~4.18 GB single-file DiT** (not
truncated); (b) FULL `train_v9.log` downloaded; (c) optional neighbor epoch as fallback. THEN **destroy** (not
stop — stop still bills storage). **~$1.3/hr; destroy the moment train+download finish.**

### Vast terminal gotcha (carried over)
Jupyter wraps lines >~95 chars + hangs on pasted heredocs → use `git fetch` + short scripts + `VAR=val` split
lines, never long pastes. Tomls force-added to `v5-build` so they checkout/curl.

## Files to produce

- `src/v9_curate.py` + `tests/test_v9_curate.py` (incl. grid bg-sharpness gate with the user-authored
  `passes_bg_sharpness` rule).
- `scripts/v9_fetch_x.py` (X timeline/search → `data/v9_x/`).
- `outputs/anima_realism_ft_v9_train_config.toml`, `outputs/anima_realism_ft_v9_dataset_config.toml`.
- `scripts/run_v9_train.sh`, `scripts/vast_fetch_v9.sh`.
- `config/pipeline.yaml` repoint (manifest/dataset/project_name → v9).
- Frozen eval-prompt doc (background + amateur-drift + NSFW canaries), alongside this spec.
- This spec.

## Safety (unchanged hard boundary)

**Legal adults only.** The WD14 underage hard-block (`loli`/`shota`/`child`/…) stays ON and non-negotiable as a
backstop, even though the user controls NSFW sourcing. Gemini core child-safety is always-on. Anything flagged
is dropped, no override.

## Out of scope (separate tracks)

- **Anime → real / preserving anime priors** — explicitly dropped this run (goal is pure photoreal).
- **Small faces in wide shots** — inference track (ADetailer/FaceDetailer + hires-fix). Not data-fixable at 1536.
- **Hands/feet structure at inference** — HandDetailer/ADetailer remains the cheap lever for residual cases.
- **Retraining realism from scratch** — not needed; ep10 has realism + fidelity. v9 only fixes backgrounds +
  extends style.

## Risks

- **Bokeh slips past the gate** → re-teaches soft backgrounds = the disease persists. Mitigation: calibrate the
  grid bg-sharp rule on known bokeh/deep-focus samples before the run; the rule is the centerpiece — get it right.
- **Amateur → stock drift** at 6e-6 / 20 ep (hotter+longer than v8). Mitigation: save-every + pick-best (keep any
  winning epoch, even early) + amateur-drift eval canary. **Primary regression risk.**
- **X yield lower than hoped / ToS** — X images mostly fail the gates (~10% yield); cost guard + targeting good
  accounts keeps spend bounded; X is optional supplement, not the dataset's backbone.
- **Over-cropping / AR miss** → `AssertionError` crash at 1536. Mechanical AR gate must run on every image.
- **Underage block bypassed by per-worker complexity** — ensure the block runs somewhere before training (curate
  worker or stage 3). Non-negotiable.
- **Checkpoint loss** — the recurring failure mode. Pre-destroy checklist is mandatory.

## Success criteria

- Eval shows, vs the `v8_epoch10` baseline on fixed prompts+seeds: **selfie/mirror-selfie/portrait backgrounds
  render sharp + coherent** (not blurry/slopish), with the **amateur aesthetic intact** (no stock drift) and
  **NSFW capability preserved**.
- A chosen best epoch downloaded + verified locally (~4.18 GB single-file DiT) before instance teardown.
- The grid bg-sharpness gate demonstrably separates bokeh from deep-focus on the calibration samples.

## Strategic note (recorded for the project memory)

The user evaluated and **rejected** an anime→real pivot this session. Reasoning: community Anima realism models
retain anime knowledge (anime character → real person) because they preserve concept knowledge (LoRA / mixed
anime+real data / regularization) and add realism as a *style*; this project instead full-finetuned on pure
photos across ~53 cumulative epochs, treating the anime prior as the enemy → ~70% of anime elements lost. The
user's goal is a **pure photoreal amateur + NSFW generator**, for which anime preservation is irrelevant, so v9
optimizes only for that. (If anime→real is ever wanted, it is a *separate* project: warm-start from a less-cooked
checkpoint that still holds anime — base / early v5 — with preservation-aware training, NOT a continuation of ep10.)
