# CLAUDE.md — Anima Realism finetune project

> Portable project memory. Lives in the project folder so it survives moving to another drive.
> **CURRENT = V10 BUILT (code done, NOT trained) — clean RESTART. Photoreal render-style finetune that PRESERVES the
> base's concept/character knowledge. Warm-start BASE DiT, gentle lr 6e-6, 50ep save-every-5, pick-best on a
> concept-retention eval.** Spec: `docs/superpowers/specs/2026-06-06-anima-realism-v10-design.md`; plan:
> `docs/superpowers/plans/2026-06-06-anima-realism-v10.md`; eval: `docs/superpowers/specs/2026-06-06-v10-eval-prompts.md`.
> (Prior: V8 trained, keeper `v8_epoch10.safetensors`; v9 background-plan superseded by v10.) v8 specs:
> `docs/superpowers/specs/2026-06-04-anima-realism-v8-fidelity-refiner-design.md`.
> V7 captioning spec: `docs/superpowers/specs/2026-06-03-anima-realism-v7-captioning-design.md`.
> Prompt guide: `docs/v7-prompt-guide.md`. History: v5 (`2026-06-02-...v5-design.md`), v6 probe, v2/LoRA (abandoned).

## Status (2026-06-06) — V10 BUILT (code complete, NOT trained yet). Clean restart, new philosophy.

**V10 = photoreal RENDER-STYLE finetune that PRESERVES base concepts** (different goal from v5–v9, which tried to
*erase* the anime prior). User reframed: output = 100% photoreal, but keep the base's concept/character *knowledge*
(prompt a known concept → get it rendered as a photo). Branch `v5-build`. **Code done + committed; pipeline NOT run,
not trained.**

**Recipe (all locked with user):**
- **Warm-start BASE** `anima-base-v1.0.safetensors` (max concept knowledge; v8 ep10 already drifted over 4 photo gens).
- **Data = `data/raw`** (6396 readable), floor **1280** short side (~2699 candidates pre-gate) → train **1536**.
- **NEW curation `src/v10_curate.py`** — measures REAL high-freq detail, not pixels (a 6k upscale can be soft; a 1.3k
  photo can be sharp). Gates: scale-aware sharpness (Laplacian on a 512px copy) + **FFT high-freq energy ratio** (the
  fake-big/upscale killer) + JPEG quant-table quality + 8×8 blockiness + phash dedup + AR-crop 0.66–1.5. Emits ALL
  metrics to `data/v10_manifest.csv`; `--calibrate N` prints percentiles. ⚠️ **Thresholds (`SHARP_MIN/FFT_MIN/
  JPEG_Q_MIN/BLOCK_MAX`) are placeholders (0/0/0/1e9) until CALIBRATED** from the real distribution (Task 4, user-run).
  Note: `blockiness` is *negative* for clean photos (compressed pushes it up); raw JPEGs ~q90 so real filtering = sharp+fft.
- **Captions = GEMINI-ONLY structured** (`src/v10_caption_gemini.py`). One `gemini-3-flash-preview` call per image
  (`response_schema` JSON) returns three layers: the **v7 enum rubric** (real-photo subset: quality/shot_type/view/
  camera_angle/capture_style/lighting/condition/color_grade/camera_lens/depth_of_field/expression/body_type/
  breast_size/ethnicity/skin_tone/setting_type) + a **real-photo tag list** + a **50–100 word NL paragraph**.
  Assembled = `<quality>, <rating>, <enums>, <tags>[, watermark], <paragraph>`. Prompt/structure
  adapted from the user's **lenstag-ai** app + the v7 rubric. **RAM++, WD14, Falconsai all DROPPED** (RAM++ = noisy flat
  keywords; WD14 underage gate = anime tagger that false-positives on real adult photos and isn't a real-photo minor
  detector → removed on the owner's assertion the set is all legal adults). **`quality` Gemini-judged per-image** (v7
  booru ladder `masterpiece..worst quality`, REQUIRED) — NOT a fixed prefix, gives real contrast; no `score_7`.
  **rating = simple `safe`/`suggestive`/`explicit`** (no `rating:` prefix; REQUIRED).
  **BLOCK_NONE → captions NSFW fine** (verified; ~45% explicit). ⚠️ Gemini 3 = THINKING model →
  `ThinkingConfig(thinking_budget=0)` or thinking tokens truncate the JSON. Thread pool + resumable cache
  (`data/v10_caption_cache.json`, path→raw JSON), idempotent rebuild. Build = **reuse `src/04_build_dataset.py`**;
  re-run build + `scripts/v10_zip.py` after captioning.
- **Train** `outputs/anima_realism_ft_v10_train_config.toml`: base warm-start, **lr 6e-6**, **50 epochs**,
  **save_every_n_epochs=5** (10 ckpts ~42GB), adamw_optimi fp32, freeze Qwen3 (`llm_adapter_lr=0`), full-FT (no [adapter]),
  AR 0.66–1.5. `scripts/run_v10_train.sh` + `scripts/vast_fetch_v10.sh` (dataset only — base DiT comes from vast_setup.sh).
- **Concept preservation lever = pick-best epoch, NOT low LR.** Two frozen eval sets (`...v10-eval-prompts.md`): photoreal
  (should climb) + concept-retention (`masterpiece, best quality, safe, <concept>` prompts) → pick the last epoch where
  concepts still render AND photoreal is strong.

**Anima HF facts (read 2026-06-06, drove v10):** base trained on anime + ~800k non-anime *artistic* images **with photos
FILTERED OUT** → zero photographic prior → photoreal is a real shift (50ep justified). Authors: keep LLM adapter LR=0;
model needs a **"light touch" due to existing diversity** (= the concepts we keep). Base-native prompt = booru/score
(`masterpiece, best quality, score_7, safe, [char] [tags]`). Author infer: steps 30–50, **CFG 4–5**, sampler `er_sde`/
`euler_a`/`dpmpp_2m_sde_gpu`, scheduler `beta57`, res ≤1536 — matches our prior low-CFG finding.

**PROGRESS (2026-06-06): curate DONE (2069 kept), Gemini caption layer BUILT + smoke-verified (rich captions).**
**NEXT (user-run runtime, local 4080 then Vast):** (1) ✅ curate done (`data/v10_clean`, 2069). (2) `python
src/v10_caption_gemini.py` (Gemini structured caption; resumable cache; uses GEMINI_API_KEY). (3) `python
src/04_build_dataset.py` → `python scripts/v10_zip.py` → upload `data/v10_dataset.zip` to Drive. (4) Rent 96GB Vast →
`vast_setup.sh` → `vast_fetch_v10.sh <ID>` → `run_v10_train.sh` → eval ep5..50 → DOWNLOAD pick-best + log BEFORE destroy.
⚠️ Must `git push origin v5-build` before the Vast clone (v10 code is local-only until pushed).

## Goal

Make the **Anima** diffusion model (anime base; Qwen3-0.6B TE + Qwen-Image VAE) output **realistic photos** —
a domain shift fought against the anime prior. Community realism finetunes of Anima exist on Civitai, so it works.
**(v10 reframe: keep the base's concept/character *knowledge*, change only the *render style* to photoreal.)**

- **Current = v5:** full finetune (not LoRA) **from base DiT at 1024**, on a rented **Vast.ai A100-80GB**,
  via **diffusion-pipe**, on ~1942 Gemini-captioned sharp photos. (LoRA on local 4080 abandoned: 16GB can't fit a
  full finetune, and the anime→photo shift is too large for LoRA.)

## Status (2026-06-05) — V8 TRAINED. ep10 = great keeper, STILL CLIMBING. Instance destroyed. NEXT = v9 (backgrounds).

**V8 SUCCESS — keeper so far = `v8_epoch10.safetensors`** (much better than ep17: more detail/texture/lighting,
"already usable"). Eval climb: **ep5 much better → ep8 > ep5 → ep10 still improving (NOT plateaued).** Same
undertraining pattern a 4th time — at **lr 4e-6** (gentler than v7's 8e-6) 10 epochs is too few to converge. Loss was
flat-noise (0.02–0.34, no trend, 40 prints) while quality climbed → **loss BLIND, reconfirmed.** Run clean (10 ep
saved, no OOM/NaN). **Downloaded locally + verified 4.18 GB: ep2/ep5/ep8/ep10 + `train_v8.log`. Instance DESTROYED**
(everything saved first; ~$1.3/hr Vast, ~12 min/epoch so 10ep ≈ 2 hr).

**⚠️ Two open wants (2026-06-05):** (1) fully learn lighting/style → MORE EPOCHS (still climbing) — warm-start ep10,
extend; (2) **messy/low-quality backgrounds in portraits → NOT an epochs fix, a DATA GAP.** bg bucket was starved
(**9 imgs, 2%**) + detail/anchor sources have blurred/bokeh backgrounds → model never saw sharp coherent backgrounds →
can't render them. **Fix = v9 dataset: add sharp deep-focus environment/scene shots + sharp-background portraits +
a real bg bucket.** Recommended: **v9 = warm-start ep10 + background-enriched dataset** → gets the extra style epochs
AND fixes backgrounds in one run (vs extending on the same bg-poor data, which can't add backgrounds).

**V8 build recap.** Goal: erase the learned **compression look** + add high-freq detail (fingers/clothes/phones/toes)
while **keeping amateur.** Warm-start `V7_epoch17.safetensors`, full-FT, lr 4e-6, save-every, pick-best.

**Core principle — fidelity ⟂ aesthetic.** Honest split-axis captioning: quality ladder + resolution = fidelity;
`capture_style` (amateur snapshot/casual phone) = aesthetic. Keeps "clean" from binding to "studio." **Inference
dial:** prompt `amateur snapshot, best quality, highres, sharp` + negative `jpeg artifacts, compressed, blurry,
low quality`. Full-FT (not LoRA) chosen because erasing compression = moving base weights (LoRA only adds a delta on
top of still-compressed ep17 → residual leaks).

**⚠️ KEY LEARNING — compression is learned PER CONCEPT, not globally.** The model outputs soft mirror-selfies because
its *mirror-selfie training data was the compressed social-media ones* → it believes "mirror selfie = soft." It
de-compresses a concept **only by seeing that concept rendered clean.** Generic clean hands teach clean *hands*, not
clean *mirror selfies*. Consequences: (a) **anchor bucket = clean versions of the user's ACTUAL outputs** (mirror
selfies, their compositions), NOT random generic people — that de-compresses real use-cases AND reinforces specifics
(kills the dilution worry). (b) **detail crops** (decontextualized hands/fabric/feet) = pure texture/fidelity, ~zero
concept dilution → safe. (c) **ep17 = concept memory; generic data = texture teacher; keep the nudge gentle (low lr,
few epochs) or the specifics erode.** (d) Do NOT train on the user's current soft images — reinforces the disease.

**Dataset — ALL-CLEAN, ZERO compressed.** Own originals = the compressed disease → EXCLUDED. Sourced HIGH-RES from
**Pexels API** (`scripts/v8_fetch_pexels.py`, 3–6k px, filtered ≥1536 before download; free key in `.env` as
`PEXELS_API_KEY`, 25k/mo). **Open Images is a DEAD END (≤1024 px)** — `scripts/v8_fetch_openimages.py` kept for
reference only. Buckets `data/v8_raw/{detail,anchor,bg}` (target 60/35/5; actual = anchor-heavy 104/311/9, user chose
proceed: anchor=their compositions=high-leverage). **Built = 405 img+txt pairs** (4 underage-blocked, 84 explicit→
Gemini-refused→WD14-tags fallback, ~15 vanished mid-run). **Known issue:** detail bucket ~1/3 contaminated (scenery/
whole-person, not body-part crops) — user chose proceed-as-is (all clean, ~8% off-target, de-compression goal intact).

**Curation = `src/v8_curate.py` (REPLACES stage 1 for v8):** gates ≥1536 short side + Laplacian sharpness ≥100 +
phash dedup + **AR-crop to 0.66–1.5** (pos-emb 120-patch / 1920px cap), bucket-labeled → `data/v8_clean/` +
`data/v8_manifest.csv`. Existing **stage 3 (caption)** + **stage 4 (build)** consume it unchanged.
`config/pipeline.yaml` repointed: `paths.manifest`→`data/v8_manifest.csv`, `paths.dataset`→`data/v8_dataset`,
`finetune.project_name`→`anima_realism_ft_v8`. (v8 uses the SAME v7 captioner → no need to delete `gemini_cache.json`.)

**V8 files (branch `v5-build`, PUSHED 890f2d6):** spec + plan + eval-prompts; `outputs/anima_realism_ft_v8_{train,
dataset}_config.toml`; `scripts/{run_v8_train.sh, vast_fetch_v8.sh, v8_fetch_pexels.py, v8_fetch_openimages.py}`;
`src/v8_curate.py` + `tests/test_v8_curate.py` (6 green). `data/v8_dataset.zip` (1.45 GB) uploaded to Drive
(ID `1GurWF_pqHHSAj_pGWQsxqWOH8Wfbd4Y2`; ep17 ID `1B7CiuSJBecf3OmEEwP3efON1Nnni9YDW`). Trained ckpts
`v8_epoch{2,5,8,10}.safetensors` downloaded to `…\ComfyUI\models\diffusion_models\` (each 4.18 GB; **ep10 = keeper**).
Pexels gotcha fixed: API 403 on default urllib UA (Cloudflare) → added `User-Agent`. Vast paste-wrap gotcha: long
arg lines split → pass IDs as short `VAR=...` lines.

**NEXT = v9 (fix backgrounds + squeeze more lighting/style).** Warm-start `v8_epoch10.safetensors` (zero penalty vs
continuing — diffusion-pipe is weights-only warm-start). Steps: (1) upload `v8_epoch10.safetensors` to Drive (get ID).
(2) **Build a background-enriched dataset** — the messy-background fix: keep the 405 (or re-curate) + ADD **sharp
DEEP-FOCUS environment/scene shots + sharp-background (not bokeh) portraits + a real bg bucket** (Pexels queries:
`living room interior`, `bedroom`, `city street`, `landscape sharp`, `office interior`; favor deep focus). Re-run
`v8_curate.py` → `03_caption.py` → `04_build_dataset.py`. (3) New train toml warm-start ep10, lr 4e-6 (or try 6e-6 —
no drift seen at 4e-6 = headroom), save-every, pick-best. (4) Rent Vast, fetch, run, eval vs **ep10 baseline**
(watch amateur-drift), DOWNLOAD best BEFORE destroy. **Cost note: ~12 min/epoch, ~$1.3/hr — destroy the instance the
moment training+download finish; never leave it idle.** ⚠️ Backgrounds are DATA-bound, not epochs — more epochs on the
bg-poor v8 data will NOT add background detail.

## Status (2026-06-04) — V7 DONE. Keeper = epoch18 label (on-disk file = V7_epoch17.safetensors). Instance destroyed after download.

**V7 SUCCESS — keeper = `epoch18`** (downloaded locally; instance shut down after verifying download + log).
Full-finetune, warm-start from the v6 keeper (`anima_v6_keeper.safetensors` = v6 epoch25), **lr 8e-6 (VALIDATED),
1536 res, adamw_optimi, save-every-epoch.** Trained ~18 epochs (~40 min/ep, ~1.12 s/iter on RTX 6000 Blackwell 96 GB).
**Eval verdict:** ep5 "slightly better" → ep10 "normal body, 5 fingers, clear face, better detail" → **ep18 "much
better"** = the keeper. Loss stayed flat-noise 0.067–0.124 the whole run (flow-matching loss is **blind to quality**;
lr fixed 8e-6 by design = no decay → judged ENTIRELY by eval images, not loss/lr). Climb ep5→ep18 confirms the v6
"undertraining, lr 8e-6 fine" finding once more.

**⚠️ INFERENCE — LOW CFG (saturation finding).** Anima = flow-matching DiT → **oversaturated / high-contrast =
CFG-too-high signature, NOT undertraining.** Fix at sample time: **CFG ~3.0–4.5** (not the SDXL 6–8 habit), optional
RescaleCFG ~0.7, sampler `euler`/`dpmpp_2m` + `simple`/`beta`, steps 20–30 (steps fix detail not color), confirm VAE =
`qwen_image_vae.safetensors`. User confirmed lower CFG fixed it.

**Known-OK weakness:** some outputs soft/low-quality — traced to training sources **<1536 that upscaled soft**
(`upscale = no new detail`, per pipeline note). Not a model fault. → fixed by the next run (V7-HD below), not more epochs.

**NEXT = V7-HD high-pass detail run (planned, not started):** warm-start from **ep18** → train on **≥1536 / 4k native
originals** to add high-frequency sharpness while keeping the realism ep18 already has. Same AR 0.66–1.5 pos-emb limit
applies. Needs: 4k dataset prep (re-run stages 1/3/4 with ≥1536 sources), upload, fresh/cheaper Vast host, new train toml
(transformer_path → ep18). lr 8e-6 + save-every + pick-best still the playbook.

**v6 PROBE RESULT:** extend-v5 @8e-6 → cum-epoch 25 much better + still climbing = **UNDERTRAINING confirmed, lr 8e-6
fine** (not LR-limited). v6b higher-LR arm dropped. (v6 keeper = epoch25, used as V7 warm-start.)

**V7 dataset (LOCAL prep done):** ~6200 new raw → stages 1/3/4 → **2103 img+txt @1024 floor** (blur 100), 1536 train.
Captions = **V7 vocab: NO `realistic photo` anchor, 18 enum axes** (shot/view/angle/quality-booru-ladder/capture/
lighting/condition/color/lens/dof/expr/body/breast/ethnicity/skin/setting/rating) + **WD14 EVA02 @0.25** booru tags +
rich NL. **80% full Gemini / 20% tags-only fallback** (NSFW refusals → EVA02 carries detail), **23 underage-blocked**.
Known-OK noise: ~20% fallback ratings from Falconsai undercall (suggestive→general). `data/v7_dataset.zip`.

**⚠️ HARD LIMIT — Anima DiT pos-emb caps at 120 patches = 1920 px/side** (VAE/8 × patch/2 = ÷16). At 1536, **AR must
stay 0.66–1.5** (0.5/2.0 → 2176 px → 136 patches → AssertionError crash, hit + fixed). Baked into dataset toml + config.

**INFERENCE:** generate at **~1536 area** (1536×1536, 1344×1728, 1856×1280; AR 0.66–1.5, dims ÷64) for full detail;
lower res OK for drafts down to ~1 MP (warm-start from 1024-v6 keeps it forgiving). Don't gen native >1536-area (dup
artifacts). Prompt guide + 10 test prompts: `docs/v7-prompt-guide.md`. Weak/dead triggers: grainy/soft/motion-blur
(blur gate), rating:questionable (22), close-up (27).

**V7 files (all on `v5-build`):** `outputs/anima_realism_ft_v7_{train,dataset}_config.toml`, `scripts/run_v7_train.sh`,
`scripts/vast_fetch_v7.sh` (gdown dataset+keeper, IDs as args). Captioner: `src/gemini_caption.py` (rewritten v7),
`src/03_caption.py`, `config/pipeline.yaml`. Tests green (40).

**NEXT SESSION (V7-HD):** keeper `epoch18` is DOWNLOADED. (1) Build 4k/≥1536-source dataset (stages 1/3/4, raise the
source-quality floor — the soft-image fix). (2) New train toml warm-starting from ep18. (3) Rent Vast, upload, run,
eval, pick best, DOWNLOAD before destroy. Same lr 8e-6 / save-every / pick-best / AR 0.66–1.5 rules.
⚠️ Confirm `train_v7.log` (full, to ep18) was pulled before the V7 instance was destroyed — hand to Claude for record.

---

## Status history (2026-06-02) — v5 (superseded by v6 probe + V7)

**v5.** Pipeline rebuilt + validated end-to-end (subagent-driven, all tests green); training
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
