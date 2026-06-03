# Anima Realism v6 — design (extend-v5 convergence probe)

> Date: 2026-06-03 · Branch: `v5-build` · Supersedes the old "v6 plan" in `CLAUDE.md`
> Predecessor spec: `2026-06-02-anima-realism-v5-design.md`

## Goal

Push **overall photorealism** further than v5. v5 (1024, from base, lr 8e-6, 20 epochs, 1942 imgs)
succeeded as a photoreal base but its quality curve was **still climbing at epoch 20** (best = last
checkpoint, no plateau). v6 answers one question for minimal Vast spend:

> **Is v5's ceiling a matter of not-enough-training, or of too-low a learning rate?**

This is NOT a fine-detail run. Small faces / hands / background were diagnosed resolution-bound and
are handled on a separate **inference track** (see Out of Scope).

## Diagnosis — what we KNOW vs ASSUME

- **KNOW (measured):** v5 quality rose monotonically; epoch 20 = best; no plateau. → direct evidence
  of **undertraining** (we stopped early).
- **ASSUME (not measured):** "8e-6 too gentle." We never observed a plateau, so there is **zero
  evidence** the LR was too low — only that we ran too few steps. The `CLAUDE.md` "LR too gentle" line
  is interpretation, not data.
- **Dataset: ruled out** for this goal. v5 produces good overall realism → data is sufficient to learn
  the photo domain. Dataset only limits fine detail (subject-focused framing), already routed to inference.

Key point: **"more epochs" vs "higher LR" is mostly a false dichotomy** — both raise effective learning
progress (≈ LR × steps). A still-climbing curve can't distinguish them; **only a plateau can.**

## Approach — extend-v5 probe (one variable)

Warm-start the v5 epoch-20 weights and **continue training with everything identical to v5 except the
starting weights.** Changing exactly one thing (more steps) makes the result interpretable.

- **Warm-start:** `transformer_path` → v5 `epoch20.safetensors` (instead of base DiT).
- **lr:** `8e-6` (same as v5 — this is the whole point).
- **epochs:** `5` (continuation; cumulative exposure ≈ 25 epochs). Fast cheap peek; extend if still climbing.
- **Everything else byte-identical to v5:** 1024 res, micro_batch 1 / grad_accum 1, `adamw_optimi` (fp32),
  betas [0.9, 0.99], weight_decay 0.01, warmup_steps 100, save_every_n_epochs 1, Qwen3 frozen
  (`llm_adapter_lr=0`), `caption_dropout_percent=0.1`, `shuffle_tags=false`, `tag_dropout_percent=0`.
- **Dataset + captions:** unchanged from v5 (same 1942-image dataset, same `.txt` captions). Identical
  content = clean test.

### Warm-start caveat
diffusion-pipe loads **weights only**, not Adam optimizer moments → the first ~100 steps re-warm the
moments (a fresh optimizer on adapted weights). `warmup_steps=100` already covers this. Expect a
possible **small epoch-1 dip** that recovers — judge the trend, not epoch 1 in isolation.

### VRAM — RTX 6000 Pro 96 GB (fp32 optimizer, byte-clean)
Run on the **RTX 6000 Pro (Blackwell, 96 GB)**. v5 used ~50 GB at 1024 → ample headroom, so keep v5's
**fp32 `adamw_optimi`** — the probe is byte-identical to v5 (no 8-bit optimizer confound). No bitsandbytes.

**Blackwell (sm_120) compat caveat:** the instance image must ship a recent CUDA (≈12.8+) + torch (≈2.7+);
older wheels lack Blackwell kernels and crash at first CUDA op. Verify the Vast image's torch sees the GPU
(`python -c "import torch; print(torch.cuda.get_device_name(0))"`) before launching. If diffusion-pipe's
pinned deps are too old for Blackwell, upgrade torch in the instance.

(History: an interim plan targeted a 40 GB A100 via `adamw8bit`; the 96 GB card makes that unnecessary.
96 GB also makes a future **1536 fine-detail v6b** fit on one card without nf4/offload.)

### Decision rule (reads the curve)

| Curve shape over the 5 epochs | Conclusion | Action |
|---|---|---|
| Still climbing | Pure **undertraining**; LR was fine | Pick best epoch; extend again (+N) if still rising |
| Plateaus / regresses | 8e-6 **ceiling** reached → LR *was* the limiter | Escalate to **v6b**: fresh-from-base at lr 1.5–2e-5, 18–20 ep |

## Config changes

### `outputs/anima_realism_ft_v6_train_config.toml`
Diff vs the currently pre-staged v6 toml (which is lr 1.5e-5 / 15ep / from-base):
- `transformer_path = '/workspace/anima/models/anima_v5_epoch20.safetensors'`  *(was base DiT)*
- `lr = 8e-06`  *(was 1.5e-05)*
- `epochs = 5`  *(was 15)*
- `[optimizer] type`: **unchanged** — `adamw_optimi` (fp32), same as v5 (96 GB card, no 8-bit needed).
- `output_dir = '/workspace/anima/outputs/anima_realism_ft_v6'`  *(unchanged)*
- all other keys unchanged (already match v5).

### `outputs/anima_realism_ft_v6_dataset_config.toml`
**No change.** `resolutions = [1024]`, same dataset path, AR buckets as v5.

## Eval protocol (makes the curve readable)

v5 was judged ad hoc. v6 fixes this so climb-vs-plateau is evidence, not vibes:

- **Frozen prompt set:** ~10 prompts spanning the v5 caption vocab — close-up / medium / full-body
  framings × varied lighting + capture-style tokens (`masterpiece, best quality`, `professional
  photograph`, `golden hour`, etc.). Each prompt leads with the `realistic photo` anchor.
- **Fixed seeds** per prompt (same seed list every epoch).
- **Baseline included:** run the SAME prompts+seeds on **v5-epoch20** so v6 epochs are compared
  apples-to-apples against the starting point (did we actually move past v5?).
- Generate every saved epoch in ComfyUI; compare side-by-side.
- **Plateau = no perceptible overall-realism gain across 2 consecutive saved epochs.**

(The frozen prompt list + seeds live alongside this spec / in the run notes so v6b reuses them.)

### Training log (quantitative corroboration)
`run_v6_train.sh` redirects all training output to `/workspace/train_v6.log`. With
`steps_per_print=100` and ~1942 steps/epoch, the log accumulates ~19 loss prints/epoch (~97 over the
5-epoch probe) — a **quantitative loss-vs-step trace**.

- **Deliverable:** after the run, download `train_v6.log` from the Jupyter file browser and hand it
  back for analysis. The loss trend cross-checks the visual climb-vs-plateau call.
- **Caveat:** flow-matching loss (`timestep_sample_method='logit_normal'`) is noisy and not strictly
  monotonic with perceptual quality. Use as **corroboration** of the visual eval, not the sole signal:
  a still-falling loss supports "still learning"; a flattened loss supports "plateau."

## Vast execution

Assume the v5 instance was destroyed → **fresh A100-80GB**.

1. `scripts/vast_setup.sh` — clone diffusion-pipe + download the 3 Anima base models.
2. Re-upload the dataset — `v5_dataset.tgz` already exists in repo root (reuse; same content as v5).
3. **Upload `epoch20.safetensors` (4.18 GB)** to `/workspace/anima/models/anima_v5_epoch20.safetensors`
   (same GDrive+gdown path used for the v5 dataset, or scp). This is the warm-start source.
4. `scripts/run_v6_train.sh` — copies tomls into place + `nohup deepspeed --num_gpus=1 train.py ...`.
   Watch `tail -f /workspace/train_v6.log`.
5. Checkpoints in `outputs/anima_realism_ft_v6/<ts>/epoch{1..5}/`.
6. **Download `train_v6.log`** (Jupyter file browser) for loss-trend analysis.
7. **Download the best epoch immediately, THEN destroy the instance** (no v3/v4 loss repeat).

Local source of the warm-start file:
`C:\Users\erede\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\epoch20.safetensors`

### Vast terminal gotcha (carried over)
Jupyter terminal wraps lines >~95 chars + hangs on pasted heredocs → use `git fetch` + the short
`scripts/run_v6_train.sh`, never long pastes. Tomls are force-added to `v5-build` so they `curl`/checkout.

## Out of scope (separate tracks — documented, not built here)

- **Fine detail (small faces / hands / background)** → **inference track**: ADetailer / FaceDetailer +
  HandDetailer + hires-fix in ComfyUI. This is the cheaper, correct lever for resolution-bound detail.
- **Richer captions + Gemini-emitted safety tag** → deferred. Confounds this probe and forces a full
  recapture (Gemini cost + re-run stages 1/3/4 + re-cache latents). Its own future experiment.
- **`min_resolution: 768` (more data) and 1536 training** → deferred. Dataset isn't the realism
  bottleneck; 1536 risks OOM on A100-80GB. Revisit only if the inference track is insufficient.

## Escalation — v6b (only if probe plateaus)

Fresh-from-base at **lr 1.5–2e-5, 18–20 epochs**, same dataset/captions, save every epoch. Tests the
higher-LR ceiling directly. Reuse the frozen eval prompt set + seeds for direct comparison.

## Risks

- **Probe still climbing at +5** (likely, given v5 climbed all 20) → not a failure; just extend. Budget
  for a possible second continuation.
- **Warm-start instability** (Adam re-warm) → mitigated by warmup_steps=100; watch epoch-1 dip.
- **Checkpoint loss** → the recurring failure mode; download best epoch before destroying the instance.
- **Eval subjectivity** → mitigated by frozen prompts + fixed seeds + v5-ep20 baseline.

## Success criteria

- A clear read on the decision-rule table: either (a) confirmed continued realism gain with a chosen
  best epoch downloaded, or (b) confirmed plateau that justifies v6b's higher LR.
- Best checkpoint downloaded locally before instance teardown.
