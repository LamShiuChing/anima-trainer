# Anima Realism Finetune — v5 Design

> Date: 2026-06-02. Supersedes v2/v3/v4 recipes. Full design history in
> `2026-06-01-anima-realism-finetune-v2-design.md`. This doc = the v5 delta + converged pipeline.

## Why v5

v3/v4 outputs were **soft, still anime-ish, and anatomically weak**. Root-cause read:
the finetune converges toward a **low-fidelity target** — if the training data is soft/mixed
and the captions are thin (tag-only), more epochs just sharpen convergence onto mush
("overcook at epoch5" was convergence onto mediocre data, not a true optimizer problem).

v5 attacks the **inputs**, not the training math:

1. **Raise the data-quality ceiling** — train at 1024, drop only *technical* defects
   (sub-1024, out-of-focus, duplicates). Keep *aesthetic* variety (tagged, not dropped).
2. **Rich, enum-locked captions** — a closed style vocabulary becomes a reliable inference
   control surface; the Qwen3-0.6B LLM text-encoder is starved by tag-only captions and
   thrives on natural language + consistent tokens.
3. **Train from base DiT at 1024** — a clean read, no inherited softness from the v3/v4 chain.

Key distinction driving the whole design: **aesthetic-bad ≠ blurry.** Two independent axes.
Drop for *broken* (out-of-focus, sub-res, dup, refused). Keep for *ugly* (tag it `low quality`).

## Free pre-check (before any GPU spend)

Generate from the current best checkpoint (v3-epoch4) **at 1024**, matched to the v5 training
resolution. v3 notes already suspected an inference-side resolution mismatch; if matching res
alone sharpens output, part of the "blur" is free to fix. Costs nothing. Do this first.
(Download v3-epoch4 off Vast before destroying the instance — it's the only copy.)

## Pipeline (converged)

All curation gating is **cheap + local** (no CLIP, no heavy VLM). The single paid/API step is
Gemini, run **only on survivors** after gating + spot-review.

### Stage 1 — ingest (local)
- phash near-dup dedup, keep highest-res in each group. **Raise `phash_hamming_threshold` 6 → 8**
  to kill social-media reposts harder (more diverse small set).
- Record `width`, `height`, `blur_var` (cv2 Laplacian variance) per image.
- Drop: corrupt (always), `min(w,h) < 1024` (resolution floor = training res, so we never pay to
  caption an image we'll later drop).

### Stage 2 — CLIP aesthetic scoring — **DELETED**
Aesthetic no longer gates (all buckets kept) — it only tags, and Gemini emits that tag directly.
Removing CLIP eliminates a model load, the aesthetic-weights download, and the two-scorer-scale
problem. The aesthetic weights config block is removed from `pipeline.yaml`.

### Stage 3 — caption (local WD14 + local safety + Gemini API)
Per survivor:
- **WD14 tagger** (local, ONNX, uncensored, sub-second): content tags **and** the underage hard-block.
  `block_tags` (loli/shota/child/...) → **hard drop** (adults-only legal boundary). WD14 stays
  precisely because the comma-tag format keeps the exact-match block working — Gemini NL would not.
- **Safety tag** (local): `safe`/`explicit` from the NSFW classifier (Falconsai ViT). Stays local —
  Gemini is unreliable/refuses on explicit, so safety classification must not depend on it.
- **Gemini** (`gemini-2.5-flash-lite`, structured output, `safety_settings = BLOCK_NONE`): one call
  returns the enum-locked style vocab + watermark flag + NL description (schema below).
  - On block/empty response → **fallback**: caption = tags only (no NL). Graceful; "the most
    explicit it can caption."
  - Resumable (cache completed images by path/hash — thousands of calls, reruns must skip done).
  - Rate-limit + retry with backoff.

### Stage 4 — build dataset (local)
- Gate survivors: **`min_blur_var` sharpness drop** (threshold tuned from the recorded `blur_var`
  distribution at build time — see Open Items), **keep all aesthetic buckets**.
- **Spot-review**: contact-sheet manual cull of remaining junk before the Gemini spend.
- Copy to flat `data/dataset/`, write `.txt` sidecars, emit diffusion-pipe `dataset.toml`
  (`resolutions=[1024]`, AR-buckets, `frame_buckets=[1]`).

### Stage 5 — emit `anima.toml` (local) + train (Vast)
diffusion-pipe **full finetune from base**:
- `init_from` empty (fresh `anima-base-v1.0.safetensors`), `project_name = anima_realism_ft_v5`.
- `resolutions=[1024]`, `lr=8e-6` constant, freeze Qwen3 adapter (`llm_adapter_lr=0`).
- `caption_dropout_percent=0.10` (CFG / generalization).
- `epochs ~20`, `save_every_n_epochs=1` → preview each epoch in ComfyUI → **stop at best**
  (from-base needs more epochs than warm-start; expect photoreal to emerge mid-run then plateau).
- VRAM ~50GB at 1024 (768 was ~33GB; ~1.8× area) → A100-80GB has headroom. Confirm at launch.

## Caption format

```
realistic photo, {quality_level}, {capture_style}, {lighting}, {condition}, {safety}, {wd14_tags}[, watermark], {description}
```

Example:
```
realistic photo, low quality, amateur snapshot, direct on-camera flash, grainy / high ISO, safe,
woman, kitchen, indoor, mug, a woman leaning on a kitchen counter holding a mug
```

- `realistic photo,` — fixed anchor leading every caption; the inference handle that pulls output
  off the anime prior.
- Style tokens come from a **closed vocabulary** → consistent → reliable triggers.
  `description` is free NL → subject/scene content.
- `watermark` appended only when `has_watermark` → **negative-promptable** at inference (no drop/crop).

## Gemini `response_schema` (enum-locked)

| Field | Type | Values | Required |
|---|---|---|---|
| `quality_level` | enum | `masterpiece, best quality` · `high quality` · `low quality` | yes (replaces CLIP) |
| `capture_style` | enum | `amateur snapshot` · `casual phone photo` · `semi-professional` · `professional photograph` · `studio portrait` | yes |
| `lighting` | array<enum> (0–2) | `direct on-camera flash` · `natural daylight` · `golden hour` · `overcast flat light` · `indoor artificial light` · `low light` · `soft window light` · `studio lighting` | no |
| `condition` | array<enum> (0–2) | `sharp focus` · `soft focus` · `grainy / high ISO` · `motion blur` · `compressed / low-res` · `overexposed` · `underexposed` | no |
| `has_watermark` | bool | — | yes |
| `description` | string | free NL, factual, no quality commentary | yes |

Design principles: **small vocab > big vocab** (each value needs ≥~100 training images to become a
strong trigger; ~2500 imgs across 5 axes supports 5–8 values/axis). **Style ≠ content** — enums carry
steerable style, `description` carries scene; separation keeps triggers clean. Arrays allow 0 picks so
Gemini won't force a wrong tag. The `amateur snapshot` end of the ladder is the primary lever against
AI/anime gloss.

`safety_settings`: `BLOCK_NONE` on the 4 adjustable categories
(`HARASSMENT`, `HATE_SPEECH`, `SEXUALLY_EXPLICIT`, `DANGEROUS`). Child-safety is **not adjustable** —
Google always blocks it at the core, a backstop on the adults-only boundary. WD14 `block_tags` remains
the primary underage filter.

## Config / code changes

**`config/pipeline.yaml`:**
- `ingest.phash_hamming_threshold: 8` (was 6).
- Remove the `quality:` (CLIP) block.
- `caption`: drop CLIP/joycaption keys; add `gemini` sub-block (`model: gemini-2.5-flash-lite`,
  the four enum vocab lists, `safety_block_none: true`, `max_output_tokens`, retry/cache settings);
  `captioner` becomes the merged WD14+Gemini path.
- `dataset`: `buckets_to_keep` → all buckets (or remove the bucket filter); `min_resolution: 1024`;
  `resolutions: [1024]`; add `min_blur_var:` (tuned at build).
- `finetune`: `project_name: anima_realism_ft_v5`; `init_from:` empty; `epochs: 20`;
  `caption_dropout_percent: 0.10`; res-related bumps.

**Code:**
- New `src/gemini_caption.py` — structured Gemini call (schema above), `BLOCK_NONE`, resumable cache,
  rate-limit/retry, empty/blocked → tags-only fallback. Reads `GEMINI_API_KEY` from `.env` (gitignored).
- `src/03_caption.py` — remove CLIP dependency; integrate WD14 (tags + underage block) + local safety
  tag + Gemini; assemble the new caption; append `watermark` when flagged.
- `src/04_build_dataset.py` — add `min_blur_var` gate to `curate()`; keep all buckets; 1024 dataset.toml.
- Delete stage-2 CLIP scoring path.
- `.env` for `GEMINI_API_KEY` (already gitignored).

## Testing

- Caption assembly: enum→string ordering; `watermark` appended iff flagged; fallback path
  (Gemini empty → tags-only) produces a valid caption.
- Schema validation: enum values reject out-of-vocab; arrays honor 0–2 bound.
- `curate()` gates: resolution floor, `min_blur_var`, dedup keep-highest-res; all-buckets retained.
- **Underage block regression test**: confirm the block still fires on WD14 comma-tags (it must,
  since WD14 stays — this was the failure mode if NL had replaced tags).
- Gemini wrapper: cache hit skips a completed image; retry on rate-limit.

## Risks / open items

- **`min_blur_var` threshold** — tune from the actual `blur_var` distribution of the ≥1024 set at
  build time (scan ~3004 imgs, pick threshold at the soft tail). Not a fixed guess.
- **Gemini quality calibration** — subjective 3-bucket call; sanity-check the first ~50 captions,
  adjust the prompt rubric if buckets skew.
- **Explicit refusal rate** — unknown until run; fallback to WD14 tags is accepted (user decision).
- **Account/ToS risk** — `BLOCK_NONE` on explicit real-person images "may be subject to review"
  (Google). Use a **throwaway/project-specific API key**, not a primary account.
- **1024 data sufficiency** — ~3004 imgs clear 1024 pre-gate; after dedup + sharpness, expect
  ~2000–2700 (no aesthetic drop). Plenty for a finetune; verify post-gate count at build.

## Hard boundaries (unchanged)

Legal adult content only — real adults, consensual. No minors, no non-consensual. WD14 `block_tags`
hard-drop + Gemini core child-safety block (non-disableable) together enforce this. Safety-tag, never
filter, for legal NSFW.
