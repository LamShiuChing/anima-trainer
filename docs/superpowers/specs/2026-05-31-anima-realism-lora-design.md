# Anima Realism LoRA — Phase 1 Design Spec

**Date:** 2026-05-31
**Status:** Approved (pipeline), pending spec review
**Author:** pair session (user + Claude)

---

## 1. Goal

Train a **realism domain-shift LoRA** for the [Anima](https://huggingface.co/circlestone-labs/Anima)
diffusion model so it can produce realistic photographs, despite Anima being an
anime/illustration model. The base under Anima is NVIDIA Cosmos-Predict2-2B (photoreal-capable),
and community realism finetunes already exist on Civitai — so this is feasible, just a domain shift.

Phase 1 = prove the pipeline + data + captions actually move Anima toward photoreal, with a LoRA.
Phase 2 (separate future spec) = full finetune on the complete 3000-image set, on RunPod.

## 2. Source material & constraints

- **Dataset:** ~3000 photos scraped from social media (Reddit, X, Threads). Implications:
  JPEG artifacts, watermarks/text overlays, screenshots/memes mixed in, heavy near-duplicates
  (reposts), wild aspect ratios, stripped EXIF, mixed good/medium/bad quality.
- **NSFW:** some NSFW present. Handled by **safety-tagging, never filtering** (keep all data,
  control at inference). **Hard boundary: legal adult content only** — real adults, consensual.
  No minors, no non-consensual material.
- **Compute (Phase 1):** local **RTX 4080, 16GB VRAM, Windows 11**. No cloud, no Google Drive,
  so NSFW carries zero TOS/account-ban risk. Colab is *out* for Phase 1 (Drive scans content;
  flags can nuke the whole Google account). RunPod deferred to Phase 2.
- **Curation:** Phase-1 LoRA uses the **best ~500–800 images** (good + medium quality buckets;
  drop "bad" for now). Full 3000 reserved for Phase 2 finetune. A style/domain LoRA's sweet
  spot is ~500 images; dumping all 3000 into a first LoRA = slow epochs + noise.

## 3. Trainer backend

**Local backend: [gazingstars/Anima-Standalone-Trainer](https://github.com/gazingstars/Anima-Standalone-Trainer)**
(Windows `setup_env.bat`, sd-scripts based, ships `anima_train_network.py`). Driven **headless via
config TOMLs** — skip its Node/Web UI (painful, unnecessary for a scripted pipeline).

Config schema + exact invocation taken from the proven reference notebook
`Copy of ANIMA_Trainer_v5.ipynb` (repo: `citronlegacy/citron-colab-anima-lora-trainer`), which uses
the same `anima_train_network.py` + `networks.lora_anima`.

**Invocation:**
```
accelerate launch --num_cpu_threads_per_process 1 \
  <trainer>/anima_train_network.py \
  --config_file  <project>_training_config.toml \
  --dataset_config <project>_dataset_config.toml
```

## 4. Model assets (download once)

| Component | File | Size | URL (HF, `resolve/main`) |
|-----------|------|------|--------------------------|
| DiT | `anima-base-v1.0.safetensors` | 4.18 GB | `split_files/diffusion_models/anima-base-v1.0.safetensors` |
| Text encoder | `qwen_3_06b_base.safetensors` | 1.19 GB | `split_files/text_encoders/qwen_3_06b_base.safetensors` |
| VAE | `qwen_image_vae.safetensors` | 254 MB | `split_files/vae/qwen_image_vae.safetensors` |

Host prefix: `https://huggingface.co/circlestone-labs/Anima/resolve/main/`.
Alt DiT options exist (`anima-preview`, `anima-preview3-base`) — default to `anima-base-v1.0`.

## 5. Caption format

```
<quality tags>, <safety tag>, realistic photo, <natural-language description>
```
Example: `masterpiece, best quality, safe, realistic photo, a woman sitting on a park bench at golden hour, soft rim light, 35mm`

- **Quality tags** from the aesthetic-score bucket (Anima's native vocabulary: `masterpiece, best quality` / `high quality` / `normal quality` / `low quality`).
- **Safety tag** from an NSFW classifier (`safe` / `sensitive` / `explicit`) — matches Anima's
  training format; gives inference-time control instead of discarding data.
- **`realistic photo`** = domain trigger (prompt it at inference to invoke the LoRA's effect).
- **NL description** from JoyCaption.
- Captions are free text; `shuffle_caption=False` keeps the prefix in place.

**Captioner: JoyCaption** (NSFW-capable, runs on the 4080, native booru+NL style). Florence-2 /
Qwen2.5-VL rejected: censored, sanitize NSFW into useless captions. (Note: the Qwen3 *text encoder*
encodes whatever text the captioner writes — there is no benefit to "matching" captioner to TE.)

## 6. Pipeline — 6 local stages

Each stage is one script, reads the previous stage's output dir, idempotent/re-runnable.
All paths + thresholds live in `config/pipeline.yaml`.

| # | Script | Input | Output | Tools |
|---|--------|-------|--------|-------|
| 1 | `src/01_ingest_clean.py` | `data/raw/` | `data/clean/` + drop log | perceptual-hash dedup; min-size + Laplacian-blur + corrupt-file filters; OCR text-ratio flag for memes/screenshots |
| 2 | `src/02_quality_score.py` | `data/clean/` | `manifest.csv` (path, aesthetic score, bucket) | CLIP aesthetic predictor → good/medium/bad buckets |
| 3 | `src/03_caption.py` | `data/clean/` + manifest | per-image caption strings | JoyCaption NL + NSFW classifier safety tag + quality tag → assembled caption |
| 4 | `src/04_build_dataset.py` | captions + curated subset | `data/dataset/` (flat: `img.ext` + `img.txt`) + dataset TOML | copy curated images, write sidecars, emit dataset config |
| 5 | `src/05_make_train_config.py` | `pipeline.yaml` | `<project>_training_config.toml` | emit training TOML (schema §7) |
| 6 | `scripts/06_train.ps1` | TOMLs + models | trained LoRA `.safetensors` + sample previews | clone+setup Standalone-Trainer, download models, `accelerate launch`, fixed-seed sample gens |

**Stage 2 is the mixed-quality trick:** bad photos aren't discarded — they're tagged `low quality`
so the model learns the quality *axis* and good-quality gens stay clean. (Phase 1 still curates to
good+medium to keep the first run small; the scoring infra carries forward to Phase 2.)

## 7. Config schemas (from proven notebook)

**Training TOML** (`05_make_train_config.py` emits this):
```toml
pretrained_model_name_or_path = "<DiT path>"
qwen3 = "<TE path>"
vae   = "<VAE path>"
network_module = "networks.lora_anima"
network_dim   = 32          # 20–32 start; → 8 if OOM
network_alpha = 32
network_train_unet_only = true   # freezes Qwen3 TE/LLM adapter (= "don't train the adapter")
learning_rate = 1e-4        # proven default for dim~20; 2e-5 = conservative (model-card)
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1", "betas=[0.9, 0.99]"]
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 1
lr_warmup_steps = 100
max_train_epochs = 10
train_batch_size = 1
gradient_accumulation_steps = 1
max_grad_norm = 1.0
seed = 42
timestep_sampling = "sigmoid"
discrete_flow_shift = 1.0
qwen3_max_token_length = 512
t5_max_token_length = 512
mixed_precision = "bf16"
gradient_checkpointing = true
cache_latents = true                 # VRAM saver: precompute + offload VAE
cache_text_encoder_outputs = true    # VRAM saver: precompute + offload TE
vae_chunk_size = 64
vae_disable_cache = true
output_dir = "<out>"
output_name = "<project>"
save_model_as = "safetensors"
save_precision = "bf16"
save_every_n_epochs = 1
save_last_n_epochs = 4
shuffle_caption = false
caption_extension = ".txt"
noise_offset = 0.03
multires_noise_discount = 0.3
```

**Dataset TOML** (`04_build_dataset.py` emits this):
```toml
[general]
resolution = 768
enable_bucket = true
bucket_no_upscale = false
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096

[[datasets]]
resolution = 768
  [[datasets.subsets]]
  num_repeats = 5
  image_dir = "<data/dataset>"
  caption_extension = ".txt"
  caption_dropout_rate = 0.1
```

## 8. Hyperparameters & VRAM strategy (16GB)

- **Fit-in-16GB recipe:** `cache_latents` + `cache_text_encoder_outputs` (precompute VAE/TE
  outputs to disk, then unload — training holds only DiT + LoRA) + `gradient_checkpointing` +
  `bf16` + `AdamW8bit` + batch 1.
- **Start:** dim/alpha 32, res 768, repeats 5, epochs 10. **OOM fallback:** dim 8, res 512.
- **Trade-off:** cached latents pin resolution (no random-crop aug); re-cache if res changes.
- **Step budget:** local, so the Colab <1000-step cap does not apply. Watch loss/sample previews
  instead; ~1500–3000 steps typical for a ~500-image domain LoRA.

## 9. Repo layout (local Windows)

```
anima training/
  config/pipeline.yaml            # all paths + thresholds, single source
  src/01_ingest_clean.py
  src/02_quality_score.py
  src/03_caption.py
  src/04_build_dataset.py
  src/05_make_train_config.py
  scripts/06_train.ps1            # Windows launcher (setup + train + sample)
  scripts/download_models.ps1     # pull Anima DiT/TE/VAE from HF
  data/{raw,clean,dataset}/       # gitignored
  outputs/                        # LoRA checkpoints + sample previews (gitignored)
  requirements.txt                # pipeline deps (separate venv from trainer)
  README.md                       # run order
  Copy of ANIMA_Trainer_v5.ipynb  # reference notebook (kept; usable for SFW Colab later)
  docs/superpowers/specs/         # this spec
```

Two Python environments: **pipeline venv** (cleaning/captioning deps) vs **trainer venv**
(Standalone-Trainer's `setup_env.bat`, PyTorch 2.7 cu128). Kept separate to avoid dependency clashes.

## 10. Success criteria (Phase 1)

1. Pipeline runs end-to-end on Windows from `data/raw/` → trained `.safetensors`.
2. Training fits in 16GB without OOM at the documented settings.
3. Sample previews at the `realistic photo` trigger show a clear shift toward photoreal vs base Anima.
4. Quality/safety tags demonstrably steer output (e.g. `low quality` vs `best quality`, `safe` vs `explicit`).

## 11. Out of scope (→ Phase 2 spec)

- Full finetune on all 3000 images.
- RunPod turnkey deploy (network volume, headless setup script).
- Colab SFW-only runner (reference notebook already covers this if wanted).

## 12. Open items to resolve at build time

- JoyCaption Windows install path (model weights ~8–12GB; confirm CUDA wheel for bitsandbytes on Win).
- Aesthetic predictor choice (CLIP+MLP improved-aesthetic vs alternative) — pick lightest that runs on 4080.
- NSFW classifier choice for safety-tagging (e.g. a nudity/NSFW detector that handles photos).
- Confirm `networks.lora_anima` exists in the local Standalone-Trainer checkout after `setup_env.bat`.
```
