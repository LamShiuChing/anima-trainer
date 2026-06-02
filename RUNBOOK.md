# RUNBOOK — Anima Realism Full Finetune on Vast.ai

End-to-end first run. All prep + training happen on one rented Linux GPU.

## 0. Prerequisites (do locally, before renting)

- A folder of your photos (any sizes/formats). Zip it: `dataset_raw.zip`.
- A Vast.ai account with credit (~$10). A 512px run is ~2–3 h on one 48 GB GPU (~$1.5–2).
- Host `dataset_raw.zip` somewhere with a direct download link (Google Drive direct link,
  Dropbox `?dl=1`, S3, or `huggingface-cli upload` to a private dataset repo). This is the
  simplest way to get it onto an ephemeral instance. (Alternative: `scp` after the instance is up.)

## 1. Rent a GPU on Vast.ai

1. vast.ai → **Search**.
2. Filters: **GPU RAM ≥ 48 GB** (A6000 / L40S / A100-40 or 80). Disk **≥ 80 GB**. Reliability high.
3. **Template:** choose a PyTorch CUDA image, e.g. `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
   (or a Vast "PyTorch" recommended template). On-demand (not interruptible) for the first run.
4. Rent → open the instance's **Jupyter** or **SSH**.

## 2. Get this repo + your data onto the instance

```bash
export ANIMA_BASE=/workspace/anima
mkdir -p $ANIMA_BASE && cd $ANIMA_BASE
git clone <YOUR_REPO_URL> repo        # this project (the pipeline + scripts)
cd repo

# your photos from Google Drive — use gdown (plain wget fails on GDrive's >100MB virus-scan page).
# GDrive file must be shared "Anyone with the link". Grab the FILE_ID from the share URL:
#   https://drive.google.com/file/d/FILE_ID/view?usp=sharing
mkdir -p data/raw && cd data/raw
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" -O dataset_raw.zip
unzip -q dataset_raw.zip && rm dataset_raw.zip
# flatten if the zip made a subfolder:
find . -mindepth 2 -type f -exec mv -t . {} + 2>/dev/null || true
cd "$ANIMA_BASE/repo"
ls data/raw | head        # sanity: your images are here
```

## 3. Install diffusion-pipe + download Anima models

```bash
bash scripts/vast_setup.sh        # clones fork, pip installs, wgets the 3 model files (~5.6 GB)
```

## 4. Install the prep pipeline deps (separate from training)

```bash
# --system-site-packages reuses the instance image's CUDA torch (no 2.5 GB reinstall / CUDA mismatch).
python -m venv --system-site-packages $ANIMA_BASE/prepvenv
source $ANIMA_BASE/prepvenv/bin/activate
pip install -r requirements.txt   # torch NOT listed here — comes from the instance image
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 5. Run prep (dedup → score → tag → build dataset + configs)

```bash
bash scripts/run_prep.sh
```

This writes:
- `data/dataset/` — flat folder of kept images + `.txt` tag captions.
- `outputs/anima_realism_ft_v1_dataset_config.toml`
- `outputs/anima_realism_ft_v1_train_config.toml`

**Sanity-check a few captions** (should look like `masterpiece, best quality, safe`):
```bash
head -c 200 "$(ls data/dataset/*.txt | head -1)"; echo
```

Move the dataset where the config expects it, and copy the configs to the outputs dir:
```bash
mkdir -p $ANIMA_BASE/data $ANIMA_BASE/outputs
cp -r data/dataset $ANIMA_BASE/data/dataset
cp outputs/anima_realism_ft_v1_*.toml $ANIMA_BASE/outputs/
```

## 6. Launch the finetune

```bash
# Activate the instance's base Python env (has torch + deepspeed). A fresh tmux/SSH shell does NOT
# auto-activate it, so `deepspeed: command not found` means this step was skipped. Path is Vast-image
# specific — if /venv/main is absent, run `which python` in a normal shell to find the active env.
source /venv/main/bin/activate
which deepspeed || pip install deepspeed
cd $ANIMA_BASE/diffusion-pipe
deepspeed --num_gpus=1 train.py --deepspeed \
  --config $ANIMA_BASE/outputs/anima_realism_ft_v1_train_config.toml
```

Watch the loss. Checkpoints save every epoch to
`$ANIMA_BASE/outputs/anima_realism_ft_v1/` and every 30 min.

## 7. Preview a checkpoint

diffusion-pipe in-training image eval is OFF for run 1. To preview: load an epoch checkpoint
in **ComfyUI** with the Anima workflow (DiT + `qwen_3_06b_base` TE + `qwen_image_vae`) and prompt
`masterpiece, best quality, safe`. Compare against the base model to see the realism shift.

## 8. Retrieve your model before stopping the instance

```bash
# from your local machine:
scp -P <SSH_PORT> root@<INSTANCE_IP>:$ANIMA_BASE/outputs/anima_realism_ft_v1/'*.safetensors' .
# or push to HuggingFace from the instance:
#   huggingface-cli login && huggingface-cli upload <you>/anima-realism outputs/anima_realism_ft_v1
```

**Then DESTROY the instance** (Vast bills while it exists, even stopped, for storage).

## Troubleshooting

- **CUDA OOM:** confirm `activation_checkpointing = true` in the toml first. Then in
  `outputs/..._train_config.toml` set `[model] qwen_nf4 = true`, or switch `[optimizer] type`
  to `CAME`. Last resort: drop `resolutions = [512]` to `[448]` in the dataset toml.
- **"qwen_path" load error:** confirm `models/qwen_3_06b_base.safetensors` downloaded fully
  (re-run `scripts/vast_setup.sh`; `wget -c` resumes). If the loader insists on an HF dir,
  `huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir models/Qwen3-0.6B` and set
  `qwen_path` to that folder.
- **NSFW tags all "safe":** the classifier's `id2label` strings may differ from the substrings in
  `config/pipeline.yaml` (`caption.nsfw_label_map`). Print them once:
  `python -c "from transformers import AutoModelForImageClassification as M; print(M.from_pretrained('MichalMlodawski/nsfw-image-detection-large').config.id2label)"`
  and adjust the map substrings.
- **Instance interrupted mid-train:** re-rent, re-run setup, and resume with diffusion-pipe's
  `--resume_from_checkpoint` pointing at the last saved global step dir.

## v5 run (1024, from base, Gemini captions)

0. FREE pre-check: generate from v3-epoch4 at 1024 (matched res) before any GPU spend.
1. `.env`: copy `.env.example` -> `.env`, set GEMINI_API_KEY (throwaway/project key).
2. Stage 1 ingest (drop_small=true, min_size=1024, drop_blurry=false): records blur_var, drops <1024 + dups.
3. Tune sharpness: inspect the blur_var column distribution of kept rows; pick a threshold at the soft
   tail. Set ingest.blur_var_threshold + ingest.drop_blurry=true, re-run stage 1.
4. Spot-review: delete obviously-bad survivors from data/clean (stage 3 skips missing files).
5. Stage 3 caption: WD14 + safety + Gemini (resumable via data/gemini_cache.json). Sanity-check the
   first ~50 captions; adjust the Gemini prompt rubric if quality buckets skew.
6. Stage 4 build (min_resolution=1024, optional min_blur_var backstop) + Stage 5 anima.toml
   (init_from="" => base DiT, epochs=20). Confirm post-gate image count is sufficient.
7. Upload data/dataset + tomls to Vast; train; preview each epoch in ComfyUI; stop at best.
