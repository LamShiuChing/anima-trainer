#!/usr/bin/env bash
# v9 launch on Vast = 1536 full-finetune BACKGROUND FIX, WARM-START from the V8 keeper (epoch10),
# lr 6e-6, epochs 20 (save-every, pick best). Requires: warm-start ckpt
# (models/anima_v8_epoch10.safetensors), VAE + Qwen3 dir (vast_setup.sh), v9 dataset (data/dataset,
# uploaded via vast_fetch_v9.sh), and the two v9 tomls (this repo).
# Log -> /workspace/train_v9.log (download from Jupyter after run for loss-trend analysis).
set -euo pipefail
BASE=/workspace/anima
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root = parent of scripts/ (no matter where it's cloned)

mkdir -p "$BASE/outputs"
cp "$REPO/outputs/anima_realism_ft_v9_dataset_config.toml" "$BASE/outputs/"
cp "$REPO/outputs/anima_realism_ft_v9_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima_v8_epoch10.safetensors" || { echo "MISSING warm-start ckpt -> upload v8_epoch10.safetensors to models/anima_v8_epoch10.safetensors"; exit 1; }
test -f "$BASE/models/qwen_image_vae.safetensors"  || { echo "MISSING VAE -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> run scripts/vast_fetch_v9.sh first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v9_train_config.toml" \
  > /workspace/train_v9.log 2>&1 &
echo "STARTED v9 pid $!  --  watch: tail -f /workspace/train_v9.log"
