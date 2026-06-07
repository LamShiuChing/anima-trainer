#!/usr/bin/env bash
# v10 launch on Vast = 1536 full-finetune, WARM-START FROM BASE DiT, lr 6e-6, 50 epochs
# (save-every-5, pick best on the concept-retention + photoreal eval sets).
# Requires: base DiT + VAE + Qwen3 dir (scripts/vast_setup.sh), v10 dataset (scripts/vast_fetch_v10.sh),
# and the two v10 tomls (this repo). Log -> /workspace/train_v10.log.
set -euo pipefail
BASE=/workspace/anima
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root = parent of scripts/ (no matter where it's cloned)

mkdir -p "$BASE/outputs"
cp "$REPO/outputs/anima_realism_ft_v10_dataset_config.toml" "$BASE/outputs/"
cp "$REPO/outputs/anima_realism_ft_v10_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima-base-v1.0.safetensors" || { echo "MISSING base DiT -> run scripts/vast_setup.sh"; exit 1; }
test -f "$BASE/models/qwen_image_vae.safetensors"  || { echo "MISSING VAE -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> run scripts/vast_fetch_v10.sh first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v10_train_config.toml" \
  > /workspace/train_v10.log 2>&1 &
echo "STARTED v10 pid $!  --  watch: tail -f /workspace/train_v10.log"
