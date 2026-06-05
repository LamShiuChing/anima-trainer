#!/usr/bin/env bash
# v8 launch on Vast = 1536 full-finetune FIDELITY REFINER, WARM-START from the V7 keeper (epoch17),
# lr 4e-6, epochs 10 (save-every, pick best). Requires: warm-start ckpt
# (models/anima_v7_epoch17.safetensors), VAE + Qwen3 dir (vast_setup.sh), v8 dataset (data/dataset,
# uploaded via vast_fetch_v8.sh), and the two v8 tomls (this repo).
# Log -> /workspace/train_v8.log (download from Jupyter after run for loss-trend analysis).
set -euo pipefail
BASE=/workspace/anima

mkdir -p "$BASE/outputs"
cp "$BASE/repo/outputs/anima_realism_ft_v8_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v8_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima_v7_epoch17.safetensors" || { echo "MISSING warm-start ckpt -> upload V7_epoch17.safetensors to models/anima_v7_epoch17.safetensors"; exit 1; }
test -f "$BASE/models/qwen_image_vae.safetensors"  || { echo "MISSING VAE -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> run scripts/vast_fetch_v8.sh first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v8_train_config.toml" \
  > /workspace/train_v8.log 2>&1 &
echo "STARTED v8 pid $!  --  watch: tail -f /workspace/train_v8.log"
