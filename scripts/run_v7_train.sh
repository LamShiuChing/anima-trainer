#!/usr/bin/env bash
# v7 launch on Vast = 1536 full-finetune, WARM-START from the v6 keeper, lr 8e-6, epochs 40 (save-every, pick best).
# Requires: warm-start ckpt (models/anima_v6_keeper.safetensors), VAE + Qwen3 dir (vast_setup.sh),
#           v7 dataset (data/dataset, uploaded), and the two v7 tomls (this repo).
# Log -> /workspace/train_v7.log (download from Jupyter after run for loss-trend analysis).
set -euo pipefail
BASE=/workspace/anima

mkdir -p "$BASE/outputs"
cp "$BASE/repo/outputs/anima_realism_ft_v7_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v7_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima_v6_keeper.safetensors" || { echo "MISSING warm-start ckpt -> upload best v6 epoch to models/anima_v6_keeper.safetensors"; exit 1; }
test -f "$BASE/models/qwen_image_vae.safetensors"  || { echo "MISSING VAE -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> upload the v7 dataset first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v7_train_config.toml" \
  > /workspace/train_v7.log 2>&1 &
echo "STARTED v7 pid $!  --  watch: tail -f /workspace/train_v7.log"
