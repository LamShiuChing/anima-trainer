#!/usr/bin/env bash
# v5 launch helper for Vast. Run AFTER vast_setup.sh + dataset upload.
# Copies the generated tomls into place and starts training under nohup.
set -euo pipefail
BASE=/workspace/anima

mkdir -p "$BASE/outputs"
cp "$BASE/repo/outputs/anima_realism_ft_v5_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v5_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima-base-v1.0.safetensors" || { echo "MISSING DiT model -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/models/Qwen3-0.6B-Base"             || { echo "MISSING Qwen3 dir -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset -> upload it first"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v5_train_config.toml" \
  > /workspace/train.log 2>&1 &
echo "STARTED pid $!  --  watch: tail -f /workspace/train.log"
