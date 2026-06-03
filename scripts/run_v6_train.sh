#!/usr/bin/env bash
# v6 launch on Vast = extend-v5 probe. WARM-START from v5 epoch20 @ 8e-6, +5 epochs.
# Requires: base models (vast_setup.sh), data/dataset (upload), models/anima_v5_epoch20.safetensors (upload).
# Log -> /workspace/train_v6.log (download from Jupyter after run for loss-trend analysis).
set -euo pipefail
BASE=/workspace/anima

cp "$BASE/repo/outputs/anima_realism_ft_v6_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v6_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima-base-v1.0.safetensors" || { echo "MISSING DiT model -> run scripts/vast_setup.sh"; exit 1; }
test -f "$BASE/models/anima_v5_epoch20.safetensors" || { echo "MISSING warm-start ckpt -> upload v5 epoch20 to models/anima_v5_epoch20.safetensors (see plan Task 6)"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset (v6 reuses v5's upload)"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
python -c "import bitsandbytes" >/dev/null 2>&1 || pip install -q bitsandbytes   # adamw8bit (40GB fit)
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v6_train_config.toml" \
  > /workspace/train_v6.log 2>&1 &
echo "STARTED v6 pid $!  --  watch: tail -f /workspace/train_v6.log"
