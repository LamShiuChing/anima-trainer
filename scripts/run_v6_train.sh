#!/usr/bin/env bash
# v6 launch on Vast. SAME dataset + models as v5 (already on the instance) — only a new train config (lr 1.5e-5).
# RUN ONLY AFTER v5 has finished or been stopped (frees the A100). Running both at once will OOM.
set -euo pipefail
BASE=/workspace/anima

cp "$BASE/repo/outputs/anima_realism_ft_v6_dataset_config.toml" "$BASE/outputs/"
cp "$BASE/repo/outputs/anima_realism_ft_v6_train_config.toml" "$BASE/outputs/"

test -f "$BASE/models/anima-base-v1.0.safetensors" || { echo "MISSING DiT model -> run scripts/vast_setup.sh"; exit 1; }
test -d "$BASE/data/dataset"                        || { echo "MISSING data/dataset (v6 reuses v5's upload)"; exit 1; }
echo "dataset files: $(ls "$BASE/data/dataset" | wc -l)"

cd "$BASE/diffusion-pipe"
which deepspeed >/dev/null 2>&1 || pip install -q deepspeed
nohup deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$BASE/outputs/anima_realism_ft_v6_train_config.toml" \
  > /workspace/train_v6.log 2>&1 &
echo "STARTED v6 pid $!  --  watch: tail -f /workspace/train_v6.log"
