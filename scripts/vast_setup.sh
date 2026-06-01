#!/usr/bin/env bash
# One-shot setup on a fresh Vast.ai Linux instance for Anima full finetune.
# Usage: bash scripts/vast_setup.sh
set -euo pipefail

BASE="${ANIMA_BASE:-/workspace/anima}"
DP_DIR="$BASE/diffusion-pipe"
MODELS="$BASE/models"
HF="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"

mkdir -p "$MODELS" "$BASE/data" "$BASE/outputs"

# 1) diffusion-pipe (Anima fork) + submodules
if [ ! -d "$DP_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/bluvoll/diffusion-pipe "$DP_DIR"
fi
cd "$DP_DIR"
git submodule update --init --recursive

# 2) Python deps (instance image already has CUDA torch). deepspeed + diffusion-pipe reqs.
pip install --upgrade pip
pip install deepspeed
pip install -r requirements.txt

# 3) Anima models (~5.6 GB)
wget -c -O "$MODELS/anima-base-v1.0.safetensors" "$HF/diffusion_models/anima-base-v1.0.safetensors"
wget -c -O "$MODELS/qwen_3_06b_base.safetensors"  "$HF/text_encoders/qwen_3_06b_base.safetensors"
wget -c -O "$MODELS/qwen_image_vae.safetensors"   "$HF/vae/qwen_image_vae.safetensors"

echo "Setup done. Models in $MODELS ; diffusion-pipe in $DP_DIR"
