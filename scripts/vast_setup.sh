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

# 3) Anima DiT + VAE (single files)
wget -c -O "$MODELS/anima-base-v1.0.safetensors" "$HF/diffusion_models/anima-base-v1.0.safetensors"
wget -c -O "$MODELS/qwen_image_vae.safetensors"   "$HF/vae/qwen_image_vae.safetensors"

# 3b) Qwen3-0.6B-Base text encoder as a HF DIR. The Anima loader calls AutoTokenizer.from_pretrained(qwen_path),
#     which needs tokenizer+config files — a single .safetensors fails. Official base repo = same weights.
pip install -q "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir "$MODELS/Qwen3-0.6B-Base"

echo "Setup done. Models in $MODELS ; diffusion-pipe in $DP_DIR"
