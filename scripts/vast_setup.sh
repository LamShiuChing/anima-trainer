#!/usr/bin/env bash
# One-shot setup on a fresh Vast.ai Linux instance for Anima full finetune.
# Usage: bash scripts/vast_setup.sh
set -euo pipefail

BASE="${ANIMA_BASE:-/workspace/anima}"
DP_DIR="$BASE/diffusion-pipe"
MODELS="$BASE/models"

# Guard: deepspeed's inference-v2 import crashes on Python >= 3.13 ("qkv_w not found"). Fail fast
# BEFORE downloading ~6 GB of models, so a wrong host image is caught in seconds, not minutes.
PYV=$(python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
case "$PYV" in
  3.10|3.11|3.12) echo "Python $PYV OK" ;;
  *) echo "ERROR: Python $PYV. deepspeed/torch need Python 3.10-3.12 (3.13/3.14 break the deepspeed import)."; \
     echo "Destroy this instance and rent one with a Python 3.10-3.12 image (standard 'PyTorch 2.x CUDA' template)."; exit 1 ;;
esac
HF="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"

# HF's Xet transport (cas-bridge.xethub.hf.co) is flaky on some Vast hosts -> force classic HTTP.
export HF_HUB_DISABLE_XET=1

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

# 3) Anima DiT + VAE (single files). -c resumes, --tries retries flaky connections.
wget -c --tries=10 --retry-connrefused --waitretry=5 -O "$MODELS/anima-base-v1.0.safetensors" "$HF/diffusion_models/anima-base-v1.0.safetensors"
wget -c --tries=10 --retry-connrefused --waitretry=5 -O "$MODELS/qwen_image_vae.safetensors"   "$HF/vae/qwen_image_vae.safetensors"

# 3b) Qwen3-0.6B-Base text encoder as a HF DIR. The Anima loader calls AutoTokenizer.from_pretrained(qwen_path),
#     which needs tokenizer+config files — a single .safetensors fails. Official base repo = same weights.
#     snapshot_download is resumable; retry up to 3x for flaky HF connectivity (xet already disabled above).
pip install -q huggingface_hub
for i in 1 2 3; do
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='$MODELS/Qwen3-0.6B-Base')" && break
  echo "Qwen snapshot attempt $i failed; retrying in 5s..."; sleep 5
done

# Verify the Qwen safetensors actually landed; fall back to a direct wget if snapshot still missed it.
if [ ! -s "$MODELS/Qwen3-0.6B-Base/model.safetensors" ]; then
  echo "snapshot missed model.safetensors -> direct wget fallback"
  wget -c --tries=10 --retry-connrefused --waitretry=5 -O "$MODELS/Qwen3-0.6B-Base/model.safetensors" \
    "https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors"
fi

echo "Setup done. Models in $MODELS ; diffusion-pipe in $DP_DIR"
