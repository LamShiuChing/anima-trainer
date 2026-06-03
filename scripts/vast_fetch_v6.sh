#!/usr/bin/env bash
# Fetch v6 inputs onto a fresh Vast instance: dataset zip + v5 warm-start checkpoint.
# IDs are passed as ARGS (never committed — repo is public, Drive links are private).
# Usage: bash scripts/vast_fetch_v6.sh <DATASET_ZIP_GDRIVE_ID> <CKPT_GDRIVE_ID>
set -euo pipefail
BASE="${ANIMA_BASE:-/workspace/anima}"
CKPT_BYTES=4182218360   # exact size of epoch20.safetensors; guards against gdown truncation

[ $# -eq 2 ] || { echo "usage: $0 <DATASET_ZIP_ID> <CKPT_ID>"; exit 1; }
DATASET_ID="$1"; CKPT_ID="$2"

mkdir -p "$BASE/data" "$BASE/models"
pip install -q gdown

# 1) warm-start checkpoint -> exact path run_v6_train.sh guards
gdown "$CKPT_ID" -O "$BASE/models/anima_v5_epoch20.safetensors"
SZ=$(stat -c%s "$BASE/models/anima_v5_epoch20.safetensors")
if [ "$SZ" != "$CKPT_BYTES" ]; then
  echo "CKPT SIZE WRONG: got $SZ, expect $CKPT_BYTES (gdown likely returned the virus-scan HTML page)."
  echo "Retry: gdown --fuzzy 'https://drive.google.com/uc?id=$CKPT_ID' -O $BASE/models/anima_v5_epoch20.safetensors"
  exit 1
fi
echo "ckpt size OK ($SZ)"

# 2) dataset zip -> flatten to data/dataset (find wherever the .txt sidecars land)
gdown "$DATASET_ID" -O "$BASE/data/dataset.zip"
rm -rf "$BASE/data/_stage" "$BASE/data/dataset"
mkdir -p "$BASE/data/_stage" "$BASE/data/dataset"
unzip -q -o "$BASE/data/dataset.zip" -d "$BASE/data/_stage"
TXT=$(find "$BASE/data/_stage" -name '*.txt' -print -quit)
[ -n "$TXT" ] || { echo "NO .txt sidecars in zip -> wrong archive?"; exit 1; }
SRC=$(dirname "$TXT")
mv "$SRC"/* "$BASE/data/dataset"/
rm -rf "$BASE/data/_stage"

CNT=$(ls "$BASE/data/dataset" | wc -l)
IMG=$(find "$BASE/data/dataset" -type f ! -name '*.txt' | wc -l)
TXTN=$(find "$BASE/data/dataset" -name '*.txt' | wc -l)
echo "dataset: $CNT files ($IMG images + $TXTN captions)  [v5 = 1942 + 1942 = 3884]"
[ "$IMG" -gt 0 ] && [ "$TXTN" -gt 0 ] || { echo "MISSING images or captions"; exit 1; }
echo "fetch OK -> now: bash $BASE/repo/scripts/run_v6_train.sh"
