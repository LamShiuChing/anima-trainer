#!/usr/bin/env bash
# Fetch v10 inputs onto a fresh Vast instance: the dataset zip only.
# (No warm-start ckpt: v10 warm-starts the BASE DiT, which scripts/vast_setup.sh already downloads.)
# Drive ID is passed as an ARG (never committed — repo is public, Drive links are private).
# Usage: bash scripts/vast_fetch_v10.sh <DATASET_ZIP_GDRIVE_ID>
set -euo pipefail
BASE="${ANIMA_BASE:-/workspace/anima}"

[ $# -eq 1 ] || { echo "usage: $0 <DATASET_ZIP_ID>"; exit 1; }
DATASET_ID="$1"

mkdir -p "$BASE/data"
pip install -q gdown

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
echo "dataset: $CNT files ($IMG images + $TXTN captions)"
[ "$IMG" -gt 0 ] && [ "$TXTN" -gt 0 ] || { echo "MISSING images or captions"; exit 1; }
echo "fetch OK -> now: bash $BASE/repo/scripts/run_v10_train.sh"
