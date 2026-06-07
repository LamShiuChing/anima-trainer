#!/usr/bin/env bash
# Fetch v10 inputs onto a fresh Vast instance: the dataset zip (bundles dataset/ + optional char/).
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
rm -rf "$BASE/data/_stage" "$BASE/data/dataset" "$BASE/data/char"
mkdir -p "$BASE/data/_stage"
# unzip returns exit 1 on harmless warnings (e.g. CJK filename local/central mismatch) -> tolerate it
# (set -e would otherwise abort the script before the files are moved). Only exit >=2 is a real error.
set +e; unzip -q -o "$BASE/data/dataset.zip" -d "$BASE/data/_stage"; rc=$?; set -e
[ "$rc" -le 1 ] || { echo "unzip failed (exit $rc)"; exit 1; }

# zip bundles top-level folders dataset/ (+ optional char/). Place each as its own diffusion-pipe dir.
[ -d "$BASE/data/_stage/dataset" ] || { echo "zip has no dataset/ folder -> wrong archive (rebuild with scripts/v10_zip.py)"; exit 1; }
mv "$BASE/data/_stage/dataset" "$BASE/data/dataset"
if [ -d "$BASE/data/_stage/char" ]; then
  mv "$BASE/data/_stage/char" "$BASE/data/char"
fi
rm -rf "$BASE/data/_stage"

IMG=$(find "$BASE/data/dataset" -type f ! -name '*.txt' | wc -l)
TXT=$(find "$BASE/data/dataset" -name '*.txt' | wc -l)
echo "dataset: $IMG images + $TXT captions"
[ "$IMG" -gt 0 ] && [ "$TXT" -gt 0 ] || { echo "MISSING images or captions in dataset/"; exit 1; }
if [ -d "$BASE/data/char" ]; then
  CIMG=$(find "$BASE/data/char" -type f ! -name '*.txt' | wc -l)
  CTXT=$(find "$BASE/data/char" -name '*.txt' | wc -l)
  echo "char:    $CIMG images + $CTXT captions"
  [ "$CIMG" -gt 0 ] && [ "$CTXT" -gt 0 ] || { echo "char/ present but empty/mismatched"; exit 1; }
fi
echo "fetch OK -> now: bash $BASE/repo/scripts/run_v10_train.sh"
