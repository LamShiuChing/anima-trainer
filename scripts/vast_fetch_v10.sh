#!/usr/bin/env bash
# Fetch v10 inputs onto a fresh Vast instance: the main dataset zip + the character set zip.
# (No warm-start ckpt: v10 warm-starts the BASE DiT, which scripts/vast_setup.sh already downloads.)
# Drive IDs are passed as ARGS (never committed — repo is public, Drive links are private).
# Usage: bash scripts/vast_fetch_v10.sh <DATASET_ZIP_ID> [CHAR_ZIP_ID]
#   - DATASET_ZIP_ID -> /workspace/anima/data/dataset
#   - CHAR_ZIP_ID    -> /workspace/anima/data/char   (optional; the oversampled trigger-word set)
# Each zip may be flat (img/txt at root) or contain one folder — both handled.
set -euo pipefail
BASE="${ANIMA_BASE:-/workspace/anima}"

[ $# -ge 1 ] || { echo "usage: $0 <DATASET_ZIP_ID> [CHAR_ZIP_ID]"; exit 1; }
mkdir -p "$BASE/data"
pip install -q gdown

unpack () {  # $1 = gdrive id, $2 = destination dir
  local id="$1" dest="$2" stage="$BASE/data/_stage" zip="$BASE/data/_dl.zip"
  rm -rf "$stage" "$dest" "$zip"; mkdir -p "$stage" "$dest"
  gdown "$id" -O "$zip"
  unzip -q -o "$zip" -d "$stage"
  local txt; txt=$(find "$stage" -name '*.txt' -print -quit)
  [ -n "$txt" ] || { echo "NO .txt sidecars in zip $id -> wrong archive?"; exit 1; }
  mv "$(dirname "$txt")"/* "$dest"/
  rm -rf "$stage" "$zip"
  local img t; img=$(find "$dest" -type f ! -name '*.txt' | wc -l); t=$(find "$dest" -name '*.txt' | wc -l)
  echo "$dest: $img images + $t captions"
  [ "$img" -gt 0 ] && [ "$t" -gt 0 ] || { echo "MISSING images or captions in $dest"; exit 1; }
}

unpack "$1" "$BASE/data/dataset"
if [ $# -ge 2 ]; then
  unpack "$2" "$BASE/data/char"
else
  echo "WARN: no CHAR_ZIP_ID given -> training WITHOUT the character set (toml expects data/char)."
fi
echo "fetch OK -> now: bash $BASE/repo/scripts/run_v10_train.sh"
