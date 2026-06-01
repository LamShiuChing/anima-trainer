#!/usr/bin/env bash
# Run the prep pipeline (stages 1-5) on Vast. Produces data/dataset + the two TOMLs.
# Assumes a pipeline venv with CUDA torch + requirements.txt installed, and data/raw populated.
# Usage: bash scripts/run_prep.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

python src/01_ingest_clean.py
python src/02_quality_score.py
python src/03_caption.py
python src/04_build_dataset.py
python src/05_make_train_config.py

echo "Prep done. Review outputs/*_train_config.toml and outputs/*_dataset_config.toml,"
echo "then copy data/dataset to \$ANIMA_BASE/data/dataset before launching training."
