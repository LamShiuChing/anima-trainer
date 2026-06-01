# Anima Realism LoRA — Phase 1 Pipeline

Turns `data/raw/` photos into a realism domain-shift LoRA for the Anima diffusion model.
Local, Windows, RTX 4080 (16GB). Full design: `docs/superpowers/specs/2026-05-31-anima-realism-lora-design.md`.

## Setup (two separate venvs)

**Pipeline venv** (stages 1-5):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
The trainer venv is created automatically by stage 6 (`setup_env.bat`) — do not mix the two.

## Run order
```powershell
# Pipeline venv active:
python src/01_ingest_clean.py
python src/02_quality_score.py
python src/03_caption.py
python src/04_build_dataset.py
python src/05_make_train_config.py
# Models + train (uses curl + the trainer venv it provisions):
.\scripts\download_models.ps1
.\scripts\06_train.ps1
```
Every stage is idempotent and reads the previous stage's output. Config lives in `config/pipeline.yaml`.

## OOM fallback
If training OOMs at the documented settings, edit `config/pipeline.yaml` `train:` → `network_dim: 8`,
`network_alpha: 8`, `resolution: 512`, then re-run `python src/05_make_train_config.py` and `.\scripts\06_train.ps1`.
(Cached latents pin resolution — changing res forces a re-cache, which the trainer does automatically.)

## Tests
```powershell
python -m pytest -v          # pure-logic suite, no GPU needed
```
