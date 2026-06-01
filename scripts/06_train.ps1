# Stage 6: idempotently provision the trainer, then run headless LoRA training with sample previews.
# Prereq: run stages 1-5 (Python) and scripts/download_models.ps1 first.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$trainerDir = Join-Path $root "trainer"
$project = "anima_realism_v1"
$outputs = Join-Path $root "outputs"
$trainToml = Join-Path $outputs "${project}_training_config.toml"
$dataToml  = Join-Path $outputs "${project}_dataset_config.toml"

foreach ($t in @($trainToml, $dataToml)) {
    if (-not (Test-Path $t)) { throw "Missing $t  - run stages 4 & 5 first." }
}

# 1. Clone the trainer if absent.
if (-not (Test-Path $trainerDir)) {
    Write-Host "Cloning Anima-Standalone-Trainer..."
    git clone https://github.com/gazingstars123/Anima-Standalone-Trainer $trainerDir
}

# 2. Run setup_env.bat once (marker file => idempotent; installs PyTorch 2.7 cu128 venv).
$marker = Join-Path $trainerDir ".env_ready"
if (-not (Test-Path $marker)) {
    Write-Host "Running setup_env.bat (first-time, ~10-15 min)..."
    Push-Location $trainerDir
    cmd /c setup_env.bat
    Pop-Location
    New-Item -ItemType File -Path $marker | Out-Null
}

# 3. Confirm the Anima LoRA network module exists.
if (-not (Test-Path (Join-Path $trainerDir "networks\lora_anima.py"))) {
    throw "networks\lora_anima.py not found in trainer checkout - aborting."
}

# 4. Launch training headless. Use the trainer venv's python via accelerate.
Push-Location $trainerDir
$venvActivate = Join-Path $trainerDir "venv\Scripts\activate.ps1"
if (Test-Path $venvActivate) { . $venvActivate }
accelerate launch --num_cpu_threads_per_process 1 `
  "$trainerDir\anima_train_network.py" `
  --config_file "$trainToml" `
  --dataset_config "$dataToml"
Pop-Location

Write-Host "Training launched. LoRA + sample previews -> $outputs"
