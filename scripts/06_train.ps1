# Stage 6: idempotently provision the trainer, then run headless LoRA training with sample previews.
# Prereq: run stages 1-5 (Python) and scripts/download_models.ps1 first.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$trainerDir = Join-Path $root "trainer"
$project = "anima_realism_v1"   # MUST match train.project_name in config/pipeline.yaml
$outputs = Join-Path $root "outputs"
$trainToml = Join-Path $outputs "${project}_training_config.toml"
$dataToml  = Join-Path $outputs "${project}_dataset_config.toml"

foreach ($t in @($trainToml, $dataToml)) {
    if (-not (Test-Path $t)) { throw "Missing $t  - run stages 4 & 5 first." }
}

# NOTE: $ErrorActionPreference="Stop" does NOT catch native-command (git/cmd/accelerate)
# failures - only $LASTEXITCODE reflects them. Each native call below is checked explicitly
# so a failed step never silently poisons idempotency or reports a false success.

# 1. Clone the trainer if absent. Remove a partial dir on failure so a re-run can retry.
if (-not (Test-Path $trainerDir)) {
    Write-Host "Cloning Anima-Standalone-Trainer..."
    git clone https://github.com/gazingstars123/Anima-Standalone-Trainer $trainerDir
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Recurse -Force $trainerDir -ErrorAction SilentlyContinue
        throw "git clone failed (exit $LASTEXITCODE). Partial directory removed; re-run to retry."
    }
}

# 2. Run setup_env.bat once. Write the .env_ready marker ONLY on success, so a failed setup
#    doesn't get skipped on the next run (which would train with a broken venv).
$marker = Join-Path $trainerDir ".env_ready"
if (-not (Test-Path $marker)) {
    Write-Host "Running setup_env.bat (first-time, ~10-15 min)..."
    Push-Location $trainerDir
    try {
        cmd /c setup_env.bat
        $setupExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($setupExit -ne 0) {
        throw "setup_env.bat failed (exit $setupExit). Fix the error, then re-run (no marker written)."
    }
    New-Item -ItemType File -Path $marker | Out-Null
}

# 3. Confirm the Anima LoRA network module exists.
if (-not (Test-Path (Join-Path $trainerDir "networks\lora_anima.py"))) {
    throw "networks\lora_anima.py not found in trainer checkout - aborting."
}

# 4. Launch training headless. Activate the trainer venv if present; its exact name isn't
#    known until setup_env.bat runs, so warn (don't silently fall back to system Python).
Push-Location $trainerDir
try {
    $venvActivate = Join-Path $trainerDir "venv\Scripts\activate.ps1"
    if (Test-Path $venvActivate) {
        . $venvActivate
    } else {
        Write-Warning "Trainer venv not found at $venvActivate - using PATH 'accelerate'. If setup_env.bat named the venv differently, update this path."
    }
    accelerate launch --num_cpu_threads_per_process 1 `
      "$trainerDir\anima_train_network.py" `
      --config_file "$trainToml" `
      --dataset_config "$dataToml"
    $trainExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($trainExit -ne 0) {
    throw "accelerate launch exited with code $trainExit. Check training logs."
}

Write-Host "Training complete. LoRA + sample previews -> $outputs"
