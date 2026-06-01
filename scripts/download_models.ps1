# Downloads the 3 Anima training assets into models/ (idempotent: skips existing files).
# Reads repo + relative paths from config/pipeline.yaml's `models:` block (kept inline here for a no-dep script).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$repo = "circlestone-labs/Anima"
$prefix = "https://huggingface.co/$repo/resolve/main"
$files = @(
    "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "split_files/vae/qwen_image_vae.safetensors"
)

foreach ($f in $files) {
    $name = Split-Path $f -Leaf
    $dest = Join-Path $modelsDir $name
    if (Test-Path $dest) {
        Write-Host "exists, skip: $name"
        continue
    }
    $url = "$prefix/$f"
    Write-Host "downloading: $name"
    # -L follows redirects to the HF CDN; resumes via -C - if interrupted.
    curl.exe -L -C - -o $dest $url
}
Write-Host "Models ready in $modelsDir"
