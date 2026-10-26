# Runs the full pipeline for this project sequentially: build data index ->
# train -> evaluate on held-out test. Stops immediately if any step fails
# (non-zero exit code), so a broken build_index or a crashed training run
# can't silently be followed by an evaluate.py against stale/missing
# checkpoints.
#
# Usage (from anywhere):
#   powershell -File "E:\Multimodal\NamanAyush\03_models_19class\unet_lovasz_ce\run_pipeline.ps1"

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& "E:\Multimodal\NamanAyush\\.venv\Scripts\Activate.ps1"

Write-Host "==================== [1/3] build_index.py ====================" -ForegroundColor Cyan
python scripts/build_index.py
if ($LASTEXITCODE -ne 0) { Write-Host "build_index.py FAILED (exit $LASTEXITCODE) -- stopping." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==================== [2/3] train.py ====================" -ForegroundColor Cyan
python src/train.py --config configs/config_256.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "train.py FAILED (exit $LASTEXITCODE) -- stopping." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==================== [3/3] evaluate.py ====================" -ForegroundColor Cyan
python src/evaluate.py --config configs/config_256.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "evaluate.py FAILED (exit $LASTEXITCODE) -- stopping." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==================== Pipeline complete ====================" -ForegroundColor Green
Write-Host "Results: $PSScriptRoot\results_256\  (training) and $PSScriptRoot\results_256\test_eval\ (held-out test)"
