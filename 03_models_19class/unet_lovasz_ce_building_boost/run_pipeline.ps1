# Runs train -> evaluate sequentially for this project. data_index/ was
# copied verbatim from the sibling project (split unchanged), so no
# build_index.py step is needed here unlike the first CE+Lovasz project.
# Stops immediately if a step fails.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File "E:\Multimodal\NamanAyush\03_models_19class\unet_lovasz_ce_building_boost\run_pipeline.ps1"

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe"

Write-Host "==================== [1/2] train.py ====================" -ForegroundColor Cyan
& $py src\train.py --config configs\config_256.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "train.py FAILED (exit $LASTEXITCODE) -- stopping." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==================== [2/2] evaluate.py ====================" -ForegroundColor Cyan
& $py src\evaluate.py --config configs\config_256.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "evaluate.py FAILED (exit $LASTEXITCODE) -- stopping." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==================== Pipeline complete ====================" -ForegroundColor Green
Write-Host "Results: $PSScriptRoot\results_256\  (training) and $PSScriptRoot\results_256\test_eval\ (held-out test)"
