# Export yolo11n.onnx (and optionally OpenVINO) for Ultralytics AutoBackend.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No .venv found. Run .\run.ps1 first."
}
& $venvPython -m pip install -q onnx onnxslim onnxruntime
& $venvPython (Join-Path $Root "scripts\export_models.py") @args
