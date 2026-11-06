# Export yolo11n.onnx (and optionally OpenVINO) for Ultralytics AutoBackend.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
if ($Root -match 'OneDrive') {
    $VenvDir = Join-Path $env:LOCALAPPDATA "traffic-tracker\venv"
} else {
    $VenvDir = Join-Path $Root ".venv"
}
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No venv found at $VenvDir. Run .\run.ps1 first."
}
& $venvPython -m pip install -q onnx onnxslim onnxruntime
& $venvPython (Join-Path $Root "scripts\export_models.py") @args
