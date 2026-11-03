# Build a Windows folder bundle: dist\Traffic Tracker\Traffic Tracker.exe
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"

function Ensure-Venv {
    if (Test-Path $venvPython) { return }
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    $py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (-not (Test-Path $py311)) {
        $py311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
    }
    if (-not $py311 -or -not (Test-Path $py311)) {
        throw "Python 3.11 not found. Run .\run.ps1 once or install Python 3.11."
    }
    & $py311 -m venv (Join-Path $Root ".venv")
}

Ensure-Venv

Write-Host "Installing build tools..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r (Join-Path $Root "requirements-dev.txt")

Write-Host "Building executable (this can take several minutes)..." -ForegroundColor Cyan
& $venvPython -m PyInstaller --noconfirm --clean (Join-Path $Root "TrafficTracker.spec")

$exe = Join-Path $Root "dist\Traffic Tracker\Traffic Tracker.exe"
if (Test-Path $exe) {
    Write-Host ""
    Write-Host "Build complete:" -ForegroundColor Green
    Write-Host "  $exe"
    Write-Host ""
    Write-Host "Ship the entire folder: dist\Traffic Tracker\"
    Write-Host "Optional: copy yolo11n.pt into the project root before building to bundle model weights."
} else {
    throw "Build finished but executable was not found at $exe"
}
