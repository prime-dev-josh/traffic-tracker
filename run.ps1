# Launch Traffic Tracker from source (creates .venv and installs deps on first run).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvPip = Join-Path $Root ".venv\Scripts\pip.exe"

function Find-Python311 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        (Get-Command py -ErrorAction SilentlyContinue | ForEach-Object { "py" })
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if ($candidate -eq "py") {
            try {
                $version = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $version) { return $version.Trim() }
            } catch { }
        } elseif (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Python 3.11 not found. Install from https://www.python.org/downloads/ or: winget install Python.Python.3.11"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    $py311 = Find-Python311
    & $py311 -m venv (Join-Path $Root ".venv")
}

if (-not (Test-Path $venvPip)) {
    throw "Virtual environment is incomplete. Delete .venv and run this script again."
}

Write-Host "Installing/updating dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r (Join-Path $Root "requirements.txt")
Write-Host "Installing CUDA PyTorch (RTX / NVIDIA GPU)..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $Root "requirements-cuda.txt")

Write-Host "Starting Traffic Tracker..." -ForegroundColor Green
& $venvPython (Join-Path $Root "run.py")
