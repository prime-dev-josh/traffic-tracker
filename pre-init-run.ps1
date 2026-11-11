# Validate venv / PyTorch / Ultralytics before launching Traffic Tracker or processing video.
# Exits 0 on success, 1 on failure (use in CI or a desktop shortcut before run.ps1).
param(
    [switch]$Quick,
    [string]$Weights
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Get-TrafficTrackerVenvDir {
    if ($Root -match 'OneDrive') {
        return Join-Path $env:LOCALAPPDATA "traffic-tracker\venv"
    }
    return Join-Path $Root ".venv"
}

$VenvDir = Get-TrafficTrackerVenvDir
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
$checkScript = Join-Path $Root "scripts\pre_init_run.py"

if (-not (Test-Path $venvPython)) {
    Write-Host "No virtual environment at $VenvDir" -ForegroundColor Red
    Write-Host "Run .\run.ps1 once to create the venv and install dependencies." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $checkScript)) {
    throw "Missing $checkScript"
}

$pyArgs = @()
if ($Quick) { $pyArgs += "--quick" }
if ($Weights) { $pyArgs += @("--weights", $Weights) }

& $venvPython $checkScript @pyArgs
exit $LASTEXITCODE
