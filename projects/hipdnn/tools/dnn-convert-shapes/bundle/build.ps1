# Build dnn-convert-shapes.pyz using shiv in an isolated venv.
# Honors BUNDLE_OUT_DIR for out-of-source builds; defaults to script dir.
# Honors BUNDLE_VENV_DIR for a shared shiv venv; defaults to $OutDir\.venv.
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PkgDir = Split-Path -Parent $ScriptDir
$OutDir = if ($env:BUNDLE_OUT_DIR) { $env:BUNDLE_OUT_DIR } else { $ScriptDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Out = Join-Path $OutDir "dnn-convert-shapes.pyz"
$VenvDir = if ($env:BUNDLE_VENV_DIR) { $env:BUNDLE_VENV_DIR } else { Join-Path $OutDir ".venv" }
$ShivVersion = "1.0.8"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ShivExe = Join-Path $VenvDir "Scripts\shiv.exe"

if (-not (Test-Path $ShivExe)) {
    & $Python -m venv $VenvDir
    & (Join-Path $VenvDir "Scripts\pip.exe") install --quiet --disable-pip-version-check "shiv==$ShivVersion"
}

& $ShivExe -c dnn-convert-shapes -o $Out $PkgDir
Write-Host "Built: $Out"
