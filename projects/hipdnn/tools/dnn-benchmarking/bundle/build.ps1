# Build dnn-benchmark.pyz using shiv.
#
# Output:   <bundle-dir>\dnn-benchmark.pyz
# Override: $env:DNN_BUNDLE_OUT = 'C:\path\to\output.pyz'
#
# torch is intentionally NOT bundled — it ships from the ROCm/CUDA nightly
# index and the user installs it separately on the target host. The bundle
# carries dnn_benchmarking + numpy + psutil + pytest only.
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PkgDir = Split-Path -Parent $ScriptDir
$Out = if ($env:DNN_BUNDLE_OUT) { $env:DNN_BUNDLE_OUT } else { Join-Path $ScriptDir 'dnn-benchmark.pyz' }

$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "$Python not found on PATH"
    exit 1
}

& $Python -m pip install --quiet shiv
& $Python -m shiv `
    --console-script dnn-benchmark `
    --output-file $Out `
    --compressed `
    $PkgDir

Write-Host "Built: $Out"
