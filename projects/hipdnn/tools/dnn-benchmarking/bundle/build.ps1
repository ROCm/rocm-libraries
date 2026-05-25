# Build dnn-benchmark.pyz using shiv
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PkgDir = Split-Path -Parent $ScriptDir
$Out = Join-Path $ScriptDir 'dnn-benchmark.pyz'

pip install shiv --quiet
shiv -c dnn-benchmark -o $Out $PkgDir
Write-Host "Built: $Out"
