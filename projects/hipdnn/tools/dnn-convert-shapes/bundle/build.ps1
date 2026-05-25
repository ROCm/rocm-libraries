# Build dnn-convert-shapes.pyz using shiv
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PkgDir = Split-Path -Parent $ScriptDir
$Out = Join-Path $ScriptDir "dnn-convert-shapes.pyz"

pip install shiv --quiet
shiv -c dnn-convert-shapes -o $Out $PkgDir
Write-Host "Built: $Out"
