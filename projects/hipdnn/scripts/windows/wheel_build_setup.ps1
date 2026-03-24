# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Sets up a ROCm Windows development environment using Python wheels.

.DESCRIPTION
    Creates a Python virtual environment and installs ROCm SDK wheels for
    building hipDNN on Windows. Supports installing from either:
      - ROCm nightlies (default)
      - S3 staging with a specific SHA

    After installation, the script initializes the ROCm SDK and prints
    the CMake variables needed to build hipDNN.

.PARAMETER SHA
    Optional. A specific build SHA to install from S3 staging.
    When omitted, wheels are installed from the ROCm nightlies index.

.PARAMETER VenvPath
    Path where the Python virtual environment will be created.
    Default: D:\develop\latest_wheels

.PARAMETER ClangPath
    Path to the Clang toolchain bin directory.
    Default: D:\develop\dist\clang\bin

.PARAMETER GpuTarget
    GPU architecture target for the CMake example.
    Default: gfx1103

.EXAMPLE
    .\wheel_build_setup.ps1
    Installs from ROCm nightlies using default paths.

.EXAMPLE
    .\wheel_build_setup.ps1 -SHA "abc123"
    Installs specific wheels from S3 staging.

.EXAMPLE
    .\wheel_build_setup.ps1 -VenvPath "C:\my_venv" -ClangPath "C:\clang\bin" -GpuTarget "gfx1100"
    Installs from nightlies with custom paths and GPU target.
#>

param(
    [string]$SHA = "",
    [string]$VenvPath = "D:\develop\latest_wheels",
    [string]$ClangPath = "D:\develop\dist\clang\bin",
    [string]$GpuTarget = "gfx1103"
)

$ErrorActionPreference = "Stop"

# --- Display configuration ---

Write-Host ""
Write-Host "=== ROCm Wheel Build Setup ===" -ForegroundColor Cyan
if ($SHA) {
    Write-Host "  SHA:        $SHA"
} else {
    Write-Host "  SHA:        (not provided - using nightlies)"
}
Write-Host "  Venv Path:  $VenvPath"
Write-Host "  Clang Path: $ClangPath"
Write-Host "  GPU Target: $GpuTarget"
Write-Host ""

# --- Create or reuse virtual environment ---

$SkipInstall = $false
if (Test-Path $VenvPath) {
    Write-Host "Existing environment found at $VenvPath" -ForegroundColor Yellow
    $response = Read-Host "Pull new wheels? (Y/N, default: N)"
    if ($response -eq 'Y') {
        Write-Host "  Removing existing venv..."
        Remove-Item -Recurse -Force $VenvPath
    } else {
        Write-Host "  Using existing wheels." -ForegroundColor Green
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & "$VenvPath\Scripts\Activate.ps1"

        # Skip straight to path setup
        $SkipInstall = $true
    }
}

if (-not $SkipInstall) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath

    # --- Activate virtual environment ---

    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "$VenvPath\Scripts\Activate.ps1"

    # --- Install ROCm wheels ---

    Write-Host "Installing ROCm wheels..." -ForegroundColor Yellow

    if ($SHA) {
        $BaseUrl = "https://therock-dev-python.s3.amazonaws.com/v2-staging/gfx110X-all"

        Write-Host "  Source: S3 staging (SHA: $SHA)" -ForegroundColor Yellow
        pip install `
            "$BaseUrl/rocm-7.12.0.dev0%2B$SHA.tar.gz" `
            "$BaseUrl/rocm_sdk_core-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_libraries_gfx110x_all-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_devel-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl"
    } else {
        Write-Host "  Source: ROCm nightlies" -ForegroundColor Yellow
        pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]"
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install ROCm wheels." -ForegroundColor Red
        exit 1
    }

    # --- Initialize ROCm SDK ---

    Write-Host "Initializing ROCm SDK..." -ForegroundColor Yellow
    rocm-sdk init

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to initialize ROCm SDK." -ForegroundColor Red
        exit 1
    }
}

# --- Configure paths ---

$SitePackages = "$VenvPath\Lib\site-packages"
$RocmDevel = "$SitePackages\_rocm_sdk_devel"
$RocmBin = "$RocmDevel\bin"

Write-Host "Adding ROCm bin to PATH..." -ForegroundColor Yellow
$env:PATH = "$RocmBin;$env:PATH"

# Convert to forward slashes for CMake compatibility
$RocmDevelUnix = $RocmDevel -replace '\\', '/'
$ClangPathUnix = $ClangPath -replace '\\', '/'

# --- Print summary ---

Write-Host ""
Write-Host "=== Environment Ready ===" -ForegroundColor Green
Write-Host ""
Write-Host "ROCm SDK paths (use these in CMake):"
Write-Host "  CMAKE_HIP_COMPILER_ROCM_ROOT:  $RocmDevelUnix"
Write-Host ""
Write-Host "=== Sample CMake command for hipDNN ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "cmake -GNinja -DGPU_TARGETS=$GpuTarget -DCMAKE_PREFIX_PATH=$RocmDevelUnix -DCMAKE_PROGRAM_PATH=$ClangPathUnix .." -ForegroundColor White
Write-Host ""
