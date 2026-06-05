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
    Default: gfx1151

.EXAMPLE
    .\wheel_build_setup.ps1
    Installs from ROCm nightlies using default paths.

.EXAMPLE
    .\wheel_build_setup.ps1 -SHA "abc123"
    Installs specific wheels from S3 staging.

.EXAMPLE
    .\wheel_build_setup.ps1 -VenvPath "C:\my_venv" -ClangPath "C:\clang\bin" -GpuTarget "gfx1151"
    Installs from nightlies with custom paths and GPU target.
#>

param(
    [string]$SHA = "",
    [string]$VenvPath = "D:\develop\latest_wheels",
    [string]$ClangPath = "D:\develop\dist\clang\bin",
    [string]$GpuTarget = "gfx1151"
)

$ErrorActionPreference = "Stop"
$OriginalPath = $env:PATH
$OriginalVirtualEnv = $env:VIRTUAL_ENV
$OriginalVirtualEnvPrompt = $env:VIRTUAL_ENV_PROMPT

function Resolve-RocmArtifactGroup {
    param([string]$Target)

    switch -Regex ($Target.ToLower()) {
        "^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])-all$" { return $Target }
        "^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])$" { return "$Target-all" }
        "^gfx115[0-9]$" { return $Target }
        default { return $Target }
    }
}

$RocmArtifactGroup = Resolve-RocmArtifactGroup -Target $GpuTarget
$LibrariesWheelTarget = $RocmArtifactGroup.ToLower().Replace('-', '_')
$VerifiedGpuTarget = $GpuTarget.ToLower() -match "^(gfx115[0-9]|gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])(-all)?)$"
if (-not $VerifiedGpuTarget) {
    Write-Warning "GPU target '$GpuTarget' is not in the verified list (gfx115x, gfx120x[-all], gfx110x[-all], gfx103x[-all], gfx90x[-all]). Wheel install may not work."
}

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
Write-Host "  Wheel Group: $RocmArtifactGroup"
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
        $BaseUrl = "https://therock-dev-python.s3.amazonaws.com/v2-staging/$RocmArtifactGroup"

        Write-Host "  Source: S3 staging (SHA: $SHA, group: $RocmArtifactGroup)" -ForegroundColor Yellow
        pip install `
            "$BaseUrl/rocm-7.12.0.dev0%2B$SHA.tar.gz" `
            "$BaseUrl/rocm_sdk_core-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_libraries_$LibrariesWheelTarget-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_devel-7.12.0.dev0%2B$SHA-py3-none-win_amd64.whl"
    } else {
        Write-Host "  Source: ROCm nightlies (group: $RocmArtifactGroup)" -ForegroundColor Yellow
        pip install --index-url "https://rocm.nightlies.amd.com/v2/$RocmArtifactGroup/" "rocm[libraries,devel]"
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
Write-Host "=== Sample CMake command for rocm-libraries superbuild ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "# Run from the rocm-libraries repository root" -ForegroundColor DarkGray
Write-Host "cmake --preset miopen-provider -DROCM_PATH=$RocmDevelUnix -DCMAKE_PROGRAM_PATH=$ClangPathUnix" -ForegroundColor White
Write-Host "cmake --build build" -ForegroundColor White
Write-Host ""

# --- Deactivate venv for this shell session ---

Write-Host "Deactivating Python virtual environment for this session..." -ForegroundColor Yellow
if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
} else {
    Write-Warning "Could not find 'deactivate' function; restoring pre-activation environment variables."
    $env:PATH = $OriginalPath
    if ($null -eq $OriginalVirtualEnv) {
        Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
    } else {
        $env:VIRTUAL_ENV = $OriginalVirtualEnv
    }

    if ($null -eq $OriginalVirtualEnvPrompt) {
        Remove-Item Env:VIRTUAL_ENV_PROMPT -ErrorAction SilentlyContinue
    } else {
        $env:VIRTUAL_ENV_PROMPT = $OriginalVirtualEnvPrompt
    }
}

# Keep ROCm bin available in this terminal session after deactivation.
$CurrentPathParts = $env:PATH -split ';'
$HasRocmBinInCurrentPath = $false
foreach ($pathEntry in $CurrentPathParts) {
    if ($pathEntry.Trim() -eq $RocmBin) {
        $HasRocmBinInCurrentPath = $true
        break
    }
}
if (-not $HasRocmBinInCurrentPath) {
    $env:PATH = "$RocmBin;$env:PATH"
}
