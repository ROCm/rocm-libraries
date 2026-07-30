# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Sets up a ROCm Windows development environment using Python wheels.

.DESCRIPTION
    Creates a Python virtual environment and installs ROCm SDK wheels for
    building hipDNN on Windows. Supports installing from either:
      - ROCm nightlies (default)
      - S3 staging with a specific SHA and version

    After installation, the script initializes the ROCm SDK and prints
    the CMake variables needed to build hipDNN.

.PARAMETER SHA
    Optional. A specific build SHA to install from S3 staging.
    When omitted, wheels are installed from the ROCm nightlies index.
    Requires -StagingVersion to specify the version.

.PARAMETER StagingVersion
    Optional. ROCm version to install from S3 staging (e.g., "7.12.0.dev0").
    Only used when -SHA is also provided. Default: 7.12.0.dev0

.PARAMETER Version
    Optional. Pin a specific version from nightlies (e.g., "7.13.0a20260425", "7.14.0a20260624").
    Only used when -SHA is NOT provided. When omitted, installs the latest nightly.

.PARAMETER VenvBase
    Base directory where versioned venvs are stored. Each version is installed
    to a subfolder (e.g., VenvBase\7.14.0a20260624), and a symlink named
    "latest_wheels" points to the active version.
    Default: D:\develop\wheels

.PARAMETER LinkName
    Name of the symlink that points to the active version.
    Default: latest_wheels

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
    .\wheel_build_setup.ps1 -Version "7.13.0a20260425"
    Installs a specific older version from nightlies.

.EXAMPLE
    .\wheel_build_setup.ps1 -SHA "abc123"
    Installs specific wheels from S3 staging with default version (7.12.0.dev0).

.EXAMPLE
    .\wheel_build_setup.ps1 -SHA "abc123" -StagingVersion "7.11.0.dev0"
    Installs specific wheels from S3 staging with an older version.

.EXAMPLE
    .\wheel_build_setup.ps1 -VenvBase "C:\rocm_wheels" -ClangPath "C:\clang\bin" -GpuTarget "gfx1151"
    Installs from nightlies with custom paths and GPU target.
#>

param(
    [string]$SHA = "",
    [string]$StagingVersion = "7.12.0.dev0",
    [string]$Version = "",
    [string]$VenvBase = "D:\develop\rocm_wheels",
    [string]$LinkName = "latest_wheels",
    [string]$ClangPath = "D:\develop\dist\clang\bin",
    [string]$GpuTarget = "gfx1151",
    [switch]$NonInteractive,
    [switch]$Help
)

function Show-Usage {
    Write-Host @"
wheel_build_setup.ps1 - Set up a ROCm Windows dev environment using Python wheels

Creates a versioned Python virtual environment, installs ROCm SDK wheels, points
a stable 'latest_wheels' junction at the active version, and prints the CMake
variables needed to build hipDNN. Does not require Administrator privileges.

Prerequisites:
  - Python 3.x on PATH
  - A Clang toolchain already installed (see windows_build_setup.ps1 or download
    manually from https://github.com/llvm/llvm-project/releases)

Usage:
  .\wheel_build_setup.ps1 [options]

Options:
  -SHA <string>             Build SHA to install from S3 staging. When omitted,
                            wheels are installed from the ROCm nightlies index.
  -StagingVersion <string>  ROCm version for S3 staging (only used with -SHA).
                            Default: 7.12.0.dev0
  -Version <string>         Pin a specific nightly version (e.g. 7.13.0a20260425).
                            Only used when -SHA is NOT provided. Omit for latest.
  -VenvBase <string>        Base directory for versioned venvs; each version lives
                            in a subfolder and the link points at the active one.
                            Default: D:\develop\rocm_wheels
  -LinkName <string>        Name of the junction pointing at the active version.
                            Default: latest_wheels
  -ClangPath <string>       Path to the Clang toolchain bin directory.
                            Default: D:\develop\dist\clang\bin
  -GpuTarget <string>       GPU architecture target (gfx115x, gfx120x[-all],
                            gfx110x[-all], gfx103x[-all], gfx90x[-all]).
                            Default: gfx1151
  -NonInteractive           Do not prompt; assume defaults for any prompts.
  -Help                     Show this help and exit.

Examples:
  .\wheel_build_setup.ps1
      Install the latest ROCm nightly using default paths.

  .\wheel_build_setup.ps1 -Version "7.13.0a20260425"
      Install a specific older version from nightlies.

  .\wheel_build_setup.ps1 -SHA "abc123"
      Install specific wheels from S3 staging (default version 7.12.0.dev0).

  .\wheel_build_setup.ps1 -SHA "abc123" -StagingVersion "7.11.0.dev0"
      Install specific wheels from S3 staging with an older version.

  .\wheel_build_setup.ps1 -VenvBase "C:\rocm_wheels" -ClangPath "C:\clang\bin" -GpuTarget "gfx1151"
      Install from nightlies with custom paths and GPU target.

For the full built-in help, run: Get-Help .\wheel_build_setup.ps1 -Detailed
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

$ErrorActionPreference = "Stop"
$OriginalPath = $env:PATH
$OriginalVirtualEnv = $env:VIRTUAL_ENV
$OriginalVirtualEnvPrompt = $env:VIRTUAL_ENV_PROMPT
$script:VenvDeactivated = $false

function Disable-RocmVenv {
    if ($script:VenvDeactivated) { return }
    Write-Host "Deactivating Python virtual environment for this session..." -ForegroundColor Yellow
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    } else {
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
    $script:VenvDeactivated = $true
}

function Rename-VenvFolder {
    param(
        [string]$Path,
        [string]$NewName
    )

    # Freshly-installed wheels (hundreds of MB of DLLs) are often still held open
    # by Windows Defender / the search indexer, which blocks the rename with
    # "Access denied". Drop our own handles and retry with backoff.
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    $maxAttempts = 10
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            Rename-Item -Path $Path -NewName $NewName -ErrorAction Stop
            return $true
        } catch {
            if ($i -eq $maxAttempts) {
                Write-Warning "Rename failed after $maxAttempts attempts: $($_.Exception.Message)"
                return $false
            }
            Write-Host "  Rename attempt $i/$maxAttempts failed (folder likely locked by AV/indexer); retrying in 3s..." -ForegroundColor DarkGray
            Start-Sleep -Seconds 3
        }
    }
    return $false
}

function Resolve-RocmArtifactGroup {
    param([string]$Target)

    switch -Regex ($Target.ToLower()) {
        "^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])-all$" { return $Target }
        "^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])$" { return "$Target-all" }
        "^gfx115[0-9]$" { return $Target }
        default { return $Target }
    }
}

function Get-InstalledRocmVersion {
    param([string]$VenvPath)

    $SitePackages = "$VenvPath\Lib\site-packages"

    # Look for rocm_sdk_libraries-*.dist-info directory to extract version
    $DistInfo = Get-ChildItem -Path $SitePackages -Filter "rocm_sdk_libraries-*.dist-info" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($DistInfo) {
        if ($DistInfo.Name -match "^rocm_sdk_libraries-(.+)\.dist-info$") {
            return $Matches[1]
        }
    }

    # Fallback: try rocm-*.dist-info
    $DistInfo = Get-ChildItem -Path $SitePackages -Filter "rocm-*.dist-info" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($DistInfo) {
        if ($DistInfo.Name -match "^rocm-(.+)\.dist-info$") {
            return $Matches[1]
        }
    }

    return $null
}

function Get-InstalledDeviceTarget {
    param([string]$VenvPath)

    $SitePackages = "$VenvPath\Lib\site-packages"

    # The per-arch device wheel encodes the GPU target in its dist-info name,
    # e.g. rocm_sdk_device_gfx1103-10.0.0a20260729.dist-info -> gfx1103.
    $DistInfo = Get-ChildItem -Path $SitePackages -Filter "rocm_sdk_device_*.dist-info" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($DistInfo) {
        if ($DistInfo.Name -match "^rocm_sdk_device_(gfx[0-9a-z]+)-") {
            return $Matches[1]
        }
    }

    return $null
}

function Update-VersionLink {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )

    # Remove existing link/junction if present
    if (Test-Path $LinkPath) {
        $item = Get-Item $LinkPath -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            # It's a symlink/junction - remove it
            cmd /c rmdir "$LinkPath" 2>$null
        } else {
            Write-Error "Path $LinkPath exists but is not a symlink. Please remove it manually."
            exit 1
        }
    }

    # Create new junction (works without admin on Windows)
    cmd /c mklink /J "$LinkPath" "$TargetPath"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create junction at $LinkPath"
        exit 1
    }
}

function Get-SymlinkTarget {
    param([string]$LinkPath)

    if (-not (Test-Path $LinkPath)) {
        return $null
    }

    $item = Get-Item $LinkPath -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        return $item.Target
    }

    return $null
}

# Compute the link path (sits alongside VenvBase, e.g., D:\develop\latest_wheels)
$VenvBaseParent = Split-Path -Parent $VenvBase
$LinkPath = Join-Path $VenvBaseParent $LinkName

# When -GpuTarget was not explicitly passed, default to the arch of the currently
# installed wheel (recorded implicitly in the device wheel's dist-info name) so
# subsequent updates stay on the same GPU without re-specifying it.
if (-not $PSBoundParameters.ContainsKey('GpuTarget')) {
    $InstalledDeviceTarget = Get-InstalledDeviceTarget -VenvPath $LinkPath
    if ($InstalledDeviceTarget) {
        $GpuTarget = $InstalledDeviceTarget
        Write-Host "Using GPU target from current install: $GpuTarget" -ForegroundColor Cyan
    }
}

$RocmArtifactGroup = Resolve-RocmArtifactGroup -Target $GpuTarget
$LibrariesWheelTarget = $RocmArtifactGroup.ToLower().Replace('-', '_')
# Multi-arch nightlies use a single index and select the GPU via a bare
# `device-<arch>` extra (no family/-all suffix), e.g. gfx942-all -> device-gfx942.
$DeviceTarget = $GpuTarget.ToLower() -replace '-all$', ''
$VerifiedGpuTarget = $GpuTarget.ToLower() -match "^(gfx115[0-9]|gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])(-all)?)$"
if (-not $VerifiedGpuTarget) {
    Write-Warning "GPU target '$GpuTarget' is not in the verified list (gfx115x, gfx120x[-all], gfx110x[-all], gfx103x[-all], gfx90x[-all]). Wheel install may not work."
}

# --- Display configuration ---

Write-Host ""
Write-Host "=== ROCm Wheel Build Setup ===" -ForegroundColor Cyan
if ($SHA) {
    Write-Host "  SHA:        $SHA"
    Write-Host "  Version:    $StagingVersion (S3 staging)"
} else {
    if ($Version) {
        Write-Host "  Version:    $Version (pinned)"
    } else {
        Write-Host "  Version:    (latest nightly)"
    }
}
Write-Host "  Venv Base:  $VenvBase"
Write-Host "  Link Path:  $LinkPath"
Write-Host "  Clang Path: $ClangPath"
Write-Host "  GPU Target: $GpuTarget"
Write-Host "  Device (nightly): device-$DeviceTarget"
Write-Host "  Wheel Group: $RocmArtifactGroup"
Write-Host ""

# --- List available versions ---

if (Test-Path $VenvBase) {
    $AvailableVersions = Get-ChildItem -Path $VenvBase -Directory | Select-Object -ExpandProperty Name
    if ($AvailableVersions) {
        Write-Host "Available versions in $VenvBase`:" -ForegroundColor Cyan
        foreach ($v in $AvailableVersions) {
            Write-Host "  - $v"
        }
        Write-Host ""
    }
}

# Show current link target
$CurrentLinkTarget = Get-SymlinkTarget -LinkPath $LinkPath
if ($CurrentLinkTarget) {
    $CurrentVersionName = Split-Path -Leaf $CurrentLinkTarget
    Write-Host "Current active version: $CurrentVersionName" -ForegroundColor Green
    Write-Host ""
}

# --- Determine target version ---

# For S3 staging, the full version string includes the SHA
if ($SHA) {
    $TargetVersion = "$StagingVersion+$SHA"
} elseif ($Version) {
    $TargetVersion = $Version
} else {
    $TargetVersion = "latest"
}

# The folder name is qualified with the GPU arch so different arches at the same
# version don't collide (e.g. 10.0.0a20260729_gfx1103). "latest" stays a bare
# placeholder here; it is renamed to <version>_<arch> after install once the
# concrete version is known.
if ($TargetVersion -eq "latest") {
    $VersionedVenvName = "latest"
} else {
    $VersionedVenvName = "${TargetVersion}_${DeviceTarget}"
}
$VersionedVenvPath = Join-Path $VenvBase $VersionedVenvName

# --- Check if version already exists ---

$SkipInstall = $false
if ($TargetVersion -eq "latest") {
    # "latest" is a sentinel, not a version identity: an existing folder named
    # "latest" says nothing about whether the remote nightly moved on. Always
    # resolve fresh, and clear any stale placeholder so the venv is created clean.
    if (Test-Path $VersionedVenvPath) {
        Write-Host "Removing stale 'latest' placeholder to force a fresh resolve..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VersionedVenvPath
    }
    Write-Host "Resolving latest nightly. Will download and install." -ForegroundColor Yellow
} elseif (Test-Path $VersionedVenvPath) {
    Write-Host "Version $VersionedVenvName already exists at $VersionedVenvPath" -ForegroundColor Green

    # Check if it's already the active version
    if ($CurrentLinkTarget -and (Split-Path -Leaf $CurrentLinkTarget) -eq $VersionedVenvName) {
        Write-Host "This version is already active." -ForegroundColor Green
        if ($NonInteractive) {
            $response = 'N'
        } else {
            $response = Read-Host "Reinstall anyway? (Y/N, default: N)"
        }
        if ($response -ne 'Y') {
            Write-Host "  Using existing installation." -ForegroundColor Green
            $SkipInstall = $true
        } else {
            Write-Host "  Removing existing version for reinstall..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $VersionedVenvPath
        }
    } else {
        Write-Host "Switching to this version..." -ForegroundColor Yellow
        Update-VersionLink -LinkPath $LinkPath -TargetPath $VersionedVenvPath
        Write-Host "  Link updated: $LinkPath -> $VersionedVenvPath" -ForegroundColor Green
        $SkipInstall = $true
    }
} else {
    Write-Host "Version $VersionedVenvName not found locally. Will download and install." -ForegroundColor Yellow
}

# For using later (VenvPath now points to the versioned path)
$VenvPath = $VersionedVenvPath

if (-not $SkipInstall) {
    # Ensure base directory exists
    if (-not (Test-Path $VenvBase)) {
        Write-Host "Creating venv base directory: $VenvBase" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $VenvBase -Force | Out-Null
    }

    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath

    # --- Activate virtual environment ---

    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "$VenvPath\Scripts\Activate.ps1"

    # --- Install ROCm wheels ---

    Write-Host "Installing ROCm wheels..." -ForegroundColor Yellow

    if ($SHA) {
        $BaseUrl = "https://therock-dev-python.s3.amazonaws.com/v2-staging/$RocmArtifactGroup"

        Write-Host "  Source: S3 staging (SHA: $SHA, version: $StagingVersion, group: $RocmArtifactGroup)" -ForegroundColor Yellow
        pip install `
            "$BaseUrl/rocm-$StagingVersion%2B$SHA.tar.gz" `
            "$BaseUrl/rocm_sdk_core-$StagingVersion%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_libraries_$LibrariesWheelTarget-$StagingVersion%2B$SHA-py3-none-win_amd64.whl" `
            "$BaseUrl/rocm_sdk_devel-$StagingVersion%2B$SHA-py3-none-win_amd64.whl"
    } else {
        if ($Version) {
            Write-Host "  Source: ROCm multi-arch nightlies (device: $DeviceTarget, version: $Version)" -ForegroundColor Yellow
            pip install --no-cache-dir --index-url "https://rocm.nightlies.amd.com/whl-multi-arch/" "rocm==$Version[libraries,devel,device-$DeviceTarget]"
        } else {
            Write-Host "  Source: ROCm multi-arch nightlies (device: $DeviceTarget, latest)" -ForegroundColor Yellow
            pip install --no-cache-dir --index-url "https://rocm.nightlies.amd.com/whl-multi-arch/" "rocm[libraries,devel,device-$DeviceTarget]"
        }
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

    # --- Handle "latest" version: detect actual version and rename folder ---

    if ($TargetVersion -eq "latest") {
        $ActualVersion = Get-InstalledRocmVersion -VenvPath $VenvPath
        if ($ActualVersion) {
            # Qualify the folder name with the GPU arch so different arches at the
            # same version don't collide (e.g. 10.0.0a20260729_gfx1103).
            $ActualVersionName = "${ActualVersion}_${DeviceTarget}"
            $ActualVersionedPath = Join-Path $VenvBase $ActualVersionName
            # Deactivate first: an active venv holds handles into the folder,
            # which makes the rename below fail with "Access denied" on Windows.
            Disable-RocmVenv
            if (Test-Path $ActualVersionedPath) {
                Write-Host "Version $ActualVersionName already exists, removing duplicate..." -ForegroundColor Yellow
                Remove-Item -Recurse -Force $VenvPath
                $VenvPath = $ActualVersionedPath
            } else {
                Write-Host "Detected version: $ActualVersionName" -ForegroundColor Green
                if (Rename-VenvFolder -Path $VenvPath -NewName $ActualVersionName) {
                    $VenvPath = $ActualVersionedPath
                } else {
                    Write-Warning "Could not rename '$VenvPath' to '$ActualVersionName'; keeping folder name as 'latest' and linking to it."
                }
            }
        } else {
            Write-Warning "Could not detect installed version. Keeping folder name as 'latest'."
        }
    }

    # --- Update symlink to point to new version ---

    Write-Host "Updating version link..." -ForegroundColor Yellow
    Update-VersionLink -LinkPath $LinkPath -TargetPath $VenvPath
    Write-Host "  Link updated: $LinkPath -> $VenvPath" -ForegroundColor Green
}

# --- Configure paths (use link path for stable references) ---

$SitePackages = "$LinkPath\Lib\site-packages"
$RocmDevel = "$SitePackages\_rocm_sdk_devel"
$RocmBin = "$RocmDevel\bin"

Write-Host "Adding ROCm bin to PATH..." -ForegroundColor Yellow
$env:PATH = "$RocmBin;$env:PATH"
$env:ROCM_PATH = $RocmDevel

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

# --- Deactivate venv for this shell session (no-op if already done above) ---

Disable-RocmVenv

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

# Publish the wheel venv path for tools that install into it (e.g. dnn-benchmark's
# setup.ps1), persisting after deactivation. Use the link path for stability.
$env:ROCM_WHEEL_VENV = $LinkPath
