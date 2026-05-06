# setup-hipdnn-windows.ps1
# PowerShell script to automate hipDNN build setup on Windows
# Run this script from an Administrator PowerShell prompt

param(
    [string]$InstallRoot = "D:\develop",
    [string]$VsBuildToolsPath = "",
    [string]$ClangPath = "",
    [Alias("GpuTarget")]
    [string]$Asic = "gfx1151",
    [switch]$SkipPrerequisites = $false,
    [switch]$SkipWindowsConfig = $false,
    [switch]$SkipToolchainDownload = $false,
    [switch]$SkipEnvironmentVariables = $false
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Resolve default install paths from InstallRoot unless explicit paths are provided.
if (-not $ClangPath) {
    $ClangPath = Join-Path $InstallRoot "dist\clang"
}
if (-not $VsBuildToolsPath) {
    $VsBuildToolsPath = Join-Path $InstallRoot "dist\vs-buildtools"
}

# Version configuration
$CLANG_VERSION = "20.1.8"
$CMAKE_VERSION = "3.31.0"

# Colors for output
function Write-Status { param($Message) Write-Host "[$([datetime]::Now.ToString('HH:mm:ss'))] $Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[!] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[X] $Message" -ForegroundColor Red }

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Error "This script must be run as Administrator!"
    Write-Host "Please run PowerShell as Administrator and try again."
    exit 1
}

Write-Host "`n===================================================" -ForegroundColor Magenta
Write-Host "  hipDNN Windows Build Setup Script" -ForegroundColor Magenta
Write-Host "===================================================" -ForegroundColor Magenta
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Install Root: $InstallRoot"
Write-Host "  VS Build Tools Path: $VsBuildToolsPath"
Write-Host "  Clang Path: $ClangPath"
Write-Host "  ASIC: $Asic"
Write-Host "===================================================`n" -ForegroundColor Magenta

# Section 1: Install Prerequisites
if (-not $SkipPrerequisites) {
    Write-Host "`n=== Section 1: Installing Prerequisites ===" -ForegroundColor Yellow

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error "winget is required but not found."
        Write-Host "Install 'App Installer' from the Microsoft Store or update Windows."
        exit 1
    }
    Write-Success "winget found"

    $wingetArgs = @("--accept-package-agreements", "--accept-source-agreements")

    # Install Visual Studio Build Tools with Windows SDK
    Write-Status "Installing Visual Studio 2022 Build Tools + Windows 11 SDK..."
    $vsOverride = "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 "
    $vsOverride += "--add Microsoft.VisualStudio.Component.VC.CMake.Project "
    $vsOverride += "--add Microsoft.VisualStudio.Component.VC.ATL "
    $vsOverride += "--add Microsoft.VisualStudio.Component.Windows11SDK.22621 "
    $vsOverride += "--installPath `"$VsBuildToolsPath`" --quiet --wait"
    winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --override "$vsOverride" @wingetArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Visual Studio Build Tools installed/updated"
    } else {
        Write-Warning "VS Build Tools install exited with code $LASTEXITCODE (may already be installed)"
    }

    # Install Git
    Write-Status "Installing Git..."
    winget install --id Git.Git -e --source winget --custom "/o:PathOption=CmdTools" @wingetArgs
    Write-Success "Git installed/updated"

    # Install CMake
    Write-Status "Installing CMake $CMAKE_VERSION..."
    winget install --id Kitware.CMake -v $CMAKE_VERSION @wingetArgs
    Write-Success "CMake installed/updated"

    # Install Ninja
    Write-Status "Installing Ninja..."
    winget install --id ninja-build.ninja @wingetArgs
    Write-Success "Ninja installed/updated"

    # Install Python
    Write-Status "Installing Python..."
    winget install --id Python.Python.3.12 @wingetArgs
    Write-Success "Python installed/updated"

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Section 2: Configure Windows Settings
if (-not $SkipWindowsConfig) {
    Write-Host "`n=== Section 2: Configuring Windows Settings ===" -ForegroundColor Yellow

    # Enable Long Paths
    Write-Status "Enabling Windows Long Path Support..."
    try {
        $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
        Set-ItemProperty -Path $regPath -Name "LongPathsEnabled" -Value 1 -Type DWord -Force
        Write-Success "Long path support enabled"
    } catch {
        Write-Warning "Could not enable long paths: $_"
    }

    # Enable Developer Mode for Symlinks
    Write-Status "Enabling Developer Mode for symlinks..."
    try {
        $devModePath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock"
        if (-not (Test-Path $devModePath)) {
            New-Item -Path $devModePath -Force | Out-Null
        }
        Set-ItemProperty -Path $devModePath -Name "AllowDevelopmentWithoutDevLicense" -Value 1 -Type DWord -Force
        Write-Success "Developer mode enabled"
    } catch {
        Write-Warning "Could not enable developer mode automatically"
        Write-Host "Please manually enable Developer Mode:"
        Write-Host "  Windows 11: Settings → System → For Developers → Developer Mode → On"
        Write-Host "  Windows 10: Settings → Update & Security → For Developers → Developer Mode → On"
    }

    # Configure Git
    Write-Status "Configuring Git for symlinks and long paths..."
    git config --global core.symlinks true
    git config --global core.longpaths true
    Write-Success "Git configured for symlinks and long paths"
}

# Section 3: Download and Install Toolchains
if (-not $SkipToolchainDownload) {
    Write-Host "`n=== Section 3: Installing Toolchains ===" -ForegroundColor Yellow

    # Download and install Clang
    Write-Status "Setting up Clang toolchain..."
    if (-not (Test-Path "$ClangPath\bin\clang.exe")) {
        Write-Status "Downloading Clang $CLANG_VERSION..."

        # Create directory
        New-Item -ItemType Directory -Path $ClangPath -Force | Out-Null

        $clangUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$CLANG_VERSION/clang+llvm-$CLANG_VERSION-x86_64-pc-windows-msvc.tar.xz"
        $clangArchive = "$env:TEMP\clang.tar.xz"

        try {
            Invoke-WebRequest -Uri $clangUrl -OutFile $clangArchive -UseBasicParsing
            Write-Success "Clang downloaded"

            Write-Status "Extracting Clang (this may take a few minutes)..."
            # Using tar (comes with Windows 10+)
            tar -xf $clangArchive -C $env:TEMP

            # Find extracted folder and move contents
            $extractedFolder = Get-ChildItem -Path $env:TEMP -Filter "clang+llvm*" -Directory | Select-Object -First 1
            if ($extractedFolder) {
                Move-Item -Path "$($extractedFolder.FullName)\*" -Destination $ClangPath -Force
                Remove-Item -Path $extractedFolder.FullName -Force -Recurse
            }

            Remove-Item -Path $clangArchive -Force
            Write-Success "Clang installed to $ClangPath"
        } catch {
            Write-Error "Failed to download/install Clang: $_"
            Write-Host "Please manually download from: $clangUrl"
            Write-Host "Extract to: $ClangPath"
            exit 1
        }
    } else {
        Write-Success "Clang already installed at $ClangPath"
    }

    Write-Status "Skipping TheRock nightly tarball download."
    Write-Host "TheRock nightly tarball download has been retired."
    Write-Host "Use the wheel-based setup to install the latest ROCm nightly SDK."
}

# Section 4: Set System Environment Variables
if (-not $SkipEnvironmentVariables) {
    Write-Host "`n=== Section 4: Setting System Environment Variables ===" -ForegroundColor Yellow

    Write-Status "Setting system environment variables..."

    # Set HIP_PLATFORM
    try {
        Write-Status "Setting HIP_PLATFORM..."
        [System.Environment]::SetEnvironmentVariable("HIP_PLATFORM", "amd", "Machine")
        Write-Success "HIP_PLATFORM set to 'amd'"
    } catch {
        Write-Warning "Could not set HIP_PLATFORM: $_"
    }

    # Set CMAKE_GENERATOR
    try {
        Write-Status "Setting CMAKE_GENERATOR..."
        [System.Environment]::SetEnvironmentVariable("CMAKE_GENERATOR", "Ninja", "Machine")
        Write-Success "CMAKE_GENERATOR set to 'Ninja'"
    } catch {
        Write-Warning "Could not set CMAKE_GENERATOR: $_"
    }

    Write-Success "System environment variables configured"
    Write-Warning "You may need to restart your terminal or system for changes to take effect"
}

Write-Host "`n===================================================" -ForegroundColor Magenta
Write-Host "  Setup Script Complete" -ForegroundColor Magenta
Write-Host "===================================================" -ForegroundColor Magenta
Write-Host "`nTo install the latest ROCm nightly SDK, run:" -ForegroundColor Yellow
Write-Host "  .\scripts\windows\wheel_build_setup.ps1"
Write-Host "Use -SHA <commit-sha> to install a specific staging build."

# Prompt for system restart if settings were changed
if (-not $SkipWindowsConfig) {
    Write-Warning "`nSystem restart may be required for all settings to take effect."
    $restart = Read-Host 'Restart now? (Y/N)'
    if ($restart -eq 'Y') {
        Restart-Computer -Force
    }
}
