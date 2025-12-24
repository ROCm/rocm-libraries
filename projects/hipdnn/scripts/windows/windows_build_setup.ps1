# setup-hipdnn-windows.ps1
# PowerShell script to automate hipDNN build setup on Windows
# Run this script from an Administrator PowerShell prompt

param(
    [string]$ClangPath = "C:\dist\clang",
    [string]$TheRockPath = "C:\dist\therock",
    [string]$ProjectPath = "C:\projects\hipdnn",
    [string]$GpuTarget = "gfx1103",
    [switch]$SkipPrerequisites = $false,
    [switch]$SkipWindowsConfig = $false,
    [switch]$SkipToolchainDownload = $false,
    [switch]$SkipEnvironmentVariables = $false,
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Version configuration
$CLANG_VERSION = "20.1.8"
$CMAKE_VERSION = "3.31.0"
$THEROCK_VERSION = "7.10.0"  # Base version, will look for latest nightly

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
Write-Host "  Clang Path: $ClangPath"
Write-Host "  TheRock Path: $TheRockPath"
Write-Host "  Project Path: $ProjectPath"
Write-Host "  GPU Target: $GpuTarget"
Write-Host "===================================================`n" -ForegroundColor Magenta

# Section 1: Install Prerequisites
if (-not $SkipPrerequisites) {
    Write-Host "`n=== Section 1: Installing Prerequisites ===" -ForegroundColor Yellow

    # Install Chocolatey if not present
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Status "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed"
    } else {
        Write-Success "Chocolatey already installed"
    }

    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    # Install Visual Studio Build Tools
    Write-Status "Installing Visual Studio 2022 Build Tools..."
    $vsParams = "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 "
    $vsParams += "--add Microsoft.VisualStudio.Component.VC.CMake.Project "
    $vsParams += "--add Microsoft.VisualStudio.Component.VC.ATL "
    $vsParams += "--add Microsoft.VisualStudio.Component.Windows11SDK.22621"

    choco install visualstudio2022buildtools -y --params "`"$vsParams`"" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 3010) {
        Write-Success "Visual Studio Build Tools installed/updated"
    }

    # Install Git
    Write-Status "Installing Git..."
    choco install git.install -y --params "'/GitAndUnixToolsOnPath'" 2>&1 | Out-Null
    Write-Success "Git installed/updated"

    # Install CMake
    Write-Status "Installing CMake..."
    choco install cmake --version=$CMAKE_VERSION -y 2>&1 | Out-Null
    Write-Success "CMake installed/updated"

    # Install Ninja
    Write-Status "Installing Ninja..."
    choco install ninja -y 2>&1 | Out-Null
    Write-Success "Ninja installed/updated"

    # Install Python
    Write-Status "Installing Python..."
    choco install python -y 2>&1 | Out-Null
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

    # Verify amdgpu-arch
    if (Test-Path "$ClangPath\bin\amdgpu-arch.exe") {
        Write-Status "Detecting GPU architecture..."
        $detectedGpu = & "$ClangPath\bin\amdgpu-arch.exe" 2>$null
        if ($detectedGpu) {
            Write-Success "Detected GPU: $detectedGpu"
            if ($detectedGpu -ne $GpuTarget) {
                Write-Warning "Detected GPU ($detectedGpu) differs from specified target ($GpuTarget)"
                $response = Read-Host 'Use detected GPU? (Y/N)'
                if ($response -eq 'Y') {
                    $GpuTarget = $detectedGpu
                }
            }
        }
    }

    # Download and install TheRock
    Write-Status "Setting up TheRock ROCm SDK..."
    if (-not (Test-Path "$TheRockPath\bin\hipconfig.exe")) {
        Write-Status "Downloading TheRock for $GpuTarget..."

        # Determine GFX family
        $gfxFamily = switch -Regex ($GpuTarget) {
            "gfx110[0-9]" { "gfx110X-all" }
            "gfx103[0-9]" { "gfx103X-all" }
            "gfx90[0-9]"  { "gfx90X-all" }
            default {
                Write-Warning "Unknown GFX family for $GpuTarget, using gfx110X-all"
                "gfx110X-all"
            }
        }

        Write-Status "Using GFX Family: $gfxFamily"

        # Create directory
        New-Item -ItemType Directory -Path $TheRockPath -Force | Out-Null

        # Find latest nightly build
        $baseUrl = "https://therock-nightly-tarball.s3.amazonaws.com"
        $indexUrl = "$baseUrl/index.html"

        try {
            Write-Status "Checking for latest TheRock build..."
            $indexContent = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing

            # Parse for Windows builds matching our GFX family
            $pattern = "therock-dist-windows-$gfxFamily-.*?\.tar\.gz"
            $matches = [regex]::Matches($indexContent.Content, $pattern)

            if ($matches.Count -gt 0) {
                # Get the most recent (last in list usually)
                $latestFile = $matches[$matches.Count - 1].Value
                $theRockUrl = "$baseUrl/$latestFile"

                Write-Status "Downloading $latestFile..."
                $theRockArchive = "$env:TEMP\therock.tar.gz"

                Invoke-WebRequest -Uri $theRockUrl -OutFile $theRockArchive -UseBasicParsing
                Write-Success "TheRock downloaded"

                Write-Status "Extracting TheRock (this may take several minutes)..."
                tar -xzf $theRockArchive -C $TheRockPath

                Remove-Item -Path $theRockArchive -Force
                Write-Success "TheRock installed to $TheRockPath"
            } else {
                throw "No Windows builds found for $gfxFamily"
            }
        } catch {
            Write-Error "Failed to download/install TheRock: $_"
            Write-Host "Please manually download from: $indexUrl"
            Write-Host "Look for: therock-dist-windows-$gfxFamily-*.tar.gz"
            Write-Host "Extract to: $TheRockPath"
            exit 1
        }
    } else {
        Write-Success "TheRock already installed at $TheRockPath"
    }
}

# Section 4: Set System Environment Variables
if (-not $SkipEnvironmentVariables) {
    Write-Host "`n=== Section 4: Setting System Environment Variables ===" -ForegroundColor Yellow

    Write-Status "Setting system environment variables..."

    # Add TheRock bin to system PATH if not already present
    try {
        $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        if (-not $currentPath.Contains("$TheRockPath\bin")) {
            Write-Status "Adding TheRock to system PATH..."
            $newPath = "$TheRockPath\bin;" + $currentPath
            [System.Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
            Write-Success "Added $TheRockPath\bin to system PATH"
        } else {
            Write-Success "TheRock already in system PATH"
        }
    } catch {
        Write-Warning "Could not modify system PATH: $_"
    }

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

# Prompt for system restart if settings were changed
if (-not $SkipWindowsConfig) {
    Write-Warning "`nSystem restart may be required for all settings to take effect."
    $restart = Read-Host 'Restart now? (Y/N)'
    if ($restart -eq 'Y') {
        Restart-Computer -Force
    }
}
