#!/usr/bin/env pwsh
# rocBLAS Windows Build Script
# Wrapper around rmake.py that handles Windows-specific environment setup
#
# This script is designed to work even when:
# - IT restrictions prevent Windows SDK registry edits
# - Environment variables aren't properly set
# - Multiple SDK versions are installed
#
# Usage: .\install.ps1 [options]
#   -d, --dependencies     Install dependencies (one-time setup)
#   -c, --clients          Build clients (tests and benchmarks)
#   -i, --install          Install rocBLAS package after build
#   -g, --debug            Build in debug mode
#   --architecture <arch>  Target GPU architecture (e.g., "auto", "gfx90a")
#   --skip-aocl            Skip AOCL recommendation
#   -h, --help             Show this help message

param(
    [Alias("d")]
    [switch]$dependencies,
    
    [Alias("c")]
    [switch]$clients,
    
    [Alias("i")]
    [switch]$install,
    
    [Alias("g")]
    [switch]$debug_build,
    
    [string]$architecture,
    
    [switch]$skip_aocl,
    
    [Alias("h")]
    [switch]$help,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$remaining_args
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }

function Show-Help {
    Write-Host @"
rocBLAS Windows Build Script

USAGE:
    .\install.ps1 [OPTIONS]

OPTIONS:
    -d, --dependencies      Install build dependencies (vcpkg, gtest, etc.)
                           Only needed once or after dependency changes
    
    -c, --clients          Build client programs (tests and benchmarks)
                           Requires AOCL-BLAS for best results
    
    -i, --install          Install rocBLAS package after building
    
    -g, --debug            Build in debug mode (default: release)
    
    --architecture <arch>  Target specific GPU architecture
                           Examples: "auto", "gfx90a", "gfx942"
                           Default: "all"
    
    --skip-aocl            Skip AOCL-BLAS installation check/warning
    
    -h, --help             Show this help message

EXAMPLES:
    # First-time setup: Install dependencies
    .\install.ps1 -d

    # Build library only (fast iteration)
    .\install.ps1

    # Build with tests and benchmarks
    .\install.ps1 -c

    # Build for specific GPU (faster)
    .\install.ps1 -c --architecture auto

    # Debug build with clients
    .\install.ps1 -g -c

ENVIRONMENT:
    HIP_PATH            Path to AMD HIP SDK installation
                        Default: C:\Program Files\AMD\ROCm\<latest>
    
    VCPKG_PATH          Path to vcpkg installation
                        Default: C:\github\vcpkg
    
    OPENBLAS_DIR        Path to OpenBLAS (if not using AOCL)
                        Not recommended - use AOCL instead

DEPENDENCIES:
    Required:
    - AMD HIP SDK (ROCm for Windows)
    - Visual Studio Build Tools 2022
    - Windows SDK (10.0.19041.0 or newer)
    - Python 3.9+
    - CMake 3.24.4+

    Recommended for clients (-c):
    - AMD AOCL-BLAS 4.2 (ILP64 version)
      Download: https://www.amd.com/en/developer/aocl.html

MORE INFO:
    See README.md or https://rocm.docs.amd.com/projects/rocBLAS/
"@
    exit 0
}

if ($help) {
    Show-Help
}

Write-Success "`n========================================`nrocBLAS Windows Build Setup`n========================================`n"

# ============================================================================
# PHASE 1: Environment Detection and Setup
# ============================================================================

Write-Info "Phase 1: Detecting build environment..."

# 1.1 Find HIP SDK
if (-not $env:HIP_PATH) {
    $rocmBase = "C:\Program Files\AMD\ROCm"
    if (Test-Path $rocmBase) {
        # Find the latest version
        $versions = Get-ChildItem $rocmBase -Directory | Sort-Object Name -Descending
        if ($versions) {
            $env:HIP_PATH = $versions[0].FullName
            Write-Info "  [Auto-detected] HIP_PATH: $env:HIP_PATH"
        } else {
            Write-Error "  [ERROR] AMD HIP SDK not found in $rocmBase"
            Write-Warning "  Please install AMD HIP SDK from: https://rocm.docs.amd.com/en/latest"
            exit 1
        }
    } else {
        Write-Error "  [ERROR] AMD HIP SDK not found"
        Write-Warning "  Please install AMD HIP SDK and set HIP_PATH environment variable"
        exit 1
    }
} else {
    Write-Success "  [OK] HIP_PATH: $env:HIP_PATH"
}

# Verify hipcc exists
$hipcc = Join-Path $env:HIP_PATH "bin\hipcc.exe"
if (-not (Test-Path $hipcc)) {
    Write-Error "  [ERROR] hipcc.exe not found at: $hipcc"
    exit 1
}
Write-Success "  [OK] hipcc found"

# Add HIP to PATH for this session
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"

# 1.2 Find Windows SDK (even without registry)
Write-Info "`n  Detecting Windows SDK..."
$sdkBasePath = "C:\Program Files (x86)\Windows Kits\10"
$sdkFound = $false
$sdkVersion = $null

if (Test-Path "$sdkBasePath\Include") {
    $sdkVersions = Get-ChildItem "$sdkBasePath\Include" -Directory | 
                   Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
                   Sort-Object Name -Descending
    
    if ($sdkVersions) {
        $sdkVersion = $sdkVersions[0].Name
        $sdkFound = $true
        Write-Success "  [OK] Windows SDK $sdkVersion found"
        
        # Set up SDK environment variables (workaround for missing registry)
        $env:WindowsSdkDir = "$sdkBasePath\"
        $env:WindowsSDKVersion = "$sdkVersion\"
        $env:WindowsSdkBinPath = "$sdkBasePath\bin\"
        $env:WindowsSdkVerBinPath = "$sdkBasePath\bin\$sdkVersion\"
        $env:UniversalCRTSdkDir = "$sdkBasePath\"
        $env:UCRTVersion = $sdkVersion
        
        # Add SDK tools to PATH (critical for vcpkg)
        $env:PATH = "$sdkBasePath\bin\$sdkVersion\x64;$env:PATH"
        
        # Verify rc.exe is accessible
        try {
            $rcPath = (Get-Command rc.exe -ErrorAction Stop).Source
            Write-Success "  [OK] rc.exe accessible at: $rcPath"
        } catch {
            Write-Warning "  [WARNING] rc.exe not in PATH - may cause build issues"
            Write-Info "  This is likely due to IT restrictions. SDK path has been added to session PATH."
        }
    }
}

if (-not $sdkFound) {
    Write-Error "  [ERROR] Windows SDK not found at: $sdkBasePath"
    Write-Warning @"
  
  Please install Windows SDK through Visual Studio Installer:
  1. Run: `"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe`"
  2. Modify 'Build Tools for Visual Studio 2022'
  3. Select 'Desktop development with C++'
  4. In 'Individual components', check a Windows SDK version
  5. Install

  After installation, you may need to reboot for proper registry setup.
"@
    exit 1
}

# 1.3 Find Visual Studio Build Tools
$vsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest
    if ($vsPath) {
        Write-Success "  [OK] Visual Studio Build Tools found: $vsPath"
    } else {
        Write-Warning "  [WARNING] Visual Studio Build Tools not detected"
        Write-Info "  vcpkg may have issues. Install 'Desktop development with C++' workload."
    }
} else {
    Write-Warning "  [WARNING] Visual Studio Installer not found"
}

# 1.4 Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Success "  [OK] Python: $pythonVersion"
} catch {
    Write-Error "  [ERROR] Python not found in PATH"
    Write-Warning "  Please install Python 3.9+ and add to PATH"
    exit 1
}

# 1.5 Check CMake
try {
    $cmakeVersion = (cmake --version 2>&1)[0]
    Write-Success "  [OK] CMake: $cmakeVersion"
} catch {
    Write-Error "  [ERROR] CMake not found in PATH"
    Write-Warning "  Please install CMake 3.24.4 or later"
    exit 1
}

# 1.6 Setup VCPKG_PATH
if (-not $env:VCPKG_PATH) {
    $defaultVcpkg = "C:\github\vcpkg"
    if (Test-Path $defaultVcpkg) {
        $env:VCPKG_PATH = $defaultVcpkg
        Write-Info "  [Auto-detected] VCPKG_PATH: $env:VCPKG_PATH"
    } else {
        Write-Info "  [INFO] VCPKG_PATH not set - will be installed by rmake.py -d"
    }
} else {
    Write-Success "  [OK] VCPKG_PATH: $env:VCPKG_PATH"
}

# ============================================================================
# PHASE 2: Dependency Checks (AOCL)
# ============================================================================

if ($clients -and -not $skip_aocl) {
    Write-Info "`nPhase 2: Checking client dependencies..."
    
    $aoclPath = "C:\Program Files\AMD\AOCL-Windows"
    $aoclBlisLib = "$aoclPath\amd-blis\lib\ILP64\AOCL-LibBlis-Win-MT.lib"
    
    if (Test-Path $aoclBlisLib) {
        Write-Success "  [OK] AOCL-BLAS found: $aoclPath"
        Write-Info "  AOCL will be used for client testing (ILP64 support)"
    } else {
        Write-Warning @"
  [RECOMMENDED] AOCL-BLAS 4.2 not found

  For reliable client testing, install AOCL-BLAS (ILP64 version):
    Download: https://www.amd.com/en/developer/aocl.html
    Install to default location: C:\Program Files\AMD\AOCL-Windows

  Without AOCL, vcpkg's OpenBLAS will be used, which may cause stress test
  failures due to 32-bit integer overflow. You can exclude these tests with:
    --gtest_filter=-*stress*

  To suppress this warning, use: --skip-aocl
"@
        
        # Give user a chance to read the warning
        Start-Sleep -Seconds 2
    }
}

# ============================================================================
# PHASE 3: Build rmake.py arguments
# ============================================================================

Write-Info "`nPhase 3: Preparing build..."

$rmakeArgs = @()

# Map install.ps1 flags to rmake.py flags
if ($dependencies) { $rmakeArgs += "-d" }
if ($clients) { $rmakeArgs += "-c" }
if ($install) { $rmakeArgs += "-i" }
if ($debug_build) { $rmakeArgs += "-g" }
if ($architecture) { $rmakeArgs += "--architecture", $architecture }

# Add any additional arguments passed through
if ($remaining_args) {
    $rmakeArgs += $remaining_args
}

# Show what we're going to build
Write-Info "  Build configuration:"
Write-Info "    Mode: $(if ($debug_build) { 'Debug' } else { 'Release' })"
Write-Info "    Install dependencies: $(if ($dependencies) { 'Yes' } else { 'No' })"
Write-Info "    Build clients: $(if ($clients) { 'Yes' } else { 'No' })"
if ($architecture) {
    Write-Info "    Architecture: $architecture"
}

# ============================================================================
# PHASE 4: Run rmake.py
# ============================================================================

Write-Success "`n========================================`nStarting Build`n========================================`n"

$rmakeCmd = "python"
$rmakeCmdArgs = @("rmake.py") + $rmakeArgs

Write-Info "Executing: $rmakeCmd $($rmakeCmdArgs -join ' ')`n"

try {
    & $rmakeCmd $rmakeCmdArgs
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Success "`n========================================`nBuild completed successfully!`n========================================`n"
        
        if ($clients) {
            $testBinary = "build\release\clients\staging\rocblas-test.exe"
            if (Test-Path $testBinary) {
                Write-Info "Test binary: $testBinary"
                Write-Info "  Run tests: .\$testBinary --gtest_filter=*gemm*"
            }
        }
        
        exit 0
    } else {
        Write-Error "`n========================================`nBuild failed with exit code: $exitCode`n========================================`n"
        
        # Provide helpful troubleshooting
        Write-Info "Troubleshooting tips:"
        Write-Info "  1. If you see 'rc.exe not found' errors, try rebooting (Windows SDK registry issue)"
        Write-Info "  2. Check that all tools are installed: Python, CMake, HIP SDK, Build Tools, Windows SDK"
        Write-Info "  3. For dependency issues, try: .\install.ps1 -d --clean"
        Write-Info "  4. See build logs in: build\release\ or build\debug\"
        
        exit $exitCode
    }
} catch {
    Write-Error "`n[ERROR] Failed to execute rmake.py: $_"
    exit 1
}

