#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows setup for the dnn-benchmark tool (PowerShell analogue of setup.sh).

.DESCRIPTION
    setup.sh targets Linux: it installs hipDNN + the MIOpen provider to /opt/rocm,
    pulls a ROCm build of PyTorch from the Linux nightly index, and uses
    LD_LIBRARY_PATH. This script does the Windows equivalents:

      1. Selects an existing Python environment (the active venv, or -PythonExe).
         On Windows this should be the ROCm-wheel env, which carries the runtime.
      2. Optionally does a FULL BUILD of hipDNN (+ the Python bindings) and the
         MIOpen provider from source via the MSVC + ROCm-wheel toolchain (-FullBuild).
      3. Installs the dnn-benchmark package in editable mode (and its PyPI deps).
      4. Makes the hipdnn_frontend bindings importable by wiring the compiled
         extension onto the env via a .pth file.
      5. Installs PyTorch (CPU) — the default PyPI wheel — which backs the PyTorch
         reference provider (--backend pytorch / --validate pytorch).

    The ROCm runtime DLLs are resolved at import time by hipdnn_frontend itself,
    either from the rocm_sdk wheel (if installed in the env) or from
    ROCM_PATH / HIP_PATH / ROCM_HOME.

.PARAMETER PythonExe
    Python interpreter to install into. Default: the active venv's python if one is
    active, otherwise 'python' on PATH. Recommended: your ROCm wheel env's python
    (it carries rocm_sdk + the _rocm_sdk_devel toolchain used for -FullBuild).

.PARAMETER FullBuild
    Build hipDNN (with the Python bindings) and the MIOpen provider from source,
    then install them to -InstallDir. Requires the MSVC toolchain and a ROCm devel
    wheel (_rocm_sdk_devel) in the selected Python env. Implies building the bindings.

.PARAMETER SkipProvider
    With -FullBuild, skip building the MIOpen provider (build only hipDNN + bindings).

.PARAMETER InstallDir
    Install prefix for -FullBuild. Default: <hipdnn>/install. (/opt/rocm does not
    exist on Windows.) The MIOpen plugin lands in <InstallDir>/lib/hipdnn_plugins/engines.

.PARAMETER GpuTargets
    GPU architecture(s) to build for. Default: gfx1150.

.PARAMETER BuildType
    CMake build type for -FullBuild. Default: Release.

.PARAMETER VcVars
    Path to vcvars64.bat. Default: auto-detected via vswhere, falling back to
    C:\develop\dist\vs-buildtools\VC\Auxiliary\Build\vcvars64.bat.

.PARAMETER WinSdkRoot
    Windows 10/11 SDK root. Default: 'C:\Program Files (x86)\Windows Kits\10'.

.PARAMETER WinSdkVersion
    Windows SDK version. Default: the newest version found under <WinSdkRoot>\Lib.

.PARAMETER CMakeExe
.PARAMETER NinjaExe
    cmake / ninja to use for -FullBuild. Default: whatever is on PATH.

.PARAMETER Force
    Clean reconfigure: wipe build dirs before -FullBuild, and rewrite the .pth wiring.

.PARAMETER Help
    Show this help and exit.

.EXAMPLE
    # Install into the ROCm wheel env and wire already-built bindings
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe

.EXAMPLE
    # Full build of hipDNN + bindings + MIOpen provider, then install the tool
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe -FullBuild

.EXAMPLE
    # Full build for a different GPU target, clean
    pwsh ./setup.ps1 -FullBuild -GpuTargets gfx942 -Force
#>
[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$FullBuild,
    [switch]$SkipProvider,
    [string]$InstallDir,
    [string]$GpuTargets = 'gfx1150',
    [string]$BuildType = 'Release',
    [string]$VcVars,
    [string]$WinSdkRoot = 'C:\Program Files (x86)\Windows Kits\10',
    [string]$WinSdkVersion,
    [string]$CMakeExe,
    [string]$NinjaExe,
    [switch]$Force,
    [switch]$Help
)

if ($Help) { Get-Help -Detailed $PSCommandPath; exit 0 }

$ErrorActionPreference = 'Stop'
# This script probes optional components by inspecting $LASTEXITCODE after running
# Python (e.g. "does rocm_sdk import?"). Since PowerShell 7.4 a native command that
# exits non-zero throws under ErrorActionPreference=Stop, which would abort those
# intentional probes — so opt out and check exit codes explicitly instead.
$PSNativeCommandUseErrorActionPreference = $false

# --- Paths -----------------------------------------------------------------
$ScriptDir   = $PSScriptRoot
$HipdnnRoot  = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$BindingsPkg = Join-Path $HipdnnRoot 'python'                              # hipdnn_frontend package
$BindingsLib = Join-Path $HipdnnRoot 'build\lib'                           # compiled .pyd lands here
$BuildDir    = Join-Path $HipdnnRoot 'build'
$ProviderDir = Join-Path $HipdnnRoot '..\..\dnn-providers\miopen-provider' # rocm-libraries/dnn-providers/...
if (-not $InstallDir) { $InstallDir = Join-Path $HipdnnRoot 'install' }

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Fwd($p) { return ($p -replace '\\', '/') }                        # backslashes -> CMake-friendly slashes

function Invoke-Native {
    # Run an external command and throw on non-zero exit.
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed (exit $LASTEXITCODE)"
    }
}

function Invoke-ToolchainBuild {
    # Run CMake command lines inside an MSVC + Windows SDK environment by writing a
    # throwaway .bat (vcvars64 cannot be sourced into PowerShell). Mirrors the proven
    # cmd pathway: call vcvars, append the SDK lib/include paths the unregistered
    # BuildTools instance can't locate, then run each command, aborting on the first
    # failure.
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string[]]$Commands,
        [switch]$BestEffort
    )
    $lines = @(
        '@echo off',
        'set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"',
        "call `"$VcVars`" >nul || (echo VCVARS FAILED & exit /b 1)",
        "set `"WINSDK=$WinSdkRoot`"",
        "set `"WINSDKVER=$WinSdkVersion`"",
        'set "LIB=%LIB%;%WINSDK%\Lib\%WINSDKVER%\um\x64;%WINSDK%\Lib\%WINSDKVER%\ucrt\x64"',
        'set "INCLUDE=%INCLUDE%;%WINSDK%\Include\%WINSDKVER%\um;%WINSDK%\Include\%WINSDKVER%\ucrt;%WINSDK%\Include\%WINSDKVER%\shared"'
    )
    foreach ($c in $Commands) { $lines += $c; $lines += 'if errorlevel 1 exit /b 1' }

    $bat = Join-Path $env:TEMP ("hipdnn_build_{0}.bat" -f ([System.Guid]::NewGuid().ToString('N')))
    Set-Content -Path $bat -Value $lines -Encoding ascii
    Write-Step "$Title"
    Write-Host "    (toolchain script: $bat)"
    & cmd /c $bat
    $code = $LASTEXITCODE
    Remove-Item $bat -ErrorAction SilentlyContinue
    if ($code -ne 0) {
        if ($BestEffort) { Write-Warn "$Title failed (exit $code); continuing."; return $false }
        throw "$Title failed (exit $code)."
    }
    return $true
}

# --- 1. Resolve the Python environment -------------------------------------
if ($PythonExe) { $Python = $PythonExe }
elseif ($env:VIRTUAL_ENV) { $Python = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
else { $Python = 'python' }

$resolved = (Get-Command $Python -ErrorAction SilentlyContinue)
if ($resolved) { $Python = $resolved.Source } else { throw "Python interpreter not found: $Python" }

$pyVersion = (& $Python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
Write-Step "Using Python $pyVersion at $Python"
$pyOk = (& $Python -c "import sys; print(1 if sys.version_info[:2] >= (3, 12) else 0)")
if ($pyOk.Trim() -ne '1') { Write-Warn "dnn-benchmark requires Python >= 3.12; found $pyVersion." }

# --- 2. Check the ROCm runtime is reachable --------------------------------
$rocmOk = $false
& $Python -c "import rocm_sdk" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Step "ROCm runtime: rocm_sdk wheel present (DLLs preloaded at import)."
    $rocmOk = $true
}
else {
    foreach ($var in @('ROCM_PATH', 'HIP_PATH', 'ROCM_HOME')) {
        $root = [Environment]::GetEnvironmentVariable($var)
        if ($root -and (Test-Path (Join-Path $root 'bin'))) {
            Write-Step "ROCm runtime: $var=$root (bin/ added to the DLL search path at import)."
            $rocmOk = $true; break
        }
    }
}
if (-not $rocmOk) {
    Write-Warn ("No ROCm runtime found (no rocm_sdk wheel, no ROCM_PATH/HIP_PATH/ROCM_HOME). " +
                "hipdnn_frontend will fail to import until one is provided.")
}

# --- 3. Build from source --------------------------------------------------
if ($FullBuild) {
    # Locate the ROCm devel wheel (provides clang++, hipcc, and the CMake configs).
    $Wheel = (& $Python -c "import os,_rocm_sdk_devel as d; print(os.path.dirname(d.__file__))" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $Wheel) {
        throw "-FullBuild needs the ROCm devel wheel (_rocm_sdk_devel) in $Python's env. Install the ROCm wheels or point -PythonExe at the wheel env."
    }
    $Wheel = $Wheel.Trim()

    # Resolve the toolchain (parameter > autodetect).
    if (-not $CMakeExe) { $CMakeExe = (Get-Command cmake -ErrorAction SilentlyContinue)?.Source }
    if (-not $CMakeExe) { throw "cmake not found on PATH; pass -CMakeExe." }
    if (-not $NinjaExe) { $NinjaExe = (Get-Command ninja -ErrorAction SilentlyContinue)?.Source }
    if (-not $NinjaExe) { throw "ninja not found on PATH; pass -NinjaExe." }

    if (-not $VcVars) {
        $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path $vswhere) {
            $vsPath = & $vswhere -latest -products * `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath 2>$null
            if ($vsPath) { $VcVars = Join-Path $vsPath.Trim() 'VC\Auxiliary\Build\vcvars64.bat' }
        }
        if (-not $VcVars -or -not (Test-Path $VcVars)) {
            $fallback = 'C:\develop\dist\vs-buildtools\VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $fallback) { $VcVars = $fallback }
        }
    }
    if (-not $VcVars -or -not (Test-Path $VcVars)) { throw "vcvars64.bat not found; pass -VcVars." }

    if (-not $WinSdkVersion) {
        $sdkLib = Join-Path $WinSdkRoot 'Lib'
        $WinSdkVersion = (Get-ChildItem $sdkLib -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^\d+\.' } | Sort-Object Name | Select-Object -Last 1).Name
    }
    if (-not $WinSdkVersion) { throw "No Windows SDK found under $WinSdkRoot\Lib; pass -WinSdkVersion." }

    Write-Step "Toolchain: cmake=$CMakeExe ninja=$NinjaExe"
    Write-Host  "           vcvars=$VcVars"
    Write-Host  "           winsdk=$WinSdkVersion  wheel=$Wheel"
    Write-Host  "           gpu=$GpuTargets  type=$BuildType  install=$InstallDir"

    $ProviderDir   = (Resolve-Path $ProviderDir -ErrorAction SilentlyContinue)?.Path
    $ProviderBuild = if ($ProviderDir) { Join-Path $ProviderDir 'build' } else { $null }

    if ($Force) {
        if (Test-Path $BuildDir) { Write-Step "Wiping $BuildDir (-Force)"; Remove-Item -Recurse -Force $BuildDir }
        if ($ProviderBuild -and (Test-Path $ProviderBuild)) { Remove-Item -Recurse -Force $ProviderBuild }
    }

    # hipDNN: configure -> build -> install (bindings included).
    $hipdnnCfg = '"{0}" -S "{1}" -B "{2}" -GNinja -DCMAKE_BUILD_TYPE={3} -DCMAKE_CXX_COMPILER="{4}/lib/llvm/bin/clang++.exe" -DCMAKE_MAKE_PROGRAM="{5}" -DCMAKE_PREFIX_PATH="{4}" -DROCM_CMAKE_PATH="{4}" -DPython_EXECUTABLE="{6}" -DGPU_TARGETS={7} -DENABLE_CLANG_FORMAT=OFF -DHIPDNN_SKIP_TESTS=ON -DHIPDNN_BUILD_PYTHON_BINDINGS=ON -DCMAKE_INSTALL_PREFIX="{8}"' -f `
        $CMakeExe, (Fwd $HipdnnRoot), (Fwd $BuildDir), $BuildType, (Fwd $Wheel), (Fwd $NinjaExe), (Fwd $Python), $GpuTargets, (Fwd $InstallDir)
    $hipdnnBuild   = '"{0}" --build "{1}"'   -f $CMakeExe, (Fwd $BuildDir)
    $hipdnnInstall = '"{0}" --install "{1}"' -f $CMakeExe, (Fwd $BuildDir)
    Invoke-ToolchainBuild -Title "Building + installing hipDNN (with Python bindings)" `
        -Commands @($hipdnnCfg, $hipdnnBuild, $hipdnnInstall) | Out-Null

    # MIOpen provider: built against the freshly installed hipDNN (best-effort —
    # Windows provider support is still maturing).
    if (-not $SkipProvider) {
        if ($ProviderDir) {
            $provCfg = '"{0}" -S "{1}" -B "{2}" -GNinja -DCMAKE_BUILD_TYPE={3} -DCMAKE_MAKE_PROGRAM="{4}" -DCMAKE_PREFIX_PATH="{5};{6}" -DROCM_CMAKE_PATH="{6}" -DROCM_PATH="{6}" -DGPU_TARGETS={7} -DMIOPENPROVIDER_SKIP_TESTS=ON -DCMAKE_INSTALL_PREFIX="{5}"' -f `
                $CMakeExe, (Fwd $ProviderDir), (Fwd $ProviderBuild), $BuildType, (Fwd $NinjaExe), (Fwd $InstallDir), (Fwd $Wheel), $GpuTargets
            $provBuild   = '"{0}" --build "{1}"'   -f $CMakeExe, (Fwd $ProviderBuild)
            $provInstall = '"{0}" --install "{1}"' -f $CMakeExe, (Fwd $ProviderBuild)
            $ok = Invoke-ToolchainBuild -Title "Building + installing MIOpen provider" `
                -Commands @($provCfg, $provBuild, $provInstall) -BestEffort
            if ($ok) { Write-Step "MIOpen plugin installed to $InstallDir\lib\hipdnn_plugins\engines" }
        }
        else {
            Write-Warn "MIOpen provider not found at $ProviderDir; skipping (pass -SkipProvider to silence)."
        }
    }
}

# --- 4. Wire the compiled bindings onto the environment via a .pth ----------
# The bindings are built out-of-tree (subdirectory CMake build), so the package
# dir (python/) and the compiled extension (build/lib/) are not on sys.path. A
# .pth in site-packages adds both persistently — no PYTHONPATH needed.
$builtPyd = Get-ChildItem -Path $BindingsLib -Filter 'hipdnn_frontend_python*.pyd' -ErrorAction SilentlyContinue |
    Select-Object -First 1

& $Python -c "import hipdnn_frontend" 2>$null
$alreadyImportable = ($LASTEXITCODE -eq 0)

if ($alreadyImportable -and -not $Force) {
    Write-Step "hipdnn_frontend already importable; leaving it as-is."
}
elseif ($builtPyd) {
    $sitePkgs = (& $Python -c "import sysconfig; print(sysconfig.get_path('purelib'))").Trim()
    $pth = Join-Path $sitePkgs 'hipdnn_frontend.pth'
    Write-Step "Wiring hipdnn_frontend onto the env via $pth"
    Set-Content -Path $pth -Value @($BindingsPkg, $BindingsLib) -Encoding ascii
}
else {
    Write-Warn ("hipdnn_frontend is not importable and no compiled extension was found " +
                "under $BindingsLib. Re-run with -FullBuild, or " +
                "pip-install the bindings from $BindingsPkg (see python/README.md).")
}

# --- 5. Install the dnn-benchmark package + PyTorch ------------------------
# torch is omitted from pyproject.toml (optional dep), so install it explicitly
# here. The default PyPI wheel on Windows is the CPU build, which backs the
# PyTorch reference provider (--backend pytorch / --validate pytorch). pip
# resolves the package's own deps (numpy / pytest / psutil) from PyPI.
Write-Step "Installing dnn-benchmark (editable) and its PyPI dependencies"
Invoke-Native $Python @('-m', 'pip', 'install', '-e', $ScriptDir)
Write-Step "Installing PyTorch (CPU)"
Invoke-Native $Python @('-m', 'pip', 'install', 'torch')

# --- 6. Best-effort amdsmi (powers the GPU SMI snapshot) -------------------
# Not on PyPI; ships with the HIP SDK. If absent, metrics/gpu_smi.py degrades to
# None fields (warn-once), so this is purely best-effort.
& $Python -c "import amdsmi" 2>$null
if ($LASTEXITCODE -ne 0) {
    $hipRoot = $env:HIP_PATH; if (-not $hipRoot) { $hipRoot = $env:ROCM_PATH }
    $amdsmiDir = if ($hipRoot) { Join-Path $hipRoot 'share\amd_smi' } else { $null }
    if ($amdsmiDir -and (Test-Path $amdsmiDir)) {
        Write-Step "Installing amdsmi Python bindings from $amdsmiDir"
        & $Python -m pip install $amdsmiDir
        if ($LASTEXITCODE -ne 0) { Write-Warn "amdsmi install failed; GPU SMI snapshot disabled." }
    }
    else { Write-Warn "amdsmi not found; GPU SMI snapshot disabled (optional)." }
}

# --- 7. Verify -------------------------------------------------------------
Write-Step "Verifying installation"
& $Python -c "import dnn_benchmarking; print('dnn_benchmarking OK')"
if ($LASTEXITCODE -ne 0) { throw "dnn_benchmarking failed to import." }

& $Python -c "import hipdnn_frontend; print('hipdnn_frontend OK')" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Warn "hipdnn_frontend could not be imported (ROCm runtime or bindings missing)." }

& $Python -m dnn_benchmarking --help > $null
if ($LASTEXITCODE -ne 0) { throw "dnn-benchmark CLI failed to run." }

# --- 8. Next steps ---------------------------------------------------------
Write-Host ""
Write-Step "Setup complete."
Write-Host "  Run benchmarks with:" -ForegroundColor Green
Write-Host "    & '$Python' -m dnn_benchmarking --graph <graph.json>"
if ($FullBuild -and -not $SkipProvider) {
    Write-Host "    & '$Python' -m dnn_benchmarking --graph <graph.json> ``"
    Write-Host "        --plugin-path '$InstallDir\lib\hipdnn_plugins\engines'"
}
Write-Host ""
Write-Host "  PyTorch (CPU) was installed: --backend pytorch and --validate pytorch work."
Write-Host "  GPU kernel-event timing via torch is unavailable (needs a ROCm/CUDA build"
Write-Host "  of torch, which isn't readily available on Windows)."
