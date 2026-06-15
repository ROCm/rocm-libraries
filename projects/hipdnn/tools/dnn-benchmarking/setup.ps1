#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows setup for the dnn-benchmark tool (PowerShell analogue of setup.sh).

.DESCRIPTION
    Installs dnn-benchmark into an existing Python env (the active venv or
    -PythonExe; on Windows this should be the ROCm-wheel env). Optionally builds
    hipDNN + the Python bindings and the MIOpen provider from source (-FullBuild),
    wires the bindings onto the env via a .pth, installs the tool (editable) plus
    CPU PyTorch, and verifies the result.

.PARAMETER PythonExe
    Python interpreter to install into. Default: the active venv, else 'python' on
    PATH. Recommended: the ROCm wheel env's python (rocm_sdk + _rocm_sdk_devel).

.PARAMETER FullBuild
    Build hipDNN (with bindings) and the MIOpen provider from source, then install
    them to -InstallDir. Needs the MSVC toolchain and the _rocm_sdk_devel wheel.

.PARAMETER InstallDir
    Install prefix for -FullBuild. Default: <hipdnn>/install.

.PARAMETER GpuTargets
    GPU architecture(s) to build for. Default: gfx1150.

.PARAMETER Force
    Clean reconfigure: wipe build dirs before -FullBuild, and rewrite the .pth wiring.

.EXAMPLE
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe

.EXAMPLE
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe -FullBuild
#>
[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$FullBuild,
    [string]$InstallDir,
    [string]$GpuTargets = 'gfx1150',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
# Let intentional exit-code probes (e.g. "does rocm_sdk import?") not throw.
$PSNativeCommandUseErrorActionPreference = $false

# --- Paths -----------------------------------------------------------------
$ScriptDir   = $PSScriptRoot
$HipdnnRoot  = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$BindingsPkg = Join-Path $HipdnnRoot 'python'
$BindingsLib = Join-Path $HipdnnRoot 'build\lib'
$BuildDir    = Join-Path $HipdnnRoot 'build'
$ProviderDir = Join-Path $HipdnnRoot '..\..\dnn-providers\miopen-provider'
$BuildType   = 'Release'
$WinSdkRoot  = 'C:\Program Files (x86)\Windows Kits\10'
if (-not $InstallDir) { $InstallDir = Join-Path $HipdnnRoot 'install' }

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Fwd($p) { return ($p -replace '\\', '/') }

function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed (exit $LASTEXITCODE)"
    }
}

function Invoke-ToolchainBuild {
    # Run CMake inside an MSVC + Windows SDK env via a throwaway .bat (vcvars64
    # can't be sourced into PowerShell); append the SDK lib/include paths the
    # unregistered BuildTools instance can't locate.
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
            Write-Step "ROCm runtime: $var=$root (bin/ on the DLL search path at import)."
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
    # ROCm devel wheel: provides clang++, hipcc, and the CMake configs.
    $Wheel = (& $Python -c "import os,_rocm_sdk_devel as d; print(os.path.dirname(d.__file__))" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $Wheel) {
        throw "-FullBuild needs the ROCm devel wheel (_rocm_sdk_devel) in $Python's env."
    }
    $Wheel = $Wheel.Trim()

    $CMakeExe = (Get-Command cmake -ErrorAction SilentlyContinue)?.Source
    if (-not $CMakeExe) { throw "cmake not found on PATH." }
    $NinjaExe = (Get-Command ninja -ErrorAction SilentlyContinue)?.Source
    if (-not $NinjaExe) { throw "ninja not found on PATH." }

    # vcvars64: prefer vswhere, fall back to the BuildTools install location.
    $VcVars = $null
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
    if (-not $VcVars -or -not (Test-Path $VcVars)) { throw "vcvars64.bat not found." }

    $WinSdkVersion = (Get-ChildItem (Join-Path $WinSdkRoot 'Lib') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d+\.' } | Sort-Object Name | Select-Object -Last 1).Name
    if (-not $WinSdkVersion) { throw "No Windows SDK found under $WinSdkRoot\Lib." }

    Write-Step "Toolchain: cmake=$CMakeExe ninja=$NinjaExe"
    Write-Host  "           vcvars=$VcVars  winsdk=$WinSdkVersion"
    Write-Host  "           wheel=$Wheel  gpu=$GpuTargets  install=$InstallDir"

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

    # MIOpen provider: built against the freshly installed hipDNN (best-effort).
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
        Write-Warn "MIOpen provider not found at $ProviderDir; skipping."
    }
}

# --- 4. Wire the compiled bindings onto the environment via a .pth ----------
# Bindings build out-of-tree, so add python/ + build/lib/ to site-packages.
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
                "under $BindingsLib. Re-run with -FullBuild, or pip-install the bindings " +
                "from $BindingsPkg (see python/README.md).")
}

# --- 5. Install the dnn-benchmark package + PyTorch ------------------------
# torch is optional (not in pyproject.toml); the default PyPI wheel on Windows is
# the CPU build, which backs --backend pytorch / --validate pytorch.
Write-Step "Installing dnn-benchmark (editable) and its PyPI dependencies"
Invoke-Native $Python @('-m', 'pip', 'install', '-e', $ScriptDir)
Write-Step "Installing PyTorch (CPU)"
Invoke-Native $Python @('-m', 'pip', 'install', 'torch')

# --- 6. Best-effort amdsmi (powers the GPU SMI snapshot) -------------------
# Ships with the HIP SDK (not on PyPI); metrics degrade gracefully if absent.
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
if ($FullBuild) {
    Write-Host "    & '$Python' -m dnn_benchmarking --graph <graph.json> ``"
    Write-Host "        --plugin-path '$InstallDir\lib\hipdnn_plugins\engines'"
}
