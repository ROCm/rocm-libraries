#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows setup for the dnn-benchmark tool (PowerShell analogue of setup.sh).

.DESCRIPTION
    Installs dnn-benchmark into an existing Python env (the active venv or
    -PythonExe; on Windows this should be the ROCm-wheel env). Optionally builds
    hipDNN + the Python bindings and the MIOpen provider from source (-ForceBuild),
    wires the bindings onto the env via a .pth, installs the tool (editable) and
    PyTorch per -TorchMode, then verifies the result.

    Parameters mirror setup.sh where they apply on Windows. setup.sh's venv
    management (--reuse-venv / --workspace) is omitted: this script installs into
    an env you select rather than creating one. --torch-mode's rocm/cuda values
    are omitted too (ROCm torch isn't available on Windows).

.PARAMETER PythonExe
    Python interpreter to install into. Default: the active venv, else 'python' on
    PATH. Recommended: the ROCm wheel env's python (rocm_sdk + _rocm_sdk_devel).

.PARAMETER TorchMode
    How torch is provided (setup.sh --torch-mode, Windows subset). Default: cpu.
      cpu:      install CPU-only torch (from -TorchIndexUrl or the PyTorch CPU index).
      existing: reuse torch already installed in the env (error if absent).
      none:     leave torch uninstalled.

.PARAMETER TorchIndexUrl
    Override the pip index URL used for torch (setup.sh --torch-index-url).

.PARAMETER ForceBuild
    Build hipDNN (with bindings) and the MIOpen provider from source, then install
    them to -InstallDir (setup.sh --force-build). Needs the MSVC toolchain and a
    ROCm devel prefix (the _rocm_sdk_devel wheel, or -RocmPrefix).

.PARAMETER RocmPrefix
    Explicit ROCm/hipDNN prefix for the binding/provider build (setup.sh
    --rocm-prefix). Takes precedence over _rocm_sdk_devel wheel discovery.

.PARAMETER InstallDir
    Install prefix for -ForceBuild. Default: <hipdnn>/install.

.PARAMETER GpuArch
    GPU architecture to build for, i.e. GPU_TARGETS (setup.sh --gpu-arch).
    Default: gfx1150.

.PARAMETER Force
    Clean reconfigure: wipe build dirs before -ForceBuild, and rewrite the .pth.

.EXAMPLE
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe

.EXAMPLE
    pwsh ./setup.ps1 -PythonExe C:/develop/latest_wheels/Scripts/python.exe -ForceBuild

.EXAMPLE
    pwsh ./setup.ps1 -TorchMode existing
#>
[CmdletBinding()]
param(
    [string]$PythonExe,
    [ValidateSet('cpu', 'existing', 'none')]
    [string]$TorchMode = 'cpu',
    [string]$TorchIndexUrl,
    [switch]$ForceBuild,
    [string]$RocmPrefix,
    [string]$InstallDir,
    [string]$GpuArch = 'gfx1150',
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

# wheel_build_setup.ps1 bootstraps a ROCm-wheel venv when the selected env has no
# ROCm runtime. We pass $WheelVenvPath as its -VenvPath (overriding the script's
# D:\ default, which assumes a drive that isn't always present) and reuse the same
# path to locate the python it creates.
$WheelSetupScript = Join-Path $HipdnnRoot 'scripts\windows\wheel_build_setup.ps1'
$WheelVenvPath    = 'C:\develop\latest_wheels'

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

function New-CMakeStages {
    # Turn one CMake project into its configure -> build -> install command lines,
    # ready to run inside the toolchain .bat. Source/build paths are forward-slashed
    # and quoted here; $ConfigureArgs carries the project-specific -D flags (each
    # already quoted as needed). Keeps the call sites a readable list of flags
    # instead of one positional format string.
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Build,
        [Parameter(Mandatory)][string[]]$ConfigureArgs
    )
    $cmake = '"{0}"' -f $CMakeExe
    $src   = '"{0}"' -f (Fwd $Source)
    $bld   = '"{0}"' -f (Fwd $Build)
    $configure = '{0} -S {1} -B {2} {3}' -f $cmake, $src, $bld, ($ConfigureArgs -join ' ')
    $build     = '{0} --build {1}'       -f $cmake, $bld
    $install   = '{0} --install {1}'     -f $cmake, $bld
    return @($configure, $build, $install)
}

function Get-TorchMode {
    # Mirror setup.sh get_torch_mode: rocm / cuda / cpu / missing.
    $code = @'
try:
    import torch
except Exception:
    print("missing")
else:
    if getattr(torch.version, "hip", None):
        print("rocm")
    elif getattr(torch.version, "cuda", None):
        print("cuda")
    else:
        print("cpu")
'@
    $mode = (& $Python -c $code 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $mode) { return 'missing' }
    return $mode.Trim()
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
        'set "INCLUDE=%INCLUDE%;%WINSDK%\Include\%WINSDKVER%\um;%WINSDK%\Include\%WINSDKVER%\ucrt;%WINSDK%\Include\%WINSDKVER%\shared"',
        # hipDNN's ClangToolChain warns when ROCM_PATH leaks via the environment;
        # clear it for the build and pass the prefix as -DROCM_PATH in the args.
        'set "ROCM_PATH="'
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

function Test-RocmRuntime {
    # rocm_sdk wheel preloads the ROCm DLLs at import; otherwise a ROCm prefix
    # env var must point at a bin/ that lands on the DLL search path.
    & $Python -c "import rocm_sdk" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Step "ROCm runtime: rocm_sdk wheel present (DLLs preloaded at import)."
        return $true
    }
    foreach ($var in @('ROCM_PATH', 'HIP_PATH', 'ROCM_HOME')) {
        $root = [Environment]::GetEnvironmentVariable($var)
        if ($root -and (Test-Path (Join-Path $root 'bin'))) {
            Write-Step "ROCm runtime: $var=$root (bin/ on the DLL search path at import)."
            return $true
        }
    }
    return $false
}

# --- 1. Resolve the Python environment -------------------------------------
# Selection order: explicit -PythonExe, then an active venv, then the ROCm wheel
# venv at $WheelVenvPath (the env this tool targets), and only then a bare system
# python. Preferring the wheel venv keeps the install out of system python even
# when the system python happens to see a ROCm runtime via HIP_PATH/ROCM_PATH.
if ($PythonExe) { $Python = $PythonExe }
elseif ($env:VIRTUAL_ENV) { $Python = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
elseif (Test-Path (Join-Path $WheelVenvPath 'Scripts\python.exe')) {
    $Python = Join-Path $WheelVenvPath 'Scripts\python.exe'
}
else { $Python = 'python' }

$resolved = (Get-Command $Python -ErrorAction SilentlyContinue)
if ($resolved) { $Python = $resolved.Source } else { throw "Python interpreter not found: $Python" }

$pyVersion = (& $Python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
Write-Step "Using Python $pyVersion at $Python"
$pyOk = (& $Python -c "import sys; print(1 if sys.version_info[:2] >= (3, 12) else 0)")
if ($pyOk.Trim() -ne '1') { Write-Warn "dnn-benchmark requires Python >= 3.12; found $pyVersion." }

# --- 2. Check the ROCm runtime is reachable --------------------------------
$rocmOk = Test-RocmRuntime

# No ROCm runtime in the selected env: fall back to the wheel venv at
# $WheelVenvPath. Prefer one that already exists; only build a fresh one with
# wheel_build_setup.ps1 (creates the venv + installs rocm[libraries,devel] and
# sets ROCM_PATH for the session) when it's absent. In either case switch $Python
# to that venv. Skipped when -PythonExe was given — that's a deliberate env choice.
if (-not $rocmOk -and -not $PythonExe) {
    $wheelPython = Join-Path $WheelVenvPath 'Scripts\python.exe'

    # Reuse an existing wheel env before building a new one.
    if (Test-Path $wheelPython) {
        $Python    = (Get-Command $wheelPython).Source
        $pyVersion = (& $Python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
        Write-Step "Found wheel env at $WheelVenvPath; using Python $pyVersion"
        $rocmOk = Test-RocmRuntime
    }

    # Still no runtime: bootstrap one.
    if (-not $rocmOk) {
        if (Test-Path $WheelSetupScript) {
            Write-Step "Bootstrapping a wheel env via wheel_build_setup.ps1"
            & $WheelSetupScript -VenvPath $WheelVenvPath -GpuTarget $GpuArch
            if ($LASTEXITCODE -ne 0) { throw "wheel_build_setup.ps1 failed (exit $LASTEXITCODE)." }
            if (-not (Test-Path $wheelPython)) {
                throw "wheel_build_setup.ps1 did not produce a python at $wheelPython."
            }
            $Python    = (Get-Command $wheelPython).Source
            $pyVersion = (& $Python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
            Write-Step "Using bootstrapped Python $pyVersion at $Python"
            $rocmOk = Test-RocmRuntime
        }
        else {
            Write-Warn "wheel_build_setup.ps1 not found at $WheelSetupScript; cannot bootstrap a ROCm env."
        }
    }
}

if (-not $rocmOk) {
    Write-Warn ("No ROCm runtime found (no rocm_sdk wheel, no ROCM_PATH/HIP_PATH/ROCM_HOME). " +
                "hipdnn_frontend will fail to import until one is provided.")
}

# --- 3. Build from source --------------------------------------------------
if ($ForceBuild) {
    # ROCm devel prefix: provides clang++, hipcc, and the CMake configs. Prefer
    # -RocmPrefix, else discover the _rocm_sdk_devel wheel in the env.
    if ($RocmPrefix) {
        $Wheel = $RocmPrefix
    }
    else {
        $Wheel = (& $Python -c "import os,_rocm_sdk_devel as d; print(os.path.dirname(d.__file__))" 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $Wheel) {
            throw "-ForceBuild needs the ROCm devel wheel (_rocm_sdk_devel) in $Python's env, or pass -RocmPrefix."
        }
        $Wheel = $Wheel.Trim()
    }

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
    Write-Host  "           rocm=$Wheel  gpu=$GpuArch  install=$InstallDir"

    $ProviderDir   = (Resolve-Path $ProviderDir -ErrorAction SilentlyContinue)?.Path
    $ProviderBuild = if ($ProviderDir) { Join-Path $ProviderDir 'build' } else { $null }

    if ($Force) {
        if (Test-Path $BuildDir) { Write-Step "Wiping $BuildDir (-Force)"; Remove-Item -Recurse -Force $BuildDir }
        if ($ProviderBuild -and (Test-Path $ProviderBuild)) { Remove-Item -Recurse -Force $ProviderBuild }
    }

    # Forward-slashed copies of the paths CMake flags embed (Windows backslashes
    # would be read as escapes inside the .bat command lines).
    $wheelFwd   = Fwd $Wheel
    $ninjaFwd   = Fwd $NinjaExe
    $pythonFwd  = Fwd $Python
    $installFwd = Fwd $InstallDir

    # hipDNN: configure -> build -> install (Python bindings included).
    $hipdnnArgs = @(
        '-GNinja'
        "-DCMAKE_BUILD_TYPE=$BuildType"
        "-DCMAKE_CXX_COMPILER=`"$wheelFwd/lib/llvm/bin/clang++.exe`""
        "-DCMAKE_MAKE_PROGRAM=`"$ninjaFwd`""
        "-DCMAKE_PREFIX_PATH=`"$wheelFwd`""
        "-DROCM_CMAKE_PATH=`"$wheelFwd`""
        "-DROCM_PATH=`"$wheelFwd`""
        "-DPython_EXECUTABLE=`"$pythonFwd`""
        "-DGPU_TARGETS=$GpuArch"
        '-DENABLE_CLANG_FORMAT=OFF'
        '-DHIPDNN_SKIP_TESTS=ON'
        '-DHIPDNN_BUILD_PYTHON_BINDINGS=ON'
        "-DCMAKE_INSTALL_PREFIX=`"$installFwd`""
    )
    $hipdnnStages = New-CMakeStages -Source $HipdnnRoot -Build $BuildDir -ConfigureArgs $hipdnnArgs
    Invoke-ToolchainBuild -Title "Building + installing hipDNN (with Python bindings)" `
        -Commands $hipdnnStages | Out-Null

    # MIOpen provider: built against the freshly installed hipDNN (best-effort).
    if ($ProviderDir) {
        $provArgs = @(
            '-GNinja'
            "-DCMAKE_BUILD_TYPE=$BuildType"
            "-DCMAKE_MAKE_PROGRAM=`"$ninjaFwd`""
            "-DCMAKE_PREFIX_PATH=`"$installFwd;$wheelFwd`""
            "-DROCM_CMAKE_PATH=`"$wheelFwd`""
            "-DROCM_PATH=`"$wheelFwd`""
            "-DGPU_TARGETS=$GpuArch"
            '-DMIOPENPROVIDER_SKIP_TESTS=ON'
            # clang-format/-tidy are required-by-default dev lints that hard-fail
            # configure when the tools aren't on PATH; off for an artifact build.
            '-DENABLE_CLANG_FORMAT=OFF'
            '-DENABLE_CLANG_TIDY=OFF'
            "-DCMAKE_INSTALL_PREFIX=`"$installFwd`""
        )
        $provStages = New-CMakeStages -Source $ProviderDir -Build $ProviderBuild -ConfigureArgs $provArgs
        Invoke-ToolchainBuild -Title "Building + installing MIOpen provider" `
            -Commands $provStages -BestEffort | Out-Null

        # Report success on the actual artifact, not the build's exit status — the
        # plugin .dll landing in the engines dir is the signal that matters. On
        # Windows the RUNTIME .dll installs under bin/; lib/ is the Linux layout.
        $PluginEnginesDir = @(
            (Join-Path $InstallDir 'bin\hipdnn_plugins\engines'),
            (Join-Path $InstallDir 'lib\hipdnn_plugins\engines')
        ) | Where-Object { Get-ChildItem $_ -Filter '*.dll' -ErrorAction SilentlyContinue } |
            Select-Object -First 1
        if ($PluginEnginesDir) {
            Write-Step "MIOpen plugin installed to $PluginEnginesDir"
        }
        else {
            Write-Warn ("MIOpen provider produced no plugin under " +
                        "$InstallDir\{bin,lib}\hipdnn_plugins\engines (see build output above).")
        }
    }
    else {
        Write-Warn "MIOpen provider not found at $ProviderDir; skipping."
    }
}

# --- 4. Wire the compiled bindings onto the environment via a .pth ----------
# Bindings build out-of-tree. Add python/ (the package) plus the directory that
# holds the compiled extension to site-packages. The .pyd can land in build/lib
# (Ninja, -ForceBuild) or build/<config>/lib (multi-config generators), so probe
# all three and prefer a release build over a debug one.
$pydDir = $null
foreach ($cand in @($BindingsLib,
                    (Join-Path $BuildDir 'release\lib'),
                    (Join-Path $BuildDir 'debug\lib'))) {
    if (Get-ChildItem -Path $cand -Filter 'hipdnn_frontend_python*.pyd' -ErrorAction SilentlyContinue) {
        $pydDir = $cand; break
    }
}

& $Python -c "import hipdnn_frontend" 2>$null
$alreadyImportable = ($LASTEXITCODE -eq 0)

if ($alreadyImportable -and -not $Force) {
    Write-Step "hipdnn_frontend already importable; leaving it as-is."
}
elseif ($pydDir) {
    # The extension links hipdnn_backend.dll, which the build drops in a sibling
    # bin/ (build[/<config>]/bin), not next to the .pyd. Extension modules load
    # with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS (no PATH, no RPATH), so register that
    # bin/ via os.add_dll_directory from the .pth — stashing the handle on sys so
    # it isn't GC'd before the import. ROCm deps are preloaded by the package
    # __init__ (rocm_sdk / ROCM_PATH) ahead of the extension import.
    $lines = @($BindingsPkg, $pydDir)
    $backendBin = Join-Path (Split-Path $pydDir -Parent) 'bin'
    if (Test-Path (Join-Path $backendBin 'hipdnn_backend.dll')) {
        $lines += "import os, sys; _p = r'$backendBin'; sys.__dict__.setdefault('_hipdnn_dll_dirs', []).append(os.add_dll_directory(_p))"
    }

    $sitePkgs = (& $Python -c "import sysconfig; print(sysconfig.get_path('purelib'))").Trim()
    $pth = Join-Path $sitePkgs 'hipdnn_frontend.pth'
    Write-Step "Wiring hipdnn_frontend onto the env via $pth (extension in $pydDir)"
    Set-Content -Path $pth -Value $lines -Encoding ascii
}
else {
    Write-Warn ("hipdnn_frontend is not importable and no compiled extension was found " +
                "under $BindingsLib or $BuildDir\<config>\lib. Re-run with -ForceBuild, or " +
                "pip-install the bindings from $BindingsPkg (see python/README.md).")
}

# --- 5. Install the dnn-benchmark package + PyTorch ------------------------
# torch is omitted from pyproject.toml so pip never replaces the selected wheel;
# install it explicitly per -TorchMode.
Write-Step "Installing dnn-benchmark (editable) and its PyPI dependencies"
Invoke-Native $Python @('-m', 'pip', 'install', '-e', $ScriptDir)

$installedTorch = Get-TorchMode
switch ($TorchMode) {
    'none' {
        Write-Step "Torch mode 'none': leaving torch uninstalled."
    }
    'existing' {
        if ($installedTorch -eq 'missing') {
            throw "-TorchMode existing requires torch already installed in the selected env."
        }
        Write-Step "Torch mode 'existing': using installed torch ($installedTorch)."
    }
    'cpu' {
        if ($installedTorch -ne 'missing') {
            Write-Step "Torch mode 'cpu': torch already present ($installedTorch); leaving as-is."
        }
        else {
            $idx = if ($TorchIndexUrl) { $TorchIndexUrl } else { 'https://download.pytorch.org/whl/cpu' }
            Write-Step "Installing CPU PyTorch from $idx"
            # The torch wheel is ~120 MB; raise pip's 15s socket timeout and retries
            # so a slow link doesn't abort the download mid-stream.
            Invoke-Native $Python @('-m', 'pip', 'install', 'torch',
                '--index-url', $idx, '--timeout', '120', '--retries', '10')
        }
    }
}

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
# pip install drops the `dnn-benchmark` console script next to the interpreter;
# echo that shorthand rather than `python -m dnn_benchmarking`.
$ScriptsDir = Split-Path $Python -Parent
$DnnExe     = Join-Path $ScriptsDir 'dnn-benchmark.exe'

Write-Host ""
Write-Step "Setup complete."
Write-Host "  Run benchmarks with:" -ForegroundColor Green
Write-Host "    & '$DnnExe' --graph <graph.json>"
if ($ForceBuild) {
    # Point at the engines dir the provider actually installed to (bin/ on Windows).
    if (-not $PluginEnginesDir) { $PluginEnginesDir = Join-Path $InstallDir 'bin\hipdnn_plugins\engines' }
    Write-Host "    & '$DnnExe' --graph <graph.json> ``"
    Write-Host "        --plugin-path '$PluginEnginesDir'"
}

# If the install landed in a venv, point at its activate script so callers can use
# a bare `dnn-benchmark` instead of the absolute path.
$activate = Join-Path $ScriptsDir 'Activate.ps1'
if (Test-Path $activate) {
    Write-Host ""
    Write-Host "  Or activate the venv first, then use 'dnn-benchmark' directly:" -ForegroundColor Green
    Write-Host "    & '$activate'"
    Write-Host "    dnn-benchmark --graph <graph.json>"
}
