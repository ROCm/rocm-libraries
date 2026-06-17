# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys
from pathlib import Path

from invoke import task

ROOT_PATH = Path(__file__).resolve().parent
BUILD_DIR = ROOT_PATH / "build"

# Fail early if invoke is running under a different Python than the active
# venv.  This happens when invoke is installed system-wide (/usr/bin/invoke)
# but a venv is active — sys.executable will be the system Python, so cmake
# gets the wrong -DPython_EXECUTABLE and venv packages like pytest won't be
# found.
_venv = os.environ.get("VIRTUAL_ENV")
if _venv and not sys.executable.startswith(_venv):
    raise SystemExit(
        f"ERROR: invoke is running under {sys.executable} but VIRTUAL_ENV "
        f"is set to {_venv}.\n"
        f"Install invoke in the venv:  pip install invoke\n"
        f"Then re-run:  invoke build"
    )


def _detect_rocm() -> Path:
    """Detect ROCm installation path.

    Priority: ROCM_PATH env > rocm-sdk path --root > /opt/rocm (Linux) or error (Windows).
    Warns if both ROCM_PATH and rocm-sdk are set but point to different paths.
    """
    env_path = os.environ.get("ROCM_PATH")

    sdk_path = None
    if shutil.which("rocm-sdk"):
        try:
            result = (
                subprocess.check_output(
                    ["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            if result:
                sdk_path = result
        except subprocess.CalledProcessError:
            pass

    # Warn if both are set but differ
    if env_path and sdk_path:
        if Path(env_path).resolve() != Path(sdk_path).resolve():
            print(
                f"Warning: ROCM_PATH ({env_path}) differs from "
                f"rocm-sdk path --root ({sdk_path}). Using ROCM_PATH."
            )

    # 1. ROCM_PATH environment variable
    if env_path:
        return Path(env_path)

    # 2. rocm-sdk CLI
    if sdk_path:
        return Path(sdk_path)

    # 3. Platform-specific fallback
    if sys.platform == "win32":
        raise RuntimeError(
            "ROCm SDK not found. Install rocm-sdk or set ROCM_PATH environment variable."
        )
    return Path("/opt/rocm")


# ---------------------------------------------------------------------------
# Windows helpers (no-ops on Linux)
# ---------------------------------------------------------------------------


def _setup_msvc_env():
    """Initialize the full MSVC build environment from vcvarsall.bat."""
    if sys.platform != "win32":
        return
    if "VSINSTALLDIR" in os.environ:
        return  # Already inside a Developer Command Prompt
    vs_root = Path("C:/Program Files (x86)/Microsoft Visual Studio")
    candidates = sorted(
        vs_root.glob("*/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"), reverse=True
    )
    if not candidates:
        print(
            "Warning: vcvarsall.bat not found. Ensure Visual Studio Build Tools are installed."
        )
        return
    vcvarsall = candidates[0]
    result = subprocess.run(
        f'"{vcvarsall}" amd64 && set',
        capture_output=True,
        text=True,
        encoding="mbcs",
        shell=True,
    )
    original_lib = os.environ.get("LIB", "")
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value
    # Restore original LIB entries so vcvarsall doesn't drop existing SDK paths
    if original_lib:
        existing = os.environ.get("LIB", "")
        merged = ";".join(dict.fromkeys(existing.split(";") + original_lib.split(";")))
        os.environ["LIB"] = merged

    # Ensure Windows SDK lib, include, and bin paths are present.
    prog86 = os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    sdk_lib_root = Path(prog86) / "Windows Kits/10/lib"
    sdk_versions = (
        sorted(sdk_lib_root.glob("*/"), reverse=True) if sdk_lib_root.exists() else []
    )
    if sdk_versions:
        sdk_ver = sdk_versions[0].name
        sdk_root = sdk_lib_root.parent

        sdk_libs = [
            str(sdk_root / "lib" / sdk_ver / "ucrt/x64"),
            str(sdk_root / "lib" / sdk_ver / "um/x64"),
        ]
        current_lib = os.environ.get("LIB", "")
        missing_libs = [p for p in sdk_libs if p not in current_lib]
        if missing_libs:
            os.environ["LIB"] = ";".join(
                missing_libs + ([current_lib] if current_lib else [])
            )

        sdk_includes = [
            str(p) for p in (sdk_root / "include" / sdk_ver).iterdir() if p.is_dir()
        ]
        current_include = os.environ.get("INCLUDE", "")
        missing_includes = [p for p in sdk_includes if p not in current_include]
        if missing_includes:
            os.environ["INCLUDE"] = ";".join(
                missing_includes + ([current_include] if current_include else [])
            )

        # Add the SDK bin directory to PATH so rc.exe is found by CMake automatically.
        sdk_bin = str(sdk_root / "bin" / sdk_ver / "x64")
        if sdk_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = sdk_bin + ";" + os.environ.get("PATH", "")


def _rmtree(path: Path):
    """Remove a directory tree, handling read-only files on Windows."""

    def _on_error(func, path, _exc):
        os.chmod(path, 0o666)
        func(path)

    shutil.rmtree(path, onexc=_on_error)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(
    help={
        "build_dir": "Override the build directory (default: build/).",
        "build_type": "CMake build type: Release, Debug, RelWithDebInfo (default: Release).",
        "tests": "Build unit tests (default: ON when building standalone).",
        "no-python": "Disable Python bindings (enabled by default).",
        "static": "Build as a static library instead of shared.",
        "jobs": "Number of parallel build jobs (default: all cores).",
        "clean": "Remove the build directory before configuring.",
        "reconfigure": "Delete CMake cache to force a fresh configure (keeps compiled objects).",
        "gcc": "Use GCC instead of amdclang.",
        "coverage": "Build with code coverage instrumentation (use `invoke coverage` instead for the full report flow).",
        "rocm_path": "Path to ROCm installation (default: ROCM_PATH env or /opt/rocm).",
    }
)
def build(
    c,
    build_dir=None,
    build_type="Release",
    tests=True,
    no_python=False,
    static=False,
    jobs=None,
    clean=False,
    reconfigure=False,
    gcc=False,
    coverage=False,
    rocm_path=None,
):
    bld = Path(build_dir).resolve() if build_dir else BUILD_DIR

    if clean and bld.exists():
        _rmtree(bld)
    elif reconfigure and bld.exists():
        for name in ("CMakeCache.txt",):
            p = bld / name
            if p.exists():
                p.unlink()

    bld.mkdir(parents=True, exist_ok=True)

    jobs = jobs or os.cpu_count()

    rocm = Path(rocm_path) if rocm_path else _detect_rocm()
    rocm_s = rocm.as_posix()

    cmake_opts = [
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DBUILD_SHARED_LIBS={'OFF' if static else 'ON'}",
        f"-DSTINKYTOFU_BUILD_TESTS={'ON' if tests else 'OFF'}",
        f"-DSTINKYTOFU_BUILD_PYTHON={'OFF' if no_python else 'ON'}",
        # Standalone dev build: build the example plugins (demo + exercised by the
        # unit tests). Default OFF in CMake so integrated/ROCm builds never ship them.
        "-DSTINKYTOFU_BUILD_EXAMPLES=ON",
        "-DSTINKYTOFU_ENABLE_WERROR=ON",
        f"-DSTINKYTOFU_CODE_COVERAGE={'ON' if coverage else 'OFF'}",
    ]

    if not no_python:
        cmake_opts.append(f"-DPython_EXECUTABLE={sys.executable}")

    # Locate ROCmCMakeBuildTools for version TWEAK (git hash) support.
    _rocm_sdk = shutil.which("rocm-sdk")
    if _rocm_sdk:
        try:
            _sdk_root = (
                subprocess.check_output(
                    ["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            if _sdk_root:
                _rocm_cmake_dir = Path(_sdk_root) / "share/rocmcmakebuildtools/cmake"
                if _rocm_cmake_dir.is_dir():
                    cmake_opts.append(
                        f"-DROCmCMakeBuildTools_DIR={_rocm_cmake_dir.as_posix()}"
                    )
        except subprocess.CalledProcessError:
            pass

    compiler_opts = []

    if sys.platform == "win32":
        _setup_msvc_env()

        _vcpkg_toolchain_path = None
        _vcpkg = shutil.which("vcpkg")
        if _vcpkg:
            _vcpkg_root = Path(_vcpkg).resolve().parent
            _vcpkg_tc = _vcpkg_root / "scripts/buildsystems/vcpkg.cmake"
            if _vcpkg_tc.exists():
                _vcpkg_toolchain_path = _vcpkg_tc.as_posix()

        sep = os.pathsep
        rocm_bin_dirs = [f"{rocm_s}/bin", f"{rocm_s}/lib/llvm/bin"]
        os.environ["PATH"] = sep.join(rocm_bin_dirs) + sep + os.environ.get("PATH", "")
        os.environ["ROCM_PATH"] = rocm_s
        os.environ["HIP_PATH"] = rocm_s

        _cxx = shutil.which("amdclang++") or shutil.which("amdclang++.exe")
        _cc = shutil.which("amdclang") or shutil.which("amdclang.exe")
        if not _cxx or not _cc:
            print(
                "Could not find amdclang++/amdclang on PATH. Ensure ROCm SDK is installed."
            )
            sys.exit(1)

        def _normalized(p: str) -> str:
            pp = Path(p)
            return (pp.parent / pp.name.lower()).as_posix()

        compiler_opts += [
            f'"-DCMAKE_CXX_COMPILER={_normalized(_cxx)}"',
            f'"-DCMAKE_C_COMPILER={_normalized(_cc)}"',
        ]

        # rc.exe is now on PATH via _setup_msvc_env(); pass it explicitly as a
        # fallback in case CMake's RC detection still fails.
        rc = shutil.which("rc") or shutil.which("rc.exe")
        if rc:
            compiler_opts.append(f'"-DCMAKE_RC_COMPILER={_normalized(rc)}"')

        # CMake's default ar-style archiver syntax (qc flags) is incompatible
        # with lib.exe. Generate a toolchain file that overrides the archive
        # command to use lib.exe's /OUT: syntax.
        lib_exe = shutil.which("lib") or shutil.which("lib.exe")
        if lib_exe:
            lib_posix = Path(lib_exe).as_posix()
            toolchain_file = bld / "windows_toolchain.cmake"
            tc_content = ""
            if _vcpkg_toolchain_path:
                tc_content += f'include("{_vcpkg_toolchain_path}")\n'
            tc_content += (
                f'set(CMAKE_AR "{lib_posix}" CACHE FILEPATH "" FORCE)\n'
                'set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")\n'
                'set(CMAKE_C_CREATE_STATIC_LIBRARY   "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")\n'
            )
            toolchain_file.write_text(tc_content)
            cmake_opts.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file.as_posix()}")
        elif _vcpkg_toolchain_path:
            cmake_opts.append(f"-DCMAKE_TOOLCHAIN_FILE={_vcpkg_toolchain_path}")
        else:
            print("Warning: lib.exe not found. Static library archiving will fail.")

        # MS Store Python installs to a path with spaces, which breaks Ninja.
        # Fall back to NMake (single-threaded) when the Python executable path
        # contains a space.
        if " " in sys.executable:
            print(
                f"Warning: Python path contains spaces ({sys.executable}). "
                "Falling back to NMake (no parallel builds). "
                "Install Python from python.org for Ninja support."
            )
            cmake_opts.append('-G "NMake Makefiles"')
        else:
            cmake_opts.append("-G Ninja")
    else:
        if shutil.which("ninja"):
            cmake_opts.append("-G Ninja")

        if gcc:
            _cxx = shutil.which("g++") or "g++"
            _cc = shutil.which("gcc") or "gcc"
        else:
            _cxx = shutil.which("amdclang++") or f"{rocm_s}/bin/amdclang++"
            _cc = shutil.which("amdclang") or f"{rocm_s}/bin/amdclang"
        compiler_opts += [
            f"-DCMAKE_CXX_COMPILER={_cxx}",
            f"-DCMAKE_C_COMPILER={_cc}",
        ]

    all_opts = " ".join(cmake_opts + compiler_opts)
    cmake_cmd = f'cmake -B "{bld.as_posix()}" -S "{ROOT_PATH.as_posix()}" {all_opts}'
    print(f"cmake command: {cmake_cmd}")
    c.run(cmake_cmd)
    c.run(f'cmake --build "{bld.as_posix()}" -j {jobs}')


@task
def tidy(c, build_dir=None):
    """Run clang-tidy on all source files. Requires a prior 'invoke build'."""
    bld = Path(build_dir).resolve() if build_dir else BUILD_DIR
    if not (bld / "compile_commands.json").exists():
        print("No compile_commands.json found. Run 'invoke build' first.")
        sys.exit(1)
    c.run(
        f'cmake -B "{bld.as_posix()}" -S "{ROOT_PATH.as_posix()}" -DENABLE_CLANG_TIDY=ON'
    )
    c.run(f'cmake --build "{bld.as_posix()}" --target tidy')


@task(
    help={
        "build_dir": "Coverage build directory (default: build-coverage/).",
        "open_report": "Open the HTML report in a browser when finished.",
        "jobs": "Number of parallel build jobs (default: all cores).",
        "rocm_path": "Path to ROCm installation (default: ROCM_PATH env or /opt/rocm).",
    }
)
def coverage(c, build_dir=None, open_report=False, jobs=None, rocm_path=None):
    """Build instrumented, run the full test suite, and produce coverage reports.

    Uses LLVM source-based coverage (requires an amdclang/Clang build). The whole
    suite is measured: gtest unit tests, the FileCheck tools, and the pytest
    bindings all load instrumented code and drop their own raw profile.

    Outputs, under the coverage build directory:
      coverage-report/index.html  -- browsable HTML report
      coverage.info               -- lcov format (for Codecov / CI upload)
    """
    bld = Path(build_dir).resolve() if build_dir else (ROOT_PATH / "build-coverage")

    # 1. Build with instrumentation. RelWithDebInfo gives -g (all that LLVM
    #    coverage mapping needs) while keeping the *release* CRT (/MD), so it
    #    matches release-built deps (gtest x64-windows, amd_comgr). A Debug build
    #    pulls in the debug CRT (/MDd) and that mismatch makes the debug runtime
    #    abort() inside the tools.
    build(
        c,
        build_dir=str(bld),
        build_type="RelWithDebInfo",
        coverage=True,
        jobs=jobs,
        rocm_path=rocm_path,
    )

    # 2. Locate the LLVM coverage tools (shipped alongside amdclang).
    rocm = Path(rocm_path) if rocm_path else _detect_rocm()
    llvm_bin = rocm / "lib" / "llvm" / "bin"

    def _llvm_tool(name):
        exe = shutil.which(name)
        if exe:
            return exe
        for cand in (llvm_bin / name, llvm_bin / f"{name}.exe"):
            if cand.exists():
                return cand.as_posix()
        raise SystemExit(
            f"ERROR: {name} not found on PATH or in {llvm_bin}. "
            "Code coverage requires an amdclang/Clang (LLVM) toolchain."
        )

    llvm_profdata = _llvm_tool("llvm-profdata")
    llvm_cov = _llvm_tool("llvm-cov")

    # 3. Run the suite, one raw profile per process (%p == pid).
    raw_dir = bld / "coverage-raw"
    if raw_dir.exists():
        _rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    profile_pattern = (raw_dir / "cov-%p.profraw").as_posix()
    with c.cd(bld.as_posix()):
        # warn=True: produce a report even if some tests fail.
        c.run(
            "ctest --output-on-failure",
            env={"LLVM_PROFILE_FILE": profile_pattern},
            warn=True,
        )

    # 4. Merge raw profiles.
    profraws = list(raw_dir.glob("*.profraw"))
    if not profraws:
        raise SystemExit(
            "ERROR: no .profraw files were produced. The build must be "
            "instrumented with amdclang/Clang (GCC --coverage is not handled "
            "by this task)."
        )
    profdata = bld / "coverage.profdata"
    # Windows caps a command line at ~32 KB; with hundreds of .profraw files the
    # inline list overflows CreateProcess (WinError 206). Pass them via an LLVM
    # response file (@file), which llvm-profdata expands into positional args.
    rsp = raw_dir / "profraw.rsp"
    rsp.write_text("\n".join(f'"{p.as_posix()}"' for p in profraws))
    c.run(
        f'"{llvm_profdata}" merge -sparse "@{rsp.as_posix()}" -o "{profdata.as_posix()}"'
    )

    # 5. Collect the instrumented binaries to report on. The library carries the
    #    code we care about; the tools/test binaries add their own coverage.
    obj_names = (
        "libstinkytofu.so",
        "stinkytofu.dll",
        "unit_tests",
        "stinkytofu-opt",
        "stinkytofu-check",
        "test_gen_instructions",
    )
    objects = []
    for name in obj_names:
        objects += [
            p
            for p in bld.rglob(f"{name}*")
            if p.is_file() and p.suffix.lower() in (".exe", ".dll")
        ]
    if not objects:
        raise SystemExit("ERROR: no instrumented binaries found to report on.")
    obj_args = " ".join(f'-object "{o.as_posix()}"' for o in objects)

    # Keep the report focused on library/tool sources, not test or 3rd-party code.
    ignore = '--ignore-filename-regex="(tests|examples|build|_deps|rocisa|/usr/)"'

    # 6. HTML report, lcov export for CI, and a console summary.
    html_dir = bld / "coverage-report"
    common = f'{obj_args} -instr-profile="{profdata.as_posix()}" {ignore}'
    c.run(
        f'"{llvm_cov}" show {common} -format=html '
        f'-output-dir="{html_dir.as_posix()}" -show-line-counts-or-regions'
    )
    lcov_file = bld / "coverage.info"
    c.run(f'"{llvm_cov}" export {common} -format=lcov > "{lcov_file.as_posix()}"')
    c.run(f'"{llvm_cov}" report {common}')

    print(f"\nHTML report: {(html_dir / 'index.html').as_posix()}")
    print(f"lcov file:   {lcov_file.as_posix()}")
    if open_report:
        import webbrowser

        webbrowser.open((html_dir / "index.html").as_uri())
