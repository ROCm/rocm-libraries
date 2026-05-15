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
        "gcc": "Use GCC instead of amdclang.",
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
    gcc=False,
    rocm_path=None,
):
    bld = Path(build_dir).resolve() if build_dir else BUILD_DIR

    if clean and bld.exists():
        _rmtree(bld)

    bld.mkdir(parents=True, exist_ok=True)

    jobs = jobs or os.cpu_count()

    rocm = Path(rocm_path) if rocm_path else _detect_rocm()
    rocm_s = rocm.as_posix()

    cmake_opts = [
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DBUILD_SHARED_LIBS={'OFF' if static else 'ON'}",
        f"-DSTINKYTOFU_BUILD_TESTS={'ON' if tests else 'OFF'}",
        f"-DSTINKYTOFU_BUILD_PYTHON={'OFF' if no_python else 'ON'}",
        "-DSTINKYTOFU_ENABLE_WERROR=ON",
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
