"""Configure-time compiler, hip-path and device-lib-path resolution.

Runs the resolution logic from hkp_require_ingestor_toolchain() under
``cmake -P``, following the same pattern as test_hkp_selected_arches.py.
Because the function uses PARENT_SCOPE, a thin wrapper captures the
variables and prints them via message().
"""

import os
import subprocess
from pathlib import Path

import pytest

_CMAKE_DIR = Path(__file__).resolve().parents[1] / "cmake"
_PROVIDER_CMAKE_DIR = _CMAKE_DIR.parents[1] / "cmake"

pytestmark = pytest.mark.quick


def _resolve(cmake, tmp_path, **settings):
    """Run the compiler resolution logic and return (compiler, hip_path, device_lib_path).

    Passes *settings* as CMake set() calls before invoking the function.
    A FATAL_ERROR returns (None, None, None) and the stderr text.
    """
    sets = ""
    for key, value in settings.items():
        if value is not None:
            sets += f'set({key} "{value}")\n'

    # hkp_require_ingestor_toolchain needs GPU_TARGETS and uses
    # hkp_selected_arches internally; provide a valid arch to avoid
    # the "no concrete gfx" fatal error when we're testing compiler resolution.
    if "GPU_TARGETS" not in settings:
        sets += 'set(GPU_TARGETS "gfx942")\n'

    script = tmp_path / "resolve.cmake"
    script.write_text(
        f'list(APPEND CMAKE_MODULE_PATH "{_CMAKE_DIR.as_posix()}" '
        f'"{_PROVIDER_CMAKE_DIR.as_posix()}")\n'
        "include(HkpPackaging)\n"
        f"{sets}"
        # Wrap in a function so PARENT_SCOPE lands in _wrapper's scope.
        "function(_wrapper)\n"
        "    hkp_require_ingestor_toolchain(_arches)\n"
        '    message(STATUS "HKP_HIP_COMPILER=[${HKP_HIP_COMPILER}]")\n'
        '    message(STATUS "HKP_HIP_PATH=[${HKP_HIP_PATH}]")\n'
        '    message(STATUS "HKP_DEVICE_LIB_PATH=[${HKP_DEVICE_LIB_PATH}]")\n'
        "endfunction()\n"
        "_wrapper()\n"
    )
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    out = result.stdout + result.stderr
    if result.returncode != 0:
        return None, None, None, out

    def _extract(tag):
        for line in out.splitlines():
            if f"{tag}=[" in line:
                return line.split(f"{tag}=[")[1].rstrip("]")
        return ""

    return (
        _extract("HKP_HIP_COMPILER"),
        _extract("HKP_HIP_PATH"),
        _extract("HKP_DEVICE_LIB_PATH"),
        out,
    )


class TestCompilerResolution:
    def test_cmake_hip_compiler_preferred(self, cmake, tmp_path):
        compiler, _, _, _ = _resolve(
            cmake,
            tmp_path,
            CMAKE_HIP_COMPILER="/usr/bin/amdclang++",
            CMAKE_CXX_COMPILER="/usr/bin/g++",
        )
        assert compiler == "/usr/bin/amdclang++"

    def test_falls_back_to_cxx_compiler(self, cmake, tmp_path):
        compiler, _, _, _ = _resolve(
            cmake, tmp_path, CMAKE_CXX_COMPILER="/usr/bin/clang++"
        )
        assert compiler == "/usr/bin/clang++"

    def test_neither_compiler_is_fatal(self, cmake, tmp_path):
        compiler, _, _, out = _resolve(cmake, tmp_path)
        assert compiler is None
        assert "requires a HIP compiler" in out


class TestHipPathDerivation:
    def test_hip_dir_derives_hip_path(self, cmake, tmp_path):
        # hip_DIR is typically <prefix>/lib/cmake/hip; hip_path should be <prefix>.
        fake_prefix = tmp_path / "opt" / "rocm"
        hip_dir = fake_prefix / "lib" / "cmake" / "hip"
        hip_dir.mkdir(parents=True)
        _, hip_path, _, _ = _resolve(
            cmake,
            tmp_path,
            CMAKE_CXX_COMPILER="/usr/bin/clang++",
            hip_DIR=str(hip_dir),
        )
        # CMake normalises paths to forward slashes; compare via Path so
        # the assertion holds on Windows where str(Path) uses backslashes.
        assert Path(hip_path) == fake_prefix

    def test_no_hip_dir_gives_empty_path(self, cmake, tmp_path):
        _, hip_path, _, _ = _resolve(
            cmake, tmp_path, CMAKE_CXX_COMPILER="/usr/bin/clang++"
        )
        assert hip_path == ""


class TestDeviceLibPathDerivation:
    def test_sibling_bitcode_dir_found(self, cmake, tmp_path):
        # Simulate compiler at <prefix>/bin/clang++ with amdgcn/bitcode sibling.
        compiler = tmp_path / "prefix" / "bin" / "clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\n")
        bitcode = tmp_path / "prefix" / "amdgcn" / "bitcode"
        bitcode.mkdir(parents=True)
        _, _, device_lib_path, _ = _resolve(
            cmake, tmp_path, CMAKE_CXX_COMPILER=str(compiler)
        )
        assert Path(device_lib_path) == bitcode

    def test_no_sibling_gives_empty_path(self, cmake, tmp_path):
        compiler = tmp_path / "prefix" / "bin" / "clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\n")
        _, _, device_lib_path, _ = _resolve(
            cmake, tmp_path, CMAKE_CXX_COMPILER=str(compiler)
        )
        assert device_lib_path == ""
