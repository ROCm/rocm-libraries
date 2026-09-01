# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path


TENSILELITE_ROOT = Path(__file__).resolve().parents[3]
_CODEGEN_MODULE_CANDIDATES = (
    # Source tree: projects/hipblaslt/tensilelite -> projects/hipblaslt/cmake.
    TENSILELITE_ROOT.parent / "cmake" / "hipblaslt_codegen.cmake",
    # Installed test artifact: <prefix>/share/hipblaslt/tensilelite -> <prefix>/lib*/cmake.
    TENSILELITE_ROOT.parents[2] / "lib" / "cmake" / "hipblaslt" / "hipblaslt_codegen.cmake",
    TENSILELITE_ROOT.parents[2] / "lib64" / "cmake" / "hipblaslt" / "hipblaslt_codegen.cmake",
)
CODEGEN_MODULE = next(
    (candidate for candidate in _CODEGEN_MODULE_CANDIDATES if candidate.is_file()),
    _CODEGEN_MODULE_CANDIDATES[0],
)


def _configure(
    tmp_path: Path, body: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    source_dir = tmp_path / "source"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(codegen_interface_test LANGUAGES NONE)\n"
        f'include("{CODEGEN_MODULE.as_posix()}")\n'
        f"{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        env=env,
        text=True,
    )


def test_include_defines_generic_function_without_cache_entries(tmp_path):
    cache_variables = (
        "TENSILELITE_BUILD_PARALLEL_LEVEL",
        "TENSILELITE_KEEP_BUILD_TMP",
        "TENSILELITE_ASM_DEBUG",
        "TENSILELITE_LOGIC_FILTER",
        "TENSILELITE_NO_COMPRESS",
        "TENSILELITE_EXPERIMENTAL",
        "TENSILELITE_ENABLE_ASM_COMMENTS",
        "TENSILELITE_OFFLOADBUNDLER",
        "TENSILELITE_LIBLOGIC_PATH",
        "TENSILELITE_LIBRARY_FORMAT",
        "Tensile_NO_LAZY_LIBRARY_LOADING",
    )
    checks = "\n".join(
        f'if(DEFINED CACHE{{{variable}}})\n'
        f'  message(FATAL_ERROR "unexpected cache entry: {variable}")\n'
        "endif()"
        for variable in cache_variables
    )
    result = _configure(
        tmp_path,
        "if(NOT COMMAND create_device_library)\n"
        '  message(FATAL_ERROR "create_device_library is not defined")\n'
        "endif()\n"
        f"{checks}",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_create_device_library_accepts_explicit_source_root(tmp_path):
    result = _configure(
        tmp_path,
        "create_device_library(\n"
        f'  CODEGEN_ROOT "{TENSILELITE_ROOT.as_posix()}"\n'
        f'  LOGIC_PATH "{(TENSILELITE_ROOT / "Tensile" / "Tests" / "unit").as_posix()}"\n'
        f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"\n'
        f'  PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}"\n'
        '  CXX_COMPILER "/usr/bin/c++"\n'
        "  ARCHES gfx950\n"
        ")",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_create_device_library_rejects_incomplete_source_root(tmp_path):
    result = _configure(
        tmp_path,
        "create_device_library(\n"
        f'  CODEGEN_ROOT "{tmp_path.as_posix()}"\n'
        f'  LOGIC_PATH "{tmp_path.as_posix()}"\n'
        f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"\n'
        f'  PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}"\n'
        '  CXX_COMPILER "/usr/bin/c++"\n'
        "  ARCHES gfx950\n"
        ")",
    )
    assert result.returncode != 0
    assert "required codegen resource not found" in result.stdout + result.stderr


def test_create_device_library_preserves_runtime_library_path(tmp_path):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/existing/runtime/path"
    result = _configure(
        tmp_path,
        "create_device_library(\n"
        f'  CODEGEN_ROOT "{TENSILELITE_ROOT.as_posix()}"\n'
        f'  LOGIC_PATH "{(TENSILELITE_ROOT / "Tensile" / "Tests" / "unit").as_posix()}"\n'
        f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"\n'
        f'  PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}"\n'
        '  CXX_COMPILER "/usr/bin/c++"\n'
        "  ARCHES gfx950\n"
        ")",
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "/existing/runtime/path" in (tmp_path / "build" / "build.ninja").read_text()


def test_create_device_library_applies_host_asan_without_device_asan(tmp_path):
    sanitizer_runtime = tmp_path / "libclang_rt.asan-x86_64.so"
    sanitizer_runtime.touch()
    compiler = tmp_path / "clang++"
    compiler.write_text(
        f"#!/bin/sh\necho '\"{sanitizer_runtime.as_posix()}\"' >&2\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)

    result = _configure(
        tmp_path,
        f'set(CMAKE_CXX_COMPILER "{compiler.as_posix()}")\n'
        "create_device_library(\n"
        f'  CODEGEN_ROOT "{TENSILELITE_ROOT.as_posix()}"\n'
        f'  LOGIC_PATH "{(TENSILELITE_ROOT / "Tensile" / "Tests" / "unit").as_posix()}"\n'
        f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"\n'
        f'  PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}"\n'
        f'  CXX_COMPILER "{compiler.as_posix()}"\n'
        "  ARCHES gfx950\n"
        "  HOST_ASAN\n"
        "  ASAN OFF\n"
        ")",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    build_ninja = (tmp_path / "build" / "build.ninja").read_text()
    assert f"LD_PRELOAD={sanitizer_runtime.as_posix()}" in build_ninja, (
        result.stdout + result.stderr + build_ninja
    )
    assert "ASAN_OPTIONS=" in build_ninja
    assert "--address-sanitizer" not in build_ninja
