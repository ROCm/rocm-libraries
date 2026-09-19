# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU regressions for runtime discovery and standalone entry points."""

import ctypes
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from dispatcher_common import hip_library_candidates, load_hip_runtime


def _loader_accepting(monkeypatch, accepted):
    """Model a loader with exactly one usable HIP library."""
    handle = object()
    attempted = []

    def cdll(name):
        attempted.append(name)
        if name == accepted:
            return handle
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    return handle, attempted


@pytest.mark.parametrize("major", [10, 42])
def test_registered_hip_soname_without_development_symlink(monkeypatch, tmp_path, major):
    soname = f"libamdhip64.so.{major}"
    lookups = []

    def find_library(name):
        lookups.append(name)
        return soname

    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "absent"))
    handle, attempted = _loader_accepting(monkeypatch, soname)
    assert load_hip_runtime() is handle
    assert lookups == ["amdhip64"]
    assert attempted == ["libamdhip64.so", soname]


@pytest.mark.parametrize("subdir", ["lib", "lib64"])
@pytest.mark.parametrize("lookup_fails", [False, True])
def test_unregistered_versioned_hip_file_without_symlinks(monkeypatch, tmp_path, subdir, lookup_fails):
    libdir = tmp_path / subdir
    libdir.mkdir()
    # Deliberately omit both .so and .so.10, as in a stripped runtime install.
    library = libdir / "libamdhip64.so.10.0.26306"
    library.touch()

    def find_library(name):
        if lookup_fails:
            raise OSError("system library lookup is unavailable")
        return None

    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    handle, attempted = _loader_accepting(monkeypatch, str(library))
    assert load_hip_runtime() is handle
    assert str(library) in attempted


def test_versioned_hip_discovery_orders_versions_and_ignores_non_libraries(monkeypatch, tmp_path):
    libdir = tmp_path / "lib"
    libdir.mkdir()
    for name in ("libamdhip64.so.9", "libamdhip64.so.10", "libamdhip64.so.10.1",
                 "libamdhip64.so.10.debug", "libamdhip64.so.old"):
        (libdir / name).touch()
    (libdir / "libamdhip64.so.99").mkdir()
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libamdhip64.so.7")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    candidates = hip_library_candidates()
    assert len(candidates) == len(set(candidates))
    discovered = [Path(p).name for p in candidates if str(libdir) in p and Path(p).is_file()]
    assert discovered == ["libamdhip64.so.10.1", "libamdhip64.so.10", "libamdhip64.so.9"]
    assert not any(p.endswith((".debug", ".old", ".99")) for p in candidates)


def test_failed_hip_load_reports_discovered_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libamdhip64.so.10")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    _loader_accepting(monkeypatch, None)
    with pytest.raises(OSError, match=r"Tried:.*libamdhip64\.so\.10"):
        load_hip_runtime()


@pytest.mark.parametrize("arch,expected", [
    ("gfx942", (32, 32, 16)),
    ("gfx1250", (16, 16, 32)),
    ("gfx1250:xnack-", (16, 16, 32)),
])
def test_multi_abd_warp_tile_in_fresh_standalone_interpreter(arch, expected, tmp_path):
    # -I prevents pytest's other modules and PYTHONPATH from hiding missing paths.
    # Suppress only hardware detection; exercise the real module and tile lookup.
    script = '''import runpy, sys
from types import SimpleNamespace
from unittest.mock import patch
with patch("subprocess.run", return_value=SimpleNamespace(stdout="")):
    namespace = runpy.run_path(sys.argv[1])
print(namespace["TestMultiAbdGemmGpu"]._fp16_warp_tile(sys.argv[2]))
'''
    result = subprocess.run([
        sys.executable, "-I", "-c", script,
        str(ROOT / "tests/test_multi_abd_gpu_correctness.py"), arch,
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(expected)


@pytest.mark.parametrize("arch,tile", [
    ("gfx942:sramecc+:xnack-", "32x32x8"),
    ("gfx950:xnack-", "32x32x8"),
    ("gfx1250:xnack-", "16x16x32"),
])
def test_cmake_fallback_generates_kernel_for_suffixed_target(tmp_path, arch, tile):
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("requires CMake and a host C++ compiler")
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.16)
project(fallback_codegen_probe LANGUAGES CXX)
set(GPU_TARGETS "{arch}")
add_library(ck_tile_dispatcher INTERFACE)
add_subdirectory("{ROOT / 'examples'}" examples EXCLUDE_FROM_ALL)
''')
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(["cmake", "-S", str(source), "-B", str(build)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    # Build only the real generation target: this invokes no HIP compiler/GPU.
    result = subprocess.run([
        "cmake", "--build", str(build), "--target", "generate_gemm_fallback_kernel",
    ], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    headers = list((build / "examples/gemm_python_fallback").glob("gemm_*.hpp"))
    assert len(headers) == 1
    assert headers[0].stem.endswith(tile)
