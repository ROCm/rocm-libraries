################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Toolchain.Source``: the pure code-object
filename computation and ``buildSourceCodeObjectFiles`` orchestration (driven
with stub compiler/bundler + the helper cache disabled — no real compilation)."""

from pathlib import Path

import pytest

from Tensile.Toolchain.Source import (
    _computeSourceCodeObjectFilename,
    buildSourceCodeObjectFiles,
    makeSourceToolchain,
    SourceToolchain,
)

pytestmark = pytest.mark.unit


def test_make_source_toolchain():
    tc = makeSourceToolchain("amdclang++", "clang-offload-bundler",
                             asan_build=True, save_temps=True)
    assert isinstance(tc, SourceToolchain)
    assert tc.compiler is not None and tc.bundler is not None


def test_compute_co_filename_fallback(snapshot):
    p = _computeSourceCodeObjectFilename("hipv4-amdgcn-amd-amdhsa--gfx942",
                                         "TensileLibrary_fallback", "/build", "gfx942")
    assert str(p) == snapshot


def test_compute_co_filename_tensilelibrary_variant(snapshot):
    # base contains arch -> baseVariant path.
    p = _computeSourceCodeObjectFilename("hipv4-amdgcn-amd-amdhsa--gfx942:xnack-",
                                         "TensileLibrary_gfx942", "/build", "gfx942")
    assert str(p) == snapshot


def test_compute_co_filename_other(snapshot):
    p = _computeSourceCodeObjectFilename("hipv4-amdgcn-amd-amdhsa--gfx942",
                                         "Kernels", "/build", "gfx942")
    assert str(p) == snapshot


class _StubCompiler:
    def __init__(self):
        self.calls = []
        # buildSourceCodeObjectFiles computes a cache key from the compiler's
        # version/rocm_version/default_args; provide them (cache is disabled).
        import types
        v = types.SimpleNamespace(major=6, minor=2, patch=0)
        self.version = v
        self.rocm_version = v
        self.default_args = []

    def __call__(self, includeDir, archs, kernelPath, objPath):
        self.calls.append((includeDir, tuple(archs), kernelPath, objPath))
        Path(objPath).write_text("obj")


class _StubBundler:
    def targets(self, objPath):
        return ["host-x86_64-unknown-linux", "hipv4-amdgcn-amd-amdhsa--gfx942"]

    def __call__(self, target, objPath, coPathRaw):
        Path(coPathRaw).write_text("co.raw")


def test_build_source_code_object_files(tmp_path, monkeypatch, snapshot):
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")  # force cache miss/no-op
    dest = tmp_path / "dest"
    tmpObj = tmp_path / "obj"
    inc = tmp_path / "inc"
    inc.mkdir()
    kernel = tmp_path / "Kernels.cpp"
    kernel.write_text("// src")

    out = buildSourceCodeObjectFiles(
        _StubCompiler(), _StubBundler(), dest, tmpObj, inc, kernel, ["gfx942"]
    )
    assert {"num_co": len(out), "co_names": sorted(Path(p).name for p in out)} == snapshot
