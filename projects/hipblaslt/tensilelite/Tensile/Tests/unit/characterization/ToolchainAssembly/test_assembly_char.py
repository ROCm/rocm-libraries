################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Toolchain.Assembly.buildAssemblyCodeObjectFiles``
— the code-object orchestration, driven with stub linker/bundler (no real
subprocess) and fake kernel dicts."""

from pathlib import Path

import pytest

from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles

pytestmark = pytest.mark.unit


class _StubLinker:
    def __init__(self):
        self.calls = []

    def __call__(self, objFiles, coFileRaw):
        self.calls.append((list(objFiles), coFileRaw))
        Path(coFileRaw).write_text("raw")  # create the raw file for the move path


class _StubBundler:
    def __init__(self):
        self.calls = []

    def compress(self, raw, co, gfx):
        self.calls.append((raw, co, gfx))
        Path(co).write_text("co")


def _kernel(base, isa=(9, 4, 2), coFile=None):
    k = {"ISA": list(isa), "BaseName": base}
    if coFile is not None:
        k["codeObjectFile"] = coFile
    return k


def test_build_compress_true(tmp_path, snapshot):
    asmDir, destDir = tmp_path / "asm", tmp_path / "dest"
    asmDir.mkdir(); destDir.mkdir()
    linker, bundler = _StubLinker(), _StubBundler()
    kernels = [_kernel("k0"), _kernel("k1"), _kernel("kc", coFile="CustomCO")]
    out = buildAssemblyCodeObjectFiles(linker, bundler, kernels, destDir, asmDir, compress=True)
    summary = {
        "co_names": sorted(p.name for p in out),
        "linker_call_count": len(linker.calls),
        "bundler_call_count": len(bundler.calls),
    }
    assert summary == snapshot


def test_build_compress_false_moves(tmp_path, snapshot):
    asmDir, destDir = tmp_path / "asm", tmp_path / "dest"
    asmDir.mkdir(); destDir.mkdir()
    linker, bundler = _StubLinker(), _StubBundler()
    kernels = [_kernel("k0")]
    out = buildAssemblyCodeObjectFiles(linker, bundler, kernels, destDir, asmDir, compress=False)
    # compress=False -> shutil.move of the raw file (no bundler call).
    assert {
        "co_names": sorted(p.name for p in out),
        "bundler_called": len(bundler.calls) > 0,
    } == snapshot


def test_build_empty_kernels(tmp_path, snapshot):
    asmDir, destDir = tmp_path / "asm", tmp_path / "dest"
    asmDir.mkdir(); destDir.mkdir()
    out = buildAssemblyCodeObjectFiles(_StubLinker(), _StubBundler(), [], destDir, asmDir)
    assert out == snapshot
