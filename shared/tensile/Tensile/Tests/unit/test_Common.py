################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from __future__ import print_function

import shutil
import tempfile
from pathlib import Path

import pytest

import Tensile.Common as Common

import os

def test_gfxArch():
    assert Common.gfxArch('gfx9') is None

    assert Common.gfxArch('gfx803') == (8,0,3)
    assert Common.gfxArch('gfx900') == (9,0,0)
    assert Common.gfxArch('gfx906') == (9,0,6)

    assert Common.gfxArch('gfx1010') == (10,1,0)

    assert Common.gfxArch('gfx90015') == (900,1,5)

    assert Common.gfxArch('blah gfx900 stuff') == (9,0,0)

def test_isGfx12():
    assert Common.isGfx12((12, 0, 0))
    assert Common.isGfx12((12, 0, 1))
    assert Common.isGfx12((12, 5, 0))
    assert not Common.isGfx12((11, 0, 0))
    assert not Common.isGfx12((13, 0, 0))

def test_paths():
    workingPathName = os.path.join("working", "path")
    Common.globalParameters["WorkingPath"] = workingPathName
    expectedWorkingPath = os.path.join("working", "path")
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    recursiveWorkingPath = "next1"
    expectedRecurrsiveWorkingPath = os.path.join("working", "path", "next1")
    Common.pushWorkingPath (recursiveWorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedRecurrsiveWorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath

    set1WorkingPath = os.path.join("working", "path", "set1")
    expectedSet1WorkingPath = os.path.join("working", "path", "set1")
    Common.setWorkingPath (set1WorkingPath)
    assert Common.globalParameters["WorkingPath"] == expectedSet1WorkingPath
    Common.popWorkingPath()
    assert Common.globalParameters["WorkingPath"] == expectedWorkingPath


@pytest.mark.parametrize(
    "exe_depth, version_str, expected_hip_clang",
    [
        pytest.param(
            1,
            "#define HIP_VERSION_MAJOR 10\n#define HIP_VERSION_MINOR 1\n#define HIP_VERSION_PATCH 0\n",
            "10.1.0",
            id="dist_bin_layout",
        ),
        pytest.param(
            3,
            "#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR 2\n#define HIP_VERSION_PATCH 53211\n",
            "7.2.53211",
            id="dist_lib_llvm_bin_layout",
        ),
    ],
)
def test_common_path_fallback_hip_version_h(
    monkeypatch, tmp_path, exe_depth, version_str, expected_hip_clang
):
    """Common.py PATH fallback parses hip_version.h when env vars are absent.

    The version detection in assignGlobalParameters mirrors the logic in
    Tensile.Toolchain.Component.get_rocm_version: when ROCM_VERSION,
    ROCM_PATH, and HIP_PATH are all unset it walks up from any amdclang++
    found on PATH, trying .info/version then include/hip/hip_version.h at
    each ancestor level. Two representative TheRock layouts are exercised:
    dist/bin/ (1 level up) and dist/lib/llvm/bin/ (3 levels up).

    Rather than calling the full assignGlobalParameters (which requires a
    complete config), we exercise the version detection section directly by
    patching Path.read_text to reject all .info/version lookups and redirect
    the hip_version.h lookup to our controlled tmpdir file, then verify that
    globalParameters["HipClangVersion"] is set correctly before the function
    proceeds to unrelated config validation.
    """
    # Build a fake executable nested exe_depth directories under tmp_path.
    parts = ["sub"] * exe_depth + ["bin"]
    bin_dir = tmp_path
    for p in parts:
        bin_dir = bin_dir / p
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "amdclang++"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)

    # Place hip_version.h at the root of tmp_path (the "dist" level).
    hip_dir = tmp_path / "include" / "hip"
    hip_dir.mkdir(parents=True)
    hip_version_h_path = hip_dir / "hip_version.h"
    hip_version_h_path.write_text(version_str)

    monkeypatch.delenv("ROCM_VERSION", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: str(exe) if name == "amdclang++" else None)

    # Patch Path.read_text: reject all .info/version reads so the code falls
    # through to the PATH-based fallback, but allow the real hip_version.h read.
    original_read_text = Path.read_text

    def selective_read_text(self, **kwargs):
        if self.name == "version" and self.parent.name == ".info":
            raise OSError(f"mocked absence: {self}")
        return original_read_text(self, **kwargs)

    monkeypatch.setattr(Path, "read_text", selective_read_text)

    # assignGlobalParameters does more than version detection; capture the
    # HipClangVersion set by the version section before the rest fails.
    try:
        Common.assignGlobalParameters({})
    except (ValueError, KeyError):
        pass  # Expected — subsequent config steps need more parameters.

    assert Common.globalParameters["HipClangVersion"] == expected_hip_clang
