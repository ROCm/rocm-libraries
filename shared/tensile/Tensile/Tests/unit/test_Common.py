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

import pytest

import Tensile.Common as Common

import os

import pytest


@pytest.fixture
def clearStrictEnv():
    """Ensure the strict compiler-target env var is unset around each test."""
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)
    yield
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)


def test_architectureMap_strict_maps_to_gfx1250():
    assert Common.architectureMap["gfx1250-strict"] == "gfx1250"
    assert Common.getArchitectureName("gfx1250-strict") == "gfx1250"


def test_configureCompilerTarget_setsEnvForStrict(clearStrictEnv):
    Common.configureCompilerTarget("gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"


def test_configureCompilerTarget_clearsEnvForNonStrict(clearStrictEnv):
    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    Common.configureCompilerTarget("gfx1250")
    assert "TENSILE_GFX1250_COMPILER_TARGET" not in Common.os.environ


def test_configureCompilerTarget_acceptsDelimiterVariants(clearStrictEnv):
    # CMake uses `_` delimiters, the CLI uses `;`.
    Common.configureCompilerTarget("gfx942_gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"
    Common.os.environ.pop("TENSILE_GFX1250_COMPILER_TARGET", None)
    Common.configureCompilerTarget("gfx942;gfx1250-strict")
    assert Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] == "gfx1250-strict"


@pytest.mark.parametrize("mixed", ["gfx1250;gfx1250-strict", "all;gfx1250-strict"])
def test_configureCompilerTarget_rejectsMixingWithGfx1250(clearStrictEnv, mixed):
    with pytest.raises(ValueError):
        Common.configureCompilerTarget(mixed)


def test_compilerTarget_rewritesGfx1250OnlyWhenStrict(clearStrictEnv):
    assert Common.compilerTarget("gfx1250") == "gfx1250"
    assert Common.compilerTarget("gfx942") == "gfx942"

    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    assert Common.compilerTarget("gfx1250") == "gfx1250-strict"
    # Only gfx1250 is rewritten; other targets are untouched.
    assert Common.compilerTarget("gfx942") == "gfx942"


def test_gfxName_appliesStrictCompilerTarget(clearStrictEnv):
    assert Common.gfxName((12, 5, 0)) == "gfx1250"

    Common.os.environ["TENSILE_GFX1250_COMPILER_TARGET"] = "gfx1250-strict"
    assert Common.gfxName((12, 5, 0)) == "gfx1250-strict"
    # A different ISA is not affected by the strict env var.
    assert Common.gfxName((9, 4, 2)) == "gfx942"


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

    _getHipVersion() checks ROCM_VERSION, ROCM_PATH, HIP_PATH, and the
    default /opt/rocm root before falling back to a PATH walk. This test
    exercises the PATH walk by unsetting all env vars and redirecting the
    default root to a non-existent path so the prefix loop falls through.

    The walk prefers share/hip/version then include/hip/hip_version.h over
    .info/version at each ancestor level. Two representative TheRock layouts
    are exercised: dist/bin/ (1 level up) and dist/lib/llvm/bin/ (3 levels up).
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
    # Redirect the default /opt/rocm root so the prefix loop falls through
    # to the PATH walk without reading from the real system installation.
    monkeypatch.setattr(Common, "_DEFAULT_ROCM_ROOT", tmp_path / "nonexistent")
    monkeypatch.setattr(shutil, "which", lambda name: str(exe) if name == "amdclang++" else None)

    # assignGlobalParameters does more than version detection; capture the
    # HipClangVersion set by the version section before the rest fails.
    try:
        Common.assignGlobalParameters({})
    except (ValueError, KeyError):
        pass  # Expected — subsequent config steps need more parameters.

    assert Common.globalParameters["HipClangVersion"] == expected_hip_clang
