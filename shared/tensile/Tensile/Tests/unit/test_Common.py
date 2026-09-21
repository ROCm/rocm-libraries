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

import os
import shutil

import pytest
import Tensile.Common as Common


def _write_hip_version(root, relative_path, version):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    major, minor, patch = version
    if path.name == "version" and path.parent.name == "hip":
        path.write_text(
            f"HIP_VERSION_MAJOR={major}\n"
            f"HIP_VERSION_MINOR={minor}\n"
            f"HIP_VERSION_PATCH={patch}\n"
        )
    else:
        path.write_text(
            f"#define HIP_VERSION_MAJOR {major}\n"
            f"#define HIP_VERSION_MINOR {minor}\n"
            f"#define HIP_VERSION_PATCH {patch}\n"
        )


@pytest.fixture
def isolated_version_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCM_VERSION", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.setattr(Common, "_DEFAULT_ROCM_ROOT", tmp_path / "missing-default")
    monkeypatch.setattr(Common.shutil, "which", lambda _name: None)

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
    monkeypatch, tmp_path, isolated_version_sources, exe_depth, version_str, expected_hip_clang
):
    """Common.py PATH fallback parses hip_version.h when env vars are absent.

    The version detection in assignGlobalParameters mirrors the logic in
    Tensile.Toolchain.Component.get_rocm_version: when ROCM_VERSION,
    ROCM_PATH, and HIP_PATH are all unset it walks up from any amdclang++
    found on PATH, trying HIP build metadata before .info/version at each
    ancestor level. Two representative TheRock layouts are exercised:
    dist/bin/ (1 level up) and dist/lib/llvm/bin/ (3 levels up).

    The full assignment requires additional configuration, so this test checks
    the version stored before later validation stops the call.
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

    monkeypatch.setattr(shutil, "which", lambda name: str(exe) if name == "amdclang++" else None)

    # assignGlobalParameters does more than version detection; capture the
    # HipClangVersion set by the version section before the rest fails.
    try:
        Common.assignGlobalParameters({})
    except (ValueError, KeyError):
        pass  # Expected — subsequent config steps need more parameters.

    assert Common.globalParameters["HipClangVersion"] == expected_hip_clang


def test_get_hip_version_prefers_env(monkeypatch, tmp_path, isolated_version_sources):
    info_dir = tmp_path / ".info"
    info_dir.mkdir()
    (info_dir / "version").write_text("7.1.0")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setenv("ROCM_VERSION", "7.1.25424-4179531dcd")
    assert Common._getHipVersion() == "7.1.25424"


@pytest.mark.parametrize(
    "version_string, expected",
    [("10.1.0a20260813", "10.1.0"), ("7.2.1rc2", "7.2.1")],
)
def test_get_hip_version_parses_prerelease_suffix(monkeypatch, version_string, expected):
    monkeypatch.setenv("ROCM_VERSION", version_string)
    assert Common._getHipVersion() == expected


@pytest.mark.parametrize(
    "relative_path",
    ["share/hip/version", "include/hip/hip_version.h"],
)
def test_get_hip_version_prefers_build_metadata(
    monkeypatch, tmp_path, isolated_version_sources, relative_path
):
    _write_hip_version(tmp_path, relative_path, (6, 4, 43482))
    info_dir = tmp_path / ".info"
    info_dir.mkdir()
    (info_dir / "version").write_text("6.4.3")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert Common._getHipVersion() == "6.4.43482"


def test_get_hip_version_prefers_share_metadata(
    monkeypatch, tmp_path, isolated_version_sources
):
    _write_hip_version(tmp_path, "share/hip/version", (7, 1, 25424))
    _write_hip_version(tmp_path, "include/hip/hip_version.h", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert Common._getHipVersion() == "7.1.25424"


def test_get_hip_version_prefers_rocm_path(
    monkeypatch, tmp_path, isolated_version_sources
):
    rocm_root = tmp_path / "rocm"
    hip_root = tmp_path / "hip"
    _write_hip_version(rocm_root, "share/hip/version", (7, 1, 25424))
    _write_hip_version(hip_root, "share/hip/version", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setenv("HIP_PATH", str(hip_root))
    assert Common._getHipVersion() == "7.1.25424"


def test_get_hip_version_falls_back_to_hip_path(
    monkeypatch, tmp_path, isolated_version_sources
):
    hip_root = tmp_path / "hip"
    _write_hip_version(hip_root, "share/hip/version", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("HIP_PATH", str(hip_root))
    assert Common._getHipVersion() == "7.2.53211"


def test_get_hip_version_rejects_malformed_explicit_version(monkeypatch):
    monkeypatch.setenv("ROCM_VERSION", "development")
    with pytest.raises(ValueError, match="Invalid HIP version"):
        Common._getHipVersion()


def test_get_hip_version_rejects_malformed_metadata(
    monkeypatch, tmp_path, isolated_version_sources
):
    version_file = tmp_path / "share" / "hip" / "version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("HIP_VERSION_MAJOR=7\nHIP_VERSION_MINOR=2\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid HIP version file"):
        Common._getHipVersion()


def test_get_hip_version_raises_without_a_source(isolated_version_sources):
    with pytest.raises(ValueError, match="Failed to get ROCm version"):
        Common._getHipVersion()
