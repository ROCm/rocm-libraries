# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import shutil

import pytest

import Tensile.Toolchain.Component as Component
from Tensile.Common import SemanticVersion
from Tensile.CustomKernels import supportsUserSgprKernargPreload


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
    monkeypatch.setattr(
        Component, "_DEFAULT_ROCM_ROOT", tmp_path / "missing-default", raising=False
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)


def test_explicit_version_takes_priority(monkeypatch, tmp_path):
    info_dir = tmp_path / ".info"
    info_dir.mkdir()
    (info_dir / "version").write_text("7.1.0")
    monkeypatch.setenv("ROCM_VERSION", "6.4.43482")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert Component.get_rocm_version() == SemanticVersion(6, 4, 43482)


@pytest.mark.parametrize(
    "version_str, expected",
    [
        ("10.1.0a20260813", SemanticVersion(10, 1, 0)),
        ("7.2.1rc2", SemanticVersion(7, 2, 1)),
    ],
)
def test_prerelease_suffix_is_removed(monkeypatch, version_str, expected):
    monkeypatch.setenv("ROCM_VERSION", version_str)
    assert Component.get_rocm_version() == expected


@pytest.mark.parametrize(
    "relative_path",
    ["share/hip/version", "include/hip/hip_version.h"],
)
def test_hip_build_metadata_precedes_rocm_release(
    monkeypatch, tmp_path, isolated_version_sources, relative_path
):
    _write_hip_version(tmp_path, relative_path, (6, 4, 43482))
    info_dir = tmp_path / ".info"
    info_dir.mkdir()
    (info_dir / "version").write_text("6.4.3")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))

    version = Component.get_rocm_version()
    assert version == SemanticVersion(6, 4, 43482)
    assert supportsUserSgprKernargPreload(version)


def test_share_metadata_precedes_header(monkeypatch, tmp_path, isolated_version_sources):
    _write_hip_version(tmp_path, "share/hip/version", (7, 1, 25424))
    _write_hip_version(tmp_path, "include/hip/hip_version.h", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert Component.get_rocm_version() == SemanticVersion(7, 1, 25424)


def test_rocm_path_precedes_hip_path(monkeypatch, tmp_path, isolated_version_sources):
    rocm_root = tmp_path / "rocm"
    hip_root = tmp_path / "hip"
    _write_hip_version(rocm_root, "share/hip/version", (7, 1, 25424))
    _write_hip_version(hip_root, "share/hip/version", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setenv("HIP_PATH", str(hip_root))
    assert Component.get_rocm_version() == SemanticVersion(7, 1, 25424)


def test_missing_rocm_path_falls_back_to_hip_path(monkeypatch, tmp_path, isolated_version_sources):
    hip_root = tmp_path / "hip"
    _write_hip_version(hip_root, "share/hip/version", (7, 2, 53211))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("HIP_PATH", str(hip_root))
    assert Component.get_rocm_version() == SemanticVersion(7, 2, 53211)


def test_malformed_explicit_version_is_rejected(monkeypatch):
    monkeypatch.setenv("ROCM_VERSION", "development")
    with pytest.raises(RuntimeError, match="Invalid HIP version"):
        Component.get_rocm_version()


def test_malformed_hip_metadata_is_rejected(monkeypatch, tmp_path, isolated_version_sources):
    version_file = tmp_path / "share" / "hip" / "version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("HIP_VERSION_MAJOR=7\nHIP_VERSION_MINOR=2\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="Invalid HIP version file"):
        Component.get_rocm_version()


def test_missing_sources_are_reported(isolated_version_sources):
    with pytest.raises(RuntimeError, match="Failed to get ROCm version"):
        Component.get_rocm_version()


@pytest.mark.parametrize("executable", ["/amdclang++", "/a/b/c/d/e/bin/amdclang++"])
def test_path_walk_stops(monkeypatch, isolated_version_sources, executable):
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: executable if name == "amdclang++" else None,
    )
    with pytest.raises(RuntimeError, match="Failed to get ROCm version"):
        Component.get_rocm_version()
