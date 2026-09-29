# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path, monkeypatch):
    """rocisa is currently provisioned from source, not resolved by pip."""
    monkeypatch.setenv("ROCM_VERSION", "7.2.4")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
            ".",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")

    component_version = re.search(
        r'^__version__ = "([^"]+)"$',
        (_PROJECT_ROOT / "tensilelite/__init__.py").read_text(encoding="utf-8"),
        re.MULTILINE,
    ).group(1)
    assert f"Version: {component_version}+rocm7.2.4" in metadata
    assert "Requires-Dist: rocisa" not in metadata


def test_uv_lock_matches_dynamic_package_metadata():
    lock = tomllib.loads((_PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(package for package in lock["package"] if package["name"] == "tensilelite")

    assert "version" not in package
    runtime_dependencies = {dependency["name"] for dependency in package["dependencies"]}
    assert "rocisa" not in runtime_dependencies
    assert set(package["optional-dependencies"]) == {"hip-query", "profile"}

    requires_dist = package["metadata"]["requires-dist"]
    assert not any(dependency["name"] == "rocisa" for dependency in requires_dist)
    assert {
        (dependency["name"], dependency.get("marker"))
        for dependency in requires_dist
        if "extra ==" in dependency.get("marker", "")
    } == {
        ("hip-python", "extra == 'hip-query'"),
        ("yappi", "extra == 'profile'"),
    }


def test_standalone_rocisa_consumes_the_preinstalled_stinkytofu_package():
    cmake = (_PROJECT_ROOT / "rocisa/CMakeLists.txt").read_text(encoding="utf-8")
    tasks = (_PROJECT_ROOT / "tasks.py").read_text(encoding="utf-8")

    assert "if(ROCISA_STANDALONE)\n        find_package(stinkytofu CONFIG QUIET)" in cmake
    assert "if(NOT TARGET stinkytofu::stinkytofu)" in cmake
    assert 'env["CMAKE_PREFIX_PATH"]' in tasks
    assert "pip install --no-build-isolation -e" in tasks


def test_removed_source_autobuild_option_is_not_referenced():
    paths = (
        _PROJECT_ROOT.parent / "CMakeLists.txt",
        _PROJECT_ROOT.parent / "README.md",
        _PROJECT_ROOT / "README.md",
        _PROJECT_ROOT / "AGENTS_reference.md",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "TENSILELITE_ENABLE_AUTOBUILD" not in text, path
        assert "--rebuild-rocisa" not in text, path

    launcher_docs = (_PROJECT_ROOT / "README.md", _PROJECT_ROOT / "AGENTS_reference.md")
    for path in launcher_docs:
        text = path.read_text(encoding="utf-8")
        for retired_name in ("TENSILE_BIN", "DEVELOP_MODE", "Tensile.sh", "Tensile.bat"):
            assert retired_name not in text, path

    rocisa_docs = (
        _PROJECT_ROOT / "rocisa/README.md",
        _PROJECT_ROOT / "rocisa/pyproject.toml",
    )
    for path in rocisa_docs:
        text = path.read_text(encoding="utf-8")
        assert "--no-rebuild-on-import" not in text, path
        assert "--rebuild-on-import" not in text, path
