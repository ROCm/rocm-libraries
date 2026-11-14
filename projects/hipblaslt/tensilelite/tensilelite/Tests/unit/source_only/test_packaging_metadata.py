# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_ci_bootstrap_requirements_include_invoke():
    """Math CI installs requirements.txt before calling invoke build-client."""
    requirements = [
        Requirement(line)
        for line in (_PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any(
        requirement.name.lower() == "invoke" and requirement.marker is None
        for requirement in requirements
    ), "Math CI's source-build bootstrap must install Invoke unconditionally"


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path, monkeypatch):
    """rocisa is currently provisioned from source, not resolved by pip."""
    # The build reads no ROCm installation; these must not change the version.
    monkeypatch.setenv("ROCM_VERSION", "7.2.4")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "no-rocm-here"))
    monkeypatch.delenv("TENSILELITE_SOURCE_REVISION", raising=False)

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
    version = re.search(r"^Version: (.+)$", metadata, re.MULTILINE).group(1)
    # A source build is a development build: labelled with the commit when git
    # knows it, unlabelled otherwise. It never claims a ROCm release.
    assert re.fullmatch(
        re.escape(component_version) + r"(\+g[0-9a-f]{40}(\.dirty)?)?", version
    ), version
    runtime_requirements = {
        Requirement(line.removeprefix("Requires-Dist: ")).name.lower()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    }
    assert "rocisa" not in runtime_requirements
    assert "invoke" not in runtime_requirements


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
        assert "SKBUILD_EDITABLE_REBUILD" not in text, path
        assert "editable.rebuild" not in text, path
