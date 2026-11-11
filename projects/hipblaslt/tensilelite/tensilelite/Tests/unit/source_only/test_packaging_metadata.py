# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import configparser
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
        entry_points = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
        ).decode("utf-8")

    component_version = (_PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
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

    parsed_entry_points = configparser.ConfigParser()
    parsed_entry_points.optionxform = str
    parsed_entry_points.read_string(entry_points)
    assert dict(parsed_entry_points["console_scripts"]) == {
        "tensilelite": "tensilelite.cli:main",
        "Tensile": "tensilelite.run:main",
        "TensileCreateLibrary": "tensilelite.create_library:main",
        "TensileVerifyStinkyElfText": "tensilelite.verify_stinky_comment_vs_elf_text:main",
        "TensileGetPath": "tensilelite:PrintTensileRoot",
        "TensileBenchmarkCluster": "tensilelite.benchmark_cluster:main",
        "TensileRetuneLibrary": "tensilelite.retune_library:main",
    }


def test_client_identity_is_scoped_and_checked_exactly():
    top_cmake = (_PROJECT_ROOT.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    tests_cmake = (_PROJECT_ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    query_check = (_PROJECT_ROOT / "tests/check_client_version.cmake").read_text(
        encoding="utf-8"
    )

    block = top_cmake.split(
        'if(TENSILELITE_ENABLE_CLIENT)\n    set(_tensilelite_source_root', 1
    )[1].split("\nendif()", 1)[0]
    assert "HIPBLASLT_ENABLE_DEVICE OR TENSILELITE_ENABLE_CLIENT" not in top_cmake
    # The version and commit come from release_metadata.py, which reads no ROCm
    # root; the selected build root feeds only the client's --build-rocm.
    assert '"${_tensilelite_source_root}/release_metadata.py" version' in block
    assert '"${_tensilelite_source_root}/release_metadata.py" source-revision' in block
    assert '"${_tensilelite_source_root}/tensilelite/_build_rocm.py"\n' \
        '            "${HIPBLASLT_BUILD_ROCM_ROOT}"' in block
    assert "release_metadata.py\"\n            \"${HIPBLASLT_BUILD_ROCM_ROOT}" not in block
    assert 'CACHE STRING' in block and "TENSILELITE_SOURCE_REVISION" in block
    assert "client pairing disabled" in block
    # A new commit must re-run the configure that records it.
    assert "--git-path" in block and "packed-refs" in block
    assert "CMAKE_CONFIGURE_DEPENDS" in block

    for query in ("version", "source-revision", "build-rocm"):
        assert query in tests_cmake
    assert '"-DQUERY=--${_query}"' in tests_cmake
    assert "actual STREQUAL EXPECTED" in query_check


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
