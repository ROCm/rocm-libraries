# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import configparser
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
    rocm_root = tmp_path / "rocm"
    (rocm_root / ".info").mkdir(parents=True)
    (rocm_root / ".info/version").write_text("7.2.4\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.delenv("ROCM_VERSION", raising=False)

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
    assert f"Version: {component_version}+rocm7.2.4" in metadata
    assert "Requires-Dist: rocisa" not in metadata

    parsed_entry_points = configparser.ConfigParser()
    parsed_entry_points.optionxform = str
    parsed_entry_points.read_string(entry_points)
    assert dict(parsed_entry_points["console_scripts"]) == {
        "tensilelite": "tensilelite.cli:main",
        "Tensile": "tensilelite.tensilelite:main",
        "TensileCreateLibrary": "tensilelite.tensilelite_create_library:run",
        "TensileVerifyStinkyElfText": "tensilelite.verify_stinky_comment_vs_elf_text:main",
        "TensileGetPath": "tensilelite:PrintTensileRoot",
        "TensileBenchmarkCluster": "tensilelite.benchmark_cluster:main",
        "TensileRetuneLibrary": "tensilelite.retune_library:main",
    }


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


def test_logic_filter_is_forwarded_to_validation_and_generation():
    cmake = (_PROJECT_ROOT.parent / "cmake/hipblaslt_codegen.cmake").read_text(encoding="utf-8")

    assert (
        'list(APPEND _tensile_logic_args "--logic-filter=**/${_cdl_LOGIC_FILTER}.yaml")'
        in cmake
    )
    assert 'list(APPEND _opts_list "--logic-filter=${_cdl_LOGIC_FILTER}")' in cmake


def test_codegen_preflight_checks_the_package_command_modules():
    cmake = (_PROJECT_ROOT.parent / "cmake/hipblaslt_codegen.cmake").read_text(encoding="utf-8")

    for relative_path in (
        "tensilelite/__main__.py",
        "tensilelite/cli.py",
        "tensilelite/tensilelite_logic/run.py",
        "tensilelite/tensilelite_create_library/run.py",
    ):
        assert f'"${{_codegen_dir}}/{relative_path}"' in cmake
    assert '"${_codegen_dir}/tensilelite/bin/TensileLogic"' not in cmake
    assert '"${_codegen_dir}/tensilelite/tensilelite_create_library/__main__.py"' not in cmake


def test_client_version_metadata_is_scoped_and_checked_exactly():
    top_cmake = (_PROJECT_ROOT.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    tests_cmake = (_PROJECT_ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    version_check = (_PROJECT_ROOT / "tests/check_client_version.cmake").read_text(
        encoding="utf-8"
    )

    version_block = top_cmake.split(
        'if(TENSILELITE_ENABLE_CLIENT)\n    set(_tensilelite_source_root', 1
    )[1].split("\nendif()", 1)[0]
    assert "TENSILELITE_DISTRIBUTION_VERSION" in version_block
    assert "HIPBLASLT_ENABLE_DEVICE OR TENSILELITE_ENABLE_CLIENT" not in top_cmake
    assert '"-DEXPECTED_VERSION=${TENSILELITE_DISTRIBUTION_VERSION}"' in tests_cmake
    assert "actual_version STREQUAL EXPECTED_VERSION" in version_check


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
