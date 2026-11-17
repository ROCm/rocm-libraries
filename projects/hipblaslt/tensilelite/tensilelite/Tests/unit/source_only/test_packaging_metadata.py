# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for TensileLite's distributable package metadata."""

import configparser
import os
import re
import runpy
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_STAGER = _PROJECT_ROOT / "scripts/stage_release_source.py"


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


def _isolated_source(tmp_path):
    destination = tmp_path / "source"
    runpy.run_path(str(_STAGER))["stage_source"](_PROJECT_ROOT, destination)
    return destination


def test_wheel_metadata_does_not_require_unpublished_rocisa(tmp_path, monkeypatch):
    """rocisa is currently provisioned from source, not resolved by pip."""
    source_root = _isolated_source(tmp_path)
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
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        archived_names = set(archive.namelist())
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")
        entry_points = archive.read(
            next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
        ).decode("utf-8")

    component_version = (source_root / "VERSION").read_text(encoding="utf-8").strip()
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
        "tensilelite-configure-client": "tensilelite_configure_client:main",
    }
    assert "_tensilelite_client_binding.py" in archived_names
    assert "tensilelite_configure_client.py" in archived_names


def _pip_wheel(source, out, env):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
         "--disable-pip-version-check", "--wheel-dir", str(out), str(source)],
        cwd=source, env=env, check=True, capture_output=True, text=True,
    )
    (wheel,) = Path(out).glob("*.whl")
    return wheel


def _clean_env(**extra):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("TENSILELITE_ROCM_VERSION", "ROCM_PATH", "TENSILELITE_SOURCE_REVISION")
    }
    env.update(extra)
    return env


def test_explicit_rocm_identity_labels_the_wheel_and_keeps_the_source_root_clean(tmp_path):
    source_root = _isolated_source(tmp_path)
    sha = "0123456789abcdef0123456789abcdef01234567"

    wheel = _pip_wheel(
        source_root,
        tmp_path / "out",
        _clean_env(TENSILELITE_ROCM_VERSION="7.12.0rc1", TENSILELITE_SOURCE_REVISION=sha),
    )

    assert wheel.name == "tensilelite-5.0.0+rocm7.12.0rc1-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        assert f"SOURCE_REVISION = {sha!r}" in archive.read("tensilelite/_build_info.py").decode()
    assert not (source_root / "tensilelite.egg-info").exists()


def test_sdist_keeps_its_rocm_identity_when_built_into_a_wheel(tmp_path):
    source_root = _isolated_source(tmp_path)
    sha = "0123456789abcdef0123456789abcdef01234567"
    sdist_dir = tmp_path / "sdist"
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from setuptools import build_meta; build_meta.build_sdist(sys.argv[1])",
         str(sdist_dir)],
        cwd=source_root,
        env=_clean_env(TENSILELITE_ROCM_VERSION="7.12.0", TENSILELITE_SOURCE_REVISION=sha),
        check=True, capture_output=True, text=True,
    )
    (sdist,) = sdist_dir.glob("*.tar.gz")
    assert sdist.name == "tensilelite-5.0.0+rocm7.12.0.tar.gz"
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / "unpacked", filter="data")
    (unpacked,) = (tmp_path / "unpacked").iterdir()

    wheel = _pip_wheel(unpacked, tmp_path / "wheel", _clean_env())
    assert wheel.name == "tensilelite-5.0.0+rocm7.12.0-py3-none-any.whl"

    mismatch = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
         "--wheel-dir", str(tmp_path / "other"), str(unpacked)],
        env=_clean_env(TENSILELITE_ROCM_VERSION="7.13.0"),
        capture_output=True, text=True,
    )
    assert mismatch.returncode != 0
    assert "this sdist was made as 5.0.0+rocm7.12.0" in mismatch.stdout + mismatch.stderr


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
