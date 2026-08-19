# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from tensilelite import GENERATOR_VERSION
import os

pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def test_distribution_version_uses_explicit_rocm_identity():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert GENERATOR_VERSION == component_version
    assert metadata["component_version"]() == component_version
    assert metadata["distribution_version"]("7.2.4") == f"{component_version}+rocm7.2.4"


def test_release_metadata_cli_uses_explicit_rocm_identity():
    metadata_path = _SOURCE_ROOT / "release_metadata.py"
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    result = subprocess.run(
        [sys.executable, str(metadata_path), "7.3.1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"{component_version}+rocm7.3.1"


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("10.1.0a20260813", "10.1.0a20260813"),
        ("7.2.4+nightly", "7.2.4.nightly"),
    ],
)
def test_distribution_version_preserves_publication_identity(value, canonical):
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert metadata["distribution_version"](value) == f"{component_version}+rocm{canonical}"


@pytest.mark.parametrize("value", ["", "7.2", "v7.2.4", "7.2.x"])
def test_distribution_version_rejects_invalid_rocm_identity(value):
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    with pytest.raises(RuntimeError, match="ROCm Python package builds require"):
        metadata["distribution_version"](value)


@pytest.mark.parametrize("value", ["5.0", "5.0.0.dev1", "v5.0.0", ""])
def test_component_version_rejects_non_release_values(tmp_path, value):
    source = tmp_path / "source"
    source.mkdir()
    (source / "VERSION").write_text(value, encoding="utf-8")
    metadata_source = (_SOURCE_ROOT / "release_metadata.py").read_text(encoding="utf-8")
    (source / "release_metadata.py").write_text(metadata_source, encoding="utf-8")
    metadata = runpy.run_path(str(source / "release_metadata.py"))

    with pytest.raises(RuntimeError, match="VERSION must contain"):
        metadata["component_version"]()

def test_tox_package_bootstrap_reads_the_selected_rocm_identity(tmp_path):
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    environment = dict(
        os.environ,
        TOX_ENV_NAME="unit",
        ROCM_PATH=str(root),
    )
    environment.pop("TENSILELITE_ROCM_VERSION", None)

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=_SOURCE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "5.0.0+rocm7.2.4"
