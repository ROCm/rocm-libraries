# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import runpy
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def test_distribution_version_comes_from_explicit_base_version():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    assert metadata["component_version"]() == "5.0.0"
    assert metadata["distribution_version"]("7.2.4") == "5.0.0+rocm7.2.4"


def test_distribution_version_preserves_development_publication_identity():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    assert (
        metadata["distribution_version"]("10.1.0.dev0+0123456789abcdef")
        == "5.0.0+devrocm10.1.0.dev0.0123456789abcdef"
    )


def test_compatibility_setup_uses_canonical_metadata():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    assert metadata["distribution_version"]("7.2.4") == "5.0.0+rocm7.2.4"
    setup_text = (_SOURCE_ROOT / "compat" / "setup.py").read_text(encoding="utf-8")
    assert 'f"tensilelite=={_version}"' in setup_text


def test_distribution_version_rejects_invalid_base_version():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    with pytest.raises(RuntimeError, match="ROCm Python package builds require"):
        metadata["distribution_version"]("not-a-release")


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
