# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from tensilelite import GENERATOR_VERSION


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def test_distribution_version_comes_from_selected_rocm_root(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("ROCM_VERSION", "7.3.0")
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert GENERATOR_VERSION == component_version
    assert metadata["component_version"]() == component_version
    assert metadata["distribution_version"]() == f"{component_version}+rocm7.2.4"


def test_explicit_rocm_root_overrides_the_ambient_root(tmp_path, monkeypatch):
    ambient_root = tmp_path / "ambient"
    explicit_root = tmp_path / "explicit"
    for root, version in ((ambient_root, "7.2.4"), (explicit_root, "7.3.1")):
        (root / ".info").mkdir(parents=True)
        (root / ".info/version").write_text(f"{version}\n", encoding="utf-8")

    monkeypatch.setenv("ROCM_PATH", str(ambient_root))
    metadata_path = _SOURCE_ROOT / "release_metadata.py"
    metadata = runpy.run_path(str(metadata_path))
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert metadata["distribution_version"](explicit_root) == (
        f"{component_version}+rocm7.3.1"
    )
    result = subprocess.run(
        [sys.executable, str(metadata_path), str(explicit_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"{component_version}+rocm7.3.1"


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
