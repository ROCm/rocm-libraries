# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import runpy

import pytest


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def test_distribution_version_comes_from_selected_rocm_root(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("ROCM_VERSION", "7.3.0")
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))

    assert metadata["component_version"]() == "5.0.0"
    assert metadata["distribution_version"]() == "5.0.0+rocm7.2.4"


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
