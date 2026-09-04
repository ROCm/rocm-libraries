################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterize Validators' selected-installation search-path seam."""

from pathlib import Path

import pytest

from tensilelite.Toolchain import Validators


pytestmark = pytest.mark.unit


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_validate_toolchain_uses_runtime_selected_paths(monkeypatch, tmp_path):
    _executable(tmp_path, "amdclang++")
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    assert Validators.validateToolchain("amdclang++") == str(tmp_path / "amdclang++")


def test_missing_component_reports_the_selected_installation(monkeypatch, tmp_path):
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        Validators.validateToolchain("amdclang++")
