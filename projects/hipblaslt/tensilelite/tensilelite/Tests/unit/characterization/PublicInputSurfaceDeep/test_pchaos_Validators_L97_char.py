################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

"""Characterize frozen-root tool search in ``Toolchain.Validators``.

The runtime root selected at ``import tensilelite`` is the sole source for
relative tool names. Ambient ``ROCM_PATH``, ``/opt/rocm``, and ``PATH`` must not
contribute competing SDK roots.
"""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _search_paths():
    from tensilelite.Toolchain import Validators

    return Validators, Validators._posixSearchPaths


def test_real_function_uses_only_the_frozen_root(monkeypatch):
    validators, posix_search_paths = _search_paths()
    root = Path("/selected/rocm")
    monkeypatch.setattr(validators, "_toolchainSearchPaths", lambda: [root / "bin", root / "lib" / "llvm" / "bin"])
    monkeypatch.setenv("ROCM_PATH", "/stale/rocm")
    monkeypatch.setenv("PATH", "/stale/bin")

    assert posix_search_paths() == [
        root / "bin",
        root / "lib" / "llvm" / "bin",
    ]


def test_relative_tool_does_not_fall_back_outside_frozen_root(tmp_path, monkeypatch):
    validators, _ = _search_paths()
    root = tmp_path / "selected"
    selected_bin = root / "bin"
    selected_bin.mkdir(parents=True)
    stale_bin = tmp_path / "stale"
    stale_bin.mkdir()
    stale_tool = stale_bin / "amdclang++"
    stale_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    stale_tool.chmod(0o755)
    monkeypatch.setattr(validators, "_toolchainSearchPaths", lambda: [root / "bin", root / "lib" / "llvm" / "bin"])
    monkeypatch.setenv("PATH", str(stale_bin))

    with pytest.raises(FileNotFoundError):
        validators.validateToolchain("amdclang++")
