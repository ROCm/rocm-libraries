# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.assets.platform_parity_dir finds the platform emitter driver."""

from __future__ import annotations

from rocke.assets import platform_parity_dir, platform_root


def test_platform_parity_dir_defaults_to_the_platform_parity_tree(monkeypatch):
    monkeypatch.delenv("ROCKE_PLATFORM_PARITY_DIR", raising=False)
    parity_dir = platform_parity_dir()
    assert parity_dir == platform_root() / "tests" / "instances" / "parity"


def test_platform_parity_dir_holds_the_emitter_driver():
    """In a checkout and, through conftest's override, in an install."""
    assert (platform_parity_dir() / "_emit_common.py").is_file()


def test_platform_parity_dir_honours_its_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_PLATFORM_PARITY_DIR", str(tmp_path))
    assert platform_parity_dir() == tmp_path
