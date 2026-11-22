# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.assets.platform_instances_dir and platform_parity_dir find the platform
instance tests' IR parity harness and emitter driver."""

from __future__ import annotations

from rocke.assets import platform_instances_dir, platform_parity_dir, platform_root


def test_platform_instances_dir_defaults_to_the_platform_instances_tree(monkeypatch):
    monkeypatch.delenv("ROCKE_PLATFORM_INSTANCES_DIR", raising=False)
    assert platform_instances_dir() == platform_root() / "tests" / "instances"


def test_platform_instances_dir_holds_the_ir_parity_harness():
    """In a checkout and, through conftest's override, in an install."""
    assert (platform_instances_dir() / "rocke_ir_parity_harness.py").is_file()


def test_platform_instances_dir_honours_its_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_PLATFORM_INSTANCES_DIR", str(tmp_path))
    assert platform_instances_dir() == tmp_path


def test_platform_parity_dir_defaults_to_the_platform_parity_tree(monkeypatch):
    monkeypatch.delenv("ROCKE_PLATFORM_INSTANCES_DIR", raising=False)
    monkeypatch.delenv("ROCKE_PLATFORM_PARITY_DIR", raising=False)
    parity_dir = platform_parity_dir()
    assert parity_dir == platform_root() / "tests" / "instances" / "parity"


def test_platform_parity_dir_holds_the_emitter_driver():
    """In a checkout and, through conftest's override, in an install."""
    assert (platform_parity_dir() / "_emit_common.py").is_file()


def test_platform_parity_dir_follows_the_instances_override(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCKE_PLATFORM_PARITY_DIR", raising=False)
    monkeypatch.setenv("ROCKE_PLATFORM_INSTANCES_DIR", str(tmp_path))
    assert platform_parity_dir() == tmp_path / "parity"


def test_platform_parity_dir_honours_its_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKE_PLATFORM_PARITY_DIR", str(tmp_path))
    assert platform_parity_dir() == tmp_path
