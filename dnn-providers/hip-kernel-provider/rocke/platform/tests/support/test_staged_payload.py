# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.staged_payload points the asset locators at an install's staged trees."""

from __future__ import annotations

import os

from rocke.tests.staged_payload import point_locators_at_staged_payload


def _stage(tests_root, *trees):
    for tree in trees:
        (tests_root / tree).mkdir(parents=True)


def test_each_staged_tree_sets_its_locator(tmp_path):
    _stage(tmp_path, "arch_domain", "instances/parity")
    env = {}
    point_locators_at_staged_payload(tmp_path, env)
    assert env == {
        "ROCKE_PLATFORM_TOOLS_DIR": str(tmp_path / "arch_domain"),
        "ROCKE_PLATFORM_INSTANCES_DIR": str(tmp_path / "instances"),
        "ROCKE_PLATFORM_PARITY_DIR": str(tmp_path / "instances" / "parity"),
    }


def test_a_tree_the_payload_lacks_leaves_its_locator_unset(tmp_path):
    """A checkout's test root stages no tools tree; its default stays in force."""
    _stage(tmp_path, "instances/parity")
    env = {}
    point_locators_at_staged_payload(tmp_path, env)
    assert "ROCKE_PLATFORM_TOOLS_DIR" not in env
    assert env["ROCKE_PLATFORM_PARITY_DIR"] == str(tmp_path / "instances" / "parity")


def test_a_value_the_caller_chose_is_kept(tmp_path):
    _stage(tmp_path, "arch_domain", "instances/parity")
    env = {"ROCKE_PLATFORM_PARITY_DIR": "chosen"}
    point_locators_at_staged_payload(tmp_path, env)
    assert env["ROCKE_PLATFORM_PARITY_DIR"] == "chosen"
    assert env["ROCKE_PLATFORM_TOOLS_DIR"] == str(tmp_path / "arch_domain")


def test_the_process_environment_is_the_default(tmp_path, monkeypatch):
    _stage(tmp_path, "instances/parity")
    # setenv first, so monkeypatch restores the original value (or absence)
    # after the helper writes the variable.
    for variable in ("ROCKE_PLATFORM_INSTANCES_DIR", "ROCKE_PLATFORM_PARITY_DIR"):
        monkeypatch.setenv(variable, "")
        monkeypatch.delenv(variable)
    point_locators_at_staged_payload(tmp_path)
    assert os.environ["ROCKE_PLATFORM_INSTANCES_DIR"] == str(tmp_path / "instances")
    assert os.environ["ROCKE_PLATFORM_PARITY_DIR"] == str(
        tmp_path / "instances" / "parity"
    )
