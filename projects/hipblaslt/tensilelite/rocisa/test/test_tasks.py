# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for rocisa's editable-install task options."""

import tasks


class _Context:
    def __init__(self):
        self.commands = []

    def run(self, command, env):
        self.commands.append((command, env))


def _editable_install(monkeypatch, tmp_path, rebuild_on_import):
    monkeypatch.setattr(tasks, "_detect_rocm", lambda: "/opt/rocm")
    monkeypatch.setattr(tasks, "_build_and_install_stinkytofu", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)
    monkeypatch.delenv("SKBUILD_EDITABLE_REBUILD", raising=False)

    context = _Context()
    tasks._pip_install_rocisa(
        context,
        rocisa_dir=tmp_path / "rocisa",
        stinkytofu_prefix=tmp_path / "stinkytofu-install",
        rebuild_on_import=rebuild_on_import,
    )
    return context.commands[-1]


def test_editable_rocisa_preserves_rebuild_on_import_by_default(monkeypatch, tmp_path):
    _, env = _editable_install(monkeypatch, tmp_path, rebuild_on_import=True)
    assert env["SKBUILD_EDITABLE_REBUILD"] == "true"


def test_editable_rocisa_can_opt_out_of_rebuild_on_import(monkeypatch, tmp_path):
    _, env = _editable_install(monkeypatch, tmp_path, rebuild_on_import=False)
    assert env["SKBUILD_EDITABLE_REBUILD"] == "false"
