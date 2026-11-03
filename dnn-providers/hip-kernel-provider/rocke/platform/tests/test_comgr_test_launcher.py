# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Only explicitly marked native lowering tests acquire compiler evidence."""
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rocke.runtime import comgr


def _launcher_path(test_file: Path) -> Path:
    candidates = (
        test_file.parents[1] / "tools" / "run_with_comgr.py",
        test_file.parent / "tools" / "run_with_comgr.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("compiler test asset run_with_comgr.py is absent")


def _load_launcher(monkeypatch):
    path = _launcher_path(Path(__file__).resolve())
    spec = importlib.util.spec_from_file_location("compiler_test_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The launcher writes its compiler default directly to os.environ. Record
    # restoration even when the parent did not provide an override.
    for key in ("ROCKE_COMGR_LIB", "ROCKE_LLVM_FLAVOR"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    return module


@pytest.fixture
def launcher(monkeypatch):
    return _load_launcher(monkeypatch)


@pytest.mark.parametrize("initial", [None, "existing-selection"])
def test_launcher_fixture_restores_environment(monkeypatch, initial):
    keys = ("ROCKE_COMGR_LIB", "ROCKE_LLVM_FLAVOR")
    for key in keys:
        monkeypatch.setenv(key, initial or "")
        if initial is None:
            monkeypatch.delenv(key)
    with pytest.MonkeyPatch.context() as fixture_patch:
        module = _load_launcher(fixture_patch)
        # Exercise the same in-process launcher mutation as the lowering case,
        # then check the environment after its fixture has been torn down.
        test_lowering_command_receives_owner_profile(module, fixture_patch)
    for key in keys:
        actual = os.environ.get(key)
        assert actual == initial


@pytest.mark.parametrize("override", [None, "missing-compiler"])
def test_host_command_never_acquires_comgr(launcher, monkeypatch, override):
    if override:
        monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    query = Mock(side_effect=AssertionError("host compiler acquisition"))
    monkeypatch.setattr(comgr, "_require_compiler_info", query)
    child = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", child)
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", "--", "host-test"])
    assert launcher.main() == 0
    query.assert_not_called()
    assert "ROCKE_LLVM_FLAVOR" not in child.call_args.kwargs["env"]


def test_lowering_command_receives_owner_profile(launcher, monkeypatch):
    info = comgr.CompilerInfo((24, 0, 0), "fixture", None, "selected", "selected")
    query = Mock(return_value=info)
    monkeypatch.setattr(comgr, "_require_compiler_info", query)
    child = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(launcher.subprocess, "run", child)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "launcher",
            "--needs-llvm-flavor",
            "--default-comgr",
            "selected",
            "--",
            "lower-test",
        ],
    )
    assert launcher.main() == 7
    query.assert_called_once_with()
    assert child.call_args.kwargs["env"]["ROCKE_LLVM_FLAVOR"] == "llvm24"
    assert child.call_args.kwargs["env"]["ROCKE_COMGR_LIB"] == "selected"


def test_explicit_offline_command_does_not_acquire_compiler(launcher, monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    query = Mock(side_effect=AssertionError("offline compiler acquisition"))
    monkeypatch.setattr(comgr, "_require_compiler_info", query)
    monkeypatch.setattr(
        launcher.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0))
    )
    monkeypatch.setattr(
        launcher.sys, "argv", ["launcher", "--needs-llvm-flavor", "--", "lower-test"]
    )
    assert launcher.main() == 0
    query.assert_not_called()


@pytest.mark.parametrize("layout", ["source", "installed"])
def test_launcher_path_resolves_both_layouts(tmp_path, layout):
    test_file = tmp_path / "tests" / "test_comgr_test_launcher.py"
    tool_root = tmp_path if layout == "source" else test_file.parent
    helper = tool_root / "tools" / "run_with_comgr.py"
    helper.parent.mkdir(parents=True)
    helper.touch()
    assert _launcher_path(test_file) == helper


def test_launcher_path_rejects_missing_test_asset(tmp_path):
    test_file = tmp_path / "tests" / "test_comgr_test_launcher.py"
    with pytest.raises(FileNotFoundError, match="run_with_comgr.py is absent"):
        _launcher_path(test_file)
