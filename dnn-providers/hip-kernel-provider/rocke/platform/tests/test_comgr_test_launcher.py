# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Only explicitly marked native lowering tests acquire compiler evidence."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rocke.runtime import comgr


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/run_with_comgr.py"
    spec = importlib.util.spec_from_file_location("compiler_test_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ROCKE_COMGR_LIB", raising=False)
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    return module


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
