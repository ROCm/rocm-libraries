# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Installed smoke remains offline by default; compiler checks are explicit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

asset_root = Path(__file__).resolve().parents[2]
script = asset_root / "tools" / "rocke_installed_smoke.py"
if not script.is_file():
    script = asset_root / "rocke_installed_smoke.py"
spec = importlib.util.spec_from_file_location("installed_smoke", script)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_default_smoke_does_not_query_compiler(monkeypatch):
    from rocke.runtime import comgr

    def unexpected():
        raise AssertionError("offline smoke queried compiler")

    monkeypatch.setattr(comgr, "loaded_compiler_info", unexpected)
    monkeypatch.setattr(sys, "argv", ["smoke"])
    assert smoke.main() == 0


@pytest.mark.parametrize("valid", [True, False])
def test_compiler_smoke_checks_evidence_and_elf(monkeypatch, valid):
    from rocke.core import lower_llvm
    from rocke.runtime import comgr

    info = comgr.CompilerInfo(
        (23, 0, 0), "LLVMGetVersion", "compiler.so", "compiler.so", "compiler.so"
    )
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: info)
    monkeypatch.setattr(comgr, "resolved_lib_path", lambda: "compiler.so")
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", lambda: "llvm23")
    seen = {}

    def build(text, *, isa):
        seen.update(text=text, isa=isa)
        return (b"\x7fELFpayload" if valid else b"invalid"), comgr.ComgrTimings()

    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", build)
    if not valid:
        with pytest.raises(RuntimeError, match="nonempty ELF HSACO"):
            smoke._smoke_comgr()
    else:
        result = smoke._smoke_comgr()
        assert result["compiler"]["comgr_path"] == "compiler.so"
        assert result["flavor"] == "llvm23" and result["hsaco_bytes"] == 11
        assert "rocke_installed_comgr_smoke" in seen["text"]
        assert seen["isa"] == "amdgcn-amd-amdhsa--gfx942"


def test_compiler_smoke_rejects_unqueryable_library(monkeypatch):
    from rocke.runtime import comgr

    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: None)
    with pytest.raises(RuntimeError, match="lacks compiler evidence"):
        smoke._smoke_comgr()


def test_check_comgr_option_reports_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke", "--check-comgr"])
    monkeypatch.setattr(smoke, "_smoke_python_engine", lambda: ("ir", "ll"))
    monkeypatch.setattr(
        smoke, "_smoke_comgr", lambda: dict(flavor="llvm23", hsaco_bytes=42)
    )
    assert smoke.main() == 0
    assert (
        'COMGR smoke: PASS {"flavor": "llvm23", "hsaco_bytes": 42}'
        in capsys.readouterr().out
    )
