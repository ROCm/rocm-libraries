# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of dependency gating; no numeric kernels are launched."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location(
        "gfx950_scaled_numeric", Path(__file__).with_name("gfx950_scaled_numeric.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "route,missing",
    [("comgr", "llvm-objdump"), ("hip", "llvm-objdump"), ("hip", "hipcc")],
)
def test_missing_tools_skip(helper, monkeypatch, route, missing):
    monkeypatch.setattr(helper, "require_gpu", lambda: None)
    monkeypatch.setattr(
        helper.shutil, "which", lambda name: None if name == missing else name
    )
    with pytest.raises(pytest.skip.Exception, match=f"{missing} is required"):
        helper.run_numeric(monkeypatch, "fp8", "bf8", "bf16", route, 32, 64, 128)


def test_check_isa_missing_objdump(helper, monkeypatch):
    monkeypatch.setattr(helper.shutil, "which", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="llvm-objdump is required"):
        helper.check_isa(None, None)


@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_available_tools_reach_harness_and_preserve_failures(
    helper, monkeypatch, route
):
    from rocke.examples.gfx950.gemm import block_scaled_gemm_verify as verifier

    monkeypatch.setattr(helper, "require_gpu", lambda: None)
    # COMGR must remain usable without hipcc.
    available = {"llvm-objdump", "hipcc"} if route == "hip" else {"llvm-objdump"}
    monkeypatch.setattr(
        helper.shutil, "which", lambda name: name if name in available else None
    )

    def fail_numeric(*args, **kwargs):
        raise AssertionError("numeric validation failed")

    monkeypatch.setattr(verifier, "run_cases", fail_numeric)
    with pytest.raises(AssertionError, match="numeric validation failed"):
        helper.run_numeric(monkeypatch, "fp8", "bf8", "bf16", route, 32, 64, 128)


def test_objdump_failure_is_not_skipped(helper, monkeypatch):
    monkeypatch.setattr(helper.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="invalid object"),
    )
    with pytest.raises(AssertionError, match="invalid object"):
        helper.check_isa(SimpleNamespace(hsaco=b"invalid"), None)
