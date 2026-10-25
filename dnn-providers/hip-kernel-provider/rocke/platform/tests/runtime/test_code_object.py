# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Code-object facade over COMGR and the direct Rust library lookup."""

from __future__ import annotations

from unittest import mock

import pytest
from rocke.runtime import code_object, comgr


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    monkeypatch.delenv("ROCKE_CODEOBJ_BACKEND", raising=False)
    monkeypatch.delenv("ROCKE_CODEOBJ_DIRECT_LIB", raising=False)


def test_default_backend_delegates_to_comgr(monkeypatch) -> None:
    timings = comgr.ComgrTimings(bc=1.0, link=2.0, relocatable=3.0, executable=4.0)
    build = mock.Mock(return_value=(b"hsaco", timings))
    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", build)

    hsaco, result = code_object.build_hsaco_from_llvm_ir(
        "IR", isa="amdgcn-amd-amdhsa--gfx942", fresh_process=True
    )

    assert hsaco == b"hsaco"
    assert result.total == 10.0
    build.assert_called_once_with(
        "IR",
        isa="amdgcn-amd-amdhsa--gfx942",
        options=["-O3"],
        fresh_process=True,
    )


def test_module_linking_delegates_to_comgr(monkeypatch) -> None:
    timings = comgr.ComgrTimings(bc=0.1, link=0.2, relocatable=0.3, executable=0.4)
    build = mock.Mock(return_value=(b"hsaco", timings))
    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_modules", build)
    modules = ["first", "second"]

    hsaco, result = code_object.build_hsaco_from_llvm_modules(
        modules, isa="amdgcn-amd-amdhsa--gfx942"
    )

    assert hsaco == b"hsaco"
    assert result.total == pytest.approx(1.0)
    build.assert_called_once_with(
        ("first", "second"), isa="amdgcn-amd-amdhsa--gfx942", options=["-O3"]
    )


def test_invalid_backend_fails_before_loading_a_compiler() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        code_object.resolve_code_object_backend("unknown")


def test_stale_backend_environment_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "rust-worker")
    with pytest.raises(ValueError, match="ROCKE_CODEOBJ_BACKEND"):
        code_object.resolve_code_object_backend()


def test_direct_library_override_must_name_a_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ROCKE_CODEOBJ_DIRECT_LIB", str(tmp_path / "missing.dll"))
    with pytest.raises(code_object.CodeObjectError, match="is not a file"):
        code_object._direct_library_path()


def test_missing_direct_library_fails_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(code_object, "_direct_library_path", lambda: None)
    with pytest.raises(code_object.CodeObjectError, match="ROCKE_BUILD_RUST_CODEOBJ"):
        code_object.build_direct_gfx1151_wmma_gemm_hsaco(
            "kernel @rocke_direct_wmma_gemm {\n"
        )


def test_comgr_errors_are_generic_compiler_errors() -> None:
    assert issubclass(comgr.ComgrError, code_object.CodeObjectError)
    assert issubclass(comgr.ComgrProcessDied, code_object.CodeObjectProcessDied)
