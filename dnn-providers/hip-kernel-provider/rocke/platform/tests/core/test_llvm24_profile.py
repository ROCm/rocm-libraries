# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM24 is a qualified compiler lane sharing LLVM23's emission profile."""

from __future__ import annotations

import pytest

from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder


def test_lane_order_and_shared_profile() -> None:
    assert lower_llvm.LLVM_FLAVORS == ("llvm20", "llvm22", "llvm23", "llvm24")
    assert tuple(lower_llvm._LLVM_EMISSION_PROFILES) == lower_llvm.LLVM_FLAVORS
    assert lower_llvm.LLVM_EMISSION_FLAVORS == ("llvm20", "llvm22", "llvm23")
    assert lower_llvm.LLVM_EMISSION_ALIASES == {"llvm24": "llvm23"}
    llvm23 = lower_llvm._emission_profile("llvm23")
    llvm24 = lower_llvm._emission_profile("llvm24")
    assert llvm24 is llvm23
    assert lower_llvm._datalayout_for_flavor("llvm24") == (
        lower_llvm._datalayout_for_flavor("llvm23")
    )
    assert lower_llvm._decls_for_flavor("llvm24") == lower_llvm._decls_for_flavor(
        "llvm23"
    )
    assert lower_llvm._supports_llvm23_target_ops("llvm24")


def test_explicit_llvm24_emission_does_not_query_comgr(monkeypatch) -> None:
    def reject_query() -> None:
        raise AssertionError("queried COMGR")

    monkeypatch.setattr("rocke.runtime.comgr._require_compiler_info", reject_query)
    builder = IRBuilder("llvm24_offline")
    llvm = lower_llvm._lower_kernel_to_llvm_python(
        builder.kernel, arch="gfx950", llvm_flavor="llvm24"
    )
    assert lower_llvm._datalayout_for_flavor("llvm24") in llvm


def test_unknown_profile_fails_with_the_supported_lanes() -> None:
    with pytest.raises(ValueError, match="unknown LLVM flavor 'llvm25'.*llvm24"):
        lower_llvm._emission_profile("llvm25")
