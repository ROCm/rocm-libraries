# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The skip policy every compile test shares (``_compiler_test_utils``).

CPU only. Each error is a real one from the committed option table: a
compiler whose LLVM major the table records as refusing the option
(``CompilerOptionNotAccepted``), or one the table has no answer for (a plain
``ValueError``). These tests pin that a refusal skips, that it fails when
``ROCKE_REQUIRE_COMPILER_TOOLS=1`` requires the tools, and that no other error
is turned into a skip. They also pin that ``compile_or_skip``, the compile
helper of the attention backward tests, applies the same policy to a real
``compile_kernel`` call.
"""

from __future__ import annotations

import unittest

import pytest
from rocke.core.ir import KernelDef, Region
from rocke.runtime import comgr
from rocke.runtime.comgr import CompilerInfo
from rocke.runtime.compiler_options import measured_answers, require_accepted

from . import _attention_bwd_resources
from ._attention_bwd_resources import compile_or_skip
from ._compiler_test_utils import (
    REQUIRE_COMPILER_TOOLS_ENV,
    skip_where_compiler_refuses_option,
)

OPTION = "-amdgpu-mfma-vgpr-form"
# How the refusal starts, and how the policy introduces it.
REFUSAL = f"kernel 'probe' needs the compiler option '-mllvm {OPTION}'"
SKIP_REASON = f"loaded compiler cannot build this kernel: {REFUSAL}"


def _compiler(llvm_major: int) -> CompilerInfo:
    """A compiler identity as ``loaded_compiler_info`` reports one."""
    return CompilerInfo(
        llvm_version=(llvm_major, 0, 0),
        source="LLVMGetVersion",
        requested_comgr="libamd_comgr.so",
        comgr_path=None,
        query_library_path=None,
    )


def _compile_on(llvm_major: int) -> None:
    """Ask the committed table whether LLVM ``llvm_major`` accepts ``OPTION``,
    as ``compile_kernel`` does before compiling a kernel that needs it."""
    require_accepted(
        ["-O3", "-mllvm", OPTION],
        compiler=_compiler(llvm_major),
        needed_by="kernel 'probe'",
    )


def _refusing_major() -> int:
    """An LLVM major the table records as refusing ``OPTION``."""
    return next(
        major
        for major, answer in measured_answers()[OPTION].items()
        if not answer.accepted
    )


def _unmeasured_major() -> int:
    """An LLVM major newer than every one the table has an answer for."""
    return 1 + max(measured_answers()[OPTION])


def test_refused_option_skips(monkeypatch):
    monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    with (
        pytest.raises(unittest.SkipTest, match=f"^{SKIP_REASON}"),
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_refusing_major())


def test_refused_option_fails_where_tools_are_required(monkeypatch):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, "1")
    required = f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but {SKIP_REASON}"
    with (
        pytest.raises(AssertionError, match=required),
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_refusing_major())


@pytest.mark.parametrize("required", ["0", "1"])
def test_other_errors_propagate(monkeypatch, required):
    # A compiler the table has no answer for raises a plain ValueError (the
    # base class of the refusal): that is a missing measurement, not a skip.
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    with (
        pytest.raises(
            ValueError, match="which has not been measured on LLVM"
        ) as caught,
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_unmeasured_major())
    assert type(caught.value) is ValueError


def _vgpr_form_kernel() -> KernelDef:
    """An empty kernel that asks for VGPR-form MFMAs, so it needs ``OPTION``."""
    kernel = KernelDef("probe", [], Region("entry"))
    kernel.attrs["mfma_vgpr_form"] = True
    return kernel


@pytest.mark.parametrize(
    "toolchain_gap",
    [None, "flavor llvm20"],
    ids=["gate-toolchain", "other-toolchain"],
)
@pytest.mark.parametrize(
    ("required", "outcome", "reason"),
    [
        ("0", unittest.SkipTest, f"^{SKIP_REASON}"),
        ("1", AssertionError, f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but {SKIP_REASON}"),
    ],
    ids=["skips", "fails-where-tools-are-required"],
)
def test_compile_or_skip_applies_the_policy(
    monkeypatch, toolchain_gap, required, outcome, reason
):
    # The loaded compiler's identity, as the option check sees it, is replaced
    # by one the table records as refusing OPTION, so compile_kernel refuses
    # before COMGR on any host. compile_or_skip's own rule depends on whether
    # the host is the resource gate's toolchain, so both answers are pinned.
    refusing = _compiler(_refusing_major())
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: refusing)
    monkeypatch.setattr(
        _attention_bwd_resources, "gate_toolchain_gap", lambda arch: toolchain_gap
    )
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    try:
        with pytest.raises(outcome, match=reason):
            # The Python backend: a helper that lost the refusal must fail here
            # without needing the C++ engine.
            compile_or_skip(_vgpr_form_kernel(), arch="gfx942", backend="python")
    except pytest.skip.Exception as skipped:
        # Its own toolchain-gap skip would hide the refusal here, and pytest
        # would report this test as skipped rather than failed.
        pytest.fail(f"compile_or_skip skipped outside the shared policy: {skipped}")
