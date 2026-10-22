# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The skip policy every compile test shares (``_compiler_test_utils``).

CPU only. Each error is a real one from the committed option table: a
compiler whose LLVM major the table records as refusing the option
(``CompilerOptionNotAccepted``), or one the table has no answer for (a plain
``ValueError``). These tests pin that a refusal skips, that it fails when
``ROCKE_REQUIRE_COMPILER_TOOLS=1`` requires the tools, and that no other error
is turned into a skip.
"""

from __future__ import annotations

import unittest

import pytest
from rocke.runtime.comgr import CompilerInfo
from rocke.runtime.compiler_options import lookup, require_accepted

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


# The LLVM majors the helpers below search the table for.
_CANDIDATE_MAJORS = range(1, 1000)


def _refusing_major() -> int:
    """An LLVM major the table records as refusing ``OPTION``."""
    return next(
        major
        for major in _CANDIDATE_MAJORS
        if (answer := lookup(OPTION, major)) is not None and not answer.accepted
    )


def _unmeasured_major() -> int:
    """An LLVM major newer than every one the table has an answer for."""
    return 1 + max(
        major for major in _CANDIDATE_MAJORS if lookup(OPTION, major) is not None
    )


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
