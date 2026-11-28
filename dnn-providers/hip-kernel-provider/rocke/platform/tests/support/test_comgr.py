# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for the COMGR test helper and the compiler-tool policy it applies.

``require_comgr_target`` skips a compile-backed test when the loaded COMGR cannot build for
its arch, and fails it instead under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``, so a gate whose tools
are required cannot pass by skipping every case. ``gfx0`` is an arch no COMGR lists; on a host
without COMGR the reason is that COMGR is unavailable, which the policy treats the same way.
``compile_kernel_or_skip`` applies that check before it compiles, and leaves any error after it
to the kernel.
"""

from __future__ import annotations

import unittest

import pytest
from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType, Region
from rocke.runtime import comgr
from rocke.tests.comgr import compile_kernel_or_skip, require_comgr_target
from rocke.tests.compiler_tools import REQUIRE_COMPILER_TOOLS_ENV
from rocke.tests.lanes import BUILD_VARIANT, lanes_waited_for

_UNLISTED_ARCH = "gfx0"


@pytest.mark.parametrize("requirement", [None, "0"])
def test_an_arch_comgr_cannot_build_skips_without_the_requirement(
    monkeypatch, requirement
):
    if requirement is None:
        monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    else:
        monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, requirement)
    with pytest.raises(unittest.SkipTest) as skipped:
        require_comgr_target(_UNLISTED_ARCH)
    # A missing tool waits for the CI test environment to supply it.
    assert lanes_waited_for(str(skipped.value)) == (BUILD_VARIANT,)


def test_an_arch_comgr_cannot_build_fails_under_the_requirement(monkeypatch):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, "1")
    # (?s): where COMGR does not load, the reason lists each library candidate
    # that failed on a line of its own.
    with pytest.raises(
        AssertionError,
        match=f"(?s)^{REQUIRE_COMPILER_TOOLS_ENV}=1 but (comgr unavailable: .*|"
        f"the loaded comgr cannot target {_UNLISTED_ARCH})$",
    ):
        require_comgr_target(_UNLISTED_ARCH)


def test_a_compile_where_comgr_does_not_load_skips(monkeypatch):
    # COMGR reports a library it cannot load as a ComgrError, the type of a
    # compile error too, so only the check before the compile can tell them
    # apart. The ISA query here raises the loader's error, as it does on a host
    # without COMGR.
    def unloadable():
        raise comgr.ComgrError("cannot load libamd_comgr.so; 1 candidate(s) failed")

    monkeypatch.setattr(comgr, "supported_isas", unloadable)
    monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    with pytest.raises(unittest.SkipTest, match="comgr unavailable: cannot load"):
        compile_kernel_or_skip(KernelDef("probe", [], Region("entry")), arch="gfx950")


def test_a_compile_error_after_the_check_belongs_to_the_kernel():
    """An instruction no AMDGPU target has makes COMGR's codegen fail with a
    ComgrError: once COMGR loads and lists the arch, that propagates."""
    require_comgr_target("gfx950")
    builder = IRBuilder("probe_unknown_instruction")
    out = builder.param("Out", PtrType(I32, "global"))
    value = builder.inline_asm("v_no_such_instruction $0", "=v", [], I32)
    builder.global_store(out, builder.thread_id_x(), value)
    with pytest.raises(comgr.ComgrError, match="CODEGEN_BC_TO_RELOCATABLE"):
        compile_kernel_or_skip(builder.kernel, arch="gfx950")
