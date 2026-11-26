# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for the COMGR test helper and the compiler-tool policy it applies.

``require_comgr_target`` skips a compile-backed test when the loaded COMGR cannot build for
its arch, and fails it instead under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``, so a gate whose tools
are required cannot pass by skipping every case. ``gfx0`` is an arch no COMGR lists; on a host
without COMGR the reason is that COMGR is unavailable, which the policy treats the same way.
"""

from __future__ import annotations

import unittest

import pytest
from rocke.tests.comgr import require_comgr_target
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
