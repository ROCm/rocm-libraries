# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: the policy for a compile-backed test whose compiler tools are missing.

A test that compiles a kernel through COMGR or reads a code object with the LLVM tools depends on
the host: the tools must be installed and able to build for the test's arch. Without them the test
skips, because that is a fact about the host, not a defect in the kernel, as waiting for the CI
test environment to supply them (``rocke.tests.lanes.BUILD_VARIANT``). A run that sets
``ROCKE_REQUIRE_COMPILER_TOOLS=1`` (a CI lane that supplies the tools) fails instead: a gate that
skips everywhere looks green and checks nothing.
"""

from __future__ import annotations

import os
import unittest
from typing import NoReturn

from ..analysis.isa import missing_hsaco_tools
from .lanes import BUILD_VARIANT, skip_until_reason

__all__ = ["REQUIRE_COMPILER_TOOLS_ENV", "require_hsaco_tools", "require_tools_or_skip"]

# platform/CMakeLists.txt sets it to "1" only for the installed CTest entries whose tests consume
# it, where ROCM_PATH supplies the tools (Linux); see TESTING.md section 5.1.4.
REQUIRE_COMPILER_TOOLS_ENV = "ROCKE_REQUIRE_COMPILER_TOOLS"


def require_tools_or_skip(reason: str) -> NoReturn:
    """Skip the running test for ``reason``, as waiting for the CI test environment's tools
    (``BUILD_VARIANT``), or fail it with ``AssertionError`` when
    ``ROCKE_REQUIRE_COMPILER_TOOLS=1`` says the tools must be there.

    The skip is raised as ``unittest.SkipTest``, which pytest test functions and fixtures and a
    ``unittest`` ``setUpClass`` all report as a skip."""
    if os.environ.get(REQUIRE_COMPILER_TOOLS_ENV) == "1":
        raise AssertionError(f"{REQUIRE_COMPILER_TOOLS_ENV}=1 but {reason}")
    raise unittest.SkipTest(skip_until_reason(BUILD_VARIANT, reason=reason))


def require_hsaco_tools() -> None:
    """Skip the running test (see :func:`require_tools_or_skip`) unless the tools
    ``rocke.analysis.isa.analyze_hsaco`` runs are on PATH. A sweep whose every
    case analyzes a code object calls it before its first case, so a host
    without them skips the sweep as a whole instead of skipping every case."""
    missing = missing_hsaco_tools()
    if missing:
        require_tools_or_skip(
            f"HSACO introspection unavailable: missing tool: {', '.join(missing)}"
        )
