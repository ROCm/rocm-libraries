# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The compiler-tool skip rule shared by the tests whose checks need comgr or llvm-objdump."""

from __future__ import annotations

import os
import unittest


def require_tools_or_skip(reason: str) -> None:
    """Skip a check whose compiler tools are missing, unless the run requires them
    (``ROCKE_REQUIRE_COMPILER_TOOLS=1``, set by the CI lanes that supply them): a gate
    that skips everywhere looks green and checks nothing.

    ``unittest.SkipTest`` skips a pytest case too, and the ``AssertionError`` fails one,
    so unittest- and pytest-style tests share this helper.
    """
    if os.environ.get("ROCKE_REQUIRE_COMPILER_TOOLS") == "1":
        raise AssertionError(f"ROCKE_REQUIRE_COMPILER_TOOLS=1 but {reason}")
    raise unittest.SkipTest(reason)
