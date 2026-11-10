# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The skip policy every compile test shares (``_compiler_test_utils``).

CPU only. These tests pin that a missing tool skips, and that it fails when
``ROCKE_REQUIRE_COMPILER_TOOLS=1`` requires the tools.
"""

from __future__ import annotations

import unittest

import pytest

from ._compiler_test_utils import REQUIRE_COMPILER_TOOLS_ENV, require_tools_or_skip

REASON = "llvm-objdump not found"


def test_missing_tool_skips(monkeypatch):
    monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    with pytest.raises(unittest.SkipTest, match=f"^{REASON}$"):
        require_tools_or_skip(REASON)


def test_missing_tool_fails_where_tools_are_required(monkeypatch):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, "1")
    try:
        with pytest.raises(
            AssertionError, match=f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but {REASON}$"
        ):
            require_tools_or_skip(REASON)
    except unittest.SkipTest as skipped:
        # A skip escaping here would be reported as a skip, not a failure,
        # hiding a helper that lost its fail branch.
        pytest.fail(f"skipped where the tools are required: {skipped}")
