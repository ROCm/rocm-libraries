# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fail-or-skip rule of the MLA resource gates, decode's and prefill's.

A resource gate compiles a kernel through comgr and reads the code object with
llvm-objdump and llvm-readelf. Where those tools are missing, a gate that skips
looks green and checks nothing, so a run that supplies them must fail instead:
the CTest lane sets ``ROCKE_REQUIRE_COMPILER_TOOLS=1`` when it does. This is the
one definition of that rule, so the two gates cannot drift apart.

A host whose LLVM version or IR flavor a kernel does not take is a different
case: no tool is missing, the gate has nothing to check, and it skips either
way. That stays with each gate.
"""

from __future__ import annotations

import os
import unittest
from typing import NoReturn

REQUIRE_ENV = "ROCKE_REQUIRE_COMPILER_TOOLS"


def _tools_required() -> bool:
    """Whether this run supplies the compiler tools, and so requires them."""
    return os.environ.get(REQUIRE_ENV) == "1"


def require_tools_or_skip(reason: str) -> NoReturn:
    """Fail where the run requires the compiler tools, else skip.

    ``reason`` names what is missing. Never returns: ``unittest.SkipTest`` is a
    skip under unittest and under pytest alike. Tests:
    ``test_mla_decode.py::TestRequireToolsOrSkip`` and ``::TestSkipIfProbeDeclined``.
    """
    if _tools_required():
        raise AssertionError(f"{REQUIRE_ENV}=1 but {reason}")
    raise unittest.SkipTest(reason)
