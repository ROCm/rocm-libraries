# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: skip a compile-backed test when the loaded COMGR cannot build for its arch.

A host test that compiles a kernel through COMGR calls :func:`require_comgr_target` first. A
missing or unloadable COMGR, or one that does not list the arch (an older ROCm lacks the newest
targets), skips the test with the reason, or fails it under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``
(:func:`~rocke.tests.compiler_tools.require_tools_or_skip`); after the check, any compile error is
the kernel's and fails the test.
"""

from __future__ import annotations

from ..runtime import comgr
from .compiler_tools import require_tools_or_skip

__all__ = ["require_comgr_target"]


def require_comgr_target(arch: str) -> None:
    """Skip unless the loaded COMGR can build code objects for ``arch`` (``"gfx942"``); fail
    instead under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``."""
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:
        require_tools_or_skip(f"comgr unavailable: {exc}")
    if arch not in {name.split("--")[-1].split(":")[0] for name in isas}:
        require_tools_or_skip(f"the loaded comgr cannot target {arch}")
