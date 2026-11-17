# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: skip a compile-backed test when the loaded COMGR cannot build for its arch.

A host test that compiles a kernel through COMGR calls :func:`require_comgr_target` first. A
missing or unloadable COMGR, or one that does not list the arch (an older ROCm lacks the newest
targets), skips the test with the reason; after the check, any compile error is the kernel's and
fails the test.
"""

from __future__ import annotations

import pytest

__all__ = ["require_comgr_target"]


def require_comgr_target(arch: str) -> None:
    """Skip unless the loaded COMGR can build code objects for ``arch`` (``"gfx942"``)."""
    comgr = pytest.importorskip("rocke.runtime.comgr")
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:
        pytest.skip(f"comgr unavailable: {exc}")
    if arch not in {name.split("--")[-1].split(":")[0] for name in isas}:
        pytest.skip(f"the loaded comgr cannot target {arch}")
