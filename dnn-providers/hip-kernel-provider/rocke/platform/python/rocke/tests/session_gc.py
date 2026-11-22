# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The garbage-collector setting of a rocKE test session.

Building and lowering kernels allocates millions of short-lived IR objects that
sit in reference cycles (a ``Value`` points at the ``Op`` that defines it, the
``Op`` at its results), so only the cyclic collector frees them. At
the interpreter's default young-generation threshold the collector runs every
few hundred allocations and spends a large share of a kernel-building suite's
time walking those objects. A test session raises that one threshold and leaves
the older generations' thresholds as they are. Those count collections of the
next younger generation, so the older generations are collected less often per
allocation too, in proportion.

Each root ``conftest.py`` of the test trees calls :func:`use_test_session_gc`
from ``pytest_configure``, before any test module is collected.
"""

from __future__ import annotations

import gc

__all__ = ["TEST_SESSION_YOUNG_GC_THRESHOLD", "use_test_session_gc"]

#: Net allocations of container objects between young-generation collections.
TEST_SESSION_YOUNG_GC_THRESHOLD = 10_000


def use_test_session_gc() -> None:
    """Raise the young-generation threshold to
    :data:`TEST_SESSION_YOUNG_GC_THRESHOLD`, keeping the older generations'."""
    _, older, oldest = gc.get_threshold()
    gc.set_threshold(TEST_SESSION_YOUNG_GC_THRESHOLD, older, oldest)
