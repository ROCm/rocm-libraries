################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Unit tests for the InitCIterWmma unrolled-loop entry threshold.

``KernelWriterAssembly.unrollLoopEntryEndCounter`` is the single source shared
by openLoop's loop-entry guard and initC's InitCIterWmma v_mov skip-branch: the
main (global-load) unrolled loop -- and therefore the cloned iter0 that zeroes C
-- runs iff ``LoopCounter > T``. The skip-branch must use the same ``T`` so the
v_mov is skipped exactly when the loop (and its WMMA-based C init) will run.
"""

import pytest

pytestmark = pytest.mark.unit


def _endCounter(pgr, suppress=False, halfPLR=False, rap=False):
    # unrollLoopEntryEndCounter reads nothing but kernel[...], so a plain dict and
    # a bare stub self are sufficient. Import lazily to keep module import light.
    from types import SimpleNamespace

    from Tensile.KernelWriterAssembly import KernelWriterAssembly

    kernel = {
        "PrefetchGlobalRead": pgr,
        "SuppressNoLoadLoop": suppress,
        "HalfPLR": halfPLR,
        "ReuseAcrossPersistent": rap,
    }
    return KernelWriterAssembly.unrollLoopEntryEndCounter(SimpleNamespace(), kernel)


def test_pgr1_threshold():
    assert _endCounter(1) == 1
    assert _endCounter(1, suppress=True) == 0


def test_pgr2_threshold():
    assert _endCounter(2) == 2
    assert _endCounter(2, suppress=True) == 1
    assert _endCounter(2, suppress=True, halfPLR=True) == 0


def test_pgr3plus_threshold():
    assert _endCounter(3) == 3
    assert _endCounter(4) == 4
    # PGR>=3 early-exits to NoGlobalLoadLoop at LoopCounter <= PGR regardless of
    # SuppressNoLoadLoop, so the entry threshold stays PGR.
    assert _endCounter(3, suppress=True) == 3


def test_pgr0_threshold():
    assert _endCounter(0) == 0


def test_rap_threshold_is_zero_whatever_the_prefetch_depth():
    """ReuseAcrossPersistent emits one body copy per resident k-tile and no drain.

    The loop therefore has to keep going until every k-tile has run, rather than
    stopping PrefetchGlobalRead short of the end and handing the rest to sections
    that no longer exist. The threshold has to agree with rapUnrolledLoopCopies, or
    the shell would close before the last copies or spin past them.
    """
    for pgr in (1, 2, 3):
        assert _endCounter(pgr, rap=True) == 0
        assert _endCounter(pgr, suppress=True, rap=True) == 0
