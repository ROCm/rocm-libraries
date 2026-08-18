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


def _endCounter(pgr, suppress=False, halfPLR=False):
    # unrollLoopEntryEndCounter reads only kernel[...] (not self), so a dummy
    # self (None) and a plain dict kernel are sufficient. Import lazily to keep
    # module import light.
    from Tensile.KernelWriterAssembly import KernelWriterAssembly

    kernel = {
        "PrefetchGlobalRead": pgr,
        "SuppressNoLoadLoop": suppress,
        "HalfPLR": halfPLR,
    }
    return KernelWriterAssembly.unrollLoopEntryEndCounter(None, kernel)


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
