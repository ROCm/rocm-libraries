################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from Tensile.AsmRegisterPool import RegisterPool
from Tensile.Utils import roundUpToNearestMultiple

S = RegisterPool.Status


def test_RegisterPool_add0():
    poolSize = 12

    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)
    for reg in vgprPool.pool:
        assert reg.status == S.Unavailable
        assert reg.tag == "init"

    vgprPool.add(0, poolSize, "tag")
    assert len(vgprPool.pool) == poolSize
    for reg in vgprPool.pool:
        assert reg.status == S.Available
        assert reg.tag == "tag"


def test_RegisterPool_add1():
    poolSize = 12
    start = 8
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(start, poolSize, "tag")
    assert len(vgprPool.pool) == poolSize + start
    for i, reg in enumerate(vgprPool.pool):
        if i < start:
            assert reg.status == S.Unavailable
            assert reg.tag == "init"
        else:
            assert reg.status == S.Available
            assert reg.tag == "tag"


def test_findFreeRange_vgprEndToEnd1_AllAvailable():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(0, poolSize, "tag")
    out_new = vgprPool.findFreeRange(4, 8)
    out_old = findFreeRange_oldLogic(vgprPool, 4, 8)

    assert out_new == out_old == 0


def test_findFreeRange_noOverflowButAvailable():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=True, printRP=0)
    vgprPool.add(6, poolSize, "tag")

    # Starting at index 3, find 4 free registers and return the starting index
    out_new = vgprPool.findFreeRange(3, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 3, 4)

    assert out_new == out_old == 8


def test_findFreeRange_variableOveflowNotAvailable():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(0, 3, "tag")
    vgprPool.add(10, 2, "tag")

    #   A       A       A
    # [ . . . | | | | | | | . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(4, 4, preventOverflow=True)
    out_old = findFreeRange_oldLogic(vgprPool, 4, 4, preventOverflow=True)

    # We expect to hit the preventOverflow here
    assert out_new == out_old == None

    out_new = vgprPool.findFreeRange(4, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 4, 4)

    # We expect to get values from the call to `startOfLastAvailableBlock`
    assert out_new == out_old == 12


def test_findFreeRange_noOverflowNotAvailable():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=True, printRP=0)

    vgprPool.add(7, 5, "tag")

    #   A       A       A
    # [ | | | | | | | . . . . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(5, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 5, 4)

    assert out_new == out_old == None


def test_findFreeRange_noOverflowButPastLastSlot():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(7, 5, "tag")

    #   A       A       A
    # [ | | | | | | | . . . . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(5, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 5, 4)

    assert out_new == out_old
    assert out_new == 8


def test_findFreeRange_overflowPastPoolLength():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=True, printRP=0)

    vgprPool.add(0, poolSize, "tag")

    #   A       A       A
    # [ . . . . . . . . . . . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(16, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 16, 4)

    assert out_new == out_old == None


def test_findFreeRange_noOverflowPastPoolLength():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(0, poolSize, "tag")

    #   A       A       A
    # [ . . . . . . . . . . . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(16, 4)
    out_old = findFreeRange_oldLogic(vgprPool, 16, 4)

    assert out_new == out_old
    assert out_new == 0


def test_findFreeRange_noOverflowPastPoolLength2():
    poolSize = 12
    vgprPool = RegisterPool(poolSize, "v", defaultPreventOverflow=False, printRP=0)

    vgprPool.add(0, poolSize - 2, "tag")

    print(vgprPool.pool)
    #   A       A       A
    # [ | | | | | | | | | | . . ]
    #   0                     11
    out_new = vgprPool.findFreeRange(16, 8)
    out_old = findFreeRange_oldLogic(vgprPool, 16, 8)

    assert out_new == out_old
    assert out_new == 16


# ----------------
# Helper functions
# ----------------
def findFreeRange_oldLogic(
    regPool, size, alignment, preventOverflow=-1, wantedStatus=RegisterPool.Status.Available
):
    if preventOverflow == -1:
        preventOverflow = regPool.defaultPreventOverflow

    for i in range(len(regPool.pool) + 1):
        if i % alignment != 0:
            continue
        if regPool.isRangeAvailable(i, size, preventOverflow, wantedStatus):
            return i

    if preventOverflow:
        return None
    else:
        loc = regPool.startOfLastAvailableBlock()
        return roundUpToNearestMultiple(loc, alignment)
