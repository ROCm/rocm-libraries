################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.instruction import SBarrier, SWaitCnt
from ..dispatch import RegisterSchedule
from ..shared import (
    ScheduleInfo,
    TileConfig,
    create_range,
    is16bit,
    isNN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 64
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isNN(kernel) and useLDSTr and TLDS == 1:
        lra0 = [create_range(min_val = 1, num = 4, step = 2, repeat = 2),
                create_range(min_val = 0, num = 4, step = 2, repeat = 2)]

        GRIncA = [create_range(min_val = 2, num = 3, step = 2, repeat = 3),
                  create_range(min_val = 1, num = 3, step = 2, repeat = 3)]

        waitLRA0 = max(lra0[1])+5
        gra = create_range(min_val = waitLRA0+1, num = 4, step = 2, repeat = 2)
        lrb0 = create_range(min_val = max(gra)+1, num = 8, step = 1, repeat = 1)
        GRIncB = create_range(min_val = max(gra)+1, num = 9, step = 1, repeat = 1)

        assert max(lrb0) < numMfma // 2, "lrb0 max {} numMfma/2 {}".format(max(lrb0), numMfma//2)

        startGRB = max(lrb0) + 5

        assert startGRB < numMfma // 2, "startGRB {} numMfma/2 {}".format(startGRB, numMfma//2)
        grb = create_range(min_val = startGRB, num = 4, step = 2, repeat = 2)
        startLRA1 = max(grb) + 3

        lra1 = create_range(min_val = startLRA1, num = 8, step = 1, repeat = 1)
        startLRB1 = max(lra1) + 1
        grb += create_range(min_val = startLRB1, num = 4, step = 2, repeat = 2)
        lrb1 = create_range(min_val = startLRB1+1, num = 4, step = 2, repeat = 1)
        lrb1 += create_range(min_val = max(lrb1)+2, num = 4, step = 1, repeat = 1)
        syncTable = [
            -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 & LRB0"),
            waitLRA0,  SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0"),
            waitLRA0, SBarrier(comment=""),

            startGRB-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
            startGRB-1, SBarrier(comment=""),
            startLRA1-1, SWaitCnt(dscnt=-1, vlcnt=16, vscnt=-1, comment="wait for previous GRA & GRB"),
            startLRA1-1, SBarrier(comment=""),

            startLRB1-1, SWaitCnt(dscnt=-1, vlcnt=8, vscnt=-1, comment="wait for previous GRA & GRB"),
            startLRB1-1, SBarrier(comment="")
        ]

        optSchedule = {
            'GRA': [gra],
            'GRB': [grb],
            'GRIncA': [*GRIncA],
            'GRIncB': [GRIncB],
            'LCC': [[numMfma-2,numMfma-2]],
            'LRA0': [*lra0],
            'LRA1': [lra1],
            'LRB0': [lrb0],
            'LRB1': [lrb1],
            'LRSA': [[startGRB-1]],
            'LRSB': [[startGRB-1]],
            'LWSA': [[numMfma-3]],
            'LWSB': [[numMfma-3]],
            'SYNC': [syncTable[::2]],
        }

        syncCode = syncTable[1::2]
        nglshift = nllshift = 12 
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        # No matching variant found
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
