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
    inflight,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 128, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x128x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isTN(kernel) and useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        kernel["UsePLRPack"] = True
        numPackInstr = 24 
        numPackIndices = numPackInstr // 2 # Assign 2 pack instructions per mfma index

        # LRA0 + PACKA0 - done before 1/4 MFMAs - index 24
        lrA0 = [0,0, 1,1, 2,2, 3,3]
        waitLRA0 = max(lrA0) + 2
        startPACKA0 = waitLRA0
        packA0 = create_range(startPACKA0, (len(lrA0)//2)*numPackIndices, numMfma//4-1)

         # LBR0 + PACKB0 - done before 2/4 MFMAs - index 48
        lrB0 = [7,7, 15,15]
        waitLRB0 = max(lrB0) + 2
        startPACKB0 = max(waitLRB0,max(packA0)) # Starts after waitLRB0 and packA0
        packB0 = create_range(startPACKB0, (len(lrB0)//2)*numPackIndices, numMfma//2-1)

        # LRB3 + PACKB3 - start after 2/4 MFMAs - index 48
        halfMFMA = numMfma//2
        startLRB3 = halfMFMA
        lrB3 = create_range(startLRB3, 1, numMfma-1)
        lrB3 += create_range(max(lrB3)+6, 1, numMfma-1)
        waitLRB3 = startLRB3 + 4
        packB3 = create_range(waitLRB3, (len(lrB3)//2)*numPackIndices, numMfma-1)

        # LRA3 + PACKA3 - start after 3/4 MFMAs - index 72
        startLRA3 = (3*numMfma)//4
        lrA3 = create_range(startLRA3, 4, numMfma-1)
        waitLRA3 = startLRA3 + 4
        packA3 = create_range(waitLRA3, (len(lrA3)//2)*numPackIndices, numMfma-1)

        syncTable = [
            waitLRA0, SWaitCnt(dscnt=inflight(lrA0, waitLRA0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRA0 to complete"),
            waitLRA0+numPackIndices, SWaitCnt(dscnt=inflight(lrA0, waitLRA0+numPackIndices), vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete"),

            waitLRB0, SWaitCnt(dscnt=inflight(lrB0, waitLRB0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRB0 to complete"),
            waitLRB0+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0 to complete"),
            waitLRB0+numPackIndices, SBarrier(comment="Barrier before GRA&GRB"),

            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRB3-1, SBarrier(comment=""),

            waitLRB3,SWaitCnt(dscnt=inflight(lrB3, waitLRB3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRB3 to complete"),
            waitLRB3+numPackIndices,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

            startLRA3, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRA3, SBarrier(comment=""),

            waitLRA3, SWaitCnt(dscnt=inflight(lrA3,waitLRA3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRA3 to complete"),
            waitLRA3+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA3 to complete")
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[0,0,0, 1,1,1, 2,2,2]],
            'GRIncB' : [[3,3,3, 4,4,4, 5,5,5]],

            'LRA0'   : [lrA0],
            'PackA0' : [packA0],
            'LRB0'   : [lrB0],
            'PackB0' : [packB0],

            'GRA': [[48, 48, 50, 50, 52, 52, 54, 54, 66, 66, 68, 68, 70, 70, 72, 72]],
            'GRB': [[30, 32, 34, 36, 40, 42, 44, 46]],

            'LRA3'   : [lrA3],
            'PackA3' : [packA3],
            'LRB3'   : [lrB3],
            'PackB3' : [packB3],

            'LRSA': [[22]],
            'LRSB': [[22]],
            'LWSA': [[70]],
            'LWSB': [[70]],
            'LCC': [[95, 95]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 12 # vmcnt shift for ngl and nll
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
