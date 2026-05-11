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
    is16bit,
    isTN,
)


@RegisterSchedule(
   tile_config=TileConfig(192, 128, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x128x64_16bit(kernel, useLDSTr, TLDS):
    """192x128x64 TN schedule (BF16/FP16)."""
    kernel["MfmaInitCVgprs"] = True

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    # 192 = 16 * MIWaveTileA(6) * MIWaveGroup0(2)
    # 128 = 16 * MIWaveTileB(4) * MIWaveGroup1(2)
    if isTN(kernel) and (not useLDSTr) and TLDS == 1:
        numMfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]
        # Number of global reads per iter (A:6, B:4) = 10
        nglshift = nllshift = 10

        # Use syncTable format (idx, wait/barrier, idx, wait/barrier, ...)
        syncTable = [
            # Loop start: must guarantee prior-iteration LR1 completion for the next iteration's early MFMA use.
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 before starting main loop"),
            5,  SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),

            14,  SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
            14,  SBarrier(comment=""),

            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            23, SBarrier(comment=""),

            # GR -> SWait(vmcnt=0) -> SBarrier -> LR1 ordering (global-read to LDS must be visible before LR1)
            28, SWaitCnt(dscnt=-1, vlcnt=10+1, vscnt=-1, comment="Wait for all GRs to complete before starting LR1"),
            28, SBarrier(comment=""),

            44, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for all GRs to complete before starting LRB1"),
            44, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,9,10,10,11,11,12,13,13]],

            'LRA0'   : [[0,1,2,3,4,5]],
            'LRB0'   : [[6,9,12,15]],

            'GRA'    : [[14,14, 16,16, 18,18, 20,20, 22,22, 24,24],
                        [15,15, 17,17, 19,19, 21,21, 23,23, 25,25]],
            'GRB'    : [[26,26, 30,30, 37,37, 43,43]],

            # Prefetch next iteration
            'LRA1'   : [[28,30,32,34,36,38],
                        [29,31,33,35,37,39]],
            'LRB1'   : [[44,45,46,47]],

            'LRSA'   : [[21]],
            'LRSB'   : [[21]],
            'LWSA'   : [[46]],
            'LWSB'   : [[46]],
            'LCC'    : [[47, 47]],
        }

        syncCode = syncTable[1::2]
    else:
        return False, None

    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
