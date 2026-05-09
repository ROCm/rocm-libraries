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
    isNN,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x96x64_16bit(kernel, useLDSTr, TLDS):

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isTN(kernel) and TLDS == 1:

        nglshift = nllshift = 11
        syncTable = [
                    -1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish all LRA1 and 1/3 LRB1"),
                    7, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Finish 2/3 LRB1"),

                    15, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="All LRB1 and LRA0 done"),
                    15, SBarrier(comment=""),

                    23, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="1/3 LRB0 done"),

                    29, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRB0 done"),
                    29, SBarrier(comment=""),

                    35, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRA launched, 3 prev GRB."),
                    35, SBarrier(comment=""),

                    42, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Only global reads for this iter"),
                    42, SBarrier(comment="")]

        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[1,1,2,2,3,3,3,4,4]],
            'GRIncB' : [[5,5,6,6,6,7,7,8,8]],
            'LRA0'   : [[1,2,3,4,5,6,  8,10],
                        [1,2,3,4,5,6,  9,11]],
            'LRB0'   : [[12,16,18],
                        [13,17,19]],
            'GRB'    : [[36,36,38,38,40,40],
                        [37,37,39,39,41,41]],
            'GRA'    : [[16,16,18,18,20,20,22,22,24,24,26,26,28,28,30,30],
                        [17,17,19,19,21,21,23,23,25,25,27,27,29,29,31,31]],
            'LRA1'   : [[36,37,38,39,40,41,42,43]],
            'LRB1'   : [[44,45,46]],
            'LRSA'   : [[30]], # this must come before next reads of A X0 - so the LRA1
            'LRSB'   : [[31]], # this must come before next reads of A X0 - so the LRB1
            'LWSA'   : [[32]],  # swap after last gr a
            'LWSB'   : [[42]],  # swap after last gr b
            'LCC'   : [[47, 47]],
        }
    elif isNN(kernel) and useLDSTr and TLDS == 1:

        nglshift = nllshift = 11

        syncTable = [
            -1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 in prev iteration"),
            
            7, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior 5 LRA0"),
            20, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRA0 is launched"),
            20, SBarrier(comment=""),
            
            21, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="All LRB0 launched"),
            21, SBarrier(comment=""),

            36, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRA launched"),
            36, SBarrier(comment=""),

            43, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRB launched"),
            43, SBarrier(comment=""),
        ]
        
        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            
            'GRIncA' : [[1,1,1, 2,2,2, 3,3,3]],
            'GRIncB' : [[4,4,4, 5,5,5, 6,6,6]],
            
            'LRA0'   : [[1, 3,3, 5,5,   7,7, 9,9, 11,11, 13,13, 15,15, 17],
                        [2, 4,4, 6,6,   8,8, 10,10, 12,12, 14,14, 16,16, 18]],
            'LRB0'   : [[13, 15, 17],
                        [14, 16, 18]],

            'GRA'    : [[21,21, 23,23, 25,25, 27,27, 29,29, 31,31, 33,33, 35,35],
                        [20,20, 22,22, 24,24, 26,26, 28,28, 30,30, 32,32, 34,34]],
            'GRB'    : [[37,37, 39,39, 41,41],
                        [38,38, 40,40, 42,42]],

            'LRSA'   : [[30]],
            'LRSB'   : [[31]],

            'LWSA'   : [[36]],
            'LWSB'   : [[43]],

            'LRA1'   : [[36,36, 37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43]],
            'LRB1'   : [[43, 44, 45]],

            'LCC'    : [[47, 47]],
        }
    else:
        return False, None

    numMfma = 48
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
