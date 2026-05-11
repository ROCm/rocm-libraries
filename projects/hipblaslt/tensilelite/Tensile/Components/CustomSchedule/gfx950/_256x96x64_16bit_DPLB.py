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
    isNT,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 96, 64, 2, 1, 1, True, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x96x64_16bit_DPLB(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isNT(kernel) and useLDSTr and TLDS == 0:
        syncTable = [
            -1, SWaitCnt(dscnt= 4, vlcnt=-1, vscnt=-1, comment="wait for all LRA1 and one LRB1"),
             7, SWaitCnt(dscnt= 6, vlcnt=-1, vscnt=-1, comment="wait the rest of LRB1"),
            23, SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="wait for all LR0 before starting 2nd sub-iteration"),
            27, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="wait for GRAs before starting LRA1"),
            27, SBarrier(comment=""),
            41, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="wait for GRBs before starting LRB1"),
            41, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC': [syncTable[::2]],

            'GRIncA': [[0,1,1, 1,2,3, 3,3,4],
                       [0,0,0, 1,2,2, 2,3,4]],
            'GRIncB': [[18,18,18, 21,21,21, 22,22,22]],

            'LRA0': [[0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14],
                     [1,1, 3,3, 5,5, 7,7, 9,9, 11,11, 13,13, 15,15]],
            'LRB0': [[9,11, 13,15, 16,16],
                     [10,12, 14,16, 17,17]],

            'GRA': [[5,5, 5,6, 7, 9, 11,11, 15,16, 19,20, 24,25, 26,27],
                    [4,5, 6,6, 7,10, 11,12, 15,16, 19,20, 24,25, 26,28]],
            'GRB': [[31,31, 35,35, 39,39],
                    [32,32, 36,36, 40,40]],

            'LRA1': [[28,28, 30,30, 32,32, 34,34, 36,36, 37,37, 38,38, 40,40],
                     [29,29, 31,31, 33,33, 35,35, 37,37, 39,39, 41,41, 42,42]],
            'LRB1': [[42,42, 43,43, 45,45],
                     [43,43, 44,44, 46,46]],

            'LRSA': [[26,26,26,27]],
            'LRSB': [[27]],
            'LWSA': [[44,44,44],
                     [44,45,45]],
            'LWSB': [[]],
            'LCC': [[46,46],
                    [45,46]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 11
    else:
        return False, None

    numMfma = 48
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
