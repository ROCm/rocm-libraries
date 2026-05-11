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
    tile_config=TileConfig(224, 320, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x320x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 140
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False

    if isTN(kernel) and useLDSTr and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=9, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=9 newLW=0 newLR=9 for iteration == 0"),
            6, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write"),
            25, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LR0 before DTL"),
            25, SBarrier(comment=""),
            78, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="wait for prev iter GR before LRA1 and LRB1"),
            78, SBarrier(comment=""),
        ]
        optSchedule = {
            'SYNC': [syncTable[::2]],
            'GRIncA': [[0, 1, 2, 3, 4, 5, 6, 7, 8]], # 9
            'GRIncB': [[9, 10, 11, 12, 13, 14, 15, 16, 17]], # 9

            'LRA0': [[0, 3, 6, 9, 12, 15, 18]], # 7
            'LRB0': [[1, 2, 4, 5, 7, 8, 10, 11, 13, 17]], # 10

            'GRA': [[25,25, 30,30, 36,36, 42,42, 48,48, 54,54, 60,60]], # 14
            'GRB': [[62,62, 67,67, 72,72, 77,77, 88,88, 94,94, 100,100, 106,106, 112,112, 118,118]], # 20

            'LRA1': [[79, 81, 82, 83, 84, 85, 86]], # 7
            'LRB1': [[120, 121, 122, 123, 124, 125, 126, 127, 128, 129]], # 10

            'LRSA': [[66]], # 1
            'LRSB': [[66]], # 1
            'LWSA': [[118]], # 1
            'LWSB': [[118]], # 1
            'LCC': [[138, 138]], # 2
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 17

    else:
        return False, None

    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
