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
    is8bit,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 256, 128, 2, 0, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is8bit,
    vector_widths=[16, 16, 16],
    matrix_inst=[16, 16, 128, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x128_8bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    plr = 3 if kernel["ForceUnrollSubIter"] else 1
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS == 1:
        optSchedule = {
            'SYNC'      : [[6,7, 20,21, 46,47, 61]],
            'GRIncA'    : [[0,1,2,3,4,4,4,4,4]],
            'GRIncB'    : [[5,5,5,5,5,6,6,6,6]],
            'LRA0'      : [[0,0, 1,1, 2,2, 3,3]],
            'GRA'       : [[8,8,9,9,10,10,11,11,12,12, 23,23,24,24,25,25]],
            'LRB0'      : [[13,13,14,14,15,15,16,16]],
            'LRA%u'%plr : [[48,48,49,49,50,50,51,51]],
            'LRB%u'%plr : [[52,52,54,54,55,55,56,56]],
            'GRB'       : [[26,26,27,27, 39,39,40,40,41,41,42,42,43,43, 53,53]],
            'LCC'       : [[60, 60]],
            'LRSA'      : [[17]],
            'LRSB'      : [[17]],
            'LWSA'      : [[57]],
            'LWSB'      : [[57]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for PLR to complete")]
        nglshift = nllshift = 16
    else:
        return False, None

    numMfma = 64
    # B0A0, B0A1, B1A0, B1A1
    mfmaReorder = []
    kernel["MfmaInitCVgprs"] = True
    if not kernel["ForceUnrollSubIter"]:
        mfmaReorder = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27,
                       4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31,
                       32,33,34,35, 40,41,42,43, 48,49,50,51, 56,57,58,59,
                       36,37,38,39, 44,45,46,47, 52,53,54,55, 60,61,62,63]
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder)
    return True, opt1
