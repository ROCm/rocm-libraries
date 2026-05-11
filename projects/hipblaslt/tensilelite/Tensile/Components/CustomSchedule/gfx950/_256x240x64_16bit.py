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
    isNT,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 240, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x240x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'GRIncA': [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            'GRIncB': [[30, 30, 31, 31, 32, 32, 33, 33, 34]],
            'LRA0': [[0, 1, 1, 2]],
            'LRB0': [[3, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]],
            'GRA': [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
            'GRB': [[35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 89, 90, 90, 91, 91]],
            'LRA1': [[93, 94, 95, 96]],
            'LRB1': [[97, 98, 99, 100, 102, 104, 106, 108, 110, 112, 114, 116, 116, 116, 116]],
            'LRSA': [[59]],
            'LRSB': [[59]],
            'LWSA': [[91]],
            'LWSB': [[91]],
            'LCC': [[119, 119]],
            'SYNC': [[-1, -1, 4, 4, 33, 33, 92, 92]],
        }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SBarrier(comment="wavefront sync at loop start"),
            SWaitCnt(dscnt=13, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 3), before GRA starts (idx 5)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 29), before GRB starts (idx 35)"),
            SWaitCnt(dscnt=-1, vlcnt=38, vscnt=-1, comment="wait for global reads before using data"),
            SBarrier(comment="earlier final barrier to reduce idle time"),
        ]
    elif isNT(kernel) and TLDS==0 and useLDSTr:
        optSchedule = {
            'LRA0': [[0, 1, 1, 2, 2, 3, 3, 4]],
            'LRB0': [[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]],
            'LRA1': [[98, 99, 99, 100, 100, 101, 101, 102]],
            'LRB1': [[98, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119]],
            'GRA': [[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
            'GRB': [[40, 41, 43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95, 97, 98, 100, 101, 103, 104, 106, 107, 109, 110, 112, 113, 115, 116, 118, 119, 119, 119, 119, 119, 119, 119]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LCC': [[119, 119]],
            'LRSA': [[57]],
            'LRSB': [[57]],
            'LWSA': [[96]],
            'LWSB': [[96]],
            'SYNC': [[-1, 6, 6, 38, 38, 96, 96]],
        }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 4), before GRA starts (idx 8)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 33), before GRB starts (idx 40)"),
            SWaitCnt(dscnt=-1, vlcnt=27, vscnt=-1, comment="wait for 54 global reads before idx 96 (16 GRA + 38 GRB). vlcnt = 38 - 11 = 27"),
            SBarrier(comment="barrier at idx 96 - before LRA1/LRB1 start at 98"),
        ]
    elif isNN(kernel) and TLDS==1 and useLDSTr:
        optSchedule = {
                'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
                'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
                'LRA0': [[0, 1, 1, 2, 2, 3, 3, 4]],
                'LRB0': [[1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],
                'GRA': [[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
                'GRB': [[26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114]],
                'LRA1': [[93, 95, 95, 96, 96, 97, 97, 98]],
                'LRB1': [[94, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]],
                'LRSA': [[57]],
                'LRSB': [[57]],
                'LWSA': [[91]],
                'LWSB': [[91]],
                'LCC': [[119, 119]],
                'SYNC': [[-1, 6, 6, 26, 26, 90, 90]],
            }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SWaitCnt(dscnt=13, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 4), before GRA starts (idx 8)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 19), before GRB starts (idx 26)"),
            SWaitCnt(dscnt=-1, vlcnt=30, vscnt=-1, comment="wait for 59 global reads before idx 90 (16 GRA + 43 GRB). vlcnt = 38 - 8 = 30"),
            SBarrier(comment="barrier at idx 90 - before LRA1/LRB1 start at 93/94"),
        ]
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 120  # Must match actual MFMA count for 256x240x64 tile
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
