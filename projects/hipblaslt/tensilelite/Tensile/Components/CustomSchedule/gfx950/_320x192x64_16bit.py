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
    tile_config=TileConfig(320, 192, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_320x192x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        syncTable = [
            -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1 "),
            19, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="before DirectToLds load, ensure LRB0 have finished"),
            19, SBarrier(comment=""),
            52, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 finish"),
            52, SBarrier(comment=""),
            53, SWaitCnt(dscnt=-1, vlcnt=16+1, vscnt=-1, comment="wait for previous GRB finish"),
            53, SBarrier(comment=""),
            71, SWaitCnt(dscnt=-1, vlcnt=16-8, vscnt=-1, comment="wait for previous GRA finish"),
            71, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC': [syncTable[::2]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'GRIncB': [[9,10,11,12,13,14,16,16,16]],

            'LRB0': [[0, 1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5, 6]],
            'LRA0': [[6, 8, 10, 12, 14, 16, 18,  20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44],
                     [7, 8, 11, 13, 15, 17, 18,  21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]],
            'GRA': [[19,19, 21,21, 28,28, 31,31, 33,33, 41,41],
                    [20,20, 22,22, 29,29, 32,32, 34,34, 42,42]],
            'GRB': [[52,52, 64,64, 74,74, 78,78, 83,83, 88,88, 93,93, 98,98, 105,105, 109,109],
                    [52,52, 65,65, 75,75, 79,79, 84,84, 89,89, 94,94, 99,99, 106,106, 110,110]],
            'LRB1': [[53, 55, 57, 59, 63, 67],
                     [54, 56, 58, 60, 64, 68]],
            'LRA1': [[71,73, 75,77, 79,81, 83,85, 87,89, 91,93, 95,97, 99,101, 103,105, 107,109],
                     [72,74, 76,78, 80,82, 84,86, 88,90, 92,94, 96,98, 100,102, 104,106, 108,110]],
            'LRSB': [[16]],
            'LRSA': [[48]],
            'LWSB': [[48]],
            'LWSA': [[112]],
            'LCC': [[119, 119]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 16
    elif isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        optSchedule = {
            'SYNC'  : [[-1, 4, 16, 16, 50, 50, 54, 55, 84, 85]],
            'GRIncA': [[0,  1,  2,  3,  4,  5,  6,  7,  8]],
            'GRIncB': [[9, 10, 11, 12, 13, 14, 16, 16, 16]],
            'LRB0'  : [[0, 2, 4, 6, 8, 10],
                       [1, 3, 5, 7, 9, 11]],
            'LRA0'  : [[12, 14, 24, 26, 28, 30, 32, 34, 36, 38],
                       [13, 15, 25, 27, 29, 31, 33, 35, 37, 39]],
            'GRA'   : [[16, 16, 18, 18, 20, 20, 22, 22, 46, 46, 48, 48],
                       [17, 17, 19, 19, 21, 21, 23, 23, 47, 47, 49, 49]],
            'GRB'   : [[50, 50, 52, 52, 74, 75, 78, 78, 80, 80, 82, 82, 106, 106, 108, 108, 110, 110, 112, 112],
                       [51, 51, 53, 53, 76, 77, 79, 79, 81, 81, 83, 83, 107, 107, 109, 109, 111, 111, 113, 113]],
            'LRB1'  : [[56, 58, 60, 62, 64, 66],
                       [57, 59, 61, 63, 65, 67]],
            'LRA1'  : [[86, 88, 90, 92, 94, 96, 98, 100, 102, 104],
                       [87, 89, 91, 93, 95, 97, 99, 101, 103, 105]],
            'LRSA'  : [[44]],
            'LRSB'  : [[45]],
            'LWSA'  : [[115]],
            'LWSB'  : [[117]],
            'LCC'   : [[119, 119]],
        }

        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior local read. Relax a bit to dscnt=4 to reduce latency") ,
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for all prior local read requested as the input for MFMA") ,
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRB0 before GRA (GR for MatrixB). Skip LRA0*2") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA0 before GRB (GR for MatrixA)") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=18, vscnt=-1, comment="Wait for GRA (GR for MatrixB) from previous iteration before LRB1. Skip GRB*10 from last iter and GRA*6 + GRB*2 from this iter.") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="Wait for GRB (GR for MatrixA) from previous iteration before LRA1. Skip GRA*6 + GRB*6 from this iter.") ,
            SBarrier(comment="") ,
        ]
        nglshift = nllshift = 16
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        kernel["SwapGlobalReadOrder"] = True
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        optSchedule = {
            'SYNC'  : [[-1, 7, 17, 17, 49, 49, 59, 59]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRB0'  : [[0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10],
                       [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11]],
            'LRA0'  : [[11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 31, 33, 33, 35, 37, 39, 41, 43, 45],
                       [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 36, 38, 38, 40, 42, 44, 46]],
            'GRA'   : [[18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28],
                       [19, 19, 21, 21, 23, 23, 25, 25, 27, 27, 29, 29]],
            'GRB'   : [[49, 49, 51, 51, 53, 53, 55, 55, 57, 57, 89, 89, 91, 91, 93, 93, 95, 95, 97, 97],
                       [50, 50, 52, 52, 54, 54, 56, 56, 58, 58, 90, 90, 92, 92, 94, 94, 96, 96, 98, 98]],
            'LRB1'  : [[60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82],
                       [61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83]],
            'LRA1'  : [[85, 87, 89, 91, 93, 95, 97,  99, 101, 103, 103, 105, 105, 107, 107, 109, 111, 113, 115, 117],
                       [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 106, 108, 108, 110, 110, 112, 114, 116, 118],],
            'LRSA'  : [[58]],
            'LRSB'  : [[58]],
            'LWSA'  : [[99]],
            'LWSB'  : [[99]],
            'LCC'   : [[119, 119]],
        }

        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior local read. Relax a bit to dscnt=4 to reduce latency") ,
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for remaining LRA1.") ,
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 prior to  LRA0*3") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=0,  vlcnt=-1, vscnt=-1, comment="Wait for prior local read") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prior GRA*6 + GRB*5 = 11 global reads") ,
            SBarrier(comment="") ,
        ]
        nglshift = nllshift = 16
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 120
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
