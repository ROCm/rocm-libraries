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
    tile_config=TileConfig(192, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isNN(kernel) and useLDSTr and TLDS==1:
        # TODO: This schedule can be improved when BC are resolved for MT192
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'    : [[12,13, 47,48,49,50,51, 52,53, 56,56, 95]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncA' : [[42,42,43,43,44,44,45,45,46]],
            'LRB0'   : [[0,0,1,1,2,2,6,8],
                        [3,3,4,4,5,5,7,9]],
            # These local reads have BC
            'LRA0'   : [[10, 15,17,19,21,23, 25,27,29,33,37,39],
                        [11, 14,16,18,20,22, 24,26,28,32,36,38]],
            'GRA'    : [[14,14, 16,16, 18,18, 20,20, 22,22, 34,34,36,36,38,38],
                        [15,15, 17,17, 19,19, 21,21, 23,23, 35,35,37,37,39,39]],
            'GRB'    : [[54,54, 56,56, 58,58, 60,60, 62,62, 64,64],
                        [55,55, 57,57, 59,59, 61,61, 63,63, 65,65]],
            'LRSA'   : [[40]],
            'LRSB'   : [[40]],
            'LWSB'   : [[41]], # For B
            'LWSA'   : [[66]], # For A
            'LRB1'   : [[57,57,59,59,61,61,63,65],
                        [58,58,60,60,62,62,64,64]],
            'LRA1'   : [[67,71,73,75,77,79,81,85,87,89,91,93],
                        [68,72,74,76,78,80,82,86,88,90,92,94]],
            'LCC'    : [[95, 95]],
        }
        syncCode = [SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=10, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRA & GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 & LRB1 to complete"),]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isTN(kernel) and not useLDSTr and TLDS == 1:
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="for LRB1-0"),
                      5, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="for LRB1"),
                     14, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="for LRA0 complete"),
                     14, SBarrier(comment="for GRA start"),
                     46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="for LRB0"),
                     46, SBarrier(comment="for GRB start"),
                     50, SWaitCnt(dscnt=-1, vlcnt=14+1, vscnt=-1, comment="for LRA1"),
                     50, SBarrier(comment="for LRA1 start"),
                     65, SWaitCnt(dscnt=-1, vlcnt=6+5, vscnt=-1, comment="for LRB0"),
                     65, SBarrier(comment="for LRB1 start"),]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[6,6,7,7,8,8,9,9,9]],
                'GRIncB': [[33,34,35,36,37,38,39,40,41]],

                'LRA0'  : [[0, 1, 2, 3, 4, 5],
                           [-1, 0, 1, 2, 3, 4]],
                'LRB0'  : [[7, 9, 11, 13, 15, 17, 19, 21],
                           [8, 10, 12, 13, 16, 18, 20, 22]],
                'GRA'   : [[14,14, 16,16, 18,18, 20,20, 25,25, 31,31],
                           [15,15, 17,17, 19,19, 21,21, 26,26, 32,32]],

                'GRB'   : [[46,46, 50,50, 54,54, 58,58, 62,62, 66,66, 70,70, 76,76],
                           [47,47, 51,51, 55,55, 59,59, 63,63, 67,67, 71,71, 77,77]],
                'LRA1'  : [[50, 52, 56, 58, 60, 62],
                            [51, 53, 57, 59, 61, 63]],
                'LRB1'  : [[65, 67, 69, 71, 73, 75, 77, 79],
                           [66, 68, 70, 72, 74, 76, 78, 80]],

                'LRSA'  : [[47]],
                'LRSB'  : [[47]],
                'LWSA'  : [[47]],
                'LWSB'  : [[80]],
                'LCC'   : [[95, 95]],
            }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isNT(kernel) and not useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC'  : [[-1, 25, 25, 46, 46, 55, 55, 72, 72]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRA0'  : [[0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22],
                       [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17, 19, 19, 21, 21, 23, 23]],
            'LRB0'  : [[24, 24, 26, 26, 28, 28, 30 ,30],
                       [25, 25, 27, 27, 29, 29, 31, 31]],
            'GRA'   : [[25, 25, 27, 27, 29, 29, 31, 31, 33, 33, 35, 35],
                       [26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 36, 36]],
            'GRB'   : [[47, 47, 49, 49, 51, 51, 53, 53, 64, 64, 66, 66, 68, 68, 70, 70],
                       [48, 48, 50, 50, 52, 52, 54, 54, 65, 65, 67, 67, 69, 69, 71, 71]],
            'LRA1'  : [[55, 55, 57, 57, 59, 59, 61, 61, 63, 63, 65, 65, 67, 67, 69, 69, 87, 87, 89, 89, 91, 91, 93, 93],
                       [56, 56, 58, 58, 60, 60, 62, 62, 64, 64, 66, 66, 68, 68, 70, 70, 88, 88, 90, 90, 92, 92, 94, 94]],
            'LRB1'  : [[72, 74, 76, 78, 80, 82, 84, 86],
                       [73, 75, 77, 79, 81, 83, 85, 87]],
            'LRSA'  : [[37]],
            'LRSB'  : [[37]],
            'LWSA'  : [[71]],
            'LWSB'  : [[71]],
            'PackB1': [[-1, -1, -1, -1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]],
            'PackA1': [[-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]],
            'PackB0': [[47, 47, 47, 47, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53]],
            'PackA0': [[47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50]],
            'LCC'   : [[95, 95]],
        }

        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete") ,
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=-1, vlcnt=14+4, vscnt=-1, comment="Wait for global reads to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="Wait for global reads to complete") ,
                    SBarrier(comment="") ,
        ]
        nglshift = nllshift = 14
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 96
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
