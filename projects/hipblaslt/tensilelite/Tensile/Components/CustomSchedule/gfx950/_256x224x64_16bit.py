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
    tile_config=TileConfig(256, 224, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x224x64_16bit(kernel, useLDSTr, TLDS):
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    optSchedule = dict()
    syncCode = []
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'SYNC'   : [[ -1,  18,  18,  51,  51,  90,  90]],
            'GRIncA' : [[  1,   1,   3,   3,   5,   5,   7,   7,   9],
                        [  0,   0,   2,   2,   4,   4,   6,   6,   8]],
            'GRIncB' : [[  9,  11,  11,  13,  13,  15,  15,  17,  17],
                        [  8,  10,  10,  12,  12,  14,  14,  16,  16]],
            'LRA0'   : [[  0,   2,   4,   6,   8,  10,  12,  14],
                        [  1,   3,   5,   7,   9,  11,  13,  15]],
            # schduling GRIncA/B and LRA0 as follow,
            # SIMD 0 | ... | MFMA | GRInc  | GRInc  | MFMA | LDS Load            | MFMA | GRInc  | GRInc  | MFMA | ...
            # SIMD 1 | ... | MFMA | LDS Load        | MFMA | GRInc  | GRInc      | MFMA | LDS Load        | MFMA | ...

            'LRB0'   : [[ 16,      19,  21,  23,  25,  27,      29],
                        [ 17,      20,  22,  24,  26,  28,      30]],
            'GRA'    : [[ 19,  20,  21,  22,  23,  24,  25,  26,  27,  28,     46,  47,  48,  49,  52,  53],
                        [ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,     47,  48,  49,  50,  53,  54]],

            'LRA1'   : [[ 52,  56,  58,       77,  79,  81,  83,  85],
                        [ 53,  57,  59,       78,  80,  82,  84,  86]],
            'GRB'    : [[ 54,  55,  56,  57,      77,  78,  79,  80,  81,  82,  83,  84,  85,  86],
                        [ 55,  56,  57,  58,      78,  79,  80,  81,  82,  83,  84,  85,  86,  87]],

            'LRB1'   : [[ 91,  93,  95,  97,  99, 101, 103],
                        [ 92,  94,  96,  98, 100, 102, 104]],
            'LRSA'   : [[ 50], [52]],
            'LRSB'   : [[ 50], [52]],
            'LWSA'   : [[108]],
            'LWSB'   : [[109]],
            'LCC'    : [[110, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="Wait for LRBs"),
            SWaitCnt(dscnt= 1, vlcnt=-1, vscnt=-1, comment="Wait for LRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt= 0, vlcnt=14, vscnt=-1, comment="Wait for LRBs and previous set of GRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous set of GRBs"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC': [[-1,6,
                       21,21,55,55,60,60]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'LRA0': [[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14],
                     [1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]],
            
            'GRIncB': [[9,10,11,12,13,14,15,16,17]],
            
            'GRA': [[21,22, 25,26, 30,31, 35,36, 40,41, 45,46, 50,51, 53,54]],
            'LRB0': [[21,22, 25,26, 30,31, 35,36, 40,41, 45,46, 50,51]],

            'GRB': [[61,61, 63,63, 65,65, 79,79, 85,85, 95,95, 100,100],
                    [62,62, 64,64, 66,66, 80,80, 91,91, 96,96, 101,101]],
            'LWSA': [[93],[99]],
            'LWSB': [[91],[87]],
            'LRA1': [[61,61, 63,63, 65,65, 79,79, 85,85, 93,93, 100,100, 104,104],
                    [62,62, 64,64, 66,66, 80,80, 91,91, 96,96, 101,101, 106,106]],
            'LRB1': [[91,91,95,95,98,98,100,100,110,110,110,111,111,111],
                     [87,87,94,94,99,99,101,101,103,103,105,105,107,107]],

            'LRSA': [[54]],
            'LRSB': [[54]],
            'LCC': [[110,111]],
        }

        syncCode = [
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SWaitCnt(dscnt=0, vlcnt=7, vscnt=-1, comment="wait for previous set of global reads and Local Reads"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=8, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        nglshift = nllshift = 15

    elif isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1"),
                     17, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),
                     17, SBarrier(comment=""),
                    #  46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                     51, SWaitCnt(dscnt=0, vlcnt=15, vscnt=-1, comment="wait for previous set of global reads"),
                     51, SBarrier(comment=""),
                     65, SWaitCnt(dscnt=-1, vlcnt=15-4, vscnt=-1, comment="wait for previous set of global reads"),
                     65, SBarrier(comment=""),
                    ]
        optSchedule = {
                    'SYNC'  : [syncTable[::2]],
                    'GRIncA': [[37,38,39,40,41,42,43,44,45]],
                    'GRIncB': [[0,1,2,3,4,5,6,7,8]],

                    'LRB0': [[-1, 0, 1, 2, 3, 4, 5],
                             [ 0, 1, 2, 3, 4, 5, 6]],
                    'LRA0': [[8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35, 37],
                             [9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36, 38]],
                    'GRA': [[17,17, 19,19, 26,26, 28,28, 30,30, 32,32, 34,34],
                            [18,18, 20,20, 27,27, 29,29, 31,31, 33,33, 35,35]],

                    'GRB': [[52,52, 54,54, 56,56, 58,58, 66,66, 68,68, 70,70, 72,72],
                            [53,53, 55,55, 57,57, 59,59, 67,67, 69,69, 71,71, 73,73]],
                    'LRB1': [[51, 53, 55, 57, 59, 61, 63],
                             [52, 54, 56, 58, 60, 62, 64]],
                    'LRA1': [[65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95],
                             [66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96]],

                    'LRSB': [[14]],
                    'LRSA': [[45]],
                    'LWSB': [[97]],
                    'LWSA': [[97]],
                    'LCC' : [[110, 110]],
                }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 15 # vmcnt shift for ngl and nll

    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
