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
    tile_config=TileConfig(224, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x256x64_16bit(kernel, useLDSTr, TLDS):
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
            'LRA0'   : [[  0,   2,   4,   6,   8,  10,  12],
                        [  1,   3,   5,   7,   9,  11,  13]],

            'LRB0'   : [[ 14,  16,      19,  21,  23,  25,  27,      29],
                        [ 15,  17,      20,  22,  24,  26,  28,      30]],
            'GRA'    : [[ 19,  20,  21,  22,  23,  24,  25,  26,  27,  28,     46,  47,  48,  49],
                        [ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,     47,  48,  49,  50]],

            'LRA1'   : [[ 56,  58,       77,  79,  81,  83,  85],
                        [ 57,  59,       78,  80,  82,  84,  86]],
            'GRB'    : [[ 52,  53,  54,  55,  56,  57,      77,  78,  79,  80,  81,  82,  83,  84,  85,  86],
                        [ 53,  54,  55,  56,  57,  58,      78,  79,  80,  81,  82,  83,  84,  85,  86,  87]],

            'LRB1'   : [[ 91,  93,  95,  97,  99, 101, 103, 105],
                        [ 92,  94,  96,  98, 100, 102, 104, 106]],
            'LRSA'   : [[ 50], [52]],
            'LRSB'   : [[ 50], [52]],
            'LWSA'   : [[108]],
            'LWSB'   : [[109]],
            'LCC'    : [[110, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="Wait for LRBs"),
            SWaitCnt(dscnt= 2, vlcnt=-1, vscnt=-1, comment="Wait for LRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt= 0, vlcnt=15, vscnt=-1, comment="Wait for LRBs and previous set of GRs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous set of GRs"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC'   : [[-1, 21, 21, 51, 51, 79, 79]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,10,11,12,13,14,15,16,17]],

            'LRA0'   : [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            'LRB0'   : [[14, 17, 20, 23, 26, 29, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]],

            'GRA'    : [[22,22, 26,26, 30,30, 34,34, 38,38, 42,42, 46,46]],
            'GRB'    : [[52,53, 56,57, 60,61, 64,65, 68,69, 71,72, 74,75, 77,78]],

            'LRA1'   : [[79,80, 81,82, 83,84, 85,86, 87,88, 89,90, 91,92]],
            'LRB1'   : [[93,94, 95,96, 97,98, 99,100, 101,102, 103,104, 105,106, 107,108]],
            'LRSA'   : [[54]],
            'LRSB'   : [[54]],
            'LWSA'   : [[91]],
            'LWSB'   : [[91]],
            'LCC'    : [[111, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNN(kernel) and useLDSTr and TLDS == 1:
        optSchedule = {
            'SYNC': [[  -1,6,
                        23,23,
                        55,55,
                        94,
                    ]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'GRIncB': [[9,10,11,12,13,14,15,16,17]],
            'LRA0': [[0,1,2,3,4,5,6,7,8,9,10,11,12,13]],
            'GRA': [[23,23,27,27,32,32,37,37,42,42,46,46,51,51],
                    [25,25,29,29,34,34,39,39,44,44,48,48,53,53]],
            'LRB0': [[23,27,32,37,42,46,51,54],
                     [25,29,34,39,44,48,53,54]],
            
            'LWSA': [[87],[86]],
            'LWSB': [[90],[89]],
            'LRA1': [[57,57,61,61,65,65,75,75,81,81,87,90,96,96],
                     [55,55,59,59,63,63,69,69,79,79,86,89,94,94]],                     
            'GRB': [[56,56, 60,60, 70,70, 80,80, 90,92, 100,100, 106,106, 109,109],
                    [58,58, 62,62, 72,72, 82,82, 91,93, 101,101, 107,107, 110,110]],                    
            'LRB1': [[91,92,97,100,102,103,106,109]],
            'LRSA': [[54]],
            'LRSB': [[54]],
            'LCC': [[109,110]],
        }

        syncCode = [
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for 8-5 local reads // oldleft=8, completed=3"),
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for 11-6 local reads // oldleft=5, new=6, completed=5"),
            SWaitCnt(dscnt=0, vlcnt=8, vscnt=-1,  comment="wait for prior global reads and local reads // oldleft=6, new=8, completed=14"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=7, vscnt=-1,  comment="wait for prior global reads and local reads // oldleft=0, new=8, completed=8"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for 14-5 local reads // oldleft=0, new=14, completed=9"),
        ]

        nglshift = nllshift = 15    
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
