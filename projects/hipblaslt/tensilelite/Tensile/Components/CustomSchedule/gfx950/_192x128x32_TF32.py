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
    create_range,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(192, 128, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x128x32_TF32(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and useLDSTr and TLDS==1:

        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        # Used the following constrains to create schedule
        #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs - index 18
        #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs - index 36
        #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs - index 36
        #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs - index 54
        syncTable = [
                    -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),

                    6, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRA0"),
                    11, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA0"),

                    18, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRB0"),
                    23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0"),
                    23, SBarrier(comment="Barrier before GRA&GRB"),

                    44, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for prev GRA&GRB"),
                    44, SBarrier(comment=""),

                    49, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRB3"),
                    56, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB3"),


                    57, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for prev GRA&GRB"),
                    57, SBarrier(comment=""),

                    61, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRA3"),
                    66, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA3"),

                    ]

        syncCode = syncTable[1::2]

        optSchedule = {
            'SYNC'  : [syncTable[::2]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs - index 18
            'LRA0': [[0,1,2,3,4,5]],
            'PackA0' : [[7]*4 + [8]*10 + [9]*10 + [12]*4 + [13]*20 + [14]*8 + [15]*8 + [16]*8],
            #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs - index 36
            'LRB0': [[12,13,14,15]],
             # First two LRB0 need to be done at 18, all LRB0 done by 23
            'PackB0' : [create_range(19,4,22, repeat=3) +  create_range(24,12,35, repeat=3) ],
            'GRB': [[36,36,38,38,40,40,42,42],
                    [37,37,39,39,41,41,43,43]],
            'GRA': [[45,45,47,47,49,49,51,51,53,53,55,55],
                    [46,46,48,48,50,50,52,52,54,54,56,56]],
            'LRSA': [[36]],
            'LRSB': [[36]],
            'LWSA': [[52]],
            'LWSB': [[52]],
            'LCC': [[71, 71]],
            #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs - index 36
            'LRB3': [[45, 46, 47, 48]],
             # First two LRB3 need to be done by 43, all LRB3 done at 48 
            'PackB3' : [create_range(50,12,71, repeat=4)],
            #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs - index 54
            'LRA3': [[58,59,60,62,63,64]],
             # First two LRA3 need to be done by 61. All LRA0 done by 66.
            'PackA3' : [create_range(62,12,71, repeat=6)],

        }

        nglshift = nllshift = 10 # vmcnt shift for ngl and nll
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 72
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
