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
    tile_config=TileConfig(240, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_240x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    if isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = False
        optSchedule = {
            'SYNC': [[-1,
                      14,
                      26,26,
                      59,
                      69,69,
                      98,98]],
            'LRA0': [[0,2,3,4,5,6,8,10,12,14, 16,18,20,22,24]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            'LRB0': [[28,30,36,38]],
            'GRA': [[26,26,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,42,42,44,44,46,46,48,48,50,50,52,52,54,54,56,56,58,58,59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80]],
            'GRB': [[82,82,84,84,86,86,88,88,90,90,92,92,94,94,96,96],
                    [81,81,83,83,85,85,87,87,91,91,93,93,95,95,97,97]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[98]],
            'LWSB': [[98]],
            'LRA1': [[70,72,74,76,77,79,80,82,84,86,88,90,92,94,96]],
            'LRB1': [[99,114,115,116]],
            'LCC': [[119, 119]]
        }

        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=9, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0"),
            SBarrier(comment=""),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),

            SWaitCnt(dscnt=-1, vlcnt=23+8, vscnt=-1, comment="wait for previous set of GRA"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=38, vscnt=-1, comment="wait for previous set of GRB"),
            SBarrier(comment="")
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and TLDS==0:
        optSchedule = {
            'SYNC': [[-1,24,24,59,59]],
            'LRA0': [[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15]],
            'LRB0': [[24,24,26,26,28,28,30,30]],
            'GRA': [[24,24,25,25,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,43,43,45,45,47,47,49,49,50,50,52,52,54,54,56,56,58,58,60,60,61,61,62,62,63,63,64,64, 65,65,66,66,67,67,68,68,69,69,70,70]],
            'GRB': [[75,75,76,76,77,77,78,78,  89,89,91,91,93,93,95,95]],
            'LRA1': [[60,60,61,61,62,62,63,63,64,64, 65,65,66,66,67,67,68,68,69,69,70,70, 75,75,76,76,77,77,78,78]],
            'LRB1': [[89,89,91,91,93,93,95,95]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[95]],
            'LWSB': [[95]],
            'LCC': [[119,119]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=19, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0, wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {
            'SYNC': [[-1,
                      26,26,
                      59,59
                    ]],
            'LRA0': [[0,2,2,3,3,4,4,5,5,6,6,7,7,9,9,11,11,13,13,15,15,17,17,19,19,21,21,23,23,24],
                     [0,2,2,3,3,4,4,5,5,6,6,7,7,8,8,10,10,12,12,14,14,16,16,18,18,20,20,22,22,25]],
            'LRB0': [[26,27,28,29]],
            'GRA': [[26,26,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,42,42,44,44,46,46,48,48,50,50,52,52,54,54,56,56,58,58,59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80]],
            'GRB': [[82,82,84,84,86,86,88,88,90,90,92,92,93,93,96,96],
                    [83,83,85,85,87,87,89,89,91,91,94,94,99,99,103,103]],
            'LRA1': [[59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80,82,82,84,84, 86,86]],
            'LRB1': [[88,90,92,94]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[95]],
            'LWSB': [[95]],
            'GRIncA': [[1,1,1,17,17,17,18,18,18]],
            'GRIncB': [[19,19,19,20,20,20,21,21,21]],
            'LCC': [[119,119]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=18, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0"),
            SBarrier(comment=""),
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
