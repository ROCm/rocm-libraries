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
    tile_config=TileConfig(160, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_160x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 80
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1, 4, 13,13, 38,39, 42,43, 70,70]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[29,30,31,32,33,34,35,36,37]],

            'LRA0'   : [[0,2,3,4,5]],  ## -2 is place holder

            'LRB0'   : [[13,15,18,21,24,26,28,30],   ## After LRA0, we can mix LRB0 and GRA
                        [14,16,19,22,25,27,29,31]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[11,14, 17,17, 20,20, 23,23, 26,27],
                        [12,15, 18,18, 21,21, 24,24, 27,28]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[40,40, 43,43, 46,46, 49,49, 59,59, 62,62, 65,65, 67,68],  # m0 inc is part of GRA/GRB
                        [41,41, 44,44, 47,47, 57,57, 60,60, 63,63, 66,66, 68,69]],
            'LRA1'   : [[44, 47, 53, 58, 63],
                        [45, 48, 54, 59, 64]],

            #After GRB is done.
            'LRB1'   : [[70,71,72,73,75,76,77,78]],

            'LRSA'   : [[33]], # after LRA0 and before LRA1
            'LRSB'   : [[33]], # after LRB0 and before LRB2
            'LWSA'   : [[74]], # For A
            'LWSB'   : [[76]],

            'LCC'    : [[79, 79]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13 + 1), vscnt=-1, comment="Wait for GRA to complete to start LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 13 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'   : [[-1,
            12, 12, # Wait for B
            24, 24, # wait LRB0.
            41, 41,
            61, 61  # wait GRA.
            ]],
            # Addr. update (be done before GRA/GRB).
            'GRIncA' : [[5,6,7,8,9,10,11,12,13]],
            'GRIncB' : [[0,0,1,1,2,2,3,3,4]],
            # Current iteration.
            'LRA0'   : [[8,9,10,11,12,13,14,15,16,17]],
            'LRB0'   : [[0,1,2,3,4,5,6,7]],
            # Buffer loads.
            'GRB'    : [[51,51, 55,55, 59,61, 76,77, 78,78]],
            'GRA'    : [[11,12, 16,16, 20,20, 24,24, 28, 28, 32,32, 36, 36, 40, 40]],
            # Prefetch next iteration.
            'LRA1'   : [[62,63,64,65,66,67,68,69,70,71]],
            'LRB1'   : [[41,42,43,44,45,46,47,49]],
            'LRSA'   : [[39]],
            'LRSB'   : [[39]],
            'LWSA'   : [[60]],
            'LWSB'   : [[60]],
            'LCC'   : [[79, 79]], # Loop control.
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 LRB1"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for previous GRA(B) to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for previous GRB(A) to complete"),
                    SBarrier(comment="")]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS==0:
        optSchedule = {
            'SYNC': [[-1,17,17,57,57]],
            'GRA': [[16,17,20,20,24,24,28,28,31,31]],
            'GRB': [[35,35,39,39,68,68,70,70,71,71,76,76,77,77,78,78]],
            'GRIncA': [[0,0,1,1,2,2,3,3,4]],
            'GRIncB': [[4,5,5,13,13,13,14,14,14]],
            'LCC': [[79,79]],
            'LRA0': [[0,1,1,2,2,3,3,4,4,5]],
            'LRA1': [[58,60,62,64,66,67,67,68,68,69]],
            'LRB0': [[0,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]],
            'LRB1': [[59,61,63,65,67,71,72,72,73,73,74,74,75,75,76,76]],
            'LRSA': [[38]],
            'LRSB': [[38]],
            'LWSA': [[61]],
            'LWSB': [[61]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=7, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
