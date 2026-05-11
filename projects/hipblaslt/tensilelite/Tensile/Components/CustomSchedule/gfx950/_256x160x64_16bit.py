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
    tile_config=TileConfig(256, 160, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x160x64_16bit(kernel, useLDSTr, TLDS):
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    numMfma = 80
    if isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'   : [[-1,
            12,12, # Wait for LRB0
            24, 24,# Wait LRA0.
            41,41,
            61, 61 # Wait previous GR.
            ]],
            # Addr. update (be done before GRA/GRB).
            'GRIncA' : [[21,21,21,22,22,22,23,23,23]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],
            # Current iteration.
            'LRA0'   : [[5,5,7,7,9,9,11,11,13,13,15,15,17,18,19,20]],
            'LRB0'   : [[0,0,1,2,3]],
            # Buffer loads.
            'GRB'    : [[30,30, 33,33, 36,36, 52,52, 56,56, 60,61, 76,77, 78,78]],
            'GRA'    : [[11,12, 16,16, 20,20, 25,25, 26, 28]],
            # Prefetch next iteration.
            'LRA1'   : [[62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]],
            'LRB1'   : [[41,42,43,44,45]],
            'LRSA'   : [[39]],
            'LRSB'   : [[39]],
            'LWSA'   : [[60]],
            'LWSB'   : [[60]],
            'LCC'   : [[79, 79]], # Loop control
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 LRB1"),
                    SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for previous GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13+8-5), vscnt=-1, comment="Wait for previous GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13+10-13), vscnt=-1, comment="Wait for previous GRA to complete"),
                    SBarrier(comment="")]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isTN(kernel) and (not useLDSTr) and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1 (partial) before starting main loop"),
             4, SWaitCnt(dscnt=0+2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1 (complete) for the remaining main loop"),
            14, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            14, SBarrier(comment=""),
            # Must be dscnt=0 here: validator requires proving all LRA0 are complete
            # before the first GRA is issued (vmfma_index window [31,41)).
            39, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 / ensure LRA0 complete before starting GRB/GRA"),
            39, SBarrier(comment=""),
            45, SWaitCnt(dscnt=-1, vlcnt=13+2, vscnt=-1, comment="Wait for GRB to complete before LRB1"),
            45, SBarrier(comment=""),
            69, SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for GRA to complete before LRA1"),
            69, SBarrier(comment=""),
        ]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[29,30,31,32,33,34,35,36,37]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],

            # Current iteration.
            'LRB0'   : [[0,2,3,4,5],
                        [1,3,4,5,6]],
            'LRA0'   : [[13,15,18,21,24,26,28,30],
                        [13,16,19,22,25,27,29,31]],

            # GRB must not start before the SYNC at idx 15 (LRB0 completion).
            'GRB'    : [[14,14, 17,17, 20,20, 23,23, 26,26],
                        [15,15, 18,18, 21,21, 24,24, 27,27]],
            # Buffer loads.
            'GRA'    : [[40,40, 43,43, 46,46, 49,49, 59,59, 62,62, 65,65, 67,67],
                        [41,41, 44,44, 47,47, 57,57, 60,60, 63,63, 66,66, 68,68]],
            # Prefetch next iteration.
            # Need 5 local reads for B (MIWaveTileB=5).
            'LRB1'   : [[45,46,47,48,49],
                        [46,47,48,49,50]],
            # Need 8 local reads for A (MIWaveTileA=8) in each code path.
            # Path1 LRA1 must be earlier than path0 (validator requirement).
            'LRA1'   : [[69, 70, 71, 72, 73, 74, 75, 76],
                        [70, 71, 72, 73, 74, 75, 76, 77]],
            'LRSA'   : [[32]],
            'LRSB'   : [[33]],
            'LWSA'   : [[74]],
            'LWSB'   : [[76]],
            'LCC'    : [[77, 78]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS==0:
        nglshift = nllshift = 0
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC': [[-1,17,17,57,57]],
            'GRA': [[16,17,20,20,24,24,28,28,31,31]],
            'GRB': [[35,35,39,39,68,68,70,70,71,71,76,76,77,77,78,78]],
            'GRIncA': [[0,0,1,1,2,2,3,3,4]],
            'GRIncB': [[4,5,5,13,13,13,14,14,14]],
            'LCC': [[79,79]],
            'LRA0': [[0,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]],
            'LRB0': [[0,1,1,2,2,3,3,4,4,5]],
            'LRA1': [[59,61,63,65,67,71,72,72,73,73,74,74,75,75,76,76]],
            'LRB1': [[58,60,62,64,66,67,67,68,68,69]],
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
