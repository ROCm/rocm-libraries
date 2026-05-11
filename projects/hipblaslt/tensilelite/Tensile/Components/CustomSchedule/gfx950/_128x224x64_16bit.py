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
    switch_A_B_schedule,
)
from ._224x128x64_16bit import _get_schedule_224x128x64_16bit


@RegisterSchedule(
    tile_config=TileConfig(128, 224, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x224x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 10,10, 26,27, 45,45]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[22,22,24,24,25,25,26,27,27]],

            'LRA0'   : [[0,2,3,4]],  ## -2 is place holder

            'LRB0'   : [[8,11,13,15,17,19,21],   ## After LRA0, we can mix LRB0 and GRA
                        [9,12,14,16,18,20,22]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[10,11, 14,14, 17,17, 20,20],
                        [11,12, 15,15, 18,18, 21,21]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[28,28, 31,31, 34,34, 37,37, 40,40, 43,43, 46,46],  # m0 inc is part of GRA/GRB
                        [29,29, 32,32, 35,35, 38,38, 41,41, 44,44, 47,47]],
            'LRA1'   : [[29, 32, 35, 38],
                        [30, 33, 36, 39]],

            #After GRB is done.
            'LRB1'   : [[45,46,47,48,49,50,51]],

            'LRSA'   : [[23]], # after LRA0 and before LRA1
            'LRSB'   : [[23]], # after LRB0 and before LRB2
            'LWSA'   : [[54]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[55, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=11, vscnt=-1, comment="Wait for LRB0/GRA to complete to start GRB/LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 12,12, 26,27, 45,45]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[22,22,24,24,25,25,26,27,27]],

            'LRA0'   : [[0,0,2,2,3,3,5,6]],  ## -2 is place holder

            'LRB0'   : [[8,10,13,15,17,19,21],   ## After LRA0, we can mix LRB0 and GRA
                        [9,11,14,16,18,20,22]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[10,12, 14,14, 17,17, 20,20],
                        [11,13, 15,15, 18,18, 21,21]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[28,28, 31,31, 34,34, 37,37, 40,40, 43,43, 46,46],  # m0 inc is part of GRA/GRB
                        [29,29, 32,32, 35,35, 38,38, 41,41, 44,44, 47,47]],
            'LRA1'   : [[29,30, 32,33, 35,36, 38,39],
                        [30,31, 33,34, 36,37, 39,40]],

            #After GRB is done.
            'LRB1'   : [[45,46,47,48,49,50,51]],

            'LRSA'   : [[23]], # after LRA0 and before LRA1
            'LRSB'   : [[23]], # after LRB0 and before LRB2
            'LWSA'   : [[54]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[55, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=11, vscnt=-1, comment="Wait for LRB0/GRA to complete to start GRB/LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        valid, opt = _get_schedule_224x128x64_16bit(kernel, useLDSTr, TLDS)
        if not valid:
            return False, None
        optSchedule = switch_A_B_schedule(opt.optSchedule)
        return True, ScheduleInfo(opt.numCodePaths, opt.numMfma, optSchedule, opt.syncCode, opt.nglshift, opt.nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 56
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
