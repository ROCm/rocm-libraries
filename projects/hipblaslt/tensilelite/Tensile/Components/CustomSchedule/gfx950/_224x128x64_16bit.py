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
    tile_config=TileConfig(224, 128, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x128x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    numMfma = 56
    numCodePaths = 1

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
        optSchedule = {
            'SYNC': [[-1, 6, 14, 14, 27,27, 47, 47]],
            'LRA0': [[0,1, 2,3,4,5,5]],
            'GRIncA': [[0, 0, 1, 1, 2,2 , 3,3, 4]],
            'LRB0': [[9, 11,13, 19]],
            'GRIncB': [[ 6,6,7,7,8,8,9,9,10]],
            'GRA': [[14, 14, 16,16,18,18,20,20,23,23, 26,26, 27, 27]],
            'LRSA': [[26]],
            'LRSB': [[26]],
            'GRB': [[33,34,36,38,38,42,42,46]],
            'LWSA': [[54]],
            'LWSB': [[54]],
            'LRA1': [[30,35,44, 45, 46, 48,51]],
            'LRB1': [[47,52,54,55]],
            'LCC': [[55, 55]],
        }

        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for all of LRA1 and the first instance of LRB1"),
            SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="wait for the second instance of LRB1"),
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete so GRA could begin. Makes sure LRB1 is completed so no need for a barrier at 21"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=10, vscnt=-1, comment="wait for all LR. All of previous GRB (4) and current GRA (6), total of 10 can be outstanding"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Outstanding LR are all LRA so no need to wait. All of GR from previous iteration must be done."),
            SBarrier(comment="")
        ]
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 16,16, 27, 35,35, 48,48]],
            'GRIncA' : [[1,2,3,4,4,5,5,6,6]],
            'GRIncB' : [[7,7,8,8,9,9,10,11,11]],

            'LRA0'   : [[0,1,1,2,2,3,4,5,6,7,8,9,10,11]],
            'LRB0'   : [[17,18,19,20]],   ## After LRA0, we can mix LRB0 and GRA

            ## GRA should start after LRA0 is done.
            'GRA'    : [[15,16, 19,19, 22,22, 25,25, 28,28, 31,31, 34,34]],
            ## GRB should start after LRB0 is done.
            'GRB'    : [[42,42, 45,45, 48,48, 51,51]],

            #After previous GRA is done.
            'LRA1'   : [[35,36,37,38,39,40,41,42,43,44,45,46,47,48]],
            #After previous GRB is done.
            'LRB1'   : [[49,50, 52,53]],

            'LRSA'   : [[24]], # after LRA0 and before LRA1
            'LRSB'   : [[24]], # after LRB0 and before LRB2
            'LWSA'   : [[53]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[54, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for previous GRA to complete to start LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for previous GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        # Global read scheduling:
        # Each GR has two instructions (addr update + buffer_load), so we list them explicitly as
        # two adjacent MFMA indices per GR.
        kernel["SwapGlobalReadOrder"] = True
        numCodePaths = 2

        syncTable = [
            # Loop start:
            # - LRB1 waits previous-iter GRB
            # - LRA1 waits previous-iter GRA
            -1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 before starting main loop"),

            # After early MFMAs (keep prior-iter LR fully fenced)
            3,  SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),

            # GRB must wait for LRB0 (interleave LRA0 + GRB safely)
            15, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            15, SBarrier(comment=""),

            # GRA must wait for LRA0; LRB1 can be interleaved with GRA after this fence
            27, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/GRB to complete to start GRA/LRB1"),
            27, SBarrier(comment=""),

            # Mid-loop global-read safety fence (GR-to-LDS)
            30, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Mid-loop fence (wait for outstanding GR-to-LDS)"),
            30, SBarrier(comment=""),

            # Ensure all GR-to-LDS are complete before LRA1 (next-iter A LDS reads)
            43, SWaitCnt(dscnt=-1, vlcnt=11-2, vscnt=-1, comment="Wait for all GR-to-LDS to complete before LRA1"),
            43, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            # Swap A/B increments
            'GRIncB' : [[0, 0, 1, 1, 2, 2, 4, 4, 5]],
            'GRIncA' : [[5, 6, 6, 7, 7, 8, 8, 9, 9]],

            'LRB0'   : [[0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 3, 4, 5, 6, 7, 8]],
            'LRA0'   : [[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],

            'GRA'    : [[14,15, 17,18, 20,21, 23,24],
                        [15,16, 18,19, 21,22, 24,25]],
            'GRB'    : [[28,29, 31,32, 34,35, 37,38, 40,41, 43,44, 46,47],
                        [29,30, 32,33, 35,36, 38,39, 41,42, 44,45, 45,46]],

            'LRB1'   : [[30, 31, 32, 33, 34, 35, 36, 37],
                        [31, 32, 33, 34, 35, 36, 37, 38]],
            'LRA1'   : [[43, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]],

            'LRSA'   : [[24]],
            'LRSB'   : [[25]],
            'LWSA'   : [[48]],
            'LWSB'   : [[49]],
            'LCC'    : [[53, 54]],
        }

        syncCode = syncTable[1::2]
        nglshift = nllshift = 11
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(numCodePaths, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
