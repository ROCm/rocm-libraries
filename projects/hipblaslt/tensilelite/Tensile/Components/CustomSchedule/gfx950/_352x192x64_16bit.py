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
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(352, 192, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_352x192x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 132
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    
    if isTN(kernel) and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=5 newLW=0 newLR=5 for iteration == 0"),
            14, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for 5 LRA0s to complete"),
            25, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for next 4 LRA0s to complete"),
            
            34, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA0s to complete"),
            34, SBarrier(comment="Barrier before GRA"),
            
            46, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for 3 LRB0s to complete"),
            
            87, SWaitCnt(dscnt=0, vlcnt=17, vscnt=-1, comment="wait for previous GRA to complete"),
            87, SBarrier(comment="Barrier before LRA1"),
            118, SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="wait for previous GRB to complete"),
            118, SBarrier(comment="Barrier before LRB1"),
        ]
        
        optSchedule = {
            'SYNC': [syncTable[::2]], # 6
            
            'GRIncA': [[0, 0, 1, 1, 2, 2, 3, 3, 4]], # 9
            'GRIncB': [[4, 5, 5, 6, 6, 7, 7, 8, 8]], # 9
            
            'LRA0': [[0, 2, 4, 6, 8 , 10, 14, 18, 20, 24, 26],
                     [1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27]], # 11
            
            'LRB0': [[31, 33, 36, 39, 41, 44],
                     [32, 34, 37, 40, 42, 45]], # 6

            'GRA': [[35, 35, 40, 40, 46, 46, 51, 51, 56, 56, 61, 61, 67, 67, 72, 72, 77, 77, 82, 82, 86, 86]], # 22
            'GRB': [[88, 88, 93, 93, 98, 98, 103, 103, 109, 109, 114, 114]], # 12

            'LRSA': [[64]], # 1
            'LRSB': [[64]], # 1
            
            'LWSA': [[109]], # 1
            'LWSB': [[109]], # 1
            'LRA1': [[87, 90, 92, 94, 96, 98, 100, 102, 104, 106, 114]], # 11
            'LRB1': [[118, 120, 122, 124, 127, 130]], # 6 
 
            'LCC': [[numMfma-3, numMfma-3]], # 2
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
