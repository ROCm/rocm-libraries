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
    isNN,
    isNT,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 192, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x192x32_TF32(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isNN(kernel) and not useLDSTr and TLDS==1:
        # TODO: Add NN schedule in upcoming PR
        return False, None
    elif isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        syncTable = [
            5,  SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Before PackA0. Wait for all LRA0. Skip 1*LRB0.") ,
            17, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackB0. Wait for all prior LRB0 for PackB0.") ,
            17, SBarrier(comment="GRA") ,
            32, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before GRB. Wait for all prior LRB0.") ,
            32, SBarrier(comment="GRB") ,
            35, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Before LRB3. Wait for GRB from previous iter. Skip 4*GRA + 2*GRB") ,
            35, SBarrier(comment="LRB") ,
            44, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackB3. Wait for all prior LRB3.") ,
            53, SWaitCnt(dscnt=0, vlcnt=10, vscnt=-1, comment="Before LRA3. Wait for GRA from previous iter. Skip 4*GRA + 6*GRB") ,
            53, SBarrier(comment="LRA") ,
            63, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackA3. Wait for all prior LRA3.") ,
        ]
        optSchedule = {
            'SYNC'  : [syncTable[::2]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRA0'  : [[0, 1, 2, 3]],
            'LRB0'  : [[4, 6, 8, 10, 12, 14],
                       [4, 7, 9, 11, 13, 15]],
            'PackA0': [create_range(5,12,17, 1, 4)],
            'PackB0': [create_range(18,18,34, 1, 4)],
            'GRA'   : [[17,17, 18,18, 19,19, 20,20]],
            'GRB'   : [[33, 33, 34, 34, 41, 42, 43, 44, 51, 51, 52, 52]],
            'LRB3'  : [[36, 37, 38, 39, 40, 41]],
            'LRA3'  : [[53, 55, 57, 59],
                       [54, 56, 58, 60]],
            'PackB3': [create_range(44,9,52, 1, 8)],
            'PackA3': [create_range(63,8,71, 1, 6)],
            'LRSA'  : [[16]],
            'LRSB'  : [[16]],
            'LWSA'  : [[61]],
            'LWSB'  : [[62]],
            'LCC'   : [[71, 71]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 10
    elif isNT(kernel) and useLDSTr and TLDS==0:
        # TODO: Add NT schedule in upcoming PR
        return False, None
    else:
        return False, None
    
    kernel["MfmaInitCVgprs"] = True
    numMfma = 72
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
