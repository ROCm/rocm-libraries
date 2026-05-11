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
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 256, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    mfmaReorder = []
    nglshift = nllshift = 0
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 4, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)

        waitLRA0 = max(grIncA)+2
        startPACKA0 = waitLRA0

        packAOffset = [
            0, 0, 1, 1,
            4, 4,
            5, 5, 6, 6,

            2, 2, 3, 3,
            4, 4,
            7, 7, 8, 8,
        ]
        packA0 = [x + startPACKA0 for x in packAOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4 , f"packA0Done {packA0Done} >= {numMfma//4}"

        # 1st part of LRB0 + GRIncB
        lrb0 = create_range(min_val = max(packA0)+1, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = max(packA0)+1, num = 4, step = 1, repeat= 2)
        grIncB += [max(grIncB)+1]

        # GRA  + 2nd part of LBR0
        grA = create_range(min_val = max(lrb0)+1, num = 4, step = 2,repeat = 2)
        lrb0 += create_range(min_val = max(lrb0)+4, num = 2, step = 1, repeat = 1)

        waitLRB0 = max(lrb0)+2
        startPACKB0 = waitLRB0

        packBOffset = [
            0, 0, 1, 1,
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3,
            8, 8,
            11, 11, 12, 12,

            4, 4, 5, 5,
            8, 8,
            13, 13, 14, 14,

            6, 6, 7, 7,
            8, 8,
            15, 15, 16, 16,
        ]
        packB0 = [x + startPACKB0 for x in packBOffset]

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA, f"max(packB0) {max(packB0)} >= halfMFMA {halfMFMA}"

        # LR3
        startLRB3 = halfMFMA
        # Interleave GRA and LBR3
        lrb3 = [create_range(min_val = startLRB3, num = 3, step = 2, repeat = 2),
                create_range(min_val = startLRB3+1, num = 3, step = 2, repeat = 2)]

        # Splitting LRB3 to avoid LDS Issue latency
        lrb3[0]+= create_range(min_val = max(lrb3[0])+5, num = 1, step = 2, repeat = 2)
        lrb3[1]+= create_range(min_val = max(lrb3[1])+5, num = 1, step = 2, repeat = 2)

        grB = [create_range(min_val = startLRB3+1,num = 4,step = 2, repeat = 2),
               create_range(min_val = startLRB3,num = 4,step = 2, repeat = 2)]

        waitLRB3 = max(lrb3[1])+2 

        # Use different PackBOffset to shift last 5 CVTs iterations after GRB/LRA3
        packB3Offset = [
            0, 0, 1, 1,
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3,
            8, 8,
            11, 11, 19, 19,

            4, 4, 5, 5,
            8, 8,
            20, 20, 21, 21,

            6, 6, 7, 7,
            8, 8,
            22, 22, 23, 23,
        ]   

        # PackB3
        packB3 = [x + waitLRB3 for x in packB3Offset]

        startLRA3 = (3*numMfma)//4 
        # GRB + LRA3 (interleaved)
        grB[0] += create_range(min_val = startLRA3,num = 4,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3+1,num = 4,step = 2, repeat = 2)

        lra3 = [create_range(min_val = startLRA3+1,num=4,step=2,repeat=1),
                create_range(min_val = startLRA3,num=4,step=2,repeat=1)]

        waitLRA3 = max(lra3[0]) + 6 
        packA3 = [x + waitLRA3 for x in packAOffset]

        syncTable = [
            -1, SBarrier(comment="Sync codepath"),
            waitLRA0, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 1st LRA0 to complete"),
            waitLRA0+1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 2nd LRA0 to complete"),
            waitLRA0+2, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 3rd LRA0 to complete"),
            waitLRA0+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA0 to complete"),
            min(grIncB), SBarrier(comment="Barrier before GRA"),

            waitLRB0, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for 1/8 LRB0 to complete"),
            waitLRB0+1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for 2/8 LRB0 to complete"),
            waitLRB0+2, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 3/8 LRB0 to complete"),
            waitLRB0+3, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
            waitLRB0+4, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 5/8 LRB0 to complete"),
            waitLRB0+5, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 6/8 LRB0 to complete"),
            waitLRB0+6, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 7/8 LRB0 to complete"),
            waitLRB0+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 8/8 LRB0 to complete"),

            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
            startLRB3-1, SBarrier(comment="Barrier before GRB and before LRBA3/LBRB3"),

            waitLRB3, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for 1/8 LRB3 to complete"),
            waitLRB3+1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for 2/8 LRB3 to complete"),
            waitLRB3+2, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 3/8 LRB3 to complete"),
            waitLRB3+3, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB3 to complete"),
            waitLRB3+4, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 5/8 LRB3 to complete"),
            waitLRB3+5, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 6/8 LRB3 to complete"),
            waitLRB3+6, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 7/8 LRB3 to complete"),
            waitLRB3+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 8/8 LRB3 to complete"),

            waitLRA3, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 1/4 LRA3 to complete"),                    
            waitLRA3+1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 2/4 LRA3 to complete"),                    
            waitLRA3+2, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 3/4 LRA3 to complete"),                    
            waitLRA3+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 4/4 LRA3 to complete"),                    
        ]

        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],

            'GRA': [grA],
            'GRB': [*grB],              
            'LRSA': [[max(lrb0)+1]],
            'LRSB': [[max(lrb0)+1]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [*lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],
        }
        nglshift = nllshift = 12 # vmcnt shift for ngl and nll
    elif isNN(kernel) and TLDS==1:
        return False, None
        # kernel["UsePLRPack"] = True
        # kernel["UseMFMAF32XEmulation"] = True
        # kernel["UseDot2F32XEmulation"] = False

        # numLrReadA = 16
        # numLrReadB = 8
        # # mfma Reordering
        # mfmaReorder = [i for i in range(0, numMfma//4)] + [i for i in range(numMfma//2, 3*numMfma//4)] + [i for i in range(numMfma//4, numMfma//2)] + [i for i in range(3*numMfma//4, numMfma)]

        # # LBR0
        # lrb0 = create_range(min_val = 0, num = numLrReadB//2, step = 1, repeat = 2)
        # grIncB = create_range(min_val = 0, num = 3, step = 1, repeat = 3)
        # grIncA = create_range(min_val = max(grIncB)+1, num = 3, step = 1, repeat = 3)
        # waitLRB0 = max(lrb0)
        # # PackB0 using mfma4x4x4_16b
        # startPACKB0 = waitLRB0 + 4
        # packBOffset = [ 
        #     0, 0, 1, 1, 
        #     8, 8,
        #     9, 9, 10, 10,

        #     2, 2, 3, 3, 
        #     8, 8,
        #     11, 11, 12, 12,

        #     4, 4, 5, 5, 
        #     8, 8,
        #     13, 13, 14, 14,

        #     6, 6, 7, 7, 
        #     8, 8,
        #     15, 15, 16, 16,
        # ]
        # packB0 = [x + startPACKB0 for x in packBOffset]
        # packB0Done = max(packB0)
        # assert packB0Done < numMfma//4

        # # LRA0 
        # lra0 = create_range(min_val = waitLRB0+4, num = numLrReadA, step = 1, repeat = 1)
        # grA = create_range(min_val = packB0Done+6, num = 8, step = 2, repeat = 1)
        # waitLRA0 = max(lra0)
        # startPACKA0 = waitLRA0 + 2
        # packAOffset = [
        #     0, 0, 1, 1,
        #     4, 4,
        #     5, 5, 6, 6,

        #     2, 2, 3, 3,
        #     4, 4,
        #     7, 7, 8, 8,
        # ]
        # packA0 = [x + startPACKA0 for x in packAOffset]
        # halfMFMA = numMfma//2
        # assert max(packA0) < halfMFMA

        # # LRA3
        # startLRA3 = halfMFMA
        # lra3 = create_range(min_val = startLRA3, num = numLrReadA, step = 1, repeat = 1)
        # grB = create_range(min_val = startLRA3+1, num = 4, step = 2, repeat = 2)
        # grB += create_range(min_val = max(lra3)+1, num = 4, step = 2, repeat = 2)
        # waitLRA3 = max(lra3)
        # startPACKA3 = waitLRA3 + 4

        # # LRB3
        # startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        # lrb3 = create_range(min_val = startLRB3, num = numLrReadB//2, step = 1, repeat = 2)
        # waitLRB3 = max(lrb3)
        # startPACKB3 = waitLRB3 + 4

        # # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        # packB3 = [x + startPACKB3 for x in packBOffset]
        # start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        # packA3 = [
        #     *create_range(min_val = startPACKA3, num = 2, step = 1, repeat = 2),
        #     start_4x4x4, start_4x4x4,
        #     *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

        #     *create_range(min_val = startPACKA3+2, num = 2, step = 1, repeat = 2),
        #     start_4x4x4, start_4x4x4,
        #     *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),
        # ]

        # syncTable = [
        #     waitLRB0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
        #     waitLRB0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
        #     waitLRB0+4, SBarrier(comment=""),

        #     waitLRA0, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
        #     waitLRA0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 to complete"),
        #     waitLRA0+4, SBarrier(comment=""),

        #     startLRA3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for previous GRA&GRB"),
        #     startLRA3-1, SBarrier(comment=""),

        #     waitLRA3, SWaitCnt(dscnt=(numLrReadB-1), vlcnt=-1, vscnt=-1, comment="Wait for 1st LRB3 to complete"),
        #     waitLRA3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

        #     startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for previous GRA&GRB"),
        #     startLRB3-1, SBarrier(comment=""),

        #     waitLRB3, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA3 to complete"),
        #     waitLRB3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
        # ]

        # optSchedule = {
        #     'SYNC': [syncTable[::2]],
        #     'GRIncA': [grIncA],
        #     'GRIncB': [grIncB],
        #     'LRA0': [lra0],
        #     'LRB0': [lrb0],
        #     'PackA0' : [packA0],
        #     'PackB0' : [packB0],
        #     'GRA': [grA],
        #     'GRB': [grB],
        #     'LRSA': [[max(lra0)+1]],
        #     'LRSB': [[max(lra0)+1]],
        #     'LWSA': [[max(lrb3)+1]],
        #     'LWSB': [[max(lrb3)+1]],
        #     'LCC': [[numMfma-1, numMfma-1]],
        #     'LRA3': [lra3],
        #     'LRB3': [lrb3],
        #     'PackA3' : [packA3],
        #     'PackB3' : [packB3],
        # }
        # syncCode = syncTable[1::2]
        # nglshift = nllshift = 12
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    return True, opt1
