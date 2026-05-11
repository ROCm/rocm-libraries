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
    tile_config=TileConfig(192, 256, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 144
    optSchedule = dict()
    syncCode = []
    mfmaReorder = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        # Used the following constrains to create schedule
        #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs
        #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs
        #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs
        #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)
        # Hide LRA0 latency behind GRIncA
        waitLRA0 = max(grIncA)+5
        startPACKA0 = waitLRA0

        # Reordering of packA instructions.
        # 4 CVT + 2 4x4x4_16B MFMAs + 4 CVTs
        # we interleave the 3 blocks together to avoid :
        # - having a 5 state wait after each 4x4x4_16B MFMA
        # - having extra latency when switching between MFMA types
        packAOffset = [ 
                   0, 0, 1, 1, 
                   6, 6,
                   7, 7, 8, 8,

                   2, 2, 3, 3, 
                   6, 6,
                   9, 9, 10, 11,

                   4, 4, 5, 5, 
                   6, 6,
                   12, 12, 13, 13,
                   ]


        packA0 = [x + startPACKA0 for x in packAOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4

        # LRB0 + GRIncB
        lrb0 = create_range(min_val = max(packA0)+1, num = 8, step = 1, repeat = 1)
        grIncB = create_range(max(lrb0)+1,3,max(lrb0)+4,1,3)
        waitLRB0 = max(grIncB)+6
        startPACKB0 = waitLRB0
        packBOffset = [ 
            0, 0, 1, 1, 
            8, 8,
            9, 9, 10, 11,

            2, 2, 3, 3, 
            8, 8,
            12, 12, 13, 13,

            4, 4, 5, 5, 
            8, 8,
            14, 14, 15, 15,

            6, 6, 7, 7, 
            8, 8,
            16, 16, 17, 17,
            ]

        packB0 = [x + startPACKB0 for x in packBOffset]

        # GRA                
        grA = [create_range(min_val = max(packB0)+1, num = 6, step = 2,repeat = 2),
               create_range(min_val = max(packB0)+2, num = 6, step = 2,repeat = 2)]

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA

        # LR3
        startLRB3 = halfMFMA
        lrb3 = create_range(min_val = startLRB3, num = 2, step = 1, repeat = 2)
        lrb3 += create_range(min_val = max(lrb3)+6,num = 2, step = 1, repeat = 2)

        # GRB (split in two blocks)
        grB = create_range(min_val = max(lrb3)+1,num = 4,step = 2, repeat = 2)
        waitLRB3 = max(grB)+1 
        grB += create_range(min_val = max(grB)+47,num = 4,step = 2, repeat = 2)

        # PackB3 (starts after 1st GRB block)
        packB3 = [x + waitLRB3 for x in packBOffset]

        # LRA3 + PACKA3
        startLRA3 = (3*numMfma)//4 # Can't start before 3/4 MFMAs
        lra3 = create_range(min_val = startLRA3,num=6,step=1,repeat=1)
        waitLRA3 = max(lra3) + 8 
        packA3 = [x + waitLRA3 for x in packAOffset]

        syncTable = [                    
                    waitLRA0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    waitLRB0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),

                    max(packB0)+1, SBarrier(comment="Barrier before GRA&GRB"),

                    startLRB3-1,SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRB3-1,SBarrier(comment=""),

                    waitLRB3,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
                    waitLRA3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    
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

            'GRA': [*grA],
            'GRB': [grB],              
            'LRSA': [[max(grIncB)+1]],
            'LRSB': [[max(grIncB)+2]],
            'LWSA': [[142]],
            'LWSB': [[142]],
            'LCC': [[143, 143]],
            'LRA3': [lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],
        }

        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        numLrReadA = 24 
        numLrReadB = 8

        # A is 24 instructions as current codegen can't generate ds_read_b128 in NN case.
        # Instead of reading A first, this schedule re-orders the mfma instructions and reads B first (less instructions)
        # Before :
        #  B0 - A0
        #  B0 - A1
        #  B1 - A0
        #  B1 - A1
        # Now :
        #  B0 - A0
        #  B1 - A0
        #  B0 - A1
        #  B1 - A1

        # mfma Reordering
        mfmaReorder = [i for i in range(0,numMfma//4)] + [i for i in range(numMfma//2,3*numMfma//4)]+[i for i in range(numMfma//4,numMfma//2)]+[i for i in range(3*numMfma//4,numMfma)]

        # Interleaving of LBR0 and GRINCB to hide LRB0 latency
        lrb0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = max(lrb0)+1, num = 3, step = 1, repeat = 3)
        lrb0 += create_range(min_val = max(grIncB)+1, num = 2, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lrb0)+1, num = 3, step = 1, repeat = 3)
        waitLRB0 = max(grIncA)+4

        # PackB0 using mfma4x4x4_16b
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
        packB0Done = max(packB0)

        # Sanity check
        assert packB0Done < numMfma//4

        # GRB (1st block) interleaved with LRA0
        grB = [create_range(min_val = packB0Done+2,num = 4,step = 4, repeat = 2),
               create_range(min_val = packB0Done+1,num = 4,step = 4, repeat = 2)]
       
        # LRA0 
        lra0 = [create_range(min_val = max(packB0)+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = max(packB0)+2, num = numLrReadA // 2, step = 2, repeat = 2)]
       
        # PackA0
        waitLRA0 = max(lra0[1])+2
        startPACKA0 = waitLRA0
        packAOffset = [ 
                   0, 0, 1, 1, 
                   6, 6,
                   7, 7, 8, 8,

                   2, 2, 3, 3, 
                   6, 6,
                   9, 9, 10, 10,

                   4, 4, 5, 5, 
                   6, 6,
                   11, 11, 12, 12,
                   ]

        packA0 = [x + startPACKA0 for x in packAOffset]

        halfMFMA = numMfma//2
        assert max(packA0) < halfMFMA

        # LRA3 interleaved with GRB (2nd half)
        startLRA3 = halfMFMA
        lra3 = [create_range(min_val = startLRA3, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = startLRA3+1, num = numLrReadA // 2, step = 2, repeat = 2)]
        grB[0] += create_range(min_val = startLRA3+1,num = 4,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3,num = 4,step = 2, repeat = 2)
        waitLRA3 = max(lra3[0])+4  
        # LRB3 + PACKA3 & PACKB3
        startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        lrb3 = create_range(min_val = startLRB3,num=numLrReadB - 2,step=1,repeat=1)
        grA = [create_range(min_val = min(lrb3)+1, num = 8, step = 1,repeat = 1),
               create_range(min_val = min(lrb3)+3, num = 8, step = 1,repeat = 1)]
        lrb3 += create_range(min_val = max(lrb3)+3,num=2,step=1,repeat=1)
        waitLRB3 = max(lrb3) + 6 

        # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        packB3 = [x + waitLRB3 for x in packBOffset]
        start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        packA3 = [ 
                   *create_range(min_val = waitLRA3, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+2, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+4, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+5, num = 2, step = 1, repeat = 2),
                   ]

        # GRA 2nd half
        grA[0] += create_range(min_val = max(packB3)+1, num = 2, step = 1,repeat = 2)
        grA[1] += create_range(min_val = max(packB3)+1, num = 2, step = 1,repeat = 2)

        syncTable = [                                      
                    waitLRB0, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
                    waitLRB0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 to complete"),
                    waitLRB0+4, SBarrier(comment="Barrier before GRB"), #Barrier can be after CVT

                    # dscnt has a max value of 15
                    waitLRA0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
                    waitLRA0+1, SWaitCnt(dscnt=min(15,numLrReadA-8), vlcnt=-1, vscnt=-1, comment="Wait for 8 LRA0 to complete"),
                    waitLRA0+2, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),

                    startLRA3-1,SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRA3-1,SBarrier(comment="Sync before GRA, LRA3 & LRB3"),

                    # incremental wait on LRA3
                    waitLRA3, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA3 to complete"),                    
                    waitLRA3+1, SWaitCnt(dscnt=min(15,numLrReadA-8), vlcnt=-1, vscnt=-1, comment="Wait for 8 LRA3 to complete"), 
                    waitLRA3+2, SWaitCnt(dscnt=min(15,numLrReadA-12), vlcnt=-1, vscnt=-1, comment="Wait for 12 LRA3 to complete"), 
                    waitLRA3+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    

                    # incremental wait on LRB3
                    waitLRB3, SWaitCnt(dscnt=(numLrReadB-1), vlcnt=-1, vscnt=-1, comment="Wait for 1st LRB3 to complete"),
                    waitLRB3+1, SWaitCnt(dscnt=(numLrReadB-2), vlcnt=-1, vscnt=-1, comment="Wait for 2nd LRB3 to complete"),
                    waitLRB3+2, SWaitCnt(dscnt=(numLrReadB-3), vlcnt=-1, vscnt=-1, comment="Wait for 3rd LRB3 to complete"),
                    waitLRB3+3, SWaitCnt(dscnt=(numLrReadB-4), vlcnt=-1, vscnt=-1, comment="Wait for 4th LRB3 to complete"),
                    waitLRB3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

                    ]

        syncCode = syncTable[1::2]
        optSchedule = {

            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [*lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],
            'GRA': [*grA],
            'GRB': [*grB],              
            'LRSA': [[max(lra0[1])+1]],
            'LRSB': [[max(lra0[1])+1]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 14
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    return True, opt1
