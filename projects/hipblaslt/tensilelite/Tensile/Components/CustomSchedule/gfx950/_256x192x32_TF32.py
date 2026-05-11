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
    inflight,
    isNN,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 192, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x192x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 144
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        numPackInstr = 24 
        numPackIndices = numPackInstr // 2 # Assign 2 pack instructions per mfma index
        
        # LRA0 + PACKA0 - done before 1/4 MFMAs - index 36
        lrA0 = [0,0, 1,1, 2,2, 3,3]
        waitLRA0 = max(lrA0) + 2
        startPACKA0 = waitLRA0
        packA0 = create_range(startPACKA0, (len(lrA0)//2)*numPackIndices, numMfma//4-1)
        
         # LBR0 + PACKB0 - done before 2/4 MFMAs - index 72
        lrB0 = [7,7, 11,11, 15,15]
        waitLRB0 = max(lrB0) + 2
        startPACKB0 = max(waitLRB0,max(packA0)) # Starts after waitLRB0 and packA0
        packB0 = create_range(startPACKB0, (len(lrB0)//2)*numPackIndices, numMfma//2-1)
        
        # LBR3 + PACKB3 - start after 2/4 MFMAs - index 72
        halfMFMA = numMfma//2
        startLRB3 = halfMFMA
        lrB3 = create_range(startLRB3, 1, numMfma-1)
        lrB3 += create_range(max(lrB3)+6, 2, numMfma-1)
        waitLRB3 = startLRB3 + 6
        packB3 = create_range(waitLRB3, (len(lrB3)//2)*numPackIndices, numMfma-1)
        
        # LRA3 + PACKA3 - start after 3/4 MFMAs - index 108
        startLRA3 = (3*numMfma)//4
        lrA3 = create_range(startLRA3, 4, numMfma-1)
        waitLRA3 = startLRA3 + 5
        packA3 = create_range(waitLRA3, (len(lrA3)//2)*numPackIndices, numMfma-1)
        
        syncTable = [
            waitLRA0, SWaitCnt(dscnt=inflight(lrA0, waitLRA0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRA0 to complete"),
            waitLRA0+numPackIndices, SWaitCnt(dscnt=inflight(lrB0, waitLRA0+numPackIndices), vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete"),
            
            waitLRB0, SWaitCnt(dscnt=inflight(lrB0, waitLRB0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRB0 to complete"),
            waitLRB0+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0 to complete"),
            waitLRB0+numPackIndices, SBarrier(comment="Barrier before GRA&GRB"),
            
            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRB3-1, SBarrier(comment=""),
            
            waitLRB3,SWaitCnt(dscnt=inflight(lrB3, waitLRB3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRB3 to complete"),
            waitLRB3+numPackIndices,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

            waitLRA3, SWaitCnt(dscnt=inflight(lrA3,waitLRA3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRA3 to complete"),
            waitLRA3+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA3 to complete")
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            
            'GRIncA' : [[0,0,0, 1,1,1, 2,2,2]],
            'GRIncB' : [[3,3,3, 4,4,4, 5,5,5]], 

            'LRA0'   : [lrA0],
            'PackA0' : [packA0],
            
            'LRB0'   : [lrB0],
            'PackB0' : [packB0],
            
            'GRA'    : [[72,72, 74,74, 76,76, 100,100, 102,102, 104,104, 106,106, 108,108],
                        [73,73, 75,75, 77,77, 101,101, 103,103, 105,105, 107,107, 109,109]],
            'GRB'    : [[38,38, 40,40, 42,42, 44,44, 46,46, 48,48],
                        [39,39, 41,41, 43,43, 45,45, 47,47, 49,49]],
            
            'LRA3'   : [lrA3],
            'PackA3' : [packA3],
            
            'LRB3'   : [lrB3],
            'PackB3' : [packB3],

            'LRSA'   : [[35]],
            'LRSB'   : [[35]],

            'LWSA'   : [[107]],
            'LWSB'   : [[107]],

            'LCC'    : [[143, 143]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)

    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        
        numLrReadA = 32 
        numLrReadB = 6

        # mfma Reordering
        mfmaReorder = [i for i in range(0,numMfma//4)] + [i for i in range(numMfma//2,3*numMfma//4)]+[i for i in range(numMfma//4,numMfma//2)]+[i for i in range(3*numMfma//4,numMfma)]

        # Interleave LBR0 and GRINCB
        lrb0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB += create_range(min_val = max(lrb0)+1, num = 1, step = 1, repeat = 3)
        
        # Interleave GRINCA and PACKB0
        grIncA = create_range(min_val = max(grIncB)+1, num = 2, step = 1, repeat = 3)
        waitLRB0 = max(grIncA)
        grIncA += create_range(min_val = max(grIncA)+2, num = 3, step = 1, repeat = 1)

        startPACKB0 = waitLRB0
        packBOffset = [ 
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

        packB0 = [x + startPACKB0 for x in packBOffset]
        packB0Done = max(packB0)

        # Sanity check
        assert packB0Done < numMfma//4, f"packB0Done {packB0Done} >= numMfma//4 {numMfma//4}"

        # GRB (1st block) interleaved with LRA0
        grB = [create_range(min_val = packB0Done+2,num = 3,step = 4, repeat = 2),
               create_range(min_val = packB0Done+1,num = 3,step = 4, repeat = 2)]
       
        # LRA0 
        lra0 = [create_range(min_val = max(packB0)+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = max(packB0)+2, num = numLrReadA // 2, step = 2, repeat = 2)]
       
        # PackA0
        waitLRA0 = max(lra0[1])+1
        startPACKA0 = waitLRA0

        packAOffset = [ 
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
        packA0 = [x + startPACKA0 for x in packAOffset]

        halfMFMA = numMfma//2
        assert max(packA0) < halfMFMA, f"max(packA0) {max(packA0)} >= halfMFMA {halfMFMA}"

        # LRA3 interleaved with GRB (2nd half)
        startLRA3 = halfMFMA
        lra3 = [create_range(min_val = startLRA3+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = startLRA3, num = numLrReadA // 2, step = 2, repeat = 2)]

        # M0 update before barrier to prevent bad interleaving between the 2 codepaths
        grB[0] += [startLRA3-2,startLRA3]
        grB[1] += [startLRA3-2,startLRA3+1]

        grB[0] += create_range(min_val = startLRA3+2,num = 2,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3+3,num = 2,step = 2, repeat = 2)
        
        # LRB3 + PACKA3 & PACKB3
        startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        lrb3 = create_range(min_val = startLRB3,num=numLrReadB - 2,step=1,repeat=1)
        
        grA = [create_range(min_val = max(lra3[0])+1, num = 8, step = 1,repeat = 1),
               create_range(min_val = max(lra3[1])+1, num = 8, step = 1,repeat = 1)]
        lrb3 += create_range(min_val = max(lrb3)+3,num=2,step=1,repeat=1)
        
        waitLRB3 = max(lrb3) + 9 

        # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        packB3 = [x + waitLRB3 for x in packBOffset]
        start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        waitLRA3 = max(lrb3)+1
        packA3 = [ 
                   *create_range(min_val = waitLRA3, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+2, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+4, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+5, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+6, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+7, num = 2, step = 1, repeat = 2),
                   ]

        # GRA 2nd half
        grA[0] += create_range(min_val = max(packB3)+1, num = 4, step = 1,repeat = 2)
        grA[1] += create_range(min_val = max(packB3)+1, num = 4, step = 1,repeat = 2)

        syncTable = [                                      
                    waitLRB0, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 1/6 LRB0 to complete"),
                    waitLRB0+1, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 2/6 LRB0 to complete"),
                    waitLRB0+2, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 3/6 LRB0 to complete"),
                    waitLRB0+3, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 4/6 LRB0 to complete"),
                    waitLRB0+4, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 5/6 LRB0 to complete"),
                    waitLRB0+5, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 6/6 LRB0 to complete"),
                    waitLRB0+5, SBarrier(comment="Barrier before GRB"), 

                    # incremental wait on LRA0
                    waitLRA0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
                    waitLRA0+4, SWaitCnt(dscnt=numLrReadA-20, vlcnt=-1, vscnt=-1, comment="Wait for 20 LRA0 to complete"),
                    waitLRA0+5, SWaitCnt(dscnt=numLrReadA-24, vlcnt=-1, vscnt=-1, comment="Wait for 24 LRA0 to complete"),
                    waitLRA0+6, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),

                    startLRA3-1,SWaitCnt(dscnt=-1, vlcnt=3, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRA3-1,SBarrier(comment="Sync before GRA, LRA3 & LRB3"),

                    # incremental wait on LRA3 & LRB3
                    waitLRA3, SWaitCnt(dscnt=15, vlcnt=-1, vscnt=-1, comment="Wait for 17/32 LRA3 to complete"),         
                    waitLRA3+4, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    
                    waitLRA3+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),                    

                    ]

        syncCode = syncTable[1::2]
        optSchedule = {

            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [*lra0],
            'LRB0': [lrb0],
            'LRSA': [[packA0[4]]],#Use slot between MFMA 16x16x32 & 4x4x4 for LRSA
            'LRSB': [[packA0[4]]],
            'PackA0' : [packA0],
            'PackB0' : [packB0],
            'GRA': [*grA],
            'GRB': [*grB],              
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 14
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
