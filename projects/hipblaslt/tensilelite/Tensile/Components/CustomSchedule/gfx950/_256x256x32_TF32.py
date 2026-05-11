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
    isNT,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 256, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 192
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False
        # This schedule follows similar pattern as 192x256x32 TF32 schedule

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 4, step = 1, repeat = 1)
        lra0 += create_range(min_val = max(lra0)+4, num = 4, step = 1, repeat = 1)

        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)
        waitLRA0 = max(grIncA)+5
        startPACKA0 = waitLRA0

        # Use a common packOffset re-ordering for both A and B
        packOffset = [ 
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


        packA0 = [x + startPACKA0 for x in packOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4

        # LRB0 + GRIncB (Split LRB0 into two halves to hide latency)
        lrb0 = create_range(min_val = max(packA0)+1, num = 4, step = 1, repeat = 1)
        lrb0 += create_range(min_val = max(lrb0)+4, num = 4, step = 1, repeat = 1)
        grIncB = create_range(max(lrb0)+1,3,max(lrb0)+4,1,3)
        waitLRB0 = max(grIncB)+6
        startPACKB0 = waitLRB0

        packB0 = [x + startPACKB0 for x in packOffset]

        # GRA - 1st half (4 reads)                
        grA = create_range(min_val = max(packB0)+1, num = 4, step = 2,repeat = 2)

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA

        # LR3
        startLRB3 = halfMFMA
        lrb3 = create_range(min_val = startLRB3, num = 4, step = 1, repeat = 1)
        lrb3 += create_range(min_val = max(lrb3)+4, num = 4, step = 1, repeat = 1)

        # GRA - 2nd half (4 reads)   
        grA += create_range(min_val = max(lrb3)+1, num = 4, step = 2,repeat = 2)
        waitLRB3 = max(grA)+1 

        # PackB3
        packB3 = [x + waitLRB3 for x in packOffset]

        # GRB - 1st half (4 reads) 
        grB = create_range(min_val = max(packB3)+1,num = 4,step = 2, repeat = 2)

        # LRA3 + PACKA3
        startLRA3 = (3*numMfma)//4 
        lra3 = create_range(min_val = startLRA3,num=4,step=1,repeat=1)
        lra3 += create_range(min_val = max(lra3)+4,num=4,step=1,repeat=1)
        waitLRA3 = max(lra3) + 10 
        packA3 = [x + waitLRA3 for x in packOffset]

        # GRB - 2nd half (4 reads) 
        grB += create_range(min_val = max(packA3)+1,num = 4,step = 2, repeat = 2)

        syncTable = [                    
                    waitLRA0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    waitLRB0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),

                    max(packB0)+1, SBarrier(comment="Barrier before GRA&GRB"),

                    startLRB3-1,SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
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

            'GRA': [grA],
            'GRB': [grB],              
            'LRSA': [[max(grIncB)+1]],
            'LRSB': [[max(grIncB)+2]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 16 # vmcnt shift for ngl and nll
    elif isNT(kernel) and not useLDSTr and TLDS==0 and kernel["VectorWidthA"] == 4 and kernel["VectorWidthB"] == 4:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False
        swap_idx =   [1,2,3, # depend on DS1 
                        7,8, # depend on DS2
                         11, # depend on DS3
                      4,5,6, # depend on DS5
                       9,10, # depend on DS6
                         12, # depend on DS7
                        ]
        optSchedule = {
            'SYNC': [[9, 14, 38, 43, 73, 95, 95, 106, 111, 157, 162]],
            'GRIncA': [[0,1,2,3, 5,6,7,8, 11]],
            'GRIncB': [[38, 39, 40, 41, 45, 46, 47, 50, 51]],
            'LRA0': [[0,1,2,3, 5,6,7,8]],
            'LRB0': [[28, 30, 31, 32, 34, 35, 36, 37]],

            'PackA0': [[*[x + 8 for x in swap_idx],
                        20,20,21,21, 28,28, 29,29,30,30,
                        22,22,23,23, 28,28, 31,31,32,32, 
                        24,24,25,25, 28,28, 33,33,34,34, 
                        26,26,27,27, 28,28, 34,35,36,36]],
            'PackB0': [[*[x + 37 for x in swap_idx],
                        55,55,56,56, 63,63, 64,64,65,65,
                        57,57,58,58, 63,63, 66,66,67,67,
                        59,59,60,60, 63,63, 68,68,69,69,
                        61,61,62,62, 63,63, 70,70,71,71]],
            'LRSA': [[50]],
            'LRSB': [[51]],
            'LRB3': [[96,96,98,98, 103,103,105,105],
                    [97,97,99,99,  102,102,104,104]],                    
            'GRA': [[75,75,80,80,85,85,90,90, 107,107,109,109,111,111,113,113],
                    [77,77,82,82,87,87,92,92, 108,108,110,110,112,112,114,114]],
            'LRA3': [[138, 138, 144, 144, 150, 150, 154, 154],
                     [139, 139, 146, 146, 152, 152, 156, 156]],
            'GRB': [[130,130,135,135,140,140,145,145, 165,165,170,170,175,175,180,180],
                    [132,132,137,137,142,142,147,147, 167,167,172,172,177,177,182,182]],
            
            'PackB3': [[*[x + 105 for x in swap_idx],
                        119,119,120,120, 128,128, 129,129,130,130,
                        121,121,122,122, 128,128, 131,131,132,132,
                        123,123,124,124, 128,128, 133,133,134,134,
                        125,125,126,126, 128,128, 135,135,136,136]], 
            'PackA3': [[*[x + 156 for x in swap_idx],
                        169,169,170,170, 177,177, 178,178,179,179,
                        171,171,172,172, 177,177, 180,180,181,181,
                        173,173,174,174, 177,177, 182,182,183,183,
                        175,175,176,176, 177,177, 184,184,185,185]],
            'LWSA': [[187]],
            'LWSB': [[188]],
            'LCC': [[189, 190]],
        }

        syncCode = [                    
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
            SBarrier(comment="Barrier before GRA&GRB"),
            SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
        ]
        nglshift = nllshift = 16 # vmcnt shift for ngl and nll
    else:
        return False, None
        
    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
