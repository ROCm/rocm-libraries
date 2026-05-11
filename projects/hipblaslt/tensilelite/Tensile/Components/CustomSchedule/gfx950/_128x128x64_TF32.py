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

from ..dispatch import RegisterSchedule
from ..shared import (
    ScheduleInfo,
    SyncSchedule,
    TileConfig,
    duplicate_list_items,
    isNN,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 128, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 96
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []   
    gr_inc_step = 1

    if isTN(kernel) and not useLDSTr and TLDS==1:
        offset=[0,0,1,1, 8,8,  9, 9,10,10, 
                2,2,3,3, 8,8, 11,11,12,12,
                4,4,5,5, 8,8, 13,13,14,14, 
                6,6,7,7, 8,8, 15,15,16,16]
        
        lra0   = [ 0,1,2,3,4,5,6,7]
        lrb0   = [   8,9,10,11,12,13,14,15]
        #                wait then read
        syncs.add(       10, dscnt=6, comment="wait for the first 2 LRAs before packing")
        syncs.add(       14, dscnt=6, comment="wait for the rest of LRAs before packing them")
        pack_a0= [          i+11 for i in offset] # last at 27
        # because of GR starting at 22, we need barrier at 21, will use that for sync too.
        syncs.add(                          21, dscnt=0, comment="wait for LRBs before the packing them",
                                            barrier=True, barrier_comment="barrier before GR")
        pack_b0= [                                i+28 for i in offset] # last at 44

        grinca = [0,0,1,1,2,2,3,3,4]
        grincb = [5,5,6,6,7,7,8,8,9]
        lrsa   = [45]
        lrsb   = [45]
        lwsa   = [72]
        lwsb   = [72]        
        
        gra    = [                            22,25,29,33, 37,41,45,49] # one index for two instructions
        grb    = [                                                    53,57,61, 64,69,75,79,84] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                 48, vlcnt=7, barrier=True, comment="wait for the previous GRs")

        lra1   =[[                                                 48,49,50,51,52,53,54,55]]
        lrb1   = [                                                    56,57,  59,60,61,62,63,64]
        syncs.add(                                                          58, dscnt=6, comment="wait for the first two LRAs before packing")
        syncs.add(                                                          63, dscnt=6, comment="wait for the rest of LRAs before packing them")
        pack_a1 = [                                                           i+59 for i in offset] # last at 75
        syncs.add(                                                                                 76, dscnt=0, comment="wait for LRBs before the packing them")
        pack_b1 =[                                                                                 i+77 for i in offset] # last at 93

    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 4:
        offset=[0,0,1,1, 8,8,  9, 9,10,10, 
                2,2,3,3, 8,8, 11,11,12,12,
                4,4,5,5, 8,8, 13,13,14,14, 
                6,6,7,7, 8,8, 15,15,16,16]
        
        lra0   = [ 0,0,2,2,4,4,6,6]
        lrb0   = [                                       11,11,13,13,15,15,17,17]
        #                wait then read
        syncs.add(             6, dscnt=2, comment="wait for the first 4 LRAs before swapping/packing")
        syncs.add(                     9, dscnt=0, comment="wait for the rest of LRAs before swapping/packing them")
        pack_a0= [              7,7,7, 9,9,9, 8,8, 10,10, 8, 10] # swap instructions
        pack_a0+=[                                       i+11 for i in offset] # last at 27
        # because of GR starting at 22, we need barrier at 21, will use that for sync too.
        syncs.add(                                                                21, dscnt=0, comment="wait for LRBs before the packing them",
                                                                                  barrier=True, barrier_comment="barrier before GR")
        pack_b0= [                                                                  i+28 for i in offset] # last at 44

        grinca = [0,1,1,1,2,3,3,3,4]
        grincb = [5,5,5,7,8,9,10,19,19]
        lrsa   = [45]
        lrsb   = [45]
        lwsa   = [72]
        lwsb   = [72]        
        
        gra    = [                            22,25,29,33, 37,41,45,49] # one index for two instructions
        grb    = [                                                    53,57,61, 64,69,75,79,84] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                 48, vlcnt=7, barrier=True, comment="wait for the previous GRs")
        lra1   =[[                                                 48,48,50,50,52,52,54,54],
                                                                   [49,49,51,51,53,53,54,54]]
        lrb1   = [                                                                                  59,59,  61,61,63,63,65,65]
        syncs.add(                                                                   54, dscnt=2, comment="wait for the first two LRAs before packing")
        syncs.add(                                                                             57, dscnt=0, comment="wait for the rest of LRAs before packing them")
        pack_a1 =[                                                                    55,55,55, 57,57,57, 55,55, 57,57, 56, 58] # swap instructions
        pack_a1+= [                                                                                 i+59 for i in offset] # last at 75
        syncs.add(                                                                                     76, dscnt=0, comment="wait for LRBs before the packing them")
        pack_b1 =[                                                                                     i+77 for i in offset] # last at 93

    else:
        return False, None  
    
    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'GRA':    [duplicate_list_items(gra,                2, gr_inc_step),
                   duplicate_list_items([i+1 for i in gra], 2, gr_inc_step)],
        'GRB':    [duplicate_list_items(grb,                2, gr_inc_step),
                   duplicate_list_items([i+1 for i in grb], 2, gr_inc_step)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'LRA1':   lra1,
        'LRB1':   [lrb1],
        'PackB1': [pack_b1],
        'PackA1': [pack_a1],
        'LCC':    [[n_mfma-2, n_mfma-2]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["UseMFMAF32XEmulation"] = True
    kernel["UseDot2F32XEmulation"] = False
    kernel["UsePLRPack"] = True
    opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
