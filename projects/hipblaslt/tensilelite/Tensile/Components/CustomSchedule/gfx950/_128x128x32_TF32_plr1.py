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
    isNT,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[32, 32, 16, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x32_TF32_plr1(kernel, useLDSTr, TLDS):
    n_mfma = 128//2//32 * 128//2//32 * 3 * 2    # 128 MT0 / 2 WT0 / 32 mfma dim  * 128/2/32 * 3 bf16 MFMAs per tf32 mfma * 2 PLR=1

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    syncs = SyncSchedule()
    gr_inc_step = 0
    num_code_paths = 1

    if isTN(kernel) and not useLDSTr and TLDS==1:
        lra0   = [0,1,2,3]
        lrb0   = [       4,5,6,7]
        #                wait then read
        syncs.add(       4, dscnt=2, comment="wait for the first 2 LRAs before packing")
        syncs.add(         5, dscnt=1, comment="wait for the rest of LRAs before packing them")
        pack_a0 = [      4,4,4,4, 6,6, 7,7,7,7,
                           5,5,5,5, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs before the packing them",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               9,9,9,9, 10,10, 11,11,11,11,
                                                 9,9,9,9, 10,10, 11,11,11,11]

        grinca = [0,0,0,1,1,1,2,2,2]
        grincb = [3,3,3,6,6,6,6,6,6]
        lrsa   = [10]
        lrsb   = [10]    
        
        gra    = [                                 10,10,11,11] # one index for two instructions
        grb    = [                                              13,13,14,14] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,13,14] # twice on 12 since we are waiting for GRA anyway at 12
        lrb1   = [                                                        15,16,16,17]
        #                                                                 wait then read
        syncs.add(                                                        15, dscnt=2, vlcnt=8, comment="wait for the first 2 LRAs before packing. Also wait for GRBs",
                                                                              barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        syncs.add(                                                            17, dscnt=3, comment="wait for the rest of LRAs before packing them")
        pack_a1 = [                                                          16,16,16,16, 20,20, 21,21,21,21,
                                                                              17,17,17,17, 20,20, 21,21,21,21]
        syncs.add(                                                              18, dscnt=2, comment="wait for 2 LRBs before the packing them")
        syncs.add(                                                               19, dscnt=0, comment="wait for the rest of LRBs before the packing them")
        pack_b1= [                                                              18,18,18,18, 20,20, 22,22,22,22,
                                                                                 19,19,19,19, 20,20, 22,22,23,23]
        lwsa   = [                                                                          20] # use delay before mfma4x4x4
        lwsb   = [                                                                          20]
        
    elif isNN(kernel) and TLDS==1  and kernel["VectorWidthA"] == 2:
        lra0   = [0,0,0,0,
                    1,1,1,1]
        lrb0   = [     3,  4,6,6]
        #                wait then read
        syncs.add(     3, dscnt=4, comment="wait for the first 2x2 LRAs before packing")
        syncs.add(         4, dscnt=1, comment="wait for the rest of LRAs")
        pack_a0 = [    3,3,4,4, # swap instructions, must come after LR and before other packs
                             4,5,5,5, 6,6, 7,7,7,7, 
                             5,5,6,6, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs before the packing them",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               10,10,10,10, 10,10, 11,11,11,11,
                                                 9,9,9,9,     10,10, 11,11,11,11]
        grinca = [0,0,1, 1,2,2, 2,2,2]
        grincb = [2,2,6, 7,7,7, 8,8,8]
        lrsa   = [10]
        lrsb   = [10]   
        
        num_code_paths = 2
        gra   = [                                9,9,   11,11]
        gra2  = [                                 10,10,11,11]
        grb    = [                                              13,        14,14,17] # one index for two instructions
        grb2   = [                                              13,         15,15,17] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,12,12,
                                                                13,13,13,13]
        syncs.add(                                                         14, vlcnt=4+1, barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        lrb1   = [                                                         14,15,16,16]
        syncs.add(                                                            15, dscnt=1, comment="wait for LRAs")
        pack_a1 =[                                                            15,15,16,16, # swap instructions, must come after LR and before other packs
                                                                                17,17,17,17, 20,20, 21,21,21,21,
                                                                                 18,18,18,18, 20,20, 21,21,21,21]
        syncs.add(                                                                19, dscnt=2, comment="wait for the first 2 LRBs before the packing them")
        syncs.add(                                                                 20, dscnt=0, comment="wait for the rest of LRBs")
        pack_b1= [                                                                19,19,19,19, 20,20, 22,22,22,22,
                                                                                   20,20,20,20, 20,20, 22,22,22,22]
        lwsa   = [                                                                            20] # use delay before mfma4x4x4
        lwsb   = [                                                                            20]    
    
    elif isNT(kernel) and useLDSTr and TLDS==0  and kernel["VectorWidthA"] == 2 and kernel["VectorWidthB"] == 2:
        lra0   = [0,0,0,0,
                    1,1,1,1]
        lrb0   = [     3,3,4,4,
                                 6,6,6,6]
        #              wait then read
        syncs.add(     3, dscnt=4, comment="wait for the first 2x2 LRAs before packing")
        syncs.add(         4, dscnt=2, comment="wait for the rest of LRAs")
        pack_a0 = [    3,3,4,4, # swap instructions, must come after LR and before other packs
                             4,5,5,5, 6,6, 7,7,7,7, 
                             5,5,6,6, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               9,9, 9,9, # swap instructions, must come after LR and before other packs
                                                 10,10,10,10, 10,10, 11,11,11,11,
                                                 9,9,9,9,     10,10, 11,11,11,11]
        grinca = [0,0,1, 1,2,2, 2,2,2]
        grincb = [2,2,6, 7,7,7, 8,8,8]
        lrsa   = [10]
        lrsb   = [10]   
        
        num_code_paths = 2
        gra   = [                                9,9,   11,11]
        gra2  = [                                 10,10,11,11]
        grb    = [                                              13,        14,14,17] # one index for two instructions
        grb2   = [                                              13,         15,15,17] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,12,12,
                                                                13,13,13,13]
        syncs.add(                                                         14, vlcnt=4+1, barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        lrb1   = [                                                         14,14,15,15,
                                                                             16,16,16,16]
        syncs.add(                                                            15, dscnt=2, comment="wait for LRAs")
        pack_a1 =[                                                            15,15,16,16, # swap instructions, must come after LR and before other packs
                                                                                17,17,17,17, 20,20, 21,21,21,21,
                                                                                 18,18,18,18, 20,20, 21,21,21,21]
        syncs.add(                                                                19, dscnt=0, comment="wait for LRBs")
        pack_b1= [                                                                19,19,19,19, # swap instructions, must come after LR and before other packs
                                                                                  19,19,19,19, 20,20, 22,22,22,22,
                                                                                   20,20,20,20, 20,20, 22,22,22,22]
        lwsa   = [                                                                            20] # use delay before mfma4x4x4
        lwsb   = [                                                                            20]    

    else:
        return False, None  
    
    final_gra = [duplicate_list_items(gra, 2, gr_inc_step)]
    if num_code_paths == 2:
        final_gra += [duplicate_list_items(gra2, 2, gr_inc_step)]
    
    final_grb = [duplicate_list_items(grb, 2, gr_inc_step)]
    if num_code_paths == 2:
        final_grb += [duplicate_list_items(grb2, 2, gr_inc_step)]

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'GRA':    final_gra,
        'GRB':    final_grb,
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'PackB1': [pack_b1],
        'PackA1': [pack_a1],
        'LCC':    [[n_mfma-1, n_mfma-1]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["UsePLRPack"] = True
    kernel["UseMFMAF32XEmulation"] = True
    kernel["UseDot2F32XEmulation"] = False
    opt1 = ScheduleInfo(num_code_paths, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
