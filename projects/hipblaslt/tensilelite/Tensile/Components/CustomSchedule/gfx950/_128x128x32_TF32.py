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

from rocisa.instruction import SNop
from ..dispatch import RegisterSchedule
from ..shared import (
    ScheduleInfo,
    SyncSchedule,
    TileConfig,
    count_items,
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x32_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 4 * 4 * 3    # 128 MT0 / 2 WT0 / 16 mfma dim  * 128/2/16 * 3 bf16 MFMAs per tf32 mfma

    optSchedule = dict()
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    syncs = SyncSchedule()
    syncCode = []   
    snops: list[tuple[int, SNop]] = []
    snopCode = []

    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True

        lra0 = [0,0,1,1]
        syncs.add( 3, dscnt=2, comment="Wait for the first 2 LRA0 to complete before pack")
        syncs.add( 5, dscnt=0, comment="Wait for the rest    LRA0 to complete before pack")
        pack_a0 = [3,3,4,4, 7,7, 8,8,9,9, 5,5,6,6, 7,7, 10,10,11,11]
        pack_b0 = [12,12,13,13, 16,16, 17,17,18,18,  14,14,15,15, 16,16,  19,19,20,20]

        lrb0 = [6,6,7,7]
        syncs.add(11, dscnt=0, comment="Wait for LRB0 to complete before pack",
                  barrier=True, barrier_comment="Wait for all waves to finish LRs before GRs")
        grinca = [0,1,2, 2,2,2, 2,4,5]
        grincb = [6,8,9, 10,11,12, 13,14,15]
        lrsa = [13]
        lrsb = [14]
        lwsa = [45]
        lwsb = [45]
        
        gra = [15,17, 18,19, 20,21, 25,26]
        grb = [27,28, 31,33, 36,37, 39,40]
        num_gr = (len(gra[1::2]) + len(grb[1::2]))
        
        gr_wait = 23
        v = count_items(gra[1::2]+grb[1::2], ev=gr_wait)
        syncs.add(gr_wait, vlcnt=v, barrier=True, comment = "Wait for previous GRA&B")

        lrb3 = [24,24,25,25]
        syncs.add( 28, dscnt=2, comment="Wait for the first 2 LRB3 to complete")
        syncs.add( 30, dscnt=0, comment="Wait for the rest    LRB3 to complete")
        pack_b3 = [28,28,29,29, 32,32,  33,33,34,34,  30,30,31,31, 32,32,  35,35,36,36]
        
        lra3 = [36,36,37,37]
        syncs.add(39, dscnt=2, comment="Wait for the first 2 LRA3 to complete")
        syncs.add(41, dscnt=0, comment="Wait for the rest    LRA3 to complete")
        pack_a3 = [39,39,40,40, 43,43, 44,44,45,45, 41,41,42,42, 43,43, 46,46,47,47]

    else:
        return False, None

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'GRA':    [gra],
        'GRB':    [grb],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA3':   [lra3],
        'LRB3':   [lrb3],
        'PackB3': [pack_b3],
        'PackA3': [pack_a3],
        'LCC':    [[n_mfma-2, n_mfma-1]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    if snops:
        optSchedule['SNOP'] = [ [s[0] for s in snops] ]
        snopCode = [s[1] for s in snops]
 
    kernel["MfmaInitCVgprs"] = True
    kernel["UsePLRPack"] = True
    opt1 = ScheduleInfo(1, n_mfma, optSchedule, syncCode, nglshift, nllshift, snopCode=snopCode)
    return True, opt1
