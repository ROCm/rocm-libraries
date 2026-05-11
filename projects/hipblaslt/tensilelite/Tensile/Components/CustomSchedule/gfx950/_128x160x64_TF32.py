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
    isTF32,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 160, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x160x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 120
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []
    gr_inc_step = 0

    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UsePLRPack"] = True

        grinca = [0,0,1,1,2,2,3,3,4]
        grincb = [4,5,5,6,6,7,7,8,8]
        lrsa   = [58]
        lrsb   = [59]
        lwsa   = [118]
        lwsb   = [118]

        pack_a = [0,0,1,1, 8,8, 9,9,10,10,
                  2,2,3,3, 8,8, 11,11,12,12,
                  4,4,5,5, 8,8, 13,13,14,14,
                  6,6,7,7, 8,8, 15,15,16,16
                  ]
        pack_b = [0,0,1,1, 10,10, 11,11,12,12,
                  2,2,3,3, 10,10, 13,13,14,14,
                  4,4,5,5, 10,10, 15,15,16,16,
                  6,6,7,7, 10,10, 17,17,18,18,
                  8,8,9,9, 10,10, 19,19,20,20
                  ]
        lra0   = [0,1,2,3,4,5,6,7]
        syncs.add(                 12, dscnt=4, barrier=True, comment="wait for LRA0 before pack to complete + barrier for GRA")
        pack_a0 = [                i+13 for i in pack_a]  ## last element = 13 + 16 = 29

        lrb0   = [               8,9,10,11, 13,14,15,16, 18,19]
        syncs.add(                                               24, dscnt=0, comment="wait for LRB0 before pack to complete")
        pack_b0 = [                                                  i+30 for i in pack_b]  ## last element = 30 + 20 = 50

        gra    = [                                    17,22,27,32, 42,47,52,57] # one index for two instructions
        grb    = [                                                               67,71,75,79, 89,93,97,101, 112,116] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                            59, vlcnt=8, barrier=True, comment="wait for previous set of global reads + barrier for GRB")

        lra1   = [60,61,62,63,64,65,66,67]
        syncs.add(                          72, dscnt=4, comment="wait for LRA1 before pack to complete")
        pack_a1 = [                         i+73 for i in pack_a]  ## last element = 73 + 16 = 89

        lrb1   = [                        68,69,70,71, 73,74,75,76, 78,79]
        syncs.add(                                                            85, dscnt=0, comment="wait for LRB1 before pack to complete")
        pack_b1 = [                                                           i+90 for i in pack_b]  ## last element = 90 + 20 = 110

        optSchedule = {
            'SYNC':   [syncs.get_indicies()],
            'GRIncA': [grinca],
            'GRIncB': [grincb],
            'LRA0':   [lra0],
            'LRB0':   [lrb0],
            'PackA0': [pack_a0],
            'PackB0': [pack_b0],
            'GRA':    [duplicate_list_items(gra, 2, gr_inc_step),
                       duplicate_list_items([x+1 for x in gra], 2, gr_inc_step)],
            'GRB':    [duplicate_list_items(grb, 2, gr_inc_step),
                       duplicate_list_items([x+1 for x in grb], 2, gr_inc_step)],
            'LRSA':   [lrsa],
            'LRSB':   [lrsb],
            'LWSA':   [lwsa],
            'LWSB':   [lwsb],
            'LRA1':   [lra1],
            'LRB1':   [lrb1],
            'PackB1': [pack_b1],
            'PackA1': [pack_a1],
            'LCC':    [[n_mfma-2, n_mfma-1]],
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = num_gr
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
