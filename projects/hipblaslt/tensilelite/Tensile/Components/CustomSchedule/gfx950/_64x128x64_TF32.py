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
    tile_config=TileConfig(64, 128, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 48
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
        lrsa   = [23]
        lrsb   = [23]
        lwsa   = [47]
        lwsb   = [47]

        pack_a_offset = [0,0,0,0, 2,2, 3,3,3,3,
                         1,1,1,1, 2,2, 4,4,4,4
                        ]
        pack_b_offset = [0,0,0,0, 4,4, 5,5,5,5,
                         1,1,1,1, 4,4, 6,6,6,6,
                         2,2,2,2, 4,4, 7,7,7,7,
                         3,3,3,3, 4,4, 8,8,8,8
                        ]
        lra0   = [0,1,2,3]
        syncs.add(                8, dscnt=5, comment="wait for necessary LRA0 before pack to start")
        syncs.add(                10, dscnt=4, barrier=True, comment="wait for remaining LRA0 before pack to complete + barrier for GRA")
        pack_a0 = [                i+9 for i in pack_a_offset]  ## last index = 9 + 4 = 13

        lrb0   = [        4,5, 7, 9,10,11,12,13]
        syncs.add(                           14, dscnt=4, comment="wait for necessary LRB0 before pack to start")
        syncs.add(                           16, dscnt=0, comment="wait for remaining LRB0 before pack to complete")
        pack_b0 = [                          i+14 for i in pack_b_offset]  ## last index = 14 + 8 = 22

        gra    = [                 10,13,17,21] # one index for two instructions
        grb    = [                            24,28,32,35,38,41,43,45] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                            24, vlcnt=4, barrier=True, comment="wait for previous set of global reads + barrier for GRB")

        lra1   = [24,25,26,27]
        syncs.add(                       32, dscnt=5, comment="wait for necessary LRA1 before pack to start")
        syncs.add(                       34, dscnt=4, comment="wait for remaining LRA1 before pack to complete")
        pack_a1 = [                         i+33 for i in pack_a_offset]  ## last index = 33 + 4 = 37

        lrb1   = [           28,29, 31,32, 34,35,36,37]
        syncs.add(                                     38, dscnt=4, comment="wait for necessary LRB1 before pack to start")
        syncs.add(                                     40, dscnt=0, comment="wait for remaining LRB1 before pack to complete")
        pack_b1 = [                                    i+38 for i in pack_b_offset]  ## last index = 38 + 8 = 46

        optSchedule = {
            'SYNC':   [syncs.get_indicies()],
            'GRIncA': [grinca],
            'GRIncB': [grincb],
            'LRA0':   [lra0],
            'LRB0':   [lrb0],
            'PackA0': [pack_a0],
            'PackB0': [pack_b0],
            'GRA':    [duplicate_list_items(gra, 2, gr_inc_step)],
            'GRB':    [duplicate_list_items(grb, 2, gr_inc_step)],
            'LRSA':   [lrsa],
            'LRSB':   [lrsb],
            'LWSA':   [lwsa],
            'LWSB':   [lwsb],
            'LRA1':   [lra1],
            'LRB1':   [lrb1],
            'PackB1': [pack_b1],
            'PackA1': [pack_a1],
            'LCC':    [[n_mfma-1, n_mfma-1]],
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = num_gr
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(1, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
