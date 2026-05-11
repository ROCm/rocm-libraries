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
    is16bit,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(128, 192, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x192x64_16bit(kernel, useLDSTr, TLDS):
    """128x192x64 TN schedule (BF16/FP16)."""
    kernel["MfmaInitCVgprs"] = True

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    numMfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"] # 48

    syncs = SyncSchedule()
    gr_inc_step = 0

    if isTN(kernel) and not useLDSTr and TLDS==1:
        grinca = [0,1,2, 3,4,5, 6,7,7]
        grincb = [7,8,9, 9,9,10, 10,10,11]

        syncs.add(-1, dscnt=5, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3]
        syncs.add(      3, dscnt=3, comment="wait for the rest of LRB1 to complete")
        lrb0   = [       4,5,6,8,11,  14]

        syncs.add(                 12, dscnt=5, barrier=True, comment="wait for LRA0 before GRA start")
        gra    = [                   13,15,17,19] # one index for two instructions
        
        syncs.add(                             21, dscnt=0, vlcnt=4+6, barrier=True, comment="wait for LRB0 before GRB start + wait for previous GRAs before LRA1")
        grb    = [                              21,24,    27,31,34,37] # one index for two instructions
        lra1   = [                                22,25,26,29]
        syncs.add(                                                35, vlcnt=4+5, barrier=True, comment="wait for previous GRBs to complete before LRB1")
        lrb1   = [                                                35,38,40,42,44,46]

        num_gr = len(gra) + len(grb)
        lrsa   = [18]
        lrsb   = [18]
        lwsa   = [30]
        lwsb   = [30]

    else:
        return False, None

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'LRA0':   [lra0],
        'GRIncA': [grinca],
        'LRB0':   [lrb0],
        'GRIncB': [grincb],
        # Note: each GRA/GRB item corresponds to two instructions (addr increment and read). So duplicate each item twice.
        'GRA':    [duplicate_list_items(gra, 2, gr_inc_step)],
        'GRB':    [duplicate_list_items(grb, 2, gr_inc_step)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'LCC':    [[numMfma-3, numMfma-2]],
    }
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)

    return True, opt1
