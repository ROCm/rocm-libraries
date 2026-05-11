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
    count_items,
    duplicate_list_items,
    is16bit,
    isNN,
    isNT,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(192, 320, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x320x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 120
    nllZeroDscnt = False
    syncs = SyncSchedule()
    gr_inc_step = 0

    if isNN(kernel) and useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=9, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,4,6,8,10,12,14,16,18]

        syncs.add(5, dscnt=5, comment="wait for the rest of LRB1 to complete")
        grinca = [7,7,7,9,9,9,11,11,11]
        grincb = [13,13,13,15,15,15,17,17,17]
        lrb0   = [20,22,24,25,27,29,31,33,35,37]

        syncs.add(26, dscnt=4, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions
        syncs.add(44, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        grb    = [53,58,63,67,72,77,82,86,91,96] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lrsa   = [58]
        lrsb   = [58]

        syncs.add(71, vlcnt=10, barrier=True, comment="wait for previous set of global reads")
        lra1   = [72,74,76,78,80,82,84,87,90,92,98,100]
        lrb1   = [99,106,107,108,109,110,111,112,113,114]
        lwsa   = [95]
        lwsb   = [95]

    elif isTN(kernel) and not useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=9, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,4,6]

        syncs.add(5, dscnt=5, comment="wait for the rest of LRB1 to complete")
        grinca = [0,1,2,3,4,5,6,7,8]
        grincb = [9,10,12,13,14,15,16,17,18]
        lrb0   = [8,10,12,14,16,18,20,22,24,27]

        syncs.add(26, dscnt=9, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions
        syncs.add(44, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        grb    = [53,58,63,67,72,77,82,86,91,96] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lrsa   = [58]
        lrsb   = [58]
        syncs.add(71, vlcnt=10, barrier=True, comment="wait for previous set of global reads")

        lra1   = [72,76,80,84,90,98]
        lrb1   = [99,106,107,108,109,110,111,112,113,114]
        lwsa   = [95]
        lwsb   = [95]

        gr_inc_step = 1

    elif isNT(kernel) and useLDSTr and TLDS == 0:
        lra0   = [0,1,3,5,7, 9,10,12,14,16, 18,19] # 12 loads
        lrb0   = [21,23,25,27,28, 30,32,34,36,37, 39,41,43,45,46, 48,50,52,54,55] # 20 loads
        # need two LRB1 items because a single LRB read gets only half of the data needed for MFMA
        # note, max dscnt value is 15, so in this case 19 will be maxed at 15, thus we will wait more than needed :(
        syncs.add(-1, dscnt=len(lrb0)-2, comment="wait for all LRA1 and two items from LRB1 before starting the sub-iteration")

        i = 5 # next LRB1 is needed at index 6, so insert wait at 5
        syncs.add(i, dscnt=count_items(lra0,ev=i), comment="wait for the rest of LRB1 to complete")
        grinca = [0,1,2,3,4,5,6,7,8]
        grincb = [9,10,12,13,14,15,16,17,18]

        i = 26
        syncs.add(i, dscnt=count_items(lrb0,ev=i), barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions

        lrsa   = [57]
        lrsb   = [58]

        # This wait serves dual purpose
        syncs.add(59, dscnt=len(lrb0)-2, vlcnt=len(gra), barrier=True,
                  comment="wait for the first LRB0 to complete to start 2nd batch of MFMAs and also make GRs from the previous iteration is done before LRA1 starts")
        lra1   = [61,62,63,64,65, 66,67,69,71,73, 75,76] # 12 loads

        i = 65 # next LRB0 is needed at index 66, so insert wait at 65
        syncs.add(i, dscnt=count_items(lra1,ev=i), barrier=True,
                  comment="wait for the rest of LRB0 to complete across all waves before GRB start")
        grb    = [66,70,75,79,84, 88,93,97,102,106] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lwsa   = [68]
        lwsb   = [77]
        lrb1   = [78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116] # 20 loads

        gr_inc_step = 1
        nllZeroDscnt = True
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
        'LCC':    [[numMfma-2, numMfma-1]],
    }

    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, nllZeroDscnt)

    return True, opt1
