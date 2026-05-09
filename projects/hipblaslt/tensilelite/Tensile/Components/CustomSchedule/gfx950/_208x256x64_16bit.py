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
    isNN,
    isNT,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(208, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_208x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 104
    syncs = SyncSchedule()

    if isTN(kernel) and not useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,5,7,9,11,13,15,17,19,20]

        syncs.add(12, dscnt=8, comment="wait for the rest of LRB1 to complete")
        grinca = [4,4,4,10,10,10,14,14,14]
        lrb0   = [22,24,26,28]
        grincb = [16,16,16,18,18,18,21,21,21]

        syncs.add(29, dscnt=4, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        # one index for two instructions
        gra    = [30,31,32,33,34,35,36,37,38,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67]

        syncs.add(40, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        grb    = [69,71,73,77,81,85,89,93]
        num_gr = len(gra) + len(grb)
        lrsa   = [50]
        lrsb   = [50]
        lwsa   = [70]
        lwsb   = [70]

        syncs.add(72, vlcnt=num_gr-6, barrier=True, comment="wait for previous set of global reads")
        lra1   = [73,74,75,76,78,82,84,86,88,90,92,94,96]
        lrb1   = [80,98,99,100]

    elif isNN(kernel) and useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        grinca = [8,9,10,11,13,14,15,16,17]
        grincb = [19,20,21,22,23,24,25,26,27]

        syncs.add(12, dscnt=12, comment="wait for the rest of LRB1 to complete")
        lra0   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # 26 loads

        syncs.add(30, dscnt=2, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        lrb0   = [26,28,30,33]

        syncs.add(38, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        gra    = [31,32,34,36,37,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67,68,69,70,71] # 26 loads
        grb    = [73,74,81,83,87,91,92,93]
        num_gr = len(gra) + len(grb)

        syncs.add(72, vlcnt=num_gr-8, barrier=True, comment="wait for previous set of global reads")
        lra1   = [73,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99] # 26 loads
        lrb1   = [74,100,101,102]

        lrsa   = [49]
        lrsb   = [51]
        lwsa   = [84]
        lwsb   = [85]

    elif isNT(kernel) and useLDSTr and TLDS==0:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        grinca = [8,9,10,11,13,14,15,16,17]
        grincb = [19,20,21,22,23,24,25,26,27]

        syncs.add(12, dscnt=12, comment="wait for the rest of LRB1 to complete")
        lra0   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # 26 loads
        lrb0   = [27,29,31,33,35,38,40,42] # 8 loads
        syncs.add(30, dscnt=2, barrier=True, comment="wait for all LRA0 to complete before GRA start")

        syncs.add(49, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        gra    = [31,32,34,36,37,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67,68,69,70,71] # 26 loads
        grb    = [73,74,81,83,87,91,92,93] # 8 loads
        num_gr = len(gra) + len(grb)

        # 8 GRBs from previous iteration + 16 GRAs from current iteration (indices 31-57) can be still pending
        syncs.add(58, vlcnt=(8+16), barrier=True, comment="wait for previous set of GRA")
        lra1   = [59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] # 26 loads

        # 20 GRAs (indices 31-64) from current iteration can be still pending
        syncs.add(65,vlcnt=20, barrier=True, comment="wait for previous set of GRB")
        lrb1   = [66,85,86,88,90,92,94,96] # 8 loads

        lrsa   = [49]
        lrsb   = [51]
        lwsa   = [84]
        lwsb   = [85]
    else:
        return False, None

    def duplicate_list_items(input_list, repeat_count):
        """Example: duplicate_list_items([1, 2, 3], 3) => [1,1,1, 2,2,2, 3,3,3]"""
        return [item for item in input_list for _ in range(repeat_count)]

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'LRA0':   [lra0],
        'GRIncA': [grinca],
        'LRB0':   [lrb0],
        'GRIncB': [grincb],
        # Note: each GRA/GRB item corresponds to two instructions (addr increment and read). So duplicate each item twice.
        'GRA':    [duplicate_list_items(gra, 2)],
        'GRB':    [duplicate_list_items(grb, 2)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'LCC':    [[numMfma-2, numMfma-1]],
    }
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
