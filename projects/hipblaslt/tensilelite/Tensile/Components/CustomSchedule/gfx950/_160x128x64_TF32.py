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
    isNN,
    isTF32,
    isTN,
    switch_A_B_schedule,
)
from ._128x160x64_TF32 import _get_schedule_128x160x64_TF32


@RegisterSchedule(
    tile_config=TileConfig(160, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_160x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 120
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []

    if isNN(kernel) and useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UsePLRPack"] = True
        syncs.add(11, dscnt=8, comment="wait for LRB0 before pack to complete")
        syncs.add(16, dscnt=8, barrier=True, comment="wait for LRB0 before pack to complete", barrier_comment="barrier for GRA")
        syncs.add(33, dscnt=0, comment="wait for LRA0 before pack to complete")
        syncs.add(59, vlcnt=8, barrier=True, comment="wait for previous set of global reads", barrier_comment="barrier for GRB")
        syncs.add(73, dscnt=8, comment="wait for LRB1 before pack to complete")
        syncs.add(78, dscnt=8, comment="wait for LRB1 before pack to complete")
        syncs.add(94, dscnt=0, comment="wait for LRA1 before pack to complete")

        optSchedule = {
            'SYNC': [syncs.get_indicies()],

            'GRIncB': [[0, 1, 2, 3, 4, 6, 7, 8, 9]],
            'LRB0'  : [[0,2,
                        4,5, 
                        6,7, 
                        9,11],
                                [0,1,
                                 3,5, 
                                 6,8,
                                 10,12]],

            'GRIncA': [[10, 11, 12, 13, 14, 15, 15, 16, 16]],

            'PackB0': [[13,13,14,14, 21,21, 22,22,23,23, 
                        15,15,16,16, 21,21, 24,24,25,25, 
                        17,17,18,18, 21,21, 26,26,27,27, 
                        19,19,20,20, 21,21, 28,28,29,29]],

            'LRA0'  : [[0, 0, 2, 2, 4, 4, 5, 5,
                        7, 7, 8, 8, 10,10,12,12,
                        13,13,13,13,14,14,14,14,
                        16,16,16,16,18,18,18,18,
                        20,20,20,20,22,24,26,28],
                                                [0, 0, 1, 1, 3, 3, 5, 5,
                                                 7, 7, 9, 9, 11,11,12,12, 
                                                 13,13,13,13,15,15,15,15,
                                                 17,17,17,17,19,19,19,19,
                                                 21,21,21,21,23,23,23,23]],

            'PackA0': [[30,30,31,31, 40,40, 41,41,42,42, 
                        32,32,33,33, 40,40, 43,43,44,44, 
                        34,34,35,35, 40,40, 45,45,46,46, 
                        36,36,37,37, 40,40, 47,47,48,48,
                        38,38,39,39, 40,40, 49,49,50,50]],

            'GRA'   : [[60, 60, 62, 62, 64, 64, 66, 66, 68, 68,  
                        81, 81, 83, 83, 85, 85, 87, 87, 90, 90]],

            'GRB'   : [[17, 17, 22, 22, 27, 27, 32, 32, 42, 42, 47, 47, 52, 52, 57, 57]],

            'LRSA'  : [[58]], 'LRSB'  : [[59]],'LWSA'  : [[118]], 'LWSB'  : [[118]],

            'LRB1'  : [[59,61,
                        63,65,
                        67,69,
                        71,73],
                                    [60,62,
                                     64,66,
                                     68,70,
                                     72,74]],

            'PackB1': [[73,73,74,74, 81,81, 82,82,83,83, 
                        75,75,76,76, 81,81, 84,84,85,85, 
                        77,77,78,78, 81,81, 86,86,87,87, 
                        79,79,80,80, 81,81, 88,88,89,89]],

            'LRA1'  : [[59,59,61,61, 63,63,65,65,
                        67,67,69,69, 71,71,73,73,
                        74,74,74,74, 75,75,75,75,
                        77,77,77,77, 79,79,79,79,
                        81,81,81,81, 83,83,83,83],
                                                    [60,60,62,62, 64,64,65,65,
                                                     66,66,68,68, 70,70,72,72,
                                                     74,74,74,74, 76,76,76,76,
                                                     78,78,78,78, 80,80,80,80,
                                                     81,81,81,81, 82,82,82,82]],

            'PackA1': [[90,90,91,91,            100,100, 101,101,102,102, 
                        92,92,93,93,            100,100, 103,103,104,104, 
                        94,94,95,95,            100,100, 105,105,106,106, 
                        96,96,97,97,            100,100, 107,107,108,108, 
                        98,98,99,99,            100,100, 109,109,110,110]],

            'LCC': [[118, 119]]
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2

        opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)

    elif isTN(kernel) and not useLDSTr and TLDS==1:
        valid, opt = _get_schedule_128x160x64_TF32(kernel, useLDSTr, TLDS)
        if not valid:
            return False, None
        optSchedule = switch_A_B_schedule(opt.optSchedule)
        opt1 = ScheduleInfo(opt.numCodePaths, opt.numMfma, optSchedule, opt.syncCode, opt.nglshift, opt.nllshift)

    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
