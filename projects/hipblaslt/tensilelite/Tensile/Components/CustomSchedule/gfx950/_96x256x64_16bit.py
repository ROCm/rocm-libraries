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

from rocisa.instruction import SBarrier, SNop, SWaitCnt
from ..dispatch import RegisterSchedule
from ..shared import (
    ScheduleInfo,
    SyncSchedule,
    TileConfig,
    is16bit,
    isNN,
    isNT,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(96, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_96x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

    numMfma = 48
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 11

    if isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True

        syncTable = [
            7, SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Finish all LRA1s and LRB1s"),

            9, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRB0 done"),
            9, SBarrier(comment=""),

            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRA0 done"),

            35, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prev GRBs"),
            35, SBarrier(comment=""),

            43, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prev GRA"),
            43, SBarrier(comment=""),

            47, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish all LRB1 and 1/3 LRA1"),
        ]

        syncCode = syncTable[1::2]
        optSchedule = {
            'GRIncA' : [[0,0,0,    2,2,2, 3, 4,4],
                        [-1,-1,-1, 1,1,1, 3, 3,3]],
            'GRIncB' : [[4, 5,5,5, 6,6,6, 7,7],
                        [4, 5,5,5, 6,6,6, 7,7]],

            'LRB0'   : [[-1,-1,-1, 1,1,1, 3,3],
                        [0,0,0,    2,2,2, 4,4]],
            'LRSB'   : [[8], [9]],
            
            'SYNC'   : [syncTable[::2]],

            # Actually loads GRB
            'GRA'    : [[8,9,  11,11, 13,13, 15,15,  20,20, 22,22, 24,24, 26,26],
                        [8,10, 12,12, 14,14, 16,16,  21,21, 23,23, 25,25, 27,27]],
            'LWSB'   : [[32]],

            'LRA0'   : [[17, 17, 17],
                        [15, 15, 15]],
            'LRSA'   : [[19]],
            
            # Actually loads GRA
            'GRB'    : [[36,36, 38,38, 40,40],
                        [37,37, 39,39, 41,41]],
            'LWSA'   : [[45]],
            
            'LRB1'   : [[35,35, 37,37, 39,39, 41,41],
                        [36,36, 38,38, 40,40, 42,42]],
            'LRA1'   : [[43,44,46],
                        [43,45,46]],
            
            'LCC'   : [[47, 47]],
        }

        # Reorder to basically create the 256x96 case
        mfmaReorder = [
             0,  3,  6,  9, 12, 15, 18, 21,
             1,  4,  7, 10, 13, 16, 19, 22,
             2,  5,  8, 11, 14, 17, 20, 23,
              # Second half
             24, 27, 30, 33, 36, 39, 42, 45,
             25, 28, 31, 34, 37, 40, 43, 46,
             26, 29, 32, 35, 38, 41, 44, 47
        ]

        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)

    elif isNT(kernel) and useLDSTr and TLDS == 0:
        # A: MIWaveTileA=3 => 2*3 = 6 local/global reads
        # B: MIWaveTileB=8 => 2*8 = 16 local/global reads
        syncTable = [
            -1, SWaitCnt(dscnt=12, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW for iteration == 0"),
             5, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW for iteration == 0"),
            12, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA"),
            12, SBarrier(comment="barrier after LRA0, before GRA"),
            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB"),
            23, SBarrier(comment="barrier after LRB0, before GRB"),

            25, SWaitCnt(dscnt=-1, vlcnt=11+1, vscnt=-1, comment="wait for global reads before next-tile LR"),
            25, SBarrier(comment="final barrier before LRA1/LRB1"),
            36, SWaitCnt(dscnt=-1, vlcnt=11-2, vscnt=-1, comment="wait for global reads before next-tile LR"),
            36, SBarrier(comment="final barrier before LRA1/LRB1"),
        ]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],

            # Address increments
            'GRIncA' : [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            'GRIncB' : [[4, 5, 5, 6, 6, 7, 7, 8, 8]],

            # Current-iteration local reads
            'LRA0'   : [[0, 1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 4, 6]],
            # Ensure LRB0 completes early enough for upcoming MFMA consumers.
            'LRB0'   : [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                        [7, 8, 9, 10, 11, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]],

            # Global reads (DirectToLds). Issue after LRA0 barrier, while finishing LRB0.
            # Do not interleave GRA between GRIncB indices (SCC hazard).
            'GRA'    : [[12, 12, 18, 18, 21, 21],
                        [13, 13, 19, 19, 22, 22]],
            # Start GRB after the LRB0 barrier. Use stride=2 starting at 24.
            'GRB'    : [[24, 24, 26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 38, 38, 40, 40],
                        [24, 24, 27, 27, 29, 29, 31, 31, 33, 33, 35, 35, 37, 37, 39, 39]],

            # Next-iteration local reads
            'LRA1'   : [[25, 26, 27, 28, 29, 30],
                        [26, 27, 28, 29, 30, 31]],
            # Keep within the mfma window; repeated indices are allowed (multiple instructions scheduled at same mfma slot).
            'LRB1'   : [[36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 43, 45, 46, 47],
                        [37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 44, 46, 47, 47]],

            # Epilogue-related
            'LRSA'   : [[22]],
            'LRSB'   : [[23]],
            'LWSA'   : [[43]],
            'LWSB'   : [[44]],
            'LCC'    : [[45, 46]],
        }

        syncCode = syncTable[1::2]
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and not useLDSTr and TLDS==1:
        snops = []
        syncs = SyncSchedule()
        syncs.add(-1, dscnt=4, comment="Wait for prior local read local write")
        syncs.add(2, dscnt=3, comment="Wait for prior local read local write")
        syncs.add(8, dscnt=6, comment="Wait for partial LRA0")
        syncs.add(16, dscnt=0, barrier=True, comment="Wait for LRA0 to complete before starting LRB0+GRA")
        syncs.add(23, dscnt=0, vlcnt=3, barrier=True, comment="Wait for LRB0+GRA")

        snopIdxs = [1, 25]
        snops = [[x, SNop(1, comment="")] for x in snopIdxs]

        lra0 = [0,0,1,1,2,2,3,3,4,5,5,6,6,7,7,8,8,9,10,11,12,13,14,15]

        # Issue LRB0+GRA after LRA0 completes.
        lrb0 = [16, 17, 18, 19, 20, 21, 22, 22]
        grA =  [18, 18, 20, 20, 21, 21]

        # Issue LRA1+GRB after LRB0+GRA complete.
        lra1 = [24,24, 25,25, 26,26, 27,27, 28, 29,29, 30,30, 31,31, 32,32, 33,34,35,36,37,38,39]
        grB  = [24,24,26,26,28,28,30,30,32,32,36,36,38,38,40,40]
        lrb1 = [39,40,41,42,43,44,45,46]

        packA1 = [
            -1,-1,-1,-1,-1,-1,
            0,0,0,0,
            1,1,
        ]
        packA0 = [
            23,23,23,23,23,23,
            24,24,24,24,
            25,25,
        ]

        # GRIncs should be ordered AFTER LRs.
        grIncA = [0,1,2,3,4,5,6,7,8]
        grIncB = [9,10,11,12,13,14,15,16,17]

        lwsa = [46]
        lwsb = [46]
        lrsa = [22]
        lrsb = [22]
        num_gr = (len(grA) + len(grB)) // 2
        optSchedule = {
            'SYNC'   : [syncs.get_indicies()],
            'LRA0'   : [lra0],
            'LRA1'   : [lra1],
            'PackA0' : [packA0],
            'PackA1' : [packA1],
            'LRB0'   : [lrb0],
            'LRB1'   : [lrb1],
            'GRIncA' : [grIncA],
            'GRIncB' : [grIncB],
            'GRA'    : [grA],
            'GRB'    : [grB],
            'LRSA'   : [lrsa],
            'LRSB'   : [lrsb],
            'LWSA'   : [lwsa],
            'LWSB'   : [lwsb],
            'LCC'    : [[47, 47]],
        }
        nllshift = nglshift = num_gr
        if snops:
            optSchedule['SNOP'] = [[s[0] for s in snops]]
            snopCode = [s[1] for s in snops]

        opt1 = ScheduleInfo(1, 48, optSchedule=optSchedule, syncCode=syncs.get_code(), nglshift=nglshift, nllshift=nllshift, snopCode=snopCode)
    else:
        return False, None

    return True, opt1
