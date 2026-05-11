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

from rocisa.instruction import SBarrier, SWaitCnt
from ..dispatch import RegisterSchedule
from ..shared import (
    ScheduleInfo,
    TileConfig,
    is16bit,
    isNN,
    isNT,
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 192, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x192x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS == 1:
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for LRB1-0"),
                     7, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for LRB1"),
                     10, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="wait for LRA0"),
                     10, SBarrier(comment="for GRA"),

                     47, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0-0"),
                     50, SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="for previous GRA"),
                     50, SBarrier(comment="for GRA"),

                     70, SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="for previous GRB"),
                     70, SBarrier(comment="for GRB"),
                     ]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[0,1,2,3,4,5,6,7,8]],
                'GRIncB': [[37,37,38,38,39,39,40,40,41]],
                'LRA0': [[-1, 0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6, 7]],
                 #interleave LRB0 , GRA
                'LRB0': [[7, 9, 11, 13, 15, 17],
                        [8, 10, 12, 14, 16, 18]],
                'GRA': [[10,10, 12,12, 14,14, 16,16, 20,20, 31,31, 33,33, 35,35],
                        [11,11, 13,13, 15,15, 17,17, 21,21, 32,32, 34,34, 36,36]],
                 #interleave GRB, LRB1
                'GRB': [[51,51, 55,55, 59,59, 63,63, 83,83, 85,85],
                        [52,52, 56,56, 60,60, 64,64, 84,84, 86,86]],
                'LRA1': [[50, 52, 57, 60, 62, 64, 66, 68],
                         [51, 53, 58, 61, 63, 65, 67, 69]],

                'LRB1': [[70, 72, 74, 76, 78, 79],
                         [71, 73, 75, 77, 79, 80]],
                'LRSA': [[20]],
                'LRSB': [[64]],
                'LWSA': [[41]],
                'LWSB': [[90]],
                'LCC' : [[95, 95]],
            }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="for LRB1"),
                     29, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for LRB0. For code path 0, this is actually wait for LRB0 + 1/16 LRA0"),
                     29, SBarrier(comment="for GRA"),
                     47, SWaitCnt(dscnt=0, vlcnt=14, vscnt=-1, comment="wait for previous GRB and LRA0"),
                     47, SBarrier(comment="for GRB"),
                     70, SWaitCnt(dscnt=-1, vlcnt=14-3, vscnt=-1, comment="wait for previous GRA"),
                     70, SBarrier(comment="for GRB"),
                     ]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[18, 19,20,21,22,23,24,25,26]],
                'GRIncB': [[9,10,11,12,13,14,15,16,17]],

                'LRB0': [[-1,1, 3,5, 7,9, 11,13, 15,17, 19,21],
                         [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]],
                'LRA0': [[23, 24, 25, 26, 27, 28, 29, 30,31, 32,33, 34,35, 36,37, 38],
                         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
                'GRA': [[29,29, 31,31, 33,33, 38,38, 40,40, 41,41],
                         [30, 30, 32, 32, 34, 34, 39, 39, 41, 41, 42, 42]],

                'GRB': [[57,57, 59,59, 61,61, 63,63, 65,65, 70,70, 75,75, 80,80],
                        [58, 58, 60, 60, 62, 62, 64, 64, 66, 66, 71, 71, 76, 76, 81, 81]],
                'LRB1': [[47,48, 50,52, 54,56, 58,60, 62,64, 66,68],
                         [48, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]],
                'LRA1': [[70,71, 72,73, 74,75, 76,77, 78,79, 80,81, 82,83, 84,85],
                         [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]],

                'LRSB': [[39]],
                'LRSA': [[46]],
                'LWSB': [[78]],
                'LWSA': [[95]],
                'LCC' : [[95, 95]],}
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1"),
                     15, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),
                     15, SBarrier(comment=""),
                     46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                     51, SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="wait for previous set of global reads"),
                     51, SBarrier(comment=""),
                     63, SWaitCnt(dscnt=-1, vlcnt=14-4, vscnt=-1, comment="wait for previous set of global reads"),
                     63, SBarrier(comment=""),
                    ]
        optSchedule = {
                    'SYNC'  : [syncTable[::2]],
                    'GRIncA': [[35,36,37,38,39,40,41,42,43]],
                    'GRIncB': [[0,1,2,3,4,5,6,7,8]],

                    'LRB0': [[-1, 0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4, 5]],
                    'LRA0': [[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35],
                             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36]],
                    'GRA': [[15,15, 17,17, 27,27, 29,29, 31,31, 33,33],
                            [16, 16, 18, 18, 28, 28, 30, 30, 32, 32, 34, 34]],

                    'GRB': [[51,51, 53,53, 55,55, 57,57, 67,67, 69,69, 71,71, 73,73],
                            [52,52, 54,54, 56,56, 58,58, 68,68, 70,70, 72,72, 74,74]],
                    'LRB1': [[51, 53, 55, 57, 59, 61],
                             [52, 54, 56, 58, 60, 62]],
                    'LRA1': [[63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93],
                             [64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]],

                    'LRSB': [[14]],
                    'LRSA': [[45]],
                    'LWSB': [[94]],
                    'LWSA': [[94]],
                    'LCC' : [[95, 95]],
                }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1
