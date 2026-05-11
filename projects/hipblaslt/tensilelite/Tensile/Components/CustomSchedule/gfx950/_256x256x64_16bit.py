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
    isTT,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 256, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS == 1:
        optSchedule = {
            'SYNC'   : [[19,20, 50,51, 67,68, 104, 105, 127]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,10,11,12,13,14,15,16,17]],
            'LRA0'   : [[0,2,4,6,8,10,12,14],
                        [1,3,5,7,9,11,13,15]],
            'LRB0'   : [[24,27,30,33,36,38,40,42],
                        [22,25,28,31,34,37,39,41]],
            'GRA'    : [[21,22, 23,25, 26,28, 29,31, 32,34, 35,52, 53,55, 56,58],
                        [21,23, 24,26, 27,29, 30,32, 33,35, 36,53, 54,56, 57,59]],
            'GRB'    : [[59,61, 62,64, 65,85, 86,87, 88,89, 94,96, 98,100, 102,124],
                        [60,62, 63,65, 66,84, 85,86, 87,88, 93,95, 97,99, 103,123]],
            'LRA1'   : [[69,71,73,75,77,79,81,83],
                        [70,72,74,76,78,80,82,90]],
            'LRB1'   : [[106,108,110,112,114,116,118,120],
                        [107,109,111,113,115,117,119,121]],
            'LRSA'   : [[16]],
            'LRSB'   : [[83]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'   : [[126, 126]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(2 + 8 + 8), vscnt=-1, comment="Wait for previous GRA to completely"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous GRB to completely"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 and 3/8 LRB1 to complete")]
        nglshift = nllshift = 16
    elif isNT(kernel) and not useLDSTr and TLDS == 0:
        kernel["UsePLRPack"] = True

        optSchedule = {
            'SYNC'   : [[12,13, 36,44, 56,59, 66,68, 73,92]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[28,29,30,31,32,33,34,35,36]],
            'LRA0'   : [[0,0,2,2,4,4,6,6],
                        [1,1,3,3,5,5,7,7]],
            'LRB0'   : [[8,8,10,10,15,15,18,18],
                        [9,9,11,11,14,14,17,17]],
            'GRA'    : [[14,14, 17,17, 20,20, 23,23, 26,26,   45,45, 48,48, 51,51],
                        [15,15, 18,18, 21,21, 24,24, 27,27,   46,46, 49,49, 52,52]],
            'GRB'    : [[54,54, 57,57, 87,87,90,90,93,93,96,96,99,99, 123,123],
                        [55,55, 58,58, 88,88,91,91,94,94,97,97,100,100, 124,124]],
            'LRA1'   : [[60,60,62,62,64,64,66,66],
                        [61,61,63,63,65,65,67,67]],
            'LRB1'   : [[69,69,71,71,73,73,75,75],
                        [70,70,72,72,74,74,76,76]],
            'LRSA'   : [[59]],
            'LRSB'   : [[59]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'    : [[126, 126]],
            'PackA0' : [[16,16, 19,19, 21,21, 22,22, 24,24, 25,25, 27,27, 28,28, 29,29, 30,30, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36],
                        [16,16, 19,19, 20,20, 22,22, 23,23, 25,25, 26,26, 28,28, 29,29, 30,30, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36]],
            'PackB0' : [[37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43, 46,46, 47,47, 49,49, 50,50, 52,52, 53,53, 55,55, 56,56, 58,58],
                        [37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43, 45,45, 47,47, 48,48, 50,50, 51,51, 53,53, 54,54, 56,56, 57,57]],
            'PackA1' : [[74,74, 76,76, 77,77, 78,78, 79,79, 80,80, 81,81, 82,82, 83,83, 84,84, 85,85, 86,86, 88,88, 89,89, 91,91, 92,92],
                        [75,75, 77,77, 78,78, 79,79, 80,80, 81,81, 82,82, 83,83, 84,84, 85,85, 86,86, 87,87, 89,89, 90,90, 92,92, 93,93]],
            'PackB1' : [[94,94, 95,95, 97,97, 98,98, 100,100, 101,101, 102,102, 103,103, 104,104, 105,105, 106,106, 107,107, 108,108, 109,109, 110,110, 111,111],
                        [95,95, 96,96, 98,98, 99,99, 101,101, 102,102, 103,103, 104,104, 105,105, 106,106, 107,107, 108,108, 109,109, 110,110, 111,111, 112,112]],
        }
        syncCode = [SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=17, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 to complete")]
        nglshift = nllshift = 16
    elif (isNN(kernel) or isTT(kernel)) and not useLDSTr and TLDS == 1:
        kernel["UsePLRPack"] = True

        optSchedule = {
            'SYNC'   : [[8, 12,13, 36,44, 56,59, 66,68, 74, 127]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[28,29,30,31,32,33,34,35,36]],
            'LRA0'   : [[0,0,2,2,4,4,6,6],
                        [1,1,3,3,5,5,7,7]],
            'LRB0'   : [[9,11, 15,18,21,24,27,30],
                        [10,12, 14,17,20,23,26,29]],
            'GRA'    : [[14,14, 17,17, 20,20, 23,23, 26,26,   45,45, 48,48, 51,51],
                        [15,15, 18,18, 21,21, 24,24, 27,27,   46,46, 49,49, 52,52]],
            'GRB'    : [[54,54, 57,57, 87,87,90,90,93,93,96,96,99,99, 123,123],
                        [55,55, 58,58, 88,88,91,91,94,94,97,97,100,100, 124,124]],
            'LRA1'   : [[60,60,62,62,64,64,66,66],
                        [61,61,63,63,65,65,67,67]],
            'LRB1'   : [[68,70,72,74,76,78,80,82],
                        [69,71,73,75,77,79,81,83]],
            'LRSA'   : [[59]],
            'LRSB'   : [[59]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'    : [[126, 126]],
            'PackA0' : [[8,8, 16,16, 19,19, 22,22, 25,25, 28,28, 29,29, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36, 37,37, 38,38, 39,39]],
            'PackA1' : [[75,75, 77,77, 79,79, 81,81, 83,83, 84,84, 85,85, 86,86, 88,88, 89,89, 91,91, 92,92, 94,94, 95,95, 97,97, 98,98],
                        [74,74, 76,76, 78,78, 80,80, 82,82, 84,84, 85,85, 86,86, 87,87, 89,89, 90,90, 92,92, 93,93, 95,95, 96,96, 98,98]],
        }
        syncCode = [SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 first half to complete"),
                    SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=17, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 to complete")]
        if isTT(kernel):
            kernel["SwapGlobalReadOrder"] = True

            optSchedule['GRIncA'], optSchedule['GRIncB'] = optSchedule['GRIncB'], optSchedule['GRIncA']
            optSchedule['LRA0'], optSchedule['LRB0'] = optSchedule['LRB0'], optSchedule['LRA0']
            optSchedule['LRA1'], optSchedule['LRB1'] = optSchedule['LRB1'], optSchedule['LRA1']
            optSchedule['PackB0'] = optSchedule['PackA0']
            optSchedule['PackB1'] = optSchedule['PackA1']
            del optSchedule['PackA0'], optSchedule['PackA1']
        nglshift = nllshift = 16
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 128
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
