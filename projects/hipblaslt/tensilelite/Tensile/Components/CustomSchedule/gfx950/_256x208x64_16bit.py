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
    isTN,
)


@RegisterSchedule(
    tile_config=TileConfig(256, 208, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x208x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'SYNC': [[-1, 3, 23, 23, 35, 35, 81, 81]],
            'LRA0': [[0, 1, 2, 4]],
            'LRB0': [[5, 6, 8, 9, 11, 14, 16, 19, 21, 24, 26, 29, 31]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'GRA': [[23, 23, 23, 23, 25, 25, 28, 28, 29, 29, 31, 31, 33, 33, 35, 35]],
            'GRB': [[36, 36, 37, 37, 39, 39, 41, 41, 42, 42, 44, 44, 47, 47, 48, 48, 50, 50, 52, 52, 54, 54, 55, 55, 57, 57, 59, 59, 60, 60, 62, 62, 64, 64, 66, 66, 67, 67, 69, 69, 71, 71, 73, 73, 74, 74, 76, 76, 78, 78, 79, 79]],
            'LRSA': [[50]],
            'LRSB': [[50]],
            'LWSA': [[79]],
            'LWSB': [[80]],
            'LRA1': [[82, 84, 87, 90]],
            'LRB1': [[83, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98]],
            'LCC': [[98, 98]],
        }

        syncCode = [
            SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LRB1-4"),
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LRB1-remaining"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0, before GRA starts"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0, before GRB starts"),
            SWaitCnt(dscnt=-1, vlcnt=34, vscnt=-1, comment="wait for all GRA/GRB before next-tile LDS reads"),
            SBarrier(comment="final barrier before LRA1/LRB1"),
        ]

        nglshift = nllshift = 34

    elif isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        nglshift = nllshift = 0

        optSchedule = {
            # last index of producer <SYNC> first index of consumer
            # SYNC[0] = -1 to align all waves at the start of the loop
            # A fence at 23, annd B fence at 36, final vmem fence at 81
            'SYNC': [[-1, 3, 7, 11, 23, 36, 36, 81, 81]],

            # Avoid interleaving of LRA0 and LRB0
            # LRA0: tightly packed at the beginning
            'LRA0': [[0, 1, 2, 3, 4, 5, 6, 7]],
            # LRB0 scheduled after the A fence, overlapping with GRA/GRB
            'LRB0': [[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 35]],

            # Address increments for GR
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],

            # first stream (A side in NN+swapped)
            'GRB': [[23, 23, 24, 24,
                    26, 26, 28, 28,
                    29, 29, 31, 31,
                    33, 33, 35, 35]],

            # second stream (B side in NN+swapped)
            'GRA': [[36, 36, 38, 38, 40, 40, 41, 41,
                    43, 43, 45, 45, 47, 47, 48, 48,
                    50, 50, 52, 52, 54, 54, 55, 55,
                    57, 57, 59, 59, 60, 60, 62, 62,
                    64, 64, 66, 66, 67, 67, 69, 69,
                    71, 71, 73, 73, 74, 74, 76, 76,
                    78, 78, 79, 79]],

            # from epilogue in the default schedule
            # these are not updated in the updated schedule
            'LRSA': [[50]],
            'LRSB': [[50]],
            'LWSA': [[80]],
            'LWSB': [[80]],
            'LRA1': [[82, 84, 84, 85, 85, 86, 86, 87]], # 8
            'LRB1': [[83, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]], # 13
            'LCC':  [[100, 100]],
        }

        syncCode = [
            SWaitCnt(dscnt=12, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=11, vlcnt=-1, vscnt=-1, comment="ensure all previous LRA1/LRB1 done before early MFMA use"),
            SWaitCnt(dscnt=10, vlcnt=-1, vscnt=-1, comment="ensure all previous LRA1/LRB1 done before early MFMA use"),


            # A fence: all LRA0 are done before DTL writes from the first GR stream startign at 23 (swapped)
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before first DTL writes"),
            # barrier after LRA0, before first global DTL phase at 23
            SBarrier(comment="barrier after LRA0 , before GR at 23"),

            # B fence : all LRB0 are done before second DTL stream
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before second DTL writes"),
            # barrier after LRB0 before second global stream at 36
            SBarrier(comment="barrier after LRB0, before GR at 36"),

            # final vmem fence: ensure all GRA/GRB are done before next tile LDS reads LRA1/LRB1
            SWaitCnt(dscnt=-1, vlcnt=34, vscnt=-1, comment="wait for all GRA/GRB before next-tile LDS reads"),
            # final barrier : all waves; make next-tile LDS visible to LRA1/LRB1
            SBarrier(comment="final barrier before LRA1/LRB1 (at 83)"),
        ]

        nglshift = nllshift = 34
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 104
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
