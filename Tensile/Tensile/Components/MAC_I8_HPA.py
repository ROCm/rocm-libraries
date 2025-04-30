################################################################################
#
# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from ..Component import Component, MAC
from ..DataType import DataType


class FMA_I8_HPA_DOT4(MAC):
    asmCaps = lambda caps: caps["v_dot4c_i32_i8"] or caps["v_dot4_i32_i8"]
    kernel = {"ProblemType": {"DataType": DataType(DataType.int8),
                              "HighPrecisionAccumulate": True},
              "LocalDotLayout": 4
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()

        priority = Component.Priority.find(writer)

        endLine     = writer.endLine
        accumulate  = writer.asmCaps["v_dot4c_i32_i8"]
        ThreadTile0 = kernel["ThreadTile0"]
        ThreadTile1 = kernel["ThreadTile1"]
        instruction = "v_dot4c_i32_i8" if accumulate else "v_dot4_i32_i8"
        inTSize     = 4

        kStr += '// C index : {blockA}*{inTSize} + {blockB}*{inTSize}*{ThreadTile0} + {indexB}*{ThreadTile0} + {indexA}' + endLine

        for blockB in range(0, ThreadTile1//inTSize):
            for blockA in range(0, ThreadTile0//inTSize):
                for indexB in range(inTSize):
                    for indexA in range(inTSize):
                        cIdxStr = f'{blockA}*{inTSize} + {blockB}*{inTSize}*{ThreadTile0} + {indexB}*{ThreadTile0} + {indexA}'
                        cIdx = eval(cIdxStr)

                        cStr = f'v[vgprValuC + {cIdxStr}]'
                        aStr = f'v[vgprValuA_X{m}_I{indexA}+{blockA}]'
                        bStr = f'v[vgprValuB_X{m}_I{indexB}+{blockB}]'
                        if accumulate:
                          kStr += f'{instruction} {cStr}, {aStr}, {bStr} // ValuC[{cIdx}]{endLine}'
                        else:
                          kStr += f'{instruction} {cStr}, {aStr}, {bStr}, {cStr} // ValuC[{cIdx}]{endLine}'

                        kStr += priority(writer, 1, "Raise priority while processing macs")

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr


class FMA_I8_HPA(MAC):
    @staticmethod
    def asmCaps(caps):
        return True

    kernel = {
        "ProblemType": {"DataType": DataType(DataType.int8), "HighPrecisionAccumulate": True},
        "LocalDotLayout": 1
    }

    def __call__(self, writer, m, innerUnroll):
        kernel      = writer.kernel
        priority    = Component.Priority.find(writer)
        spacePerReg = writer.bpr // writer.bpeAB
        elemPerReg  = min(kernel['VectorWidth'], spacePerReg)
        endLine     = writer.endLine

        kStr = self.commentHeader()

        for a in range(kernel["ThreadTile0"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = a // elemPerReg
                idx  = a %  elemPerReg
                sStr = f'vgprValuA_X{m}_I{iui}+{src}'
                tStr = f'vgprValuA_X{m}_I{iui}+{a}'
                kStr += f'v_lshlrev_b32 v[{tStr}], {(spacePerReg-idx-1)*8}, v[{sStr}]{endLine}'
                kStr += priority(writer, 1, "Raise priority while processing macs")
                kStr += f'v_ashrrev_i32 v[{tStr}], {(spacePerReg    -1)*8}, v[{tStr}]{endLine}'

        for b in range(kernel["ThreadTile1"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = b // elemPerReg
                idx  = b %  elemPerReg
                sStr = f'vgprValuB_X{m}_I{iui}+{src}'
                tStr = f'vgprValuB_X{m}_I{iui}+{b}'
                kStr += f'v_lshlrev_b32 v[{tStr}], {(spacePerReg-idx-1)*8}, v[{sStr}]{endLine}'
                kStr += priority(writer, 1, "Raise priority while processing macs")
                kStr += f'v_ashrrev_i32 v[{tStr}], {(spacePerReg    -1)*8}, v[{tStr}]{endLine}'

        ThreadTile0 = kernel["ThreadTile0"]
        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    cStr = f'v[vgprValuC + {a} + {b}*{ThreadTile0}]'
                    aStr = f'v[vgprValuA_X{m}_I{iui} + {a}]'
                    bStr = f'v[vgprValuB_X{m}_I{iui} + {b}]'
                    kStr += f'v_mad_i32_i24 {cStr}, {aStr}, {bStr}, {cStr}{endLine}'
                    kStr += priority(writer, 1, "Raise priority while processing macs")

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr
