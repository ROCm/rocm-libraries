################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import DataType, Module
from ..Component import Component, MAC

class FMA_I8_HPA(MAC):
    @staticmethod
    def asmCaps(caps):
        return True

    kernel = {
        "ProblemType": {"DataType": DataType(DataType.int8), "HighPrecisionAccumulate": True},
    }

    def __call__(self, writer, m, innerUnroll):
        kernel      = writer.states.kernel
        priority    = Component.Priority.find(writer)
        spacePerReg = writer.states.bpr // writer.states.bpeAB
        elemPerReg  = min(kernel['VectorWidth'], spacePerReg)

        module = Module("FMA_I8_HPA")
        module.addComment(self.commentHeader())

        for a in range(kernel["ThreadTile0"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = a // elemPerReg
                idx  = a %  elemPerReg
                sStr = f'v[vgprValuA_X{m}_I{iui}+{src}]'
                tStr = f'v[vgprValuA_X{m}_I{iui}+{a}]'
                module.addInst("v_lshlrev_b32", tStr, {(spacePerReg-idx-1)*8}, sStr, "")
                module.add(priority(writer, 1, "Raise priority while processing macs"))
                module.addInst("v_ashrrev_i32", tStr, {(spacePerReg    -1)*8}, tStr, "")

        for b in range(kernel["ThreadTile1"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = b // elemPerReg
                idx  = b %  elemPerReg
                sStr = f'v[vgprValuB_X{m}_I{iui}+{src}]'
                tStr = f'v[vgprValuB_X{m}_I{iui}+{b}]'
                module.addInst("v_lshlrev_b32", tStr, {(spacePerReg-idx-1)*8}, sStr, "")
                module.add(priority(writer, 1, "Raise priority while processing macs"))
                module.addInst("v_ashrrev_i32", tStr, {(spacePerReg    -1)*8}, tStr, "")

        ThreadTile0 = kernel["ThreadTile0"]
        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    cStr = f'v[vgprValuC + {a} + {b}*{ThreadTile0}]'
                    aStr = f'v[vgprValuA_X{m}_I{iui} + {a}]'
                    bStr = f'v[vgprValuB_X{m}_I{iui} + {b}]'
                    module.addInst("v_mad_i32_i24", cStr, aStr, bStr, cStr, "")
                    module.add(priority(writer, 1, "Raise priority while processing macs"))

        module.add(priority(writer, 0, "Reset priority after macs"))

        return module
