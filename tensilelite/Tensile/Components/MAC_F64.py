################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import Module
from rocisa.container import vgpr
from rocisa.enum import DataTypeEnum
from rocisa.instruction import VFmaF64, SSetPrior
from ..Common.DataType import DataType
from ..Component import MAC

class FMA_F64_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    asmCaps = {"v_fma_f64": True}
    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.Double)}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("MAC_F64_Plain")
        module.addComment(self.commentHeader())

        vars = {}
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]

        for b in range(0, kernel["ThreadTile1"]):
            vars["b"] = b
            for a in range(0, kernel["ThreadTile0"]):
                vars["a"] = a
                for iui in range(0, innerUnroll):
                    vars["iui"] = iui

                    cStr = "ValuC+%d" % ((vars["a"]+vars["b"]*vars["ThreadTile0"])*2)
                    aStr = "ValuA_X%d_I%d+%d" % (vars["m"], vars["iui"], vars["a"]*2)
                    bStr = "ValuB_X%d_I%d+%d" % (vars["m"], vars["iui"], vars["b"]*2)

                    module.add(VFmaF64(dst=vgpr(cStr, 2), src0=vgpr(aStr, 2),
                                       src1=vgpr(bStr, 2), src2=vgpr(cStr, 2)))
                    if (b == 0) and (a == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module
