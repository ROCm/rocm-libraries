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
from rocisa.instruction import VMacF32, SSetPrior
from ..Common.DataType import DataType
from ..Component import MAC

class MAC_F32_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    @staticmethod
    def asmCaps(caps):
        return caps["v_mac_f32"] or caps["v_fma_f32"]

    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.Float)}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("MAC_F32_Plain")
        module.addComment(self.commentHeader())

        vars = {}
        vars["m"] = m
        vars["kernel"] = kernel
        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        for idx1 in range(0, kernel["ThreadTile1"]):
            for idx0 in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["idx0"] = idx0
                    vars["idx1"] = idx1
                    vars["a"] = idx0 if tPB["tile01Idx"] else idx1
                    vars["b"] = idx1 if tPB["tile01Idx"] else idx0
                    vars["iui"] = iui

                    cStr = "ValuC+%d+%d"%(vars["idx0"], vars["idx1"]*vars["ThreadTile0"])
                    aStr = "ValuA_X{m}_I{iui}+{a}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{b}".format_map(vars)

                    module.add(VMacF32(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr)))
                    if (idx1 == 0) and (idx0 == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module
