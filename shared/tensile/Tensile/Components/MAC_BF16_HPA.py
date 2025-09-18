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

from ..Component import Component, MAC
from ..DataType import DataType

class FMA_BF16_HPA(MAC):
    asmCaps = {"v_fma_f32": True}
    kernel = {"ProblemType": {"DataType": DataType(DataType.bfloat16),
                              "HighPrecisionAccumulate": True}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel
        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}
        vars["endLine"] = writer.endLine
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile0Half"] = kernel["ThreadTile0"] // 2

        for iui in range(0, innerUnroll):
            vars["iui"] = iui
            for blockA in range(kernel["ThreadTileA"]//2-1, -1, -1):
                vars["blockA"] = blockA
                kStr += "v_and_b32 v[vgprValuA_X{m}_I{iui}+{blockA}*2+1], 0xffff0000, v[vgprValuA_X{m}_I{iui}+{blockA}]{endLine}".format_map(vars)
                kStr += "v_lshlrev_b32 v[vgprValuA_X{m}_I{iui}+{blockA}*2], 16, v[vgprValuA_X{m}_I{iui}+{blockA}]{endLine}".format_map(vars)
            for blockB in range(kernel["ThreadTileB"]//2-1, -1, -1):
                vars["blockB"] = blockB
                kStr += "v_and_b32 v[vgprValuB_X{m}_I{iui}+{blockB}*2+1], 0xffff0000, v[vgprValuB_X{m}_I{iui}+{blockB}]{endLine}".format_map(vars)
                kStr += "v_lshlrev_b32 v[vgprValuB_X{m}_I{iui}+{blockB}*2], 16, v[vgprValuB_X{m}_I{iui}+{blockB}]{endLine}".format_map(vars)

        for block1 in range(0, kernel["ThreadTile1"]//2):
            vars["block1"] = block1
            for block0 in range(0, kernel["ThreadTile0"]//2):
                vars["block0"] = block0
                if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                    # we treat HighPrecisionAccumulate as expanded packed math
                    for iui in range(0, innerUnroll):
                        vars["iui"] = iui

                        vars["blockA"] = block0 if writer.tPB["tile01Idx"] else block1
                        vars["blockB"] = block1 if writer.tPB["tile01Idx"] else block0

                        vars["aStr0"] = "v[vgprValuA_X{m}_I{iui}+{blockA}*2+0]".format_map(vars)
                        vars["aStr1"] = "v[vgprValuA_X{m}_I{iui}+{blockA}*2+1]".format_map(vars)
                        vars["bStr0"] = "v[vgprValuB_X{m}_I{iui}+{blockB}*2+0]".format_map(vars)
                        vars["bStr1"] = "v[vgprValuB_X{m}_I{iui}+{blockB}*2+1]".format_map(vars)

                        vars["cidx"] = block0*2 + block1*kernel["ThreadTile0"]*2 + 0
                        vars["cStr"] = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+0*2+0]".format_map(vars)
                        kStr += "v_fma_f32 {cStr}, {aStr0}, {bStr0}, {cStr} //ValuC[{cidx}]{endLine}".format_map(vars)

                        kStr += priority(writer, 1, "Raise priority while processing macs")

                        vars["aStr"] = vars["aStr1"] if writer.tPB["tile01Idx"] else vars["aStr0"]
                        vars["bStr"] = vars["bStr0"] if writer.tPB["tile01Idx"] else vars["bStr1"]
                        vars["cidx"] = block0*2 + block1*kernel["ThreadTile0"]*2 + 1
                        vars["cStr"] = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+0*2+1]".format_map(vars) # *2 b/c of fp32
                        kStr += "v_fma_f32 {cStr}, {aStr}, {bStr}, {cStr} //ValuC[{cidx}]{endLine}".format_map(vars)

                        vars["aStr"] = vars["aStr0"] if writer.tPB["tile01Idx"] else vars["aStr1"]
                        vars["bStr"] = vars["bStr1"] if writer.tPB["tile01Idx"] else vars["bStr0"]
                        vars["cidx"] = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                        vars["cStr"] = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+{ThreadTile0Half}*2+0]".format_map(vars)
                        kStr += "v_fma_f32 {cStr}, {aStr}, {bStr}, {cStr} //ValuC[{cidx}]{endLine}".format_map(vars)

                        vars["cidx"] = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                        vars["cStr"] = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+{ThreadTile0Half}*2+1]".format_map(vars)
                        kStr += "v_fma_f32 {cStr}, {aStr1}, {bStr1}, {cStr} //valuC[{cidx}]{endLine}".format_map(vars)
                        """
                        ignore this, not quite correct for mixed precision
                        D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                        D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                        C[0] = A[0]*B[0]+D[0]
                        C[1] = A[1]*B[1]+D[1]
                        """
                        #kStr += self.bomb(-13)

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr
