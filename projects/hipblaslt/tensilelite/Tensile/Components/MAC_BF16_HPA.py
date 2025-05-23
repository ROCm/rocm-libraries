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
from rocisa.instruction import SSetPrior, VDot2F32BF16, VDot2CF32BF16

from rocisa.enum import DataTypeEnum
from ..Common.DataType import DataType
from ..Component import Component, MAC

class FMA_BF16_HPA(MAC):
    asmCaps = {"v_fma_f32": True}
    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.BFloat16),
                              "HighPrecisionAccumulate": True},
              "UseDotInstruction": False,
              }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("FMA_BF16_HPA")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        vars = {}
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile0Half"] = kernel["ThreadTile0"] // 2

        for iui in range(0, innerUnroll):
            vars["iui"] = iui
            for blockA in range(kernel["ThreadTileA"]//2-1, -1, -1):
                vars["blockA"] = blockA
                dst = "v[vgprValuA_X{m}_I{iui}+{blockA}*2+1]".format_map(vars)
                src = "v[vgprValuA_X{m}_I{iui}+{blockA}]".format_map(vars)
                module.addInst("v_and_b32", dst, "0xffff0000", src, "")
                dst = "v[vgprValuA_X{m}_I{iui}+{blockA}*2]".format_map(vars)
                module.addInst("v_lshlrev_b32", dst, 16, src, "")
            for blockB in range(kernel["ThreadTileB"]//2-1, -1, -1):
                vars["blockB"] = blockB
                dst = "v[vgprValuB_X{m}_I{iui}+{blockB}*2+1]".format_map(vars)
                src = "v[vgprValuB_X{m}_I{iui}+{blockB}]".format_map(vars)
                module.addInst("v_and_b32", dst, "0xffff0000", src, "")
                dst = "v[vgprValuB_X{m}_I{iui}+{blockB}*2]".format_map(vars)
                module.addInst("v_and_b32", dst, 16, src, "")

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

                        aStr0 = "v[vgprValuA_X{m}_I{iui}+{blockA}*2+0]".format_map(vars)
                        aStr1 = "v[vgprValuA_X{m}_I{iui}+{blockA}*2+1]".format_map(vars)
                        bStr0 = "v[vgprValuB_X{m}_I{iui}+{blockB}*2+0]".format_map(vars)
                        bStr1 = "v[vgprValuB_X{m}_I{iui}+{blockB}*2+1]".format_map(vars)

                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 0
                        cStr = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+0*2+0]".format_map(vars)
                        module.addInst("v_fma_f32", cStr, aStr0, bStr0, cStr, "ValuC[%u]" % cidx)

                        module.add(priority(writer, 1, "Raise priority while processing macs"))

                        aStr = aStr1 if writer.tPB["tile01Idx"] else aStr0
                        bStr = bStr0 if writer.tPB["tile01Idx"] else bStr1
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 1
                        cStr = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+0*2+1]".format_map(vars) # *2 b/c of fp32
                        module.addInst("v_fma_f32", cStr, aStr, bStr, cStr, "ValuC[%u]" % cidx)

                        aStr = aStr0 if writer.tPB["tile01Idx"] else aStr1
                        bStr = bStr1 if writer.tPB["tile01Idx"] else bStr0
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                        cStr = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+{ThreadTile0Half}*2+0]".format_map(vars)
                        module.addInst("v_fma_f32", cStr, aStr, bStr, cStr, "ValuC[%u]" % cidx)

                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                        cStr = "v[vgprValuC+{block0}*2+{block1}*{ThreadTile0}*2+{ThreadTile0Half}*2+1]".format_map(vars)
                        module.addInst("v_fma_f32", cStr, aStr1, bStr1, cStr, "ValuC[%u]" % cidx)
                        """
                        ignore this, not quite correct for mixed precision
                        D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                        D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                        C[0] = A[0]*B[0]+D[0]
                        C[1] = A[1]*B[1]+D[1]
                        """
                        #module.add(self.getBomb(-13))

        module.add(priority(writer, 0, "Reset priority after macs"))
        return module

#dot2
class FMA_BF16_HPA_DOT2(MAC):
    asmCaps = lambda caps: caps['v_dot2_f32_bf16'] or caps['v_dot2c_f32_bf16']
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.BFloat16),
                              "HighPrecisionAccumulate": True},
              "UseDotInstruction": True,
             }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_BF16_HPA_DOT2")
        module.addComment(self.commentHeader())

        vars = {}

        if writer.states.asmCaps["v_dot2_f32_bf16"]:
            instruction = VDot2F32BF16
        else:
            instruction = VDot2CF32BF16

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        for block1 in range(0, kernel["ThreadTile1"]):
            for block0 in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["block0"] = block0
                    vars["block1"] = block1
                    vars["blockA"] = block0 if tPA["tileIdx"] == 0 else block1
                    vars["blockB"] = block1 if tPB["tileIdx"] != 0 else block0
                    vars["iui"] = iui

                    vars["cIdxExpr"] = "%d+%d" % (vars["block0"], vars["block1"]*vars["ThreadTile0"])
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "ValuC+{cIdxExpr}".format_map(vars)
                    aStr = "ValuA_X{m}_I{iui}+{blockA}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{blockB}".format_map(vars)
                    if instruction == VDot2F32BF16:
                        module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
                                            comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))
                    else:
                        module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), \
                                            comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))

                    if (block1 == 0) and (block0 == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module
