################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
from rocisa.instruction import SSetPrior, VDot2F32F16, VDot2CF32F16, \
    VDualDot2AccF32F16

from ..Common.DataType import DataType
from ..Component import Component, MAC

# dot2
class FMA_F16_HPA_DOT2(MAC):
    asmCaps = lambda caps: caps['v_dot2_f32_f16'] or caps['v_dot2c_f32_f16']
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Half),
                              "MacDataTypeB": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": True},
              "UseDotInstruction": True,
             }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_HPA_DOT2")
        module.addComment(self.commentHeader())

        vars = {}

        # VOPD pairs two dot2-accumulates into one issue slot. Only the
        # accumulate form has a dual encoding, so this path ignores the
        # non-accumulating v_dot2_f32_f16 even where it exists.
        if kernel["UseDualFMAC"]:
            return self._callVopd(module, tPA, tPB, m, innerUnroll,
                                  kernel["ThreadTile0"], kernel["ThreadTile1"])

        if writer.states.asmCaps["v_dot2_f32_f16"]:
            instruction = VDot2F32F16
        else:
            instruction = VDot2CF32F16

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
                    if instruction == VDot2F32F16:
                        module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
                                            comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))
                    else:
                        module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), \
                                            comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))

                    if (block1 == 0) and (block0 == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module

    def _callVopd(self, module, tPA, tPB, m, innerUnroll, TT0, TT1):
        # Same 2x2 block-diagonal pairing as the f32 VOPD path: partition the
        # ThreadTile into 2x2 blocks and emit the two diagonals,
        #   (i,j)+(i+1,j+1)  and  (i+1,j)+(i,j+1)
        # which satisfy the VOPD constraints (the two destinations differ in
        # parity, and the sources fall on different VGPR banks) for an even x
        # even ThreadTile. SolutionStructs restricts UseDualFMAC to that, but
        # any leftover odd dimension falls back to single-issue below.
        def cell(block0, block1, iui):
            blockA = block0 if tPA["tileIdx"] == 0 else block1
            blockB = block1 if tPB["tileIdx"] != 0 else block0
            return (vgpr("ValuC+%d" % (block0 + block1 * TT0)),
                    vgpr("ValuA_X%d_I%d+%d" % (m, iui, blockA)),
                    vgpr("ValuB_X%d_I%d+%d" % (m, iui, blockB)))

        for iui in range(0, innerUnroll):
            paired = [[False] * TT1 for _ in range(TT0)]
            for j in range(0, TT1 - 1, 2):
                for i in range(0, TT0 - 1, 2):
                    for (i0, j0), (i1, j1) in (((i, j), (i + 1, j + 1)),
                                               ((i + 1, j), (i, j + 1))):
                        cX, aX, bX = cell(i0, j0, iui)
                        cY, aY, bY = cell(i1, j1, iui)
                        module.add(VDualDot2AccF32F16(dstX=cX, src0X=aX, src1X=bX,
                                                      dstY=cY, src0Y=aY, src1Y=bY,
                                                      comment="VOPD dual-issue dot2"))
                        paired[i0][j0] = paired[i1][j1] = True
            for block1 in range(TT1):
                for block0 in range(TT0):
                    if not paired[block0][block1]:
                        c, a, b = cell(block0, block1, iui)
                        module.add(VDot2CF32F16(dst=c, src0=a, src1=b))

        return module


class FMA_F16_HPA_MAD_MIX(MAC):
    asmCaps = lambda caps: caps['v_mad_mix_f32'] or caps['v_fma_mix_f32']
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Half),
                              "MacDataTypeB": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": True},
              "UseDotInstruction": False,
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_HPA_MAD_MIX")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        vars = {}

        if writer.states.asmCaps["v_fma_mix_f32"]:
            instruction = "v_fma_mix_f32"
        else:
            instruction = "v_mad_mix_f32"

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for block1 in range(0, kernel["ThreadTile1"]//2):
            for block0 in range(0, kernel["ThreadTile0"]//2):
                for iui in range(0, innerUnroll):
                    vars["block0"] = block0
                    vars["block1"] = block1
                    vars["blockA"] = block0 if writer.tPA["tileIdx"] == 0 else block1
                    vars["blockB"] = block1 if writer.tPB["tileIdx"] != 0 else block0
                    vars["iui"] = iui

                    vars["aBase"] = "vgprValuA_X{m}_I{iui}".format_map(vars)
                    vars["bBase"] = "vgprValuB_X{m}_I{iui}".format_map(vars)

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + 0*2 + 0".format_map(vars)
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    aStr = "v[{aBase}+{blockA}]".format_map(vars)
                    bStr = "v[{bBase}+{blockB}]".format_map(vars)
                    module.addInst(instruction, cStr, aStr, bStr, cStr, "op_sel:[0,0,0]", "op_sel_hi:[1,1,0]", "ValuC[%u] iui=%u" % (cidx, vars["iui"]))

                    module.add(priority(writer, 1, "Raise priority while processing macs"))

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + 0*2 + 1".format_map(vars)
                    cidx  = eval(vars["cIdxExpr"])
                    cStr  = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    opSel = "op_sel:[1,0,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[0,1,0]"
                    module.addInst(instruction, cStr, aStr, bStr, cStr, opSel, "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 0".format_map(vars)
                    cidx  = eval(vars["cIdxExpr"])
                    cStr  = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    opSel = "op_sel:[0,1,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[1,0,0]"
                    module.addInst(instruction, cStr, aStr, bStr, cStr, opSel, "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

                    vars["cIdxExpr"] = "{block0}*2+{block1}*{ThreadTile0}*2+{Half_ThreadTile0}*2+1".format_map(vars)
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    module.addInst(instruction, cStr, aStr, bStr, cStr, "op_sel:[1,1,0]", "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

        module.add(priority(writer, 0, "Reset priority after macs"))

        return module
