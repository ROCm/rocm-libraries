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
from rocisa.instruction import SSetPrior, VDot2F32BF16, VDot2CF32BF16, VAndB32, VFmaF32, VLShiftLeftB32

from rocisa.enum import DataTypeEnum
from ..Common.DataType import DataType
from ..Component import Component, MAC

class FMA_BF16_HPA(MAC):
    asmCaps = {"v_fma_f32": True}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.BFloat16),
                              "MacDataTypeB": DataType(DataTypeEnum.BFloat16),
                              "HighPrecisionAccumulate": True},
              "UseDotInstruction": False,
              }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("FMA_BF16_HPA")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        for iui in range(innerUnroll):
            for tensor in ("A", "B"):
                for block in range(kernel[f"ThreadTile{tensor}"] // 2 - 1, -1, -1):
                    base = f"Valu{tensor}_X{m}_I{iui}"
                    src = vgpr(f"{base}+{block}")
                    module.add(VAndB32(dst=vgpr(f"{base}+{block * 2 + 1}"),
                                       src0="0xffff0000", src1=src))
                    module.add(VLShiftLeftB32(dst=vgpr(f"{base}+{block * 2}"),
                                              shiftHex=hex(16), src=src))

        tt0 = kernel["ThreadTile0"]
        for block1 in range(kernel["ThreadTile1"] // 2):
            for block0 in range(tt0 // 2):
                for iui in range(innerUnroll):
                    first_dim_a = bool(tPB["tile01Idx"])
                    block_a = block0 if first_dim_a else block1
                    block_b = block1 if first_dim_a else block0
                    a = [vgpr(f"ValuA_X{m}_I{iui}+{block_a * 2 + j}") for j in range(2)]
                    b = [vgpr(f"ValuB_X{m}_I{iui}+{block_b * 2 + j}") for j in range(2)]
                    pairs = ((a[0], b[0]),
                             (a[1], b[0]) if first_dim_a else (a[0], b[1]),
                             (a[0], b[1]) if first_dim_a else (a[1], b[0]),
                             (a[1], b[1]))
                    base = block0 * 2 + block1 * tt0 * 2
                    for offset, (src0, src1) in zip((0, 1, tt0, tt0 + 1), pairs):
                        cidx = base + offset
                        c = vgpr(f"ValuC+{cidx}")
                        module.add(VFmaF32(dst=c, src0=src0, src1=src1, src2=c,
                                           comment=f"ValuC[{cidx}]"))
                        if offset == 0:
                            prio = priority(writer, 1, "Raise priority while processing macs")
                            if prio:
                                module.add(prio)

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)
        return module

#dot2
class FMA_BF16_HPA_DOT2(MAC):
    asmCaps = lambda caps: caps['v_dot2_f32_bf16'] or caps['v_dot2c_f32_bf16']
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.BFloat16),
                              "MacDataTypeB": DataType(DataTypeEnum.BFloat16),
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
