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

class MAC_F16_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    asmCaps = {"v_mac_f16": True,
               "v_pk_fma_f16": False,
               "v_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("MAC_F16_Plain")
        module.addComment(self.commentHeader())

        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                for b in range(blockB*2, (blockB+1)*2):
                    for a in range(blockA*2, (blockA+1)*2):
                        for iui in range(0, innerUnroll):
                            vars["blockA"] = blockA
                            vars["blockB"] = blockB
                            vars["iui"] = iui

                            cStr = "v[vgprValuC+{blockA}+{blockB}*{ThreadTile0}+0]".format_map(vars)
                            aStr = "v[vgprValuA_X{m}_I{iui}+{blockA}]".format_map(vars)
                            bStr = "v[vgprValuB_X{m}_I{iui}+{blockB}]".format_map(vars)
                            module.addInst("v_mac_f16", cStr, aStr, bStr, "")
                            module.add(priority(writer, 1, "Raise priority while processing macs"))

        module.add(priority(writer, 0, "Reset priority after macs"))
        return module


class FMA_F16_NonPacked(MAC):
    asmCaps = {"v_fma_f16": True,
               "v_pk_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_NonPacked")
        module.addComment(self.commentHeader())

        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                for iui in range(0, innerUnroll):
                    vars["blockA"] = blockA
                    vars["blockB"] = blockB
                    vars["iui"] = iui

                    cIdxExpr0 = "{blockA} + {blockB}*{ThreadTile0} + 0".format_map(vars)
                    cStr0 = "v[vgprValuC + {}]".format(eval(cIdxExpr0))

                    # /2 b/c of 2 f16's per 32-bit vgpr
                    cIdxExpr1 = "{blockA} + {blockB}*{ThreadTile0} + {Half_ThreadTile0}".format_map(vars)
                    cStr1 = "v[vgprValuC + {}]".format(eval(cIdxExpr1))

                    aStr = "v[vgprValuA_X{m}_I{iui} + {blockA}]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui} + {blockB}]".format_map(vars)

                    module.addInst("v_fma_f16", cStr0, aStr, bStr, cStr0, "op_sel:[0,0,0,0]", cIdxExpr0)
                    module.add(priority(writer, 1, "Raise priority while processing macs"))
                    module.addInst("v_fma_f16", cStr1, aStr, bStr, cStr1, "op_sel:[0,1,0,0]", cIdxExpr1)
                    module.addInst("v_fma_f16", cStr0, aStr, bStr, cStr0, "op_sel:[1,0,1,1]", cIdxExpr0)
                    module.addInst("v_fma_f16", cStr1, aStr, bStr, cStr1, "op_sel:[1,1,1,1]", cIdxExpr1)

        module.add(priority(writer, 0, "Reset priority after macs"))
        return module

class FMA_F16_Packed(MAC):
    asmCaps = {"v_pk_fma_f16": True}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_Packed")
        module.addComment(self.commentHeader())

        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                for iui in range(0, innerUnroll):
                    vars["blockA"] = blockA
                    vars["blockB"] = blockB
                    vars["iui"] = iui

                    cIdxExpr = "{blockA} + {blockB}*{ThreadTile0} + 0".format_map(vars)

                    # /2 b/c of 2 f16's per 32-bit vgpr
                    cStr = "v[vgprValuC + {}]".format(eval(cIdxExpr))

                    aStr = "v[vgprValuA_X{m}_I{iui} + {blockA}]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui} + {blockB}]".format_map(vars)

                    module.addInst("v_pk_fma_f16", cStr, aStr, bStr, cStr, "op_sel:[0,0,0]", "op_sel_hi:[1,0,1]", cIdxExpr)
                    module.add(priority(writer, 1, "Raise priority while processing macs"))

                    cIdxExpr = "{blockA} + {blockB}*{ThreadTile0} + {Half_ThreadTile0}".format_map(vars)
                    cIdxVal  = eval(vars["cIdxExpr"])

                    cStr = "v[vgprValuC + {cIdxExpr}]".format_map(vars)

                    module.addInst("v_pk_fma_f16", cStr, aStr, bStr, cStr, "op_sel:[0,1,0]", "op_sel_hi:[1,1,1]", cIdxExpr)

        module.add(priority(writer, 0, "Reset priority after macs"))
        return module
