################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
        kernel = writer.kernel

        kStr = self.commentHeader()

        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

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

                            vars["cStr"] = "v[vgprValuC+{blockA}+{blockB}*{ThreadTile0}+0]".format_map(vars)
                            vars["aStr"] = "v[vgprValuA_X{m}_I{iui}+{blockA}]".format_map(vars)
                            vars["bStr"] = "v[vgprValuB_X{m}_I{iui}+{blockB}]".format_map(vars)
                            kStr += "v_mac_f16 {cStr}, {aStr}, {bStr}{endLine}".format_map(vars)
                            kStr += priority(writer, 1, "Raise priority while processing macs")

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr


class FMA_F16_NonPacked(MAC):
    asmCaps = {"v_fma_f16": True,
               "v_pk_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

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

                    vars["cIdxExpr0"] = "{blockA} + {blockB}*{ThreadTile0} + 0".format_map(vars)
                    vars["cIdxVal0"] = eval(vars["cIdxExpr0"])
                    vars["cStr0"] = "v[vgprValuC + {cIdxExpr0}]".format_map(vars)

                    # /2 b/c of 2 f16's per 32-bit vgpr
                    vars["cIdxExpr1"] = "{blockA} + {blockB}*{ThreadTile0} + {Half_ThreadTile0}".format_map(vars)
                    vars["cIdxVal1"] = eval(vars["cIdxExpr1"])
                    vars["cStr1"] = "v[vgprValuC + {cIdxExpr1}]".format_map(vars)

                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui} + {blockA}]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui} + {blockB}]".format_map(vars)

                    kStr += "v_fma_f16 {cStr0}, {aStr}, {bStr}, {cStr0} op_sel:[0,0,0,0] // {cIdxExpr0}{endLine}".format_map(vars)
                    kStr += priority(writer, 1, "Raise priority while processing macs")
                    kStr += "v_fma_f16 {cStr1}, {aStr}, {bStr}, {cStr1} op_sel:[0,1,0,0] // {cIdxExpr1}{endLine}".format_map(vars)
                    kStr += "v_fma_f16 {cStr0}, {aStr}, {bStr}, {cStr0} op_sel:[1,0,1,1] // {cIdxExpr0}{endLine}".format_map(vars)
                    kStr += "v_fma_f16 {cStr1}, {aStr}, {bStr}, {cStr1} op_sel:[1,1,1,1] // {cIdxExpr1}{endLine}".format_map(vars)

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr

class FMA_F16_Packed(MAC):
    asmCaps = {"v_pk_fma_f16": True}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

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

                    vars["cIdxExpr"] = "{blockA} + {blockB}*{ThreadTile0} + 0".format_map(vars)
                    vars["cIdxVal"] = eval(vars["cIdxExpr"])

                    # /2 b/c of 2 f16's per 32-bit vgpr
                    vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)

                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui} + {blockA}]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui} + {blockB}]".format_map(vars)

                    kStr += "v_pk_fma_f16 {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,0,1] // {cIdxVal}{endLine}".format_map(vars)

                    kStr += priority(writer, 1, "Raise priority while processing macs")

                    vars["cIdxExpr"] = "{blockA} + {blockB}*{ThreadTile0} + {Half_ThreadTile0}".format_map(vars)
                    vars["cIdxVal"] = eval(vars["cIdxExpr"])

                    vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)

                    kStr += "v_pk_fma_f16 {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,1,0] op_sel_hi:[1,1,1] // {cIdxVal}{endLine}".format_map(vars)

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr
