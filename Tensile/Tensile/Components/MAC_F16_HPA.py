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

class FMA_F16_HPA_MAD_MIX_LDL(MAC):
    @staticmethod
    def asmCaps(caps):
        return (caps['v_mad_mix_f32'] or caps['v_fma_mix_f32']) \
            and not caps["v_dot2c_f32_f16"] \
            and not caps["v_dot2_f32_f16"]
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
              "LocalDotLayout": lambda ldl: ldl > 1,
              "InnerUnroll": 2
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}

        if writer.asmCaps["v_fma_mix_f32"]:
            vars["instruction"] = "v_fma_mix_f32"
        else:
            vars["instruction"] = "v_mad_mix_f32"

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                vars["blockA"] = blockA
                vars["blockB"] = blockB

                vars["aBase0"] = "vgprValuA_X{m}_I0".format_map(vars)
                vars["bBase0"] = "vgprValuB_X{m}_I0".format_map(vars)

                vars["aBase1"] = "vgprValuA_X{m}_I1".format_map(vars)
                vars["bBase1"] = "vgprValuB_X{m}_I1".format_map(vars)

                # we treat HighPrecisionAccumulate as expanded packed math
                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + 0*2 + 0".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32

                vars["aStr"] = "v[{aBase0}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase0}+{blockB}]".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

                kStr += priority(writer, 1, "Raise priority while processing macs")

                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + 0*2 + 1".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                vars["aStr"] = "v[{aBase1}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase0}+{blockB}]".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)
                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)


                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 0".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)
                vars["aStr"] = "v[{aBase0}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase1}+{blockB}]".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)
                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

                vars["cidx"] = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1

                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + {Half_ThreadTile0}*2+1".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)
                vars["aStr"] = "v[{aBase1}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase1}+{blockB}]".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,1,0] //valuC[{cidx}]{endLine}".format_map(vars)
                kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[{cidx}]{endLine}".format_map(vars)
                #kStr += writer.bomb(-13)

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr


class FMA_F16_HPA_MAD_MIX(MAC):
    asmCaps = lambda caps: caps['v_mad_mix_f32'] or caps['v_fma_mix_f32']
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
              "LocalDotLayout": 1
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}

        if writer.asmCaps["v_fma_mix_f32"]:
            vars["instruction"] = "v_fma_mix_f32"
        else:
            vars["instruction"] = "v_mad_mix_f32"

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

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
                    vars["cidx"] = eval(vars["cIdxExpr"])

                    vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    vars["aStr"] = "v[{aBase}+{blockA}]".format_map(vars)
                    vars["bStr"] = "v[{bBase}+{blockB}]".format_map(vars)
                    kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[{cidx}] iui={iui}{endLine}".format_map(vars)

                    kStr += priority(writer, 1, "Raise priority while processing macs")

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + 0*2 + 1".format_map(vars)
                    vars["cidx"] = eval(vars["cIdxExpr"])

                    vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    vars["opSel"] = "op_sel:[1,0,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[0,1,0]"
                    kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} {opSel} op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 0".format_map(vars)
                    vars["cidx"] = eval(vars["cIdxExpr"])

                    vars["cStr"] = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    vars["opSel"] = "op_sel:[0,1,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[1,0,0]"
                    kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} {opSel} op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

                    vars["cIdxExpr"] = "{block0}*2+{block1}*{ThreadTile0}*2+{Half_ThreadTile0}*2+1".format_map(vars)
                    vars["cidx"] = eval(vars["cIdxExpr"])

                    vars["cStr"] = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    kStr += "{instruction} {cStr}, {aStr}, {bStr}, {cStr} op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[{cidx}]{endLine}".format_map(vars)

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr

class FMA_F16_DOT2(MAC):
    asmCaps = lambda caps: caps["v_dot2c_f32_f16"] or caps["v_dot2_f32_f16"]
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
              "LocalDotLayout": lambda ldl: ldl > 1
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()

        accumulate = writer.asmCaps["v_dot2c_f32_f16"]

        priority = Component.Priority.find(writer)

        vars = {}

        if accumulate:
            vars["instruction"] = "_v_dot2acc_f32_f16"
        else:
            vars["instruction"] = "v_dot2_f32_f16"

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2
        vars["cSrc"] = ""

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                vars["blockA"] = blockA
                vars["blockB"] = blockB

                vars["aBase0"] = "vgprValuA_X{m}_I0".format_map(vars)
                vars["bBase0"] = "vgprValuB_X{m}_I0".format_map(vars)

                vars["aBase1"] = "vgprValuA_X{m}_I1".format_map(vars)
                vars["bBase1"] = "vgprValuB_X{m}_I1".format_map(vars)

                # we treat HighPrecisionAccumulate as expanded packed math
                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + 0*2 + 0".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                vars["aStr"] = "v[{aBase0}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase0}+{blockB}]".format_map(vars)

                if not accumulate:
                    vars["cSrc"] = ", {cStr}".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}{cSrc} //ValuC[{cidx}]{endLine}".format_map(vars)

                kStr += priority(writer, 1, "Raise priority while processing macs")

                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + 0*2 + 1".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)
                vars["aStr"] = "v[{aBase1}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase0}+{blockB}]".format_map(vars)

                if not accumulate:
                    vars["cSrc"] = ", {cStr}".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}{cSrc} //ValuC[{cidx}]{endLine}".format_map(vars)

                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 0".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)
                vars["aStr"] = "v[{aBase0}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase1}+{blockB}]".format_map(vars)

                if not accumulate:
                    vars["cSrc"] = ", {cStr}".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}{cSrc} //ValuC[{cidx}]{endLine}".format_map(vars)

                vars["cIdxExpr"] = "{blockA}*2 + {blockB}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 1".format_map(vars)
                vars["cidx"] = eval(vars["cIdxExpr"])

                vars["cStr"] = "v[vgprValuC + {cIdxExpr}]".format_map(vars)
                vars["aStr"] = "v[{aBase1}+{blockA}]".format_map(vars)
                vars["bStr"] = "v[{bBase1}+{blockB}]".format_map(vars)

                if not accumulate:
                    vars["cSrc"] = ", {cStr}".format_map(vars)

                kStr += "{instruction} {cStr}, {aStr}, {bStr}{cSrc} //ValuC[{cidx}]{endLine}".format_map(vars)
                #kStr += writer.bomb(-13)

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr
