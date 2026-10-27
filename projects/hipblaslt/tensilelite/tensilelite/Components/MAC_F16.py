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
from rocisa.container import vgpr, VOP3PModifiers
from rocisa.enum import DataTypeEnum
from rocisa.instruction import VFmaF16, VFmaPKF16
from ..Common.DataType import DataType
from ..Component import Component, MAC

class MAC_F16_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    asmCaps = {"v_mac_f16": True,
               "v_pk_fma_f16": False,
               "v_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Half),
                              "MacDataTypeB": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
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
                            prio = priority(writer, 1, "Raise priority while processing macs")
                            if prio:
                                module.add(prio)

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)
        return module


class FMA_F16_NonPacked(MAC):
    asmCaps = {"v_fma_f16": True,
               "v_pk_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Half),
                              "MacDataTypeB": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
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

                    aStr = "ValuA_X{m}_I{iui}+{blockA}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{blockB}".format_map(vars)

                    cIdx0 = blockA + blockB * kernel["ThreadTile0"]
                    cIdx1 = cIdx0 + kernel["ThreadTile0"] // 2
                    cStr0 = "ValuC+%d" % cIdx0
                    cStr1 = "ValuC+%d" % cIdx1

                    module.add(VFmaF16(dst=vgpr(cStr0), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr0), \
                                        vop3=VOP3PModifiers(op_sel=[0,0,0,0]), comment="ValuC[%u] iui=%u" % (cIdx0, vars["iui"])))
                    prio = priority(writer, 1, "Raise priority while processing macs")
                    if prio:
                        module.add(prio)
                    module.add(VFmaF16(dst=vgpr(cStr1), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr1), \
                                        vop3=VOP3PModifiers(op_sel=[0,1,0,0]), comment="ValuC[%u] iui=%u" % (cIdx1, vars["iui"])))
                    module.add(VFmaF16(dst=vgpr(cStr0), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr0), \
                                        vop3=VOP3PModifiers(op_sel=[1,0,1,1]), comment="ValuC[%u] iui=%u" % (cIdx0, vars["iui"])))
                    module.add(VFmaF16(dst=vgpr(cStr1), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr1), \
                                        vop3=VOP3PModifiers(op_sel=[1,1,1,1]), comment="ValuC[%u] iui=%u" % (cIdx1, vars["iui"])))

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)
        return module

class FMA_F16_Packed(MAC):
    asmCaps = {"v_pk_fma_f16": True}
    #archCaps = {}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Half),
                              "MacDataTypeB": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
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

                    cIdx0 = blockA + blockB * kernel["ThreadTile0"]
                    cIdx1 = cIdx0 + kernel["ThreadTile0"] // 2

                    aStr = "ValuA_X{m}_I{iui}+{blockA}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{blockB}".format_map(vars)

                    # /2 b/c of 2 f16's per 32-bit vgpr
                    module.add(VFmaPKF16(dst=vgpr("ValuC+%d" % cIdx0), src0=vgpr(aStr), src1=vgpr(bStr), \
                                            src2=vgpr("ValuC+%d" % cIdx0), \
                                            vop3=VOP3PModifiers(op_sel=[0,0,0], op_sel_hi=[1,0,1]), \
                                            comment="ValuC[%u] iui=%u" % (cIdx0, vars["iui"])))
                    prio = priority(writer, 1, "Raise priority while processing macs")
                    if prio:
                        module.add(prio)

                    module.add(VFmaPKF16(dst=vgpr("ValuC+%d" % cIdx1), src0=vgpr(aStr), src1=vgpr(bStr), \
                                            src2=vgpr("ValuC+%d" % cIdx1), \
                                            vop3=VOP3PModifiers(op_sel=[0,1,0], op_sel_hi=[1,1,1]), \
                                            comment="ValuC[%u] iui=%u" % (cIdx1, vars["iui"])))

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)
        return module
