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
from rocisa.instruction import VAShiftRightI32, VLShiftLeftB32, VMadI32I24, VDot4CI32I8, VDot4I32I8

from ..Common.DataType import DataType
from ..Component import Component, MAC

class FMA_I8_HPA_DOT4(MAC):
    @staticmethod
    def asmCaps(caps):
        return caps['v_dot4c_i32_i8'] or caps['v_dot4_i32_i8']

    kernel = {
        "ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Int8),
                        "MacDataTypeB": DataType(DataTypeEnum.Int8),
                        "HighPrecisionAccumulate": True},
        "UseDotInstruction": True,
    }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel   = writer.states.kernel
        priority = Component.Priority.find(writer)

        module = Module("FMA_I8_HPA_DOT4")
        module.addComment(self.commentHeader())

        # v_dot4c_i32_i8 accumulates into its destination (dst = dst + dot4(src0,src1)),
        # v_dot4_i32_i8 takes the accumulator as an explicit third source.
        accumulate  = writer.states.asmCaps["v_dot4c_i32_i8"]
        ThreadTile0 = kernel["ThreadTile0"]
        ThreadTile1 = kernel["ThreadTile1"]

        for b in range(ThreadTile1):
            for a in range(ThreadTile0):
                for iui in range(innerUnroll):
                    # A carries the output dimension it owns and B the one it
                    # owns, which the contraction decides: tPA['tileIdx'] says
                    # which of the two output dimensions A belongs to.
                    blockA = a if tPA["tileIdx"] == 0 else b
                    blockB = b if tPB["tileIdx"] != 0 else a
                    cStr = f'ValuC+{a + b*ThreadTile0}'
                    aStr = f'ValuA_X{m}_I{iui}+{blockA}'
                    bStr = f'ValuB_X{m}_I{iui}+{blockB}'
                    if accumulate:
                        module.add(VDot4CI32I8(dst=vgpr(cStr),
                                               src0=vgpr(aStr),
                                               src1=vgpr(bStr),
                                               comment="Accumulate"))
                    else:
                        module.add(VDot4I32I8(dst=vgpr(cStr),
                                               src0=vgpr(aStr),
                                               src1=vgpr(bStr),
                                               src2=vgpr(cStr),
                                               comment="Accumulate"))
                    prio = priority(writer, 1, "Raise priority while processing macs")
                    if prio:
                        module.add(prio)

        prio = priority(writer, 0, "Reset priority after processing macs")
        if prio:
            module.add(prio)

        return module

class FMA_I8_HPA(MAC):
    @staticmethod
    def asmCaps(caps):
        return True

    kernel = {
        "ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Int8),
                        "MacDataTypeB": DataType(DataTypeEnum.Int8),
                        "HighPrecisionAccumulate": True},
        "UseDotInstruction": False,
    }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel      = writer.states.kernel
        priority    = Component.Priority.find(writer)
        spacePerReg = writer.states.bpr
        elemPerReg  = min(kernel['VectorWidth'], spacePerReg)

        module = Module("FMA_I8_HPA")
        module.addComment(self.commentHeader())

        for a in range(kernel["ThreadTile0"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = a // elemPerReg
                idx  = a %  elemPerReg
                sStr = f'ValuA_X{m}_I{iui}+{src}'
                tStr = f'ValuA_X{m}_I{iui}+{a}'
                module.add(VLShiftLeftB32(dst=vgpr(tStr),
                                          shiftHex=hex((spacePerReg-idx-1)*8),
                                          src=vgpr(sStr),
                                          comment="Sign extend A"))
                prio = priority(writer, 1, "Raise priority while processing macs")
                if prio:
                    module.add(prio)
                module.add(VAShiftRightI32(dst=vgpr(tStr),
                                           shiftHex=hex((spacePerReg-1)*8),
                                           src=vgpr(tStr),
                                           comment="Sign extend A"))

        for b in range(kernel["ThreadTile1"]-1, -1, -1):
            for iui in range(0, innerUnroll):
                src  = b // elemPerReg
                idx  = b %  elemPerReg
                sStr = f'ValuB_X{m}_I{iui}+{src}'
                tStr = f'ValuB_X{m}_I{iui}+{b}'
                module.add(VLShiftLeftB32(dst=vgpr(tStr),
                                          shiftHex=hex((spacePerReg-idx-1)*8),
                                          src=vgpr(sStr),
                                          comment="Sign extend B"))
                prio = priority(writer, 1, "Raise priority while processing macs")
                if prio:
                    module.add(prio)
                module.add(VAShiftRightI32(dst=vgpr(tStr),
                                           shiftHex=hex((spacePerReg-1)*8),
                                           src=vgpr(tStr),
                                           comment="Sign extend B"))

        ThreadTile0 = kernel["ThreadTile0"]
        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    cStr = f'ValuC+{a + b*ThreadTile0}'
                    aStr = f'ValuA_X{m}_I{iui}+{a}'
                    bStr = f'ValuB_X{m}_I{iui}+{b}'
                    module.add(VMadI32I24(dst=vgpr(cStr),
                                          src0=vgpr(aStr),
                                          src1=vgpr(bStr),
                                          src2=vgpr(cStr),
                                          comment="Accumulate"))
                    prio = priority(writer, 1, "Raise priority while processing macs")
                    if prio:
                        module.add(prio)

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)

        return module
