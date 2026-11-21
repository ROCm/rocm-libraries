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
from rocisa.instruction import VDot4I32I8
from ..Common.DataType import DataType
from ..Component import Component, MAC

class MAC_I8X4_Plain(MAC):
    asmCaps = {"VOP3v_dot4_i32_i8": True}
    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.Int8x4)}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("MAC_I8X4_Plain")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    cidx = a + b * kernel["ThreadTile0"]
                    cStr = vgpr(f"ValuC+{cidx}")
                    aStr = vgpr(f"ValuA_X{m}_I{iui}+{a}")
                    bStr = vgpr(f"ValuB_X{m}_I{iui}+{b}")
                    module.add(VDot4I32I8(cStr, aStr, bStr, cStr,
                                          f"valuC[{cidx}]"))
                    prio = priority(writer, 1, "Raise priority while processing macs")
                    if prio:
                        module.add(prio)

        prio = priority(writer, 0, "Reset priority after macs")
        if prio:
            module.add(prio)
        return module
