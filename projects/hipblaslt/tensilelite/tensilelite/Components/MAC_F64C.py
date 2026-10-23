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
from rocisa.instruction import VFmaF64, SSetPrior
from ..Common.DataType import DataType
from ..Component import MAC

class FMA_F64C_Plain(MAC):
    asmCaps = {"v_fma_f64": True}
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.ComplexDouble),
                              "MacDataTypeB": DataType(DataTypeEnum.ComplexDouble)}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("FMA_F64C_Plain")
        module.addComment(self.commentHeader())

        ccA = kernel["ProblemType"]["ComplexConjugateA"]
        ccB = kernel["ProblemType"]["ComplexConjugateB"]
        tt0 = kernel["ThreadTile0"]

        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, tt0):
                for iui in range(0, innerUnroll):
                    # each complex-double element is 4 vgprs: real=[+0:+1], imag=[+2:+3]
                    cReal = vgpr("ValuC+%d" % ((a + b*tt0)*4 + 0), 2)
                    cImag = vgpr("ValuC+%d" % ((a + b*tt0)*4 + 2), 2)
                    aReal = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*4 + 0), 2)
                    aImag = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*4 + 2), 2)
                    bReal = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*4 + 0), 2)
                    bImag = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*4 + 2), 2)

                    # c.real += a.real * b.real
                    module.add(VFmaF64(dst=cReal, src0=aReal, src1=bReal, src2=cReal))
                    # c.real -= a.imag * b.imag  (sign flips under a single conjugate)
                    bImagRR = bImag.getMinus() if (ccA == ccB) else bImag
                    module.add(VFmaF64(dst=cReal, src0=aImag, src1=bImagRR, src2=cReal))
                    # c.imag += a.real * b.imag  (negate b.imag when conjugating B)
                    bImagIR = bImag.getMinus() if ccB else bImag
                    module.add(VFmaF64(dst=cImag, src0=aReal, src1=bImagIR, src2=cImag))
                    # c.imag += a.imag * b.real  (negate a.imag when conjugating A)
                    aImagIR = aImag.getMinus() if ccA else aImag
                    module.add(VFmaF64(dst=cImag, src0=aImagIR, src1=bReal, src2=cImag))

                    if (b == 0) and (a == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module
