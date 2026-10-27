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
from rocisa.instruction import VMacF32, VDualFMACF32, VXorB32, SSetPrior
from ..Common.DataType import DataType
from ..Component import MAC

class MAC_F32C_Plain(MAC):
    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.ComplexFloat),
                              "MacDataTypeB": DataType(DataTypeEnum.ComplexFloat)}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("MAC_F32C_Plain")
        module.addComment(self.commentHeader())

        ccA = kernel["ProblemType"]["ComplexConjugateA"]
        ccB = kernel["ProblemType"]["ComplexConjugateB"]
        tt0 = kernel["ThreadTile0"]

        if kernel["UseDualFMAC"]:
            return self._callVopd(module, writer, m, innerUnroll, ccA, ccB)

        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, tt0):
                for iui in range(0, innerUnroll):
                    # each complex-single element is 2 vgprs: real=+0, imag=+1
                    cReal = vgpr("ValuC+%d" % ((a + b*tt0)*2 + 0))
                    cImag = vgpr("ValuC+%d" % ((a + b*tt0)*2 + 1))
                    aReal = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2 + 0))
                    aImag = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2 + 1))
                    bReal = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*2 + 0))
                    bImag = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*2 + 1))

                    # c.real += a.real * b.real
                    module.add(VMacF32(dst=cReal, src0=aReal, src1=bReal))
                    # c.real -= a.imag * b.imag  (sign flips under a single conjugate)
                    module.add(VMacF32(dst=cReal, src0=aImag.getMinus() if (ccA == ccB) else aImag, src1=bImag))
                    # c.imag += a.real * b.imag  (negate b.imag when conjugating B)
                    module.add(VMacF32(dst=cImag, src0=aReal, src1=bImag.getMinus() if ccB else bImag))
                    # c.imag += a.imag * b.real  (negate a.imag when conjugating A)
                    module.add(VMacF32(dst=cImag, src0=aImag.getMinus() if ccA else aImag, src1=bReal))

                    if (b == 0) and (a == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module

    def _callVopd(self, module, writer, m, innerUnroll, ccA, ccB):
        # A complex MAC is two v_dual_fmac_f32 pairs:
        #   pair 1: c.re += a.re*b.re      c.im += a.im*b.re
        #   pair 2: c.re -= a.im*b.im      c.im += a.re*b.im
        # VOPD has no negate, so the minus goes on b: xor into a temp placed after ValuB (KernelWriter),
        # -b.im at an even offset and -b.re at an odd one, so paired src1 are in different banks.
        # Pair 2 reads what pair 1 wrote, so a row of one pair is emitted before the other.
        kernel = writer.states.kernel
        tt0 = kernel["ThreadTile0"]
        tempOffset = writer.states.b.valuNegTempOffset
        assert tempOffset is not None, "UseDualFMAC complex MAC needs the negated-b temps from KernelWriter"

        # which products use the negated b
        negRealOfAImagBReal = ccA
        negImagOfAImagBImag = ccA == ccB
        negImagOfARealBImag = ccB
        # only pair 1 needs a fresh xor when A alone is conjugated, so pair 2 goes first
        pairOrder = (2, 1) if (ccA and not ccB) else (1, 2)

        for b in range(0, kernel["ThreadTile1"]):
            for iui in range(0, innerUnroll):
                bReal = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*2))
                bImag = vgpr("ValuB_X%d_I%d+%d" % (m, iui, b*2 + 1))
                negImag = vgpr("ValuB_X0_I0+%d" % (tempOffset + 2*iui))
                negReal = vgpr("ValuB_X0_I0+%d" % (tempOffset + 2*iui + 1))
                if negRealOfAImagBReal:
                    module.add(VXorB32(dst=negReal, src0=hex(0x80000000), src1=bReal, comment="-b.real"))
                if negImagOfAImagBImag or negImagOfARealBImag:
                    module.add(VXorB32(dst=negImag, src0=hex(0x80000000), src1=bImag, comment="-b.imag"))

                for pair in pairOrder:
                    for a in range(0, tt0):
                        cReal = vgpr("ValuC+%d" % ((a + b*tt0)*2))
                        cImag = vgpr("ValuC+%d" % ((a + b*tt0)*2 + 1))
                        aReal = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2))
                        aImag = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2 + 1))
                        if pair == 1:
                            x, y = (cReal, aReal, bReal), (cImag, aImag, negReal if negRealOfAImagBReal else bReal)
                        else:
                            x = (cReal, aImag, negImag if negImagOfAImagBImag else bImag)
                            y = (cImag, aReal, negImag if negImagOfARealBImag else bImag)
                        module.add(VDualFMACF32(dstX=x[0], src0X=x[1], src1X=x[2],
                                                dstY=y[0], src0Y=y[1], src1Y=y[2],
                                                comment="VOPD pair %d" % pair))
                        if (b, iui, pair, a) == (0, 0, pairOrder[0], 0):
                            module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module
