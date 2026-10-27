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
        # Two v_dual_fmac_f32 pairs per complex MAC; X works on column b, Y on its neighbour column b^1 ('):
        #   pair 1: c.re += a.re*b.re      c'.im += a.im*b'.re
        #   pair 2: c.re -= a.im*b.im      c'.im += a.re*b'.im
        # b and b' are 2 vgprs apart, so src1 banks differ (gfx11 needs it). VOPD can't negate: -b is xor'ed
        # into vgprValuBNeg, [re, im] per column parity, so each copy keeps its source's bank.
        kernel = writer.states.kernel
        tt0 = kernel["ThreadTile0"]
        assert writer.states.b.valuNegTempOffset is not None, \
            "UseDualFMAC complex MAC needs vgprValuBNeg from KernelWriter"

        # which products use the negated b
        negRealOfAImagBReal = ccA
        negImagOfAImagBImag = ccA == ccB
        negImagOfARealBImag = ccB
        # only pair 1 needs a fresh xor when A alone is conjugated, so pair 2 goes first
        pairOrder = (2, 1) if (ccA and not ccB) else (1, 2)

        def bReg(col, iui, k, negated=False):  # k: 0 real, 1 imag
            if negated:
                return vgpr("ValuBNeg+%d" % (4*iui + 2*(col % 2) + k))
            return vgpr("ValuB_X%d_I%d+%d" % (m, iui, col*2 + k))

        for b in range(0, kernel["ThreadTile1"]):
            nb = b ^ 1
            for iui in range(0, innerUnroll):
                if b % 2 == 0:  # negate both neighbour columns once
                    for col in (b, nb):
                        if negRealOfAImagBReal:
                            module.add(VXorB32(dst=bReg(col, iui, 0, True), src0=hex(0x80000000), src1=bReg(col, iui, 0), comment="-b.real"))
                        if negImagOfAImagBImag or negImagOfARealBImag:
                            module.add(VXorB32(dst=bReg(col, iui, 1, True), src0=hex(0x80000000), src1=bReg(col, iui, 1), comment="-b.imag"))

                # all rows of one pair before the other: pair 2 accumulates into what pair 1 just wrote
                for pair in pairOrder:
                    for a in range(0, tt0):
                        cReal = vgpr("ValuC+%d" % ((a + b*tt0)*2))
                        cImag = vgpr("ValuC+%d" % ((a + nb*tt0)*2 + 1))
                        aReal = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2))
                        aImag = vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2 + 1))
                        if pair == 1:
                            x = (cReal, aReal, bReg(b, iui, 0))
                            y = (cImag, aImag, bReg(nb, iui, 0, negRealOfAImagBReal))
                        else:
                            x = (cReal, aImag, bReg(b, iui, 1, negImagOfAImagBImag))
                            y = (cImag, aReal, bReg(nb, iui, 1, negImagOfARealBImag))
                        module.add(VDualFMACF32(*x, *y, comment="VOPD pair %d" % pair))
                        if (b, iui, pair, a) == (0, 0, pairOrder[0], 0):
                            module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module
