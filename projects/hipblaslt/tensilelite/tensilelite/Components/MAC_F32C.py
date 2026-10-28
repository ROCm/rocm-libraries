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

from itertools import product

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
        # A complex MAC c += a*b is two v_dual_fmac_f32 pairs; each VOPD does two real FMAs, an X half and a Y half.
        # As in MAC_F32, the two halves work on different cells: any two FMAs of one complex cell share a dst
        # parity, src0 or src1, which VOPD forbids. B columns are taken in blocks of two, (b0, b0+1). The X half
        # works on column bX and the Y half on the other column bY of the block (2 vgprs apart, so the src1
        # banks differ, which gfx11 requires):
        #   pair 1:  cX.re  += a.re * bX.re      cY.im += a.im * bY.re
        #   pair 2:  cX.re  -= a.im * bX.im      cY.im += a.re * bY.im
        # Running (bX, bY) = (b0, b0+1) and then (b0+1, b0) updates both accumulators (re, im) of both columns.
        # VOPD cannot negate an operand, so each minus comes from a negated copy of B made with xor (vgprValuBNeg).
        kernel = writer.states.kernel
        tt0, tt1 = kernel["ThreadTile0"], kernel["ThreadTile1"]
        assert tt1 % 2 == 0, "UseDualFMAC complex MAC takes B columns in blocks of two: ThreadTile1 must be even"
        assert writer.states.b.valuNegTempOffset is not None, \
            "UseDualFMAC complex MAC needs vgprValuBNeg from KernelWriter"

        RE, IM = 0, 1
        # Per pair, the (A part, B part, takes a minus) of its X and Y half. Conjugating A or B flips the sign
        # of an imaginary part, which only changes which products take the minus.
        halves = {1: ((RE, RE, False),      (IM, RE, ccA)),
                  2: ((IM, IM, ccA == ccB), (RE, IM, ccB))}
        # Only pair 1 needs a fresh xor when A alone is conjugated, so then pair 2 goes first.
        pairOrder = (2, 1) if (ccA and not ccB) else (1, 2)
        # The B parts that take a minus somewhere, so they need a negated copy.
        negParts = {bPart for half in halves.values() for _, bPart, minus in half if minus}

        def aReg(a, iui, part):
            return vgpr("ValuA_X%d_I%d+%d" % (m, iui, a*2 + part))

        def cReg(a, col, part):
            return vgpr("ValuC+%d" % ((a + col*tt0)*2 + part))

        def bReg(col, iui, part, minus=False):
            if minus:  # 4 vgprs per iui: [re, im] of even columns then of odd ones, so a copy keeps its source's bank
                return vgpr("ValuBNeg+%d" % (4*iui + 2*(col % 2) + part))
            return vgpr("ValuB_X%d_I%d+%d" % (m, iui, col*2 + part))

        firstVopd = True
        for iui, b0 in product(range(innerUnroll), range(0, tt1, 2)):
            for col, part in product((b0, b0 + 1), (RE, IM)):  # negated copies of both columns of the block
                if part in negParts:
                    module.add(VXorB32(dst=bReg(col, iui, part, minus=True), src0=hex(0x80000000),
                                       src1=bReg(col, iui, part), comment="-b.real" if part == RE else "-b.imag"))

            # Both column roles, and all rows (a) of one pair before the other: pair 2 accumulates into what pair 1 just wrote.
            for (bX, bY), pair, a in product(((b0, b0 + 1), (b0 + 1, b0)), pairOrder, range(tt0)):
                (xA, xB, xMinus), (yA, yB, yMinus) = halves[pair]
                module.add(VDualFMACF32(dstX=cReg(a, bX, RE), src0X=aReg(a, iui, xA), src1X=bReg(bX, iui, xB, xMinus),
                                        dstY=cReg(a, bY, IM), src0Y=aReg(a, iui, yA), src1Y=bReg(bY, iui, yB, yMinus),
                                        comment="VOPD pair %d" % pair))
                if firstVopd:
                    firstVopd = False
                    module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module
