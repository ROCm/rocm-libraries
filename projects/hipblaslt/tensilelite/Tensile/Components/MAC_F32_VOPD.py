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
from rocisa.instruction import VMacF32, VDualFmacF32, SSetPrior
from ..Common.DataType import DataType
from ..Component import MAC


class MAC_F32_VOPD(MAC):
    """
    VOPD dual-issue MAC for FP32 non-MI kernels (RDNA3+, wave32).
    Uses 2x2 block diagonal pairing for 100% VOPD coverage.

    RDNA3 VOPD requires ALL THREE operand pairs to differ in bank:
      - dst: one even, one odd VGPR
      - src0: different VGPR banks
      - src1: different VGPR banks

    2x2 block diagonal partitions TT into 2x2 blocks, each yielding 2 VOPD pairs:
      Pair A: (i,j)+(i+1,j+1) — dst diff=1+tt0 (odd), src0 diff=+1, src1 diff=+1
      Pair B: (i+1,j)+(i,j+1) — dst diff=tt0-1 (odd), src0 diff=-1, src1 diff=+1

    Coverage: 100% for any even×even TT. Zero leftovers.
    """
    @staticmethod
    def asmCaps(caps):
        return caps.get("v_dual_fmac_f32", False)

    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Float),
                              "MacDataTypeB": DataType(DataTypeEnum.Float)},
              "EnableVOPD": 1}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("MAC_F32_VOPD")
        module.addComment(self.commentHeader())

        tt0 = kernel["ThreadTile0"]
        tt1 = kernel["ThreadTile1"]

        prioritySet = False
        for iui in range(0, innerUnroll):
            # Build grid of FMACs
            grid = {}
            for idx1 in range(tt1):
                for idx0 in range(tt0):
                    a = idx0 if tPB["tile01Idx"] else idx1
                    b = idx1 if tPB["tile01Idx"] else idx0
                    cStr = "ValuC+%d+%d" % (idx0, idx1 * tt0)
                    aStr = "ValuA_X{m}_I{iui}+{a}".format(m=m, iui=iui, a=a)
                    bStr = "ValuB_X{m}_I{iui}+{b}".format(m=m, iui=iui, b=b)
                    grid[(idx0, idx1)] = (cStr, aStr, bStr)

            # 2x2 block diagonal: 100% coverage for even×even TT
            paired = set()
            vopd_pairs = []
            for block_j in range(0, tt1, 2):
                for block_i in range(0, tt0, 2):
                    vopd_pairs.append(((block_i, block_j), (block_i + 1, block_j + 1)))
                    paired.add((block_i, block_j))
                    paired.add((block_i + 1, block_j + 1))
                    vopd_pairs.append(((block_i + 1, block_j), (block_i, block_j + 1)))
                    paired.add((block_i + 1, block_j))
                    paired.add((block_i, block_j + 1))

            # Emit VOPD pairs
            for (i0a, i1a), (i0b, i1b) in vopd_pairs:
                c0, a0, b0 = grid[(i0a, i1a)]
                c1, a1, b1 = grid[(i0b, i1b)]
                module.add(VDualFmacF32(
                    dstX=vgpr(c0), src0X=vgpr(a0), src1X=vgpr(b0),
                    dstY=vgpr(c1), src0Y=vgpr(a1), src1Y=vgpr(b1)))
                if not prioritySet:
                    module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))
                    prioritySet = True

            # Emit unpaired leftovers as single v_fmac_f32
            for idx1 in range(tt1):
                for idx0 in range(tt0):
                    if (idx0, idx1) not in paired:
                        c, a, b = grid[(idx0, idx1)]
                        module.add(VMacF32(dst=vgpr(c), src0=vgpr(a), src1=vgpr(b)))
                        if not prioritySet:
                            module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))
                            prioritySet = True

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module
