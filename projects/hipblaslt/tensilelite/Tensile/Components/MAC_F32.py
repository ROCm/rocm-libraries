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
from rocisa.instruction import VMacF32, SSetPrior, VDualFMACF32
from ..Common.DataType import DataType
from ..Component import MAC

class MAC_F32_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    @staticmethod
    def asmCaps(caps):
        return caps["v_mac_f32"] or caps["v_fma_f32"]

    kernel = {"ProblemType": {"MacDataTypeA": DataType(DataTypeEnum.Float),
                              "MacDataTypeB": DataType(DataTypeEnum.Float),}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("MAC_F32_Plain")
        module.addComment(self.commentHeader())

        # UseDualFMAC: emit RDNA3/3.5/4 VOPD v_dual_fmac_f32 pairs instead of single-issue
        # v_fmac_f32, ~doubling the FMA issue rate.  The parameter is validated/auto-disabled
        # in SolutionStructs to f32 source (non-MFMA) kernels on gfx11/gfx12.
        if kernel["UseDualFMAC"]:
            return self._callVopd(module, tPB, m, innerUnroll,
                                  kernel["ThreadTile0"], kernel["ThreadTile1"])

        vars = {}
        vars["m"] = m
        vars["kernel"] = kernel
        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        for idx1 in range(0, kernel["ThreadTile1"]):
            for idx0 in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["idx0"] = idx0
                    vars["idx1"] = idx1
                    vars["a"] = idx0 if tPB["tile01Idx"] else idx1
                    vars["b"] = idx1 if tPB["tile01Idx"] else idx0
                    vars["iui"] = iui

                    cStr = "ValuC+%d+%d"%(vars["idx0"], vars["idx1"]*vars["ThreadTile0"])
                    aStr = "ValuA_X{m}_I{iui}+{a}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{b}".format_map(vars)

                    module.add(VMacF32(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr)))
                    if (idx1 == 0) and (idx0 == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module

    def _callVopd(self, module, tPB, m, innerUnroll, TT0, TT1):
        # Direct 2x2 block-diagonal pairing (no matching search): partition the ThreadTile
        # into 2x2 blocks, each emitting two VOPD pairs --
        #   (i,j)+(i+1,j+1)  and  (i+1,j)+(i,j+1)
        # both of which satisfy the VOPD constraints (dst parity differs; src0/src1 on
        # different VGPR banks) for even x even ThreadTiles -> 100% coverage, deterministic.
        # UseDualFMAC is validated to even ThreadTile dims in SolutionStructs; any leftover
        # (defensive, odd dim) falls back to single-issue v_fmac_f32.
        tile01 = tPB["tile01Idx"]

        def cell(idx0, idx1, iui):
            a = idx0 if tile01 else idx1
            b = idx1 if tile01 else idx0
            return (vgpr("ValuC+%d" % (idx0 + idx1 * TT0)),
                    vgpr("ValuA_X%d_I%d+%d" % (m, iui, a)),
                    vgpr("ValuB_X%d_I%d+%d" % (m, iui, b)))

        for iui in range(0, innerUnroll):
            paired = [[False] * TT1 for _ in range(TT0)]
            for j in range(0, TT1 - 1, 2):
                for i in range(0, TT0 - 1, 2):
                    for (i0, j0), (i1, j1) in (((i, j), (i + 1, j + 1)),
                                               ((i + 1, j), (i, j + 1))):
                        cX, aX, bX = cell(i0, j0, iui)
                        cY, aY, bY = cell(i1, j1, iui)
                        module.add(VDualFMACF32(dstX=cX, src0X=aX, src1X=bX,
                                                dstY=cY, src0Y=aY, src1Y=bY,
                                                comment="VOPD dual-issue FMA"))
                        paired[i0][j0] = paired[i1][j1] = True
            for idx1 in range(TT1):
                for idx0 in range(TT0):
                    if not paired[idx0][idx1]:
                        c, a, b = cell(idx0, idx1, iui)
                        module.add(VMacF32(dst=c, src0=a, src1=b))

        return module
