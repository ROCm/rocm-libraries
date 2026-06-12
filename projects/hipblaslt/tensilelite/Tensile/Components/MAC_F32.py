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

    @staticmethod
    def _vopdOk(x, y):
        # VOPD v_dual_fmac_f32 constraints enforced by llvm-mc (gfx11/12), base-independent
        # in offset terms: dst parity must differ, and src0 (ValuA) / src1 (ValuB) of the
        # two ops must be on different VGPR banks (reg % 4).
        return (((x[0] - y[0]) % 2) == 1 and
                ((x[1] - y[1]) % 4) != 0 and
                ((x[2] - y[2]) % 4) != 0)

    @classmethod
    def _vopdMatch(cls, ops):
        # Maximum (near-perfect) matching of MACs into VOPD-legal pairs: greedy then
        # length-3 augmenting paths to a fixpoint.  Reaches 100% pairing for all even
        # ThreadTiles.
        n = len(ops)
        adj = [[k for k in range(n) if k != i and cls._vopdOk(ops[i], ops[k])]
               for i in range(n)]
        mate = [-1] * n
        for i in range(n):
            if mate[i] == -1:
                for k in adj[i]:
                    if mate[k] == -1:
                        mate[i] = k; mate[k] = i; break
        improved = True
        while improved:
            improved = False
            for u in range(n):
                if mate[u] != -1:
                    continue
                done = False
                for a in adj[u]:
                    b = mate[a]
                    if b == -1:
                        mate[u] = a; mate[a] = u; improved = done = True; break
                    for w in adj[b]:
                        if w != u and mate[w] == -1:
                            mate[u] = a; mate[a] = u; mate[b] = w; mate[w] = b
                            improved = done = True; break
                    if done:
                        break
        return mate

    def _callVopd(self, module, tPB, m, innerUnroll, TT0, TT1):
        tile01 = tPB["tile01Idx"]
        items = []
        for iui in range(0, innerUnroll):
            # ops for this iui as (cOff, aOff, bOff); different iui accumulate into the
            # same ValuC, so they are kept in separate sequential blocks.
            ops = []
            for idx1 in range(0, TT1):
                for idx0 in range(0, TT0):
                    a = idx0 if tile01 else idx1
                    b = idx1 if tile01 else idx0
                    ops.append((idx0 + idx1 * TT0, a, b))

            mate = self._vopdMatch(ops)
            for i in range(len(ops)):
                if mate[i] != -1 and mate[i] < i:
                    continue  # already emitted as the X-op of its pair
                if mate[i] > i:
                    j = mate[i]
                    cX, aX, bX = ops[i]
                    cY, aY, bY = ops[j]
                    items.append(VDualFMACF32(
                        dstX=vgpr("ValuC+%d" % cX),
                        src0X=vgpr("ValuA_X%d_I%d+%d" % (m, iui, aX)),
                        src1X=vgpr("ValuB_X%d_I%d+%d" % (m, iui, bX)),
                        dstY=vgpr("ValuC+%d" % cY),
                        src0Y=vgpr("ValuA_X%d_I%d+%d" % (m, iui, aY)),
                        src1Y=vgpr("ValuB_X%d_I%d+%d" % (m, iui, bY)),
                        comment="VOPD dual-issue FMA"))
                else:
                    c, a, b = ops[i]
                    items.append(VMacF32(dst=vgpr("ValuC+%d" % c),
                                         src0=vgpr("ValuA_X%d_I%d+%d" % (m, iui, a)),
                                         src1=vgpr("ValuB_X%d_I%d+%d" % (m, iui, b))))

        # Raise priority on the first mac and reset after the block, matching the
        # single-issue path above.
        for n, it in enumerate(items):
            module.add(it)
            if n == 0:
                module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))
        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module
