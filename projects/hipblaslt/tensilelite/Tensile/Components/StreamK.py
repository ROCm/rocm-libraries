################################################################################
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.enum import CacheScope
from rocisa.code import Module, Label
from rocisa.container import vgpr, sgpr, mgpr, SMEMModifiers, MUBUFModifiers, replaceHolder, EXEC,\
    VOP3PModifiers, ContinuousRegister, DSModifiers
from rocisa.instruction import GlobalInv, GlobalWb, SAddCU32, SAddU32, SAndB32, SBarrier, \
    SBranch, SCBranchSCC0, SCBranchSCC1, SCMovB32, SCSelectB32, SCmpEQU32, SCmpEQU64, \
    SCmpGtU32, SCmpLeU32, SCmpLtU32, SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, VLShiftLeftB32, SLoadB32, \
    SMaxI32, SMinU32, SMovB32, SMovB64, SMulI32, SNop, SOrB32, SSleep, SStoreB32, SSubU32, \
    SWaitCnt, SWaitXCnt, VAddF32, VAddF64, VAddPKF16, VAddU32, VSubU32, VLShiftRightB32, VMovB32, \
    VReadfirstlaneB32, VCvtBF16toFP32, BufferLoadB32, BufferStoreB32, SAtomicInc, DSLoadB32, DSStoreB32, \
    SLongBranch, SLongBranchPositive
from rocisa.functions import scalarStaticDivideAndRemainder, sMagicDiv2, \
    vectorStaticMultiply, BranchIfNotZero, scalarUInt24DivideAndRemainder, scalarUInt32DivideAndRemainder

from .Subtile.SubtileLREmit import localReadResetOffsetsSubtile

from ..Common import print2, ceilDivide, log2
from ..Component import Component
from ..AsmStoreState import StoreState, VectorDataTypes
from ..AsmAddressCalculation import AddrCalculation
import abc

from copy import deepcopy

class XCCMapping(Component):
    """
    XCC mapping code.
    """

class XCCMappingOff(XCCMapping):
    kernel = {"StreamKXCCMapping": 0}

    def __call__(self, writer, kernel):
        module = Module("XCCMapping Off")
        return module

class XCCMappingOn(XCCMapping):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["StreamKXCCMapping"] > 0

    def __call__(self, writer, kernel):
        module = Module("XCCMapping On")

        with writer.allocTmpSgpr(4, tag="StreamKXCCMappingOn_tmpSgprRes") as tmpSgprRes:
            sXCC   = tmpSgprRes.idx
            sGridC = tmpSgprRes.idx + 1
            sGridF = tmpSgprRes.idx + 2
            sGridM = tmpSgprRes.idx + 3
            sTmp = None
            sTmpRes = None
            sqTmp = writer.sgprPool.checkOut(1, "sqTmp")
            divisor = kernel["StreamKXCCMapping"]
            if ((divisor & (divisor - 1)) != 0): # Need temp registers if not power of 2
                sTmp = writer.sgprPool.checkOutAligned(2, 2, "sTmp", preventOverflow=not kernel.get("UseSubtileImpl", False))
                sTmpRes  = ContinuousRegister(idx=sTmp, size=2)

            # sGridC = ceil(grid / xccm)
            sGrid = writer.acquireStreamKConstSgpr(kernel, "skGrid")
            if writer.isStreamKConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.skConstVgprs["skGrid"])))
            module.add(SAddU32(dst=sgpr(sGridC), src0=sgpr(sGrid), src1=hex(kernel["StreamKXCCMapping"] - 1), comment="ceil(grid/xccm)"))
            module.add(scalarStaticDivideAndRemainder(qReg=sGridC, rReg=-1, dReg=sGridC, divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            # sGridF = floor(grid / xccm)
            # sGridM = grid % xccm
            module.add(scalarStaticDivideAndRemainder(qReg=sGridF, rReg=sGridM, dReg=sGrid, divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes))
            writer.releaseStreamKConstSgpr(sGrid)
            # sXCC = wg0 % xccm
            # sqtmp is temp register for quotient for non-power-of-2 case
            # sqtmp overlaps temp registers, works in this case and output is discarded
            module.add(scalarStaticDivideAndRemainder(qReg=sqTmp, rReg=sXCC, dReg="WorkGroup0", divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=2))
            # Check if current XCC requires a remainder WG or not
            module.add(SCmpLtU32(src0=sgpr(sXCC), src1=sgpr(sGridM), comment="XCCM < Remainder"))
            module.add(SCSelectB32(dst=sgpr(sGridC), src0=sgpr(sGridC), src1=sgpr(sGridF), comment="Select multiplier"))
            module.add(SCSelectB32(dst=sgpr(sGridM), src0=0, src1=sgpr(sGridM), comment="Select remainder"))
            # WG = floor(wg0 / xccm) * xccm + XCCoffset + optional remainder
            module.add(scalarStaticDivideAndRemainder(qReg="WorkGroup0", rReg=-1, dReg="WorkGroup0", divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            module.add(SMulI32(dst=sgpr(sXCC), src0=sgpr(sXCC), src1=sgpr(sGridC), comment="XCC group id"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sXCC), comment="Add XCC group offset"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sGridM), comment="Add remainder offset"))

            writer.sgprPool.checkIn(sqTmp)
            if sTmp is not None:
                writer.sgprPool.checkIn(sTmp)

        return module


class StreamKMemoryOrdering(Component):
    """
    Memory-ordering fences and flag accessors for the StreamK partial-tile
    handshake.

    StreamK uses a producer/consumer protocol: one workgroup writes a partial
    tile to a workspace and signals completion via a flag, and other
    workgroups poll the flag and read the partials. The required cross-CU
    memory ordering depends on the target ISA:

    - Most arches: `s_waitcnt vscnt(0)` before the flag store and an SMEM
      flag load with `glc/dlc/scope:SCOPE_DEV` are sufficient because
      ordering between L1/L2 and the device-scope coherence point is
      implicit.

    - gfx1250: the L2 has independent partitions and SMEM is not coherent
      with the VMEM flag store, so an explicit `global_wb scope:SCOPE_DEV`
      is required on the release side and a `global_inv scope:SCOPE_DEV`
      on the acquire side. Additionally, XNACK-replay can reorder a
      volatile/atomic VMEM op past in-flight VMEM, so `s_wait_xcnt 0` must
      precede such ops. The flag itself must be read via VMEM (not SMEM)
      to observe the producer's release-side fence.

    Selection is driven by the `HasInvWbDevFences` arch capability. The
    XNACK-replay drain in `preVolatileVmem` is gated separately on
    `RequiresXCntForVolatileVMEM` and lives on the abstract base so a
    future arch needing only one of the two can be supported by adding a
    single capability flag.
    """
    def __call__(self):
        assert(0)

    def preVolatileVmem(self, writer, comment="") -> Module:
        """Drain in-flight VMEM (XNACK-replay) before a volatile/atomic VMEM op.

        Required on arches with `RequiresXCntForVolatileVMEM`. No-op
        elsewhere.
        """
        module = Module("StreamK pre-volatile VMEM drain")
        if writer.states.archCaps["RequiresXCntForVolatileVMEM"]:
            module.add(SWaitXCnt(xcnt=0, comment=comment))
        return module

    @abc.abstractmethod
    def releaseFence(self, writer) -> Module:
        """Memory fence ordering prior partial-tile stores before the flag store."""
        pass

    @abc.abstractmethod
    def acquireFence(self, writer) -> Module:
        """Memory fence after observing the flag and before reading partials."""
        pass

    @abc.abstractmethod
    def readFlag(self, writer, dst, soffset) -> Module:
        """Read the StreamK completion flag into SGPR `dst` for compare."""
        pass

    @abc.abstractmethod
    def flagBufferMubuf(self) -> MUBUFModifiers:
        """MUBUF modifiers for buffer load/store of the flag word."""
        pass


class StreamKMemoryOrderingDefault(StreamKMemoryOrdering):
    """No-op cross-CU fences; SMEM flag with glc/dlc/SCOPE_DEV.

    Used on every arch that does not require explicit cross-L2 fences.
    """
    archCaps = {"HasInvWbDevFences": False}

    def releaseFence(self, writer) -> Module:
        module = Module("StreamK release fence (default)")
        module.add(SWaitCnt(vscnt=0, comment="wait for data store"))
        return module

    def acquireFence(self, writer) -> Module:
        return Module("StreamK acquire fence (default, no-op)")

    def readFlag(self, writer, dst, soffset) -> Module:
        module = Module("StreamK read flag (SMEM)")
        module.add(SLoadB32(dst=sgpr(dst), base=sgpr("AddressFlags", 2),
                            soffset=soffset,
                            smem=SMEMModifiers(glc=True, dlc=True,
                                               scope=CacheScope.SCOPE_DEV),
                            comment="get flag"))
        module.add(SWaitCnt(kmcnt=0, comment="wait for flag load"))
        return module

    def flagBufferMubuf(self) -> MUBUFModifiers:
        return MUBUFModifiers(offen=True, glc=True, dlc=True,
                              scope=CacheScope.SCOPE_DEV)


class StreamKMemoryOrderingDevScopeFences(StreamKMemoryOrdering):
    """Explicit cross-L2 release/acquire fences via global_wb/global_inv
    scope:SCOPE_DEV plus a VMEM-coherent flag read.

    Selected on arches whose L2 is partitioned across CUs/XCDs and whose
    SMEM is not coherent with the VMEM flag write (e.g. gfx1250).
    """
    archCaps = {"HasInvWbDevFences": True}

    def releaseFence(self, writer) -> Module:
        module = Module("StreamK release fence (dev-scope)")
        module.add(SWaitCnt(vlcnt=0,
            comment="release: drain in-flight loads before global_wb"))
        module.add(SWaitCnt(vscnt=0, comment="wait for data store"))
        module.add(GlobalWb(scope=CacheScope.SCOPE_DEV,
            comment="release: writeback partials to L2-coherent point"))
        module.add(SWaitCnt(vlcnt=0, vscnt=0,
            comment="release: wait for global_wb"))
        return module

    def acquireFence(self, writer) -> Module:
        module = Module("StreamK acquire fence (dev-scope)")
        module.add(GlobalInv(scope=CacheScope.SCOPE_DEV,
            comment="acquire: invalidate partials after flag"))
        module.add(SWaitCnt(vlcnt=0, comment="acquire: wait for global_inv"))
        return module

    def readFlag(self, writer, dst, soffset) -> Module:
        streamk = Component.StreamK.find(writer)
        module = Module("StreamK read flag (VMEM)")
        flagVgpr = writer.vgprPool.checkOut(1, "flagAcq")
        module.add(streamk.getFlagValue(writer, dst=vgpr(flagVgpr),
            soffset=soffset, comment="acquire: get flag (VMEM)"))
        module.add(SWaitCnt(vlcnt=0, comment="acquire: wait VMEM flag load"))
        module.add(VReadfirstlaneB32(dst=sgpr(dst), src=vgpr(flagVgpr),
            comment="move VMEM flag to SGPR for compare"))
        writer.vgprPool.checkIn(flagVgpr)
        return module

    def flagBufferMubuf(self) -> MUBUFModifiers:
        return MUBUFModifiers(offen=True, scope=CacheScope.SCOPE_DEV)


class StreamK(Component):
    """
    StreamK code.
    """
    # --- Variant feature flags. Each flag is snapshotted onto
    # writer.states.streamK by KernelWriter._initKernel and queried by
    # call sites that previously branched on the integer value of
    # kernel["StreamK"]. Defaults are False; each concrete StreamK*
    # subclass overrides the flags it sets.
    #
    # emitsParallelReductionSgprAliases: emit the SkSplit/skTiles +
    #     SkPartialIdx/Beta SGPR aliases pre-epilogue
    # borrowsSrdWsInEpilogue: epilogue may borrow the SrdWS SGPR pool
    # emitsWorkspaceReductionBpe: epilogue allocates dtype-aware Log2Bpe
    #     SGPRs for the workspace reduction
    # requiresWorkspaceReductionStorePath: the global-write elements
    #     emit must emit the workspace-reduction store branch
    #     (disables noGSUBranch fast path)
    # keepsConstantsInSgpr: the dynamic per-XCD path references SK
    #     kernarg constants directly, so they cannot be cached in VGPRs
    #     on gfx1250
    # supportsSubtileImpl: variant is accepted by UseSubtileImpl=1
    emitsParallelReductionSgprAliases: bool = False
    borrowsSrdWsInEpilogue: bool = False
    emitsWorkspaceReductionBpe: bool = False
    requiresWorkspaceReductionStorePath: bool = False
    keepsConstantsInSgpr: bool = False
    supportsSubtileImpl: bool = False

    def __call__(self):
        assert(0)

    @staticmethod
    def _depthUForTc(kernel, tc):
        """Return the per-tensor-character DepthU (element count along unroll).

        For MX scale tensors, DepthU is divided by the MX block size because
        there is one scale element per MXBlock data elements.
        For MXSA/MXSB (MX swizzled/pre-shuffle case), the swizzled block size
        is 32 * 256 so an additional *32 multiplier is needed.
        """
        key = "_DepthU%s" % tc
        if key in kernel:
            _DepthU = kernel[key]
            if tc in ("MXSA", "MXSB") and kernel.get("UseSubtileImpl"):
                # UseSubtileImpl MX swizzled(pre shuffle) case: swizzled block size is 32 * 256,
                # so the effective K stride for the scale tensor is DepthU * 32.
                # Non-subtile MX kernels use the raw _DepthU (scale elements per tile in K).
                _DepthU = (_DepthU * 32)
            return _DepthU
        return kernel["DepthU"]

    def shiftSrd(self, writer, srdIdx) -> Module:
      module = Module("shiftSrd")
      if writer.states.version[:2] == (12, 5):
        with writer.allocTmpSgpr(1, tag="shiftSrd_tmpSgprRes") as stmpRes:
          module.addComment("Shift num records for gfx125x")
          module.add(SAndB32(sgpr(stmpRes.idx), sgpr(srdIdx+2), 0x7F))
          module.add(SLShiftLeftB32(sgpr(stmpRes.idx), 25, sgpr(stmpRes.idx)))
          module.add(SAndB32(sgpr(srdIdx+1), sgpr(srdIdx+1), 0x1FFFFFF))
          module.add(SOrB32(sgpr(srdIdx+1), sgpr(srdIdx+1), sgpr(stmpRes.idx)))
          module.add(SLShiftRightB32(sgpr(srdIdx+2), 7, sgpr(srdIdx+2)))

      return module

    def _skv(self, writer, name):
        """Return the VGPR index holding a StreamK constant."""
        return writer.states.skConstVgprs[name]

    @abc.abstractmethod
    def preLoop(self, writer, kernel):
        pass

    @abc.abstractmethod
    def graWorkGroup(self, writer, kernel, tPA, tPB):
        pass

    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tPA, tPB, skipLroReset=False):
        """Recompute StreamK tile locals and map tile index to WorkGroup* for the *next* tile.

        After each persistent iteration's main body, ``StreamKIter`` already holds the starting
        global iteration index for the next chunk (set at the beginning of ``graWorkGroup``).
        Running ``skTileIndex`` + ``skIndexToWG`` + WGM remapping here matches the start of the
        next ``setupNewTile`` / ``graWorkGroup`` (without advancing ``StreamKIter`` again), so
        SGPRs are warm before the persistent back-edge.

        When ``skipLroReset`` is True the local-read-offset reset inside
        ``skTileIndex`` is suppressed.  This is needed when PAP runs *before*
        the NLL body: the NLL still needs the current tile's read pointers."""
        from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM, SpaceFillingCurveWalk

        module = Module("StreamK prefetchAcrossPersistentSetupNextTile")
        with writer.allocTmpSgpr(4, 2, "SKPrefetchTemp") as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB, skipLroReset=skipLroReset))
            module.add(self.skIndexToWG(writer, kernel, sTmp))
        if len(kernel["SpaceFillingAlgo"]):
            writer.states.WGMTransformLevels = len(kernel["SpaceFillingAlgo"])
            module.add(SpaceFillingCurveWalk(writer, kernel, "WGM"))
        else:
            module.add(DefaultWGM(writer, kernel, "WGM"))
        return module

    def computeTotalTiles(self, writer, kernel, dstSgpr):
        """Compute totalTiles = NumWorkGroups0 * NumWorkGroups1 * batchCount into dstSgpr."""
        module = Module("StreamK computeTotalTiles")
        module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="totalTiles = nwg0 * nwg1"))
        for i in range(kernel["ProblemType"]["NumIndicesC"] - kernel["ProblemType"]["NumIndicesFree"]):
            batchIdx = kernel["ProblemType"]["NumIndicesFree"] + i
            module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr(dstSgpr), src1=sgpr("SizesFree+%u" % batchIdx), comment="totalTiles *= batch dim %u" % i))
        return module

    def computeTotalIters(self, writer, kernel, dstSgpr):
        """Compute totalIters = NumWorkGroups0 * NumWorkGroups1 * batchCount * ItersPerTile into dstSgpr."""
        module = Module("StreamK computeTotalIters")
        module.add(self.computeTotalTiles(writer, kernel, dstSgpr))
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if writer.isStreamKConstantsToVgprEnabled(kernel):
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr(dstSgpr), src1=sgpr(sIpt), comment="totalIters = totalTiles * itersPerTile"))
        writer.releaseStreamKConstSgpr(sIpt)
        return module

    def skTileIndex(self, writer, kernel, sTmp, tPA, tPB, skipLroReset=False):
        module = Module("StreamK skTileIndex")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        # Always reset pointers to handle odd-exit case which moves LRO to the upper bank.
        # Skipped when PAP calls this before the NLL body: the current
        # tile's local read pointers must stay intact for the remaining MACs.
        if kernel["PrefetchGlobalRead"] and not skipLroReset:
            if not kernel["UseSubtileImpl"]:
                module.add(writer.localReadResetOffsets(kernel, tPA))
                if kernel["ProblemType"]["MXBlockA"] and "MX" in tPA:
                    module.add(writer.localReadResetOffsets(kernel, tPA["MX"]))
                if kernel["ProblemType"]["MXBlockB"] and "MX" in tPB:
                    module.add(writer.localReadResetOffsets(kernel, tPB["MX"]))
                module.add(writer.localReadResetOffsets(kernel, tPB))
            else:
                module.add(localReadResetOffsetsSubtile(writer, kernel))

        module.addComment0("StreamK calculate tile idx and map to WG")

        # sTmp = tile index
        sMagicNum = writer.acquireStreamKConstSgpr(kernel, "MagicNumberItersPerTile")
        sMagicShift = writer.acquireStreamKConstSgpr(kernel, "MagicShiftItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sMagicNum), src=vgpr(writer.states.skConstVgprs["MagicNumberItersPerTile"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sMagicShift), src=vgpr(writer.states.skConstVgprs["MagicShiftItersPerTile"])))
        # SK5: mode bit (30) already cleared at preLoop; keep magic add + shift.
        if kernel["StreamK"] == 5:
            module.add(SAndB32(dst=sgpr(sMagicShift), src0=sgpr(sMagicShift), src1=hex(0x8000001F),
                               comment="SK5: keep magic add bit (31) + 5-bit shift, drop mode bit (30)"))
        module.add(sMagicDiv2(sgpr(sTmp), sgpr(sTmp+1), sgpr("StreamKIter"), sgpr(sMagicNum), sgpr(sMagicShift), sgpr(sTmp+2)))
        writer.releaseStreamKConstSgpr(sMagicNum)
        writer.releaseStreamKConstSgpr(sMagicShift)
        # sTmp+1 = tile start, sTmp+2 = tile end
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sTmp), src1=sgpr(sIpt), comment="Tile start iteration"))
        module.add(SAddU32(dst=sgpr(sTmp+2), src0=sgpr(sTmp+1), src1=sgpr(sIpt), comment="Tile end iteration"))
        writer.releaseStreamKConstSgpr(sIpt)
        # local start
        module.add(SSubU32(dst=sgpr("StreamKLocalStart"), src0=sgpr("StreamKIter"), src1=sgpr(sTmp+1), comment="Local iteration start"))
        # local end (SK tile)
        module.add(SMinU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp+2), comment="1. (Local) iteration end (SK tile)"))
        module.add(SSubU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"), src1=sgpr(sTmp+1), comment="2. Local iteration end (SK tile)"))

        return module

    def skIndexToWG(self, writer, kernel, sTmp):
        # Note: There's one unused sgpr passed with sTmp.
        module = Module("StreamK skIndexToWG")

        # Map StreamK tile index to wg0/1
        module.addComment0("Map StreamK tile index to wg0/1/2")
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="Total tiles"))
        tmpVgpr = writer.vgprPool.checkOut(2, "div")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder(qReg="WorkGroup2", dReg=sTmp, divReg=sTmp+1, rReg=sTmp+2, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="TileID // nWG0*nWG1"))

        # Store tileID for use later in general WGM algo
        if kernel["SpaceFillingAlgo"]:
            module.add(SNop(waitState=1, comment=""))
            module.add(SMovB32(dst=sgpr("StreamKTileID"), src=sgpr(sTmp+2), comment=""))

        module.add(scalarUInt32DivideAndRemainder(qReg="WorkGroup1", dReg=sTmp+2, divReg="NumWorkGroups0", rReg="WorkGroup0", tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="TileID // nWG0"))
        tmpVgprRes = None
        writer.vgprPool.checkIn(tmpVgpr)
        module.addSpaceLine()

        return module

    def skExtraIters(self, writer, kernel, sSkExtraIters, sTmp):
        # skExtraIters = skTiles * ItersPerTile - SKItersPerWG * skGrid
        # Use sSkExtraIters/sTmp as readfirstlane destinations to reduce SGPR pressure
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)
        module = Module("StreamK skExtraIters")

        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkExtraIters), src=vgpr(writer.states.skConstVgprs["skTiles"])))
        sT = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sT), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            skTilesSrc = sSkExtraIters
        else:
            skTilesSrc = "skTiles"

        module.add(SMulI32(dst=sgpr(sSkExtraIters), src0=sgpr(skTilesSrc), src1=sgpr(sT)))
        writer.releaseStreamKConstSgpr(sT)

        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sTmp), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
        sT = writer.acquireStreamKConstSgpr(kernel, "skGrid")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sT), src=vgpr(writer.states.skConstVgprs["skGrid"])))
            skItersPerWgSrc = sTmp
        else:
            skItersPerWgSrc = "SKItersPerWG"

        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(skItersPerWgSrc), src1=sgpr(sT)))
        writer.releaseStreamKConstSgpr(sT)

        module.add(SSubU32(dst=sgpr(sSkExtraIters), src0=sgpr(sSkExtraIters), src1=sgpr(sTmp), comment="skTiles * ItersPerTile - SKItersPerWG * skGrid"))

        return module

    @abc.abstractmethod
    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        pass

    def computeLoadSrdCommon(self, writer, kernel, tP, sTmp):
        module = Module("StreamK Common computeLoadSrd")

        tileStart = sTmp + 2
        tc = tP["tensorChar"]
        depthU = self._depthUForTc(kernel, tc)
        # StreamK partial tile - offset to tile start index
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("StreamKLocalStart"), src1=depthU, comment="StreamK tile start offset"))
        strideL = writer.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        module.add(writer.s_mul_u64_u32(sgpr(sTmp), sgpr(sTmp+1), sgpr(sTmp), strideL, comment="StreamK tile start offset"))
        # Overflow check removed
        # if kernel["CheckDimOverflow"] >=2:
        #     kStr += self.assert_eq(sgpr(sTmp+1),0)
        module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(sTmp+0), comment="accum GsuOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(sTmp+1), comment="accum GsuOffset term to tilestart"))

        return module

    @abc.abstractmethod
    def computeStoreSrdStart(self, writer, kernel):
        pass

    def computeStoreSrdStartCommon(self, writer, kernel):
        module = Module("StreamK Common computeStoreSrdStart")
        if kernel["StreamKForceDPOnly"]:
            return module
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        # Check for parallel reduction
        # Paralell reduction stores to SrdD in split format, fixup happens in post kernel
        skSplitSrd = Label("SK_SplitSrd", "")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=skSplitSrd.getLabelName(), comment="Skip this block if using single-kernel stream-k fixup"))
        # Alpha/Beta will be applied in post kernel if necessary
        # module.add(SMovB32(dst=sgpr("Alpha"), src=1.0, comment="For parallel reduction, alpha applied in post kernel"))
        # module.add(SMovB32(dst=sgpr("Beta"), src=0.0, comment="For parallel reduction, beta applied in post kernel"))

        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)

        sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
        module.add(SCmpEQU32(src0=sgpr(sSkt), src1=1, comment="split == 1 ?"))
        writer.releaseStreamKConstSgpr(sSkt)
        module.add(SCBranchSCC1(labelName=skSplitSrd.getLabelName(), comment="branch if split == 1"))
        # Parallel reduction: adjust output buffer address to per split buffer
        with writer.allocTmpSgpr(4, alignment=1, tag="computeStoreSrdStartCommon_tmpSgprInfo") as tmpSgprInfo:
            if tmpSgprInfo.idx % 2 == 0:
                tmpSgprX2  = tmpSgprInfo.idx+0
                tmpSgpr0   = tmpSgprInfo.idx+0
                tmpSgpr1   = tmpSgprInfo.idx+1
                tmpSgpr2   = tmpSgprInfo.idx+2
                tmpSgpr3   = tmpSgprInfo.idx+3
            else:
                tmpSgprX2  = tmpSgprInfo.idx+1
                tmpSgpr0   = tmpSgprInfo.idx+1
                tmpSgpr1   = tmpSgprInfo.idx+2
                tmpSgpr2   = tmpSgprInfo.idx+0
                tmpSgpr3   = tmpSgprInfo.idx+3
            module.addComment("Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s")
            # PartialIdx was saved in sgprBeta for re-use
            module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr0), sgpr(tmpSgpr1), sgpr("SizesFree+0"), sgpr("SkPartialIdx"), comment="Free0"))
            for i in range(1, numDim):
                module.add(SSubU32(dst=sgpr(tmpSgpr2), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
                module.add(SMulI32(dst=sgpr(tmpSgpr2), src0=sgpr(tmpSgpr2), src1=sgpr("SkPartialIdx"), comment="Free%u" % i))
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr2), sgpr(tmpSgpr3), sgpr(tmpSgpr2), sgpr("StrideC%s"%writer.states.indexChars[i]), comment="Free%u" % i))
                module.add(SAddU32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr2), comment="Free%u" % i))
                module.add(SAddCU32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(tmpSgpr3), comment="Free%u" % i))
            module.add(SLShiftLeftB64(dst=sgpr(tmpSgprX2,2), src=sgpr(tmpSgprX2,2), shiftHex=log2(writer.states.bpeCinternal), comment="scale by bpe"))
            module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("SrdD+0"), src1=sgpr(tmpSgprX2), comment="add lo GSU offset to SRD"))
            module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("SrdD+1"), src1=sgpr(tmpSgpr1), comment="add hi GSU offset to SRD"))

        module.add(skSplitSrd)

        return module

    @abc.abstractmethod
    def graAddresses(self, writer, kernel, tP, vTmp):
        pass

    def graAddressesCommon(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Common graAddresses")

        tc = tP["tensorChar"]
        depthU = self._depthUForTc(kernel, tc)
        # StreamK partial tile - offset to tile start index
        tmpOffset = writer.sgprPool.checkOut(2, "skStartOffset")
        module.add(SMulI32(dst=sgpr(tmpOffset), src0=sgpr("StreamKLocalStart"), src1=int(depthU * tP["bpe"]), comment="StreamK tile start offset"))
        strideL = writer.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        module.add(writer.s_mul_u64_u32(sgpr(tmpOffset), sgpr(tmpOffset+1), sgpr(tmpOffset), strideL, "StreamK tile start offset"))
        # Overflow check removed
        # if kernel["CheckDimOverflow"] >=2:
        #     kStr += self.assert_eq(sgpr(tmpOffset+1),0)
        module.add(SAddU32(dst=sgpr(tmpOffset+0), src0=sgpr(tmpOffset+0), src1=sgpr("Address%s+0" % tc), comment="accum skOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tmpOffset+1), src0=sgpr(tmpOffset+1), src1=sgpr("Address%s+1" % tc), comment="accum skOffset term to tilestart"))
        module.add(VMovB32(dst=vgpr(vTmp+0), src=sgpr(tmpOffset+0)))
        module.add(VMovB32(dst=vgpr(vTmp+1), src=sgpr(tmpOffset+1)))
        writer.sgprPool.checkIn(tmpOffset)

        return module

    @abc.abstractmethod
    def declareStaggerParms(self, writer, kernel):
        pass

    def declareStaggerParmsCommon(self, writer, kernel):
        module = Module("StreamK Common declareStaggerParms")

        # Set stagger=0 for partial tiles to avoid using stagger larger than workload
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if writer.isStreamKConstantsToVgprEnabled(kernel):
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SCmpGtU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        module.add(SCMovB32(dst=sgpr("StaggerUIter"), src=0, comment="set stagger=0 for partial tiles"))
        module.add(SCmpLtU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr(sIpt), comment="does wg finish tile?"))
        module.add(SCMovB32(dst=sgpr("StaggerUIter"), src=0, comment="set stagger=0 for partial tiles"))
        writer.releaseStreamKConstSgpr(sIpt)

        return module

    @abc.abstractmethod
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        pass

    def tailLoopNumIterCommon(self, writer, kernel, loopCounter):
        module = Module("StreamK Common tailLoopNumIter")

        # skip tail loop if StreamK WG not processing final iteration
        # Check if tile finished
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if writer.isStreamKConstantsToVgprEnabled(kernel):
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SCmpLtU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr(sIpt), comment="Check if WG processes final iteration of tile"))
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SCMovB32(dst=loopCounter, src=0, comment="This WG not completing tile"))

        return module

    @abc.abstractmethod
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        pass

    def calculateLoopNumIterCommon(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Common calculateLoopNumIter")

        # Use StreamK params for loop count
        module.add(SSubU32(dst=sgpr(loopCounterName), src0=sgpr("StreamKLocalEnd"), src1=sgpr("StreamKLocalStart"), comment="StreamK loop counter = localEnd - localStart"))
        # Short circuit if alpha==0 (set loopCounter to 0 to skip main loop)
        alphaLabel2 = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
        module.add(BranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), alphaLabel2))
        module.add(SMovB32(dst=sgpr(loopCounterName), src=0, comment="Skip iterations"))
        module.add(alphaLabel2)

        # Adjust loop count for tail loop
        if not kernel["NoTailLoop"]:
            tmpSgpr = tmpSgprInfo.idx
            unrollIdx = writer.states.unrollIdx
            loopChar = writer.states.indexChars[kernel["ProblemType"]["IndicesSummation"][unrollIdx]]

            assert kernel["DepthU"] % 2 == 0 # Assuming DepthU is power of 2, if odd DepthU were supported this divide would need 2 more temp registers for divide
            maxUnit = writer.states.tailloopInNllmaxUnit
            # tailloopInNll + maxUnit == 1 case, tailloopInNll is always used and no need to adjust loopCounter
            if not (writer.states.tailloopInNll and maxUnit == 1):
                if ((kernel["DepthU"] & (kernel["DepthU"] - 1)) == 0):
                    module.add(scalarStaticDivideAndRemainder(qReg=tmpSgpr, rReg=tmpSgpr+1, dReg=("SizesSum+%u" % unrollIdx), divisor=kernel["DepthU"], tmpSgprRes=None, doRemainder=2))
                else:
                    with writer.allocTmpSgpr(4, tag="calculateLoopNumIterCommon_tmpSgpr1") as tmpSgpr1:
                        module.add(scalarStaticDivideAndRemainder(qReg=tmpSgpr, rReg=tmpSgpr+1, dReg=("SizesSum+%u" % unrollIdx), divisor=kernel["DepthU"], tmpSgprRes=tmpSgpr1, doRemainder=2))
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr+1), src1=0, comment="numIter%s == 0"%loopChar ))
                module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=0, src1=1, comment="check if size uses tail loop"))
                sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
                if writer.isStreamKConstantsToVgprEnabled(kernel):
                    module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
                module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr(sIpt), comment="Check if WG processes final iteration of tile"))
                writer.releaseStreamKConstSgpr(sIpt)
                module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=0, comment="this WG runs tail loop"))

                if writer.states.tailloopInNll and maxUnit > 1:
                    # tailloopInNll + maxUnit > 1 case, we need to check if SizesSum is multiple of maxUnit at runtime.
                    # if SizesSum is not multiple of maxUnit, we do not use tailloopInNll and need to decrement loopCounter for StreamK
                    # if SizesSum is multiple of maxUnit, we  use tailloopInNll and need to increment loopCounter by 1.
                    # With considering both increment and decrement, we do not need to adjust loopCounter.
                    module.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr("SizesSum+%u" % unrollIdx), src1=maxUnit-1, \
                                       comment="if summation is not multiple of %u, skip tailloopInNll"%maxUnit))
                    module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=0, comment="do not decrement in tailloopInNll case"))

                module.add(SSubU32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), src1=sgpr(tmpSgpr), comment="Adjust loop counter for tail loop"))
                module.add(SMaxI32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), src1=0, comment="Avoid setting negative value to loopCounter"))

        return module

    @abc.abstractmethod
    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        pass

    def storeBranchesCommon(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Common storeBranches")

        # No branches when no StreamK partial/fixup path can be reached.
        if kernel["StreamKAtomic"] or kernel["StreamKForceDPOnly"]:
            return module

        memOrder = Component.StreamKMemoryOrdering.find(writer)
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)
        skStoreLabel = Label(label=writer.labels.getNameInc("SK_Store"), comment="")

        if kernel["StreamKFixupTreeReduction"] == 1:
            skFixupTreeLabel = Label(label=writer.labels.getNameInc("SK_Fixup_Tree"), comment="")
            skFixupTreeLoopStart = Label(label=writer.labels.getNameInc("SK_Fixup_TreeLoop_Start"), comment="")
            skFixupWaitForFlag = Label(label=writer.labels.getNameInc("SK_Fixup_Wait_Flag"), comment="")
            endFixupLoop = Label(label=writer.labels.getNameInc("endFixupLoop"), comment="")
            skFixupCalcPartialIdx = Label(label=writer.labels.getNameInc("SK_Fixup_CalcPartialIdx"), comment="")

            # sIter = writer.sgprPool.checkOut(2, "SKIter")
            sPartialIdx = writer.sgprPool.checkOut(1, "SK_Fixup_Partial_idx")

            sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
            tmpSgpr = writer.sgprPool.checkOut(1, tag="StreamKCommon_storeBranches_tmpSgpr")
            module.add(self.skExtraIters(writer, kernel, sSkExtraIters, tmpSgpr))
            writer.sgprPool.checkIn(tmpSgpr)

            # Skip to global write if WG started and finished tile
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
            module.add(SCBranchSCC0(labelName=skFixupTreeLabel.getLabelName(), comment="If we didn't start the tile, always to SK Tree fixup"))
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr(sIpt), comment="does wg finish tile?"))
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if started and finished tile, go to regular store code"))

            # Start Tree Fixup
            module.add(skFixupTreeLabel)

            # Calculate partialIdx
            tmpVgpr = writer.vgprPool.checkOutAligned(4, 2, tag="StreamKCommon_storeBranches_tmpVgpr")
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=4)

            # Get Iter of the start of the tile
            tmpSgpr = writer.sgprPool.checkOut(3, tag="StreamKCommon_storeBranches_tmpSgpr2")

            # Compute dpSectionSize = (totalTiles - skTiles) * ItersPerTile
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
            sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
            module.add(self.computeTotalTiles(writer, kernel, tmpSgpr))
            module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=sgpr(sSkt), comment="dpTiles = totalTiles - skTiles"))
            writer.releaseStreamKConstSgpr(sSkt)
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=sgpr(sIpt), comment="Offset to first SK tile"))
            module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr("StreamKIter"), src1=sgpr(tmpSgpr), comment="Iter relative to starting SK iter"))

            module.add(SSubU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr), src1=1, comment="minus 1 to get Iter in current tile"))
            module.add(scalarUInt24DivideAndRemainder(qReg=tmpSgpr+0, dReg=tmpSgpr+1, divReg=sIpt, rReg=-1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="wgCount = tileStart / (itersPerTile)"))
            module.add(SMulI32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+0), src1=sgpr(sIpt), comment="tileStart=tileIdx * ItersPerTile"))
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(sIpw), src1=1, comment="ItersPerWG w/ extraIter"))
            module.add(scalarUInt24DivideAndRemainder(qReg=tmpSgpr+2, dReg=tmpSgpr+1, divReg=tmpSgpr+0, rReg=-1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="wgCount = tileStart / (itersPerWG+1)"))
            module.add(SCmpLtU32(src0=sgpr(tmpSgpr+2), src1=sgpr(sSkExtraIters), comment="find co-op group start"))
            module.add(SCBranchSCC1(labelName=skFixupCalcPartialIdx.getLabelName(), comment="All WG have extra iter so far, skip following calcs"))
            module.add(SSubU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+1), src1=sgpr(sSkExtraIters), comment="tileStart - extraIters"))
            module.add(scalarUInt24DivideAndRemainder(qReg=tmpSgpr+2, dReg=tmpSgpr+0, divReg=sIpw, rReg=-1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="wgExtraIters = (tileStart - extraIters) / itersPerWG"))
            writer.releaseStreamKConstSgpr(sIpw)
            module.add(skFixupCalcPartialIdx)

            sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
            module.add(SSubU32(dst=sgpr(sPartialIdx), src0=sgpr(sIdx), src1=sgpr(tmpSgpr+2), comment="partialIdx = streamkidx - coopGroupStart"))
            writer.releaseStreamKConstSgpr(sIdx)

            tmpVgprRes = None
            writer.vgprPool.checkIn(tmpVgpr)

            sFlagIdx = writer.sgprPool.checkOut(1, "FlagIdx")
            sIdxOffset = writer.sgprPool.checkOut(1, "IdxOffset")
            module.add(SMovB32(dst=sgpr(sIdxOffset), src=1, comment="Init IdxOffset=1"))

            module.add(skFixupTreeLoopStart) # start tree fixup loop

            # First, jump to partial write if (partialIdx//2)*2 != partialIdx, i.e. branch if last bit is 1
            module.add(SAndB32(dst=sgpr(tmpSgpr+0), src0=sgpr(sPartialIdx), src1=1))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr+0), src1=1, comment="partialIdx&1==1?"))
            module.add(writer.longBranchScc1(skPartialsLabel, posNeg=1))
            module.add(SLShiftRightB32(dst=sgpr(sPartialIdx), src=sgpr(sPartialIdx), shiftHex=log2(2), comment="sPartialIdx // 2"))
            sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
            if writer.isStreamKConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
            module.add(SAddU32(dst=sgpr(sFlagIdx), src0=sgpr(sIdx), src1=sgpr(sIdxOffset), comment="flagIdx=StreamKIdx+IdxOffset"))
            writer.releaseStreamKConstSgpr(sIdx)

            # If the flag we're waiting for is past this tile we can finish the fixup step
            # LocalEnd + 1 + (sIdxOffset-1) * SKItersPerWG + (Extras)
            sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
            if writer.isStreamKConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
            module.add(SSubU32(dst=sgpr(tmpSgpr+1), src0=sgpr(sIdxOffset), src1=1, comment="Starting on next WG so offset-1"))
            module.add(SMulI32(dst=sgpr(tmpSgpr+2), src0=sgpr(sIpw), src1=sgpr(tmpSgpr+1), comment="Before extra iters"))
            writer.releaseStreamKConstSgpr(sIpw)

            module.add(SSubU32(dst=sgpr(tmpSgpr+0), src0=sgpr(sFlagIdx), src1=sgpr(sSkExtraIters), comment="TargetWG-ExtraIters"))
            module.add(SMinU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+1), comment="min of above and (offset-1)"))
            module.add(SCmpLtU32(src0=sgpr(sFlagIdx), src1=sgpr(sSkExtraIters), comment="TargetWG < extraIters?"))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr+0), src0=0, src1=sgpr(tmpSgpr+0), comment="If True, don't sub any iters"))
            module.add(SSubU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+0), comment="extras = (offset-1) - (possible extras)"))
            module.add(SAddU32(dst=sgpr(tmpSgpr+2), src0=sgpr(tmpSgpr+2), src1=sgpr(tmpSgpr+1), comment="Add possible extra iters"))
            module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr("StreamKLocalEnd"), src1=1, comment="Start of next wg"))
            module.add(SAddU32(dst=sgpr(tmpSgpr+2), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2)))
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            if writer.isStreamKConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(SCmpGtU32(src0=sgpr(tmpSgpr+2), src1=sgpr(sIpt)))
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(SCBranchSCC1(labelName=endFixupLoop.getLabelName()))
            writer.sgprPool.checkIn(tmpSgpr)

            # check flag
            tmpSgpr = writer.sgprPool.checkOut(2, "globalWriteElements")
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sFlagIdx), shiftHex=log2(4), comment="flag offset based on wg index"))

            module.add(skFixupWaitForFlag) # loop to wait for flag
            module.add(memOrder.readFlag(writer, dst=tmpSgpr+1, soffset=sgpr(tmpSgpr)))
            if kernel["DebugStreamK"] & 2 == 0: # Don't wait for partials if not being written
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr+1), src1=1, comment="check if ready"))
                module.add(SCBranchSCC0(labelName=skFixupWaitForFlag.getLabelName(), comment="if flag not set, wait and check again"))
                module.add(memOrder.acquireFence(writer))

            module.add(SBarrier(comment="wait for all workgroups before resetting flag"))
            skipFlagReset = Label(label=writer.labels.getNameInc("SK_SkipFlagReset"), comment="")
            module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr+2), src=vgpr("Serial"), comment="Wave 0 updates flags"))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=0, comment="Check for wave 0"))
            module.add(SCBranchSCC0(labelName=skipFlagReset.getLabelName(), comment="Skip flag reset"))
            if writer.states.asmCaps["HasScalarStore"]:
                # (tmpSgpr+2) contains a vlue of 0, use it to reset the flag
                module.add(SStoreB32(src=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=True), comment="reset flag"))
            else:
                module.add(VMovB32(dst=vgpr(tmpVgpr), src=0, comment="move 0 to tmpVgpr"))
                module.add(self.setFlagValue(writer, src=vgpr(tmpVgpr), soffset=sgpr(tmpSgpr), comment="reset flag"))
            module.add(skipFlagReset)

            writer.sgprPool.checkIn(tmpSgpr)

            # fixup step
            if kernel["DebugStreamK"] & 1 == 0: # Skip fixup reads if set, need to do the loop if partial writes are enabled
                fixupEdge = [False] # Test no edge variant
                module.add(self.fixupStep(writer, kernel, vectorWidths, elements, fixupEdge, tmpVgpr, cvtVgprStruct, sFlagIdx))

            # Could branch if our new offset puts us off the tile, but we essentially do that when calculating if our target wg is off the tile earlier
            module.add(SLShiftLeftB32(dst=sgpr(sIdxOffset), src=sgpr(sIdxOffset), shiftHex=log2(2), comment="IdxOffset *= 2 for Tree reduction"))
            module.add(SBranch(labelName=skFixupTreeLoopStart.getLabelName(), comment="Branch to continue fixup loop"))

            # If we started the tile, we reduced the partial results to that WG, so global write
            # Otherwise, partial write
            module.add(endFixupLoop)
            # Done fixup loop
            writer.sgprPool.checkIn(sIdxOffset)
            writer.sgprPool.checkIn(sFlagIdx)
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
            module.add(writer.longBranchScc0(skPartialsLabel, posNeg=1))
        else: # linear reduction
            skFixupLabel = Label(label=writer.labels.getNameInc("SK_Fixup"), comment="")

            # StreamK store branches
            # if we're doing parallel reduction, jump to global write
            # module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
            # module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if using parallel reduction, go to regular store code"))

            tmpSgpr = writer.sgprPool.checkOut(4, "globalWriteElements")
            # if we did not start the tile, store partials
            # branch to beta == 0 store path
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
            module.add(writer.longBranchScc0(skPartialsLabel, posNeg=1))
            # module.add(SCBranchSCC0(labelName=skPartialsLabel.getLabelName(), comment="Branch if not start tile, store partials"))

            if kernel["DebugStreamK"] & 1 == 0:
                # if we started and finished the tile, regular store code
                # branch to regular store code, skip fixup step
                sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
                if skConstsInVgprs:
                    module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
                module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr(sIpt), comment="does wg finish tile?"))
                module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if started and finished tile, go to regular store code"))

                # if we started the tile but did not finish it, fix up step
                # run fixup code before regular store code
                sCtaIdx = writer.sgprPool.checkOut(1, "CtaIdx") # self.defineSgpr("CtaIdx", 1)
                sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
                if skConstsInVgprs:
                    module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(SAddU32(dst=sgpr(sCtaIdx), src0=sgpr(sIdx), src1=1, comment="input partial tile index"))
                writer.releaseStreamKConstSgpr(sIdx)

                sFixupEnd = writer.sgprPool.checkOut(1, "FixupEnd") # self.defineSgpr("CtaEnd", 1)
                sMagicNum = writer.acquireStreamKConstSgpr(kernel, "MagicNumberItersPerTile")
                sMagicShift = writer.acquireStreamKConstSgpr(kernel, "MagicShiftItersPerTile")
                if skConstsInVgprs:
                    module.add(VReadfirstlaneB32(dst=sgpr(sMagicNum), src=vgpr(writer.states.skConstVgprs["MagicNumberItersPerTile"])))
                    module.add(VReadfirstlaneB32(dst=sgpr(sMagicShift), src=vgpr(writer.states.skConstVgprs["MagicShiftItersPerTile"])))
                module.add(sMagicDiv2(sgpr(tmpSgpr), sgpr(tmpSgpr+1), sgpr("StreamKIterEnd"), sgpr(sMagicNum), sgpr(sMagicShift), sgpr(tmpSgpr+2)))
                writer.releaseStreamKConstSgpr(sMagicNum)
                writer.releaseStreamKConstSgpr(sMagicShift)
                module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=sgpr(sIpt), comment="start iteration of partial tile"))
                writer.releaseStreamKConstSgpr(sIpt)
                module.add(SSubU32(dst=sgpr(sFixupEnd), src0=sgpr("StreamKIterEnd"), src1=sgpr(tmpSgpr), comment="calc iterations completed by this WG"))

                module.add(skFixupLabel)

                # Check flag
                module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sCtaIdx), shiftHex=log2(4), comment="flag offset based on CTA index"))
                module.add(memOrder.readFlag(writer, dst=tmpSgpr+2, soffset=sgpr(tmpSgpr)))
                if kernel["DebugStreamK"] & 2 == 0:
                    module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=1, comment="check if ready"))
                    module.add(SCBranchSCC0(labelName=skFixupLabel.getLabelName(), comment="if flag not set, wait and check again"))
                    module.add(memOrder.acquireFence(writer))

                # TODO Barrier here to sync all threads in workgroup, but maybe better to have separate flag for each wavefront (to be tested)
                module.add(SBarrier(comment="wait for all workgroups before resetting flag"))
                skipFlagReset = Label(label=writer.labels.getNameInc("SK_SkipFlagReset"), comment="")
                module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr+2), src=vgpr("Serial"), comment="Wave 0 updates flags"))
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=0, comment="Check for wave 0"))
                module.add(SCBranchSCC0(labelName=skipFlagReset.getLabelName(), comment="Skip flag reset"))
                if writer.states.asmCaps["HasScalarStore"]:
                    # (tmpSgpr+2) contains a vlue of 0, use it to reset the flag
                    module.add(SStoreB32(src=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=True), comment="reset flag"))
                else:
                    module.add(VMovB32(dst=vgpr(tmpVgpr), src=0, comment="move 0 to tmpVgpr"))
                    module.add(self.setFlagValue(writer, src=vgpr(tmpVgpr), soffset=sgpr(tmpSgpr), comment="reset flag"))
                module.add(skipFlagReset)
                writer.sgprPool.checkIn(tmpSgpr)

                fixupEdge = [False] # Test no edge variant
                # Fixup writes to workspace (no bias LDS barriers), safe to defer.
                deferFixup = (
                    kernel.get("UseSubtileImpl")
                )
                if deferFixup:
                    fixupDeferredLabel = Label(label=writer.labels.getNameInc("Fixup_E0_Deferred"), comment="")
                    fixupReturnLabel = Label(label=writer.labels.getNameInc("Fixup_E0_Deferred_Return"), comment="")
                    # Keep original Fixup_E0 label inline as a stub
                    fixupInlineLabel = Label(label=writer.labels.getNameInc("Fixup_E%u" % 0), comment="")
                    module.add(fixupInlineLabel)
                    with writer.allocTmpSgpr(3, tag="StreamKOn_fixupInline_tmpSgprInfo") as tmpSgprInfo:
                        module.add(SLongBranchPositive(fixupDeferredLabel, tmpSgprInfo, comment="jump to deferred fixup block"))
                    module.addComment0("=" * 60)
                    module.addComment0(" Fixup block deferred to after persistent loop")
                    module.addComment0(" (would have been inline here in non-deferred version)")
                    module.addComment0("=" * 60)
                    module.add(fixupReturnLabel)
                    # Collect fixup code in deferred module
                    fixupModule = Module("Fixup_DeferredBlock")
                    fixupModule.add(fixupDeferredLabel)
                    fixupModule.add(self.fixupStep(writer, kernel, vectorWidths, elements, fixupEdge, tmpVgpr, cvtVgprStruct, sCtaIdx))
                    with writer.allocTmpSgpr(3, tag="StreamKOn_fixupDeferred_tmpSgprInfo") as tmpSgprInfo:
                        posLabel = writer.labels.getNameInc("FixupDeferredReturnDir")
                        fixupModule.add(SLongBranch(fixupReturnLabel, tmpSgprInfo, posLabel, comment="return from deferred fixup block"))
                    writer.states.deferredFixupModule = fixupModule
                else:
                    fixupModule = None
                    module.add(self.fixupStep(writer, kernel, vectorWidths, elements, fixupEdge, tmpVgpr, cvtVgprStruct, sCtaIdx))

                if kernel["StreamK"] >= 2:
                    sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
                    sIterCount = writer.sgprPool.checkOut(1, "iterCount")
                    module.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIterCount)) # sIterCount is a temp register
                    sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
                    if writer.isStreamKConstantsToVgprEnabled(kernel):
                        module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
                    module.add(SAddU32(dst=sgpr(sIterCount), src0=sgpr(sIpw), src1=1, comment="Add extra iter"))
                    module.add(SCmpLtU32(src0=sgpr(sCtaIdx), src1=sgpr(sSkExtraIters), comment="Check if next WG had an extra iteration"))
                    module.add(SCSelectB32(dst=sgpr(sIterCount), src0=sgpr(sIterCount), src1=sgpr(sIpw), comment="Select correct number of iterations for next WG"))
                    writer.releaseStreamKConstSgpr(sIpw)
                    module.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sFixupEnd), src1=sgpr(sIterCount), comment="next partial tile iteration"))
                    writer.sgprPool.checkIn(sSkExtraIters)
                    writer.sgprPool.checkIn(sIterCount)
                module.add(SAddU32(dst=sgpr(sCtaIdx), src0=sgpr(sCtaIdx), src1=1, comment="next partial tile index"))
                if kernel["StreamK"] == 1:
                    sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
                    if writer.isStreamKConstantsToVgprEnabled(kernel):
                        module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
                    module.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sFixupEnd), src1=sgpr(sIpw), comment="next partial tile iteration"))
                    writer.releaseStreamKConstSgpr(sIpw)
                sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
                if writer.isStreamKConstantsToVgprEnabled(kernel):
                    module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
                module.add(SCmpLtU32(src0=sgpr(sFixupEnd), src1=sgpr(sIpt), comment="done loading partial tiles?"))
                writer.releaseStreamKConstSgpr(sIpt)
                module.add(SCBranchSCC1(labelName=skFixupLabel.getLabelName(), comment="Branch to continue fixup loop"))

                writer.sgprPool.checkIn(sFixupEnd)
                writer.sgprPool.checkIn(sCtaIdx)

        module.add(skStoreLabel)

        return module

    @abc.abstractmethod
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        pass

    def writePartialsCommon(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Common writePartials")

        # No partial writes for atomic or DP-only StreamK.
        if kernel["StreamKAtomic"] or kernel["StreamKForceDPOnly"]:
            return module

        module.add(skPartialsLabel)
        if kernel["DebugStreamK"] & 2 != 0:
            return module

        # fixupEdge = [False] # Temporary hack to test no edge variant
        edges = [False]

        partialsLabels = {}
        for edge in edges:
            partialsLabels[edge] = Label(writer.labels.getNameInc("GW_Partials_E%u" % ( 1 if edge else 0)), comment="")

        if False in edges and True in edges:
            with self.allocTmpSgpr(4, tag="StreamKCommon_writePartials_tmpSgprInfo") as tmpSgprInfo:
                module.add(writer.checkIsEdge(kernel, tmpSgprInfo, partialsLabels[True], partialsLabels[True]))

        # WritePartials writes to workspace (no bias LDS barriers), safe to defer.
        deferPartials = (
            kernel.get("UseSubtileImpl")
        )
        if deferPartials:
            partialsDeferredLabel = Label(label=writer.labels.getNameInc("GW_Partials_E0_Deferred"), comment="")
            partialsReturnLabel = Label(label=writer.labels.getNameInc("GW_Partials_E0_Deferred_Return"), comment="")
            # Inline stub
            for edge in edges:
                module.add(partialsLabels[edge])
            with writer.allocTmpSgpr(3, tag="StreamKCommon_writePartials_tmpSgprInfo2") as tmpSgprInfo:
                module.add(SLongBranchPositive(partialsDeferredLabel, tmpSgprInfo, comment="writePartials (deferred)"))
            module.addComment0("=" * 60)
            module.addComment0(" WritePartials block deferred to after persistent loop")
            module.addComment0(" (would have been inline here in non-deferred version)")
            module.addComment0("=" * 60)
            module.add(partialsReturnLabel)
            module.add(SBranch(labelName=endLabel.getLabelName(), comment="jump to end"))
            # Deferred block
            partialsModule = Module("Partials_DeferredBlock")
            partialsModule.add(partialsDeferredLabel)
            for edge in edges:
                sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
                if writer.isStreamKConstantsToVgprEnabled(kernel):
                    partialsModule.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                partialsModule.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sIdx)))
                writer.releaseStreamKConstSgpr(sIdx)
                partialsModule.add(self.partialsWriteProcedure(writer, kernel, vectorWidths, elements, False, False, edge, tmpVgpr, cvtVgprStruct, partialsReturnLabel))
            writer.states.deferredPartialsModule = partialsModule
        else:
            for edge in edges:
                module.add(partialsLabels[edge])
                sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
                if writer.isStreamKConstantsToVgprEnabled(kernel):
                    module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sIdx)))
                writer.releaseStreamKConstSgpr(sIdx)
                module.add(self.partialsWriteProcedure(writer, kernel, vectorWidths, elements, False, False, edge, tmpVgpr, cvtVgprStruct, endLabel))

        return module

    def computeWorkspaceSrd(self, writer, kernel, sPartialIdx, tmpSgpr = None):
        module = Module("StreamK Common computeWorkspaceSrd")

        # Base Address
        module.add(SMovB64(dst=sgpr("SrdWS", 2), src=sgpr("AddressWS", 2)))
        module.add(SMovB32(dst=sgpr("SrdWS+2"), src="BufferOOB"))
        module.add(SMovB32(dst=sgpr("SrdWS+3"), src="Srd127_96"))
        module.add(writer.shiftSrd("WS"))

        tmpLocal = None
        if tmpSgpr == None:
            tmpLocal = writer.sgprPool.checkOut(1, "SKMappingTemp")
            tmpSgpr = tmpLocal

        assert kernel["BufferStore"]
        module.addSpaceLine()
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=hex(kernel["MacroTile0"]*kernel["MacroTile1"]*writer.states.bpeCinternal), src1=sPartialIdx, comment="Offset to correct partials tile"))
        module.add(SAddU32(dst=sgpr("SrdWS+0"), src0=sgpr("SrdWS+0"), src1=sgpr(tmpSgpr), comment="add lo to SRD"))
        module.add(SAddCU32(dst=sgpr("SrdWS+1"), src0=sgpr("SrdWS+1"), src1=0, comment="add hi to SRD"))

        if tmpLocal is not None:
            writer.sgprPool.checkIn(tmpLocal)

        return module

    def partialsWriteProcedure(self, writer, kernel, vectorWidths, elements, alpha, beta, edge, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Common partialsWriteProcedure")
        memOrder = Component.StreamKMemoryOrdering.find(writer)

        # PreLoopVmcntCaseStr = ""
        # # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
        # if self.canOptimizePreLoopLWVmcnt:
        #     if beta:
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_B1_Store
        #     elif edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
        #     else:
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
        #     PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
        #         "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
        #     # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
        #     # OrdNLL_B1_Store is excluded
        #     if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

        edgeI = edge
        #edgeI = True    # set to True to disable vector stores
        gwvw = vectorWidths[edgeI]
        #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        vectorDataTypes = VectorDataTypes()
        ss = StoreState(writer, kernel, gwvw, edge, beta, False, elements[edgeI], vectorDataTypes, dim=0, isWorkspace=True)

        #print self.vgprPool.state()
        # Use VGPR up to next occupancy threshold:
        maxVgprs, _ = writer.getMaxRegsForOccupancy(kernel["NumThreads"], writer.vgprPool.size(), writer.sgprPool.size(), \
            writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
        if writer.states.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
            # len(elements[edgeI])
            # tl = []
            # for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
            #     tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
            # for t in tl:
            #     self.vgprPool.checkIn(t)
            writer.vgprPool.growPool(writer.vgprPool.size()-writer.vgprPool.available(), maxVgprs, 1, \
                "grow-pool up to next occupancy for GlobalWrite")
        # align = 1
        # # align adjustment
        # if self.ss.cfg.numVgprsPerAddr > 1:
        #     align = max(align, self.ss.cfg.numVgprsPerAddr)
        # if self.ss.cfg.numVgprPerValuC*gwvw > 1:
        #     align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
        # if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
        #     align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
        numVgprAvailable = writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel.
        # Before growing the pool, see if we can shrink the write vector width instead?
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.    Maybe want to move vgprSerial to first vgpr?

        # TODO: Minimum elems for StoreRemap
        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        minElements = 1
        if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
            minElements = 2
        elif kernel["ProblemType"]["DataType"].is8bitFloat():
            # TODO STREAM-K check if needed
            minElements = 4
        minNeeded = minElements * ss.numVgprsPerElement

        shrinkDb = 0
        if shrinkDb:
            print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)

        if numVgprAvailable < minNeeded:
            gwvwOrig = gwvw
            currentOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size(), \
                writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
            futureOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size() - numVgprAvailable + minNeeded, \
                writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)

            if shrinkDb:
                print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                    % (currentOccupancy, futureOccupancy, writer.vgprPool.size(), \
                    numVgprAvailable, minElements*ss.numVgprsPerElement))
            if futureOccupancy > currentOccupancy:
                if shrinkDb:
                    print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                        (writer.states.kernelName))
                    print("     numVgprAvailable=", numVgprAvailable, \
                        "numVgprsPerElement=", ss.numVgprsPerElement, \
                        "beta=", beta, "gwvw=", gwvw)
            elif gwvw != gwvwOrig:
                ss.gwvw = gwvw # make both representations consistent
                if shrinkDb:
                    print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                        % (writer.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

            if numVgprAvailable < minElements*ss.numVgprsPerElement:
                print2("info: growing pool += %d * %d for GlobalWrite\n" \
                    % (minElements,ss.numVgprsPerElement))
                print2(writer.vgprPool.state())
                # tl = []
                # for i in range(0,minElements):
                #     tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
                # for t in tl:
                #     self.vgprPool.checkIn(t)
                writer.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
                    "grow-pool for GlobalWrite")
                numVgprAvailable = writer.vgprPool.available()
                print2(writer.vgprPool.state())

        # set atomicW after we potentially resize GWVW
        # atomicW = min(gwvw, kernel["VectorAtomicWidth"])
        atomicW = min(gwvw, writer.getVectorAtomicWidth(kernel))

        # print("NumVgprAvailable", numVgprAvailable)
        if ss.numVgprsPerElement:
            numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
        else:
            numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

        # Cap batch size to align on MIWaveTile[0] boundaries (see refineOccupancy).
        if kernel.get("UseSubtileImpl") and kernel.get("EnableMatrixInstruction"):
            miwt0 = kernel["MIWaveTile"][0]
            totalElems = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]
            if numElementsPerBatch >= totalElems:
                numElementsPerBatch = totalElems
            elif miwt0 > 1 and numElementsPerBatch >= miwt0:
                numElementsPerBatch = (numElementsPerBatch // miwt0) * miwt0

        # assert(writer.states.numVgprValuC % gwvw == 0) # sanity check

        numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

        if shrinkDb:
            print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
                "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
        if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
            numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
            # only do an even number of halves - since these share hi/lo pieces of some registers?
            if numElementsPerBatch > 1:
                numElementsPerBatch = int(numElementsPerBatch/2)*2
            elif not kernel["EnableMatrixInstruction"]:
                # (excluding MFMA+LSU case. It can work without an issue)
                # The globalWriteBatch routine below can't handle odd elements per batch
                # and 0 elements per batch is illegal.
                # so if we don't have *GPR resources to handle a larger batch then need
                # to mark overflowedResources rather than generate a kernel that won't work.
                # It might be possible to fix globalWriteBatch to handle this case but these
                # are likely to be low-performing so likely not worth optimizing.
                if shrinkDb:
                    print("WARNING: half requires at least two elements per batch")
                self.overflowedResources = 3
        #elif kernel["ProblemType"]["DataType"].is8bitFloat():
        #    if numElementsPerBatch > 1:
        #        numElementsPerBatch = int(numElementsPerBatch/4)*4

        assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%writer.states.kernelName

        #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
        # if atomic and (ss.optSingleColVgpr or ss.optSharedColVgpr):
        #     # hack to avoid re-using address vgpr across rows
        #     # atomics need to perform several memory operations
        #     # if the batch spans multiple rows, need multiple address vgpr
        #     # which is not currently supported in the two opt*ColVgpr modes
        #     firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
        #     numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

        numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

        numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

        # TODO STREAM-K activation code

        if writer.db["PrintStoreRegisterDb"]:
            print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
            print("numSgprs=", numSgprs, "sgprPool.size()=", writer.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
            print(writer.sgprPool.state())
        module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
            (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
        #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        with writer.allocTmpSgpr(numSgprs, 2, tag="StreamKCommon_partialsWriteBatch_tmpSgprRes") as tmpSgprRes:
            tmpSgpr = tmpSgprRes.idx
            elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch

            codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None
            # TODO STREAM-K remove this?
            useCodeMulAlpha = kernel["MIArchVgpr"] and alpha and not (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1)
            if useCodeMulAlpha: # do not set codeAccVgprRead=None if GSU>1
                codeAccVgprRead = None

            for batchIdx in range(0, numBatches):
                elementStartIdx = batchIdx * numElementsPerBatch
                elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
                # elementVgprs can be large and should be perfectly tuned to the number of available
                # VGPRS.    We do not want to accidentally overflow and grow the pool here:

                module.add(self.partialsWriteBatch(writer, kernel, ss, batchIdx, alpha, beta, edge, gwvw, atomicW, \
                        elementsThisBatch, writer.vgprs.addrD, writer.vgprs.addrC, \
                        tmpVgpr, cvtVgprStruct, \
                        elementSgprs, tmpSgpr, codeAccVgprRead))
            # delay PreLoopVmcntCase code after globalWrite
            # if self.canOptimizePreLoopLWVmcnt:
            #     kStr += PreLoopVmcntCaseStr

            # Set flag
            module.add(memOrder.releaseFence(writer))
            module.add(SBarrier(comment="store all data before setting flag"))

            if kernel["StreamK"] == 4:
                # TODO modularize this section into abstract function
                module.add(self.calculatePartialIdx(tmpSgpr))
                module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(tmpSgpr), shiftHex=log2(4), comment="flag offset based on partial index"))
                module.add(SAddU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(256*8), comment="Offset flags to come after the work queues"))
            elif kernel["StreamK"] == 5:
                # SK5 hybrid: dispatch on StreamKHybridMode bit
                # (0 = static SK3 -> use StreamKIdx, 1 = dynamic SK4 -> use calculatePartialIdx).
                sk5FlagStatic = Label(writer.labels.getNameInc("SK5_PartialsFlagStatic"), "")
                sk5FlagDone   = Label(writer.labels.getNameInc("SK5_PartialsFlagDone"), "")
                module.add(SCmpEQU32(src0=sgpr("StreamKHybridMode"), src1=0,
                                     comment="SK5: mode bit == 0 -> SK3 (static) flag offset"))
                module.add(SCBranchSCC1(labelName=sk5FlagStatic.getLabelName(),
                                        comment="SK5: branch to static flag offset"))
                # SK4 (dynamic) flag offset
                module.add(self.calculatePartialIdx(tmpSgpr))
                module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(tmpSgpr), shiftHex=log2(4),
                                          comment="SK5/SK4: flag offset based on partial index"))
                module.add(SAddU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(256*8),
                                   comment="SK5/SK4: offset flags to come after the work queues"))
                module.add(SBranch(labelName=sk5FlagDone.getLabelName(),
                                   comment="SK5: skip static flag offset"))
                # SK3 (static) flag offset
                module.add(sk5FlagStatic)
                module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr("StreamKIdx"), shiftHex=log2(4),
                                          comment="SK5/SK3: flag offset based on CTA index"))
                module.add(sk5FlagDone)
            else:
                sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
                if writer.isStreamKConstantsToVgprEnabled(kernel):
                    module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sIdx), shiftHex=log2(4), comment="flag offset based on CTA index"))
                writer.releaseStreamKConstSgpr(sIdx)

            with writer.allocTmpSgpr(1, tag="StreamKCommon_setFlag_tmpSgprRes") as flagSgprRes:
                flagSgpr = flagSgprRes.idx
                skipFlagSet = Label(label=writer.labels.getNameInc("SK_SkipFlagSet"), comment="")
                module.add(VReadfirstlaneB32(dst=sgpr(flagSgpr), src=vgpr("Serial"), comment="Wave 0 updates flags"))
                module.add(SCmpEQU32(src0=sgpr(flagSgpr), src1=0, comment="Check for wave 0"))
                module.add(SCBranchSCC0(labelName=skipFlagSet.getLabelName(), comment="Skip flag set"))
                if writer.states.asmCaps["HasScalarStore"]:
                    module.add(SMovB32(dst=sgpr(flagSgpr), src=1, comment="flag data"))
                    module.add(SStoreB32(src=sgpr(flagSgpr), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=True), comment="set flag"))
                else:
                    module.add(VMovB32(dst=vgpr(tmpVgpr), src=1, comment="move 1 to tmpVgpr"))
                    module.add(self.setFlagValue(writer, src=vgpr(tmpVgpr), soffset=sgpr(tmpSgpr), comment="set flag"))
                module.add(skipFlagSet)
            module.add(SWaitCnt(kmcnt=0, comment="wait for flag")) # TODO just for testing

        if "Deferred" in endLabel.getLabelName():
            posLabel = writer.labels.getNameInc("PartialsDeferredReturnDir")
            with writer.allocTmpSgpr(3, tag="StreamKCommon_partialsDeferredReturn_tmpSgprInfo") as tmpSgprInfo:
                module.add(SLongBranch(endLabel, tmpSgprInfo, posLabel, comment="jump to end"))
        else:
            module.add(SBranch(labelName=endLabel.getLabelName(), comment="jump to end"))

        # Finish one write path, reset currPreLoopVmcntCase to Undefined
        # self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

        return module

    def setFlagValue(self, writer, src, soffset, comment=""):
        module = Module("Buffer Store Flag Value")
        memOrder = Component.StreamKMemoryOrdering.find(writer)
        tmpSgprBuffer = writer.sgprPool.checkOutAligned(4, 4, tag="StreamKCommon_setFlagValue_tmpSgprBuffer", preventOverflow=False)
        tmpVgprOff = writer.vgprPool.checkOut(1, "vaddr_off")
        module.add(VMovB32(dst=vgpr(tmpVgprOff), src=0, comment="zero vaddr offset"))
        module.add(SMovB64(dst=sgpr(tmpSgprBuffer, 2), src=sgpr("AddressFlags", 2)))
        module.add(SMovB32(dst=sgpr(tmpSgprBuffer+2), src="BufferOOB"))
        module.add(SMovB32(dst=sgpr(tmpSgprBuffer+3), src="Srd127_96"))
        module.add(self.shiftSrd(writer, tmpSgprBuffer))
        module.add(memOrder.preVolatileVmem(writer, comment="drain xnacks before volatile VMEM store"))
        module.add(BufferStoreB32(src=src, vaddr=vgpr(tmpVgprOff), saddr=sgpr(tmpSgprBuffer, 4), soffset=soffset,
                                  mubuf=memOrder.flagBufferMubuf(), comment=comment))
        module.add(SWaitCnt(vscnt=0, comment="wait for data store"))
        writer.vgprPool.checkIn(tmpVgprOff)
        writer.sgprPool.checkIn(tmpSgprBuffer)

        return module

    def getFlagValue(self, writer, dst, soffset, comment=""):
        """Buffer-load primitive for the StreamK flag.

        Used by `StreamKMemoryOrderingDevScopeFences.readFlag` to perform a
        VMEM-coherent flag load. Default arches read the flag via SMEM
        directly in `StreamKMemoryOrderingDefault.readFlag` and never call
        this helper.
        """
        module = Module("Buffer Load Flag Value")
        memOrder = Component.StreamKMemoryOrdering.find(writer)
        tmpSgprBuffer = writer.sgprPool.checkOutAligned(4, 4, tag="StreamKCommon_getFlagValue_tmpSgprBuffer", preventOverflow=False)
        tmpVgprOff = writer.vgprPool.checkOut(1, "vaddr_off")
        module.add(VMovB32(dst=vgpr(tmpVgprOff), src=0, comment="zero vaddr offset"))
        module.add(SMovB64(dst=sgpr(tmpSgprBuffer, 2), src=sgpr("AddressFlags", 2)))
        module.add(SMovB32(dst=sgpr(tmpSgprBuffer+2), src="BufferOOB"))
        module.add(SMovB32(dst=sgpr(tmpSgprBuffer+3), src="Srd127_96"))
        module.add(self.shiftSrd(writer, tmpSgprBuffer))
        module.add(memOrder.preVolatileVmem(writer, comment="drain xnacks before volatile VMEM load"))
        module.add(BufferLoadB32(dst=dst, vaddr=vgpr(tmpVgprOff), saddr=sgpr(tmpSgprBuffer, 4), soffset=soffset,
                                 mubuf=memOrder.flagBufferMubuf(),
                                 comment=comment))
        writer.vgprPool.checkIn(tmpVgprOff)
        writer.sgprPool.checkIn(tmpSgprBuffer)

        return module

    def partialsWriteBatch(self, writer, kernel, ss, batchIdx, applyAlpha, beta, edge, gwvw, atomicW, \
            batchElements, addrD, addrC, \
            tmpVgpr, cvtVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead):
        module = Module("StreamK Common partialsWriteBatch")

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Partials Write%s%s%s Batch #%u (d1,d0,vc1,vc0) =\n     " \
            % (" Alpha" if applyAlpha else "", " Beta" if beta else "", " Edge" if edge else "", batchIdx)
        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = (not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=isOptNLL, isWorkspace=True)
        ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, factorDim=0, isWorkspace=True)

        storesIssued = 0
        tmpS01 = tmpSgpr # scratch sgprs

        ########################################
        # calculate addr and masks
        module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(writer.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(writer.getBomb()) # should not get here

        ## create code Module to push mov vgpr,acc instructions
        # if kernel["StoreCInUnroll"] and not edge:
        #     accVgprRead = Code.Module("movaccVgpr")
        #     self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

        ########################################
        # AccVgpr read
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprRead is not None and kernel["LocalSplitU"] == 1:
            # CompactLoopStore makes accVgprRead use v_movrelsd_2_b32 (src VGPR
            # index offset by M0) so the same body can cover the full thread
            # tile inside a CLS loop. The StreamK partials-write path runs
            # OUTSIDE that CLS loop, but reuses the same precomputed
            # writer.codes.accVgprRead module -- M0 still holds whatever the
            # last CLS loop left in it (sgprWorkGroup2 + step), so the source
            # VGPR index gets a random offset and the partials wrote into D
            # come out scrambled. Force M0=0 here so v_movrelsd_2_b32 behaves
            # like the plain v_mov_b32 the non-CLS path used to emit.
            if kernel.get("CompactLoopStore", False):
                module.add(SMovB32(dst=mgpr(0), src=0,
                    comment="reset M0 for v_movrelsd_2_b32 outside CLS loop"))
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        startVgprValuOffset = 0 if kernel.get("UseSubtileImpl") else writer.states.c.startVgprValu
                        module.add(replaceHolder(codeAccVgprRead.popFirstItem(), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - startVgprValuOffset))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        ########################################
        # Not Atomic
        ########################################
        # else:
        # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
        for elementIdx in range(len(batchElements)):
            for vi in range(gwvw):
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                # TODO STREAM-K is start value needed now?
                # TODO KUPO!!!!!!!!!!!!!!!!
                # newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
                if kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
                    if writer.db["ForceExpectedValue"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=writer.db["ValueCExpectedValue"], comment="force expected value"))
                        # module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdxV), src=self.debugConfig["ValueCExpectedValue"], comment="force expected value" ))
                    if writer.db["ForceVSerial"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("Serial"), comment="force expected value to serial"))
                        # module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdxV), src=vgpr("Serial"), comment="force expected value to serial" ))
                    if writer.db["CheckValueC"]:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=writer.db["ValueCExpectedValue"], comment="Move expected value"))
                        module.add(writer.getCmpAssert(writer.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        module.addComment1("apply mask, calc new C and issue writes")

        # if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #     vgprBf16Temp = tmpCVTVgpr
        #     vgprBf16Mask = vgprBf16Temp + 1
        #     vgprFp32Nan = vgprBf16Temp + 2
        #     vgprBf16Inc = vgprBf16Temp + 3
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" )
        #     kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" )
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" )
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8_fnuz() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43700000", comment="Fp8 Max value 240 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3700000", comment="Fp8 Min value -240 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43E00000", comment="Fp8 Max value 448 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3E00000", comment="Fp8 Min value -448 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isAnyBFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Max), "0x47600000", comment="BF8 Max value 57344 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Min), "0xc7600000", comment="BF8 Min value -57344 as float32" ))

        if kernel["EnableMatrixInstruction"]:
            WaveNum = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1] * kernel["WorkGroup"][2]
        else:
            WaveNum = kernel["NumThreads"] // kernel["WavefrontSize"]

        storeCode = Module("Partials GroupLoadStore")
        for elementIdx in range(len(batchElements)):
            element = batchElements[elementIdx]
            addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
            addr = addrCalc.addrDVgpr
            # For UseSubtileImpl, vgprValuC is remapped; add the base offset so the
            # WS store reads from the correct accumulator VGPRs.  For the regular path
            # (non-subtile), startVgprValu is already accounted for by the vgprValuC
            # assembler macro, so no offset is needed (matches rebase behaviour).
            if kernel.get("UseSubtileImpl"):
                sumIdx = ss.elementSumIdx[elementIdx] + writer.states.c.startVgprValu
            else:
                sumIdx = ss.elementSumIdx[elementIdx]
            storeWidth = kernel["StoreVectorWidth"]
            # storeWidth = 2
            if batchIdx == 0 and elementIdx == 0:
                tmpSgprRes = ContinuousRegister(idx=tmpS01, size=1)
                module.add(vectorStaticMultiply(vgpr(addr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, tmpSgprRes))
                # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
                module.add(SMovB32(dst=sgpr(tmpS01), src=0, comment="Init sgpr offset"))
            else:
                increment = (kernel["WavefrontSize"] * WaveNum) * storeWidth * writer.states.bpeCinternal
                # module.addComment1("WavefrontSize={}, WaveNum={}, storeWidth={}, bpeC={}".format(kernel["WavefrontSize"], WaveNum, storeWidth, writer.states.bpeCinternal))
                module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=increment, comment="Inc sgpr offset"))

            # TODO StreamK need this packing code???
            # if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"] and kernel["ProblemType"]["DestDataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"]):
            #     for vi in range(0, gwvw):
            #         sumIdxV = ss.elementSumIdx[elementIdx] + vi
            #         if vi%2 == 1:
            #             d = ss.elementSumIdx[elementIdx] + vi//2
            #             kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

            # if not kernel["StoreRemapVectorWidth"]:
            tmpStoreCode = writer.addStore(kernel, ss, 'WS', addrCalc, sumIdx, tmpS01, edge, wsOffset=sgpr(tmpS01))
            if kernel["GroupLoadStore"]:
                storeCode.add(tmpStoreCode)
            else:
                module.add(tmpStoreCode)
            storesIssued += 1

        module.add(storeCode)

        # return registers to pool:
        lastData = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastData:
                    writer.vgprPool.checkIn(data)
                lastData = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        if writer.states.serializedStore:
            module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

        # Update the store cnt to preLoopVmcntDict for Case2/3
        # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
        # TODO STREAM-K Need this?
        # if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #     if not self.archCaps["SeparateVscnt"]:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

        return module

    def fixupStep(self, writer, kernel, vectorWidths, elements, edges, tmpVgpr, cvtVgprStruct, sPartialIdx):
        module = Module("StreamK Common fixupStep")

        fixupLabels = {}
        for edge in edges:
            fixupLabels[edge] = Label(writer.labels.getNameInc("Fixup_E%u" % ( 1 if edge else 0)), comment="")

        # branch if Edge0 or Edge1
        if False in edges and True in edges:
            module.add(writer.checkIsEdge(kernel, tmpSgprInfo, fixupLabels[True], fixupLabels[True]))

        # by now we either jumped to E1 or stayed at E0
        for edge in edges:
            # write label for batch case
            module.add(fixupLabels[edge])

            # PreLoopVmcntCaseStr = ""
            # # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
            # if self.canOptimizePreLoopLWVmcnt:
            #     if edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
            #     else:
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
            #     PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
            #         "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
            #     # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
            #     # OrdNLL_B1_Store is excluded
            #     if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
            #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

            edgeI = edge
            #edgeI = True    # set to True to disable vector stores
            gwvw = vectorWidths[edgeI]

            ########################################
            # Calculate Vgprs for Write Batching
            ########################################

            vectorDataTypes = VectorDataTypes()
            ss = StoreState(writer, kernel, gwvw, edge, True, False, elements[edgeI], vectorDataTypes, dim=0, isWorkspace=True)

            # how many vgprs are needed for zero elements
            # 2 for addressC in vgpr for addition - already checked out
            # 2 for coord0,1 of thread - already checked out
            # 2 for tmp - already checked out

            # 5 = how many vgprs are needed per element (flat)
            #    - 2 for addr
            #    - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
            #    - if beta gwvw*rpe for new value
            #    - if atomic 2*rpe for old and cmp values

            # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(self.ss.cfg.numVgprsPerAddr, self.ss.cfg.numVgprsPerDataPerVI, self.ss.cfg.numVgprPerValuC))
            # numVgprsPerElement = self.ss.cfg.numVgprPerValuC*gwvw + self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

            # if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
            #     numVgprsPerElement += self.ss.cfg.numVgprsPerAddr

            #print self.vgprPool.state()
            # Use VGPR up to next occupancy threshold:
            maxVgprs, _ = writer.getMaxRegsForOccupancy(kernel["NumThreads"], writer.vgprPool.size(), writer.sgprPool.size(), \
                writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
            if writer.states.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
                # len(elements[edgeI])
                # tl = []
                # for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
                #     tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
                # for t in tl:
                #     self.vgprPool.checkIn(t)
                writer.vgprPool.growPool(writer.vgprPool.size()-writer.vgprPool.available(), maxVgprs, 1, \
                    "grow-pool up to next occupancy for GlobalWrite")
            # align = 1
            # # align adjustment
            # if self.ss.cfg.numVgprsPerAddr > 1:
            #     align = max(align, self.ss.cfg.numVgprsPerAddr)
            # if self.ss.cfg.numVgprPerValuC*gwvw > 1:
            #     align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
            # if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
            #     align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
            numVgprAvailable = writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

            # Grow the register pool if needed - we need enough regs for at least one element
            # Unfortunate since this means the write logic is setting the VGPR requirement
            # for the entire kernel but at least we have a functional kernel.
            # Before growing the pool, see if we can shrink the write vector width instead?
            # TODO : the vgprSerial is needed for-ever and if we grow here will split the
            # range of the tmps.    Maybe want to move vgprSerial to first vgpr?

            # TODO: Minimum elems for StoreRemap
            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            minElements = 1
            if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
                minElements = 2
            elif kernel["ProblemType"]["DataType"].is8bitFloat():
                minElements = 4
            minNeeded = minElements * ss.numVgprsPerElement

            shrinkDb = 0
            if shrinkDb:
                print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)

            if numVgprAvailable < minNeeded:
                gwvwOrig = gwvw
                currentOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size(), \
                        writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
                futureOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size() - numVgprAvailable + minNeeded, \
                        writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)

                if shrinkDb:
                    print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                        % (currentOccupancy, futureOccupancy, writer.vgprPool.size(), \
                        numVgprAvailable, minElements*ss.numVgprsPerElement))
                if futureOccupancy > currentOccupancy:
                    if shrinkDb:
                        print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                            (writer.states.kernelName))
                        print("     numVgprAvailable=", numVgprAvailable, \
                            "numVgprsPerElement=", ss.numVgprsPerElement, \
                            "gwvw=", gwvw)
                elif gwvw != gwvwOrig:
                    ss.gwvw = gwvw # make both representations consistent
                    if shrinkDb:
                        print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                            % (writer.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

                if numVgprAvailable < minElements*ss.numVgprsPerElement:
                    print2("info: growing pool += %d * %d for GlobalWrite\n" \
                        % (minElements,ss.numVgprsPerElement))
                    print2(writer.vgprPool.state())
                    # tl = []
                    # for i in range(0,minElements):
                    #     tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
                    # for t in tl:
                    #     self.vgprPool.checkIn(t)
                    writer.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
                        "grow-pool for GlobalWrite")
                    numVgprAvailable = writer.vgprPool.available()
                    print2(writer.vgprPool.state())

            # print("NumVgprAvailable", numVgprAvailable)
            if ss.numVgprsPerElement:
                numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
            else:
                numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

            # assert(self.numVgprValuC % gwvw == 0) # sanity check

            numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

            if shrinkDb:
                print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
                        "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
            if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
                numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
                # only do an even number of halves - since these share hi/lo pieces of some registers?
                if numElementsPerBatch > 1:
                    numElementsPerBatch = int(numElementsPerBatch/2)*2
                elif not kernel["EnableMatrixInstruction"]:
                    # (excluding MFMA+LSU case. It can work without an issue)
                    # The globalWriteBatch routine below can't handle odd elements per batch
                    # and 0 elements per batch is illegal.
                    # so if we don't have *GPR resources to handle a larger batch then need
                    # to mark overflowedResources rather than generate a kernel that won't work.
                    # It might be possible to fix globalWriteBatch to handle this case but these
                    # are likely to be low-performing so likely not worth optimizing.
                    if shrinkDb:
                        print("WARNING: half requires at least two elements per batch")
                    self.overflowedResources = 3
            #elif kernel["ProblemType"]["DataType"].is8bitFloat():
            #    if numElementsPerBatch > 1:
            #        numElementsPerBatch = int(numElementsPerBatch/4)*4

            assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%writer.states.kernelName

            # if no atomics and no edge, then write whole vectors
            # ERROR commented out in globalWriteELements, causes numVectorsPerBatch to not be int
            # if not edge: # not atomic and
            #    numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
            #    #print "    NumVectorsPerBatch", numVectorsPerBatch
            #    numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
            numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

            numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

            if writer.db["PrintStoreRegisterDb"]:
                print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
                print ("numSgprs=", numSgprs, "sgprPool.size()=", writer.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                    "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
                print(writer.sgprPool.state())
            module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
                    (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
            #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
            # so if we don't have *GPR resources to handle a larger batch then need
            # to mark overflowedResources rather than generate a kernel that won't work.

            with writer.allocTmpSgpr(numSgprs, 2, tag="StreamKCommon_fixupStep_tmpSgprRes") as tmpSgprRes:
                tmpSgpr = tmpSgprRes.idx
                elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch

                codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None
                # codeAccVgprRead = deepcopy(writer.codes.codeAccVgprRead) if writer.states.serializedStore else None
                codeAccVgprWrite = deepcopy(writer.codes.accVgprWrite) if writer.states.serializedStore else None

                module.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sPartialIdx), tmpSgpr))

                for batchIdx in range(0, numBatches):
                    elementStartIdx = batchIdx * numElementsPerBatch
                    elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                    elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                    #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
                    # elementVgprs can be large and should be perfectly tuned to the number of available
                    # VGPRS.    We do not want to accidentally overflow and grow the pool here:

                    module.add(self.fixupBatch(writer, kernel, ss, batchIdx, edge, gwvw, \
                            elementsThisBatch, writer.vgprs.addrD, writer.vgprs.addrC, \
                            tmpVgpr, cvtVgprStruct, \
                            elementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite,
                            elementStartIdx))
                # delay PreLoopVmcntCase code after globalWrite
                # if self.canOptimizePreLoopLWVmcnt:
                #     kStr += PreLoopVmcntCaseStr

            # Finish one write path, reset currPreLoopVmcntCase to Undefined
            # self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

            # kStr += inst("s_branch", skStoreLabel, "jump to store")

        return module

    def fixupBatch(self, writer, kernel, ss, batchIdx, edge, gwvw, \
            batchElements, addrD, addrC, \
            tmpVgpr, cvtVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite,
            elementStartIdx=0):
        module = Module("StreamK Common fixupBatch")

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Fixup%s Batch #%u (d1,d0,vc1,vc0) =\n     " \
            % (" Edge" if edge else "", batchIdx)
        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)
        # print(self.kernelName)
        # print(commentStr)

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = True #(not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, preventOverflow=preventOverflow, isWorkspace=True)
        ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, False, 0, True, elementStartIdx)

        loadsIssued = 0
        storesIssued = 0
        tmpS01 = tmpSgpr # scratch sgprs

        # laneSGPRC = writer.states.laneSGPRCount
        # always use gwvw for buffer load C for atomic_cmpswap
        # bpm = self.bpeCexternal * atomicW
        # bpm = self.bpeCexternal * gwvw
        # vgprLoadDW = 1*(bpm//4)
        # atomic oparation width. 1 for b32, 2 for b64
        # atomicOpW = (atomicW * self.bpeCexternal) // 4
        # if atomicOpW > 2:
        #     # should not exceeding 2.
        #     atomicOpW = 2

        ########################################
        # calculate addr and masks
        module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(writer.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(writer.getBomb()) # should not get here

        # atomicAddC = kernel["AtomicAddC"] and not edge

        ## create code Module to push mov vgpr,acc instructions
        # if kernel["StoreCInUnroll"] and not edge:
        #     accVgprRead = Code.Module("movaccVgpr")
        #     self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

        if kernel["EnableMatrixInstruction"]:
            WaveNum = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1] * kernel["WorkGroup"][2]
        else:
            WaveNum = kernel["NumThreads"] // kernel["WavefrontSize"]

        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
            # addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
            addrCalc = ss.elementAddr[elementIdx]
            data = ss.elementData[elementIdx]
            # mask = ss.elementMask[elementIdx]
            # sumIdx = ss.elementSumIdx[elementIdx]
            # d1 = element[0]
            # d0 = element[1]
            # vc1 = element[2]
            vc0 = element[3]

            storeWidth = kernel["StoreVectorWidth"]
            # storeWidth = 2
            if batchIdx == 0 and elementIdx == 0:
                tmpS01Res = ContinuousRegister(idx=tmpS01, size=1)
                module.add(vectorStaticMultiply(vgpr(addrCVgpr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, tmpS01Res))
                # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
                module.add(SMovB32(dst=sgpr(tmpS01), src=0, comment="Init sgpr offset"))
            else:
                increment = (kernel["WavefrontSize"] * WaveNum) * storeWidth * writer.states.bpeCinternal
                # module.addComment1("WavefrontSize={}, WaveNum={}, storeWidth={}, bpeC={}".format(kernel["WavefrontSize"], WaveNum, storeWidth, writer.states.bpeCinternal))
                module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=increment, comment="Inc sgpr offset"))

            module.add(writer.readInput(kernel, ss, 'WS', kernel["ProblemType"]["ComputeDataType"], addrCalc, vc0, data, gwvw, addrCVgpr, sgpr(tmpS01)))
            loadsIssued += 1

        ########################################
        # AccVgpr read
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprRead is not None and kernel["LocalSplitU"] == 1:
            # Same M0 reset as partialsWriteBatch: the SK fixup path runs after
            # the CLS loop has left M0 = sgprWorkGroup2+step. accVgprRead is the
            # precomputed v_movrelsd_2_b32 module (when CompactLoopStore=True),
            # which would pick up that stale M0 and reorder accumulator vregs.
            if kernel.get("CompactLoopStore", False):
                module.add(SMovB32(dst=mgpr(0), src=0,
                    comment="reset M0 for v_movrelsd_2_b32 outside CLS loop"))
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        module.add(replaceHolder(codeAccVgprRead.popFirstItem(), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - writer.states.c.startVgprValu))
                        # tempStr = str(codeAccVgprRead.popFirstItem())
                        # kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        ########################################
        # Not Atomic
        ########################################
        # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
        interleaveStoreVmcnt = writer.states.interleaveStoreVmcnt and not edge
        for elementIdx in range(0, len(batchElements)):
            for vi in range(0, gwvw):
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
                if kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
                    if writer.db["ForceExpectedValue"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=writer.db["ValueCExpectedValue"], comment="force expected value"))
                    if writer.db["ForceVSerial"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("Serial"), comment="force expected value to serial"))
                    if writer.db["CheckValueC"]:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=writer.db["ValueCExpectedValue"], comment="Move expected value"))
                        module.add(writer.getCmpAssert(writer.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        ########################################
        # wait for batched load
        if not interleaveStoreVmcnt: # beta and
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="wait C"))

            # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
            # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
            # And this only occurs when beta=true, so case must not be 2 or 3
            # assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
            #     "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

        module.addComment1("apply mask, calc new C and issue writes")

        # if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #     vgprBf16Temp = tmpCVTVgpr
        #     vgprBf16Mask = vgprBf16Temp + 1
        #     vgprFp32Nan = vgprBf16Temp + 2
        #     vgprBf16Inc = vgprBf16Temp + 3
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" )
        #     kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" )
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" )
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43E00000", comment="OCP Fp8 Max value 448 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3E00000", comment="OCP Fp8 Min value -448 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8_fnuz() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43700000", comment="Fp8 Max value 240 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3700000", comment="Fp8 Min value -240 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isAnyBFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8NanInf), "0x207", comment="Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Max), "0x47600000", comment="BF8 Max value 57344 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Min), "0xc7600000", comment="BF8 Min value -57344 as float32" ))

        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            addr = ss.elementAddr[elementIdx].addrDVgpr
            mask = ss.elementMask[elementIdx]
            addrCalc = ss.elementAddr[elementIdx]
            # d1 = element[0]
            # d0 = element[1]
            # vc1 = element[2]
            vc0 = element[3]
            sumIdx = ss.elementSumIdx[elementIdx]

            # apply in-bounds exec mask
            if edge and not kernel["BufferStore"]:
                module.add(writer.getEdgeMovInstType()(EXEC(), sgpr(mask, writer.states.laneSGPRC), "sgprs -> exec"))
                # kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )

            # if beta:
            # if GWVW=1 the half path still assumes we have
            # at least two stores so does some combining across VI -
            # for example assuming we can have two elements and can use pk_mul
            # here:
            if interleaveStoreVmcnt: # beta and
                vlcnt = loadsIssued - elementIdx - 1
                # we are waiting for loads to finish, so no need to wait for stores if counted separately
                if writer.states.asmCaps["SeparateVscnt"] or writer.states.asmCaps["SeparateVMcnt"]:
                    vscnt = -1
                    vmComment = "{} = {} - {} - 1".format(vlcnt, loadsIssued, elementIdx)
                else:
                    waitStoreCnt = storesIssued if not kernel["GroupLoadStore"] else 0
                    vscnt = waitStoreCnt
                    vmComment = "{} = {} - {} + {} - 1".format(vlcnt, loadsIssued, elementIdx, waitStoreCnt)

                #print "wmvcnt=", vmcnt
                module.addSpaceLine()
                # if not atomicAddC:
                module.add(SWaitCnt(vlcnt=vlcnt, vscnt=vscnt, comment="wait C (interleaved) {}".format(vmComment)))

                # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
                # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
                # And this only occurs when beta=true, so case must not be 2 or 3
                # assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

            for vi in range(0, gwvw):
                dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                if kernel["ProblemType"]["ComputeDataType"].isHalf():
                    if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        if writer.states.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"]:
                            dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
                            # if (vi % 2) == 0:
                            #         kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                            #                 "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                            # else:
                            if (vi % 2) != 0:
                                module.add(VLShiftRightB32(dst=vgpr(dataV), shiftHex=16, src=vgpr(dataV), \
                                    comment="shift 16bit to get next half of packed ValueC"))
                            # dataV+0 = new c = old c*beta + rC
                            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(sumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(sumIdxV)), \
                                comment="sum*alpha + C*beta"))
                        elif sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
                            newSumIdxV = sumIdxV // 2 - writer.states.c.startVgprValu
                            # dataV+0 = new c = old c*beta
                            # kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                            #         "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                            # dataV+0 = new c = old c*beta + rC
                            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(newSumIdxV)), \
                                comment="sum*alpha + C*beta"))
                        else:
                            pass # add will have been done previously
                    else: # HPA
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        # dataV+0 = new c = old c*beta + rC
                        # src0 = beta = f32 = opsel 00
                        # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                        # src2 = sumIdxV = f32 = opsel 00
                        dataCExternal = ss.elementData[elementIdx] + vi//2
                        hi16 = (vi + gwvw*vc0) % 2
                        # TODO try to replace with add? need opsel for f16 src
                        # kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
                        # module.add(writer.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=sgpr("Beta"), \
                        #     src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
                        #     vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
                        #     comment="//C*=beta"))
                        module.add(writer.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=1, \
                            src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
                            vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
                            comment="//C*=beta"))
                        # kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), 1, \
                        #         vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
                        #         "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                        #         "//C*=beta")

                elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
                    if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        # dataV+0 = new c = old c*beta + rC
                        # src0 = beta = f32 = opsel 00
                        # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                        # src2 = sumIdxV = f32 = opsel 00
                        dataCExternal = ss.elementData[elementIdx] + vi//2
                        # if (vi%2) == 1:
                        #     kStr += inst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(vgprBf16Mask), "convert bf16 to fp32")
                        # else:
                        #     kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
                        module.add(VCvtBF16toFP32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vgprMask=vgpr(cvtVgprStruct.vgprBf16Mask), vi=(vi)))
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        module.add(VAddF32(dst=vgpr("ValuC+%u"%sumIdxV), src0=vgpr("ValuC+%u"%sumIdxV), src1=vgpr(tmpVgpr), comment="accum partials"))

                elif kernel["ProblemType"]["ComputeDataType"].isSingle():
                    if kernel["ProblemType"]["DataType"].isInt8():
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=vgpr("ValuC+%u"%newSumIdxV), comment="accum partials"))
                    else:
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        module.add(VAddF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr("ValuC+%u"%newSumIdxV), src1=vgpr(dataV+0), comment="accum partials"))

                elif kernel["ProblemType"]["ComputeDataType"].isInt32():
                    newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                    # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
                    # v_mad_i32_i24
                    module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=vgpr("ValuC+%u"%newSumIdxV), comment="accum partials"))

                elif kernel["ProblemType"]["ComputeDataType"].isDouble():
                    newSumIdxV = sumIdxV * 2 - writer.states.c.startVgprValu
                    # dataV+0 = new c = old c*beta
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV),2), src0=vgpr("ValuC+%u"%(newSumIdxV),2), src1=vgpr(dataV+0,2), comment="accum partials"))

                # single precision complex
                elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
                    newSumIdxV = sumIdxV * 2 - writer.states.c.startVgprValu
                    module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr("ValuC+%u"%(newSumIdxV)), src1=vgpr(dataV+0), comment="accum partials real"))
                    module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr("ValuC+%u"%(newSumIdxV+1)), src1=vgpr(dataV+1), comment="accum partials imag"))

                # double precision complex
                elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
                    newSumIdxV = sumIdxV * 4 - writer.states.c.startVgprValu
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV+0),2), src0=vgpr("ValuC+%u"%(newSumIdxV+0),2), src1=vgpr(dataV+0,2), comment="accum partials real"))
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV+2),2), src0=vgpr("ValuC+%u"%(newSumIdxV+2),2), src1=vgpr(dataV+2,2), comment="accum partials imag"))

        ########################################
        # AccVgpr write
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprWrite is not None and kernel["LocalSplitU"] == 1:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        module.add(replaceHolder(codeAccVgprWrite.popFirstItem(), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - writer.states.c.startVgprValu))
                        # tempStr = str(codeAccVgprWrite.popFirstItem())
                        # kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        # if self.db["CheckStoreC"]>=0:
        #     useBuffer = kernel["BufferStore"]
        #     # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
        #     # Note - TODO- CheckStoreC also won't work for StoreRemap
        #     kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        #     if self.archCaps["SeparateVscnt"]:
        #         kStr += inst("s_waitcnt_vscnt", -2, "0", "writes")
        #     for elementIdx in range(0, len(batchElements)):
        #         addr = ss.elementAddr[elementIdx].addrDVgpr
        #         sumIdx = ss.elementSumIdx[elementIdx]

        #         bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
        #         if kernel["BufferStore"]:
        #             addr0 = vgpr(addr)
        #             addr1 = sgpr("SrdC", 4)
        #         else:
        #             addr0 = vgpr(addr,2)
        #             addr1 = ""

        #         if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():
        #             if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #                 kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
        #                                     addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=sumIdx%2).toStr()
        #             else:
        #                 kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
        #                                     addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=0).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex() :
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #     kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        #     if self.archCaps["SeparateVscnt"]:
        #         kStr += inst("s_waitcnt_vscnt", -2, "0", "writes")

        #     # Add checks for expected values:
        #     kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
        #     for elementIdx in range(0, len(batchElements)):
        #         sumIdx = ss.elementSumIdx[elementIdx]
        #         # Need to fix for other types:
        #         assert (kernel["ProblemType"]["DestDataType"].isSingle() or kernel["ProblemType"]["DestDataType"].isInt32())
        #         kStr += self.assert_eq(vgpr(sumIdx), sgpr(tmpS01))


        if edge and (not kernel["BufferStore"]): # atomic or
            # subsequent batch must start with full exec mask
            # BufferStore doesn't need exec since it used buffer range checking when
            # possible
            module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

        if writer.db["ConservativeWaitCnt"] & 0x40:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        ########################################
        # End Not Atomic
        ########################################

        # return registers to pool:
        lastData = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastData:
                    writer.vgprPool.checkIn(data)
                lastData = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        if writer.states.serializedStore:
            module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

        # Update the store cnt to preLoopVmcntDict for Case2/3
        # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
        # if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #     if not self.archCaps["SeparateVscnt"]:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

        return module
    
    def stridedBatchOrGeneralBatch(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK stridedBatchOrGeneralBatch")
        if kernel["ProblemType"]["SupportUserArgs"]:
            module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=3, comment="ArgType == 3 for General Batched GEMM"))
            module.add(SCBranchSCC0(labelName=stridedBatchedGemmLoad.getLabelName())) 
            # Check for StreamK Kernel when ArgType == 3 (General Batched GEMM)
            # AddressFlags == 0, then parallel reduction in StreamK and SrdC/D is not dereferenced as pointer array
            # AddressFlags != 0, then not parallel reduction in StreamK and SrdC/D is dereferenced as pointer array                   
            module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
            module.add(SCBranchSCC0(labelName=generalBatchedGemmLoad.getLabelName()))
        return module

    @abc.abstractmethod
    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        pass

    @abc.abstractmethod
    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        pass

    @abc.abstractmethod
    def kernelEnd(self, writer, kernel):
        pass

class StreamKOff(StreamK):
    kernel = {"StreamK": 0}

    def preLoop(self, writer, kernel):
        module = Module("StreamK Off openLoop")
        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Off graWorkGroup")
        return module

    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tPA, tPB, skipLroReset=False):
        module = Module("StreamK Off prefetchAcrossPersistentSetupNextTile")
        return module

    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("StreamK Off computeLoadSrd")
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK Off computeStoreSrdStart")
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Off graAddresses")

        tc = tP["tensorChar"]
        module.add(VMovB32(dst=vgpr(vTmp+0), src=sgpr("Address%s+0" % tc)))
        module.add(VMovB32(dst=vgpr(vTmp+1), src=sgpr("Address%s+1" % tc)))

        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Off declareStaggerParms")
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Off tailLoopNumIter")
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Off calculateLoopNumIter")

        quotient = loopCounterName
        dividend = "SizesSum+%u" % loopIdx #sumSize = self.sumSize(kernel, loopIdx)
        divisor = kernel["DepthU"]

        module.add(scalarStaticDivideAndRemainder(qReg=quotient, rReg=-1, dReg=dividend, divisor=divisor, tmpSgprRes=tmpSgprInfo, doRemainder=False))
        if writer.states.tailloopInNll:
            maxUnit = writer.states.tailloopInNllmaxUnit
            sgprSizesSum = sgpr("SizesSum+%u" % writer.states.unrollIdx)
            depthU = kernel["DepthU"]
            tmpSgpr = tmpSgprInfo.idx
            assert (depthU > 1 and (depthU & (depthU - 1) == 0)), "DepthU should be power of 2 with tailloopInNll"
            # We need to increment loop counter by 1 if we use tailloopInNll code
            # We do not use tailloopInNll code if
            #   summation is multiple of depthU, or
            #   maxUnit > 1 and summation is not multiple of maxUnit
            module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgprSizesSum, src1=depthU-1, \
                               comment="tailloopInNll: check if summation is multiple of DepthU(%u)"%depthU))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=1, src1=0, \
                                   comment="tailloopInNll: select loopcounter increment value (0 if summation is multiple of DepthU else 1"))
            if maxUnit > 1:
                # maxUnit > 1 case, check if summation is multiple of maxUnit or not
                # if summation is multiple of maxUnit or not, tailloopInNll is used and increment loopCounter
                module.add(SAndB32(dst=sgpr(tmpSgpr+1), src0=sgprSizesSum, src1=maxUnit-1, \
                                   comment="if summation is multiple of %u, use tailloopInNll"%maxUnit))
                module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=0, src1=sgpr(tmpSgpr), \
                                       comment="select loopcounter (0 if summation is multiple of %u)"%maxUnit))
            if kernel["GlobalSplitU"] != 0:
                # skip tailloopInNll code if GSU>1
                module.add(SAndB32(dst=sgpr(tmpSgpr+1), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SCmpGtU32(src0=sgpr(tmpSgpr+1), src1=1, comment="GSU > 1 ?"))
                module.add(SCMovB32(dst=sgpr(tmpSgpr), src=0, comment="do not increment loopcounter if GSU > 1"))
            module.add(SAddU32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), \
                               src1=sgpr(tmpSgpr), comment="increment loopcounter for tailloopInNll" ))


        return module

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Off storeBranches")
        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Off writePartials")
        return module

    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK Off initializeSrdAddressFlagsCheck")
        module.add(SBranch(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="General Batched GEMM, Srd initialized to 0"))
        return module

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK Off routeToGeneralBatchedOrStridedBatched")
        return module

    def kernelEnd(self, writer, kernel):
        module = Module("StreamK Off kernelEnd")
        return module

class StreamKBasic(StreamK):
    kernel = {"StreamK": 1}

    def preLoop(self, writer, kernel):
        module = Module("StreamK Basic openLoop")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Workaround for gfx12
        if writer.states.archCaps["WorkGroupIdFromTTM"]:
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        if skConstsInVgprs:
            module.add(VMovB32(dst=vgpr(self._skv(writer, "StreamKIdx")), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index to VGPR"))
        else:
            module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index"))
        # Basic SK
        sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
        sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sIpw), comment="StreamK starting iteration"))
        writer.releaseStreamKConstSgpr(sIdx)
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sIpw), comment="StreamK ending iteration"))
        writer.releaseStreamKConstSgpr(sIpw)
        with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.computeTotalIters(writer, kernel, sTmp))
            module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp), comment="Cap ending iter at total iters"))
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Basic graWorkGroup")

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=not kernel.get("UseSubtileImpl", False))

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        # Increment StreamK iteration
        module.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+2), comment="Increment StreamK Iteration"))

        module.add(self.skIndexToWG(writer, kernel, sTmp))

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("StreamK Basic computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tP, sTmp))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK Basic computeStoreSrdStart")
        # Parallel reduction currently only available for two-tile kernel
        # module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Basic graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Basic declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Basic tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Basic calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Basic storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct))
        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Basic writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
        return module

    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK Basic initializeSrdAddressFlagsCheck")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="Parallel Reduction for General Batched GEMM, Srd initialized to workspace"))
        return module

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK Basic routeToGeneralBatchedOrStridedBatched")
        module.add(self.stridedBatchOrGeneralBatch(stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel))
        return module

    def kernelEnd(self, writer, kernel):
        module = Module("StreamK Basic kernelEnd")
        return module

class StreamKTwoTileOriginal(StreamK):
    kernel = {"StreamK": 2}
    emitsParallelReductionSgprAliases = True

    def preLoop(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal openLoop")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Workaround for gfx12
        if writer.states.archCaps["WorkGroupIdFromTTM"]:
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        if skConstsInVgprs:
            module.add(VMovB32(dst=vgpr(self._skv(writer, "StreamKIdx")), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index to VGPR"))
        else:
            module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index"))
        # Two-tile SK (SK first)
        # iter count after all extra iters have been distributed
        sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
        sIter = writer.sgprPool.checkOut(2, "SKIter")
        module.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIter)) # sIter used as tmp
        sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
        sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sIpw), comment="StreamK starting iteration (case: after extra iters)"))
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(sSkExtraIters), comment="Add extra iters"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sIpw), comment="StreamK ending iteration (case: after extra iters)"))
        # iter count before all extra iters have been distributed
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIpw), src1=1, comment="Spread out extra iterations"))
        writer.releaseStreamKConstSgpr(sIpw)
        module.add(SMulI32(dst=sgpr(sIter), src0=sgpr(sIdx), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
        # select correct start/end iteration index
        module.add(SCmpLtU32(src0=sgpr(sIdx), src1=sgpr(sSkExtraIters), comment="Check if lane gets an extra iteration"))
        writer.releaseStreamKConstSgpr(sIdx)
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
        module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        writer.sgprPool.checkIn(sSkExtraIters)
        writer.sgprPool.checkIn(sIter)
        # clamp to end of sk iterations
        sTmp = writer.sgprPool.checkOut(1, "TotalSKIters")
        sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(sSkt), src1=sgpr(sIpt), comment="Total SK iters"))
        writer.releaseStreamKConstSgpr(sSkt)
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp), comment="Cap ending iter at total SK iters"))
        # check if this WG has no work to do
        module.add(self.computeTotalIters(writer, kernel, sTmp))
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTmp), comment="Make sure there's work to do"))
        writer.sgprPool.checkIn(sTmp)
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK TwoTileOriginal graWorkGroup")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=not kernel.get("UseSubtileImpl", False))

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        # local end (DP tile)
        # TODO This line isnt needed?
        module.add(SSubU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+2), src1=sgpr(sTmp+1), comment="Local iteration end (DP tile)"))
        # select correct local end
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Check if in SK or DP section"))
        module.add(SCSelectB32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"), src1=sgpr(sTmp+3), comment="Apply SK or DP end iteration"))

        # Increment StreamK iteration
        # If moving from SK to DP, next iteration is first DP
        # sTmp = offset to first DP tile
        sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr(sSkt), src1=sgpr(sIpt), comment="Offset to first DP tile"))
        writer.releaseStreamKConstSgpr(sSkt)
        sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sIdx), src1=sgpr(sIpt), comment="WG tile offset"))
        writer.releaseStreamKConstSgpr(sIdx)
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SAddU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sTmp+1), comment="DP start offset + WG offset"))
        # If already in DP, add dpShift
        sGrid = writer.acquireStreamKConstSgpr(kernel, "skGrid")
        sIpt2 = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.skConstVgprs["skGrid"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt2), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sGrid), src1=sgpr(sIpt2), comment="DP iterations shift"))
        writer.releaseStreamKConstSgpr(sGrid)
        writer.releaseStreamKConstSgpr(sIpt2)
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"), comment="Add DP shift"))
        # Save DP iter in sTmp
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Check if in SK or DP section"))
        module.add(SCSelectB32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sTmp+1), comment="Select first DP tile, or add DP shift"))
        # If staying in SK portion, next iteration is sTmp+2
        module.add(SCmpLtU32(src0=sgpr(sTmp+2), src1=sgpr("StreamKIterEnd"), comment="Check if there are more SK tiles"))
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sTmp+2), src1=sgpr(sTmp+3), comment="Select next SK or DP tile"))

        module.add(self.skIndexToWG(writer, kernel, sTmp))

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("StreamK TwoTileOriginal computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tP, sTmp))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal computeStoreSrdStart")
        module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK TwoTileOriginal graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK TwoTileOriginal tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK TwoTileOriginal calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK TwoTileOriginal storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct))
        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK TwoTileOriginal writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
        return module

    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK TwoTileOriginal initializeSrdAddressFlagsCheck")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="Parallel Reduction for General Batched GEMM, Srd initialized to workspace"))
        return module

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK TwoTileOriginal routeToGeneralBatchedOrStridedBatched")
        module.add(self.stridedBatchOrGeneralBatch(stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel))
        return module

    def kernelEnd(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal kernelEnd")
        return module

class StreamKTwoTileDPFirst(StreamK):
    kernel = {"StreamK": 3}
    emitsParallelReductionSgprAliases = True
    borrowsSrdWsInEpilogue = True
    emitsWorkspaceReductionBpe = True
    requiresWorkspaceReductionStorePath = True
    supportsSubtileImpl = True

    def preLoop(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst openLoop")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Workaround for gfx12
        if writer.states.archCaps["WorkGroupIdFromTTM"]:
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        if skConstsInVgprs:
            module.add(VMovB32(dst=vgpr(self._skv(writer, "StreamKIdx")), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index to VGPR"))
        else:
            module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index"))

        if kernel["StreamKForceDPOnly"]:
            sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sIpt), comment="DP starting iteration"))
            writer.releaseStreamKConstSgpr(sIdx)
            with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
                sTmp = sTmpRes.idx
                module.add(self.computeTotalTiles(writer, kernel, sTmp))
                module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(sTmp), src1=sgpr(sIpt), comment="totalIters = totalTiles * itersPerTile"))
                module.add(SMovB32(dst=sgpr("StreamKIterEnd"), src=sgpr(sTmp), comment="DP ending iteration"))
                module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTmp), comment="Make sure there's work to do"))
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))
            return module

        # Two-tile SK (DP first)
        # Do DP tiles before SK
        skInitDone = Label("SK_InitDone", "")

        # Choose reduction strategy
        # If synchronizer buffer exists, then do single-kernel stream-k fixup step with tree reduction
        # If there's no synchronizer, parallel reduction is done in a post-kernel
        skSplitInit = Label("SK_SplitInit", "")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=skSplitInit.getLabelName(), comment="Jump to single kernel init"))

        ################
        # Parallel reduction init
        ################
        # WGsPerTile = skTiles (would be WGsPerTile = grid / tiles)
        # tile = Idx / WGsPerTile
        # partialIndex = Idx % WGsPerTile
        stmpTileIdx = writer.sgprPool.checkOut(1, "TileIdx")
        stmpPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
        tmpVgpr = writer.vgprPool.checkOut(2, "div")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
        module.add(scalarUInt32DivideAndRemainder(qReg=stmpTileIdx, dReg=sIdx, divReg="SkSplit", rReg=stmpPartialIdx, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile"))
        writer.releaseStreamKConstSgpr(sIdx)
        tmpVgprRes = None
        writer.vgprPool.checkIn(tmpVgpr)

        # if (partialIdx < extraIters) then (skIter = partialIdx * (itersPerWG + 1)) else (skIter = partialIdx * itersPerWG + extraIters)
        skHasExtraLabel = Label("SK_HasExtra", "")
        skDoneExtraLabel = Label("SK_DoneExtra", "")

        # PartialIdx = itersPerTile % skSplit (skSplit is passed as SkSplit)
        # extraIters = ItersPerTile - SkSplit * skItersPerWG
        sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
        sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sSkExtraIters), src0=sgpr("SkSplit"), src1=sgpr(sIpw)))
        module.add(SSubU32(dst=sgpr(sSkExtraIters), src0=sgpr(sIpt), src1=sgpr(sSkExtraIters), comment="extraIters = itersPerTile - SkSplit * skItersPerWG"))

        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(stmpPartialIdx), src1=sgpr(sIpw), comment="StreamK starting iteration (case: after extra iters)"))
        module.add(SCmpLtU32(src0=sgpr(stmpPartialIdx), src1=sgpr(sSkExtraIters), comment="Check if WG gets an extra iteration"))
        module.add(SCBranchSCC1(labelName=skHasExtraLabel.getLabelName(), comment="Has extra iter"))
        # No extra
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(sSkExtraIters), comment="This WG does not have an extra iteration"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sIpw), comment="StreamK ending iteration (case: after extra iters)"))
        module.add(SBranch(labelName=skDoneExtraLabel.getLabelName(), comment="Done init for parallel reduction"))
        # Has extra
        module.add(skHasExtraLabel)
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(stmpPartialIdx), comment="This WG has an extra iteration"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sIpw), comment="StreamK ending iteration (case: after extra iters)"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=1, comment="StreamK ending iteration (case: after extra iters)"))
        module.add(skDoneExtraLabel)
        writer.releaseStreamKConstSgpr(sIpw)
        # Offset to tile
        module.add(SMulI32(dst=sgpr(stmpTileIdx), src0=sgpr(stmpTileIdx), src1=sgpr(sIpt), comment="Tile offset = tilesIdx * itersPerTile"))
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(stmpTileIdx), comment="Offset to correct tile"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(stmpTileIdx), comment="Offset to correct tile"))
        # Save partial idx for later
        module.add(SMovB32(dst=sgpr("SkPartialIdx"), src=sgpr(stmpPartialIdx), comment="Save partial idx for SrdD calculation"))
        # Done init
        module.add(SBranch(labelName=skInitDone.getLabelName(), comment="Done init for parallel reduction"))

        # # Save PratialIdx for later, skExtraIters is unused for partial reduction
        # module.add(SMovB32(dst=sgpr("skExtraIters"), src=sgpr(stmpPartialIdx), comment="Save partial idx for SrdD calculation"))
        # # StreamKIter = tile * itersPerTile + itersPerWG * partialIndex
        # module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(stmpTileIdx), src1=sgpr("ItersPerTile"), comment="Tile offset = tilesIdx * itersPerTile"))
        # module.add(SMulI32(dst=sgpr(stmpPartialIdx), src0=sgpr("SKItersPerWG"), src1=sgpr(stmpPartialIdx), comment="Offset within tile = itersPerWG * partialIdx"))
        # module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(stmpPartialIdx), comment="StreamKIter = tileIdx * itersPerTile + partialIdx * itersPerWG"))
        # # if itersPerWG * partialIndex > itersPerTile jump to end
        # module.add(SCmpLtU32(src0=sgpr(stmpPartialIdx), src1=sgpr("ItersPerTile"), comment="Make sure there's work to do"))
        # module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))
        # # StreamKIterEnd = StreamKIter + itersPerWG
        # module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr("SKItersPerWG"), comment="StreamKIterEnd = StreamKIter + itersPerWG"))
        # # tileEnd = (tile + 1) * itersPerTile
        # module.add(SAddU32(dst=sgpr(stmpTileIdx), src0=sgpr(stmpTileIdx), src1=1, comment="Find end of tile"))
        # module.add(SMulI32(dst=sgpr(stmpTileIdx), src0=sgpr(stmpTileIdx), src1=sgpr("ItersPerTile"), comment="Find end of tile"))
        # # StreamKIterEnd = min(StreamKIterEnd, tileEnd)
        # # TODO SMin instruciton
        # module.add(SCmpLtU32(src0=sgpr("StreamKIterEnd"), src1=sgpr(stmpTileIdx), comment="StreamKIterEnd = min(StreamKIterEnd, tileEnd)"))
        # module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(stmpTileIdx), comment="Set start iter"))
        # # Done init
        # module.add(SBranch(labelName=skInitDone.getLabelName(), comment="Done init for parallel reduction"))
        module.add(skSplitInit)
        writer.sgprPool.checkIn(sSkExtraIters)
        writer.sgprPool.checkIn(stmpPartialIdx)
        writer.sgprPool.checkIn(stmpTileIdx)

        ################
        # Tree reduction init
        ################
        sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sIpt), comment="DP starting iteration (case: DP work to do)"))
        writer.releaseStreamKConstSgpr(sIdx)
        with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.computeTotalIters(writer, kernel, sTmp))
            module.add(SMovB32(dst=sgpr("StreamKIterEnd"), src=sgpr(sTmp), comment="DP ending iteration (case: only DP work to do)"))
            sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
            module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(sSkt), src1=sgpr(sIpt), comment="Total SK iters"))
            writer.releaseStreamKConstSgpr(sSkt)
            module.add(SCmpLtU32(src0=sgpr(sTmp), src1=sgpr("StreamKIterEnd"), comment="Check if there are DP tiles to do"))
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SCBranchSCC1(labelName=skInitDone.getLabelName(), comment="Done init"))

        # If there are no DP tiles to do, regular SK init
        # iter count after all extra iters have been distributed
        with writer.allocTmpSgpr(1, tag="extraIters") as extraItersRes, \
             writer.allocTmpSgpr(2, alignment=1, tag="SKIter") as skIterRes:
            sSkExtraIters = extraItersRes.idx
            sIter = skIterRes.idx
            module.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIter)) # sIter used as tmp
            sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
            sIpw = writer.acquireStreamKConstSgpr(kernel, "SKItersPerWG")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sIpw), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
            module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sIpw), comment="StreamK starting iteration (case: after extra iters)"))
            module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(sSkExtraIters), comment="Add extra iters"))
            module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sIpw), comment="StreamK ending iteration (case: after extra iters)"))
            # iter count before all extra iters have been distributed
            module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIpw), src1=1, comment="Spread out extra iterations"))
            writer.releaseStreamKConstSgpr(sIpw)
            module.add(SMulI32(dst=sgpr(sIter), src0=sgpr(sIdx), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
            module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
            # select correct start/end iteration index
            module.add(SCmpLtU32(src0=sgpr(sIdx), src1=sgpr(sSkExtraIters), comment="Check if lane gets an extra iteration"))
            writer.releaseStreamKConstSgpr(sIdx)
            module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
            module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        sTmp = writer.sgprPool.checkOut(1, "TotalSKIters")
        sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(sSkt), src1=sgpr(sIpt), comment="Total SK iters"))
        writer.releaseStreamKConstSgpr(sSkt)
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp), comment="Cap ending iter at total SK iters"))
        writer.sgprPool.checkIn(sTmp)

        module.add(skInitDone)
        # check if this WG has no work to do
        with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.computeTotalIters(writer, kernel, sTmp))
            module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTmp), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK TwoTileDPFirst graWorkGroup")
        skConstsInVgprs = writer.isStreamKConstantsToVgprEnabled(kernel)

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=not kernel.get("UseSubtileImpl", False))

        if kernel["StreamKForceDPOnly"]:
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(self.computeTotalTiles(writer, kernel, sTmp+3))
            module.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sIpt), comment="dpSectionSize = totalTiles * ItersPerTile"))
            module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTmp+3), comment="Make sure there's DP work to do"))
            module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))
            writer.releaseStreamKConstSgpr(sIpt)

            module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            sGrid = writer.acquireStreamKConstSgpr(kernel, "skGrid")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.skConstVgprs["skGrid"])))
            module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sGrid), src1=sgpr(sIpt), comment="DP iterations shift"))
            writer.releaseStreamKConstSgpr(sGrid)
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"), comment="Add DP shift"))
            module.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+1), comment="Store next DP iteration"))

            module.add(self.skIndexToWG(writer, kernel, sTmp))

            alphaLabel = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
            module.add(BranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), alphaLabel))
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
            skCloseLoopLabel = Label("SK_CloseLoop", "")
            module.add(writer.longBranchScc0(skCloseLoopLabel, posNeg=1))
            sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
            if writer.isStreamKConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr(sIpt), comment="Skip iterations"))
            writer.releaseStreamKConstSgpr(sIpt)
            module.add(alphaLabel)

            writer.sgprPool.checkIn(sTmp)
            return module

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        skUpdateDone = Label("SK_UpdateDone", "")

        # Choose reduction strategy
        skSplitUpdate = Label("SK_SplitUpdate", "")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=skSplitUpdate.getLabelName(), comment="Jump to single kernel update"))
        # Parallel reduction doesn't cross tile boundaries, move to end
        module.add(SMovB32(dst=sgpr(sTmp+1), src=sgpr("StreamKIterEnd"), comment="Parallel reduction, work contained to single partial tile"))
        # Done update
        module.add(SBranch(labelName=skUpdateDone.getLabelName(), comment="Done update for parallel reduction"))
        module.add(skSplitUpdate)

        module.add(self.computeTotalTiles(writer, kernel, sTmp+3))
        sSkt = writer.acquireStreamKConstSgpr(kernel, "skTiles")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sSkt), src=vgpr(writer.states.skConstVgprs["skTiles"])))
        module.add(SSubU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sSkt), comment="dpTiles = totalTiles - skTiles"))
        writer.releaseStreamKConstSgpr(sSkt)

        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        sGrid = writer.acquireStreamKConstSgpr(kernel, "skGrid")
        if skConstsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
            module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.skConstVgprs["skGrid"])))
        module.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sIpt), comment="dpSectionSize = dpTiles * ItersPerTile"))

        # If in DP, add dpShift
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sGrid), src1=sgpr(sIpt), comment="DP iterations shift"))
        writer.releaseStreamKConstSgpr(sGrid)
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"), comment="Add DP shift"))
        # if sTmp+1 < sTmp+3, continue DP (add dpShift)
        module.add(SCmpLtU32(src0=sgpr(sTmp+1), src1=sgpr(sTmp+3), comment="Check if still in DP section"))
        module.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(), comment="Done update"))
        # if StreamKIter >= sTmp+3, continue SK (add skShift?)
        module.add(SMovB32(dst=sgpr(sTmp+1), src=sgpr(sTmp+2), comment="SK iterations shift"))
        module.add(SCmpLeU32(src0=sgpr(sTmp+3), src1=sgpr("StreamKIter"), comment="Check if continuing in SK section"))
        module.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(), comment="Done update"))
        # if sTmp+1 > sTmp+3 and StreamKIter < sTmp+3, switch from DP to SK (add dpShift)
        # iter count after all extra iters have been distributed
        with writer.allocTmpSgpr(1, tag="extraIters") as extraItersRes, \
             writer.allocTmpSgpr(2, alignment=1, tag="SKIter") as skIterRes:
            sSkExtraIters = extraItersRes.idx
            sIter = skIterRes.idx
            module.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIter)) # sIter used as tmp
            sIdx = writer.acquireStreamKConstSgpr(kernel, "StreamKIdx")
            if skConstsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.skConstVgprs["StreamKIdx"])))
                module.add(VReadfirstlaneB32(dst=sgpr(sIter), src=vgpr(writer.states.skConstVgprs["SKItersPerWG"])))
                sItersPerWg = sIter
            else:
                sItersPerWg = "SKItersPerWG"
            module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(sIdx), src1=sgpr(sItersPerWg), comment="StreamK starting iteration (case: after extra iters)"))
            module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr(sSkExtraIters), comment="Add extra iters"))
            module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr(sItersPerWg), comment="StreamK ending iteration (case: after extra iters)"))
            # iter count before all extra iters have been distributed
            module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sItersPerWg), src1=1, comment="Spread out extra iterations"))
            module.add(SMulI32(dst=sgpr(sIter), src0=sgpr(sIdx), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
            module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
            # select correct start/end iteration index
            module.add(SCmpLtU32(src0=sgpr(sIdx), src1=sgpr(sSkExtraIters), comment="Check if lane gets an extra iteration"))
            writer.releaseStreamKConstSgpr(sIdx)
            module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
            module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr("StreamKIter"), src1=sgpr(sTmp+3), comment="Offset to start of SK section"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp+3), comment="Offset to start of SK section"))
        with writer.allocTmpSgpr(1, tag="TotalIters") as tmpTotalIters:
            sTotalIters = tmpTotalIters.idx
            module.add(self.computeTotalIters(writer, kernel, sTotalIters))
            # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
            module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTotalIters), comment="Cap ending iter at total SK iters"))
            # check if this WG has no work to do
            module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTotalIters), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        # If in SK, next iteration is sTmp+2
        # Increment StreamK iteration
        module.add(skUpdateDone)
        module.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+1), comment="Store current iteration"))

        # Map SK index to WG
        module.add(self.skIndexToWG(writer, kernel, sTmp))

        # Short circuit if alpha==0 (skip main loop and reading A/B, only do beta * C)
        # To skip main loop in stream-k, we check if this WG is responsible for writing results (ie: WG starts tile)
        # If WG starts tile then set LocalEnd=ItersPerTile to skip fixup step, and set loopCounter to 0 to skip main loop
        # If WG does not start tile, skip to end of persistent loop to check for other SK tile
        alphaLabel = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
        module.add(BranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), alphaLabel))
        # Skip to end if not doing the global write
        module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        skCloseLoopLabel = Label("SK_CloseLoop", "")
        module.add(writer.longBranchScc0(skCloseLoopLabel, posNeg=1))
        sIpt = writer.acquireStreamKConstSgpr(kernel, "ItersPerTile")
        if writer.isStreamKConstantsToVgprEnabled(kernel):
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.skConstVgprs["ItersPerTile"])))
        module.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr(sIpt), comment="Skip iterations"))
        writer.releaseStreamKConstSgpr(sIpt)
        module.add(alphaLabel)

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("StreamK TwoTileDPFirst computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tP, sTmp))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst computeStoreSrdStart")
        module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK TwoTileDPFirst graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK TwoTileDPFirst tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK TwoTileDPFirst calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK TwoTileDPFirst storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct))
        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK TwoTileDPFirst writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
        return module

    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK TwoTileDPFirst initializeSrdAddressFlagsCheck")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="Parallel Reduction for General Batched GEMM, Srd initialized to workspace"))
        return module        

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK TwoTileDPFirst routeToGeneralBatchedOrStridedBatched")
        module.add(self.stridedBatchOrGeneralBatch(stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel))
        return module

    def kernelEnd(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst kernelEnd")
        return module

class StreamKDynamic(StreamK):
    kernel = {"StreamK": 4}
    requiresWorkspaceReductionStorePath = True
    keepsConstantsInSgpr = True
    supportsSubtileImpl = True

    def preLoop(self, writer, kernel):
        module = Module("StreamK Dynamic openLoop")

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Workaround for gfx12
        if writer.states.archCaps["WorkGroupIdFromTTM"]:
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"), comment="Save original StreamK index"))
        # Two-tile SK (DP first)
        # Do DP tiles before SK
        skInitDone = Label("SK_InitDone", "")
        module.add(skInitDone)

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Dynamic graWorkGroup")

        skFullTile = Label("SK_FullTile", "")
        skPartialTile = Label("SK_PartialTile", "")
        skDone = Label("SK_Done", "")

        # Local address for sharing work id
        vLocalAddress = writer.vgprPool.checkOut(1, "LocalAddress")
        # TODO reorganize to reduce waits
        module.add(VMovB32(dst=vgpr(vLocalAddress), src=vgpr("Serial"), comment="Move local address to vgpr"))
        module.add(VLShiftLeftB32(dst=vgpr(vLocalAddress), src=vgpr(vLocalAddress), shiftHex=log2(4), comment="Scale by BPE"))
        sFirstLane = writer.sgprPool.checkOut(1, "FirstLane")
        module.add(SNop(waitState=4, comment="4 wait required between VALU op and readfirstlane using the value"))
        module.add(VReadfirstlaneB32(dst=sgpr(sFirstLane), src=vgpr(vLocalAddress), comment="Read first lane of local address"))
        module.add(SNop(waitState=2, comment="2 wait required between readfirstlane and VALU op using the value"))
        module.add(VSubU32(dst=vgpr(vLocalAddress), src0=vgpr(vLocalAddress), src1=sgpr(sFirstLane)))
        writer.sgprPool.checkIn(sFirstLane)

        # Only first wave reads next work item index
        skSkipWorkItem = Label("SK_SkipWorkItem", "")
        sWave = writer.sgprPool.checkOut(1, "Wave")
        # module.add(SNop(waitState=4, comment="4 wait required between VALU op and readfirstlane using the value"))
        module.add(VReadfirstlaneB32(dst=sgpr(sWave), src=vgpr("Serial"), comment="Wave 0 updates flags"))
        # module.add(SNop(waitState=2, comment="2 wait required between VALU op and readfirstlane using the value"))
        module.add(SCmpEQU32(src0=sgpr(sWave), src1=0, comment="Check for wave 0"))
        module.add(SCBranchSCC0(labelName=skSkipWorkItem.getLabelName(), comment="Skip work item"))
        writer.sgprPool.checkIn(sWave)

        # Default queue index
        sQueueIdx = writer.sgprPool.checkOut(1, "QueueIdx")
        module.add(SLShiftRightB32(dst=sgpr(sQueueIdx), src=sgpr("StreamKIdx"), shiftHex=log2(8)))
        module.add(SLShiftLeftB32(dst=sgpr(sQueueIdx), src=sgpr(sQueueIdx), shiftHex=log2(8)))
        module.add(SSubU32(dst=sgpr(sQueueIdx), src0=sgpr("StreamKIdx"), src1=sgpr(sQueueIdx), comment="Default queue index"))

        # Queue address
        sAddress = writer.sgprPool.checkOutAligned(2, 2, "Address")
        module.add(SLShiftLeftB32(dst=sgpr(sAddress), src=sgpr(sQueueIdx), shiftHex=log2(256), comment="Stride queues to different cache lines"))
        module.add(SAddU32(dst=sgpr(sAddress+0), src0=sgpr(sAddress+0), src1=sgpr("AddressFlags+0")))
        module.add(SAddCU32(dst=sgpr(sAddress+1), src0=0, src1=sgpr("AddressFlags+1")))

        # Tiles in queue
        sTilesInQueue = writer.sgprPool.checkOut(1, "tilesInQueue")
        module.add(SLShiftRightB32(dst=sgpr(sTilesInQueue), src=sgpr("TotalItems"), shiftHex=log2(8)))
        sRemainder = writer.sgprPool.checkOut(1, "remainder tiles")
        module.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sTilesInQueue), shiftHex=log2(8)))
        module.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr("TotalItems"), src1=sgpr(sRemainder), comment="Remainder tiles"))
        module.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder), comment="Check if queue gets an extra tile"))
        module.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
        module.add(SAddU32(dst=sgpr(sTilesInQueue), src0=sgpr(sTilesInQueue), src1=sgpr(sRemainder)))
        writer.sgprPool.checkIn(sRemainder)

        # Workgroups in queue
        sWorkgroupsInQueue = writer.sgprPool.checkOut(1, "workgroupsInQueue")
        module.add(SLShiftRightB32(dst=sgpr(sWorkgroupsInQueue), src=sgpr("skGrid"), shiftHex=log2(8)))
        sRemainder = writer.sgprPool.checkOut(1, "remainder workgroups")
        module.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sWorkgroupsInQueue), shiftHex=log2(8)))
        module.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr("skGrid"), src1=sgpr(sRemainder), comment="Remainder workgroups"))
        module.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder), comment="Check if queue gets an extra tile"))
        module.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
        module.add(SAddU32(dst=sgpr(sWorkgroupsInQueue), src0=sgpr(sWorkgroupsInQueue), src1=sgpr(sRemainder)))
        writer.sgprPool.checkIn(sRemainder)

        # Fetch next work item index
        sWorkItemIdx = writer.sgprPool.checkOut(1, "nextWorkItemIdx")
        module.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sTilesInQueue), src1=sgpr(sWorkgroupsInQueue), comment="Queue reset"))
        module.add(SSubU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=1))
        writer.sgprPool.checkIn(sTilesInQueue)
        writer.sgprPool.checkIn(sWorkgroupsInQueue)

        # Fetch next work item
        module.add(SAtomicInc(dst=sgpr(sWorkItemIdx), base=sgpr(sAddress, 2), soffset=0, smem=SMEMModifiers(glc=True), comment="Fetch next work item index"))
        # module.add(SMovB32(dst=sgpr(sWorkItemIdx), src=sgpr("WorkGroup0")))
        module.add(SWaitCnt(kmcnt=0, comment="Wait for scalar memory op"))
        writer.sgprPool.checkIn(sAddress)

        # Convert to global work item index
        module.add(SLShiftLeftB32(dst=sgpr(sWorkItemIdx), src=sgpr(sWorkItemIdx), shiftHex=log2(8)))
        module.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=sgpr(sQueueIdx)))
        writer.sgprPool.checkIn(sQueueIdx)

        # Share work item index with all waves
        vWaveWorkItemIdx = writer.vgprPool.checkOut(1, "WaveWorkItemIdx")
        module.add(VMovB32(dst=vgpr(vWaveWorkItemIdx), src=sgpr(sWorkItemIdx), comment="Move work item index to vgpr"))
        module.add(DSStoreB32(dstAddr=vgpr(vLocalAddress), src=vgpr(vWaveWorkItemIdx), ds=DSModifiers(offset=0)))
        module.add(SWaitCnt(dscnt=0))

        module.add(skSkipWorkItem)
        module.add(SBarrier())

        module.add(DSLoadB32(dst=vgpr(vWaveWorkItemIdx), src=vgpr(vLocalAddress), ds=DSModifiers(offset=0)))
        module.add(SWaitCnt(dscnt=0))
        # module.add(SNop(waitState=4, comment="4 wait states required before reading vgpr by lane"))
        module.add(VReadfirstlaneB32(dst=sgpr(sWorkItemIdx), src=vgpr(vWaveWorkItemIdx), comment="Read work item index from vgpr"))
        # module.add(SNop(waitState=2, comment="2 wait states required before reading vgpr by lane"))
        module.add(SBarrier())

        writer.vgprPool.checkIn(vLocalAddress)
        writer.vgprPool.checkIn(vWaveWorkItemIdx)

        # Check if work item index is valid
        module.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr("TotalItems"), comment="Check if work item index is valid"))
        # If work item index is not valid, skip to end of kernel
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        # Check if work item is a full tile. The full-tile work-item count
        # spans all batches (as TotalItems does), so it must use the
        # batch-inclusive total tile count (nWG0 * nWG1 * batchCount).
        sFullTile = writer.sgprPool.checkOut(1, "fullTile")
        module.add(self.computeTotalTiles(writer, kernel, sFullTile))
        module.add(SSubU32(dst=sgpr(sFullTile), src0=sgpr(sFullTile), src1=sgpr("skTiles"), comment="Get number of full-tile work items (across all batches)"))
        module.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr(sFullTile), comment="Check if work item is a full tile"))
        module.add(SCBranchSCC0(labelName=skPartialTile.getLabelName(), comment="Work item is a partial tile"))

        # Calculate iteration range for full tile
        module.add(skFullTile)
        module.add(SMovB32(dst=sgpr("StreamKTileIdx"), src=sgpr(sWorkItemIdx), comment="StreamKTileIdx = nextWorkItemIdx"))
        module.add(SMovB32(dst=sgpr("StreamKLocalStart"), src=0, comment="StreamKLocalStart = 0"))
        module.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr("ItersPerTile"), comment="StreamKLocalEnd = ItersPerTile"))
        module.add(SBranch(labelName=skDone.getLabelName(), comment="Done"))

        # Calculate iteration range for partial tile
        module.add(skPartialTile)
        # Calculate tile index of partial work item = floor((WorkItem - FullTiles) / skSplit) + FullTiles
        module.add(SSubU32(dst=sgpr("StreamKTileIdx"), src0=sgpr(sWorkItemIdx), src1=sgpr(sFullTile), comment="Tile index of partial work item"))
        tmpVgpr = writer.vgprPool.checkOut(2, "div")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder(qReg="StreamKTileIdx", dReg="StreamKTileIdx", divReg="SKSplit", rReg="StreamKPartialIdx", tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))
        tmpVgprRes = None
        writer.vgprPool.checkIn(tmpVgpr)
        module.add(SAddU32(dst=sgpr("StreamKTileIdx"), src0=sgpr("StreamKTileIdx"), src1=sgpr(sFullTile), comment="Offset to first partial tile"))
        module.add(SMulI32(dst=sgpr("StreamKLocalStart"), src0=sgpr("StreamKPartialIdx"), src1=sgpr("SKItersPerWI"), comment="StreamKLocalStart = PartialIdx * SKItersPerWI"))
        module.add(SAddU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalStart"), src1=sgpr("SKItersPerWI"), comment="StreamKLocalEnd = StreamKLocalStart + SKItersPerWI"))
        module.add(SMinU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="Cap ending iter at ItersPerTile"))

        module.add(skDone)
        writer.sgprPool.checkIn(sFullTile)
        writer.sgprPool.checkIn(sWorkItemIdx)

        # Map StreamK tile index to wg0/1
        module.addComment0("Map StreamK tile index to wg0/1/2")
        tmpVgpr = writer.vgprPool.checkOut(2, "div")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        sRemainder = writer.sgprPool.checkOut(1, "StreamKTileIdxRemainder")
        # Per-batch tile count (NOT batch-inclusive): splits the global tile
        # index into batch (WorkGroup2) and the in-batch tile.
        sTilesPerBatch = writer.sgprPool.checkOut(1, "TilesPerBatch")
        module.add(SMulI32(dst=sgpr(sTilesPerBatch), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="tiles per batch = nWG0 * nWG1"))
        module.add(scalarUInt32DivideAndRemainder(qReg="WorkGroup2", dReg="StreamKTileIdx", divReg=sTilesPerBatch, rReg=sRemainder, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="TileID // nWG0*nWG1"))
        # Store tileID for use later in general WGM algo
        # if kernel["SpaceFillingAlgo"]:
        #     module.add(SNop(waitState=4, comment=""))
        #     module.add(SMovB32(dst=sgpr("StreamKTileID"), src=sgpr(sTmp+2), comment=""))
        module.add(scalarUInt32DivideAndRemainder(qReg="WorkGroup1", dReg=sRemainder, divReg="NumWorkGroups0", rReg="WorkGroup0", tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="TileID // nWG0"))
        tmpVgprRes = None
        writer.vgprPool.checkIn(tmpVgpr)
        writer.sgprPool.checkIn(sRemainder)
        module.addSpaceLine()

        writer.sgprPool.checkIn(sTilesPerBatch)

        # Map SK index to WG
        # module.add(self.skIndexToWG(writer, kernel, sTmp))

        # Short circuit if alpha==0 (skip main loop and reading A/B, only do beta * C)
        # To skip main loop in stream-k, we check if this WG is responsible for writing results (ie: WG starts tile)
        # If WG starts tile then set LocalEnd=ItersPerTile to skip fixup step, and set loopCounter to 0 to skip main loop
        # If WG does not start tile, skip to end of persistent loop to check for other SK tile
        # TODO verify alpha check is correct for dynamic + streamk
        # Use getNameInc (like the other SKAlphaCheck sites) so this label is
        # unique: calculateLoopNumIterCommon also emits an "SKAlphaCheck" label
        # in the same kernel, and a hardcoded name here collides with it
        # ("symbol already defined") on the dynamic StreamK path.
        alphaLabel = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
        module.add(BranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"].toEnum(), alphaLabel))
        # Skip to end if not doing the global write
        module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        skCloseLoopLabel = Label("SK_CloseLoop", "")
        module.add(writer.longBranchScc0(skCloseLoopLabel, posNeg=1))
        module.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr("ItersPerTile"), comment="Skip iterations"))
        module.add(alphaLabel)

        # writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        module = Module("StreamK Dynamic computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tc, sTmp))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK Dynamic computeStoreSrdStart")
        module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Dynamic graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Dynamic declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Dynamic tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Dynamic calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def calculateFirstPartialIdx(self, sPartialIdx):
        module = Module("StreamK Dynamic calculateFirstPartialIdx")

        module.add(SMulI32(dst=sgpr(sPartialIdx), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="Total tiles"))
        module.add(SSubU32(dst=sgpr(sPartialIdx), src0=sgpr(sPartialIdx), src1=sgpr("skTiles"), comment="Number of full tiles"))
        module.add(SSubU32(dst=sgpr(sPartialIdx), src0=sgpr("StreamKTileIdx"), src1=sgpr(sPartialIdx), comment="PartialTile = (TileIdx - #FullTiles)"))
        module.add(SMulI32(dst=sgpr(sPartialIdx), src0=sgpr(sPartialIdx), src1=sgpr("SKSplit"), comment="PartialIdxBase = PartialTile * SKSplit"))

        return module

    def calculatePartialIdx(self, sPartialIdx):
        module = Module("StreamK Dynamic calculatePartialIdx")

        module.add(self.calculateFirstPartialIdx(sPartialIdx))
        module.add(SAddU32(dst=sgpr(sPartialIdx), src0=sgpr(sPartialIdx), src1=sgpr("StreamKPartialIdx"), comment="Offset to correct partials tile"))

        return module

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Dynamic storeBranches")

        # No branches for atomic mode
        if kernel["StreamKAtomic"]:
            return module

        skStoreLabel = Label(label=writer.labels.getNameInc("SK_Store"), comment="")
        skFixupLabel = Label(label=writer.labels.getNameInc("SK_Fixup"), comment="")

        # StreamK store branches
        # if we're doing parallel reduction, jump to global write
        # module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        # module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if using parallel reduction, go to regular store code"))

        tmpSgpr = writer.sgprPool.checkOut(4, "globalWriteElements")
        # if we did not finish the tile, store partials
        # branch to beta == 0 store path
        module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="does wg finish tile?"))
        module.add(writer.longBranchScc0(skPartialsLabel, posNeg=1))

        if kernel["DebugStreamK"] & 1 == 0:
            # if we started and finished the tile, regular store code
            # branch to regular store code, skip fixup step
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
            module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if started and finished tile, go to regular store code"))

            # if we finished the tile but did not start it, fix up step
            # run fixup code before regular store code
            sPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
            module.add(self.calculateFirstPartialIdx(sPartialIdx))

            sFixupEnd = writer.sgprPool.checkOut(1, "FixupEnd")
            module.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sPartialIdx), src1=sgpr("StreamKPartialIdx"), comment="Final partial tile index"))

            module.add(skFixupLabel)

            # Check flag
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sPartialIdx), shiftHex=log2(4), comment="flag offset based on partial index"))
            module.add(SAddU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(256*8), comment="Offset flags to come after the work queues"))
            module.add(SLoadB32(dst=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=True, dlc=True, scope=CacheScope.SCOPE_DEV), comment="get flag"))

            module.add(SWaitCnt(kmcnt=0, comment="wait for flag load"))
            if kernel["DebugStreamK"] & 2 == 0:
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=1, comment="check if ready"))
                module.add(SCBranchSCC0(labelName=skFixupLabel.getLabelName(), comment="if flag not set, wait and check again"))

            # TODO Barrier here to sync all threads in workgroup, but maybe better to have separate flag for each wavefront (to be tested)
            module.add(SBarrier(comment="wait for all workgroups before resetting flag"))
            skipFlagReset = Label(label=writer.labels.getNameInc("SK_SkipFlagReset"), comment="")
            module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr+2), src=vgpr("Serial"), comment="Wave 0 updates flags"))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=0, comment="Check for wave 0"))
            module.add(SCBranchSCC0(labelName=skipFlagReset.getLabelName(), comment="Skip flag reset"))
            if writer.states.asmCaps["HasScalarStore"]:
                # (tmpSgpr+2) contains a vlue of 0, use it to reset the flag
                module.add(SStoreB32(src=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=True), comment="reset flag"))
            else:
                module.add(VMovB32(dst=vgpr(tmpVgpr), src=0, comment="move 0 to tmpVgpr"))
                module.add(self.setFlagValue(writer, src=vgpr(tmpVgpr), soffset=sgpr(tmpSgpr), comment="reset flag"))
            module.add(skipFlagReset)
            writer.sgprPool.checkIn(tmpSgpr)

            fixupEdge = [False] # Test no edge variant
            module.add(self.fixupStep(writer, kernel, vectorWidths, elements, fixupEdge, tmpVgpr, cvtVgprStruct, sPartialIdx))

            module.add(SAddU32(dst=sgpr(sPartialIdx), src0=sgpr(sPartialIdx), src1=1, comment="next partial tile index"))
            module.add(SCmpLtU32(src0=sgpr(sPartialIdx), src1=sgpr(sFixupEnd), comment="done loading partial tiles?"))
            module.add(SCBranchSCC1(labelName=skFixupLabel.getLabelName(), comment="Branch to continue fixup loop"))

            writer.sgprPool.checkIn(sFixupEnd)
            writer.sgprPool.checkIn(sPartialIdx)

        module.add(skStoreLabel)

        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        # TODO this can be combined with common case, only part that's different is workspace SRD
        module = Module("StreamK Dynamic writePartials")

        # No partials for atomic mode
        if kernel["StreamKAtomic"]:
            return module

        module.add(skPartialsLabel)
        if kernel["DebugStreamK"] & 2 != 0:
            return module

        # fixupEdge = [False] # Temporary hack to test no edge variant
        edges = [False]

        partialsLabels = {}
        for edge in edges:
            partialsLabels[edge] = Label(writer.labels.getNameInc("GW_Partials_E%u" % ( 1 if edge else 0)), comment="")

        if False in edges and True in edges:
            with self.allocTmpSgpr(4, tag="StreamKDynamic_writePartials_tmpSgprInfo") as tmpSgprInfo:
                module.add(writer.checkIsEdge(kernel, tmpSgprInfo, partialsLabels[True], partialsLabels[True]))

        for edge in edges:
            module.add(partialsLabels[edge])
            sPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
            module.add(self.calculatePartialIdx(sPartialIdx))
            module.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sPartialIdx)))
            writer.sgprPool.checkIn(sPartialIdx)
            module.add(self.partialsWriteProcedure(writer, kernel, vectorWidths, elements, False, False, edge, tmpVgpr, cvtVgprStruct, endLabel))

        return module
        
    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK Dynamic initializeSrdAddressFlagsCheck")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="Parallel Reduction for General Batched GEMM, Srd initialized to workspace"))
        return module        

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK Dynamic routeToGeneralBatchedOrStridedBatched")
        module.add(self.stridedBatchOrGeneralBatch(stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel))
        return module
        
    def kernelEnd(self, writer, kernel):
        module = Module("StreamK Dynamic kernelEnd")

        # We don't need to track completed kernels if we know total tiles and grid size
        # Reset is baked into the atomic_inc at the top of the loop
        # TODO will need to reset the rest of the synchronizer if tiles were split
        # Remaining reset can be done if workitem = grid + total - 1

        return module

def _extract_hybrid_mode():
    """Extract SK5 mode bit 30 into StreamKHybridMode; clear it in MagicShiftItersPerTile."""
    module = Module("SK5 mode extraction")
    module.add(SLShiftRightB32(dst=sgpr("StreamKHybridMode"),
                               src=sgpr("MagicShiftItersPerTile"),
                               shiftHex=hex(30),
                               comment="SK5: shift mode bit (bit 30) down"))
    module.add(SAndB32(dst=sgpr("StreamKHybridMode"),
                       src0=sgpr("StreamKHybridMode"),
                       src1=hex(0x1),
                       comment="SK5: isolate mode bit -> StreamKHybridMode"))
    module.add(SAndB32(dst=sgpr("MagicShiftItersPerTile"),
                       src0=sgpr("MagicShiftItersPerTile"),
                       src1=hex(0xBFFFFFFF),
                       comment="SK5: clear bit 30; keep shift bits and magic add bit"))
    return module

class StreamKHybrid(StreamK):
    """
    Hybrid SK3 + SK4: emits both the static (TwoTileDPFirst) and dynamic
    (Dynamic work-queue) code paths in a single kernel. A runtime mode bit
    packed into bit 30 of the MagicShiftItersPerTile kernel arg selects
    which path executes. The bit is extracted once at preLoop entry into
    the StreamKHybridMode SGPR; every divergent SK3-vs-SK4 callsite emits
    both fragments back-to-back gated by an s_cmp_eq_u32 + s_cbranch on
    that single SGPR.

    Kernel-argument layout (see Tensile/Components/Signature.py SK5 branch
    and tensilelite/src/ContractionSolution.cpp SK5 branch):

        Slot   SK3 (primary, defineSgpr)   SK4 (RegSet alias)
        ----   --------------------------  ---------------------
        0      ItersPerTile                ItersPerTile (shared)
        1      MagicNumberItersPerTile     TotalItems
        2      MagicShiftItersPerTile      SKTiles
        3      SKItersPerWG                SKSplit
        4      skGrid                      SKItersPerWI
        5      skTiles                     SKGrid

    The host pushes only the 6 args matching the active mode; the inactive
    path's code is dead (never executed at runtime) but still references
    the SK4 names, which are resolved to the SK3 slots via RegSet aliases
    emitted in KernelWriterAssembly.py (SK5 block, line ~1502).
    """
    kernel = {"StreamK": 5}
    emitsParallelReductionSgprAliases = True
    borrowsSrdWsInEpilogue = True
    emitsWorkspaceReductionBpe = True
    requiresWorkspaceReductionStorePath = True
    keepsConstantsInSgpr = True
    supportsSubtileImpl = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emitModeExtraction(self, writer, kernel):
        return _extract_hybrid_mode()

    def _emitSk3Sk4Branch(self, writer, module, tag, emitDynamic, emitStatic):
        """Emit mode-gated dual path: dynamic (SK4) first, static (SK3) second."""
        sk5Static = Label(writer.labels.getNameInc(f"SK5_Static{tag}"), "")
        sk5Done   = Label(writer.labels.getNameInc(f"SK5_{tag}Done"), "")

        module.add(SCmpEQU32(src0=sgpr("StreamKHybridMode"), src1=0,
                             comment=f"SK5: mode bit == 0 -> SK3 (static) {tag}"))
        module.add(SCBranchSCC1(labelName=sk5Static.getLabelName(),
                                comment=f"SK5: branch to static {tag}"))

        module.addComment2(f"SK5 dynamic (SK4) {tag}")
        emitDynamic(module)

        module.add(SBranch(labelName=sk5Done.getLabelName(),
                           comment=f"SK5: skip static {tag}"))

        module.add(sk5Static)
        module.addComment2(f"SK5 static (SK3) {tag}")
        emitStatic(module)

        module.add(sk5Done)

    # ------------------------------------------------------------------
    # preLoop
    # ------------------------------------------------------------------
    def preLoop(self, writer, kernel):
        module = Module("StreamK Hybrid openLoop")

        # ----- Common prologue: XCC mapping, gfx12 workaround, save WG0 -----
        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        if writer.states.archCaps["WorkGroupIdFromTTM"]:
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        # SK5 always has isStreamKConstantsToVgprEnabled(kernel) == False,
        # so save directly to the StreamKIdx SGPR (no VGPR-cache path).
        module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"),
                           comment="SK5: save original StreamK index"))

        # ----- Extract the mode bit once for the whole kernel -----
        module.add(self._emitModeExtraction(writer, kernel))

        def emitDynamicPreLoop(mod):
            sk4InitDone = Label(writer.labels.getNameInc("SK_InitDone"), "")
            mod.add(sk4InitDone)

        def emitStaticPreLoop(mod):
            sk3InitDone  = Label(writer.labels.getNameInc("SK_InitDone"), "")
            sk3SplitInit = Label(writer.labels.getNameInc("SK_SplitInit"), "")

            # Choose reduction strategy: parallel (no synchronizer) vs tree (synchronizer)
            mod.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0),
                              comment="Check for synchronizer"))
            mod.add(SCBranchSCC0(labelName=sk3SplitInit.getLabelName(),
                                 comment="Jump to single kernel init"))

            # ---- Parallel reduction init ----
            stmpTileIdx    = writer.sgprPool.checkOut(1, "TileIdx")
            stmpPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
            tmpVgpr        = writer.vgprPool.checkOut(2, "div")
            tmpVgprRes     = ContinuousRegister(idx=tmpVgpr, size=2)
            mod.add(scalarUInt32DivideAndRemainder(
                qReg=stmpTileIdx, dReg="StreamKIdx", divReg="SkSplit",
                rReg=stmpPartialIdx, tmpVgprRes=tmpVgprRes,
                wavewidth=kernel["WavefrontSize"], doRemainder=True,
                comment="TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile"))
            tmpVgprRes = None
            writer.vgprPool.checkIn(tmpVgpr)

            skHasExtraLabel  = Label(writer.labels.getNameInc("SK_HasExtra"), "")
            skDoneExtraLabel = Label(writer.labels.getNameInc("SK_DoneExtra"), "")

            sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
            mod.add(SMulI32(dst=sgpr(sSkExtraIters),
                            src0=sgpr("SkSplit"), src1=sgpr("SKItersPerWG")))
            mod.add(SSubU32(dst=sgpr(sSkExtraIters),
                            src0=sgpr("ItersPerTile"), src1=sgpr(sSkExtraIters),
                            comment="extraIters = itersPerTile - SkSplit * skItersPerWG"))

            mod.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr(stmpPartialIdx),
                            src1=sgpr("SKItersPerWG"),
                            comment="StreamK starting iteration (case: after extra iters)"))
            mod.add(SCmpLtU32(src0=sgpr(stmpPartialIdx), src1=sgpr(sSkExtraIters),
                              comment="Check if WG gets an extra iteration"))
            mod.add(SCBranchSCC1(labelName=skHasExtraLabel.getLabelName(),
                                 comment="Has extra iter"))
            # No extra
            mod.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"),
                            src1=sgpr(sSkExtraIters),
                            comment="This WG does not have an extra iteration"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"),
                            src1=sgpr("SKItersPerWG"),
                            comment="StreamK ending iteration (case: after extra iters)"))
            mod.add(SBranch(labelName=skDoneExtraLabel.getLabelName(),
                            comment="Done init for parallel reduction"))
            # Has extra
            mod.add(skHasExtraLabel)
            mod.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"),
                            src1=sgpr(stmpPartialIdx),
                            comment="This WG has an extra iteration"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"),
                            src1=sgpr("SKItersPerWG"),
                            comment="StreamK ending iteration (case: after extra iters)"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"),
                            src1=1,
                            comment="StreamK ending iteration (case: after extra iters)"))
            mod.add(skDoneExtraLabel)

            # Offset to tile
            mod.add(SMulI32(dst=sgpr(stmpTileIdx), src0=sgpr(stmpTileIdx),
                            src1=sgpr("ItersPerTile"),
                            comment="Tile offset = tilesIdx * itersPerTile"))
            mod.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"),
                            src1=sgpr(stmpTileIdx), comment="Offset to correct tile"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"),
                            src1=sgpr(stmpTileIdx), comment="Offset to correct tile"))
            # Save partial idx for SrdD calculation
            mod.add(SMovB32(dst=sgpr("SkPartialIdx"), src=sgpr(stmpPartialIdx),
                            comment="Save partial idx for SrdD calculation"))
            mod.add(SBranch(labelName=sk3InitDone.getLabelName(),
                            comment="Done init for parallel reduction"))

            mod.add(sk3SplitInit)
            writer.sgprPool.checkIn(sSkExtraIters)
            writer.sgprPool.checkIn(stmpPartialIdx)
            writer.sgprPool.checkIn(stmpTileIdx)

            # ---- Tree reduction init ----
            mod.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"),
                            src1=sgpr("ItersPerTile"),
                            comment="DP starting iteration (case: DP work to do)"))
            with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
                sTmp = sTmpRes.idx
                mod.add(self.computeTotalIters(writer, kernel, sTmp))
                mod.add(SMovB32(dst=sgpr("StreamKIterEnd"), src=sgpr(sTmp),
                                comment="DP ending iteration (case: only DP work to do)"))
                mod.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("skTiles"),
                                src1=sgpr("ItersPerTile"), comment="Total SK iters"))
                mod.add(SCmpLtU32(src0=sgpr(sTmp), src1=sgpr("StreamKIterEnd"),
                                  comment="Check if there are DP tiles to do"))
            mod.add(SCBranchSCC1(labelName=sk3InitDone.getLabelName(),
                                 comment="Done init"))

            # No DP tiles to do, regular SK init
            sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
            sIter = writer.sgprPool.checkOut(2, "SKIter")
            mod.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIter))
            mod.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"),
                            src1=sgpr("SKItersPerWG"),
                            comment="StreamK starting iteration (case: after extra iters)"))
            mod.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"),
                            src1=sgpr(sSkExtraIters), comment="Add extra iters"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"),
                            src1=sgpr("SKItersPerWG"),
                            comment="StreamK ending iteration (case: after extra iters)"))
            mod.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr("SKItersPerWG"), src1=1,
                            comment="Spread out extra iterations"))
            mod.add(SMulI32(dst=sgpr(sIter), src0=sgpr("StreamKIdx"),
                            src1=sgpr(sIter+1),
                            comment="StreamK starting iteration (case: before extra iters)"))
            mod.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1),
                            comment="StreamK ending iteration (case: before extra iters)"))
            mod.add(SCmpLtU32(src0=sgpr("StreamKIdx"), src1=sgpr(sSkExtraIters),
                              comment="Check if lane gets an extra iteration"))
            mod.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter),
                                src1=sgpr("StreamKIter"), comment="Set start iter"))
            mod.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1),
                                src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
            writer.sgprPool.checkIn(sSkExtraIters)
            writer.sgprPool.checkIn(sIter)
            sTmp = writer.sgprPool.checkOut(1, "TotalSKIters")
            mod.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("skTiles"),
                            src1=sgpr("ItersPerTile"), comment="Total SK iters"))
            mod.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"),
                            src1=sgpr(sTmp), comment="Cap ending iter at total SK iters"))
            writer.sgprPool.checkIn(sTmp)

            mod.add(sk3InitDone)
            with writer.allocTmpSgpr(1, tag="TotalIters") as sTmpRes:
                sTmp = sTmpRes.idx
                mod.add(self.computeTotalIters(writer, kernel, sTmp))
                mod.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTmp),
                                  comment="Make sure there's work to do"))
            mod.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        self._emitSk3Sk4Branch(writer, module, "PreLoop", emitDynamicPreLoop, emitStaticPreLoop)
        return module

    # ------------------------------------------------------------------
    # graWorkGroup
    # ------------------------------------------------------------------
    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Hybrid graWorkGroup")

        def emitDynamicGRA(mod):

            skFullTile    = Label(writer.labels.getNameInc("SK_FullTile"), "")
            skPartialTile = Label(writer.labels.getNameInc("SK_PartialTile"), "")
            skDone        = Label(writer.labels.getNameInc("SK_Done"), "")

            # Local address for sharing work id
            vLocalAddress = writer.vgprPool.checkOut(1, "LocalAddress")
            mod.add(VMovB32(dst=vgpr(vLocalAddress), src=vgpr("Serial"),
                               comment="Move local address to vgpr"))
            mod.add(VLShiftLeftB32(dst=vgpr(vLocalAddress), src=vgpr(vLocalAddress),
                                      shiftHex=log2(4), comment="Scale by BPE"))
            sFirstLane = writer.sgprPool.checkOut(1, "FirstLane")
            mod.add(SNop(waitState=4,
                            comment="4 wait required between VALU op and readfirstlane using the value"))
            mod.add(VReadfirstlaneB32(dst=sgpr(sFirstLane), src=vgpr(vLocalAddress),
                                         comment="Read first lane of local address"))
            mod.add(SNop(waitState=2,
                            comment="2 wait required between readfirstlane and VALU op using the value"))
            mod.add(VSubU32(dst=vgpr(vLocalAddress), src0=vgpr(vLocalAddress),
                               src1=sgpr(sFirstLane)))
            writer.sgprPool.checkIn(sFirstLane)

            # Only first wave reads next work item index
            skSkipWorkItem = Label(writer.labels.getNameInc("SK_SkipWorkItem"), "")
            sWave = writer.sgprPool.checkOut(1, "Wave")
            mod.add(VReadfirstlaneB32(dst=sgpr(sWave), src=vgpr("Serial"),
                                         comment="Wave 0 updates flags"))
            mod.add(SCmpEQU32(src0=sgpr(sWave), src1=0, comment="Check for wave 0"))
            mod.add(SCBranchSCC0(labelName=skSkipWorkItem.getLabelName(),
                                    comment="Skip work item"))
            writer.sgprPool.checkIn(sWave)

            # Default queue index
            sQueueIdx = writer.sgprPool.checkOut(1, "QueueIdx")
            mod.add(SLShiftRightB32(dst=sgpr(sQueueIdx), src=sgpr("StreamKIdx"),
                                       shiftHex=log2(8)))
            mod.add(SLShiftLeftB32(dst=sgpr(sQueueIdx), src=sgpr(sQueueIdx),
                                      shiftHex=log2(8)))
            mod.add(SSubU32(dst=sgpr(sQueueIdx), src0=sgpr("StreamKIdx"),
                               src1=sgpr(sQueueIdx), comment="Default queue index"))

            # Queue address
            sAddress = writer.sgprPool.checkOutAligned(2, 2, "Address")
            mod.add(SLShiftLeftB32(dst=sgpr(sAddress), src=sgpr(sQueueIdx),
                                      shiftHex=log2(256),
                                      comment="Stride queues to different cache lines"))
            mod.add(SAddU32(dst=sgpr(sAddress+0), src0=sgpr(sAddress+0),
                               src1=sgpr("AddressFlags+0")))
            mod.add(SAddCU32(dst=sgpr(sAddress+1), src0=0, src1=sgpr("AddressFlags+1")))

            # Tiles in queue
            sTilesInQueue = writer.sgprPool.checkOut(1, "tilesInQueue")
            mod.add(SLShiftRightB32(dst=sgpr(sTilesInQueue), src=sgpr("TotalItems"),
                                       shiftHex=log2(8)))
            sRemainder = writer.sgprPool.checkOut(1, "remainder tiles")
            mod.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sTilesInQueue),
                                      shiftHex=log2(8)))
            mod.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr("TotalItems"),
                               src1=sgpr(sRemainder), comment="Remainder tiles"))
            mod.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder),
                                 comment="Check if queue gets an extra tile"))
            mod.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
            mod.add(SAddU32(dst=sgpr(sTilesInQueue), src0=sgpr(sTilesInQueue),
                               src1=sgpr(sRemainder)))
            writer.sgprPool.checkIn(sRemainder)

            # Workgroups in queue
            sWorkgroupsInQueue = writer.sgprPool.checkOut(1, "workgroupsInQueue")
            # SK5: SKGrid is the SK4-dedicated grid SGPR (uppercase).
            mod.add(SLShiftRightB32(dst=sgpr(sWorkgroupsInQueue), src=sgpr("SKGrid"),
                                       shiftHex=log2(8)))
            sRemainder = writer.sgprPool.checkOut(1, "remainder workgroups")
            mod.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sWorkgroupsInQueue),
                                      shiftHex=log2(8)))
            mod.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr("SKGrid"),
                               src1=sgpr(sRemainder), comment="Remainder workgroups"))
            mod.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder),
                                 comment="Check if queue gets an extra tile"))
            mod.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
            mod.add(SAddU32(dst=sgpr(sWorkgroupsInQueue),
                               src0=sgpr(sWorkgroupsInQueue), src1=sgpr(sRemainder)))
            writer.sgprPool.checkIn(sRemainder)

            # Fetch next work item index
            sWorkItemIdx = writer.sgprPool.checkOut(1, "nextWorkItemIdx")
            mod.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sTilesInQueue),
                               src1=sgpr(sWorkgroupsInQueue), comment="Queue reset"))
            mod.add(SSubU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=1))
            writer.sgprPool.checkIn(sTilesInQueue)
            writer.sgprPool.checkIn(sWorkgroupsInQueue)

            mod.add(SAtomicInc(dst=sgpr(sWorkItemIdx), base=sgpr(sAddress, 2),
                                  soffset=0, smem=SMEMModifiers(glc=True),
                                  comment="Fetch next work item index"))
            mod.add(SWaitCnt(kmcnt=0, comment="Wait for scalar memory op"))
            writer.sgprPool.checkIn(sAddress)

            # Convert to global work item index
            mod.add(SLShiftLeftB32(dst=sgpr(sWorkItemIdx), src=sgpr(sWorkItemIdx),
                                      shiftHex=log2(8)))
            mod.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx),
                               src1=sgpr(sQueueIdx)))
            writer.sgprPool.checkIn(sQueueIdx)

            # Share work item index with all waves
            vWaveWorkItemIdx = writer.vgprPool.checkOut(1, "WaveWorkItemIdx")
            mod.add(VMovB32(dst=vgpr(vWaveWorkItemIdx), src=sgpr(sWorkItemIdx),
                               comment="Move work item index to vgpr"))
            mod.add(DSStoreB32(dstAddr=vgpr(vLocalAddress), src=vgpr(vWaveWorkItemIdx),
                                  ds=DSModifiers(offset=0)))
            mod.add(SWaitCnt(dscnt=0))

            mod.add(skSkipWorkItem)
            mod.add(SBarrier())

            mod.add(DSLoadB32(dst=vgpr(vWaveWorkItemIdx), src=vgpr(vLocalAddress),
                                 ds=DSModifiers(offset=0)))
            mod.add(SWaitCnt(dscnt=0))
            mod.add(VReadfirstlaneB32(dst=sgpr(sWorkItemIdx), src=vgpr(vWaveWorkItemIdx),
                                         comment="Read work item index from vgpr"))
            mod.add(SBarrier())

            writer.vgprPool.checkIn(vLocalAddress)
            writer.vgprPool.checkIn(vWaveWorkItemIdx)

            # Check if work item index is valid
            mod.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr("TotalItems"),
                                 comment="Check if work item index is valid"))
            mod.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

            # Full tile vs partial tile. The full-tile work-item count spans all
            # batches (as TotalItems does), so it must use the batch-inclusive
            # total tile count (nWG0 * nWG1 * batchCount). SK5: SKTiles is the
            # SK4-dedicated tiles SGPR (uppercase).
            sFullTile = writer.sgprPool.checkOut(1, "fullTile")
            mod.add(self.computeTotalTiles(writer, kernel, sFullTile))
            mod.add(SSubU32(dst=sgpr(sFullTile), src0=sgpr(sFullTile), src1=sgpr("SKTiles"),
                               comment="Get number of full-tile work items (across all batches)"))
            mod.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr(sFullTile),
                                 comment="Check if work item is a full tile"))
            mod.add(SCBranchSCC0(labelName=skPartialTile.getLabelName(),
                                    comment="Work item is a partial tile"))

            # Full tile
            mod.add(skFullTile)
            mod.add(SMovB32(dst=sgpr("StreamKTileIdx"), src=sgpr(sWorkItemIdx),
                               comment="StreamKTileIdx = nextWorkItemIdx"))
            mod.add(SMovB32(dst=sgpr("StreamKLocalStart"), src=0,
                               comment="StreamKLocalStart = 0"))
            mod.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr("ItersPerTile"),
                               comment="StreamKLocalEnd = ItersPerTile"))
            mod.add(SBranch(labelName=skDone.getLabelName(), comment="Done"))

            # Partial tile
            mod.add(skPartialTile)
            mod.add(SSubU32(dst=sgpr("StreamKTileIdx"), src0=sgpr(sWorkItemIdx),
                               src1=sgpr(sFullTile),
                               comment="Tile index of partial work item"))
            tmpVgpr = writer.vgprPool.checkOut(2, "div")
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            mod.add(scalarUInt32DivideAndRemainder(
                qReg="StreamKTileIdx", dReg="StreamKTileIdx", divReg="SKSplit",
                rReg="StreamKPartialIdx", tmpVgprRes=tmpVgprRes,
                wavewidth=kernel["WavefrontSize"], doRemainder=True))
            tmpVgprRes = None
            writer.vgprPool.checkIn(tmpVgpr)
            mod.add(SAddU32(dst=sgpr("StreamKTileIdx"), src0=sgpr("StreamKTileIdx"),
                               src1=sgpr(sFullTile), comment="Offset to first partial tile"))
            mod.add(SMulI32(dst=sgpr("StreamKLocalStart"), src0=sgpr("StreamKPartialIdx"),
                               src1=sgpr("SKItersPerWI"),
                               comment="StreamKLocalStart = PartialIdx * SKItersPerWI"))
            mod.add(SAddU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalStart"),
                               src1=sgpr("SKItersPerWI"),
                               comment="StreamKLocalEnd = StreamKLocalStart + SKItersPerWI"))
            mod.add(SMinU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"),
                               src1=sgpr("ItersPerTile"),
                               comment="Cap ending iter at ItersPerTile"))

            mod.add(skDone)
            writer.sgprPool.checkIn(sFullTile)
            writer.sgprPool.checkIn(sWorkItemIdx)

            # Map StreamK tile index to wg0/1/2
            mod.addComment0("Map StreamK tile index to wg0/1/2")
            tmpVgpr = writer.vgprPool.checkOut(2, "div")
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            sRemainder = writer.sgprPool.checkOut(1, "StreamKTileIdxRemainder")
            # Per-batch tile count (NOT batch-inclusive): splits the global tile
            # index into batch (WorkGroup2) and the in-batch tile.
            sTilesPerBatch = writer.sgprPool.checkOut(1, "TilesPerBatch")
            mod.add(SMulI32(dst=sgpr(sTilesPerBatch), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="tiles per batch = nWG0 * nWG1"))
            mod.add(scalarUInt32DivideAndRemainder(
                qReg="WorkGroup2", dReg="StreamKTileIdx", divReg=sTilesPerBatch,
                rReg=sRemainder, tmpVgprRes=tmpVgprRes,
                wavewidth=kernel["WavefrontSize"], doRemainder=True,
                comment="TileID // nWG0*nWG1"))
            mod.add(scalarUInt32DivideAndRemainder(
                qReg="WorkGroup1", dReg=sRemainder, divReg="NumWorkGroups0",
                rReg="WorkGroup0", tmpVgprRes=tmpVgprRes,
                wavewidth=kernel["WavefrontSize"], doRemainder=True,
                comment="TileID // nWG0"))
            tmpVgprRes = None
            writer.vgprPool.checkIn(tmpVgpr)
            writer.sgprPool.checkIn(sRemainder)
            mod.addSpaceLine()

            writer.sgprPool.checkIn(sTilesPerBatch)

            # alpha == 0 short-circuit
            alphaLabelD = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
            mod.add(BranchIfNotZero("Alpha",
                                       kernel["ProblemType"]["ComputeDataType"].toEnum(),
                                       alphaLabelD))
            mod.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0,
                                 comment="does wg start tile?"))
            skCloseLoopLabelD = Label("SK_CloseLoop", "")
            mod.add(writer.longBranchScc0(skCloseLoopLabelD, posNeg=1))
            mod.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr("ItersPerTile"),
                               comment="Skip iterations"))
            mod.add(alphaLabelD)


        def emitStaticGRA(mod):

            sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp")

            mod.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

            skUpdateDone  = Label(writer.labels.getNameInc("SK_UpdateDone"), "")
            skSplitUpdate = Label(writer.labels.getNameInc("SK_SplitUpdate"), "")

            mod.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0),
                                 comment="Check for synchronizer"))
            mod.add(SCBranchSCC0(labelName=skSplitUpdate.getLabelName(),
                                    comment="Jump to single kernel update"))
            # Parallel reduction
            mod.add(SMovB32(dst=sgpr(sTmp+1), src=sgpr("StreamKIterEnd"),
                               comment="Parallel reduction, work contained to single partial tile"))
            mod.add(SBranch(labelName=skUpdateDone.getLabelName(),
                               comment="Done update for parallel reduction"))
            mod.add(skSplitUpdate)

            mod.add(self.computeTotalTiles(writer, kernel, sTmp+3))
            mod.add(SSubU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr("skTiles"),
                               comment="dpTiles = totalTiles - skTiles"))

            mod.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr("ItersPerTile"),
                               comment="dpSectionSize = dpTiles * ItersPerTile"))

            mod.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr("skGrid"), src1=sgpr("ItersPerTile"),
                               comment="DP iterations shift"))
            mod.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"),
                               comment="Add DP shift"))
            mod.add(SCmpLtU32(src0=sgpr(sTmp+1), src1=sgpr(sTmp+3),
                                 comment="Check if still in DP section"))
            mod.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(),
                                    comment="Done update"))
            mod.add(SMovB32(dst=sgpr(sTmp+1), src=sgpr(sTmp+2),
                               comment="SK iterations shift"))
            mod.add(SCmpLeU32(src0=sgpr(sTmp+3), src1=sgpr("StreamKIter"),
                                 comment="Check if continuing in SK section"))
            mod.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(),
                                    comment="Done update"))

            # Switch from DP to SK
            sSkExtraIters = writer.sgprPool.checkOut(1, "extraIters")
            sIter = writer.sgprPool.checkOut(2, "SKIter")
            mod.add(self.skExtraIters(writer, kernel, sSkExtraIters, sIter))
            mod.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"),
                               src1=sgpr("SKItersPerWG"),
                               comment="StreamK starting iteration (case: after extra iters)"))
            mod.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"),
                               src1=sgpr(sSkExtraIters), comment="Add extra iters"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"),
                               src1=sgpr("SKItersPerWG"),
                               comment="StreamK ending iteration (case: after extra iters)"))
            mod.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr("SKItersPerWG"), src1=1,
                               comment="Spread out extra iterations"))
            mod.add(SMulI32(dst=sgpr(sIter), src0=sgpr("StreamKIdx"), src1=sgpr(sIter+1),
                               comment="StreamK starting iteration (case: before extra iters)"))
            mod.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1),
                               comment="StreamK ending iteration (case: before extra iters)"))
            mod.add(SCmpLtU32(src0=sgpr("StreamKIdx"), src1=sgpr(sSkExtraIters),
                                 comment="Check if lane gets an extra iteration"))
            writer.sgprPool.checkIn(sSkExtraIters)
            mod.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter),
                                   src1=sgpr("StreamKIter"), comment="Set start iter"))
            mod.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1),
                                   src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
            writer.sgprPool.checkIn(sIter)
            mod.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr("StreamKIter"), src1=sgpr(sTmp+3),
                               comment="Offset to start of SK section"))
            mod.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"),
                               src1=sgpr(sTmp+3), comment="Offset to start of SK section"))
            with writer.allocTmpSgpr(1, tag="TotalIters") as tmpTotalIters:
                sTotalIters = tmpTotalIters.idx
                mod.add(self.computeTotalIters(writer, kernel, sTotalIters))
                mod.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"),
                                   src1=sgpr(sTotalIters),
                                   comment="Cap ending iter at total SK iters"))
                mod.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr(sTotalIters),
                                     comment="Make sure there's work to do"))
            mod.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

            mod.add(skUpdateDone)
            mod.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+1),
                               comment="Store current iteration"))

            # Map SK index to WG
            mod.add(self.skIndexToWG(writer, kernel, sTmp))

            # alpha == 0 short-circuit (static path)
            alphaLabelS = Label(writer.labels.getNameInc("SKAlphaCheck"), "")
            mod.add(BranchIfNotZero("Alpha",
                                       kernel["ProblemType"]["ComputeDataType"].toEnum(),
                                       alphaLabelS))
            mod.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0,
                                 comment="does wg start tile?"))
            skCloseLoopLabelS = Label("SK_CloseLoop", "")
            mod.add(writer.longBranchScc0(skCloseLoopLabelS, posNeg=1))
            mod.add(SMovB32(dst=sgpr("StreamKLocalEnd"), src=sgpr("ItersPerTile"),
                               comment="Skip iterations"))
            mod.add(alphaLabelS)

            writer.sgprPool.checkIn(sTmp)


        self._emitSk3Sk4Branch(writer, module, "GRA", emitDynamicGRA, emitStaticGRA)
        return module

    # ------------------------------------------------------------------
    # Common delegations
    # ------------------------------------------------------------------
    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("StreamK Hybrid computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tP, sTmp))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("StreamK Hybrid computeStoreSrdStart")
        module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Hybrid graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Hybrid declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Hybrid tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Hybrid calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    # ------------------------------------------------------------------
    # SK4-style partial-index helpers (used by the dynamic side of
    # partialsWriteProcedure and the dynamic SRD setup in writePartials).
    # Note: SK5 uses SKTiles (uppercase) as the SK4-dedicated tile count.
    # ------------------------------------------------------------------
    def calculateFirstPartialIdx(self, sPartialIdx):
        module = Module("StreamK Hybrid calculateFirstPartialIdx")
        module.add(SMulI32(dst=sgpr(sPartialIdx),
                           src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"),
                           comment="Total tiles"))
        module.add(SSubU32(dst=sgpr(sPartialIdx),
                           src0=sgpr(sPartialIdx), src1=sgpr("SKTiles"),
                           comment="Number of full tiles"))
        module.add(SSubU32(dst=sgpr(sPartialIdx),
                           src0=sgpr("StreamKTileIdx"), src1=sgpr(sPartialIdx),
                           comment="PartialTile = (TileIdx - #FullTiles)"))
        module.add(SMulI32(dst=sgpr(sPartialIdx),
                           src0=sgpr(sPartialIdx), src1=sgpr("SKSplit"),
                           comment="PartialIdxBase = PartialTile * SKSplit"))
        return module

    def calculatePartialIdx(self, sPartialIdx):
        module = Module("StreamK Hybrid calculatePartialIdx")
        module.add(self.calculateFirstPartialIdx(sPartialIdx))
        module.add(SAddU32(dst=sgpr(sPartialIdx),
                           src0=sgpr(sPartialIdx), src1=sgpr("StreamKPartialIdx"),
                           comment="Offset to correct partials tile"))
        return module

    # ------------------------------------------------------------------
    # storeBranches: runtime dispatch between SK4 inlined body and SK3
    # storeBranchesCommon. Both paths terminate with their own internal
    # SK_Store label and fall through to the actual store sequence, so we
    # need an explicit SBranch over the static body after the dynamic
    # body completes.
    # ------------------------------------------------------------------
    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Hybrid storeBranches")

        if kernel["StreamKAtomic"]:
            return module

        def emitDynamicStore(mod):
            skStoreLabel = Label(writer.labels.getNameInc("SK_Store"), "")
            skFixupLabel = Label(writer.labels.getNameInc("SK_Fixup"), "")

            tmpSgpr = writer.sgprPool.checkOut(4, "globalWriteElements")
            mod.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"),
                              comment="does wg finish tile?"))
            mod.add(writer.longBranchScc0(skPartialsLabel, posNeg=1))

            if kernel["DebugStreamK"] & 1 == 0:
                mod.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0,
                                  comment="does wg start tile?"))
                mod.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(),
                                     comment="Branch if started and finished tile, go to regular store code"))

                sPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
                mod.add(self.calculateFirstPartialIdx(sPartialIdx))

                sFixupEnd = writer.sgprPool.checkOut(1, "FixupEnd")
                mod.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sPartialIdx),
                                src1=sgpr("StreamKPartialIdx"),
                                comment="Final partial tile index"))

                mod.add(skFixupLabel)

                mod.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sPartialIdx),
                                       shiftHex=log2(4),
                                       comment="flag offset based on partial index"))
                mod.add(SAddU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(256*8),
                                comment="Offset flags to come after the work queues"))
                mod.add(SLoadB32(dst=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2),
                                 soffset=sgpr(tmpSgpr),
                                 smem=SMEMModifiers(glc=True, dlc=True, scope=CacheScope.SCOPE_DEV),
                                 comment="get flag"))

                mod.add(SWaitCnt(kmcnt=0, comment="wait for flag load"))
                if kernel["DebugStreamK"] & 2 == 0:
                    mod.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=1, comment="check if ready"))
                    mod.add(SCBranchSCC0(labelName=skFixupLabel.getLabelName(),
                                         comment="if flag not set, wait and check again"))

                mod.add(SBarrier(comment="wait for all workgroups before resetting flag"))
                skipFlagReset = Label(writer.labels.getNameInc("SK_SkipFlagReset"), "")
                mod.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr+2), src=vgpr("Serial"),
                                          comment="Wave 0 updates flags"))
                mod.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=0, comment="Check for wave 0"))
                mod.add(SCBranchSCC0(labelName=skipFlagReset.getLabelName(),
                                    comment="Skip flag reset"))
                if writer.states.asmCaps["HasScalarStore"]:
                    mod.add(SStoreB32(src=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2),
                                      soffset=sgpr(tmpSgpr),
                                      smem=SMEMModifiers(glc=True), comment="reset flag"))
                else:
                    mod.add(VMovB32(dst=vgpr(tmpVgpr), src=0, comment="move 0 to tmpVgpr"))
                    mod.add(self.setFlagValue(writer, src=vgpr(tmpVgpr), soffset=sgpr(tmpSgpr),
                                              comment="reset flag"))
                mod.add(skipFlagReset)
                writer.sgprPool.checkIn(tmpSgpr)

                fixupEdge = [False]
                mod.add(self.fixupStep(writer, kernel, vectorWidths, elements,
                                       fixupEdge, tmpVgpr, cvtVgprStruct, sPartialIdx))

                mod.add(SAddU32(dst=sgpr(sPartialIdx), src0=sgpr(sPartialIdx), src1=1,
                                comment="next partial tile index"))
                mod.add(SCmpLtU32(src0=sgpr(sPartialIdx), src1=sgpr(sFixupEnd),
                                  comment="done loading partial tiles?"))
                mod.add(SCBranchSCC1(labelName=skFixupLabel.getLabelName(),
                                     comment="Branch to continue fixup loop"))

                writer.sgprPool.checkIn(sFixupEnd)
                writer.sgprPool.checkIn(sPartialIdx)
            else:
                writer.sgprPool.checkIn(tmpSgpr)

            mod.add(skStoreLabel)

        def emitStaticStore(mod):
            mod.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel,
                                             vectorWidths, elements, tmpVgpr, cvtVgprStruct))

        self._emitSk3Sk4Branch(writer, module, "Store", emitDynamicStore, emitStaticStore)
        return module

    # ------------------------------------------------------------------
    # writePartials: runtime dispatch only for the workspace SRD setup;
    # partialsWriteProcedure itself handles the SK5 flag-offset branch
    # internally (modified above), so we call it once per edge.
    # ------------------------------------------------------------------
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Hybrid writePartials")

        if kernel["StreamKAtomic"]:
            return module

        module.add(skPartialsLabel)
        if kernel["DebugStreamK"] & 2 != 0:
            return module

        edges = [False]
        partialsLabels = {}
        for edge in edges:
            partialsLabels[edge] = Label(writer.labels.getNameInc("GW_Partials_E%u" % (1 if edge else 0)), comment="")

        for edge in edges:
            module.add(partialsLabels[edge])

            def emitDynamicSrd(mod):
                sPartialIdx = writer.sgprPool.checkOut(1, "PartialIdx")
                mod.add(self.calculatePartialIdx(sPartialIdx))
                mod.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sPartialIdx)))
                writer.sgprPool.checkIn(sPartialIdx)

            def emitStaticSrd(mod):
                mod.add(self.computeWorkspaceSrd(writer, kernel, sgpr("StreamKIdx")))

            self._emitSk3Sk4Branch(writer, module, "PartialsSrd", emitDynamicSrd, emitStaticSrd)

            module.add(self.partialsWriteProcedure(writer, kernel, vectorWidths, elements,
                                                   False, False, edge, tmpVgpr, cvtVgprStruct,
                                                   endLabel))

        return module

    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("StreamK Hybrid initializeSrdAddressFlagsCheck")
        module.add(SCmpEQU64(src0=sgpr("AddressFlags", 2), src1=hex(0), comment="Check for synchronizer"))
        module.add(SCBranchSCC0(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="Parallel Reduction for General Batched GEMM, Srd initialized to workspace"))
        return module

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("StreamK Hybrid routeToGeneralBatchedOrStridedBatched")
        module.add(self.stridedBatchOrGeneralBatch(stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel))
        return module

    def kernelEnd(self, writer, kernel):
        module = Module("StreamK Hybrid kernelEnd")
        return module


# Mapping from kernel["StreamK"] int -> variant class. Lets non-
# KernelWriter consumers (e.g. Solution validation) read variant
# feature flags without an instantiated KernelWriter.
_STREAMK_VARIANT_BY_INT = {
    0: StreamKOff,
    1: StreamKBasic,
    2: StreamKTwoTileOriginal,
    3: StreamKTwoTileDPFirst,
    4: StreamKDynamic,
    5: StreamKHybrid,
}


def streamKVariantClass(streamK):
    return _STREAMK_VARIANT_BY_INT[streamK]
