from ..Component import GL2Prefetch
from ..Common import INDEX_CHARS
from typing import Mapping, Optional
from rocisa.code import Module, Label
from rocisa.instruction import SMulI32, SAddU64, VMovB32, VAddU32, VAddCOU32, \
    VAddCCOU32, VAddNCU64, VLShiftRightB32, VMulLOU32, VMulHIU32, GlobalPrefetchB8, \
    VCmpGtU32, VCndMaskB32, SSubI32, SMovB32, SMovB64, SAddU32, SAddCU32, SAndB32, SBranch, \
    SCBranchSCC1, SCMovB32, SCMovB64, SLShiftRightB32, SMulHIU32
from rocisa.container import sgpr, vgpr, RegisterContainer, VCC, GLOBALModifiers, ContinuousRegister
from rocisa.functions import vectorMultiply64Bpe, vectorMultiplyBpe, scalarMultiplyBpe, \
    vectorStaticDivideAndRemainder, scalarStaticRemainder
from rocisa.enum import TemporalHint, CacheScope
from math import log2, ceil

# Bit 15 of the packed GSU kernel argument selects how the summation loop is cut
# up between the GSU groups, and the two layouts need different start offsets and
# strides:
#   GSUC == 0: the groups interleave every DepthU, so group g starts at iteration
#              g and then steps GSU iterations at a time.
#   GSUC == 1: each group owns a contiguous run, so group g starts after every
#              lower group's run and then steps one iteration at a time.
GSUC_BIT = 0x8000

class GL2PrefetchLoad(GL2Prefetch):
    asmCaps = {"HasGlobalPrefetch": True}
    globalModifiers = GLOBALModifiers(th=TemporalHint.TH_NT, scope=CacheScope.SCOPE_SE)

    def __call__(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping):
        pass

    def init(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping):
        globalPrefetchSize: int = writer.states.regCaps["GlobalPrefetchSize"]
        tc: str = tp["tensorChar"]
        isMX: bool = tc.startswith("MX")
        isM: bool = tp.get("isM", False)
        # Cooperative prefetch spans the *whole* cluster: every workgroup in the
        # cluster contributes threads, and together they cover all the distinct
        # macro-tiles the cluster consumes rather than only the single tile one
        # workgroup uses for its own computation. Along the MT-selector axis
        # (WorkGroup0 for A, WorkGroup1 for B) the cluster spans numTileWGs
        # contiguous macro-tiles, so the tile dimension of the prefetched block
        # is scaled accordingly.
        # TODO: boundary clusters from the padded-WG edge-size path have fewer
        # than ClusterDim live workgroups, so this full-cluster count over-counts
        # the cooperative tile span and thread population. The effect is perf-only
        # (padded WGs early-exit and just skip their prefetch slice; real compute
        # data is loaded by each WG's own TDM load), so it is left unfixed for now.
        numCooperativeWGs: int = kernel["ClusterDim"][0] * kernel["ClusterDim"][1]
        numCooperativeThreads: int = numCooperativeWGs * kernel["NumThreads"]

        subTc: str = tc if isM else tc[-1]
        mt: int = kernel["MacroTile%s" % subTc]
        numTileWGs: int = kernel["ClusterDim"][tp["idx"]] if isM else (kernel["ClusterDim"][0] if subTc == "A" else kernel["ClusterDim"][1])
        bpe: float = tp["bpeGR"]

        if isMX:
            coalescedDim = mt * numTileWGs * kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]
            perpendicularDim = kernel["DepthU"] // kernel["MatrixInstK"]
        else:
            du: int = kernel["_DepthU%s" % subTc]
            coalescedDim, perpendicularDim = (mt * numTileWGs, du) if tp["tlu"] else (du, mt * numTileWGs)

        tp["gl2ncp"] = perpendicularDim
        tp["gl2ncc"] = max(1, ceil(coalescedDim * bpe / globalPrefetchSize))
        tp["gl2nc"] = tp["gl2ncp"] * tp["gl2ncc"]
        tp["gl2nl"] = max(1, ceil(tp["gl2nc"] / numCooperativeThreads))

    @staticmethod
    def numIncSgpr(kernel: Mapping) -> int:
        """Width of GL2PrefetchInc{tc}: 2 sgprs (64-bit) if PrefetchGL2Inc64Bit, else 1."""
        return 2 if kernel["PrefetchGL2Inc64Bit"] else 1

    @staticmethod
    def useSAddr(kernel: Mapping) -> bool:
        """True if each tensor uses one sgpr pair base (GL2PrefetchBase{tc}) plus 32-bit vgpr offsets."""
        return kernel["PrefetchGL2SAddr"]

    @staticmethod
    def numAddrVgpr(kernel: Mapping) -> int:
        """Vgprs per GL2PrefetchAddr{tc}_{i}: a 32-bit offset in SAddr mode, else a 64-bit address."""
        return 1 if GL2PrefetchLoad.useSAddr(kernel) else 2

    def clearIncrement(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        """Zero the addr increment if SCC is set."""
        mod = Module()
        incName: str = f"GL2PrefetchInc{tp['tensorChar']}"
        if self.numIncSgpr(kernel) == 2:
            mod.add(SCMovB64(sgpr(incName, 2), 0))
        else:
            mod.add(SCMovB32(sgpr(incName), 0))
        return mod

    def isGSUEnabled(self, kernel: Mapping) -> bool:
        """True when the kernel emits the GSUOn paths, so GSU/GSUSumIdx are live."""
        return kernel["GlobalSplitU"] > 0 or kernel["GlobalSplitU"] == -1

    def calculateGSUIterOffset(self, writer: "KernelWriterAssembly", kernel: Mapping, \
                               dstSgprIdx: int, tmpSgprRes: ContinuousRegister) -> Module:
        """Unroll iteration at which this workgroup's GSU chunk starts.

        One unroll iteration is one DepthU step for every tensor, so the result is
        tensor independent and callers scale it by each tensor's own per-iteration
        byte increment. That keeps the chunk-layout math in one place instead of
        repeating it per tensor and per TLU/MX/metadata layout.

        Clobbers GSUSumIdx+1, which the GSU component also uses as scratch;
        computeLoadSrd and calculateLoopNumIterGsu both recompute it later.
        """
        mod = Module("gl2 prefetch GSU start iteration")
        depthU: int = kernel["DepthU"]
        gsucLabel = Label(writer.labels.getNameInc("GL2PrefetchGSUC"), "")
        gsucLabelEnd = Label(writer.labels.getNameInc("GL2PrefetchGSUC_End"), "")

        mod.addComment("gl2 prefetch GSU start iteration")
        mod.add(SAndB32(dst=sgpr(dstSgprIdx), src0=sgpr("GSU"), src1=hex(GSUC_BIT), \
            comment="SCC = (GSUC == 1) ?"))
        mod.add(SCBranchSCC1(labelName=gsucLabel.getLabelName(), comment="branch if GSUC == 1"))
        mod.add(SMovB32(dst=sgpr(dstSgprIdx), src=sgpr("GSUSumIdx"), \
            comment="interleaved chunks: startIter = GSUSumIdx"))
        mod.add(SBranch(gsucLabelEnd.getLabelName()))
        mod.add(gsucLabel)
        mod.add(SLShiftRightB32(dst=sgpr(dstSgprIdx), shiftHex=int(log2(depthU)), src=sgpr("SizesSum"), \
            comment="numIter = SizesSum / DepthU(%u)" % depthU))
        mod.add(writer.calculateLoopNumIterOffsetGsu(kernel, dstSgprIdx, tmpSgprRes))
        mod.add(SMovB32(dst=sgpr(dstSgprIdx), src=sgpr(tmpSgprRes.idx), \
            comment="contiguous chunks: startIter = accumulated iters of lower groups"))
        mod.add(gsucLabelEnd)
        return mod

    def applyGSUChunk(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, \
                      gsuIterSgpr: int, baseSgprIdx: int, tmpSgprIdx: int, tmpVgprIdx: int) -> Module:
        """Move the prefetch base onto this workgroup's GSU chunk and widen the step.

        Both are multiples of the one-DepthU increment setIncrement produced, so
        the layout only has to be decoded once (calculateGSUIterOffset). The start
        offset must consume the unscaled increment, so the scaling happens after
        it and before the PGR pre-skip, which already steps by whole chunks.
        """
        mod = Module("gl2 prefetch GSU chunk offset")
        tc: str = tp["tensorChar"]
        incName: str = f"GL2PrefetchInc{tc}"
        is64: bool = self.numIncSgpr(kernel) == 2

        mod.addComment(f"gl2 prefetch GSU chunk offset of {tc}")
        mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
            sgpr(tmpSgprIdx), sgpr(tmpSgprIdx + 1),
            sgpr(gsuIterSgpr), sgpr(incName),
            tmpVgprIdx, comment="gsuOffset = startIter * inc"))
        mod.add(SAddU64(sgpr(baseSgprIdx, 2), sgpr(baseSgprIdx, 2), sgpr(tmpSgprIdx, 2), \
            comment="skip to this WG's GSU chunk"))
        if is64:
            mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(gsuIterSgpr), sgpr(f"{incName}+1"), \
                comment="gsuOffset.hi = startIter * inc.hi"))
            mod.add(SAddU32(sgpr(baseSgprIdx + 1), sgpr(baseSgprIdx + 1), sgpr(tmpSgprIdx)))
        # Widen the step to the chunk stride. In 32-bit mode this mirrors
        # GlobalReadIncs on the real load path (GSU.graIncrements): a stride that
        # overflows 32 bits is already broken there.
        mod.add(SAndB32(dst=sgpr(tmpSgprIdx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), \
            comment="Restore GSU"))
        mod.add(SAndB32(dst=sgpr(tmpSgprIdx + 1), src0=sgpr("GSU"), src1=hex(GSUC_BIT), \
            comment="SCC = (GSUC == 1) ?"))
        mod.add(SCMovB32(dst=sgpr(tmpSgprIdx), src=1, comment="stride stays DepthU if GSUC == 1"))
        if is64:
            mod.add(SMulI32(sgpr(tmpSgprIdx + 1), sgpr(f"{incName}+1"), sgpr(tmpSgprIdx), \
                comment="inc.hi * GSU chunk stride"))
            mod.add(SMulHIU32(sgpr(f"{incName}+1"), sgpr(incName), sgpr(tmpSgprIdx), \
                comment="carry of inc.lo * GSU chunk stride"))
            mod.add(SAddU32(sgpr(f"{incName}+1"), sgpr(f"{incName}+1"), sgpr(tmpSgprIdx + 1)))
        mod.add(SMulI32(sgpr(incName), sgpr(incName), sgpr(tmpSgprIdx), \
            comment="addr increment *= GSU chunk stride"))
        return mod

    def setIncrement(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        """Bytes the prefetch address advances for one DepthU step along K.

        This is the *unscaled* step. Under GSU, applyGSUChunk widens it to the
        workgroup's chunk stride once the start offset has consumed it.
        """
        mod = Module()
        tc: str = tp["tensorChar"]
        tIdx: int = tp['idx']
        isM: bool = tp.get("isM", False)
        subTc: str = tc if isM else tc[-1]
        bpe: float = tp["bpeGR"]
        du: int = kernel["_DepthU%s" % subTc]
        incName: str = f"GL2PrefetchInc{tc}"
        is64: bool = self.numIncSgpr(kernel) == 2
        if tc.startswith("MX"):
            src0, src1 = sgpr("Size%s"%INDEX_CHARS[tIdx]), \
                round(kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"] * bpe)
        elif tp["tlu"]:
            src0, src1 = writer.strideRef(subTc, 3), round(du * bpe)
        else:
            mod.add(SMovB32(dst=sgpr(incName), src=round(du * bpe), comment="addr increment"))
            if is64:
                mod.add(SMovB32(dst=sgpr(f"{incName}+1"), src=0, comment="addr increment hi"))
            return mod
        if is64:
            mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                sgpr(incName), sgpr(f"{incName}+1"), src0, src1, comment="addr increment"))
        else:
            mod.add(SMulI32(sgpr(incName), src0, src1, comment="addr increment"))
        return mod

    def calculateStartAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, \
                           gsuIterSgpr: Optional[int] = None) -> Module:
        """Compute this workgroup's prefetch start addresses.

        gsuIterSgpr holds the shared GSU chunk start iteration from
        calculateGSUIterOffset, or None when the kernel has no GSU paths.
        """
        mod = Module()
        globalPrefetchSize: int = writer.states.regCaps["GlobalPrefetchSize"]
        tc: str = tp["tensorChar"]
        tIdx: int = tp['idx']
        tlu: bool = tp["tlu"]
        isMX: bool = tc.startswith("MX")
        isM: bool = tp.get("isM", False)
        subTc: str = tc if isM else tc[-1]
        mt: int = kernel["MacroTile%s" % subTc]
        bpe: float = tp["bpeGR"]
        tileStride: str | RegisterContainer = writer.strideRef(subTc, tIdx)
        unrollStride: str | RegisterContainer = writer.strideRef(subTc, 3)
        perpStride: str | RegisterContainer = unrollStride if tlu else tileStride
        # WorkGroup{tIdx} selects the macro-tile; the other cluster axis is the
        # cooperative sharing axis. The whole cluster cooperates on the prefetch.
        sgprTileWgName: str = f"WorkGroup{tIdx}"
        sgprShareWgName: str = f"WorkGroup{1 - tIdx}"
        sgprSizeFreeName: str = f"Size{INDEX_CHARS[tIdx]}"
        numThreads: int = kernel["NumThreads"]
        vgprAddrBaseName: str = f"GL2PrefetchAddr{tc}"
        vgprAddrName0: str = f"{vgprAddrBaseName}_0"
        numTileWGs: int = kernel["ClusterDim"][tIdx]
        numShareWGs: int = kernel["ClusterDim"][1 - tIdx]
        numCooperativeWGs: int = numTileWGs * numShareWGs
        numCooperativeThreads: int = numCooperativeWGs * numThreads
        ncc: int = tp["gl2ncc"]
        nc: int = tp["gl2nc"]
        nl: int = tp["gl2nl"]
        ncPerInst: int = ceil(nc / tp["gl2nl"])
        inactiveShiftBits: int = int(log2(numCooperativeThreads // ncPerInst))
        useSAddr: bool = self.useSAddr(kernel)
        numTmpSgpr = 4
        tmpVgprIdx = writer.vgprPool.checkOutAligned(2, 2)
        tmpVgprCoalIdx = writer.vgprPool.checkOutAligned(1, 1)
        if isMX:
            mxUnit: int = kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]

        mod.addComment(f"gl2 prefetch calc start addr of {tc}")
        with writer.allocTmpSgpr(numTmpSgpr, 2) as tmpSgprRes:
            tmpSgprIdx0 = tmpSgprRes.idx
            tmpSgprIdx1 = tmpSgprRes.idx + 1
            tmpSgprIdx2 = tmpSgprRes.idx + 2
            tmpSgprIdx3 = tmpSgprRes.idx + 3
            # Cooperative thread index over the whole cluster. Flatten this
            # workgroup's cluster-local (tile, share) position into a single index
            # and offset the wave's Serial by it, so the cluster's threads jointly
            # enumerate all cooperative chunks. tmpSgprIdx3 keeps the cluster-local
            # tile index; the cluster's base macro-tile (WorkGroup{tIdx} minus it)
            # is recovered from it for the MT offset below.
            mod.add(scalarStaticRemainder(tmpSgprIdx0, tmpSgprIdx3, sgprTileWgName, numTileWGs, \
                tmpSgprRes, comment="cluster-local tile idx"))
            mod.add(scalarStaticRemainder(tmpSgprIdx0, tmpSgprIdx0, sgprShareWgName, numShareWGs, \
                tmpSgprRes, comment="cluster-local share idx"))
            mod.add(SMulI32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx3), numShareWGs, \
                comment="tile idx * shareWGs"))
            mod.add(SAddU32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx0), \
                comment="flattened cluster WG idx"))
            mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx1), numThreads, \
                comment="cluster WG idx * numThreads"))
            mod.add(VAddU32(vgpr(vgprAddrName0), vgpr("Serial"), sgpr(tmpSgprIdx0), \
                comment="cooperative thread idx"))
            if inactiveShiftBits > 0:
                assert nl == 1, "Should only have one inst if inactiveShiftBits > 0"
                mod.add(VLShiftRightB32(vgpr(vgprAddrName0), inactiveShiftBits, vgpr(vgprAddrName0), \
                    comment="shift inactive index"))
            else:
                for i in range(1, nl):
                    src = f"{vgprAddrBaseName}_{i-1}"
                    dst = f"{vgprAddrBaseName}_{i}"
                    mod.add(VAddU32(vgpr(dst), vgpr(src), ncPerInst, comment="inst index"))
            # the last inst may contain overflow address, we need to mask it
            vgprAddrNameLast = f"{vgprAddrBaseName}_{(nl-1)}"
            mod.add(VCmpGtU32(VCC(), vgpr(vgprAddrNameLast), nc-1, comment="overflow number of needed cachelines?"))
            mod.add(VCndMaskB32(vgpr(vgprAddrNameLast), vgpr(vgprAddrNameLast), nc-1, VCC()))

            # MT offset & edge limit (in units of elements). The offset is the
            # cluster's base macro-tile (WorkGroup{tIdx} floored to the cluster,
            # i.e. minus the cluster-local tile idx kept in tmpSgprIdx3), since the
            # cooperative block now spans all numTileWGs tiles the cluster covers.
            mod.add(SSubI32(sgpr(tmpSgprIdx0), sgpr(sgprTileWgName), sgpr(tmpSgprIdx3), \
                comment="cluster base tile"))
            if isMX:
                mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx0), mxUnit * mt, \
                    comment=f"clusterBaseTile * mxUnit({mxUnit}) * MT({mt})"))
                mod.add(SSubI32(sgpr(tmpSgprIdx1), sgpr(sgprSizeFreeName), 1))
                mod.add(SMulI32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), mxUnit))
                mod.add(SSubI32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx0), comment="max offset inside cluster tiles"))
            else:
                mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx0), mt, comment=f"clusterBaseTile * MT({mt})"))
                mod.add(SSubI32(sgpr(tmpSgprIdx1), sgpr(sgprSizeFreeName), 1))
                mod.add(SSubI32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx0), comment="max offset inside cluster tiles"))

            # will we have MX stride later?
            if isMX:
                perpStride = sgpr(tmpSgprIdx2)
                mod.add(SMulI32(perpStride, sgpr(sgprSizeFreeName), mxUnit, f"MX perp stride"))
            for i in range(nl):
                vgprAddrName = f"{vgprAddrBaseName}_{i}"
                vgprAddrNameHi = vgprAddrName + "+1"
                if ncc > 1:
                    mod.add(VMovB32(vgpr(tmpVgprCoalIdx), vgpr(vgprAddrName)))
                    mod.add(vectorStaticDivideAndRemainder(vgprAddrName, tmpVgprCoalIdx, tmpVgprCoalIdx, \
                        ncc, ContinuousRegister(tmpVgprIdx, 2), comment="coal/perp index calc"))
                    mod.add(VMulLOU32(vgpr(tmpVgprCoalIdx), vgpr(tmpVgprCoalIdx), round(globalPrefetchSize / bpe), \
                        comment="coal * globalPrefetchSize / bpe"))
                else:
                    mod.add(VMovB32(vgpr(tmpVgprCoalIdx), 0, comment="coalesced index"))
                
                # edge protection
                if isMX or tlu:
                    mod.add(VCmpGtU32(VCC(), vgpr(tmpVgprCoalIdx), sgpr(tmpSgprIdx1), comment="> edge limit?"))
                    mod.add(VCndMaskB32(vgpr(tmpVgprCoalIdx), vgpr(tmpVgprCoalIdx), sgpr(tmpSgprIdx1), VCC()))
                else:
                    mod.add(VCmpGtU32(VCC(), vgpr(vgprAddrName), sgpr(tmpSgprIdx1), comment="> edge limit?"))
                    mod.add(VCndMaskB32(vgpr(vgprAddrName), vgpr(vgprAddrName), sgpr(tmpSgprIdx1), VCC()))
                if useSAddr:
                    # the per-lane byte offset must fit in 32 bits
                    mod.add(VMulLOU32(vgpr(vgprAddrName), vgpr(vgprAddrName), perpStride, comment="perp *= stride"))
                    mod.add(VAddU32(vgpr(vgprAddrName), vgpr(vgprAddrName), vgpr(tmpVgprCoalIdx), comment="coal + perp"))
                    mod.add(vectorMultiplyBpe(vgprAddrName, vgprAddrName, bpe, comment="scale by bpe"))
                    continue
                # perp stride
                mod.add(VMulHIU32(vgpr(vgprAddrNameHi), vgpr(vgprAddrName), perpStride, comment="perp *= stride"))
                mod.add(VMulLOU32(vgpr(vgprAddrName), vgpr(vgprAddrName), perpStride))
                # coal + perp
                mod.add(VAddCOU32(vgpr(vgprAddrName), VCC(), vgpr(vgprAddrName), vgpr(tmpVgprCoalIdx), comment="coal + perp"))
                mod.add(VAddCCOU32(vgpr(vgprAddrNameHi), VCC(), vgpr(vgprAddrNameHi), 0, VCC()))
                mod.add(vectorMultiply64Bpe(vgprAddrName, vgprAddrName, bpe, tmpVgprIdx, comment="scale by bpe"))

            # base address + MT offset (in units of bytes)
            mod.add(scalarMultiplyBpe(tmpSgprIdx0, tmpSgprIdx0, bpe))
            if isMX or tlu:
                mod.add(SAddU32(sgpr(tmpSgprIdx0), sgpr("Address%s"%tc), sgpr(tmpSgprIdx0), comment="base address + MT offset"))
                mod.add(SAddCU32(sgpr(tmpSgprIdx1), sgpr("Address%s+1"%tc), 0))
            else:
                mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                    sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx1),
                    sgpr(tmpSgprIdx0), perpStride,
                    tmpVgprIdx, comment="*= stride"))
                mod.add(SAddU64(sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx0, 2), sgpr("Address%s"%tc, 2), comment="base address + MT offset"))
                
            # strided batch offset
            if kernel["ProblemType"]["Batched"]:
                assert kernel["ProblemType"]["StridedBatched"], "Currently GL2Prefetch does not support general batch"
                for batchIdx in kernel["ProblemType"]["IndicesBatch"]:
                    # packed index check
                    if batchIdx in kernel["ProblemType"]["IndicesFree"] or batchIdx not in tp['ia']:
                        continue
                    assert(batchIdx==2) # can only have one wg2 with a batch. Other dimensions should be packed into wg0/wg1
                    batchStrideName = "Stride%s%s"%(tc, writer.states.indexChars[batchIdx])
                    mod.add(scalarMultiplyBpe(tmpSgprIdx2, batchStrideName, bpe, comment="batchStride * bpe"))
                    mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                        sgpr(tmpSgprIdx2), sgpr(tmpSgprIdx3),
                        sgpr("WorkGroup2"), sgpr(tmpSgprIdx2),
                        tmpVgprIdx, comment="batch offset * wg2"))
                    mod.add(SAddU64(sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx2, 2)))
            # GSU chunk offset. Must precede the PGR pre-skip: it consumes the
            # unscaled increment and leaves behind the chunk-strided one that the
            # pre-skip and every in-loop increment then use.
            if gsuIterSgpr is not None:
                mod.add(self.applyGSUChunk(writer, kernel, tp, gsuIterSgpr, \
                    tmpSgprIdx0, tmpSgprIdx2, tmpVgprIdx))

            # add all together
            if useSAddr:
                mod.add(SMovB64(sgpr(f"GL2PrefetchBase{tc}", 2), sgpr(tmpSgprIdx0, 2), comment="scalar base addr"))
            else:
                for i in range(tp["gl2nl"]):
                    dst = f"{vgprAddrBaseName}_{i}"
                    mod.add(VAddNCU64(vgpr(dst, 2), vgpr(dst, 2), sgpr(tmpSgprIdx0, 2)))

        writer.vgprPool.checkIn(tmpVgprIdx)
        writer.vgprPool.checkIn(tmpVgprCoalIdx)
        return mod

    def issueLoad(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        if self.useSAddr(kernel):
            vaddrs = [vgpr(f"GL2PrefetchAddr{tc}_{i}") for i in range(tp["gl2nl"])]
            saddr = sgpr(f"GL2PrefetchBase{tc}", 2)
        else:
            vaddrs = [vgpr(f"GL2PrefetchAddr{tc}_{i}", 2) for i in range(tp["gl2nl"])]
            saddr = sgpr("off", isOff=True)
        for vaddr in vaddrs:
            mod.add(GlobalPrefetchB8(vaddr, saddr, self.globalModifiers))
        return mod

    def addOffset(self, kernel: Mapping, tp: Mapping, offsetSgpr: str | int, is64: bool) -> Module:
        """Advance every prefetch address of tp by the 32-bit (or 64-bit if is64) sgpr offsetSgpr."""
        mod = Module()
        tc: str = tp["tensorChar"]
        if self.useSAddr(kernel):
            base: str = f"GL2PrefetchBase{tc}"
            if is64:
                mod.add(SAddU64(sgpr(base, 2), sgpr(base, 2), sgpr(offsetSgpr, 2)))
            else:
                mod.add(SAddU32(sgpr(base), sgpr(base), sgpr(offsetSgpr)))
                mod.add(SAddCU32(sgpr(f"{base}+1"), sgpr(f"{base}+1"), 0))
            return mod
        for i in range(tp["gl2nl"]):
            addr = f"GL2PrefetchAddr{tc}_{i}"
            if is64:
                mod.add(VAddNCU64(vgpr(addr, 2), vgpr(addr, 2), sgpr(offsetSgpr, 2)))
            else:
                mod.add(VAddCOU32(vgpr(addr), VCC(), vgpr(addr), sgpr(offsetSgpr)))
                mod.add(VAddCCOU32(vgpr(f"{addr}+1"), VCC(), vgpr(f"{addr}+1"), 0, VCC()))
        return mod

    def incrementAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        return self.addOffset(kernel, tp, f"GL2PrefetchInc{tp['tensorChar']}", self.numIncSgpr(kernel) == 2)
    
    def skipPGR(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        """Skip PGR loads.

        PGR tiles are already loaded into vgpr/lds, no need to load it into cache again.
        """
        mod = Module()
        tc: str = tp["tensorChar"]
        inc = sgpr(f"GL2PrefetchInc{tc}")
        is64: bool = self.numIncSgpr(kernel) == 2
        pgr = kernel["PrefetchGlobalRead"]
        if pgr > 0:
            if pgr > 1:
                with writer.allocTmpSgpr(3 if is64 else 2, 2) as tmpSgprRes:
                    tmpSgprIdx0 = tmpSgprRes.idx
                    tmpSgprIdx1 = tmpSgprRes.idx + 1
                    mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                        sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx1),
                        inc, pgr, comment="*= PGR"))
                    if is64:
                        tmpSgprIdx2 = tmpSgprRes.idx + 2
                        mod.add(SMulI32(sgpr(tmpSgprIdx2), sgpr(f"GL2PrefetchInc{tc}+1"), pgr, \
                            comment="inc.hi *= PGR"))
                        mod.add(SAddU32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx2)))
                    mod.addModuleAsFlatItems(self.addOffset(kernel, tp, tmpSgprIdx0, True))
            else:
                mod.addModuleAsFlatItems(self.incrementAddr(writer, kernel, tp))
        return mod
