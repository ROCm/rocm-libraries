from ..Component import GL2Prefetch
from ..Common import INDEX_CHARS
from typing import Mapping, Optional
from rocisa.code import Module, Label
from rocisa.instruction import SMulI32, SAddU64, VMovB32, VAddU32, VAddCOU32, \
    VAddCCOU32, VAddNCU64, VLShiftRightB32, VMulLOU32, VMulHIU32, GlobalPrefetchB8, \
    VCmpGtU32, VCndMaskB32, SSubI32, SMovB32, SAddU32, SAddCU32, SAndB32, SBranch, \
    SCBranchSCC1, SCMovB32, SLShiftRightB32, SSubU32, SCmpEQU32, SCmpGeU32, \
    SCmpLeU32, SCSelectB32
from rocisa.container import sgpr, vgpr, RegisterContainer, VCC, GLOBALModifiers, ContinuousRegister
from rocisa.functions import vectorMultiply64Bpe, scalarMultiplyBpe, vectorStaticDivideAndRemainder, \
    scalarStaticRemainder
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
        tp["gl2ncc"] = max(1, round(coalescedDim * bpe) // globalPrefetchSize)
        tp["gl2nc"] = tp["gl2ncp"] * tp["gl2ncc"]
        tp["gl2nl"] = max(1, ceil(tp["gl2nc"] / numCooperativeThreads))

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

        mod.addComment(f"gl2 prefetch GSU chunk offset of {tc}")
        mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
            sgpr(tmpSgprIdx), sgpr(tmpSgprIdx + 1),
            sgpr(gsuIterSgpr), sgpr(incName),
            tmpVgprIdx, comment="gsuOffset = startIter * inc"))
        mod.add(SAddU64(sgpr(baseSgprIdx, 2), sgpr(baseSgprIdx, 2), sgpr(tmpSgprIdx, 2), \
            comment="skip to this WG's GSU chunk"))
        # Widen the step to the chunk stride. Kept 32-bit to mirror GlobalReadIncs
        # on the real load path (GSU.graIncrements), which the prefetch has to
        # track: a stride that overflows 32 bits is already broken there.
        mod.add(SAndB32(dst=sgpr(tmpSgprIdx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), \
            comment="Restore GSU"))
        mod.add(SAndB32(dst=sgpr(tmpSgprIdx + 1), src0=sgpr("GSU"), src1=hex(GSUC_BIT), \
            comment="SCC = (GSUC == 1) ?"))
        mod.add(SCMovB32(dst=sgpr(tmpSgprIdx), src=1, comment="stride stays DepthU if GSUC == 1"))
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
        if tc.startswith("MX"):
            mod.add(SMulI32(sgpr(f"GL2PrefetchInc{tc}"), sgpr("Size%s"%INDEX_CHARS[tIdx]), \
                round(kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"] * bpe), comment="addr increment"))
        elif tp["tlu"]:
            perpStride: str | RegisterContainer = writer.strideRef(subTc, 3)
            mod.add(SMulI32(sgpr(f"GL2PrefetchInc{tc}"), perpStride, round(du * bpe), comment="addr increment"))
        else:
            mod.add(SMovB32(dst=sgpr(f"GL2PrefetchInc{tc}"), src=round(du * bpe), comment="addr increment"))
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

            # skip PGR loads (uses GSU-adjusted increment)
            if kernel["PrefetchGlobalRead"] > 0:
                if kernel["PrefetchGlobalRead"] > 1:
                    mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                        sgpr(tmpSgprIdx2), sgpr(tmpSgprIdx3),
                        sgpr(f"GL2PrefetchInc{tc}"), kernel["PrefetchGlobalRead"],
                        tmpVgprIdx, comment="*= PGR"))
                    mod.add(SAddU64(sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx2, 2), \
                        comment="skip PGR loads"))
                else:
                    mod.add(SAddU32(sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx0), sgpr(f"GL2PrefetchInc{tc}"), \
                        comment="skip PGR loads"))
                    mod.add(SAddCU32(sgpr(tmpSgprIdx1), sgpr(tmpSgprIdx1), 0, \
                        comment="skip PGR loads"))

            # add all together
            for i in range(tp["gl2nl"]):
                dst = f"{vgprAddrBaseName}_{i}"
                mod.add(VAddNCU64(vgpr(dst, 2), vgpr(dst, 2), sgpr(tmpSgprIdx0, 2)))

        writer.vgprPool.checkIn(tmpVgprIdx)
        writer.vgprPool.checkIn(tmpVgprCoalIdx)
        return mod

    # ----------------------------------------------------------------------
    # StaggerU. The unroll loop is rotated so that step j of the summation reads
    # iteration (StaggerUIter + j) % numIter instead of j, which spreads the K
    # start of each workgroup and keeps them off the same cache lines. The
    # prefetch stream has to follow the same rotation or it would warm lines the
    # loads never touch, so it needs the two pieces the real load stream uses:
    #   - a start shift onto the rotated position (staggerStartIterDelta), and
    #   - a one-off wrap step that jumps from the last iteration back to the
    #     first when the rotation rolls over (incrementAddr).
    # The prefetch runs PrefetchGL2 steps ahead of the real loads, which is the
    # only thing that separates its wrap point from theirs; see incrementAddr.
    # The wrap distance itself is not one of them: calculateStagger's WrapU{tc}
    # is the same byte count over the same span of K, so it is reused directly.
    # ----------------------------------------------------------------------

    def staggerStartIterDelta(self, writer: "KernelWriterAssembly", kernel: Mapping, \
                              dstSgprIdx: int, tmpSgprIdx: int) -> Module:
        """Unroll iterations to shift the prefetch start by so it lands on the
        rotated K position.

        calculateStartAddr left the prefetch at unroll position PrefetchGlobalRead
        of an unrotated loop. Rotating by StaggerUIter moves it that many
        iterations further, except when the rotation has already carried that
        position past the end of K, where it lands numIter earlier instead. The
        result counts iterations, so it is tensor independent and each tensor
        scales it by its own increment.
        """
        mod = Module()
        numIter = writer.loopCounter(kernel, writer.states.unrollIdx)
        pgr: int = kernel["PrefetchGlobalRead"]

        mod.addComment("gl2 prefetch stagger start offset")
        mod.add(SAddU32(dst=sgpr(tmpSgprIdx), src0=sgpr("StaggerUIter"), src1=pgr, \
            comment="start position after the PGR pre-skip"))
        mod.add(SSubU32(dst=sgpr(dstSgprIdx), src0=sgpr("StaggerUIter"), src1=numIter, \
            comment="rolled-over shift = StaggerUIter - numIter"))
        mod.add(SCmpGeU32(src0=sgpr(tmpSgprIdx), src1=numIter, \
            comment="does the rotated start run past the end of K?"))
        mod.add(SCSelectB32(dst=sgpr(dstSgprIdx), src0=sgpr(dstSgprIdx), src1=sgpr("StaggerUIter"), \
            comment="startIter shift"))
        # StaggerUIter is 0 when StaggerU is off at runtime. Without this the
        # roll-over select above would still fire on a loop shorter than PGR and
        # drag the start below the tensor base.
        mod.add(SCmpEQU32(src0=sgpr("StaggerUIter"), src1=0, comment="StaggerU off?"))
        mod.add(SCMovB32(dst=sgpr(dstSgprIdx), src=0, comment="no rotation, keep the plain start"))
        return mod

    def applyStaggerStart(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, \
                          deltaIterSgpr: int, tmpSgprIdx: int) -> Module:
        """Move this tensor's prefetch addresses onto the rotated K start."""
        mod = Module()
        tc: str = tp["tensorChar"]

        mod.addComment(f"gl2 prefetch stagger start of {tc}")
        mod.addModuleAsFlatItems(writer.s_mul_i64_i32(
            sgpr(tmpSgprIdx), sgpr(tmpSgprIdx + 1),
            sgpr(deltaIterSgpr), sgpr(f"GL2PrefetchInc{tc}"), "stagger byte offset"))
        for i in range(tp["gl2nl"]):
            addrName = f"GL2PrefetchAddr{tc}_{i}"
            mod.add(VAddNCU64(vgpr(addrName, 2), vgpr(addrName, 2), sgpr(tmpSgprIdx, 2)))
        return mod

    def issueLoad(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        for i in range(tp["gl2nl"]):
            addrName = f"GL2PrefetchAddr{tc}_{i}"
            mod.add(GlobalPrefetchB8(vgpr(addrName, 2), sgpr("off", isOff=True), self.globalModifiers))
        return mod

    def incrementAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tpList: list, \
                      staggerWrapOffset: Optional[int] = None, \
                      freezeIter: Optional[int] = None) -> Module:
        """Advance every tensor's prefetch addresses by one unroll iteration.

        staggerWrapOffset turns on the StaggerU rotation: on the one iteration
        where the rotation rolls over, the step becomes WrapU instead. That is
        calculateStagger's own register -- the prefetch rolls over the same span
        of K with the same increment as the real loads, so the byte count is the
        same one. The real load stream wraps when the loop counter reaches
        StaggerUIter; the prefetch is staggerWrapOffset iterations further along
        the same stream, so it reaches the roll-over that many iterations of the
        counter earlier.

        freezeIter stops the streams at the end of K. Where that stop lands is
        the only thing the rotation changes about it. Without the rotation
        nothing reads GL2PrefetchInc{tc} afterwards, so the stop is a one-way
        latch on the increment registers themselves and a single compare covers
        every tensor; later iterations just re-zero registers that already hold
        0. Under the rotation those same registers still feed the roll-over
        step, so the stop cannot land on them and each tensor takes it on its
        own temporary instead, after the select, so that it also swallows the
        roll-over the rotation would otherwise take on the very iteration the
        stream stops.
        """
        mod = Module()
        loopCounter = writer.loopCounter(kernel, writer.states.unrollIdx)

        if staggerWrapOffset is None:
            if freezeIter is not None:
                mod.add(SCmpLeU32(src0=loopCounter, src1=freezeIter, comment="counterL<=PGR+GL2"))
                for tp in tpList:
                    mod.add(SCMovB32(dst=sgpr(f"GL2PrefetchInc{tp['tensorChar']}"), src=0, \
                        comment="stop at the end of K"))
            for tp in tpList:
                tc: str = tp["tensorChar"]
                inc = sgpr(f"GL2PrefetchInc{tc}")
                for i in range(tp["gl2nl"]):
                    addrName = f"GL2PrefetchAddr{tc}_{i}"
                    addrNameHi = addrName + "+1"
                    mod.add(VAddCOU32(vgpr(addrName), VCC(), vgpr(addrName), inc))
                    mod.add(VAddCCOU32(vgpr(addrNameHi), VCC(), vgpr(addrNameHi), 0, VCC()))
            return mod

        for tp in tpList:
            tc: str = tp["tensorChar"]
            with writer.allocTmpSgpr(2, 2, tag="gl2PrefetchIncrementAddr_stagger") as tmpSgprRes:
                incLo: int = tmpSgprRes.idx
                incHi: int = tmpSgprRes.idx + 1
                # incHi doubles as scratch for the compare operand; the select below
                # overwrites it only after SCC has been set.
                mod.add(SAddU32(dst=sgpr(incHi), src0=sgpr("StaggerUIter"), src1=staggerWrapOffset, \
                    comment="counter value at which the prefetch rolls over"))
                mod.add(SCmpEQU32(src0=loopCounter, src1=sgpr(incHi), comment="Is this the wrapIter?"))
                mod.add(SCSelectB32(dst=sgpr(incLo), src0=sgpr(f"WrapU{tc}+0"), \
                    src1=sgpr(f"GL2PrefetchInc{tc}"), comment="select WrapU or normal inc (lo)"))
                mod.add(SCSelectB32(dst=sgpr(incHi), src0=sgpr(f"WrapU{tc}+1"), src1=0, \
                    comment="select WrapU or normal inc (hi)"))
                if freezeIter is not None:
                    mod.add(SCmpLeU32(src0=loopCounter, src1=freezeIter, comment="counterL<=PGR+GL2"))
                    mod.add(SCMovB32(dst=sgpr(incLo), src=0, comment="stop at the end of K"))
                    mod.add(SCMovB32(dst=sgpr(incHi), src=0, comment="stop at the end of K"))
                for i in range(tp["gl2nl"]):
                    addrName = f"GL2PrefetchAddr{tc}_{i}"
                    mod.add(VAddNCU64(vgpr(addrName, 2), vgpr(addrName, 2), sgpr(incLo, 2)))
        return mod