from ..Component import GL2Prefetch
from ..Common import INDEX_CHARS
from typing import Mapping
from rocisa.code import Module
from rocisa.instruction import SMulI32, SAddU64, VMovB32, VAddU32, VAddCOU32, \
    VAddCCOU32, VAddNCU64, VLShiftRightB32, VMulLOU32, VMulHIU32, GlobalPrefetchB8, \
    VCmpGtU32, VCndMaskB32, SSubI32, SMovB32, SAddU32, SAddCU32
from rocisa.container import sgpr, vgpr, RegisterContainer, VCC, GLOBALModifiers, ContinuousRegister
from rocisa.functions import vectorMultiply64Bpe, scalarMultiplyBpe, vectorStaticDivideAndRemainder, \
    scalarStaticRemainder
from rocisa.enum import TemporalHint, CacheScope
from math import log2, ceil

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

    def setIncrement(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
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

    def calculateStartAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
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

    def issueLoad(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        for i in range(tp["gl2nl"]):
            addrName = f"GL2PrefetchAddr{tc}_{i}"
            mod.add(GlobalPrefetchB8(vgpr(addrName, 2), sgpr("off", isOff=True), self.globalModifiers))
        return mod

    def incrementAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        inc = sgpr(f"GL2PrefetchInc{tc}")
        for i in range(tp["gl2nl"]):
            addrName = f"GL2PrefetchAddr{tc}_{i}"
            addrNameHi = addrName + "+1"
            mod.add(VAddCOU32(vgpr(addrName), VCC(), vgpr(addrName), inc))
            mod.add(VAddCCOU32(vgpr(addrNameHi), VCC(), vgpr(addrNameHi), 0, VCC()))

        return mod