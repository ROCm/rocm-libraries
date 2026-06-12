from ..Component import GL2Prefetch
from ..Common import INDEX_CHARS
from typing import Mapping
from rocisa.code import Module
from rocisa.instruction import SMulI32, SAndB32, SLShiftLeftB32, SAddU64, VMovB32,\
    VAndB32, VAddU32, VAddCOU32, VAddCCOU32, VAddNCU64, VLShiftRightB32, VMulLOU32, VMulHIU32, VLShiftLeftB32,\
    GlobalPrefetchB8, SMovB64, VCmpGtU32, VCndMaskB32, SSubU32
from rocisa.container import sgpr, vgpr, RegisterContainer, VCC, GLOBALModifiers
from rocisa.functions import vectorMultiply64Bpe, scalarMultiplyBpe
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
        subTc: str = tc[-1]
        isMX: bool = tc.startswith("MX")
        mt: int = kernel["MacroTile%s" % subTc]
        numCooperativeWGs: int = kernel["ClusterDim"][1] if subTc == "A" else kernel["ClusterDim"][0]
        numCooperativeThreads: int = numCooperativeWGs * kernel["NumThreads"]
        
        if isMX:
            coalescedDim = mt * kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]
            perpendicularDim = kernel["DepthU"] // kernel["MatrixInstK"]
        else:
            coalescedDim, perpendicularDim = (mt, kernel["DepthU"]) if tp["tlu"] else (kernel["DepthU"], mt)

        tp["gl2ncp"] = perpendicularDim
        tp["gl2ncc"] = max(1, round(coalescedDim * tp["bpeGR"]) // globalPrefetchSize)
        tp["gl2nc"] = tp["gl2ncp"] * tp["gl2ncc"]
        tp["gl2nl"] = max(1, tp["gl2nc"] // numCooperativeThreads)
        tp["gl2nlc"] = max(1, tp["gl2ncc"] // numCooperativeThreads)
        tp["gl2nlp"] = tp["gl2nl"] // tp["gl2nlc"]

    def setIncrement(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        tIdx: int = tp['idx']
        subTc: str = tc[-1]
        bpe: float = tp["bpeGR"]
        if tc.startswith("MX"):
            mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                sgpr(f"GL2PrefetchInc{tc}"), sgpr(f"GL2PrefetchInc{tc}+1"),
                sgpr("Size%s"%INDEX_CHARS[tIdx]), round(kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"] * bpe),
                None, comment="addr increment"))
        elif tp["tlu"]:
            perpStride: str | RegisterContainer = writer.strideRef(subTc, 3)
            mod.addModuleAsFlatItems(writer.s_mul_u64_u32(
                sgpr(f"GL2PrefetchInc{tc}"), sgpr(f"GL2PrefetchInc{tc}+1"),
                perpStride, round(kernel["DepthU"] * bpe),
                None, comment="addr increment"))
        else:
            mod.add(SMovB64(dst=sgpr(f"GL2PrefetchInc{tc}", 2), src=round(kernel["DepthU"] * bpe), comment="addr increment"))
        return mod

    def calculateStartAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        globalPrefetchSize: int = writer.states.regCaps["GlobalPrefetchSize"]
        tc: str = tp["tensorChar"]
        subTc: str = tc[-1]
        tIdx: int = tp['idx']
        mt: int = kernel["MacroTile%s" % subTc]
        bpe: float = tp["bpeGR"]
        tlu: bool = tp["tlu"]
        isMX: bool = tc.startswith("MX")
        tileStride: str | RegisterContainer = writer.strideRef(subTc, tIdx)
        unrollStride: str | RegisterContainer = writer.strideRef(subTc, 3)
        perpStride: str | RegisterContainer = unrollStride if tlu else tileStride
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        sgprCooperativeWgName: str = f"WorkGroup{1 - tIdx}"
        sgprSizeFreeName: str = f"Size{INDEX_CHARS[tIdx]}"
        vgprThreadIdName: str = "Serial"
        numThreads: int = kernel["NumThreads"]
        vgprAddrBaseName: str = f"GL2PrefetchAddr{tc}"
        vgprAddrName: str = f"GL2PrefetchAddr{tc}_0_0"
        numCooperativeWGs: int = kernel["ClusterDim"][1] if subTc == "A" else kernel["ClusterDim"][0]
        numCooperativeThreads: int = numCooperativeWGs * numThreads
        ncc: int = tp["gl2ncc"]
        nc: int = tp["gl2nc"]
        ncInOneInst: int = nc // tp["gl2nl"]
        nccInOneInst: int = min(ncc, ncInOneInst)
        ncpInOneInst: int = ncInOneInst // nccInOneInst
        inactiveShiftBits: int = ceil(log2(numCooperativeThreads // ncInOneInst))
        numTmpSgpr = 4
        tmpVgprIdx = writer.vgprPool.checkOutAligned(2, 2)
        tmpVgprCoalIdx = []
        for i in range(tp["gl2nlc"]):
            tmpVgprCoalIdx.append(writer.vgprPool.checkOutAligned(1, 1))
        if isMX:
            mxUnit: int = kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]
        

        mod.addComment(f"gl2 prefetch calc start addr of {tc}")
        with writer.allocTmpSgpr(numTmpSgpr) as tmpSgprRes:
            tmpSgprIdx0 = tmpSgprRes.idx
            tmpSgprIdx1 = tmpSgprRes.idx + 1
            tmpSgprIdx2 = tmpSgprRes.idx + 2
            tmpSgprIdx3 = tmpSgprRes.idx + 3
            # will we have MX stride later?
            if isMX:
                perpStride = sgpr(tmpSgprIdx2)
                mod.add(SMulI32(perpStride, sgpr(sgprSizeFreeName), mxUnit, f"MX perp stride"))
            # offset inside MT
            mod.add(SAndB32(sgpr(tmpSgprIdx0), sgpr(sgprCooperativeWgName), numCooperativeWGs-1, \
                comment="WG index in cluster"))
            mod.add(SLShiftLeftB32(sgpr(tmpSgprIdx0), ceil(log2(numThreads)), sgpr(tmpSgprIdx0), \
                comment="cooperative thread idx"))
            mod.add(VAddU32(vgpr(vgprAddrName), vgpr(vgprThreadIdName), sgpr(tmpSgprIdx0), \
                comment="cooperative thread idx"))
            if inactiveShiftBits > 0:
                mod.add(VLShiftRightB32(vgpr(vgprAddrName), inactiveShiftBits, vgpr(vgprAddrName), \
                    comment="shift inactive index"))
            if ncc > 1:
                assert ncc & (ncc - 1) == 0, "gl2ncc must be power of 2"
                mod.add(VAndB32(vgpr(tmpVgprCoalIdx[0]), vgpr(vgprAddrName), ncc-1, \
                    comment="coalesced index"))
                mod.add(VLShiftLeftB32(vgpr(tmpVgprCoalIdx[0]), \
                    ceil(log2(globalPrefetchSize // bpe)), vgpr(tmpVgprCoalIdx[0]), \
                    comment=f" *= {globalPrefetchSize // bpe} (elements)"))
                mod.add(VLShiftRightB32(vgpr(vgprAddrName), ceil(log2(ncc)), vgpr(vgprAddrName), \
                    comment="perpendicular index"))
            else:
                mod.add(VMovB32(vgpr(tmpVgprCoalIdx[0]), 0, comment="coalesced index"))
            # WG MT offset
            if isMX:
                mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(sgprWorkgroupName), mxUnit * mt, \
                    comment=f"wgId * mxUnit({mxUnit}) * MT({mt})"))
                mod.add(VAddU32(vgpr(tmpVgprCoalIdx[0]), vgpr(tmpVgprCoalIdx[0]), sgpr(tmpSgprIdx0), \
                    comment="coal += MT offset"))
            else:
                mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(sgprWorkgroupName), mt, \
                    comment=f"wgId * MT({mt})"))
                if tlu:
                    mod.add(VAddU32(vgpr(tmpVgprCoalIdx[0]), vgpr(tmpVgprCoalIdx[0]), sgpr(tmpSgprIdx0), \
                        comment="coal += MT offset"))
                else:
                    mod.add(VAddU32(vgpr(vgprAddrName), vgpr(vgprAddrName), sgpr(tmpSgprIdx0), \
                        comment="perp += MT offset"))
            # edge limit
            mod.add(SSubU32(sgpr(tmpSgprIdx0), sgpr(sgprSizeFreeName), 1, comment="edge limit"))
            if isMX:
                mod.add(SMulI32(sgpr(tmpSgprIdx0), sgpr(tmpSgprIdx0), mxUnit, comment="edge limit *= mxUnit"))
            # Inst offset    
            # coal
            for j in range(tp["gl2nlc"]):
                dst = tmpVgprCoalIdx[j]
                if j > 0:
                    src = tmpVgprCoalIdx[j-1]
                    mod.add(VAddU32(vgpr(dst), vgpr(src), round(nccInOneInst * globalPrefetchSize / bpe)))
                if isMX or tlu:
                    # edge protection
                    mod.add(VCmpGtU32(VCC(), vgpr(dst), sgpr(tmpSgprIdx0), comment="> edge limit?"))
                    mod.add(VCndMaskB32(vgpr(dst), vgpr(dst), sgpr(tmpSgprIdx0), VCC()))
            # perp
            for i in range(tp["gl2nlp"]):
                dst = f"{vgprAddrBaseName}_{i}_0"
                if i > 0:
                    src = f"{vgprAddrBaseName}_{(i-1)}_0"
                    mod.add(VAddU32(vgpr(dst), vgpr(src), ncpInOneInst))
                if not (isMX or tlu):
                    mod.add(VCmpGtU32(VCC(), vgpr(dst), sgpr(tmpSgprIdx0), comment="> edge limit?"))
                    mod.add(VCndMaskB32(vgpr(dst), vgpr(dst), sgpr(tmpSgprIdx0), VCC()))
            for i in range(tp["gl2nlp"]):
                dst = f"{vgprAddrBaseName}_{i}_0"
                dstHi = dst + "+1"
                mod.add(VMulHIU32(vgpr(dstHi), vgpr(dst), perpStride, comment="perp *= stride"))
                mod.add(VMulLOU32(vgpr(dst), vgpr(dst), perpStride))
            # coal + perp & scale by bpe
            for i in range(tp["gl2nlp"]-1, -1, -1):
                for j in range(tp["gl2nlc"]-1, -1, -1):
                    dst = f"{vgprAddrBaseName}_{i}_{j}"
                    dstHi = dst + "+1"
                    srcCoal = tmpVgprCoalIdx[j]
                    srcPerp = f"{vgprAddrBaseName}_{i}_0"
                    srcPerpHi = srcPerp + "+1"
                    mod.add(VAddCOU32(vgpr(dst), VCC(), vgpr(srcPerp), vgpr(srcCoal), comment="coal + perp"))
                    mod.add(VAddCCOU32(vgpr(dstHi), VCC(), vgpr(srcPerpHi), 0, VCC()))
                    mod.add(vectorMultiply64Bpe(dst, dst, bpe, tmpVgprIdx, comment="scale by bpe"))
            # base address
            mod.add(SMovB64(sgpr(tmpSgprIdx0, 2), sgpr("Address%s"%tc, 2), comment="base address"))
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
                    mod.add(SAddU64(sgpr(tmpSgprIdx0, 2), sgpr(tmpSgprIdx0, 2), sgpr(f"GL2PrefetchInc{tc}", 2), \
                        comment="skip PGR loads"))

            # add all together
            for i in range(tp["gl2nlp"]):
                for j in range(tp["gl2nlc"]):
                    dst = f"{vgprAddrBaseName}_{i}_{j}"
                    mod.add(VAddNCU64(vgpr(dst, 2), vgpr(dst, 2), sgpr(tmpSgprIdx0, 2)))

        writer.vgprPool.checkIn(tmpVgprIdx)
        for vgprIdx in tmpVgprCoalIdx:
            writer.vgprPool.checkIn(vgprIdx)
        return mod

    def issueLoad(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        for i in range(tp["gl2nlp"]):
            for j in range(tp["gl2nlc"]):
                addrName = f"GL2PrefetchAddr{tc}_{i}_{j}"
                mod.add(GlobalPrefetchB8(vgpr(addrName, 2), sgpr("off", isOff=True), self.globalModifiers))
        return mod

    def incrementAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        inc = sgpr(f"GL2PrefetchInc{tc}", 2)
        for i in range(tp["gl2nlp"]):
            for j in range(tp["gl2nlc"]):
                addrName = f"GL2PrefetchAddr{tc}_{i}_{j}"
                mod.add(VAddNCU64(vgpr(addrName, 2), vgpr(addrName, 2), inc))

        return mod