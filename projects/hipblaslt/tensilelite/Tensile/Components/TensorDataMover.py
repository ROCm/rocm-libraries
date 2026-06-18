from ..Component import TensorDataMover
from ..Common.DataType import DataType
from ..Common import INDEX_CHARS
from typing import Mapping, Optional
from rocisa.code import Module
from rocisa.instruction import SMovB32, SMovB64, SOrB32, SAndB32, SLShiftLeftB32, SLShiftLeftB64, \
    SLShiftRightB32, SAddU32, SAddCU32, SMulI32, TensorLoadToLds, VReadfirstlaneB32, SMulLOU32
from rocisa.container import sgpr, vgpr, RegisterContainer, MemTokenData
from rocisa.functions import scalarMultiply64Bpe
from math import log2, ceil, prod
# from ..KernelWriterAssembly import KernelWriterAssembly

class TensorDataMoverLoad(TensorDataMover):
    kernel = {"TDMInst": 3}
    asmCaps = {"HasTDM": True}
    GROUP0_NUM_SGPR = 4
    GROUP1_NUM_SGPR = 8
    GROUP2_NUM_SGPR = 4
    GROUP3_NUM_SGPR = 4
    mem_token = None

    def setMemToken(self, mem_token: list[int]):
        self.mem_token = mem_token

    def __call__(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping):
        pass

    def calculateStartAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, sgprAddr: int | str) -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        tlu: int = tp["tlu"]
        tIdx: int = 0 if tp["isA"] else 1
        if kernel["ProblemType"]["Sparse"] and tp["isM"]:
            # Metadata follows the sparse tensor's free dimension, but A/B data tensors
            # keep their normal A->WG0 and B->WG1 mapping. Remapping data tensors here
            # would request nonexistent strides such as StrideA1J or StrideB0I.
            tIdx = 0 if kernel["ProblemType"]["Sparse"] == 1 else 1
        bpe: float = tp["bpeGR"] if not tp["isM"] else 0.25
        assert bpe > 0, "bpe must > 0"
        tileStride: str | RegisterContainer = writer.strideRef(tc, tIdx)
        unrollSummation = [i for i in tp["ia"] if i in kernel["ProblemType"]["IndicesSummation"]]
        tdmSeparateStride: str | RegisterContainer = writer.strideRef(tc, unrollSummation[-1]) if tlu else writer.strideRef(tc, tIdx)
        if tp["isM"]:
            ia = kernel["ProblemType"]["IndexAssignmentsMetadata"]
            sgprStrideName: str = f"Stride{tc}{writer.states.indexChars[ia[1]]}"
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        vgprThreadIdName: str = "Serial"
        #TODO: temp hack
        numWaves: int = kernel["NumWaves"]
        wavelen: int = kernel["WavefrontSize"]
        mt: int = kernel["MacroTile0"] if tc == "A" else kernel["MacroTile1"]
        tdmSplit: int = 2 if (kernel["TDMSplit"] and not ("MXS" in tc) and not kernel["ProblemType"]["Sparse"]) else 1
        du: int = kernel["DepthU"]
        if "MXS" in tc:
            subTc0 = tc[3]
            depthU: int = kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc0}"]
        else:
            depthU: int = kernel["DepthU"]
        gsuOffsetBytes: int = round(depthU * bpe)

        mod.addComment(f"TDM calc start addr of {tc}")

        with writer.allocTmpSgpr(3, tag="TensorDataMoverLoad_tmpSgprRes") as tmpSgprRes:
            tmpSgprIdx = tmpSgprRes.idx
            waveOffsetSgprIdx = tmpSgprRes.idx + 2
            mod.add(SMovB64(sgpr(tmpSgprIdx, 2), 0))
            if tp['isM']:
                if not kernel["ProblemType"]["MetadataLayout"]:
                    mod.add(SMovB32(sgpr(tmpSgprIdx), sgpr("SizeL"), "stride = sizeL"))
                    mod.add(SLShiftRightB32(sgpr(tmpSgprIdx), hex(3), sgpr(tmpSgprIdx), "stride = SizeL / 2 (sparse) / 4 (bpe = 0.25)"))
                    mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), round(mt), f"stride *= MT({mt})"))
                else:
                    mod.add(SMovB32(sgpr(tmpSgprIdx), mt, f"stride = MT({mt})"))
            else:
                mod.add(SMulI32(sgpr(tmpSgprIdx), tileStride, round(mt * bpe), f"stride * MT({mt}) * bpe({bpe})"))
            mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx+1), sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), comment="*= wgId"))
            #add wave offset
            if tp['isM']:
                mod.add(VReadfirstlaneB32(sgpr(waveOffsetSgprIdx), vgpr(vgprThreadIdName), "first tId"))
                mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), ceil(log2(wavelen)), sgpr(waveOffsetSgprIdx), f"wId=fTid // {wavelen}"))
                if not kernel["ProblemType"]["MetadataLayout"]:
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numWaves), "woffset = wId * mt // numWaves"))
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr("SizeL"), f"woffset *= stride (SizeL / 8 for metadata)"))
                    mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), hex(3), sgpr(waveOffsetSgprIdx), "stride = SizeL / 2 (sparse) / 4 (bpe = 0.25)"))
                else:
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(du * bpe // 2 // numWaves), "woffset = wId * du * bpe / 2 (sparse) // numWaves"))
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr(sgprStrideName), f"woffset *= stride"))
            else:
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr("WaveIdx"), round(mt // numWaves * bpe // tdmSplit), "woffset = wId * mt // numWaves * bpe // tdmSplit"))
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tdmSeparateStride, f"woffset *= stride"))
            mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(waveOffsetSgprIdx), "+= woffset"))
            mod.add(SAddCU32(sgpr(tmpSgprIdx+1), sgpr(tmpSgprIdx+1), 0, "+= woffset carry"))
            #add GSU offset
            if kernel["GlobalSplitU"] > 0 or kernel["GlobalSplitU"] == -1:
                gsuOffsetSgprIdx = waveOffsetSgprIdx
                mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr("GSUSumIdx"), gsuOffsetBytes, f"gsuOffset = GSUSumIdx * DepthU({depthU}) * bpe({bpe})"))
                if "MXS" in tc:
                    mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr(gsuOffsetSgprIdx), sgpr(f"Size{INDEX_CHARS[tIdx]}"), f"MXS: scale GSU offset by tile size Size{INDEX_CHARS[tIdx]}"))
                elif tlu:
                    unrollStride = writer.strideRef(tc, unrollSummation[-1])
                    mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr(gsuOffsetSgprIdx), unrollStride, "tlu=1, scale GSU offset by unroll stride"))
                mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(gsuOffsetSgprIdx), "+= gsuOffset"))
            mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
            mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
            if kernel["ProblemType"]["Batched"]:
                if kernel["ProblemType"]["StridedBatched"]:
                    batchStrideName = f"Stride{tc}{writer.states.indexChars[tp['ia'][2]]}"
                    mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx+1), sgpr(batchStrideName), sgpr("WorkGroup2"), comment="Batch: Stride*WG"))
                    with writer.allocTmpSgpr(1, tag="TensorDataMoverLoad_tmpSgprBpe") as bpeTmp:
                        mod.add(scalarMultiply64Bpe(tmpSgprIdx, tmpSgprIdx, bpe, bpeTmp.idx, comment="scale by bpe"))
                    mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
                    mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
                else:
                    #TODO: support general batch
                    assert False, "Currently, TDM does not support general batch"
            #TODO: support stagger U
        return mod

    def calculateStartAddrWaveSeparated(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, sgprAddr: int | str, dstGroup0: str = None, waveIdxSgpr: int | str = "WaveIdx") -> Module:
        mod = Module()
        tc: str = tp["tensorChar"]
        tIdx: int = tp["idx"]
        bpe: float = tp["bpeGR"]
        tlu: int = tp["tlu"]
        assert bpe > 0, "bpe must > 0"
        tileStride: str | RegisterContainer = writer.strideRef(tc, tIdx)
        unrollSummation = [i for i in tp["ia"] if i in kernel["ProblemType"]["IndicesSummation"]]
        tdmSeparateStride: str | RegisterContainer = writer.strideRef(tc, unrollSummation[-1]) if tlu else writer.strideRef(tc, tIdx)
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        vgprThreadIdName: str = "Serial"
        #TODO: temp hack
        numWaves: int = kernel["NumWaves"]
        assert numWaves > 1
        wavelen: int = kernel["WavefrontSize"]
        mt: int = kernel["MacroTile0"] if tc.endswith("A") else kernel["MacroTile1"]
        du: int = kernel["DepthU"]
        tile1Size: int = du if tlu else mt
        tdmSplit: int = 2 if (kernel["TDMSplit"] and not ("MXS" in tc) and not kernel["ProblemType"]["Sparse"]) else 1
        if tlu and ((kernel["ProblemType"]["Sparse"] == 1 and tc.endswith("A")) or (kernel["ProblemType"]["Sparse"] == 2 and tc.endswith("B"))):
            tile1Size = tile1Size // 2
        if ("MXS" in tc):
            subTc = tc[3]
            mxUnit: int = kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]
        if "MXS" in tc:
            depthU: int = kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"]
        else:
            depthU: int = kernel["DepthU"]
        gsuOffsetBytes: int = round(depthU * bpe)

        mod.addComment(f"TDM wave separated calc start addr of {tc}")

        with writer.allocTmpSgpr(3, tag="TensorDataMoverLoadWaveSeparated_tmpSgprRes") as tmpSgprRes:
            numComp: int = numWaves // 2
            assert numComp & (numComp - 1) == 0, "numComp must be power of 2"
            tmpSgprIdx = tmpSgprRes.idx
            waveOffsetSgprIdx = tmpSgprRes.idx + 2
            mod.add(SMovB64(sgpr(tmpSgprIdx, 2), 0))
            if ("MXS" in tc):
                mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), round(mxUnit * mt * bpe), f"wgId * mxUnit({mxUnit}) * MT({mt}) * bpe({bpe})"))
            else:
                mod.add(SMulI32(sgpr(tmpSgprIdx), tileStride, round(mt * bpe), f"tileStride * MT({mt}) * bpe({bpe})"))
                mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx+1), sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), comment="*= wgId"))
            #add wave offset
            mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), 1, sgpr(waveIdxSgpr), f"wCompId = fTid // wavelen({wavelen}) // 2)"))
            if ("MXS" in tc):
                mxDU = kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"]
                numMxKGroups = mxDU // mxUnit
                if numMxKGroups >= numComp:
                    # K-splitting: offset by stride to next k_group
                    scale = numMxKGroups // numComp
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), (mxUnit * numMxKGroups // numComp), f"woffset = wCompId * mxUnit({mxUnit}) * numMxKGroups({numMxKGroups}) // numComp({numComp})"))
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr(f"Size{INDEX_CHARS[tIdx]}"), f"woffset *= Size{INDEX_CHARS[tIdx]}"))
                else:
                    # M/N-splitting: offset within same k_group along tile dimension
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numComp * mxUnit * bpe), f"woffset = wCompId * mt//numComp({mt // numComp}) * mxUnit({mxUnit}) * bpe({bpe})"))
            else:
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(tile1Size // numComp * bpe // tdmSplit), f"woffset = wCompId * mt // numComp({numComp}) * bpe({bpe}) // tdmSplit({tdmSplit})"))
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tdmSeparateStride, f"woffset *= tdmSeparateStride"))
            mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(waveOffsetSgprIdx), "+= woffset"))
            mod.add(SAddCU32(sgpr(tmpSgprIdx+1), sgpr(tmpSgprIdx+1), 0, "+= woffset carry"))
            #add GSU offset
            if kernel["GlobalSplitU"] > 0 or kernel["GlobalSplitU"] == -1:
                gsuOffsetSgprIdx = waveOffsetSgprIdx
                mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr("GSUSumIdx"), gsuOffsetBytes, f"gsuOffset = GSUSumIdx * DepthU({depthU}) * bpe({bpe})"))
                if "MXS" in tc:
                    mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr(gsuOffsetSgprIdx), sgpr(f"Size{INDEX_CHARS[tIdx]}"), f"MXS: scale GSU offset by tile size Size{INDEX_CHARS[tIdx]}"))
                elif tlu:
                    unrollStride = writer.strideRef(tc, unrollSummation[-1])
                    mod.add(SMulI32(sgpr(gsuOffsetSgprIdx), sgpr(gsuOffsetSgprIdx), unrollStride, "tlu=1, scale GSU offset by unroll stride"))
                mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(gsuOffsetSgprIdx), "+= gsuOffset"))
            if dstGroup0 is not None:
                mod.add(SAddU32(sgpr(f"{dstGroup0}+2"), sgpr(f"{dstGroup0}+2"), sgpr(tmpSgprIdx), "+= tileOffset(lo)"))
                mod.add(SAddCU32(sgpr(f"{dstGroup0}+3"), sgpr(f"{dstGroup0}+3"), sgpr(tmpSgprIdx+1), "+= tileOffset(hi)"))
            else:
                mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
                mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))

            if kernel["ProblemType"]["Batched"]:
                if kernel["ProblemType"]["StridedBatched"]:
                    batchStrideName = f"Stride{tc}{writer.states.indexChars[tp['ia'][2]]}"
                    mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx+1), sgpr(batchStrideName), sgpr("WorkGroup2"), comment="Batch: Stride*WG"))
                    with writer.allocTmpSgpr(1, tag="TensorDataMoverLoadWaveSeparated_tmpSgprBpe") as bpeTmp:
                        mod.add(scalarMultiply64Bpe(tmpSgprIdx, tmpSgprIdx, bpe, bpeTmp.idx, comment="scale by bpe"))
                    if dstGroup0 is not None:
                        # For wave-separated path: descriptor was set from base AddressA before this runs.
                        # Add batch offset directly to descriptor to match where tile offset goes.
                        mod.add(SAddU32(sgpr(f"{dstGroup0}+2"), sgpr(f"{dstGroup0}+2"), sgpr(tmpSgprIdx), "+= batchOffset(lo)"))
                        mod.add(SAddCU32(sgpr(f"{dstGroup0}+3"), sgpr(f"{dstGroup0}+3"), sgpr(tmpSgprIdx+1), "+= batchOffset(hi)"))
                    else:
                        mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
                        mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
                else:
                    #TODO: support general batch
                    assert False, "Currently, TDM does not support general batch"
            #TODO: support stagger U
        return mod

    def issueLoad(self, group0: int | str, group1: int | str, group2: Optional[int | str], group3: Optional[int | str]) -> Module:
        mod = Module("tensor load")
        if len(self.mem_token) > 1:
            comment = f"sync LDS {self.mem_token}"
        elif len(self.mem_token) == 1:
            comment = f"sync LDS%u"%(self.mem_token[0])
        else:
            comment = "No MemToken for TDM load"
        if all(g is not None for g in [group2, group3]):
            tensorLoadToLds = TensorLoadToLds(sgpr(group0, self.GROUP0_NUM_SGPR), sgpr(group1, self.GROUP1_NUM_SGPR),\
                                              sgpr(group2, self.GROUP2_NUM_SGPR), sgpr(group3, self.GROUP3_NUM_SGPR), comment=comment)
        else:
            tensorLoadToLds = TensorLoadToLds(sgpr(group0, self.GROUP0_NUM_SGPR), sgpr(group1, self.GROUP1_NUM_SGPR),\
                                              None, None, comment=comment)

        if self.mem_token is not None:
            tensorLoadToLds.setMemToken(MemTokenData(self.mem_token))

        mod.add(tensorLoadToLds)
        return mod

    def initOperands(self, group0: int | str, group1: int | str, group2: Optional[int | str], group3: Optional[int | str]) -> Module:
        mod = Module("tdmInit")
        for i in range(self.GROUP0_NUM_SGPR):
            val = 1 if i == 0 else 0
            mod.add(SMovB32(sgpr(f"{group0}+{i}"), val))

        mod.add(SOrB32(sgpr(f"{group0}+3"), sgpr(f"{group0}+3"), hex(2 << 30), "set type field to 2(image)"))

        for i in range(self.GROUP1_NUM_SGPR):
            mod.add(SMovB32(sgpr(f"{group1}+{i}"), 0))

        if group2 is not None:
            for i in range(self.GROUP2_NUM_SGPR):
                mod.add(SMovB32(sgpr(f"{group2}+{i}"), 0))

        if group3 is not None:
            for i in range(self.GROUP3_NUM_SGPR):
                mod.add(SMovB32(sgpr(f"{group3}+{i}"), 0))

        return mod

    @staticmethod
    def dataSizeShift(dtype: DataType, isMetadata: bool = False) -> int:
        if isMetadata or dtype.isInt8() or dtype.is8bitFloat() or dtype.isFloat4() or dtype.is6bitFloat():
            return 0
        if dtype.isBFloat16() or dtype.isHalf():
            return 1
        if dtype.isSingle():
            return 2
        if dtype.isDouble():
            return 3
        raise AssertionError(f"unsupported dtype for TDM data_size: {dtype}")

    def setDataType(self, dtype: DataType, group1: str | int, isMetadata: bool = False) -> Module:
        mod = Module()
        dataSizeOp = self.dataSizeShift(dtype, isMetadata)
        mod.add(SAndB32(sgpr(group1), sgpr(group1), hex(0xFFFCFFFF), "Reset data_size"))
        mod.add(SOrB32(sgpr(group1), sgpr(group1), hex(dataSizeOp << 16), f"Set data_size to {dataSizeOp}"))
        return mod

    def setLdsAddr(self, group0: int | str, ldsAddr: int | RegisterContainer) -> Module:
        mod = Module()
        mod.addComment("TDM set LDS addr")
        mod.add(SMovB32(sgpr(f"{group0}+1"), ldsAddr))
        return mod

    def getLdsAddrSgprName(self, group0: int | str) -> str:
        return f"{group0}+1"

    def setPadding(self, group1: int | str, ldsBlockSizePerPad: int, ldsPadSize: int) -> Module:
        mod = Module()
        if ldsBlockSizePerPad != 0 and ldsPadSize != 0:
          mod.addComment("TDM set padding")
          pad_interval = self.calPadInterval(ldsBlockSizePerPad)
          pad_amount = self.calPadAmount(ldsPadSize)
          value = hex((1 << 20) | (pad_interval << 22) | (pad_amount << 25))
          mod.add(SOrB32(sgpr(f"{group1}+0"), sgpr(f"{group1}+0"), value, f"set padding {ldsPadSize} per block {ldsBlockSizePerPad}"))
        return mod

    def setGlobalAddr(self, group0: int | str, sgprGlobalAddr: int | str) -> Module:
        mod = Module()
        mod.addComment("TDM set global addr")
        mod.add(SMovB64(sgpr(f"{group0}+2", 2), sgpr(sgprGlobalAddr, 2)))
        mod.add(SOrB32(sgpr(f"{group0}+3"), sgpr(f"{group0}+3"), hex(2 << 30), "set type field to 2(image)"))
        return mod

    def incrementGlobalAddr(self, writer: "KernelWriterAssembly", group0: int | str, sgprIncrement: int | str) -> Module:
        mod = Module()
        mod.addComment("TDM increment global addr")
        mod.add(SAddU32(sgpr(f"{group0}+2"), sgpr(f"{group0}+2"), sgpr(sgprIncrement), "TDM increment lo"))
        mod.add(SAddCU32(sgpr(f"{group0}+3"), sgpr(f"{group0}+3"), 0, "TDM increment hi (carry)"))
        return mod

    def setIterationEnabled(self, group1, enabled: bool) -> Module:
        mod = Module()
        if enabled:
            mod.add(SOrB32(sgpr(group1), sgpr(group1), hex(1 << 19),
                           "set iterate_enable (D# Group 1 bit 19)"))
        else:
            mod.add(SAndB32(sgpr(group1), sgpr(group1), hex(0xFFF7FFFF),
                            "clear iterate_enable (D# Group 1 bit 19)"))
        return mod

    def resetTensorDimForTail(self, group1: int | str, sgprTail: int, tdmDescIdx: int, writer: "KernelWriterAssembly", constShifter: int=0, isMXS: bool=False, isSparseTrack: bool=False) -> Module:
        mod = Module()
        mod.addComment("TDM reset tensor dim for tail")

        def descSgprName(idx: int) -> str:
          assert idx < 4
          return f"{group1}+{idx}"

        # Clear the previous data for L
        mod.add(SAndB32(sgpr(descSgprName(tdmDescIdx)), sgpr(descSgprName(tdmDescIdx)), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(descSgprName(tdmDescIdx+1)), sgpr(descSgprName(tdmDescIdx+1)), hex(0xFFFF0000)))

        with writer.allocTmpSgpr(1, tag="resetTensorDimForTail_tmpSgpr") as tmpSgpr:
            if isMXS:
                tmp = (1 << constShifter) - 1
                mod.add(SAddU32(sgpr(tmpSgpr.idx), sgpr(sgprTail), tmp))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx)), sgpr(descSgprName(tdmDescIdx)), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprTail)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx+1)), sgpr(descSgprName(tdmDescIdx+1)), sgpr(tmpSgpr.idx)))
            elif isSparseTrack:
                # sparse A/B: tensor_dim0 must be K/2 (compressed), divide SizeL by 2
                mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprTail)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx)), sgpr(descSgprName(tdmDescIdx)), sgpr(tmpSgpr.idx)))
                mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprTail)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx+1)), sgpr(descSgprName(tdmDescIdx+1)), sgpr(tmpSgpr.idx)))
            else:
              if constShifter:
                  mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprTail)))
                  mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
              else:
                  mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprTail)))
              mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx)), sgpr(descSgprName(tdmDescIdx)), sgpr(tmpSgpr.idx)))

              if constShifter:
                  mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprTail)))
                  mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
              else:
                  mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprTail)))
              mod.add(SOrB32(sgpr(descSgprName(tdmDescIdx+1)), sgpr(descSgprName(tdmDescIdx+1)), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorDim0(self, group1: int | str, sgprDim0: int | str, writer: "KernelWriterAssembly", constShifter: int=0, isMXS: bool=False, isSparseTrack: int=0, isMetadata: bool=False) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 0")
        mod.add(SAndB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0xFFFF0000)))

        with writer.allocTmpSgpr(1, tag="setTensorDim0_tmpSgpr") as tmpSgpr:
            if isMXS:
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
            elif isSparseTrack or isMetadata:
                # lo 16 bits
                mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprDim0)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                if isMetadata:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(2), sgpr(tmpSgpr.idx), "sizeL /= 4 for metadata (bpe = 0.25)"))
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
                # hi 16 bits
                mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprDim0)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                if isMetadata:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(2), sgpr(tmpSgpr.idx), "sizeL /= 4 for metadata (bpe = 0.25)"))
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
            else:
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                else:
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
                mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                else:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorDim1(self, group1: int | str, sgprDim1: int | str, writer: "KernelWriterAssembly", constShifter: int=0, isMXS: bool=False, isSparseTrack: int=0, isMetadata: bool=False) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 1")
        mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(0xFFFF0000)))
        with writer.allocTmpSgpr(1, tag="setTensorDim1_tmpSgpr") as tmpSgpr:
            if isMXS:
                tmp = (1 << constShifter) - 1
                mod.add(SAddU32(sgpr(tmpSgpr.idx), sgpr(sgprDim1), tmp))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim1)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), sgpr(tmpSgpr.idx)))
            else:
                if isSparseTrack or isMetadata:
                    # lo 16 bits
                    mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprDim1)))
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                    if isMetadata:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(2), sgpr(tmpSgpr.idx), "sizeL /= 4 for metadata (bpe = 0.25)"))
                    if constShifter:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                    mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
                    # hi 16 bits
                    mod.add(SMovB32(sgpr(tmpSgpr.idx), sgpr(sgprDim1)))
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(1), sgpr(tmpSgpr.idx), "sizeL /= 2 for sparse matrix"))
                    if isMetadata:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(2), sgpr(tmpSgpr.idx), "sizeL /= 4 for metadata (bpe = 0.25)"))
                    if constShifter:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(tmpSgpr.idx)))
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                    mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), sgpr(tmpSgpr.idx)))
                else:
                    if constShifter:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim1)))
                        mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                    else:
                        mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim1)))
                    mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
        
                    if constShifter:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim1)))
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                    else:
                        mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim1)))
                    mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorTile0(self, group1: int | str, tile0: int, writer: "KernelWriterAssembly", constShifter: int=0) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor tile 0")
        mod.add(SAndB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(0x0000FFFF)))
        mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(((tile0 >> constShifter) & 0xFFFF) << 16), f"set tile0 to {tile0 >> constShifter}"))
        return mod

    def setTensorTile1(self, group1: int | str, tile1, writer: "KernelWriterAssembly", constShifter: int=0) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor tile 1")
        mod.add(SAndB32(sgpr(f"{group1}+4"), sgpr(f"{group1}+4"), hex(0xFFFF0000)))
        mod.add(SOrB32(sgpr(f"{group1}+4"), sgpr(f"{group1}+4"), hex((tile1 >> constShifter) & 0xFFFF), f"set tile1 to {tile1 >> constShifter}"))
        return mod

    def setTensorStride0(self, group1: int | str, sgprStride0: int | str | RegisterContainer, constShifter: int=0, isMXS: bool=False) -> Module:
        mod = Module()
        if isMXS:
            mod.add(SLShiftLeftB32(sgpr(f"{group1}+5"), hex(constShifter), sgpr(sgprStride0)))
        elif constShifter:
            if isinstance(sgprStride0, RegisterContainer):
                mod.add(SLShiftRightB32(sgpr(f"{group1}+5"), hex(constShifter), sgprStride0))
            else:
                mod.add(SLShiftRightB32(sgpr(f"{group1}+5"), hex(constShifter), sgpr(sgprStride0)))
        else:
            if isinstance(sgprStride0, RegisterContainer):
                mod.add(SMovB32(sgpr(f"{group1}+5"), sgprStride0))
            else:
                mod.add(SMovB32(sgpr(f"{group1}+5"), sgpr(sgprStride0)))
        #TODO: support 48-bit stride
        mod.add(SMovB32(sgpr(f"{group1}+6"), 0))
        return mod

    def setTensorStride0Metadata(self, group1: int | str, sgprMetadataStride0I: str) -> Module:
        mod = Module()
        mod.add(SMovB32(sgpr(f"{group1}+5"), sgpr(sgprMetadataStride0I))) # Currently use SizeL for metadata
        mod.add(SLShiftRightB32(sgpr(f"{group1}+5"), hex(3), sgpr(f"{group1}+5"), "stride = SizeL / 2 (sparse) / 4 (bpe = 0.25)"))
        mod.add(SMovB32(sgpr(f"{group1}+6"), 0))
        return mod

    def setMulticastMask(self, group1: int | str, mask: str, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.add(SOrB32(sgpr(f"{group1}"), sgpr(f"{group1}"), sgpr(f"{mask}")))
        return mod

    def setIterationIncrements(self, group2: int | str, ldsInc: int, sgprGlobalInc: int | str) -> Module:
        mod = Module()
        mod.add(SMovB32(sgpr(f"{group2}+1"), hex(ldsInc), f"set lds increment to {ldsInc}"))
        mod.add(SMovB32(sgpr(f"{group2}+2"), sgpr(sgprGlobalInc)))
        #TODO: skip high 16-bit
        return mod

    def setIterations(self, group2: int | str, sgprNumIters: int | str) -> Module:
        mod = Module()
        mod.add(SMovB32(sgpr(f"{group2}+3"), sgpr(sgprNumIters)))
        mod.add(SLShiftLeftB32(sgpr(f"{group2}+3"), hex(16), sgpr(f"{group2}+3")))
        return mod

    @staticmethod
    def calPadInterval(ldsBlockSizePerPad: int) -> int:
        ldsBlockDwordsPerPad = ldsBlockSizePerPad // 4 # bytes to dwords
        assert ldsBlockDwordsPerPad > 0
        assert (ldsBlockDwordsPerPad & (ldsBlockDwordsPerPad - 1)) == 0, \
            f"LdsBlockSizePerPad//4 ({ldsBlockDwordsPerPad}) must be a power of 2 for TDM hardware encoding"
        return int(log2(ldsBlockDwordsPerPad)) - 1

    @staticmethod
    def calPadAmount(ldsPadSize: int) -> int:
        ldsPadDwords =  ldsPadSize // 4 # bytes to dwords
        assert ldsPadDwords > 0
        return ldsPadDwords - 1
