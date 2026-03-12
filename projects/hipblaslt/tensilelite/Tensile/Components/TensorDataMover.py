from ..Component import TensorDataMover
from ..Common.DataType import DataType
from ..Common import INDEX_CHARS
from typing import Mapping, Optional
from rocisa.code import Module
from rocisa.instruction import SMovB32, SMovB64, SOrB32, SAndB32, SLShiftLeftB32, \
    SLShiftRightB32, SAddU32, SAddCU32, SMulI32, TensorLoadToLds, VReadfirstlaneB32, SMulLOU32
from rocisa.container import sgpr, vgpr, RegisterContainer, MemTokenData
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
        #TODO here we assume TN
        mod = Module()
        tc: str = tp["tensorChar"]
        tlu: int = tp["tlu"]
        tIdx: int = 0 if tp["isA"] else 1
        bpe: float = tp["bpeGR"]
        assert bpe > 0, "bpe must > 0"
        tileStride: str | RegisterContainer = writer.strideRef(tc, tIdx)
        tdmSeparateStride: str | RegisterContainer = writer.strideRef(tc, 3) if tlu else writer.strideRef(tc, tIdx)
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        vgprThreadIdName: str = "Serial"
        #TODO: temp hack
        numWaves: int = prod(kernel["MIWaveGroup"])
        wavelen: int = kernel["WavefrontSize"]
        mt: int = kernel["MacroTile0"] if tc == "A" else kernel["MacroTile1"]
        tdmSplit: int = 2 if (kernel["TDMSplit"] and not ("MXS" in tc)) else 1

        mod.addComment(f"TDM calc start addr of {tc}")

        with writer.allocTmpSgpr(3) as tmpSgprRes:
            tmpSgprIdx = tmpSgprRes.idx
            waveOffsetSgprIdx = tmpSgprRes.idx + 2
            mod.add(SMovB64(sgpr(tmpSgprIdx, 2), 0))
            mod.add(SMulI32(sgpr(tmpSgprIdx), tileStride, round(mt * bpe), f"stride * MT({mt}) * bpe({bpe})"))
            mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), "*= wgId)"))
            #add wave offset
            mod.add(VReadfirstlaneB32(sgpr(waveOffsetSgprIdx), vgpr(vgprThreadIdName), "first tId"))
            mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), ceil(log2(wavelen)), sgpr(waveOffsetSgprIdx), f"wId=fTid // {wavelen}"))
            mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numWaves * bpe // tdmSplit), "woffset = wId * mt // numWaves * bpe // tdmSplit"))
            mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tdmSeparateStride, f"woffset *= stride"))
            mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(waveOffsetSgprIdx), "+= woffset"))
            mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
            mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
            #TODO: support strided batch
            #TODO: support GSU
            #TODO: support stagger U
        return mod

    def calculateStartAddrWaveSeparated(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, sgprAddr: int | str, dstGroup0: str = None) -> Module:
        #TODO here we assume TN
        mod = Module()
        tc: str = tp["tensorChar"]
        tIdx: int = tp["idx"]
        bpe: float = tp["bpeGR"]
        tlu: int = tp["tlu"]
        assert bpe > 0, "bpe must > 0"
        tileStride: str | RegisterContainer = writer.strideRef(tc, tIdx)
        tdmSeparateStride: str | RegisterContainer = writer.strideRef(tc, 3) if tlu else writer.strideRef(tc, tIdx)
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        vgprThreadIdName: str = "Serial"
        #TODO: temp hack
        numWaves: int = prod(kernel["MIWaveGroup"])
        assert numWaves > 1
        wavelen: int = kernel["WavefrontSize"]
        mt: int = kernel["MacroTile0"] if tc.endswith("A") else kernel["MacroTile1"]
        du: int = kernel["DepthU"]
        tile1Size: int = du if tlu else mt
        tdmSplit: int = 2 if (kernel["TDMSplit"] and not ("MXS" in tc)) else 1
        if ("MXS" in tc):
            subTc = tc[3]
            mxUnit: int = kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]

        mod.addComment(f"TDM wave separated calc start addr of {tc}")

        with writer.allocTmpSgpr(3) as tmpSgprRes:
            numComp: int = numWaves // 2
            assert numComp & (numComp - 1) == 0, "numComp must be power of 2"
            tmpSgprIdx = tmpSgprRes.idx
            waveOffsetSgprIdx = tmpSgprRes.idx + 2
            mod.add(SMovB64(sgpr(tmpSgprIdx, 2), 0))
            if ("MXS" in tc):
                mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), round(mxUnit * mt * bpe), f"wgId * mxUnit({mxUnit}) * MT({mt}) * bpe({bpe})"))
            else:
                mod.add(SMulI32(sgpr(tmpSgprIdx), tileStride, round(mt * bpe), f"tileStride * MT({mt}) * bpe({bpe})"))
                mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), "*= wgId)"))
            #add wave offset
            mod.add(VReadfirstlaneB32(sgpr(waveOffsetSgprIdx), vgpr(vgprThreadIdName), "first tId"))
            mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), ceil(log2(wavelen*numComp)), sgpr(waveOffsetSgprIdx), f"wCompId = fTid // wavelen({wavelen}) // numComp({numComp})"))
            if ("MXS" in tc):
                mxDU = kernel["DepthU"] // kernel["ProblemType"][f"MXBlock{subTc}"]
                numMxKGroups = mxDU // mxUnit
                if numMxKGroups >= numComp:
                    # K-splitting: offset by stride to next k_group
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), mxUnit, f"woffset = wCompId * mxUnit({mxUnit})"))
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr("Size%s"%INDEX_CHARS[tIdx]), f"woffset *= Size{INDEX_CHARS[tIdx]}"))
                else:
                    # M/N-splitting: offset within same k_group along tile dimension
                    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numComp * mxUnit * bpe), f"woffset = wCompId * mt//numComp({mt // numComp}) * mxUnit({mxUnit}) * bpe({bpe})"))
            else:
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(tile1Size // numComp * bpe // tdmSplit), f"woffset = wCompId * mt // numComp({numComp}) * bpe({bpe}) // tdmSplit({tdmSplit})"))
                mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tdmSeparateStride, f"woffset *= tdmSeparateStride"))
            mod.add(SAddU32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(waveOffsetSgprIdx), "+= woffset"))
            if dstGroup0 is not None:
                mod.add(SAddU32(sgpr(f"{dstGroup0}+2"), sgpr(f"{dstGroup0}+2"), sgpr(tmpSgprIdx), "+= tileOffset(lo)"))
                mod.add(SAddCU32(sgpr(f"{dstGroup0}+3"), sgpr(f"{dstGroup0}+3"), sgpr(tmpSgprIdx+1), "+= tileOffset(hi)"))
            else:
                mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
                mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
            #TODO: support strided batch
            #TODO: support GSU
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

    def setDataType(self, dtype: DataType, group1: str | int) -> Module:
        mod = Module()
        dataSizeOp = None

        if dtype.is8bitFloat() or dtype.isFloat4():
            dataSizeOp = 0
        elif dtype.isBFloat16() or dtype.isHalf():
            dataSizeOp = 1
        elif dtype.isSingle():
            dataSizeOp = 2
        elif dtype.isDouble():
            dataSizeOp = 3

        assert dataSizeOp is not None

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

    def incrementGlobalAddr(self, group0: int | str, sgprIncrement: int | str) -> Module:
        """
        Handle lower 32-bit only
        """
        mod = Module()
        mod.addComment("TDM increment global addr")
        mod.add(SAddU32(sgpr(f"{group0}+2"), sgpr(f"{group0}+2"), sgpr(sgprIncrement), "TDM increment"))
        return mod

    def setIterationEnabled(self, group1, enabled: bool) -> Module:
        mod = Module()
        mask = 1 << 19 if enabled else 0xFFF7FFFF
        mod.add(SAndB32(sgpr(group1), sgpr(group1), hex(mask)))
        return mod

    def setTensorDim0(self, group1: int | str, sgprDim0: int | str, writer: "KernelWriterAssembly", constShifter: int=0, isMXS: bool=False) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 0")
        mod.add(SAndB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0xFFFF0000)))

        with writer.allocTmpSgpr(1) as tmpSgpr:
            if isMXS:
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
            else:
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim0)))
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                else:
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
                mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
                mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorDim1(self, group1: int | str, sgprDim1: int | str, writer: "KernelWriterAssembly", constShifter: int=0, isMXS: bool=False) -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 1")

        mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(0xFFFF0000)))
        with writer.allocTmpSgpr(1) as tmpSgpr:
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
                if constShifter:
                    mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(constShifter), sgpr(sgprDim1)))
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(tmpSgpr.idx)))
                else:
                    mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim1)))
                mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
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
        return int(log2(ldsBlockDwordsPerPad)) - 1

    @staticmethod
    def calPadAmount(ldsPadSize: int) -> int:
        ldsPadDwords =  ldsPadSize // 4 # bytes to dwords
        assert ldsPadDwords > 0
        return ldsPadDwords - 1
