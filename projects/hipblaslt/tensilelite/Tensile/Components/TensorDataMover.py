from ..Component import TensorDataMover
from ..Common.DataType import DataType
from typing import Mapping
from rocisa.code import Module
from rocisa.instruction import SMovB32, SMovB64, SOrB32, SAndB32, SLShiftLeftB32, \
    SLShiftRightB32, SAddU32, SAddCU32, SMulI32, TensorLoadToLds
from rocisa.container import sgpr, vgpr
# from ..KernelWriterAssembly import KernelWriterAssembly

class TensorDataMoverLoad(TensorDataMover):
    kernel = {"TDMInst": 3}
    asmCaps = {"HasTDM": True}
    GROUP0_NUM_SGPR = 4
    GROUP1_NUM_SGPR = 8
    GROUP2_NUM_SGPR = 4
    GROUP3_NUM_SGPR = 4

    def __call__(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping):
        pass

    def calculateStartAddr(self, writer: "KernelWriterAssembly", kernel: Mapping, tp: Mapping, sgprAddr: int | str) -> Module:
        #here we assume TN
        mod = Module()
        tc: str = tp["tensorChar"]
        tIdx: int = 0 if tp["isA"] else 1
        bpe: int = int(tp["bpeGR"])
        sgprStrideName: str = f"Stride{tc}{writer.states.indexChars[tp['idx']]}"
        sgprWorkgroupName: str = f"WorkGroup{tIdx}"
        mt: int = kernel["MacroTile0"] if tc == "A" else kernel["MacroTile1"]

        mod.addComment(f"TDM calc start addr of {tc}")

        with writer.allocTmpSgpr(2) as tmpSgprRes:
            tmpSgprIdx = tmpSgprRes.idx
            mod.add(SMovB64(sgpr(tmpSgprIdx, 2), 0))
            mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(sgprStrideName), mt * bpe, f"stride * MT({mt}) * bpe({bpe})"))
            mod.add(SMulI32(sgpr(tmpSgprIdx), sgpr(tmpSgprIdx), sgpr(sgprWorkgroupName), "*= wgId)"))
            mod.add(SAddU32(sgpr(sgprAddr), sgpr(tmpSgprIdx), sgpr(sgprAddr), "+= baseAddr(lo)"))
            mod.add(SAddCU32(sgpr(f"{sgprAddr}+1"), sgpr(tmpSgprIdx+1), sgpr(f"{sgprAddr}+1"), "+= baseAddr(hi)"))
            #TODO: add wave offset
            #TODO: support strided batch
            #TODO: support GSU
        return mod

    def issueLoad(self, group0: int | str, group1: int | str, group2: int | str, group3: int | str) -> Module:
        mod = Module("tensor load")
        mod.add(TensorLoadToLds(sgpr(group0, self.GROUP0_NUM_SGPR), sgpr(group1, self.GROUP1_NUM_SGPR),\
                                sgpr(group2, self.GROUP2_NUM_SGPR), sgpr(group3, self.GROUP3_NUM_SGPR)))
        return mod

    def initOperands(self, group0: int | str, group1: int | str, group2: int | str, group3: int | str) -> Module:
        mod = Module("tdmInit")
        for i in range(self.GROUP0_NUM_SGPR):
            val = 1 if i == 0 else 0
            mod.add(SMovB32(sgpr(f"{group0}+{i}"), val))

        mod.add(SOrB32(sgpr(f"{group0}+3"), sgpr(f"{group0}+3"), hex(2 << 30), "set type field to 2(image)"))

        for i in range(self.GROUP1_NUM_SGPR):
            mod.add(SMovB32(sgpr(f"{group1}+{i}"), 0))

        for i in range(self.GROUP2_NUM_SGPR):
            mod.add(SMovB32(sgpr(f"{group2}+{i}"), 0))

        for i in range(self.GROUP3_NUM_SGPR):
            mod.add(SMovB32(sgpr(f"{group3}+{i}"), 0))

        return mod

    def setDataType(self, dtype: DataType, group1: str | int) -> Module:
        mod = Module()
        dataSizeOp = None

        if dtype.is8bitFloat():
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

    def setLdsAddr(self, group0: int | str, ldsAddr: int) -> Module:
        mod = Module()
        mod.addComment("TDM set LDS addr")
        mod.add(SMovB32(sgpr(f"{group0}+1"), hex(ldsAddr)))
        return mod

    def getLdsAddrSgprName(self, group0: int | str) -> str:
        return f"{group0}+1"

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
        mask = 1 << 19 if enabled else ~(1 << 19)
        mod.add(SAndB32(sgpr(group1), sgpr(group1), hex(mask)))
        return mod

    def setTensorDim0(self, group1: int | str, sgprDim0: int | str, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 0")
        mod.add(SAndB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), hex(0x0000FFFF)))
        mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0xFFFF0000)))

        with writer.allocTmpSgpr(1) as tmpSgpr:
            mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
            mod.add(SOrB32(sgpr(f"{group1}+1"), sgpr(f"{group1}+1"), sgpr(tmpSgpr.idx)))
            mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim0)))
            mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorDim1(self, group1: int | str, sgprDim1: int | str, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.addComment("TDM set tensor dim 1")

        with writer.allocTmpSgpr(1) as tmpSgpr:
            mod.add(SAndB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), hex(0x0000FFFF)))
            mod.add(SAndB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(0xFFFF0000)))
            mod.add(SLShiftLeftB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim1)))
            mod.add(SOrB32(sgpr(f"{group1}+2"), sgpr(f"{group1}+2"), sgpr(tmpSgpr.idx)))
            mod.add(SLShiftRightB32(sgpr(tmpSgpr.idx), hex(16), sgpr(sgprDim1)))
            mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), sgpr(tmpSgpr.idx)))

        return mod

    def setTensorTile0(self, group1: int | str, tile0: int, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.addComment("TDM set tensor tile 0")
        with writer.allocTmpSgpr(1) as tmpSgpr:
            mod.add(SAndB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex(0x0000FFFF)))
            mod.add(SOrB32(sgpr(f"{group1}+3"), sgpr(f"{group1}+3"), hex((tile0 & 0xFFFF) << 16), f"set tile0 to {tile0}"))
        return mod

    def setTensorTile1(self, group1: int | str, tile1, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.addComment("TDM set tensor tile 1")
        mod.add(SAndB32(sgpr(f"{group1}+4"), sgpr(f"{group1}+4"), hex(0xFFFF0000)))
        mod.add(SOrB32(sgpr(f"{group1}+4"), sgpr(f"{group1}+4"), hex(tile1 & 0xFFFF), f"set tile1 to {tile1}"))
        return mod

    def setTensorStride0(self, group1: int | str, sgprStride0: int | str) -> Module:
        mod = Module()
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
