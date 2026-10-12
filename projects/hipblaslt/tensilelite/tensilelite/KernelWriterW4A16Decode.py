# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct wave32 dot-product emission for grouped int4 decode."""

from dataclasses import dataclass

from rocisa import instruction as ins
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import KernelBody, Module, Label, SignatureBase
from rocisa.container import ContinuousRegister, GLOBALModifiers, VCC
from rocisa.enum import InstType, SignatureValueKind as SVK
from rocisa.functions import scalarStaticDivideAndRemainder, scalarUInt32DivideAndRemainder
from rocisa.container import sgpr, vgpr


@dataclass(frozen=True)
class DecodeConfig:
    loadWidth: int = 4
    unroll: int = 4
    accumulators: int = 1
    threads: int = 128
    maxN: int = 1
    linearK: bool = False
    nativePermute: bool = True
    zeroPoint: bool = False
    prefetch: int = 1

    def __post_init__(self):
        if type(self.prefetch) is not int or self.prefetch not in (1, 4):
            raise ValueError("Decode prefetch must be 1 or 4")
        if self.prefetch > 1 and (self.loadWidth != 4 or self.unroll != 4):
            raise ValueError("Decode prefetch requires width 4 and unroll 4")
        if self.loadWidth not in (2, 4) or self.unroll not in (1, 4):
            raise ValueError("Unsupported decode load width or unroll")
        if self.accumulators not in (1, 4) or self.threads not in (128, 512):
            raise ValueError("Unsupported decode accumulation or workgroup size")
        if self.maxN not in (1, 4):
            raise ValueError("Decode supports N=1 or N<=4")

    @classmethod
    def fromSolution(cls, state):
        values = state["W4A16Decode"]
        if not isinstance(values, (list, tuple)) or len(values) != 7:
            raise ValueError("W4A16Decode requires width, unroll, accumulators, threads, maxN, linearK, nativePermute")
        if any(type(v) is not int for v in values[:5]) or any(type(v) is not bool for v in values[5:]):
            raise ValueError("W4A16Decode has invalid parameter types")
        return cls(*values, zeroPoint=state["ProblemType"]["ScaleZeroPointA"],
                   prefetch=state.get("W4A16DecodePrefetch", 1))

    def kernelName(self):
        return (f"W4A16_Decode_W{self.loadWidth}_U{self.unroll}_A{self.accumulators}"
                f"_T{self.threads}_N{self.maxN}_L{int(self.linearK)}"
                f"_P{int(self.nativePermute)}_Z{int(self.zeroPoint)}"
                + (f"_PF{self.prefetch}" if self.prefetch > 1 else "") + "_gfx1151")


def configureSolution(state):
    config = DecodeConfig.fromSolution(state)
    pt = state["ProblemType"]
    if (tuple(state["ISA"]) != (11, 5, 1) or pt["OperationType"] != "GEMM"
            or not pt["DataTypeA"].isInt4()
            or not all(pt[key].isHalf() for key in ("DataTypeB", "MacDataTypeA", "DestDataType"))
            or not pt["ComputeDataType"].isSingle()
            or pt["Int4EncodingA"] != "UnsignedBias8"
            or not pt["TransposeA"] or pt["TransposeB"]
            or not pt["StridedBatched"] or pt["GroupedGemm"]
            or pt["UseBias"] or pt["UseE"] or pt["Gradient"]
            or pt["ActivationType"] != "none" or not pt["HighPrecisionAccumulate"]
            or any(pt[key] for key in ("UseInitialStridesAB", "UseInitialStridesCD",
                                      "FusedGemmA2A", "SwizzleTensorA", "SwizzleTensorB",
                                      "StochasticRounding"))
            or pt["UseScaleAB"] != "Block" or pt["UseScaleCD"] or pt["UseScaleAlphaVec"]
            or list(pt["ScaleBlockSizesA"]) != [32, 64, 128]):
        raise ValueError("W4A16Decode requires gfx1151 unsigned int4/FP16 TN runtime-group GEMM")
    state.update(WorkGroup=[config.threads // 32, 1, 32], ThreadTile=[1, 1],
                 WavefrontSize=32, WaveSplitK=True, DepthU=256,
                 EnableMatrixInstruction=False, MatrixInstruction=[],
                 GlobalSplitU=1, GlobalSplitUAlgorithm="MultipleBuffer", StaggerU=0,
                 WorkGroupMapping=1, AssertSummationElementMultiple=256,
                 SourceSwap=False, ConvertAfterDS=False, CustomKernelName="")
    state["InternalSupportParams"].update(KernArgsVersion=3, UseUniversalArgs=True,
                                        SupportUserGSU=False, SupportCustomWGM=False,
                                        SupportCustomStaggerU=False)
    state["CustomKernel"] = dict(name=config.kernelName(), generated=True, args=[],
                                 macrotile=[config.threads // 32, 1, 256],
                                 threads=[config.threads, 1, 1],
                                 grid=["TilesXYBatchGSU", "One", "One"], workspaceType="None")


class KernelWriterW4A16Decode:
    """Each wave streams one row and reduces FP32 dot2 accumulators."""

    def __init__(self, config):
        self.config = config
        self.widePrefetch = config.prefetch == 4
        # Four wide iterations overlap memory loads with computation.
        self.computeBase = 120 if self.widePrefetch else 56
        self.body = Module("W4A16Decode")

    def emit(self, cls, **kwargs):
        if "labelName" in kwargs:
            kwargs["labelName"] = Label(kwargs["labelName"], "").getLabelName()
        self.body.add(cls(**kwargs))

    def label(self, name):
        self.body.add(Label(name, ""))

    def address(self, dst, base, offset):
        self.emit(ins.VAddCOU32, dst=vgpr(dst), dst1=VCC(),
                  src0=sgpr(base.regIdx) if base.regType == "s" else vgpr(base.regIdx), src1=vgpr(offset))
        self.emit(ins.VAddCCOU32, dst=vgpr(dst + 1), dst1=VCC(),
                  src0=sgpr(base.regIdx + 1) if base.regType == 's' else vgpr(base.regIdx + 1),
                  src1=vgpr(offset + 1), src2=VCC())

    def rowAddress(self, dst, base, row, stride, shift):
        self.emit(ins.VMulLOU32, dst=vgpr(dst), src0=stride, src1=vgpr(row))
        self.emit(ins.VMulHIU32, dst=vgpr(dst + 1), src0=stride, src1=vgpr(row))
        if shift:
            self.emit(ins.VLShiftLeftB64, dst=vgpr(dst, 2), src=vgpr(dst, 2), shiftHex=shift)
        self.address(dst, sgpr(base, 2), dst)

    @staticmethod
    def signature(name, threads, zeroPoint=True, vgprs=64):
        sig = SignatureBase(name, 3, "4", 0, [1, 0, 0], 0, threads, vgprs, 0, 64)
        for name in ("Gemm info", "kernel info0", "kernel info1", "numWG",
                     "SizesFree0", "SizesFree1", "SizesFree2", "SizesSum0"):
            sig.addArg(name, SVK.SIG_VALUE, "u32")
        for name in ("A", "B"):
            sig.addArg(name, SVK.SIG_GLOBALBUFFER, "u8", "generic")
        for name in ("strideA0", "strideA1", "strideB0", "strideB1"):
            sig.addArg(name, SVK.SIG_VALUE, "u32")
        for name in ("alpha", "beta"):
            sig.addArg(name, SVK.SIG_VALUE, "f32")
        for name in ("D", "C"):
            sig.addArg(name, SVK.SIG_GLOBALBUFFER, "f16", "generic")
        for name in ("strideD0", "strideD1", "strideC0", "strideC1"):
            sig.addArg(name, SVK.SIG_VALUE, "u32")
        for name in ("AddressScaleA", "AddressScaleB") + (("AddressScaleZeroA",) if zeroPoint else ()):
            sig.addArg(name, SVK.SIG_GLOBALBUFFER, "u8", "generic")
        for name in ("batchOffsetD", "batchOffsetC", "batchOffsetA", "batchOffsetB"):
            sig.addArg(name, SVK.SIG_VALUE, "u64")
        sig.addArg("groupSize", SVK.SIG_VALUE, "u32")
        return sig

    def prologue(self):
        e = self.emit
        # s4:s31 cover M through the optional zero-point pointer.
        for index in range(4, 32, 4):
            e(ins.SLoadB128, dst=sgpr(index, 4), base=sgpr(0, 2), soffset=index * 4)
        e(ins.SLoadB32, dst=sgpr(40), base=sgpr(0, 2), soffset=160 if self.config.zeroPoint else 152)
        e(ins.SWaitCnt, kmcnt=0)
        e(ins.SMovB32, dst=sgpr(54), src=0)
        if self.config.maxN > 1:
            for n in (2, 3, 4):
                e(ins.SCmpEQU32, src0=sgpr(5), src1=n)
                e(ins.SCBranchSCC1, labelName=f"Columns{n}")
            e(ins.SBranch, labelName="ColumnsDone")
            for n in (2, 3, 4):
                self.label(f"Columns{n}")
                self.body.add(scalarStaticDivideAndRemainder(55, 54, 2, n,
                              ContinuousRegister(56, 2)))
                e(ins.SMovB32, dst=sgpr(2), src=sgpr(55))
                e(ins.SBranch, labelName="ColumnsDone")
            self.label("ColumnsDone")
        e(ins.VLShiftRightB32, dst=vgpr(1), src=vgpr(0), shiftHex=5)
        e(ins.VLShiftLeftOrB32, dst=vgpr(1), src0=sgpr(2),
          shiftHex=(self.config.threads // 32).bit_length() - 1, src1=vgpr(1))
        e(ins.VCmpLtU32, dst=VCC(), src0=vgpr(1), src1=sgpr(4))
        e(ins.SAndSaveExecB32, dst=sgpr(42), src=VCC())
        e(ins.SCBranchExecZ, labelName="DecodeEnd")
        e(ins.VAndB32, dst=vgpr(0), src0=31, src1=vgpr(0))
        for j in range(self.config.accumulators):
            e(ins.VMovB32, dst=vgpr(32 + j), src=0)
        e(ins.SAndB32, dst=sgpr(52), src0=sgpr(16), src1="0x7fffffff")
        e(ins.SCmpEQU32, src0=sgpr(52), src1=0)
        e(ins.SCBranchSCC1, labelName="Reduce")
        e(ins.SFf1B32, dst=sgpr(44), src=sgpr(40))
        e(ins.SLShiftRightB32, dst=sgpr(45), src=sgpr(7), shiftHex=sgpr(44))
        e(ins.SLShiftRightB32, dst=sgpr(52), src=sgpr(12), shiftHex=1)
        self.rowAddress(4, 8, 1, sgpr(52), 0)
        self.rowAddress(6, 26, 1, sgpr(45), 1)
        if self.config.zeroPoint:
            e(ins.VLShiftRightB32, dst=vgpr(12), src=vgpr(1), shiftHex=3)
            self.rowAddress(8, 30, 12, sgpr(45), 2)
            e(ins.VAndB32, dst=vgpr(12), src0=7, src1=vgpr(1))
            e(ins.VLShiftRightB32, dst=vgpr(12), src=vgpr(12), shiftHex=1)
            e(ins.VMovB32, dst=vgpr(13), src=0)
            self.address(8, vgpr(8, 2), 12)
            e(ins.VAndB32, dst=vgpr(3), src0=1, src1=vgpr(1))
            e(ins.VLShiftLeftB32, dst=vgpr(3), src=vgpr(3), shiftHex=2)
        # B's column base is uniform across the workgroup.
        e(ins.SMulI32, dst=sgpr(56), src0=sgpr(54), src1=sgpr(14))
        e(ins.SMulHIU32, dst=sgpr(57), src0=sgpr(54), src1=sgpr(14))
        e(ins.SLShiftLeftB64, dst=sgpr(56, 2), src=sgpr(56, 2), shiftHex=1)
        e(ins.SAddU32, dst=sgpr(10), src0=sgpr(10), src1=sgpr(56))
        e(ins.SAddCU32, dst=sgpr(11), src0=sgpr(11), src1=sgpr(57))
        e(ins.VMovB32, dst=vgpr(10), src=0)
        if not self.config.linearK:
            e(ins.SLShiftRightB32, dst=sgpr(52), src=sgpr(7), shiftHex=8)
            e(ins.VReadfirstlaneB32, dst=sgpr(53), src=vgpr(1))
            self.body.add(scalarUInt32DivideAndRemainder(55, 53, 52, 58,
                          ContinuousRegister(16, 4), 32))
            e(ins.VMovB32, dst=vgpr(10), src=sgpr(58))
            e(ins.VLShiftLeftB32, dst=vgpr(10), src=vgpr(10), shiftHex=8)
        for reg, value in ((48, "0x000f000f"), (49, "0x00f000f0"),
                           (50, "0xe408e408"), (51, "0xd480d480")):
            e(ins.SMovB32, dst=sgpr(reg), src=value)

    def loadIteration(self, width, slot=0):
        e = self.emit
        # A lane reads width packed dwords and the matching 8*width activations.
        e(ins.VAddU32, dst=vgpr(11), src0=sgpr(46), src1=vgpr(2))
        if not self.config.linearK:
            e(ins.VAddU32, dst=vgpr(11), src0=vgpr(10), src1=vgpr(11))
            e(ins.VSubU32, dst=vgpr(12), src0=vgpr(11), src1=sgpr(7))
            e(ins.VCmpGEU32, dst=VCC(), src0=vgpr(11), src1=sgpr(7))
            e(ins.VCndMaskB32, dst=vgpr(11), src0=vgpr(11), src1=vgpr(12), src2=VCC())
        e(ins.VLShiftRightB32, dst=vgpr(12), src=vgpr(11), shiftHex=1)
        e(ins.VMovB32, dst=vgpr(13), src=0)
        self.address(14, vgpr(4, 2), 12)
        load = {1: ins.GlobalLoadB32, 2: ins.GlobalLoadB64, 4: ins.GlobalLoadB128}[width]
        e(load, dst=vgpr((104 if self.widePrefetch else 24) + slot * width, width), vaddr=vgpr(14, 2), saddr=vgpr("off", 1, False, False, True),
          modifier=GLOBALModifiers(slc=True, dlc=True))
        e(ins.VMovB32, dst=vgpr(12), src=vgpr(11))
        e(ins.VLShiftLeftB64, dst=vgpr(12, 2), src=vgpr(12, 2), shiftHex=1)
        self.address(14, sgpr(10, 2), 12)
        for j in range(width):
            e(ins.GlobalLoadB128, dst=vgpr(40 + (j + slot * width) * 4, 4), vaddr=vgpr(14, 2),
              saddr=vgpr("off", 1, False, False, True), modifier=GLOBALModifiers(offset=j * 16))
        e(ins.VLShiftRightB32, dst=vgpr(12), src=vgpr(11), shiftHex=sgpr(44))
        e(ins.VMovB32, dst=vgpr(13), src=0)
        e(ins.VLShiftLeftB64, dst=vgpr(12, 2), src=vgpr(12, 2), shiftHex=1)
        self.address(14, vgpr(6, 2), 12)
        # The prologue's division temporaries are now free for pipelined metadata.
        e(ins.GLOBALLoadInstruction, instType=InstType.INST_U16, dst=vgpr(16 + slot), vaddr=vgpr(14, 2), saddr=vgpr("off", 1, False, False, True))
        if self.config.zeroPoint:
            e(ins.VLShiftLeftB64, dst=vgpr(12, 2), src=vgpr(12, 2), shiftHex=1)
            self.address(14, vgpr(8, 2), 12)
            e(ins.GLOBALLoadInstruction, instType=InstType.INST_U8, dst=vgpr(20 + slot), vaddr=vgpr(14, 2), saddr=vgpr("off", 1, False, False, True))

    def computeIteration(self, width, slot=0, pending=0):
        e = self.emit
        e(ins.SWaitCnt, vlcnt=pending)
        e(ins.VMovB32, dst=vgpr(self.computeBase + 6), src=vgpr(16 + slot))
        if self.config.zeroPoint:
            e(ins.VMovB32, dst=vgpr(self.computeBase + 4), src=vgpr(20 + slot))
        e(ins.VPackF16toB32, dst=vgpr(self.computeBase + 6), src0=vgpr(self.computeBase + 6), src1=vgpr(self.computeBase + 6))
        if self.config.zeroPoint:
            e(ins.VBfeU32, dst=vgpr(self.computeBase + 4), src0=vgpr(self.computeBase + 4), src1=vgpr(3), src2=4)
            e(ins.VLShiftLeftOrB32, dst=vgpr(self.computeBase + 4), src0=vgpr(self.computeBase + 4), shiftHex=16, src1=vgpr(self.computeBase + 4))
            e(ins.VLShiftLeftOrB32, dst=vgpr(self.computeBase + 5), src0=vgpr(self.computeBase + 4), shiftHex=4, src1="0xd400d400")
            e(ins.VOrB32, dst=vgpr(self.computeBase + 4), src0="0xe400e400", src1=vgpr(self.computeBase + 4))
        for j in range(width):
            q = (104 if self.widePrefetch else 24) + j + slot * width
            b = 40 + 4 * (j + slot * width)
            e(ins.VLShiftRightB32, dst=vgpr(self.computeBase + 7), src=vgpr(q), shiftHex=8)
            for pair in range(4):
                e(ins.VAndOrB32, dst=vgpr(self.computeBase + pair), src0=vgpr(q if pair < 2 else self.computeBase + 7),
                  src1=sgpr(48 + pair % 2), src2=("0x64006400", "0x54005400")[pair % 2])
            for pair in range(4):
                e(ins.VAddPKF16, dst=vgpr(self.computeBase + pair), src0=vgpr(self.computeBase + pair),
                  src1=vgpr(self.computeBase + 4 + pair % 2) if self.config.zeroPoint else sgpr(50 + pair % 2))
            for pair in range(4):
                e(ins.VMulPKF16, dst=vgpr(self.computeBase + pair), src0=vgpr(self.computeBase + pair), src1=vgpr(self.computeBase + 6))
            for pair in range(4):
                lo, hi, dst = vgpr(b + pair // 2), vgpr(b + 2 + pair // 2), vgpr(12 + pair)
                if self.config.nativePermute:
                    e(ins.VPermB32, dst=dst, src0=hi, src1=lo,
                      src2=("0x05040100", "0x07060302")[pair % 2])
                elif pair % 2 == 0:
                    e(ins.VAndB32, dst=dst, src0="0xffff", src1=lo)
                    e(ins.VLShiftLeftOrB32, dst=dst, src0=hi, shiftHex=16, src1=dst)
                else:
                    e(ins.VLShiftRightB32, dst=dst, src=lo, shiftHex=16)
                    e(ins.VAndOrB32, dst=dst, src0=hi, src1="0xffff0000", src2=dst)
            for pair in range(4):
                acc = 32 + j % self.config.accumulators
                e(ins.VDot2F32F16, dst=vgpr(acc), src0=vgpr(self.computeBase + pair),
                  src1=vgpr(12 + pair), src2=vgpr(acc))

    def iterationBatch(self, width, batch):
        loads = width + 2 + int(self.config.zeroPoint)
        for slot in range(batch):
            self.loadIteration(width, slot)
            self.emit(ins.SAddU32, dst=sgpr(46), src0=sgpr(46), src1=256 * width)
        for slot in range(batch):
            self.computeIteration(width, slot, (batch - slot - 1) * loads)

    def loop(self, width):
        e = self.emit
        e(ins.VLShiftLeftB32, dst=vgpr(2), src=vgpr(0), shiftHex=(8 * width).bit_length() - 1)
        e(ins.SMovB32, dst=sgpr(46), src=0)
        self.label(f"LoopW{width}")
        if self.config.unroll == 4:
            e(ins.SAddU32, dst=sgpr(53), src0=sgpr(46), src1=1024 * width)
            e(ins.SCmpGtU32, src0=sgpr(53), src1=sgpr(7))
            e(ins.SCBranchSCC1, labelName=f"TailW{width}")
            batch = 4 if width == 1 else self.config.prefetch
            for _ in range(4 // batch):
                self.iterationBatch(width, batch)
            e(ins.SCmpGeU32, src0=sgpr(46), src1=sgpr(7))
            e(ins.SCBranchSCC1, labelName="Reduce")
            e(ins.SBranch, labelName=f"LoopW{width}")
            self.label(f"TailW{width}")
            if width == 1:
                # Pair short tails without changing the order of dot products.
                e(ins.SCmpGeU32, src0=sgpr(7), src1=4096)
                e(ins.SCBranchSCC1, labelName="ScalarTail")
                e(ins.SAddU32, dst=sgpr(53), src0=sgpr(46), src1=512)
                e(ins.SCmpGtU32, src0=sgpr(53), src1=sgpr(7))
                e(ins.SCBranchSCC1, labelName="ScalarTail")
                self.iterationBatch(width, 2)
                e(ins.SCmpGeU32, src0=sgpr(46), src1=sgpr(7))
                e(ins.SCBranchSCC1, labelName="Reduce")
                self.label("ScalarTail")
        self.loadIteration(width)
        self.computeIteration(width)
        e(ins.SAddU32, dst=sgpr(46), src0=sgpr(46), src1=256 * width)
        e(ins.SCmpGeU32, src0=sgpr(46), src1=sgpr(7))
        e(ins.SCBranchSCC1, labelName="Reduce")
        e(ins.SBranch, labelName=f"TailW{width}" if self.config.unroll == 4 else f"LoopW{width}")

    def epilogue(self):
        e = self.emit
        self.label("Reduce")
        if self.config.accumulators == 4:
            for dst, a, b in ((32, 32, 33), (34, 34, 35), (32, 32, 34)):
                e(ins.VAddF32, dst=vgpr(dst), src0=vgpr(a), src1=vgpr(b))
        for distance in (16, 8, 4, 2, 1):
            e(ins.VXorB32, dst=vgpr(12), src0=distance, src1=vgpr(0))
            e(ins.VLShiftLeftB32, dst=vgpr(12), src=vgpr(12), shiftHex=2)
            e(ins.DSBPermuteB32, dst=vgpr(13), src0=vgpr(12), src1=vgpr(32))
            e(ins.SWaitCnt, dscnt=0)
            e(ins.VAddF32, dst=vgpr(32), src0=vgpr(32), src1=vgpr(13))
        e(ins.VCmpEQU32, dst=VCC(), src0=0, src1=vgpr(0))
        e(ins.SAndSaveExecB32, dst=sgpr(42), src=VCC())
        e(ins.VMulF32, dst=vgpr(32), src0=sgpr(16), src1=vgpr(32))
        e(ins.SAndB32, dst=sgpr(52), src0=sgpr(17), src1="0x7fffffff")
        e(ins.SCmpEQU32, src0=sgpr(52), src1=0)
        e(ins.SCBranchSCC1, labelName="Store")
        self.outputAddress(20, 24)
        e(ins.GLOBALLoadInstruction, instType=InstType.INST_U16, dst=vgpr(14), vaddr=vgpr(12, 2), saddr=vgpr("off", 1, False, False, True))
        e(ins.SWaitCnt, vlcnt=0)
        e(ins.VCvtF16toF32, dst=vgpr(14), src=vgpr(14))
        e(ins.VFmaF32, dst=vgpr(32), src0=sgpr(17), src1=vgpr(14), src2=vgpr(32))
        self.label("Store")
        self.outputAddress(18, 22)
        e(ins.VCvtF32toF16, dst=vgpr(32), src=vgpr(32))
        e(ins.GlobalStoreB16, vaddr=vgpr(12, 2), src=vgpr(32), saddr=vgpr("off", 1, False, False, True))
        self.label("DecodeEnd")
        e(ins.SEndpgm)

    def outputAddress(self, base, ld):
        e = self.emit
        e(ins.SMulI32, dst=sgpr(56), src0=sgpr(54), src1=sgpr(ld))
        e(ins.SMulHIU32, dst=sgpr(57), src0=sgpr(54), src1=sgpr(ld))
        e(ins.VAddCOU32, dst=vgpr(12), dst1=VCC(), src0=sgpr(56), src1=vgpr(1))
        e(ins.VAddCCOU32, dst=vgpr(13), dst1=VCC(), src0=sgpr(57), src1=0, src2=VCC())
        e(ins.VLShiftLeftB64, dst=vgpr(12, 2), src=vgpr(12, 2), shiftHex=1)
        self.address(12, sgpr(base, 2), 12)

    def generate(self, name):
        self.prologue()
        e = self.emit
        width = self.config.loadWidth
        e(ins.SAndB32, dst=sgpr(52), src0=sgpr(7), src1=256 * width - 1)
        e(ins.SAndB32, dst=sgpr(53), src0=sgpr(12), src1=8 * width - 1)
        e(ins.SOrB32, dst=sgpr(52), src0=sgpr(52), src1=sgpr(53))
        e(ins.SCBranchSCC1, labelName="Fallback")
        self.loop(width)
        self.label("Fallback")
        self.loop(1)
        self.epilogue()
        kernel = KernelBody(name)
        # The N=1 scalar schedule runs faster with fewer resident waves.
        vgprs = 128 if self.widePrefetch else (104 if self.config.maxN == 1 and self.config.unroll == 4 else 64)
        kernel.addSignature(self.signature(name, self.config.threads, self.config.zeroPoint, vgprs))
        kernel.addBody(self.body)
        kernel.setGprs(vgprs, 0, 64)
        options = rocIsaPassOption()
        options.insertDelayAlu = True
        options.numWaves = self.config.threads // 32
        rocIsaPass(kernel, options)
        return str(kernel)
