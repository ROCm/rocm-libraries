################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa import rocIsa
from rocisa.code import Module, Label, TextBlock
from rocisa.container import vgpr, sgpr, DSModifiers, SDWAModifiers, RegisterContainer, VCC
from rocisa.enum import SelectBit
from rocisa.instruction import DSLoadB32, DSStoreB32, FlatLoadB32, FlatStoreB32, \
    Instruction, PVCvtBF16toFP32, SAddU32, SAndB32, SAndB64, SAndSaveExecB32, \
    SAndSaveExecB64, SBarrier, SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ, \
    SCBranchVCCZ, SCMovB32, SCMovB64, SCmpEQU32, SCmpEQU64, SCmpLtU32, SLoadB128, \
    SLoadB256, SLoadB32, SLoadB512, SLoadB64, SMovB32, SMovB64, SMovkI32, SMulHII32, \
    SMulHIU32, SMulI32, SOrSaveExecB32, SOrSaveExecB64, SWaitCnt, VAddCOU32, VAndB32, \
    VCmpEQF32, VCmpEQF64, VCmpXEqU32, VCmpXGeU32, VCmpXGtU32, VCmpXInstruction, \
    VCmpXLeU32, VCmpXLtI32, VCmpXLtU32, VCmpXNeU16, VCmpXNeU32, VLShiftLeftB32, \
    VMaxI32, VMed3I32, VMinI32, VMovB32, VMulHII32, VMulHIU32, VMulLOU32, VReadfirstlaneB32

from .DataType import DataType
from .RegisterPool import ContinuousRegister
from .Utils import log2

from enum import Enum


from typing import Union
import sys

################################################################################
################################################################################
###
###   Function ExtInstructions
###
################################################################################
################################################################################
########################################
# If else
########################################

def SBranchIfZero(sgprName, computeDataType: DataType, tmpSgpr, laneSC, label, waveFrontSize):
    module = Module("SBranchIfZero")
    sgprStr = "s[{}]".format(sgprName)
    if computeDataType.isDoubleComplex():
        module.add(VCmpEQF64(dst=sgpr(tmpSgpr, laneSC), src0=sgpr(sgprName, 2), src1=0.0, comment="%s.real == 0.0 ?" % sgprStr))
        sgprVar = "%s+2" % sgprName if isinstance(sgprName, str) else sgprName + 2
        module.add(VCmpEQF64(dst=VCC(), src0=sgpr(sgprVar, 2), src1=0.0, comment="%s.imag == 0.0 ?" % sgprStr))
        if waveFrontSize == 32:
            module.add(SAndB32(dst=sgpr(tmpSgpr, laneSC), src0=VCC(), src1=sgpr(tmpSgpr, laneSC), comment="%s == 0 ?" % sgprStr))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr, laneSC), src1=0, comment="branch if %s == 0" % sgprStr))
        else:
            module.add(SAndB64(dst=sgpr(tmpSgpr, laneSC), src0=VCC(), src1=sgpr(tmpSgpr, laneSC), comment="%s == 0 ?" % sgprStr))
            module.add(SCmpEQU64(src0=sgpr(tmpSgpr, laneSC), src1=0, comment="branch if %s == 0" % sgprStr))
        module.add(SCBranchSCC0(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    elif computeDataType.isDouble():
        module.add(VCmpEQF64(dst=VCC(), src0=sgpr(sgprName, 2), src1=0.0, comment="%s == 0.0 ?" % sgprStr))
        module.add(SCBranchVCCNZ(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    elif computeDataType.isSingleComplex():
        module.add(VCmpEQF32(dst=sgpr(tmpSgpr, laneSC), src0=sgpr(sgprName), src1=0.0, comment="%s.real == 0.0f ?" % sgprStr))
        sgprVar = "%s+1" % sgprName if isinstance(sgprName, str) else sgprName + 1
        module.add(VCmpEQF32(dst=VCC(), src0=sgpr(sgprVar), src1=0.0, comment="%s.imag == 0.0f ?" % sgprStr))
        if waveFrontSize == 32:
            module.add(SAndB32(dst=sgpr(tmpSgpr, laneSC), src0=VCC(), src1=sgpr(tmpSgpr, laneSC), comment="%s == 0 ?" % sgprStr))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr, laneSC), src1=0, comment="branch if %s == 0" % sgprStr))
        else:
            module.add(SAndB64(dst=sgpr(tmpSgpr, laneSC), src0=VCC(), src1=sgpr(tmpSgpr, laneSC), comment="%s == 0 ?" % sgprStr))
            module.add(SCmpEQU64(src0=sgpr(tmpSgpr, laneSC), src1=0, comment="branch if %s == 0" % sgprStr))
        module.add(SCBranchSCC0(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    elif computeDataType.isSingle() or computeDataType.isHalf() or computeDataType.isBFloat16():
        module.add(VCmpEQF32(dst=VCC(), src0=sgpr(sgprName), src1=0.0, comment="%s == 0.0f ?" % sgprStr))
        module.add(SCBranchVCCNZ(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    elif computeDataType.isInt32(): # int32
        module.add(SCmpEQU32(src0=sgpr(sgprName), src1=0, comment="%s == 0 ?" % sgprStr))
        module.add(SCBranchSCC1(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    elif computeDataType.isInt64(): # int64
        module.add(SCmpEQU64(src0=sgpr(sgprName,2), src1=0, comment="%s == 0 ?" % sgprStr))
        module.add(SCBranchSCC1(labelName=label.getLabelName(), comment="branch if %s == 0" % sgprStr))
    else:
        print("Unsupported compute data type: %s" % str(computeDataType))
        sys.stdout.flush()
        sys.exit(-1)
    return module

def SBranchIfNotZero(sgprName, computeDataType: DataType, label):
    module = Module("SBranchIfNotZero")
    sgprStr = "s[{}]".format(sgprName)
    if computeDataType.isDoubleComplex():
        module.add(VCmpEQF64(dst=VCC(), src0=sgpr(sgprName, 2), src1=0.0, comment="%s.real == 0.0 ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s.real != 0" % sgprStr))
        sgprVar = "%s+2" % sgprName if isinstance(sgprName, str) else sgprName + 2
        module.add(VCmpEQF64(dst=VCC(), src0=sgpr(sgprVar, 2), src1=0.0, comment="%s.imag == 0.0 ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s.imag != 0" % sgprStr))
    elif computeDataType.isDouble():
        module.add(VCmpEQF64(dst=VCC(), src0=sgpr(sgprName, 2), src1=0.0, comment="%s == 0.0 ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s != 0" % sgprStr))
    elif computeDataType.isSingleComplex():
        module.add(VCmpEQF32(dst=VCC(), src0=sgpr(sgprName), src1=0.0, comment="%s.real == 0.0f ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s.real != 0" % sgprStr))
        sgprVar = "%s+1" % sgprName if isinstance(sgprName, str) else sgprName + 1
        module.add(VCmpEQF32(dst=VCC(), src0=sgpr(sgprVar), src1=0.0, comment="%s.imag == 0.0f ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s.imag != 0" % sgprStr))
    elif computeDataType.isSingle() or computeDataType.isHalf() or computeDataType.isBFloat16():
        module.add(VCmpEQF32(dst=VCC(), src0=sgpr(sgprName), src1=0.0, comment="%s == 0.0f ?" % sgprStr))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(), comment="branch if %s != 0" % sgprStr))
    elif computeDataType.isInt64():
        module.add(SCmpEQU64(src0=sgpr(sgprName, 2), src1=0, comment="%s == 0 ?" % sgprStr))
        module.add(SCBranchSCC0(labelName=label.getLabelName(), comment="branch if %s != 0" % sgprStr))
    else:
        module.add(SCmpEQU32(src0=sgpr(sgprName), src1=0, comment="%s == 0 ?" % sgprStr))
        module.add(SCBranchSCC0(labelName=label.getLabelName(), comment="branch if %s != 0" % sgprStr))
    return module

########################################
# Saturate Cast Integer
########################################

class SaturateCastType(Enum):
    NORMAL = 1
    DO_NOTHING = 2
    UPPER = 3
    LOWER = 4

def VSaturateCastInt(vgprSumIdxV, tmpVgpr, tmpSgpr, lowerBound, upperBound, type=SaturateCastType.NORMAL, initGpr=True):
    # SaturateCastType = 0, normal case
    # SaturateCastType = 1, do nothing
    # SaturateCastType = 2, upperbound only
    # SaturateCastType = 3, lowerbound only
    initGprStr = "with init gpr" if initGpr else "without init gpr"
    module = Module("SaturateCastInt %s"%(initGprStr))
    if type == SaturateCastType.NORMAL:
        tmpLowerBound = tmpSgpr
        tmpUpperBound = tmpVgpr
        if initGpr:
            lowerBoundHex = hex(lowerBound)
            upperBoundHex = hex(upperBound)
            module.add(SMovkI32(dst=sgpr(tmpLowerBound), src=lowerBoundHex, comment="%d"%lowerBound ))
            module.add(VMovB32(dst=vgpr(tmpUpperBound), src=upperBoundHex, comment="%d"%upperBound ))
        module.add(VMed3I32(dst=vgprSumIdxV, src0=vgprSumIdxV, src1=sgpr(tmpLowerBound), src2=vgpr(tmpUpperBound), comment="x= min(%d, max(%d, x))"%(upperBound, lowerBound)))
    elif type == SaturateCastType.DO_NOTHING:
        pass
    elif type == SaturateCastType.UPPER:
        module.add(VMinI32(dst=vgprSumIdxV, src0=upperBound, src1=vgprSumIdxV, comment="x = min(%d, x)"%upperBound))
    elif type == SaturateCastType.LOWER:
        module.add(VMaxI32(dst=vgprSumIdxV, src0=lowerBound, src1=vgprSumIdxV, comment="x = max(%d, x)"%lowerBound))
    return module

########################################
# Cvt
########################################

def VCvtBF16toFP32(dst, src, vgprMask, vi, additionalCmts=""):
    ti = rocIsa.getInstance()
    if ti.getAsmCaps()["HasBF16CVT"]:
        select_bit = SelectBit.WORD_0 if vi%2 == 0 else SelectBit.WORD_1
        sdwa=SDWAModifiers(src0_sel=select_bit);
        return PVCvtBF16toFP32(dst=vgpr(dst), src=vgpr(src), sdwa=sdwa, comment="cvt bf16 to f32")
    else:
        if (vi % 2) == 1:
            return VAndB32(dst=vgpr(dst), src0=vgpr(src), src1=vgpr(vgprMask), comment="cvt bf16 to fp32. " + additionalCmts) # mask = hex(0xffff0000)
        else:
            return VLShiftLeftB32(dst=vgpr(dst), shiftHex=16, src=vgpr(src), comment="cvt bf16 to fp32. " + additionalCmts)


########################################
# init lds state
########################################
def DSInit(tmpVgprRes: ContinuousRegister, numThreads: int, \
            ldsNumElements: int, initValue):
    assert tmpVgprRes.size > 1
    tmp = tmpVgprRes.idx
    tmpAddr = tmp + 1
    module = Module("initLds")
    module.addComment1("init lds state")
    module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment=""))
    module.add(SBarrier(comment="init LDS"))
    module.add(VMovB32(dst=vgpr(tmp), src=hex(initValue), comment="Init value"))
    module.add(VLShiftLeftB32(dst=vgpr(tmpAddr), shiftHex=2, src=vgpr("Serial"), \
                comment="set per-thread address to init LDS"))
    writesPerThread = ((ldsNumElements-1)//numThreads//4) + 1
    for i in range(0, writesPerThread):
        module.add(DSStoreB32(dstAddr=vgpr(tmpAddr), src=vgpr(tmp),
                    ds=DSModifiers(offset=(i*numThreads*4)), comment="init lds"))
    module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="wait for LDS init to complete"))
    module.add(SBarrier(comment="init LDS exit"))
    return module

################################################################################
################################################################################
###
###   Class ExtInstructions
###
################################################################################
################################################################################

##############################################################################
# Load arguments
##############################################################################

class ArgumentLoader:
    def __init__(self) -> None:
        self.kernArgOffset = 0

    def resetOffset(self) -> None:
        self.kernArgOffset = 0

    def setOffset(self, offset: int) -> None:
        self.kernArgOffset = offset

    def getOffset(self) -> int:
        return self.kernArgOffset

    ##############################################################################
    # getKernArg
    # Write an argument to specified SGPR and move the kernArgOffset
    # if writeSgpr==0, just move the kernArgOffset - this is used to skip
    # unused parms
    ##############################################################################
    def loadKernArg(self, dst: Union[int, str], srcAddr: Union[int, str], sgprOffset = None, dword=1,\
                        writeSgpr=True) -> Union[Instruction, TextBlock]:
        item = None
        size = dword*4
        if writeSgpr:
            SLoadBX = { 512: SLoadB512,
                        256: SLoadB256,
                        128: SLoadB128,
                        64:  SLoadB64,
                        32:  SLoadB32
                    }[dword * 32]
            item = SLoadBX(dst=sgpr(dst, dword), base=sgpr(srcAddr, 2), soffset=hex(self.kernArgOffset) if sgprOffset == None else sgprOffset )
        else:
            item = TextBlock("Move offset by %u\n" % size)
        self.kernArgOffset += size if sgprOffset == None else 0
        return item

    def loadAllKernArg(self, sgprStartIndex: int, srcAddr: Union[int, str], \
                    numSgprToLoad: int, numSgprPreload: int=0) -> Module:
        module = Module("LoadAllKernArg")
        actualLoad = numSgprToLoad - numSgprPreload
        sgprStartIndex += numSgprPreload
        self.kernArgOffset += numSgprPreload * 4
        while actualLoad > 0:
            i = 16 # 16, 8, 4, 2, 1
            while i >= 1:
                isSgprAligned = False
                if (i >= 4) and (sgprStartIndex % 4 == 0):
                  isSgprAligned = True
                elif (i == 2) and (sgprStartIndex % 2 == 0):
                  isSgprAligned = True
                elif i == 1:
                  isSgprAligned = True

                if isSgprAligned and actualLoad >= i:
                    actualLoad -= i
                    SLoadBX = { 512: SLoadB512,
                                256: SLoadB256,
                                128: SLoadB128,
                                64:  SLoadB64,
                                32:  SLoadB32
                            }[i * 32]
                    module.add(SLoadBX(dst=sgpr(sgprStartIndex, i), base=sgpr(srcAddr, 2), soffset=hex(self.kernArgOffset)))
                    sgprStartIndex += i
                    self.kernArgOffset += i * 4
                    break
                i = i // 2
        return module
        # currently align sgpr to kernel argument memory, and use s_load_bxxx to load argument as large as possible in one instruction
        # however, in order to match sgpr to kernel argument memory, some unnecessarily sgpr will also be defined, and caused wasting of sgpr.
        # TODO: more efficient way is to organize both sgpr and kernel argument memory in API

##############################################################################
# Assert
##############################################################################

def bomb(scratchVgpr, cookie=None):
    """
    Cause a GPUVM fault.
    Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the
    backtrace. Useful for locating which spot in code generated the bomb
    vgprAddr controls which vgpr to overwrite with the null pointer address
    """

    module = Module("bomb")
    vgprAddr = scratchVgpr

    if cookie != None:
        if cookie < 0:
            module.add(Label("bomb_neg%u" % abs(cookie), ""))
        else:
            module.add(Label("bomb_%u" % abs(cookie), ""))
    module.add(VMovB32(dst=vgpr(vgprAddr+0), src=0))
    module.add(VMovB32(dst=vgpr(vgprAddr+1), src=0))
    module.add(FlatLoadB32(dst=vgpr(vgprAddr), vaddr=vgpr(vgprAddr,2), comment="bomb - force fault" ))

    # This move does not execute but appears in the instruction stream immediately following
    # the faulting load:
    if cookie != None:
        module.add(SMovB32(dst=sgpr(0), src=cookie, comment="bomb cookie=%d(0x%x)"%(cookie,cookie&0xffffffff)))

    return module

class Assert():
    def __init__(self, laneSGPRCount, wavefrontSize, enableAsserts):
        self.printedAssertCnt = 0
        self.laneSGPRCount = laneSGPRCount
        self.wavefrontSize = wavefrontSize
        self.enableAsserts = enableAsserts

    ##############################################################################
    # assertCommon : Common routine for all assert functions.
    # On entry, we have already set the exec-mask so any enabled lanes should bomb
    ##############################################################################
    def assertCommon(self, vtmp, cookie=-1):
        module = Module("assertCommon")
        if self.enableAsserts:
            self.printedAssertCnt += 1
            # Default cookie for asserts is negative of printed #asserts
            # Can be used to roughly identify which assert in the code is firing
            module.add(bomb(vtmp, cookie if cookie != -1 else -self.printedAssertCnt))
        return module

    ##############################################################################
    # assertCmpCommon : Common routine for all assert comparison functions
    ##############################################################################
    def assertCmpCommon(self, inst, val0, val1, vtmp, cookie=-1):
        assert issubclass(inst, VCmpXInstruction)
        module = Module("assertCmpCommon")
        if self.enableAsserts:
            SOrSaveExecBX = SOrSaveExecB64 if self.wavefrontSize == 64 else SOrSaveExecB32
            module.add(SOrSaveExecBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=0, \
                comment="assert: saved execmask"))
            module.add(inst(dst=VCC(), src0=val0, src1=val1, comment="v_cmp")) # type: ignore
            module.add(self.assertCommon(vtmp, cookie))
            module.add(SOrSaveExecBX(dst=VCC(), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: restore execmask"))
        return module

    ##############################################################################
    # Handle different conditions for the asserts:
    # These support uin32 compare, float could be added later
    # Asserts currently modify vcc
    ##############################################################################
    def eq(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXNeU32, val0, val1, vtmp, cookie)

    def eq_u16(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXNeU16, val0, val1, vtmp, cookie)

    def ne(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXEqU32, val0, val1, vtmp, cookie)

    def lt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXGeU32, val0, val1, vtmp, cookie)

    def gt_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLeU32, val0, val1, vtmp, cookie)

    def le_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXGtU32, val0, val1, vtmp, cookie)

    def ge_u32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLtU32, val0, val1, vtmp, cookie)

    def ge_i32(self, val0, val1, vtmp, cookie=-1):
        return self.assertCmpCommon(VCmpXLtI32, val0, val1, vtmp, cookie)

    # can left shift w/o losing non-zero bits:
    def no_shift_of(self, val0, shift, stmp, vtmp, cookie=-1):
        module = Module("Assert no shift of")
        # TODO - use BFE here:
        module.add(SMovB32(dst=stmp, src=hex((shift-1) << (32-log2(shift))), comment="assert_no_shift_of - compute mask"))
        module.add(SAndB32(dst=stmp, src0=stmp, src1=val0, comment="assert_no_shift_of"))
        module.add(self.eq(stmp, 0, vtmp, cookie))
        return module

    # asserts if val0 is not an integer multiple of multiple2
    # multiple2 must be a constant and power of 2
    # for example assert_multiple(A, 8) will assert if A is not multiple of 8
    def multiple_b32(self, sval, multiple2, vtmp, cookie=-1):
        module = Module("Assert multiple b32")
        if self.enableAsserts:

            stmp = sgpr("SaveExecMask") # repurpose to get a tmp sgpr
            SAndBX = SAndB64 if self.wavefrontSize else SAndB32
            module.add(SAndBX(dst=stmp, src0=sval, src1=multiple2-1, comment="mask" ))
            module.add(SCmpEQU32(src0=stmp, src1=0, comment="if maskedBits==0 then SCC=1 == no fault" ))
            SMovBX = SMovB64 if self.wavefrontSize else SMovB32
            module.add(SMovBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=-1))
            SCMovBX= SCMovB64 if self.wavefrontSize else SCMovB32
            module.add(SCMovBX(dst=sgpr("SaveExecMask", self.laneSGPRCount),  src=0, comment="Clear exec mask"))

            SAndSaveExecBX = SAndSaveExecB64 if self.wavefrontSize else SAndSaveExecB32
            module.add(SAndSaveExecBX(dst=sgpr("SaveExecMask",self.laneSGPRCount), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: saved execmask"))

            module.add(self.assertCommon(vtmp, cookie))

            SOrSaveExecBX = SOrSaveExecB64 if self.wavefrontSize else SOrSaveExecB32
            module.add(SOrSaveExecBX(dst=VCC(), src=sgpr("SaveExecMask",self.laneSGPRCount), \
                comment="assert: restore execmask"))

        return module

    # assert v0 + expectedScalarDiff == v1
    # Verify that each element in v1 is scalar offset from v0
    def assert_vector_diff(self, v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie=-1):
        module = Module("assert_vector_diff")
        module.add(VAddCOU32(dst=vgpr(cmpvtmp), \
                       dst1=VCC(), \
                       src0=expectedScalarDiff, \
                       src1=v0, \
                       comment="assert_vector_diff add expectedScalarDiff"))
        module.add(self.eq(vgpr(cmpvtmp), v1, vtmp, cookie))
        return module

##############################################################################
# Dump (Store to debug buffer)
##############################################################################

class Dump:
    def __init__(self, sgprDebugKernelItems: Union[int, str], vgprAddressDbg: Union[int, str], \
                maxItem: int, enableDump) -> None:
        self.sgprDebugKernelItems = sgprDebugKernelItems
        self.vgprAddressDbg       = vgprAddressDbg
        self.maxItem              = maxItem
        self.enableDump           = enableDump

    def dumpVgpr(self, vgprStore: Union[int, str], labelName: str) -> Module:
        module = Module("dump vgpr[%s]"%str(vgprStore))
        if self.enableDump:
            afterDump = -1
            if self.maxItem != -1:
                afterDump = labelName
                afterDump = Label(afterDump, "skip debug target")
                module.add(SCmpLtU32(src0=sgpr(self.sgprDebugKernelItems), src1=16))
                module.add(SCBranchSCC0(labelName=afterDump.getLabelName(), \
                        comment="skip if already wrote enough work-items" ))
                module.add(SAddU32(dst=sgpr(self.sgprDebugKernelItems), \
                        src0=sgpr(self.sgprDebugKernelItems), src1=hex(1), \
                        comment="inc items written" ))

            module.add(FlatStoreB32(vaddr=vgpr(self.vgprAddressDbg, 2), src=vgprStore, comment="debug dump store"))
            module.add(VAddCOU32(dst=vgpr(self.vgprAddressDbg), dst1=VCC(), src0=vgpr(self.vgprAddressDbg), src1=hex(4), comment="debug dump inc"))

            if self.maxItem != -1:
                assert(isinstance(afterDump, Label)) # Dummy guard in case someone remove the if above
                module.add(afterDump)

        return module

    def dumpLds(self, startU: int, numU: int, tmpVgprRes: ContinuousRegister, bpeAB: int, \
                numThreads: int, labelName: str) -> Module:
        module = Module("dump lds")
        if self.enableDump:
            assert tmpVgprRes.size > 1
            tmp     = tmpVgprRes.idx
            tmpAddr = tmp + 1
            module.addComment1("dump lds state")
            module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment=""))
            module.add(SBarrier(comment="dump LDS"))
            module.add(VLShiftLeftB32(
                dst=vgpr(tmpAddr), \
                shiftHex=hex(bpeAB), \
                src=vgpr("Serial"), \
                comment="dump lds"))
            for i in range(startU, startU+numU):
                module.add(DSLoadB32(dst=vgpr(tmp), src=vgpr(tmpAddr),
                        ds=DSModifiers(offset=(i*numThreads*4)), comment="dump lds"))
                module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="dump"))
                module.add(self.dumpVgpr(tmp, labelName))
        return module

    def dumpSgpr(self, sgprStore: Union[int, str], tmpVgprRes: ContinuousRegister, \
                labelName: str) -> Module:
        module = Module("dump sgpr[%s]"%sgprStore)
        if self.enableDump:
            assert tmpVgprRes.size > 0
            tmp = tmpVgprRes.idx
            module.add(VMovB32(dst=vgpr(tmp), src=sgprStore, comment="debug dump sgpr store"))
            module.add(self.dumpVgpr(tmp, labelName))
        return module
