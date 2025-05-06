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

################################################################################
################################################################################
###
###   Class ExtInstructions
###
################################################################################
################################################################################

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
