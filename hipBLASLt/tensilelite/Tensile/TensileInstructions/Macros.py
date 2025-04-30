################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from .Code import Macro, Module
from .Containers import VCC
from .Instructions import SAndB32, SAndB64, VAddCOU32, VAddU32, \
                        VCmpGEU32, VCmpLeU32, VCmpNeI32, VCndMaskB32, \
                        VCvtF32toU32, VCvtU32toF32, VMulF32, \
                        VMulHIU32, VMulLOU32, VRcpF32, VLShiftRightB32, \
                        VLShiftRightB64, VSubCoU32, \
                        VXorB32, VMulU32U24, VAndB32, VLShiftLeftB32, _VLShiftLeftOrB32
from .Utils import vgpr, sgpr

# Performs a division using 'magic number' computed on host
# Argument requirements:
#   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
#   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
#   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
def MacroVMagicDiv(magicDivAlg) -> Module:
    module = Module("defineMagicDivMacros")
    module.addComment1("Magic div and mod functions")
    macro = Macro("V_MAGIC_DIV", "dstIdx:req", "dividend:req", "magicNumber:req", "magicShift:req", "magicA:req")
    if magicDivAlg==1: # TODO: remove me
        macro.add(VMulHIU32(dst="v[\\dstIdx+1]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst="v[\\dstIdx+0]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VLShiftRightB64(dst="v[\\dstIdx:\\dstIdx+1]", shiftHex="\\magicShift", src="v[\\dstIdx:\\dstIdx+1]"))
    elif magicDivAlg==2:
        macro.add(VMulHIU32(dst="v[\\dstIdx+1]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst="v[\\dstIdx+0]", src0="\\dividend", src1="\\magicA"))
        macro.add(VAddU32(dst="v[\\dstIdx+0]", src0="v[\\dstIdx+0]", src1="v[\\dstIdx+1]"))
        macro.add(VLShiftRightB32(dst="v[\\dstIdx+0]", shiftHex="\\magicShift", src="v[\\dstIdx+0]"))
    module.add(macro)
    return module

def MacroVDynamicScalarDiv(wavefrontSize) -> Module:
    module = Module("Dynamic scalar divide macros")
    module.addComment1("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
    macro = Macro("DYNAMIC_VECTOR_DIVIDE", "vQuotient", "vRemainder", "vDividend", "vDivisor", "vTmp0", "vTmp1", "sTmp")
    sTmpStr = "s[\\sTmp]" if (wavefrontSize == 32) else "s[\\sTmp:\\sTmp+1]"
    macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))
    macro.add(VRcpF32(dst="v[\\vQuotient]", src="v[\\vQuotient]"))
    macro.add(VMulF32(dst="v[\\vQuotient]", src0=hex(0x4f800000), src1="v[\\vQuotient]"))
    macro.add(VCvtF32toU32(dst="v[\\vQuotient]", src="v[\\vQuotient]"))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vDivisor]", src1="v[\\vQuotient]"))
    macro.add(VMulHIU32(dst="v[\\vTmp0]", src0="v[\\vDivisor]", src1="v[\\vQuotient]"))
    macro.add(VSubCoU32(dst="v[\\vTmp1]", dst1=VCC(), src0=hex(0), src1="v[\\vRemainder]"))
    macro.add(VCmpNeI32(dst=sTmpStr, src0=hex(0), src1="v[\\vTmp0]"))
    macro.add(VCndMaskB32(dst="v[\\vRemainder]", src0="v[\\vTmp1]", src1="v[\\vRemainder]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst="v[\\vRemainder]", src0="v[\\vRemainder]", src1="v[\\vQuotient]"))
    macro.add(VSubCoU32(dst="v[\\vTmp0]", dst1=VCC(), src0="v[\\vQuotient]", src1="v[\\vRemainder]"))
    macro.add(VAddCOU32(dst="v[\\vQuotient]", dst1=VCC(), src0="v[\\vQuotient]", src1="v[\\vRemainder]"))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vTmp0]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vDividend]"))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vQuotient]", src1="v[\\vDivisor]"))
    macro.add(VSubCoU32(dst="v[\\vTmp0]", dst1=VCC(), src0="v[\\vDividend]", src1="v[\\vRemainder]"))
    macro.add(VCmpGEU32(dst=sTmpStr, src0="v[\\vDividend]", src1="v[\\vRemainder]"))
    macro.add(VAddCOU32(dst="v[\\vRemainder]", dst1=VCC(), src0=hex(1), src1="v[\\vQuotient]"))
    macro.add(VAddCOU32(dst="v[\\vTmp1]", dst1=VCC(), src0=-1, src1="v[\\vQuotient]"))
    macro.add(VCmpLeU32(dst=VCC(), src0="v[\\vDivisor]", src1="v[\\vTmp0]"))
    SAndBX = SAndB32 if wavefrontSize == 32 else SAndB64
    macro.add(SAndBX(dst=VCC(), src0=sTmpStr, src1=VCC()))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vRemainder]", src2=VCC()))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vTmp1]",     src1="v[\\vQuotient]", src2=sTmpStr)) # type: ignore
    macro.add(VCmpNeI32(dst=VCC(), src0=hex(0), src1="v[\\vDivisor]"))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0=-1, src1="v[\\vQuotient]", src2=VCC(), comment="final result" ))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vQuotient]", src1="v[\\vDivisor]"))
    macro.add(VSubCoU32(dst="v[\\vRemainder]", dst1=VCC(), src0="v[\\vDividend]", src1="v[\\vRemainder]", comment="final result" ))
    module.add(macro)
    return module


def PseudoRandomGenerator() -> Module:
    ### modified from Tensile/.../PseudoRandomGenerator.py

    module = Module("Custom Pseudo Random Generator") # Custom?
    module.addComment1("PRND_GENERATOR: vRand=RND(vAcc, sSeed, vTid)")
    macro = Macro("PRND_GENERATOR", "vRand", "vAcc", "vTemp0", "vTemp1")

    # V Logic
    macro.add(VAndB32(dst="v[\\vTemp0]", src0="0xFFFF", src1="\\vAcc", comment="vTemp0 = vAcc & 0xFFFF"))
    macro.add(VLShiftRightB32(dst="v[\\vTemp1]", shiftHex=hex(16), src="\\vAcc", comment="vTemp1 = vAcc >> 16"))
    macro.add(VXorB32(dst="v[\\vTemp0]", src0="v[\\vTemp0]", src1="v[\\vTemp1]", comment="VTemp0 = vTemp0 ^ vTemp1"))
    macro.add(VAndB32("v[\\vTemp1]", "v[\\vTemp0]", "31", comment="vTemp1 = vTemp0 & 31"))
    macro.add(VLShiftLeftB32(dst="v[\\vTemp1]", shiftHex=hex(11), src="v[\\vTemp1]", comment="vTemp1 = vTemp1 << 11"))
    macro.add(_VLShiftLeftOrB32(dst="v[\\vTemp0]", shiftHex="v[\\vTemp0]", src0=hex(5), src1="v[\\vTemp1]", comment="vTemp0 = vTemp0 << 5 | vTemp1"))
    macro.add(VMulU32U24(dst="v[\\vTemp0]", src0="0x700149" , src1="v[\\vTemp0]", comment="VTemp0 = vTemp0 * 0x700149"))   # mult lower 24 bits should be enough??
    macro.add(VMulU32U24(dst="v[\\vTemp1]", src0=229791 , src1=vgpr("Serial"), comment="VTemp1 = vTid * 229791"))  # TODO: use index of C/D instead of local Tid
    macro.add(VXorB32(dst="v[\\vRand]", src0="0x1337137", src1="v[\\vTemp0]", comment="VRand = vTemp0 ^ 0x1337137"))
    macro.add(VXorB32(dst="v[\\vRand]", src0="v[\\vRand]", src1="v[\\vTemp1]", comment="VRand = vRand ^ vTemp1"))
    macro.add(VXorB32(dst="v[\\vRand]", src0="v[\\vRand]", src1=sgpr("RNDSeed"), comment="VRand = vRand ^ sSeed"))

    ## NOTE: Some ideas on validation:
    #     1. to test with existing validator: if we use integer initialization pattern and the output is <=16, it will work since no rounding for int up to 16.0 for fp8.
    #     2. We can use same RND (e.g., 0) in both reference and gpu kernel by commenting out following line.
    #     3. If we use 0xFFFFFFFF, cvt_sr will always round the value up. So, tests with existing validator may fail if we don't ensure this in reference kernel of Tensile host
    #     4. A better way to validate:
    #        Fix the value of RNDSeed from the caller, Save the output of this macro-function and compare it with quantization kernel's (TF-SIM's) output.
    #macro.add("v_mov_b32", "v[\\vRand]", "0x0", "vRand = 0x0" )
    #macro.add("v_mov_b32", "v[\\vRand]", "0xFFFFFFFF", "VRand = 0xffffffff" )
    ###macro.add("v_mov_b32", "v[\\vRand]", sgpr("RNDSeed"), "vRand = RNDSeed" )

    module.add(macro)
    return module
