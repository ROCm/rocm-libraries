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

from rocisa.code import Macro, Module
from rocisa.container import VCC, vgpr, sgpr
from rocisa.instruction import SAndB32, SAndB64, VAddCOU32, VAddU32, \
                        VCmpGEU32, VCmpLeU32, VCmpNeI32, VCndMaskB32, \
                        VCvtF32toU32, VCvtU32toF32, VMulF32, \
                        VMulHIU32, VMulLOU32, VRcpF32, VLShiftRightB32, \
                        VLShiftRightB64, VSubCoU32, \
                        VXorB32, VMulU32U24, VAndB32, VLShiftLeftB32, _VLShiftLeftOrB32

# Performs a division using 'magic number' computed on host
# Argument requirements:
#   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
#   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
#   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
def MacroVMagicDiv(magicDivAlg) -> Module:
    module = Module("defineMagicDivMacros")
    module.addComment1("Magic div and mod functions")
    macro = Macro("V_MAGIC_DIV", ["vgprDstIdx:req", "dividend:req", "magicNumber:req", "magicShift:req", "magicA:req"])
    if magicDivAlg==1: # TODO: remove me
        macro.add(VMulHIU32(dst=vgpr("DstIdx+1", isMacro=True), src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst=vgpr("DstIdx+0", isMacro=True), src0="\\dividend", src1="\\magicNumber"))
        macro.add(VLShiftRightB64(dst=vgpr("DstIdx", 2, isMacro=True), shiftHex="\\magicShift", src="v[\\vgprDstIdx:\\vgprDstIdx+1]"))
    elif magicDivAlg==2:
        macro.add(VMulHIU32(dst=vgpr("DstIdx+1", isMacro=True), src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst=vgpr("DstIdx+0", isMacro=True), src0="\\dividend", src1="\\magicA"))
        macro.add(VAddU32(dst=vgpr("DstIdx+0", isMacro=True), src0="v[\\vgprDstIdx+0]", src1="v[\\vgprDstIdx+1]"))
        macro.add(VLShiftRightB32(dst=vgpr("DstIdx+0", isMacro=True), shiftHex="\\magicShift", src="v[\\vgprDstIdx+0]"))
    module.add(macro)
    return module

def MacroVDynamicScalarDiv(wavefrontSize) -> Module:
    module = Module("Dynamic scalar divide macros")
    module.addComment1("Dynamic Scalar Divide: vgprQuotient=vgprDividend/vgprDivisor; vgprRemainder=vgprDividend%vgprDivisor;")
    macro = Macro("DYNAMIC_VECTOR_DIVIDE", ["vgprQuotient", "vgprRemainder", "vgprDividend", "vgprDivisor", "vgprTmp0", "vgprTmp1", "sgprTmp"])
    sTmpStr = sgpr("Tmp", isMacro=True) if (wavefrontSize == 32) else sgpr("Tmp", 2, isMacro=True)
    macro.add(VCvtU32toF32(dst=vgpr("Quotient", isMacro=True), src="v[\\vgprDivisor]"))
    macro.add(VRcpF32(dst=vgpr("Quotient", isMacro=True), src="v[\\vgprQuotient]"))
    macro.add(VMulF32(dst=vgpr("Quotient", isMacro=True), src0=hex(0x4f800000), src1="v[\\vgprQuotient]"))
    macro.add(VCvtF32toU32(dst=vgpr("Quotient", isMacro=True), src="v[\\vgprQuotient]"))
    macro.add(VMulLOU32(dst=vgpr("Remainder", isMacro=True), src0="v[\\vgprDivisor]", src1="v[\\vgprQuotient]"))
    macro.add(VMulHIU32(dst=vgpr("Tmp0", isMacro=True), src0="v[\\vgprDivisor]", src1="v[\\vgprQuotient]"))
    macro.add(VSubCoU32(dst=vgpr("Tmp1", isMacro=True), dst1=VCC(), src0=0, src1="v[\\vgprRemainder]"))
    macro.add(VCmpNeI32(dst=sTmpStr, src0=0, src1="v[\\vgprTmp0]"))
    macro.add(VCndMaskB32(dst=vgpr("Remainder", isMacro=True), src0="v[\\vgprTmp1]", src1="v[\\vgprRemainder]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst=vgpr("Remainder", isMacro=True), src0="v[\\vgprRemainder]", src1="v[\\vgprQuotient]"))
    macro.add(VSubCoU32(dst=vgpr("Tmp0", isMacro=True), dst1=VCC(), src0="v[\\vgprQuotient]", src1="v[\\vgprRemainder]"))
    macro.add(VAddCOU32(dst=vgpr("Quotient", isMacro=True), dst1=VCC(), src0="v[\\vgprQuotient]", src1="v[\\vgprRemainder]"))
    macro.add(VCndMaskB32(dst=vgpr("Quotient", isMacro=True), src0="v[\\vgprQuotient]", src1="v[\\vgprTmp0]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst=vgpr("Quotient", isMacro=True), src0="v[\\vgprQuotient]", src1="v[\\vgprDividend]"))
    macro.add(VMulLOU32(dst=vgpr("Remainder", isMacro=True), src0="v[\\vgprQuotient]", src1="v[\\vgprDivisor]"))
    macro.add(VSubCoU32(dst=vgpr("Tmp0", isMacro=True), dst1=VCC(), src0="v[\\vgprDividend]", src1="v[\\vgprRemainder]"))
    macro.add(VCmpGEU32(dst=sTmpStr, src0="v[\\vgprDividend]", src1="v[\\vgprRemainder]"))
    macro.add(VAddCOU32(dst=vgpr("Remainder", isMacro=True), dst1=VCC(), src0=hex(1), src1="v[\\vgprQuotient]"))
    macro.add(VAddCOU32(dst=vgpr("Tmp1", isMacro=True), dst1=VCC(), src0=-1, src1="v[\\vgprQuotient]"))
    macro.add(VCmpLeU32(dst=VCC(), src0="v[\\vgprDivisor]", src1="v[\\vgprTmp0]"))
    SAndBX = SAndB32 if wavefrontSize == 32 else SAndB64
    macro.add(SAndBX(dst=VCC(), src0=sTmpStr, src1=VCC()))
    macro.add(VCndMaskB32(dst=vgpr("Quotient", isMacro=True), src0="v[\\vgprQuotient]", src1="v[\\vgprRemainder]", src2=VCC()))
    macro.add(VCndMaskB32(dst=vgpr("Quotient", isMacro=True), src0="v[\\vgprTmp1]",     src1="v[\\vgprQuotient]", src2=sTmpStr)) # type: ignore
    macro.add(VCmpNeI32(dst=VCC(), src0=0, src1="v[\\vgprDivisor]"))
    macro.add(VCndMaskB32(dst=vgpr("Quotient", isMacro=True), src0=-1, src1="v[\\vgprQuotient]", src2=VCC(), comment="final result" ))
    macro.add(VMulLOU32(dst=vgpr("Remainder", isMacro=True), src0="v[\\vgprQuotient]", src1="v[\\vgprDivisor]"))
    macro.add(VSubCoU32(dst=vgpr("Remainder", isMacro=True), dst1=VCC(), src0="v[\\vgprDividend]", src1="v[\\vgprRemainder]", comment="final result" ))
    module.add(macro)
    return module


def PseudoRandomGenerator() -> Module:
    ### modified from Tensile/.../PseudoRandomGenerator.py

    module = Module("Custom Pseudo Random Generator") # Custom?
    module.addComment1("PRND_GENERATOR: vgprRand=RND(vgprAcc, sgprSeed, vgprTid)")
    macro = Macro("PRND_GENERATOR", ["vgprRand", "vgprAcc", "vgprTemp0", "vgprTemp1"])

    # V Logic
    macro.add(VAndB32(dst=vgpr("Temp0", isMacro=True), src0="0xFFFF", src1="\\vgprAcc", comment="vgprTemp0 = vgprAcc & 0xFFFF"))
    macro.add(VLShiftRightB32(dst=vgpr("Temp1", isMacro=True), shiftHex=hex(16), src="\\vgprAcc", comment="vgprTemp1 = vgprAcc >> 16"))
    macro.add(VXorB32(dst=vgpr("Temp0", isMacro=True), src0="v[\\vgprTemp0]", src1="v[\\vgprTemp1]", comment="VTemp0 = vgprTemp0 ^ vgprTemp1"))
    macro.add(VAndB32(vgpr("Temp1", isMacro=True), "v[\\vgprTemp0]", "31", comment="vgprTemp1 = vgprTemp0 & 31"))
    macro.add(VLShiftLeftB32(dst=vgpr("Temp1", isMacro=True), shiftHex=hex(11), src="v[\\vgprTemp1]", comment="vgprTemp1 = vgprTemp1 << 11"))
    macro.add(_VLShiftLeftOrB32(dst=vgpr("Temp0", isMacro=True), shiftHex="v[\\vgprTemp0]", src0=hex(5), src1="v[\\vgprTemp1]", comment="vgprTemp0 = vgprTemp0 << 5 | vgprTemp1"))
    macro.add(VMulU32U24(dst=vgpr("Temp0", isMacro=True), src0="0x700149" , src1="v[\\vgprTemp0]", comment="VTemp0 = vgprTemp0 * 0x700149"))   # mult lower 24 bits should be enough??
    macro.add(VMulU32U24(dst=vgpr("Temp1", isMacro=True), src0=229791 , src1=vgpr("Serial"), comment="VTemp1 = vTid * 229791"))  # TODO: use index of C/D instead of local Tid
    macro.add(VXorB32(dst=vgpr("Rand", isMacro=True), src0="0x1337137", src1="v[\\vgprTemp0]", comment="VRand = vgprTemp0 ^ 0x1337137"))
    macro.add(VXorB32(dst=vgpr("Rand", isMacro=True), src0="v[\\vgprRand]", src1="v[\\vgprTemp1]", comment="VRand = vgprRand ^ vgprTemp1"))
    macro.add(VXorB32(dst=vgpr("Rand", isMacro=True), src0="v[\\vgprRand]", src1=sgpr("RNDSeed"), comment="VRand = vgprRand ^ sSeed"))

    ## NOTE: Some ideas on validation:
    #     1. to test with existing validator: if we use integer initialization pattern and the output is <=16, it will work since no rounding for int up to 16.0 for fp8.
    #     2. We can use same RND (e.g., 0) in both reference and gpu kernel by commenting out following line.
    #     3. If we use 0xFFFFFFFF, cvt_sr will always round the value up. So, tests with existing validator may fail if we don't ensure this in reference kernel of Tensile host
    #     4. A better way to validate:
    #        Fix the value of RNDSeed from the caller, Save the output of this macro-function and compare it with quantization kernel's (TF-SIM's) output.
    #macro.add("v_mov_b32", "v[\\vgprRand]", "0x0", "vgprRand = 0x0" )
    #macro.add("v_mov_b32", "v[\\vgprRand]", "0xFFFFFFFF", "VRand = 0xffffffff" )
    ###macro.add("v_mov_b32", "v[\\vgprRand]", sgpr("RNDSeed"), "vgprRand = RNDSeed" )

    module.add(macro)
    return module
