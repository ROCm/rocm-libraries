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

from rocisa.code import Module
from rocisa.container import vgpr, sgpr, EXEC, HWRegContainer, RegisterContainer
from rocisa.instruction import SAddCU32, SAddU32, SAndB32, SCmpLgU32, SGetRegB32, \
    SLShiftLeftB64, SLShiftRightB32, SLShiftRightB64, SMovB32, SMovB64, SMulHIU32, \
    SMulI32, SNop, SSetRegB32, SSetRegIMM32B32, SSubU32, VAddF32, VAddU32, \
    VCmpXEqU32, VCmpXGtU32, VCvtF32toU32, VCvtU32toF32, VLShiftLeftAddU32, \
    VLShiftLeftB32, VMadU32U24, VMovB32, VMulF32, VMulLOU32, VMulU32U24, \
    VRcpIFlagF32, VReadfirstlaneB32, VSubU32

from typing import Optional
from .ExtInstructions import SMulInt64to32
from .RegisterPool import ContinuousRegister
from .Utils import log2

########################################
# Scalar Magic Div
# product register, operand register, multiplier
########################################

# dividend is a symbol (constant or sgpr).  Used directly not inside automatic sgpr(..)
# dst is 2 consecutive SGPR
#   result returned in dst0. dst1 is used as a temp,
# dst[1] cannot be same as divident, dst[0] can be same as dividend and this can be useful
def scalarMagicDivExplicit(dst, dividend, magicNumber, magicAbit, magicShift):
    module = Module("scalarMagicDivExplicit")
    module.addComment1("dst1:0 = dividend(%s) / magicTag(%s)" % (dividend, magicNumber))
    module.add(SMulHIU32(dst=sgpr(dst+1), src0=dividend, src1=sgpr(magicNumber), comment="scalar magic div (magicnum)"))
    module.add(SMulI32(dst=sgpr(dst+0), src0=dividend, src1=sgpr(magicAbit), comment="scalar magic div (abit)"))
    module.add(SAddU32(dst=sgpr(dst+0), src0=sgpr(dst+0), src1=sgpr(dst+1), comment="scalar magic div (combine)"))
    module.add(SLShiftRightB32(dst=sgpr(dst+0), shiftHex=sgpr(magicShift), src=sgpr(dst+0), \
                   comment="scalar magic div (shift), quotient in s%s"%dst))
    return module

def scalarMagicDiv(dst, dividend, magicTag):
    return scalarMagicDivExplicit(dst, dividend,
                                  magicNumber="MagicNumberSize"+magicTag,
                                  magicAbit="MagicAbitSize"+magicTag,
                                  magicShift="MagicShiftSize"+magicTag)

##############################################################################
# Perform a magic division (mul by magic number and shift)
# dest is two consec SGPR, used for intermediate temp as well as final result
# result quotient returned in sgpr(dest,1)
# tmpVgpr: Size 2
##############################################################################
def sMagicDiv(dest, hasSMulHi, dividend, magicNumber, magicShift, tmpVgpr):
    module = Module("sMagicDiv")
    module.addModuleAsFlatItems(SMulInt64to32(hasSMulHi, \
                                sgpr(dest), sgpr(dest+1), dividend, magicNumber, \
                                False, tmpVgpr, "s_magic mul"))
    module.add(SLShiftRightB64(dst=sgpr(dest,2), shiftHex=magicShift, src=sgpr(dest,2), comment="sMagicDiv"))
    return module

##############################################################################
# Perform a sgpr version of magic division algo 2 (mul by magic number, Abit and shift)
# dest is three consec SGPR, used for intermediate temp as well as final result
# result quotient returned in sgpr(dest,1)
##############################################################################
def sMagicDivAlg2(dest, dividend, magicNumber, magicShiftAbit):
    # dest+0: q,
    # dest+1: intermediate for magic div
    # dest+2: A tmpS to store the 'Abit' and the final Shift (use tmpS to save sgpr)
    tmpS = dest+2

    module = Module("sMagicDivAlg2")
    module.add(SMulHIU32(dst=sgpr(dest+1), src0=dividend, src1=magicNumber, comment=" s_magic mul, div alg 2"))
    module.add(SLShiftRightB32(dst=sgpr(tmpS), shiftHex=31, src=magicShiftAbit, comment=" tmpS = extract abit"))                             # tmpS = MagicAbit
    module.add(SMulI32(dst=sgpr(dest), src0=dividend, src1=sgpr(tmpS), comment=" s_magic mul, div alg 2"))
    module.add(SAddU32(dst=sgpr(dest), src0=sgpr(dest), src1=sgpr(dest+1), comment=""))

    module.add(SAndB32(dst=sgpr(tmpS), src0=magicShiftAbit, src1=hex(0x7fffffff), comment=" tmpS = remove abit to final shift"))   # tmpS = MagicShift
    module.add(SLShiftRightB32(dst=sgpr(dest), shiftHex=sgpr(tmpS), src=sgpr(dest), comment=" sMagicDiv Alg 2"))
    return module

########################################
# Multiply
# product register, operand register, multiplier
########################################

def staticMultiply(product, operand, multiplier, tmpSgprRes: Optional[ContinuousRegister], comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    module = Module("staticMultiply")
    if multiplier == 0:
        module.add(VMovB32(dst=product, src=hex(multiplier), comment=comment))
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and ((str(product) == operand or product == operand) or product == operand):
            module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
        else:
            module.add(VLShiftLeftB32(dst=product, shiftHex=hex(multiplier_log2), src=operand, comment=comment))
    else:
        if multiplier <= 64 and multiplier >= -16:
            module.add(VMulLOU32(dst=product, src0=hex(multiplier), src1=operand, comment=comment))
        else:
            assert tmpSgprRes and tmpSgprRes.size >= 1
            tmpSgpr = tmpSgprRes.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(multiplier), comment=comment))
            module.add(VMulLOU32(dst=product, src0=sgpr(tmpSgpr), src1=operand, comment=comment))
    return module

########################################
# MultiplyAdd
# product register, operand register, multiplier, accumulator
########################################

def staticMultiplyAdd(product, operand, multiplier, accumulator, tmpSgprRes: Optional[ContinuousRegister], comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    module = Module("staticMultiply")
    if multiplier == 0:
        module.add(VMovB32(dst=product, src=hex(multiplier), comment=comment))
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and (str(product) == operand or product == operand):
            module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
        else:
            module.add(VLShiftLeftAddU32(dst=product, shiftHex=hex(multiplier_log2), src0=operand, src1=accumulator, comment=comment))
    else: # not pow of 2
        if multiplier <= 64 and multiplier >= -16:
            module.add(VMadU32U24(dst=product, src0=hex(multiplier), src1=operand, src2=accumulator, comment=comment))
        else:
            assert tmpSgprRes and tmpSgprRes.size >= 1
            tmpSgpr = tmpSgprRes.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(multiplier), comment=comment))
            module.add(VMadU32U24(dst=product, src0=sgpr(tmpSgpr), src1=operand, src2=accumulator, comment=comment))
    return module

########################################
# Multiply scalar for 64bit
# product register, operand register, multiplier
########################################

def scalarStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    module = Module("scalarStaticMultiply")
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
        module.add(SMovB64(dst=product, src=hex(multiplier), comment=comment))
        return module

    # TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
    assert ((multiplier & (multiplier - 1)) == 0) # assert pow of 2

    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and (str(product) == operand or product == operand):
        module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
    else:
        # notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
        module.add(SLShiftLeftB64(dst=product, shiftHex=hex(multiplier_log2), src=operand, comment=comment))
    return module
