################################################################################
#
# Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from math import log

########################################
# Format Instruction
########################################

def inst(*args):
    # exclude the last parameter (before comment)
    # if it is empty (needed for clang++ assembler)
    if len(args) > 2 and args[len(args)-2] == "":
        params = args[0:len(args)-2]
    else:
        params = args[0:len(args)-1]
    comment = args[len(args)-1]
    formatting = "%s"
    if len(params) > 1:
        formatting += " %s"
    for _ in range(0, len(params)-2):
        formatting += ", %s"
    instStr = formatting % (params)
    line = "%-50s // %s\n" % (instStr, comment)
    return line

########################################
# Format Trailing Comment Only
########################################

def instCommentOnly(comment=""):
    # Aligned with inst (50 chars)
    return "%-50s // %s\n" % ("", comment)

########################################
# Format GPRs
########################################

def gpr(*args):
    gprType = args[0]
    args = args[1]
    if isinstance(args[0], int):
        if len(args) == 1:
            return "%s%u"%(gprType, args[0])
        elif len(args) == 2:
            if args[1] == 1:
                return "%s%u"%(gprType, args[0])
            else:
                return "%s[%u:%u]"%(gprType, args[0], args[0]+args[1]-1)
    if isinstance(args[0], str):
        if len(args) == 1:
            return "%s[%sgpr%s]"%(gprType, gprType, args[0])
        elif len(args) == 2:
            if args[1] == 1:
                return "%s[%sgpr%s]"%(gprType, gprType, args[0])
            else:
                return "%s[%sgpr%s:%sgpr%s+%u]"%(gprType, gprType, args[0], \
                        gprType, args[0], args[1]-1)

def vgpr(*args):
    return gpr("v", args)

def sgpr(*args):
    return gpr("s", args)

def accvgpr(*args):
    return gpr("acc", args)

########################################
# Log 2
########################################

def log2(x):
    return int(log(x, 2) + 0.5)

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmpSgpr
########################################

def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, doRemainder=True, comment=""):

    dComment = "%s = %s / %s"    % (vgpr(qReg), vgpr(dReg), divisor) if (comment=="") else comment
    rComment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor) if (comment=="") else comment

    kStr = ""
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        # does not work with doRemainder and (qReg==dReg)
        assert (not (doRemainder and (qReg == dReg)))
        divisor_log2 = log2(divisor)
        kStr += inst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), dComment)
        if doRemainder:
            kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), rComment)
    else:
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        # does not work with doRemainder and (qReg==dReg or rReg==dReg or qReg==rReg)
        assert (not (doRemainder and (qReg == dReg or rReg == dReg or qReg==rReg)))
        shift = 32+1
        shiftMinus32 = shift - 32
        magic = ((2**shift) // divisor) + 1
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), dComment)
        kStr += inst("v_mul_hi_u32", vgpr(qReg), vgpr(dReg), sgpr(tmpSgpr), dComment)
        kStr += inst("v_lshrrev_b32", vgpr(qReg), hex(shiftMinus32), vgpr(qReg), dComment)
        if doRemainder:
            kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), rComment)
            kStr += inst("v_mul_lo_u32", vgpr(rReg), vgpr(qReg), sgpr(tmpSgpr), rComment)
            kStr += inst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(rReg), rComment)
    return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpSgpr, comment=""):
    rReg = -1 # unused
    kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, False, comment)
    return kStr

def vectorStaticRemainder(rReg, dReg, divisor, tmpSgpr, comment=""):
    if comment == "":
        comment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor)

    kStr = ""
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), comment)
    else:
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        # does not work with qReg==rReg
        assert (rReg != dReg)
        shift = 32+1
        shiftMinus32 = shift - 32
        magic = ((2**shift) // divisor) + 1
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), comment)
        kStr += inst("v_mul_hi_u32", vgpr(rReg), vgpr(dReg), sgpr(tmpSgpr), comment)
        kStr += inst("v_lshrrev_b32", vgpr(rReg), hex(shiftMinus32), vgpr(rReg), comment)
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), comment)
        kStr += inst("v_mul_lo_u32", vgpr(rReg), vgpr(rReg), sgpr(tmpSgpr), comment)
        kStr += inst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(rReg), comment)
    return kStr

# only used for loop unroll and GlobalSplitU and XCC mapping
# doRemainder==0 : compute quotient only
# doRemainder==1 : compute quotient and remainder
# doRemainder==2 : only compute remainder (not quotient unless required for remainder)
# dreg == dividend
# tmpSgpr must be 2 SPGRs (can be None if divisor is power of 2)
# qReg and dReg can be "sgpr[..]" or names of sgpr (will call sgpr)
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
        doRemainder=1):

    qRegSgpr = qReg if type(qReg) == str and qReg.startswith("s[") else sgpr(qReg)

    dRegSgpr = dReg if type(dReg) == str and dReg.startswith("s[") else sgpr(dReg)

    kStr = ""
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        if doRemainder != 2:
            kStr += inst("s_lshr_b32", qRegSgpr, dRegSgpr, divisor_log2, \
                    "%s = %s / %u"%(qRegSgpr, dRegSgpr, divisor) )
        if doRemainder:
            kStr += inst("s_and_b32", sgpr(rReg), (divisor-1), dRegSgpr, \
                    "%s = %s %% %u"%(sgpr(rReg), dRegSgpr, divisor) )
    else:
        # Temp register required if divisor is not power of 2
        assert qReg != tmpSgpr
        assert tmpSgpr != None
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 6:
            shift = 32+3
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        magicHi = magic // (2**16)
        magicLo = magic & (2**16-1)

        kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), hex(0), "STATIC_DIV: divisior=%s"%divisor)
        kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), hex(magicHi), dRegSgpr, "tmp1 = dividend * magic hi")
        kStr += inst("s_lshl_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(16), "left shift 16 bits")
        kStr += inst("s_mul_i32", qRegSgpr, dRegSgpr, hex(magicLo), "tmp0 = dividend * magic lo")
        kStr += inst("s_add_u32", sgpr(tmpSgpr+0), qRegSgpr, sgpr(tmpSgpr+0), "add lo")
        kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), hex(0), "add hi")
        kStr += inst("s_lshr_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(shift), "tmp1 = (dividend * magic) << shift")
        kStr += inst("s_mov_b32", qRegSgpr, sgpr(tmpSgpr), "quotient")
        if doRemainder:
            kStr += inst("s_mul_i32", sgpr(tmpSgpr), qRegSgpr, hex(divisor), "quotient*divisor")
            kStr += inst("s_sub_u32", sgpr(rReg), dRegSgpr, sgpr(tmpSgpr), "rReg = dividend - quotient*divisor")
    return kStr

########################################
# Multiply
# product register, operand register, multiplier
########################################

# vgpr operand only version
# support multiplier < 1
def vectorStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (vgpr(product), vgpr(operand), multiplier)

    if multiplier == 0:
            return inst("v_mov_b32", vgpr(product), hex(multiplier), comment)
    elif multiplier < 1:
            # use division if multiplier < 1
            multiplier = int(1/multiplier)
            return vectorStaticDivide(product, operand, multiplier, tmpSgpr, comment)
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and product == operand:
            return instCommentOnly(comment + " (multiplier is 1, do nothing)")
        else:
            return inst("v_lshlrev_b32", vgpr(product), hex(multiplier_log2), vgpr(operand), comment)
    else:
        kStr = ""
        if product == operand:
            kStr += inst("s_mov_b32", tmpSgpr, hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", vgpr(product), tmpSgpr, vgpr(operand), comment)
        else:
            kStr += inst("v_mov_b32", vgpr(product), hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", vgpr(product), vgpr(product), vgpr(operand), comment)
        return kStr


def staticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
            return inst("v_mov_b32", product, hex(multiplier), comment)
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and product == operand:
            return instCommentOnly(comment + " (multiplier is 1, do nothing)")
        else:
            return inst("v_lshlrev_b32", product, hex(multiplier_log2), operand, comment)
    else:
        kStr = ""
        if product == operand:
            kStr += inst("s_mov_b32", tmpSgpr, hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", product, tmpSgpr, operand, comment)
        else:
            kStr += inst("v_mov_b32", product, hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", product, product, operand, comment)
        return kStr


########################################
# Multiply scalar for 64bit
# product register, operand register, multiplier
########################################

def scalarStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
            return inst("s_mov_b64", product, hex(multiplier), comment)

    # TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
    assert ((multiplier & (multiplier - 1)) == 0) # assert pow of 2

    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and product == operand:
        return instCommentOnly(comment + " (multiplier is 1, do nothing)")
    else:
        # notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
        return inst("s_lshl_b64", product, operand, hex(multiplier_log2), comment)
