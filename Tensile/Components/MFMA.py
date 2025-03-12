################################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import MFMA
from ..DataType import DataType

class WMMASelection(MFMA):
    asmCaps = {"HasWMMA": True}

    def __call__(self, writer, accOutStart, accOutEnd, in0, in1, accInStart, accInEnd, accStoreCIdx, firstIter):
        kernel = writer.kernel
        inType = kernel["ProblemType"]["DataType"].toNameAbbrev()
        neg = " neg_lo:[1,1,1]" if (inType == "i8") else ""
        inType = "iu8" if inType == "i8" else inType
        outType = kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()
        if kernel["ProblemType"]["DataType"].isComplex():
            inType = outType
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        miK = kernel["MatrixInstK"]
        # miB = kernel["MatrixInstB"]
        str0 = in1 if kernel["SourceSwap"] else in0
        str1 = in0 if kernel["SourceSwap"] else in1
        # use const 0 for src2 in firstIter case
        src2 = "0" if firstIter else "v[%u:%u]"%(accOutStart, accOutEnd)

        kStr = "v_wmma_%s_%ux%ux%u_%s v[%u+%u:%u+%u], %s, %s, %s%s%s" \
            % (outType, miM, miN, miK, inType, accInStart, accStoreCIdx, accInEnd, accStoreCIdx, str0, str1, src2, neg, writer.endLine)

        return kStr

class MFMASelection950(MFMA):
    versions = [(9,5,0)]

    def WaitCount(self, writer):
        kernel = writer.kernel
        dataType = kernel["ProblemType"]["DataType"]
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        if dataType.isSingle() or dataType.isSingleComplex() or dataType.isHalf() or dataType.isBFloat16():
            if miM == 4 and miN == 4:
                return 2
        elif dataType.isDouble() or dataType.isDoubleComplex():
            if miM == 4 and miN == 4:
                return 4
        return 0

    def __call__(self, writer, accOutStart, accOutEnd, in0, in1, accInStart, accInEnd, accStoreCIdx, firstIter):
        kernel = writer.kernel
        inType = kernel["ProblemType"]["DataType"].toNameAbbrev()
        # for F8 hybrid cases, we need to change the inType of VMFMA inst as well
        if kernel["SourceSwap"]:
            dataType = kernel["ProblemType"]["DataType"]
            if dataType.isFloat8BFloat8():
                inType = DataType("B8F8").toNameAbbrev() # change the intype from F8B8 to B8F8
            if dataType.isBFloat8Float8():
                inType = DataType("F8B8").toNameAbbrev() # change the intype from B8F8 to F8B8

        outType = kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev()
        if kernel["ProblemType"]["DataType"].isComplex():
            inType = outType
        accType = "a" if not kernel["MIArchVgpr"] else "v"
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        miK = kernel["MatrixInstK"]
        miB = kernel["MatrixInstB"]
        str0 = in1 if kernel["SourceSwap"] else in0
        str1 = in0 if kernel["SourceSwap"] else in1

        strB = ""
        if miB > 1:
            strB = "%ub_" % miB

        # use const 0 for src2 in firstIter case
        src2 = "0" if firstIter else "%s[%u:%u]"%(accType, accOutStart, accOutEnd)

        kStr = "v_mfma_%s_%ux%ux%u_%s%s %s[%u+%u:%u+%u], %s, %s, %s%s" \
            % (outType, miM, miN, miK, strB, inType, accType, accInStart, accStoreCIdx, accInEnd, accStoreCIdx, str0, str1, src2, writer.endLine)

        return kStr

class MFMASelection940(MFMA):
    versions = [(9,4,2)]

    def WaitCount(self, writer):
        kernel = writer.kernel
        dataType = kernel["ProblemType"]["DataType"]
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        if dataType.isSingle() or dataType.isSingleComplex() or dataType.isHalf() or dataType.isBFloat16():
            if miM == 4 and miN == 4:
                return 2
        elif dataType.isDouble() or dataType.isDoubleComplex():
            if miM == 4 and miN == 4:
                return 4
        return 0

    def __call__(self, writer, accOutStart, accOutEnd, in0, in1, accInStart, accInEnd, accStoreCIdx, firstIter):
        kernel = writer.kernel
        inType = kernel["ProblemType"]["F32XdlMathOp"].toNameAbbrev() if kernel["EnableF32XdlMathOp"] else kernel["ProblemType"]["DataType"].toNameAbbrev()
        # for F8 hybrid cases, we need to change the inType of VMFMA inst as well
        if kernel["SourceSwap"]:
            dataType = kernel["ProblemType"]["DataType"]
            if dataType.isFloat8BFloat8():
                inType = DataType("B8F8").toNameAbbrev() # change the intype from F8B8 to B8F8
            if dataType.isBFloat8Float8():
                inType = DataType("F8B8").toNameAbbrev() # change the intype from B8F8 to F8B8

        outType = kernel["ProblemType"]["F32XdlMathOp"].MIOutputTypeNameAbbrev() if kernel["EnableF32XdlMathOp"] else kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev()
        if kernel["ProblemType"]["DataType"].isComplex():
            inType = outType
        accType = "a" if not kernel["MIArchVgpr"] else "v"
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        miK = kernel["MatrixInstK"]
        miB = kernel["MatrixInstB"]
        str0 = in1 if kernel["SourceSwap"] else in0
        str1 = in0 if kernel["SourceSwap"] else in1

        strB = ""
        if miB > 1:
            strB = "%ub_" % miB

        # use const 0 for src2 in firstIter case
        src2 = "0" if firstIter else "%s[%u:%u]"%(accType, accOutStart, accOutEnd)

        kStr = "v_mfma_%s_%ux%ux%u_%s%s %s[%u+%u:%u+%u], %s, %s, %s%s" \
            % (outType, miM, miN, miK, strB, inType, accType, accInStart, accStoreCIdx, accInEnd, accStoreCIdx, str0, str1, src2, writer.endLine)

        return kStr

class MFMASelection(MFMA):
    versions = [(9,0,8), (9,0,10)]

    def __call__(self, writer, accOutStart, accOutEnd, in0, in1, accInStart, accInEnd, accStoreCIdx, firstIter):
        kernel = writer.kernel
        inType = "bf16" if kernel["ProblemType"]["Fp16AltImpl"] else kernel["ProblemType"]["DataType"].toNameAbbrev()
        outType = kernel["ProblemType"]["DataType"].MIOutputTypeNameAbbrev()
        if kernel["ProblemType"]["DataType"].isComplex():
            inType = outType
        accType = "a" if not kernel["MIArchVgpr"] else "v"
        mfma1k = "_1k" if (kernel["MFMA_BF16_1K"] or kernel["ProblemType"]["Fp16AltImpl"]) else ""
        miM = kernel["MatrixInstM"]
        miN = kernel["MatrixInstN"]
        miK = kernel["MatrixInstK"]
        str0 = in1 if kernel["SourceSwap"] else in0
        str1 = in0 if kernel["SourceSwap"] else in1

        # use const 0 for src2 in firstIter case
        src2 = "0" if firstIter else "%s[%u:%u]"%(accType, accOutStart, accOutEnd)

        kStr = "v_mfma_%s_%ux%ux%u%s%s %s[%u+%u:%u+%u], %s, %s, %s%s" \
                % (outType, miM, miN, miK, inType, mfma1k, accType, accInStart, accStoreCIdx, accInEnd, accStoreCIdx, str0, str1, src2, writer.endLine)

        return kStr
