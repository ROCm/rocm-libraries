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
from rocisa.code import Module
from rocisa.container import vgpr, sgpr,SDWAModifiers, VOP3PModifiers
from rocisa.enum import DataTypeEnum, SelectBit, UnusedBit, SaturateCastType
from rocisa.instruction import VAdd3U32, VCvtF32toF16, VLShiftRightB32, \
                            VCmpUF32, VCndMaskB32, VCvtPkF32toFP8, VCvtPkF32toBF8, \
                            VCmpClassF32, VOrB32, VPackF16toB32, \
                            VAndOrB32, VBfeU32, VLShiftLeftB16, SNop, VMed3F32, \
                            VCvtPkF32toBF16, VCvtPkF32toF16, VAndB32, \
                            VMovB32, VLShiftLeftB32, VCvtScalePk8F32toFP8, VCvtScalePk8F32toBF8, \
                            VCvtSRF32toFP8, VCvtScaleSRPkF32toFP8, VPrngB32, MacroInstruction
from rocisa.functions import VSaturateCastInt
from rocisa.macro import PseudoRandomGeneratorModule

from ..Common.DataType import DataType
from ..Component import PackData
from rocisa.instruction import ECvtF32toF16

def formatting(idx, inputPrefix, prefixOffset):
    if inputPrefix:
        return (inputPrefix + "%u"%(idx-prefixOffset))
    else:
        return idx

class PackData_F16(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Half)}}
    def __call__(self, gwvw, destIdx, elementSumIdx, tmpVgpr=None, inputPrefix="", prefixOffset=0):
        module = Module("PackData F16")
        ti = rocIsa.getInstance()
        if gwvw == 1:
            formatVgpr = formatting(elementSumIdx, inputPrefix, prefixOffset)
            module.add(ECvtF32toF16(dst=vgpr(destIdx), src=vgpr(formatVgpr), comment="convert C to fp16"))
            return module

        assert (gwvw % 2 == 0)
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            formatVgpr_1 = formatting(sumIdxV - 1, inputPrefix, prefixOffset)
            if tmpVgpr:
                tmpDst   = tmpVgpr + vi
                tmpDst_1 = tmpVgpr + vi - 1
            else:
                tmpDst   = formatVgpr
                tmpDst_1 = formatting(sumIdxV-1, inputPrefix, prefixOffset)

            if ti.getAsmCaps()["HasPkF16CVT"]:
                # VCvtPkF32toF16: convert and pack a pair of elements by one instruction
                if vi%2 == 1:
                    d = destIdx + vi//2
                    module.add(VCvtPkF32toF16(dst=vgpr(d), src0=vgpr(formatVgpr_1), src1=vgpr(formatVgpr), comment="convert C to fp16 and pack with neighbor"))
            else:
                module.add(ECvtF32toF16(dst=vgpr(tmpDst), src=vgpr(formatVgpr), comment="convert C to fp16"))
                if vi%2 == 1:
                    d = destIdx + vi//2
                    module.add(VPackF16toB32(dst=vgpr(d), src0=vgpr(tmpDst_1), src1=vgpr(tmpDst), \
                            comment="Pack with neighbor"))
        return module

class PackData_BF16(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.BFloat16)}}
    def __call__(self, gwvw, destIdx, elementSumIdx, bf16CVTVgprStruct, tmpS01, laneSGPRC, tmpVgpr=None, inputPrefix="", prefixOffset=0):
        ti = rocIsa.getInstance()

        module = Module("PackData BF16")
        if gwvw == 1:
            vgprBf16Temp = bf16CVTVgprStruct.vgprBf16Temp
            vgprBf16Inc = bf16CVTVgprStruct.vgprBf16Inc
            vgprFp32Nan = bf16CVTVgprStruct.vgprFp32Nan
            vgprBf16Mask = bf16CVTVgprStruct.vgprBf16Mask

            formatVgpr = formatting(elementSumIdx, inputPrefix, prefixOffset)
            if ti.getAsmCaps()["HasBF16CVT"]:
                module.add(VCvtPkF32toBF16(dst=vgpr(destIdx), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), \
                                               comment="convert C to bf16 in gwvw==1"))
            else:
                module.add(VCmpUF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), comment="check Nan"))
                module.add(VBfeU32(dst=vgpr(vgprBf16Temp), src0=vgpr(formatVgpr), src1=16, src2=1, comment="Non-Nan case: store lsb of bf16" ))
                module.add(VAdd3U32(dst=vgpr(vgprBf16Temp), src0=vgpr(formatVgpr), src1=vgpr(vgprBf16Temp), src2=vgpr(vgprBf16Inc), comment="Non-Nan case: add lsb and the increment for rounding" ))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprBf16Temp), src1=vgpr(vgprFp32Nan), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VLShiftRightB32(dst=vgpr(destIdx), shiftHex=16, src=vgpr(formatVgpr), comment="convert C to bf16"))
            return module

        assert (gwvw % 2 == 0)
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            if tmpVgpr:
                tmpDst   = tmpVgpr + vi
                tmpDst_1 = tmpVgpr + vi - 1
            else:
                tmpDst   = formatVgpr
                tmpDst_1 = formatting(sumIdxV-1, inputPrefix, prefixOffset)

            if ti.getAsmCaps()["HasBF16CVT"]:
                if vi%2 == 1:
                    d = destIdx + vi//2
                    module.add(VCvtPkF32toBF16(dst=vgpr(d), src0=vgpr(tmpDst_1), src1=vgpr(tmpDst), \
                                comment="convert C to bf16 and Pack with neighbor"))
            else:
                vgprBf16Temp = bf16CVTVgprStruct.vgprBf16Temp
                vgprBf16Inc = bf16CVTVgprStruct.vgprBf16Inc
                vgprFp32Nan = bf16CVTVgprStruct.vgprFp32Nan
                vgprBf16Mask = bf16CVTVgprStruct.vgprBf16Mask

                module.add(VCmpUF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), comment="check Nan"))
                module.add(VBfeU32(dst=vgpr(vgprBf16Temp), src0=vgpr(formatVgpr), src1=16, src2=1, comment="Non-Nan case: store lsb of bf16" ))
                module.add(VAdd3U32(dst=vgpr(vgprBf16Temp), src0=vgpr(formatVgpr), src1=vgpr(vgprBf16Temp), src2=vgpr(vgprBf16Inc), comment="Non-Nan case: add lsb and the increment for rounding" ))
                module.add(VCndMaskB32(dst=vgpr(tmpDst), src0=vgpr(vgprBf16Temp), src1=vgpr(vgprFp32Nan), src2=sgpr(tmpS01,laneSGPRC)))
                if vi%2 == 0:
                    module.add(VLShiftRightB32(dst=vgpr(tmpDst), shiftHex=16, src=vgpr(tmpDst), comment="convert C to bf16"))
                elif vi%2 == 1:
                    d = destIdx + vi//2
                    module.add(VAndOrB32(dst=vgpr(d), src0=vgpr(tmpDst), src1=vgpr(vgprBf16Mask), src2=vgpr(tmpDst_1), comment="pack two bf16 to dword"))

        return module

class PackData_FLOAT8(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Float8), "StochasticRounding": False}}
    def __call__(self, gwvw, destIdx, elementSumIdx, fp8CVTVgprStruct, tmpS01, laneSGPRC, inputPrefix="", prefixOffset=0):
        vgprFp8NanInf = fp8CVTVgprStruct.vgprFp8NanInf
        vgprFp8Temp   = fp8CVTVgprStruct.vgprFp8Temp
        vgprFp8Min    = fp8CVTVgprStruct.vgprFp8Min
        vgprFp8Max    = fp8CVTVgprStruct.vgprFp8Max

        module = Module("PackData float8")

        # Use v_cvt_scalef32_pk8_fp8_f32 when gwvw is a multiple of 8 and hardware supports it
        if gwvw % 8 == 0:
            ti = rocIsa.getInstance()
            if ti.getAsmCaps().get("HasCvtScalePk8Fp8F32", False):
                for groupIdx in range(gwvw // 8):
                    srcVgpr = formatting(elementSumIdx + groupIdx * 8, inputPrefix, prefixOffset)
                    d = destIdx + groupIdx * 2
                    module.add(VCvtScalePk8F32toFP8(dst=vgpr(d, 2), src=vgpr(srcVgpr, 8),
                                                    scale=0x3f800000, comment="convert 8xF32 to 8xFP8"))
                return module

        pos = 0
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            d = destIdx + vi//4
            if (vi + 1 >= gwvw) and (gwvw % 2 == 1):
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toFP8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,0])))
            if vi%2 == 1:
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1= vgpr(vgprFp8Min),src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src0=vgpr(vgprFp8Temp), src1=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toFP8(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,pos])))
                pos = int(not pos)
        return module

class PackData_FLOAT8_fnuz(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Float8_fnuz), "StochasticRounding": False}}
    def __call__(self, gwvw, destIdx, elementSumIdx, fp8CVTVgprStruct, tmpS01, laneSGPRC, inputPrefix="", prefixOffset=0):
        vgprFp8NanInf = fp8CVTVgprStruct.vgprFp8NanInf
        vgprFp8Temp   = fp8CVTVgprStruct.vgprFp8Temp
        vgprFp8Min    = fp8CVTVgprStruct.vgprFp8Min
        vgprFp8Max    = fp8CVTVgprStruct.vgprFp8Max

        module = Module("PackData float8_fnuz")
        pos = 0
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            d = destIdx + vi//4
            if (vi + 1 >= gwvw) and (gwvw % 2 == 1):
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toFP8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,0])))
            if vi%2 == 1:
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1= vgpr(vgprFp8Min),src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src0=vgpr(vgprFp8Temp), src1=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toFP8(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,pos])))
                pos = int(not pos)
        return module

class PackData_BF8(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.BFloat8)}}
    def __call__(self, gwvw, destIdx, elementSumIdx, bf8CVTVgprStruct, tmpS01, laneSGPRC, inputPrefix="", prefixOffset=0):
        vgprBF8NanInf = bf8CVTVgprStruct.vgprBF8NanInf
        vgprBF8Temp   = bf8CVTVgprStruct.vgprBF8Temp
        vgprBF8Min    = bf8CVTVgprStruct.vgprBF8Min
        vgprBF8Max    = bf8CVTVgprStruct.vgprBF8Max

        module = Module("PackData bfloat8")

        # Use v_cvt_scalef32_pk8_bf8_f32 when gwvw is a multiple of 8 and hardware supports it
        if gwvw % 8 == 0:
            ti = rocIsa.getInstance()
            if ti.getAsmCaps().get("HasCvtScalePk8Bf8F32", False):
                for groupIdx in range(gwvw // 8):
                    srcVgpr = formatting(elementSumIdx + groupIdx * 8, inputPrefix, prefixOffset)
                    d = destIdx + groupIdx * 2
                    module.add(VCvtScalePk8F32toBF8(dst=vgpr(d, 2), src=vgpr(srcVgpr, 8),
                                                    scale=0x3f800000, comment="convert 8xF32 to 8xBF8"))
                return module

        pos = 0
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            d = destIdx + vi//4
            if (vi + 1 >= gwvw) and (gwvw % 2 == 1):
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprBF8Min), src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprBF8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toBF8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,0])))
            if vi%2 == 1:
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1= vgpr(vgprBF8Min),src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src0=vgpr(vgprBF8Temp), src1=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src2=sgpr(tmpS01,laneSGPRC)))

                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprBF8Min), src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprBF8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toBF8(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,pos])))
                pos = int(not pos)
        return module

class PackData_BF8_fnuz(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.BFloat8_fnuz)}}
    def __call__(self, gwvw, destIdx, elementSumIdx, bf8CVTVgprStruct, tmpS01, laneSGPRC, inputPrefix="", prefixOffset=0):
        vgprBF8NanInf = bf8CVTVgprStruct.vgprBF8NanInf
        vgprBF8Temp   = bf8CVTVgprStruct.vgprBF8Temp
        vgprBF8Min    = bf8CVTVgprStruct.vgprBF8Min
        vgprBF8Max    = bf8CVTVgprStruct.vgprBF8Max

        module = Module("PackData bfloat8")
        pos = 0
        for vi in range(0, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            d = destIdx + vi//4
            if (vi + 1 >= gwvw) and (gwvw % 2 == 1):
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprBF8Min), src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprBF8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toBF8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,0])))
            if vi%2 == 1:
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1= vgpr(vgprBF8Min),src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src0=vgpr(vgprBF8Temp), src1=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src2=sgpr(tmpS01,laneSGPRC)))

                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprBF8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprBF8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprBF8Min), src2=vgpr(vgprBF8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprBF8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
                module.add(VCvtPkF32toBF8(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(formatVgpr), vop3=VOP3PModifiers(op_sel=[0,0,pos])))
                pos = int(not pos)
        return module

class PackData_INT8(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Int32), "DestDataType": DataType(DataTypeEnum.Int8)}}
    def __call__(self, gwvw, destIdx, elementSumIdx, i8CVTVgprStruct, tmpS01, SaturateTypeInt8 = SaturateCastType.NORMAL, inputPrefix="", prefixOffset=0):
        vgprI8Mask0 = i8CVTVgprStruct.vgprI8Mask0
        vgprI8Mask1 = i8CVTVgprStruct.vgprI8Mask1
        vgprI8Temp0 = i8CVTVgprStruct.vgprI8Temp0
        vgprI8Temp1 = i8CVTVgprStruct.vgprI8Temp1

        ti = rocIsa.getInstance()
        module = Module("PackData int8")
        gwvw4 = (gwvw // 4) * 4
        for vi in range(0, gwvw4):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)

            if vi%4 == 3:
                d = destIdx + vi//4
                for i in reversed(range(0, 4)):
                    module.add(VSaturateCastInt(vgpr(formatting(sumIdxV-i, inputPrefix, prefixOffset)), vgprI8Temp0, tmpS01, -128, 127, type=SaturateTypeInt8, initGpr=(i%4 == 3)))
                module.add(VLShiftLeftB16(dst=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), shiftHex=8, src=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset))))
                module.add(VLShiftLeftB16(dst=vgpr(formatting(sumIdxV-0, inputPrefix, prefixOffset)), shiftHex=8, src=vgpr(formatting(sumIdxV-0, inputPrefix, prefixOffset))))
                if ti.getArchCaps()["NoSDWA"]:
                    module.add(VMovB32(vgpr(vgprI8Mask0), "0xFF", comment="bits 7:0")) # src0_sel=SelectBit.BYTE_0
                    module.add(VAndB32(dst=vgpr(vgprI8Temp0), src0=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), \
                                       src1=vgpr(vgprI8Mask0)))
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), src0=vgpr(vgprI8Temp0), \
                                      src1=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), sdwa=None))
                else:
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), \
                                      src0=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), \
                                      src1=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), \
                                      sdwa=SDWAModifiers(dst_sel=SelectBit.DWORD, dst_unused=UnusedBit.UNUSED_PAD, \
                                                         src0_sel=SelectBit.BYTE_0, src1_sel=SelectBit.DWORD)))
                if ti.getArchCaps()["SDWAWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
                if ti.getArchCaps()["NoSDWA"]:
                    module.add(VMovB32(vgpr(vgprI8Mask0), "0xFF", comment="bits 7:0")) # src0_sel=SelectBit.BYTE_0
                    module.add(VAndB32(dst=vgpr(vgprI8Temp0), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), \
                                       src1=vgpr(vgprI8Mask0)))
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), src0=vgpr(vgprI8Temp0), src1=vgpr(formatVgpr), sdwa=None))
                    module.add(VLShiftLeftB32(dst=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), src=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), shiftHex=16, comment=""))
                else:
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), \
                                      src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), src1=vgpr(formatVgpr), \
                                      sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, dst_unused=UnusedBit.UNUSED_PAD, \
                                                         src0_sel=SelectBit.BYTE_0, src1_sel=SelectBit.DWORD)))
                if ti.getArchCaps()["SDWAWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
                if ti.getArchCaps()["NoSDWA"]:
                    module.add(VMovB32(vgpr(vgprI8Mask0), "0xFFFF", comment="bits 15:0")) # src0_sel=SelectBit.WORD_0
                    module.add(VAndB32(dst=vgpr(vgprI8Temp0), src0=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), \
                                       src1=vgpr(vgprI8Mask0)))
                    module.add(VOrB32(dst=vgpr(d), src0=vgpr(vgprI8Temp0), \
                                      src1=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), sdwa=None))
                else:
                    module.add(VOrB32(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-3, inputPrefix, prefixOffset)), \
                                      src1=vgpr(formatting(sumIdxV-2, inputPrefix, prefixOffset)), \
                                      sdwa=SDWAModifiers(dst_sel=SelectBit.DWORD, dst_unused=UnusedBit.UNUSED_PAD, \
                                                         src0_sel=SelectBit.WORD_0, src1_sel=SelectBit.DWORD)))
                if ti.getArchCaps()["SDWAWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
        # Left
        for vi in range(gwvw4, gwvw):
            sumIdxV = elementSumIdx + vi
            formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
            d = destIdx + vi//4

            if vi%2 == 1:
                for i in reversed(range(0, 2)):
                    module.add(VSaturateCastInt(vgpr(formatting(sumIdxV-i, inputPrefix, prefixOffset)), vgprI8Temp0, tmpS01, -128, 127, type=SaturateTypeInt8, initGpr=(i%2 == 1)))
                module.add(VLShiftLeftB16(dst=vgpr(formatVgpr), shiftHex=8, src=vgpr(formatVgpr)))
                if ti.getArchCaps()["NoSDWA"]:
                    module.add(VMovB32(vgpr(vgprI8Mask0), "0xFF", comment="bits 7:0")) # src0_sel=SelectBit.BYTE_0
                    module.add(VAndB32(dst=vgpr(vgprI8Temp0), src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), \
                                       src1=vgpr(vgprI8Mask0)))
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), \
                                      src0=vgpr(vgprI8Temp0), src1=vgpr(formatVgpr), sdwa=None))
                else:
                    module.add(VOrB32(dst=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), \
                                      src0=vgpr(formatting(sumIdxV-1, inputPrefix, prefixOffset)), \
                                      src1=vgpr(formatVgpr), \
                                      sdwa=SDWAModifiers(dst_sel=SelectBit.DWORD, dst_unused=UnusedBit.UNUSED_PAD, \
                                                         src0_sel=SelectBit.BYTE_0, src1_sel=SelectBit.DWORD)))
                if ti.getArchCaps()["SDWAWait"]:
                    module.add(SNop(waitState=0, comment="1 wait states"))
            elif vi + 1 >= gwvw:
                module.add(VSaturateCastInt(vgpr(formatVgpr), vgprI8Temp0, tmpS01, -128, 127, type=SaturateTypeInt8, initGpr=True))
        return module

class PackData_FLOAT8_SR(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Float8), "StochasticRounding": True}}
    def __call__(self, gwvw, destIdx, elementSumIdx, fp8CVTVgprStruct, tmpS01, laneSGPRC, vgprTmp, inputPrefix="", prefixOffset=0, alphaScale=1.0):
        """
        Pack F32 data to FP8 with stochastic rounding.
        """
        ti = rocIsa.getInstance()
        vgprFp8NanInf = fp8CVTVgprStruct.vgprFp8NanInf
        vgprFp8Temp   = fp8CVTVgprStruct.vgprFp8Temp
        vgprFp8Min    = fp8CVTVgprStruct.vgprFp8Min
        vgprFp8Max    = fp8CVTVgprStruct.vgprFp8Max
        vRand = vgprTmp #seed
        if not ti.getAsmCaps()["v_prng_b32"]:
            vTemp0 = vgprTmp+1
            vTemp1 = vgprTmp+2

        module = Module("PackData float8 with stochastic rounding")

        def addTruncateAndPRNG(formatVgpr, skipTruncate=False):
            if not skipTruncate:
                module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
                module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
                module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))
            if ti.getAsmCaps()["v_prng_b32"]:
                # NOTE: Current PRNG seed implementation simply uses the value to be converted directly as seed.
                # For thread ID-based seed design, see the legacy PRND_GENERATOR approach in tensilelite/rocisa/rocisa/include/macro.hpp
                module.add(VPrngB32(dst=vgpr(vRand), src=vgpr(formatVgpr), comment="Pseudo Random Number Generator"))
            else:
                if ti.getAsmCaps()["HasVgprMSB"]:
                    module.add(PseudoRandomGeneratorModule(vRand, vgprFp8Temp, vTemp0, vTemp1))
                else:
                    module.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprFp8Temp, vTemp0, vTemp1]))

        if gwvw % 8 == 0 and ti.getAsmCaps()["HasScaleSRPk8Cvt"]:
            # Packed path: convert 8 F32 elements to FP8 at once
            for groupIdx in range(gwvw // 8):
                srcStartVgpr = formatting(elementSumIdx + groupIdx * 8, inputPrefix, prefixOffset)
                addTruncateAndPRNG(srcStartVgpr, skipTruncate=True)
                d = destIdx + groupIdx * 2  # Each group outputs 2 VGPRs (64-bit aligned)
                module.add(VCvtScaleSRPkF32toFP8(dst=vgpr(d, 2), src0=vgpr(srcStartVgpr, 8), src1=vgpr(vRand), scale=alphaScale))
        else:
            # Scalar path: convert one element at a time
            for vi in range(gwvw):
                formatVgpr = formatting(elementSumIdx + vi, inputPrefix, prefixOffset)
                addTruncateAndPRNG(formatVgpr)
                d = destIdx + vi // 4
                module.add(VCvtSRF32toFP8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(vRand), sels=[vi % 4]))

        return module

class PackData_FLOAT8_fnuz_SR(PackData_FLOAT8_SR):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Float8_fnuz), "StochasticRounding": True}}

# Cvt is outside of this component, this is just a wrapper for ComputeDataType == float
class PackData_INT8_F32(PackData):
    kernel = {"ProblemType": {"ComputeDataType": DataType(DataTypeEnum.Float), "DestDataType": DataType(DataTypeEnum.Int8)}}
    packdata = PackData_INT8()
    def __call__(self, gwvw, destIdx, elementSumIdx, i8CVTVgprStruct, tmpS01, SaturateTypeInt8 = SaturateCastType.NORMAL, inputPrefix="", prefixOffset=0):
        return self.packdata(gwvw, destIdx, elementSumIdx, i8CVTVgprStruct, tmpS01, SaturateTypeInt8, inputPrefix, prefixOffset)
