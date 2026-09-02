# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Composite KernelWriter helpers (ArgumentLoader, math, branch).

Real: ArgumentLoader, vector/scalar divide, magic division, branch helpers.
Not yet done: VSaturateCastInt, DSInit.
"""

from __future__ import annotations

import math
from typing import Any

from ._dummy import make_dummy_func
from .code import Module, TextBlock
from .container import ContinuousRegister, EXEC, VCC, sgpr, vgpr
from .enum import DataTypeEnum
from .instruction import (
    SAddCU32, SAddU32, SAndB32, SAndB64, SCBranchSCC0, SCBranchSCC1,
    SCBranchVCCNZ, SCBranchVCCZ, SCmpEQU32, SCmpEQU64, SCmpLgU32,
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SLShiftRightB64,
    SLoadB32, SLoadB64, SLoadB128, SLoadB256, SLoadB512,
    SMulHIU32, SMulI32, SMovB32, SMovB64, SNop, SSubU32,
    VAddCCOU32, VAddLShiftLeftU32, VAddU32, VAndB32, VCmpEQF32,
    VCmpEQF64, VCmpNeU32, VCmpXEqU32, VCmpXGeU32, VCmpXGtU32,
    VCvtF32toU32, VCvtF64toU32, VCvtU32toF32, VCvtU32toF64,
    VLShiftLeftAddU32, VLShiftLeftB32, VLShiftLeftB64,
    VLShiftRightB32, VLShiftRightB64, VMadU32U24,
    VMovB32, VMulF32, VMulF64, VMulHIU32, VMulLOU32, VMulU32U24,
    VRcpF64, VRcpIFlagF32, VReadfirstlaneB32, VSubU32,
)

_P = "rocisa.functions"

_SLOAD_BY_BITS = {
    32: SLoadB32,
    64: SLoadB64,
    128: SLoadB128,
    256: SLoadB256,
    512: SLoadB512,
}


class ArgumentLoader:
    """Port of ``rocisa::ArgumentLoader`` (``functions/argument.hpp``).

    Manages kernel-argument offset bookkeeping and emits ``SLoadB*``
    instructions matching the C++ logic exactly.
    """

    __slots__ = ("_kernArgOffset",)

    def __init__(self) -> None:
        self._kernArgOffset: int = 0

    def resetOffset(self) -> None:
        self._kernArgOffset = 0

    def setOffset(self, offset: int) -> None:
        self._kernArgOffset = int(offset)

    def getOffset(self) -> int:
        return self._kernArgOffset

    def loadKernArg(self, dst, srcAddr, sgprOffset=None, dword: int = 1,
                    writeSgpr: bool = True) -> Any:
        """Emit an ``SLoadB*`` and advance ``kernArgOffset``.

        Mirrors ``functions/argument.hpp:60-121``.
        """
        size = int(dword) * 4

        if writeSgpr:
            if sgprOffset is not None:
                comment = ""
            else:
                comment = str(self._kernArgOffset)

            dst_sgpr = sgpr(dst, dword)
            src_sgpr = sgpr(srcAddr, 2)
            offset = sgprOffset if sgprOffset is not None else self._kernArgOffset

            bits = int(dword) * 32
            cls = _SLOAD_BY_BITS.get(bits)
            if cls is None:
                raise ValueError(f"Invalid dword size {dword}")
            item = cls(dst=dst_sgpr, base=src_sgpr, soffset=offset, comment=comment)
        else:
            item = TextBlock("Move offset by " + str(size) + "\n")

        if sgprOffset is None:
            self._kernArgOffset += size
        return item

    def loadAllKernArg(self, sgprStartIndex: int, srcAddr, numSgprToLoad: int,
                       numSgprPreload: int = 0) -> Module:
        """Emit a ``Module`` of greedy-packed ``SLoadB*`` instructions.

        Mirrors ``functions/argument.hpp:126-199``.
        """
        module = Module("LoadAllKernArg")
        actual_load = int(numSgprToLoad) - int(numSgprPreload)
        sgpr_idx = int(sgprStartIndex) + int(numSgprPreload)
        self._kernArgOffset += int(numSgprPreload) * 4

        while actual_load > 0:
            i = 16
            while i >= 1:
                is_aligned = False
                if i >= 4 and sgpr_idx % 4 == 0:
                    is_aligned = True
                elif i == 2 and sgpr_idx % 2 == 0:
                    is_aligned = True
                elif i == 1:
                    is_aligned = True

                if is_aligned and actual_load >= i:
                    actual_load -= i
                    dst_sgpr = sgpr(sgpr_idx, i)
                    src_sgpr = sgpr(srcAddr, 2)
                    comment = str(self._kernArgOffset)

                    bits = i * 32
                    cls = _SLOAD_BY_BITS.get(bits)
                    if cls is None:
                        raise ValueError(f"Invalid SGPR size {i}")
                    module.add(cls(dst=dst_sgpr, base=src_sgpr,
                                   soffset=self._kernArgOffset, comment=comment))
                    sgpr_idx += i
                    self._kernArgOffset += i * 4
                    break
                i //= 2

        return module


def _sgpr_with_offset(sgprName: str, offset: int) -> str:
    try:
        base = int(sgprName)
        return str(base + offset)
    except ValueError:
        return sgprName + "+" + str(offset)


def BranchIfZero(sgprName: str, computeDataType, tmpSgprIdx: int,
                 laneSC: int, label, waveFrontSize: int) -> Module:
    module = Module("BranchIfZero")
    sgprStr = "s[" + sgprName + "]"
    pVCC = VCC()

    if computeDataType == DataTypeEnum.ComplexDouble:
        tmpSgpr = sgpr(tmpSgprIdx, laneSC)
        module.add(VCmpEQF64(dst=tmpSgpr, src0=sgpr(sgprName, 2), src1=0.0,
                              comment=sgprStr + ".real == 0.0 ?"))
        sgprVar = _sgpr_with_offset(sgprName, 2)
        module.add(VCmpEQF64(dst=pVCC, src0=sgpr(sgprVar, 2), src1=0.0,
                              comment=sgprStr + ".imag == 0.0 ?"))
        if waveFrontSize == 32:
            module.add(SAndB32(dst=tmpSgpr, src0=pVCC, src1=tmpSgpr,
                               comment=sgprStr + " == 0 ?"))
            module.add(SCmpEQU32(src0=tmpSgpr, src1=0,
                                 comment="branch if " + sgprStr + " == 0"))
        else:
            module.add(SAndB64(dst=tmpSgpr, src0=pVCC, src1=tmpSgpr,
                               comment=sgprStr + " == 0 ?"))
            module.add(SCmpEQU64(src0=tmpSgpr, src1=0,
                                 comment="branch if " + sgprStr + " == 0"))
        module.add(SCBranchSCC0(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " == 0"))

    elif computeDataType == DataTypeEnum.Double:
        module.add(VCmpEQF64(dst=pVCC, src0=sgpr(sgprName, 2), src1=0.0,
                              comment=sgprStr + " == 0.0 ?"))
        module.add(SCBranchVCCNZ(labelName=label.getLabelName(),
                                 comment="branch if " + sgprStr + " == 0"))

    elif computeDataType == DataTypeEnum.ComplexFloat:
        tmpSgpr = sgpr(tmpSgprIdx, laneSC)
        module.add(VCmpEQF32(dst=tmpSgpr, src0=sgpr(sgprName), src1=0.0,
                              comment=sgprStr + ".real == 0.0f ?"))
        sgprVar = _sgpr_with_offset(sgprName, 1)
        module.add(VCmpEQF32(dst=pVCC, src0=sgpr(sgprVar), src1=0.0,
                              comment=sgprStr + ".imag == 0.0f ?"))
        if waveFrontSize == 32:
            module.add(SAndB32(dst=tmpSgpr, src0=pVCC, src1=tmpSgpr,
                               comment=sgprStr + " == 0 ?"))
            module.add(SCmpEQU32(src0=tmpSgpr, src1=0,
                                 comment="branch if " + sgprStr + " == 0"))
        else:
            module.add(SAndB64(dst=tmpSgpr, src0=pVCC, src1=tmpSgpr,
                               comment=sgprStr + " == 0 ?"))
            module.add(SCmpEQU64(src0=tmpSgpr, src1=0,
                                 comment="branch if " + sgprStr + " == 0"))
        module.add(SCBranchSCC0(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " == 0"))

    elif computeDataType in (DataTypeEnum.Float, DataTypeEnum.Half,
                             DataTypeEnum.BFloat16):
        module.add(VCmpEQF32(dst=pVCC, src0=sgpr(sgprName), src1=0.0,
                              comment=sgprStr + " == 0.0f ?"))
        module.add(SCBranchVCCNZ(labelName=label.getLabelName(),
                                 comment="branch if " + sgprStr + " == 0"))

    elif computeDataType == DataTypeEnum.Int32:
        module.add(SCmpEQU32(src0=sgpr(sgprName), src1=0,
                             comment=sgprStr + " == 0 ?"))
        module.add(SCBranchSCC1(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " == 0"))

    elif computeDataType == DataTypeEnum.Int64:
        module.add(SCmpEQU64(src0=sgpr(sgprName, 2), src1=0,
                             comment=sgprStr + " == 0 ?"))
        module.add(SCBranchSCC1(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " == 0"))
    else:
        raise RuntimeError("Unsupported compute data type")

    return module


def BranchIfNotZero(sgprName: str, computeDataType, label) -> Module:
    module = Module("BranchIfNotZero")
    sgprStr = "s[" + sgprName + "]"
    pVCC = VCC()

    if computeDataType == DataTypeEnum.ComplexDouble:
        module.add(VCmpEQF64(dst=pVCC, src0=sgpr(sgprName, 2), src1=0.0,
                              comment=sgprStr + ".real == 0.0 ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + ".real != 0"))
        sgprVar = _sgpr_with_offset(sgprName, 2)
        module.add(VCmpEQF64(dst=pVCC, src0=sgpr(sgprVar, 2), src1=0.0,
                              comment=sgprStr + ".imag == 0.0 ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + ".imag != 0"))

    elif computeDataType == DataTypeEnum.Double:
        module.add(VCmpEQF64(dst=pVCC, src0=sgpr(sgprName, 2), src1=0.0,
                              comment=sgprStr + " == 0.0 ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " != 0"))

    elif computeDataType == DataTypeEnum.ComplexFloat:
        module.add(VCmpEQF32(dst=pVCC, src0=sgpr(sgprName), src1=0.0,
                              comment=sgprStr + ".real == 0.0f ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + ".real != 0"))
        sgprVar = _sgpr_with_offset(sgprName, 1)
        module.add(VCmpEQF32(dst=pVCC, src0=sgpr(sgprVar), src1=0.0,
                              comment=sgprStr + ".imag == 0.0f ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + ".imag != 0"))

    elif computeDataType in (DataTypeEnum.Float, DataTypeEnum.Half,
                             DataTypeEnum.BFloat16):
        module.add(VCmpEQF32(dst=pVCC, src0=sgpr(sgprName), src1=0.0,
                              comment=sgprStr + " == 0.0f ?"))
        module.add(SCBranchVCCZ(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " != 0"))

    elif computeDataType == DataTypeEnum.Int64:
        module.add(SCmpEQU64(src0=sgpr(sgprName, 2), src1=0,
                             comment=sgprStr + " == 0 ?"))
        module.add(SCBranchSCC0(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " != 0"))
    else:
        module.add(SCmpEQU32(src0=sgpr(sgprName), src1=0,
                             comment=sgprStr + " == 0 ?"))
        module.add(SCBranchSCC0(labelName=label.getLabelName(),
                                comment="branch if " + sgprStr + " != 0"))

    return module

def VSaturateCastInt(SumIdxVgpr: Any, tmpVgprIdx: int, tmpSgprIdx: int,
                     lowerBound: int, upperBound: int,
                     type: Any = None, initGpr: bool = True) -> Module:
    """Saturate-cast integer (f_cast.cpp)."""
    from .enum import SaturateCastType  # noqa: WPS433
    from .instruction import SMovkI32, VMed3I32, VMinI32, VMaxI32  # noqa: WPS433

    if type is None:
        type = SaturateCastType.NORMAL

    initGprStr = "with init gpr" if initGpr else "without init gpr"
    module = Module("SaturateCastInt " + initGprStr)

    if type == SaturateCastType.NORMAL:
        tmpLowerBoundSgpr = sgpr(tmpSgprIdx)
        tmpUpperBoundVgpr = vgpr(tmpVgprIdx)

        if initGpr:
            module.add(SMovkI32(dst=tmpLowerBoundSgpr, src=lowerBound,
                                comment=str(lowerBound)))
            module.add(VMovB32(dst=tmpUpperBoundVgpr, src=upperBound,
                               comment=str(upperBound)))

        module.add(VMed3I32(
            dst=SumIdxVgpr, src0=SumIdxVgpr, src1=tmpLowerBoundSgpr,
            src2=tmpUpperBoundVgpr,
            comment=f"x= min({upperBound}, max({lowerBound}, x))",
        ))
    elif type == SaturateCastType.DO_NOTHING:
        pass
    elif type == SaturateCastType.UPPER:
        module.add(VMinI32(
            dst=SumIdxVgpr, src0=upperBound, src1=SumIdxVgpr,
            comment=f"x = min({upperBound}, x)",
        ))
    elif type == SaturateCastType.LOWER:
        module.add(VMaxI32(
            dst=SumIdxVgpr, src0=lowerBound, src1=SumIdxVgpr,
            comment=f"x = max({lowerBound}, x)",
        ))

    return module

# ---------------------------------------------------------------------------
# Math helper utilities
# ---------------------------------------------------------------------------


def _to_sgpr(arg):
    if isinstance(arg, (int, str)):
        return sgpr(arg)
    return arg


def _to_vgpr(arg):
    if isinstance(arg, (int, str)):
        return vgpr(arg)
    return arg


def _get_vgpr(arg, idx):
    if isinstance(arg, int):
        return vgpr(arg + idx)
    elif isinstance(arg, str):
        return vgpr(arg + "+" + str(idx))
    return vgpr(-1)


def _get_sgpr(arg, idx):
    if isinstance(arg, int):
        return sgpr(arg + idx)
    elif isinstance(arg, str):
        return sgpr(arg + "+" + str(idx))
    return sgpr(-1)


def _is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def _to_str(val):
    if isinstance(val, str):
        return val
    return str(val)


# ---------------------------------------------------------------------------
# SMulInt64to32 — internal helper (mirrors extension.hpp)
# ---------------------------------------------------------------------------


def _SMulInt64to32(dst0, dst1, src0, src1, tmpVgprRes, hasSMulHi,
                   sign, comment=""):
    module = Module("SMulInt64to32")
    if tmpVgprRes.size < 2:
        raise RuntimeError("ContinuousRegister size must be at least 2.")
    if hasSMulHi:
        module.add(SMulHIU32(dst=dst1, src0=src0, src1=src1, comment=comment))
        module.add(SMulI32(dst=dst0, src0=src0, src1=src1, comment=comment))
    else:
        vgprTmp = vgpr(tmpVgprRes.idx)
        vgprTmp1 = vgpr(tmpVgprRes.idx + 1)
        module.add(VMovB32(dst=vgprTmp, src=src0, comment=comment))
        module.add(VMulHIU32(dst=vgprTmp1, src0=vgprTmp, src1=src1, comment=comment))
        module.add(VReadfirstlaneB32(dst=dst1, src=vgprTmp1, comment=comment))
        module.add(VMulLOU32(dst=vgprTmp1, src0=vgprTmp, src1=src1, comment=comment))
        module.add(VReadfirstlaneB32(dst=dst0, src=vgprTmp1, comment=comment))
    return module


# ---------------------------------------------------------------------------
# Vector divide & remainder
# ---------------------------------------------------------------------------


def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor,
                                   tmpVgprRes=None, doRemainder=True,
                                   comment=""):
    qRegStr = _to_str(qReg)
    dRegStr = _to_str(dReg)
    dComment = comment if comment else f"{qRegStr} = {dRegStr} / {divisor}"
    rComment = comment if comment else f"{rReg} = {dRegStr} % {divisor}"

    module = Module("vectorStaticDivideAndRemainder")
    qRegVgpr = vgpr(qReg)
    rRegVgpr = vgpr(rReg)
    dRegVgpr = vgpr(dReg)

    if _is_power_of_2(divisor):
        divisor_log2 = int(math.log2(divisor))
        module.add(VLShiftRightB32(dst=qRegVgpr, shiftHex=divisor_log2,
                                   src=dRegVgpr, comment=dComment))
        if doRemainder:
            module.add(VAndB32(dst=rRegVgpr, src0=divisor - 1,
                               src1=dRegVgpr, comment=rComment))
    else:
        if not tmpVgprRes or tmpVgprRes.size < 2:
            raise RuntimeError("Invalid tmpVgprRes, must be at least 2")
        tmpVgprIdx = tmpVgprRes.idx
        tmpVgpr = vgpr(tmpVgprIdx)
        tmpVgpr1 = vgpr(tmpVgprIdx + 1)

        shift = 33
        magic = ((1 << shift) // divisor) + 1

        if -16 <= magic <= 64:
            module.add(VMulHIU32(dst=tmpVgpr1, src0=dRegVgpr,
                                 src1=magic, comment=dComment))
            module.add(VMulLOU32(dst=tmpVgpr, src0=dRegVgpr,
                                 src1=magic, comment=dComment))
        else:
            module.add(VMovB32(dst=tmpVgpr, src=magic))
            module.add(VMulHIU32(dst=tmpVgpr1, src0=dRegVgpr,
                                 src1=tmpVgpr, comment=dComment))
            module.add(VMulLOU32(dst=tmpVgpr, src0=dRegVgpr,
                                 src1=tmpVgpr, comment=dComment))

        module.add(VLShiftRightB64(dst=vgpr(tmpVgprIdx, 2), shiftHex=shift,
                                   src=vgpr(tmpVgprIdx, 2), comment=dComment))
        module.add(VMovB32(dst=qRegVgpr, src=tmpVgpr, comment=dComment))

        if doRemainder:
            if -16 <= divisor <= 64:
                module.add(VMulLOU32(dst=tmpVgpr, src0=qRegVgpr,
                                     src1=divisor, comment=rComment))
            else:
                module.add(VMovB32(dst=tmpVgpr1, src=divisor))
                module.add(VMulLOU32(dst=tmpVgpr, src0=qRegVgpr,
                                     src1=tmpVgpr1, comment=rComment))
            module.add(VSubU32(dst=rRegVgpr, src0=dRegVgpr,
                               src1=tmpVgpr, comment=rComment))

    return module


def vectorStaticDivide(qReg, dReg, divisor, tmpVgprRes=None, comment=""):
    module = vectorStaticDivideAndRemainder(
        qReg, -1, dReg, divisor, tmpVgprRes, False, comment)
    module.name = "vectorStaticDivide (reg=-1)"
    return module


def vectorUInt32DivideAndRemainder(qReg, dReg, divReg, rReg,
                                   doRemainder=True, comment=""):
    qRegVgpr = vgpr(qReg)
    rRegVgpr = vgpr(rReg)
    dRegVgpr = vgpr(dReg)
    divRegVgpr = vgpr(divReg)

    dComment = (comment if comment else
                f"{qRegVgpr.toString()} = {dRegVgpr.toString()}"
                f" / {divRegVgpr.toString()}")
    rComment = (comment if comment else
                f"{rRegVgpr.toString()} = {dRegVgpr.toString()}"
                f" % {divRegVgpr.toString()}")

    pEXEC = EXEC()
    module = Module("vectorUInt32DivideAndRemainder")
    module.add(VCvtU32toF32(dst=qRegVgpr, src=divRegVgpr, comment=dComment))
    module.add(VRcpIFlagF32(dst=qRegVgpr, src=qRegVgpr, comment=dComment))
    module.add(VCvtU32toF32(dst=rRegVgpr, src=dRegVgpr, comment=dComment))
    module.add(VMulF32(dst=qRegVgpr, src0=qRegVgpr, src1=rRegVgpr,
                       comment=dComment))
    module.add(VCvtF32toU32(dst=qRegVgpr, src=qRegVgpr, comment=dComment))
    module.add(VMulU32U24(dst=rRegVgpr, src0=qRegVgpr, src1=divRegVgpr,
                          comment=dComment))
    module.add(VSubU32(dst=rRegVgpr, src0=dRegVgpr, src1=rRegVgpr,
                       comment=dComment))
    module.add(VCmpXEqU32(dst=pEXEC, src0=rRegVgpr, src1=divRegVgpr,
                          comment=dComment))
    module.add(VAddU32(dst=qRegVgpr, src0=1, src1=qRegVgpr, comment=dComment))
    if doRemainder:
        module.add(VMovB32(dst=rRegVgpr, src=0, comment=rComment))
    module.add(SMovB64(dst=pEXEC, src=-1, comment=dComment))
    return module


def vectorUInt32CeilDivideAndRemainder(qReg, dReg, divReg, rReg,
                                       doRemainder=True, comment=""):
    qRegVgpr = vgpr(qReg)
    rRegVgpr = vgpr(rReg)
    dRegVgpr = vgpr(dReg)
    divRegVgpr = vgpr(divReg)

    dComment = (comment if comment else
                f"{qRegVgpr.toString()} = ceil({dRegVgpr.toString()}"
                f" / {divRegVgpr.toString()})")
    rComment = (comment if comment else
                f"{rRegVgpr.toString()} = {dRegVgpr.toString()}"
                f" % {divRegVgpr.toString()}")

    pVCC = VCC()
    pEXEC = EXEC()
    module = Module("vectorUInt32CeilDivideAndRemainder")
    module.add(VCvtU32toF32(dst=qRegVgpr, src=divRegVgpr, comment=dComment))
    module.add(VRcpIFlagF32(dst=qRegVgpr, src=qRegVgpr, comment=dComment))
    module.add(VCvtU32toF32(dst=rRegVgpr, src=dRegVgpr, comment=dComment))
    module.add(VMulF32(dst=qRegVgpr, src0=qRegVgpr, src1=rRegVgpr,
                       comment=dComment))
    module.add(VCvtF32toU32(dst=qRegVgpr, src=qRegVgpr, comment=dComment))
    module.add(VMulU32U24(dst=rRegVgpr, src0=qRegVgpr, src1=divRegVgpr,
                          comment=dComment))
    module.add(VSubU32(dst=rRegVgpr, src0=dRegVgpr, src1=rRegVgpr,
                       comment=dComment))
    module.add(VCmpNeU32(dst=pVCC, src0=rRegVgpr, src1=0, comment=dComment))
    module.add(VAddCCOU32(dst=qRegVgpr, src0=qRegVgpr, src1=0,
                          comment="ceil"))
    if doRemainder:
        module.add(VCmpXEqU32(dst=pEXEC, src0=rRegVgpr, src1=divRegVgpr,
                              comment=rComment))
        module.add(VMovB32(dst=rRegVgpr, src=0, comment=rComment))
        module.add(SMovB64(dst=pEXEC, src=-1, comment=dComment))
    return module


def vectorStaticRemainder(qReg, rReg, dReg, divisor, tmpVgprRes=None,
                          tmpSgprRes=None, comment=""):
    qRegVgpr = vgpr(qReg)
    rRegVgpr = vgpr(rReg)
    dRegVgpr = vgpr(dReg)

    dComment = (comment if comment else
                f"{rRegVgpr.toString()} = {dRegVgpr.toString()} % {divisor}")

    module = Module("vectorStaticRemainder")

    if _is_power_of_2(divisor):
        module.add(VAndB32(dst=rRegVgpr, src0=divisor - 1, src1=dRegVgpr,
                           comment=dComment))
    else:
        if not tmpVgprRes or tmpVgprRes.size < 2:
            raise RuntimeError("Invalid tmpVgprRes, must be at least 2")
        tmpVgprIdx = tmpVgprRes.idx
        tmpVgpr = vgpr(tmpVgprIdx)
        tmpVgpr1 = vgpr(tmpVgprIdx + 1)

        if not tmpSgprRes or tmpSgprRes.size < 1:
            raise RuntimeError("Invalid tmpSgprRes, must be at least 1")
        tmpSgprIdx = tmpSgprRes.idx
        tmoSgpr = sgpr(tmpSgprIdx)

        shift = 33
        magic = ((1 << shift) // divisor) + 1

        if -16 <= magic <= 64:
            module.add(VMulHIU32(dst=tmpVgpr1, src0=dRegVgpr, src1=magic,
                                 comment=dComment))
            module.add(VMulLOU32(dst=tmpVgpr, src0=dRegVgpr, src1=magic,
                                 comment=dComment))
        else:
            module.add(SMovB32(dst=tmoSgpr, src=magic, comment=dComment))
            module.add(VMulHIU32(dst=tmpVgpr1, src0=dRegVgpr, src1=tmoSgpr,
                                 comment=dComment))
            module.add(VMulLOU32(dst=tmpVgpr, src0=dRegVgpr, src1=tmoSgpr,
                                 comment=dComment))

        module.add(VLShiftRightB64(dst=vgpr(tmpVgprIdx, 2), shiftHex=shift,
                                   src=vgpr(tmpVgprIdx, 2), comment=dComment))
        module.add(VMovB32(dst=qRegVgpr, src=tmpVgpr, comment=dComment))

        if -16 <= divisor <= 64:
            module.add(VMulLOU32(dst=tmpVgpr, src0=qRegVgpr, src1=divisor,
                                 comment=dComment))
        else:
            module.add(SMovB32(dst=tmoSgpr, src=divisor, comment=dComment))
            module.add(VMulLOU32(dst=tmpVgpr, src0=qRegVgpr, src1=tmoSgpr,
                                 comment=dComment))

        module.add(VSubU32(dst=rRegVgpr, src0=dRegVgpr, src1=tmpVgpr,
                           comment=dComment))

    return module


# ---------------------------------------------------------------------------
# Scalar divide & remainder
# ---------------------------------------------------------------------------


def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor,
                                   tmpSgprRes=None, doRemainder=1):
    qRegSgpr = _to_sgpr(qReg)
    rRegSgpr = _to_sgpr(rReg)
    dRegSgpr = _to_sgpr(dReg)

    module = Module("scalarStaticDivideAndRemainder")

    if _is_power_of_2(divisor):
        divisor_log2 = int(math.log2(divisor))
        if doRemainder != 2:
            module.add(SLShiftRightB32(
                dst=qRegSgpr, shiftHex=divisor_log2, src=dRegSgpr,
                comment=f"{qRegSgpr.toString()} = {dRegSgpr.toString()}"
                        f" / {divisor}"))
        if doRemainder:
            module.add(SAndB32(
                dst=rRegSgpr, src0=divisor - 1, src1=dRegSgpr,
                comment=f"{rRegSgpr.toString()} = {dRegSgpr.toString()}"
                        f" % {divisor}"))
    else:
        if not tmpSgprRes or tmpSgprRes.size < 2:
            raise RuntimeError("Invalid tmpSgprRes, must be at least 2")
        tmpSgprIdx = tmpSgprRes.idx
        tmpSgpr = sgpr(tmpSgprIdx)
        tmpSgpr1 = sgpr(tmpSgprIdx + 1)
        tmp2Sgpr = sgpr(tmpSgprIdx, 2)

        shift = 33
        magic = ((1 << shift) // divisor) + 1
        magicHi = magic >> 16
        magicLo = magic & 0xFFFF

        module.add(SMovB32(dst=tmpSgpr1, src=0,
                           comment=f"STATIC_DIV: divisor={divisor}"))
        module.add(SMulI32(dst=tmpSgpr, src0=magicHi, src1=dRegSgpr,
                           comment="tmp1 = dividend * magic hi"))
        module.add(SLShiftLeftB64(dst=tmp2Sgpr, shiftHex=16, src=tmp2Sgpr,
                                  comment="left shift 16 bits"))
        module.add(SMulI32(dst=qRegSgpr, src0=dRegSgpr, src1=magicLo,
                           comment="tmp0 = dividend * magic lo"))
        module.add(SAddU32(dst=tmpSgpr, src0=qRegSgpr, src1=tmpSgpr,
                           comment="add lo"))
        module.add(SAddCU32(dst=tmpSgpr1, src0=tmpSgpr1, src1=0,
                            comment="add hi"))
        module.add(SLShiftRightB64(
            dst=tmp2Sgpr, shiftHex=shift, src=tmp2Sgpr,
            comment="tmp1 = (dividend * magic) << shift"))
        module.add(SMovB32(dst=qRegSgpr, src=tmpSgpr, comment="quotient"))

        if doRemainder:
            module.add(SMulI32(dst=tmpSgpr, src0=qRegSgpr, src1=divisor,
                               comment="quotient*divisor"))
            module.add(SSubU32(
                dst=rRegSgpr, src0=dRegSgpr, src1=tmpSgpr,
                comment="rReg = dividend - quotient*divisor"))

    return module


def scalarStaticCeilDivide(qReg, dReg, divisor, tmpSgprRes=None):
    qRegSgpr = _to_sgpr(qReg)
    dRegSgpr = _to_sgpr(dReg)

    module = Module("scalarStaticCeilDivide")

    if _is_power_of_2(divisor):
        divisor_log2 = int(math.log2(divisor))
        module.add(SLShiftRightB32(
            dst=qRegSgpr, shiftHex=divisor_log2, src=dRegSgpr,
            comment=f"{qRegSgpr.toString()} = {dRegSgpr.toString()}"
                    f" / {divisor}"))
        tmpSgpr = sgpr(tmpSgprRes.idx)
        module.add(SAndB32(
            dst=tmpSgpr, src0=divisor - 1, src1=dRegSgpr,
            comment=f"{tmpSgpr.toString()} = {dRegSgpr.toString()}"
                    f" % {divisor}"))
        module.add(SAddCU32(dst=qRegSgpr, src0=qRegSgpr, src1=0))
    else:
        if not tmpSgprRes or tmpSgprRes.size < 2:
            raise RuntimeError("Invalid tmpSgprRes, must be at least 2")
        tmpSgprIdx = tmpSgprRes.idx
        tmpSgpr = sgpr(tmpSgprIdx)
        tmpSgpr1 = sgpr(tmpSgprIdx + 1)
        tmp2Sgpr = sgpr(tmpSgprIdx, 2)

        shift = 33
        magic = ((1 << shift) // divisor) + 1
        magicHi = magic >> 16
        magicLo = magic & 0xFFFF

        module.add(SMovB32(dst=tmpSgpr1, src=0,
                           comment=f"STATIC_DIV: divisor={divisor}"))
        module.add(SMulI32(dst=tmpSgpr, src0=magicHi, src1=dRegSgpr,
                           comment="tmp1 = dividend * magic hi"))
        module.add(SLShiftLeftB64(dst=tmp2Sgpr, shiftHex=16, src=tmp2Sgpr,
                                  comment="left shift 16 bits"))
        module.add(SMulI32(dst=qRegSgpr, src0=dRegSgpr, src1=magicLo,
                           comment="tmp0 = dividend * magic lo"))
        module.add(SAddU32(dst=tmpSgpr, src0=qRegSgpr, src1=tmpSgpr,
                           comment="add lo"))
        module.add(SAddCU32(dst=tmpSgpr1, src0=tmpSgpr1, src1=0,
                            comment="add hi"))
        module.add(SLShiftRightB64(dst=tmp2Sgpr, shiftHex=shift,
                                   src=tmp2Sgpr, comment="tmp0 = quotient"))
        module.add(SMulI32(dst=tmpSgpr1, src0=tmpSgpr, src1=divisor,
                           comment="tmp1 = quotient * divisor"))
        module.add(SCmpLgU32(
            src0=tmpSgpr1, src1=dRegSgpr,
            comment="if (quotient * divisor != dividend), result+=1"))
        module.add(SAddCU32(
            dst=qRegSgpr, src0=tmpSgpr, src1=0,
            comment="if (quotient * divisor != dividend), result+=1"))

    return module


def scalarStaticRemainder(qReg, rReg, dReg, divisor, tmpSgprRes=None,
                          comment=""):
    module = Module("scalarStaticRemainder")

    rRegSgpr = sgpr(rReg)
    dRegSgpr = sgpr(dReg)

    dComment = (comment if comment else
                f"{rRegSgpr.toString()} = {dRegSgpr.toString()} % {divisor}")

    if _is_power_of_2(divisor):
        module.add(SAndB32(dst=rRegSgpr, src0=divisor - 1, src1=dRegSgpr,
                           comment=dComment))
    else:
        qRegSgpr = sgpr(qReg)

        if not tmpSgprRes or tmpSgprRes.size < 3:
            raise RuntimeError("Invalid tmpSgprRes, must be at least 3")
        tmpSgprIdx = tmpSgprRes.idx
        tmpSgpr = sgpr(tmpSgprIdx)
        tmpSgpr1 = sgpr(tmpSgprIdx + 1)
        tmpSgpr2 = sgpr(tmpSgprIdx + 2)
        tmp2Sgpr = sgpr(tmpSgprIdx, 2)

        shift = 33
        magic = ((1 << shift) // divisor) + 1

        if -16 <= magic <= 64:
            module.add(SMulHIU32(dst=tmpSgpr1, src0=dRegSgpr, src1=magic,
                                 comment=dComment))
            module.add(SMulI32(dst=tmpSgpr, src0=dRegSgpr, src1=magic,
                               comment=dComment))
        else:
            module.add(SMovB32(dst=tmpSgpr2, src=magic, comment=dComment))
            module.add(SMulHIU32(dst=tmpSgpr1, src0=dRegSgpr, src1=tmpSgpr2,
                                 comment=dComment))
            module.add(SMulI32(dst=tmpSgpr, src0=dRegSgpr, src1=tmpSgpr2,
                               comment=dComment))

        module.add(SLShiftRightB64(dst=tmp2Sgpr, shiftHex=shift,
                                   src=tmp2Sgpr, comment=dComment))
        module.add(SMovB32(dst=qRegSgpr, src=tmpSgpr, comment=dComment))

        if -16 <= divisor <= 64:
            module.add(SMulI32(dst=tmpSgpr, src0=qRegSgpr, src1=divisor,
                               comment=dComment))
        else:
            module.add(SMovB32(dst=tmpSgpr2, src=divisor, comment=dComment))
            module.add(SMulI32(dst=tmpSgpr, src0=qRegSgpr, src1=tmpSgpr2,
                               comment=dComment))

        module.add(SSubU32(dst=rRegSgpr, src0=dRegSgpr, src1=tmpSgpr,
                           comment=dComment))

    return module


def scalarUInt24DivideAndRemainder(qReg, dReg, divReg, rReg, tmpVgprRes,
                                   wavewidth, doRemainder=True,
                                   doQuotient=True, comment=""):
    module = Module("scalarUInt24DivideAndRemainder")

    qRegSgpr = sgpr(qReg)
    rRegSgpr = sgpr(rReg)
    dRegSgpr = sgpr(dReg)
    divRegSgpr = sgpr(divReg)

    dComment = (comment if comment else
                f"{qRegSgpr.toString()} = {dRegSgpr.toString()}"
                f" / {divRegSgpr.toString()}")
    rComment = ""
    if doRemainder:
        rComment = (comment if comment else
                    f"{sgpr(rReg).toString()} = {dRegSgpr.toString()}"
                    f" % {divRegSgpr.toString()}")

    if tmpVgprRes.size < 4:
        raise RuntimeError("Invalid tmpVgprRes, must be at least 4")
    tmpVgpr = tmpVgprRes.idx
    tmpVgpr1 = tmpVgprRes.idx + 2

    pEXEC = EXEC()

    module.add(VCvtU32toF64(dst=vgpr(tmpVgpr, 2), src=divRegSgpr,
                            comment=dComment))
    module.add(VRcpF64(dst=vgpr(tmpVgpr, 2), src=vgpr(tmpVgpr, 2),
                       comment=dComment))
    module.add(VCvtU32toF64(dst=vgpr(tmpVgpr1, 2), src=dRegSgpr,
                            comment=dComment))
    module.add(VMulF64(dst=vgpr(tmpVgpr, 2), src0=vgpr(tmpVgpr, 2),
                       src1=vgpr(tmpVgpr1, 2), comment=dComment))
    module.add(VCvtF64toU32(dst=vgpr(tmpVgpr), src=vgpr(tmpVgpr, 2),
                            comment=dComment))

    module.add(VMulLOU32(dst=vgpr(tmpVgpr + 1), src0=vgpr(tmpVgpr),
                         src1=divRegSgpr, comment=dComment))
    module.add(VSubU32(dst=vgpr(tmpVgpr1), src0=dRegSgpr,
                       src1=vgpr(tmpVgpr + 1), comment=dComment))
    module.add(VCmpXGeU32(dst=pEXEC, src0=vgpr(tmpVgpr1), src1=divRegSgpr,
                          comment=dComment))
    module.add(VAddU32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr), src1=1,
                       comment=dComment))

    if wavewidth == 64:
        module.add(SMovB64(dst=pEXEC, src=-1, comment="Reset exec"))
    else:
        module.add(SMovB32(dst=pEXEC, src=-1, comment="Reset exec"))

    if doRemainder:
        module.add(VMulLOU32(dst=vgpr(tmpVgpr + 1), src0=vgpr(tmpVgpr),
                             src1=divRegSgpr, comment=dComment))
        module.add(VSubU32(dst=vgpr(tmpVgpr1), src0=dRegSgpr,
                           src1=vgpr(tmpVgpr + 1), comment=dComment))

    if doQuotient:
        module.add(VReadfirstlaneB32(dst=qRegSgpr, src=vgpr(tmpVgpr),
                                     comment="quotient"))
    else:
        module.add(SNop(0))

    if doRemainder:
        module.add(VReadfirstlaneB32(dst=sgpr(rReg), src=vgpr(tmpVgpr1),
                                     comment="remainder"))

    return module


def scalarUInt32DivideAndRemainder(qReg, dReg, divReg, rReg, tmpVgprRes,
                                   wavewidth, doRemainder=True, comment=""):
    module = Module("scalarUInt32DivideAndRemainder")

    qRegSgpr = sgpr(qReg)
    rRegSgpr = sgpr(rReg)
    dRegSgpr = sgpr(dReg)
    divRegSgpr = sgpr(divReg)

    dComment = (comment if comment else
                f"{qRegSgpr.toString()} = {dRegSgpr.toString()}"
                f" / {divRegSgpr.toString()}")
    rComment = ""
    if doRemainder:
        rComment = (comment if comment else
                    f"{sgpr(rReg).toString()} = {dRegSgpr.toString()}"
                    f" % {divRegSgpr.toString()}")

    if tmpVgprRes.size < 2:
        raise RuntimeError("Invalid tmpVgprRes, must be at least 2")
    tmpVgpr = vgpr(tmpVgprRes.idx)
    tmpVgpr1 = vgpr(tmpVgprRes.idx + 1)

    pEXEC = EXEC()

    module.add(VCvtU32toF32(dst=tmpVgpr, src=divRegSgpr, comment=dComment))
    module.add(VRcpIFlagF32(dst=tmpVgpr, src=tmpVgpr, comment=dComment))
    module.add(VCvtU32toF32(dst=tmpVgpr1, src=dRegSgpr, comment=dComment))
    module.add(VMulF32(dst=tmpVgpr, src0=tmpVgpr, src1=tmpVgpr1,
                       comment=dComment))
    module.add(VCvtF32toU32(dst=tmpVgpr, src=tmpVgpr, comment=dComment))
    module.add(VMulU32U24(dst=tmpVgpr1, src0=tmpVgpr, src1=divRegSgpr,
                          comment=dComment))
    module.add(VSubU32(dst=tmpVgpr1, src0=dRegSgpr, src1=tmpVgpr1,
                       comment=dComment))
    module.add(VCmpXEqU32(dst=pEXEC, src0=tmpVgpr1, src1=divRegSgpr,
                          comment=dComment))
    module.add(VAddU32(dst=tmpVgpr, src0=1, src1=tmpVgpr, comment=dComment))

    if doRemainder:
        module.add(VMovB32(dst=tmpVgpr1, src=0, comment=rComment))

    def _resetExec():
        if wavewidth == 64:
            module.add(SMovB64(dst=pEXEC, src=-1, comment="Reset exec"))
        else:
            module.add(SMovB32(dst=pEXEC, src=-1, comment="Reset exec"))

    _resetExec()
    module.add(VCmpXGtU32(dst=pEXEC, src0=tmpVgpr1, src1=divRegSgpr,
                          comment="overflow happened in remainder"))
    module.add(VSubU32(dst=tmpVgpr, src0=tmpVgpr, src1=1,
                       comment="quotient - 1"))

    if doRemainder:
        module.add(VMulU32U24(dst=tmpVgpr1, src0=tmpVgpr, src1=divRegSgpr,
                              comment="re-calculate remainder"))
        module.add(VSubU32(dst=tmpVgpr1, src0=dRegSgpr, src1=tmpVgpr1,
                           comment="re-calculate remainder"))

    _resetExec()
    module.add(VReadfirstlaneB32(dst=qRegSgpr, src=tmpVgpr,
                                 comment="quotient"))

    if doRemainder:
        module.add(VReadfirstlaneB32(dst=sgpr(rReg), src=tmpVgpr1,
                                     comment="remainder"))

    return module


# ---------------------------------------------------------------------------
# Magic division
# ---------------------------------------------------------------------------


def sMagicDiv(dest, hasSMulHi, dividend, magicNumber, magicShift, tmpVgpr):
    module = Module("sMagicDiv")

    destSgpr = sgpr(dest, 2)
    destSgpr0 = sgpr(dest)
    destSgpr1 = _get_sgpr(dest, 1)
    continuousReg = ContinuousRegister(tmpVgpr.idx, 2)

    module.addModuleAsFlatItems(_SMulInt64to32(
        destSgpr0, destSgpr1, dividend, magicNumber,
        continuousReg, hasSMulHi, False, "s_magic mul"))
    module.add(SLShiftRightB64(dst=destSgpr, shiftHex=magicShift,
                               src=destSgpr, comment="sMagicDiv"))
    return module


def sMagicDiv2(dst, dst2, dividend, magicNumber, magicShiftAbit, tmpSgpr):
    module = Module("sMagicDiv2")

    module.add(SMulHIU32(dst=dst2, src0=dividend, src1=magicNumber,
                         comment="s_magic mul, div alg 2"))
    module.add(SLShiftRightB32(dst=tmpSgpr, shiftHex=31, src=magicShiftAbit,
                               comment="tmpS = extract abit"))
    module.add(SMulI32(dst=dst, src0=dividend, src1=tmpSgpr,
                       comment="s_magic mul, div alg 2"))
    module.add(SAddU32(dst=dst, src0=dst, src1=dst2))

    module.add(SAndB32(dst=tmpSgpr, src0=magicShiftAbit, src1=0x7fffffff,
                       comment="tmpS = remove abit to final shift"))
    module.add(SLShiftRightB32(dst=dst, shiftHex=tmpSgpr, src=dst,
                               comment="sMagicDiv Alg 2"))

    return module


# ---------------------------------------------------------------------------
# Multiply helpers
# ---------------------------------------------------------------------------


def vectorStaticMultiply(product, operand, multiplier, tmpSgprRes=None,
                         comment=""):
    dComment = (comment if comment else
                f"{product.toString()} = {operand.toString()} * {multiplier}")
    module = Module("vectorStaticMultiply")
    if multiplier == 0:
        module.add(VMovB32(dst=product, src=multiplier, comment=dComment))
    elif _is_power_of_2(multiplier):
        multiplier_log2 = int(math.log2(multiplier))
        if multiplier_log2 == 0 and product == operand:
            module.addCommentAlign(dComment + " (multiplier is 1, do nothing)")
        else:
            module.add(VLShiftLeftB32(dst=product, shiftHex=multiplier_log2,
                                     src=operand, comment=dComment))
    else:
        if -16 <= multiplier <= 64:
            module.add(VMulLOU32(dst=product, src0=multiplier, src1=operand,
                                 comment=dComment))
        else:
            if not tmpSgprRes or tmpSgprRes.size < 1:
                raise RuntimeError("Invalid tmpSgprRes, must be at least 1")
            tmpSgpr = sgpr(tmpSgprRes.idx)
            module.add(SMovB32(dst=tmpSgpr, src=multiplier, comment=dComment))
            module.add(VMulLOU32(dst=product, src0=tmpSgpr, src1=operand,
                                 comment=dComment))
    return module


def vectorStaticMultiplyAdd(product, operand, multiplier, accumulator,
                            tmpSgprRes=None, comment=""):
    dComment = (comment if comment else
                f"{product.toString()} = {operand.toString()} * {multiplier}")
    module = Module("vectorStaticMultiplyAdd")
    if multiplier == 0:
        module.add(VMovB32(dst=product, src=accumulator, comment=dComment))
    elif _is_power_of_2(multiplier):
        multiplier_log2 = int(math.log2(multiplier))
        if multiplier_log2 == 0:
            module.add(VAddU32(dst=product, src0=operand, src1=accumulator,
                               comment=dComment))
        else:
            module.add(VLShiftLeftAddU32(
                dst=product, shiftHex=multiplier_log2, src0=operand,
                src1=accumulator, comment=dComment))
    else:
        if -16 <= multiplier <= 64:
            module.add(VMadU32U24(dst=product, src0=multiplier, src1=operand,
                                  src2=accumulator, comment=dComment))
        else:
            if not tmpSgprRes or tmpSgprRes.size < 1:
                raise RuntimeError("Invalid tmpSgprRes, must be at least 1")
            tmpSgpr = sgpr(tmpSgprRes.idx)
            module.add(SMovB32(dst=tmpSgpr, src=multiplier, comment=dComment))
            module.add(VMadU32U24(dst=product, src0=tmpSgpr, src1=operand,
                                  src2=accumulator, comment=dComment))
    return module


def scalarStaticMultiply64(product, operand, multiplier, tmpSgprRes=None,
                           comment=""):
    commentStr = (comment if comment else
                  f"{product.toString()} = {operand.toString()}"
                  f" * {multiplier}")
    module = Module("scalarStaticMultiply64")
    if multiplier == 0:
        module.add(SMovB64(dst=product, src=0, comment=commentStr))
        return module

    if not _is_power_of_2(multiplier):
        raise RuntimeError("Multiplier must be a power of 2")

    multiplier_log2 = int(math.log2(multiplier))
    if multiplier_log2 == 0 and product == operand:
        module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
    else:
        module.add(SLShiftLeftB64(dst=product, shiftHex=multiplier_log2,
                                  src=operand, comment=commentStr))
    return module


# ---------------------------------------------------------------------------
# Bpe multiply helpers
# ---------------------------------------------------------------------------


def vectorAddMultiplyBpe(dst, src0, src1, bpe, comment=""):
    module = Module("vectorAddMultiplyBpe")
    mcomment = comment + " (multiple bpe)"
    dstVgpr = vgpr(dst)
    src0Vgpr = vgpr(src0)
    src1Vgpr = vgpr(src1)
    if bpe == 0.5:
        module.add(VAddU32(dst=dstVgpr, src0=src0Vgpr, src1=src1Vgpr,
                           comment=mcomment))
        module.add(VLShiftRightB32(dst=dstVgpr, shiftHex=1, src=dstVgpr,
                                   comment=mcomment))
    elif bpe == 0.75:
        module.add(VAddU32(dst=dstVgpr, src0=src0Vgpr, src1=src1Vgpr,
                           comment=mcomment))
        module.add(VMulLOU32(dst=dstVgpr, src0=6, src1=dstVgpr,
                             comment=mcomment))
        module.add(VLShiftRightB32(dst=dstVgpr, shiftHex=3, src=dstVgpr,
                                   comment=mcomment))
    else:
        bpe_log2 = int(math.log2(bpe))
        if bpe_log2 == 0:
            module.add(VAddU32(dst=dstVgpr, src0=src0Vgpr, src1=src1Vgpr,
                               comment=mcomment))
            module.addCommentAlign(comment + " (bpe is 1, no mul)")
        else:
            module.add(VAddLShiftLeftU32(
                dst=dstVgpr, shiftHex=bpe_log2, src0=src0Vgpr, src1=src1Vgpr,
                comment=mcomment))
    return module


def vectorMultiplyBpe(dst, src, bpe, comment=""):
    module = Module("vectorMultiplyBpe")
    mcomment = comment + " (multiple bpe)"
    dstVgpr = vgpr(dst)
    srcVgpr = vgpr(src)
    if bpe == 0.5:
        module.add(VLShiftRightB32(dst=dstVgpr, shiftHex=1, src=srcVgpr,
                                   comment=mcomment))
    elif bpe == 0.75:
        module.add(VMulLOU32(dst=dstVgpr, src0=6, src1=srcVgpr,
                             comment=mcomment))
        module.add(VLShiftRightB32(dst=dstVgpr, shiftHex=3, src=dstVgpr,
                                   comment=mcomment))
    else:
        bpe_log2 = int(math.log2(bpe))
        dst_str = _to_str(dst)
        src_str = _to_str(src)
        if bpe_log2 == 0 and dst_str == src_str:
            module.addCommentAlign(comment + " (bpe is 1, do nothing)")
        else:
            module.add(VLShiftLeftB32(dst=dstVgpr, shiftHex=bpe_log2,
                                     src=srcVgpr, comment=mcomment))
    return module


def vectorMultiply64Bpe(dst, src, bpe, tmp, comment=""):
    module = Module("vectorMultiply64Bpe")
    mcomment = comment + " (multiple bpe)"
    dstVgpr = vgpr(dst, 2)
    srcVgpr = vgpr(src, 2)
    tmpVgpr = vgpr(tmp)
    if bpe == 0.5:
        module.add(VLShiftRightB64(dst=dstVgpr, shiftHex=1, src=srcVgpr,
                                   comment=mcomment))
    elif bpe == 0.75:
        dstVgpr0 = vgpr(dst)
        dstVgpr1 = _get_vgpr(dst, 1)
        srcVgpr0 = vgpr(src)
        srcVgpr1 = _get_vgpr(src, 1)
        module.add(VMovB32(dst=tmpVgpr, src=srcVgpr1, comment=mcomment))
        module.add(VMulHIU32(dst=dstVgpr1, src0=6, src1=srcVgpr0,
                             comment=mcomment))
        module.add(VMulLOU32(dst=dstVgpr0, src0=6, src1=srcVgpr0,
                             comment=mcomment))
        module.add(VMulLOU32(dst=tmpVgpr, src0=6, src1=tmpVgpr,
                             comment=mcomment))
        module.add(VAddU32(dst=dstVgpr1, src0=dstVgpr1, src1=tmpVgpr,
                           comment=mcomment))
        module.add(VLShiftRightB64(dst=dstVgpr, shiftHex=3, src=dstVgpr,
                                   comment=mcomment))
    else:
        bpe_log2 = int(math.log2(bpe))
        dst_str = _to_str(dst)
        src_str = _to_str(src)
        if bpe_log2 == 0 and dst_str == src_str:
            module.addCommentAlign(comment + " (bpe is 1, do nothing)")
        else:
            module.add(VLShiftLeftB64(dst=dstVgpr, shiftHex=bpe_log2,
                                     src=srcVgpr, comment=mcomment))
    return module


def scalarMultiplyBpe(dst, src, bpe, comment=""):
    module = Module("scalarMultiplyBpe")
    mcomment = comment + " (multiple bpe)"
    dstSgpr = sgpr(dst)
    srcSgpr = sgpr(src)
    if bpe == 0.5:
        module.add(SLShiftRightB32(dst=dstSgpr, shiftHex=1, src=srcSgpr,
                                   comment=mcomment))
    elif bpe == 0.75:
        module.add(SMulI32(dst=dstSgpr, src0=6, src1=srcSgpr,
                           comment=mcomment))
        module.add(SLShiftRightB32(dst=dstSgpr, shiftHex=3, src=dstSgpr,
                                   comment=mcomment))
    else:
        bpe_log2 = int(math.log2(bpe))
        dst_str = _to_str(dst)
        src_str = _to_str(src)
        if bpe_log2 == 0 and dst_str == src_str:
            module.addCommentAlign(comment + " (bpe is 1, do nothing)")
        else:
            module.add(SLShiftLeftB32(dst=dstSgpr, shiftHex=bpe_log2,
                                     src=srcSgpr, comment=mcomment))
    return module


def scalarMultiply64Bpe(dst, src, bpe, tmp, comment=""):
    module = Module("scalarMultiply64Bpe")
    mcomment = comment + " (multiple bpe)"
    dstSgpr = sgpr(dst, 2)
    srcSgpr = sgpr(src, 2)
    tmpSgpr = sgpr(tmp)
    if bpe == 0.5:
        module.add(SLShiftRightB64(dst=dstSgpr, shiftHex=1, src=srcSgpr,
                                   comment=mcomment))
    elif bpe == 0.75:
        dstSgpr0 = sgpr(dst)
        dstSgpr1 = _get_sgpr(dst, 1)
        srcSgpr0 = sgpr(src)
        srcSgpr1 = _get_sgpr(src, 1)
        module.add(SMovB32(dst=tmpSgpr, src=srcSgpr1, comment=mcomment))
        module.add(SMulHIU32(dst=dstSgpr1, src0=6, src1=srcSgpr0,
                             comment=mcomment))
        module.add(SMulI32(dst=dstSgpr0, src0=6, src1=srcSgpr0,
                           comment=mcomment))
        module.add(SMulI32(dst=tmpSgpr, src0=6, src1=tmpSgpr,
                           comment=mcomment))
        module.add(SAddU32(dst=dstSgpr1, src0=dstSgpr1, src1=tmpSgpr,
                           comment=mcomment))
        module.add(SLShiftRightB64(dst=dstSgpr, shiftHex=3, src=dstSgpr,
                                   comment=mcomment))
    else:
        bpe_log2 = int(math.log2(bpe))
        dst_str = _to_str(dst)
        src_str = _to_str(src)
        if bpe_log2 == 0 and dst_str == src_str:
            module.addCommentAlign(comment + " (bpe is 1, do nothing)")
        else:
            module.add(SLShiftLeftB64(dst=dstSgpr, shiftHex=bpe_log2,
                                     src=srcSgpr, comment=mcomment))
    return module


DSInit = make_dummy_func(f"{_P}.DSInit")
