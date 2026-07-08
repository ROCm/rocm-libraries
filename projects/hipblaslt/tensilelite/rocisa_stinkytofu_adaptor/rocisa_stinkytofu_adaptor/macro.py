# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Composite macro builders (MacroVMagicDiv, PseudoRandomGenerator).

Macro/MacroInstruction IR types are real; builder callables are still dummy.
"""

from __future__ import annotations

from typing import Any

from .code import Macro, Module, TextBlock
from .container import sgpr, vgpr
from .instruction import (
    VAddU32, VAndB32, VLShiftLeftB32, VLShiftRightB32, VLShiftRightB64,
    VMulHIU32, VMulLOU32, VMulU32U24, VXorB32, _VLShiftLeftOrB32,
)


def MacroVMagicDiv(algo: int) -> Macro:
    """Build the V_MAGIC_DIV macro definition (macro.hpp:36-58)."""
    macro = Macro(
        "V_MAGIC_DIV",
        ["vgprDstIdx:req", "dividend:req", "magicNumber:req",
         "magicShift:req", "magicA:req"],
    )
    if algo == 1:
        macro.addT(VMulHIU32, dst=vgpr("DstIdx+1", 1, True),
                   src0="\\dividend", src1="\\magicNumber")
        macro.addT(VMulLOU32, dst=vgpr("DstIdx+0", 1, True),
                   src0="\\dividend", src1="\\magicNumber")
        macro.addT(VLShiftRightB64, dst=vgpr("DstIdx", 2, True),
                   shiftHex="\\magicShift", src=vgpr("DstIdx", 2, True))
    elif algo == 2:
        macro.addT(VMulHIU32, dst=vgpr("DstIdx+1", 1, True),
                   src0="\\dividend", src1="\\magicNumber")
        macro.addT(VMulLOU32, dst=vgpr("DstIdx+0", 1, True),
                   src0="\\dividend", src1="\\magicA")
        macro.addT(VAddU32, dst=vgpr("DstIdx+0", 1, True),
                   src0=vgpr("DstIdx+0", 1, True), src1=vgpr("DstIdx+1", 1, True))
        macro.addT(VLShiftRightB32, dst=vgpr("DstIdx+0", 1, True),
                   shiftHex="\\magicShift", src=vgpr("DstIdx+0", 1, True))
    return macro


def VMagicDiv(algo: int, dstIdx: int, dividend: Any,
              magicNumber: Any, magicShift: Any, magicA: Any) -> Module:
    """Inlined V_MAGIC_DIV — module form (macro.hpp:121-149)."""
    module = Module("V_MAGIC_DIV")
    module.addComment0("V_MAGIC_DIV start")
    if algo == 1:
        module.add(VMulHIU32(dst=vgpr(dstIdx + 1, 1), src0=dividend,
                             src1=magicNumber))
        module.add(VMulLOU32(dst=vgpr(dstIdx, 1), src0=dividend,
                             src1=magicNumber))
        module.add(VLShiftRightB64(dst=vgpr(dstIdx, 2), shiftHex=magicShift,
                                   src=vgpr(dstIdx, 2)))
    elif algo == 2:
        module.add(VMulHIU32(dst=vgpr(dstIdx + 1, 1), src0=dividend,
                             src1=magicNumber))
        module.add(VMulLOU32(dst=vgpr(dstIdx, 1), src0=dividend,
                             src1=magicA))
        module.add(VAddU32(dst=vgpr(dstIdx, 1), src0=vgpr(dstIdx, 1),
                           src1=vgpr(dstIdx + 1, 1)))
        module.add(VLShiftRightB32(dst=vgpr(dstIdx, 1), shiftHex=magicShift,
                                   src=vgpr(dstIdx, 1)))
    module.addComment0("V_MAGIC_DIV end")
    return module


def PseudoRandomGenerator() -> Macro:
    """Build the PRND_GENERATOR macro definition (macro.hpp:60-119)."""
    macro = Macro(
        "PRND_GENERATOR",
        ["vgprRand:req", "vgprAcc:req", "vgprTemp0:req", "vgprTemp1:req"],
    )
    macro.addComment0("PRND_GENERATOR: vgprRand=RND(vgprAcc, sgprSeed, vgprTid)")

    macro.addT(VAndB32, dst=vgpr("Temp0", 1, True), src0="0xFFFF",
               src1=vgpr("Acc", 1, True), comment="vgprTemp0 = vgprAcc & 0xFFFF")
    macro.addT(VLShiftRightB32, dst=vgpr("Temp1", 1, True), shiftHex=16,
               src=vgpr("Acc", 1, True), comment="vgprTemp1 = vgprAcc >> 16")
    macro.addT(VXorB32, dst=vgpr("Temp0", 1, True), src0=vgpr("Temp0", 1, True),
               src1=vgpr("Temp1", 1, True), comment="VTemp0 = vgprTemp0 ^ vgprTemp1")
    macro.addT(VAndB32, dst=vgpr("Temp1", 1, True), src0=vgpr("Temp0", 1, True),
               src1=31, comment="vgprTemp1 = vgprTemp0 & 31")
    macro.addT(VLShiftLeftB32, dst=vgpr("Temp1", 1, True), shiftHex=11,
               src=vgpr("Temp1", 1, True), comment="vgprTemp1 = vgprTemp1 << 11")
    macro.addT(_VLShiftLeftOrB32, dst=vgpr("Temp0", 1, True),
               src0=vgpr("Temp0", 1, True), src1=5, src2=vgpr("Temp1", 1, True),
               comment="vgprTemp0 = vgprTemp0 << 5 | vgprTemp1")
    macro.addT(VMulU32U24, dst=vgpr("Temp0", 1, True), src0="0x700149",
               src1=vgpr("Temp0", 1, True), comment="VTemp0 = vgprTemp0 * 0x700149")
    macro.addT(VMulU32U24, dst=vgpr("Temp1", 1, True), src0=229791,
               src1=vgpr("Serial"), comment="VTemp1 = vTid * 229791")
    macro.addT(VXorB32, dst=vgpr("Rand", 1, True), src0="0x1337137",
               src1=vgpr("Temp0", 1, True), comment="VRand = vgprTemp0 ^ 0x1337137")
    macro.addT(VXorB32, dst=vgpr("Rand", 1, True), src0=vgpr("Rand", 1, True),
               src1=vgpr("Temp1", 1, True), comment="VRand = vgprRand ^ vgprTemp1")
    macro.addT(VXorB32, dst=vgpr("Rand", 1, True), src0=vgpr("Rand", 1, True),
               src1=sgpr("RNDSeed"), comment="VRand = vgprRand ^ sSeed")
    return macro


def PseudoRandomGeneratorModule(Rand: int, Acc: int,
                                Temp0: int, Temp1: int) -> Module:
    """Inlined PRND_GENERATOR — module form (macro.hpp:151-210)."""
    module = Module("PRND_GENERATOR")
    module.addComment0("PRND_GENERATOR: vgprRand=RND(vgprAcc, sgprSeed, vgprTid)")

    module.add(VAndB32(dst=vgpr(Temp0, 1), src0="0xFFFF",
                       src1=vgpr(Acc, 1), comment="vgprTemp0 = vgprAcc & 0xFFFF"))
    module.add(VLShiftRightB32(dst=vgpr(Temp1, 1), shiftHex=16,
                               src=vgpr(Acc, 1), comment="vgprTemp1 = vgprAcc >> 16"))
    module.add(VXorB32(dst=vgpr(Temp0, 1), src0=vgpr(Temp0, 1),
                       src1=vgpr(Temp1, 1), comment="VTemp0 = vgprTemp0 ^ vgprTemp1"))
    module.add(VAndB32(dst=vgpr(Temp1, 1), src0=vgpr(Temp0, 1),
                       src1=31, comment="vgprTemp1 = vgprTemp0 & 31"))
    module.add(VLShiftLeftB32(dst=vgpr(Temp1, 1), shiftHex=11,
                              src=vgpr(Temp1, 1), comment="vgprTemp1 = vgprTemp1 << 11"))
    module.add(_VLShiftLeftOrB32(dst=vgpr(Temp0, 1), src0=vgpr(Temp0, 1),
                                 src1=5, src2=vgpr(Temp1, 1),
                                 comment="vgprTemp0 = vgprTemp0 << 5 | vgprTemp1"))
    module.add(VMulU32U24(dst=vgpr(Temp0, 1), src0="0x700149",
                          src1=vgpr(Temp0, 1), comment="VTemp0 = vgprTemp0 * 0x700149"))
    module.add(VMulU32U24(dst=vgpr(Temp1, 1), src0=229791,
                          src1=vgpr("Serial"), comment="VTemp1 = vTid * 229791"))
    module.add(VXorB32(dst=vgpr(Rand, 1), src0="0x1337137",
                       src1=vgpr(Temp0, 1), comment="VRand = vgprTemp0 ^ 0x1337137"))
    module.add(VXorB32(dst=vgpr(Rand, 1), src0=vgpr(Rand, 1),
                       src1=vgpr(Temp1, 1), comment="VRand = vgprRand ^ vgprTemp1"))
    module.add(VXorB32(dst=vgpr(Rand, 1), src0=vgpr(Rand, 1),
                       src1=sgpr("RNDSeed"), comment="VRand = vgprRand ^ sSeed"))
    module.addComment0("PRND_GENERATOR end")
    return module
