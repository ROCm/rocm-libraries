# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Data type to rocisa InstType mapping, and the mnemonic a matrix instruction emits.

``matrixInstructionMnemonic`` asks the assembly backend for the mnemonic it would
emit for a MatrixInstruction, so callers that need to know whether an instruction
exists (see ``SolutionStructs.Validators.MatrixInstruction``) ask the backend
rather than carrying their own opcode table.
"""

from typing import Tuple

from rocisa import rocIsa
from rocisa.container import vgpr
from rocisa.enum import InstType
from rocisa.instruction import MFMAInstruction

from .DataType import DataType


def dataTypeNameAbbrevToInstType(abbrev: str, sourceSwap: bool = False) -> InstType:
    if abbrev == 'f64':
        return InstType.INST_F64
    elif abbrev == 'f32':
        return InstType.INST_F32
    elif abbrev == 'f16':
        return InstType.INST_F16
    elif abbrev == 'i32':
        return InstType.INST_I32
    elif abbrev == 'i8':
        return InstType.INST_I8
    elif abbrev == 'bf16':
        return InstType.INST_BF16
    elif abbrev == 'xf32':
        return InstType.INST_XF32
    elif abbrev == 'fp8':
        return InstType.INST_F8
    elif abbrev == 'bf8':
        return InstType.INST_BF8
    elif (abbrev == 'fp8_bf8' and sourceSwap == False) or \
        (abbrev == 'bf8_fp8' and sourceSwap == True):
        return InstType.INST_F8_BF8
    elif (abbrev == 'bf8_fp8' and sourceSwap == False) or \
        (abbrev == 'fp8_bf8' and sourceSwap == True):
        return InstType.INST_BF8_F8
    elif abbrev == 'fp6':
        return InstType.INST_F6
    elif abbrev == 'bf6':
        return InstType.INST_BF6
    elif (abbrev == 'fp6_bf6' and sourceSwap == False) or \
        (abbrev == 'bf6_fp6' and sourceSwap == True):
        return InstType.INST_F6_B6
    elif (abbrev == 'bf6_fp6' and sourceSwap == False) or \
        (abbrev == 'fp6_bf6' and sourceSwap == True):
        return InstType.INST_B6_F6
    elif abbrev == 'fp4':
        return InstType.INST_F4
    elif (abbrev == 'fp8_fp4' and sourceSwap == False) or \
        (abbrev == 'fp4_fp8' and sourceSwap == True):
        return InstType.INST_F8_F4
    elif (abbrev == 'fp4_fp8' and sourceSwap == False) or \
        (abbrev == 'fp8_fp4' and sourceSwap == True):
        return InstType.INST_F4_F8
    elif (abbrev == 'fp6_fp4' and sourceSwap == False) or \
        (abbrev == 'fp4_fp6' and sourceSwap == True):
        return InstType.INST_F6_F4
    elif (abbrev == 'fp4_fp6' and sourceSwap == False) or \
        (abbrev == 'fp6_fp4' and sourceSwap == True):
        return InstType.INST_F4_F6
    elif (abbrev == 'fp8_fp6' and sourceSwap == False) or \
        (abbrev == 'fp6_fp8' and sourceSwap == True):
        return InstType.INST_F8_F6
    elif (abbrev == 'fp6_fp8' and sourceSwap == False) or \
        (abbrev == 'fp8_fp6' and sourceSwap == True):
        return InstType.INST_F6_F8
    elif (abbrev == 'fp8_bf6' and sourceSwap == False) or \
        (abbrev == 'bf6_fp8' and sourceSwap == True):
        return InstType.INST_F8_B6
    elif (abbrev == 'bf6_fp8' and sourceSwap == False) or \
        (abbrev == 'fp8_bf6' and sourceSwap == True):
        return InstType.INST_B6_F8
    elif (abbrev == 'bf8_fp4' and sourceSwap == False) or \
        (abbrev == 'fp4_bf8' and sourceSwap == True):
        return InstType.INST_B8_F4
    elif (abbrev == 'fp4_bf8' and sourceSwap == False) or \
        (abbrev == 'bf8_fp4' and sourceSwap == True):
        return InstType.INST_F4_B8
    elif (abbrev == 'bf6_fp4' and sourceSwap == False) or \
        (abbrev == 'fp4_bf6' and sourceSwap == True):
        return InstType.INST_B6_F4
    elif (abbrev == 'fp4_bf6' and sourceSwap == False) or \
        (abbrev == 'bf6_fp4' and sourceSwap == True):
        return InstType.INST_F4_B6
    elif (abbrev == 'bf8_fp6' and sourceSwap == False) or \
        (abbrev == 'fp6_bf8' and sourceSwap == True):
        return InstType.INST_B8_F6
    elif (abbrev == 'fp6_bf8' and sourceSwap == False) or \
        (abbrev == 'bf8_fp6' and sourceSwap == True):
        return InstType.INST_F6_B8
    elif (abbrev == 'bf8_bf6' and sourceSwap == False) or \
        (abbrev == 'bf6_bf8' and sourceSwap == True):
        return InstType.INST_B8_B6
    elif (abbrev == 'bf6_bf8' and sourceSwap == False) or \
        (abbrev == 'bf8_bf6' and sourceSwap == True):
        return InstType.INST_B6_B8
    elif abbrev == 'e8':
        return InstType.INST_E8
    elif abbrev == 'e5m3':
        return InstType.INST_E5M3
    else:
        assert("Unsupported data type.")
    return InstType.INST_NOTYPE


def dataTypeToMfmaInstTypePair(
    dataTypeA: DataType, dataTypeB: DataType, sourceSwap: bool
) -> Tuple[InstType, InstType]:
    miInTypeStrA  = dataTypeA.toNameAbbrev()
    miInTypeStrB  = dataTypeB.toNameAbbrev()
    miInTypeStr = miInTypeStrA + "_" + miInTypeStrB if miInTypeStrA != miInTypeStrB else miInTypeStrA
    miInInstType = dataTypeNameAbbrevToInstType(miInTypeStr, sourceSwap) # v_mfma_[...xK]<InType>
    miOutInstType = dataTypeNameAbbrevToInstType(dataTypeA.MIOutputTypeNameAbbrev()) # v_mfma_<OutType>..
    return miInInstType, miOutInstType


def matrixInstructionTypes(solution: dict, hasMFMA: bool):
    """Return the (input, output, negFlag) instruction types the emitter uses.

    The MAC data types are not the whole story: F32XdlMathOp replaces them, WMMA
    spells i8 as iu8, and a WMMA output type comes from ComputeDataType. Callers
    that need to know which instruction a solution emits have to agree with the
    emitter on all of it, so this is the one place it is decided.
    """
    ptype = solution["ProblemType"]
    # Validation runs before a solution is fully assigned, so these two can be absent
    # there; a kernel reaching the emitter always carries them.
    enableF32Xdl = solution.get("EnableF32XdlMathOp", False)
    sourceSwap = solution.get("SourceSwap", False)

    miInputTypeA = ptype["F32XdlMathOp"] if enableF32Xdl else ptype["MacDataTypeA"]
    miInputTypeB = ptype["F32XdlMathOp"] if enableF32Xdl else ptype["MacDataTypeB"]

    miInInstType, miOutInstType = dataTypeToMfmaInstTypePair(
        miInputTypeA, miInputTypeB, sourceSwap
    )
    negFlag = True if ((not hasMFMA) and (miInInstType == InstType.INST_I8)) else False
    miInInstType = InstType.INST_U8 if ((not hasMFMA) and miInInstType == InstType.INST_I8) else miInInstType
    computeDataType = ptype["ComputeDataType"]
    # complex WMMA is emulated with real matrix ops, so the output inst type is the
    # real base (f32/f64), not the complex abbrev (f32c/f64c) which has no InstType.
    computeOutAbbrev = computeDataType.MIOutputTypeNameAbbrev() if computeDataType.isComplex() else computeDataType.toNameAbbrev()
    miOutInstType = miOutInstType if (hasMFMA or ptype["Sparse"]) else dataTypeNameAbbrevToInstType(computeOutAbbrev)
    return miInInstType, miOutInstType, negFlag


def matrixInstructionMnemonic(
    solution: dict,
    isa,
    mi4: list,
    hasMFMA: bool,
) -> str:
    """Return the mnemonic the backend emits for *mi4* in this solution.

    The mnemonic depends on the ISA's capabilities (which suffix a type maps to,
    whether a scaled WMMA encoding is forced), so the thread's kernel ISA is set
    for the duration of the query and restored afterwards.
    """
    miInInstType, miOutInstType, _ = matrixInstructionTypes(solution, hasMFMA)
    # F32 emulation multiplies bf16 halves, so bf16 is what actually gets emitted.
    if solution.get("UseF32XEmulation", False):
        miInInstType = InstType.INST_BF16

    ti = rocIsa.getInstance()
    prevKernel = ti.getKernel()
    ti.setKernel(tuple(isa), solution["WavefrontSize"])
    try:
        # Registers do not affect the mnemonic; preStr() reads only the types and variant.
        inst = MFMAInstruction(
            instType=miInInstType,
            accType=miOutInstType,
            variant=list(mi4),
            mfma1k=solution.get("MFMA_BF16_1K", False),
            acc=vgpr(0, 1),
            a=vgpr(0, 1),
            b=vgpr(0, 1),
        )
        return inst.preStr()
    finally:
        ti.setKernel(tuple(prevKernel.isa), prevKernel.wavefrontSize)
