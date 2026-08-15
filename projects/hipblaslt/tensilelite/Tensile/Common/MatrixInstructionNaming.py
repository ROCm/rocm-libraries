# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Data type to rocisa InstType mapping, and the mnemonic a matrix instruction emits.

``matrixInstructionMnemonic`` asks the assembly backend for the mnemonic it would
emit for a MatrixInstruction, so callers that need to know whether an instruction
exists (see ``SolutionStructs.Validators.MatrixInstruction``) ask the backend
rather than carrying their own opcode table.
"""

from contextlib import contextmanager
from typing import Tuple

from rocisa import rocIsa
from rocisa.container import vgpr
from rocisa.enum import InstType
from rocisa.instruction import MFMAInstruction, MXMFMAInstruction

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


def matrixInstructionTypes(
    miInputTypeA: DataType,
    miInputTypeB: DataType,
    computeDataType: DataType,
    sourceSwap: bool,
    isSparse,
    hasMFMA: bool,
):
    """Return the (input, output, negFlag) instruction types the emitter uses.

    The input data types are not the whole story: WMMA spells i8 as iu8, and a WMMA
    output type comes from ComputeDataType rather than the input type. Callers that
    need to know which instruction a solution emits have to agree with the emitter
    on all of it, so this is the one place it is decided.

    Inputs are passed already resolved (F32XdlMathOp substituted, coerced to
    DataType) so each caller keeps its own rules for reading them out of a solution.
    """
    miInInstType, miOutInstType = dataTypeToMfmaInstTypePair(
        miInputTypeA, miInputTypeB, sourceSwap
    )
    negFlag = True if ((not hasMFMA) and (miInInstType == InstType.INST_I8)) else False
    miInInstType = InstType.INST_U8 if ((not hasMFMA) and miInInstType == InstType.INST_I8) else miInInstType
    # complex WMMA is emulated with real matrix ops, so the output inst type is the
    # real base (f32/f64), not the complex abbrev (f32c/f64c) which has no InstType.
    computeOutAbbrev = computeDataType.MIOutputTypeNameAbbrev() if computeDataType.isComplex() else computeDataType.toNameAbbrev()
    miOutInstType = miOutInstType if (hasMFMA or isSparse) else dataTypeNameAbbrevToInstType(computeOutAbbrev)
    return miInInstType, miOutInstType, negFlag


def pinnedIsa(kernelInfo):
    """The ISA *kernelInfo* has pinned, or None when the thread never pinned one.

    The two backends spell "no kernel" differently. The stinkytofu adaptor uses
    ``isa=None``; rocisa's ``KernelInfo`` is a C++ aggregate that the per-thread
    map value-initialises on lookup, so an unpinned thread reads back ``(0, 0, 0)``.
    Neither is an ISA anyone is running, and putting ``(0, 0, 0)`` back would leave
    the thread pinned to one whose capability map is empty -- ``getAsmCaps`` indexes
    ``m_isainfo`` with ``operator[]`` -- so everything the thread generated
    afterwards would silently read capability defaults instead of the target's.
    """
    isa = getattr(kernelInfo, "isa", None)
    if isa is None:
        return None
    isa = tuple(isa)
    return isa if any(isa) else None


def backendCapsLoaded(isa) -> bool:
    """Whether the backend can name instructions for *isa* accurately.

    ``matrixInstructionMnemonic`` takes the spelling from the ISA's assembler
    capabilities -- which K an f8f6f4 encoding starts at, whether a scaled WMMA is
    forced -- and rocisa serves those from a per-process map that only
    ``rocIsa.init`` fills, read with ``operator[]``. A process that never ran init
    for *isa*, such as a joblib/loky worker (handed globalParameters and nothing
    else), gets an empty map and names an instruction the emitter never emits, so
    a caller must not turn an answer from one into a rejection.
    """
    try:
        return bool(rocIsa.getInstance().getIsaInfo(tuple(isa)).asmCaps)
    except (AttributeError, RuntimeError):
        # A backend that cannot be asked cannot be trusted to have answered either.
        return False


@contextmanager
def _pinnedKernelIsa(isa, wavefrontSize: int):
    """Pin the thread's kernel ISA, then put back everything setKernel disturbs.

    rocisa's setKernel does not only switch ISA: it also clears the thread's VGPR
    index map and MSB (rocisa base.hpp). Saving and restoring those keeps this a
    query rather than a mutation of whatever the thread was in the middle of.
    """
    ti = rocIsa.getInstance()
    prevKernel = ti.getKernel()
    prevIsa = pinnedIsa(prevKernel)
    prevVgprIdx = ti.getVgprIdx()
    prevVgprMsb = ti.getVgprMsb()

    ti.setKernel(tuple(isa), wavefrontSize)
    try:
        yield
    finally:
        if prevIsa is not None:
            ti.setKernel(prevIsa, prevKernel.wavefrontSize)
        for name, idx in prevVgprIdx.items():
            ti.setVgprIdx(name, idx)
        ti.setVgprMsb(prevVgprMsb)


def matrixInstructionMnemonic(
    isa,
    wavefrontSize: int,
    mi4: list,
    miInputTypeA: DataType,
    miInputTypeB: DataType,
    computeDataType: DataType,
    sourceSwap: bool = False,
    isSparse=0,
    hasMFMA: bool = False,
    mfma1k: bool = False,
    useF32XEmulation: bool = False,
    mxBlock: int = 0,
) -> str:
    """Return the mnemonic the backend emits for *mi4* with these types.

    The mnemonic depends on the ISA's capabilities (which suffix a type maps to,
    whether a scaled WMMA encoding is forced), so the thread's kernel ISA is pinned
    for the duration of the query.

    Raises RuntimeError from the backend for a data type that has no matrix
    instruction spelling at all (complex, for one).
    """
    miInInstType, miOutInstType, _ = matrixInstructionTypes(
        miInputTypeA, miInputTypeB, computeDataType, sourceSwap, isSparse, hasMFMA
    )

    with _pinnedKernelIsa(isa, wavefrontSize):
        # Registers do not affect the mnemonic; preStr() reads only types, variant
        # and block. The branch order mirrors the emitter (KernelWriterAssembly
        # mfmaIter): emulation first, then MX, then the plain instruction.
        if useF32XEmulation:
            # F32 emulation multiplies bf16 halves, so bf16 is what gets emitted.
            miInInstType = InstType.INST_BF16
        elif mxBlock:
            return MXMFMAInstruction(
                instType=miInInstType,
                accType=miOutInstType,
                variant=list(mi4),
                acc=vgpr(0, 1),
                a=vgpr(0, 1),
                b=vgpr(0, 1),
                block=mxBlock,
            ).preStr()

        return MFMAInstruction(
            instType=miInInstType,
            accType=miOutInstType,
            variant=list(mi4),
            # Solution dicts carry this as 0/1; the binding's bool is strict.
            mfma1k=bool(mfma1k),
            acc=vgpr(0, 1),
            a=vgpr(0, 1),
            b=vgpr(0, 1),
        ).preStr()
