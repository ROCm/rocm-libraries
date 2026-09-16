# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validator for the ``UseScaleAB="Block"`` in-kernel dequantization path
(w4a16: int4 weights in A, bf16 activations in B, per-group scales).

The kernel loads A in its narrow in-memory type (``DataTypeA``), multiplies each
element by the scale of its K-group, converts to ``MacDataTypeA`` and writes
*that* to LDS.  Everything downstream of the LDS write is an ordinary
``MacDataTypeA`` GEMM.

Public entry point: :func:`validateBlockDequantCombination`.
"""

from ..Problem import INT4_ENCODINGS_A, usesBlockDequantA, usesBlockDequantZeroPointA
from ..Utilities import reject


#: K-group sizes the prototype generates code for.
SUPPORTED_BLOCK_SIZES = (32, 64, 128)


def validateBlockDequantCombination(state, printRejectionReason):
    """Reject solutions that ask for block dequantization in a shape the kernel
    generator does not emit.

    Must run late in ``assignDerivedParameters`` — it reads ``DepthU``,
    ``GlobalReadVectorWidthA`` and ``UnrollMajorLDSA``, which are derived there.

    Returns ``True`` when the state is acceptable (including when block
    dequantization is off), ``False`` when a reject was emitted.
    """
    problemType = state["ProblemType"]

    if not usesBlockDequantA(problemType):
        # ScaleBlockSizeA without UseScaleAB="Block" is a config mistake that
        # would otherwise be silently ignored.
        if problemType["ScaleBlockSizeA"]:
            reject(state, printRejectionReason,
                   "ScaleBlockSizeA=%d requires UseScaleAB=Block"
                   % problemType["ScaleBlockSizeA"])
            return False
        if problemType["ScaleZeroPointA"]:
            reject(state, printRejectionReason,
                   "ScaleZeroPointA requires UseScaleAB=Block with ScaleBlockSizeA != 0")
            return False
        if problemType["Int4EncodingA"] != "Signed":
            reject(state, printRejectionReason,
                   "Int4EncodingA=%s requires UseScaleAB=Block with ScaleBlockSizeA != 0"
                   % problemType["Int4EncodingA"])
            return False
        return True

    blockSize = problemType["ScaleBlockSizeA"]

    if problemType["Int4EncodingA"] not in INT4_ENCODINGS_A:
        reject(state, printRejectionReason,
               "Int4EncodingA must be one of %s (got %r)"
               % (", ".join(INT4_ENCODINGS_A), problemType["Int4EncodingA"]))
        return False

    if blockSize not in SUPPORTED_BLOCK_SIZES:
        reject(state, printRejectionReason,
               "UseScaleAB=Block supports ScaleBlockSizeA in %s (got %d)"
               % (str(SUPPORTED_BLOCK_SIZES), blockSize))
        return False

    # --- what the dequantize emitter can actually convert ---
    if not problemType["DataTypeA"].isInt4():
        reject(state, printRejectionReason,
               "UseScaleAB=Block only implements DataTypeA=I4 (got %s)"
               % problemType["DataTypeA"].toChar())
        return False
    # A is dequantized into MacDataTypeA and then MACed against B, so the two
    # must agree; bf16 and fp16 are both implemented.
    macType = problemType["MacDataTypeA"]
    if not (macType.isBFloat16() or macType.isHalf()):
        reject(state, printRejectionReason,
               "UseScaleAB=Block only implements MacDataTypeA in (B, H) (got %s)"
               % macType.toChar())
        return False
    if macType.toChar() != problemType["DataType"].toChar():
        reject(state, printRejectionReason,
               "UseScaleAB=Block requires MacDataTypeA (%s) to match DataType (%s)"
               % (macType.toChar(), problemType["DataType"].toChar()))
        return False
    # --- the conversion has to happen on the way *into* LDS ---
    if state["ConvertAfterDS"]:
        reject(state, printRejectionReason,
               "UseScaleAB=Block dequantizes before the LDS write; "
               "ConvertAfterDS must be False")
        return False
    if not state["UnrollMajorLDSA"]:
        reject(state, printRejectionReason,
               "UseScaleAB=Block requires UnrollMajorLDSA (A stored K-contiguous, i.e. TN)")
        return False
    if not problemType["TransposeA"]:
        reject(state, printRejectionReason, "UseScaleAB=Block requires TransposeA (TN layout)")
        return False

    # --- one scale per A global-load instruction ---
    # A is unroll-major, so a thread's GlobalReadVectorWidthA elements are
    # contiguous along K.  Keeping that inside one K-group is what lets a single
    # scalar scale cover the whole load.
    if state["GlobalReadVectorWidthA"] > blockSize:
        reject(state, printRejectionReason,
               "UseScaleAB=Block requires GlobalReadVectorWidthA (%d) <= ScaleBlockSizeA (%d)"
               % (state["GlobalReadVectorWidthA"], blockSize))
        return False
    # The dequantize emitter expands exactly one loaded dword (8 int4) into one
    # 16-byte LDS write (8 bf16). Wider loads would straddle several ds_writes,
    # which localWriteDo splits across its `s` loop with a different g2lIdx each
    # time; that mapping is not implemented yet.
    grBytesA = state["GlobalReadVectorWidthA"] * problemType["DataTypeA"].numBytes()
    if grBytesA != 4:
        reject(state, printRejectionReason,
               "UseScaleAB=Block currently needs exactly one dword per A load: "
               "GlobalReadVectorWidthA (%d) * %g bytes = %g, want 4"
               % (state["GlobalReadVectorWidthA"], problemType["DataTypeA"].numBytes(), grBytesA))
        return False

    # StaggerU shifts A's K start per workgroup without a matching shift of the
    # scale pointer, so the two would disagree.
    if state["StaggerU"] != 0:
        reject(state, printRejectionReason, "UseScaleAB=Block requires StaggerU=0")
        return False

    # NOTE: there is deliberately no DepthU/ScaleBlockSizeA parity constraint.
    # Zero-points sit at byte (m/2)*kGroups + g -- [M][kGroups] order, but packed
    # two rows per byte -- so one K group is one byte: a K iteration advances the
    # SRD by DepthU/G whole bytes and never changes a thread's nibble. Packing
    # along K instead would need DepthU/G even, because an odd group advance is
    # half a byte per row and flips every thread's nibble parity.

    # Asymmetric: the kernel takes the nibble from the parity of the
    # workgroup-local row. That is only the global row's parity when the
    # workgroup's first row wgM*MacroTile0 is even, i.e. when MacroTile0 is even
    # -- which it always is in practice, but the kernel would be silently wrong
    # if not.
    if usesBlockDequantZeroPointA(problemType) and state["MacroTile0"] % 2 != 0:
        reject(state, printRejectionReason,
               "UseScaleAB=Block with zero-points requires an even MacroTile0 (got %d)"
               % state["MacroTile0"])
        return False

    # The per-iteration scale-pointer advance is DepthU/blockSize scale
    # elements; a fractional advance would need a modulo counter.
    if state["DepthU"] % blockSize != 0:
        reject(state, printRejectionReason,
               "UseScaleAB=Block requires DepthU (%d) %% ScaleBlockSizeA (%d) == 0"
               % (state["DepthU"], blockSize))
        return False

    # --- paths that bypass the localWrite conversion entirely ---
    if not state["BufferLoad"]:
        reject(state, printRejectionReason, "UseScaleAB=Block requires BufferLoad")
        return False
    if state["DirectToVgprA"] or state["DirectToLdsA"]:
        reject(state, printRejectionReason,
               "UseScaleAB=Block is incompatible with DirectToVgprA/DirectToLdsA")
        return False
    if state["enableTDMA"] or state["TDMInst"]:
        reject(state, printRejectionReason,
               "UseScaleAB=Block is incompatible with TDM (no localWrite to hook)")
        return False
    if problemType["Sparse"]:
        reject(state, printRejectionReason, "UseScaleAB=Block is incompatible with Sparse")
        return False
    if problemType["SwizzleTensorA"]:
        reject(state, printRejectionReason, "UseScaleAB=Block is incompatible with SwizzleTensorA")
        return False
    if problemType["MXBlockA"] or problemType["MXBlockB"]:
        reject(state, printRejectionReason,
               "UseScaleAB=Block and MXBlockA/B are two different scaling paths")
        return False
    # The GSU conversion kernel declares ScaleA/ScaleB as ComputeDataType
    # pointers and applies them in the epilogue; a block scale is neither.
    if state["GlobalSplitU"] != 1:
        reject(state, printRejectionReason, "UseScaleAB=Block requires GlobalSplitU=1")
        return False

    # --- tail loop is a separate global-read path with no scale support yet ---
    if state["AssertSummationElementMultiple"] % state["DepthU"] != 0:
        reject(state, printRejectionReason,
               "UseScaleAB=Block has no tail-loop support yet; "
               "AssertSummationElementMultiple (%d) must be a multiple of DepthU (%d)"
               % (state["AssertSummationElementMultiple"], state["DepthU"]))
        return False

    # Accepted. Only now mutate the state: per-load scale offsets need a per-load
    # A offset VGPR to derive from, and the SGPR-offset optimization would
    # collapse those into one VGPR plus SGPR deltas.
    state["_UseSgprForGRO"] = 0
    return True
