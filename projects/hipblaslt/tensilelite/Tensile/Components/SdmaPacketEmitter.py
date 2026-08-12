# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter.
#
# Packet-DEPENDENT counterpart to SdmaRingEmitter (which is packet-INDEPENDENT
# ring plumbing): this module turns the all-to-all geometry (kernarg values +
# WG ids) into the 13-dword COPY_SUBWIN and 8-dword ATOMIC ADD_RTN_32 packet
# dword arrays, laid out in VGPRs; SdmaRingEmitter.emitPlacePacket then writes
# those dwords into the ring.
#
# The C++ structs / byte layout / minus-one + element-scaling conventions are
# the SAME ones frozen in client/src/SdmaPktSubwin.hpp and its golden-vector
# gtest (SdmaPktSubwin_test.cpp), validated byte-for-byte on MI355X.
#
# TWO surfaces, cross-checked against each other and against the golden vectors:
#   * encodeCopyDwords / encodeAtomicDwords -- pure-Python integer encoders that
#     reproduce the exact 13/8 dword vectors; the source of truth for what the
#     assembly must build.
#   * emitBuildCopyPacket / emitBuildAtomicPacket -- rocisa emitters that build
#     the same dwords at runtime in VGPRs from the runtime inputs (p, j, myRank,
#     M, N, nShard), mirroring the pure-Python encoders field for field.
#
# The live caller is GlobalWriteBatch._emitFusedA2ASdmaIssue.
#
# Packet geometry, per (peer p, token-tile j), with this card == myRank:
#   COPY_SUBWIN (bf16 elements, elementsize header = log2(2) = 1):
#     src  = D + (j*MT1)*ldd + p*nShard
#     dst  = peer_ptr[p] + recvOffset + (myRank*N + j*MT1)*nShard
#     src pitch = ldd  ;  dst pitch = nShard
#     rect X = nShard (feature, contiguous) ; rect Y = min(MT1, N - j*MT1) (token)
#   ATOMIC ADD_RTN_32 -> peer_ptr[p] + myRank*4, addend 1 (raise the dest flag).
#
# COORDINATES ARE FOLDED INTO THE BASE ADDRESSES: src_x/src_y/dst_x/dst_y are all
# emitted as a literal 0 and the whole offset is added into the 64-bit base
# instead (addr(x, y) = base + y*pitch*elem + x*elem, so this is the same byte
# address written two ways). This keeps the 14-bit coordinate fields from
# overflowing at large world size x N; see checkA2AFieldsFit for what remains
# field-encoded and its bit-width bounds.
################################################################################

from rocisa.container import vgpr, sgpr
from rocisa.code import Module
from rocisa.instruction import (
    VMovB32,
    SMulI32, SMulHIU32, SAddU32, SAddCU32, SAddU64, SSubU32, SMinU32,
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SOrB32,
)


# ---- COPY_SUBWIN header op/sub_op (mirror SdmaPktSubwin.hpp) ----------------
SDMA_OP_COPY_SUBWIN         = 1
SDMA_SUBOP_COPY_LINEAR_RECT = 4
COPY_PACKET_DWORDS          = 13

# ---- ATOMIC header op/operation (mirror SdmaPktSubwin.hpp) ------------------
# operation is a 7-bit index into the TC atomic op table (ADD_RTN_32 = 15,
# ADD_RTN_64 = 47); RTN means the op returns the pre-op value, which SDMA drops.
SDMA_OP_ATOMIC         = 10
SDMA_ATOMIC_ADD_RTN_32 = 15
ATOMIC_PACKET_DWORDS   = 8

# TWO DIFFERENT "ELEMENT SIZES". Conflating them is a silent 8x address error,
# so they are separate constants with separate jobs:
#
#   D_DATA_ELEMENT_LOG2 = log2(sizeof(bf16)). The width of one D element in
#     BYTES. It converts the element-unit geometry into the byte offset
#     folded into the base address. A byte offset is a byte offset at any
#     packet elementsize, so this NEVER changes with the one below.
#
#   PACKET_ELEMENT_SIZE_LOG2 = the packet's ADDRESSING GRANULARITY, the unit that
#     x, the pitches, the slice pitches and rect_x are counted in (header field
#     [31:29]). Widening it from 1 to 4 divides all of those field values by 8,
#     which is what buys the headroom: rect_x = nShard/8 lifts the AM ceiling
#     from 16384*W to 131072*W, and src_pitch = ldd/8 lifts the ldd ceiling from
#     524288 to 4194304. It does NOT touch y (a row index) or rect_y.
#
# 16-byte elements are supported by the engine: validated on MI355X (MED, W=4,
# recv byte-exact). The field is 3 bits wide, but "3 bits wide" is not evidence
# that the hardware accepts every encoding.
D_DATA_ELEMENT_LOG2      = 1
PACKET_ELEMENT_SIZE_LOG2 = 4

# How far to shift a bf16-element count down into packet-element units. Derived,
# never hardcoded: if either constant moves, every scaled field follows.
ELEMENT_SHIFT      = PACKET_ELEMENT_SIZE_LOG2 - D_DATA_ELEMENT_LOG2   # 3
ELEMENT_MULTIPLE   = 1 << ELEMENT_SHIFT                               # 8

# Back-compat alias: the pure-Python encoder's default and the hardware-backed
# golden vectors are bf16-granular, and stay that way (see encodeCopyDwords).
BF16_ELEMENT_SIZE_LOG2 = D_DATA_ELEMENT_LOG2

# Field widths (bits) shared by src/dst coordinate + rect dwords. These match
# the bit-fields declared in client/src/SdmaPktSubwin.hpp; kept here so the
# pure-Python encoder and the rocisa masks share one definition.
_XY_BITS    = 14   # src_x/src_y, dst_x/dst_y, rect_x/rect_y
_Z_BITS     = 11   # src_z/dst_z/rect_z
_PITCH_BITS = 19   # src_pitch/dst_pitch (start at bit 13, after z)
_SLICE_BITS = 28   # src/dst slice pitch


def _mask(bits):
    return (1 << bits) - 1


XY_FIELD_LIMIT    = 1 << _XY_BITS      # 16384; rect_x/rect_y (x/y are folded to 0)
PITCH_FIELD_LIMIT = 1 << _PITCH_BITS   # 524288; src_pitch/dst_pitch


def checkA2AFieldsFit(numRanks, nShard, macroTile1, srcPitch):
    """Raise ValueError if a fused-A2A geometry cannot be encoded safely.

    Python mirror of the field-fit guards in
    client/src/FusedA2AClient.cpp::runFusedA2A. Not called from codegen (W,
    nShard and ldd are runtime kernargs, unknown at codegen time); used only by
    this module's unit tests. THE TWO CAN DRIFT SILENTLY -- nothing links them,
    so keep them in sync by hand.

    Rank bound: the kernarg segment reserves exactly FUSED_A2A_MAX_RANKS
    peer_ptr slots (Signature.py), so ranks >= that have no pointer.

    src_x, src_y and dst_y are folded into the 64-bit base addresses (see
    emitComputeCopyFields) and are not checked here; N is unconstrained. What
    remains -- rect_x, rect_y, src_pitch, dst_pitch and the ELEMENT_SHIFT
    divisibility precondition -- is derived in the raise() messages below.
    """
    from .Signature import FUSED_A2A_MAX_RANKS
    if numRanks < 1 or numRanks > FUSED_A2A_MAX_RANKS:
        raise ValueError(
            "fused-A2A world size W=%d is out of range: the kernarg segment "
            "reserves exactly FUSED_A2A_MAX_RANKS=%d peer_ptr slots, so "
            "ranks >= %d have no pointer and a PUSH to them reads garbage."
            % (numRanks, FUSED_A2A_MAX_RANKS, FUSED_A2A_MAX_RANKS))
    if ELEMENT_SHIFT and (nShard % ELEMENT_MULTIPLE or srcPitch % ELEMENT_MULTIPLE):
        raise ValueError(
            "fused-A2A geometry is not addressable at the packet's %d-byte "
            "element: nShard=%d and ldd=%d must both be multiples of %d. The "
            "emitter scales them by >>%d, which would TRUNCATE a non-multiple "
            "and silently copy a short band."
            % (1 << PACKET_ELEMENT_SIZE_LOG2, nShard, srcPitch,
               ELEMENT_MULTIPLE, ELEMENT_SHIFT))
    # `>=` deliberately: the rect extents are minus-one encoded so a field value
    # of exactly 16384 would in fact encode, but one lost value is worth keeping
    # the terms uniform and identical to the C++ guard.
    rectXField = nShard >> ELEMENT_SHIFT
    if max(rectXField, macroTile1) >= XY_FIELD_LIMIT:
        raise ValueError(
            "fused-A2A geometry overflows the SDMA packet's %d-bit rect fields: "
            "W=%d nShard=%d MT1=%d -> rect_x=nShard>>%d=%d, max rect_y=%d; both "
            "must be < %d. The emitter packs these unmasked, so the copy would "
            "silently move the wrong band. rect_x is the X extent (AM/W) -- it "
            "cannot be folded into the base address the way the coordinates "
            "were; reduce AM or raise W (the bound is AM < %d*W)."
            % (_XY_BITS, numRanks, nShard, macroTile1, ELEMENT_SHIFT,
               rectXField, macroTile1, XY_FIELD_LIMIT,
               XY_FIELD_LIMIT << ELEMENT_SHIFT))
    pitchField = srcPitch >> ELEMENT_SHIFT
    if pitchField >= PITCH_FIELD_LIMIT:
        raise ValueError(
            "fused-A2A src_pitch overflows the SDMA packet's %d-bit pitch "
            "field: ldd=%d -> ldd>>%d=%d must be < %d (i.e. ldd < %d). "
            "_packPitchMinus1 shifts it left by 13 unmasked, so an over-range "
            "pitch ORs into the neighbouring field."
            % (_PITCH_BITS, srcPitch, ELEMENT_SHIFT, pitchField,
               PITCH_FIELD_LIMIT, PITCH_FIELD_LIMIT << ELEMENT_SHIFT))


# ---------------------------------------------------------------------------
# Pure-Python encoders (golden-vector source of truth; no rocisa).
# ---------------------------------------------------------------------------

def encodeCopyDwords(srcBase, srcX, srcY, srcPitch, srcSlicePitch,
                     dstBase, dstX, dstY, dstPitch, dstSlicePitch,
                     rectX, rectY, elementSizeLog2=BF16_ELEMENT_SIZE_LOG2):
    """Return the 13 dwords of a COPY_LINEAR_SUBWIN packet, encoding the two
    conventions the whole route depends on: every extent/pitch is stored MINUS
    ONE, and all coords/extents/pitches are in ELEMENTS (element size carried in
    the header). This is the reference the rocisa emitter must reproduce, pinned
    by test_sdma_packet_emitter.py to golden dwords with MI355X backing.

    Bit positions are transcribed from AMD OSS 4.4 sdma.pkt field positions,
    cross-checked against ROCR's sdma_registers.h and the kernel's
    vega10_sdma_pkt_open.h -- all three agree. The layout was then validated
    byte-for-byte on MI355X (24 packets, every dword bit-accurate).

    GFX12+ uses a DIFFERENT layout of the same size: this encoder is gfx9xx /
    gfx95x ONLY. Do not reorder fields or "clean up" the reserved gaps -- the
    minus-one convention and the <<13 pitch placement look arbitrary because
    they are hardware-mandated, not derivable."""
    dw = [0] * COPY_PACKET_DWORDS
    dw[0] = ((SDMA_OP_COPY_SUBWIN & 0xFF)
             | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8)
             | ((elementSizeLog2 & 0x7) << 29))
    dw[1] = srcBase & 0xFFFFFFFF
    dw[2] = (srcBase >> 32) & 0xFFFFFFFF
    dw[3] = (srcX & _mask(_XY_BITS)) | ((srcY & _mask(_XY_BITS)) << 16)
    dw[4] = (0 & _mask(_Z_BITS)) | (((srcPitch - 1) & _mask(_PITCH_BITS)) << 13)
    dw[5] = (srcSlicePitch - 1) & _mask(_SLICE_BITS)
    dw[6] = dstBase & 0xFFFFFFFF
    dw[7] = (dstBase >> 32) & 0xFFFFFFFF
    dw[8] = (dstX & _mask(_XY_BITS)) | ((dstY & _mask(_XY_BITS)) << 16)
    dw[9] = (0 & _mask(_Z_BITS)) | (((dstPitch - 1) & _mask(_PITCH_BITS)) << 13)
    dw[10] = (dstSlicePitch - 1) & _mask(_SLICE_BITS)
    dw[11] = ((rectX - 1) & _mask(_XY_BITS)) | (((rectY - 1) & _mask(_XY_BITS)) << 16)
    dw[12] = 0  # rect_z=0 (one plane) + default swizzle/cache policy
    return dw


def encodeAtomicDwords(dstAddr, addend=1):
    """Return the 8 dwords of an ADD_RTN_32 fetch-add ATOMIC packet (MORI
    CreateAtomicIncPacket form): op=ATOMIC, operation=ADD_RTN_32, ADDR=dstAddr,
    SRC_DATA=addend; compare + loop dwords stay zero. Mirrors
    makeAtomicAdd32Packet in SdmaPktSubwin.hpp."""
    dw = [0] * ATOMIC_PACKET_DWORDS
    dw[0] = ((SDMA_OP_ATOMIC & 0xFF)
             | ((SDMA_ATOMIC_ADD_RTN_32 & 0x7F) << 25))  # l bit (16) stays 0 (fetch-add)
    dw[1] = dstAddr & 0xFFFFFFFF
    dw[2] = (dstAddr >> 32) & 0xFFFFFFFF
    dw[3] = addend & 0xFFFFFFFF
    # dw[4] = 0 (src_data hi, 64-bit ops only).
    # dw[5..7] = 0 (cmp_data lo/hi, loop_interval): unused for a plain fetch-add.
    return dw


# Compile-time header dwords (op/sub_op/elementsize are all immediates), shared
# by the pure-Python encoder and the rocisa emitter so the two cannot drift.
COPY_HEADER_DW0 = ((SDMA_OP_COPY_SUBWIN & 0xFF)
                   | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8)
                   | ((PACKET_ELEMENT_SIZE_LOG2 & 0x7) << 29))
ATOMIC_HEADER_DW0 = ((SDMA_OP_ATOMIC & 0xFF) | ((SDMA_ATOMIC_ADD_RTN_32 & 0x7F) << 25))


class SdmaPacketEmitter:
    """Builds the COPY_SUBWIN + ATOMIC packet dword arrays in VGPRs from runtime
    inputs, matching client/src/SdmaPktSubwin.hpp field for field.

    Stateless like SdmaRingEmitter: every method takes the registers it uses
    (caller owns the pools) plus a `w` context exposing `.sgprPool`/`.vgprPool`.
    The dword VGPR block it fills is what SdmaRingEmitter.emitPlacePacket writes
    to the ring, so the two emitters compose without either knowing the other's
    internals.

    The three encoding conventions (minus-one extents/pitches, element units,
    field bit positions) are isolated in the small `_pack*` helpers below so a
    future encoding change touches one place -- the same discipline
    SdmaRingEmitter uses for its CAS primitive.
    """

    def __init__(self, macroTile1: int, elementSizeLog2: int = PACKET_ELEMENT_SIZE_LOG2):
        # MT1 (token extent / rect_y) and the element-size header are compile-time
        # solution constants; the geometric fields (p, j, myRank, M, N, nShard)
        # are runtime SGPRs.
        self.mt1 = macroTile1
        self.elementSizeLog2 = elementSizeLog2

    # ---- field-packing helpers (isolate the encoding conventions) -----------

    def _toPacketElements(self, module, dstS, srcS, comment):
        """Convert a bf16-element count into packet-element units (>> 3).

        Applies to X-DIRECTION quantities ONLY: the pitches, the slice pitches
        and rect_x. It must NOT be applied to y or rect_y, which are row indices
        the hardware does not scale by ELEMENTSIZE, nor to the folded base
        addresses, which are byte offsets.

        Emitted even when the shift is 0 would be wasteful, so the callers skip
        the whole helper in that case -- keeping the bf16-granular encoding
        byte-identical to what it was if PACKET_ELEMENT_SIZE_LOG2 is ever wound
        back. Divisibility (every scaled quantity a multiple of ELEMENT_MULTIPLE)
        is a launch-time precondition, enforced in FusedA2AClient.cpp and mirrored
        by checkA2AFieldsFit; a non-multiple would truncate here and shrink the
        copy."""
        module.add(SLShiftRightB32(dst=sgpr(dstS), src=sgpr(srcS),
                                   shiftHex=ELEMENT_SHIFT,
                                   comment=comment + " (bf16 elems -> packet elems)"))
        return module

    def _packPitchMinus1(self, module, dstV, pitchS, tmpS, comment):
        """dword = ((pitch - 1) & 0x7FFFF) << 13, z field ([10:0]) left 0.
        Minus-one is the hardware pitch convention (it adds one back). The pitch
        arrives in bf16 elements and is scaled to packet elements first."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, tmpS, pitchS, comment)
            src = tmpS
        else:
            src = pitchS
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(src), src1=1,
                           comment=comment + " (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(tmpS), shiftHex=13,
                                  comment=comment + " (<< 13)"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _packSliceMinus1(self, module, dstV, sliceS, tmpS, comment):
        """dword = (slice_pitch - 1) & 0x0FFFFFFF, at bit 0 (28-bit field).
        Scaled to packet elements like the pitches."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, tmpS, sliceS, comment)
            src = tmpS
        else:
            src = sliceS
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(src), src1=1,
                           comment=comment + " (slice - 1)"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _packRectMinus1(self, module, dstV, rectXS, rectYS, tmpS, comment):
        """dword = ((rectX - 1) & 0x3FFF) | (((rectY - 1) & 0x3FFF) << 16).
        BOTH extents are runtime SGPRs. rectY cannot be the compile-time MT1: the
        last token-tile is partial whenever N is not a multiple of MT1, and an
        unclamped MT1 would make the engine read past the end of D (the recv side
        has room -- it is allocated to ceil(N/MT1)*MT1 -- but the source does not).
        emitComputeCopyFields clamps it to min(MT1, N - j*MT1). tmpS is TWO
        consecutive scratch SGPRs (tmpS, tmpS+1).

        rect_x IS scaled to packet elements; rect_y is NOT. rect_y counts ROWS,
        and ELEMENTSIZE scales only the X direction -- scaling it would shorten
        the copy to an eighth of the band."""
        if ELEMENT_SHIFT:
            self._toPacketElements(module, tmpS, rectXS, comment + " (rectX)")
            rectXsrc = tmpS
        else:
            rectXsrc = rectXS
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(rectXsrc), src1=1,
                           comment=comment + " (rectX - 1)"))
        module.add(SSubU32(dst=sgpr(tmpS + 1), src0=sgpr(rectYS), src1=1,
                           comment=comment + " (rectY - 1, rows: NOT scaled)"))
        module.add(SLShiftLeftB32(dst=sgpr(tmpS + 1), src=sgpr(tmpS + 1), shiftHex=16,
                                  comment=comment + " ((rectY-1) << 16)"))
        module.add(SOrB32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(tmpS + 1),
                          comment=comment + " | (rectY-1) << 16"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _movImm(self, module, dstV, imm, comment):
        # Hex string, not int: rocisa renders an int above INT32_MAX as a float.
        module.add(VMovB32(dst=vgpr(dstV), src=hex(imm), comment=comment))
        return module

    def _movSgpr(self, module, dstV, srcS, comment):
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(srcS), comment=comment))
        return module

    # ---- COPY_SUBWIN builder -----------------------------------------------

    def emitBuildCopyPacket(self, module, w, pktV,
                            srcBaseS, srcPitchS, srcSliceS,
                            dstBaseS, dstPitchS, dstSliceS,
                            rectXS, rectYS, tmpS):
        """Build the 13 COPY_SUBWIN dwords into pktV[0:13] from runtime SGPR
        inputs (pitches and extents in element units; caller does the
        arithmetic that produces them -- see emitComputeCopyFields). tmpS is TWO
        consecutive scratch SGPRs (the rect dword packs two runtime extents).

        All four coordinates are ZERO: emitComputeCopyFields folded them into
        srcBaseS / dstBaseS, so DW3 and DW8 are literal-0 moves rather than
        field packing. They are still written (the ring copies a fixed 13-dword
        block, and a stale VGPR would be read as a coordinate).

        Field -> dword map (mirrors encodeCopyDwords / SdmaPktSubwin.hpp):
          DW0 header (immediate), DW1/2 srcBase, DW3 0, DW4 srcPitch-1,
          DW5 srcSlice-1, DW6/7 dstBase, DW8 0, DW9 dstPitch-1,
          DW10 dstSlice-1, DW11 (rectX-1|rectY-1), DW12 0.
        """
        self._movImm(module, pktV + 0, COPY_HEADER_DW0,
                     "SUBWIN DW0: op=COPY sub_op=RECT elementsize=log2(%dB)"
                     % (1 << PACKET_ELEMENT_SIZE_LOG2))
        self._movSgpr(module, pktV + 1, srcBaseS + 0, "SUBWIN DW1: srcBase lo")
        self._movSgpr(module, pktV + 2, srcBaseS + 1, "SUBWIN DW2: srcBase hi")
        self._movImm(module, pktV + 3, 0, "SUBWIN DW3: src_x=0|src_y=0 (folded into srcBase)")
        self._packPitchMinus1(module, pktV + 4, srcPitchS, tmpS, "SUBWIN DW4: src_pitch-1")
        self._packSliceMinus1(module, pktV + 5, srcSliceS, tmpS, "SUBWIN DW5: src_slice-1")
        self._movSgpr(module, pktV + 6, dstBaseS + 0, "SUBWIN DW6: dstBase lo")
        self._movSgpr(module, pktV + 7, dstBaseS + 1, "SUBWIN DW7: dstBase hi")
        self._movImm(module, pktV + 8, 0, "SUBWIN DW8: dst_x=0|dst_y=0 (folded into dstBase)")
        self._packPitchMinus1(module, pktV + 9, dstPitchS, tmpS, "SUBWIN DW9: dst_pitch-1")
        self._packSliceMinus1(module, pktV + 10, dstSliceS, tmpS, "SUBWIN DW10: dst_slice-1")
        self._packRectMinus1(module, pktV + 11, rectXS, rectYS, tmpS,
                             "SUBWIN DW11: rect_x-1|rect_y-1")
        self._movImm(module, pktV + 12, 0, "SUBWIN DW12: rect_z=0, default cache/swizzle")
        return module

    # ---- ATOMIC ADD_RTN_32 builder ------------------------------------------

    def emitBuildAtomicPacket(self, module, w, pktV, dstAddrS, addend=1):
        """Build the 8 ATOMIC ADD_RTN_32 dwords into pktV[0:8]: raise peer_ptr[p]
        [myRank] by `addend` (== 1). dstAddrS is a 2-SGPR pointer to the flag
        slot (caller computes peer_ptr[p] + myRank*4 -- see emitComputeFlagAddr;
        the stride is 4 because this ADD_RTN_32 writes 4 bytes). Mirrors
        encodeAtomicDwords / makeAtomicAdd32Packet. addend is a compile-time
        immediate (1)."""
        self._movImm(module, pktV + 0, ATOMIC_HEADER_DW0,
                     "ATOMIC DW0: op=ATOMIC operation=ADD_RTN_32")
        self._movSgpr(module, pktV + 1, dstAddrS + 0, "ATOMIC DW1: addr lo")
        self._movSgpr(module, pktV + 2, dstAddrS + 1, "ATOMIC DW2: addr hi")
        self._movImm(module, pktV + 3, addend & 0xFFFFFFFF, "ATOMIC DW3: src_data lo (addend)")
        self._movImm(module, pktV + 4, 0, "ATOMIC DW4: src_data hi (unused by ADD_RTN_32)")
        self._movImm(module, pktV + 5, 0, "ATOMIC DW5: cmp_data lo (unused)")
        self._movImm(module, pktV + 6, 0, "ATOMIC DW6: cmp_data hi (unused)")
        self._movImm(module, pktV + 7, 0, "ATOMIC DW7: loop_interval=0")
        return module

    def _mulU32toU64(self, module, dstS, aS, bS, comment):
        """dstS[0:1] (64-bit) = aS * bS, both operands unsigned 32-bit.

        Emitted as the bare s_mul_hi_u32 / s_mul_i32 pair rather than through
        KernelWriterAssembly.s_mul_u64_u32 on purpose. That wrapper picks between
        s_mul_hi_u32 and a VALU fallback using asmCaps["HasSMulHi"], which is a
        LIVE ASSEMBLER PROBE (rocisa hardware_caps.hpp) -- so the instruction
        sequence, and therefore the handshake golden, would depend on the machine
        that regenerated it. The COPY_SUBWIN encoder in this file is gfx9xx/gfx95x
        only (see encodeCopyDwords), and s_mul_hi_u32 is unconditional across that
        range, so there is nothing to select between here.

        UNSIGNED on purpose: the operands are extents and strides. s_mul_hi_i32
        would read a stride with bit 31 set as negative and corrupt the high word.
        """
        module.add(SMulHIU32(dst=sgpr(dstS + 1), src0=sgpr(aS), src1=sgpr(bS),
                             comment=comment + " (hi)"))
        module.add(SMulI32(dst=sgpr(dstS + 0), src0=sgpr(aS), src1=sgpr(bS),
                           comment=comment + " (lo)"))
        return module

    # ---- field arithmetic (runtime geometry -> the SGPR inputs above) --

    def emitComputeCopyFields(self, module, w,
                              pS, jS, myRankS, mS, nS, nShardS,
                              addressDS, srcPitchS, recvBaseS,
                              outSrcBaseS, outSrcYS, outSrcSliceS,
                              outDstSliceS, outRectYS, tmpS, tmp64S):
        """Compute the runtime COPY inputs from (p, j, myRank, M, N, nShard),
        FOLDING the four coordinates into the two 64-bit base addresses:

          outSrcBase = AddressD + (j*MT1*ldd    + p*nShard) * sizeof(bf16)
          recvBase  += (myRank*N + j*MT1) * nShard          * sizeof(bf16)
          src_slice  = M * N                 (single-plane slice pitch, don't-care)
          dst_slice  = MT1 * nShard          (one band's plane)
          rect_y     = min(MT1, N - j*MT1)   (clamped: last token-tile is partial)

        The hardware addresses a sub-window as base + y*pitch*elem + x*elem, so
        adding those terms into the base and leaving x/y at 0 reaches the exact
        same byte -- see the module docstring for why we want that (the 14-bit
        coordinate fields would overflow at large W).

        src_pitch = ldd and dst_pitch = nShard are passed straight through by the
        caller, as is rect_x = nShard. MT1 is the compile-time token extent.

        BOTH folds are 64-BIT and must stay that way. Neither product is bounded
        by anything now: j*MT1*ldd grows with N (unconstrained since the fold) and
        ldd (19-bit), and (myRank*N + j*MT1)*nShard grows with W, N and nShard.
        At W=8/N=65536/nShard=16383 the dst term alone is 8.6e9. See _mulU32toU64
        for why the widening multiply is emitted bare and unsigned.

        The elements->bytes shift uses D_DATA_ELEMENT_LOG2, NOT self.elementSizeLog2:
        a byte offset does not scale with the packet's addressing granularity. See
        the constant's comment.

        (a+b)<<k == (a<<k)+(b<<k), so the src side shifts the SUM once rather than
        each term -- one s_lshl_b64 instead of two.

        outSrcYS still holds j*MT1 on return: it is read three times (the src fold,
        the dst_y precursor, and the rect_y clamp below), so it cannot be scratch.
        recvBaseS is updated IN PLACE; addressDS is read-only (it is the persistent
        AddressD pair and must survive). tmpS is one scratch SGPR; tmp64S is a
        2-ALIGNED scratch pair (the 64-bit ops need SReg_64 alignment) and is dead
        on return.

        rect_y is clamped rather than left at MT1 because N need not be a multiple
        of MT1: the tail tile then covers only N - j*MT1 tokens, and copying a full
        MT1 rows would read past the end of D.

        Kept separate from emitBuildCopyPacket so the address arithmetic (what the
        formulas mean) and the bit-packing (how the hardware wants them) are
        each auditable on their own.
        """
        module.add(SMulI32(dst=sgpr(outSrcYS), src0=sgpr(jS), src1=self.mt1,
                           comment="src_y = j * MT1 (folded into the base, not a field)"))
        module.add(SMulI32(dst=sgpr(outSrcSliceS), src0=sgpr(mS), src1=sgpr(nS),
                           comment="src_slice = M * N (single-plane, don't-care)"))

        # --- src fold: AddressD + (j*MT1*ldd + p*nShard) * sizeof(bf16) ---
        module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr(pS), src1=sgpr(nShardS),
                           comment="src_x = p * nShard (folded into the base, not a field)"))
        self._mulU32toU64(module, tmp64S, outSrcYS, srcPitchS,
                          "src row offset = j*MT1 * ldd (64-bit: unbounded in N and ldd)")
        module.add(SAddU32(dst=sgpr(tmp64S + 0), src0=sgpr(tmp64S + 0), src1=sgpr(tmpS),
                           comment="+ p*nShard (feature offset)"))
        module.add(SAddCU32(dst=sgpr(tmp64S + 1), src0=sgpr(tmp64S + 1), src1=0,
                            comment="propagate carry into the high word"))
        module.add(SLShiftLeftB64(dst=sgpr(tmp64S, 2), src=sgpr(tmp64S, 2),
                                  shiftHex=D_DATA_ELEMENT_LOG2,
                                  comment="src offset: elements -> bytes (sizeof(bf16))"))
        module.add(SAddU64(dst=sgpr(outSrcBaseS, 2), src0=sgpr(addressDS, 2),
                           src1=sgpr(tmp64S, 2),
                           comment="srcBase = D + src offset (src_x/src_y now 0)"))

        # --- dst fold: recvBase += (myRank*N + j*MT1) * nShard * sizeof(bf16) ---
        module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr(myRankS), src1=sgpr(nS),
                           comment="myRank * N"))
        module.add(SAddU32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(outSrcYS),
                           comment="dst row = myRank*N + j*MT1 (folded, not a field)"))
        self._mulU32toU64(module, tmp64S, tmpS, nShardS,
                          "dst row offset = dst row * nShard (64-bit: unbounded in W and N)")
        module.add(SLShiftLeftB64(dst=sgpr(tmp64S, 2), src=sgpr(tmp64S, 2),
                                  shiftHex=D_DATA_ELEMENT_LOG2,
                                  comment="dst offset: elements -> bytes (sizeof(bf16))"))
        module.add(SAddU64(dst=sgpr(recvBaseS, 2), src0=sgpr(recvBaseS, 2),
                           src1=sgpr(tmp64S, 2),
                           comment="dstBase = recv slot + dst offset (dst_x/dst_y now 0)"))

        module.add(SMulI32(dst=sgpr(outDstSliceS), src0=sgpr(nShardS), src1=self.mt1,
                           comment="dst_slice = MT1 * nShard (one band's plane)"))
        # rect_y = min(MT1, N - j*MT1): the tail token-tile is partial when
        # N % MT1 != 0; an unclamped MT1 would read past the end of D.
        module.add(SSubU32(dst=sgpr(outRectYS), src0=sgpr(nS), src1=sgpr(outSrcYS),
                           comment="N - j*MT1 (tokens left in this tile)"))
        module.add(SMinU32(dst=sgpr(outRectYS), src0=sgpr(outRectYS), src1=self.mt1,
                           comment="rect_y = min(MT1, N - j*MT1) (clamp tail tile)"))
        return module

    def emitComputeFlagAddr(self, module, w, flagBaseS, myRankS, outAddrS, tmpS):
        """Compute the ATOMIC target peer_ptr[p] + myRank*4 into outAddrS (2
        SGPRs), a 64-bit add. flagBaseS is peer_ptr[p] (already selected by the
        caller via _fusedA2ALoadFlagBaseByRank). tmpS is one scratch SGPR.

        Stride is 4: the ATOMIC is an ADD_RTN_32, a 4-byte write.

        The flag is indexed by SOURCE rank only -- source j's tokenTiles ATOMICs
        accumulate into one slot, matching the "== tokenTiles" drain predicate.

        Three things must move with this stride: the host allocates flag as W u32
        slots (FusedA2AClient.cpp flagBytes), the DRAIN poll strides its self-flag
        address by 4, and that poll compares against tokenTiles (an accumulated
        count) rather than a one-shot sentinel -- the SDMA ATOMIC adds, it does
        not store.
        """
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(myRankS), shiftHex=2,
                                  comment="myRank * 4 (u32 flag-slot byte offset: the ATOMIC is an ADD_RTN_32)"))
        module.add(SAddU32(dst=sgpr(outAddrS + 0), src0=sgpr(flagBaseS + 0), src1=sgpr(tmpS),
                           comment="flag addr lo = peer_ptr[p] + myRank*4"))
        module.add(SAddCU32(dst=sgpr(outAddrS + 1), src0=sgpr(flagBaseS + 1), src1=0,
                            comment="flag addr hi (carry)"))
        return module
