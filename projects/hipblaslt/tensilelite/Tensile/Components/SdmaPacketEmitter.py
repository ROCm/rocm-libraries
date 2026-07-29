# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter (ROCM-27524, SDMA offload codegen Task 5).
#
# Packet-DEPENDENT counterpart to Task 4's SdmaRingEmitter (which is packet-
# INDEPENDENT ring plumbing). This module turns the §1.3 all-to-all geometry
# (kernarg values + WG ids) into the 13-dword COPY_SUBWIN and 8-dword ATOMIC
# ADD64 packet dword arrays, laid out in VGPRs; Task 4's emitPlacePacket then
# writes those dwords into the ring. It is deliberately split from the ring
# emitter: ring reserve/place/submit is packet-agnostic, packet field encoding
# is packet-specific -- two responsibilities, two files (SdmaRingEmitter.py was
# already 505 lines and its reviewer flagged that adding packet logic there
# would overload it).
#
# The C++ structs / byte layout / minus-one + element-scaling conventions are
# the SAME ones frozen in client/src/SdmaPktSubwin.hpp and its golden-vector
# gtest (SdmaPktSubwin_test.cpp). The COPY_SUBWIN encoding was validated
# byte-for-byte on MI355X; the ATOMIC encoding is MORI's production struct but
# is NOT yet hardware-verified (its first real run is Task 7/8).
#
# TWO surfaces, cross-checked against each other and against the T1 golden:
#   * encodeCopyDwords / encodeAtomicDwords -- pure-Python integer encoders that
#     reproduce the exact 13/8 dword vectors. Used by the unit test to prove the
#     field math matches the C++ golden bit-for-bit, and (conceptually) the ONE
#     source of truth for what the assembly must build.
#   * emitBuildCopyPacket / emitBuildAtomicPacket -- rocisa emitters that build
#     the same dwords at runtime in VGPRs from the runtime inputs (p, j, myRank,
#     M, N, nShard). They mirror the pure-Python encoders field for field.
#
# This round emits nothing into a live kernel (Task 6/7 wire it in); it is
# verified by (a) immediate cross-check vs the T1 golden, (b) rendering each
# emitter's Module and running it through the gfx950 assembler, and (c)
# structural asserts on the field-packing instruction sequence.
#
# §1.3 packet geometry, per (peer p, token-tile j), with this card == myRank:
#   COPY_SUBWIN (bf16 elements, elementsize header = log2(2) = 1):
#     src  = D + (j*MT1)*M + p*nShard      -> base=D, src_x=p*nShard, src_y=j*MT1
#     dst  = recv_ptr[p] + myRank*N*nShard + (j*MT1)*nShard
#            -> base=recv_ptr[p], dst_x=0, dst_y=myRank*N + j*MT1
#     src pitch = M (18432)  ;  dst pitch = nShard (2560, unpadded production)
#     rect X = nShard (feature, contiguous) ; rect Y = MT1 (token)
#   ATOMIC ADD64 -> flag_ptr[p] + myRank*4, addend 1 (raise the dest flag).
#
# Coordinate form (base + src_x/src_y) is used rather than folding the whole
# offset into the base address: it is exactly the form the MI355X golden
# validated, keeps every field in the same units the hardware documents, and
# lets the same encoder cover the harness (padded dst pitch) and production
# (unpadded) shapes by changing only the pitch argument.
################################################################################

from rocisa.container import vgpr, sgpr
from rocisa.code import Module
from rocisa.instruction import (
    VMovB32,
    SMulI32, SAddU32, SAddCU32, SSubU32,
    SLShiftLeftB32, SOrB32,
)


# ---- COPY_SUBWIN header op/sub_op (mirror SdmaPktSubwin.hpp) ----------------
SDMA_OP_COPY_SUBWIN         = 1
SDMA_SUBOP_COPY_LINEAR_RECT = 4
COPY_PACKET_DWORDS          = 13

# ---- ATOMIC header op/operation (mirror SdmaPktSubwin.hpp) ------------------
SDMA_OP_ATOMIC     = 10
SDMA_ATOMIC_ADD64  = 47
ATOMIC_PACKET_DWORDS = 8

# bf16 destination: 2-byte elements -> elementsize header field = log2(2) = 1.
BF16_ELEMENT_SIZE_LOG2 = 1

# Field widths (bits) shared by src/dst coordinate + rect dwords. These match
# the bit-fields declared in client/src/SdmaPktSubwin.hpp; kept here so the
# pure-Python encoder and the rocisa masks share one definition.
_XY_BITS    = 14   # src_x/src_y, dst_x/dst_y, rect_x/rect_y
_Z_BITS     = 11   # src_z/dst_z/rect_z
_PITCH_BITS = 19   # src_pitch/dst_pitch (start at bit 13, after z)
_SLICE_BITS = 28   # src/dst slice pitch


def _mask(bits):
    return (1 << bits) - 1


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
    """Return the 8 dwords of an ADD64 fetch-add ATOMIC packet (MORI
    CreateAtomicIncPacket form): op=ATOMIC, operation=ADD64, ADDR=dstAddr,
    SRC_DATA=addend; compare + loop dwords stay zero. Mirrors
    makeAtomicAdd64Packet in SdmaPktSubwin.hpp."""
    dw = [0] * ATOMIC_PACKET_DWORDS
    dw[0] = ((SDMA_OP_ATOMIC & 0xFF)
             | ((SDMA_ATOMIC_ADD64 & 0x7F) << 25))  # l bit (16) stays 0 (fetch-add)
    dw[1] = dstAddr & 0xFFFFFFFF
    dw[2] = (dstAddr >> 32) & 0xFFFFFFFF
    dw[3] = addend & 0xFFFFFFFF
    dw[4] = (addend >> 32) & 0xFFFFFFFF
    # dw[5..7] = 0 (cmp_data lo/hi, loop_interval): unused for a plain fetch-add.
    return dw


# Compile-time header dwords (op/sub_op/elementsize are all immediates), shared
# by the pure-Python encoder and the rocisa emitter so the two cannot drift.
COPY_HEADER_DW0 = ((SDMA_OP_COPY_SUBWIN & 0xFF)
                   | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8)
                   | ((BF16_ELEMENT_SIZE_LOG2 & 0x7) << 29))
ATOMIC_HEADER_DW0 = ((SDMA_OP_ATOMIC & 0xFF) | ((SDMA_ATOMIC_ADD64 & 0x7F) << 25))


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
    future encoding change touches one place -- the same discipline Task 4 used
    for its CAS primitive.
    """

    def __init__(self, macroTile1: int, elementSizeLog2: int = BF16_ELEMENT_SIZE_LOG2):
        # MT1 (token extent / rect_y) and the element-size header are compile-time
        # solution constants; the geometric fields (p, j, myRank, M, N, nShard)
        # are runtime SGPRs.
        self.mt1 = macroTile1
        self.elementSizeLog2 = elementSizeLog2

    # ---- field-packing helpers (isolate the encoding conventions) -----------

    def _packXY(self, module, dstV, xS, yS, tmpS, comment):
        """dword = (x & 0x3FFF) | ((y & 0x3FFF) << 16), from two SGPR inputs.
        The x/y here are already in element units and are NOT minus-one encoded
        (only pitches and rect extents are). x/y come in < 2^14 by construction
        (§1.2: dst_y max = myRank*N + j*MT1 <= 3*2048+1792 = 7936), so no mask
        instruction is needed -- the field simply occupies [13:0] and [29:16]."""
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(yS), shiftHex=16,
                                  comment=comment + " (y << 16)"))
        module.add(SOrB32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(xS),
                          comment=comment + " | x"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _packPitchMinus1(self, module, dstV, pitchS, tmpS, comment):
        """dword = ((pitch - 1) & 0x7FFFF) << 13, z field ([10:0]) left 0.
        Minus-one is the hardware pitch convention (it adds one back)."""
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(pitchS), src1=1,
                           comment=comment + " (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(tmpS), shiftHex=13,
                                  comment=comment + " (<< 13)"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _packSliceMinus1(self, module, dstV, sliceS, tmpS, comment):
        """dword = (slice_pitch - 1) & 0x0FFFFFFF, at bit 0 (28-bit field)."""
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(sliceS), src1=1,
                           comment=comment + " (slice - 1)"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _packRectMinus1(self, module, dstV, rectXS, rectYimm, tmpS, comment):
        """dword = ((rectX - 1) & 0x3FFF) | (((rectY - 1) & 0x3FFF) << 16).
        rectX (== nShard) is runtime; rectY (== MT1) is a compile-time immediate,
        so its minus-one is folded at codegen time."""
        module.add(SSubU32(dst=sgpr(tmpS), src0=sgpr(rectXS), src1=1,
                           comment=comment + " (rectX - 1)"))
        rectYm1 = (rectYimm - 1) & _mask(_XY_BITS)
        module.add(SOrB32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=(rectYm1 << 16),
                          comment=comment + " | (rectY-1) << 16"))
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(tmpS), comment=comment))
        return module

    def _movImm(self, module, dstV, imm, comment):
        module.add(VMovB32(dst=vgpr(dstV), src=imm, comment=comment))
        return module

    def _movSgpr(self, module, dstV, srcS, comment):
        module.add(VMovB32(dst=vgpr(dstV), src=sgpr(srcS), comment=comment))
        return module

    # ---- COPY_SUBWIN builder -----------------------------------------------

    def emitBuildCopyPacket(self, module, w, pktV,
                            srcBaseS, srcXS, srcYS, srcPitchS, srcSliceS,
                            dstBaseS, dstYS, dstPitchS, dstSliceS,
                            rectXS, tmpS):
        """Build the 13 COPY_SUBWIN dwords into pktV[0:13] from runtime SGPR
        inputs (all in element units; caller does the §1.3 arithmetic that
        produces them -- see emitComputeCopyFields). dst_x is always 0 (the recv
        slot base already points at the shard's first feature), so it is not an
        argument. tmpS is one scratch SGPR.

        Field -> dword map (mirrors encodeCopyDwords / SdmaPktSubwin.hpp):
          DW0 header (immediate), DW1/2 srcBase, DW3 (srcX|srcY), DW4 srcPitch-1,
          DW5 srcSlice-1, DW6/7 dstBase, DW8 (0|dstY), DW9 dstPitch-1,
          DW10 dstSlice-1, DW11 (rectX-1|rectY-1), DW12 0.
        """
        self._movImm(module, pktV + 0, COPY_HEADER_DW0,
                     "SUBWIN DW0: op=COPY sub_op=RECT elementsize=bf16")
        self._movSgpr(module, pktV + 1, srcBaseS + 0, "SUBWIN DW1: srcBase lo")
        self._movSgpr(module, pktV + 2, srcBaseS + 1, "SUBWIN DW2: srcBase hi")
        self._packXY(module, pktV + 3, srcXS, srcYS, tmpS, "SUBWIN DW3: src_x|src_y")
        self._packPitchMinus1(module, pktV + 4, srcPitchS, tmpS, "SUBWIN DW4: src_pitch-1")
        self._packSliceMinus1(module, pktV + 5, srcSliceS, tmpS, "SUBWIN DW5: src_slice-1")
        self._movSgpr(module, pktV + 6, dstBaseS + 0, "SUBWIN DW6: dstBase lo")
        self._movSgpr(module, pktV + 7, dstBaseS + 1, "SUBWIN DW7: dstBase hi")
        # DW8: dst_x==0, so the dword is just (dst_y << 16). Reuse _packXY with a
        # zero x would need a zero SGPR; instead shift dst_y directly.
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(dstYS), shiftHex=16,
                                  comment="SUBWIN DW8: dst_y << 16 (dst_x=0)"))
        self._movSgpr(module, pktV + 8, tmpS, "SUBWIN DW8: dst_x=0|dst_y")
        self._packPitchMinus1(module, pktV + 9, dstPitchS, tmpS, "SUBWIN DW9: dst_pitch-1")
        self._packSliceMinus1(module, pktV + 10, dstSliceS, tmpS, "SUBWIN DW10: dst_slice-1")
        self._packRectMinus1(module, pktV + 11, rectXS, self.mt1, tmpS,
                             "SUBWIN DW11: rect_x-1|rect_y-1")
        self._movImm(module, pktV + 12, 0, "SUBWIN DW12: rect_z=0, default cache/swizzle")
        return module

    # ---- ATOMIC ADD64 builder ----------------------------------------------

    def emitBuildAtomicPacket(self, module, w, pktV, dstAddrS, addend=1):
        """Build the 8 ATOMIC ADD64 dwords into pktV[0:8]: raise flag_ptr[p]
        [myRank] by `addend` (== 1). dstAddrS is a 2-SGPR pointer to the flag
        slot (caller computes flag_ptr[p] + myRank*4). Mirrors encodeAtomicDwords
        / makeAtomicAdd64Packet. addend is a compile-time immediate (1) so its
        hi dword is 0."""
        self._movImm(module, pktV + 0, ATOMIC_HEADER_DW0,
                     "ATOMIC DW0: op=ATOMIC operation=ADD64")
        self._movSgpr(module, pktV + 1, dstAddrS + 0, "ATOMIC DW1: addr lo")
        self._movSgpr(module, pktV + 2, dstAddrS + 1, "ATOMIC DW2: addr hi")
        self._movImm(module, pktV + 3, addend & 0xFFFFFFFF, "ATOMIC DW3: src_data lo (addend)")
        self._movImm(module, pktV + 4, (addend >> 32) & 0xFFFFFFFF, "ATOMIC DW4: src_data hi")
        self._movImm(module, pktV + 5, 0, "ATOMIC DW5: cmp_data lo (unused)")
        self._movImm(module, pktV + 6, 0, "ATOMIC DW6: cmp_data hi (unused)")
        self._movImm(module, pktV + 7, 0, "ATOMIC DW7: loop_interval=0")
        return module

    # ---- §1.3 field arithmetic (runtime geometry -> the SGPR inputs above) --

    def emitComputeCopyFields(self, module, w,
                              pS, jS, myRankS, mS, nS, nShardS,
                              outSrcXS, outSrcYS, outSrcSliceS,
                              outDstYS, outDstSliceS, tmpS):
        """Compute the runtime COPY fields from (p, j, myRank, M, N, nShard) per
        §1.3, all in element units:
          src_x        = p * nShard
          src_y        = j * MT1
          src_slice    = M * N                 (single-plane slice pitch, don't-care)
          dst_y        = myRank * N + j * MT1
          dst_slice    = MT1 * nShard          (one band's plane)
        src_pitch = M and dst_pitch = nShard are passed straight through by the
        caller (they ARE mS / nShardS), as is rect_x = nShard. MT1 is the
        compile-time token extent. tmpS is one scratch SGPR.

        Kept separate from emitBuildCopyPacket so the field arithmetic (what the
        §1.3 formulas mean) and the bit-packing (how the hardware wants them) are
        each auditable on their own.
        """
        module.add(SMulI32(dst=sgpr(outSrcXS), src0=sgpr(pS), src1=sgpr(nShardS),
                           comment="src_x = p * nShard"))
        module.add(SMulI32(dst=sgpr(outSrcYS), src0=sgpr(jS), src1=self.mt1,
                           comment="src_y = j * MT1"))
        module.add(SMulI32(dst=sgpr(outSrcSliceS), src0=sgpr(mS), src1=sgpr(nS),
                           comment="src_slice = M * N (single-plane, don't-care)"))
        # dst_y = myRank * N + j * MT1  (== myRank*N + src_y).
        module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr(myRankS), src1=sgpr(nS),
                           comment="myRank * N"))
        module.add(SAddU32(dst=sgpr(outDstYS), src0=sgpr(tmpS), src1=sgpr(outSrcYS),
                           comment="dst_y = myRank*N + j*MT1"))
        module.add(SMulI32(dst=sgpr(outDstSliceS), src0=sgpr(nShardS), src1=self.mt1,
                           comment="dst_slice = MT1 * nShard (one band's plane)"))
        return module

    def emitComputeFlagAddr(self, module, w, flagBaseS, myRankS, outAddrS, tmpS):
        """Compute the ATOMIC target flag_ptr[p] + myRank*8 into outAddrS (2
        SGPRs), a 64-bit add. flagBaseS is flag_ptr[p] (already selected by the
        caller via _fusedA2ALoadFlagBaseByRank). tmpS is one scratch SGPR.

        Stride is 8, NOT 4: the route raises a flag with an ADD64 (MORI ships
        ADD64 but no ADD32, so the atomic write is 8 bytes wide), so the flag
        buffer must be a u64 array with 8-byte slots. A u32 array + *4 stride
        would let myRank=3's 8-byte write run 4 bytes past the W*4-byte
        allocation -- a heap overrun, not merely a neighbor-slot clobber. This
        is the plan §1.3 corrected form (myRank*8; the earlier *4 was a plan
        defect, and §1.1's flag[myRank][j] was a typo -- the flag is indexed by
        SOURCE rank only, tokenTiles packets accumulating into one slot, per the
        §1.1 "== tokenTiles" drain predicate).

        DEPENDENCY -- T7 must change three things ATOMICALLY with wiring this in
        (changing any one alone breaks the currently-correct non-SDMA path, which
        still uses *4 + a READY sentinel):
          1. host: flagBytes = W * sizeof(uint64_t)   (FusedA2AClient.cpp)
          2. kernel: every flag address calc shift 2->3 (_emitFusedA2AHandshake /
             _fusedA2ALoadFlagBase*)
          3. DRAIN: poll "== tokenTiles" (an accumulated count) instead of the
             one-shot READY sentinel -- the SDMA ATOMIC ADDs, it does not store.
        This emitter is not yet wired into any kernel, so changing it here is
        inert until T7 lands the other two.
        """
        module.add(SLShiftLeftB32(dst=sgpr(tmpS), src=sgpr(myRankS), shiftHex=3,
                                  comment="myRank * 8 (u64 flag-slot byte offset; see T7 dependency)"))
        module.add(SAddU32(dst=sgpr(outAddrS + 0), src0=sgpr(flagBaseS + 0), src1=sgpr(tmpS),
                           comment="flag addr lo = flag_ptr[p] + myRank*8"))
        module.add(SAddCU32(dst=sgpr(outAddrS + 1), src0=sgpr(flagBaseS + 1), src1=0,
                            comment="flag addr hi (carry)"))
        return module
