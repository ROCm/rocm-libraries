# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA packet-construction emitter: builds the 13-dword COPY_SUBWIN and
# 8-dword ATOMIC ADD_RTN_32 dword arrays in SGPRs out of runtime field values.
# SdmaRingEmitter -- the packet-INDEPENDENT ring plumbing -- then writes them
# into the ring with s_store.  The dwords live in SGPRs because every input is
# wave-uniform, which is what lets the whole submit path avoid v_readfirstlane
# and VCC.
#
# THIS FILE KNOWS THE PACKET FORMAT AND NOTHING ABOUT WHAT IS BEING COPIED, so
# a second caller with a different data layout reuses it as-is.  The geometry
# that turns an all-to-all into these field values lives with its caller, in
# GlobalWriteBatch's _fusedA2A* helpers, and every X-direction input arrives
# ALREADY IN PACKET ELEMENTS -- the emitter does no unit conversion, so it need
# not know the caller's data type.  `packetElementLog2` is published for the
# caller to scale with.
#
# Encoding conventions, none of them derivable from the field names: every
# extent and pitch is stored MINUS ONE (the hardware adds it back); coords,
# extents and pitches are in ELEMENTS of the size named in the header, not in
# bytes; and the four coordinates are FOLDED INTO THE BASE ADDRESSES
# (addr(x, y) = base + y*pitch*elem + x*elem), so src_x/src_y/dst_x/dst_y are
# emitted as a literal 0.  The reserved gaps and the <<13 pitch placement look
# arbitrary because they are hardware-mandated.  Every field is packed
# UNMASKED, so an over-range value ORs into its neighbour; keeping the geometry
# in range is the caller's job (for the A2A path, the launch-time guards in
# client/src/FusedA2AClient.cpp::runFusedA2A).
#
# This encoding is gfx9xx / gfx95x ONLY -- GFX12+ uses a different layout of
# the same size.
################################################################################

from rocisa.container import sgpr
from rocisa.instruction import (
    SMovB32, SMovB64, SSubU32, SOrB32, SLShiftLeftB32,
)


SDMA_OP_COPY_SUBWIN         = 1
SDMA_SUBOP_COPY_LINEAR_RECT = 4
COPY_PACKET_DWORDS          = 13

# operation is a 7-bit index into the TC atomic op table (ADD_RTN_32 = 15,
# ADD_RTN_64 = 47); RTN means the op returns the pre-op value, which SDMA drops.
SDMA_OP_ATOMIC         = 10
SDMA_ATOMIC_ADD_RTN_32 = 15
ATOMIC_PACKET_DWORDS   = 8

# The packet's addressing granularity, header field [31:29].  Only the 16-byte
# encoding is hardware-validated; the field being 3 bits wide is not evidence
# that every encoding works, hence the allow-list rather than a range check.
VALIDATED_PACKET_ELEMENT_BYTES = (16,)

# DW0 minus the element-size field, which __init__ ORs in per instance.
_COPY_HEADER_DW0_BASE = ((SDMA_OP_COPY_SUBWIN & 0xFF)
                         | ((SDMA_SUBOP_COPY_LINEAR_RECT & 0xFF) << 8))
ATOMIC_HEADER_DW0 = ((SDMA_OP_ATOMIC & 0xFF) | ((SDMA_ATOMIC_ADD_RTN_32 & 0x7F) << 25))


class SdmaPacketEmitter:
    """Builds the COPY_SUBWIN + ATOMIC packet dword arrays in SGPRs, in the
    layout documented at the top of this file.

    Allocation-free: every method takes the registers it uses, and unlike
    SdmaRingEmitter this class never touches a pool and takes no `w`.  The only
    state is the packet element size.

    UNITS: every X-direction input -- the pitches, the slice pitches and rect_x
    -- must arrive ALREADY IN PACKET ELEMENTS.  rect_y must NOT: it counts
    ROWS, which the hardware does not scale by ELEMENTSIZE.  Bases are byte
    addresses.  Nothing here converts, so a caller that hands over its own
    element counts writes a silently wrong packet; scale with
    `packetElementLog2`.

    ALIASING CONTRACT: the packet block must be disjoint from every field
    argument.  Fields are read as the block is written, in field order, so a
    field register that is also an ALREADY-WRITTEN packet slot is read back as
    packet data -- dstSliceS == pktS+5 makes DW10 encode (srcSlice-1)-1.  It is
    not checked and does not fault.

    The block is written in field order and never re-read here, so an
    overlapping second packet must not be built until the first one's stores
    have been emitted (see emitPlacePacket's reuse note).
    """

    def __init__(self, packetElementBytes: int = 16):
        assert packetElementBytes in VALIDATED_PACKET_ELEMENT_BYTES, (
            "packet element size %r is not hardware-validated; only %r is"
            % (packetElementBytes, VALIDATED_PACKET_ELEMENT_BYTES))
        # Published so the caller can scale its X-direction counts to match.
        self.packetElementLog2 = packetElementBytes.bit_length() - 1
        self.copyHeaderDw0 = (_COPY_HEADER_DW0_BASE
                              | ((self.packetElementLog2 & 0x7) << 29))

    # ---- COPY_SUBWIN builder -----------------------------------------------

    def emitBuildCopyPacket(self, module, pktS,
                            srcBaseS, srcPitchS, srcSliceS,
                            dstBaseS, dstPitchS, dstSliceS,
                            rectXS, rectYS):
        """Build the 13 COPY_SUBWIN dwords into pktS[0:13].  Pitches, slice
        pitches and rect_x arrive in PACKET ELEMENTS and rect_y in rows; the
        coordinates are already folded into the two bases by the caller.  No
        scratch register is needed -- every dword is built in its own slot.

        The literal-0 coordinates are still written: the ring copies a fixed
        13-dword block, and a stale register would be read as a coordinate.

        The two 64-bit bases are relayed a dword at a time rather than moved as
        pairs, because this method requires no particular pktS alignment. At
        the caller's 4-aligned pktS, DW1/DW2 starts odd and cannot form an
        SReg_64 at all; DW6/DW7 does land 2-aligned there, but pairing it would
        also constrain dstBaseS, which no caller is asked to align.
        """
        module.add(SMovB32(dst=sgpr(pktS + 0), src=hex(self.copyHeaderDw0),
                           comment="SUBWIN DW0: op=COPY sub_op=RECT elementsize=log2(%dB)"
                                   % (1 << self.packetElementLog2)))
        module.add(SMovB32(dst=sgpr(pktS + 1), src=sgpr(srcBaseS + 0),
                           comment="SUBWIN DW1: srcBase lo"))
        module.add(SMovB32(dst=sgpr(pktS + 2), src=sgpr(srcBaseS + 1),
                           comment="SUBWIN DW2: srcBase hi"))
        module.add(SMovB32(dst=sgpr(pktS + 3), src=hex(0),
                           comment="SUBWIN DW3: src_x=0|src_y=0 (folded into srcBase)"))
        # DW4: 19-bit pitch field at [31:13]; the z field [10:0] is left 0.
        module.add(SSubU32(dst=sgpr(pktS + 4), src0=sgpr(srcPitchS), src1=1,
                           comment="SUBWIN DW4: src_pitch-1 (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(pktS + 4), src=sgpr(pktS + 4), shiftHex=13,
                                  comment="SUBWIN DW4: src_pitch-1 (<< 13)"))
        # DW5: 28-bit slice field at [27:0].  Not a free field despite rect_z
        # being 0 -- the reference implementation asserts
        # RECT_X * RECT_Y <= SLICE_PITCH and nothing at launch checks it, so the
        # caller has to hand over a value that satisfies it.
        module.add(SSubU32(dst=sgpr(pktS + 5), src0=sgpr(srcSliceS), src1=1,
                           comment="SUBWIN DW5: src_slice-1 (slice - 1)"))
        module.add(SMovB32(dst=sgpr(pktS + 6), src=sgpr(dstBaseS + 0),
                           comment="SUBWIN DW6: dstBase lo"))
        module.add(SMovB32(dst=sgpr(pktS + 7), src=sgpr(dstBaseS + 1),
                           comment="SUBWIN DW7: dstBase hi"))
        module.add(SMovB32(dst=sgpr(pktS + 8), src=hex(0),
                           comment="SUBWIN DW8: dst_x=0|dst_y=0 (folded into dstBase)"))
        module.add(SSubU32(dst=sgpr(pktS + 9), src0=sgpr(dstPitchS), src1=1,
                           comment="SUBWIN DW9: dst_pitch-1 (pitch - 1)"))
        module.add(SLShiftLeftB32(dst=sgpr(pktS + 9), src=sgpr(pktS + 9), shiftHex=13,
                                  comment="SUBWIN DW9: dst_pitch-1 (<< 13)"))
        module.add(SSubU32(dst=sgpr(pktS + 10), src0=sgpr(dstSliceS), src1=1,
                           comment="SUBWIN DW10: dst_slice-1 (slice - 1)"))
        # DW11: two 14-bit extents at [13:0] and [29:16].
        module.add(SLShiftLeftB32(dst=sgpr(pktS + 11), src=sgpr(rectYS), shiftHex=16,
                                  comment="SUBWIN DW11: rect_x|rect_y (rectY << 16, rows: NOT scaled)"))
        module.add(SOrB32(dst=sgpr(pktS + 11), src0=sgpr(pktS + 11), src1=sgpr(rectXS),
                          comment="SUBWIN DW11: rect_x|rect_y (| rectX)"))
        module.add(SSubU32(dst=sgpr(pktS + 11), src0=sgpr(pktS + 11), src1=hex(0x00010001),
                           comment="SUBWIN DW11: rect_x-1|rect_y-1 (both minus one; exact for rect_x >= 1)"))
        module.add(SMovB32(dst=sgpr(pktS + 12), src=hex(0),
                           comment="SUBWIN DW12: rect_z=0, default cache/swizzle"))

    # ---- ATOMIC ADD_RTN_32 builder ------------------------------------------

    def emitBuildAtomicPacket(self, module, pktS, dstAddrS):
        """Build the 8 ATOMIC ADD_RTN_32 dwords into pktS[0:8]: raise
        peer_flagPtr[p][myRank] by 1.  dstAddrS is a 2-SGPR pointer to the flag
        slot, which the caller strides by 4 because this ADD_RTN_32 writes
        4 bytes.

        The caller may hand this the SAME block it used for the COPY packet --
        8 dwords against the COPY's 13.  See emitPlacePacket's reuse note for
        why that is safe and what ordering it requires."""
        assert pktS % 2 == 0, (
            "pktS %r is not 2-aligned, which DW4-DW7's 64-bit zeroing requires"
            % (pktS,))
        module.add(SMovB32(dst=sgpr(pktS + 0), src=hex(ATOMIC_HEADER_DW0),
                           comment="ATOMIC DW0: op=ATOMIC operation=ADD_RTN_32"))
        module.add(SMovB32(dst=sgpr(pktS + 1), src=sgpr(dstAddrS + 0),
                           comment="ATOMIC DW1: addr lo"))
        module.add(SMovB32(dst=sgpr(pktS + 2), src=sgpr(dstAddrS + 1),
                           comment="ATOMIC DW2: addr hi"))
        module.add(SMovB32(dst=sgpr(pktS + 3), src=hex(1),
                           comment="ATOMIC DW3: src_data lo (addend)"))
        module.add(SMovB64(dst=sgpr(pktS + 4, 2), src=0,
                           comment="ATOMIC DW4: src_data hi (unused by ADD_RTN_32), DW5: cmp_data lo (unused)"))
        module.add(SMovB64(dst=sgpr(pktS + 6, 2), src=0,
                           comment="ATOMIC DW6: cmp_data hi (unused), DW7: loop_interval=0"))
