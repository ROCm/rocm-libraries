################################################################################
#
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################
"""LDS bank-conflict swizzle for TLU=1 (NT) subtile transpose reads.

The baseline K-major LDS image maps a read half's 32 lanes onto only 32 of the
64 banks, giving a 2-way conflict.  Two transforms recover 1-way, selected by
stack size ``subtileShape[0]``: a chunk-index XOR plus load-block pad for 2x1
and 4x1, the column-scatter layout for 8x1 and 16x1.  Either way GR write and LR
read agree on the LDS image -- the XOR is an involution so both sides apply it
unchanged, while col_scatter has GR de-interleave what LR interleaves -- so A
round-trips.  Unvalidated stacks fall back to no swizzle (``None``).
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class TLUSwizzle:
    """A chunk-index XOR swizzle plus load-block pad for one TLU stack."""
    xorFromBit: int       # chunk[xorToBit] ^= chunk[xorFromBit], on the 16B chunk index
    xorToBit: int
    padBytes: int         # added once per block: (chunk >> blockChunkBits) * padBytes
    blockChunkBits: int   # log2(chunks per load-block)


@dataclass(frozen=True)
class TLUColScatter:
    """Column-scatter layout for a taller TLU fp4 stack (8x1 and up).

    A single-bit XOR cannot reach 1-way once stackM >= 8: the two ds_read phases
    are stackM loads apart and the pad-induced bank-pair shift wraps.  Instead,
    deal the K-columns across the loads and place each column's group at the
    thread position that lands its distinguishing bit on the bank-pair bit::

        K-column k  ->  load  k % N,   col_group = k // N      (N = stackM)

            load 0  <-  k = 0,  8, 16, 24, ...
            load 1  <-  k = 1,  9, 17, 25, ...        N = 8 shown
              ...
            load 7  <-  k = 7, 15, 23, 31, ...

        thread bit    5      4      3      2      1      0
                    cg[3]  cg[1]  cg[2]  cg[0]  mc[1]  mc[0]
                                  =====
                     bit 3 is the bank-pair bit, so it always
                     carries col_group[gdBit]

    With 8B inter-load padding the phases then cover complementary halves of the
    even bank pairs.  Derived for stackM in {8,16}; all fields come from
    N = stackM.
    """
    N: int                    # loads per strip (= stackM)
    cpc: int                  # chunks per K-column (= N/2 for fp4 b128)
    gGroups: int              # col_groups per load (= instK / N)
    cBits: int                # thread bits carrying m_chunk (= log2(N) - 1)
    gBits: int                # thread bits carrying col_group (= 7 - log2(N))
    gdBit: int                # col_group bit separating co-accessed groups
    padBytes: int             # inter-load pad (DS_READ_B64_TR_B4 8B alignment)
    blkBytes: int             # padded load-block (= wavesize*16 + padBytes)
    mChunkThreadBits: tuple   # thread bit position per m_chunk bit, len cBits
    cgThreadBits: tuple       # thread bit position per col_group bit, len gBits
    readStrideBytes: int      # LR ds_read immediate step for readIdx (+16 in K-column)
    mTileBytes: int           # LR ds_read immediate step for mTile


def _buildColScatter(stackM: int, instM: int, instK: int, bpe: float,
                     waveSize: int) -> TLUColScatter:
    """Derive the col_scatter parameters for one TLU fp4 stack (all from N)."""
    N = stackM
    # At 32 the cgDelta below is 0, so readStrideBytes comes out 0 and the LR
    # read stops stepping in K: a wrong-answer kernel, not a rejected one.
    assert N in _SHARED_STRIP_COL_SCATTER_STACKS, \
        "col_scatter derives stacks %s, got %u" % (sorted(_SHARED_STRIP_COL_SCATTER_STACKS), N)
    logN = int(math.log2(N))
    cpc = int(stackM * instM * bpe) // 16          # chunks per K-column
    gGroups = instK // N                            # col_groups per load
    cBits = logN - 1                                # m_chunk bits
    gBits = 7 - logN                                # col_group bits
    gdBit = 5 - logN                                # distinguishing group bit
    padBytes = 8
    blkBytes = waveSize * 16 + padBytes
    # Thread bits [5:0]: bit 3 is reserved for col_group[gdBit] (bank-pair
    # separation); 0,1,2,4,5 take m_chunk first, then the other col_group bits.
    positions = [0, 1, 2, 4, 5]
    mChunkThreadBits = tuple(positions[:cBits])
    others = [i for i in range(gBits) if i != gdBit]
    cgThreadBits = [0] * gBits
    for j, i in enumerate(others):
        cgThreadBits[i] = positions[cBits + j]
    cgThreadBits[gdBit] = 3
    mTileBytes = int(instM * bpe)
    # k_col += 16 leaves the load unchanged (16 % N == 0 for N in {8,16}) and
    # steps cg by 16//N, so the byte step is a per-lane constant.
    cgDelta = 16 // N
    readStrideBytes = 0
    for i in range(gBits):
        if (cgDelta >> i) & 1:
            readStrideBytes += (1 << cgThreadBits[i]) * 16
    return TLUColScatter(N=N, cpc=cpc, gGroups=gGroups, cBits=cBits, gBits=gBits,
                         gdBit=gdBit, padBytes=padBytes, blkBytes=blkBytes,
                         mChunkThreadBits=mChunkThreadBits,
                         cgThreadBits=tuple(cgThreadBits),
                         readStrideBytes=readStrideBytes, mTileBytes=mTileBytes)


# Keyed by stack size subtileShape[0]. Each entry is intended to be 1-way,
# bijective and to reconstruct A. Unlisted stacks -> no swizzle yet.
_SWIZZLE_BY_STACK = {
    # 2x1 fp4: chunk[6] ^= chunk[5], 8B pad per 64-chunk (1024B) load-block.
    2: TLUSwizzle(xorFromBit=5, xorToBit=6, padBytes=8, blockChunkBits=6),
    # 4x1 fp4: chunk[7] ^= chunk[4] (chunk[4]=frow bit3, chunk[7]=kGroup bit1).
    # Both bits are per-lane and outside the per-read mTile/readIdx field, so the
    # 2x1 base swizzle applies unchanged with no per-read correction.
    4: TLUSwizzle(xorFromBit=4, xorToBit=7, padBytes=8, blockChunkBits=6),
}


def _sharedStrip(tileInfo) -> bool:
    """True when a strip is split across waves, so the XOR path cannot be used.

    The XOR acts on the physical chunk index, and a shared strip gives each wave
    a sub-strip offset landing in that same index, which no post-XOR offset can
    express.  col_scatter is unaffected: its load index enters additively.
    """
    return (int(tileInfo.grWavesPerStrip) > 1
            or int(tileInfo.grKSplit) > 1)


def _stackOf(tileInfo) -> Optional[int]:
    """Stack size for this tile, or None if it is not an fp4 TLU stack."""
    try:
        stack = int(tileInfo.subtileShape[0])
    except (AttributeError, TypeError, ValueError):
        # Narrow on purpose: returning None here means "no swizzle", so a wider
        # catch would turn a rename into silently bank-conflicting kernels.
        return None
    return stack if float(tileInfo.bpe) == 0.5 else None


def selectTLUSwizzle(tileInfo) -> Optional[TLUSwizzle]:
    """Return the TLUSwizzle for this tile's stack, or None if unsupported.

    Guarded to the fp4 (bpe 0.5) TLU stacks with a derived layout; anything
    else returns None so the emit paths keep their baseline addressing.
    """
    if _sharedStrip(tileInfo):
        return None
    stack = _stackOf(tileInfo)
    return _SWIZZLE_BY_STACK.get(stack) if stack is not None else None



# Stacks using column-scatter instead of a single-bit XOR.  On a shared strip
# the XOR is unusable (see _sharedStrip) so the short stacks route here too;
# elsewhere the XOR wins only on VALU cost.
_COL_SCATTER_STACKS = frozenset({8, 16})
_SHARED_STRIP_COL_SCATTER_STACKS = _COL_SCATTER_STACKS | frozenset(_SWIZZLE_BY_STACK)


def selectTLUColScatter(tileInfo) -> Optional[TLUColScatter]:
    """Return the col_scatter layout for this tile's stack, or None.

    Mutually exclusive with selectTLUSwizzle: the XOR path handles 2x1/4x1 and
    the col_scatter path 8x1 and 16x1.  Guarded to fp4.
    """
    stack = _stackOf(tileInfo)
    if stack is None:
        return None
    supported = (_SHARED_STRIP_COL_SCATTER_STACKS if _sharedStrip(tileInfo)
                 else _COL_SCATTER_STACKS)
    if stack not in supported:
        return None
    instM = int(tileInfo.mmaTileShape[0])
    instK = int(tileInfo.mmaTileShape[1])
    waveSize = int(tileInfo.waveSize)
    return _buildColScatter(stack, instM, instK, float(tileInfo.bpe), waveSize)


def tluPadBytes(tileInfo) -> int:
    """Inter-load-block LDS pad this tile's layout inserts, or 0 for neither.

    The XOR and col_scatter layouts both pad between DTL load-blocks and are
    mutually exclusive, so one selector pair answers for every caller.
    """
    swz = selectTLUSwizzle(tileInfo)
    cs = selectTLUColScatter(tileInfo)
    if swz:
        return int(swz.padBytes)
    return int(cs.padBytes) if cs else 0


def grLoadBlockBytes(waveSize: int, tileInfo) -> int:
    """LDS bytes one wave's DTL load-block occupies, pad included.

    A block is one wavesize-wide load at the tile's load width, plus the pad that
    separates it from the next block.
    """
    return int(waveSize * tileInfo.gr.config.loadWidth + tluPadBytes(tileInfo))


def swizzlePadPerStrip(tileInfo) -> int:
    """Extra LDS bytes a swizzled subtile strip occupies beyond subtileSize.

    One pad per load-block above block 0.  GR write, LR read and the LDS size
    computation must all fold this in so adjacent strips do not overlap.
    """
    padBytes = tluPadBytes(tileInfo)
    if not padBytes:
        return 0
    # Per-K-window, so derive from instK and NOT DepthU: a strip spans exactly
    # one MFMA K-window and DepthU > instK just adds further strips (sId1).
    instK = int(tileInfo.mmaTileShape[1])
    stackK = int(tileInfo.subtileShape[1])
    waveSize = int(tileInfo.waveSize)
    instM = int(tileInfo.mmaTileShape[0])
    stackM = int(tileInfo.subtileShape[0])
    mStripBytes = int(stackM * instM * tileInfo.bpe)
    chunksPerK = max(1, mStripBytes // 16)
    numBlocks = max(1, (instK * stackK * chunksPerK) // waveSize)
    return (numBlocks - 1) * padBytes


def stripStrideBytes(tileInfo) -> int:
    """LDS bytes between the start of consecutive subtile strips (M/N direction).

    Equals the nominal contiguous strip size plus any swizzle pad.  Used as the
    per-subtile-row LDS stride on both the GR write and LR read sides.
    """
    return int(tileInfo.subtileSize) + swizzlePadPerStrip(tileInfo)
