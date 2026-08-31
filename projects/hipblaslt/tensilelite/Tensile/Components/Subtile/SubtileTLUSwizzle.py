################################################################################
#
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################
"""LDS bank-conflict swizzle for TLU=1 (NT) subtile transpose reads.

The NT path writes each operand free-dim contiguous, K-major, into LDS: chunk
index ``c`` (one 128-bit / mStripBytes block) holds logical K-row ``c``.  The
``ds_read_b64_tr_b4`` transpose read then addresses those chunks by K.  In the
baseline (no swizzle) layout the 32 lanes of a read half map onto only 32 of
the 64 banks in a repeating pattern, producing a 2-way bank conflict (verified
in ``format.md`` and reproduced by the standalone bank model).

A per-chunk XOR permutation plus a byte pad inserted between load-blocks moves
the colliding chunks onto distinct banks, recovering a 1-way (conflict-free)
access.  The XOR is an involution, so the *same* transform is applied on the GR
write side (which global K-row each lane fetches) and the LR read side (which
chunk each lane addresses); the LDS image round-trips A exactly.

The transform is selected by the stack size ``subtileShape[0]`` (number of MMA
tiles stacked along the free dim): 2x1 and 4x1 use the XOR above, 8x1 and 16x1
use the column-scatter layout instead (see ``TLUColScatter``), and any stack
whose rules are not validated falls back to no swizzle (``None``).
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class TLUSwizzle:
    """A chunk-index XOR swizzle plus load-block pad for one TLU stack.

    xorFromBit / xorToBit: ``chunk[xorToBit] ^= chunk[xorFromBit]`` on the
        chunk index (units of mStripBytes LDS blocks).
    padBytes:  bytes inserted per load-block (chunkBlockBits high chunk bits).
    blockChunkBits: log2(chunks per load-block) -- the pad is added once per
        block, i.e. ``(chunk >> blockChunkBits) * padBytes``.
    """
    xorFromBit: int
    xorToBit: int
    padBytes: int
    blockChunkBits: int


@dataclass(frozen=True)
class TLUColScatter:
    """Column-scatter layout for a taller TLU fp4 stack (8x1 and up).

    A single-bit XOR can no longer reach 1-way once stackM >= 8: the two
    ds_read phases are stackM loads apart and the pad-induced bank-pair shift
    wraps, so no within-load permutation separates them.  Instead of each DTL
    load owning a contiguous block of K-columns, spread them: K-column k goes to
    load ``k % N`` (N = stackM), and its ``col_group = k // N`` is placed at a
    thread position by a bit-interleave that lands the distinguishing group bit
    at thread bit 3 (the LDS bank-pair bit).  With 8B inter-load padding the two
    phases cover complementary halves of the even bank pairs -> 1-way.  Verified
    against the bank model (1-way, exact A round-trip) for stackM in {8,16}.
    Not 32: readStrideBytes derives from cgDelta = 16 // N, which is 0 there, so
    the K-column step would come out as no step at all.

    Fields are all derived from N = stackM:
      N:            loads / buffer_load instructions per strip (= stackM)
      cpc:          chunks per K-column (= N/2 for fp4 b128)
      gGroups:      col_groups per load (= instK / N)
      cBits:        thread bits carrying m_chunk (= log2(N) - 1)
      gBits:        thread bits carrying col_group (= 7 - log2(N))
      gdBit:        the col_group bit that separates co-accessed groups
      padBytes:     inter-load pad (8B, DS_READ_B64_TR_B4 8B alignment)
      blkBytes:     bytes per padded load-block (= wavesize*16 + padBytes)
      mChunkThreadBits: thread bit position for m_chunk bit j (list, len cBits)
      cgThreadBits:     thread bit position for col_group bit i (list, len gBits)
      readStrideBytes:  LR ds_read immediate step for readIdx (+16 in K-column)
      mTileBytes:       LR ds_read immediate step for mTile
    """
    N: int
    cpc: int
    gGroups: int
    cBits: int
    gBits: int
    gdBit: int
    padBytes: int
    blkBytes: int
    mChunkThreadBits: tuple
    cgThreadBits: tuple
    readStrideBytes: int
    mTileBytes: int


def _buildColScatter(stackM: int, instM: int, instK: int, bpe: float,
                     waveSize: int) -> TLUColScatter:
    """Derive the col_scatter parameters for one TLU fp4 stack (all from N)."""
    N = stackM
    logN = int(math.log2(N))
    cpc = int(stackM * instM * bpe) // 16          # chunks per K-column
    gGroups = instK // N                            # col_groups per load
    cBits = logN - 1                                # m_chunk bits
    gBits = 7 - logN                                # col_group bits
    gdBit = 5 - logN                                # distinguishing group bit
    padBytes = 8
    blkBytes = waveSize * 16 + padBytes
    # Thread bit layout [5:0]: bit 3 is reserved for col_group[gdBit] (bank-pair
    # separation).  The remaining positions 0,1,2,4,5 are filled sequentially,
    # first with m_chunk[0..cBits-1], then with col_group[i != gdBit].
    positions = [0, 1, 2, 4, 5]
    mChunkThreadBits = tuple(positions[:cBits])
    others = [i for i in range(gBits) if i != gdBit]
    cgThreadBits = [0] * gBits
    for j, i in enumerate(others):
        cgThreadBits[i] = positions[cBits + j]
    cgThreadBits[gdBit] = 3
    mTileBytes = int(instM * bpe)
    # readIdx advances the logical K-column by 16; that maps to a fixed LDS byte
    # step because the interleave is affine in the changing col-group bits.  The
    # model verifies it is a single constant across all lanes; recompute it here
    # closed-form from the col-group bits that flip when k_col += 16.
    #   k_col += 16 -> load unchanged (16 % N == 0 for N in {8,16}), cg += 16//N.
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


# Keyed by stack size subtileShape[0]. Values verified against the bank model
# (1-way, bijective, reconstructs A). Unlisted stacks -> no swizzle yet.
_SWIZZLE_BY_STACK = {
    # 2x1 fp4: chunk[6] ^= chunk[5], 8B pad per 64-chunk (1024B) load-block.
    2: TLUSwizzle(xorFromBit=5, xorToBit=6, padBytes=8, blockChunkBits=6),
    # 4x1 fp4: chunk[7] ^= chunk[4], 8B pad per 64-chunk load-block.  Both bits
    # are pure per-lane (chunk[4]=frow bit3, chunk[7]=kGroup bit1), so the same
    # per-lane base swizzle the 2x1 stack uses applies unchanged; only the pad
    # block count grows (a 4x1 strip spans chunksPerK=2 blocks per K row).  This
    # single-bit choice keeps both swizzle bits out of the per-read mTile/readIdx
    # field, avoiding a per-read base correction.  Verified 1-way + bijective.
    4: TLUSwizzle(xorFromBit=4, xorToBit=7, padBytes=8, blockChunkBits=6),
}


def _sharedStrip(tileInfo) -> bool:
    """True when a strip is split across waves, so the XOR path cannot be used.

    The XOR acts on the *physical* chunk index, and a shared strip gives each
    wave a sub-strip offset that lands in that same index -- axis-waves sharing
    a strip (grWavesPerStrip) or other-axis waves taking K slices of one
    (grKSplit).  Either way the wave is not expressible as an offset applied
    after the XOR.  col_scatter has no such coupling: there the load index
    enters purely additively as a K-column shift.
    """
    return (int(getattr(tileInfo, "grWavesPerStrip", 1)) > 1
            or int(getattr(tileInfo, "grKSplit", 1)) > 1)


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

    Guarded to the fp4 (bpe 0.5) TLU stacks the bank model covers; anything
    else returns None so the emit paths keep their baseline addressing.
    """
    if _sharedStrip(tileInfo):
        return None
    stack = _stackOf(tileInfo)
    return _SWIZZLE_BY_STACK.get(stack) if stack is not None else None


# Stacks that use the column-scatter layout instead of a single-bit XOR.
_COL_SCATTER_STACKS = frozenset({8, 16})


def selectTLUColScatter(tileInfo) -> Optional[TLUColScatter]:
    """Return the col_scatter layout for this tile's stack, or None.

    Mutually exclusive with selectTLUSwizzle: the XOR path handles 2x1/4x1 and
    the col_scatter path 8x1 and 16x1.  Guarded to fp4.
    """
    stack = _stackOf(tileInfo)
    if stack is None:
        return None
    if _sharedStrip(tileInfo):
        # A shared strip rules out the XOR (see _sharedStrip), so every stack
        # falls here; the bank model reaches 1-way at 2, 4, 8 and 16 alike (the
        # XOR wins elsewhere only on VALU cost).
        if stack not in (2, 4, 8, 16):
            return None
    elif stack not in _COL_SCATTER_STACKS:
        return None
    instM = int(tileInfo.mmaTileShape[0])
    instK = int(tileInfo.mmaTileShape[1])
    waveSize = int(getattr(tileInfo, "waveSize", 0)) or 64
    return _buildColScatter(stack, instM, instK, float(tileInfo.bpe), waveSize)


def tluPadBytes(tileInfo) -> int:
    """Inter-load-block LDS pad this tile's layout inserts, or 0 for neither.

    Both the XOR swizzle and the col_scatter layout separate consecutive DTL
    load-blocks by a pad, and they are mutually exclusive, so every caller that
    needs the pad is asking the same question of the same two selectors.
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

    The pad is inserted once per load-block above block 0, so a strip that spans
    ``numGRPerSubtile`` blocks grows by ``(numGRPerSubtile - 1) * padBytes``.
    Returns 0 when the stack has no swizzle.  GR write, LR read, and the LDS
    size computation must all fold this in so adjacent strips do not overlap.
    """
    padBytes = tluPadBytes(tileInfo)
    if not padBytes:
        return 0
    # Block count is per-K-window, so derive it from instK and NOT from DepthU:
    # a strip spans exactly one MFMA K-window, and DepthU > instK just stacks
    # further K-windows as further strips (sId1 in the emit paths).
    # A taller stack widens each K row to chunksPerK = mStripBytes/16 chunks, so
    # the window holds instK*stackK*chunksPerK (2x1: 1 block; 4x1: 2 -> 4).
    instK = int(tileInfo.mmaTileShape[1])
    stackK = int(tileInfo.subtileShape[1])
    waveSize = int(getattr(tileInfo, "waveSize", 0)) or 64
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
