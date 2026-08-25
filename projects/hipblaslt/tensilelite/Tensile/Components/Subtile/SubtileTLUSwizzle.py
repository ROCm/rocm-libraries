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
tiles stacked along the free dim).  Only the 2x1 fp4 stack is wired today; other
stacks fall back to no swizzle (``None``) until their rules are validated.
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
    against the bank model (1-way, exact A round-trip) for stackM in {8,16,32}.

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


def selectTLUSwizzle(tileInfo) -> Optional[TLUSwizzle]:
    """Return the TLUSwizzle for this tile's stack, or None if unsupported.

    Guarded to the fp4 (bpe 0.5) TLU stacks the bank model covers; anything
    else returns None so the emit paths keep their baseline addressing.
    """
    try:
        stack = int(tileInfo.subtileShape[0])
    except Exception:
        return None
    if float(tileInfo.bpe) != 0.5:
        return None
    return _SWIZZLE_BY_STACK.get(stack)


# Stacks that use the column-scatter layout instead of a single-bit XOR.
_COL_SCATTER_STACKS = frozenset({8})


def selectTLUColScatter(tileInfo) -> Optional[TLUColScatter]:
    """Return the col_scatter layout for this tile's stack, or None.

    Mutually exclusive with selectTLUSwizzle: the XOR path handles 2x1/4x1, the
    col_scatter path handles 8x1 (and, once wired, 16x1/32x1).  Guarded to fp4.
    """
    try:
        stack = int(tileInfo.subtileShape[0])
    except Exception:
        return None
    if float(tileInfo.bpe) != 0.5:
        return None
    if stack not in _COL_SCATTER_STACKS:
        return None
    instM = int(tileInfo.mmaTileShape[0])
    instK = int(tileInfo.mmaTileShape[1])
    waveSize = int(getattr(tileInfo, "waveSize", 0)) or 64
    return _buildColScatter(stack, instM, instK, float(tileInfo.bpe), waveSize)


def swizzlePadPerStrip(tileInfo) -> int:
    """Extra LDS bytes a swizzled subtile strip occupies beyond subtileSize.

    The pad is inserted once per load-block above block 0, so a strip that spans
    ``numGRPerSubtile`` blocks grows by ``(numGRPerSubtile - 1) * padBytes``.
    Returns 0 when the stack has no swizzle.  GR write, LR read, and the LDS
    size computation must all fold this in so adjacent strips do not overlap.
    """
    swz = selectTLUSwizzle(tileInfo)
    cs = selectTLUColScatter(tileInfo)
    if not swz and not cs:
        return 0
    padBytes = int(swz.padBytes) if swz else int(cs.padBytes)
    # A subtile strip spans exactly ONE MFMA K-window (subtileShape[1] MFMA
    # tiles of instK K-rows), regardless of DepthU: DepthU > instK just stacks
    # additional K-windows as further strips (sId1 in the emit paths).  One DTL
    # load-block covers wavesize chunks and a pad is inserted above each block
    # boundary except the first, so the block count is per-K-window.  Deriving it
    # from instK (not DepthU) keeps stripStride correct for DepthU > instK.  A
    # taller stack makes each K row chunksPerK = mStripBytes/16 b128 chunks wide,
    # so a K-window holds instK*stackK*chunksPerK chunks -- the block count must
    # scale by chunksPerK (2x1: chunksPerK=1, unchanged; 4x1: 2 -> 4 blocks).
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
