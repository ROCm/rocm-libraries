################################################################################
#
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################
"""LDS bank-conflict swizzle for TLU=1 (NT) subtile transpose reads.

The baseline K-major LDS image maps a read half's 32 lanes onto only 32 of the
64 banks, giving a 2-way conflict.  Two transforms recover 1-way, selected by
the strip width in chunks per K-column (``cpc``): a chunk-index XOR plus
load-block pad for the narrow widths, the column-scatter layout for cpc 4 and 8.
Either way GR write and LR read agree on the LDS image -- the XOR is an
involution so both sides apply it unchanged, while col_scatter has GR
de-interleave what LR interleaves -- so A round-trips.  Unvalidated widths fall
back to no swizzle (``None``).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TLUSwizzle:
    """A chunk-index XOR swizzle plus load-block pad for one TLU stack."""
    xorFromBit: int       # chunk[xorToBit] ^= chunk[xorFromBit], on the 16B chunk index
    xorToBit: int
    padBytes: int         # added once per block: (chunk >> blockChunkBits) * padBytes
    blockChunkBits: int   # log2(chunks per load-block)


@dataclass(frozen=True)
class TLUColScatter:
    """Column-scatter layout for a wide TLU strip (cpc 4 and 8).

    A single-bit XOR cannot reach 1-way at these widths: the two ds_read phases
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

    With inter-load padding the phases then cover complementary halves of the
    even bank pairs.  Derived for cpc in {4,8}, which fp4 reaches at stacks 8
    and 16 and bf16 at stacks 2 and 4; all fields come from N = stackM.
    """
    N: int                    # loads per strip (= stackM)
    cpc: int                  # chunks per K-column (= stackM*instM*bpe/16)
    gGroups: int              # col_groups per load (= instK / N)
    cBits: int                # thread bits carrying m_chunk (= log2(cpc))
    gBits: int                # thread bits carrying col_group (= 6 - cBits)
    gdBit: int                # col_group bit separating co-accessed groups
    padBytes: int             # inter-load pad (DS_READ_B64_TR_B4 8B alignment)
    blkBytes: int             # padded load-block (= wavesize*16 + padBytes)
    mChunkThreadBits: tuple   # thread bit position per m_chunk bit, len cBits
    cgThreadBits: tuple       # thread bit position per col_group bit, len gBits
    readStrideBytes: int      # LR ds_read immediate step for readIdx (+16 in K-column)
    mTileBytes: int           # LR ds_read immediate step for mTile


def _buildColScatter(stackM: int, instM: int, instK: int, bpe: float,
                     waveSize: int, elemsPerRead: int) -> TLUColScatter:
    """Derive the col_scatter parameters for one TLU stack.

    Everything in the bit layout is a function of ``cpc`` (chunks per K-column),
    not of the stack: both wired dtypes satisfy ``instM*instK*bpe == waveSize*16``,
    so ``cpc * gGroups == waveSize`` and the thread field is always 6 bits.  That
    makes fp4 8x1 and bf16 2x1 the same layout (cpc 4), as are fp4 16x1 and bf16
    4x1 (cpc 8) -- only the load count N and the byte strides differ.
    """
    N = stackM
    cpc = int(stackM * instM * bpe) // 16          # chunks per K-column
    gGroups = instK // N                            # col_groups per load
    cBits = cpc.bit_length() - 1                    # m_chunk bits = log2(cpc)
    gBits = (waveSize.bit_length() - 1) - cBits     # col_group bits
    gdBit = gBits - 2                               # distinguishing group bit
    # A transpose read covers elemsPerRead K-columns of freeRuns free-dim runs.
    # The pad has to step a whole run span, or consecutive loads overlap instead
    # of landing on the next bank pair: fp4 spans one run (8 B), bf16 four (32 B).
    freeRuns = max(1, instM // elemsPerRead)
    padBytes = int(freeRuns * elemsPerRead * bpe)
    blkBytes = waveSize * 16 + padBytes
    # Guard the shape assumptions the bit layout rests on; a dtype or MMA shape
    # that breaks one would emit a silently wrong permutation rather than fail.
    # The elemsPerRead check subsumes the old stack whitelist: it is what stops
    # cgDelta reaching 0, which would leave the LR read not stepping in K.
    if cpc & (cpc - 1) or cBits > 3:
        raise ValueError("col_scatter needs a power-of-two cpc with at most 3 "
                         "m_chunk bits (contiguous in bytes), got cpc=%d" % cpc)
    if elemsPerRead % N:
        raise ValueError("col_scatter needs the transpose read to step whole "
                         "col-groups: elemsPerRead=%d, N=%d" % (elemsPerRead, N))
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
    # k_col += elemsPerRead leaves the load unchanged (elemsPerRead % N == 0) and
    # steps cg by elemsPerRead//N, so the byte step is a per-lane constant.
    cgDelta = elemsPerRead // N
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


# VGPR return count of the TLU=1 transpose read, keyed by bytes per element.
# The opcode itself lives in SubtileLREmit; only the register count feeds the
# layout math, so it sits here and keeps this module free of any dependency on
# the emit modules (SubtileLREmit already imports this one).
_TLU_TR_REGS_PER_READ = {0.5: 2, 2.0: 2}


def tluElemsPerRead(bpe) -> Optional[int]:
    """Elements one TLU=1 transpose read covers per lane, or None if unwired."""
    regs = _TLU_TR_REGS_PER_READ.get(float(bpe))
    return None if regs is None else int(regs * 4 / float(bpe))


def _stackOf(tileInfo) -> Optional[int]:
    """Stack size for this tile, or None if it is not an fp4 TLU stack."""
    try:
        stack = int(tileInfo.subtileShape[0])
    except (AttributeError, TypeError, ValueError):
        # Narrow on purpose: returning None here means "no swizzle", so a wider
        # catch would turn a rename into silently bank-conflicting kernels.
        return None
    # The XOR table below is fp4-verified only; bf16 takes col_scatter at every
    # stack it can select, so it must never reach _SWIZZLE_BY_STACK.
    return stack if float(tileInfo.bpe) == 0.5 else None


def _cpcOf(tileInfo) -> Optional[int]:
    """Chunks per K-column for this tile, or None if the dtype is not wired."""
    try:
        stack = int(tileInfo.subtileShape[0])
        instM = int(tileInfo.mmaTileShape[0])
        bpe = float(tileInfo.bpe)
    except (AttributeError, TypeError, ValueError):
        return None
    if bpe not in _TLU_TR_REGS_PER_READ:
        return None
    return int(stack * instM * bpe) // 16


def selectTLUSwizzle(tileInfo) -> Optional[TLUSwizzle]:
    """Return the TLUSwizzle for this tile's stack, or None if unsupported.

    Guarded to the fp4 (bpe 0.5) TLU stacks with a derived layout; anything
    else returns None so the emit paths keep their baseline addressing.
    """
    if _sharedStrip(tileInfo):
        return None
    stack = _stackOf(tileInfo)
    return _SWIZZLE_BY_STACK.get(stack) if stack is not None else None


# Chunks-per-K-column values that take the column-scatter layout instead of a
# single-bit XOR.  Keyed on cpc rather than the stack because the layout is a
# function of cpc alone: fp4 8x1 and bf16 2x1 both land on cpc 4, fp4 16x1 and
# bf16 4x1 on cpc 8.  Above 8 the layout degenerates -- cgDelta falls to 0 (the
# transpose read stops stepping K-columns) and m_chunk stops being contiguous in
# bytes -- so wider strips stay unswizzled.  On a shared strip the XOR is
# unusable (see _sharedStrip), so the narrow widths route here too.
_COL_SCATTER_CPC = frozenset({4, 8})
# The narrow widths the XOR normally owns, added rather than restated so the
# shared-strip set cannot fall behind _COL_SCATTER_CPC.  A miss here is a
# bank-conflicting kernel, not an error.
_COL_SCATTER_CPC_SHARED = _COL_SCATTER_CPC | frozenset({1, 2})


def selectTLUColScatter(tileInfo) -> Optional[TLUColScatter]:
    """Return the col_scatter layout for this tile, or None.

    Mutually exclusive with selectTLUSwizzle: for fp4 the XOR path handles the
    narrow strips (cpc 1 and 2, i.e. 2x1 and 4x1) and col_scatter the wide ones.
    bf16 strips are 4x wider per stack, so both its stacks land in col_scatter.
    """
    cpc = _cpcOf(tileInfo)
    if cpc is None:
        return None
    allowed = _COL_SCATTER_CPC_SHARED if _sharedStrip(tileInfo) else _COL_SCATTER_CPC
    if cpc not in allowed:
        return None
    stack = int(tileInfo.subtileShape[0])
    instM = int(tileInfo.mmaTileShape[0])
    instK = int(tileInfo.mmaTileShape[1])
    waveSize = int(tileInfo.waveSize)
    elemsPerRead = tluElemsPerRead(tileInfo.bpe)
    return _buildColScatter(stack, instM, instK, float(tileInfo.bpe), waveSize,
                            elemsPerRead)


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
