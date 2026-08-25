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


# Keyed by stack size subtileShape[0]. Values verified against the bank model
# (1-way, bijective, reconstructs A). Unlisted stacks -> no swizzle yet.
_SWIZZLE_BY_STACK = {
    # 2x1 fp4: chunk[6] ^= chunk[5], 8B pad per 64-chunk (1024B) load-block.
    2: TLUSwizzle(xorFromBit=5, xorToBit=6, padBytes=8, blockChunkBits=6),
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


def swizzlePadPerStrip(tileInfo) -> int:
    """Extra LDS bytes a swizzled subtile strip occupies beyond subtileSize.

    The pad is inserted once per load-block above block 0, so a strip that spans
    ``numGRPerSubtile`` blocks grows by ``(numGRPerSubtile - 1) * padBytes``.
    Returns 0 when the stack has no swizzle.  GR write, LR read, and the LDS
    size computation must all fold this in so adjacent strips do not overlap.
    """
    swz = selectTLUSwizzle(tileInfo)
    if not swz:
        return 0
    numBlocks = max(1, int(getattr(tileInfo, "numGRPerSubtile", 1)))
    return (numBlocks - 1) * int(swz.padBytes)


def stripStrideBytes(tileInfo) -> int:
    """LDS bytes between the start of consecutive subtile strips (M/N direction).

    Equals the nominal contiguous strip size plus any swizzle pad.  Used as the
    per-subtile-row LDS stride on both the GR write and LR read sides.
    """
    return int(tileInfo.subtileSize) + swizzlePadPerStrip(tileInfo)
