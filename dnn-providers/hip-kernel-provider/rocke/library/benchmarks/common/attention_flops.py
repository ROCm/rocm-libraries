# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Architecture-independent theoretical attention FLOP accounting."""

from __future__ import annotations


def attended_pairs(
    seqlen_q: int,
    seqlen_k: int,
    *,
    causal: bool,
    sliding_window: int = 0,
) -> int:
    """Return logical query/key pairs under right-aligned mask semantics."""
    sq = int(seqlen_q)
    sk = int(seqlen_k)
    window = int(sliding_window)
    if sq <= 0 or sk <= 0:
        return 0
    if not causal and window <= 0:
        return sq * sk

    offset = sk - sq
    total = 0
    for qi in range(sq):
        last = qi + offset if causal else sk - 1
        last = min(sk - 1, max(-1, last))
        if window > 0:
            first = max(0, qi + offset - window + 1)
        else:
            first = 0
        if last >= first:
            total += last - first + 1
    return total


def attention_flops(
    batch: int,
    nhead_q: int,
    head_size: int,
    seqlen_q: int,
    seqlen_k: int,
    *,
    causal: bool,
    sliding_window: int = 0,
) -> float:
    """Count QK^T and PV as ``4 * B * Hq * D * attended_pairs``."""
    pairs = attended_pairs(
        seqlen_q,
        seqlen_k,
        causal=causal,
        sliding_window=sliding_window,
    )
    return 4.0 * int(batch) * int(nhead_q) * int(head_size) * float(pairs)
