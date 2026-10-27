# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packed-varlen query-block decode shared by the attention prefill kernels.

A packed-varlen launch numbers its query blocks the AITER way: sequence ``i``
owns the blocks starting at ``cu_q[i] // block_q + i``.
:func:`rocke.helpers.attention.binary_search_seq_idx` inverts that numbering to
the owning sequence; :func:`varlen_q_block` adds the six ops every caller ran
after it -- the sequence's ``cu_q`` bounds, its query length, and the block's
index inside the sequence.
"""

from __future__ import annotations

from typing import NamedTuple

from rocke.core.ir import IRBuilder, Value
from rocke.helpers.attention import binary_search_seq_idx

__all__ = ["VarlenQBlock", "varlen_q_block"]


class VarlenQBlock(NamedTuple):
    seq_idx: Value
    cu_q_start: Value
    q_len: Value
    q_block_local_idx: Value


def varlen_q_block(
    b: IRBuilder,
    cu_q: Value,
    q_block_global_idx: Value,
    num_seqs: Value,
    *,
    block_q: int,
    iters: int,
) -> VarlenQBlock:
    """Locate packed q-block ``q_block_global_idx`` inside its sequence.

    ``q_block_local_idx * block_q`` is the block's first query row within the
    sequence; it is at or past ``q_len`` for the padding blocks the AITER
    count launches, and callers early-exit on that.
    """
    seq_idx = binary_search_seq_idx(
        b, cu_q, q_block_global_idx, num_seqs, block_q=block_q, iterations=iters
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_q_stop, cu_q_start)
    # The binary search's loop invariant is ``cu_q[i] // block_q + i <= target``,
    # so the sequence's first block sits at ``cu_q_start // block_q + seq_idx``.
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(block_q)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    return VarlenQBlock(seq_idx, cu_q_start, q_len, q_block_local_idx)
