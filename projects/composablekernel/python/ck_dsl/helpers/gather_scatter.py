# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gather / scatter address helpers for MoE-shaped pipelines.

The fused-MoE forward orchestrates three indirect-addressing patterns
that aren't captured by the existing tile / sweep / transform DAG
abstractions:

* **gather-by-bucket-row**: each *output row* of the per-expert input
 tensor reads from one *input row* of the original token tensor,
 selected by the per-bucket ``sorted_token_ids[bucket_idx]`` indirect.
 Used by the gate / up MLP input gather.

* **scatter-by-bucket-row**: each *input row* of the per-expert output
 tensor writes to one *output row* of the per-token MLP output,
 scaled by the matching ``sorted_topk_weights[bucket_idx]`` and
 accumulated atomically. Used by the down-projection's topk-weighted
 reduce.

* **scatter-by-token-pair**: each ``(token, k_topk)`` pair has its own
 bucket position in the sorted output -- computed at sort time by
 :mod:`ck_dsl.instances.moe_sorting`. The launcher reads the sort
 pass's ``SortedTokenIds`` / ``SortedTopkIds`` arrays back into
 metadata buffers for the gather / scatter steps.

This module exposes the per-element address-calc helpers; the actual
gather / scatter kernel bodies live in
:mod:`ck_dsl.instances.fused_moe`.

What v1 ships:

* :func:`gather_row_offset` -- ``sorted_token_ids[bucket_idx] * hidden
 + col`` as one SSA chain.
* :func:`scatter_token_offset` -- inverse: ``token_id * hidden + col``
 for the topk-weighted reduce write side.
* :func:`load_sorted_token_id` -- one i32 global load with the right
 alignment annotation.
"""

from __future__ import annotations

from ..core.ir import IRBuilder, PtrType, Value


__all__ = [
    "gather_row_offset",
    "load_sorted_token_id",
    "load_sorted_topk_weight",
    "scatter_token_offset",
]


def gather_row_offset(
    b: IRBuilder,
    sorted_token_ids: Value,
    bucket_idx: Value,
    *,
    hidden: int,
    col: Value,
) -> Value:
    """Return the flat element offset for one column of one row in the
    pre-sort token tensor, gathered by ``bucket_idx``.

    Pipeline::

    token_id = sorted_token_ids[bucket_idx] # i32 load
    offset = token_id * hidden + col # flat i32 index

    Used by the fused-MoE gate / up MLP input gather: the kernel walks
    its bucket-indexed rows and pulls the original token's hidden vector
    via this indirect.
    """
    token_id = load_sorted_token_id(b, sorted_token_ids, bucket_idx)
    return b.add(b.mul(token_id, b.const_i32(hidden)), col)


def scatter_token_offset(
    b: IRBuilder, token_id: Value, *, hidden: int, col: Value
) -> Value:
    """Return the flat element offset for one column of one token in
    the per-token output tensor.

    Trivial helper -- exposed for parity with :func:`gather_row_offset`
    so the fused-MoE topk-weighted reduce kernel reads symmetrically.
    """
    return b.add(b.mul(token_id, b.const_i32(hidden)), col)


def load_sorted_token_id(
    b: IRBuilder, sorted_token_ids: Value, bucket_idx: Value
) -> Value:
    """One ``i32`` global load from the moe-sort output, with the
    pointer alignment annotation tightened to 4 bytes.
    """
    if not isinstance(sorted_token_ids.type, PtrType):
        raise ValueError(
            "load_sorted_token_id expects a pointer to int32 "
            "(the SortedTokenIds buffer from moe_sort)"
        )
    if sorted_token_ids.type.pointee.name != "i32":
        raise ValueError(
            f"SortedTokenIds must be ptr<i32>, got "
            f"ptr<{sorted_token_ids.type.pointee.name}>"
        )
    return b.global_load_i32(sorted_token_ids, bucket_idx)


def load_sorted_topk_weight(
    b: IRBuilder, sorted_weights: Value, bucket_idx: Value
) -> Value:
    """One ``f32`` global load of the per-bucket topk weight.

    The fused-MoE down-projection's topk-weighted reduce multiplies its
    output by this weight before the atomic accumulate into the
    per-token output.
    """
    if not isinstance(sorted_weights.type, PtrType):
        raise ValueError(
            "load_sorted_topk_weight expects a pointer to f32 "
            "(the SortedWeights buffer from moe_sort)"
        )
    if sorted_weights.type.pointee.name != "f32":
        raise ValueError(
            f"SortedWeights must be ptr<f32>, got "
            f"ptr<{sorted_weights.type.pointee.name}>"
        )
    return b.global_load_f32(sorted_weights, bucket_idx)
