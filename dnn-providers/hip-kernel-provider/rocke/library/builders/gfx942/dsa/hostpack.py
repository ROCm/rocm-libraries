# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free numpy inputs, bf16 packing, and the lightning-indexer score oracle.

This module is the correctness reference for the DSA lightning indexer. It has no
torch dependency (numpy only) so it runs in any environment, and it computes the
scores exactly as the kernel does: bf16-rounded operands, an f32 dot product per
head, a ReLU, a per-head weighting, a sum across heads, and a causal mask that
drives future keys to the sentinel.

Layout matches the kernel signature (see
``kernels.gfx942.lightning_indexer.build_lightning_indexer``):

* ``index_q``  [seqlen_q, H_I, D_I]  bf16   -- per-head index-query
* ``index_k``  [seqlen_k, D_I]       bf16   -- shared index-key (no head axis)
* ``w``        [H_I]                 f32    -- per-head indexer weights
* scores       [seqlen_q, seqlen_k]  f32    -- output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Must match kernels.gfx942.lightning_indexer.NEG_INF_SCORE. Kept as a local
# literal so the oracle has no dependency back up into the kernels layer.
NEG_INF_SCORE = -3.0e38


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round an f32 array to bf16 precision, returned as f32 values.

    Round-to-nearest-even on the top 16 bits, matching the truncation the
    hardware bf16 load + f32 promote produces. The kernel loads bf16 and casts
    to f32, so the oracle must score the bf16-rounded operands, not the full-
    precision ones, or the two would disagree by more than the tolerance.
    """
    u = x.astype(np.float32).view(np.uint32)
    # add rounding bias (0x7FFF + LSB of the retained mantissa), then truncate.
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Pack an f32 array into the uint16 bf16 bit pattern a launcher uploads."""
    u = x.astype(np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Unpack a uint16 bf16 bit pattern back to f32."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_inputs(
    seqlen_q: int,
    n_index_heads: int,
    index_head_dim: int,
    seqlen_k: int,
    *,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate a reproducible set of bf16-rounded indexer inputs (as f32)."""
    rng = np.random.default_rng(seed)
    index_q = to_bf16(rng.standard_normal((seqlen_q, n_index_heads, index_head_dim), dtype=np.float32))
    index_k = to_bf16(rng.standard_normal((seqlen_k, index_head_dim), dtype=np.float32))
    # Weights stay f32 (higher precision at negligible cost, per the dtype plan);
    # keep them non-negative so no head's contribution is inverted.
    w = np.abs(rng.standard_normal(n_index_heads, dtype=np.float32))
    return {"index_q": index_q, "index_k": index_k, "w": w}


def ref_indexer_scores(
    index_q: np.ndarray,
    index_k: np.ndarray,
    w: np.ndarray,
    q_pos_base: int = 0,
    *,
    neg_inf: float = NEG_INF_SCORE,
) -> np.ndarray:
    """Compute the reference score matrix [seqlen_q, seqlen_k].

    I(t, s) = sum over heads h of  w_h * ReLU( index_q[t, h] . index_k[s] )

    with the causal bound: for a query at absolute position ``q_pos_base + t``,
    keys ``s > q_pos_base + t`` are future keys and get the sentinel so top-k
    never selects them.
    """
    qf = to_bf16(index_q).astype(np.float32)
    kf = to_bf16(index_k).astype(np.float32)
    wf = w.astype(np.float32)
    # per-head dot products: [Q, H, D] . [Sk, D] -> [Q, H, Sk]
    dots = np.einsum("qhd,sd->qhs", qf, kf, optimize=True)
    relu = np.maximum(dots, 0.0)
    # weight per head and sum over heads -> [Q, Sk]
    scores = np.einsum("h,qhs->qs", wf, relu, optimize=True)
    # causal mask
    seqlen_q, seqlen_k = scores.shape
    q_pos = q_pos_base + np.arange(seqlen_q)[:, None]
    s_idx = np.arange(seqlen_k)[None, :]
    scores = np.where(s_idx <= q_pos, scores, neg_inf)
    return scores.astype(np.float32)


@dataclass(frozen=True)
class PackedIndexer:
    """bf16-as-uint16 device buffers plus the f32 reference scores."""

    index_q_bits: np.ndarray  # uint16 [seqlen_q, H_I, D_I]
    index_k_bits: np.ndarray  # uint16 [seqlen_k, D_I]
    w: np.ndarray  # f32 [H_I]
    q_pos_base: int
    ref_scores: np.ndarray  # f32 [seqlen_q, seqlen_k]


def pack(inputs: Dict[str, np.ndarray], q_pos_base: int = 0) -> PackedIndexer:
    """Round + pack inputs to device layout and precompute the reference scores."""
    return PackedIndexer(
        index_q_bits=f32_to_bf16_bits(inputs["index_q"]),
        index_k_bits=f32_to_bf16_bits(inputs["index_k"]),
        w=inputs["w"].astype(np.float32),
        q_pos_base=q_pos_base,
        ref_scores=ref_indexer_scores(inputs["index_q"], inputs["index_k"], inputs["w"], q_pos_base),
    )
