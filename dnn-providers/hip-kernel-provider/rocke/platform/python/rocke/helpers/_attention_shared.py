# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pieces shared by the MFMA and WMMA attention forward inner bodies.

Both :mod:`rocke.helpers.mfma_attention` (wave64 MFMA) and
:mod:`rocke.helpers.wmma_attention` (wave32 WMMA) need the same online-softmax
row reduction, K-tile size and dtype mapping. Keeping them in this leaf module
lets the MFMA module import the WMMA body without an import cycle: this module
imports neither of them. The old private names stay importable from
``mfma_attention``.
"""

from __future__ import annotations

from ..core.ir import F16, F32, BF16, IRBuilder, Value
from .distribution import (
    TileDistributionEncoding,
    block_tile_reduce_sync,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)

MFMA_ATTN_BLOCK_K = 16  # K positions per K-tile


def _ir_type_for_dtype(dtype: str):
    if dtype in ("f16", "fp16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"mfma_attention currently supports f16/bf16; got {dtype!r}")


# --- Distribution-driven softmax row reduce (CK Tile BlockReduce2dSync) -------
#
# The online-softmax row max / row sum fold each lane's per-row scalar across
# the 16 lanes that share that tile row. CK Tile expresses this as a
# ``block_tile_reduce_sync`` over a *reduce distribution*: a single keep-row Y
# (length 1) with the reduce axis collapsed into a lane-owned R level of length
# 16, derivative 1. ``_r_butterfly_plan`` then emits XOR masks ``1,2,4,8`` --
# byte-for-byte the historical ``wave_reduce_max/sum(lanes_per_row=16)``
# 4-stage butterfly (verified via the subset IR digest gate). This single
# encoding serves both the wave64 MFMA and the wave32 WMMA softmax: the lane
# butterfly stride is wave-size-independent (the ``wave_size`` arg only drives
# the cross-warp LDS stage, which a single-warp row reduce skips).
_SOFTMAX_ROW_REDUCE_ENC = TileDistributionEncoding(
    Rs=(16,),
    Hs=((1,),),
    Ps2RHs_major=((0,),),  # the single (lane) P feeds R (major 0)
    Ps2RHs_minor=((0,),),
    Ys2RHs_major=(1,),  # one keep-row Y on the M H-dim (length 1)
    Ys2RHs_minor=(0,),
)
_SOFTMAX_ROW_REDUCE_DIST = make_static_tile_distribution(_SOFTMAX_ROW_REDUCE_ENC)


def _softmax_row_reduce(b: IRBuilder, scalar: Value, *, combine: str) -> Value:
    """Reduce ``scalar`` across the 16 lanes sharing one tile row.

    Wraps the per-lane f32 ``scalar`` in a one-element
    :class:`StaticDistributedTensor` over :data:`_SOFTMAX_ROW_REDUCE_DIST`
    and folds it with :func:`block_tile_reduce_sync`. ``combine`` is
    ``"max"`` (row max) or ``"sum"`` (row sum). The emitted op stream is the
    same 4-stage XOR butterfly (masks ``1,2,4,8``) the legacy
    ``wave_reduce_max/sum(lanes_per_row=16)`` produced.
    """
    dt = make_static_distributed_tensor(_SOFTMAX_ROW_REDUCE_DIST, F32)
    dt.storage[0] = scalar
    block_tile_reduce_sync(b, dt, combine=combine)
    return dt.storage[0]
