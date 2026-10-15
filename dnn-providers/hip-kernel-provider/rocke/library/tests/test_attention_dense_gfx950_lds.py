# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-budget gate for the gfx950 dense attention kernel.

``supports_attention_dense`` rejects specs whose static LDS exceeds the gfx950
capacity. These tests pin the closed-form byte count against the allocation the
builder really emits, including non-default ``lds_*_pad`` values, and check that
``gfx950_dense_tile_legality`` agrees with the gate over the dispatcher's tile axes.

Pure text lowering -- no GPU and no comgr required.
"""

import pytest

from dispatch.attention import axes as attn_axes
from kernels.gfx950 import attention_dense as dense
from kernels.gfx950.attention_dense import AttentionDenseSpec

from test_attention_dense_d64_lds import _lds_pool_bytes


def _spec(head_size, block_m, block_n, **kw):
    defaults = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=head_size,
        causal=True,
        dtype="bf16",
        block_m=block_m,
        block_n=block_n,
        persistent=False,
    )
    defaults.update(kw)
    return AttentionDenseSpec(**defaults)


def _formula(spec):
    return dense.gfx950_dense_static_lds_bytes(
        head_size=spec.head_size,
        block_n=spec.block_n,
        k_row_pad=spec.lds_k_row_pad,
        v_row_pad=spec.lds_v_row_pad,
        k_group_pad=spec.lds_k_group_pad,
        wide_lds_dma=spec.wide_lds_dma,
    )


_WITHIN_BUDGET = [
    (128, 128, 64, {}),
    (128, 128, 64, {"lds_v_row_pad": 64}),
    (128, 128, 64, {"lds_k_row_pad": 24}),
    (64, 128, 128, {}),
    (64, 128, 128, {"lds_k_group_pad": 40}),
    # wide_lds_dma=True uses a different layout (K/V slab lines of 64*8 elems)
    # and should produce a SMALLER footprint than the non-wide D128/BN64 path.
    # Non-wide: 2*64*(128+8 + 128+32)*2 = 75,776 B; wide: 68,096 B.
    # persistent=True required by __post_init__ when wide_lds_dma=True.
    (128, 128, 64, {"wide_lds_dma": True, "persistent": True}),
]

# Over-budget layouts that the default-pad formula used to accept.
_OVER_BUDGET = [
    (64, 256, 256, {"lds_k_group_pad": 72}),
    (128, 256, 128, {"lds_v_row_pad": 64}),
    (128, 256, 128, {"lds_k_row_pad": 64}),
]


@pytest.mark.parametrize("head_size,block_m,block_n,pads", _WITHIN_BUDGET)
def test_formula_matches_real_lds_pool(head_size, block_m, block_n, pads):
    spec = _spec(head_size, block_m, block_n, **pads)
    ok, reason = dense.supports_attention_dense(spec)
    assert ok, reason
    assert _formula(spec) == _lds_pool_bytes(spec)


@pytest.mark.parametrize("head_size,block_m,block_n,pads", _OVER_BUDGET)
def test_over_budget_pads_are_rejected(head_size, block_m, block_n, pads):
    spec = _spec(head_size, block_m, block_n, **pads)
    assert _formula(spec) > dense.GFX950_LDS_CAPACITY_BYTES
    ok, reason = dense.supports_attention_dense(spec)
    assert not ok
    assert "LDS" in reason


@pytest.mark.parametrize("head_size", [64, 128])
def test_tile_legality_matches_gate_over_dispatcher_axes(head_size):
    """The LDS-budget predicate and gfx950_dense_tile_legality must agree.

    The dispatcher axes (_GFX950_BLOCK_M, _GFX950_BLOCK_N) are the union of
    all legal tiles across head_sizes, so they intentionally cover more
    (block_m, block_n) pairs than any single head_size allows.  The
    per-head-size gate is supports_attention_dense itself.

    This test verifies two properties:
    1. Every legal tile for this head_size is covered by the dispatcher axes
       (no legal tile is missing from the axis).
    2. supports_attention_dense (the pack-time gate) agrees exactly with
       gfx950_dense_tile_legality for this head_size.
    """
    legal = set(dense.gfx950_dense_tile_legality(head_size=head_size))
    axis_tiles = {
        (bm, bn) for bm in attn_axes._GFX950_BLOCK_M for bn in attn_axes._GFX950_BLOCK_N
    }

    # Every legal tile must be on the axes (axes are a superset by design).
    assert legal <= axis_tiles, (
        f"head_size={head_size}: legal tiles from gfx950_dense_tile_legality "
        f"contain tiles NOT on the dispatcher axes — add them to axes.py:\n"
        f"  missing: {legal - axis_tiles}"
    )

    # supports_attention_dense (the gate used at pack time) must agree exactly
    # with gfx950_dense_tile_legality for this head_size.
    accepted = {
        tile
        for tile in axis_tiles
        if dense.supports_attention_dense(_spec(head_size, *tile))[0]
    }
    assert accepted == legal, (
        f"head_size={head_size}: supports_attention_dense gate disagrees with "
        f"gfx950_dense_tile_legality.\n"
        f"  legal but not accepted: {legal - accepted}\n"
        f"  accepted but not legal: {accepted - legal}"
    )
    assert accepted, "no legal tile on the dispatcher axes"
