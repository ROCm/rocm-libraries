# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-budget gate for the gfx950 dense attention kernel.

``supports_attention_dense`` rejects specs whose static LDS exceeds the gfx950
capacity. These tests pin the closed-form byte count against the allocation the
builder really emits -- for every tile the catalog ships and for non-default
``lds_*_pad`` values -- and check that ``gfx950_dense_tile_legality`` agrees with
the gate. The C++ ``Gfx950AttentionDenseGeometry.hpp`` that restates the model is
exercised by the gfx950 attention-dense census, which requires every shipped
kernel's tile to pass ``isSupportedGfx950AttentionDenseTile``.

Pure text lowering -- no GPU and no comgr required.
"""

import pytest

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


_LEGAL_TILES = [
    (head_size, block_m, block_n)
    for head_size in (64, 128)
    for block_m, block_n in dense.gfx950_dense_tile_legality(head_size=head_size)
]


@pytest.mark.parametrize("head_size,block_m,block_n", _LEGAL_TILES)
def test_every_legal_tile_fits_and_matches_real_lds_pool(head_size, block_m, block_n):
    """Includes the tiles the LDS rule turns on: D128/BN128 (151,552 B) and
    D64/BM256/BN256 (135,168 B)."""
    spec = _spec(head_size, block_m, block_n)
    ok, reason = dense.supports_attention_dense(spec)
    assert ok, reason
    pool = _lds_pool_bytes(spec)
    assert pool == _formula(spec)
    assert pool <= dense.GFX950_LDS_CAPACITY_BYTES


@pytest.mark.parametrize("head_size,block_m,block_n,pads", _OVER_BUDGET)
def test_over_budget_pads_are_rejected(head_size, block_m, block_n, pads):
    spec = _spec(head_size, block_m, block_n, **pads)
    assert _formula(spec) > dense.GFX950_LDS_CAPACITY_BYTES
    ok, reason = dense.supports_attention_dense(spec)
    assert not ok
    assert "LDS" in reason


@pytest.mark.parametrize("head_size", [64, 128])
def test_gate_accepts_exactly_the_legal_tiles(head_size):
    """The gate (spec validation + shared preflight) and gfx950_dense_tile_legality
    state the structural tile rules separately; they must pick the same tiles out of
    every block_m/block_n the spec could name. The LDS term they share is pinned
    against the real pool above."""
    legal = set(dense.gfx950_dense_tile_legality(head_size=head_size))
    # lcm of every block_n below, so no tile is refused for its extents alone.
    extent = 256 * 3 * 5 * 7
    accepted = {
        (block_m, block_n)
        for block_m in (128, 256)
        for block_n in range(32, 257, 32)
        if dense.supports_attention_dense(
            _spec(head_size, block_m, block_n, seqlen_q=extent, seqlen_kv=extent)
        )[0]
    }
    assert accepted == legal
    assert accepted, "no legal tile"
