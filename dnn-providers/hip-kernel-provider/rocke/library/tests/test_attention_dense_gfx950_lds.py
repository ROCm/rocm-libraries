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
    return AttentionDenseSpec(
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
        **kw,
    )


def _formula(spec):
    return dense.gfx950_dense_static_lds_bytes(
        head_size=spec.head_size,
        block_n=spec.block_n,
        k_row_pad=spec.lds_k_row_pad,
        v_row_pad=spec.lds_v_row_pad,
        k_group_pad=spec.lds_k_group_pad,
    )


_WITHIN_BUDGET = [
    (128, 128, 64, {}),
    (128, 128, 64, {"lds_v_row_pad": 64}),
    (128, 128, 64, {"lds_k_row_pad": 24}),
    (64, 128, 128, {}),
    (64, 128, 128, {"lds_k_group_pad": 40}),
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
    legal = set(dense.gfx950_dense_tile_legality(head_size=head_size))
    accepted = {
        (bm, bn)
        for bm in attn_axes._GFX950_BLOCK_M
        for bn in attn_axes._GFX950_BLOCK_N
        if dense.supports_attention_dense(_spec(head_size, bm, bn))[0]
    }
    axis_tiles = {
        (bm, bn) for bm in attn_axes._GFX950_BLOCK_M for bn in attn_axes._GFX950_BLOCK_N
    }
    assert accepted == legal & axis_tiles
    assert accepted, "no legal tile on the dispatcher axes"
