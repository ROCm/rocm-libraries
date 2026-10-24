# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dense short-chunk rule: KV-segment count for an underfilled default grid."""

from dataclasses import replace

import pytest

import dispatch.attention as attention
from dispatch.attention.dense_rules import dense_kv_segments


@pytest.mark.parametrize(
    "work,cus,expected",
    [
        (128, 256, 4),
        (64, 256, 8),
        (160, 256, 4),
        (112, 256, 4),
        (256, 256, 1),
        (640, 256, 1),
        (1, 256, 8),
        (0, 256, 1),
    ],
)
def test_dense_kv_segments(work, cus, expected):
    assert dense_kv_segments(work, cus) == expected


def _gfx950_grid_spec(sq, sk, hq, hkv=8, **over):
    req = attention.AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        mask_type=2,
        kv_block_size=64,
        dtype="bf16",
        num_cus=256,
        **over,
    )
    cand = next(
        c
        for c in attention.ATTENTION_EXECUTION_REGISTRY.candidates()
        if c.spec_id == "gfx950_dense_grid"
    )
    res = attention.dispatch_attention(
        replace(req, algorithm=cand.algorithm, spec_id=cand.spec_id)
    )
    return getattr(res.spec, "kernel_spec", res.spec)


@pytest.mark.parametrize(
    "sq,sk,hq,block_m,segments",
    [
        (512, 8192, 32, 128, 4),
        (256, 4096, 32, 128, 8),
        (2048, 8192, 32, 256, 1),
        (2048, 2048, 32, 256, 1),
    ],
)
def test_gfx950_short_bottom_right_chunk_splits_kv(sq, sk, hq, block_m, segments):
    spec = _gfx950_grid_spec(sq, sk, hq)
    assert (spec.block_m, spec.kv_segments) == (block_m, segments)


def test_gfx950_split_is_off_with_sinks():
    assert _gfx950_grid_spec(512, 8192, 32, use_sinks=True).kv_segments == 1
