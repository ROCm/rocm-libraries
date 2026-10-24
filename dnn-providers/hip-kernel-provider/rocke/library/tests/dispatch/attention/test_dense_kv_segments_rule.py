# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dense short-chunk rule: KV-segment count for an underfilled default grid."""

from dataclasses import replace

import pytest

import dispatch.attention as attention
from dispatch.attention.dense_rules import dense_kv_segments, dense_wave_segments


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


def _gfx942_spec(sq, sk, hq, hkv=8, mask_type=2, **over):
    req = attention.AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        mask_type=mask_type,
        kv_block_size=64,
        dtype="bf16",
        num_cus=304,
        **over,
    )
    return attention.attention_tuning_spec(req, "gfx942_dense").kernel_spec


@pytest.mark.parametrize(
    "sq,sk,hq,block_m,persistent,segments",
    [
        (512, 8192, 32, 256, False, 4),
        (256, 4096, 32, 256, False, 8),
        (2048, 8192, 32, 256, True, 1),
        (2048, 32768, 40, 256, False, 4),
    ],
)
def test_gfx942_bottom_right_geometry(sq, sk, hq, block_m, persistent, segments):
    spec = _gfx942_spec(sq, sk, hq)
    assert spec.causal_bottom_right and spec.causal_diag_split
    assert (spec.block_m, spec.persistent, spec.kv_segments) == (
        block_m,
        persistent,
        segments,
    )


def test_gfx942_equal_length_keeps_default_geometry():
    spec = _gfx942_spec(2048, 2048, 32, mask_type=1)
    assert not spec.causal_bottom_right and not spec.causal_diag_split
    assert spec.kv_segments == 1 and spec.persistent


@pytest.mark.parametrize(
    "work,cus,expected",
    [
        (64, 304, 4),
        (56, 304, 4),
        (32, 304, 8),
        (80, 304, 8),
        (256, 304, 1),
        (320, 304, 4),
        (512, 304, 1),
        (304, 304, 1),
        (0, 304, 1),
    ],
)
def test_dense_wave_segments(work, cus, expected):
    assert dense_wave_segments(work, cus) == expected
