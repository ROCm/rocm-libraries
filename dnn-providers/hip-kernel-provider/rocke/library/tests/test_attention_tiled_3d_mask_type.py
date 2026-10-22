# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``UnifiedAttention3DTiledSpec.mask_type()`` names the band the way the dispatcher's
``AttentionMaskType`` does, and agrees with the spec the dispatcher builds for each mask.

The ingestor catalog carries this value as its ``mask_type`` metadata, observed from the
compiled spec at pack time, so a mismatch ships a kernel under another mask's key.
"""

from __future__ import annotations

import pytest

from dispatch.attention import attention_candidates
from dispatch.attention.common import AttentionMaskType, AttentionRequest
from kernels.gfx950.attention_tiled_3d import UnifiedAttention3DTiledSpec

_SPEC_ID = "gfx950_u3d_splitkv_seg16_t1xb"


def _spec(**band) -> UnifiedAttention3DTiledSpec:
    return UnifiedAttention3DTiledSpec(
        head_size=128,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=False,
        sliding_window=band.pop("sliding_window", 0),
        has_softcap=False,
        num_segments=16,
        **band,
    )


@pytest.mark.parametrize(
    "band, expected",
    [
        ({"right_bound": -1}, 0),
        ({"right_bound": -1, "causal_top_left": True}, 0),
        ({"right_bound": 0, "causal_top_left": True}, 1),
        ({"right_bound": 0}, 2),
        ({}, 2),
        ({"right_bound": 3}, 3),
        ({"right_bound": 0, "sliding_window": 128}, 3),
    ],
)
def test_mask_type_names_the_band(band, expected):
    """Each band: not the neighbouring one, which is what a swapped alignment gives. The
    spec's default band is bottom-right causal."""
    assert _spec(**band).mask_type() == expected


def _request(mask, seqlen_q):
    candidate = next(
        c for c in attention_candidates() if getattr(c, "spec_id", "") == _SPEC_ID
    )
    request = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=seqlen_q,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        mask_type=mask,
        dtype="bf16",
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate, request


@pytest.mark.parametrize(
    "mask",
    [AttentionMaskType.TOP_LEFT_CAUSAL, AttentionMaskType.BOTTOM_RIGHT_CAUSAL],
)
def test_mask_type_matches_what_the_dispatcher_builds_for_decode(mask):
    candidate, request = _request(mask, seqlen_q=1)
    assert candidate.select_spec(request).kernel_spec.mask_type() == int(mask)


def test_decode_without_a_mask_is_the_bottom_right_causal_kernel():
    """One query row sees every key under either band, so the dispatcher builds NO_MASK
    decode as bottom-right causal; a catalog keyed on mask_type must match it there."""
    candidate, request = _request(AttentionMaskType.NO_MASK, seqlen_q=1)
    assert candidate.select_spec(request).kernel_spec.mask_type() == int(
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    )
