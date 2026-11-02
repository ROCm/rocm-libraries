# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``UnifiedAttention2DTiledSpec.mask_type()`` names the band the way the dispatcher's
``AttentionMaskType`` does, and agrees with the spec the dispatcher builds for each mask.

The ingestor catalog carries this value as its ``mask_type`` metadata, observed from the
compiled spec at pack time, so a mismatch ships a kernel under another mask's key.
"""

from __future__ import annotations

import pytest

from dispatch.attention import attention_candidates
from dispatch.attention.common import AttentionMaskType, AttentionRequest
from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec

_SPEC_ID = "gfx950_u2d_narrow_nw4_mw16_t4xb_llvm"


def _spec(**band) -> UnifiedAttention2DTiledSpec:
    return UnifiedAttention2DTiledSpec(
        head_size=128,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=False,
        sliding_window=band.pop("sliding_window", 0),
        has_softcap=False,
        **band,
    )


@pytest.mark.parametrize(
    "band, expected",
    [
        ({"right_bound": -1}, 0),
        ({"right_bound": -1, "causal_top_left": True}, 0),
        ({"right_bound": 0, "causal_top_left": True}, 1),
        ({"right_bound": 0}, 2),
        ({"right_bound": 3}, 3),
        ({"right_bound": 0, "sliding_window": 128}, 3),
        ({"right_bound": -1, "sliding_window": 128}, 3),
    ],
)
def test_mask_type_names_the_band(band, expected):
    """Each band: not the neighbouring one, which is what a swapped alignment gives."""
    assert _spec(**band).mask_type() == expected


@pytest.mark.parametrize(
    "mask",
    [
        AttentionMaskType.NO_MASK,
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ],
)
def test_mask_type_matches_what_the_dispatcher_builds(mask):
    """The registry's spec for a request carries the same value as the request's mask."""
    candidate = next(
        c for c in attention_candidates() if getattr(c, "spec_id", "") == _SPEC_ID
    )
    request = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=128,
        seqlen_k=256,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        mask_type=mask,
        dtype="bf16",
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    assert candidate.select_spec(request).kernel_spec.mask_type() == int(mask)
