# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``UnifiedAttention2DTiledSpec.mask_type()`` names the band by the ``AttentionMaskType``
ordinals, and the spec the dispatcher builds for each mask round-trips to that mask.

The ordinals and the band -> kind mapping live once, in
``kernels.common.attention_mask_type`` (the dispatcher re-exports the enum; both tiled
specs' ``mask_type()`` delegate to ``band_mask_type``). The table below pins the mapping
through the 2D spec. The ingestor catalog carries this value as its ``mask_type``
metadata, observed from the compiled spec at pack time, so a mismatch ships a kernel
under another mask's key.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention import attention_candidates
from dispatch.attention.common import AttentionMaskType, AttentionRequest
from kernels.common import attention_mask_type
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942Spec,
)
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
    """Each band: not the neighbouring one, which is what a swapped alignment gives. An
    exact ``int``: hkp_pack accepts no other type (an ``IntEnum`` member included)."""
    mask = _spec(**band).mask_type()
    assert type(mask) is int and mask == expected


def test_mask_ordinals_have_one_home():
    """The dispatcher's mask kinds are the kernels' enum, re-exported rather than
    redeclared, and keep the ``plan_utils::MaskType`` ordinals."""
    assert AttentionMaskType is attention_mask_type.AttentionMaskType
    assert [(m.name, int(m)) for m in AttentionMaskType] == [
        ("NO_MASK", 0),
        ("TOP_LEFT_CAUSAL", 1),
        ("BOTTOM_RIGHT_CAUSAL", 2),
        ("SLIDING_WINDOW", 3),
    ]


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


# gfx942: the tiled 2D body computes the bottom-right causal band only, so its
# mask_type() is bottom-right causal or, with a window, a sliding window.

_GFX942_SPEC_ID = "gfx942_u2d_transposed_x8_nw4_mw32_t4xb_llvm"


def _gfx942_spec(**band) -> Gfx942Spec:
    return Gfx942Spec(
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


@pytest.mark.parametrize("window, expected", [(0, 2), (128, 3)])
def test_gfx942_mask_type_names_the_band(window, expected):
    mask = _gfx942_spec(sliding_window=window).mask_type()
    assert type(mask) is int and mask == expected


@pytest.mark.parametrize(
    "band",
    [
        {"right_bound": -1},
        {"right_bound": -1, "causal_top_left": True},
        {"right_bound": 0, "causal_top_left": True},
        {"right_bound": 0},
        {"right_bound": 3},
        {"right_bound": 0, "sliding_window": 128},
        {"right_bound": -1, "sliding_window": 128},
    ],
)
def test_gfx942_mask_type_reads_every_band_field_the_spec_has(band):
    """Each band field the gfx942 spec declares must reach ``mask_type()``.

    Today the spec has none of ``right_bound`` / ``causal_top_left``, so only the
    window varies. A spec that gains them while ``mask_type()`` still assumes the
    bottom-right causal band would ship top-left and no-mask kernels under the
    bottom-right key; this table then fails.
    """
    declared = {f.name for f in dataclasses.fields(Gfx942Spec)}
    kept = {k: v for k, v in band.items() if k in declared or k == "sliding_window"}
    try:
        spec = _gfx942_spec(**dict(kept))
    except ValueError:
        pytest.skip(f"the gfx942 spec refuses {kept}")
    expected = attention_mask_type.band_mask_type(
        sliding_window=kept.get("sliding_window", 0),
        right_bound=kept.get("right_bound", 0),
        causal_top_left=kept.get("causal_top_left", False),
    )
    assert spec.mask_type() == expected


def test_gfx942_mask_type_matches_what_the_dispatcher_builds():
    candidate = next(
        c
        for c in attention_candidates()
        if getattr(c, "spec_id", "") == _GFX942_SPEC_ID
    )
    request = AttentionRequest(
        batch=1,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=128,
        seqlen_k=256,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        dtype="bf16",
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    spec = candidate.select_spec(request).kernel_spec
    assert spec.mask_type() == int(AttentionMaskType.BOTTOM_RIGHT_CAUSAL)
