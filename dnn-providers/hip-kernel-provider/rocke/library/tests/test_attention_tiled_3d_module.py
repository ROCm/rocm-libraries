# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``build_unified_attention_3d_tiled_module`` packs the split-KV segment kernel with the
reduce kernel the dispatcher pairs it with, and its predicate gates both.

hkp_pack compiles the module once and ships the reduce as the ``reduce`` launch beside the
segment launch. The reduce spec is derived from the segment spec (``reduce_spec_for``), so
a derivation that drifts from the dispatcher's own pairing ships a reduce that combines a
different number of segments, heads or head dims than the segment kernel wrote.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from dispatch.attention import attention_candidates
from dispatch.attention.common import AttentionMaskType, AttentionRequest
from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    build_unified_attention_3d_tiled_module,
    reduce_spec_for,
    supports_tiled_3d_spec,
    supports_unified_attention_3d_tiled_module,
)


def _gfx950_3d_candidates():
    return [
        c
        for c in attention_candidates()
        if str(getattr(c, "spec_id", "")).startswith("gfx950_u3d_")
    ]


def _request(candidate, *, hq, hkv, d, dtype, mask, batch=2, seqlen_k=4096):
    return AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=1,
        seqlen_k=seqlen_k,
        hdim_q=d,
        hdim_v=d,
        arch="gfx950",
        mask_type=mask,
        dtype=dtype,
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )


def test_every_gfx950_3d_candidate_is_compared():
    """The parity test below must not pass by finding no candidate."""
    assert len(_gfx950_3d_candidates()) == 32


@pytest.mark.parametrize(
    "heads, d, dtype, mask",
    list(
        itertools.product(
            [(8, 1), (16, 2), (32, 8)],
            [64, 128],
            ["bf16", "fp16"],
            [AttentionMaskType.TOP_LEFT_CAUSAL, AttentionMaskType.BOTTOM_RIGHT_CAUSAL],
        )
    ),
    ids=lambda v: str(v),
)
def test_derived_reduce_is_the_one_the_dispatcher_pairs(heads, d, dtype, mask):
    """For every gfx950 3D tuning candidate, ``reduce_spec_for`` of the segment spec the
    dispatcher builds equals the reduce spec it builds beside it."""
    hq, hkv = heads
    compared = 0
    for candidate in _gfx950_3d_candidates():
        request = _request(candidate, hq=hq, hkv=hkv, d=d, dtype=dtype, mask=mask)
        tuning = candidate.select_spec(request)
        if tuning is None:
            continue
        assert tuning.reduce_spec is not None, candidate.spec_id
        assert (
            reduce_spec_for(tuning.kernel_spec) == tuning.reduce_spec
        ), candidate.spec_id
        compared += 1
    assert compared, "no gfx950 3D candidate admitted the request"


def _segment(**overrides) -> UnifiedAttention3DTiledSpec:
    base = dict(
        head_size=128,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=16,
        right_bound=0,
    )
    return UnifiedAttention3DTiledSpec(**{**base, **overrides})


def test_module_predicate_admits_a_spec_both_halves_admit():
    spec = _segment()
    assert supports_tiled_3d_spec(spec, arch="gfx950")[0]
    assert supports_unified_attention_3d_tiled_module(spec, arch="gfx950") == (
        True,
        "supported",
    )


def test_module_predicate_refuses_what_only_the_reduce_refuses():
    """A segment count the segment kernel builds but whose reduce cannot cache its
    per-segment factors in LDS: the segment predicate alone would ship it."""
    spec = _segment(num_segments=4097)
    assert supports_tiled_3d_spec(spec, arch="gfx950")[0]
    ok, why = supports_unified_attention_3d_tiled_module(spec, arch="gfx950")
    assert not ok and "exceeds 4096" in why


def test_module_predicate_refuses_what_the_segment_refuses():
    spec = _segment(dtype="fp32")
    segment = supports_tiled_3d_spec(spec, arch="gfx950")
    assert not segment[0]
    assert supports_unified_attention_3d_tiled_module(spec, arch="gfx950") == segment


def test_module_exports_the_segment_as_primary_and_the_reduce_by_role():
    spec = _segment()
    module = build_unified_attention_3d_tiled_module(spec, arch="gfx950")
    assert set(module.companions) == {"reduce"}
    assert module.primary.name == spec.kernel_name()
    assert module.companions["reduce"].name == reduce_spec_for(spec).kernel_name()


def test_reduce_spec_follows_every_segment_field_it_reads():
    """Each field the reduce reads, changed on the segment, changes the derived reduce:
    a derivation that hard-codes one would pair every variant with the same reduce."""
    spec = _segment(waves_per_eu=2)
    base = reduce_spec_for(spec)
    for field, value in [
        ("head_size", 64),
        ("num_query_heads", 16),
        ("num_kv_heads", 1),
        ("dtype", "fp16"),
        ("num_segments", 32),
        ("waves_per_eu", 4),
    ]:
        changed = reduce_spec_for(dataclasses.replace(spec, **{field: value}))
        assert getattr(changed, field) == value and changed != base, field
