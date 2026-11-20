# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``build_unified_attention_3d_tiled_module`` packs the split-KV segment kernel with the
reduce kernel the dispatcher pairs it with, and its predicate gates both.

hkp_pack compiles the module once and ships the reduce as the ``reduce`` launch beside the
segment launch. The reduce spec is derived from the segment spec (``reduce_spec_for``), and
the dispatcher builds its pair through the same derivation
(``kernels.common.attention_split_kv_reduce.split_kv_reduce_spec``). The round trip below
pins that the module packs the reduce the dispatcher launches: otherwise it would combine a
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


def _request(
    candidate, *, hq, hkv, d, dtype, mask, batch=2, seqlen_k=4096, emit_lse=False
):
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
        emit_lse=emit_lse,
    )


def test_every_gfx950_3d_candidate_is_compared():
    """The parity test below must not pass by finding no candidate."""
    assert len(_gfx950_3d_candidates()) == 32


@pytest.mark.parametrize(
    "heads, d, dtype, mask, emit_lse",
    list(
        itertools.product(
            [(8, 1), (16, 2), (32, 8)],
            [64, 128],
            ["bf16", "fp16"],
            [AttentionMaskType.TOP_LEFT_CAUSAL, AttentionMaskType.BOTTOM_RIGHT_CAUSAL],
            [False, True],
        )
    ),
    ids=lambda v: str(v),
)
def test_derived_reduce_is_the_one_the_dispatcher_pairs(
    heads, d, dtype, mask, emit_lse
):
    """For every gfx950 3D tuning candidate, ``reduce_spec_for`` of the segment spec the
    dispatcher builds (with the request's ``emit_lse``, a problem field) equals the
    reduce spec it builds beside it."""
    hq, hkv = heads
    compared = 0
    for candidate in _gfx950_3d_candidates():
        request = _request(
            candidate, hq=hq, hkv=hkv, d=d, dtype=dtype, mask=mask, emit_lse=emit_lse
        )
        tuning = candidate.select_spec(request)
        if tuning is None:
            continue
        assert tuning.reduce_spec is not None, candidate.spec_id
        assert tuning.reduce_spec.emit_lse is emit_lse, candidate.spec_id
        assert (
            reduce_spec_for(tuning.kernel_spec, emit_lse=emit_lse) == tuning.reduce_spec
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
    from kernels.gfx950.attention_tiled_3d import reduce_max_segments

    spec = _segment(num_segments=reduce_max_segments("gfx950") + 1)
    assert supports_tiled_3d_spec(spec, arch="gfx950")[0]
    ok, why = supports_unified_attention_3d_tiled_module(spec, arch="gfx950")
    assert not ok and "LDS" in why


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


def test_reduce_spec_takes_an_explicit_emit_lse_from_the_caller():
    """The dispatcher's explicit value wins over the segment's, both ways, and is the
    only difference."""
    spec = _segment(waves_per_eu=2)
    assert reduce_spec_for(spec).emit_lse is False
    with_lse = reduce_spec_for(spec, emit_lse=True)
    assert with_lse == dataclasses.replace(reduce_spec_for(spec), emit_lse=True)
    on = _segment(waves_per_eu=2, emit_lse=True)
    assert reduce_spec_for(on, emit_lse=False) == reduce_spec_for(spec)


def test_reduce_spec_defaults_to_the_segment_emit_lse():
    """A packaged module is built from the segment spec alone."""
    assert reduce_spec_for(_segment(emit_lse=True)).emit_lse is True
    assert reduce_spec_for(_segment()).emit_lse is False


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_segment_spec_rejects_non_boolean_emit_lse(bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _segment(emit_lse=bad)


def test_segment_name_gains_lse_suffix_only_when_on():
    off = _segment()
    assert _segment(emit_lse=False).kernel_name() == off.kernel_name()
    assert not off.kernel_name().endswith("_lse")
    assert _segment(emit_lse=True).kernel_name() == off.kernel_name() + "_lse"


def _ir(kernel, name=None):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    if name is not None:
        kernel.name = name
    return lower_kernel_to_llvm(kernel, arch="gfx950")


@pytest.mark.parametrize("num_segments", [16, 64])
def test_lse_module_keeps_the_segment_ir_and_packs_the_lse_reduce(num_segments):
    """Only the reduce changes: the segment IR is the LSE-off IR (bar its name), and
    the reduce is the LSE reduce the dispatcher launches for an LSE request."""
    from kernels.gfx950.attention_tiled_3d import build_unified_attention_reduce_tiled

    off_spec = _segment(num_segments=num_segments)
    on_spec = _segment(num_segments=num_segments, emit_lse=True)
    off = build_unified_attention_3d_tiled_module(off_spec, arch="gfx950")
    on = build_unified_attention_3d_tiled_module(on_spec, arch="gfx950")
    assert on.primary.name == on_spec.kernel_name()
    assert _ir(on.primary, off.primary.name) == _ir(off.primary)
    reduce = on.companions["reduce"]
    assert reduce.name == reduce_spec_for(off_spec, emit_lse=True).kernel_name()
    assert reduce.params[-1].name == "lse_ptr"
    expected = build_unified_attention_reduce_tiled(
        reduce_spec_for(off_spec, emit_lse=True), arch="gfx950"
    )
    assert _ir(reduce) == _ir(expected)
    assert "lse_ptr" not in _ir(off.companions["reduce"])


def test_module_predicate_admits_the_lse_module():
    assert supports_unified_attention_3d_tiled_module(
        _segment(emit_lse=True), arch="gfx950"
    ) == (True, "supported")
