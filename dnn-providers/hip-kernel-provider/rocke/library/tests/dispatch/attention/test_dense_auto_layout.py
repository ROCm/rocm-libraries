# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Auto dispatch of contiguous-layout (``layout='dense'``) and paged-layout
(``layout='paged'``) requests (host only).

An unpinned request that declares contiguous tensors and has more than one
query row gets the dense kernel's default body, and single-row requests take
the strided decode kernel. A paged request goes to a dense paged body where its
arch's paged order takes it and falls back to unified elsewhere; an unspecified
layout keeps the unified route.
"""

import dataclasses

import pytest

from dispatch.attention import AttentionRequest, dispatch_attention


def _request(arch, **overrides):
    fields = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=4096,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        mask_type=1,
        dtype="bf16",
        kv_block_size=64,
        num_cus=256 if arch == "gfx950" else 304,
        layout="dense",
    )
    fields.update(overrides)
    return AttentionRequest(**fields)


@pytest.mark.parametrize(
    "overrides, candidate",
    [
        # Enough work for the persistent grid, plain causal D128 tile multiples.
        ({}, "attention_gfx950_dense_persist_widedma"),
        # Persistent, but a window rules out wide DMA.
        ({"sliding_window": 1024}, "attention_gfx950_dense_persist"),
        ({"use_sinks": True}, "attention_gfx950_dense_persist"),
        ({"hdim_q": 64, "hdim_v": 64}, "attention_gfx950_dense_persist"),
        # Too few work items for 256 persistent CTAs: plain causal D128 still
        # takes wide DMA (at the half-height tile); anything else takes grid.
        ({"seqlen_q": 512, "seqlen_k": 512}, "attention_gfx950_dense_persist_widedma"),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "sliding_window": 128},
            "attention_gfx950_dense_grid",
        ),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "hdim_q": 64, "hdim_v": 64},
            "attention_gfx950_dense_grid",
        ),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "mask_type": 0},
            "attention_gfx950_dense_grid",
        ),
        # The moving bottom-right diagonal runs only on the grid body.
        (
            {"seqlen_q": 512, "seqlen_k": 8192, "mask_type": 2},
            "attention_gfx950_dense_grid",
        ),
    ],
)
def test_gfx950_dense_layout_picks_default_body(overrides, candidate):
    result = dispatch_attention(_request("gfx950", **overrides))
    assert result.candidate.name == candidate
    assert result.spec.path == "dense"
    # The selection is the candidate's default spec, reproducible by pinning it.
    pinned = dispatch_attention(
        dataclasses.replace(
            _request("gfx950", **overrides),
            algorithm=result.candidate.algorithm,
            spec_id=result.candidate.spec_id,
        )
    )
    assert pinned.spec.kernel_spec == result.spec.kernel_spec


@pytest.mark.parametrize(
    "overrides, candidate",
    [
        # Each body's own admission sends what it refuses down the order.
        ({}, "attention_gfx950_dense_persist_widedma"),
        # Wide DMA needs a page per KV tile at least.
        ({"kv_block_size": 16}, "attention_gfx950_dense_persist"),
        ({"sliding_window": 1024}, "attention_gfx950_dense_persist_widedma"),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "sliding_window": 128},
            "attention_gfx950_dense_persist_widedma",
        ),
        (
            {"sliding_window": 1024, "kv_block_size": 16},
            "attention_gfx950_dense_persist",
        ),
        ({"sliding_window": 1024, "use_sinks": True}, "attention_gfx950_dense_persist"),
        ({"sliding_window": 1024, "batch": 4}, "attention_gfx950_dense_grid"),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "hdim_q": 64, "hdim_v": 64},
            "attention_gfx950_dense_persist",
        ),
        (
            {
                "seqlen_q": 512,
                "seqlen_k": 512,
                "hdim_q": 64,
                "hdim_v": 64,
                "sliding_window": 128,
                "kv_block_size": 16,
            },
            "attention_gfx950_dense_persist",
        ),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "hdim_q": 64, "hdim_v": 64, "batch": 4},
            "attention_gfx950_dense_grid",
        ),
        # The moving bottom-right diagonal runs only on the grid body.
        (
            {
                "seqlen_q": 512,
                "seqlen_k": 8192,
                "hdim_q": 64,
                "hdim_v": 64,
                "mask_type": 2,
            },
            "attention_gfx950_dense_grid",
        ),
        # Sinks rule out wide DMA, as for contiguous K/V.
        ({"use_sinks": True}, "attention_gfx950_dense_persist"),
        # The persistent bodies take one sequence.
        ({"batch": 4}, "attention_gfx950_dense_grid"),
        ({"batch": 4, "use_sinks": True}, "attention_gfx950_dense_grid"),
        ({"seqlen_q": 512, "seqlen_k": 512}, "attention_gfx950_dense_persist_widedma"),
        (
            {"seqlen_q": 512, "seqlen_k": 512, "kv_block_size": 16},
            "attention_gfx950_dense_grid",
        ),
    ],
)
def test_gfx950_paged_layout_picks_default_body(overrides, candidate):
    req = _request("gfx950", layout="paged", **overrides)
    result = dispatch_attention(req)
    assert result.candidate.name == candidate
    assert result.spec.path == "dense"
    pinned = dispatch_attention(
        dataclasses.replace(
            req, algorithm=result.candidate.algorithm, spec_id=result.candidate.spec_id
        )
    )
    assert pinned.spec.kernel_spec == result.spec.kernel_spec


@pytest.mark.parametrize(
    "overrides, persistent",
    [
        ({}, True),
        # Plain causal fp16 runs faster on the grid, windowed fp16 does not;
        # narrow windows keep their grid geometry.
        ({"dtype": "fp16"}, False),
        ({"dtype": "fp16", "sliding_window": 4096}, True),
        # Sinks leave the choice as it is without them.
        ({"use_sinks": True}, True),
        ({"use_sinks": True, "dtype": "fp16"}, False),
        ({"use_sinks": True, "batch": 4}, False),
        ({"dtype": "fp16", "sliding_window": 256}, False),
        ({"sliding_window": 128}, False),
        ({"hdim_q": 64, "hdim_v": 64}, False),
        # The persistent body takes one sequence on whole KV tiles.
        ({"batch": 4}, False),
        ({"seqlen_q": 4000, "seqlen_k": 4000}, False),
        ({"batch": 4, "sliding_window": 1024, "hdim_q": 64, "hdim_v": 64}, False),
    ],
)
def test_gfx942_paged_layout_picks_default_body(overrides, persistent):
    req = _request("gfx942", layout="paged", **overrides)
    result = dispatch_attention(req)
    assert result.candidate.name == "attention_gfx942_dense"
    assert ("_persist" in result.spec.kernel_name()) == persistent
    pinned = dispatch_attention(
        dataclasses.replace(
            req, algorithm=result.candidate.algorithm, spec_id=result.candidate.spec_id
        )
    )
    assert pinned.spec.kernel_spec == result.spec.kernel_spec


@pytest.mark.parametrize("use_sinks", [False, True])
def test_gfx942_paged_several_windowed_d128_sequences_take_dense(use_sinks):
    result = dispatch_attention(
        _request(
            "gfx942", layout="paged", batch=4, sliding_window=1024, use_sinks=use_sinks
        )
    )
    assert result.candidate.name == "attention_gfx942_dense"


def test_gfx942_dense_layout_picks_dense():
    result = dispatch_attention(_request("gfx942"))
    assert result.candidate.name == "attention_gfx942_dense"
    assert result.spec.path == "dense"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_unspecified_layout_keeps_unified(arch):
    result = dispatch_attention(_request(arch, layout="auto"))
    assert result.spec.path in ("2d", "3d")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_single_query_row_is_routed_to_strided_decode(arch):
    result = dispatch_attention(_request(arch, seqlen_q=1, mask_type=0))
    assert result.candidate.name == "attention_strided_decode"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_dense_refusal_does_not_fall_back_to_another_kernel(arch):
    # The dense kernels have no head-size-256 body, and no other CDNA kernel
    # serves contiguous tensors.
    with pytest.raises(ValueError):
        dispatch_attention(_request(arch, hdim_q=256, hdim_v=256))


def test_explicit_pin_is_unchanged():
    req = _request(
        "gfx950", algorithm="attention_dense_grid", spec_id="gfx950_dense_grid"
    )
    assert dispatch_attention(req).candidate.name == "attention_gfx950_dense_grid"


def test_fp8_dense_layout_auto_pins_to_the_grid_body():
    """An unpinned fp8 request that declares contiguous tensors gets the dense
    fp8 kernel.

    This IS a routing change: before fp8 was a declared feature the same request
    was refused outright, because no candidate served it. It is confined to
    ``layout='dense'`` (see the companion test), where asking for contiguous
    tensors already means asking for the dense kernel. fp8 is grid-only, so the
    persistent bodies the bf16 request would win do not admit it.
    """
    result = dispatch_attention(_request("gfx950", use_fp8=True))
    assert result.candidate.name == "attention_gfx950_dense_grid"
    assert result.spec.kernel_spec.kv_storage_dtype == "fp8e4m3"


def test_fp8_keeps_the_unified_route_off_the_dense_layout():
    """The fp8 auto-pin above is scoped to ``layout='dense'``.

    The dense paged body has no fp8 K/V, so a paged fp8 request falls back to
    unified too.
    """
    for layout in ("auto", "paged"):
        result = dispatch_attention(_request("gfx950", use_fp8=True, layout=layout))
        assert result.candidate.name == "attention_unified_2d"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "overrides",
    [
        # Paged dense is causal-only, with or without sinks.
        pytest.param({"use_sinks": True, "mask_type": 0}, id="non_causal_sinks"),
        pytest.param({"seqlen_q": 1, "mask_type": 0}, id="decode"),
        pytest.param({"batch": 4, "emit_lse": True}, id="multiseq_lse"),
    ],
)
def test_paged_outside_dense_falls_back_to_unified(arch, overrides):
    result = dispatch_attention(_request(arch, layout="paged", **overrides))
    assert result.candidate.name in ("attention_unified_2d", "attention_unified_3d")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "lengths",
    [{"seqlen_q": 512}, {"seqlen_k": 1024}],
    ids=["q_lt_kv", "q_gt_kv"],
)
def test_paged_top_left_cross_length_routes_to_dense(arch, lengths):
    # gfx942 unified folds top-left into bottom-right (normalize_band), so dense is
    # the only correct body there.
    result = dispatch_attention(_request(arch, layout="paged", batch=4, **lengths))
    spec = result.spec.kernel_spec
    assert spec.paged and spec.varlen and not spec.causal_bottom_right


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_page_size_no_kernel_serves_is_refused(arch):
    # Page size 8: below dense's minimum and outside unified's block sizes.
    with pytest.raises(ValueError):
        dispatch_attention(_request(arch, layout="paged", kv_block_size=8))
