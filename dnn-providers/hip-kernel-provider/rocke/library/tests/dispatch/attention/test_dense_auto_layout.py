# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Auto dispatch of contiguous-layout (``layout='dense'``) requests (host only).

An unpinned request that declares contiguous tensors and has more than one
query row gets the dense kernel's default body; paged / unspecified layouts keep
the unified route, and single-row requests take the strided decode kernel.
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

    Only ``auto`` is checked: ``paged`` and the unset layout are refused for
    every dtype on this path, so they say nothing about fp8 either way.
    """
    result = dispatch_attention(_request("gfx950", use_fp8=True, layout="auto"))
    assert result.candidate.name == "attention_unified_2d"
