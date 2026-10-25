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
        # Too few work items for 256 persistent CTAs.
        ({"seqlen_q": 512, "seqlen_k": 512}, "attention_gfx950_dense_grid"),
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
