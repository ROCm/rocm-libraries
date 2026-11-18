# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx942 tiled 2D kernel bakes the GQA ratio, not the head counts or the batch.

The kernel reads ``num_query_heads`` and ``num_kv_heads`` as kernargs and derives its
``cu_q`` search trip count from ``num_seqs`` at run time, so specs that differ only in
those fields are one binary. The ingestor catalog relies on it: the engine manifest
declares them ``neutral_fields`` and keys its cells by
:meth:`UnifiedAttention2DTiledSpec.resolved_num_queries_per_kv`, one kernel per ratio.
The specs here are one tuning candidate per codepath (narrow, transposed x8, wide 32x8), built
through the dispatcher's ``select_spec``; the property is per codepath, not per shipped geometry.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention import attention_candidates
from dispatch.attention.common import AttentionMaskType, AttentionRequest
from kernels.gfx942.attention_tiled_2d import build_unified_attention_2d_tiled
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python

_NEUTRAL = {"num_query_heads", "num_kv_heads", "num_seqs"}

# One geometry per codepath, each legal at both head sizes.
_SPEC_IDS = (
    "gfx942_u2d_narrow_nw4_mw16_t2xb_llvm",
    "gfx942_u2d_transposed_x8_nw2_mw32_t2xb_llvm",
    "gfx942_u2d_wide32x8_nw1_mw32_t2xb_llvm",
)

# Two head pairs and two batches per GQA ratio.
_PAIRS = {
    1: ((8, 8, 4), (32, 32, 64)),
    4: ((8, 2, 4), (32, 8, 64)),
    16: ((16, 1, 4), (128, 8, 64)),
}


def _spec(spec_id: str, head_size: int, hq: int, hkv: int, batch: int):
    candidate = next(
        c for c in attention_candidates() if getattr(c, "spec_id", "") == spec_id
    )
    request = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=head_size,
        hdim_v=head_size,
        arch="gfx942",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        dtype="bf16",
        kv_block_size=16,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate.select_spec(request).kernel_spec


def _lower(spec, flavor: str) -> str:
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    return _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("ratio", sorted(_PAIRS))
@pytest.mark.parametrize("head_size", (64, 128))
@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_head_counts_and_batch_are_ir_neutral_at_a_fixed_gqa_ratio(
    spec_id, head_size, ratio, flavor
):
    (hq_a, hkv_a, b_a), (hq_b, hkv_b, b_b) = _PAIRS[ratio]
    a = _spec(spec_id, head_size, hq_a, hkv_a, b_a)
    b = _spec(spec_id, head_size, hq_b, hkv_b, b_b)
    changed = {
        f.name
        for f in dataclasses.fields(a)
        if getattr(a, f.name) != getattr(b, f.name)
    }
    # The pair differs in the neutral fields and nothing else, so equal IR proves
    # them neutral rather than unread.
    assert changed == _NEUTRAL
    assert a.resolved_num_queries_per_kv() == b.resolved_num_queries_per_kv() == ratio
    assert a.kernel_name() == b.kernel_name()
    assert _lower(a, flavor) == _lower(b, flavor)


@pytest.mark.parametrize("ratio", sorted(_PAIRS))
def test_resolved_num_queries_per_kv_is_the_baked_ratio(ratio):
    (hq, hkv, batch), _ = _PAIRS[ratio]
    spec = _spec(_SPEC_IDS[0], 128, hq, hkv, batch)
    resolved = spec.resolved_num_queries_per_kv()
    # hkp_pack records int metadata only as an exact int.
    assert type(resolved) is int and resolved == ratio == spec.num_queries_per_kv
    assert f"_g{ratio}_" in spec.kernel_name()


def test_ratios_split_ir():
    """The control: the ratio is baked, so two ratios are two binaries."""
    a = _spec(_SPEC_IDS[1], 128, 8, 8, 4)
    b = _spec(_SPEC_IDS[1], 128, 32, 8, 4)
    assert a.kernel_name() != b.kernel_name()
    assert _lower(a, LLVM_FLAVORS[0]) != _lower(b, LLVM_FLAVORS[0])
