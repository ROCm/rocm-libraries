# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Inventory of the classes that carry an ``emit_lse`` field (host only).

Each carrier refuses a non-bool value in ``__post_init__`` through ``require_bool``,
and its own test pins that refusal. This test fails when a class reachable from the
dispatcher gains an ``emit_lse`` field without being listed here, so the new guard
and its refusal test are not forgotten.
"""

import dataclasses

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    attention_problem,
    dispatch_attention,
)

_ARCHES = ("gfx942", "gfx950")

# Every class with an ``emit_lse`` field and its own guard.
_EXPECTED_CLASSES = {
    "dispatch.attention.common.AttentionSpec",
    "kernels.common.attention_unified.UnifiedAttentionProblem",
    "kernels.gfx942.attention_dense.Gfx942AttentionDenseSpec",
    "kernels.gfx942.attention_tiled_2d.UnifiedAttention2DTiledSpec",
    "kernels.gfx942.attention_tiled_3d.UnifiedAttentionReduceTiledSpec",
    "kernels.gfx950.attention_dense.Gfx950AttentionDenseSpec",
    "kernels.gfx950.attention_tiled_2d.UnifiedAttention2DTiledSpec",
    "kernels.gfx950.attention_tiled_3d.UnifiedAttention3DTiledSpec",
    "kernels.gfx950.attention_tiled_3d.UnifiedAttentionReduceTiledSpec",
}

_DENSE_SPEC = {"gfx942": "gfx942_dense", "gfx950": "gfx950_dense_grid"}


def _unified_request(arch, **overrides):
    """One-token decode over a long paged context: routes to split-KV (3D)."""
    fields = dict(
        batch=4,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        mask_type=2,  # BOTTOM_RIGHT_CAUSAL
        dtype="bf16",
        emit_lse=True,
    )
    fields.update(overrides)
    return AttentionRequest(**fields)


def _dense_request(arch, spec_id):
    candidate = next(
        c for c in attention_execution_candidates() if c.spec_id == spec_id
    )
    return AttentionRequest(
        batch=2,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        mask_type=1,
        dtype="bf16",
        emit_lse=True,
        algorithm=candidate.algorithm,
        spec_id=spec_id,
    )


def _tuning(req, spec_id):
    return dataclasses.replace(req, algorithm="unified_tuning", spec_id=spec_id)


def _requests(arch):
    """One request per route that returns a different ``emit_lse`` class."""
    prefill = _unified_request(arch, batch=1, seqlen_q=256, seqlen_k=256)
    return [
        _unified_request(arch),  # unpinned: AttentionSpec
        _tuning(_unified_request(arch), f"{arch}_u3d_splitkv_seg16_t1xb"),
        _tuning(prefill, f"{arch}_u2d_narrow_nw1_mw16_t1xb_llvm"),
        _dense_request(arch, _DENSE_SPEC[arch]),
    ]


def _carriers(obj):
    """``obj`` and every dataclass nested in it that has an ``emit_lse`` field."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return
    fields = dataclasses.fields(obj)
    if any(f.name == "emit_lse" for f in fields):
        yield obj
    for f in fields:
        yield from _carriers(getattr(obj, f.name))


def _qualified_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


def test_every_emit_lse_carrier_is_listed():
    """A new class with an ``emit_lse`` field must be added to the list above."""
    found = []
    for arch in _ARCHES:
        for req in _requests(arch):
            found.extend(_carriers(dispatch_attention(req).spec))
        found.append(attention_problem(_unified_request(arch)))
    assert {_qualified_name(c) for c in found} == _EXPECTED_CLASSES
