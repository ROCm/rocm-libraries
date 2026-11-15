# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every attention spec or problem that carries ``emit_lse`` refuses a non-bool value.

``bool("false")`` is ``True``, so a truthy non-bool must not turn the LSE output on:
it would change the kernel's identity without the caller asking for it. The guard
lives in several classes, and a direct caller of a public spec class reaches each one
without passing the request validator. This test takes an instance of each class from
the dispatcher, rebuilds it with a non-bool ``emit_lse`` and expects the class itself
to refuse, so a class that loses its guard, or a new class that never gets one, fails
here. The request-level refusal is pinned in ``test_unified_lse_dispatch.py`` and
``test_dense_lse_dispatch.py`` (host only).
"""

import dataclasses

import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    attention_problem,
    dispatch_attention,
)

_ARCHES = ("gfx942", "gfx950")
_BAD_VALUES = ["false", 0, 1, None]

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


_DENSE_SPEC = {"gfx942": "gfx942_dense", "gfx950": "gfx950_dense_grid"}


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


def _all_carriers():
    found = []
    for arch in _ARCHES:
        for req in _requests(arch):
            found.extend(_carriers(dispatch_attention(req).spec))
        found.append(attention_problem(_unified_request(arch)))
    return found


def _qualified_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


def test_every_emit_lse_carrier_is_covered():
    """A new class with an ``emit_lse`` field must be added to this test, with its guard."""
    assert {_qualified_name(c) for c in _all_carriers()} == _EXPECTED_CLASSES


@pytest.mark.parametrize("bad", _BAD_VALUES, ids=repr)
@pytest.mark.parametrize("name", sorted(_EXPECTED_CLASSES))
def test_emit_lse_rejects_non_bool_values(name, bad):
    carriers = [c for c in _all_carriers() if _qualified_name(c) == name]
    assert carriers, f"no instance of {name} was reachable"
    for carrier in carriers:
        with pytest.raises(
            ValueError, match=f"^emit_lse must be a bool, got {type(bad).__name__}$"
        ):
            dataclasses.replace(carrier, emit_lse=bad)
