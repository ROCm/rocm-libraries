# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for the optional dense LSE output (host only)."""

import dataclasses

import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    dispatch_attention,
    dispatch_attention_all,
)

_DENSE_SPEC_IDS = {
    "gfx942": ("gfx942_dense",),
    "gfx950": (
        "gfx950_dense_grid",
        "gfx950_dense_persist",
        "gfx950_dense_persist_widedma",
    ),
}
_CASES = [(arch, sid) for arch, ids in _DENSE_SPEC_IDS.items() for sid in ids]


def _request(arch, **overrides):
    fields = dict(
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
    )
    fields.update(overrides)
    return AttentionRequest(**fields)


def _pinned(req, spec_id):
    candidate = next(
        c for c in attention_execution_candidates() if c.spec_id == spec_id
    )
    return dataclasses.replace(req, algorithm=candidate.algorithm, spec_id=spec_id)


@pytest.mark.parametrize("bad_emit_lse", ["false", 0, 1, None])
def test_emit_lse_rejects_non_boolean_values(bad_emit_lse):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        dispatch_attention(_request("gfx950", emit_lse=bad_emit_lse))


@pytest.mark.parametrize("arch", sorted(_DENSE_SPEC_IDS))
def test_unpinned_lse_request_is_refused_not_served_without_lse(arch):
    """No default (unified) kernel writes LSE, so ``auto`` must fail loudly
    instead of returning a kernel that silently drops the requested output."""
    dispatch_attention(_request(arch))  # the same problem without LSE routes
    with pytest.raises(ValueError, match="lse"):
        dispatch_attention(_request(arch, emit_lse=True))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_pinned_dense_spec_follows_the_request(arch, spec_id):
    on = dispatch_attention(_pinned(_request(arch, emit_lse=True), spec_id))
    off = dispatch_attention(_pinned(_request(arch), spec_id))
    assert on.spec.kernel_spec.emit_lse is True
    assert off.spec.kernel_spec.emit_lse is False
    # Distinct binaries: an LSE kernel takes one more pointer argument.
    assert on.spec.kernel_spec.kernel_name() != off.spec.kernel_spec.kernel_name()


@pytest.mark.parametrize("arch", sorted(_DENSE_SPEC_IDS))
@pytest.mark.parametrize("emit_lse", [False, True])
def test_dense_sweep_keeps_the_requested_output(arch, emit_lse):
    """A tuning sweep varies performance knobs only; it must never add or drop
    the LSE output the caller asked for."""
    results = [
        r
        for r in dispatch_attention_all(
            _request(arch, emit_lse=emit_lse), candidate_prefix="attention_"
        )
        if hasattr(r.spec, "kernel_spec") and r.spec.path == "dense"
    ]
    assert results, "no dense configuration was enumerated"
    assert {r.spec.kernel_spec.emit_lse for r in results} == {emit_lse}
