# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for dense attention dropout: the request field, the dense
candidates and their specs, and the dense binding's dropout inputs (host only)."""

import dataclasses
import math

import numpy as np
import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    dispatch_attention,
    dispatch_attention_all,
)
from dispatch.attention import bindings
from dispatch.attention.dense_rules import dense_axes_for

_DENSE_SPEC_IDS = {
    "gfx942": (
        "gfx942_dense",
        "gfx942_dense_qbminor",
        "gfx942_dense_persist_aot",
        "gfx942_dense_btminor_aot",
        "gfx942_dense_hqswz_aot",
    ),
    "gfx950": (
        "gfx950_dense_grid",
        "gfx950_dense_qbminor",
        "gfx950_dense_persist",
        "gfx950_dense_persist_widedma",
    ),
}
_CASES = [(arch, sid) for arch, ids in _DENSE_SPEC_IDS.items() for sid in ids]
_ARCHES = sorted(_DENSE_SPEC_IDS)


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


def _chunk_request(arch, **overrides):
    """A short bottom-right query chunk over a long KV prefix: with dropout off
    the dense base spec splits each work item's KV range across CTAs."""
    return _request(
        arch,
        batch=1,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=8192,
        mask_type=2,
        layout="dense",
        **overrides,
    )


def _pinned(req, spec_id):
    candidate = next(
        c for c in attention_execution_candidates() if c.spec_id == spec_id
    )
    return dataclasses.replace(req, algorithm=candidate.algorithm, spec_id=spec_id)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, math.nan, math.inf, True, "0.1", None])
def test_dropout_p_outside_its_range_is_refused(bad):
    with pytest.raises(
        ValueError, match=r"dropout_p=.*expected a finite number in \[0, 1\)"
    ):
        _request("gfx950", dropout_p=bad)


def test_dropout_p_takes_a_numpy_scalar():
    req = _request("gfx950", dropout_p=np.float32(0.25))
    assert req.dropout_p == 0.25 and type(req.dropout_p) is float


def test_dropout_feature_and_request_identity():
    """Dropout is reported only when on, and an off request keeps the
    ``normalized()`` payload (so the request_hash) it had before the field."""
    off = _request("gfx950")
    on = _request("gfx950", dropout_p=0.1)
    assert "dropout" not in off.features() and "dropout" in on.features()
    assert "dropout_p" not in off.normalized()
    assert on.normalized()["dropout_p"] == 0.1
    assert _request("gfx950", dropout_p=0).dropout_p == 0.0


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_pinned_dense_spec_follows_the_request(arch, spec_id):
    on = dispatch_attention(_pinned(_request(arch, dropout_p=0.1), spec_id))
    off = dispatch_attention(_pinned(_request(arch), spec_id))
    assert on.spec.kernel_spec.dropout is True
    assert off.spec.kernel_spec.dropout is False
    # Distinct binaries: the dropout kernel takes the dropout arguments.
    assert on.spec.kernel_spec.kernel_name() != off.spec.kernel_spec.kernel_name()
    # Same configuration: dropout is the caller's, not a knob, so a stored pin
    # serves both.
    assert on.spec.config_key == off.spec.config_key
    assert on.spec.tuning_id == off.spec.tuning_id


@pytest.mark.parametrize("arch", _ARCHES)
def test_dense_layout_request_is_served_by_dense_with_dropout(arch):
    result = dispatch_attention(_request(arch, layout="dense", dropout_p=0.1))
    assert result.spec.path == "dense"
    assert result.spec.kernel_spec.dropout is True


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_keeps_one_kv_segment(arch):
    off = dispatch_attention(_chunk_request(arch))
    on = dispatch_attention(_chunk_request(arch, dropout_p=0.1))
    assert off.spec.kernel_spec.kv_segments > 1
    assert on.spec.kernel_spec.kv_segments == 1
    assert on.spec.kernel_spec.dropout is True


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_takes_the_kv_split_out_of_the_knob_space(arch):
    spec = dispatch_attention(_chunk_request(arch, dropout_p=0.1)).spec.kernel_spec
    assert "kv_segments" not in {a.name for a in dense_axes_for(spec, arch)}


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_dense_sweep_keeps_the_requested_dropout(arch, dropout_p):
    """A tuning sweep varies performance knobs only; it must never add or drop
    dropout, nor reach a KV split with it."""
    results = [
        r
        for r in dispatch_attention_all(
            _chunk_request(arch, dropout_p=dropout_p), candidate_prefix="attention_"
        )
        if hasattr(r.spec, "kernel_spec") and r.spec.path == "dense"
    ]
    assert results, "no dense configuration was enumerated"
    assert {r.spec.kernel_spec.dropout for r in results} == {dropout_p > 0}
    if dropout_p:
        assert {r.spec.kernel_spec.kv_segments for r in results} == {1}


def test_fp8_dropout_is_refused_on_the_gfx950_grid():
    req = _pinned(_request("gfx950", use_fp8=True, dropout_p=0.1), "gfx950_dense_grid")
    with pytest.raises(
        ValueError, match="dropout is not supported with fp8 KV storage"
    ):
        dispatch_attention(req)


# --- binding --------------------------------------------------------------


def _tensors(**extra):
    # The dense binding forwards these untouched; the runner is replaced.
    return {name: object() for name in ("q", "k", "v", "out")} | extra


@pytest.fixture
def runner(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bindings, "_dense_runner", lambda arch: lambda **kw: calls.append(kw)
    )
    return calls


def _bound(arch, dropout_p, tensors, **kwargs):
    req = _pinned(_request(arch, dropout_p=dropout_p), _DENSE_SPEC_IDS[arch][0])
    return dispatch_attention(req).bind_torch(tensors, **kwargs)


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_forwards_seed_offset_and_p(arch, runner):
    seed, offset = object(), 7
    _bound(arch, 0.25, _tensors(dropout_seed=seed, dropout_offset=offset)).launch()
    (call,) = runner
    assert call["dropout_seed"] is seed
    assert call["dropout_offset"] == 7
    assert call["dropout_p"] == 0.25
    assert call["spec"].dropout is True


@pytest.mark.parametrize("arch", _ARCHES)
def test_launch_kwargs_give_each_call_its_seed(arch, runner):
    binding = _bound(arch, 0.25, _tensors())
    binding.launch(dropout_seed=1, dropout_offset=0)
    binding.launch(dropout_seed=2, dropout_offset=64)
    assert [(c["dropout_seed"], c["dropout_offset"]) for c in runner] == [
        (1, 0),
        (2, 64),
    ]


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_off_passes_no_dropout_inputs(arch, runner):
    _bound(arch, 0.0, _tensors()).launch()
    (call,) = runner
    assert not {"dropout_seed", "dropout_offset", "dropout_p"} & call.keys()


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("given", [{}, {"dropout_seed": 1}, {"dropout_offset": 1}])
def test_dropout_without_seed_and_offset_is_refused(arch, given, runner):
    binding = _bound(arch, 0.25, _tensors(**given))
    with pytest.raises(
        ValueError, match="dropout_p=0.25 requires dropout_seed and dropout_offset"
    ):
        binding.launch()
    assert not runner


@pytest.mark.parametrize("arch", _ARCHES)
def test_seed_bound_with_dropout_off_is_refused(arch, runner):
    with pytest.raises(
        ValueError, match=r"dropout_seed given but request.dropout_p is 0"
    ):
        _bound(arch, 0.0, _tensors(dropout_seed=1))


@pytest.mark.parametrize("arch", _ARCHES)
def test_seed_launched_with_dropout_off_is_refused(arch, runner):
    binding = _bound(arch, 0.0, _tensors())
    with pytest.raises(
        ValueError, match=r"dropout_offset given but request.dropout_p is 0"
    ):
        binding.launch(dropout_offset=3)
    assert not runner


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_p_is_not_a_launch_option(arch, runner):
    """The probability belongs to the request (it selects the kernel), so a
    launch cannot change it."""
    binding = _bound(arch, 0.25, _tensors(dropout_seed=1, dropout_offset=0))
    with pytest.raises(TypeError, match="unexpected keyword.*dropout_p"):
        binding.launch(dropout_p=0.5)


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_refuses_a_spec_that_disagrees_on_dropout(arch):
    req = _pinned(_request(arch, dropout_p=0.25), _DENSE_SPEC_IDS[arch][0])
    spec = dispatch_attention(req).spec
    off = dataclasses.replace(
        spec, kernel_spec=dataclasses.replace(spec.kernel_spec, dropout=False)
    )
    with pytest.raises(ValueError, match="kernel_spec.dropout=False disagrees"):
        bindings.bind_dense_attention_torch(req, off, _tensors())
