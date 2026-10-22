# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for the optional unified split-KV LSE output (host only)."""

import dataclasses

import pytest

from dispatch.attention import (
    AttentionRequest,
    dispatch_attention,
    dispatch_attention_all,
)

_ARCHES = ("gfx942", "gfx950")
# One explicit split-KV geometry per arch (tuning candidates; gfx942's are opt-in).
_SPLIT_KV_SPEC = {
    "gfx942": "gfx942_u3d_splitkv_seg16_t1xb",
    "gfx950": "gfx950_u3d_splitkv_seg16_t1xb",
}
_TUNING_ALGORITHM = "unified_tuning"


def _decode_request(arch, **overrides):
    """One-token decode over a long paged context: routes to split-KV (3D).

    Bottom-right causal is the paged decode mask (the query sees every cached
    key); top-left would restrict a single query to key 0, which the gfx942
    tiled kernels do not implement."""
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
    )
    fields.update(overrides)
    return AttentionRequest(**fields)


def _pinned(req, spec_id):
    return dataclasses.replace(req, algorithm=_TUNING_ALGORITHM, spec_id=spec_id)


@pytest.mark.parametrize("arch", _ARCHES)
def test_unpinned_lse_decode_request_routes_to_split_kv(arch):
    result = dispatch_attention(_decode_request(arch, emit_lse=True))
    assert result.spec.path == "3d"


@pytest.mark.parametrize("arch", _ARCHES)
def test_unpinned_lse_spec_has_its_own_identity(arch):
    """The LSE reduce takes one more pointer: a distinct kernel, so the routed
    spec must not share kernel_name / spec_hash with the LSE-off spec (consumers
    key compile caches and manifests on them)."""
    on = dispatch_attention(_decode_request(arch, emit_lse=True))
    off = dispatch_attention(_decode_request(arch))
    assert on.spec.kernel_name() == off.spec.kernel_name() + "_lse"
    assert on.kernel_id.spec_hash != off.kernel_id.spec_hash


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_attention_spec_rejects_non_boolean_emit_lse(bad):
    """AttentionSpec is public: a truthy non-bool must not become an LSE identity."""
    spec = dispatch_attention(_decode_request("gfx950")).spec
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        dataclasses.replace(spec, emit_lse=bad)


@pytest.mark.parametrize("arch", _ARCHES)
def test_pinned_split_kv_spec_follows_the_request(arch):
    spec_id = _SPLIT_KV_SPEC[arch]
    on = dispatch_attention(_pinned(_decode_request(arch, emit_lse=True), spec_id))
    off = dispatch_attention(_pinned(_decode_request(arch), spec_id))
    assert on.spec.reduce_spec.emit_lse is True
    assert off.spec.reduce_spec.emit_lse is False
    # Distinct binaries and cache entries: the LSE reduce takes one more pointer.
    assert on.spec.reduce_spec.kernel_name() != off.spec.reduce_spec.kernel_name()
    assert on.spec.cache_key() != off.spec.cache_key()


# What a long-lived cache stored for ``_decode_request`` before emit_lse existed:
# the unpinned spec_hash, and the split-KV pin with its spec_hash. These are
# develop's own values (gfx950's split-KV config_key moved with #13120's
# additive-bias field); emit_lse must leave them exactly as develop computes them.
_PRE_LSE_UNPINNED_SPEC_HASH = "65416b2a8b9ffa90"
_PRE_LSE_PIN = {
    "gfx942": ("splitkv_seg16_t1xb_wpenone@d50574be81e67c73", "8dc6060ed42707f6"),
    "gfx950": ("splitkv_seg16_t1xb_wpenone@79ee1d118140d024", "b9c537be7298900c"),
}


@pytest.mark.parametrize("arch", _ARCHES)
def test_pre_lse_cache_entries_still_resolve_with_lse_off(arch):
    """A cache replays a stored pin and treats a spec_hash mismatch as a miss.
    The LSE field must leave every LSE-off identity as it was (spec payload,
    tuning cache key, config defaults fingerprint), or every stored entry stops
    resolving."""
    unpinned = dispatch_attention(_decode_request(arch))
    assert unpinned.kernel_id.spec_hash == _PRE_LSE_UNPINNED_SPEC_HASH
    tuning_id, spec_hash = _PRE_LSE_PIN[arch]
    pinned = dataclasses.replace(
        _pinned(_decode_request(arch), _SPLIT_KV_SPEC[arch]), tuning_id=tuning_id
    )
    assert dispatch_attention(pinned).kernel_id.spec_hash == spec_hash


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("emit_lse", [False, True])
def test_split_kv_sweep_keeps_the_requested_output(arch, emit_lse):
    """A tuning sweep varies performance knobs only; it must never add or drop
    the LSE output. With LSE requested, no 2D configuration may be offered."""
    results = dispatch_attention_all(
        _decode_request(arch, emit_lse=emit_lse), candidate_prefix="attention_"
    )
    split_kv = [r for r in results if r.spec.path == "3d"]
    assert split_kv, "no split-KV configuration was enumerated"
    assert {r.spec.reduce_spec.emit_lse for r in split_kv} == {emit_lse}
    if emit_lse:
        assert not [r for r in results if r.spec.path == "2d"]


@pytest.mark.parametrize("arch", _ARCHES)
def test_pinned_2d_spec_refuses_lse(arch):
    prefill = _decode_request(arch, batch=1, seqlen_q=256, seqlen_k=256)
    spec_id = f"{arch}_u2d_narrow_nw1_mw16_t1xb_llvm"
    dispatch_attention(_pinned(prefill, spec_id))  # served without LSE
    # Refused by the missing ``lse`` feature or, should a 2D variant ever declare
    # it, by the 2D gate ("LSE output is not implemented ..."): either is a refusal.
    with pytest.raises(ValueError, match="(?i)lse"):
        dispatch_attention(
            _pinned(dataclasses.replace(prefill, emit_lse=True), spec_id)
        )


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("pin", [False, True], ids=["auto", "pinned"])
def test_lse_with_sinks_is_refused(arch, pin):
    req = _decode_request(arch, emit_lse=True, use_sinks=True)
    if pin:
        req = _pinned(req, _SPLIT_KV_SPEC[arch])
    with pytest.raises(ValueError, match="LSE output with attention sinks"):
        dispatch_attention(req)


def test_lse_on_gfx1250_split_kv_is_refused():
    with pytest.raises(ValueError, match="LSE output is not implemented on gfx1250"):
        dispatch_attention(_decode_request("gfx1250", emit_lse=True))


class _FakeTensor:
    """Metadata-only stand-in the real binding validator accepts (no torch)."""

    def __init__(self, shape, dtype, values=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "cuda:0"
        self._values = values

    def is_contiguous(self):
        return True

    def stride(self, dim):
        stride = 1
        for extent in self.shape[dim + 1 :]:
            stride *= extent
        return stride

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def _paged_tensors(problem):
    """Paged decode tensors matching ``problem``, every sequence at full length."""
    seqs, pages = problem.num_seqs, -(-problem.max_seqlen_k // problem.block_size)
    act = (problem.total_q, problem.num_query_heads, problem.head_size)
    kv = (seqs * pages, problem.block_size, problem.num_kv_heads, problem.head_size)
    return {
        "q": _FakeTensor(act, "torch.bfloat16"),
        "out": _FakeTensor(act, "torch.bfloat16"),
        "k": _FakeTensor(kv, "torch.bfloat16"),
        "v": _FakeTensor(kv, "torch.bfloat16"),
        "cu_seqlens_q": _FakeTensor((seqs + 1,), "torch.int32", list(range(seqs + 1))),
        "seqused_k": _FakeTensor((seqs,), "torch.int32", [problem.max_seqlen_k] * seqs),
        "block_table": _FakeTensor(
            (seqs, pages),
            "torch.int32",
            [list(range(s * pages, (s + 1) * pages)) for s in range(seqs)],
        ),
    }


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_forwards_the_lse_buffer(arch, monkeypatch):
    """The bound launch hands the caller's lse tensor to the runtime, after the
    real paged-tensor validation (only the GPU runner is replaced)."""
    from dispatch.attention.common import _tuning_problem
    from kernels.common import attention_unified as au

    seen = {}
    monkeypatch.setattr(au, "run_unified_attention_torch", lambda **kw: seen.update(kw))
    request = _pinned(_decode_request(arch, emit_lse=True), _SPLIT_KV_SPEC[arch])
    lse = object()
    tensors = {**_paged_tensors(_tuning_problem(request)), "lse": lse}
    dispatch_attention(request).bind_torch(tensors).launch()
    assert seen["lse"] is lse
    assert seen["problem"].emit_lse is True


@pytest.mark.parametrize("emit_lse", [True, False], ids=["missing", "stray"])
@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_refuses_an_lse_buffer_that_does_not_match_the_request(arch, emit_lse):
    """missing: LSE requested without tensors['lse']. stray: an lse buffer with
    LSE off. Both fail at bind_torch, not at the first launch."""
    from dispatch.attention.common import _tuning_problem

    request = _pinned(_decode_request(arch, emit_lse=emit_lse), _SPLIT_KV_SPEC[arch])
    tensors = _paged_tensors(_tuning_problem(request))
    if not emit_lse:
        tensors["lse"] = object()
    with pytest.raises(ValueError, match=r"tensors\['lse'\]"):
        dispatch_attention(request).bind_torch(tensors)


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_rejects_a_problem_that_disagrees_on_lse(arch, monkeypatch):
    from dispatch.attention import bindings

    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda problem, t, **kw: 64
    )
    result = dispatch_attention(
        _pinned(_decode_request(arch, emit_lse=True), _SPLIT_KV_SPEC[arch])
    )
    from dispatch.attention.common import _tuning_problem

    problem = dataclasses.replace(
        _tuning_problem(_decode_request(arch, emit_lse=True)), emit_lse=False
    )
    with pytest.raises(ValueError, match="emit_lse"):
        result.bind_torch({"problem": problem})
