# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for unified attention dropout: tiled 2D routing, the
paths that decline it, and the tuning binding's dropout inputs (host only)."""

import dataclasses

import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_problem,
    dispatch_attention,
    dispatch_attention_all,
)
from rocke.tests.host_arrays import HostArray

_ARCHES = ("gfx942", "gfx950")
_TUNING_ALGORITHM = "unified_tuning"
_NARROW_2D_SPEC = "{arch}_u2d_narrow_nw1_mw16_t1xb_llvm"
_TRANSPOSED_2D_SPEC = {
    "gfx942": "gfx942_u2d_transposed_x8_nw1_mw32_t2xb_llvm",
    "gfx950": "gfx950_u2d_transposed32_nw1_mw32_t2xb_llvm",
}
_WIDE_2D_SPEC = {
    "gfx942": "gfx942_u2d_wide32x8_nw1_mw32_t1xb_llvm",
    "gfx950": "gfx950_u2d_wide32_nw1_mw32_t1xb_llvm",
}
_SPLIT_KV_SPEC = "{arch}_u3d_splitkv_seg16_t1xb"
_GFX942_4WARP_SPEC = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


def _decode_request(arch, **overrides):
    """One-token decode over a long paged context: auto routes it to split-KV
    with dropout off."""
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


def _prefill_request(arch, **overrides):
    """Single-sequence 256-token prefill: routes to tiled 2D."""
    return _decode_request(arch, batch=1, seqlen_q=256, seqlen_k=256, **overrides)


def _pinned(req, spec_id):
    return dataclasses.replace(req, algorithm=_TUNING_ALGORITHM, spec_id=spec_id)


@pytest.mark.parametrize("arch", _ARCHES)
def test_unpinned_prefill_is_served_on_tiled_2d_with_dropout(arch):
    on = dispatch_attention(_prefill_request(arch, dropout_p=0.1))
    off = dispatch_attention(_prefill_request(arch))
    assert on.spec.path == "2d" and on.spec.dropout is True
    assert on.kernel_id.candidate == off.kernel_id.candidate
    assert on.spec.kernel_name() == off.spec.kernel_name() + "_do"
    assert on.kernel_id.spec_hash != off.kernel_id.spec_hash
    # Dropout off keeps the identity payload it had before the field existed.
    assert "dropout" not in off.spec.identity()


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_attention_spec_rejects_non_boolean_dropout(bad):
    spec = dispatch_attention(_prefill_request("gfx950")).spec
    with pytest.raises(ValueError, match="dropout must be a bool"):
        dataclasses.replace(spec, dropout=bad)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "request_fn",
    [
        _decode_request,
        # A long single-sequence chunk: few query blocks, a long KV range.
        lambda arch, **o: _decode_request(
            arch, batch=1, nhead_q=8, nhead_k=8, seqlen_q=64, seqlen_k=16384, **o
        ),
    ],
    ids=["decode", "long_chunk"],
)
def test_row_auto_sends_to_split_kv_runs_tiled_2d(arch, request_fn):
    """Only tiled 2D has dropout, so a row ``auto`` would split over KV stays
    on tiled 2D when its gate admits it."""
    assert dispatch_attention(request_fn(arch)).spec.path == "3d"
    result = dispatch_attention(request_fn(arch, dropout_p=0.1))
    assert result.spec.path == "2d" and result.spec.dropout is True


@pytest.mark.parametrize("arch", _ARCHES)
def test_auto_path_is_tiled_2d_for_dropout(arch):
    from kernels.common.attention_unified import select_auto_path

    problem = attention_problem(_decode_request(arch, dropout_p=0.1))
    assert problem.select_path(arch) == "3d"
    assert select_auto_path(problem, arch) == "2d"


def test_row_tiled_2d_declines_is_refused():
    """gfx942 D256 bf16 prefill at block size 32: its default tiled spec
    exceeds the LDS budget, so dropout has no path (LSE would go to split-KV)."""
    req = _prefill_request(
        "gfx942", hdim_q=256, hdim_v=256, kv_block_size=32, dropout_p=0.1
    )
    with pytest.raises(ValueError) as info:
        dispatch_attention(req)
    message = str(info.value)
    assert "no candidate supports request" in message
    assert "attention_unified_2d: tiled 2D kernel: estimated LDS" in message
    assert "attention_unified_3d: capability: cannot serve features ['dropout']" in (
        message
    )


@pytest.mark.parametrize("arch", _ARCHES)
def test_fp8_dropout_is_refused(arch):
    req = _decode_request(arch, use_fp8=True, fp8_fnuz=arch == "gfx942", dropout_p=0.1)
    with pytest.raises(
        ValueError,
        match="attention_unified_2d: dropout is not implemented with FP8 K/V",
    ):
        dispatch_attention(req)


def test_dropout_on_an_arch_without_it_is_refused():
    with pytest.raises(
        ValueError, match="dropout is not implemented on gfx1250 tiled 2D attention"
    ):
        dispatch_attention(_prefill_request("gfx1250", dropout_p=0.1))


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("spec_id", ["narrow", "transposed"])
def test_pinned_2d_spec_follows_the_request(arch, spec_id):
    spec_id = (
        _NARROW_2D_SPEC.format(arch=arch)
        if spec_id == "narrow"
        else _TRANSPOSED_2D_SPEC[arch]
    )
    on = dispatch_attention(_pinned(_prefill_request(arch, dropout_p=0.1), spec_id))
    off = dispatch_attention(_pinned(_prefill_request(arch), spec_id))
    assert on.spec.kernel_spec.dropout is True
    assert off.spec.kernel_spec.dropout is False
    # Same configuration (dropout is the caller's, not a knob), distinct binary.
    assert on.spec.config_key == off.spec.config_key
    assert on.spec.tuning_id == off.spec.tuning_id
    assert on.spec.cache_key() != off.spec.cache_key()
    assert ("dropout", "False") not in off.spec.cache_key()[4]


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "spec_id",
    [lambda a: _WIDE_2D_SPEC[a], lambda a: _SPLIT_KV_SPEC.format(arch=a)],
    ids=["wide32", "split_kv"],
)
def test_pinned_spec_without_dropout_is_refused(arch, spec_id):
    req = _pinned(_decode_request(arch, dropout_p=0.1), spec_id(arch))
    with pytest.raises(ValueError, match=r"cannot serve features \['dropout'\]"):
        dispatch_attention(req)


def test_pinned_gfx942_4warp_spec_refuses_dropout():
    req = _prefill_request("gfx942", hdim_q=256, hdim_v=256, dropout_p=0.1)
    with pytest.raises(ValueError, match=r"cannot serve features \['dropout'\]"):
        dispatch_attention(_pinned(req, _GFX942_4WARP_SPEC))


def test_explicit_4warp_spec_builder_refuses_dropout():
    from dispatch.attention.common import _tuning_problem
    from dispatch.attention.tuning_specs import (
        ExplicitAttention2DConfig,
        make_explicit_attention_2d_spec,
    )

    problem = _tuning_problem(
        _prefill_request("gfx942", hdim_q=256, hdim_v=256, dropout_p=0.1)
    )
    config = ExplicitAttention2DConfig(
        num_warps=1,
        block_m_per_warp=32,
        tile_policy="64",
        builder_kind="gfx942_4warp_gqa",
    )
    with pytest.raises(
        ValueError, match="dropout is not implemented on the gfx942 4-warp GQA builder"
    ):
        make_explicit_attention_2d_spec(problem, config, arch="gfx942")


@pytest.mark.parametrize("arch", _ARCHES)
def test_explicit_split_kv_spec_builder_refuses_dropout(arch):
    from dispatch.attention.common import _tuning_problem
    from dispatch.attention.tuning_specs import (
        ExplicitAttention3DConfig,
        make_explicit_attention_3d_specs,
    )

    problem = _tuning_problem(_decode_request(arch, dropout_p=0.1))
    with pytest.raises(
        ValueError, match="dropout is not implemented on the split-KV attention path"
    ):
        make_explicit_attention_3d_specs(
            problem, ExplicitAttention3DConfig(num_segments=16), arch=arch
        )


@pytest.mark.parametrize("arch", _ARCHES)
def test_sweep_offers_only_tiled_2d_dropout_kernels(arch):
    results = dispatch_attention_all(
        _prefill_request(arch, dropout_p=0.1), candidate_prefix="attention_"
    )
    unified = [
        r for r in results if hasattr(r.spec, "kernel_spec") and r.spec.path != "dense"
    ]
    assert unified, "no tiled 2D configuration was enumerated"
    assert {r.spec.path for r in unified} == {"2d"}
    assert {r.spec.kernel_spec.dropout for r in unified} == {True}
    assert not [r for r in unified if r.spec.builder_kind == "gfx942_4warp_gqa"]


# --- binding --------------------------------------------------------------


def _array(shape, dtype, values=None):
    """A host-only device-array stand-in the real binding validator accepts."""
    return HostArray(shape, dtype, values=values)


def _paged_tensors(problem, **extra):
    """Paged tensors matching ``problem``, every sequence at full length."""
    seqs, pages = problem.num_seqs, -(-problem.max_seqlen_k // problem.block_size)
    act = (problem.total_q, problem.num_query_heads, problem.head_size)
    kv = (seqs * pages, problem.block_size, problem.num_kv_heads, problem.head_size)
    rows = problem.max_seqlen_q
    return {
        "q": _array(act, "bf16"),
        "out": _array(act, "bf16"),
        "k": _array(kv, "bf16"),
        "v": _array(kv, "bf16"),
        "cu_seqlens_q": _array((seqs + 1,), "i32", [s * rows for s in range(seqs + 1)]),
        "seqused_k": _array((seqs,), "i32", [problem.max_seqlen_k] * seqs),
        "block_table": _array(
            (seqs, pages),
            "i32",
            [list(range(s * pages, (s + 1) * pages)) for s in range(seqs)],
        ),
    } | extra


@pytest.fixture
def runner(monkeypatch):
    from kernels.common import attention_unified as au

    calls = []
    monkeypatch.setattr(
        au, "run_unified_attention_torch", lambda **kw: calls.append(kw)
    )
    return calls


def _bound(arch, dropout_p, **extra):
    from dispatch.attention.common import _tuning_problem

    request = _pinned(
        _prefill_request(arch, dropout_p=dropout_p), _NARROW_2D_SPEC.format(arch=arch)
    )
    tensors = _paged_tensors(_tuning_problem(request), **extra)
    return dispatch_attention(request).bind_torch(tensors)


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_forwards_seed_offset_and_p(arch, runner):
    seed = object()
    _bound(arch, 0.25, dropout_seed=seed, dropout_offset=5).launch()
    (call,) = runner
    assert call["dropout_seed"] is seed and call["dropout_offset"] == 5
    assert call["dropout_p"] == 0.25
    assert call["problem"].dropout is True and call["backend"] == "tiled"


@pytest.mark.parametrize("arch", _ARCHES)
def test_launch_kwargs_give_each_call_its_seed(arch, runner):
    binding = _bound(arch, 0.25)
    binding.launch(dropout_seed=1, dropout_offset=0)
    binding.launch(dropout_seed=2, dropout_offset=64)
    assert [(c["dropout_seed"], c["dropout_offset"]) for c in runner] == [
        (1, 0),
        (2, 64),
    ]


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_off_passes_no_dropout_inputs(arch, runner):
    _bound(arch, 0.0).launch()
    (call,) = runner
    assert not {"dropout_seed", "dropout_offset", "dropout_p"} & call.keys()


@pytest.mark.parametrize("arch", _ARCHES)
def test_dropout_without_seed_and_offset_is_refused(arch, runner):
    binding = _bound(arch, 0.25, dropout_seed=1)
    with pytest.raises(
        ValueError, match="dropout_p=0.25 requires dropout_seed and dropout_offset"
    ):
        binding.launch()
    assert not runner


@pytest.mark.parametrize("arch", _ARCHES)
def test_seed_bound_with_dropout_off_is_refused(arch, runner):
    with pytest.raises(
        ValueError,
        match="dropout_seed, dropout_offset given but request.dropout_p is 0",
    ):
        _bound(arch, 0.0, dropout_seed=1, dropout_offset=0)


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_rejects_a_problem_that_disagrees_on_dropout(arch, monkeypatch):
    from dispatch.attention import bindings
    from dispatch.attention.common import _tuning_problem

    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda problem, t, **kw: 64
    )
    request = _pinned(
        _prefill_request(arch, dropout_p=0.1), _NARROW_2D_SPEC.format(arch=arch)
    )
    problem = dataclasses.replace(_tuning_problem(request), dropout=False)
    with pytest.raises(
        ValueError, match="request.dropout_p=0.1 disagrees with problem.dropout=False"
    ):
        dispatch_attention(request).bind_torch({"problem": problem})


@pytest.mark.parametrize("arch", _ARCHES)
def test_binding_rejects_a_2d_kernel_spec_that_disagrees_on_dropout(arch):
    from dispatch.attention.bindings import validate_tuning_attention_contract
    from dispatch.attention.common import _tuning_problem

    request = _prefill_request(arch, dropout_p=0.1)
    result = dispatch_attention(_pinned(request, _NARROW_2D_SPEC.format(arch=arch)))
    spec = dataclasses.replace(
        result.spec,
        kernel_spec=dataclasses.replace(result.spec.kernel_spec, dropout=False),
    )
    with pytest.raises(ValueError, match="kernel_spec.dropout=False disagrees"):
        validate_tuning_attention_contract(request, _tuning_problem(request), spec)


# --- carriers ---------------------------------------------------------------

# Every class reachable from the dispatcher with a ``dropout`` field; each refuses
# a non-bool value in ``__post_init__``. A new carrier fails the inventory below
# until it is listed, so its guard is not forgotten.
_DROPOUT_CARRIERS = {
    "dispatch.attention.common.AttentionSpec",
    "kernels.common.attention_unified.UnifiedAttentionProblem",
    "kernels.gfx942.attention_dense.Gfx942AttentionDenseSpec",
    "kernels.gfx942.attention_tiled_2d.UnifiedAttention2DTiledSpec",
    "kernels.gfx950.attention_dense.Gfx950AttentionDenseSpec",
    "kernels.gfx950.attention_tiled_2d.UnifiedAttention2DTiledSpec",
}
_DENSE_SPEC = {"gfx942": "gfx942_dense", "gfx950": "gfx950_dense_grid"}


def _carriers(obj):
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return
    fields = dataclasses.fields(obj)
    if any(f.name == "dropout" for f in fields):
        yield obj
    for f in fields:
        yield from _carriers(getattr(obj, f.name))


def test_every_dropout_carrier_is_listed():
    found = []
    for arch in _ARCHES:
        prefill = _prefill_request(arch, dropout_p=0.1)
        dense = dataclasses.replace(
            prefill, algorithm="attention_dense", spec_id=_DENSE_SPEC[arch]
        )
        if arch == "gfx950":
            dense = dataclasses.replace(dense, algorithm="attention_dense_grid")
        for req in (
            prefill,
            _pinned(prefill, _NARROW_2D_SPEC.format(arch=arch)),
            dense,
        ):
            found.extend(_carriers(dispatch_attention(req).spec))
        found.append(attention_problem(prefill))
    names = {f"{type(c).__module__}.{type(c).__name__}" for c in found}
    assert names == _DROPOUT_CARRIERS
    for carrier in found:
        with pytest.raises(ValueError, match="dropout must be a bool"):
            dataclasses.replace(carrier, dropout=1)
