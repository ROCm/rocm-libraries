# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side contract of the gfx950 dense kernel's additive bias: the spec field and
its refusals, the kernel name and cache identity, the kernel-param / launcher ABI, the
runner's bias validation, the dispatcher wiring and the bias load shape (vector on
the aligned body, scalar on the ragged one). No GPU; the numerics are
``test_attention_dense_bias_gfx950_numeric.py``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernels.common._attention_additive_bias import additive_bias_strides
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    attention_dense_signature,
    build_attention_dense,
    run_attention_dense_torch,
)

_BIAS_PARAMS = [
    "additive_bias_ptr",
    "additive_bias_batch_stride",
    "additive_bias_head_stride",
    "additive_bias_sq_stride",
]

_BASE = Gfx950AttentionDenseSpec(
    batch=2,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
    causal=True,
    dtype="bf16",
)

# The grid-body modes the bias composes with.
_MODES = {
    "default": {},
    "noncausal": {"causal": False},
    "d64_fp16": {"head_size": 64, "dtype": "fp16"},
    "window": {"sliding_window": 128},
    "bottom_right": {"seqlen_kv": 1024, "causal_bottom_right": True},
    "ragged": {"seqlen_q": 500, "seqlen_kv": 700, "ragged": True},
    "varlen": {"varlen": True},
    "sinks": {"use_sinks": True},
    "lse": {"emit_lse": True},
    "lse_d64": {"emit_lse": True, "head_size": 64, "dtype": "fp16"},
    "lookahead": {"right_bound": 16},
}


def _bias(spec_kw):
    return replace(_BASE, **spec_kw, use_additive_bias=True)


@pytest.mark.parametrize("mode", list(_MODES))
def test_kernel_params_match_the_launcher_signature(mode):
    """KernelLauncher packs kernargs by position, so the builder's params and
    attention_dense_signature must name the same list, the four bias params last."""
    spec = _bias(_MODES[mode])
    params = [p.name for p in build_attention_dense(spec).params]
    assert params == [a["name"] for a in attention_dense_signature(spec)]
    assert params[-4:] == _BIAS_PARAMS
    plain = replace(spec, use_additive_bias=False)
    plain_params = [p.name for p in build_attention_dense(plain).params]
    assert plain_params == params[:-4]


@pytest.mark.parametrize("mode", list(_MODES))
def test_bias_is_in_the_name_and_the_cache_key(mode):
    spec = _bias(_MODES[mode])
    plain = replace(spec, use_additive_bias=False)
    assert "_addb" in spec.kernel_name()
    assert "addb" not in plain.kernel_name()
    assert attention_dense_cache_key(spec, arch="gfx950") != attention_dense_cache_key(
        plain, arch="gfx950"
    )


_BIAS_X4_LOAD = re.compile(r"load <4 x float>, ptr addrspace\(1\) [^,]+, align (\d+)")
# The tile's bias pointer: additive_bias_elem_ptr's byte GEP off the bias kernarg.
_BIAS_TILE_PTR = re.compile(
    r"%(\S+) = getelementptr inbounds i8, ptr addrspace\(1\) %additive_bias_ptr, i64 %\S+\n"
)
# An x4 bias load at a constant element offset from a tile pointer.
_BIAS_X4_AT = re.compile(
    r"%(\S+) = getelementptr inbounds float, ptr addrspace\(1\) %(\S+), i32 (\d+)\n"
    r"\s*%\S+ = load <4 x float>, ptr addrspace\(1\) %\1, align 4\n"
)


@pytest.mark.parametrize("mode", list(_MODES))
def test_the_aligned_body_reads_the_bias_four_keys_per_element_aligned_load(mode):
    """The non-ragged body reads the bias with one <4 x float> load per four keys,
    element-aligned only (the host contract promises no 16-byte alignment), all
    from ONE pointer per KV tile at constant element offsets 0, 8, ... (so the
    backend folds them into load immediates); the ragged body keeps masked scalar
    loads, the D128 LSE body plain scalar ones (the vector form would spill it),
    and bias off emits no such load."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    def llvm(spec):
        return _lower_kernel_to_llvm_python(
            build_attention_dense(spec, arch="gfx950"),
            arch="gfx950",
            llvm_flavor="llvm22",
        )

    spec = _bias(_MODES[mode])
    ll = llvm(spec)
    aligns = _BIAS_X4_LOAD.findall(ll)
    if spec.ragged or (spec.emit_lse and spec.head_size == 128):
        assert aligns == []
    else:
        assert aligns and set(aligns) == {"4"}
        groups: dict[str, list[int]] = {}
        for m in _BIAS_X4_AT.finditer(ll):
            groups.setdefault(m.group(2), []).append(int(m.group(3)))
        # Every x4 load is a constant offset from a tile pointer ...
        assert sum(map(len, groups.values())) == len(aligns)
        assert set(groups) <= set(_BIAS_TILE_PTR.findall(ll))
        # ... and each tile pointer serves the whole tile: BN // 32 * 4 groups, 8 keys
        # apart.
        per_tile = spec.block_n // 32 * 4
        for offsets in groups.values():
            assert offsets == list(range(0, 8 * per_tile, 8))
    assert _BIAS_X4_LOAD.findall(llvm(replace(spec, use_additive_bias=False))) == []


def test_bias_keeps_the_runtime_shape_cohort():
    """The bias strides are kernargs, so an aligned bias kernel still serves every
    batch and length from one binary."""
    spec = _bias({})
    assert spec.runtime_shape
    other = replace(spec, batch=5, seqlen_q=1024, seqlen_kv=2048)
    assert attention_dense_cache_key(spec, arch="gfx950") == attention_dense_cache_key(
        other, arch="gfx950"
    )


@pytest.mark.parametrize("sliding_window", [0, 128])
@pytest.mark.parametrize("block_m", [256, 128])
def test_bias_head_counts_share_one_kernel(block_m, sliding_window):
    """The head counts are kernargs too, so one aligned bias binary serves every head
    configuration at one block order: the bias head index and the GQA mapping are
    computed at launch. A head count baked into the bias addressing would split the
    IR under one cache key, and the cache would serve one head configuration's
    kernel to all of them. The block order is pinned (the catalog's qb_minor)
    because auto picks it partly from the head counts."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    specs = [
        replace(
            _bias({"sliding_window": sliding_window}),
            num_query_heads=hq,
            num_kv_heads=hkv,
            block_m=block_m,
            nonpersist_decode="qb_minor",
        )
        for hq, hkv in ((128, 8), (40, 8), (28, 4), (32, 32), (64, 64))
    ]
    assert all(s.runtime_shape for s in specs)
    keys = {attention_dense_cache_key(s, arch="gfx950") for s in specs}
    names = {s.kernel_name() for s in specs}
    irs = {
        _lower_kernel_to_llvm_python(
            build_attention_dense(s, arch="gfx950"), arch="gfx950", llvm_flavor="llvm22"
        )
        for s in specs
    }
    assert len(keys) == 1
    assert len(names) == 1 and not re.search(r"_(hq|kv)\d+", names.pop())
    assert len(irs) == 1


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_use_additive_bias_must_be_a_bool(value):
    with pytest.raises(ValueError, match="must be a bool"):
        replace(_BASE, use_additive_bias=value)


def test_bias_is_refused_on_the_persistent_body():
    replace(_BASE, persistent=True, num_persistent=256)  # the persistent spec is valid
    with pytest.raises(
        ValueError, match=r"use_additive_bias is implemented on the grid body only"
    ):
        replace(_BASE, persistent=True, num_persistent=256, use_additive_bias=True)


def test_bias_is_refused_with_paged_kv():
    paged = dict(
        batch=1,
        paged=True,
        block_size=64,
        num_kv_blocks=8,
        seqlen_q=512,
        seqlen_kv=512,
        sliding_window=128,
    )
    replace(_BASE, **paged)  # the paged spec itself is valid
    with pytest.raises(
        ValueError, match=r"use_additive_bias is not supported with paged K/V"
    ):
        replace(_BASE, **paged, use_additive_bias=True)


def test_bias_is_refused_with_a_kv_segment_split():
    # The split's segment store returns before the bias epilogue's keyless-row
    # handling, so the two do not compose.
    replace(_BASE, kv_segments=2)  # the split spec itself is valid
    with pytest.raises(
        ValueError, match=r"use_additive_bias is not supported with kv_segments > 1"
    ):
        replace(_BASE, kv_segments=2, use_additive_bias=True)


# --------------------------------------------------------------------------- #
# Runner validation (raised before any compile or launch). The bias checks run on
# duck-typed tensors, so they need no torch; the varlen offset checks read int32
# torch tensors and import torch per test.
# --------------------------------------------------------------------------- #

_B, _H, _SQ, _SKV = 2, 8, 512, 512


def _fake_bias(
    shape, strides, *, storage_elems, offset=0, device="cpu", dtype="torch.float32"
):
    """A duck-typed tensor: lets a case state strides no real allocation could back
    (past the i32 limit) without allocating, and runs without torch."""
    return SimpleNamespace(
        dtype=dtype,
        device=device,
        shape=shape,
        stride=lambda axis=None: strides if axis is None else strides[axis],
        untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 4 * storage_elems),
        element_size=lambda: 4,
        storage_offset=lambda: offset,
    )


def _packed(shape):
    """Contiguous element strides of ``shape``."""
    strides, step = [], 1
    for extent in reversed(shape):
        strides.insert(0, step)
        step *= extent
    return tuple(strides), step


def _fake_packed(shape, **kw):
    strides, elems = _packed(shape)
    return _fake_bias(shape, strides, storage_elems=elems, **kw)


def _fake_operands():
    """Stand-ins for q/k/v/out: the runner reads only ``q.device`` (and the packed
    extents under varlen) before the bias checks raise."""
    q = SimpleNamespace(device="cpu")
    return dict(q=q, k=q, v=q, out=q, scale=0.125)


# id -> (bias or None, error, message regex)
_MALFORMED = {
    "missing": (
        None,
        ValueError,
        r"spec\.use_additive_bias=True requires additive_bias",
    ),
    "fp16": (
        _fake_packed((_B, _H, _SQ, _SKV), dtype="torch.float16"),
        ValueError,
        r"additive_bias must be float32, got torch\.float16",
    ),
    "rank3": (_fake_packed((_H, _SQ, _SKV)), ValueError, r"must be rank-4"),
    "key-extent": (
        _fake_packed((_B, _H, _SQ, _SKV // 2)),
        ValueError,
        r"key dim must equal seqlen_kv=512, got 256",
    ),
    "head-extent": (
        _fake_packed((_B, 2, _SQ, _SKV)),
        ValueError,
        r"dim 1 \(query head\) must be 1 or 8, got 2",
    ),
    "batch-extent": (
        _fake_packed((3, _H, _SQ, _SKV)),
        ValueError,
        r"dim 0 \(batch\) must be 1 or 2, got 3",
    ),
    "query-extent": (
        _fake_packed((_B, _H, 7, _SKV)),
        ValueError,
        r"dim 2 \(query\) must be 1 or 512, got 7",
    ),
    # A [B, H, Skv, Sq] tensor transposed to [B, H, Sq, Skv].
    "key-stride": (
        _fake_bias(
            (_B, _H, _SQ, _SKV),
            (_H * _SQ * _SKV, _SQ * _SKV, 1, _SQ),
            storage_elems=_B * _H * _SQ * _SKV,
        ),
        ValueError,
        r"contiguous along Skv",
    ),
}


@pytest.mark.parametrize("case", list(_MALFORMED))
def test_runner_refuses_a_malformed_bias(case):
    bias, error, match = _MALFORMED[case]
    with pytest.raises(error, match=match):
        run_attention_dense_torch(
            spec=_bias({}), additive_bias=bias, **_fake_operands()
        )


def test_runner_refuses_a_bias_for_a_plain_spec():
    with pytest.raises(
        ValueError, match=r"additive_bias provided but spec\.use_additive_bias is False"
    ):
        run_attention_dense_torch(
            spec=_BASE,
            additive_bias=_fake_packed((_B, _H, _SQ, _SKV)),
            **_fake_operands(),
        )


# id -> (cu_seqlens_q, cu_seqlens_kv, message regex). Spec maxima Sq = Skv = 512,
# block_m 256, block_n 64; 1024 packed query and 1280 packed key tokens.
_BAD_OFFSETS = {
    "kv-past-the-maximum": (
        [0, 512, 1024],
        [0, 256, 1280],
        r"cu_seqlens_kv lengths must be monotone, tile-aligned, and within spec",
    ),
    "q-past-the-maximum": (
        [0, 256, 1024],
        [0, 640, 1280],
        r"cu_seqlens_q lengths must be monotone, tile-aligned, and within spec",
    ),
    "q-not-tile-aligned": (
        [0, 500, 1024],
        [0, 640, 1280],
        r"cu_seqlens_q lengths must be monotone, tile-aligned, and within spec",
    ),
    "q-wrong-entry-count": (
        [0, 512, 768, 1024],
        [0, 640, 1280],
        r"cu_seqlens_q must be int32 \[batch\+1\]",
    ),
    "kv-short-of-the-packed-tensor": (
        [0, 512, 1024],
        [0, 256, 512],
        r"cu_seqlens_kv endpoints must span its packed tensor",
    ),
    # The prologue always scores KV tile 0: for an empty sequence that is the next
    # sequence's keys, or K/V past the packed tensor for the last one.
    "kv-empty-sequence": (
        [0, 512, 1024],
        [0, 0, 1280],
        r"cu_seqlens_kv sequences must be nonempty",
    ),
}


@pytest.mark.parametrize("case", list(_BAD_OFFSETS))
def test_runner_refuses_varlen_offsets_the_bias_cannot_serve(case):
    """The varlen body loads the bias unmasked by in-sequence position, so the
    offsets are checked as for LSE (here with LSE off) before any compile."""
    torch = pytest.importorskip("torch", reason="the offsets are torch tensors")
    cu_q, cu_kv, match = _BAD_OFFSETS[case]
    with pytest.raises(ValueError, match=match):
        _run_varlen_bias(
            torch.tensor(cu_q, dtype=torch.int32),
            torch.tensor(cu_kv, dtype=torch.int32),
        )


def _run_varlen_bias(cu_seqlens_q, cu_seqlens_kv):
    """A varlen bias launch with LSE off; q, k, v, out and the bias are on the CPU
    (torch devices compare unequal to the string "cpu")."""
    import torch

    cpu = torch.device("cpu")
    q = SimpleNamespace(device=cpu, shape=(1024, _H, 128))
    kv = SimpleNamespace(device=cpu, shape=(1280, 2, 128))
    run_attention_dense_torch(
        spec=_bias({"varlen": True}),
        q=q,
        k=kv,
        v=kv,
        out=q,
        scale=0.125,
        additive_bias=_fake_packed((_B, _H, _SQ, _SKV), device=cpu),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )


@pytest.mark.parametrize("name", ["cu_seqlens_q", "cu_seqlens_kv"])
def test_runner_refuses_varlen_offsets_off_q_device(name):
    """The kernel dereferences the offsets, so they must be on q's device. With LSE
    on, the LSE buffer check already requires it; this is the LSE-off path. A
    meta tensor stands in for any other device (no GPU needed)."""
    torch = pytest.importorskip("torch", reason="the offsets are torch tensors")
    offsets = {
        "cu_seqlens_q": torch.tensor([0, 512, 1024], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 640, 1280], dtype=torch.int32),
    }
    offsets[name] = offsets[name].to("meta")
    with pytest.raises(ValueError, match=rf"{name} must be on the same device as q"):
        _run_varlen_bias(offsets["cu_seqlens_q"], offsets["cu_seqlens_kv"])


def test_the_lse_overlap_check_sees_the_bias(monkeypatch):
    """The LSE buffer may not overlap any input the kernel reads, the bias included.
    The check itself needs GPU tensors (the numeric suite runs it for real); here
    the inputs it receives are captured."""
    import kernels.common._lse_store as lse_store

    seen = {}

    def capture(lse, q, expected, inputs):
        seen.update(inputs)
        raise RuntimeError("captured")

    monkeypatch.setattr(lse_store, "_check_lse_buffer", capture)
    bias = _fake_packed((_B, _H, _SQ, _SKV))
    operands = _fake_operands()
    operands["q"] = SimpleNamespace(device="cpu", shape=(_B, _SQ, _H, 128))
    with pytest.raises(RuntimeError, match="captured"):
        run_attention_dense_torch(
            spec=_bias({"emit_lse": True}),
            additive_bias=bias,
            lse=object(),
            **operands,
        )
    assert seen["additive_bias"] is bias


_DIMS = dict(batch=_B, num_query_heads=_H, seqlen_q=_SQ, seqlen_kv=_SKV, device=None)
_PACKED = (_H * _SQ * _SKV, _SQ * _SKV, _SKV, 1)
_PACKED_ELEMS = _B * _H * _SQ * _SKV


def test_strides_broadcast_and_pass_through():
    full = _fake_bias((_B, _H, _SQ, _SKV), _PACKED, storage_elems=_PACKED_ELEMS)
    assert additive_bias_strides(full, **_DIMS) == _PACKED[:3]
    # Broadcast axes give 0 whatever stride the tensor writes there.
    row = _fake_bias((1, 1, 1, _SKV), (99, 99, 99, 1), storage_elems=_SKV)
    assert additive_bias_strides(row, **_DIMS) == (0, 0, 0)


def test_a_single_key_bias_needs_no_unit_key_stride():
    fake = _fake_bias(
        (_B, _H, _SQ, 1), (_H * _SQ * 7, _SQ * 7, 7, 7), storage_elems=_B * _H * _SQ * 7
    )
    dims = {**_DIMS, "seqlen_kv": 1}
    assert additive_bias_strides(fake, **dims) == (_H * _SQ * 7, _SQ * 7, 7)


def test_strides_refuse_a_bias_on_another_device():
    fake = _fake_bias((1, 1, 1, _SKV), (0, 0, 0, 1), storage_elems=_SKV, device="cpu")
    with pytest.raises(ValueError, match="same device as q"):
        additive_bias_strides(fake, **{**_DIMS, "device": "cuda:0"})


def test_strides_past_the_i32_limit_are_not_implemented():
    fake = _fake_bias((_B, _H, _SQ, _SKV), (2**30, 2**28, _SKV, 1), storage_elems=2**40)
    with pytest.raises(NotImplementedError, match=r"exceeds the kernels' i32"):
        additive_bias_strides(fake, **_DIMS)


@pytest.mark.parametrize(
    "reach, ok", [(2**31 - 1, True), (2**31, False)], ids=["at-limit", "past-limit"]
)
def test_the_i32_limit_is_on_the_largest_element_index(reach, ok):
    """The bound is on the largest index the kernel reads (here seqlen_kv - 1)."""
    dims = dict(
        batch=1, num_query_heads=1, seqlen_q=1, seqlen_kv=reach + 1, device=None
    )
    fake = _fake_bias((1, 1, 1, reach + 1), (0, 0, 0, 1), storage_elems=reach + 1)
    if ok:
        assert additive_bias_strides(fake, **dims) == (0, 0, 0)
    else:
        with pytest.raises(NotImplementedError, match=r"max element index 2147483648"):
            additive_bias_strides(fake, **dims)


def test_strides_beyond_the_storage_are_refused():
    fake = _fake_bias((_B, _H, _SQ, _SKV), _PACKED, storage_elems=1024)
    with pytest.raises(ValueError, match="only backs"):
        additive_bias_strides(fake, **_DIMS)


def test_the_storage_offset_counts_against_the_backing():
    exact = _fake_bias((_B, _H, _SQ, _SKV), _PACKED, storage_elems=_PACKED_ELEMS)
    assert additive_bias_strides(exact, **_DIMS) == _PACKED[:3]
    shifted = _fake_bias(
        (_B, _H, _SQ, _SKV), _PACKED, storage_elems=_PACKED_ELEMS, offset=1
    )
    with pytest.raises(ValueError, match="only backs"):
        additive_bias_strides(shifted, **_DIMS)


# --------------------------------------------------------------------------- #
# Dispatcher wiring
# --------------------------------------------------------------------------- #


def _request(**kw):
    from dispatch.attention import AttentionRequest

    fields = dict(
        batch=2,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        mask_type=1,
        dtype="bf16",
        use_attn_bias=True,
    )
    fields.update(kw)
    return AttentionRequest(**fields)


def test_the_grid_candidate_builds_a_bias_spec():
    from dispatch.attention import attention_tuning_spec

    spec = attention_tuning_spec(_request(), "gfx950_dense_grid").kernel_spec
    assert spec.use_additive_bias and not spec.persistent
    plain = attention_tuning_spec(_request(use_attn_bias=False), "gfx950_dense_grid")
    assert not plain.kernel_spec.use_additive_bias


@pytest.mark.parametrize("spec_id", ["gfx950_dense_grid", "gfx950_dense_qbminor"])
def test_a_short_bottom_right_bias_chunk_keeps_one_kv_segment(spec_id):
    # A short query chunk over a long KV prefix splits the KV range when bias is
    # off; with bias on the factory keeps the unsplit grid the bias path supports.
    from dispatch.attention import attention_tuning_spec

    chunk = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=8192,
        mask_type=2,
        kv_block_size=64,
        num_cus=256,
    )
    plain = attention_tuning_spec(_request(**chunk, use_attn_bias=False), spec_id)
    assert plain.kernel_spec.kv_segments > 1
    spec = attention_tuning_spec(_request(**chunk), spec_id).kernel_spec
    assert spec.use_additive_bias and spec.kv_segments == 1


def test_a_bias_request_sweeps_bias_specs_only():
    from dispatch.attention import attention_candidates
    from rocke.dispatch.tuning import walk

    (grid,) = [c for c in attention_candidates() if c.spec_id == "gfx950_dense_grid"]
    for use_bias in (True, False):
        req = _request(
            algorithm=grid.algorithm, spec_id=grid.spec_id, use_attn_bias=use_bias
        )
        with walk.sweep_level("production"):
            specs = [s.kernel_spec for s in grid.sweep_space(req)]
        assert specs and all(s.use_additive_bias is use_bias for s in specs), use_bias


def test_only_the_grid_dense_candidates_declare_bias():
    from dispatch.attention import attention_candidates

    refused = {
        c.spec_id: c.admits(_request())
        for c in attention_candidates()
        if c.spec_id.startswith("gfx950_dense")
        and c.spec_id not in ("gfx950_dense_grid", "gfx950_dense_qbminor")
    }
    assert set(refused) == {"gfx950_dense_persist", "gfx950_dense_persist_widedma"}
    for spec_id, (ok, why) in refused.items():
        assert not ok and "cannot serve features ['attn_bias']" in why, spec_id


def test_a_q_dtype_bias_is_refused():
    from dispatch.attention import attention_tuning_spec
    from rocke.dispatch.core import PinRefused

    with pytest.raises(
        PinRefused, match=r"additive bias must be fp32, got attn_bias_dtype='q'"
    ):
        attention_tuning_spec(_request(attn_bias_dtype="q"), "gfx950_dense_grid")


@pytest.mark.parametrize("value", ["false", None, 1, 0])
def test_a_non_bool_use_attn_bias_is_refused(value):
    # bool("false") is True: coercing would enable the bias the caller turned off.
    # The request refuses it on construction, before any cache: 1 == True and
    # hashes alike, so a request selected earlier with True would otherwise
    # serve a request with 1 from the candidate's spec cache.
    from dispatch.attention import attention_tuning_spec

    attention_tuning_spec(_request(use_attn_bias=bool(value)), "gfx950_dense_grid")
    with pytest.raises(ValueError, match=r"use_attn_bias must be a bool"):
        _request(use_attn_bias=value)


def test_the_binding_hands_attn_bias_to_the_runner(monkeypatch):
    import kernels.gfx950.attention_dense as dense
    from dispatch.attention import attention_tuning_spec
    from dispatch.attention.bindings import bind_dense_attention_torch

    calls = []
    monkeypatch.setattr(
        dense, "run_attention_dense_torch", lambda **kw: calls.append(kw)
    )
    req = _request()
    ts = attention_tuning_spec(req, "gfx950_dense_grid")
    bias, override = object(), object()
    tensors = {"q": "q", "k": "k", "v": "v", "out": "o", "attn_bias": bias}
    binding = bind_dense_attention_torch(req, ts, tensors)
    binding.launch()
    binding.launch(additive_bias=override)
    no_bias = {k: v for k, v in tensors.items() if k != "attn_bias"}
    bind_dense_attention_torch(req, ts, no_bias).launch()
    assert calls[0]["additive_bias"] is bias and calls[0]["spec"].use_additive_bias
    assert calls[1]["additive_bias"] is override
    assert "additive_bias" not in calls[2]


def test_the_gfx942_binding_refuses_attn_bias():
    from dispatch.attention.bindings import bind_dense_attention_torch

    req = _request(arch="gfx942", use_attn_bias=False)
    ts = SimpleNamespace(
        kernel_spec=SimpleNamespace(head_size=128),
        launch_grid=lambda: (1, 1, 1),
        launch_block=lambda: (256, 1, 1),
    )
    binding = bind_dense_attention_torch(
        req, ts, {"q": 0, "k": 0, "v": 0, "out": 0, "attn_bias": object()}
    )
    with pytest.raises(
        NotImplementedError,
        match=r"gfx942 dense runner does not accept tensors\['attn_bias'\]",
    ):
        binding.launch()
