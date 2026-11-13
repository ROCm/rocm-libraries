# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp32 attention sinks on the gfx950 dense forward (``sink_dtype``): the spec
contract, the kernel name and cache key, the emitted IR, and proof that the GPU
tests' controls break their bound. Host only: no GPU, no compiler."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
)

from ._dense_sink_cases import (
    EXACT_LSE_ATOL,
    EXACT_SINKS,
    ROUND,
    exact_counts,
    exact_lse,
    two_byte_read,
)

_CONTROLS = {
    "bf16-rounded": ROUND["bf16"],
    "fp16-rounded": ROUND["fp16"],
    "bf16-two-byte": lambda s: two_byte_read(s, "bf16"),
    "fp16-two-byte": lambda s: two_byte_read(s, "fp16"),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_exact_sinks_are_not_input_dtype_values(dtype):
    gap = np.abs(ROUND[dtype](EXACT_SINKS).astype(np.float64) - EXACT_SINKS)
    assert gap.min() >= 4 * EXACT_LSE_ATOL, gap


@pytest.mark.parametrize("rows", ["keyed", "keyless"])
@pytest.mark.parametrize("control", sorted(_CONTROLS))
def test_every_control_breaks_the_exact_bound(control, rows):
    """keyed rows guard the running-max seed loads, keyless rows the LSE-store load."""
    counts = exact_counts()
    picked = counts > 0 if rows == "keyed" else counts == 0
    err = np.abs(exact_lse(_CONTROLS[control](EXACT_SINKS)) - exact_lse(EXACT_SINKS))
    err = np.where(np.isfinite(err), err, np.inf)[:, picked]
    assert err.max() >= 4 * EXACT_LSE_ATOL, (control, rows, err.max())


_BASE = dict(
    batch=2,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
    dtype="bf16",
    use_sinks=True,
)


def _spec(**over):
    return Gfx950AttentionDenseSpec(**{**_BASE, **over})


def test_default_sink_dtype_is_the_query_dtype():
    assert _spec().sink_dtype == "q"


@pytest.mark.parametrize("bad", ["F32", "fp32", "bf16", "", None, 1])
def test_unknown_sink_dtype_is_rejected(bad):
    with pytest.raises(ValueError, match="sink_dtype must be 'q' or 'f32'"):
        _spec(sink_dtype=bad)


def test_f32_sinks_require_use_sinks():
    with pytest.raises(ValueError, match=r"sink_dtype='f32' requires use_sinks=True"):
        _spec(use_sinks=False, sink_dtype="f32")


def test_gfx942_has_no_sink_dtype():
    """Scope: the fp32 sink is a gfx950 mode; gfx942 keeps input-dtype sinks."""
    assert "sink_dtype" not in {f.name for f in fields(Gfx942AttentionDenseSpec)}


_NAME_CASES = {
    "grid": {},
    "grid-lse": {"emit_lse": True},
    "grid-d64-window": {"head_size": 64, "sliding_window": 128},
    "persistent": {"persistent": True, "num_persistent": 256},
    "wide-dma": {
        "batch": 1,
        "dtype": "fp16",
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "block_n": 64,
        "persistent": True,
        "num_persistent": 16,
        "persist_decode": "gqa_pair",
        "wide_lds_dma": True,
    },
}


@pytest.mark.parametrize("over", list(_NAME_CASES.values()), ids=list(_NAME_CASES))
def test_f32_adds_one_trailing_name_token(over):
    q = _spec(**over)
    f = replace(q, sink_dtype="f32")
    assert "sinkf32" not in q.kernel_name()
    assert f._algorithm_name_parts() == (*q._algorithm_name_parts(), "sinkf32")
    assert f.kernel_name().replace("_sinkf32", "", 1) == q.kernel_name()


def test_cache_keys_differ_only_in_sink_dtype():
    q = _spec()
    f = replace(q, sink_dtype="f32")
    kq = attention_dense_cache_key(q, arch="gfx950")
    kf = attention_dense_cache_key(f, arch="gfx950")
    assert kq != kf
    assert {a for a, b in zip(kq[2], kf[2]) if a != b} == {("sink_dtype", "q")}
    assert q.runtime_param_fields == f.runtime_param_fields


_KEYLESS = dict(
    batch=1,
    seqlen_q=256,
    seqlen_kv=64,
    num_query_heads=4,
    num_kv_heads=1,
    sliding_window=64,
    emit_lse=True,
)
_IR_CASES = {
    "grid": {},
    "grid-d64": {"head_size": 64},
    "grid-lse-keyless": _KEYLESS,
    "persistent": {"persistent": True, "num_persistent": 256},
    "persistent-lse-keyless-d64": {
        **_KEYLESS,
        "head_size": 64,
        "persistent": True,
        "num_persistent": 256,
    },
}


def _walk(region):
    for op in region.ops:
        yield op
        for inner in op.regions:
            yield from _walk(inner)


def _sink_loads(kernel):
    return [
        op
        for op in _walk(kernel.body)
        if op.name == "memref.global_load_typed" and op.operands[0].name == "%sink_ptr"
    ]


@pytest.mark.parametrize("over", list(_IR_CASES.values()), ids=list(_IR_CASES))
def test_f32_sink_is_one_four_byte_load_per_site(over):
    q = _spec(**over)
    f = replace(q, sink_dtype="f32")
    kq = build_attention_dense(q, arch="gfx950")
    kf = build_attention_dense(f, arch="gfx950")
    lq, lf = _sink_loads(kq), _sink_loads(kf)
    assert len(lq) == len(lf) == (2 if q.emit_lse else 1)
    assert {(o.attrs["elem_type"], o.attrs["align"]) for o in lq} == {
        (q.dtype_ir.name, 2)
    }
    assert {(o.attrs["elem_type"], o.attrs["align"]) for o in lf} == {("f32", 4)}
    loaded = {id(o.result) for o in lf}
    assert not [
        o
        for o in _walk(kf.body)
        if o.name == "arith.cast_to_f32" and id(o.operands[0]) in loaded
    ]
    (param,) = [p for p in kf.params if p.name == "sink_ptr"]
    assert param.type.pointee.name == "f32"
