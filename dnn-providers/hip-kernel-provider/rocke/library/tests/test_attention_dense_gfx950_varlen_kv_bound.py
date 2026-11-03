# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Varlen K/V buffer bound of the gfx950 dense attention grid body, on the IR (no GPU).

Packed varlen K/V tensors hold ``cu_seqlens_kv[batch]`` rows. A K/V buffer resource
bounded by anything larger, such as the spec maxima ``batch * seqlen_kv``, leaves a band
of rows between the packed end and that bound. A load that lands in the band is not
dropped by the hardware bounds check, so it reads past the allocation. Two load sites
reach it: the one-tile-ahead prefetch of the last sequence (every varlen launch whose
sequences are not all full length) and a sliding-window start tile past a sequence's
keys.

The outputs of rows that have keys do not change (the prefetched tile is never
consumed, and a window start tile past the keys feeds only q-blocks that have no key),
so the numeric tests cannot see it. These tests pin the bound on the kernel IR instead,
so they run in every lane, GPU or not.

The checks follow def-use edges from each K/V ``tile.buffer_rsrc`` bound back to the
kernel parameters and global loads it is computed from. Parameters are matched by their
ABI names, which ``attention_dense_signature`` fixes.
"""

from __future__ import annotations

import pytest

from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

# (dtype, head_size, num_kv_heads). The bound's row size is num_kv_heads * head_size * 2
# bytes; two shapes with different products catch a hard-coded or D-specific term.
_SHAPES = {"bf16_d128_kv2": ("bf16", 128, 2), "fp16_d64_kv1": ("fp16", 64, 1)}

# The window start tile is the second load site; LSE changes the varlen epilogue.
_VARIANTS = {
    "causal": {},
    "window": {"sliding_window": 64},
    "lse": {"emit_lse": True},
    "window_lse": {"sliding_window": 64, "emit_lse": True},
}


def _spec(varlen, shape, **kw):
    dtype, head_size, num_kv_heads = _SHAPES[shape]
    return AttentionDenseSpec(
        batch=2,
        seqlen_q=256,
        seqlen_kv=256,
        num_query_heads=8,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        causal=True,
        dtype=dtype,
        persistent=False,
        varlen=varlen,
        **kw,
    )


def _kv_row_bytes(shape):
    """Bytes per KV row: the multiplier that turns a row count into the bound."""
    _, head_size, num_kv_heads = _SHAPES[shape]
    return num_kv_heads * head_size * 2


def _walk(region):
    for op in region.ops:
        yield op
        for inner in op.regions:
            yield from _walk(inner)


def _kv_bounds(kernel):
    """The byte bound of the K and the V buffer resource, as (pointer name, Value)."""
    return [
        (op.operands[0].name, op.operands[1])
        for op in _walk(kernel.body)
        if op.name == "tile.buffer_rsrc" and op.operands[0].name in ("%k_ptr", "%v_ptr")
    ]


def _sources(value):
    """What ``value`` is computed from: ``(params, loads)``.

    ``params`` holds the names of the kernel parameters reached by def-use edges.
    ``loads`` holds the ``(pointer, index)`` names of every global load reached; a load
    is a leaf, its own operands are not followed.
    """
    params, loads, seen = set(), set(), set()
    pending = [value]
    while pending:
        v = pending.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if v.op is None:
            params.add(v.name)
        elif v.op.name == "memref.global_load_typed":
            loads.add(tuple(o.name for o in v.op.operands))
        else:
            pending.extend(v.op.operands)
    return params, loads


def _split_bound(bound):
    """(rows, row_bytes) of a bound that must be exactly rows * row_bytes."""
    assert bound.op is not None, "bound is a kernel parameter, not a row scaling"
    assert bound.op.name == "arith.mul", f"bound is {bound.op.name}, not a row scaling"
    consts = [
        o
        for o in bound.op.operands
        if o.op is not None and o.op.name == "arith.constant"
    ]
    assert len(consts) == 1, "expected one constant multiplier"
    (rows,) = [o for o in bound.op.operands if o is not consts[0]]
    return rows, consts[0].op.attrs["value"]


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_varlen_kv_bound_is_the_packed_extent(variant, shape):
    """Both K and V are bounded by exactly cu_seqlens_kv[batch] rows, nothing added."""
    spec = _spec(True, shape, **_VARIANTS[variant])
    bounds = _kv_bounds(build_attention_dense(spec))
    assert sorted(name for name, _ in bounds) == ["%k_ptr", "%v_ptr"]
    for name, bound in bounds:
        rows, row_bytes = _split_bound(bound)
        # The row count must BE the load: an add, max or select around it (say one extra
        # block_n of slack) still reads only cu_seqlens_kv[batch] but re-opens the band.
        assert rows.op is not None and rows.op.name == "memref.global_load_typed", (
            f"{name} bound counts rows with {rows.op.name if rows.op else rows.name}; "
            "it must be cu_seqlens_kv[batch] itself"
        )
        assert tuple(o.name for o in rows.op.operands) == ("%cu_seqlens_kv", "%batch")
        assert row_bytes == _kv_row_bytes(shape)


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_dense_kv_bound_stays_the_spec_extent(variant, shape):
    """Control: a non-varlen bound stays batch * seqlen_kv, with no global load."""
    spec = _spec(False, shape, **_VARIANTS[variant])
    bounds = _kv_bounds(build_attention_dense(spec))
    assert sorted(name for name, _ in bounds) == ["%k_ptr", "%v_ptr"]
    for name, bound in bounds:
        params, loads = _sources(bound)
        assert params == {"%batch", "%seqlen_kv"}, f"{name}: {sorted(params)}"
        assert not loads, f"{name} bound loads {sorted(loads)}"
        assert _split_bound(bound)[1] == _kv_row_bytes(shape)
