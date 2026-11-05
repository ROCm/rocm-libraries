# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Varlen K/V buffer bound of the gfx950 dense attention grid body, on the IR (no GPU).

Packed varlen K/V tensors hold sequence ``s`` in rows ``cu_seqlens_kv[s]`` up to
``cu_seqlens_kv[s + 1]``. A K/V buffer resource bounded by anything past the end of the
workgroup's own sequence, such as the spec maxima ``batch * seqlen_kv``, lets two loads
read beyond it: the one-tile-ahead prefetch after the sequence's last KV tile, and a
sliding-window start tile past the sequence's keys. For the last sequence those reads
land past the allocation; for the others, in the next sequence's rows.

The outputs of rows that have keys do not change (the prefetched tile is never
consumed, and a window start tile past the keys feeds only q-blocks that have no key),
so the numeric tests cannot see it. These tests pin the bound on the kernel IR instead,
so they run in every lane, GPU or not. If the varlen test fails, look at the
``elif varlen:`` branch of the K/V ``buffer_rsrc`` block in ``build_attention_dense``.

The varlen check follows def-use edges from each K/V ``tile.buffer_rsrc`` bound back to
the ``cu_seqlens_kv`` load it is computed from; the non-varlen control evaluates the
bound at fixed kernel parameter values. Parameters are matched by their ABI names,
which ``attention_dense_signature`` fixes.
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

# Non-varlen controls, as (spec fields, runtime_shape). The _VARIANTS specs are
# runtime_shape (the row size comes from the num_kv_heads parameter); ragged bakes the
# shape, so it covers the constant row size. Ragged at whole-tile lengths takes no
# sliding window.
_CONTROLS = {
    **{name: (kw, True) for name, kw in _VARIANTS.items()},
    "ragged": ({"ragged": True}, False),
    "ragged_lse": ({"ragged": True, "emit_lse": True}, False),
}

# Kernel parameter values at which the control evaluates a non-varlen bound.
# num_kv_heads matches no _SHAPES head count, so a runtime_shape bound that bakes the
# spec's row size instead of reading the parameter evaluates to a different number.
_PARAMS = {"%batch": 3, "%seqlen_kv": 5, "%num_kv_heads": 7}


def _spec(shape, *, varlen, **kw):
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
    bounds = [
        (op.operands[0].name, op.operands[1])
        for op in _walk(kernel.body)
        if op.name == "tile.buffer_rsrc" and op.operands[0].name in ("%k_ptr", "%v_ptr")
    ]
    names = sorted(name for name, _ in bounds)
    assert names == [
        "%k_ptr",
        "%v_ptr",
    ], f"want one K and one V tile.buffer_rsrc, found {names}"
    return bounds


def _load_indices(kernel, pointer):
    """The index Value of every global load from the pointer parameter ``pointer``."""
    return [
        op.operands[1]
        for op in _walk(kernel.body)
        if op.name == "memref.global_load_typed" and op.operands[0].name == pointer
    ]


def _peel(index):
    """``index`` as ``(base, offset)``, peeling every ``arith.add`` of a constant:
    ``(bt + 1) + 1`` reads as ``(bt, 2)``, whichever Values the adds share."""
    offset = 0
    while index.op is not None and index.op.name == "arith.add":
        consts = [
            o
            for o in index.op.operands
            if o.op is not None and o.op.name == "arith.constant"
        ]
        if len(consts) != 1:
            break
        (index,) = [o for o in index.op.operands if o is not consts[0]]
        offset += consts[0].op.attrs["value"]
    return index, offset


def _evaluate(value, name):
    """The integer ``value`` computes when the kernel parameters hold ``_PARAMS``;
    ``name`` names the bound in failures."""
    if value.op is None:
        assert (
            value.name in _PARAMS
        ), f"{name} bound reads parameter {value.name}; want only {sorted(_PARAMS)}"
        return _PARAMS[value.name]
    if value.op.name == "arith.constant":
        return value.op.attrs["value"]
    assert value.op.name == "arith.mul", (
        f"{name} bound computes {value.op.name} ({value.name}); a non-varlen bound is "
        "a product of kernel parameters and constants, with no global load"
    )
    lhs, rhs = (_evaluate(o, name) for o in value.op.operands)
    return lhs * rhs


def _split_const(value, what):
    """(other operand, constant) of ``value``, a binary op whose operands are one
    ``arith.constant`` and one other Value; ``what`` names ``value`` in failures."""
    assert value.op is not None, f"{what} is a kernel parameter, not computed"
    consts = [
        o
        for o in value.op.operands
        if o.op is not None and o.op.name == "arith.constant"
    ]
    assert len(consts) == 1, (
        f"{what} is {value.op.name}({', '.join(o.name for o in value.op.operands)}); "
        "expected one constant operand"
    )
    (other,) = [o for o in value.op.operands if o is not consts[0]]
    return other, consts[0].op.attrs["value"]


def _split_bound(bound, name):
    """(rows, row_bytes) of a bound that must be exactly rows * row_bytes."""
    assert bound.op is not None, f"{name} bound is a parameter, not a row scaling"
    assert (
        bound.op.name == "arith.mul"
    ), f"{name} bound is {bound.op.name}, not a row scaling"
    return _split_const(bound, f"{name} bound")


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_varlen_kv_bound_is_the_sequence_end(variant, shape):
    """Both K and V are bounded by exactly cu_seqlens_kv[s + 1] rows, nothing added,
    where s is the sequence whose queries the workgroup reads (cu_seqlens_q[s])."""
    spec = _spec(shape, varlen=True, **_VARIANTS[variant])
    kernel = build_attention_dense(spec)
    bounds = _kv_bounds(kernel)
    # The workgroup's own sequence s: its lowest cu_seqlens_q index (its Q start).
    q_seq, q_off = min(
        map(_peel, _load_indices(kernel, "%cu_seqlens_q")), key=lambda p: p[1]
    )
    for name, bound in bounds:
        rows, row_bytes = _split_bound(bound, name)
        # The row count must BE the load: an add, max or select around it (say one extra
        # block_n of slack) re-opens reads past the sequence's end.
        assert rows.op is not None and rows.op.name == "memref.global_load_typed", (
            f"{name} bound counts rows with "
            f"{rows.op.name if rows.op is not None else rows.name}; "
            "it must be cu_seqlens_kv[bt + 1] itself"
        )
        pointer, index = rows.op.operands
        assert pointer.name == "%cu_seqlens_kv", f"{name} bound loads {pointer.name}"
        # Peeled, so (bt + 1) + 1 cannot pass as bt' + 1. Rules out cu_seqlens_kv[batch]
        # (the packed total), [bt] and [bt + 2].
        seq, off = _peel(index)
        assert seq is q_seq and off == q_off + 1, (
            f"{name} bound loads cu_seqlens_kv[{seq.name} + {off}]; want "
            f"cu_seqlens_kv[{q_seq.name} + {q_off + 1}], the end of the sequence whose "
            f"queries start at cu_seqlens_q[{q_seq.name} + {q_off}]"
        )
        assert row_bytes == _kv_row_bytes(shape), (
            f"{name} row size is {row_bytes} bytes; want num_kv_heads * head_size * 2 "
            f"= {_kv_row_bytes(shape)}"
        )


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("variant", list(_CONTROLS))
def test_nonvarlen_kv_bound_stays_the_spec_extent(variant, shape):
    """Control: a non-varlen bound stays batch * seqlen_kv rows of num_kv_heads *
    head_size * 2 bytes, read from the kernel parameters, with no global load.

    A runtime_shape kernel (one binary serves every head count) must read the row size
    from the num_kv_heads parameter, so a bound that bakes the spec's head count
    evaluates to a different number at _PARAMS; a baked kernel multiplies by the
    spec's constant row size."""
    kw, runtime_shape = _CONTROLS[variant]
    spec = _spec(shape, varlen=False, **kw)
    assert spec.runtime_shape == runtime_shape, (
        f"control spec has runtime_shape={spec.runtime_shape}, picked for "
        f"{runtime_shape}; keep both kinds covered (AttentionDenseSpec.runtime_shape)"
    )
    bounds = _kv_bounds(build_attention_dense(spec))
    _, head_size, num_kv_heads = _SHAPES[shape]
    heads = _PARAMS["%num_kv_heads"] if runtime_shape else num_kv_heads
    want = _PARAMS["%batch"] * _PARAMS["%seqlen_kv"] * heads * head_size * 2
    for name, bound in bounds:
        got = _evaluate(bound, name)
        assert got == want, (
            f"{name} bound is {got} bytes at {_PARAMS}; want batch * seqlen_kv * "
            f"{'num_kv_heads' if runtime_shape else num_kv_heads} * {head_size} * 2 "
            f"= {want}. The non-varlen branch must keep develop's _kv_elem_bytes_ir(): "
            "it reads num_kv_heads under runtime_shape"
        )
