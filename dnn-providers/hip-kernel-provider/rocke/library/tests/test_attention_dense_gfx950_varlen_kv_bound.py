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

The bodies that read their strides bound each (batch, KV head) slab instead, on gfx950
and gfx942 alike; an over-wide slab bound is just as invisible to the numeric tests,
so :func:`test_strided_kv_bound_is_the_slab_extent` pins it here for both, and
:func:`test_strided_kv_bound_never_passes_the_view_end` ties it to the end of K/V views
of several layouts, at key counts that are and are not a multiple of ``block_n``: the
last slab ends on the allocation's end, never past it.

The host admits a stride only if the furthest row a K/V load may address,
``dense_kv_reach_rows`` (the padded length plus ``DENSE_SLAB_SLACK_TILES`` tiles), stays
inside 32-bit buffer offsets. A body that loaded further would wrap the offset with no
error. :func:`test_kv_loads_stay_inside_the_admitted_reach` runs the integer part of
each strided body's IR for every workgroup and finds the furthest row it loads, and
:func:`test_the_slack_is_what_the_deepest_prefetch_needs` checks that some body needs
all of the slack, so the constant follows the kernels both ways. The walk measures
every K and V access or fails: a K/V pointer or buffer resource that reaches anything
but the loads it knows fails the test, so a new kind of load cannot hide a deeper
prefetch (:func:`test_the_reach_walk_refuses_a_kv_access_it_cannot_measure`).

The same integer walk checks which slab each work item reads
(:func:`test_each_work_item_reads_its_own_slabs`): the stride probe of
``test_attention_dense_stride_reads.py`` only ties each stride to its tensor, so a slab
found with the tensor's own strides on the wrong axes, or for the wrong head, would
pass it, and a batch-1 device lane cannot see every such mistake.
"""

from __future__ import annotations

import functools
import operator
import types

import pytest
from rocke.core.ir import Op, Value

from kernels.common._attention_dense_strides import (
    DENSE_ELEM_BYTES,
    DENSE_SLAB_SLACK_TILES,
    dense_kv_reach_rows,
)
from kernels.gfx942 import attention_dense as k942
from kernels.gfx950 import attention_dense as k950
from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

_MODULES = {"gfx942": k942, "gfx950": k950}
_SPEC_TYPES = {"gfx942": k942.Gfx942AttentionDenseSpec, "gfx950": AttentionDenseSpec}

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

# Non-varlen controls that keep packed addressing and the packed bound, as (spec
# fields, runtime_shape). The additive-bias body is runtime_shape (the row size comes
# from the num_kv_heads parameter); ragged bakes the shape, so it covers the constant
# row size. Ragged at whole-tile lengths takes no sliding window.
_CONTROLS = {
    "bias": ({"use_additive_bias": True}, True),
    "ragged": ({"ragged": True}, False),
    "ragged_lse": ({"ragged": True, "emit_lse": True}, False),
}

# The bodies that read their strides, as (arch, body) -> spec fields: the grid body
# (the _VARIANTS), its KV-segment form, the persistent body and the wide-DMA body
# (head size 128 only) on gfx950; the grid body (without a mask too: at BF16 D128 that
# is the body with the per-row LDS destination), its KV-segment form and the
# persistent body on gfx942.
_PERSIST_950 = {"persistent": True, "num_persistent": 256}
_PERSIST_942 = {"persistent": True, "num_persistent": 304}
_SLAB_BODIES = {
    **{("gfx950", name): kw for name, kw in _VARIANTS.items()},
    ("gfx950", "kv_segments"): {"kv_segments": 2},
    ("gfx950", "persistent"): _PERSIST_950,
    ("gfx950", "persistent_lse"): {**_PERSIST_950, "emit_lse": True},
    ("gfx950", "persistent_window"): {**_PERSIST_950, "sliding_window": 64},
    ("gfx950", "wide_dma"): {
        **_PERSIST_950,
        "wide_lds_dma": True,
        "lds_k_group_pad": 8,
    },
    ("gfx942", "causal"): {},
    ("gfx942", "noncausal"): {"causal": False},
    ("gfx942", "window"): {"sliding_window": 64},
    ("gfx942", "lse"): {"emit_lse": True},
    ("gfx942", "kv_segments"): {"kv_segments": 2},
    ("gfx942", "persistent"): _PERSIST_942,
    ("gfx942", "persistent_window"): {**_PERSIST_942, "sliding_window": 64},
}

# Kernel parameter values at which the control evaluates a non-varlen bound.
# num_kv_heads matches no _SHAPES head count, so a runtime_shape bound that bakes the
# spec's row size instead of reading the parameter evaluates to a different number.
# The two token strides differ, so a V bound that borrows K's stride is caught.
_PARAMS = {
    "%batch": 3,
    "%seqlen_kv": 5,
    "%num_kv_heads": 7,
    "%stride_k_token": 24,
    "%stride_v_token": 40,
}


def _spec(shape, *, varlen, persistent=False, arch="gfx950", **kw):
    dtype, head_size, num_kv_heads = _SHAPES[shape]
    fields = {
        "batch": 2,
        "seqlen_q": 256,
        "seqlen_kv": 256,
        "num_query_heads": 8,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "causal": True,
        "dtype": dtype,
        "persistent": persistent,
        "varlen": varlen,
    }
    return _SPEC_TYPES[arch](**{**fields, **kw})


def _kv_row_bytes(shape):
    """Bytes per KV row: the multiplier that turns a row count into the bound."""
    _, head_size, num_kv_heads = _SHAPES[shape]
    return num_kv_heads * head_size * 2


def _walk(region):
    for op in region.ops:
        yield op
        for inner in op.regions:
            yield from _walk(inner)


def _base_pointer(pointer):
    """The kernel parameter ``pointer`` is, or the one it is moved from by a
    ``tile.global_ptr_add`` (a strided body bounds the slab, not the whole tensor)."""
    while pointer.op is not None and pointer.op.name == "tile.global_ptr_add":
        pointer = pointer.op.operands[0]
    return pointer


def _kv_bounds(kernel):
    """The byte bound of the K and the V buffer resource, as (pointer name, Value)."""
    bounds = [
        (_base_pointer(op.operands[0]).name, op.operands[1])
        for op in _walk(kernel.body)
        if op.name == "tile.buffer_rsrc"
        and _base_pointer(op.operands[0]).name in ("%k_ptr", "%v_ptr")
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


_ARITH = {
    "arith.mul": lambda a, b: a * b,
    "arith.add": lambda a, b: a + b,
    "arith.sub": lambda a, b: a - b,
    "arith.cmp": None,
    "arith.select": None,
}


def _evaluate(value, name, params=_PARAMS):
    """The integer ``value`` computes when the kernel parameters hold ``params``;
    ``name`` names the bound in failures."""
    if value.op is None:
        assert (
            value.name in params
        ), f"{name} bound reads parameter {value.name}; want only {sorted(params)}"
        return params[value.name]
    if value.op.name == "arith.constant":
        return value.op.attrs["value"]
    assert value.op.name in _ARITH, (
        f"{name} bound computes {value.op.name} ({value.name}); a non-varlen bound is "
        "arithmetic on kernel parameters and constants, with no global load"
    )
    operands = [_evaluate(o, name, params) for o in value.op.operands]
    if value.op.name == "arith.cmp":
        assert value.op.attrs["pred"] == "gt", f"{name} bound compares with a non-gt"
        return operands[0] > operands[1]
    if value.op.name == "arith.select":
        return operands[1] if operands[0] else operands[2]
    return _ARITH[value.op.name](*operands)


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


@functools.lru_cache(maxsize=1)
def _build(arch, spec):
    """The kernel of ``spec``. Consecutive calls for one spec share it (the reach
    tests walk one kernel at several key counts), so no caller may edit it."""
    return _MODULES[arch].build_attention_dense(spec, arch=arch)


# Every (arch, body, shape) that reads its strides; the wide-DMA body is head size 128 only.
_SLAB_CASES = [
    pytest.param(arch, body, shape, id=f"{arch}-{body}-{shape}")
    for arch, body in _SLAB_BODIES
    for shape in _SHAPES
    if body != "wide_dma" or _SHAPES[shape][1] == 128
]


@pytest.mark.parametrize(("arch", "body", "shape"), _SLAB_CASES)
def test_strided_kv_bound_is_the_slab_extent(arch, body, shape):
    """A body that reads its strides bounds each (batch, KV head) slab of K and of V
    by its last valid row: ``seqlen_kv > 0 ? 2 * ((seqlen_kv - 1) * stride + D) : 0``
    bytes with the TENSOR'S OWN token stride, read from the kernel parameters, with
    no global load. The slab base carries the batch and head offsets, so no batch
    term appears. A persistent body forms the bound once per workgroup (it does not
    depend on the work item) for a buffer resource rebuilt per work item. Evaluated
    at a second point with no keys, where the bound must be 0."""
    spec = _spec(shape, varlen=False, arch=arch, **_SLAB_BODIES[(arch, body)])
    assert _MODULES[arch].body_serves_strides(spec), "not a body that reads strides"
    _, head_size, _ = _SHAPES[shape]
    bounds = _kv_bounds(_build(arch, spec))
    for params in (_PARAMS, {**_PARAMS, "%seqlen_kv": 0}):
        n = params["%seqlen_kv"]
        for name, bound in bounds:
            token = "stride_k_token" if name == "%k_ptr" else "stride_v_token"
            got = _evaluate(bound, name, params)
            want = 2 * ((n - 1) * params[f"%{token}"] + head_size) if n > 0 else 0
            assert got == want, (
                f"{name} bound is {got} bytes at {params}; want seqlen_kv > 0 ? "
                f"2 * ((seqlen_kv - 1) * {token} + {head_size}) : 0 = {want}"
            )


# A K/V view of dims [B, Skv, Hkv, D] as (batch, head, token) element strides, by
# layout: contiguous [B, Skv, Hkv, D]; contiguous [B, Hkv, Skv, D] read through the same
# logical dims; and token rows padded by 8 elements (a multiple the host admits).
_VIEW_LAYOUTS = {
    "packed": lambda s, h, d: (s * h * d, d, h * d),
    "head_major": lambda s, h, d: (h * s * d, s * d, d),
    "padded_rows": lambda s, h, d: (s * (h * d + 8), d, h * d + 8),
}


@pytest.mark.parametrize(("arch", "body", "shape"), _SLAB_CASES)
def test_strided_kv_bound_never_passes_the_view_end(arch, body, shape):
    """No K or V slab bound reaches past the end of its view, and the last (batch, head)
    slab ends exactly on it. The view is the whole allocation: its last byte is the
    last element of the last row of the last slab. So the padded tail of the last tile
    and the host's slack tiles, which a load still forms offsets into, read as 0 and
    never touch memory past the allocation: the bound is the valid rows, not the reach
    :func:`dense_kv_reach_rows` admits strides for.

    The bound of each body's IR is evaluated at key counts that are a multiple of
    ``block_n`` (one tile, three tiles), that are not (one past a tile, five short of
    three tiles) and one key, over every pair of K and V layouts, so a V bound that
    borrows K's token stride, a bound that rounds up to ``block_n`` and one that adds
    slack tiles each end past the view for some slab."""
    spec = _spec(shape, varlen=False, arch=arch, **_SLAB_BODIES[(arch, body)])
    assert _MODULES[arch].body_serves_strides(spec), "not a body that reads strides"
    _, head_size, hkv = _SHAPES[shape]
    block_n, batch = spec.block_n, 3
    bounds = _kv_bounds(_build(arch, spec))
    keys = (block_n, 3 * block_n, block_n + 1, 3 * block_n - 5, 1)
    for skv in keys:
        for k_layout in _VIEW_LAYOUTS:
            for v_layout in _VIEW_LAYOUTS:
                views = {
                    "%k_ptr": _VIEW_LAYOUTS[k_layout](skv, hkv, head_size),
                    "%v_ptr": _VIEW_LAYOUTS[v_layout](skv, hkv, head_size),
                }
                params = {
                    **_PARAMS,
                    "%seqlen_kv": skv,
                    "%stride_k_token": views["%k_ptr"][2],
                    "%stride_v_token": views["%v_ptr"][2],
                }
                for name, bound in bounds:
                    sb, sh, st = views[name]
                    # One past the view's last element, in bytes.
                    view_end = DENSE_ELEM_BYTES * (
                        (batch - 1) * sb + (hkv - 1) * sh + (skv - 1) * st + head_size
                    )
                    nrec = _evaluate(bound, name, params)
                    where = f"{name} at seqlen_kv={skv} (K {k_layout}, V {v_layout})"
                    for b_idx in range(batch):
                        for h_idx in range(hkv):
                            end = DENSE_ELEM_BYTES * (b_idx * sb + h_idx * sh) + nrec
                            assert end <= view_end, (
                                f"{where}: the slab of batch {b_idx}, head {h_idx} "
                                f"ends at byte {end}, {end - view_end} past the "
                                f"view's end {view_end}; the bound is {nrec} bytes"
                            )
                    last_end = (
                        DENSE_ELEM_BYTES * ((batch - 1) * sb + (hkv - 1) * sh) + nrec
                    )
                    assert last_end == view_end, (
                        f"{where}: the last slab ends at byte {last_end}, the view "
                        f"at {view_end}: the bound is not the extent of the valid rows"
                    )


# ---- The furthest K/V row a strided body loads, against the host's slack ----

# Token strides for the reach walk: larger than any head size, so the row of a load is
# its element offset divided by the stride (the column stays below it). K's and V's
# differ, so a load that steps by the other tensor's stride lands on another row.
_REACH_STRIDES = {"%k_ptr": 4096, "%v_ptr": 6144}
# The K/V loads of the strided bodies, as op name -> index of the voffset operand
# (the soffset follows it): the LDS-DMA copies, and the gfx942 cfvst V loads.
_KV_LOADS = {"tile.async_buffer_load_lds_addr": 2, "tile.buffer_load_vN": 1}
_WAVE = 64
_U32 = 0xFFFFFFFF


def _sdiv(a, b):
    """i32 ``sdiv``: the quotient rounded toward zero."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


_INT_OPS = {
    "arith.add": operator.add,
    "arith.sub": operator.sub,
    "arith.mul": operator.mul,
    "arith.div": _sdiv,
    "arith.mod": lambda a, b: a - b * _sdiv(a, b),
    "arith.and": operator.and_,
    "arith.or": operator.or_,
    "arith.shl": operator.lshift,
    "arith.lshr": lambda a, b: (a & _U32) >> b,
    "arith.umul_hi_i32": lambda a, b: ((a & _U32) * (b & _U32)) >> 32,
    "arith.smax": max,
}
_PREDICATES = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "ult": lambda a, b: (a & _U32) < (b & _U32),
}


class _Thread:
    """The integer Values of one thread of one workgroup: kernel parameters, block
    and thread ids, the induction variables of the loops being walked, and the
    integer ops between them. Anything else on an address path fails the test with
    its op name, so the walk never guesses."""

    def __init__(self, params, block, tid):
        self._params, self._block, self._tid = params, block, tid
        self._ivs = {}
        self._memo = {}

    def set_iv(self, loop, value):
        self._ivs[id(loop)] = value
        self._memo.clear()

    def value(self, v, tid=None):
        tid = self._tid if tid is None else tid
        key = (id(v), tid)
        if key not in self._memo:
            self._memo[key] = self._compute(v, tid)
        return self._memo[key]

    def _compute(self, v, tid):
        op = v.op
        if op is None:
            assert v.name in self._params, f"address reads parameter {v.name}"
            return self._params[v.name]
        name = op.name
        operands = op.operands
        if name == "scf.for":
            assert v.name == op.attrs["iv"], f"address reads loop-carried {v.name}"
            return self._ivs[id(op)]
        if name == "arith.constant":
            return op.attrs["value"]
        if name == "gpu.thread_id":
            return tid
        if name == "gpu.block_id":
            return self._block["xyz".index(op.attrs["axis"])]
        if name == "tile.lane_id":
            return tid % _WAVE
        if name == "tile.readfirstlane":
            # The value of the wave's first lane.
            return self.value(operands[0], tid - tid % _WAVE)
        if name == "tile.pin_sgpr":
            return self.value(operands[0], tid)
        if name == "arith.zext":
            return self.value(operands[0], tid) & _U32
        if name == "arith.cmp":
            a, b = (self.value(o, tid) for o in operands)
            return _PREDICATES[op.attrs["pred"]](a, b)
        if name == "arith.select":
            cond, a, b = (self.value(o, tid) for o in operands)
            return a if cond else b
        assert name in _INT_OPS, f"address computes {name} ({v.name})"
        a, b = (self.value(o, tid) for o in operands)
        return _INT_OPS[name](a, b)


def _kv_load_tensor(op):
    """``"%k_ptr"`` or ``"%v_ptr"`` when ``op`` loads from a K or V buffer resource,
    else None."""
    if op.name not in _KV_LOADS:
        return None
    rsrc = op.operands[0]
    if rsrc.op is None or rsrc.op.name != "tile.buffer_rsrc":
        return None
    name = _base_pointer(rsrc.op.operands[0]).name
    return name if name in _REACH_STRIDES else None


def _kv_pointer(value):
    """``"%k_ptr"`` or ``"%v_ptr"`` when ``value`` is that kernel parameter or a
    pointer ``tile.global_ptr_add`` moves from it, else None."""
    if value.op is not None and value.op.name != "tile.global_ptr_add":
        return None
    base = _base_pointer(value)
    return base.name if base.op is None and base.name in _REACH_STRIDES else None


def _refuse_unmeasured_kv_access(kernel):
    """Fail unless the reach walk measures every K and V access of ``kernel``. A K
    or V pointer may only move (``tile.global_ptr_add``) or found a buffer resource
    (``tile.buffer_rsrc``), and such a resource may only be the resource operand of
    a load in ``_KV_LOADS``, whose voffset the walk reads. Any other use, a new load
    op or a resource passed through another op, would read K or V at offsets the
    walk never sees."""
    for op in _walk(kernel.body):
        for at, operand in enumerate(op.operands):
            pointer = _kv_pointer(operand)
            resource = (
                _kv_pointer(operand.op.operands[0])
                if operand.op is not None and operand.op.name == "tile.buffer_rsrc"
                else None
            )
            if pointer and (at, op.name) not in (
                (0, "tile.global_ptr_add"),
                (0, "tile.buffer_rsrc"),
            ):
                what = f"the {pointer} pointer"
            elif resource and (at != 0 or op.name not in _KV_LOADS):
                what = f"the {resource} buffer resource"
            else:
                continue
            raise AssertionError(
                f"{op.name} takes {what} as operand {at}; the reach walk measures K "
                f"and V only through a tile.buffer_rsrc read by {sorted(_KV_LOADS)}. "
                "Teach it this op (_KV_LOADS), or its rows go unmeasured"
            )


def _path_walk(wanted, what):
    """``walk(region, thread, visit)``: one thread's run of ``region`` along the ops
    ``wanted`` accepts. Each ``scf.for`` that holds one runs its real trip count, an
    ``scf.if`` (a then region only; rocKE spells an if/else ``scf.if_else``) runs
    when its condition holds, and ``visit(op, thread, loops)`` sees each accepted op,
    ``loops`` naming the iterations around it. Any other op that holds one fails,
    naming ``what``, so the walk never skips one."""
    on_path = {}

    def ops_to_walk(region):
        """The ops of ``region`` that ``wanted`` accepts or that hold one."""
        if id(region) not in on_path:
            on_path[id(region)] = [
                op
                for op in region.ops
                if wanted(op) or any(ops_to_walk(r) for r in op.regions)
            ]
        return on_path[id(region)]

    def walk(region, thread, visit, loops=()):
        for op in ops_to_walk(region):
            if wanted(op):
                visit(op, thread, loops)
            elif op.name == "scf.for":
                lo, hi, step = (thread.value(o) for o in op.operands[:3])
                for i in range(lo, hi, step):
                    thread.set_iv(op, i)
                    walk(op.regions[0], thread, visit, (*loops, (id(op), i)))
            elif op.name == "scf.if":
                if thread.value(op.operands[0]):
                    walk(op.regions[0], thread, visit, loops)
            else:
                raise AssertionError(f"{what} inside {op.name}; teach the walk")

    return walk


def _workgroup_threads(module, spec):
    """``(block ids, thread id)`` of the first and last lane of the first and last
    wave of every workgroup of ``spec``'s launch."""
    threads = module.attention_dense_block(spec)[0]
    grid = module.attention_dense_grid(spec)
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                for tid in (0, _WAVE - 1, threads - _WAVE, threads - 1):
                    yield (x, y, z), tid


def _kv_rows(arch, spec, kernel, seqlen_kv):
    """:func:`_furthest_kv_rows` of ``kernel``, a build of ``spec``."""
    _refuse_unmeasured_kv_access(kernel)
    module = _MODULES[arch]
    params = {f"%{k}": v for k, v in module.attention_dense_runtime_args(spec).items()}
    params["%seqlen_kv"] = seqlen_kv
    params["%stride_k_token"] = _REACH_STRIDES["%k_ptr"]
    params["%stride_v_token"] = _REACH_STRIDES["%v_ptr"]
    rows = {}

    def visit(op, thread, _loops):
        tensor = _kv_load_tensor(op)
        at = _KV_LOADS[op.name]
        offset = sum(thread.value(o) for o in op.operands[at : at + 2])
        row = offset // DENSE_ELEM_BYTES // _REACH_STRIDES[tensor]
        rows[tensor] = max(rows.get(tensor, row), row)

    walk = _path_walk(lambda op: _kv_load_tensor(op) is not None, "a K/V load")
    for block, tid in _workgroup_threads(module, spec):
        walk(kernel.body, _Thread(params, block, tid), visit)
    return rows


@functools.cache
def _furthest_kv_rows(arch, spec, seqlen_kv):
    """The furthest K row and V row any workgroup of ``spec``'s launch loads when the
    kernel runs with ``seqlen_kv`` keys, as ``{pointer name: row}`` (no entry: no
    load). Each workgroup is walked for the first and last lane of its first and last
    wave; every loop that holds a K/V load runs its real trip count. Cached, read
    only: the slack test reads the launches the reach test walks."""
    return types.MappingProxyType(_kv_rows(arch, spec, _build(arch, spec), seqlen_kv))


def _reach_spec(arch, body, shape):
    """The ``_SLAB_BODIES`` spec at one workgroup per query block and head pair (batch
    1, as many query heads as KV heads, two persistent workgroups), so the walk
    stays fast; the KV loop and its prefetch do not depend on those counts."""
    kw = {**_SLAB_BODIES[(arch, body)], "batch": 1}
    kw["num_query_heads"] = _SHAPES[shape][2]
    if kw.get("persistent"):
        kw["num_persistent"] = 2
    return _spec(shape, varlen=False, arch=arch, **kw)


def _reach_points(block_n):
    """Key counts the reach is walked at: none (an empty KV loop, where a prologue
    still prefetches), one tile (the second of two KV segments is empty) and three
    tiles."""
    return (0, block_n, 3 * block_n)


_REACH_CASES = _SLAB_CASES


@pytest.mark.parametrize(("arch", "body", "shape"), _REACH_CASES)
def test_kv_loads_stay_inside_the_admitted_reach(arch, body, shape):
    """No K or V load of a body that reads its strides goes past the rows the host
    admits strides for, ``dense_kv_reach_rows(seqlen_kv, block_n)``: the padded
    length plus ``DENSE_SLAB_SLACK_TILES`` tiles. Past it, a buffer offset could
    wrap past 2^31 bytes with no error. Walked at every workgroup with no keys, one
    tile and three tiles of keys."""
    spec = _reach_spec(arch, body, shape)
    assert _MODULES[arch].body_serves_strides(spec), "not a body that reads strides"
    for n in _reach_points(spec.block_n):
        reach = dense_kv_reach_rows(n, spec.block_n)
        rows = _furthest_kv_rows(arch, spec, n)
        if n:
            assert sorted(rows) == ["%k_ptr", "%v_ptr"], (
                f"found K/V loads of {sorted(rows)} at seqlen_kv={n}; want both: "
                "the walk no longer recognizes the body's K/V loads (_KV_LOADS)"
            )
        for name, row in rows.items():
            assert row < reach, (
                f"{name} loads row {row} at seqlen_kv={n}, block_n={spec.block_n}; "
                f"the host admits strides for rows below {reach} "
                f"(DENSE_SLAB_SLACK_TILES={DENSE_SLAB_SLACK_TILES}). A body that "
                "prefetches deeper must raise DENSE_SLAB_SLACK_TILES and its C++ twin"
            )


def test_the_slack_is_what_the_deepest_prefetch_needs():
    """Some body loads the last row the host admits, so DENSE_SLAB_SLACK_TILES is no
    larger than the kernels need. The gfx950 grid, KV-segment and persistent bodies
    prime their K/V double buffer with tiles ``start_tile`` and ``start_tile + 1``,
    and an empty KV range starts at its upper end ``n_upper``, so they load tile
    ``n_upper + 1``, two tiles past the padded length: with no keys, and in an empty
    KV segment. The gfx942 bodies load tile ``j`` inside the loop and stop at
    ``n_upper - 1``; the gfx950 sliding-window bodies clamp their start tile to the
    last tile that holds keys, so they stop one tile past the padded length."""
    beyond = {}
    for case in _REACH_CASES:
        arch, body, shape = case.values
        spec = _reach_spec(arch, body, shape)
        bn = spec.block_n
        for n in _reach_points(bn):
            padded_tiles = -(-n // bn)
            for row in _furthest_kv_rows(arch, spec, n).values():
                tiles = row // bn + 1 - padded_tiles
                beyond[case.id] = max(beyond.get(case.id, tiles), tiles)
    deepest = max(beyond.values())
    assert deepest == DENSE_SLAB_SLACK_TILES, (
        f"the deepest K/V load is {deepest} tiles past the padded length; "
        f"DENSE_SLAB_SLACK_TILES is {DENSE_SLAB_SLACK_TILES}. Tiles past it per "
        f"body: {beyond}"
    )


def _first_kv_load(kernel, tensor):
    """``(region, op)`` of the first load from the ``tensor`` buffer resource."""
    for region in (kernel.body, *(r for op in _walk(kernel.body) for r in op.regions)):
        for op in region.ops:
            if _kv_load_tensor(op) == tensor:
                return region, op
    raise AssertionError(f"no load of {tensor}")


def _load_by_an_unknown_op(kernel):
    """A K load through one more LDS-DMA op, one the walk has no voffset for."""
    _, load = _first_kv_load(kernel, "%k_ptr")
    load.name = "tile.async_buffer_load_lds"


def _resource_through_another_op(kernel):
    """A K load through a copy of K's buffer resource."""
    region, load = _first_kv_load(kernel, "%k_ptr")
    rsrc = load.operands[0]
    copy = Value("%rsrc_copy", rsrc.type)
    copy.op = Op("tile.readfirstlane", operands=[rsrc], results=[copy])
    at = next(i for i, op in enumerate(region.ops) if op is load)
    region.ops.insert(at, copy.op)
    load.operands[0] = copy


def _load_from_the_slab_pointer(kernel):
    """A V load from V's slab pointer, past no buffer resource."""
    _, load = _first_kv_load(kernel, "%v_ptr")
    load.operands[0] = load.operands[0].op.operands[0]


def _load_inside_an_unmodelled_loop(kernel):
    """The KV loop is a loop op the walk has no trip count for."""
    loop = next(
        op
        for op in _walk(kernel.body)
        if op.name == "scf.for" and any(map(_kv_load_tensor, _walk(op.regions[0])))
    )
    loop.name = "scf.while"


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_load_by_an_unknown_op, "tile.async_buffer_load_lds takes the %k_ptr buffer"),
        (_resource_through_another_op, "tile.readfirstlane takes the %k_ptr buffer"),
        (_load_from_the_slab_pointer, "tile.buffer_load_vN takes the %v_ptr pointer"),
        (_load_inside_an_unmodelled_loop, "a K/V load inside scf.while"),
    ],
    ids=[
        "unknown-load-op",
        "resource-through-another-op",
        "load-from-the-slab-pointer",
        "unmodelled-loop",
    ],
)
def test_the_reach_walk_refuses_a_kv_access_it_cannot_measure(edit, match):
    """Edited into the gfx942 grid body (both kinds of K/V load), each K or V access
    the walk cannot measure fails the walk, rather than being left out of the
    furthest row: such a load could prefetch deeper than the host admits."""
    arch, spec = "gfx942", _reach_spec("gfx942", "causal", "bf16_d128_kv2")
    kernel = _MODULES[arch].build_attention_dense(spec, arch=arch)
    assert _kv_rows(arch, spec, kernel, spec.block_n), "the unedited body loads no K/V"
    edit(kernel)
    with pytest.raises(AssertionError, match=match):
        _kv_rows(arch, spec, kernel, spec.block_n)


# Strides for the slab-choice walk, as (batch, head, token) elements per tensor. No
# two are equal and a head stride times any head count stays below the batch stride,
# so a slab offset decodes to one (batch, head) pair, and an offset that steps by the
# wrong axis's stride, or another tensor's, does not decode to the right pair.
_SLAB_CHOICE_STRIDES = {
    "q": (1_000_000, 1_000, 7),
    "k": (2_000_000, 3_000, 11),
    "v": (3_000_000, 5_000, 13),
    "o": (4_000_000, 7_000, 17),
}


def _slab_choices(arch, spec):
    """``{work item: {tensor: [(batch, head) or the raw offset]}}``: the slab each
    ``tile.global_ptr_add`` of a tensor pointer moves to, for every thread of
    :func:`_workgroup_threads` and every iteration of the loops around it, at the
    strides of ``_SLAB_CHOICE_STRIDES``."""
    module = _MODULES[arch]
    params = {f"%{k}": v for k, v in module.attention_dense_runtime_args(spec).items()}
    for tensor, values in _SLAB_CHOICE_STRIDES.items():
        for axis, value in zip(("batch", "head", "token"), values):
            params[f"%stride_{tensor}_{axis}"] = value
    pointers = {f"%{tensor}_ptr": tensor for tensor in _SLAB_CHOICE_STRIDES}
    items = {}
    walk = _path_walk(
        lambda op: op.name == "tile.global_ptr_add"
        and op.operands[0].op is None
        and op.operands[0].name in pointers,
        "a slab pointer",
    )
    kernel = _build(arch, spec)
    for block, tid in _workgroup_threads(module, spec):

        def visit(op, thread, loops, at=(block, tid)):
            tensor = pointers[op.operands[0].name]
            batch_stride, head_stride, _ = _SLAB_CHOICE_STRIDES[tensor]
            offset = thread.value(op.operands[1])
            batch, rest = divmod(offset // DENSE_ELEM_BYTES, batch_stride)
            head, odd = divmod(rest, head_stride)
            exact = offset % DENSE_ELEM_BYTES == 0 and odd == 0
            slabs = items.setdefault((*at, loops), {})
            slabs.setdefault(tensor, []).append((batch, head) if exact else offset)

        walk(kernel.body, _Thread(params, block, tid), visit)
    return items


@pytest.mark.parametrize(("arch", "body", "shape"), _REACH_CASES)
def test_each_work_item_reads_its_own_slabs(arch, body, shape):
    """Each work item moves Q and O to the slab of its (batch, query head), and K
    and V to the slab of the same batch and that query head's KV head (query head //
    group size); the work items cover every (batch, query head). A KV-segment body
    may leave O alone: the combine kernel writes it. Batch 2 and grouped heads, so a
    slab found with its batch and head strides swapped, or for the wrong head of a
    group, decodes to another pair; persistent bodies run three workgroups, so a
    workgroup carries several work items."""
    kw = {**_SLAB_BODIES[(arch, body)]}
    if kw.get("persistent"):
        kw["num_persistent"] = 3
    spec = _spec(shape, varlen=False, arch=arch, **kw)
    assert _MODULES[arch].body_serves_strides(spec), "not a body that reads strides"
    group = spec.num_query_heads // spec.num_kv_heads
    assert spec.batch > 1 and group > 1, "a swap of batch and head must show"
    need = {"q", "k", "v"} | (set() if spec.kv_segments > 1 else {"o"})
    covered = set()
    items = _slab_choices(arch, spec)
    assert items, "the walk found no slab pointer"
    for item, slabs in items.items():
        assert need <= set(slabs), f"work item {item} moves only {sorted(slabs)}"
        assert len(set(slabs["q"])) == 1, f"work item {item} reads Q at {slabs['q']}"
        q = slabs["q"][0]
        assert isinstance(q, tuple), f"work item {item}: Q offset {q} is no slab"
        batch, head = q
        want = {
            "q": (batch, head),
            "o": (batch, head),
            "k": (batch, head // group),
            "v": (batch, head // group),
        }
        wrong = {t: got for t, got in slabs.items() if set(got) != {want[t]}}
        assert not wrong, (
            f"work item {item} reads Q of (batch, head) = {q}; want K and V at "
            f"{want['k']} and O at {want['o']}, but these move to (batch, head) or a "
            f"raw byte offset: {wrong}"
        )
        covered.add(q)
    every = {(b, h) for b in range(spec.batch) for h in range(spec.num_query_heads)}
    assert covered == every, f"no work item reads Q of {sorted(every - covered)}"


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("variant", list(_CONTROLS))
def test_nonvarlen_kv_bound_stays_the_spec_extent(variant, shape):
    """Control: a packed-addressing non-varlen bound stays batch * seqlen_kv rows of
    num_kv_heads * head_size * 2 bytes, read from the kernel parameters, with no
    global load.

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
            f"= {want}. The non-varlen packed branch must read num_kv_heads under "
            "runtime_shape (_kv_elem_bytes_ir)"
        )
