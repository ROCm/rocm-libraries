# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Block-level reductions, lifted from the inline copies in norm/reduce kernels.

CK Tile exposes ``BlockReduce2dDefaultPolicy`` / ``block_tile_reduce_xor_sync``
in :file:`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`. The DSL
counterpart here is a thin LDS tree reduction over a single f32
broadcast value: each thread writes its partial value to a shared
:class:`ck_dsl.helpers.tensor_view.TensorView` in LDS, the reduction
halves the active lane set on each step, and the final value at index
0 is broadcast back to every lane.

The combine semantics are parameterised; today we support ``"sum"``
(LayerNorm/RMSNorm/Reduce-sum/Reduce-mean) and ``"max"`` (Reduce-max,
attention online softmax). The wave-butterfly form via ``ds_bpermute``
that the attention kernels use is a *different* algorithm (wave-only,
no LDS round-trip) and intentionally lives in
:mod:`ck_dsl.instances.attention_tiled_2d` next to the softmax it
serves.

Why a separate module: every call site that needed a block reduction
copied a 15-30 line ``_block_reduce_sum`` from
:mod:`ck_dsl.instances.layernorm2d`. We now have one canonical
implementation that the norm / reduce / pooling kernels share.
"""

from __future__ import annotations

from typing import Literal

from ..core.ir import IRBuilder, Value


__all__ = [
    "ReduceCombine",
    "block_lds_reduce",
]


ReduceCombine = Literal["sum", "max", "min", "prod"]


def _emit_combine(b: IRBuilder, combine: ReduceCombine, a: Value, c: Value) -> Value:
    """Apply the reduction combiner to two f32 partials."""
    if combine == "sum":
        return b.fadd(a, c)
    if combine == "max":
        return b.fmax(a, c)
    if combine == "min":
        return b.fmin(a, c)
    if combine == "prod":
        return b.fmul(a, c)
    raise ValueError(f"unknown combine {combine!r}")


def block_lds_reduce(
    b: IRBuilder,
    val: Value,
    lds_buf: Value,
    tid: Value,
    *,
    block_size: int,
    combine: ReduceCombine = "sum",
) -> Value:
    """LDS tree reduction across all ``block_size`` lanes.

    ``val`` is the per-thread partial; ``lds_buf`` is a
    ``block_size`` x f32 LDS allocation owned by the caller. The
    reduced value is broadcast back to every lane (i.e. the return
    value is the same across all threads in the workgroup).

    Supported combiners: ``sum`` (LayerNorm / RMSNorm / Reduce-sum /
    Reduce-mean), ``max`` (Reduce-max, attention online softmax),
    ``min`` (Reduce-min), ``prod`` (Reduce-prod). The combiner is
    applied in f32 regardless of the storage dtype the caller is
    accumulating from.

    The barrier between halving steps is :func:`IRBuilder.sync`, which
    now correctly emits an ``s_waitcnt lgkmcnt(0) vmcnt(0)`` before
    ``s_barrier`` (see ``_op_tile_sync`` in ``core/lower_llvm.py``).
    """
    if combine not in ("sum", "max", "min", "prod"):
        raise ValueError(
            f"unknown combine {combine!r}; expected one of {{'sum','max','min','prod'}}"
        )
    if val.type.name != "f32":
        raise ValueError(f"block_lds_reduce expects f32 input, got {val.type.name}")

    b.smem_store_vN_f32(lds_buf, [tid], val, 1)
    b.sync()

    n = block_size
    while n > 1:
        half = n // 2
        c_half = b.const_i32(half)
        in_first = b.cmp_lt(tid, c_half)
        with b.scf_if(in_first):
            j = b.add(tid, c_half)
            a_vec = b.smem_load_vN_f32(lds_buf, tid, n=1)
            c_vec = b.smem_load_vN_f32(lds_buf, j, n=1)
            a = b.vec_extract(a_vec, 0)
            c = b.vec_extract(c_vec, 0)
            combined = _emit_combine(b, combine, a, c)
            b.smem_store_vN_f32(lds_buf, [tid], combined, 1)
        b.sync()
        n = half

    out = b.smem_load_vN_f32(lds_buf, b.const_i32(0), n=1)
    return b.vec_extract(out, 0)
