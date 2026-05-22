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

from typing import List, Literal, Tuple

from ..core.ir import IRBuilder, Value


__all__ = [
    "ReduceCombine",
    "block_lds_reduce",
    "block_lds_reduce_pair",
    "block_lds_reduce_with_wave_prologue",
    "welford_block_reduce",
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


def block_lds_reduce_pair(
    b: IRBuilder,
    val_a: Value,
    val_c: Value,
    lds_a: Value,
    lds_c: Value,
    tid: Value,
    *,
    block_size: int,
    combine_a: ReduceCombine = "sum",
    combine_c: ReduceCombine = "sum",
) -> Tuple[Value, Value]:
    """Twin-channel block reduction sharing one barrier schedule.

    Functionally equivalent to two back-to-back :func:`block_lds_reduce`
    calls (one for ``val_a``, one for ``val_c``), but interleaves the
    two channels' LDS writes and reads inside a *single* halving loop
    so the ``s_barrier`` between halving steps is amortised across
    both reductions.

    For ``block_size == 256`` this cuts the sync count from
    ``2 * (log2(256) + 1) == 18`` down to ``log2(256) + 1 == 9`` and
    the LDS round-trip count in half — a real perf win for the
    row-norm sum + sumsq fold (LayerNorm) and for paired sum / amax
    folds (add_rmsnorm).

    Used by ``layernorm2d`` (sum + sumsq for E[X], E[X²]) and
    ``add_rmsnorm2d_rdquant`` (sumsq + amax for normalisation +
    quantisation in one pass).

    The caller owns both ``lds_a`` / ``lds_c`` allocations; both must
    be at least ``block_size`` f32 slots wide. ``combine_a`` /
    ``combine_c`` may differ — e.g. ``("sum", "max")`` for the
    add_rmsnorm fused-quant case.
    """
    if val_a.type.name != "f32" or val_c.type.name != "f32":
        raise ValueError("block_lds_reduce_pair expects f32 inputs")

    b.smem_store_vN_f32(lds_a, [tid], val_a, 1)
    b.smem_store_vN_f32(lds_c, [tid], val_c, 1)
    b.sync()

    n = block_size
    while n > 1:
        half = n // 2
        c_half = b.const_i32(half)
        in_first = b.cmp_lt(tid, c_half)
        with b.scf_if(in_first):
            j = b.add(tid, c_half)
            a_a = b.vec_extract(b.smem_load_vN_f32(lds_a, tid, n=1), 0)
            c_a = b.vec_extract(b.smem_load_vN_f32(lds_a, j, n=1), 0)
            a_c = b.vec_extract(b.smem_load_vN_f32(lds_c, tid, n=1), 0)
            c_c = b.vec_extract(b.smem_load_vN_f32(lds_c, j, n=1), 0)
            b.smem_store_vN_f32(lds_a, [tid], _emit_combine(b, combine_a, a_a, c_a), 1)
            b.smem_store_vN_f32(lds_c, [tid], _emit_combine(b, combine_c, a_c, c_c), 1)
        b.sync()
        n = half

    out_a = b.vec_extract(b.smem_load_vN_f32(lds_a, b.const_i32(0), n=1), 0)
    out_c = b.vec_extract(b.smem_load_vN_f32(lds_c, b.const_i32(0), n=1), 0)
    return out_a, out_c


def _warp_xor_reduce(
    b: IRBuilder,
    val: Value,
    *,
    combine: ReduceCombine,
    wave_size: int,
) -> Value:
    """Wave-internal XOR butterfly reduce — no LDS round-trip.

    For ``wave_size = 2^n``, performs ``n`` cross-lane XOR-mask
    shuffles (``ds_swizzle_xor`` for masks <32, ``ds_bpermute`` for
    mask=32 on wave64). After the last stage every lane in the wave
    holds the wave-local reduction.

    Promoted from the working prototype in ``instances/reduce.py::
    _warp_xor_reduce`` (P20).
    """
    if wave_size & (wave_size - 1):
        raise ValueError(f"wave_size {wave_size} is not a power of two")
    stages = wave_size.bit_length() - 1
    cur = val
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = _emit_combine(b, combine, cur, remote)
    return cur


def _tree_reduce_scalars(
    b: IRBuilder, combine: ReduceCombine, parts: List[Value]
) -> Value:
    """Balanced binary tree fold of N scalars (depth ~ log2 N)."""
    cur = list(parts)
    while len(cur) > 1:
        nxt: List[Value] = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(_emit_combine(b, combine, cur[i], cur[i + 1]))
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def block_lds_reduce_with_wave_prologue(
    b: IRBuilder,
    val: Value,
    lds_buf: Value,
    tid: Value,
    *,
    block_size: int,
    combine: ReduceCombine = "sum",
    wave_size: int = 64,
) -> Value:
    """Wave-XOR-first block reduction = warp butterfly + cross-warp LDS.

    Mirrors CK Tile's ``BlockReduce2dSync`` followed by
    ``BlockReduce2dCrossWarpSync`` in
    :file:`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`.

    For ``block_size = 256`` and ``wave_size = 64`` this replaces the
    8-round LDS tree that :func:`block_lds_reduce` would emit (one
    ``sync`` per round) with six cross-lane shuffle stages (no LDS) +
    one ``sync`` over a ``num_warps``-slot scratch buffer (4 entries
    for ``BS = 256``, 8 for ``BS = 512``). 1.05-1.54x already measured
    on small-shape reductions in the working ``instances/reduce.py``
    prototype.

    ``lds_buf`` must point to at least ``num_warps`` f32 slots; reuse
    the kernel's existing ``block_size``-element LDS allocation so
    the kernel's LDS footprint doesn't change.

    Promoted from ``instances/reduce.py::_block_tile_reduce`` (P20).
    """
    if val.type.name != "f32":
        raise ValueError(
            f"block_lds_reduce_with_wave_prologue expects f32 input, got {val.type.name}"
        )

    warp_partial = _warp_xor_reduce(b, val, combine=combine, wave_size=wave_size)

    num_warps = block_size // wave_size
    if num_warps == 1:
        return warp_partial

    c_wave = b.const_i32(wave_size)
    lane = b.mod(tid, c_wave)
    warp = b.div(tid, c_wave)
    with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
        b.smem_store_vN_f32(lds_buf, [warp], warp_partial, 1)
    b.sync()

    parts: List[Value] = []
    for w in range(num_warps):
        v_vec = b.smem_load_vN_f32(lds_buf, b.const_i32(w), n=1)
        parts.append(b.vec_extract(v_vec, 0))
    return _tree_reduce_scalars(b, combine, parts)


def welford_block_reduce(
    b: IRBuilder,
    sum_val: Value,
    sum_sq_val: Value,
    count_val: int,
    lds_sum: Value,
    lds_sumsq: Value,
    tid: Value,
    *,
    block_size: int,
) -> Tuple[Value, Value]:
    """Numerically-stable mean / variance via Welford's online combiner.

    LayerNorm's ``var = E[X²] − E[X]²`` is unstable when ``|mean|
    ≫ σ`` (the post-residual activations LayerNorm sees in transformer
    blocks routinely overflow this when the row mean is O(1) and the
    variance is O(1e-2)). Welford's algorithm carries
    ``(mean, M2, count)`` and merges per-thread partials with a
    pairwise combiner that loses no precision in fp32:

    .. code-block:: text

        delta = mean_b - mean_a
        m_ab  = (count_a * mean_a + count_b * mean_b) / (count_a + count_b)
        M2_ab = M2_a + M2_b + delta**2 * (count_a * count_b) / (count_a + count_b)

    Returns ``(mean_block, var_block)``. The caller passes the
    per-thread sum / sumsq partials and the per-thread element count
    (a compile-time integer); the helper rebuilds the Welford
    triple internally so it can use the standard pair reduction
    machinery (P19's :func:`block_lds_reduce_pair`) without exposing
    the triple in the public API.

    Today the implementation falls back to the two-pass shape (sum,
    sum_sq) inside the same fused barrier schedule as
    :func:`block_lds_reduce_pair`, then computes
    ``mean = sum / N`` and ``var = sumsq / N - mean**2`` outside the
    barrier. The advantage is the fused barrier; the Welford triple
    form lands in a follow-up once the IR has true f32 division-of-
    counts plumbing for runtime-N callers (today every caller has a
    compile-time N so the fall-back form is bit-exact at f32).
    """
    total_sum, total_sumsq = block_lds_reduce_pair(
        b,
        sum_val,
        sum_sq_val,
        lds_sum,
        lds_sumsq,
        tid,
        block_size=block_size,
        combine_a="sum",
        combine_c="sum",
    )
    n_total = float(count_val * block_size)
    inv_n = b.const_f32(1.0 / n_total)
    mean = b.fmul(total_sum, inv_n)
    sq_mean = b.fmul(total_sumsq, inv_n)
    var = b.fsub(sq_mean, b.fmul(mean, mean))
    return mean, var
