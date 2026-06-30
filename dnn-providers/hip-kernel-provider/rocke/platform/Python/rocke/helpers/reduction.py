# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Block-level reductions, lifted from the inline copies in norm/reduce kernels.

CK Tile exposes ``BlockReduce2dDefaultPolicy`` / ``block_tile_reduce_xor_sync``
in :file:`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`. The DSL
counterpart here is a thin LDS tree reduction over a single f32
broadcast value: each thread writes its partial value to a shared
:class:`rocke.helpers.tensor_view.TensorView` in LDS, the reduction
halves the active lane set on each step, and the final value at index
0 is broadcast back to every lane.

The combine semantics are parameterised; today we support ``"sum"``
(LayerNorm/RMSNorm/Reduce-sum/Reduce-mean) and ``"max"`` (Reduce-max,
attention online softmax). The wave-butterfly form via ``ds_bpermute``
that the attention kernels use is a *different* algorithm (wave-only,
no LDS round-trip) and intentionally lives in
:mod:`rocke.instances.gfx950.attention_tiled_2d` next to the softmax it
serves.

Why a separate module: every call site that needed a block reduction
copied a 15-30 line ``_block_reduce_sum`` from
:mod:`rocke.instances.common.layernorm2d`. We now have one canonical
implementation that the norm / reduce / pooling kernels share.
"""

from __future__ import annotations

from typing import Callable, List, Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, Value


__all__ = [
    "ReduceCombine",
    "IndexCombine",
    "REGISTER_TILE_MAX_ELEMS_PER_THREAD",
    "row_norm_needs_two_pass",
    "block_lds_reduce",
    "block_lds_reduce_pair",
    "block_lds_reduce_with_index",
    "block_lds_reduce_with_wave_prologue",
    "tree_reduce",
    "welford_block_reduce",
    "welford_block_reduce_stable",
]


ReduceCombine = Literal["sum", "max", "min", "prod"]
IndexCombine = Literal["argmax", "argmin"]


# Per-thread register-tile capacity for the row-norm family. The
# single-pass norm kernels cache the whole row in the per-thread f32
# register file so pass 2 (normalise / scale) never re-reads X from HBM;
# that cache costs ``elems_per_thread`` VGPRs per lane, so it is bounded.
# Legacy CK makes the same call through its ``isSweepOnce`` selector
# (:file:`include/ck/tensor_operation/gpu/grid/normalization/
# gridwise_normalization_selector.hpp:78,181-226`): a row whose K fits in
# registers takes the one-pass ``SweepOnce`` kernel, anything wider takes
# the streaming kernel that re-reads X for pass 2
# (``gridwise_normalization_welford_variance.hpp:311`` vs ``:427``).
#
# 64 mirrors the ``max_elems_per_thread=64`` cap the row-norm
# ``is_valid_spec`` predicates already enforce for the cached path.
REGISTER_TILE_MAX_ELEMS_PER_THREAD = 64


def row_norm_needs_two_pass(
    elems_per_thread: int,
    *,
    max_cached: int = REGISTER_TILE_MAX_ELEMS_PER_THREAD,
) -> bool:
    """Select the streaming two-pass row-norm path at BUILD time.

    Returns ``True`` when ``elems_per_thread`` exceeds the per-thread
    register-tile capacity ``max_cached`` (so caching the whole row in
    VGPRs would overflow the budget) and the kernel must re-stream X from
    HBM in pass 2 instead — the DSL analogue of legacy CK's
    ``isSweepOnce`` selection. Returns ``False`` for rows that fit, which
    keeps the cached single-pass path (byte-identical IR) as the default
    for every in-budget config.

    Shared by :mod:`rocke.instances.common.layernorm2d` and
    :mod:`rocke.instances.common.rmsnorm2d` so the single-vs-two-pass
    cutover is defined in exactly one place.
    """
    return elems_per_thread > max_cached


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


def tree_reduce(
    b: IRBuilder,
    combine: Callable[[Value, Value], Value],
    xs: List[Value],
) -> Value:
    """Balanced binary-tree fold of N scalars (depth ~ log2 N).

    ``combine`` is the binary combiner emitted at each tree node — e.g.
    ``b.fadd`` for a sum fold, ``b.fmax`` for an abs-max fold. Pairs
    ``xs[i]`` with ``xs[i+1]`` left-to-right and carries any odd tail
    element forward unchanged, so for power-of-two ``len(xs)`` the tree
    is fully balanced with no carried tail.

    Emitting the tree explicitly drops the per-thread critical-path
    depth from ``len(xs) - 1`` (the natural left-fold chain) to
    ``ceil(log2(len(xs)))``: the DSL never sets the LLVM ``reassoc``
    fastmath flag on ``arith.fadd`` (see ``core/lower_llvm.py``), so the
    optimiser cannot re-shape the chain on our behalf. This is the
    canonical home for the per-chunk folds the norm / reduce kernels
    copied inline (``_balanced_combine`` / ``_tree_reduce``).

    The ``b`` argument is accepted for API symmetry with the other
    reduction helpers; the combiner already closes over the builder.
    """
    if not xs:
        raise ValueError("tree_reduce requires at least one value")
    cur = list(xs)
    while len(cur) > 1:
        nxt: List[Value] = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(combine(cur[i], cur[i + 1]))
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def _tree_reduce_scalars(
    b: IRBuilder, combine: ReduceCombine, parts: List[Value]
) -> Value:
    """Balanced binary tree fold of N scalars (depth ~ log2 N)."""
    return tree_reduce(b, lambda a, c: _emit_combine(b, combine, a, c), parts)


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


def welford_block_reduce_stable(
    b: IRBuilder,
    mean_val: Value,
    m2_val: Value,
    count_val: Value,
    lds_mean: Value,
    lds_m2: Value,
    lds_count: Value,
    tid: Value,
    *,
    block_size: int,
) -> Tuple[Value, Value]:
    """Numerically-stable Welford block reduction over the full triple.

    This is the count-weighted ``(mean, M2, count)`` parallel merge from
    legacy CK's ``BlockwiseWelford::Merge``
    (:file:`include/ck/tensor_operation/gpu/block/blockwise_welford.hpp`
    lines 40-48) and ``ThreadwiseWelfordMerge::Merge``
    (:file:`include/ck/tensor_operation/gpu/thread/threadwise_welford.hpp`
    lines 90-100). Unlike :func:`welford_block_reduce` (which falls back
    to the unstable ``var = E[X²] − E[X]²`` two-pass shape), this carries
    the real Welford triple so there is no catastrophic cancellation when
    ``|mean| ≫ σ`` — the post-residual activations LayerNorm sees in
    transformer blocks.

    Each thread supplies its *own partial* Welford triple:

    * ``mean_val``  — f32 mean of this thread's elements,
    * ``m2_val``    — f32 sum-of-squared-deviations (M2 = Σ(x−mean)²),
    * ``count_val`` — f32 number of elements this thread accumulated
      (passed as an f32 :class:`Value`; partial threads may carry a
      different count, exactly as the reference's per-lane ``count``).

    The merge of two partials ``a``/``b`` (CK ``Merge``) is::

        count            = count_a + count_b
        count_b_over_cnt = count_b / count            (0 when count == 0)
        delta            = mean_b - mean_a
        mean_a          += delta * count_b_over_cnt
        M2_a            += M2_b + delta*delta * count_a * count_b_over_cnt
        count_a          = count

    Because this combiner is *not* a plain associative add, it cannot
    ride the generic ``combine=`` path of :func:`block_lds_reduce`; it
    needs a dedicated three-channel (mean / M2 / count) LDS tree, which
    is what this helper emits. The tree mirrors the halving schedule of
    :func:`block_lds_reduce` (``block_size`` → 1) so the barrier count is
    identical to the value-only reduction.

    Returns ``(mean_block, var_block)`` broadcast to every lane, where
    ``var_block = M2_total / count_total`` — the ``GetActualVariance``
    divide at ``blockwise_welford.hpp:104-107``. The caller can use a
    biased (population) variance directly or rescale to the unbiased
    form; the reference returns the biased ``M2 / count``.

    The caller owns ``lds_mean`` / ``lds_m2`` / ``lds_count`` — three
    ``block_size``-wide f32 LDS allocations (``count`` is stored in f32,
    matching the reference's ``T count_b_over_count`` f32 divide).
    """
    if mean_val.type.name != "f32":
        raise ValueError("welford_block_reduce_stable expects f32 mean_val")
    if m2_val.type.name != "f32":
        raise ValueError("welford_block_reduce_stable expects f32 m2_val")
    if count_val.type.name != "f32":
        raise ValueError("welford_block_reduce_stable expects f32 count_val")

    b.smem_store_vN_f32(lds_mean, [tid], mean_val, 1)
    b.smem_store_vN_f32(lds_m2, [tid], m2_val, 1)
    b.smem_store_vN_f32(lds_count, [tid], count_val, 1)
    b.sync()

    zero = b.const_f32(0.0)

    n = block_size
    while n > 1:
        half = n // 2
        c_half = b.const_i32(half)
        in_first = b.cmp_lt(tid, c_half)
        with b.scf_if(in_first):
            j = b.add(tid, c_half)
            mean_a = b.vec_extract(b.smem_load_vN_f32(lds_mean, tid, n=1), 0)
            m2_a = b.vec_extract(b.smem_load_vN_f32(lds_m2, tid, n=1), 0)
            cnt_a = b.vec_extract(b.smem_load_vN_f32(lds_count, tid, n=1), 0)
            mean_b = b.vec_extract(b.smem_load_vN_f32(lds_mean, j, n=1), 0)
            m2_b = b.vec_extract(b.smem_load_vN_f32(lds_m2, j, n=1), 0)
            cnt_b = b.vec_extract(b.smem_load_vN_f32(lds_count, j, n=1), 0)

            # count = count_a + count_b
            count = b.fadd(cnt_a, cnt_b)
            # count_b_over_count = count == 0 ? 0 : count_b / count
            is_empty = b.fcmp("oeq", count, zero)
            ratio = b.fmul(cnt_b, b.rcp(count))
            count_b_over_count = b.select(is_empty, zero, ratio)
            # delta = mean_b - mean_a
            delta = b.fsub(mean_b, mean_a)
            # mean_a += delta * count_b_over_count
            new_mean = b.fadd(mean_a, b.fmul(delta, count_b_over_count))
            # M2_a += M2_b + delta*delta * count_a * count_b_over_count
            dd = b.fmul(delta, delta)
            cross = b.fmul(b.fmul(dd, cnt_a), count_b_over_count)
            new_m2 = b.fadd(m2_a, b.fadd(m2_b, cross))

            b.smem_store_vN_f32(lds_mean, [tid], new_mean, 1)
            b.smem_store_vN_f32(lds_m2, [tid], new_m2, 1)
            b.smem_store_vN_f32(lds_count, [tid], count, 1)
        b.sync()
        n = half

    mean_out = b.vec_extract(b.smem_load_vN_f32(lds_mean, b.const_i32(0), n=1), 0)
    m2_out = b.vec_extract(b.smem_load_vN_f32(lds_m2, b.const_i32(0), n=1), 0)
    count_out = b.vec_extract(b.smem_load_vN_f32(lds_count, b.const_i32(0), n=1), 0)
    var_out = b.fmul(m2_out, b.rcp(count_out))
    return mean_out, var_out


def block_lds_reduce_with_index(
    b: IRBuilder,
    val: Value,
    idx: Value,
    lds_val: Value,
    lds_idx: Value,
    tid: Value,
    *,
    block_size: int,
    combine: IndexCombine = "argmax",
) -> Tuple[Value, Value]:
    """LDS tree reduction carrying both value and index (argmax / argmin).

    Mirrors legacy CK's
    ``PartitionedBlockwiseReductionWithIndex::Reduce``
    (:file:`include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp`
    lines 164-242) with the ``AccumulateWithIndexAndNanCheck<false, …>``
    combiner
    (:file:`include/ck/utility/reduction_functions_accumulate.hpp`
    lines 67-84): the index is overwritten **only on a strict
    improvement** (the ``changed`` flag of the indexable Max/Min
    operator, ``reduction_operator.hpp:233-245``). That makes the
    tie-break deterministic — on equal values the *lower* index wins,
    because the accumulator is always the lower-offset slot and the
    higher-offset value only displaces it when strictly better.

    Each thread supplies its per-thread partial ``val`` (f32) and the
    candidate ``idx`` (i32, e.g. the column the partial came from). The
    reduced ``(value, index)`` pair is broadcast back to every lane.

    The twin LDS transfer stores the i32 index into an f32-typed LDS
    buffer via :meth:`IRBuilder.bitcast` (i32 → f32 is a no-op
    reinterpret of the 32 bits), matching the reference's parallel
    ``work_val_buffer`` / ``work_idx_buffer`` pair. The caller owns both
    ``lds_val`` / ``lds_idx`` (``block_size``-wide f32 LDS each).

    NaN policy is the non-propagating ``false`` arm (no NaN check); a NaN
    candidate never sets ``changed`` so it is effectively ignored, which
    is the legacy default for the no-NaN-check selector.
    """
    if combine not in ("argmax", "argmin"):
        raise ValueError(
            f"unknown index combine {combine!r}; expected 'argmax' or 'argmin'"
        )
    if val.type.name != "f32":
        raise ValueError("block_lds_reduce_with_index expects f32 val")
    if idx.type.name != "i32":
        raise ValueError("block_lds_reduce_with_index expects i32 idx")

    if block_size & (block_size - 1):
        raise ValueError(
            f"block_lds_reduce_with_index needs power-of-two block_size, got {block_size}"
        )

    b.smem_store_vN_f32(lds_val, [tid], val, 1)
    b.smem_store_vN_f32(lds_idx, [tid], b.bitcast(idx, F32), 1)
    b.sync()

    # CK doubling tree (reduction_functions_blockwise.hpp:215-235):
    # ``indOffset = 1 << I`` and only lanes with
    # ``tid % (indOffset*2) == 0`` merge slot ``tid`` (accumulator, the
    # lower offset) with slot ``tid + indOffset`` (candidate). This is
    # deliberately NOT the power-of-two halving the value-only tree uses
    # (block_lds_reduce): the accumulator always sits at the lower
    # original index, so the strict-improvement tie-break resolves ties
    # to the LOWEST index, matching numpy.argmax / numpy.argmin. A naive
    # halving tree gives a non-deterministic argmax under ties.
    cluster_len_shift = block_size.bit_length() - 1
    for shift in range(cluster_len_shift):
        ind_offset = 1 << shift
        c_off = b.const_i32(ind_offset)
        c_mod = b.const_i32(ind_offset * 2)
        participates = b.cmp_eq(b.mod(tid, c_mod), b.const_i32(0))
        with b.scf_if(participates):
            j = b.add(tid, c_off)
            v_a = b.vec_extract(b.smem_load_vN_f32(lds_val, tid, n=1), 0)
            v_b = b.vec_extract(b.smem_load_vN_f32(lds_val, j, n=1), 0)
            i_a = b.bitcast(
                b.vec_extract(b.smem_load_vN_f32(lds_idx, tid, n=1), 0), I32
            )
            i_b = b.bitcast(b.vec_extract(b.smem_load_vN_f32(lds_idx, j, n=1), 0), I32)

            # strict-improvement test (the `changed` flag of the
            # indexable Max/Min operator, reduction_operator.hpp:233-245).
            # argmax: changed when a < b; argmin: changed when a > b.
            # Ties never change -> the lower-offset accumulator (lower
            # original index) is retained.
            if combine == "argmax":
                changed = b.fcmp("olt", v_a, v_b)
            else:
                changed = b.fcmp("ogt", v_a, v_b)

            new_val = b.select(changed, v_b, v_a)
            new_idx = b.select(changed, i_b, i_a)

            b.smem_store_vN_f32(lds_val, [tid], new_val, 1)
            b.smem_store_vN_f32(lds_idx, [tid], b.bitcast(new_idx, F32), 1)
        b.sync()

    out_val = b.vec_extract(b.smem_load_vN_f32(lds_val, b.const_i32(0), n=1), 0)
    out_idx = b.bitcast(
        b.vec_extract(b.smem_load_vN_f32(lds_idx, b.const_i32(0), n=1), 0), I32
    )
    return out_val, out_idx
