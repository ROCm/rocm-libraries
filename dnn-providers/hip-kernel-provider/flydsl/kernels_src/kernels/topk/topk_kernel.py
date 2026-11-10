# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Written for the hip-kernel-provider against flydsl 0.3.4; not a vendored file.
#
# Derived in part from AITER's FlyDSL per-row top-k kernels:
#   upstream: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   paths:    aiter/ops/flydsl/kernels/topk/topk_per_row_argmax.py
#             aiter/ops/flydsl/kernels/kernels_common.py (ord_signed_f32)
#   taken:    ordering scores by an int32 key instead of by float compares (AITER's
#             ord_signed_f32: -0.0 folded onto +0.0, every NaN above +inf; the key here
#             is a sign-magnitude form whose range excludes INT32_MIN); the per-row
#             buffer descriptor built over the sliced row, so the 32-bit `num_records`
#             never wraps on a large tensor; k = 1 as a (key, column) reduction rather
#             than a selection.
#   changed:  - one workgroup per row for every k (AITER's argmax splits a row across
#               workgroups and folds the partials in a second launch);
#             - bf16 / f16 / f32 input, widened to f32 in-kernel, and the values output
#               copied from the input element bit for bit;
#             - `largest` as a runtime argument (smallest = the bitwise complement of
#               the key, which also moves NaN last);
#             - int32 or int64 indices, and an optional values output;
#             - general k by an exact radix select (AITER's small-k selector breaks
#               ties by lane, not by column, and is not used): ties go to the lower
#               column, as torch.topk(sorted=True, stable) and the hipDNN CPU
#               reference require;
#             - wave reductions by shuffle instead of AITER's LDS tree.
#
# Upstream-first: nothing here is AITER code verbatim; tools/diff_upstream.py does not
# track this file.

"""Per-row top-k along the last dim, sorted best-first.

Semantics (hipDNN `TopkAttributes`): for every row, the positions of the k largest
(`largest != 0`) or smallest entries, best first; equal values keep the lower position
first; NaN compares greater than every number (first for largest, last for smallest; NaNs
keep position order among themselves); -0.0 equals +0.0. Values, when written, are copied
from the input without conversion.

Every element maps to a 32-bit key whose signed order is the requested order: the input
widened to f32 (exact for bf16 and f16), -0.0 and +0.0 one key, every NaN one key above
+inf, then complemented when selecting the smallest. Keys tie exactly when the CPU
reference's comparison ties, so `(key desc, column asc)` is the reference's order.

Two bodies, chosen at build time by `k_max`:

- `k_max == 1`: a (key, column) max-reduction. One pass over the row.
- `k_max > 1`: an exact radix select. Four 8-bit histogram passes over the keys find the
  k-th best key T, how many keys beat it (G), and how many equal it (E). A collect pass
  takes every key above T, and every key equal to T when E == k - G; otherwise a
  column-ordered pass takes the first k - G keys equal to T. The k survivors are then
  ranked by `(key desc, column asc)` and the rank is the output slot. A row of up to
  CACHE_KEYS columns is read once and its keys kept in LDS; a longer one is re-read by
  every pass.

Row width, row count, k and `largest` are runtime. The row is read with 128-bit loads when
`VecOk` says the row starts are 16-byte aligned (the host checks the base pointer and the
row stride), with a scalar tail; otherwise element by element.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import (
    Float32,
    Int32,
    Int64,
    as_ir_value,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T
from kernels.common.kernels_common import dtype_to_elem_type, get_warp_size

KERNEL_NAME = "topk"

# The general body's histogram has one bin per thread of the block.
RADIX_BITS = 8
RADIX = 1 << RADIX_BITS
GENERAL_BLOCK_THREADS = RADIX
# The largest k a general object serves: its survivor buffer, two int32 per survivor.
GENERAL_K_MAX = 1024
# 128-bit loads a thread issues before it uses any of them. Measured on gfx90a over one
# row of 248320 bf16, 2 to 16 run within 5% of each other: one workgroup per row is bound
# by its CU's VALU (the per-element key), not by the loads.
LOADS_IN_FLIGHT = 4
# The longest row whose keys the general body keeps in LDS between passes (8 KiB).
CACHE_KEYS = 2048

_INT32_MIN = -2147483648
_F32_ABS_MASK = 0x7FFFFFFF
# Every NaN's magnitude is clamped to this, one above +inf's.
_NAN_MAGNITUDE = 0x7F800001

# Scratch slots shared by the block.
_ST_PREFIX = 0  # high bits of T fixed so far (unsigned key bits)
_ST_NEED = 1  # how many keys equal to the prefix are still to be taken
_ST_EQ = 2  # after the last pass: E, the count of keys equal to T
_ST_COUNT = 3  # survivors appended so far
_ST_RUN = 4  # column-ordered pass: keys equal to T seen in earlier tiles
_ST_SLOTS = 8

_INDEX_TYPES = {"i32": Int32, "i64": Int64}


def _lds_atomic_add(view, offset, value):
    """`view[offset] += value` in LDS, returning the previous value."""
    ptr = fx.to_llvm_ptr(fx.get_iter(view) + offset)
    old = _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.add,
        ptr,
        as_ir_value(Int32(value) if isinstance(value, int) else value),
        _llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    ).result
    return Int32(old)


def _ord_key(value_f32):
    """Signed int32 key in the order of the f32 value: -0.0 == +0.0, NaN above +inf.

    The key is the magnitude, negated for a negative value, with every NaN's magnitude
    clamped to one above +inf's and its sign dropped. So -0.0 and +0.0 are one key, all
    NaNs are one key, and keys span [-0x7F800000, 0x7F800001]: neither the key nor its
    complement (the smallest-first key) is ever INT32_MIN, which the k = 1 body uses as
    its empty value.
    """
    bits = value_f32.bitcast(Int32)
    magnitude = fx.min(bits & Int32(_F32_ABS_MASK), Int32(_NAN_MAGNITUDE))
    sign = (magnitude == Int32(_NAN_MAGNITUDE)).select(Int32(0), bits >> Int32(31))
    return (magnitude ^ sign) - sign


def build_topk_module(
    dtype_str: str, index_dtype_str: str, k_max: int, block_threads: int
):
    """Build the per-row top-k launcher for one (dtype, index dtype, k bound, block).

    `k_max == 1` builds the reduction body; any larger bound builds the radix-select body,
    which serves every k in [1, k_max] and needs `block_threads == RADIX`.
    """
    if index_dtype_str not in _INDEX_TYPES:
        raise ValueError(
            f"index dtype must be one of {sorted(_INDEX_TYPES)}, got {index_dtype_str!r}"
        )
    warp = get_warp_size()
    if warp != 64:
        raise ValueError(f"topk kernels are built for wave64 targets, got wave{warp}")
    if block_threads % warp or block_threads & (block_threads - 1):
        raise ValueError(
            f"block must be a power-of-two number of waves, got {block_threads}"
        )
    if k_max < 1 or k_max > GENERAL_K_MAX:
        raise ValueError(f"k_max must be in [1, {GENERAL_K_MAX}], got {k_max}")
    reduce_only = k_max == 1
    if not reduce_only and block_threads != RADIX:
        raise ValueError(
            f"the radix-select body needs {RADIX} threads, got {block_threads}"
        )

    elem_bits = 32 if dtype_str == "f32" else 16
    vec = 128 // elem_bits
    index_type = _INDEX_TYPES[index_dtype_str]
    index_bits = 64 if index_dtype_str == "i64" else 32
    n_waves = block_threads // warp

    if reduce_only:

        @fx.struct
        class SharedStorage:
            red_key: fx.Array[Int32, n_waves, 16]
            red_col: fx.Array[Int32, n_waves, 16]

    else:

        @fx.struct
        class SharedStorage:
            hist: fx.Array[Int32, RADIX, 16]
            wave_tot: fx.Array[Int32, n_waves, 16]
            state: fx.Array[Int32, _ST_SLOTS, 16]
            surv_key: fx.Array[Int32, k_max, 16]
            surv_col: fx.Array[Int32, k_max, 16]
            keys: fx.Array[Int32, CACHE_KEYS, 16]

    def _scalar_atom(bits):
        return {
            16: fx.rocdl.BufferCopy16b,
            32: fx.rocdl.BufferCopy32b,
            64: fx.rocdl.BufferCopy64b,
        }[bits]()

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def topk_kernel(
        X: fx.Tensor,
        Indices: fx.Tensor,
        Values: fx.Tensor,
        K: fx.Int32,
        Largest: fx.Int32,
        WriteValues: fx.Int32,
        VecOk: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % Int32(warp)
        wave = tid // Int32(warp)
        zero = Int32(0)
        one = Int32(1)

        elem_dtype = dtype_to_elem_type(dtype_str)
        n = X.shape.to_py_value()[1]
        # Smallest-first is the largest-first order of the complemented key.
        flip = (Largest != zero).select(zero, Int32(-1))

        # Descriptors over this row alone (see the header).
        x_row = fx.rocdl.make_buffer_tensor(fx.slice(X, (row, None)), max_size=False)
        x_vec = fx.logical_divide(x_row, fx.make_layout(vec, 1))
        x_one = fx.logical_divide(x_row, fx.make_layout(1, 1))
        i_row = fx.rocdl.make_buffer_tensor(
            fx.slice(Indices, (row, None)), max_size=False
        )
        i_one = fx.logical_divide(i_row, fx.make_layout(1, 1))
        v_row = fx.rocdl.make_buffer_tensor(
            fx.slice(Values, (row, None)), max_size=False
        )
        v_one = fx.logical_divide(v_row, fx.make_layout(1, 1))

        vec_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        elem_atom = fx.make_copy_atom(_scalar_atom(elem_bits), elem_bits)
        index_atom = fx.make_copy_atom(_scalar_atom(index_bits), index_bits)

        # Whole vectors when the row starts are aligned, the rest element by element.
        n_vec = (VecOk != zero).select(n // Int32(vec), zero)
        tail_start = n_vec * Int32(vec)

        def key_of(value):
            widened = value if const_expr(dtype_str == "f32") else value.to(Float32)
            return _ord_key(widened) ^ flip

        def load_vec(v):
            frag = fx.make_rmem_tensor(vec, elem_dtype)
            fx.copy(vec_atom, fx.slice(x_vec, (None, v)), frag)
            return fx.memref_load_vec(frag)

        def load_one(col):
            frag = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy(elem_atom, fx.slice(x_one, (None, col)), frag)
            return fx.memref_load_vec(frag)[0]

        def load_group(v0):
            # Every load of a group is issued before any is used. A vector past the row
            # reads zeros (the descriptor is bounded by this row) and the caller drops it.
            group = []
            for u in range_constexpr(LOADS_IN_FLIGHT):
                group.append(load_vec(v0 + Int32(u * block_threads)))
            return group

        def store_one(atom, dtype, divided, col, value):
            frag = fx.make_rmem_tensor(1, dtype)
            fx.memref_store_vec(fx.Vector.filled(1, value, dtype), frag)
            fx.copy(atom, frag, fx.slice(divided, (None, col)))

        def write_result(place, col):
            store_one(index_atom, index_type, i_one, place, index_type(col))
            if WriteValues != zero:
                store_one(elem_atom, elem_dtype, v_one, place, load_one(col))

        if const_expr(reduce_only):
            # ---- k = 1: the best (key, column) of the row. ----------------------
            # A thread's columns ascend, so a strict `>` keeps the earliest of a tie.
            # INT32_MIN is the empty value: no element's key is INT32_MIN (`_ord_key`),
            # so the first live element always replaces it.
            best_key = Int32(_INT32_MIN)
            best_col = Int32(-1)
            for v0 in range(tid, n_vec, Int32(block_threads * LOADS_IN_FLIGHT)):
                group = load_group(v0)
                for u in range_constexpr(LOADS_IN_FLIGHT):
                    v = v0 + Int32(u * block_threads)
                    live = v < n_vec
                    for j in range_constexpr(vec):
                        key = key_of(group[u][j])
                        better = live & (key > best_key)
                        best_key = better.select(key, best_key)
                        best_col = better.select(v * Int32(vec) + Int32(j), best_col)
            for col in range(tail_start + tid, n, Int32(block_threads)):
                key = key_of(load_one(col))
                better = key > best_key
                best_key = better.select(key, best_key)
                best_col = better.select(col, best_col)

            # Across lanes and waves, ties go to the smaller column, so the answer does
            # not depend on which thread held an element.
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            red_key = lds.red_key.view(fx.make_layout(n_waves, 1))
            red_col = lds.red_col.view(fx.make_layout(n_waves, 1))

            def fold(key_a, col_a, key_b, col_b):
                take_b = (key_b > key_a) | ((key_b == key_a) & (col_b < col_a))
                return take_b.select(key_b, key_a), take_b.select(col_b, col_a)

            for sh in range_constexpr(6):
                off = warp >> (sh + 1)
                peer_key = best_key.shuffle_xor(off, warp)
                peer_col = best_col.shuffle_xor(off, warp)
                best_key, best_col = fold(best_key, best_col, peer_key, peer_col)
            if lane == zero:
                red_key[wave] = best_key
                red_col[wave] = best_col
            gpu.barrier()
            if tid == zero:
                top_key = red_key[zero]
                top_col = red_col[zero]
                for w in range_constexpr(1, n_waves):
                    top_key, top_col = fold(
                        top_key, top_col, red_key[Int32(w)], red_col[Int32(w)]
                    )
                write_result(zero, top_col)
        else:
            # ---- general k: radix select on the unsigned key. --------------------
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            hist = lds.hist.view(fx.make_layout(RADIX, 1))
            wave_tot = lds.wave_tot.view(fx.make_layout(n_waves, 1))
            state = lds.state.view(fx.make_layout(_ST_SLOTS, 1))
            surv_key = lds.surv_key.view(fx.make_layout(k_max, 1))
            surv_col = lds.surv_col.view(fx.make_layout(k_max, 1))
            keys = lds.keys.view(fx.make_layout(CACHE_KEYS, 1))

            if tid == zero:
                state[_ST_PREFIX] = zero
                state[_ST_NEED] = K
                state[_ST_COUNT] = zero
                state[_ST_RUN] = zero

            def for_each_loaded(body):
                # body(col, key, live) for every column of the row, read from memory.
                for v0 in range(tid, n_vec, Int32(block_threads * LOADS_IN_FLIGHT)):
                    group = load_group(v0)
                    for u in range_constexpr(LOADS_IN_FLIGHT):
                        v = v0 + Int32(u * block_threads)
                        live = v < n_vec
                        for j in range_constexpr(vec):
                            body(v * Int32(vec) + Int32(j), key_of(group[u][j]), live)
                for col in range(tail_start + tid, n, Int32(block_threads)):
                    body(col, key_of(load_one(col)), col < n)

            # A row of up to CACHE_KEYS columns is read once, its keys kept in LDS for
            # every later pass; a longer one is re-read by each pass. Either way each
            # pass is one compact loop: unrolling a pass per element, or the passes
            # themselves, makes an object large enough that instruction fetch, not the
            # row, dominates a short row.
            cached = n <= Int32(CACHE_KEYS)

            def cache_key(col, key, live):
                if live:
                    fx.memref_store(key, keys, col)

            if cached:
                for_each_loaded(cache_key)

            def for_each_key(body):
                if cached:
                    for col in range(tid, n, Int32(block_threads)):
                        body(col, keys[col], col < n)
                else:
                    for_each_loaded(body)

            for p in range(zero, Int32(32 // RADIX_BITS), one):
                p_shift = Int32(32 - RADIX_BITS) - p * Int32(RADIX_BITS)
                # Bits above this digit, fixed by earlier passes (none on the first: a
                # shift by 32 is undefined, so the mask is shifted in two steps).
                p_mask = (Int32(-1) << (p_shift + Int32(RADIX_BITS - 1))) << one
                hist[tid] = zero
                gpu.barrier()
                p_prefix = state[_ST_PREFIX]
                p_need = state[_ST_NEED]

                def count(col, key, live):
                    ukey = key ^ Int32(_INT32_MIN)
                    if live & ((ukey & p_mask) == p_prefix):
                        _lds_atomic_add(hist, (ukey >> p_shift) & Int32(RADIX - 1), one)

                for_each_key(count)
                gpu.barrier()

                # Keys above bin t: a suffix sum, within the wave by shuffle and then
                # across the waves' totals. Exactly one bin holds the need-th key.
                mine = hist[tid]
                above = mine
                for sh in range_constexpr(6):
                    off = 1 << sh
                    peer = gpu.shuffle_down(above, off, warp)
                    above = above + (lane + Int32(off) < Int32(warp)).select(
                        Int32(peer), zero
                    )
                if lane == zero:
                    wave_tot[wave] = above
                gpu.barrier()
                for w in range_constexpr(n_waves):
                    above = above + (Int32(w) > wave).select(wave_tot[Int32(w)], zero)
                above = above - mine
                if (above < p_need) & (above + mine >= p_need):
                    state[_ST_PREFIX] = p_prefix | (tid << p_shift)
                    state[_ST_NEED] = p_need - above
                    state[_ST_EQ] = mine
                gpu.barrier()

            # T, as a signed key; how many keys equal to T are still to be taken; and E.
            threshold = state[_ST_PREFIX] ^ Int32(_INT32_MIN)
            need = state[_ST_NEED]
            take_all_ties = state[_ST_EQ] == need

            def append(col, key):
                at = _lds_atomic_add(state, _ST_COUNT, one)
                surv_key[at] = key
                surv_col[at] = col

            def collect(col, key, live):
                if live & ((key > threshold) | (take_all_ties & (key == threshold))):
                    append(col, key)

            for_each_key(collect)

            # More keys equal T than places remain: take the lowest columns, in tiles of
            # one column per thread, ranking each tie by a block-wide prefix count.
            if state[_ST_EQ] != need:
                for base in range(zero, n, Int32(block_threads)):
                    col = base + tid
                    live = col < n
                    safe = live.select(col, zero)
                    key = cached.select(
                        keys[fx.min(safe, Int32(CACHE_KEYS - 1))],
                        key_of(load_one(safe)),
                    )
                    tied = live & (key == threshold)
                    mask = fly_rocdl.ballot(T.i64, tied)
                    below = Int32(
                        fx.math.ctpop(mask & ((Int64(1) << Int64(lane)) - Int64(1)))
                    )
                    if lane == zero:
                        wave_tot[wave] = Int32(fx.math.ctpop(mask))
                    gpu.barrier()
                    run = state[_ST_RUN]
                    earlier = zero
                    total = zero
                    for w in range_constexpr(n_waves):
                        count_w = wave_tot[Int32(w)]
                        earlier = earlier + (Int32(w) < wave).select(count_w, zero)
                        total = total + count_w
                    if tied & (run + earlier + below < need):
                        append(col, key)
                    gpu.barrier()
                    if tid == zero:
                        state[_ST_RUN] = run + total
            gpu.barrier()

            # Rank the k survivors; the rank is the output slot.
            for mine_at in range(tid, K, Int32(block_threads)):
                my_key = surv_key[mine_at]
                my_col = surv_col[mine_at]
                place = zero
                for other in range(zero, K, one):
                    peer_key = surv_key[other]
                    peer_col = surv_col[other]
                    ahead = (peer_key > my_key) | (
                        (peer_key == my_key) & (peer_col < my_col)
                    )
                    place = place + ahead.select(one, zero)
                write_result(place, my_col)

    @flyc.jit
    def launch_topk(
        X: fx.Tensor,
        Indices: fx.Tensor,
        Values: fx.Tensor,
        K: fx.Int32,
        Largest: fx.Int32,
        WriteValues: fx.Int32,
        VecOk: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        topk_kernel(X, Indices, Values, K, Largest, WriteValues, VecOk).launch(
            grid=(rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk
