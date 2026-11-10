# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Not vendored: a port to flydsl 0.3.4 (pip wheel), written for this provider.
#   toolchain: ROCm 7.1.0
#
# Ported from upstream FlyDSL
#   https://github.com/ROCm/FlyDSL
#   commit:   226cde8d2a84bc2687c8d69587607fddf0c16f2f  (v0.5.0)
#   path:     kernels/norm/softmax_kernel.py
# which is newer than the FlyDSL commit the other vendored kernels are pinned
# to, so tools/diff_upstream.py cannot compare it and skips it. Its
# register-buffered compile-time-N path is kept as the baked tier
# (`N` an int); everything else is modified:
#   1. A runtime-N tier (`N=None`): two streaming passes over global memory,
#      an online (max, sum) pass and a normalize pass, in scf.for loops whose
#      trip count comes from the input descriptor (P.shape[1]). One object
#      serves every width, up to a vocabulary-sized row, since nothing is held
#      in registers across the row. Upstream's generic path unrolled a
#      compile-time N into registers.
#   2. Fully masked rows (every element -inf) give s = 0 and stats = -inf
#      instead of NaN, as hipDNN's CPU reference does: the exponent is taken
#      against max(p) with -inf replaced by 0, and the reciprocal sum is 0 when
#      max(p) is -inf. Max reductions are NaN-propagating (arith.maximumf), so
#      a NaN anywhere in a row makes the whole row and its stats NaN, as the
#      reference does. No fast-math flag that assumes away NaN or infinity is
#      used (upstream applied `fast` to exp2 and the sum).
#   3. A per-row log-sum-exp output, switched at run time (`StoreStats`, an
#      i32): stats = (max + ln(sum)) * StatsScale, a runtime f32, so one
#      object serves no stats, natural-log stats and log2 stats
#      (StatsScale = 1/ln 2). Lane 0 of each row stores it.
#   4. The output dtype may differ from the input's (`out_dtype_str`); the
#      output is stored in up-to-128-bit pieces.
#   5. Kernel arguments: (P, S, Stats, MIn, StoreStats, StatsScale).
#      Upstream's two padding tensors are gone; Stats is a 1-D tensor, which
#      the launcher fills with any valid mapping when no stats are stored.
#   6. The vector width of the runtime tier is a builder argument (`VEC`):
#      VEC > 1 needs N % VEC == 0 (the host matcher checks it), VEC = 1 serves
#      every N with scalar copies. `UNROLL` vectors are loaded per thread per
#      loop iteration so several loads are in flight.
#   7. Warp/block reductions come from fx.coop (present in 0.3.4) with the
#      width THREADS_PER_ROW; the autotune entry point `softmax_direct` and
#      TUNING_SCHEMA are dropped (AOT objects are built from the instance
#      table).
#
# Upstream-first: a runtime-N softmax with stats belongs in FlyDSL; this copy
# exists so the provider is not blocked on that.

"""Softmax over the last dimension, with an optional log-sum-exp output.

    s[r, j]  = exp(p[r, j] - max_j p[r, :]) / sum_j exp(p[r, j] - max_j p[r, :])
    stats[r] = (max_j p[r, :] + ln(sum_j exp(p[r, j] - max))) * StatsScale

stats is written only when StoreStats is non-zero.

Arithmetic is f32. Rows are independent; ``THREADS_PER_ROW`` threads serve one
row and a block of ``BLOCK_THREADS`` threads serves
``BLOCK_THREADS // THREADS_PER_ROW`` consecutive rows, so the grid is
``ceil(M / rows_per_block)``.

Two tiers:
  - Baked (``N`` an int, ``N % (THREADS_PER_ROW * VEC) == 0``): upstream's
    register-buffered path. Each thread holds its slice of the row, so p is
    read once.
  - Runtime (``N`` is None): p is read twice, once for the online max and sum
    and once to normalize, so any width is served with constant registers.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from kernels.common.kernels_common import LOG2E, dtype_to_elem_type, get_warp_size

KERNEL_NAME = "softmax_kernel"

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
LN2 = math.log(2.0)

# Loads per thread per runtime-tier loop iteration.
UNROLL = 4


def _elem_bits(dtype_str: str) -> int:
    if dtype_str == "f32":
        return 32
    if dtype_str in ("f16", "bf16"):
        return 16
    raise ValueError(f"unsupported softmax dtype {dtype_str!r}")


def build_softmax_module(
    N: int | None,
    dtype_str: str,
    out_dtype_str: str | None = None,
    BLOCK_THREADS: int = BLOCK_THREADS,
    THREADS_PER_ROW: int | None = None,
    VEC: int | None = None,
    UNROLL: int = UNROLL,
):
    """Build a softmax launcher.

    ``N`` is the row width: an int bakes it (the register-buffered tier), None
    reads it from the input descriptor at run time. ``VEC`` is the elements per
    copy: 128 bits of input by default; 1 for the runtime tier's any-width
    scalar form. ``launch(P, S, Stats, m_in, store_stats, stats_scale, stream)``.
    """
    RUNTIME_N = N is None
    out_dtype_str = dtype_str if out_dtype_str is None else out_dtype_str
    THREADS_PER_ROW = BLOCK_THREADS if THREADS_PER_ROW is None else THREADS_PER_ROW
    if THREADS_PER_ROW <= 0 or THREADS_PER_ROW & (THREADS_PER_ROW - 1):
        raise ValueError(
            f"THREADS_PER_ROW must be a positive power of two, got {THREADS_PER_ROW}"
        )
    if BLOCK_THREADS % THREADS_PER_ROW:
        raise ValueError(
            f"BLOCK_THREADS {BLOCK_THREADS} is not a multiple of THREADS_PER_ROW "
            f"{THREADS_PER_ROW}"
        )
    ROWS_PER_BLOCK = BLOCK_THREADS // THREADS_PER_ROW
    if THREADS_PER_ROW > WARP_SIZE and ROWS_PER_BLOCK != 1:
        raise ValueError("multi-row blocks require THREADS_PER_ROW <= WARP_SIZE")

    elem_bits = _elem_bits(dtype_str)
    out_bits = _elem_bits(out_dtype_str)
    full_vec = 128 // elem_bits
    VEC = full_vec if VEC is None else VEC
    if VEC not in (1, full_vec):
        raise ValueError(f"VEC must be 1 or {full_vec} for {dtype_str}, got {VEC}")
    if UNROLL < 1:
        raise ValueError(f"UNROLL must be positive, got {UNROLL}")
    tile_cols = THREADS_PER_ROW * VEC
    if not RUNTIME_N:
        if N <= 0 or N % tile_cols:
            raise ValueError(
                f"a baked N must be a positive multiple of THREADS_PER_ROW * VEC = "
                f"{tile_cols}, got {N}"
            )
        num_tiles = N // tile_cols

    # The output is stored in pieces of at most 128 bits: bf16 in, f32 out moves
    # 256 bits per input vector.
    out_piece = min(VEC, 128 // out_bits)
    out_pieces = VEC // out_piece

    def _copy_op(bits):
        return {
            16: fx.rocdl.BufferCopy16b,
            32: fx.rocdl.BufferCopy32b,
            64: fx.rocdl.BufferCopy64b,
            128: fx.rocdl.BufferCopy128b,
        }[bits]()

    BLOCK_REDUCE = THREADS_PER_ROW > WARP_SIZE
    if BLOCK_REDUCE:
        BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]
        SharedStorage = BlockReduce.SharedStorage
    _kernel_kwargs = (
        {} if BLOCK_THREADS <= 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    @flyc.kernel(**_kernel_kwargs)
    def softmax_kernel(
        P: fx.Tensor,
        S: fx.Tensor,
        Stats: fx.Tensor,
        MIn: fx.Int32,
        StoreStats: fx.Int32,
        StatsScale: fx.Float32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        in_elem = dtype_to_elem_type(dtype_str)
        out_elem = dtype_to_elem_type(out_dtype_str)
        # Only the approximate-function flag: exp2 lowers straight to v_exp_f32.
        # nnan/ninf would let the compiler fold away the -inf and NaN handling.
        fm_afn = arith.FastMathFlags.afn

        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))

        lane = tid
        row = bid
        row_valid = bid < MIn
        if const_expr(ROWS_PER_BLOCK > 1):
            lane = tid % THREADS_PER_ROW
            row = bid * ROWS_PER_BLOCK + tid // THREADS_PER_ROW
            row_valid = row < MIn
        row_safe = row_valid.select(row, 0)

        if const_expr(BLOCK_REDUCE):
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def row_reduce(val, op):
            if const_expr(not BLOCK_REDUCE):
                return fx.coop.warp_reduce(val, op, width=THREADS_PER_ROW)
            # Order the preceding reduction's readers before scratch reuse.
            gpu.barrier()
            return BlockReduce(val, op, storage=lds)

        def vec_max(x):
            # NaN-propagating, unlike Vector.reduce(MAX) (maxnumf).
            m = x[0]
            for i in range_constexpr(1, VEC):
                m = fx.max(m, x[i])
            return m

        def vec_exp(x, shift):
            return fmath.exp2((x - shift) * LOG2E, fastmath=fm_afn)

        P_buf = fx.rocdl.make_buffer_tensor(P)
        S_buf = fx.rocdl.make_buffer_tensor(S)
        row_p = fx.slice(P_buf, (row_safe, None))
        row_s = fx.slice(S_buf, (row_safe, None))
        p_div = fx.logical_divide(row_p, fx.make_layout(VEC, 1))
        s_div = fx.logical_divide(row_s, fx.make_layout(out_piece, 1))
        load_atom = fx.make_copy_atom(_copy_op(VEC * elem_bits), elem_bits)
        store_atom = fx.make_copy_atom(_copy_op(out_piece * out_bits), out_bits)

        def load_f32(vidx):
            r = fx.make_rmem_tensor(VEC, in_elem)
            fx.copy(load_atom, fx.slice(p_div, (None, vidx)), r)
            x = fx.memref_load_vec(r)
            if const_expr(dtype_str == "f32"):
                return x
            return x.to(fx.Float32)

        def store_f32(y, vidx):
            out = y
            if const_expr(out_dtype_str != "f32"):
                out = y.to(out_elem)
            for k in range_constexpr(out_pieces):
                piece = out
                if const_expr(out_pieces > 1):
                    piece = fx.Vector.from_elements(
                        [out[k * out_piece + i] for i in range(out_piece)], out_elem
                    )
                r = fx.make_rmem_tensor(out_piece, out_elem)
                fx.memref_store_vec(piece, r)
                fx.copy(store_atom, r, fx.slice(s_div, (None, vidx * out_pieces + k)))

        def finish(row_max, row_sum_of):
            """The row's shift, reciprocal sum and stats from its max and sum.

            A fully masked row has max -inf; shifting by 0 instead makes every
            exponent 0 rather than NaN, and its reciprocal sum is 0, so s is 0.
            """
            masked = row_max == c_neg_inf
            shift = masked.select(c_zero, row_max)
            row_sum = row_sum_of(shift)
            inv_sum = masked.select(c_zero, c_one / row_sum)
            lse = (shift + fmath.log2(row_sum) * LN2) * StatsScale
            lse = masked.select(c_neg_inf, lse)
            if row_valid:
                if lane == 0:
                    if StoreStats != 0:
                        stats_div = fx.logical_divide(
                            fx.rocdl.make_buffer_tensor(Stats), fx.make_layout(1, 1)
                        )
                        r = fx.make_rmem_tensor(1, fx.Float32)
                        fx.memref_store_vec(fx.Vector.filled(1, lse, fx.Float32), r)
                        fx.copy(
                            fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32),
                            r,
                            fx.slice(stats_div, (None, row)),
                        )
            return shift, inv_sum

        if const_expr(not RUNTIME_N):
            # Upstream's register-buffered path: the row is read once.
            row_buffer = []
            thread_max = c_neg_inf
            for t in range_constexpr(num_tiles):
                x = load_f32(lane + t * THREADS_PER_ROW)
                row_buffer.append(x)
                thread_max = fx.max(thread_max, vec_max(x))
            row_max = row_reduce(thread_max, ReductionOp.MAX)

            def baked_sum(shift):
                thread_sum = c_zero
                for t in range_constexpr(num_tiles):
                    e = vec_exp(row_buffer[t], shift)
                    row_buffer[t] = e
                    thread_sum = thread_sum + e.reduce(ReductionOp.ADD)
                return row_reduce(thread_sum, ReductionOp.ADD)

            _, inv_sum = finish(row_max, baked_sum)
            if row_valid:
                for t in range_constexpr(num_tiles):
                    store_f32(row_buffer[t] * inv_sum, lane + t * THREADS_PER_ROW)
        else:
            # Runtime N: dim 1 of the input descriptor. Rows past MIn (the tail
            # of the last multi-row block) get no iterations at all.
            n_cols = P.shape.to_py_value()[1]
            n_vec = n_cols // VEC
            n_vec = row_valid.select(n_vec, 0)
            step = THREADS_PER_ROW * UNROLL

            # Pass 1: each thread keeps a running max and a sum scaled to it,
            # rescaling the sum once per iteration when the max moves.
            for base, state in range(0, n_vec, step, init=[c_neg_inf, c_zero]):
                m_run = fx.Float32(state[0])
                s_run = fx.Float32(state[1])
                xs = []
                valids = []
                group_max = c_neg_inf
                for u in range_constexpr(UNROLL):
                    vidx = lane + fx.Int32(base) + u * THREADS_PER_ROW
                    valid = vidx < n_vec
                    x = load_f32(valid.select(vidx, 0))
                    group_max = fx.max(group_max, valid.select(vec_max(x), c_neg_inf))
                    xs.append(x)
                    valids.append(valid)
                m_new = fx.max(m_run, group_max)
                m_shift = (m_new == c_neg_inf).select(c_zero, m_new)
                s_new = s_run * fmath.exp2((m_run - m_shift) * LOG2E, fastmath=fm_afn)
                for u in range_constexpr(UNROLL):
                    e_sum = vec_exp(xs[u], m_shift).reduce(ReductionOp.ADD)
                    s_new = s_new + valids[u].select(e_sum, c_zero)
                results = yield [m_new, s_new]
            thread_max = fx.Float32(results[0])
            thread_sum = fx.Float32(results[1])

            row_max = row_reduce(thread_max, ReductionOp.MAX)

            def runtime_sum(shift):
                # A thread that saw no element has max -inf and sum 0: scaled to 0.
                scaled = thread_sum * fmath.exp2(
                    (thread_max - shift) * LOG2E, fastmath=fm_afn
                )
                return row_reduce(scaled, ReductionOp.ADD)

            shift, inv_sum = finish(row_max, runtime_sum)

            # Pass 2: normalize and store.
            for base, _state in range(0, n_vec, step, init=[]):
                for u in range_constexpr(UNROLL):
                    vidx = lane + fx.Int32(base) + u * THREADS_PER_ROW
                    if vidx < n_vec:
                        store_f32(vec_exp(load_f32(vidx), shift) * inv_sum, vidx)

    @flyc.jit
    def launch_softmax(
        P: fx.Tensor,
        S: fx.Tensor,
        Stats: fx.Tensor,
        m_in: fx.Int32,
        store_stats: fx.Int32,
        stats_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = softmax_kernel(P, S, Stats, m_in, store_stats, stats_scale)
        launcher.launch(
            grid=((m_in + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_softmax
