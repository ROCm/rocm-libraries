# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/FlyDSL
#   commit:   89ad52fbbb9e252396829a8a44b12e18b7acd970  (v0.3.4.1-19-g89ad52f)
#   path:     kernels/norm/rmsnorm_bwd_kernel.py
#   compiler: flydsl 0.3.4 (pip wheel)   toolchain: ROCm 7.13.0
#
# Modifications (all in the two-stage builder, _build_rmsnorm_bwd_two_stage_module):
#   - N=None with an `n_max` bound selects a bounded runtime-N variant. The
#     per-thread dweight accumulators and the unrolled column loops are sized
#     for n_max; the row width is read from the input descriptor (Source.shape[1])
#     and masks every column past it, so one AOT object serves every N <= n_max.
#     The vec8 path keeps its compile-time N, as in rmsnorm_kernel.py.
#   - The dweight partial workspace is a 2-D (programs, N) tensor, and both
#     kernels read the program count from its descriptor (shape[0]) instead of
#     the baked num_programs: the partial kernel strides rows by it and writes
#     its own row by slicing, the reduce kernel sums that many rows. The caller
#     chooses the program count per launch (and launches that many partial
#     blocks); the `num_programs` argument now only sizes the convenience
#     launcher's grid. The reduce kernel likewise reads N from DWeight.shape[0].
#
# Upstream-first: prefer landing these changes in FlyDSL; this copy exists so
# the provider is not blocked on that. Re-diff with tools/diff_upstream.py.

"""RMSNorm backward kernel builders for plain and fused-add variants."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import atomic_add, dtype_to_elem_type
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    VEC_WIDTH,
    WARP_SIZE,
    load_scalar,
    load_vec,
    load_weight_vec,
    make_single_reduction_storage,
    resolve_rmsnorm_weight_dtype,
    store_scalar,
    store_vec,
    to_elem_vec,
    weight_vec_width,
)

# The staged path is selected by the Python wrapper only when M is large
# enough to keep one persistent program per CU busy.  NUM_PROGRAMS is a build
# parameter (a small multiple of the device CU count); M remains a runtime
# argument so a compiled callable can be reused across row counts.
DWEIGHT_REDUCE_COLS = 64
DWEIGHT_REDUCE_ROW_LANES = 4
DWEIGHT_REDUCE_THREADS = DWEIGHT_REDUCE_COLS * DWEIGHT_REDUCE_ROW_LANES
TWO_STAGE_PARTIAL_THREADS = 512


def is_rmsnorm_bwd_two_stage_vec_config(N: int, dtype_str: str) -> bool:
    """Whether the staged main kernel can use full-width vec8 column I/O."""
    return (
        dtype_str in ("f16", "bf16")
        and N >= TWO_STAGE_PARTIAL_THREADS * VEC_WIDTH
        and N % VEC_WIDTH == 0
    )


def build_rmsnorm_bwd_module(
    N: int, dtype_str: str, weight_dtype_str: str | None = None
):
    """RMSNorm backward atomic fallback: one block per row.

    Pass 1: c1 = mean_N(x_hat * wdy), x_hat = x*rstd, wdy = dy*gamma.
    Pass 2: dx = (wdy - x_hat*c1) * rstd  -> DX (elem dtype);
            dw_elem = dy * x_hat (fp32)   -> atomicAdd into DWeight[idx] (fp32).
    eps is baked into Rstd by the forward, so it is not needed here.
    """
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    SharedStorage = make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_bwd_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_in = fx.slice(Input_buf, (bid, None))
        row_dy = fx.slice(DY_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        gamma_copy_atom_s = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy16b()
                if weight_elem_bits <= 16
                else fx.rocdl.BufferCopy32b()
            ),
            weight_elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

        rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, bid)

        # Pass 1: c1 = mean( x_hat * wdy ) = mean( (x*rstd) * (dy*gamma) )
        thread_acc = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            x_e = load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
            dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            g_e = load_scalar(gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx_safe)
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
            x_hat = x * rstd
            wdy = dy * g
            prod = x_hat * wdy
            thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

        sum_prod = block_reduce_add(thread_acc)
        c1 = sum_prod / n_float

        # Pass 2: dx = (wdy - x_hat*c1) * rstd ; dw = dy * x_hat (atomicAdd fp32)
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                x_e = load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                g_e = load_scalar(gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = x * rstd
                wdy = dy * g
                dx = (wdy - x_hat * c1) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                store_scalar(copy_atom_s, elem_dtype, dx_div, idx, dx_e)

                dw = dy * x_hat
                atomic_add(DWeight, idx, dw, dtype_bytes=4)

    @flyc.jit
    def launch_rmsnorm_bwd(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_bwd_kernel(Input, Gamma, DY, Rstd, DX, DWeight)
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_rmsnorm_bwd


def build_fused_add_rmsnorm_bwd_module(
    N: int, dtype_str: str, weight_dtype_str: str | None = None
):
    """Fused-add / prenorm RMSNorm backward: grid=(M,), one block per row.

    Inputs: Added (= residual_out from fwd), Gamma, DY (= dout), DResidualOut
    (grad flowing into residual_out from downstream), Rstd.

    With a_hat = Added*rstd, wdy = dy*gamma, c1 = mean_N(a_hat*wdy):
      d_added = (wdy - a_hat*c1) * rstd          (gradient through the norm)
      total   = d_added + dresidual_out_elem     (downstream residual grad)
    Since added = x + residual_in, dx == dresidual == total unconditionally, so
    the kernel writes `total` to DX only; the python wrapper returns dx as both
    grads (aliased). dweight = sum_rows(dy * a_hat) (fp32 atomicAdd).

    eps is baked into Rstd by the forward, so it is not needed here.

    DResidualOut is always a real tensor: the Python wrapper passes a zero
    tensor when the caller has no downstream residual grad (pure-norm case).
    This keeps the kernel branch-free wrt None.
    """
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    SharedStorage = make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_bwd_kernel(
        Added: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Added_buf = fx.rocdl.make_buffer_tensor(Added)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        DResidualOut_buf = fx.rocdl.make_buffer_tensor(DResidualOut)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_added = fx.slice(Added_buf, (bid, None))
        row_dy = fx.slice(DY_buf, (bid, None))
        row_dres_out = fx.slice(DResidualOut_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        gamma_copy_atom_s = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy16b()
                if weight_elem_bits <= 16
                else fx.rocdl.BufferCopy32b()
            ),
            weight_elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        added_div = fx.logical_divide(row_added, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        dres_out_div = fx.logical_divide(row_dres_out, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

        rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, bid)

        # Pass 1: c1 = mean( a_hat * wdy ) = mean( (added*rstd) * (dy*gamma) )
        thread_acc = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            a_e = load_scalar(copy_atom_s, elem_dtype, added_div, idx_safe)
            dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            g_e = load_scalar(gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx_safe)
            a = a_e if dtype_str == "f32" else a_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
            a_hat = a * rstd
            wdy = dy * g
            prod = a_hat * wdy
            thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

        sum_prod = block_reduce_add(thread_acc)
        c1 = sum_prod / n_float

        # Pass 2: d_added = (wdy - a_hat*c1) * rstd ; total = d_added + dres_out
        #         store total -> DX only (the wrapper aliases dx as dresidual, since
        #         added = x + residual_in => dx == dresidual) ; dw = dy*a_hat (atomicAdd fp32)
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                a_e = load_scalar(copy_atom_s, elem_dtype, added_div, idx)
                dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                g_e = load_scalar(gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx)
                dres_out_e = load_scalar(copy_atom_s, elem_dtype, dres_out_div, idx)
                a = a_e if dtype_str == "f32" else a_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                dres_out = (
                    dres_out_e if dtype_str == "f32" else dres_out_e.to(fx.Float32)
                )
                a_hat = a * rstd
                wdy = dy * g
                d_added = (wdy - a_hat * c1) * rstd
                total = d_added + dres_out
                total_e = total if dtype_str == "f32" else total.to(elem_dtype)
                store_scalar(copy_atom_s, elem_dtype, dx_div, idx, total_e)

                dw = dy * a_hat
                atomic_add(DWeight, idx, dw, dtype_bytes=4)

    @flyc.jit
    def launch_fused_add_rmsnorm_bwd(
        Added: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_bwd_kernel(
            Added, Gamma, DY, DResidualOut, Rstd, DX, DWeight
        )
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fused_add_rmsnorm_bwd


def _build_rmsnorm_bwd_two_stage_module(
    N: int | None,
    dtype_str: str,
    num_programs: int,
    *,
    fused_add: bool,
    weight_dtype_str: str | None = None,
    n_max: int | None = None,
):
    """Build the large-M persistent backward + dweight finalizer.

    The 512-thread main kernel walks rows in a grid-stride loop, writes every
    assigned ``dx`` row, and accumulates one FP32 dweight partial in registers.
    Full-width 16-bit rows use vec8 I/O and retain source, dy, and gamma across
    the row reduction; other shapes keep the scalar fallback.  Each program writes
    one ``N``-wide partial row, then a second kernel reduces all partial rows and
    stores dweight directly in the parameter dtype.

    This replaces M competing atomic adds per dweight element with a bounded
    ``num_programs x N`` workspace and a deterministic final reduction.  The
    caller owns the workspace for the duration of one invocation; the builder
    never caches writable storage.
    """
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    RUNTIME_N = N is None
    if RUNTIME_N and (n_max is None or n_max <= 0):
        raise ValueError("a runtime-N build (N=None) needs a positive n_max bound")
    # The width the unrolled loops and register accumulators are sized for.
    N_STATIC = n_max if RUNTIME_N else N

    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    RED_SLOTS = max(1, (TWO_STAGE_PARTIAL_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC = not RUNTIME_N and is_rmsnorm_bwd_two_stage_vec_config(N, dtype_str)
    IO_WIDTH = VEC_WIDTH if USE_VEC else 1
    WEIGHT_IO_WIDTH = weight_vec_width(weight_dtype_str) if USE_VEC else 1
    NUM_IO_TILES = (N_STATIC + IO_WIDTH - 1) // IO_WIDTH
    NUM_IO_ITERS = (
        NUM_IO_TILES + TWO_STAGE_PARTIAL_THREADS - 1
    ) // TWO_STAGE_PARTIAL_THREADS
    PARTIAL_ACC_SIZE = NUM_IO_ITERS * IO_WIDTH
    arch = get_rocm_arch() if USE_VEC else ""
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    SharedStorage = make_single_reduction_storage(RED_SLOTS)
    DWeightReduceStorage = make_single_reduction_storage(DWEIGHT_REDUCE_THREADS)

    @flyc.kernel(known_block_size=[TWO_STAGE_PARTIAL_THREADS, 1, 1])
    def rmsnorm_bwd_partial_kernel(
        Source: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeightPartial: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        if const_expr(RUNTIME_N):
            # IO_WIDTH is 1 here, so the row width is also the column-tile bound.
            n_rt = Source.shape.to_py_value()[1]
            n_float = n_rt.to(fx.Float32)
            io_tiles = n_rt
        else:
            n_float = float(N)
            io_tiles = NUM_IO_TILES
        programs = DWeightPartial.shape.to_py_value()[0]
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Source_buf = fx.rocdl.make_buffer_tensor(Source)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        DResidualOut_buf = fx.rocdl.make_buffer_tensor(DResidualOut)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)
        DWeightPartial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)

        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
        partial_div = fx.logical_divide(
            fx.slice(DWeightPartial_buf, (bid, None)), fx.make_layout(1, 1)
        )

        copy_atom_io = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if USE_VEC
                else (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                )
            ),
            elem_bits,
        )
        gamma_copy_atom = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if USE_VEC
                else (
                    fx.rocdl.BufferCopy16b()
                    if weight_elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                )
            ),
            weight_elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(WEIGHT_IO_WIDTH, 1))
        gamma_local = []
        if const_expr(USE_VEC):
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < io_tiles
                io_idx_safe = is_valid.select(io_idx, 0)
                gamma_local.append(
                    load_weight_vec(
                        gamma_copy_atom,
                        weight_dtype_str,
                        weight_elem_dtype,
                        gamma_div,
                        io_idx_safe,
                    )
                )

        dweight_partial = fx.Vector.filled(PARTIAL_ACC_SIZE, 0.0, fx.Float32)
        for row in range(bid, MIn, programs):
            row_source = fx.slice(Source_buf, (row, None))
            row_dy = fx.slice(DY_buf, (row, None))
            row_dx = fx.slice(DX_buf, (row, None))
            if const_expr(fused_add):
                row_dres_out = fx.slice(DResidualOut_buf, (row, None))

            source_div = fx.logical_divide(row_source, fx.make_layout(IO_WIDTH, 1))
            dy_div = fx.logical_divide(row_dy, fx.make_layout(IO_WIDTH, 1))
            dx_div = fx.logical_divide(row_dx, fx.make_layout(IO_WIDTH, 1))
            if const_expr(fused_add):
                dres_out_div = fx.logical_divide(
                    row_dres_out, fx.make_layout(IO_WIDTH, 1)
                )

            rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, row)
            thread_acc = c_zero_f
            source_local = []
            dy_local = []
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < io_tiles
                io_idx_safe = is_valid.select(io_idx, 0)
                if const_expr(USE_VEC):
                    source_e = load_vec(
                        copy_atom_io, IO_WIDTH, elem_dtype, source_div, io_idx_safe
                    )
                    dy_e = load_vec(
                        copy_atom_io, IO_WIDTH, elem_dtype, dy_div, io_idx_safe
                    )
                    source_local.append(source_e)
                    dy_local.append(dy_e)
                    gamma_e = gamma_local[tile_i]
                else:
                    source_e = load_scalar(
                        copy_atom_io, elem_dtype, source_div, io_idx_safe
                    )
                    dy_e = load_scalar(copy_atom_io, elem_dtype, dy_div, io_idx_safe)
                    gamma_e = load_scalar(
                        gamma_copy_atom, weight_elem_dtype, gamma_div, io_idx_safe
                    )

                source = source_e if dtype_str == "f32" else source_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                gamma = (
                    gamma_e
                    if USE_VEC or weight_dtype_str == "f32"
                    else gamma_e.to(fx.Float32)
                )
                source_hat = source * rstd
                wdy = dy * gamma
                prod = source_hat * wdy
                if const_expr(USE_VEC):
                    prod = prod.reduce(ReductionOp.ADD, fastmath=fm_fast)

                thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

            sum_prod = block_reduce_add(thread_acc)
            c1 = sum_prod / n_float
            row_dweight = []
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < io_tiles
                io_idx_safe = is_valid.select(io_idx, 0)
                if const_expr(USE_VEC):
                    source_e = source_local[tile_i]
                    dy_e = dy_local[tile_i]
                    gamma_e = gamma_local[tile_i]
                else:
                    source_e = load_scalar(
                        copy_atom_io, elem_dtype, source_div, io_idx_safe
                    )
                    dy_e = load_scalar(copy_atom_io, elem_dtype, dy_div, io_idx_safe)
                    gamma_e = load_scalar(
                        gamma_copy_atom, weight_elem_dtype, gamma_div, io_idx_safe
                    )

                source = source_e if dtype_str == "f32" else source_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                gamma = (
                    gamma_e
                    if USE_VEC or weight_dtype_str == "f32"
                    else gamma_e.to(fx.Float32)
                )
                source_hat = source * rstd
                wdy = dy * gamma

                dx = (wdy - source_hat * c1) * rstd
                if const_expr(fused_add):
                    if const_expr(USE_VEC):
                        dres = load_vec(
                            copy_atom_io,
                            IO_WIDTH,
                            elem_dtype,
                            dres_out_div,
                            io_idx_safe,
                        ).to(fx.Float32)
                    else:
                        dres_e = load_scalar(
                            copy_atom_io, elem_dtype, dres_out_div, io_idx_safe
                        )
                        dres = dres_e if dtype_str == "f32" else dres_e.to(fx.Float32)
                    dx = dx + dres

                if io_idx < io_tiles:
                    if const_expr(USE_VEC):
                        dx_e = to_elem_vec(
                            dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, dx
                        )
                        store_vec(
                            copy_atom_io, IO_WIDTH, elem_dtype, dx_e, dx_div, io_idx
                        )
                    else:
                        dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                        store_scalar(copy_atom_io, elem_dtype, dx_div, io_idx, dx_e)

                dw = dy * source_hat
                if const_expr(USE_VEC):
                    for lane in range_constexpr(IO_WIDTH):
                        row_dweight.append(is_valid.select(dw[lane], c_zero_f))
                else:
                    row_dweight.append(is_valid.select(dw, c_zero_f))

            dweight_partial = dweight_partial + fx.Vector.from_elements(
                row_dweight, fx.Float32
            )
            gpu.barrier()

        for tile_i in range_constexpr(NUM_IO_ITERS):
            io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
            if io_idx < io_tiles:
                for lane in range_constexpr(IO_WIDTH):
                    idx = io_idx * IO_WIDTH + lane
                    partial_value = dweight_partial[tile_i * IO_WIDTH + lane]
                    store_scalar(
                        copy_atom_f32,
                        fx.Float32,
                        partial_div,
                        idx,
                        partial_value,
                    )

    @flyc.kernel
    def rmsnorm_bwd_dweight_reduce_kernel(
        DWeightPartial: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        col_lane = tid % DWEIGHT_REDUCE_COLS
        partial_lane = tid // DWEIGHT_REDUCE_COLS
        col = bid * DWEIGHT_REDUCE_COLS + col_lane
        n_cols = DWeight.shape.to_py_value()  # 1-D: the extent itself, not a tuple
        programs = DWeightPartial.shape.to_py_value()[0]
        is_valid = col < n_cols
        col_safe = is_valid.select(col, 0)

        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        DWeightPartial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)
        DWeight_buf = fx.rocdl.make_buffer_tensor(DWeight)
        dweight_div = fx.logical_divide(DWeight_buf, fx.make_layout(1, 1))
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        weight_copy_atom_s = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy16b()
                if weight_elem_bits <= 16
                else fx.rocdl.BufferCopy32b()
            ),
            weight_elem_bits,
        )

        lds = fx.SharedAllocator().allocate(DWeightReduceStorage).peek()
        s_partial = lds.s_red.view(fx.make_layout(DWEIGHT_REDUCE_THREADS, 1))

        c_zero_f = fx.Float32(0.0)
        acc = c_zero_f
        for partial_base in range(0, programs, DWEIGHT_REDUCE_ROW_LANES):
            partial_row = partial_base + partial_lane
            partial_valid = partial_row < programs
            partial_row_safe = partial_valid.select(partial_row, 0)
            partial_div = fx.logical_divide(
                fx.slice(DWeightPartial_buf, (partial_row_safe, None)),
                fx.make_layout(1, 1),
            )
            value = load_scalar(copy_atom_f32, fx.Float32, partial_div, col_safe)
            acc = acc + partial_valid.select(value, c_zero_f)
        fx.memref_store(acc, s_partial, tid)
        gpu.barrier()

        if partial_lane == 0:
            if col < n_cols:
                total = c_zero_f
                for lane in range_constexpr(DWEIGHT_REDUCE_ROW_LANES):
                    total = total + fx.memref_load(
                        s_partial, lane * DWEIGHT_REDUCE_COLS + col_lane
                    )
                out = (
                    total if weight_dtype_str == "f32" else total.to(weight_elem_dtype)
                )
                store_scalar(
                    weight_copy_atom_s,
                    weight_elem_dtype,
                    dweight_div,
                    col,
                    out,
                )

    # A runtime-N build launches for n_max; columns past the real N are masked.
    reduce_grid = (N_STATIC + DWEIGHT_REDUCE_COLS - 1) // DWEIGHT_REDUCE_COLS

    if fused_add:

        @flyc.jit
        def launch_fused_add_rmsnorm_bwd_two_stage(
            Added: fx.Tensor,
            Gamma: fx.Tensor,
            DY: fx.Tensor,
            DResidualOut: fx.Tensor,
            Rstd: fx.Tensor,
            DX: fx.Tensor,
            DWeight: fx.Tensor,
            DWeightPartial: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            partial_launcher = rmsnorm_bwd_partial_kernel(
                Added,
                Gamma,
                DY,
                DResidualOut,
                Rstd,
                DX,
                DWeightPartial,
                m_in,
            )
            partial_launcher.launch(
                grid=(num_programs, 1, 1),
                block=(TWO_STAGE_PARTIAL_THREADS, 1, 1),
                stream=stream,
            )
            reduce_launcher = rmsnorm_bwd_dweight_reduce_kernel(DWeightPartial, DWeight)
            reduce_launcher.launch(
                grid=(reduce_grid, 1, 1),
                block=(DWEIGHT_REDUCE_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_bwd_two_stage

    @flyc.jit
    def launch_rmsnorm_bwd_two_stage(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        DWeightPartial: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # DResidualOut is compile-time dead in the plain variant; DY fills the
        # otherwise-unused tensor slot so both variants share one main kernel.
        partial_launcher = rmsnorm_bwd_partial_kernel(
            Input,
            Gamma,
            DY,
            DY,
            Rstd,
            DX,
            DWeightPartial,
            m_in,
        )
        partial_launcher.launch(
            grid=(num_programs, 1, 1),
            block=(TWO_STAGE_PARTIAL_THREADS, 1, 1),
            stream=stream,
        )
        reduce_launcher = rmsnorm_bwd_dweight_reduce_kernel(DWeightPartial, DWeight)
        reduce_launcher.launch(
            grid=(reduce_grid, 1, 1),
            block=(DWEIGHT_REDUCE_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd_two_stage


def build_rmsnorm_bwd_two_stage_module(
    N: int | None,
    dtype_str: str,
    num_programs: int,
    weight_dtype_str: str | None = None,
    n_max: int | None = None,
):
    """Large-M plain RMSNorm backward without global dweight atomics.

    ``N=None`` with ``n_max`` builds the bounded runtime-N variant (see the
    header). DWeightPartial is a (programs, N) f32 tensor.
    """
    return _build_rmsnorm_bwd_two_stage_module(
        N,
        dtype_str,
        num_programs,
        fused_add=False,
        weight_dtype_str=weight_dtype_str,
        n_max=n_max,
    )


def build_fused_add_rmsnorm_bwd_two_stage_module(
    N: int,
    dtype_str: str,
    num_programs: int,
    weight_dtype_str: str | None = None,
):
    """Large-M fused-add RMSNorm backward without global dweight atomics."""
    return _build_rmsnorm_bwd_two_stage_module(
        N,
        dtype_str,
        num_programs,
        fused_add=True,
        weight_dtype_str=weight_dtype_str,
    )
