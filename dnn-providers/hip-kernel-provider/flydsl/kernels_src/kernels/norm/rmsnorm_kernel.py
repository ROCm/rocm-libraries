# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/FlyDSL
#   commit:   89ad52fbbb9e252396829a8a44b12e18b7acd970  (v0.3.4.1-19-g89ad52f)
#   path:     kernels/norm/rmsnorm_kernel.py
#   compiler: flydsl 0.3.4 (pip wheel)   toolchain: ROCm 7.13.0
#
# Modifications:
#   - build_rmsnorm_module(): accepts N=None, selecting a runtime-N generic
#     variant. The scalar fallback's range_constexpr loops become dynamic
#     scf.for loops whose trip count is read from the input tensor descriptor
#     (Input.shape[1]), so one AOT instance covers every N. The vectorized fast
#     path is unchanged and still requires a compile-time N -- it caches tiles
#     in a Python list of registers, which only exists under unrolling.
#   - Removed the `from kernels.norm.rmsnorm_bwd_kernel import (...)`
#     backward-compat re-export. Increment 1 is forward-only and the re-export
#     pulls in a module this provider never builds.
#   - Removed the trailing `if torch is not None:` PyTorch wrapper/autograd
#     section (upstream lines 1498-1902). This copy is imported by an AOT
#     generator to build modules, never for eager use, and the section drags in
#     kernels.common.tensor_shim plus the backward builders above.
#
# Upstream-first: prefer landing these changes in FlyDSL; this copy exists so
# the provider is not blocked on that. Re-diff with tools/diff_upstream.py.

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Three paths:
  - Fast path (compile-time N, N % tile_cols == 0): 128-bit buffer copies.
  - Generic path (compile-time N, arbitrary): scalar guarded copies, unrolled.
  - Runtime-N path (N is None): scalar guarded copies over an scf.for whose
    bound comes from the tensor descriptor. One instance covers every N.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import dtype_to_elem_type

from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    EPS,
    VEC_WIDTH,
    WARP_SIZE,
)
from kernels.norm.rmsnorm_common import load_scalar as _load_scalar
from kernels.norm.rmsnorm_common import load_vec as _load_vec
from kernels.norm.rmsnorm_common import load_weight_vec as _load_weight_vec
from kernels.norm.rmsnorm_common import (
    make_reduction_storage as _make_reduction_storage,
)
from kernels.norm.rmsnorm_common import (
    resolve_rmsnorm_weight_dtype as _resolve_rmsnorm_weight_dtype,
)
from kernels.norm.rmsnorm_common import store_scalar as _store_scalar
from kernels.norm.rmsnorm_common import store_vec as _store_vec
from kernels.norm.rmsnorm_common import to_elem_scalar as _to_elem_scalar
from kernels.norm.rmsnorm_common import to_elem_vec as _to_elem_vec
from kernels.norm.rmsnorm_common import weight_vec_width as _weight_vec_width

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "rmsnorm"

# The small-N path derives its own block geometry and is not tuned.
SMALL_N_THRESHOLD = 2048


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(
        f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')"
    )


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(
        f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')"
    )


def build_rmsnorm_module(
    N: int | None,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    BLOCK_THREADS: int = BLOCK_THREADS,
    weight_dtype_str: str | None = None,
):
    """Build an RMSNorm forward module.

    ``N`` is the normalized dimension. Passing an int specializes on it, which
    is what the vectorized fast path needs. Passing ``None`` builds the
    runtime-N instance: the trip count is read from the input descriptor, so a
    single AOT binary serves every N at the cost of the fast path.
    """
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    RUNTIME_N = N is None
    if not RUNTIME_N and N <= SMALL_N_THRESHOLD:
        return _build_rmsnorm_large_m_small_n_module(
            N, dtype_str, store_rstd, eps, weight_dtype_str
        )

    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    # BLOCK_THREADS controls storage, tiling, and launch geometry.
    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    _kernel_kwargs = (
        {} if BLOCK_THREADS <= 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel(**_kernel_kwargs)
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        if const_expr(RUNTIME_N):
            # Dim 1 of the input descriptor is the normalized dimension. Taking
            # the bound and the tail predicate from there -- rather than from a
            # Python int -- is the whole of the runtime-N variant: one binary
            # covers every N. Both extents are dynamic in the layout type, so
            # this reads a real kernarg field, not a folded constant.
            N_bound = Input.shape.to_py_value()[1]
            n_float = N_bound.to(fx.Float32)
        else:
            N_bound = N
            n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # Fast path for complete 128-bit tiles. Unreachable when N is runtime:
        # in_local below is a Python list of registers, which exists only while
        # the tile loop is unrolled, and unrolling needs a compile-time count.
        if const_expr(
            not RUNTIME_N and N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16
        ):
            num_tiles = N // tile_cols
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(
                Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1)
            )

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            gamma_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), weight_elem_bits
            )

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, rstd_div, bid, rrms)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_weight_vec(
                    gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx
                )
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                ),
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

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            if const_expr(RUNTIME_N):
                # Runtime trip count, so this lowers to a real scf.for carrying
                # the partial sum. state[0] is the incoming accumulator and the
                # yield both closes the iteration and binds the loop's result.
                for base_idx_int, state in range(
                    0, N_bound, BLOCK_THREADS, init=[c_zero_f]
                ):
                    idx = tid + fx.Int32(base_idx_int)
                    is_valid = idx < N_bound
                    idx_safe = is_valid.select(idx, 0)
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    x2 = x * x
                    x2_safe = is_valid.select(x2, c_zero_f)
                    thread_sumsq = yield [state[0] + x2_safe]
            else:
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    idx = tid + base_idx_int
                    is_valid = idx < N
                    idx_safe = is_valid.select(idx, 0)
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    x2 = x * x
                    x2_safe = is_valid.select(x2, c_zero_f)
                    thread_sumsq = thread_sumsq + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, rstd_div, bid, rrms)

            if const_expr(RUNTIME_N):
                # Nothing is carried across iterations here -- the stores are the
                # loop's only effect -- but init=[] is still passed so the loop
                # routes to scf_range rather than to iter_args inference, which
                # would try to promote the body's temporaries to loop state.
                for base_idx_int, _state in range(0, N_bound, BLOCK_THREADS, init=[]):
                    idx = tid + fx.Int32(base_idx_int)
                    if idx < N_bound:
                        x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                        g_e = _load_scalar(
                            gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx
                        )
                        x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                        g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                        norm = x * rrms
                        y = norm * g
                        y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                        _store_scalar(copy_atom_s, elem_dtype, out_div, idx, y_e)
            else:
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    idx = tid + base_idx_int
                    if idx < N:
                        x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                        g_e = _load_scalar(
                            gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx
                        )
                        x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                        g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                        norm = x * rrms
                        y = norm * g
                        y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                        _store_scalar(copy_atom_s, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(Input, Gamma, Rstd, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


@flyc.jit
def rmsnorm_direct(
    Input: fx.Tensor,
    Gamma: fx.Tensor,
    Output: fx.Tensor,
    m_in: fx.Int32,
    N: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
    weight_dtype_str: fx.Constexpr[str] = "",
):
    """Specialize the existing RMSNorm factory through JIT Constexpr inputs."""
    resolved_weight_dtype_str = (
        dtype_str if weight_dtype_str == "" else weight_dtype_str
    )
    launch = build_rmsnorm_module(
        N,
        dtype_str,
        BLOCK_THREADS=BLOCK_THREADS,
        weight_dtype_str=resolved_weight_dtype_str,
    )
    launch(Input, Gamma, Output, m_in, stream)


def _build_rmsnorm_large_m_small_n_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_M = max(min(16384 // BLOCK_N, 32), 8)
    THREADS_PER_ROW = min(WARP_SIZE, 1024 // BLOCK_M)
    BLOCK_THREADS_SPECIAL = BLOCK_M * THREADS_PER_ROW
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16

    @flyc.kernel(known_block_size=[BLOCK_THREADS_SPECIAL, 1, 1])
    def rmsnorm_large_m_small_n_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lane = tid % THREADS_PER_ROW
        row_local = tid // THREADS_PER_ROW
        row = bid * BLOCK_M + row_local

        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
            fm_fast = arith.FastMathFlags.fast
            eps_c = eps
            n_float = float(N)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(store_rstd):
                Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
                rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
                rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))

            copy_atom_s = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                ),
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

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def group_reduce_add(x):
                w = x
                for _sh_exp in range_constexpr(int(math.log2(THREADS_PER_ROW))):
                    off = THREADS_PER_ROW // (2 << _sh_exp)
                    peer = w.shuffle_xor(off, THREADS_PER_ROW)
                    w = w.addf(peer, fastmath=fm_fast)
                return w

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = group_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if lane == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, rstd_div, row, rrms)

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(
                        gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx
                    )
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_large_m_small_n(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Rstd, Output, m_in)
            launcher.launch(
                grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                block=(BLOCK_THREADS_SPECIAL, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_large_m_small_n

    @flyc.jit
    def launch_rmsnorm_large_m_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Gamma, Output, m_in)
        launcher.launch(
            grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
            block=(BLOCK_THREADS_SPECIAL, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_large_m_small_n


def build_fused_add_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # Fast path for complete 128-bit tiles.
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(
                row_residual_in, fx.make_layout(VEC_WIDTH, 1)
            )
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(
                row_residual_out, fx.make_layout(VEC_WIDTH, 1)
            )
            gamma_div = fx.logical_divide(
                Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1)
            )

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            gamma_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), weight_elem_bits
            )

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(
                    fx.Float32
                )
                residual = _load_vec(
                    copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx
                ).to(fx.Float32)
                added_e = _to_elem_vec(
                    dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual
                )
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)

                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

                _store_vec(
                    copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx
                )

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, rstd_div, bid, rrms)

            # Pass 2: normalize + gamma + store (reuse cached added values)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                g = _load_weight_vec(
                    gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx
                )
                added = (
                    add_local[tile_i]
                    if dtype_str == "f32"
                    else add_local[tile_i].to(fx.Float32)
                )
                y = (added * rrms) * g
                y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                ),
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

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                residual_e = _load_scalar(
                    copy_atom_s, elem_dtype, residual_in_div, idx_safe
                )
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = (
                    residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                )
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(
                        copy_atom_s, elem_dtype, residual_out_div, idx, added_e
                    )
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, rstd_div, bid, rrms)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = _load_scalar(
                        gamma_copy_atom_s, weight_elem_dtype, gamma_div, idx
                    )
                    added_e = _load_scalar(
                        copy_atom_s, elem_dtype, residual_out_div, idx
                    )
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_fused_add_rmsnorm(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_kernel(
                Input, ResidualIn, Gamma, Output, ResidualOut, Rstd
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm

    @flyc.jit
    def launch_fused_add_rmsnorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_kernel(
            Input, ResidualIn, Gamma, Output, ResidualOut, Gamma
        )
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_rmsnorm


def _build_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = fx.max(w, peer)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # Fast path for complete 128-bit tiles.
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            quant_half_width = VEC_WIDTH // 2
            abs_mask = fx.Vector.filled(VEC_WIDTH, 0x7FFFFFFF, fx.Uint32)
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(
                    fx.Float32
                )
                x = in_local[tile_i].to(fx.Float32)
                y = (x * rrms) * g
                if const_expr(is_smooth):
                    s = _load_vec(
                        copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx
                    ).to(fx.Float32)
                    y = y * s

                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_scalar(scale_copy_atom, fx.Float32, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for tile_i in range_constexpr(num_tiles):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(
                    copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx
                )
                _store_vec(
                    copy_atom_q,
                    quant_half_width,
                    quant_dtype,
                    q_hi,
                    out_div_q,
                    out_idx + 1,
                )

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            copy_atom_s = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                ),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(
                    (
                        fx.rocdl.BufferCopy16b()
                        if elem_bits <= 16
                        else fx.rocdl.BufferCopy32b()
                    ),
                    elem_bits,
                )

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                y = (x * rrms) * g
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = fx.max(
                    thread_row_max, is_valid.select(y_abs, c_zero_f)
                )

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_scalar(scale_copy_atom, fx.Float32, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_rmsnorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_quant_kernel(Input, Gamma, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_quant_kernel(Input, Gamma, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_dynamicquant


def build_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )


def _build_fused_add_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = fx.max(w, peer)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # Fast path for complete 128-bit tiles.
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            quant_half_width = VEC_WIDTH // 2
            abs_mask = fx.Vector.filled(VEC_WIDTH, 0x7FFFFFFF, fx.Uint32)
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(
                row_residual_in, fx.make_layout(VEC_WIDTH, 1)
            )
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(
                row_residual_out, fx.make_layout(VEC_WIDTH, 1)
            )
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(
                    fx.Float32
                )
                residual = _load_vec(
                    copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx
                ).to(fx.Float32)
                added_e = _to_elem_vec(
                    dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual
                )
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2
                _store_vec(
                    copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx
                )

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(
                    fx.Float32
                )
                added = (
                    add_local[tile_i]
                    if dtype_str == "f32"
                    else add_local[tile_i].to(fx.Float32)
                )
                y = (added * rrms) * g
                if const_expr(is_smooth):
                    s = _load_vec(
                        copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx
                    ).to(fx.Float32)
                    y = y * s

                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_scalar(scale_copy_atom, fx.Float32, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for tile_i in range_constexpr(num_tiles):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(
                    copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx
                )
                _store_vec(
                    copy_atom_q,
                    quant_half_width,
                    quant_dtype,
                    q_hi,
                    out_div_q,
                    out_idx + 1,
                )

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            copy_atom_s = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy16b()
                    if elem_bits <= 16
                    else fx.rocdl.BufferCopy32b()
                ),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(
                    (
                        fx.rocdl.BufferCopy16b()
                        if elem_bits <= 16
                        else fx.rocdl.BufferCopy32b()
                    ),
                    elem_bits,
                )

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: add, write residual_out, and accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                residual_e = _load_scalar(
                    copy_atom_s, elem_dtype, residual_in_div, idx_safe
                )
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = (
                    residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                )
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(
                        copy_atom_s, elem_dtype, residual_out_div, idx, added_e
                    )
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                added_e = _load_scalar(
                    copy_atom_s, elem_dtype, residual_out_div, idx_safe
                )
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                y = (added * rrms) * g
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = fx.max(
                    thread_row_max, is_valid.select(y_abs, c_zero_f)
                )

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_scalar(scale_copy_atom, fx.Float32, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    added_e = _load_scalar(
                        copy_atom_s, elem_dtype, residual_out_div, idx
                    )
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_rmsnorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_quant_kernel(
                Input, ResidualIn, Gamma, XScale, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_quant_kernel(
                Input, ResidualIn, Gamma, Gamma, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_dynamicquant


def build_fused_add_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_fused_add_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )
