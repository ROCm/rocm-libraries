# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""FlyDSL dBias reduction for the split backward (gfx11-generic / gfx12-generic): the broadcast-dim sum of the full-shape dB workspace.

The dQ kernel writes `dB = dS` for the full (B, Hq, Sq, Sk) problem, one writer per element (the dQ kernel). When the bias is
broadcast over batch and / or head and that dim is larger than 1, the user's dBias has fewer elements and the sum over the broadcast
dims is this kernel's job (PyTorch + AOTriton do the same: `grad_bias` is full shape and autograd's `expand` backward is
`sum_to`). Bias heads are the query heads (GQA bias is per query head).

    out[bo, ho, q, k] = sum over the reduced dims of ws[b, h, q, k]            (b = bo or all B, h = ho or all Hq)

**Accumulation = PyTorch's.** PyTorch's CUDA/HIP sum reduces a 16-bit tensor with an fp32 accumulator and rounds once to the
tensor dtype (`aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:212-219`, `reduce_dispatch`: `OpFunctor<at::BFloat16, float>` /
`<at::Half, float>`, commit ece869f8d2). This kernel does the same: fp32 add of the 16-bit workspace values, one round to nearest even
into the bias dtype.

**Deterministic.** One thread owns one output element group (8 consecutive columns) and sums its terms itself in a fixed order,
batch-major then head: term t = ib * (reduced head count) + ih. No atomics, no cross-thread combine, no dependence on the launch
geometry, so the result is bitwise identical run to run. The order differs from PyTorch's tree-ish order, so the two agree to fp32
accumulation (one output rounding apart at worst), not bit for bit.

**ABI** (one object per dtype; every size and stride is a launch value):
    DB            full-shape workspace in the bias dtype, (B, Hq, Sq, Sk) with strides stride_ws_{batch,head,seq} (elements), unit column stride
    DBias         the user's dBias in the bias dtype, (B or 1, Hq or 1, Sq, Sk) with strides stride_out_{batch,head,seq}, unit column stride
    num_batch, num_head, seqlen_q, seqlen_k   the problem's B, Hq, Sq, Sk
    reduce_batch, reduce_head                  0 / 1: sum over that dim (DBias has extent 1 there) or copy it through
    Both pointers are 16-byte aligned (the ABI convention of every kernel pointer in this package; `variant.alignment_for`).
Reducing no dim is a strided copy; the caller skips the launch (and the workspace) in that case.

**Layout and loads.** Grid = (cdiv(Sq * cdiv(Sk, 8), 256), planes), planes = output (batch, head) planes. A thread owns the 8-column
vector `i % cdiv(Sk, 8)` of row `i // cdiv(Sk, 8)`, consecutive threads taking consecutive vectors, so a wave32 reads 512 B contiguous
per term. When Sk is not a multiple of 8, or any stride is not a multiple of 8 elements, the vector is not 16-byte aligned or not
entirely inside the row: those threads take the element-wise path (2-byte loads and stores, masked at Sk). The path is chosen per
thread from the launch values; both produce the same sums.

The gfx950 `fmha_bwd_dbias_reduce_gfx950.py` with the same ABI and the same fixed-order sum. The kernel has no cross-lane
operation, no LDS and no wave-size constant, so wave32 changes nothing but the contiguous-read width above; the 256-thread
block is 8 waves. Derived from `fmha_bwd_dbias_reduce_gfx950.py`.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_dbias_reduce_gfx11"
BLOCK_THREADS = 256
VEC = 8  # 16-bit elements per 128-bit access

# IEEE adds, no fast-math: the sum is the contract.
_COMPILE_HINTS = {}


def dbias_reduce_arg_values(
    ws, out, B, Hq, Sq, Sk, reduce_batch, reduce_head, ws_strides, out_strides
):
    """Launch values keyed by kernel argument name. `ws` / `out` are device addresses; strides are (batch, head, seq) in elements."""
    v = dict(
        DB=ws,
        DBias=out,
        num_batch=B,
        num_head=Hq,
        seqlen_q=Sq,
        seqlen_k=Sk,
        reduce_batch=int(bool(reduce_batch)),
        reduce_head=int(bool(reduce_head)),
    )
    for a, x in zip(("batch", "head", "seq"), ws_strides):
        v[f"stride_ws_{a}"] = x
    for a, x in zip(("batch", "head", "seq"), out_strides):
        v[f"stride_out_{a}"] = x
    return v


def _carried(values):
    """The one value an `scf.for` with a single carried value hands back (it is unwrapped, not a one-element list)."""
    if const_expr(isinstance(values, (list, tuple))):
        return values[0]
    return values


def _elem(dtype_str):
    return {"f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]


def build_fmha_bwd_dbias_reduce_gfx11_module_primary(dtype_str):
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(dtype_str)
    elem = _elem(dtype_str)
    _cache_tag = (KERNEL_NAME, dtype_str)

    def _ptr_ty(ty, align):
        return fx.PointerType.get(
            elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_dbias_reduce_gfx11_kernel(
        DB: fx.Pointer,
        DBias: fx.Pointer,
        num_batch: fx.Int32,
        num_head: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        reduce_batch: fx.Int32,
        reduce_head: fx.Int32,
        stride_ws_batch: fx.Int64,
        stride_ws_head: fx.Int64,
        stride_ws_seq: fx.Int64,
        stride_out_batch: fx.Int64,
        stride_out_head: fx.Int64,
        stride_out_seq: fx.Int64,
    ):
        tid = fx.Index(fx.thread_idx.x)
        plane = fx.Index(fx.block_idx.y)

        # output plane -> (bo, ho); the reduced dim contributes extent 1 to the plane count and all of its extent to the terms
        out_heads = (reduce_head != fx.Int32(0)).select(fx.Int32(1), num_head)
        n_batch_t = (reduce_batch != fx.Int32(0)).select(num_batch, fx.Int32(1))
        n_head_t = (reduce_head != fx.Int32(0)).select(num_head, fx.Int32(1))
        bo = plane // fx.Index(out_heads)
        ho = plane - bo * fx.Index(out_heads)
        b0 = (reduce_batch != fx.Int32(0)).select(fx.Index(0), bo)
        h0 = (reduce_head != fx.Int32(0)).select(fx.Index(0), ho)

        nvec = (fx.Index(seqlen_k) + fx.Index(VEC - 1)) // fx.Index(VEC)
        total = fx.Index(seqlen_q) * nvec
        i = fx.Index(fx.block_idx.x) * fx.Index(BLOCK_THREADS) + tid
        valid = i < total
        # an out-of-range thread works on vector 0 of row 0 (always in bounds) and stores nothing
        i_c = valid.select(i, fx.Index(0))
        q = i_c // nvec
        col0 = (i_c - q * nvec) * fx.Index(VEC)

        ws_row = q * fx.Index(stride_ws_seq) + col0
        out_row = (
            (b0 * fx.Index(stride_out_batch) + h0 * fx.Index(stride_out_head))
            + q * fx.Index(stride_out_seq)
            + col0
        )
        ws_term0 = (
            b0 * fx.Index(stride_ws_batch) + h0 * fx.Index(stride_ws_head) + ws_row
        )

        ws_p = fx.recast_iter(_ptr_ty(elem, 16), DB)
        out_p = fx.recast_iter(_ptr_ty(elem, 16), DBias)
        ws_ep = fx.recast_iter(_ptr_ty(elem, 2), DB)
        out_ep = fx.recast_iter(_ptr_ty(elem, 2), DBias)

        strides_al = (
            (
                stride_ws_batch
                | stride_ws_head
                | stride_ws_seq
                | stride_out_batch
                | stride_out_head
                | stride_out_seq
            )
            & fx.Int64(VEC - 1)
        ) == fx.Int64(0)
        use_vec = strides_al & ((col0 + fx.Index(VEC)) <= fx.Index(seqlen_k))

        n_batch_i = fx.Index(n_batch_t)
        n_head_i = fx.Index(n_head_t)
        ws_b = fx.Index(stride_ws_batch)
        ws_h = fx.Index(stride_ws_head)

        if use_vec:
            vt = Vec.make_type(VEC, elem)
            init_args = [Vec.filled(VEC, 0.0, fx.Float32).ir_value()]
            loop_results = init_args
            for ib, b_args in range(
                fx.Index(0), n_batch_i, fx.Index(1), init=init_args
            ):
                for ih, h_args in range(
                    fx.Index(0), n_head_i, fx.Index(1), init=b_args
                ):
                    acc = _carried(h_args)
                    off = ws_term0 + ib * ws_b + ih * ws_h
                    acc = acc + fx.ptr_load(ws_p + fx.Int64(off), vt).to(fx.Float32)
                    h_results = yield [acc]
                b_results = yield h_results
            res = _carried(b_results)
            if valid:
                fx.ptr_store(res.to(elem), out_p + fx.Int64(out_row))
        else:
            # element-wise: 8 independent fp32 accumulators, columns masked at Sk
            sk = fx.Index(seqlen_k)
            init_args = [fx.Float32(0.0) for _ in range_constexpr(VEC)]
            for ib, b_args in range(
                fx.Index(0), n_batch_i, fx.Index(1), init=init_args
            ):
                for ih, h_args in range(
                    fx.Index(0), n_head_i, fx.Index(1), init=b_args
                ):
                    base = ws_term0 - col0 + ib * ws_b + ih * ws_h
                    nxt = []
                    for j in range_constexpr(VEC):
                        ok = (col0 + fx.Index(j)) < sk
                        cc = ok.select(col0 + fx.Index(j), fx.Index(0))
                        nxt.append(
                            h_args[j]
                            + fx.ptr_load(ws_ep + fx.Int64(base + cc), elem).to(
                                fx.Float32
                            )
                        )
                    h_results = yield nxt
                b_results = yield h_results
            for j in range_constexpr(VEC):
                if valid & ((col0 + fx.Index(j)) < sk):
                    fx.ptr_store(
                        fx.Float32(b_results[j]).to(elem),
                        out_ep + fx.Int64(out_row + fx.Index(j)),
                    )

    @flyc.jit
    def launch_fmha_bwd_dbias_reduce_gfx950(
        DB: fx.Pointer,
        DBias: fx.Pointer,
        num_batch: fx.Int32,
        num_head: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        reduce_batch: fx.Int32,
        reduce_head: fx.Int32,
        stride_ws_batch: fx.Int64,
        stride_ws_head: fx.Int64,
        stride_ws_seq: fx.Int64,
        stride_out_batch: fx.Int64,
        stride_out_head: fx.Int64,
        stride_out_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        nvec = (fx.Index(seqlen_k) + fx.Index(VEC - 1)) // fx.Index(VEC)
        nblk = (fx.Index(seqlen_q) * nvec + fx.Index(BLOCK_THREADS - 1)) // fx.Index(
            BLOCK_THREADS
        )
        out_b = (reduce_batch != fx.Int32(0)).select(fx.Int32(1), num_batch)
        out_h = (reduce_head != fx.Int32(0)).select(fx.Int32(1), num_head)
        planes = fx.Index(out_b) * fx.Index(out_h)
        fmha_bwd_dbias_reduce_gfx11_kernel(
            DB,
            DBias,
            num_batch,
            num_head,
            seqlen_q,
            seqlen_k,
            reduce_batch,
            reduce_head,
            stride_ws_batch,
            stride_ws_head,
            stride_ws_seq,
            stride_out_batch,
            stride_out_head,
            stride_out_seq,
        ).launch(grid=(nblk, planes, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_dbias_reduce_gfx950, *args, **kwargs)

    return _launch
