# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA dropout keep mask, written out: the graph's ``rng_dump``.

Every FlyDSL attention kernel that drops (RDNA prefill, aiw, CDNA) keeps or drops key ``j`` of
row ``i`` of a ``(batch b, query head h)`` plane on word ``j % 4`` of the Philox4x32-10 draw at
counter ``offset1 + offset2 + ((b * Hq + h) * Sq + i) * cdiv(Skv, 4) + j / 4``, keyed by the
64-bit seed, kept iff the word read as an int32 exceeds the threshold (AOTriton >= 0.15's
scheme). ``Sq`` and ``Skv`` are the launch's extents, so the mask is a function of
``(seed, offset, B, Hq, Sq, Skv, threshold)`` alone: no tile, wave size or per-batch length
enters it. This kernel draws the same counters through the same ``philox.py``
(``Philox.grid_plane`` / ``grid_offset``, the helpers that exist so a producer and a checker of
one mask cannot disagree about which counter an element gets), and writes FLOAT
``[B, Hq, Sq, Skv]`` packed row-major: 1.0 kept, 0.0 dropped.

It writes the raw keep bit at **every** element, masked keys, padded rows and keys past a
batch's length included: those are the draws the attention kernel makes and discards, and the
reference harness counts the whole buffer when it checks the keep rate.

One thread per counter: thread ``t`` of ``num_planes * Sq * cdiv(Skv, 4)`` makes one draw and
stores the up to four keys it covers, as four f32 stores (a row's base is not 16-byte aligned
unless ``Skv`` is a multiple of 4). The grid is ``cdiv(threads, BLOCK_THREADS)`` workgroups; the
native launches it on the attention launch's stream, after it, only when the graph names an RNG
dump, so the attention objects are untouched.

Arguments, in order: ``MASK`` (the dump), ``SEED`` and ``OFFSET1`` (pointers to 8 bytes; a null
``OFFSET1`` reads as 0, as the attention kernels read it), ``offset2`` (Int64), ``threshold``
(Int32), ``num_planes`` (B * Hq), ``seq_len_q``, ``seq_len_kv`` (Int32), then the stream.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from kernels.attention_rdna4.philox import Philox, keep_mask

KERNEL_NAME = "sdpa_dropout_mask_kernel"


def _load_u64_or_zero(ptr):
    """``*ptr`` as an i64, or 0 when ``ptr`` is null: a branch, so a null pointer is never
    loaded. The attention kernels read the seed and AOTriton's offset1 this way."""
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    if_op = _scf.IfOp(fx.as_ir_value(nonnull), results_=[T.i64], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        i64_ptr = fx.recast_iter(
            fx.PointerType.get(
                elem_ty=fx.Int64.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=8,
            ),
            ptr,
        )
        _scf.YieldOp([fx.as_ir_value(fx.ptr_load(i64_ptr, fx.Int64))])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(fx.Int64(0))])
    return fx.Int64(if_op.results[0])


def build_sdpa_dropout_mask_module(block_threads=256):
    """Build the mask writer's launcher (``@flyc.jit``); its arguments are the module's."""
    BLOCK_THREADS = int(block_threads)
    PHILOX = Philox(width=32)
    RN = PHILOX.randoms_per_offset

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def sdpa_dropout_mask_kernel(
        MASK: fx.Pointer,
        SEED: fx.Pointer,
        OFFSET1: fx.Pointer,
        offset2: fx.Int64,
        threshold: fx.Int32,
        num_planes: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
    ):
        groups = fx.Int64((fx.Int32(seq_len_kv) + fx.Int32(RN - 1)) // fx.Int32(RN))
        rows = fx.Int64(num_planes) * fx.Int64(seq_len_q)
        t = fx.Int64(gpu.block_idx.x) * fx.Int64(BLOCK_THREADS) + fx.Int64(
            gpu.thread_idx.x
        )
        if t < rows * groups:
            row = t // groups
            group = t - row * groups
            plane = row // fx.Int64(seq_len_q)
            q_row = row - plane * fx.Int64(seq_len_q)
            seed = _load_u64_or_zero(SEED)
            offset = fx.Int64(offset2) + _load_u64_or_zero(OFFSET1)
            plane_base, row_stride = PHILOX.grid_plane(
                offset, fx.Int32(plane), seq_len_q, seq_len_kv
            )
            first_col = group * fx.Int64(RN)
            counter = PHILOX.grid_offset(plane_base, row_stride, q_row, first_col)
            keep = keep_mask(PHILOX.u32(seed, counter), threshold)
            mask_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, MASK.address_space), MASK
            )
            base = row * fx.Int64(seq_len_kv) + first_col
            for r in range_constexpr(RN):
                if first_col + fx.Int64(r) < fx.Int64(seq_len_kv):
                    value = keep[r].select(fx.Float32(1.0), fx.Float32(0.0))
                    view = fx.make_view(
                        fx.add_offset(mask_ptr, base + fx.Int64(r)),
                        fx.make_layout(1, 1),
                    )
                    view.store(Vec.from_elements([value], fx.Float32))

    @flyc.jit
    def launch_sdpa_dropout_mask(
        MASK: fx.Pointer,
        SEED: fx.Pointer,
        OFFSET1: fx.Pointer,
        offset2: fx.Int64,
        threshold: fx.Int32,
        num_planes: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: evaluated once on purpose
            None
        ),
    ):
        groups = (fx.Uint64(seq_len_kv) + RN - 1) // RN
        threads = fx.Uint64(num_planes) * fx.Uint64(seq_len_q) * groups
        grid_x = (threads + BLOCK_THREADS - 1) // BLOCK_THREADS
        launcher = sdpa_dropout_mask_kernel(
            MASK, SEED, OFFSET1, offset2, threshold, num_planes, seq_len_q, seq_len_kv
        )
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    launch_sdpa_dropout_mask.block_size = BLOCK_THREADS
    return launch_sdpa_dropout_mask
