# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
#   compiler: flydsl 0.3.4 (pip wheel)   toolchain: ROCm 7.13.0
#
# Upstream is the RDNA4 (gfx1201) forward kernel. This copy targets RDNA3.5
# (gfx1151) and widens what one compiled object serves.
#
# Modifications:
#   1. Ported to the gfx11 WMMA ABI. RDNA4 passes 8 K-values per lane for A
#      and B (lanes 16-31 carry the upper K half) and lays accumulator element
#      r of lane l at row r + 8*(l/16). gfx11 passes all 16 K-values per lane
#      (lanes 16-31 duplicate lanes 0-15) and lays element r at row
#      2r + (l/16). So:
#        - Q, K and V^T operands are loaded 16 wide, at the K-step origin.
#        - P^T can no longer be fed straight from the S accumulator: a lane
#          holds only the even or only the odd kv rows of its tile, and its
#          xor-16 peer holds the rest. Each lane trades its 8 values with that
#          peer and interleaves them into the v16 operand.
#        - The causal mask indexes kv as 2r + (l/16).
#        - The O store trades with the peer once at the end so each lane
#          still writes 8 contiguous head-dim elements.
#      The online softmax is unchanged: each lane still owns one query row,
#      and its row reduction is still its own 8 values plus its xor-16 peer.
#   2. num_heads is a runtime kernel argument rather than a baked one. It was
#      used only for the token stride and the grid decomposition, and baking it
#      meant one object per model head count.
#   3. GQA/MQA: a runtime kv_group (query heads per kv head) maps a query head
#      to its K head, and a runtime v_group (appended last) to its V head, so K
#      and V may have different head counts. Upstream is MHA only.
#   4. Q, K, V and O each take runtime batch / sequence / head strides, in
#      elements. Upstream assumed packed BSHD. Only the head-dim stride must be
#      1, so BSHD, BHSD and packed-QKV views all run on one object.
#   5. Separate seq_len_q and seq_len_kv, and the reference's two-sided mask
#      as runtime arguments (CpuFpReferenceSdpa isMasked): kv is masked when
#      kv > q + align + right_bound (the causal variant) or
#      kv < q + align - left_bound (left_bound >= 0), where align is 0
#      top-left and seq_len_kv - seq_len_q bottom-right (align_br). So one
#      causal object serves both corners and any band to the right of the
#      diagonal, and both objects serve sliding windows; a window also lifts
#      the KV loop's start. align is computed in the kernel from the lengths,
#      so per-batch lengths can later move it without an argument-list change.
#      Each score is masked by one unsigned range compare. Upstream was
#      self-attention only (seq_len_q == seq_len_kv), causal top-left.
#   6. Any seq_len_kv is correct. K is now read through a bounds-checked
#      buffer descriptor like V, scoped to one (batch, kv head) slice, and
#      every score with kv >= seq_len_kv is masked. Upstream read K unguarded
#      and required callers to pad the sequence to the tile, which
#      mis-normalises a non-causal softmax (padded keys contribute exp(0) to
#      its denominator).
#   7. attn_scale is a runtime float rather than a baked 1/sqrt(head_dim).
#   8. Dropped the no-nans / unsafe-fp-math function attributes and the
#      fast/unsafe compile hints. Masking now produces -inf scores on every
#      tail tile, and a fully masked row is guarded against -inf - -inf; both
#      are the inputs those flags license the compiler to assume away. DAZ
#      (denormal-fp-math-f32) is kept.
#   9. Removed the torch launch wrapper (_launch / _compile / _wrap_qkvo) and
#      its tensor_shim import. This copy is imported by an AOT generator and a
#      validation script, which drive the jit launcher directly.
#  10. Softmax statistics: an optional LSE output (natural log of the scaled
#      scores, f32, [B, H, Sq, 1] with runtime strides), written when lse_on
#      (bit 0 of the flags argument, modification 26).
#      A row no key reaches gets O = 0 and LSE = -inf, as the reference
#      defines it, rather than 0 * inf.
#  11. Waves whose 16 query rows all lie past seq_len_q -- 7 of 8 in a decode
#      step -- skip the matrix work (they still join the cooperative loads and
#      barriers).
#  12. P is narrowed to bf16 by round-to-nearest rather than truncation
#      (truncate_p=False by default), which halves the bf16 output error.
#  13. head_dim above 128 (256): Q is re-read per K-step, one step ahead and
#      fenced by a scheduling barrier, instead of held for the whole KV loop
#      (q_resident), and the output columns are split across DV_SPLIT
#      workgroups that each recompute QK^T (dv_split). Together they bring the
#      register demand under 256 VGPRs; 128 and below are built as before.
#  14. Additive bias (has_bias): an f32 tensor (or see 19), broadcast through its (batch,
#      head, query, key) element strides -- 0 on a broadcast axis -- added to
#      the scaled scores before the mask, as the reference does. Its pointer
#      and strides are appended after the LSE strides, so the arguments before
#      them keep their offsets; objects without bias ignore them and compile to
#      the same code. At head_dim 128 a bias object re-reads Q per K-step.
#  15. A runtime head_dim (generic_head_dim): head_dim is then the largest
#      served, and the actual one -- any multiple of 8 up to it -- is a last,
#      appended argument. Q and K columns at or past it are zeroed as they are
#      loaded (both: 0 * NaN read from a neighbouring row is still NaN), the
#      K/V slice and every Q read stop at it, and O columns past it are not
#      stored. Objects built without it ignore the argument.
#  16. K/V tiles of any width: the cooperative load covers a tile in as many
#      passes of the block's threads as it takes, rounding up, and the last
#      pass stores only the rows inside the tile. Upstream rounds down, which
#      leaves rows unloaded when head_dim / 16 does not divide the block's
#      threads into whole rows (160, 192, 224); every width it does divide
#      builds as before.
#  17. Both WMMA ABIs, chosen by the arch FlyDSL compiles for: gfx11 as in
#      modification 1, and gfx12 (RDNA4) as upstream has it -- Q, K and V^T
#      loaded 8 wide at K offset 8*(l/16), P^T fed straight from the S
#      accumulator, kv indexed as r + 8*(l/16) in the mask and the bias, and O
#      stored with no trade. Every other modification applies to both.
#  18. Head-dim pairs (hd_pair): two adjacent waves share each 16 query rows.
#      Each computes QK^T over its half of head_dim, publishes its partial
#      scores to LDS behind the barrier that already publishes V (which moves
#      ahead of the softmax), adds its partner's, runs the same softmax, and
#      accumulates half of O's columns. QK^T is then computed once per row
#      rather than once per output-column workgroup (dv_split), at the same
#      thread count when block_m halves. The pair index is read from the first
#      lane so the column offsets it feeds stay scalar. Above 128 this replaces
#      most of dv_split's recomputation; objects without it build as before.
#  19. A bias in the query's dtype (bias_query_dtype): the bias pointer is
#      read as bf16/f16 rather than f32, its slice offsets halve, and each
#      value is widened to f32 before it is added, so the scores see the same
#      arithmetic as an f32 bias. Objects without it build as before.
#  20. A V head_dim of its own (head_dim_v, MLA): Q and K keep head_dim, while
#      V, O and the PV product are head_dim_v wide (at most head_dim). V tiles
#      are staged with K's load mapping; the PV product, the output-column
#      split and the O store cover only head_dim_v columns, and the V slice
#      ends at them, so columns past it are never consumed. Objects without
#      it build as before.
#  21. Attention sinks (has_sink): one logit per query head, in the query's
#      dtype, read through its head stride. It joins each row's softmax as an
#      extra key with no value: the row's sum gains exp(sink - max) and its
#      LSE the same term, and a row no key reaches gets O = 0 and LSE = sink.
#      The pointer and stride are appended last; objects without it ignore
#      them and build as before.
#  22. Rows with no reachable key under extreme masks: a row whose window starts
#      past its end (top-left, seq_len_q > seq_len_kv + left_bound) gets an
#      empty span rather than a wrapped one that kept every key, and a bias
#      object exponentiates (s - m) * log2(e), which stays finite when a bias
#      near -FLT_MAX masks the whole row (s * log2(e) - m * log2(e) overflowed
#      to inf - inf).
#  23. A sink of -inf on a row no key reaches: its max is floored at -FLT_MAX and
#      1/l at 0 when l is 0, so O = 0 and LSE = -inf rather than NaN (sink objects
#      only).
#  24. A row whose softmax sum is NaN (NaN in Q, or NaN/inf in a key it sees) gets
#      O = NaN and LSE = NaN, as PyTorch's logsumexp gives: the keyless test is
#      l != 0 (unordered, true for NaN) rather than l > 0, which sent NaN rows to
#      the keyless LSE of -inf. A row no key reaches still has l = 0 exactly. The
#      decode kernel beside this one (flash_attn_decode_rdna.py) does the same,
#      in its main kernel and in its split merge.
#  25. Decode kernel only (flash_attn_decode_rdna.py, not this file's code): its
#      bias objects issue the tile's bias loads before the QK product rather than
#      after it, where the scores waited on them with nothing left to overlap.
#      This was already so at head_dim 128 and up; it now holds at 64 and on the
#      generic tiers too (up to 20% faster with a bias on long caches), and only
#      the specialized head_dim 96 objects keep the late loads, which measured a
#      fraction of a percent faster there on short caches. Objects without a
#      bias, and every bias object that already loaded early, build as before.
#  26. Per-batch sequence lengths (padding), in this kernel and the decode kernel
#      (with its split merge). Two appended pointers, SEQLEN_Q and SEQLEN_KV, each
#      an int32 per batch; read once per workgroup after the batch index under a
#      uniform branch on bit 1 of the flags argument (formerly lse_on, still its
#      bit 0), so a dense launch loads nothing, they replace seq_len_q /
#      seq_len_kv for everything past the
#      grid: the K/V slice extent (so no neighbouring batch is read), the
#      diagonal's offset (bottom-right per batch, as modification 5 foresaw), the
#      key span and the KV loop's end, the query bounds and the stores. A query
#      tile wholly past its batch's length runs no KV iteration and writes
#      nothing; rows past it are not written. Also appended, for later changes so
#      the argument list moves once: ROW_OFF_Q / ROW_OFF_KV (packed sequences),
#      and the prefill kernel's dropout arguments (SEED, OFFSET1 pointers, offset2
#      i64, idropout_p i32, dropout_scale f32) -- read by no object yet.
#  27. Dropout (has_dropout), AOTriton's scheme bit for bit (philox.py, imported
#      from the aiw kernel's directory): each score of row i, key j of a (batch,
#      query head) plane keeps or drops on word j % 4 of the Philox4x32-10 draw at
#      counter offset1 + offset2 + ((b * Hq + h) * Sq + i) * cdiv(Skv, 4) + j / 4,
#      with Sq and Skv the launch's lengths, against the f32 threshold the host
#      passes (idropout_p). Applied to P after the row sum is formed, so l and the
#      LSE stay undropped; 1 / (1 - p) folds into 1 / l. On gfx11 a lane holds the
#      even or the odd keys of 16, so each lane draws the 8 consecutive keys of its
#      half (two calls) and trades the packed keep bits with its xor-16 peer: one
#      shuffle per 16 keys rather than four calls per lane. Objects without it
#      build as before.
#  28. Padded rows filled, and lengths clamped, in this kernel and the decode kernel
#      (with its split merge). With bit 2 of the flags argument, a row past its
#      batch's query length but inside the launch's is written as O = 0 and LSE =
#      -inf, hipDNN's reference contract (cuDNN >= 9.14), on sink objects too (a
#      padded row is not a keyless row): a store of constants beside the live rows'
#      store, so the live path is unchanged. Without the bit such rows keep what the
#      buffer held (AOTriton's behaviour). The per-batch lengths are clamped to
#      [0, the launch's length] where they are read, as hipDNN's GPU reference clamps
#      them, so a dense launch pays neither.
#  29. Packed (THD) sequences, in this kernel only (the decode kernel and its merge
#      decline them). With bit 3 of the flags argument, batch b's tokens are rows
#      ROW_OFF_Q[b] .. ROW_OFF_Q[b + 1] (and ROW_OFF_KV for K and V) of tensors
#      whose batch stride the native passes as their sequence stride: the operands'
#      batch bases, and the LSE's, are the first token's row times that stride. The
#      batch's query and key counts are the tables' differences, clamped to [0, the
#      launch's lengths]; with bit 1 as well, the per-batch lengths are read too and
#      clamped within them (cuDNN's packed form, each table entry a slot whose rows
#      past the length are padded rows). Q's loads are bounded by the rows the batch
#      owns (its slot packed, the launch's length otherwise), so no load reaches the
#      next sequence or past the last; the K/V slices already stop at the batch's key
#      count. The lengths read moves into one branch on the flags word (load_sequence;
#      load_lengths stays the decode kernel's), which yields the batch index as the row
#      for a dense launch, so a dense launch still loads nothing.
#  30. (The decode kernel's generic tier up to 512: flash_attn_decode_rdna.py.)
#  31. An f32 sink, in this kernel and the decode kernel and its merge: cuDNN and
#      hipDNN's reference take the sink as FLOAT. Bit 32 of sink_stride (SINK_F32_BIT)
#      says the sink is f32, clear it is in the query's dtype as before; the low 32
#      bits stay the head stride in elements. load_sink_logit reads either without a
#      branch, as two 16-bit loads at the element (the same one twice for a 16-bit
#      sink), and widens it exactly, so a query-dtype sink gives the same logit as
#      before and an f32 one its full value. Sink objects only: the argument list is
#      unchanged, and objects without a sink build as before.
#
# Upstream-first: prefer landing these changes in AITER/FlyDSL; this copy exists
# so the provider is not blocked on that.

"""Flash Attention forward kernel for gfx11 (RDNA3 / RDNA3.5), wave32 WMMA.

Uses 16x16x16 WMMA, online softmax and pipelined V loads. Requires a baked
``head_dim >= 64`` with ``head_dim % 32 == 0`` (a generic object serves any multiple
of 8 up to it), and a ``head_dim_v`` (V and O, default ``head_dim``) at most
``head_dim``, a multiple of 16.

Baked per object: ``head_dim``, ``head_dim_v``, ``causal`` (whether a right bound is applied),
``dtype``, tile sizes, whether an additive bias or an attention sink is applied, the
output-column split (the grid is
``batch * ceil(seq_len_q / block_m) * num_heads * dv_split``), and whether
waves work in head-dim pairs (``block_m / 16 * (2 if hd_pair else 1)`` waves
of 32). Runtime: batch, both sequence lengths, head counts,
strides, scale, mask bounds and alignment, and the optional LSE output.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from flydsl.runtime.device import get_rocm_arch
from kernels.attention_rdna4.philox import Philox
from kernels.common.kernels_common import LOG2E as _LOG2E

KERNEL_NAME = "flash_attn_func_rdna_kernel"

# Stands for "no lower edge": a window start far enough below any row that
# q + _NO_WINDOW never reaches a real kv column, yet small enough that
# kv - (q + _NO_WINDOW) cannot overflow int32.
_NO_WINDOW = -(1 << 30)


def load_lengths(flags, pointers, index, defaults):
    """``ptr[index]`` as an i32 for each of ``pointers`` when bit 1 of ``flags`` says the
    launch passes per-batch lengths, else ``defaults`` (modification 26).

    A real branch, on a flag the kernel reads with its first arguments (bit 1 of the
    ``flags`` word whose bit 0 is ``lse_on``), so a dense launch loads nothing and waits on
    nothing. Two forms measured slower on short launches: a branch on the pointers
    themselves made the kernel wait for them before issuing its other argument loads, and a
    branchless form (an address that falls back to a pointer the launch always passes) read
    a cold cache line on every dense launch. Built as an explicit ``scf.if`` because the
    ``if`` rewrite is lexical per kernel function and the decode kernel shares this helper.
    """
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    on = (fx.Int32(flags) & fx.Int32(2)) != fx.Int32(0)
    if_op = _scf.IfOp(
        fx.as_ir_value(on), results_=[T.i32] * len(pointers), has_else=True
    )
    with ir.InsertionPoint(if_op.then_block):
        values = []
        for ptr, default in zip(pointers, defaults):
            i32_ptr = fx.recast_iter(
                fx.PointerType.get(
                    elem_ty=fx.Int32.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=4,
                ),
                ptr,
            )
            # Clamped to [0, the launch's length] (modification 28), as hipDNN's GPU
            # reference clamps them: a length past it would read past the batch's slice.
            length = fx.Int32(fx.ptr_load(i32_ptr + fx.Int64(index), fx.Int32))
            length = (length < fx.Int32(0)).select(fx.Int32(0), length)
            length = (length > fx.Int32(default)).select(fx.Int32(default), length)
            values.append(fx.as_ir_value(length))
        _scf.YieldOp(values)
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(fx.Int32(d)) for d in defaults])
    return [fx.Int32(r) for r in if_op.results]


def _i32_at(ptr, index):
    """``ptr[index]`` as an i32 (an int32 table in global memory)."""
    i32_ptr = fx.recast_iter(
        fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4,
        ),
        ptr,
    )
    return fx.Int32(fx.ptr_load(i32_ptr + fx.Int64(index), fx.Int32))


def _clamp_i32(value, high):
    """``value`` clamped to [0, ``high``]."""
    value = (value < fx.Int32(0)).select(fx.Int32(0), value)
    return fx.Int32((value > fx.Int32(high)).select(fx.Int32(high), value))


def load_sequence(flags, batch, lengths, seq_len_pointers, row_off_pointers):
    """This batch's sequence (modification 29): ``(len_q, len_kv, slot_q, row_q, row_kv)``.

    ``len_*`` are the rows and keys the batch computes with, ``slot_q`` the query rows it
    owns in Q and O (rows in [len_q, slot_q) are padded rows), and ``row_*`` the first
    token's index: each operand's batch base is ``row * batch_stride``. A dense launch
    gets the launch's lengths, slot ``len_q`` and row ``batch``. Bit 1 of ``flags``: the
    lengths come from the per-batch arrays (padding, modification 26). Bit 3: the
    sequences are packed (THD): batch b's tokens are rows ``off[b] .. off[b + 1]`` of
    the ragged-offset tables, the native passes each batch stride as the operand's sequence
    stride, and the slot is the table's length. With both, the length is the array's within
    the slot (cuDNN's packed form with per-batch lengths). Every length is clamped to [0,
    the launch's] (modification 28) and to the slot. One branch on the flags word, so a
    dense launch loads nothing.
    """
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    seq_len_q, seq_len_kv = lengths
    batch_i64 = fx.Int64(batch)
    results = [T.i32, T.i32, T.i32, T.i64, T.i64]
    on = (fx.Int32(flags) & fx.Int32(10)) != fx.Int32(0)
    if_op = _scf.IfOp(fx.as_ir_value(on), results_=results, has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        packed = (fx.Int32(flags) & fx.Int32(8)) != fx.Int32(0)
        slot_op = _scf.IfOp(fx.as_ir_value(packed), results_=results[1:], has_else=True)
        with ir.InsertionPoint(slot_op.then_block):
            slot = []
            rows = []
            for ptr, high in zip(row_off_pointers, lengths):
                first = _i32_at(ptr, batch_i64)
                last = _i32_at(ptr, batch_i64 + fx.Int64(1))
                slot.append(fx.as_ir_value(_clamp_i32(last - first, high)))
                rows.append(fx.as_ir_value(fx.Int64(first)))
            _scf.YieldOp(slot + rows)
        with ir.InsertionPoint(slot_op.else_block):
            _scf.YieldOp(
                [
                    fx.as_ir_value(fx.Int32(seq_len_q)),
                    fx.as_ir_value(fx.Int32(seq_len_kv)),
                ]
                + [fx.as_ir_value(batch_i64)] * 2
            )
        slot_q, slot_kv = (fx.Int32(r) for r in slot_op.results[:2])
        row_q, row_kv = (fx.Int64(r) for r in slot_op.results[2:])
        padded = (fx.Int32(flags) & fx.Int32(2)) != fx.Int32(0)
        len_op = _scf.IfOp(fx.as_ir_value(padded), results_=[T.i32] * 2, has_else=True)
        with ir.InsertionPoint(len_op.then_block):
            _scf.YieldOp(
                [
                    fx.as_ir_value(_clamp_i32(_i32_at(ptr, batch_i64), high))
                    for ptr, high in zip(seq_len_pointers, (slot_q, slot_kv))
                ]
            )
        with ir.InsertionPoint(len_op.else_block):
            _scf.YieldOp([fx.as_ir_value(slot_q), fx.as_ir_value(slot_kv)])
        len_q, len_kv = (fx.Int32(r) for r in len_op.results)
        _scf.YieldOp(
            [fx.as_ir_value(v) for v in (len_q, len_kv, slot_q, row_q, row_kv)]
        )
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp(
            [
                fx.as_ir_value(fx.Int32(seq_len_q)),
                fx.as_ir_value(fx.Int32(seq_len_kv)),
                fx.as_ir_value(fx.Int32(seq_len_q)),
                fx.as_ir_value(batch_i64),
                fx.as_ir_value(batch_i64),
            ]
        )
    r = if_op.results
    return (
        fx.Int32(r[0]),
        fx.Int32(r[1]),
        fx.Int32(r[2]),
        fx.Int64(r[3]),
        fx.Int64(r[4]),
    )


def load_u64_or_zero(ptr):
    """``*ptr`` as an i64, or 0 when ``ptr`` is null (modification 27): a branch, so a null
    pointer is never loaded. The dropout seed and AOTriton's offset1 pointer, read once per
    workgroup by a dropout object only."""
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


# The sink_stride argument's bit that marks an f32 sink (modification 31): set, the sink is
# f32 (cuDNN's and hipDNN's reference's dtype); clear, it is in the query's dtype. The low 32
# bits are the head stride in elements.
SINK_F32_BIT = 32


def load_sink_logit(sink, sink_stride, head, elem_dtype):
    """Query head ``head``'s attention-sink logit, as f32 (modification 31).

    ``sink_stride`` packs the head stride in elements (bits 0-31) and the sink's dtype
    (``SINK_F32_BIT``: f32, else the query's). The logit is read as two 16-bit halves at
    16-bit index ``head * stride`` scaled to the dtype's width: an f32 sink's low and high
    halves, a 16-bit sink's one element twice. Neither load reaches past the element, so a
    16-bit sink is never over-read as f32, and there is no branch. Both widen exactly.
    """
    packed = fx.Int64(sink_stride)
    wide = fx.Int32(packed >> fx.Int64(SINK_F32_BIT)) & fx.Int32(1)
    stride = packed & fx.Int64(0xFFFFFFFF)
    first = fx.Int32(fx.Int64(head) * stride) << wide
    half_ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int16.ir_type, sink.address_space), sink
    )
    elem_ptr = fx.recast_iter(
        fx.PointerType.get(elem_dtype.ir_type, sink.address_space), sink
    )
    low = fx.ptr_load(elem_ptr + first)
    high = fx.Int32(fx.ptr_load(half_ptr + (first + wide), fx.Int16)) & fx.Int32(0xFFFF)
    low_bits = fx.Int32(low.bitcast(fx.Int16)) & fx.Int32(0xFFFF)
    as_f32 = ((high << fx.Int32(16)) | low_bits).bitcast(fx.Float32)
    return (wide != fx.Int32(0)).select(as_f32, fx.Float32(low.to(fx.Float32)))


def build_flash_attn_func_module(
    head_dim,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    daz=True,
    truncate_p=False,
    q_resident=None,
    dv_split=None,
    has_bias=False,
    generic_head_dim=False,
    hd_pair=False,
    bias_query_dtype=False,
    head_dim_v=None,
    has_sink=False,
    has_dropout=False,
):
    """Build the gfx11 Flash Attention launcher.

    Returns the ``@flyc.jit`` launcher, whose arguments are, in order:
    ``Q, K, V, O, LSE`` (pointers), ``batch_size, seq_len_q, seq_len_kv,
    num_heads, kv_group, right_bound, left_bound, align_br, flags`` (Int32; flags bit 0 is
    lse_on, bit 1 per-batch lengths, bit 2 padded rows filled, bit 3 packed sequences),
    ``scale`` (Float32), then the fifteen element strides ``(batch, seq, head)``
    of Q, K, V, O and LSE (Int64), the ``BIAS`` pointer and its four element
    strides ``(batch, head, query, key)`` (Int64), the runtime ``head_dim``
    (Int32; read only by a ``generic_head_dim`` build), ``v_group`` (Int32: query
    heads per V head), the ``SINK`` pointer and its head stride (Int64, with the sink's
    dtype in bit ``SINK_F32_BIT``; read only by a ``has_sink`` build), the per-batch length pointers ``SEQLEN_Q``/``SEQLEN_KV`` and
    the packed-sequence offset pointers ``ROW_OFF_Q``/``ROW_OFF_KV``, the dropout seed and
    offset1 pointers, offset2 (Int64), the keep threshold (Int32) and the scale (Float32)
    (read only by a ``has_dropout`` build), then the stream. The kernel itself takes the
    same list without ``batch_size``.
    """

    # Q is the B operand of every QK WMMA. Held in registers for the whole KV loop
    # it costs head_dim / 2 VGPRs per lane; above 128 that and the O accumulators
    # fill the register file, so wider heads re-read it per K-step instead.
    # An additive f32 bias, broadcast by strides, added to the scaled scores
    # before the mask. Baked: it is per-score work in the inner loop, and its
    # per-tile values (16 VGPRs) tip a register-resident Q past the file at 128.
    HAS_BIAS = bool(has_bias)
    # The bias in the query's dtype rather than f32 (bias_query_dtype): read as such and
    # widened to f32 as it is added.
    BIAS_Q = bool(bias_query_dtype)
    # An attention sink (has_sink): a per-head logit, f32 or in the query's dtype
    # (modification 31), that joins every row's softmax as a key with no value.
    HAS_SINK = bool(has_sink)
    # Dropout (has_dropout, modification 27): a Philox draw per score, so baked.
    HAS_DROPOUT = bool(has_dropout)
    PHILOX = Philox(width=32)
    # head_dim is then the largest served; the actual one (a multiple of 8) is a
    # runtime argument. Columns at or past it are zeroed as Q and K are loaded --
    # both, since 0 * NaN read from a neighbouring row is still NaN -- and not
    # stored from O.
    GENERIC_D = bool(generic_head_dim)
    # Head-dim pairs (hd_pair): two waves share each 16 query rows. Each computes
    # QK^T over its half of head_dim, the halves are summed through LDS, and each
    # accumulates half of O's columns -- so QK^T is computed once rather than once
    # per output-column workgroup, and each wave holds half of Q.
    # A pair re-reads its half of Q per K-step: held, it and the exchange push the
    # half-width O accumulators into spills.
    HD_PAIR = bool(hd_pair)
    PAIR = 2 if HD_PAIR else 1
    QK_DIM = head_dim // PAIR
    if q_resident is None:
        # At 128 a bias's or a dropout mask's per-tile values tip a resident Q past the
        # register file (modifications 14 and 27).
        Q_RESIDENT = not HD_PAIR and (
            head_dim < 128 or (head_dim == 128 and not HAS_BIAS and not HAS_DROPOUT)
        )
    else:
        Q_RESIDENT = bool(q_resident)
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert (
        BLOCK_N % K_SUB_N == 0
    ), f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert (
        BLOCK_M % ROWS_PER_WAVE == 0
    ), f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE * PAIR
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    BLOCK_N_OUT = BLOCK_N

    NUM_PREFETCH_K = 1
    NUM_PREFETCH_V = 1

    K_STEP_QK = WMMA_K
    K_STEPS_QK = head_dim // K_STEP_QK
    # The K-steps one wave computes: all of them, or its half with hd_pair.
    K_STEPS_QK_WAVE = K_STEPS_QK // PAIR

    # V and O are head_dim_v wide (MLA: narrower than Q and K); everything on the PV
    # side -- the output chunks, their split across workgroups and waves, the store --
    # is sized by it, the QK side by head_dim.
    HEAD_DIM_V = head_dim if head_dim_v is None else int(head_dim_v)
    MLA = HEAD_DIM_V != head_dim
    assert 16 <= HEAD_DIM_V <= head_dim and HEAD_DIM_V % 16 == 0
    assert not (MLA and generic_head_dim), "a V head_dim of its own is baked"
    D_CHUNK = WMMA_N
    D_CHUNKS = HEAD_DIM_V // D_CHUNK
    # The output columns may be split across workgroups: each computes the whole
    # of QK^T and the softmax but only DV_TILE columns of O, so it holds O_CHUNKS
    # accumulators instead of D_CHUNKS, and the grid grows by DV_SPLIT. Above 128
    # that, with Q re-read per K-step, is what fits the register file. hd_pair
    # halves each wave's columns again, inside the workgroup.
    if dv_split is None:
        DV_SPLIT = 2 if HEAD_DIM_V > 128 and not HD_PAIR else 1
    else:
        DV_SPLIT = int(dv_split)
    assert D_CHUNKS % (DV_SPLIT * PAIR) == 0
    assert K_STEPS_QK % PAIR == 0
    O_CHUNKS = D_CHUNKS // (DV_SPLIT * PAIR)
    DV_TILE = O_CHUNKS * D_CHUNK

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert dtype_str in ("f16", "bf16")

    HEAD_DIM = head_dim
    # The WMMA operand layout (modification 17): gfx12 passes 8 K-values per lane
    # for A and B, gfx11 all 16.
    WMMA_V8 = get_rocm_arch().lower().startswith("gfx12")
    WMMA_AB = 8 if WMMA_V8 else 16
    CAUSAL = causal
    # P (the softmax numerators, f32) is narrowed to bf16 for the second product.
    # Upstream truncates; rounding to nearest halves the bf16 output error, for a
    # cost inside measurement noise on gfx1151.
    BF16_TRUNCATE_P = bool(truncate_p)

    # Padding reduces LDS bank conflicts.
    K_STRIDE = HEAD_DIM + 4
    V_STRIDE = HEAD_DIM + 4

    VEC_WIDTH = 16
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
        KV_TAIL_GUARD = False
    else:
        # Enough passes to cover the tile. When the passes overrun it (a head_dim
        # whose rows do not tile the block's threads, e.g. 160), the last pass
        # stores only the rows inside it.
        NUM_BATCHES_KV = -(-BLOCK_N // ROWS_PER_BATCH_LOAD)
        KV_NEEDS_GUARD = False
        KV_TAIL_GUARD = NUM_BATCHES_KV * ROWS_PER_BATCH_LOAD > BLOCK_N

    # Buffer loads cap at dwordx4, so K and V rows are fetched in 8-element pieces.
    KV_SUBVECS = VEC_WIDTH // 8
    NUM_V_VECS = NUM_BATCHES_KV * KV_SUBVECS

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    _NUMERIC_MAP = {
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }
    elem_numeric_cls = _NUMERIC_MAP[dtype_str]
    ELEM_BYTES = (elem_numeric_cls.width + 7) // 8

    # hd_pair: each lane's partial scores, NUM_S_VALS f32 per lane per wave, laid
    # out as 4-wide pieces lane-contiguous so a wave's stores and its partner's
    # loads are conflict-free.
    LDS_XS_SIZE = NUM_WAVES * WARP_SIZE * NUM_S_VALS

    if HD_PAIR:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]
            xs: fx.Array[fx.Float32, LDS_XS_SIZE, 16]

    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_rdna_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        num_heads: fx.Int32,
        kv_group: fx.Int32,
        right_bound: fx.Int32,
        left_bound: fx.Int32,
        align_br: fx.Int32,
        flags: fx.Int32,
        scale: fx.Float32,
        q_sb: fx.Int64,
        q_ss: fx.Int64,
        q_sh: fx.Int64,
        k_sb: fx.Int64,
        k_ss: fx.Int64,
        k_sh: fx.Int64,
        v_sb: fx.Int64,
        v_ss: fx.Int64,
        v_sh: fx.Int64,
        o_sb: fx.Int64,
        o_ss: fx.Int64,
        o_sh: fx.Int64,
        lse_sb: fx.Int64,
        lse_ss: fx.Int64,
        lse_sh: fx.Int64,
        BIAS: fx.Pointer,
        bias_sb: fx.Int64,
        bias_sh: fx.Int64,
        bias_sq: fx.Int64,
        bias_skv: fx.Int64,
        head_dim_rt: fx.Int32,
        v_group: fx.Int32,
        SINK: fx.Pointer,
        sink_stride: fx.Int64,
        SEQLEN_Q: fx.Pointer,
        SEQLEN_KV: fx.Pointer,
        ROW_OFF_Q: fx.Pointer,
        ROW_OFF_KV: fx.Pointer,
        DROPOUT_SEED: fx.Pointer,
        DROPOUT_OFFSET1: fx.Pointer,
        dropout_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
    ):
        elem_dtype = elem_numeric_cls

        def _fadd(a, b):
            return a + b

        def _fsub(a, b):
            return a - b

        def _fmul(a, b):
            return a * b

        def _fmax(a, b):
            return fx.Float32(a).maximumf(fx.Float32(b))

        def _as_elem_ptr(ptr):
            return fx.recast_iter(
                fx.PointerType.get(elem_dtype.ir_type, ptr.address_space),
                ptr,
            )

        q_elem_ptr = _as_elem_ptr(Q)
        k_elem_ptr = _as_elem_ptr(K)
        v_elem_ptr = _as_elem_ptr(V)
        o_elem_ptr = _as_elem_ptr(O)

        def _bounds_checked_buf_ptr(ptr, num_records_bytes):
            # OOB_SELECT=3 zero-fills accesses beyond num_records.
            flags = (7 << 12) | (4 << 15) | (1 << 24) | (3 << 28)
            buf_ptr_ty = fx.PointerType.get(
                elem_ty=ptr.element_type.ir_type,
                address_space=fx.rocdl.TargetAddressSpace.BufferDesc,
                alignment=ptr.alignment,
            )
            return fx.make_ptr(
                buf_ptr_ty,
                [
                    ptr,
                    fx.Int16(0).ir_value(),
                    fx.Int64(num_records_bytes).ir_value(),
                    fx.Int32(flags).ir_value(),
                ],
            )

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32)
        )

        def wmma_acc(a_v16, b_v16, c_v8):
            # gfx11 ABI: 16 K-values per lane for A and B, 8 f32 accumulators;
            # gfx12: 8 K-values per lane.
            a_frag = fx.make_rmem_tensor(WMMA_AB, elem_dtype)
            b_frag = fx.make_rmem_tensor(WMMA_AB, elem_dtype)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_v16))
            b_frag.store(Vec(b_v16))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, a_frag, b_frag, c_frag)
            return Vec(c_frag.load())

        seq_q_v = fx.Int64(seq_len_q)
        seq_kv_v = fx.Int64(seq_len_kv)
        num_heads_v = fx.Int64(num_heads)
        kv_group_v = fx.Int64(kv_group)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr

        def lds_view(offset, width):
            return fx.make_view(
                lds_kv + fx.Int32(offset),
                fx.make_layout(width, 1),
            )

        def lds_load(offset, width=1):
            return lds_view(offset, width).load()

        def lds_store(offset, value):
            lds_view(offset, value.numel).store(value)

        block_id = fx.Int64(gpu.block_idx.x)
        tid = fx.Int64(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        if const_expr(HD_PAIR):
            # Partners are adjacent waves: pair_idx picks the head-dim half. Read
            # from the first lane, so it and every column offset it feeds are
            # known wave-uniform and stay scalar.
            pair_idx = fx.Int64(
                fx.Int32(
                    fx.rocdl.readfirstlane(
                        fx.Int32.ir_type, fx.Int32(wave_id % PAIR).ir_value()
                    )
                )
            )
            wave_q_offset = (wave_id // PAIR) * ROWS_PER_WAVE
            qk_col_base = pair_idx * QK_DIM
            lds_xs = lds.xs.ptr
        else:
            wave_q_offset = wave_id * ROWS_PER_WAVE

        # Output-column tile innermost: the workgroups sharing a Q tile run back to
        # back and share its K/V reads in L2.
        dv_idx = block_id % DV_SPLIT
        if const_expr(HD_PAIR):
            dv_col_base = (dv_idx * PAIR + pair_idx) * DV_TILE
        else:
            dv_col_base = dv_idx * DV_TILE
        block_rest = block_id // DV_SPLIT
        head_idx = block_rest % num_heads_v
        batch_q_tile_id = block_rest // num_heads_v
        num_q_tiles = (seq_q_v + BLOCK_M - 1) // BLOCK_M
        _q_tile_linear = batch_q_tile_id % num_q_tiles
        if const_expr(CAUSAL):
            # Dispatch longer causal tiles first.
            q_tile_idx = num_q_tiles - fx.Int64(1) - _q_tile_linear
        else:
            q_tile_idx = _q_tile_linear
        batch_idx = batch_q_tile_id // num_q_tiles
        # Modification 26: this batch's lengths, or the launch's when the pointers are
        # null. Everything past the grid reads these; the grid and a bias's extents keep
        # the launch's (the longest sequence). Modification 29: packed (THD), the batch's
        # rows start at its table entry (row_q / row_kv, the batch index when dense) and
        # it owns slot_q query rows (the launch's when not packed).
        seq_q_max = seq_q_v
        seq_kv_max = seq_kv_v
        seq_q_b_i32, seq_kv_b_i32, slot_q_i32, row_q, row_kv = load_sequence(
            flags,
            batch_idx,
            (seq_len_q, seq_len_kv),
            (SEQLEN_Q, SEQLEN_KV),
            (ROW_OFF_Q, ROW_OFF_KV),
        )
        seq_q_v = fx.Int64(seq_q_b_i32)
        seq_kv_v = fx.Int64(seq_kv_b_i32)
        kv_head_idx = head_idx // kv_group_v
        v_head_idx = head_idx // fx.Int64(v_group)
        q_start = q_tile_idx * BLOCK_M

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        q_base = row_q * q_sb + head_idx * q_sh
        o_base = row_q * o_sb + head_idx * o_sh

        def q_idx(token_idx, col):
            return q_base + token_idx * q_ss + col

        def o_idx(token_idx, col):
            return o_base + token_idx * o_ss + col

        # K and V are each read through a buffer descriptor scoped to one
        # (batch, kv head) slice, so a row at or past seq_len_kv -- including a
        # tail prefetch -- reads zero instead of a neighbouring head or past the
        # allocation. The slice must fit the descriptor's 32-bit num_records.
        def _kv_buf_ptr(elem_ptr, sb, ss, sh, slice_head, row_elems):
            slice_base = row_kv * sb + slice_head * sh
            # A batch with no key still gets one row's extent: its scores are all
            # masked, and the extent must not go negative.
            _last_row = fx.Int64(
                (seq_kv_v > fx.Int64(0)).select(seq_kv_v - fx.Int64(1), fx.Int64(0))
            )
            if const_expr(GENERIC_D):
                slice_elems = _last_row * ss + fx.Int64(head_dim_rt)
            else:
                slice_elems = _last_row * ss + fx.Int64(row_elems)
            return _bounds_checked_buf_ptr(
                fx.add_offset(elem_ptr, fx.Int64(slice_base)),
                fx.Int64(slice_elems) * fx.Int64(ELEM_BYTES),
            )

        k_buf_ptr = _kv_buf_ptr(k_elem_ptr, k_sb, k_ss, k_sh, kv_head_idx, HEAD_DIM)
        # A V row is head_dim_v wide: the staged columns past it read the next row (or
        # zero past the slice) and are never consumed.
        v_buf_ptr = _kv_buf_ptr(v_elem_ptr, v_sb, v_ss, v_sh, v_head_idx, HEAD_DIM_V)

        def k_idx(token_idx, col):
            return token_idx * k_ss + col

        def v_idx(token_idx, col):
            return token_idx * v_ss + col

        def _load_global_half_vec(elem_ptr, base_idx, width):
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(width, 1),
            )
            return Vec(view.load())

        def _store_global_half(elem_ptr, base_idx, val):
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(val.numel, 1),
            )
            view.store(Vec(val))

        def _bitcast_i32(value):
            return fx.Float32(value).bitcast(fx.Int32)

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack(f32_vals):
            """Pack f32 values into bf16 via bitwise truncation (upper 16 bits)."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(len(f32_vals) // 2):
                pairs.append(
                    _pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask)
                )
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype)

        def to_elem_vec(f32_vals):
            if const_expr(dtype_str == "bf16" and BF16_TRUNCATE_P):
                return bf16_trunc_pack(f32_vals)
            elem_list = []
            for j in range_constexpr(len(f32_vals)):
                elem_list.append(fx.Float32(f32_vals[j]).to(elem_dtype))
            return Vec.from_elements(elem_list, elem_dtype)

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Int64(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Int64(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Int64(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        # Clamp surplus loader rows; their LDS stores are discarded.
        load_row_kv = load_row_in_batch
        if const_expr(KV_NEEDS_GUARD):
            row_cap = fx.Int64(BLOCK_N - 1)
            load_row_kv = fx.Int64(
                (load_row_in_batch < row_cap).select(load_row_in_batch, row_cap)
            )

        def coop_load_kv_global(buf_ptr, idx_fn, tile_start):
            tile_start = fx.Int64(tile_start)
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_kv + row_offset
                for sv in range_constexpr(KV_SUBVECS):
                    g_idx = idx_fn(row_idx, load_col_base + fx.Int64(sv * 8))
                    raw = _load_global_half_vec(buf_ptr, g_idx, 8)
                    if const_expr(GENERIC_D):
                        col_ok = (load_col_base + fx.Int64(sv * 8)) < fx.Int64(
                            head_dim_rt
                        )
                        raw = col_ok.select(raw, Vec.filled(8, 0.0, elem_dtype))
                    vecs.append(raw)
            return vecs

        def _store_row_major(base, stride, lds_row, col_extra, vec):
            lds_idx = base + lds_row * stride + load_col_base + col_extra
            fx.ptr_store(Vec(vec), lds_kv + fx.Int32(lds_idx))

        def coop_store_kv_lds(vecs, base, stride):
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Int64(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        for sv in range_constexpr(KV_SUBVECS):
                            _store_row_major(
                                base,
                                stride,
                                lds_row,
                                sv * 8,
                                vecs[batch * KV_SUBVECS + sv],
                            )
                elif const_expr(KV_TAIL_GUARD and batch == NUM_BATCHES_KV - 1):
                    lds_row = load_row_in_batch + row_offset
                    if lds_row < fx.Int64(BLOCK_N):
                        for sv in range_constexpr(KV_SUBVECS):
                            _store_row_major(
                                base,
                                stride,
                                lds_row,
                                sv * 8,
                                vecs[batch * KV_SUBVECS + sv],
                            )
                else:
                    lds_row = load_row_in_batch + row_offset
                    for sv in range_constexpr(KV_SUBVECS):
                        _store_row_major(
                            base, stride, lds_row, sv * 8, vecs[batch * KV_SUBVECS + sv]
                        )

        def coop_load_k(tile_start, buf_id=0):
            vecs = coop_load_kv_global(k_buf_ptr, k_idx, tile_start)
            coop_store_kv_lds(vecs, k_buf_base(buf_id), K_STRIDE)

        def coop_load_v_global(tile_start):
            return coop_load_kv_global(v_buf_ptr, v_idx, tile_start)

        def coop_store_v_lds(vecs, buf_id=0):
            coop_store_kv_lds(vecs, v_buf_base(buf_id), V_STRIDE)

        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)

        q_in_bounds = q_row < seq_q_v
        # Q's loads are bounded by the rows the batch owns, not its length: a row past the
        # batch's length is still inside Q and its values are never stored (modification
        # 26). Dense or padded that is the launch's length; packed, the batch's slot, so
        # no load reaches the next sequence or past the last (modification 29).
        q_load_ok = q_row < fx.Int64(slot_q_i32)
        q_row_safe = fx.Int64(q_load_ok.select(q_row, fx.Int64(0)))

        # The mask, in the reference's terms (CpuFpReferenceSdpa isMasked): kv is
        # masked when kv > q + align + right_bound (the causal variant only) or
        # kv < q + align - left_bound (left_bound >= 0). align is the diagonal's
        # offset: 0 top-left, seq_len_kv - seq_len_q bottom-right. Computed here
        # from the lengths rather than passed in, so per-batch lengths can later
        # move the diagonal without another change to the argument list.
        seq_kv_i32 = seq_kv_b_i32
        align_i32 = fx.Int32(
            (fx.Int32(align_br) != fx.Int32(0)).select(
                seq_kv_i32 - seq_q_b_i32, fx.Int32(0)
            )
        )
        causal_offset_i32 = align_i32 + fx.Int32(right_bound)
        window_lo_i32 = fx.Int32(
            (fx.Int32(left_bound) >= fx.Int32(0)).select(
                align_i32 - fx.Int32(left_bound), fx.Int32(_NO_WINDOW)
            )
        )
        # Last kv column a query row of this lane may attend to.
        q_kv_limit_i32 = q_row_i32 + causal_offset_i32
        # First kv column it may attend to: a sliding window's lower edge. Far
        # below zero when there is no window, so it never masks.
        q_kv_first_i32 = q_row_i32 + window_lo_i32
        # The row attends to kv in [first, end): end is the kv tail, and for the
        # causal variant also one past its diagonal limit. Held as one unsigned
        # span so each score is masked by a single compare: kv - first wraps to
        # a huge unsigned value below the window, and lands at or past the span
        # beyond its end.
        if const_expr(CAUSAL):
            _q_kv_end_i32 = q_kv_limit_i32 + fx.Int32(1)
            q_kv_end_i32 = fx.Int32(
                (_q_kv_end_i32 < seq_kv_i32).select(_q_kv_end_i32, seq_kv_i32)
            )
        else:
            q_kv_end_i32 = seq_kv_i32
        # A row whose window starts past its end (top-left, seq_len_q > seq_len_kv
        # + left_bound) sees no key: its span is 0, not a wrapped difference that
        # would keep every key (modification 22).
        _q_kv_span_i32 = q_kv_end_i32 - q_kv_first_i32
        q_kv_span_u32 = fx.Uint32(
            (_q_kv_span_i32 > fx.Int32(0)).select(_q_kv_span_i32, fx.Int32(0))
        )

        # First KV column fully masked for this wave.
        wave_kv_limit_i32 = (
            fx.Int32(q_start + wave_q_offset + fx.Int64(ROWS_PER_WAVE))
            + causal_offset_i32
        )
        # A wave whose 16 rows all lie past seq_len_q -- 7 of 8 waves in a decode
        # step -- still takes part in the cooperative loads and barriers, but skips
        # the matrix work.
        wave_has_rows = (q_start + wave_q_offset) < seq_q_v

        # B operand of S^T = K @ Q^T: lane l carries query row l%16 and, on gfx11,
        # all 16 head-dim values of the K-step (lanes 16-31 duplicate lanes 0-15);
        # on gfx12, the 8 at 8*(l/16).
        c_zero_v16 = Vec.filled(16, 0.0, elem_dtype)

        def qk_col(offset):
            # Head-dim column `offset` of this wave's QK^T share.
            if const_expr(HD_PAIR):
                return qk_col_base + fx.Int64(offset)
            return fx.Int64(offset)

        def load_q_pack(ks):
            if const_expr(WMMA_V8):
                # gfx12: this lane's 8 head-dim values of the K-step, at 8*(l/16).
                col = qk_col(ks * K_STEP_QK) + klane * 8
                if const_expr(GENERIC_D):
                    col_ok = q_load_ok & (col < fx.Int64(head_dim_rt))
                    safe = col_ok.select(col, fx.Int64(0))
                    raw = Vec(
                        _load_global_half_vec(q_elem_ptr, q_idx(q_row_safe, safe), 8)
                    )
                    return col_ok.select(raw, Vec.filled(8, 0.0, elem_dtype))
                raw = _load_global_half_vec(q_elem_ptr, q_idx(q_row_safe, col), 8)
                return q_load_ok.select(raw, Vec.filled(8, 0.0, elem_dtype))
            if const_expr(GENERIC_D):
                # Two 8-column halves, each wholly in or out (head_dim % 8 == 0);
                # an out half reads column 0 instead, so nothing reads past a row.
                halves = []
                for half in range_constexpr(2):
                    col = qk_col(ks * K_STEP_QK + half * 8)
                    col_ok = q_load_ok & (col < fx.Int64(head_dim_rt))
                    safe = col_ok.select(col, fx.Int64(0))
                    raw = Vec(
                        _load_global_half_vec(q_elem_ptr, q_idx(q_row_safe, safe), 8)
                    )
                    halves.append(col_ok.select(raw, Vec.filled(8, 0.0, elem_dtype)))
                return Vec.from_elements(
                    [Vec(halves[0])[i] for i in range(8)]
                    + [Vec(halves[1])[i] for i in range(8)],
                    elem_dtype,
                )
            q_col = qk_col(ks * K_STEP_QK)
            raw = _load_global_half_vec(q_elem_ptr, q_idx(q_row_safe, q_col), 16)
            return q_load_ok.select(raw, c_zero_v16)

        q_b_packs = []
        if const_expr(Q_RESIDENT):
            for ks in range_constexpr(K_STEPS_QK_WAVE):
                q_b_packs.append(load_q_pack(ks))

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        if const_expr(HAS_BIAS):
            # Scores are scaled (and biased) before the softmax, so the
            # exponent and the running max are in the scaled domain already.
            c_scale_f = fx.Float32(scale)
            c_sm_scale_log2e = fx.Float32(_LOG2E)
            bias_f32_ptr = fx.recast_iter(
                fx.PointerType.get(
                    (elem_dtype if BIAS_Q else fx.Float32).ir_type, BIAS.address_space
                ),
                BIAS,
            )
            # One (batch, head) slice of the broadcast bias; a size-1 axis has
            # stride 0. Reads past it -- the kv tail -- come back as zero.
            _bias_slice_elems = (
                (seq_q_max - fx.Int64(1)) * bias_sq
                + (seq_kv_max - fx.Int64(1)) * bias_skv
                + fx.Int64(1)
            )
            bias_buf_ptr = _bounds_checked_buf_ptr(
                fx.add_offset(
                    bias_f32_ptr, fx.Int64(batch_idx * bias_sb + head_idx * bias_sh)
                ),
                fx.Int64(_bias_slice_elems) * fx.Int64(2 if BIAS_Q else 4),
            )
            bias_skv_i32 = fx.Int32(bias_skv)
            bias_row_i32 = fx.Int32(q_row_safe * bias_sq)
        else:
            c_sm_scale_log2e = fx.Float32(scale) * fx.Float32(_LOG2E)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)
        klane_is_zero = klane == fx.Int64(0)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        if const_expr(CAUSAL):
            _q_end_lim = q_start + BLOCK_M + fx.Int64(causal_offset_i32)
            kv_upper = fx.Int64((_q_end_lim < seq_kv_v).select(_q_end_lim, seq_kv_v))
        else:
            kv_upper = seq_kv_v

        # Tiles wholly below the window's lower edge for every row of this block
        # hold nothing any row attends to; start the loop at the first tile that
        # can. Without a window window_lo_i32 is far negative and this is 0.
        _kv_lo_raw = fx.Int64(q_start) + fx.Int64(window_lo_i32)
        _kv_lo_pos = fx.Int64(
            (_kv_lo_raw > fx.Int64(0)).select(_kv_lo_raw, fx.Int64(0))
        )
        kv_lower = (_kv_lo_pos // BLOCK_N_OUT) * BLOCK_N_OUT

        if const_expr(HAS_DROPOUT):
            # This (batch, query head) plane's first counter and the counters per row
            # (modification 27): the launch's lengths, not the batch's, as AOTriton's
            # Max_seqlen_q/k, so dense, padded and packed storage share one mask.
            _ph_seed = load_u64_or_zero(DROPOUT_SEED)
            _ph_offset = fx.Int64(dropout_offset2) + load_u64_or_zero(DROPOUT_OFFSET1)
            _ph_base, _ph_stride = PHILOX.grid_plane(
                _ph_offset,
                batch_idx * num_heads_v + head_idx,
                seq_len_q,
                seq_len_kv,
            )
            _ph_threshold = fx.Int32(idropout_p)
        # A query tile wholly past this batch's length runs no KV iteration
        # (modification 26); its rows are out of bounds, so nothing is stored.
        kv_upper = fx.Int64((q_start < seq_q_v).select(kv_upper, kv_lower))

        # Non-causal carries prefetched V across iterations; causal avoids the
        # extra VGPR lifetime and loads V in the current iteration.
        PREFETCH_V_ACROSS_ITERS = not CAUSAL

        if const_expr(PREFETCH_V_ACROSS_ITERS):
            _v_vecs_init = coop_load_v_global(kv_lower)

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(O_CHUNKS):
            init_args.append(c_zero_v8f32)
        if const_expr(PREFETCH_V_ACROSS_ITERS):
            for vi in range_constexpr(NUM_V_VECS):
                init_args.append(_v_vecs_init[vi])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            kv_lower, kv_upper, fx.Int64(BLOCK_N_OUT), init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(O_CHUNKS)]
            if const_expr(PREFETCH_V_ACROSS_ITERS):
                _v_vecs_tile = [
                    inner_iter_args[2 + O_CHUNKS + b]
                    for b in range_constexpr(NUM_V_VECS)
                ]

            coop_load_k(kv_block_start, 0)
            gpu.barrier()
            k_base = k_buf_base(0)

            if const_expr(not PREFETCH_V_ACROSS_ITERS):
                # Overlap the current V load with GEMM1 and softmax.
                _v_vecs_tile = coop_load_v_global(kv_block_start)

            if const_expr(CAUSAL):
                wave_needs_kv_tile = wave_has_rows & (
                    fx.Int32(kv_block_start) < wave_kv_limit_i32
                )
            else:
                wave_needs_kv_tile = wave_has_rows

            # S^T = K @ Q^T. A operand: lane l carries kv row l%16 of the
            # sub-tile, all 16 head-dim values of the K-step.
            s_accs = [c_zero_v8f32 for _ in range(NUM_S_ACCS)]

            if const_expr(HAS_BIAS):
                # Issued before the QK product so its latency hides behind it.
                # Element (st, r) is kv = kv_start + 16 st + 2 r + klane.
                _bias_tile_i32 = (
                    bias_row_i32
                    + (fx.Int32(kv_block_start) + fx.Int32(klane)) * bias_skv_i32
                )
                if const_expr(WMMA_V8):
                    # gfx12: element (st, r) is kv = kv_start + 16 st + r + 8 klane.
                    _bias_tile_i32 = (
                        bias_row_i32
                        + (fx.Int32(kv_block_start) + fx.Int32(klane) * fx.Int32(8))
                        * bias_skv_i32
                    )
                bias_vals = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        _kv = st * 16 + (r if const_expr(WMMA_V8) else 2 * r)
                        _off = _bias_tile_i32 + fx.Int32(_kv) * bias_skv_i32
                        _bv = fx.ptr_load(bias_buf_ptr + _off)
                        bias_vals.append(
                            _bv.to(fx.Float32) if const_expr(BIAS_Q) else _bv
                        )

            if wave_needs_kv_tile:
                if const_expr(not Q_RESIDENT):
                    q_next = load_q_pack(0)
                for ks in range_constexpr(K_STEPS_QK_WAVE):
                    k_col = qk_col(ks * K_STEP_QK)
                    if const_expr(Q_RESIDENT):
                        q_pack = q_b_packs[ks]
                    else:
                        # One step ahead, and fenced: left to itself the
                        # scheduler issues all of Q's loads at once and holds
                        # them, which is the register cost this path avoids.
                        q_pack = q_next
                        if const_expr(ks + 1 < K_STEPS_QK_WAVE):
                            q_next = load_q_pack(ks + 1)

                    for st_idx in range_constexpr(N_SUB_TILES):
                        st_base_row = st_idx * K_SUB_N

                        k_row_a = lane16 + fx.Int64(st_base_row)
                        k_row_b = lane16 + fx.Int64(st_base_row + 16)
                        if const_expr(WMMA_V8):
                            k_col_l = k_col + klane * 8
                            k_pack_a = Vec(
                                lds_load(k_base + k_row_a * K_STRIDE + k_col_l, 8)
                            )
                            k_pack_b = Vec(
                                lds_load(k_base + k_row_b * K_STRIDE + k_col_l, 8)
                            )
                        else:
                            k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                            k_pack_a = Vec(lds_load(k_lds_a, 16))

                            k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                            k_pack_b = Vec(lds_load(k_lds_b, 16))

                        acc_idx_a = st_idx * 2
                        acc_idx_b = st_idx * 2 + 1
                        s_accs[acc_idx_a] = wmma_acc(
                            k_pack_a, q_pack, s_accs[acc_idx_a]
                        )
                        s_accs[acc_idx_b] = wmma_acc(
                            k_pack_b, q_pack, s_accs[acc_idx_b]
                        )
                    if const_expr(not Q_RESIDENT):
                        fx.rocdl.sched_barrier(0)

            if const_expr(HD_PAIR):
                # Publish this wave's partial scores, then stage V: the barrier
                # that publishes V also publishes them, and the partner's are
                # added in after it. Unconditional: a wave without the tile
                # publishes zeros, and its partner shares its rows, so skips it too.
                def _xs_index(wave, piece):
                    return (
                        wave * fx.Int64(NUM_S_ACCS * 2) + fx.Int64(piece)
                    ) * fx.Int64(WARP_SIZE * 4) + lane * fx.Int64(4)

                for st in range_constexpr(NUM_S_ACCS):
                    for h in range_constexpr(2):
                        piece = Vec.from_elements(
                            [Vec(s_accs[st])[h * 4 + j] for j in range(4)], fx.Float32
                        )
                        fx.make_view(
                            lds_xs + fx.Int32(_xs_index(wave_id, st * 2 + h)),
                            fx.make_layout(4, 1),
                        ).store(piece)
                coop_store_v_lds(_v_vecs_tile, 0)
                gpu.barrier()
                partner = wave_id + fx.Int64(1) - pair_idx * fx.Int64(2)
                for st in range_constexpr(NUM_S_ACCS):
                    halves = []
                    for h in range_constexpr(2):
                        halves.append(
                            Vec(
                                fx.make_view(
                                    lds_xs + fx.Int32(_xs_index(partner, st * 2 + h)),
                                    fx.make_layout(4, 1),
                                ).load()
                            )
                        )
                    peer_s = Vec.from_elements(
                        [halves[j // 4][j % 4] for j in range(8)], fx.Float32
                    )
                    s_accs[st] = Vec(s_accs[st]) + peer_s

            # Element r of accumulator a holds kv = kv_start + 16a + 2r + klane
            # (gfx12: + r + 8*klane) for this lane's query row. Mask the kv tail and, when causal, the
            # columns past this row's limit. Unconditional: a select per score
            # is cheap beside the WMMA work, and it keeps the scores out of a
            # data-dependent region.
            # Offset of this lane's first score from the row's first attendable
            # column; each score adds only a constant to it.
            if const_expr(WMMA_V8):
                # gfx12: element r holds kv = kv_start + 16a + r + 8*klane.
                kv_rel_base_i32 = (
                    fx.Int32(kv_block_start)
                    + fx.Int32(klane) * fx.Int32(8)
                    - q_kv_first_i32
                )
            else:
                kv_rel_base_i32 = (
                    fx.Int32(kv_block_start) + fx.Int32(klane) - q_kv_first_i32
                )
            s_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_val = Vec(s_accs[st])[r]
                    if const_expr(HAS_BIAS):
                        s_val = fx.math.fma(s_val, c_scale_f, bias_vals[st * 8 + r])
                    _kv = st * 16 + (r if const_expr(WMMA_V8) else 2 * r)
                    kv_rel_u32 = fx.Uint32(kv_rel_base_i32 + fx.Int32(_kv))
                    s_val = (kv_rel_u32 < q_kv_span_u32).select(s_val, c_neg_inf)
                    s_raw.append(s_val)

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            # A row with every column so far masked has m = -inf; keep its
            # exponents finite so it contributes zeros rather than NaNs.
            m_safe = (m_new_raw == c_neg_inf).select(c_zero_f, m_new_raw)

            diff_m_raw = _fsub(m_running, m_safe)
            diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
            corr = fx.Float32(
                fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff_m_scaled).ir_value())
            )

            scaled_max = _fmul(c_sm_scale_log2e, m_safe)
            neg_scaled_max = _fsub(c_zero_f, scaled_max)

            p_vals = []
            local_sum = c_zero_f
            for r in range_constexpr(NUM_S_VALS):
                if const_expr(HAS_BIAS):
                    # A row the bias masks with values near -FLT_MAX has a max whose
                    # product with log2(e) overflows, making neg_scaled_max +inf and
                    # every p inf/NaN; the difference first stays finite
                    # (modification 22).
                    diff = _fmul(_fsub(s_raw[r], m_safe), c_sm_scale_log2e)
                else:
                    diff = fx.math.fma(
                        s_raw[r],
                        c_sm_scale_log2e,
                        neg_scaled_max,
                    )
                p = fx.Float32(
                    fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff).ir_value())
                )
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            peer_sum = reduction_peer(local_sum)
            tile_sum = _fadd(local_sum, peer_sum)
            l_corr = _fmul(corr, l_running)
            l_new = _fadd(l_corr, tile_sum)

            if const_expr(HAS_DROPOUT):
                # After the row sum, before P is packed: the sum and the LSE stay undropped
                # (modification 27). Element (st, r) is key kv_start + 16 st + 2 r + klane on
                # gfx11, kv_start + 16 st + r + 8 klane on gfx12.
                for st in range_constexpr(NUM_S_ACCS):
                    _kb = fx.Int64(kv_block_start) + fx.Int64(st * 16)
                    # The 8 consecutive keys of this lane's half of the 16: klane 0 the first,
                    # klane 1 the second (on gfx12 these are the lane's own 8).
                    _first = PHILOX.grid_offset(
                        _ph_base, _ph_stride, q_row, _kb + klane * fx.Int64(8)
                    )
                    _words = PHILOX.u32(_ph_seed, _first) + PHILOX.u32(
                        _ph_seed, _first + fx.Int64(1)
                    )
                    _keep_half = [fx.Int32(w) > _ph_threshold for w in _words]
                    if const_expr(WMMA_V8):
                        _keeps = _keep_half
                    else:
                        _own = fx.Int32(0)
                        for k in range_constexpr(8):
                            _own = _own | fx.Int32(
                                _keep_half[k].select(fx.Int32(1 << k), fx.Int32(0))
                            )
                        _peer = fx.Int32(_own).shuffle_xor(shuf_16_i32, width_i32)
                        # klane 0 holds keys 0, 2, .., 14 of the 16: bits 0, 2, 4, 6 of its
                        # own half, then of its peer's; klane 1 the odd keys, peer first.
                        _lo = fx.Int32(klane_is_zero.select(_own, _peer))
                        _hi = fx.Int32(klane_is_zero.select(_peer, _own))
                        _bit0 = fx.Int32(klane)
                        _keeps = []
                        for r in range_constexpr(8):
                            _src = _lo if const_expr(r < 4) else _hi
                            _pos = fx.Int32(2 * (r % 4)) + _bit0
                            _keeps.append(((_src >> _pos) & fx.Int32(1)) != fx.Int32(0))
                    for r in range_constexpr(8):
                        _i = st * 8 + r
                        p_vals[_i] = fx.Float32(
                            _keeps[r].select(fx.Float32(p_vals[_i]), c_zero_f)
                        )

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
            for dc in range_constexpr(O_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            if const_expr(not HD_PAIR):
                coop_store_v_lds(_v_vecs_tile, 0)
                gpu.barrier()

            # B operand of O^T += V^T @ P^T: lane l needs query row l%16 and, on
            # gfx11, all 16 kv values of the step. It holds the even (klane 0) or
            # odd (klane 1) half; its xor-16 peer holds the other. On gfx12 it needs
            # the 8 it holds.
            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    own = [p_vals[p_base + j] for j in range(8)]
                    if const_expr(WMMA_V8):
                        # gfx12: the lane's 8 kv values are its operand as they are.
                        p_packs_st.append(to_elem_vec(own))
                    else:
                        peer = [reduction_peer(own[j]) for j in range(8)]
                        full = []
                        for j in range_constexpr(8):
                            full.append(klane_is_zero.select(own[j], peer[j]))
                            full.append(klane_is_zero.select(peer[j], own[j]))
                        p_packs_st.append(to_elem_vec(full))
                p_packs_all.append(p_packs_st)

            v_base = v_buf_base(0)

            # A operand: lane l carries head-dim row dc*16 + l%16 and all 16 kv
            # values of the step, read transposed out of the row-major V tile.
            def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val, v_base=v_base):
                d_pos = dv_col_base + fx.Int64(dc_val * D_CHUNK) + lane16
                v_elems = []
                if const_expr(WMMA_V8):
                    # gfx12: this lane's 8 kv values of the step, at 8*(l/16).
                    for k_sub in range_constexpr(8):
                        kv_row = (
                            fx.Int64(st_kv_base_val + pks_val * PV_K_STEP + k_sub)
                            + klane * 8
                        )
                        v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                        v_elems.append(fx.ptr_load(lds_kv + fx.Int32(v_lds_idx)))
                    return Vec.from_elements(v_elems, elem_dtype)
                for k_sub in range_constexpr(16):
                    kv_row = fx.Int64(st_kv_base_val + pks_val * PV_K_STEP + k_sub)
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(fx.ptr_load(lds_kv + fx.Int32(v_lds_idx)))
                return Vec.from_elements(v_elems, elem_dtype)

            if wave_needs_kv_tile:
                o_tmp = list(o_accs)

                cur_v_packs = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    cur_v_packs.append(_load_v_rowmajor(st_idx * K_SUB_N, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(O_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= O_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v_packs.append(
                                    _load_v_rowmajor(
                                        st_idx * K_SUB_N, next_pks, next_dc
                                    )
                                )

                        for st_idx in range_constexpr(N_SUB_TILES):
                            o_tmp[dc] = wmma_acc(
                                cur_v_packs[st_idx],
                                p_packs_all[st_idx][pks],
                                o_tmp[dc],
                            )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs

                o_accs = o_tmp

            m_running = m_new_raw
            l_running = l_new

            if const_expr(PREFETCH_V_ACROSS_ITERS):
                next_kv_start = fx.Int64(kv_block_start) + fx.Int64(BLOCK_N_OUT)
                _v_vecs_tile = coop_load_v_global(next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(PREFETCH_V_ACROSS_ITERS):
                for vi in range_constexpr(NUM_V_VECS):
                    _yield_args.append(_v_vecs_tile[vi])
            loop_results = yield _yield_args

        m_final = loop_results[0]
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(O_CHUNKS)]

        # A row no key reaches -- possible with a window, or bottom-right causal
        # with more queries than keys -- has l = 0. Its output is zero and its LSE
        # -inf, as the reference defines them, rather than 0 * inf. Unordered, so a
        # NaN sum is not keyless: O and the LSE stay NaN, as PyTorch's logsumexp
        # (modification 24).
        row_has_keys = l_final != c_zero_f
        if const_expr(HAS_SINK):
            # The max in natural-log units: unscaled without a bias, scaled with one.
            m_natural = (
                fx.Float32(m_final)
                if const_expr(HAS_BIAS)
                else fx.Float32(scale) * fx.Float32(m_final)
            )
            # The sink joins the sum as one more key with no value, against the
            # larger of it and the row's max; a row with no keys keeps only it.
            # In f32 or the query's dtype (modification 31).
            sink_val = load_sink_logit(SINK, sink_stride, head_idx, elem_dtype)
            # A sink of -inf on a row with no keys: m_top floored at -FLT_MAX so the
            # sink's term is exp(-inf) = 0 rather than exp(-inf - -inf) = NaN; then
            # l_total is 0, O is 0 and the LSE -inf (modification 23).
            m_top = row_has_keys.select(
                _fmax(m_natural, sink_val),
                _fmax(sink_val, fx.Float32(-3.4028234663852886e38)),
            )
            keep = row_has_keys.select(fx.math.exp(m_natural - m_top), c_zero_f)
            l_total = l_final * keep + fx.math.exp(sink_val - m_top)
            inv_l = (l_total != c_zero_f).select(keep / l_total, c_zero_f)
            lse_val = m_top + fx.math.log(l_total)
        else:
            inv_l = row_has_keys.select(c_one_f / l_final, c_zero_f)
        if const_expr(HAS_DROPOUT):
            # 1 / (1 - p) folds into 1 / l; l itself stays the undropped sum.
            inv_l = fx.Float32(inv_l) * fx.Float32(dropout_scale)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)

        # Element r of an O accumulator is head-dim row 2r + klane on gfx11. Trade
        # with the xor-16 peer so lane klane writes rows [8*klane, 8*klane + 8),
        # which is where gfx12 has them already.
        o_rows = []
        for dc in range_constexpr(O_CHUNKS):
            o_norm = Vec(_fmul(o_finals[dc], inv_l_vec))
            own = [o_norm[r] for r in range(8)]
            if const_expr(WMMA_V8):
                # gfx12: element r is head-dim row r + 8*klane already.
                o_rows.append(own)
            else:
                peer = [reduction_peer(own[r]) for r in range(8)]
                rows = []
                for j in range_constexpr(8):
                    if const_expr(j % 2 == 0):
                        lo_src, hi_src = own, peer
                    else:
                        lo_src, hi_src = peer, own
                    # klane 0 -> row j = 2*(j//2) + j%2; klane 1 -> row 8 + j.
                    rows.append(
                        klane_is_zero.select(lo_src[j // 2], hi_src[4 + j // 2])
                    )
                o_rows.append(rows)

        # Natural-log log-sum-exp of the scaled scores, [B, H, Sq, 1] f32: the
        # running max is unscaled, and l is a sum of exp((s - m) * scale). A sink
        # object computed it above, with inv_l.
        if const_expr(not HAS_SINK):
            lse_val = row_has_keys.select(
                (
                    fx.Float32(m_final)
                    if const_expr(HAS_BIAS)
                    else fx.Float32(scale) * fx.Float32(m_final)
                )
                + fx.math.log(l_final),
                c_neg_inf,
            )
        write_lse = q_in_bounds & ((fx.Int32(flags) & fx.Int32(1)) != fx.Int32(0))
        # Modification 28: a padded row (past the batch's query length, inside the
        # launch's) is written as O = 0 and LSE = -inf when bit 2 of flags says so.
        pad_row = (
            (q_row >= seq_q_v)
            & q_load_ok
            & ((fx.Int32(flags) & fx.Int32(4)) != fx.Int32(0))
        )

        if q_in_bounds:
            for dc in range_constexpr(O_CHUNKS):
                o_trunc = Vec.from_elements(o_rows[dc], fx.Float32).to(elem_dtype)
                d_col = dv_col_base + fx.Int64(dc * D_CHUNK) + klane * 8
                if const_expr(GENERIC_D):
                    if d_col < fx.Int64(head_dim_rt):
                        _store_global_half(o_elem_ptr, o_idx(q_row, d_col), o_trunc)
                else:
                    _store_global_half(o_elem_ptr, o_idx(q_row, d_col), o_trunc)

        if pad_row:
            o_zero = Vec.filled(8, 0.0, elem_dtype)
            for dc in range_constexpr(O_CHUNKS):
                d_col = dv_col_base + fx.Int64(dc * D_CHUNK) + klane * 8
                if const_expr(GENERIC_D):
                    if d_col < fx.Int64(head_dim_rt):
                        _store_global_half(o_elem_ptr, o_idx(q_row, d_col), o_zero)
                else:
                    _store_global_half(o_elem_ptr, o_idx(q_row, d_col), o_zero)
            if (fx.Int32(flags) & fx.Int32(1)) != fx.Int32(0):
                pad_lse_ptr = fx.recast_iter(
                    fx.PointerType.get(fx.Float32.ir_type, LSE.address_space), LSE
                )
                _store_global_half(
                    pad_lse_ptr,
                    row_q * lse_sb + head_idx * lse_sh + q_row * lse_ss,
                    Vec.from_elements([c_neg_inf], fx.Float32),
                )

        # Both lanes holding a row write the same value to the same address.
        if write_lse:
            lse_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, LSE.address_space), LSE
            )
            lse_index = row_q * lse_sb + head_idx * lse_sh + q_row * lse_ss
            _store_global_half(
                lse_ptr, lse_index, Vec.from_elements([lse_val], fx.Float32)
            )

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        num_heads: fx.Int32,
        kv_group: fx.Int32,
        right_bound: fx.Int32,
        left_bound: fx.Int32,
        align_br: fx.Int32,
        flags: fx.Int32,
        scale: fx.Float32,
        q_sb: fx.Int64,
        q_ss: fx.Int64,
        q_sh: fx.Int64,
        k_sb: fx.Int64,
        k_ss: fx.Int64,
        k_sh: fx.Int64,
        v_sb: fx.Int64,
        v_ss: fx.Int64,
        v_sh: fx.Int64,
        o_sb: fx.Int64,
        o_ss: fx.Int64,
        o_sh: fx.Int64,
        lse_sb: fx.Int64,
        lse_ss: fx.Int64,
        lse_sh: fx.Int64,
        BIAS: fx.Pointer,
        bias_sb: fx.Int64,
        bias_sh: fx.Int64,
        bias_sq: fx.Int64,
        bias_skv: fx.Int64,
        head_dim_rt: fx.Int32,
        v_group: fx.Int32,
        SINK: fx.Pointer,
        sink_stride: fx.Int64,
        SEQLEN_Q: fx.Pointer,
        SEQLEN_KV: fx.Pointer,
        ROW_OFF_Q: fx.Pointer,
        ROW_OFF_KV: fx.Pointer,
        DROPOUT_SEED: fx.Pointer,
        DROPOUT_OFFSET1: fx.Pointer,
        dropout_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        ctx = CompilationContext.get_current()

        bs_idx = fx.Uint64(batch_size)
        sl_idx = fx.Uint64(seq_len_q)
        nh_idx = fx.Uint64(num_heads)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * nh_idx * DV_SPLIT

        launcher = flash_attn_func_rdna_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            seq_len_q,
            seq_len_kv,
            num_heads,
            kv_group,
            right_bound,
            left_bound,
            align_br,
            flags,
            scale,
            q_sb,
            q_ss,
            q_sh,
            k_sb,
            k_ss,
            k_sh,
            v_sb,
            v_ss,
            v_sh,
            o_sb,
            o_ss,
            o_sh,
            lse_sb,
            lse_ss,
            lse_sh,
            BIAS,
            bias_sb,
            bias_sh,
            bias_sq,
            bias_skv,
            head_dim_rt,
            v_group,
            SINK,
            sink_stride,
            SEQLEN_Q,
            SEQLEN_KV,
            ROW_OFF_Q,
            ROW_OFF_KV,
            DROPOUT_SEED,
            DROPOUT_OFFSET1,
            dropout_offset2,
            idropout_p,
            dropout_scale,
        )

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    launch_flash_attn_func.compile_hints = {
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    # Tile geometry the dispatcher must reproduce; see the generator.
    launch_flash_attn_func.block_m = BLOCK_M
    launch_flash_attn_func.block_size = BLOCK_SIZE
    launch_flash_attn_func.dv_split = DV_SPLIT
    launch_flash_attn_func.head_dim_v = HEAD_DIM_V
    launch_flash_attn_func.hd_pair = int(HD_PAIR)
    launch_flash_attn_func.has_sink = int(HAS_SINK)
    launch_flash_attn_func.has_dropout = int(HAS_DROPOUT)
    return launch_flash_attn_func
