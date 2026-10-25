# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/FlyDSL
#   commit:   89ad52fbbb9e252396829a8a44b12e18b7acd970 (v0.3.4.1+19)
#   path:     kernels/attention/flash_attn_utils.py
#   compiler: flydsl 0.3.4 (pip wheel)
#
# CDNA (gfx90a, gfx942, gfx950) attention for the hipkernel:flydsl_sdpa_cdna engine.
# Modifications are listed in README.md beside this file.

"""Shared helpers for the gfx950 dual-wave, software-pipelined flash-attention kernels."""

import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cdna3
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from kernels.common import buffer_ops
from kernels.common.kernels_common import LOG2E as _LOG2E
from kernels.common.kernels_common import dtype_to_elem_type

# gfx950 (MI350/MI355X): 8 XCDs, each with a private ~4 MB L2.
NUM_XCD_GFX950 = 8
LDS_BYTES_GFX950 = 160 * 1024
MIN_Q_BLOCKS_XCD_SWIZZLE = 64
# The dual-wave 8-wave CTA fixes the q-block height; callers need it to count
# q-blocks before any traits object exists.
DUALWAVE_SWP_BLOCK_M = 256
# Reserve an aligned out-of-bounds vector offset below the signed-i32 limit.
PAGED_FP8_BUFFER_LIMIT_BYTES = (1 << 31) - 16
# A bias build's floor for the running max of a row no key has reached (modification
# 28): below every finite score a bf16 or f16 bias gives, the bf16 lowest value
# (-3.39e38) included, so it lifts only a row of -inf. Not -FLT_MAX: under the kernels'
# no-infs fast math, max(x, -FLT_MAX) is a no-op LLVM removes, as it removes a -inf bit
# test on a fast-math result.
_BIAS_NEG_FLOOR = -3.4e38
# What a bias build's generic kernel writes over a score its own masks remove (key
# padding, causal, window; modification 28): -FLT_MAX, below the floor, so its exponent
# is 0 against any running max. Not -inf: the bias build's exponent subtracts the max
# first, and under the no-infs fast math LLVM folds `select(c, -inf, s) - m` to `s - m`
# (the -inf arm is poison), which dropped the key-padding mask and let the keys past a
# ragged Skv into the softmax.
_BIAS_MASKED_SCORE = -3.4028234663852886e38
# A strided bias build (modification 31) keeps its scores halved, (q.k * scale + bias)
# / 2, so an f32 bias at its lowest finite value (-FLT_MAX) stays above the floor and the
# masked score: the floor is -FLT_MAX / 2, every halved score's lower bound, and the
# exponent and the LSE take a factor of 2 back. Halving is exact, so every exponent and
# LSE is bit-for-bit the unhalved build's.
_BIAS_HALVED_NEG_FLOOR = -1.7014117331926443e38


def _no_keys(l_sum):
    """Whether a row's softmax sum is exactly +0, the "no key reached it" test of every
    O and LSE select (modification 29). Tested on the bits: a row whose sum is NaN (a NaN
    or +inf score) must keep its NaN O and LSE, as PyTorch's do, but under the kernels'
    no-NaNs fast math LLVM rewrites the ordered `l == 0` select as `l != 0` (v_cmp_lg,
    false for NaN), and `l > 0` is false for NaN too. An integer compare it leaves alone.
    The sum starts at +0 and only adds non-negative terms, so +0 is its only zero."""
    return fx.Float32(l_sum).bitcast(fx.Int32) == fx.Int32(0)


# The `sink_on` argument of a sink object (modification 36): no sink (the logit is -inf), a
# sink in the query's dtype, or an f32 one (cuDNN's and hipDNN's reference's dtype).
SINK_OFF = 0
SINK_QUERY_DTYPE = 1
SINK_F32 = 2


def load_sink_logit(sink, sink_stride, sink_on, head, elem_dtype, neg_inf):
    """Query head ``head``'s attention-sink logit as f32, or ``neg_inf`` when ``sink_on`` is
    SINK_OFF (modification 36). ``sink_stride`` is the head stride in elements of the sink's
    dtype, ``sink_on`` SINK_QUERY_DTYPE or SINK_F32. Read as two 16-bit loads at the element
    -- an f32 sink's low and high halves, a 16-bit sink's one element twice -- so neither
    dtype's load reaches past it, without a branch; both widen exactly."""
    rsrc = buffer_ops.create_buffer_resource(sink, max_size=True)
    on = fx.Int32(sink_on)
    wide = (on == fx.Int32(SINK_F32)).select(fx.Int32(1), fx.Int32(0))
    first = fx.Int32(head * fx.Index(sink_stride)) << wide
    low = fx.Int16(
        buffer_ops.buffer_load(rsrc, as_mlir_value(first), vec_width=1, dtype=fx.Int16)
    )
    high = fx.Int16(
        buffer_ops.buffer_load(
            rsrc, as_mlir_value(first + wide), vec_width=1, dtype=fx.Int16
        )
    )
    low_bits = fx.Int32(low) & fx.Int32(0xFFFF)
    high_bits = fx.Int32(high) & fx.Int32(0xFFFF)
    as_f32 = ((high_bits << fx.Int32(16)) | low_bits).bitcast(fx.Float32)
    as_query = fx.Float32(low.bitcast(elem_dtype).to(fx.Float32))
    logit = (wide != fx.Int32(0)).select(as_f32, as_query)
    return (on != fx.Int32(SINK_OFF)).select(logit, neg_inf)


# The generic kernel's `varlen_flags` bits (modification 32).
# Bit 0: per-batch lengths are passed (padding); a padded query row gets O = 0, LSE = -inf.
VARLEN_PADDED = 1
# Bit 1: the diagonal is bottom-right, so it moves with each batch's own lengths.
VARLEN_BOTTOM_RIGHT = 2
# Bit 2: Q/O and K/V are packed (THD) through row-offset tables, INT32 [B + 1]: sequence b
# starts at token offset[b] and holds offset[b + 1] - offset[b] tokens (modification 34).
VARLEN_PACKED = 4


def _load_i32(tensor, index):
    """The wave-uniform int32 at @p tensor's base + @p index."""
    gep = buffer_ops.get_element_ptr(
        _extract_aligned_pointer(tensor, address_space=1),
        fx.Int64(index),
        elem_type=T.i32,
    )
    return fx.Int32(rocdl.readfirstlane(T.i32, _pointer_load(T.i32, gep)))


@dataclass
class _Varlen:
    """One batch's sequences as a launch with per-batch lengths sees them (modifications
    32, 34), each an i32: the rows (`len_*`) that are real; the rows a store may write
    (`view_*`: the launch's on a padded or dense one, the sequence's own when packed,
    where the rows past it are the next sequence's); the first token of a packed sequence
    (`tok_*`, 0 otherwise); the bottom-right diagonal's shift; and whether packed."""

    len_q: object
    len_kv: object
    view_q: object
    view_kv: object
    tok_q: object
    tok_kv: object
    shift: object
    packed: object


def _load_varlen(varlen_flags, seq_lens, row_offs, batch_idx, seq_len, seq_len_kv):
    """Batch @p batch_idx's `_Varlen` (modifications 32, 34).

    Under `VARLEN_PACKED` the row-offset tables @p row_offs give each sequence's first
    token and its token count; under `VARLEN_PADDED` the INT32 per-batch arrays
    @p seq_lens give the real rows within it (or within the launch's rows, unpacked).
    Every count is clamped to [0, the launch's length], as hipDNN's GPU reference clamps
    lengths. The shift is (Skv_b - Skv) - (Sq_b - Sq) under `VARLEN_BOTTOM_RIGHT`. Each
    read sits under a real branch on its flag, which arrives with the kernel's arguments,
    so a dense launch loads nothing and waits on nothing (a branch on the pointers, or a
    branchless read of a fallback pointer, cost the RDNA kernels' short launches several
    percent). Built as explicit ``scf.if``s: the ``if`` rewrite is lexical to a kernel's
    body."""
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    flags = fx.Int32(varlen_flags)
    full_q = fx.Int32(seq_len)
    full_kv = fx.Int32(seq_len_kv)
    zero = fx.Int32(0)

    def clamp(value, full):
        low = fx.Int32((value > zero).select(value, zero))
        return fx.Int32((low < full).select(low, full))

    packed = (flags & fx.Int32(VARLEN_PACKED)) != zero
    pack_if = _scf.IfOp(as_mlir_value(packed), results_=[T.i32] * 4, has_else=True)
    with ir.InsertionPoint(pack_if.then_block):
        values = []
        for table, full in zip(row_offs, (full_q, full_kv)):
            first = _load_i32(table, batch_idx)
            end = _load_i32(table, fx.Index(batch_idx) + fx.Index(1))
            values += [first, clamp(end - first, full)]
        tok_q, view_q, tok_kv, view_kv = values
        _scf.YieldOp([as_mlir_value(v) for v in (tok_q, tok_kv, view_q, view_kv)])
    with ir.InsertionPoint(pack_if.else_block):
        _scf.YieldOp([as_mlir_value(v) for v in (zero, zero, full_q, full_kv)])
    tok_q, tok_kv, view_q, view_kv = [fx.Int32(r) for r in pack_if.results]

    padded = (flags & fx.Int32(VARLEN_PADDED)) != zero
    pad_if = _scf.IfOp(as_mlir_value(padded), results_=[T.i32] * 2, has_else=True)
    with ir.InsertionPoint(pad_if.then_block):
        lengths = [
            clamp(_load_i32(tensor, batch_idx), view)
            for tensor, view in zip(seq_lens, (view_q, view_kv))
        ]
        _scf.YieldOp([as_mlir_value(v) for v in lengths])
    with ir.InsertionPoint(pad_if.else_block):
        _scf.YieldOp([as_mlir_value(v) for v in (view_q, view_kv)])
    len_q, len_kv = [fx.Int32(r) for r in pad_if.results]

    bottom_right = (flags & fx.Int32(VARLEN_BOTTOM_RIGHT)) != zero
    shift = fx.Int32(bottom_right.select((len_kv - full_kv) - (len_q - full_q), zero))
    return _Varlen(len_q, len_kv, view_q, view_kv, tok_q, tok_kv, shift, packed)


def _load_seq_lens(varlen_flags, seq_lens, batch_idx, seq_len, seq_len_kv):
    """Batch @p batch_idx's query and key lengths and its diagonal's shift, as i32
    (modification 32).

    With `VARLEN_PADDED` set in @p varlen_flags, each length is read from its INT32
    per-batch array in @p seq_lens and clamped to [0, the launch's length], as hipDNN's
    GPU reference clamps them; the shift is (Skv_b - Skv) - (Sq_b - Sq) under
    `VARLEN_BOTTOM_RIGHT`, else 0. Otherwise the launch's lengths and no shift. A real
    branch on a flag that arrives with the kernel's arguments, so a dense launch loads
    nothing and waits on nothing (a branch on the pointers, or a branchless read of a
    fallback pointer, cost the RDNA kernels' short launches several percent). Built as
    an explicit ``scf.if``: the ``if`` rewrite is lexical to a kernel's body."""
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    flags = fx.Int32(varlen_flags)
    full_q = fx.Int32(seq_len)
    full_kv = fx.Int32(seq_len_kv)
    zero = fx.Int32(0)
    padded = (flags & fx.Int32(VARLEN_PADDED)) != zero
    if_op = _scf.IfOp(as_mlir_value(padded), results_=[T.i32] * 3, has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        lengths = []
        for tensor, full in zip(seq_lens, (full_q, full_kv)):
            gep = buffer_ops.get_element_ptr(
                _extract_aligned_pointer(tensor, address_space=1),
                fx.Int64(batch_idx),
                elem_type=T.i32,
            )
            raw = fx.Int32(rocdl.readfirstlane(T.i32, _pointer_load(T.i32, gep)))
            low = fx.Int32((raw > zero).select(raw, zero))
            lengths.append(fx.Int32((low < full).select(low, full)))
        len_q, len_kv = lengths
        bottom_right = (flags & fx.Int32(VARLEN_BOTTOM_RIGHT)) != zero
        shift = fx.Int32(
            bottom_right.select((len_kv - full_kv) - (len_q - full_q), zero)
        )
        _scf.YieldOp([as_mlir_value(v) for v in (len_q, len_kv, shift)])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([as_mlir_value(v) for v in (full_q, full_kv, zero)])
    return [fx.Int32(r) for r in if_op.results]


# Philox4x32-10, AOTriton's dropout RNG (modification 33): the round of the aiw kernel's
# `attention_rdna4/philox.py` (Triton's, verbatim), restated here so the CDNA objects
# digest only their own directory. The constants and the order of the round are the
# stream's contract; the provider's tests pin the mask it draws bit for bit against a
# host oracle and AOTriton's own masks.
_PHILOX_MUL_A = 0xD2511F53
_PHILOX_MUL_B = 0xCD9E8D57
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_ROUNDS = 10


def _philox4x32(c0, c1, k0, k1):
    """The four 32-bit words of Philox4x32-10 at counter (c0, c1, 0, 0) and key
    (k0, k1). Raw ``arith`` ops wrap at 32 bits, which is the PRNG's arithmetic; the
    ``fx`` operators would fold a constant key through unbounded Python ints."""
    from flydsl._mlir.dialects import arith as _arith  # noqa: PLC0415

    def i32(value):
        return fx.Int32(value)

    def raw(value):
        return as_mlir_value(fx.Int32(value))

    mul_a, mul_b = i32(_PHILOX_MUL_A), i32(_PHILOX_MUL_B)
    key_a, key_b = i32(_PHILOX_KEY_A), i32(_PHILOX_KEY_B)
    c2 = c3 = i32(0)
    for _ in range(_PHILOX_ROUNDS):
        hi_b = _arith.MulUIExtendedOp(raw(mul_b), raw(c2)).high
        hi_a = _arith.MulUIExtendedOp(raw(mul_a), raw(c0)).high
        lo_b = _arith.muli(raw(mul_b), raw(c2))
        lo_a = _arith.muli(raw(mul_a), raw(c0))
        c0 = i32(_arith.xori(_arith.xori(hi_b, raw(c1)), raw(k0)))
        c2 = i32(_arith.xori(_arith.xori(hi_a, raw(c3)), raw(k1)))
        c1 = i32(lo_b)
        c3 = i32(lo_a)
        k0 = i32(_arith.addi(raw(k0), raw(key_a)))
        k1 = i32(_arith.addi(raw(k1), raw(key_b)))
    return c0, c1, c2, c3


def _load_u64_words(tensor):
    """The 8 bytes at @p tensor's base, as two wave-uniform i32 words (low, high)."""
    base = _extract_aligned_pointer(tensor, address_space=1)
    words = []
    for index in (0, 1):
        gep = buffer_ops.get_element_ptr(base, fx.Int64(index), elem_type=T.i32)
        words.append(fx.Int32(rocdl.readfirstlane(T.i32, _pointer_load(T.i32, gep))))
    return words


def _rows_bytes(rows, span):
    """A buffer view's byte count @p span for @p rows rows, or 0 for none: a padded
    batch's length may be 0 (modification 32), where (rows - 1) * stride + width would
    go negative and wrap the record count."""
    return fx.Index((rows > fx.Index(0)).select(span, fx.Index(0)))


_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'


def _if_then(cond, then_fn):
    """Run ``then_fn`` under a runtime ``cond`` from a helper method.

    The AST rewriter lowers ``if`` to ``scf.if`` only inside the decorated
    kernel body; a plain ``if`` on a dynamic condition here would reach
    ``Boolean.__bool__`` during tracing (FlyDSL #452). This emits the same
    ``scf.if`` the rewriter would.
    """
    ReplaceIfWithDispatch.scf_if_dispatch(cond, then_fn)


# Wait and low-level ROCDL wrappers


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only.

    In the GFX9 encoding every CDNA target shares, emitted directly: FlyDSL's
    arch-dispatched form knows only gfx942 and gfx950 (modification 24).
    """
    cdna3.s_waitcnt(vmcnt=n)


def _is_gfx942_class(gpu_arch):
    """gfx942 or gfx90a: the generic kernel's 64 KiB-LDS path (modification 24).

    The two share what that path uses -- 64 KiB of LDS, global-to-LDS DMA of at most
    four bytes, the 32x32x8 MFMA -- so gfx90a takes every gfx942 branch; gfx950 adds
    the 16-byte DMA, the LDS transpose read and the 32x32x16 MFMA.
    """
    return gpu_arch.startswith(("gfx942", "gfx90a"))


def _s_waitcnt(val):
    rocdl.s_waitcnt(val)


def _sched_barrier(val):
    rocdl.sched_barrier(val)


def _s_barrier():
    rocdl.s_barrier()


def _s_setprio(val):
    rocdl.s_setprio(val)


def _dualwave_sync_barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _s_nop(x):
    if not isinstance(x, int) or not 0 <= x <= 15:
        raise ValueError("s_nop immediate must be a Python int in [0, 15]")
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"), [], f"s_nop {x}", "", has_side_effects=True
    )


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with DUALWAVE_SWP immediate byte offset."""
    imm = int(imm_offset)
    raw_type = T.vec(2, T.i32)
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return (
        fx.Vector(raw)
        .bitcast(fx.Numeric.from_ir_type(ir.VectorType(result_type).element_type))
        .ir_value()
    )


def _ds_read_tr8_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b8 (8-bit transpose) with immediate byte offset.

    Returns 64 bits = 8 fp8 (the fp8 analog of ds_read_b64_tr_b16's 4 bf16),
    used for the fp8 V transpose load.
    """
    imm = int(imm_offset)
    raw_type = T.vec(2, T.i32)
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b8 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return (
        fx.Vector(raw)
        .bitcast(fx.Numeric.from_ir_type(ir.VectorType(result_type).element_type))
        .ir_value()
    )


# Arithmetic and inline-asm primitives


def _tree_reduce(vals, binop):
    items = list(vals)
    while len(items) > 1:
        nxt = [binop(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
        if len(items) % 2 == 1:
            nxt.append(items[-1])
        items = nxt
    return items[0]


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return fx.Float32(value).bitcast(fx.Int32).ir_value()


def _bitcast_f32(value):
    return fx.Int32(value).bitcast(fx.Float32).ir_value()


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    """DUALWAVE_SWP pair mask asm: 2 compares followed by 2 cndmasks."""
    asm_str = (
        f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
        f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
        "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
        "v_cndmask_b32_e64 $3, $5, $7, $1"
    )
    ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
    ret = llvm.inline_asm(
        ret_struct_ty,
        [
            as_mlir_value(x_ref_i32),
            as_mlir_value(y_ref_i32),
            as_mlir_value(rel_i32),
            as_mlir_value(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


def _reduction_pair(v_f32):
    v_i32 = _bitcast_i32(v_f32)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(
        pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
    )
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b):
    is_hi_half = lane_div_32 != fx.Index(0)
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = is_hi_half.select(y0_a_lo, y0_a_hi), is_hi_half.select(
        y1_a_lo, y1_a_hi
    )
    y0_b, y1_b = is_hi_half.select(y0_b_lo, y0_b_hi), is_hi_half.select(
        y1_b_lo, y1_b_hi
    )
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return w0, w1, w2, w3


def _o_pack_2dw(traits, v_o, dc, store_group, elem_dtype):
    r_base = store_group * 4
    if const_expr(traits.DTYPE_STR == "bf16"):
        lo = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base],
            Vec(v_o[dc])[r_base + 1],
        )
        hi = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base + 2],
            Vec(v_o[dc])[r_base + 3],
        )
        return lo, hi

    o_f16 = []
    for i in range_constexpr(4):
        o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
    return as_mlir_value(pack[0]), as_mlir_value(pack[1])


def _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    return _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b)


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    return Vec.from_elements(
        [
            fx.Int32(w)
            for w in _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype)
        ],
        fx.Int32,
    )


def _anchor_scalar_f32(x):
    """Pin a scalar f32 at the current source position (no-op asm)."""
    x_ir = as_mlir_value(x)
    return llvm.inline_asm(
        x_ir.type,
        [x_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )


def _anchor_v_o(traits, v_o):
    """Pin v_o accumulators at the current source position."""
    acc_irs = [as_mlir_value(v_o[dc]) for dc in range_constexpr(traits.D_CHUNKS)]
    ret_ty = ir.Type.parse(
        f"!llvm.struct<({', '.join(['vector<16xf32>'] * traits.D_CHUNKS)})>"
    )
    constraints = ",".join(
        ["=v"] * traits.D_CHUNKS + [str(i) for i in range(traits.D_CHUNKS)]
    )
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [
        llvm.extractvalue(acc_irs[dc].type, ret, [dc])
        for dc in range_constexpr(traits.D_CHUNKS)
    ]


def _anchor_v_p(traits, v_p, elem_dtype):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    p_all = _concat_vectors(p_lo_all, p_hi_all)
    p_all_ir = as_mlir_value(p_all)
    p_all_anchored = llvm.inline_asm(
        p_all_ir.type,
        [p_all_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )
    p_vec = Vec(p_all_anchored, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    anchored_lo = []
    anchored_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        anchored_lo.append(
            p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value()
        )
        anchored_hi.append(
            p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value()
        )
    return anchored_lo, anchored_hi


def _v_pair_to_vec32(v):
    return _concat_vectors(v[0], v[1]).ir_value()


def _v_vec32_to_pair(v):
    v_vec = Vec(v, (32,), fx.Float32)
    v_lo = v_vec.shuffle(v_vec, [i for i in range(16)]).ir_value()
    v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
    return v_lo, v_hi


def _v_p_to_vec32(v_p):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    return _concat_vectors(p_lo_all, p_hi_all).ir_value()


def _v_vec32_to_p(traits, v_p_all, elem_dtype):
    p_vec = Vec(v_p_all, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    p_lo = []
    p_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        p_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        p_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
    return p_lo, p_hi


def _rescale_value_types(traits, elem_dtype):
    v32bf16_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, elem_dtype)
    v32f32_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, fx.Float32)
    return v32bf16_type, v32f32_type


def _scale_v_p(traits, v_p, scale_scalar, elem_dtype, fm_fast):
    v32bf16_type, v32f32_type = _rescale_value_types(traits, elem_dtype)
    fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
    p_all = _v_p_to_vec32(v_p)
    p_all_f32_op = llvm.FPExtOp(v32f32_type, as_mlir_value(p_all))
    p_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(
        traits.PV_K_STEPS * 2 * 8
    )
    p_scaled_f32 = as_mlir_value(scale_vec * Vec(p_all_f32_op.result))
    p_scaled_bf16_op = llvm.FPTruncOp(v32bf16_type, p_scaled_f32)
    p_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
    return _v_vec32_to_p(traits, p_scaled_bf16_op.result, elem_dtype=elem_dtype)


def _bf16_trunc_pack_v8(traits, f32_vals, elem_dtype):
    if const_expr(traits.DTYPE_STR == "bf16"):
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
    # fp16: truncate each f32 -> f16 (RNE) and build the v8 pack directly.
    f16_vals = []
    for i in range_constexpr(8):
        f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
    return Vec.from_elements(f16_vals, elem_dtype).ir_value()


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _reduce_score_pair(v_s, initial, reducer, fm_fast):
    s_lo, s_hi = v_s
    acc = initial
    for r in range_constexpr(16):
        acc = reducer(acc, s_lo[r], fm_fast)
    for r in range_constexpr(16):
        acc = reducer(acc, s_hi[r], fm_fast)
    return acc


def _lane_pair_reduce(v, reducer, fm_fast):
    lhs, rhs = _reduction_pair(v)
    return reducer(lhs, rhs, fm_fast)


def _score_pair_max(v_s, neg_inf, fm_fast):
    reducer = lambda a, b, _fm: fx.maxnumf(a, b)  # noqa: E731
    return _lane_pair_reduce(
        _reduce_score_pair(v_s, neg_inf, reducer, fm_fast), reducer, fm_fast
    )


def _score_pair_sum(v_s, zero_f, fm_fast):
    s_lo, s_hi = _score_lists_to_vecs(v_s)
    tile = Vec(s_lo) + Vec(s_hi)
    reducer = lambda a, b, _fm: a + b  # noqa: E731
    return _lane_pair_reduce(
        tile.reduce("add", init_val=zero_f, fastmath=fm_fast), reducer, fm_fast
    )


def _sub_score_pair(v_s, row_max, fm_fast):
    s_lo, s_hi = v_s
    lo_sub = []
    hi_sub = []
    for r in range_constexpr(16):
        lo_sub.append(s_lo[r] - row_max)
    for r in range_constexpr(16):
        hi_sub.append(s_hi[r] - row_max)
    return (
        Vec.from_elements(lo_sub, fx.Float32).ir_value(),
        Vec.from_elements(hi_sub, fx.Float32).ir_value(),
    )


def _scale_sub_score_pair(
    v_s, row_max_raw, scale, zero_f, fm_fast, bias=None, explicit_rounding=False
):
    """Fused softmax-scale + row-max subtraction (optimization 1-A).

    Returns ``scale * (v_s - row_max_raw) + bias`` per element via a single FMA
    (``fma(s, scale, bias - scale*row_max_raw)``), so the fp8 QK MMA can emit raw
    (un-scaled) logits and reduce_max can run in the raw domain (scale > 0 is
    order-preserving). Replaces the separate post-QK scale multiply + subtract.
    ``-inf`` masked lanes stay ``-inf`` (scale > 0), matching the un-fused path.

    ``bias`` lands in the FMA's addend, so a caller needing ``exp2`` to produce
    ``2**bias * P`` pays nothing -- see ``DualwaveFp8SoftmaxHelper.sub_m``.
    """
    s_lo, s_hi = v_s
    neg_scaled_max = zero_f - scale * row_max_raw
    if bias is not None:
        # exp2 lands on 2**bias * P instead of P, at no extra instruction: the
        # FMA's addend absorbs it.
        neg_scaled_max = neg_scaled_max + bias
    if const_expr(explicit_rounding):
        # Match the kernel descriptor's FP32 nearest-even mode through the public
        # math API. Explicit rounding retains unpacked FMA in gfx950 codegen.

        def _center(values):
            centered = []
            for i in range_constexpr(16):
                centered.append(
                    fx.fma(
                        Vec(values)[i],
                        scale,
                        neg_scaled_max,
                        fastmath=fx.arith.FastMathFlags.none,
                        roundingmode=fx.RoundingMode.to_nearest_even,
                    )
                )
            return Vec.from_elements(centered, fx.Float32)

        lo, hi = _center(s_lo), _center(s_hi)
    else:
        scale_v = Vec.from_elements([scale], fx.Float32).broadcast_to(16)
        nsm_v = Vec.from_elements([neg_scaled_max], fx.Float32).broadcast_to(16)
        lo = fx.fma(Vec(s_lo), scale_v, nsm_v, fastmath=fm_fast)
        hi = fx.fma(Vec(s_hi), scale_v, nsm_v, fastmath=fm_fast)
    return as_mlir_value(lo), as_mlir_value(hi)


def _exp2_score_slice(v_s, start, length):
    if const_expr(start == 0):
        s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        lo_partial = []
        for r in range_constexpr(16):
            lo_partial.append(rocdl.exp2(T.f32, as_mlir_value(s_lo[r])))
        return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

    lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
    hi_full = []
    for r in range_constexpr(16):
        hi_full.append(rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])))
    return lo_partial, hi_full


def _pack_p_v8_slices(traits, v_p, pack_v8_fn):
    lo_partial_list, hi_full = v_p
    p_lo_packs = []
    p_hi_packs = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        p_base = pks * 8
        lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        hi_slice = hi_full[p_base : p_base + 8]
        p_lo_packs.append(pack_v8_fn(lo_slice))
        p_hi_packs.append(pack_v8_fn(hi_slice))
    return p_lo_packs, p_hi_packs


def _safe_l_inv(l_row, zero_f):
    l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
    return (fx.Float32(l_row) > zero_f).select(l_inv, zero_f)


def _rescale_from_tile_max(m_row, m_tile_max, fm_fast):
    row_max = fx.maxnumf(m_row, m_tile_max)
    rescale = rocdl.exp2(T.f32, as_mlir_value(m_row - row_max))
    return row_max, rescale


def _scale_o_accs(v_o, scale_scalar, traits, fm_fast):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = Vec(v_o[dc]) * scale_vec


def _causal_pair_thresholds(kv_vectorized):
    if const_expr(kv_vectorized):
        return [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
        ]
    return [
        (0, 1),
        (2, 3),
        (8, 9),
        (10, 11),
        (16, 17),
        (18, 19),
        (24, 25),
        (26, 27),
    ]


def _apply_dualwave_causal_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        new_x, new_y = _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_bits, y_bits
        )
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


# Descriptor, LDS, and buffer helpers


def _llvm_value(value):
    """Unwrap FlyDSL values before passing them to LLVM dialect ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    ptr_type = ir.Type.parse(
        "!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>"
    )
    return fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _lds_alias_scope_array(names):
    attrs = [
        f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>'
        for name in names
    ]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def _dualwave_lds_scope(kind, buf_id):
    return f"lds_{kind}{buf_id}"


def _dualwave_lds_alias_scopes(name):
    return _lds_alias_scope_array([name])


def _dualwave_lds_noalias_scopes(name, scope_names):
    return _lds_alias_scope_array(
        [scope_name for scope_name in scope_names if scope_name != name]
    )


def _cu_load(div, idx, cu_atom, cu_v1i32):
    """Load cu_seqlens[idx] into an SGPR. ``idx`` must be wave-uniform."""
    v = fly.copy_atom_call_ssa(
        [cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx)))
    )
    return fx.Index(
        rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0])))
    )


def _make_page_view(
    base_iter,
    base_iter_ty,
    align,
    page_id,
    page_byte_stride,
    page_nrec_bytes,
    page_layout,
    elem_ir,
    buf_flags_i32,
):
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    off_i64 = fx.Int64(page_id * page_byte_stride)
    shifted = fx.inttoptr(base_iter_ty, base_i64 + off_i64)
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=elem_ir,
        address_space=rocdl.TargetAddressSpace.BufferDesc,
        alignment=align,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [
            shifted,
            fx.Int16(0).ir_value(),
            page_nrec_bytes.ir_value(),
            buf_flags_i32.ir_value(),
        ],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, page_layout), fx.make_layout(1, 1))


def _make_ws_rsrc(ws_base_i64, byte_offset, nrec_bytes):
    addr_i64 = as_mlir_value(ws_base_i64 + fx.Int64(byte_offset))
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64, num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes))
    )


def _make_rebased_view(
    base_iter, byte_off, nrec_bytes, layout, _buf_flags_i32, _elem_ir
):
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=_elem_ir,
        address_space=rocdl.TargetAddressSpace.BufferDesc,
        alignment=base_iter.alignment,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [
            shifted,
            fx.Int16(0).ir_value(),
            fx.Int64(nrec_bytes).ir_value(),
            _buf_flags_i32.ir_value(),
        ],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))


def _make_raw_buffer_rsrc(tensor):
    base_ptr = _extract_aligned_pointer(tensor)
    base_i64 = llvm.PtrToIntOp(T.i64, base_ptr).result
    base_lo = arith.trunci(T.i32, fx.Int64(base_i64).ir_value())
    base_hi = arith.trunci(T.i32, fx.Int64(base_i64).shrui(fx.Int64(32)).ir_value())
    return Vec.from_elements(
        [
            base_lo,
            base_hi,
            buffer_ops._create_i32_constant(0xFFFFFFFF),
            buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
        ],
        fx.Int32,
    ).ir_value()


def _read_v8f16_off(traits, v_base_ptr, const_off, kv_mfma_pack_type):
    ptr = buffer_ops.get_element_ptr(
        v_base_ptr,
        byte_offset=as_mlir_value(fx.Int32(const_off * traits.BF16_BYTES)),
        elem_type=T.i8,
    )
    return llvm.LoadOp(kv_mfma_pack_type, ptr, alignment=16).result


def _load_k_pack_aligned(traits, lds_kv_base_ptr, elem_idx, buf_id, kv_mfma_pack_type):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * traits.BF16_BYTES
    ptr = buffer_ops.get_element_ptr(
        lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8
    )
    return llvm.LoadOp(
        kv_mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


# ---------------- attention-bias LDS staging ----------------
# The per-lane score layout wants 16 bias values from one q row, so a direct global
# read makes every lane in a wave touch a different bias row (one cache line per
# lane). Instead each wave DMAs its own [ROWS_PER_WAVE, BLOCK_N] tile with 8 lanes
# per row -- fully coalesced -- and reads the per-lane pattern back out of LDS.
# Column granules are XOR-swizzled by row so the read side spreads across banks;
# without it all 32 lanes of a half-wave would hit one bank (row stride is 128B).


def _bias_lanes_per_row(traits):
    return traits.BLOCK_N * traits.BF16_BYTES // traits.DMA_BYTES


def _bias_gran_elems(traits):
    return traits.DMA_BYTES // traits.BF16_BYTES


def _bias_wave_bytes(traits):
    return traits.ROWS_PER_WAVE * traits.BLOCK_N * traits.BF16_BYTES


def _bias_buf_bytes(traits):
    return traits.NUM_WAVES * _bias_wave_bytes(traits)


def _num_bias_dma(traits):
    return _bias_wave_bytes(traits) // (traits.WARP_SIZE * traits.DMA_BYTES)


def _bias_dma_m0_base(traits, buf, d, wave_id_uni, lds_bias_base_idx):
    lds_addr = (
        lds_bias_base_idx
        + buf * _bias_buf_bytes(traits)
        + wave_id_uni * _bias_wave_bytes(traits)
        + d * (traits.WARP_SIZE * traits.DMA_BYTES)
    )
    return fx.Int32(lds_addr)


def _bias_dma_src_elem(
    traits, row_base, tile_col_base, d, lane_in_warp, bias_stride0_v
):
    lanes_per_row = _bias_lanes_per_row(traits)
    stride = fx.Int32(bias_stride0_v)
    lane_i32 = fx.Int32(lane_in_warp)
    row_in_group = lane_i32 // fx.Int32(lanes_per_row)
    gran = (lane_i32 % fx.Int32(lanes_per_row)) ^ row_in_group
    uni = (
        fx.Int32(row_base) + fx.Int32(d * (traits.WARP_SIZE // lanes_per_row))
    ) * stride + fx.Int32(tile_col_base)
    uni_s = rocdl.readfirstlane(T.i32, as_mlir_value(uni))
    return (
        fx.Int32(uni_s)
        + row_in_group * stride
        + gran * fx.Int32(_bias_gran_elems(traits))
    )


BIAS_MAX_OFFSET_ELEMS = 2**31 - 1
BIAS_MAX_DESCRIPTOR_BYTES = 0xFFFFFFFF


def bias_addressing_error(elems, elem_size):
    """Reason an `elems`-element bias cannot be addressed, or None if it fits.

    A bias past either limit wraps the i32 offset or overruns the descriptor, so
    the kernel would silently read the wrong rows instead of failing. Callers
    prefix their own entry-point name and the bias shape.
    """
    if elems > BIAS_MAX_OFFSET_ELEMS:
        return (
            f"has {elems} elements, exceeding the {BIAS_MAX_OFFSET_ELEMS} "
            f"addressable by the kernel's i32 bias element offsets"
        )
    if elems * elem_size > BIAS_MAX_DESCRIPTOR_BYTES:
        return (
            f"occupies {elems * elem_size} bytes, exceeding the "
            f"{BIAS_MAX_DESCRIPTOR_BYTES} byte range of the bias buffer descriptor"
        )
    return None


def _bias_lds_lane_base(traits, buf, wave_id, lane_mod_32, lane_div_32, vec_elems):
    return (
        buf * _bias_buf_bytes(traits)
        + wave_id * _bias_wave_bytes(traits)
        + lane_mod_32 * (traits.BLOCK_N * traits.BF16_BYTES)
        + lane_div_32 * (vec_elems * traits.BF16_BYTES)
    )


def _make_bias_lds_ptr(lds_bias_base_idx):
    return buffer_ops.create_llvm_ptr(lds_bias_base_idx, address_space=3)


def _load_bias_frag_lds(lds_bias_base_ptr, byte_offset, frag_type, align):
    ptr = buffer_ops.get_element_ptr(
        lds_bias_base_ptr, byte_offset=byte_offset, elem_type=T.i8
    )
    return llvm.LoadOp(
        frag_type,
        ptr,
        alignment=align,
        alias_scopes=_dualwave_lds_alias_scopes(_dualwave_lds_scope("bias", 0)),
    ).result


def _ws_store_f32(f32_val, local_elem_index, rsrc):
    """32-bit f32 store into a per-split-z workspace region via raw buffer descriptor."""
    f32_ir = as_mlir_value(fx.Float32(f32_val))
    buffer_ops.buffer_store(f32_ir, rsrc, as_mlir_value(fx.Int32(local_elem_index)))


def _ws_store_quad_i32(dwords, local_elem_index, rsrc):
    """128-bit i32x4 store (buffer_store_dwordx4) into a per-split-z workspace region."""
    vec_ir = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32).ir_value()
    buffer_ops.buffer_store(vec_ir, rsrc, as_mlir_value(fx.Int32(local_elem_index)))


def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa(
        [q_load_i32x4_type],
        _load_atom_128,
        fx.slice(q_div, (None, fx.Int32(elem_index))),
    )


def _buffer_load_lds_128(
    src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty
):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_128(
    pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div
):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(
        _store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index)))
    )


# Mapping and address helpers


def _k_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
    # runtime buf_id (rare): K0=0, K1=DUALWAVE_SWP_KV_PER_BUFFER
    return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _kv_tile_addr(
    traits, tile_start, kv_gmem_elem_offset, kv_head_elem_offset, stride_kv_n_v
):
    """Return (src_base, soffset): dense uses tile_start*stride; paged folds page offset into the descriptor."""
    if const_expr(traits.PAGED):
        return kv_head_elem_offset, 0
    return kv_gmem_elem_offset, tile_start * stride_kv_n_v


def _linear_kv_src_elem(
    traits, src_base, d, n_in_warp, wave_id, d_bucket, stride_kv_n_v
):
    """Global element index for lane's d-th 128-bit chunk in a linear KV tile."""
    n_in_tile = n_in_warp * traits.NUM_WAVES + wave_id
    global_d = d_bucket * traits.VEC_KV + d * traits.D_128B_SIZE
    return src_base + n_in_tile * stride_kv_n_v + global_d


def _vec_v_elem(n, d, kv_head_idx, TRAITS_BLOCK_N, TRAITS_KV_VEC_SIZE, TRAITS_HEAD_DIM):
    return (
        kv_head_idx
        * (TRAITS_BLOCK_N // TRAITS_KV_VEC_SIZE)
        * TRAITS_HEAD_DIM
        * TRAITS_KV_VEC_SIZE
        + (n // TRAITS_KV_VEC_SIZE) * TRAITS_HEAD_DIM * TRAITS_KV_VEC_SIZE
        + d * TRAITS_KV_VEC_SIZE
        + (n % TRAITS_KV_VEC_SIZE)
    )


def _vec_k_dma_oct_idx(traits, d, wave_id_uni, lane_in_warp):
    """Flat octet index for this wave/lane's d-th DMA slot in vectorized K layout."""
    return (
        wave_id_uni * (traits.WARP_SIZE * traits.SMEM_D_RPT)
        + d * traits.WARP_SIZE
        + lane_in_warp
    )


def _sigma_k_tile_n(ni):
    """Sigma permutation applied to K tile-n during vectorized DMA (bit-shuffle)."""
    return (ni & 3) | ((ni & 8) >> 1) | ((ni & 4) << 1) | (ni & ~15)


def _vec_k_src_elem(traits, d, wave_id_uni, lane_in_warp, kv_head_idx):
    """Global element index for vectorized K DMA slot d (sigma remap applied)."""
    oct_idx = _vec_k_dma_oct_idx(
        traits, d, wave_id_uni=wave_id_uni, lane_in_warp=lane_in_warp
    )
    ni, dg = oct_idx % traits.BLOCK_N, oct_idx // traits.BLOCK_N
    src_oct = dg * traits.BLOCK_N + _sigma_k_tile_n(ni)
    return (
        kv_head_idx
        * (traits.HEAD_DIM // traits.KV_VEC_SIZE)
        * traits.BLOCK_N
        * traits.KV_VEC_SIZE
        + src_oct * traits.KV_VEC_SIZE
    )


def _vec_v_src_elem(traits, d, wave_id_uni, lane_in_warp, kv_head_idx):
    """Global element index for vectorized V DMA slot d (NO-major wave rows)."""
    row = wave_id_uni * traits.SMEM_D_RPT + d
    no = lane_in_warp // traits.SMEM_N_PER_WAVE
    d_col = row * traits.SMEM_N_PER_WAVE + lane_in_warp % traits.SMEM_N_PER_WAVE
    return _vec_v_elem(
        no * traits.KV_VEC_SIZE,
        d_col,
        kv_head_idx,
        traits.BLOCK_N,
        traits.KV_VEC_SIZE,
        traits.HEAD_DIM,
    )


def _paged_bt_byte_offset(tile_idx, split_t0):
    """Byte offset of `tile_idx`'s page-id entry in the LDS block-table cache."""
    return fx.Int32((tile_idx - split_t0) * fx.Index(4))


def _q_pack_col(traits, ks, lane_div_32):
    """K-dimension column for Q pack at MFMA k-step `ks` for this lane."""
    return ks * traits.K_STEP_QK + lane_div_32 * traits.MFMA_LANE_K


def _q_pack_global_idx(traits, q_row_in_block, ks, lane_div_32, stride_q_n_v):
    """Flat global element index for Q pack (q_row, ks)."""
    return q_row_in_block * stride_q_n_v + _q_pack_col(
        traits, ks, lane_div_32=lane_div_32
    )


def _get_q_pack(traits, q_all_scaled_bf16, ks):
    q_vec = Vec(q_all_scaled_bf16)
    base = ks * traits.MFMA_LANE_K
    return q_vec.shuffle(
        q_vec, [base + i for i in range(traits.MFMA_LANE_K)]
    ).ir_value()


def _vec_k_lds_idx_lo(traits, k_base, ks, lane_div_32, lane_mod_32):
    """LDS element index for vectorized K pack lo at k-step `ks` for this lane."""
    return (
        k_base
        + (ks * 2 + lane_div_32) * (traits.BLOCK_N * traits.KV_VEC_SIZE)
        + lane_mod_32 * traits.KV_VEC_SIZE
    )


def _swizzled_ks_offset(traits, ks):
    """Non-vectorized K LDS offset for k-step `ks` (outer/inner swizzle pattern)."""
    return (ks // 4) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
        ks % 4
    ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _k_lds_read_base_per_lane(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // 8) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _v_lds_read_base_per_lane(traits, lane, lane_div_32):
    return (
        lane_div_32 * traits.V_LDS_TO_REG_HALF_WAVE_STRIDE
        + ((lane % 16) // 4) * traits.V_LDS_TO_REG_LANE_QUAD_STRIDE
        + ((lane // 16) % 2) * traits.V_LDS_TO_REG_N_GROUP_STRIDE
        + (lane % 4) * traits.V_LDS_TO_REG_LANE_IN_QUAD_STRIDE
    )


def _vec_v_lds_addr_base(traits, v_base, lane_div_32, lane_mod_32):
    """Per-lane LDS element base (in elements) for vectorized V reads."""
    lm = lane_mod_32
    return (
        v_base
        + (lm // traits.SMEM_N_PER_WAVE) * traits.VEC_V_ROW_STRIDE
        + lane_div_32 * traits.D_128B_SIZE
        + (lm % traits.SMEM_N_PER_WAVE) * traits.KV_VEC_SIZE
    )


def _vec_v_const_off(traits, dc, k_substep):
    """Constant element offset from lane base for vectorized V pack (dc, k_substep)."""
    return dc * (
        traits.D_CHUNK // traits.SMEM_N_PER_WAVE
    ) * traits.VEC_V_ROW_STRIDE + k_substep * (2 * traits.D_128B_SIZE)


def _swizzled_v_dc_off(traits, dc):
    """Non-vectorized V LDS dc-axis offset (swizzled axis0/axis1 decomposition)."""
    return (dc // 2) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
        dc % 2
    ) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _swizzled_v_imm_lo(traits, dc, k_substep):
    """Non-vectorized V LDS byte offset for (dc, k_substep) in bf16 elements."""
    return (
        k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE
        + _swizzled_v_dc_off(traits, dc)
    ) * traits.BF16_BYTES


def _ds_read_tr_v4f16_imm(
    lds_base_elem_idx, imm_bytes, lds_kv_base_idx, v_lds_read_vec4_type
):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(v_lds_read_vec4_type, addr_i32, imm_bytes)


def _seq_pad_col_base(traits, tile_idx, lane_div_32):
    """Base KV column for this lane's lo-half scores at `tile_idx`."""
    _lane_n_off = 8 if traits.KV_VECTORIZED else 4
    return fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(lane_div_32) * fx.Int32(
        _lane_n_off
    )


def _seq_pad_score_threshold(traits, r):
    """Column threshold offset for score row `r` (layout-dependent swizzle)."""
    if const_expr(traits.KV_VECTORIZED):
        return (r // 8) * 16 + (r % 8)
    return (r // 4) * 8 + (r % 4)


def _k_dma_m0_base(traits, buf_id, d, lane_in_warp, lds_kv_base_idx, wave_id_uni):
    k_lds_byte_base = lds_kv_base_idx + _k_buf_base(traits, buf_id) * traits.BF16_BYTES
    if const_expr(traits.KV_VECTORIZED):
        oct_idx = (
            wave_id_uni * (traits.WARP_SIZE * traits.SMEM_D_RPT)
            + d * traits.WARP_SIZE
            + lane_in_warp
        )
        lds_addr = k_lds_byte_base + oct_idx * (traits.KV_VEC_SIZE * traits.BF16_BYTES)
    else:
        lds_addr = (
            k_lds_byte_base
            + wave_id_uni * (traits.SMEM_K_LINE_STRIDE * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * traits.SMEM_K_LINE_STRIDE * traits.BF16_BYTES)
        )
    return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))


def _v_dma_m0_base(traits, buf_id, d, lane_in_warp, lds_kv_base_idx, wave_id_uni):
    v_lds_byte_base = lds_kv_base_idx + _v_buf_base(traits, buf_id) * traits.BF16_BYTES
    if const_expr(traits.KV_VECTORIZED):
        row = wave_id_uni * traits.SMEM_D_RPT + d
        lds_elem = row * traits.VEC_V_ROW_STRIDE + lane_in_warp * traits.KV_VEC_SIZE
        lds_addr = v_lds_byte_base + lds_elem * traits.BF16_BYTES
    else:
        lds_addr = (
            v_lds_byte_base
            + wave_id_uni * (traits.SMEM_V_LINE_STRIDE * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE * traits.BF16_BYTES)
        )
    return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))


def _splitk_workspace_split_z(traits, batch_idx, split_idx):
    return batch_idx * traits.NUM_KV_SPLITS + split_idx


def _splitk_workspace_resources(
    ws_base_i64,
    split_z,
    ws_opart_per_split_bytes,
    ws_ml_per_split_bytes,
    ws_mrow_abs_bytes,
    ws_lrow_abs_bytes,
):
    opart_rsrc = _make_ws_rsrc(
        ws_base_i64, split_z * ws_opart_per_split_bytes, ws_opart_per_split_bytes
    )
    mrow_rsrc = _make_ws_rsrc(
        ws_base_i64,
        ws_mrow_abs_bytes + split_z * ws_ml_per_split_bytes,
        ws_ml_per_split_bytes,
    )
    lrow_rsrc = _make_ws_rsrc(
        ws_base_i64,
        ws_lrow_abs_bytes + split_z * ws_ml_per_split_bytes,
        ws_ml_per_split_bytes,
    )
    return opart_rsrc, mrow_rsrc, lrow_rsrc


def _splitk_local_opart_row_base(traits, q_head_idx, seq_len_v, q_row):
    return (q_head_idx * seq_len_v + q_row) * fx.Index(traits.HEAD_DIM // 2)


def _splitk_local_ml_idx(q_head_idx, seq_len_v, q_row):
    return q_head_idx * seq_len_v + q_row


def _splitk_o_partial_dword_col(traits, dc, g, lane_div_32):
    return dc * (traits.D_CHUNK // 2) + (2 * g + lane_div_32) * 4


def _store_empty_splitk_o_partial_row(
    traits, local_opart_base, lane_div_32, opart_rsrc
):
    c_zero_i = fx.Int32(0)
    for dc in range_constexpr(traits.D_CHUNKS):
        for g in range_constexpr(2):
            _ws_store_quad_i32(
                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                local_opart_base
                + _splitk_o_partial_dword_col(traits, dc, g, lane_div_32),
                opart_rsrc,
            )


def _store_splitk_ml_row(m_row, l_row, local_ml_idx, mrow_rsrc, lrow_rsrc):
    _ws_store_f32(m_row, local_ml_idx, mrow_rsrc)
    _ws_store_f32(l_row, local_ml_idx, lrow_rsrc)


def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Shared verbatim by DualwaveKernelContext and DualwaveFp8KernelContext."""
    traits = ctx.traits
    batch_interleave_group = getattr(traits, "BATCH_INTERLEAVE_GROUP", 1)
    # Swizzled Head-first Mapping (arXiv:2511.02132): the grid is head-fast, so one
    # head's q-blocks scatter across all XCDs and each re-streams its K/V. Re-derive
    # (head, q_block) with head as the slow axis to keep them on one XCD. Bijective,
    # so output is bit-identical; split-K's third grid axis would not survive it.
    # Non-causal only: under a causal mask q-block i does work proportional to i, so
    # making q_block the fast axis clusters unequal work and costs 7% (measured).
    if const_expr(
        traits.XCD_SWIZZLE
        and not traits.SPLITK
        and not traits.CAUSAL
        and traits.NUM_HEADS_Q % NUM_XCD_GFX950 == 0
    ):
        num_q_blocks = fx.Index(gpu.grid_dim.y)
        linear_wg = fx.Index(gpu.block_idx.x) + fx.Index(gpu.block_idx.y) * fx.Index(
            traits.NUM_HEADS_Q
        )
        ctx.h_idx = linear_wg // num_q_blocks
        ctx.q_block_idx = linear_wg % num_q_blocks
    elif const_expr(batch_interleave_group > 1 and not traits.SPLITK):
        # Keep a bounded batch group in grid X so the final causal q-block
        # occupies more CUs without abandoning K/V locality across all batches.
        linear_head_batch = fx.Index(gpu.block_idx.x)
        if const_expr(
            getattr(traits, "PAGED", False)
            and getattr(traits, "PAIRED_PAGE_IDS", False)
        ):
            ctx.h_idx = linear_head_batch // batch_interleave_group
            ctx.batch_idx = (
                fx.Int64(gpu.block_idx.z) * batch_interleave_group
                + linear_head_batch % batch_interleave_group
            )
        else:
            ctx.h_idx = linear_head_batch % traits.NUM_HEADS_Q
            batch_in_group = linear_head_batch // traits.NUM_HEADS_Q
            ctx.batch_idx = (
                fx.Int64(gpu.block_idx.z) * batch_interleave_group + batch_in_group
            )
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    else:
        ctx.h_idx = fx.Index(gpu.block_idx.x)
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(traits.SPLITK):
        ctx.bz_idx = fx.Index(gpu.block_idx.z)
        ctx.batch_idx = ctx.bz_idx // traits.NUM_KV_SPLITS
        ctx.split_idx = ctx.bz_idx % traits.NUM_KV_SPLITS
    else:
        if const_expr(batch_interleave_group <= 1):
            ctx.batch_idx = fx.Int64(gpu.block_idx.z)
        ctx.split_idx = None
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = fx.Int32(ctx.tid)
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        (_tid_i32 // fx.Int32(traits.WARP_SIZE)).ir_value(),
    )
    # Two stagger groups, whatever the wave count.
    ctx.stagger_i32 = (
        fx.Int32(_wave_id_uni_i32) // fx.Int32(traits.NUM_WAVES // 2)
    ).ir_value()
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    num_heads_kv = getattr(ctx, "num_heads_kv", None)
    if num_heads_kv is None:
        ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
        ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
        ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    else:
        # Runtime head counts (modification 3). Only the default mapping, where the
        # grid's x is the query head, reads them; the swizzled and batch-interleaved
        # mappings still use the build-time count.
        if const_expr(
            traits.XCD_SWIZZLE or batch_interleave_group > 1 or traits.SPLITK
        ):
            raise ValueError(
                "runtime head counts serve only the default dual-wave grid mapping"
            )
        # In 32 bits: an Index divide is a 64-bit one, many times the instructions,
        # and every workgroup pays it before its first load.
        h_i32 = fx.Int32(ctx.h_idx)
        num_heads_kv_i32 = fx.Int32(num_heads_kv)
        group_size_i32 = fx.Int32(ctx.num_heads_q) // num_heads_kv_i32
        h_kv_i32 = h_i32 % num_heads_kv_i32
        group_id_i32 = h_i32 // num_heads_kv_i32
        ctx.h_kv_idx = fx.Index(h_kv_i32)
        ctx.group_id = fx.Index(group_id_i32)
        ctx.q_head_idx = fx.Index(h_kv_i32 * group_size_i32 + group_id_i32)
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


# Traits and factory


@dataclass(frozen=True)
class FlashAttnGenericTraits:
    """Compile-time constants for the generic flash-attention kernel."""

    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    HEAD_DIM: int
    DTYPE_STR: str
    CAUSAL: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    PAGED: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    PATH_TAG: str
    N_SUBTILES: int
    ENABLE_PREFETCH_3BUF: bool
    ENABLE_DMA: bool
    ENABLE_LDS_VEC16: bool
    REDUCE_MODE: str
    NUM_PREFETCH_K: int
    NUM_PREFETCH_V: int
    CK_LDS_SEQ: tuple[int, ...]
    USE_HW_TR: bool
    USE_K16: bool
    USE_PERMLANE_OSTORE: bool
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    KV_VEC_SIZE: int
    PAGE_SIZE: int
    PAGE_STRIDE_VEC: int
    HEAD_STRIDE_VEC: int
    STRIDE_TOKEN_Q: int
    STRIDE_TOKEN_KV: int
    K_STRIDE: int
    K_SWZ_ROWMASK: int
    VT_STRIDE: int
    V_STRIDE: int
    VEC_V_LINE: int
    VEC_V_D128: int
    VEC_V_NGROUPS: int
    TOTAL_V8: int
    NV8_PER_THREAD: int
    V_NOMAJOR_DMA: bool
    VEC_WIDTH: int
    THREADS_PER_ROW_LOAD: int
    ROWS_PER_BATCH_LOAD: int
    NUM_BATCHES_KV: int
    KV_NEEDS_GUARD: bool
    LDS_K_TILE_SIZE: int
    LDS_V_TILE_SIZE: int
    LDS_K_TOTAL_SIZE: int
    LDS_V_BASE: int
    LDS_V_TOTAL_SIZE: int
    LDS_KV_TOTAL_SIZE: int
    WAVES_PER_EU: int
    DAZ: bool
    FAST_FP_MATH: bool
    UNSAFE_FP_MATH: bool
    SM_SCALE: float | None
    SKIP_KV_PAD_MASK: bool
    ENABLE_GFX942_DMA: bool
    ENABLE_GFX942_KV_GPFETCH: bool
    ENABLE_GFX942_VEC_K: bool
    V_PERM_TR: bool
    K_VEC_SIZE: int
    K_VEC_N_STRIDE: int
    K_VEC_HI_N_OFFSET: int
    QK_PREFETCH_DEPTH: int
    RETURN_LSE: bool = False
    # V's and O's head dim (modification 13); HEAD_DIM's unless narrower (MLA).
    HEAD_DIM_V: int = 0
    STRIDE_TOKEN_V: int = 0
    # Q and K columns at or past a runtime head dim read as zero (modification 13).
    QK_MASK: bool = False
    # Keys split across workgroups, each writing a partial O and LSE (modification 14).
    SPLIT_KV: bool = False
    # A GQA group's query heads packed into the row tile (modification 15).
    PACK_GQA: bool = False
    # O's columns split across DV_SPLIT workgroups (grid z), each DV_TILE wide, staging only
    # its slice of V through V_WIDTH-wide rows with a load mapping of its own; Q re-read
    # per K-step past head_dim 256 (modification 16).
    DV_SPLIT: int = 1
    DV_TILE: int = 0
    V_WIDTH: int = 0
    THREADS_PER_ROW_LOAD_V: int = 0
    ROWS_PER_BATCH_LOAD_V: int = 0
    NUM_BATCHES_V: int = 0
    V_NEEDS_GUARD: bool = False
    Q_STREAM: bool = False
    # V and O as wide as Q and K's runtime head dim (modification 17): a qk_mask build
    # whose V is the bucket's width serves any head dim up to it, V included.
    V_FOLLOWS_QK: bool = False
    # K staged K_CHUNKS column chunks at a time, each K_STEPS_QK steps of the QK product
    # (modification 18): a head too wide for the LDS takes several passes per KV tile.
    K_CHUNKS: int = 1
    # A K or V row whose width is not a power of two is loaded as the next power of two
    # wide: lanes whose columns start at or past K_COLS / V_WIDTH read nothing past the
    # row and store nothing to the LDS (modification 26).
    K_COLS: int = 0
    K_LOAD_GUARD: bool = False
    V_LOAD_GUARD: bool = False
    # f32 (modification 35): bytes per Q/K/V/O element (2 for f16/bf16, 4 for f32).
    ELEM_BYTES: int = 2

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.GQA_GROUP_SIZE,
            self.HEAD_DIM,
            self.HEAD_DIM_V,
            self.QK_MASK,
            self.SPLIT_KV,
            self.PACK_GQA,
            self.DV_SPLIT,
            self.Q_STREAM,
            self.V_FOLLOWS_QK,
            self.K_CHUNKS,
            self.DTYPE_STR,
            self.CAUSAL,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.PAGED,
            self.KV_CACHE_LAYOUT,
            self.KV_VECTORIZED,
            False,  # SPLITK is not supported by the generic builder.
            1,
            self.BLOCK_M,
            self.BLOCK_N,
            self.BLOCK_N_OUT,
            self.K_SUB_N,
            self.WARP_SIZE,
            self.NUM_WAVES,
            self.BLOCK_SIZE,
            self.ROWS_PER_WAVE,
            self.PATH_TAG,
            self.N_SUBTILES,
            self.ENABLE_PREFETCH_3BUF,
            self.ENABLE_DMA,
            self.ENABLE_LDS_VEC16,
            self.REDUCE_MODE,
            self.NUM_PREFETCH_K,
            self.NUM_PREFETCH_V,
            self.CK_LDS_SEQ,
            self.USE_HW_TR,
            self.USE_K16,
            self.USE_PERMLANE_OSTORE,
            self.K_STEP_QK,
            self.K_STEPS_QK,
            self.D_CHUNK,
            self.D_CHUNKS,
            self.PV_K_STEP,
            self.PV_K_STEPS,
            self.MFMA_LANE_K,
            self.KV_VEC_SIZE,
            self.PAGE_SIZE,
            self.PAGE_STRIDE_VEC,
            self.HEAD_STRIDE_VEC,
            self.STRIDE_TOKEN_Q,
            self.STRIDE_TOKEN_KV,
            self.K_STRIDE,
            self.K_SWZ_ROWMASK,
            self.VT_STRIDE,
            self.V_STRIDE,
            self.VEC_V_LINE,
            self.VEC_V_D128,
            self.VEC_V_NGROUPS,
            self.TOTAL_V8,
            self.NV8_PER_THREAD,
            self.V_NOMAJOR_DMA,
            self.VEC_WIDTH,
            self.THREADS_PER_ROW_LOAD,
            self.ROWS_PER_BATCH_LOAD,
            self.NUM_BATCHES_KV,
            self.KV_NEEDS_GUARD,
            self.LDS_K_TILE_SIZE,
            self.LDS_V_TILE_SIZE,
            self.LDS_K_TOTAL_SIZE,
            self.LDS_V_BASE,
            self.LDS_V_TOTAL_SIZE,
            self.LDS_KV_TOTAL_SIZE,
            self.WAVES_PER_EU,
            self.DAZ,
            self.FAST_FP_MATH,
            self.UNSAFE_FP_MATH,
            self.SM_SCALE,
            self.SKIP_KV_PAD_MASK,
            self.ENABLE_GFX942_DMA,
            self.ENABLE_GFX942_KV_GPFETCH,
            self.ENABLE_GFX942_VEC_K,
            self.V_PERM_TR,
            self.K_VEC_SIZE,
            self.K_VEC_N_STRIDE,
            self.K_VEC_HI_N_OFFSET,
            self.QK_PREFETCH_DEPTH,
            self.RETURN_LSE,
        )


def _make_flash_attn_generic_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    gpu_arch,
    causal=True,
    dtype_str="f16",
    flat_work_group_size=None,
    block_m=None,
    path_tag="auto",
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    waves_per_eu=2,
    daz=True,
    fast_fp_math=True,
    unsafe_fp_math=True,
    sm_scale=None,
    skip_kv_pad_mask=None,
    return_lse=False,
    k_pad=None,
    vt_pad=None,
    head_dim_v=None,
    qk_mask=False,
    split_kv=False,
    pack_gqa=False,
    dv_split=1,
    k_chunk=None,
):
    """Build compile-time traits for ``flash_attn_generic``."""
    # f32 (modification 35): exact-f32 operands take the gfx942-class path on every arch (no HW
    # transpose, no K16, no permlane O store, no DMA, no KV global prefetch).
    is_f32 = dtype_str == "f32"
    block_n = 64
    k_sub_n = 32
    warp_size = 64

    block_m = 128 if block_m is None else block_m
    if flat_work_group_size is None:
        flat_work_group_size = 256 if block_m <= 128 else 512
    num_waves = flat_work_group_size // warp_size
    block_size = flat_work_group_size
    rows_per_wave = block_m // num_waves

    if path_tag.upper() in ("N32", "N128"):
        path = path_tag.upper()
    elif (
        dtype_str in ("f16", "bf16")
        and causal
        and head_dim == 128
        # The masked and narrow-V loads are on the N32 path (modification 13).
        and not qk_mask
        and head_dim_v in (None, head_dim)
        and int(dv_split) == 1
    ):
        path = "N128"
    else:
        path = "N32"
    block_n_out = 128 if path == "N128" else block_n
    n_subtiles = block_n_out // block_n

    enable_prefetch_3buf = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    )
    has_lds_load_b128 = not _is_gfx942_class(gpu_arch)
    enable_gfx942_dma = _is_gfx942_class(gpu_arch) and (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_DMA", "0") == "1"
    )
    enable_dma = enable_gfx942_dma or (
        has_lds_load_b128
        and (
            path == "N128"
            or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
        )
    )
    if is_f32:
        enable_gfx942_dma = False
        enable_dma = False
    enable_gfx942_kv_gpfetch = (
        not is_f32
        and _is_gfx942_class(gpu_arch)
        and not enable_prefetch_3buf
        and (not enable_dma or enable_gfx942_dma)
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_KV_GPFETCH", "1") == "1"
    )
    enable_gfx942_vec_k = (
        _is_gfx942_class(gpu_arch)
        and enable_dma
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_VEC_K", "1") == "1"
    )
    enable_lds_vec16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    reduce_mode = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if reduce_mode not in ("xor", "ds_bpermute"):
        reduce_mode = "xor"
    if skip_kv_pad_mask is None:
        skip_kv_pad_mask = (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_SKIP_KV_PAD_MASK", "0") == "1"
        )
    else:
        skip_kv_pad_mask = bool(skip_kv_pad_mask)
    num_prefetch_k = (
        3
        if enable_prefetch_3buf
        else (2 if (enable_dma and not enable_gfx942_dma) else 1)
    )
    num_prefetch_v = 3 if enable_prefetch_3buf else 1
    ck_lds_seq = (1, 2, 0, 1, 0, 1, 2, 0) if enable_prefetch_3buf else (0,)

    use_hw_tr = gpu_arch.startswith("gfx950") and not is_f32
    use_k16 = gpu_arch.startswith("gfx950") and not is_f32
    use_permlane_ostore = gpu_arch.startswith("gfx950") and not is_f32
    k_step_qk = 16 if use_k16 else 8
    # K's side -- its LDS tile, load mapping, swizzle and the QK steps per pass -- is a
    # k_chunk-wide column chunk of the head, staged K_CHUNKS times per KV tile
    # (modification 18); the whole head otherwise.
    k_side = head_dim if k_chunk is None else int(k_chunk)
    assert head_dim % k_side == 0
    k_chunks = head_dim // k_side
    k_steps_qk = k_side // k_step_qk
    d_chunk = 32
    # V and O are head_dim_v wide (modification 13): the PV product and the O
    # accumulators cover only those columns; V is still staged head_dim wide.
    head_dim_v = head_dim if head_dim_v is None else int(head_dim_v)
    assert 32 <= head_dim_v <= head_dim and head_dim_v % d_chunk == 0
    # O's columns split across workgroups (modification 16): each holds DV_TILE of them.
    dv_split = int(dv_split)
    assert head_dim_v % (dv_split * d_chunk) == 0
    dv_tile = head_dim_v // dv_split
    d_chunks = dv_tile // d_chunk
    # V rows are staged V_WIDTH wide: the whole head as before, or, when V is narrower or
    # split, only this workgroup's slice.
    v_width = head_dim if (head_dim_v == head_dim and dv_split == 1) else dv_tile
    # Past 256 the Q packs would fill the register file: re-read per K-step.
    # f32's packs are twice as wide: from 256 they are re-read too (modification 35).
    q_stream = head_dim > 256 or (is_f32 and head_dim >= 256)
    # A Q/K bucket whose V is as wide: V and O follow the runtime head dim (modification
    # 17). The merge kernel and the column split know only baked widths.
    v_follows_qk = bool(qk_mask) and head_dim_v == head_dim
    assert not (v_follows_qk and (split_kv or dv_split > 1))
    pv_k_step = 16 if use_k16 else 8
    pv_k_steps = k_sub_n // pv_k_step
    mfma_lane_k = 8 if use_k16 else 4

    paged = bool(paged)
    kv_vectorized = paged and (kv_cache_layout == "vectorized")
    kv_vec_size = 8
    page_size = block_n
    page_stride_vec = num_kv_heads * head_dim * page_size
    head_stride_vec = head_dim * page_size

    stride_token_q = num_heads * head_dim
    stride_token_kv = num_kv_heads * head_dim
    stride_token_v = num_kv_heads * head_dim_v
    _k_pad_default = (
        4 if ((_is_gfx942_class(gpu_arch) or is_f32) and not enable_dma) else 0
    )
    _k_pad = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_K_PAD", str(_k_pad_default)))
    if k_pad is not None:
        _k_pad = int(k_pad)
    k_stride = k_side + _k_pad
    # The K tile's XOR swizzle, keyed by key row. The reader keys both halves of the tile
    # by lane % 32, so the mask must leave the half's bit (32) out: past head_dim 512 it
    # would not, and the upper 32 rows would be read through another swizzle
    # (modification 16).
    k_swz_rowmask = min(k_side // 16 - 1, 31)
    # And at a width that is not a power of two, inside the largest power of two dividing
    # it, so the XOR keeps every column in its row (modification 26).
    k_swz_rowmask = min(k_swz_rowmask, (k_side & -k_side) // 16 - 1)
    if use_hw_tr:
        vt_stride = 0
        v_stride = v_width if enable_dma else v_width + 4
    else:
        vt_stride = block_n + (2 if vt_pad is None else int(vt_pad))
        v_stride = vt_stride

    vec_v_line = 544
    vec_v_d128 = 64
    vec_v_ngroups = block_n // 8
    if kv_vectorized:
        total_v8 = vec_v_ngroups * head_dim
        assert total_v8 % block_size == 0
        nv8_per_thread = total_v8 // block_size
        assert (
            not enable_prefetch_3buf
        ), "KV_VECTORIZED no-major V unsupported with 3-buffer prefetch"
        v_nomajor_dma = os.getenv("FLYDSL_FLASH_ATTN_FUNC_VEC_V_DMA", "1") == "1"
    else:
        total_v8 = 0
        nv8_per_thread = 0
        v_nomajor_dma = False

    vec_width = 16 if enable_lds_vec16 else 8
    if is_f32:
        # 8 f32 = 32 bytes per lane, the 16-bit path's 16 elements' bytes.
        vec_width = 8
    assert k_side % vec_width == 0
    # A row that is not a power of two wide is loaded as the next power of two wide, the
    # lanes past its end guarded (modification 26).
    k_load_width = 1 << (k_side - 1).bit_length()
    k_load_guard = k_load_width != k_side
    assert not (k_load_guard and k_chunks > 1), "a K chunk must be a power of two wide"
    threads_per_row_load = k_load_width // vec_width
    assert block_size % threads_per_row_load == 0
    rows_per_batch_load = block_size // threads_per_row_load
    if rows_per_batch_load >= block_n:
        num_batches_kv = 1
        kv_needs_guard = rows_per_batch_load > block_n
    else:
        assert block_n % rows_per_batch_load == 0
        num_batches_kv = block_n // rows_per_batch_load
        kv_needs_guard = False

    # The masked and narrow-V loads (modification 13) are on the plain load paths.
    narrow = head_dim_v != head_dim or bool(qk_mask) or dv_split > 1 or k_chunks > 1
    # Guarded rows run on the plain load paths too (modification 26).
    narrow = narrow or k_load_guard or (v_width & (v_width - 1)) != 0
    if narrow and (enable_dma or paged):
        raise NotImplementedError(
            "a narrow V or a runtime Q/K head dim needs the dense, non-DMA path"
        )
    # V's own load mapping over V_WIDTH (modification 16); K's when V is staged whole.
    v_load_width = 1 << (v_width - 1).bit_length()
    v_load_guard = v_load_width != v_width
    assert not (
        v_load_guard and dv_split > 1
    ), "a V column slice must be a power of two wide"
    threads_per_row_load_v = v_load_width // vec_width
    assert block_size % threads_per_row_load_v == 0
    rows_per_batch_load_v = block_size // threads_per_row_load_v
    if rows_per_batch_load_v >= block_n:
        num_batches_v = 1
        v_needs_guard = rows_per_batch_load_v > block_n
    else:
        assert block_n % rows_per_batch_load_v == 0
        num_batches_v = block_n // rows_per_batch_load_v
        v_needs_guard = False

    v_perm_tr_env = os.getenv("FLYDSL_FLASH_ATTN_FUNC_V_PERM_TR", "1") == "1"
    v_perm_tr = (
        not narrow
        and v_perm_tr_env
        and (not use_hw_tr)
        and (not enable_dma)
        and (not enable_prefetch_3buf)
        and dtype_str == "bf16"
        and vec_width == 16
        and num_batches_kv == 1
        and (not kv_needs_guard)
        and (vt_stride == block_n + 2)
        and (block_n % 2 == 0)
    )
    k_vec_size = k_step_qk
    k_vec_n_stride = block_n * k_vec_size
    k_vec_hi_n_offset = k_sub_n * k_vec_size
    if enable_gfx942_kv_gpfetch and not enable_gfx942_vec_k:
        qk_prefetch_depth = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_QK_PF_DEPTH", "3"))
    else:
        qk_prefetch_depth = (
            4
            if (enable_gfx942_kv_gpfetch or enable_gfx942_vec_k)
            else (3 if enable_dma else 2)
        )

    lds_k_tile_size = block_n * k_stride
    if kv_vectorized:
        lds_v_tile_size = (head_dim // 8) * vec_v_line
    elif use_hw_tr:
        lds_v_tile_size = block_n * v_stride
    else:
        lds_v_tile_size = v_width * vt_stride
    lds_k_total_size = num_prefetch_k * lds_k_tile_size
    lds_v_base = lds_k_total_size
    lds_v_total_size = num_prefetch_v * lds_v_tile_size
    lds_kv_total_size = lds_k_total_size + lds_v_total_size

    return FlashAttnGenericTraits(
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        DTYPE_STR=dtype_str,
        CAUSAL=bool(causal),
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        PAGED=paged,
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n_out,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        PATH_TAG=path,
        N_SUBTILES=n_subtiles,
        ENABLE_PREFETCH_3BUF=enable_prefetch_3buf,
        ENABLE_DMA=enable_dma,
        ENABLE_LDS_VEC16=enable_lds_vec16,
        REDUCE_MODE=reduce_mode,
        NUM_PREFETCH_K=num_prefetch_k,
        NUM_PREFETCH_V=num_prefetch_v,
        CK_LDS_SEQ=ck_lds_seq,
        USE_HW_TR=use_hw_tr,
        USE_K16=use_k16,
        USE_PERMLANE_OSTORE=use_permlane_ostore,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEP=pv_k_step,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=mfma_lane_k,
        KV_VEC_SIZE=kv_vec_size,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_VEC=page_stride_vec,
        HEAD_STRIDE_VEC=head_stride_vec,
        STRIDE_TOKEN_Q=stride_token_q,
        STRIDE_TOKEN_KV=stride_token_kv,
        K_STRIDE=k_stride,
        K_SWZ_ROWMASK=k_swz_rowmask,
        VT_STRIDE=vt_stride,
        V_STRIDE=v_stride,
        VEC_V_LINE=vec_v_line,
        VEC_V_D128=vec_v_d128,
        VEC_V_NGROUPS=vec_v_ngroups,
        TOTAL_V8=total_v8,
        NV8_PER_THREAD=nv8_per_thread,
        V_NOMAJOR_DMA=v_nomajor_dma,
        VEC_WIDTH=vec_width,
        THREADS_PER_ROW_LOAD=threads_per_row_load,
        ROWS_PER_BATCH_LOAD=rows_per_batch_load,
        NUM_BATCHES_KV=num_batches_kv,
        KV_NEEDS_GUARD=kv_needs_guard,
        LDS_K_TILE_SIZE=lds_k_tile_size,
        LDS_V_TILE_SIZE=lds_v_tile_size,
        LDS_K_TOTAL_SIZE=lds_k_total_size,
        LDS_V_BASE=lds_v_base,
        LDS_V_TOTAL_SIZE=lds_v_total_size,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        FAST_FP_MATH=bool(fast_fp_math),
        UNSAFE_FP_MATH=bool(unsafe_fp_math),
        SM_SCALE=sm_scale,
        SKIP_KV_PAD_MASK=skip_kv_pad_mask,
        ENABLE_GFX942_DMA=enable_gfx942_dma,
        ENABLE_GFX942_KV_GPFETCH=enable_gfx942_kv_gpfetch,
        ENABLE_GFX942_VEC_K=enable_gfx942_vec_k,
        V_PERM_TR=v_perm_tr,
        K_VEC_SIZE=k_vec_size,
        K_VEC_N_STRIDE=k_vec_n_stride,
        K_VEC_HI_N_OFFSET=k_vec_hi_n_offset,
        QK_PREFETCH_DEPTH=qk_prefetch_depth,
        RETURN_LSE=bool(return_lse),
        HEAD_DIM_V=head_dim_v,
        STRIDE_TOKEN_V=stride_token_v,
        QK_MASK=bool(qk_mask),
        SPLIT_KV=bool(split_kv),
        PACK_GQA=bool(pack_gqa),
        DV_SPLIT=dv_split,
        DV_TILE=dv_tile,
        V_WIDTH=v_width,
        THREADS_PER_ROW_LOAD_V=threads_per_row_load_v,
        ROWS_PER_BATCH_LOAD_V=rows_per_batch_load_v,
        NUM_BATCHES_V=num_batches_v,
        V_NEEDS_GUARD=v_needs_guard,
        Q_STREAM=q_stream,
        V_FOLLOWS_QK=v_follows_qk,
        K_COLS=k_side,
        K_LOAD_GUARD=k_load_guard,
        V_LOAD_GUARD=v_load_guard,
        K_CHUNKS=k_chunks,
        ELEM_BYTES=4 if is_f32 else 2,
    )


@dataclass(frozen=True)
class DualwaveSwpTraits:
    """Pure compile-time tile/layout constants for gfx950 DUALWAVE_SWP."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    PAGED: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DMA_BYTES: int
    BF16_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    SMEM_LINEAR_WAVE: int
    SMEM_N_PER_WAVE: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_PAD: int
    SMEM_V_PAD: int
    SMEM_K_LINE_STRIDE: int
    SMEM_V_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    SMEM_V_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    K_LDS_TO_REG_N_STRIP_STRIDE: int
    K_LDS_TO_REG_KSTEP_INNER_STRIDE: int
    K_LDS_TO_REG_KSTEP_OUTER_STRIDE: int
    V_LDS_TO_REG_HALF_WAVE_STRIDE: int
    V_LDS_TO_REG_LANE_QUAD_STRIDE: int
    V_LDS_TO_REG_N_GROUP_STRIDE: int
    V_LDS_TO_REG_LANE_IN_QUAD_STRIDE: int
    V_LDS_TO_REG_K_SUBSTEP_STRIDE: int
    V_LDS_TO_REG_DCHUNK_PAIR_STRIDE: int
    V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE: int
    V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE: int
    PAGED_BT_LDS_SIZE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    KV_VEC_SIZE: int
    VEC_V_ROW_STRIDE: int
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int
    RETURN_LSE: bool = False
    XCD_SWIZZLE: bool = False

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.PAGED,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.KV_CACHE_LAYOUT,
            self.KV_VECTORIZED,
            self.RETURN_LSE,
            self.XCD_SWIZZLE,
        )


def _make_dualwave_swp_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    kv_vectorized=None,
    return_lse=False,
    xcd_swizzle=False,
):
    """Build gfx950 DUALWAVE_SWP compile-time layout traits."""
    # Tile shape and wave geometry follow the gfx950 dual-wave 8-wave CTA.
    block_m = DUALWAVE_SWP_BLOCK_M
    block_n = 64
    block_n_out = 64
    k_sub_n = 32
    warp_size = 64
    num_waves = 8
    block_size = num_waves * warp_size
    rows_per_wave = 32

    # QK walks D in 16-wide MFMA K steps; PV consumes K_SUB_N in two 16-token steps.
    k_step_qk = 16
    k_steps_qk = head_dim // k_step_qk
    d_chunk = 32
    d_chunks = head_dim // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step
    mfma_lane_k = 8

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    # Global K/V DMA is 16B per lane; D_128B_SIZE is one 128B row in bf16 elements.
    dma_bytes = 16
    bf16_bytes = 2
    d_128b_size = 64
    vec_kv = 8
    smem_linear_wave = warp_size * 16 // bf16_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // d_128b_size
    # K/V LDS tiles are wave-linear rows with padding to avoid bank-aligned repeats.
    smem_k_pad = 16 // bf16_bytes
    smem_v_pad = 64 // bf16_bytes
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride
    num_prefetch_k = 2
    dualwave_swp_kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = (0, dualwave_swp_kv_per_buffer)
    dualwave_swp_v_buf_base = (
        smem_k_tile_elems,
        smem_k_tile_elems + dualwave_swp_kv_per_buffer,
    )

    # K LDS->VGPR reads: hi half jumps one N strip; ks uses inner stride then d_rpt stride.
    k_lds_to_reg_n_strip_stride = 256
    k_lds_to_reg_kstep_inner_stride = 16
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride
    # V LDS->VGPR base is decomposed by half-wave, lane quad, N group, and lane-in-quad.
    v_lds_to_reg_half_wave_stride = 2176
    v_lds_to_reg_lane_quad_stride = smem_v_line_stride
    v_lds_to_reg_n_group_stride = 16
    v_lds_to_reg_lane_in_quad_stride = 4
    # V read immediates step across K substeps, D-chunk pairs, and transpose-load pairs.
    v_lds_to_reg_k_substep_stride = 128
    v_lds_to_reg_dchunk_pair_stride = smem_n_rpt * smem_v_line_stride
    v_lds_to_reg_dchunk_in_pair_stride = 32
    v_lds_to_reg_transpose_pair_stride = d_128b_size
    # Vectorized KV path keeps one 16B vector per lane and reuses the V LDS row stride.
    kv_vec_size = 16 // bf16_bytes
    vec_v_row_stride = smem_v_line_stride
    splitk = num_kv_splits > 1
    paged = bool(paged)
    varlen = bool(varlen)
    cross_seqlen = bool(cross_seqlen)

    return DualwaveSwpTraits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n_out,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEP=pv_k_step,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=mfma_lane_k,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(dualwave_swp_debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        PAGED=paged,
        VARLEN=varlen,
        CROSS_SEQLEN=cross_seqlen,
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=dma_bytes,
        BF16_BYTES=bf16_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=vec_kv,
        SMEM_LINEAR_WAVE=smem_linear_wave,
        SMEM_N_PER_WAVE=smem_n_per_wave,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_PAD=smem_k_pad,
        SMEM_V_PAD=smem_v_pad,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_lds_to_reg_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=v_lds_to_reg_lane_quad_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_lds_to_reg_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=v_lds_to_reg_lane_in_quad_stride,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_lds_to_reg_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=v_lds_to_reg_dchunk_pair_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_lds_to_reg_dchunk_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=v_lds_to_reg_transpose_pair_stride,
        PAGED_BT_LDS_SIZE=2048,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        KV_VEC_SIZE=kv_vec_size,
        VEC_V_ROW_STRIDE=vec_v_row_stride,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
        RETURN_LSE=bool(return_lse),
        XCD_SWIZZLE=bool(xcd_swizzle),
    )


@dataclass(frozen=True)
class DualwaveSwpFp8Traits:
    """Pure compile-time tile/layout constants for the gfx950 DUALWAVE_SWP fp8 kernel."""

    BLOCK_M: int
    BLOCK_N: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DEFAULT_STRIDE_V_N: int
    DEFAULT_STRIDE_O_N: int
    QLDS: bool
    K_BAND_CHUNK: tuple[int, ...]
    K_BAND_BASE: tuple[int, ...]
    K_BAND_LINE_STRIDE: tuple[int, ...]
    K_BAND_GLOBAL_D: tuple[int, ...]
    K_WS_BAND: tuple[int, ...]
    K_WS_OFF: tuple[int, ...]
    DMA_BYTES: int
    ELEM_BYTES: int
    OUT_ELEM_BYTES: int
    VEC_KV: int
    LANE_SPLIT_KV: int
    SMEM_K_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    VT_BF16_TOTAL: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    SCHED_DS_READ_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int
    XCD_SWIZZLE: bool = False
    BATCH_INTERLEAVE_GROUP: int = 1
    PAGED: bool = False
    KV_VECTORIZED: bool = False
    FP8_PV_SEGMENTED: bool = False

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.HEAD_DIM_V,
            self.QLDS,
            self.K_BAND_CHUNK,
            "fp8_wide_qk_hiprec_pv",
            self.ELEM_BYTES,
            self.OUT_ELEM_BYTES,
            self.LANE_SPLIT_KV,
            self.VT_BF16_TOTAL,
            self.NUM_PREFETCH_K,
            self.XCD_SWIZZLE,
            self.BATCH_INTERLEAVE_GROUP,
            self.BLOCK_M,
            self.BLOCK_SIZE,
            self.NUM_WAVES,
        )


def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    rescale_threshold,
    head_dim_v=None,
    block_m=256,
    causal=True,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    xcd_swizzle=False,
    batch_interleave_group=1,
):
    """Build gfx950 DUALWAVE_SWP fp8 compile-time layout traits.

    ``head_dim`` is the QK reduction width (a multiple of 64: the QK MFMA is
    32x32x64) and ``head_dim_v`` the V/output width, tiled in 32-wide D_CHUNKs.
    """
    if head_dim_v is None:
        head_dim_v = head_dim
    if head_dim % 64:
        raise RuntimeError(
            f"fp8 flash attention needs head_dim % 64 == 0, got head_dim={head_dim}"
        )
    # D_CHUNKS == head_dim_v // 32 must land in [2, 6]: below 2 `_anchor_v_o`
    # aborts LLVM, above 6 the high D_CHUNKs come back wrong.
    if head_dim_v % 32 or not 64 <= head_dim_v <= 192:
        raise RuntimeError(
            "fp8 flash attention needs 64 <= head_dim_v <= 192 and head_dim_v % 32 == 0, "
            f"got head_dim_v={head_dim_v} (head_dim={head_dim})"
        )
    block_n = 64
    k_sub_n = 32
    warp_size = 64
    rows_per_wave = 32
    if block_m % rows_per_wave or block_m // rows_per_wave not in (4, 8):
        raise RuntimeError(
            f"fp8 flash attention supports block_m 128 (4 waves) or 256 (8 waves), got {block_m}"
        )
    num_waves = block_m // rows_per_wave
    block_size = num_waves * warp_size

    d_chunk = 32
    d_chunks = head_dim_v // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim
    default_stride_v_n = num_kv_heads * head_dim_v
    default_stride_o_n = num_heads * head_dim_v

    # fp8: Q/K/V are 1B; O is bf16 (2B). ELEM_BYTES=1 drives the fp8 address math.
    elem_bytes = 1
    out_elem_bytes = 2
    vec_kv = 16 // elem_bytes
    lane_split_kv = 8
    smem_linear_wave = warp_size * 16 // elem_bytes
    smem_k_pad = 16 // elem_bytes
    smem_k_line_stride = smem_linear_wave + smem_k_pad

    rows_per_wave_dma = -(-block_n // num_waves)
    k_band_chunk, k_band_base, k_band_line_stride, k_band_global_d = [], [], [], []
    _off, _cursor = 0, 0
    while _off < head_dim:
        chunk = 128 if head_dim - _off >= 128 else head_dim - _off
        line_stride = rows_per_wave_dma * chunk + smem_k_pad
        k_band_chunk.append(chunk)
        k_band_base.append(_cursor)
        k_band_line_stride.append(line_stride)
        k_band_global_d.append(_off)
        _cursor += num_waves * line_stride
        _off += chunk
    smem_k_tile_elems = _cursor
    k_ws_band, k_ws_off = [], []
    for _bi, chunk in enumerate(k_band_chunk):
        for _o in range(0, chunk, 64):
            k_ws_band.append(_bi)
            k_ws_off.append(_o)
    num_prefetch_k = 6
    dualwave_swp_kv_per_buffer = smem_k_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = tuple(
        i * dualwave_swp_kv_per_buffer for i in range(num_prefetch_k)
    )
    dualwave_swp_v_buf_base = tuple(
        smem_k_tile_elems + i * dualwave_swp_kv_per_buffer
        for i in range(num_prefetch_k)
    )

    # The +128 covers the alignment the DMA base is rounded up to.
    eb_bf = 2
    fp8_v_tile_bytes = (block_n // 8) * (head_dim_v // 16) * 128
    vt_bf16_total = num_prefetch_k * (fp8_v_tile_bytes // eb_bf) + 128

    splitk = num_kv_splits > 1

    qlds = head_dim <= 128

    lds_bytes = lds_kv_total_size * elem_bytes + vt_bf16_total * eb_bf
    if qlds:
        lds_bytes += block_m * head_dim * elem_bytes
    if lds_bytes > LDS_BYTES_GFX950:
        raise RuntimeError(
            f"fp8 flash attention head_dim={head_dim}/head_dim_v={head_dim_v} at block_m={block_m} "
            f"needs {lds_bytes} B of LDS, over the {LDS_BYTES_GFX950} B gfx950 workgroup limit. "
            "Largest head_dim_v that fits: 192 at head_dim 64/128/192, 160 at 256, 96 at 320; "
            "head_dim 384 and above never fits."
        )

    return DualwaveSwpFp8Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR="fp8",
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(dualwave_swp_debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DEFAULT_STRIDE_V_N=default_stride_v_n,
        DEFAULT_STRIDE_O_N=default_stride_o_n,
        QLDS=bool(qlds),
        K_BAND_CHUNK=tuple(k_band_chunk),
        K_BAND_BASE=tuple(k_band_base),
        K_BAND_LINE_STRIDE=tuple(k_band_line_stride),
        K_BAND_GLOBAL_D=tuple(k_band_global_d),
        K_WS_BAND=tuple(k_ws_band),
        K_WS_OFF=tuple(k_ws_off),
        DMA_BYTES=16,
        ELEM_BYTES=elem_bytes,
        OUT_ELEM_BYTES=out_elem_bytes,
        VEC_KV=vec_kv,
        LANE_SPLIT_KV=lane_split_kv,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        VT_BF16_TOTAL=vt_bf16_total,
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        SCHED_DS_READ_MASK=0x100,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
        XCD_SWIZZLE=bool(xcd_swizzle),
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
    )


def dualwave_fp8_dma_per_iter(traits):
    rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
    k_instr = sum(
        -(-(rows_per_wave * (chunk // traits.VEC_KV)) // traits.WARP_SIZE)
        for chunk in traits.K_BAND_CHUNK
    )
    num_dma_v = (traits.BLOCK_N * (traits.HEAD_DIM_V // 16) * 16) // (
        traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
    )
    v_instr_min = num_dma_v // traits.NUM_WAVES
    return 2 * k_instr + 2 * v_instr_min


@dataclass(frozen=True)
class PagedDualwaveSwpFp8Traits:
    """Layouts for native paged FP8 QK/PV with register-resident Q."""

    HEAD_DIM: int
    HEAD_DIM_V: int
    D_CHUNKS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    FP8_PV_SEGMENTED: bool
    FP8_V_H1: int
    FP8_V_H2: int
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    SMEM_D_RPT: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, ...]
    VT_BF16_TOTAL: int
    FP8_V_ROW_STRIDE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    VARLEN: bool = True
    PAIRED_PAGE_IDS: bool = True
    BATCH_INTERLEAVE_GROUP: int = 1
    BLOCK_M: int = 256
    BLOCK_N: int = 64
    WARP_SIZE: int = 64
    NUM_WAVES: int = 8
    BLOCK_SIZE: int = 512
    ROWS_PER_WAVE: int = 32
    D_CHUNK: int = 32
    PV_K_STEPS: int = 2
    CAUSAL: bool = True
    DTYPE_STR: str = "fp8"
    NUM_KV_SPLITS: int = 1
    SPLITK: bool = False
    CROSS_SEQLEN: bool = True
    PAGED: bool = True
    KV_VECTORIZED: bool = True
    PAGE_SIZE: int = 64
    KV_CACHE_LAYOUT: str = "vectorized"
    CACHE_BUFFERED: bool = False
    DMA_BYTES: int = 16
    ELEM_BYTES: int = 1
    OUT_ELEM_BYTES: int = 2
    VEC_KV: int = 16
    KV_VEC_SIZE: int = 16
    LANE_SPLIT_KV: int = 8
    SCHED_MFMA_MASK: int = 0x008
    SCHED_DS_READ_MASK: int = 0x100
    NEG_INF_F32_BITS: int = 0xFF800000
    XCD_SWIZZLE: bool = False
    QLDS: bool = False

    @property
    def cache_tag(self):
        return (
            "paged_fp8_native",
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.HEAD_DIM_V,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            self.VARLEN,
            self.NUM_PREFETCH_K,
            self.FP8_V_ROW_STRIDE,
            self.BATCH_INTERLEAVE_GROUP,
            self.PAIRED_PAGE_IDS,
            self.PAGE_SIZE,
            self.KV_CACHE_LAYOUT,
            self.CACHE_BUFFERED,
        )

    @property
    def K_LDS_PAGE_GROUPED(self):
        return self.PAGE_SIZE == 16 and self.HEAD_DIM_V == 192

    @property
    def V_LOAD_LAYOUT(self):
        """Producer word order; the matching store and byte mask must agree."""
        if self.PAGE_SIZE == 1:
            return (
                "token_transpose"
                if (self.HEAD_DIM, self.HEAD_DIM_V) == (192, 128)
                else "token_words"
            )
        if self.PAGE_SIZE == 16 and self.HEAD_DIM == 192:
            if self.HEAD_DIM_V == 192 or not self.CACHE_BUFFERED:
                return "page_waves"
        return "lane_groups"

    @property
    def SHARED_PAGE_ID_LAYOUT(self):
        if self.PAGE_SIZE == 1 and self.HEAD_DIM_V == 192:
            return "tokens"
        if self.PAGE_SIZE == 16 and self.CACHE_BUFFERED:
            return self.V_LOAD_LAYOUT
        return None


def _make_paged_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    value_head_dim,
    rescale_threshold,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    batch_interleave_group=1,
    paired_page_ids=True,
    page_size=64,
    kv_cache_layout="vectorized",
    cache_buffered=False,
):
    """Build layouts after the dedicated builder validates the paged contract."""
    block_n = 64
    smem_d_rpt = (head_dim + 127) // 128
    # Retain padded K-slot spacing; V192 below compacts the unused D256 tail.
    smem_k_tile_elems = 8 * smem_d_rpt * 1040
    compact_v192 = value_head_dim == 192
    if compact_v192:
        # Six K/V slots fit in 160 KiB only when K omits its unused D256 tail.
        smem_k_tile_elems = block_n * head_dim
    num_prefetch_k = 8 if head_dim == 128 else 6
    slot_elems = smem_k_tile_elems
    # The builder accepts V128/V192: one 128-channel prefix and an optional V64 tail.
    fp8_v_h1, fp8_v_h2 = 128, value_head_dim - 128
    fp8_v_row_stride = block_n if compact_v192 else block_n + 16
    fp8_v_tile_bytes = value_head_dim * fp8_v_row_stride

    return PagedDualwaveSwpFp8Traits(
        HEAD_DIM=head_dim,
        HEAD_DIM_V=value_head_dim,
        D_CHUNKS=value_head_dim // 32,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=num_heads // num_kv_heads,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        FP8_PV_SEGMENTED=fp8_v_h2 > 0,
        FP8_V_H1=fp8_v_h1,
        FP8_V_H2=fp8_v_h2,
        PAIRED_PAGE_IDS=bool(paired_page_ids),
        PAGE_SIZE=page_size,
        KV_CACHE_LAYOUT=kv_cache_layout,
        CACHE_BUFFERED=bool(cache_buffered),
        DEFAULT_STRIDE_Q_N=num_heads * head_dim,
        DEFAULT_STRIDE_KV_N=num_kv_heads * head_dim,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=slot_elems,
        LDS_KV_TOTAL_SIZE=num_prefetch_k * slot_elems,
        DUALWAVE_SWP_K_BUF_BASE=tuple(i * slot_elems for i in range(num_prefetch_k)),
        VT_BF16_TOTAL=num_prefetch_k * (fp8_v_tile_bytes // 2) + 128,
        FP8_V_ROW_STRIDE=fp8_v_row_stride,
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
    )


# Kernel context


def _lse_records(lse_on, lse_bytes):
    """The LSE buffer's record count: all of it, or none when the runtime `lse_on`
    argument is 0, so every LSE store falls outside the buffer and is dropped
    (modification 4). None keeps the build-time choice."""
    if lse_on is None:
        return lse_bytes
    return fx.Index(
        (fx.Int32(lse_on) != fx.Int32(0)).select(fx.Int32(lse_bytes), fx.Int32(0))
    )


class GenericFlashAttnContext:
    """Runtime setup state for the generic flash-attention kernel."""

    def __init__(
        self,
        traits,
        K,
        V,
        seq_len,
        seq_len_kv,
        num_heads_q=None,
        num_heads_kv=None,
        stride_token_q=None,
        stride_token_kv=None,
        lse_on=None,
        diag_offset=None,
        view_strides=None,
        window_lo=None,
        stride_token_v=None,
        head_dim_qk=None,
        num_splits=None,
        ws_o=None,
        ws_lse=None,
        sink=None,
        sink_stride=None,
        sink_on=None,
        lse_strides=None,
        varlen_flags=None,
    ):
        self.traits = traits
        # Per-batch lengths (modification 32): `VARLEN_*` bits, or None for a build that
        # takes the launch's lengths alone. Set by init_sequence_lengths: the diagonal's
        # per-batch shift.
        self.varlen_flags = varlen_flags
        self.diag_shift_i32 = None
        # Dropout (modification 33): set by init_dropout on a dropout build.
        self.dropout = False
        # The LSE output's (batch, head, query) element strides (modification 20); None
        # keeps the packed [B, H, Sq] layout.
        self.lse_strides = lse_strides
        # Attention sinks (modification 19): a per-query-head logit in the softmax's
        # denominator, read unless sink_on is SINK_OFF, in the dtype it names
        # (modification 36).
        self.sink = sink
        self.sink_stride = sink_stride
        self.sink_on = sink_on
        # The KV split (modification 14): how many, and the partials' workspace.
        self.num_splits = num_splits
        self.ws_o = ws_o
        self.ws_lse = ws_lse
        self.K = K
        self.V = V
        # V's own token stride (modification 13); None keeps K's.
        self.stride_token_v = stride_token_v
        # The actual Q/K head dim, read by a QK_MASK build (modification 13).
        self.head_dim_qk = head_dim_qk
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        # Runtime head counts and token strides (modification 3); None keeps the
        # traits' build-time value.
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.stride_token_q = stride_token_q
        self.stride_token_kv = stride_token_kv
        self.lse_on = lse_on
        self.diag_offset = diag_offset
        # Runtime (batch, head) strides of Q, O and K/V, and O's token stride
        # (modification 6); None keeps the BSHD layout the token strides imply.
        self.view_strides = view_strides
        # Sliding window (modification 11): query row i sees no key j < i + window_lo.
        # None, or any value at or below -seq_len_kv, is no window.
        self.window_lo = window_lo

    def init_types_and_pointers(self):
        traits = self.traits
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.elem_type = self.elem_dtype.ir_type
        self.compute_type = fx.Float32.ir_type
        self.k_ptr = _extract_aligned_pointer(self.K)
        self.v_ptr = _extract_aligned_pointer(self.V)
        self.load_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.v1i32_type = Vec.make_type(1, fx.Int32)

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v8f16_type = Vec.make_type(8, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.mfma_pack_type = self.v8f16_type if traits.USE_K16 else self.v4f16_type
        self.MFMA_LANE_K = traits.MFMA_LANE_K
        self.mma_atom_k16 = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))

    def init_sequence_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        traits = self.traits

        def _shape(value, built):
            return fx.Index(built if value is None else value)

        self.num_heads_q_v = _shape(self.num_heads_q, traits.NUM_HEADS_Q)
        self.num_heads_kv_v = _shape(self.num_heads_kv, traits.NUM_HEADS_KV)
        self.runtime_heads = self.num_heads_q is not None
        if self.runtime_heads:
            # Head arithmetic in 32 bits: an Index divide is a 64-bit one.
            self.num_heads_q_i32 = fx.Int32(self.num_heads_q)
            self.gqa_group_i32 = self.num_heads_q_i32 // fx.Int32(self.num_heads_kv)
        self.stride_token_q_v = _shape(self.stride_token_q, traits.STRIDE_TOKEN_Q)
        self.stride_token_kv_v = _shape(self.stride_token_kv, traits.STRIDE_TOKEN_KV)
        self.stride_token_v_v = (
            self.stride_token_kv_v
            if self.stride_token_v is None
            else fx.Index(self.stride_token_v)
        )
        if const_expr(traits.QK_MASK):
            self.head_dim_qk_v = fx.Index(self.head_dim_qk)

    def init_lds_view(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds_kv = lds.kv.ptr
        self.lds_kv_offset = fx.Index(fx.ptrtoint(lds.kv.ptr))

    def init_thread_mapping(self):
        traits = self.traits
        self.block_id = fx.Index(gpu.block_idx.x)
        self.tid = fx.Index(gpu.thread_idx.x)

        self.wave_id = self.tid // traits.WARP_SIZE
        self.lane = self.tid % traits.WARP_SIZE
        self.lane_mod_32 = self.lane % 32
        self.lane_div_32 = self.lane // 32

        self.tr_k_group = (self.lane % 16) // 4
        self.tr_col_sub = self.lane % 4
        self.tr_col_half = (self.lane % 32) // 16
        self.wave_q_offset = self.wave_id * traits.ROWS_PER_WAVE

    def init_block_mapping(self):
        traits = self.traits
        # This workgroup's slice of O's columns (modification 16): grid z.
        if const_expr(traits.DV_SPLIT > 1):
            self.dv_idx = fx.Index(gpu.block_idx.z)
            self.dv_col_base = self.dv_idx * fx.Index(traits.DV_TILE)
        else:
            self.dv_idx = None
            self.dv_col_base = fx.Index(0)
        if const_expr(traits.SPLIT_KV):
            # One grid row per split (modification 14).
            self.num_splits_v = fx.Index(self.num_splits)
            self.split_idx = fx.Index(gpu.block_idx.y)
        if const_expr(traits.PACK_GQA):
            # One workgroup per (batch, row tile, KV head); its rows are the group's
            # query heads by position, head-major: row r is query head
            # kv_head * group + r // Sq at position r % Sq (modification 15).
            group = fx.Index(self.gqa_group_i32)
            self.pack_rows = group * self.seq_len_v
            row_tiles = (self.pack_rows + fx.Index(traits.BLOCK_M - 1)) // fx.Index(
                traits.BLOCK_M
            )
            num_kv = fx.Index(self.num_heads_kv)
            self.kv_head_idx = self.block_id % num_kv
            batch_tile = self.block_id // num_kv
            self.q_tile_idx = batch_tile % row_tiles
            self.batch_idx = batch_tile // row_tiles
            self.num_q_tiles = row_tiles
            self.q_start = self.q_tile_idx * traits.BLOCK_M
            # The group's first query head: Q's, O's and the LSE's bases.
            self.q_head_idx = self.kv_head_idx * group
            # Every row of a tile may sit at any position.
            self.pos_first = fx.Index(0)
            self.pos_end = self.seq_len_v
            return
        if self.runtime_heads:
            block_id_i32 = fx.Int32(self.block_id)
            q_head_i32 = block_id_i32 % self.num_heads_q_i32
            self.q_head_idx = fx.Index(q_head_i32)
            self.batch_q_tile_id = fx.Index(block_id_i32 // self.num_heads_q_i32)
            self.kv_head_idx_i32 = q_head_i32 // self.gqa_group_i32
        else:
            self.q_head_idx = self.block_id % traits.NUM_HEADS_Q
            self.batch_q_tile_id = self.block_id // traits.NUM_HEADS_Q
        self.num_q_tiles = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_tile_idx = self.batch_q_tile_id % self.num_q_tiles
        self.batch_idx = self.batch_q_tile_id // self.num_q_tiles
        self.q_start = self.q_tile_idx * traits.BLOCK_M
        self.pos_first = self.q_start
        self.pos_end = self.q_start + traits.BLOCK_M
        if self.runtime_heads:
            self.kv_head_idx = fx.Index(self.kv_head_idx_i32)
        else:
            self.kv_head_idx = (
                self.q_head_idx
                if traits.GQA_GROUP_SIZE == 1
                else self.q_head_idx // traits.GQA_GROUP_SIZE
            )

    def init_load_mapping(self):
        traits = self.traits
        self.load_row_in_batch = self.tid // traits.THREADS_PER_ROW_LOAD
        self.load_lane_in_row = self.tid % traits.THREADS_PER_ROW_LOAD
        self.load_col_base = self.load_lane_in_row * traits.VEC_WIDTH
        # V's own mapping over its staged width (modification 16); K's when they agree.
        if const_expr(traits.THREADS_PER_ROW_LOAD_V == traits.THREADS_PER_ROW_LOAD):
            self.load_row_in_batch_v = self.load_row_in_batch
            self.load_col_base_v = self.load_col_base
        else:
            self.load_row_in_batch_v = self.tid // traits.THREADS_PER_ROW_LOAD_V
            self.load_col_base_v = (
                self.tid % traits.THREADS_PER_ROW_LOAD_V
            ) * traits.VEC_WIDTH
        if const_expr(traits.V_PERM_TR):
            self.vp_rows_per_thread = 2
            self.vp_num_row_groups = traits.BLOCK_N // self.vp_rows_per_thread
            self.vp_active_threads = (
                traits.THREADS_PER_ROW_LOAD * self.vp_num_row_groups
            )
            self.vp_col_group = self.tid % traits.THREADS_PER_ROW_LOAD
            self.vp_row_group = self.tid // traits.THREADS_PER_ROW_LOAD
            self.vp_col_base = self.vp_col_group * traits.VEC_WIDTH
            self.vp_row_base = self.vp_row_group * self.vp_rows_per_thread
            self.vp_sel_lo = fx.Int32(0x05040100)
            self.vp_sel_hi = fx.Int32(0x07060302)

    def init_constants(self, sm_scale):
        traits = self.traits
        self.c_neg_inf = fx.Float32(float("-inf"))
        # The running max of a row no key has reached yet, over raw scores. -3e38 times
        # sm_scale * log2e overflows to -inf once the scale passes about 0.78, and
        # fma(-inf, scale, +inf) is NaN; -1e30 stays finite (modification 22).
        self.c_neg_floor = fx.Float32(-1.0e30)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_i32 = fx.Int32(0)
        self.c_sm_scale = fx.Float32(sm_scale)
        if isinstance(sm_scale, float):
            self.c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        else:
            # A runtime scale argument (modification 2).
            self.c_sm_scale_log2e = self.c_sm_scale * fx.Float32(_LOG2E)
        if getattr(self, "bias_rsrc", None) is not None:
            # A bias build scales its scores and adds the bias before the softmax, as
            # the reference does (modification 28), so the running max, the exponent
            # and the LSE are in the scaled domain: the scale left for them is 1. A
            # row the bias masks at the dtype's lowest value stays finite there, and
            # only a row of -inf meets the floor.
            self.c_score_scale = self.c_sm_scale
            self.c_sm_scale = fx.Float32(1.0)
            self.c_sm_scale_log2e = fx.Float32(_LOG2E)
            self.c_neg_floor = fx.Float32(_BIAS_NEG_FLOOR)
            self.c_masked_score = fx.Float32(_BIAS_MASKED_SCORE)
            if self.bias_kstrided:
                # Halved scores (modification 31): an f32 bias reaches -FLT_MAX.
                self.c_score_scale = self.c_score_scale * fx.Float32(0.5)
                self.c_sm_scale = fx.Float32(2.0)
                self.c_sm_scale_log2e = fx.Float32(2.0 * _LOG2E)
                self.c_neg_floor = fx.Float32(_BIAS_HALVED_NEG_FLOOR)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        self.width_i32 = fx.Int32(traits.WARP_SIZE)
        self.shuf_32_i32 = fx.Int32(32)
        self.lane_xor_32_byte = (fx.Int32(self.lane) ^ self.shuf_32_i32) * fx.Int32(4)

    def init_sequence_lengths(self, CuSeqQ, CuSeqKv, row_offs=None):
        traits = self.traits
        # Packed sequences' first tokens and the rows a store may write (modification
        # 34): None for a build that reads no row offsets.
        self.packed_tok = None
        self.view_q_b = self.seq_len_v
        if const_expr(traits.VARLEN):
            cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1)
            )
            cuk_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1)
            )
            self.q_tok_base = _cu_load(
                cuq_div, self.batch_idx, self.load_atom_32, self.v1i32_type
            )
            self.kv_tok_base = _cu_load(
                cuk_div, self.batch_idx, self.load_atom_32, self.v1i32_type
            )
            self.seqlen_q_b = (
                _cu_load(
                    cuq_div,
                    self.batch_idx + fx.Index(1),
                    self.load_atom_32,
                    self.v1i32_type,
                )
                - self.q_tok_base
            )
            self.seqlen_kv_b = (
                _cu_load(
                    cuk_div,
                    self.batch_idx + fx.Index(1),
                    self.load_atom_32,
                    self.v1i32_type,
                )
                - self.kv_tok_base
            )
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.seqlen_q_b = self.seq_len_v
            self.seqlen_kv_b = self.seq_len_kv_v
            if self.varlen_flags is not None and row_offs is not None:
                # Packed sequences too (modification 34): a prefill build reads each
                # sequence's first token and token count from the row-offset tables when
                # the launch says so. Its Q, O, K, V and LSE views start at those tokens,
                # over the batch's rows only -- the rows after them are the next
                # sequence's, which no store may reach.
                varlen = _load_varlen(
                    self.varlen_flags,
                    (CuSeqQ, CuSeqKv),
                    row_offs,
                    self.batch_idx,
                    self.seq_len,
                    self.seq_len_kv,
                )
                self.diag_shift_i32 = varlen.shift
                self.seqlen_q_b = fx.Index(varlen.len_q)
                self.seqlen_kv_b = fx.Index(varlen.len_kv)
                self.view_q_b = fx.Index(varlen.view_q)
                self.packed_tok = (fx.Index(varlen.tok_q), fx.Index(varlen.tok_kv))
            elif self.varlen_flags is not None:
                # Padding (modification 32): this batch's own lengths, read when the
                # launch says so, replace the launch's for everything past the grid --
                # the Q, K and V views, the KV loop's bounds, the masks and the bias's
                # view -- and its bottom-right diagonal moves with them. The batch and
                # head bases keep the launch's strides.
                len_q, len_kv, self.diag_shift_i32 = _load_seq_lens(
                    self.varlen_flags,
                    (CuSeqQ, CuSeqKv),
                    self.batch_idx,
                    self.seq_len,
                    self.seq_len_kv,
                )
                self.seqlen_q_b = fx.Index(len_q)
                self.seqlen_kv_b = fx.Index(len_kv)

    def init_kv_batch_pointers(self):
        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        traits = self.traits
        strides = self.view_strides
        if strides is not None and const_expr(traits.PAGED or traits.VARLEN):
            raise ValueError("runtime strides serve the dense kernel only")
        # The head's offset lives in the K/V base under runtime strides, and in
        # each lane's index otherwise.
        self.kv_head_byte_offset = (
            fx.Index(0)
            if strides is not None
            else self.kv_head_idx * fx.Index(traits.HEAD_DIM * traits.ELEM_BYTES)
        )
        if const_expr(not traits.PAGED):
            off = self.kv_tok_base * self.stride_token_kv_v
            v_off = self.kv_tok_base * self.stride_token_v_v
            if strides is not None:
                off = fx.Index(self.batch_idx) * fx.Index(
                    strides["kv_b"]
                ) + self.kv_head_idx * fx.Index(strides["kv_h"])
                # V's own batch and head strides (modification 13).
                v_off = off
                if "v_b" in strides:
                    v_off = fx.Index(self.batch_idx) * fx.Index(
                        strides["v_b"]
                    ) + self.kv_head_idx * fx.Index(strides["v_h"])
                if self.packed_tok is not None:
                    # A packed sequence starts at its first token (modification 34); 0
                    # unpacked, where the batch stride places it.
                    tok_kv = self.packed_tok[1]
                    off = off + tok_kv * self.stride_token_kv_v
                    v_off = v_off + tok_kv * self.stride_token_v_v
                if self.varlen_flags is not None:
                    # A batch or sequence with no keys runs no KV iteration, but a build
                    # that loads the first K tile before its loop still reads its clamped
                    # row, -1: before K's base for a view at the tensor's start. So a
                    # keyless view at token 0 (every unpacked one) moves one token on,
                    # making that row its own first (modifications 32, 34). A packed one
                    # further on reads the previous sequence's last token, in the tensor.
                    first_tok = (
                        fx.Index(0) if self.packed_tok is None else self.packed_tok[1]
                    )
                    at_start = (self.seqlen_kv_b == fx.Index(0)) & (
                        first_tok == fx.Index(0)
                    )
                    off = off + fx.Index(
                        at_start.select(self.stride_token_kv_v, fx.Index(0))
                    )
                    v_off = v_off + fx.Index(
                        at_start.select(self.stride_token_v_v, fx.Index(0))
                    )
            self.k_ptr = buffer_ops.get_element_ptr(
                self.k_ptr, off, elem_type=self.elem_type
            )
            self.v_ptr = buffer_ops.get_element_ptr(
                self.v_ptr, v_off, elem_type=self.elem_type
            )

    def init_descriptors(self, Q, O, LSE=None):  # noqa: E741
        # Per-batch descriptors keep global indices 0-based and bounded to one batch,
        # keeping 32-bit offsets small while preserving arbitrary-seqlen OOB behavior.
        traits = self.traits
        if self.view_strides is not None:
            self._init_strided_descriptors(Q, O)
        else:
            self._init_bshd_descriptors(Q, O)
        # LSE output: fp32 [batch, num_heads, seq_len], per-batch descriptor.
        if const_expr(traits.RETURN_LSE):
            if self.lse_strides is None:
                self.lse_stride_h_v = self.seq_len_v
                self.lse_stride_n_v = fx.Index(1)
                self.lse_per_batch_elems = self.num_heads_q_v * self.seq_len_v
                _lse_batch_elems = self.lse_per_batch_elems
            else:
                # One batch's rows, through the LSE's own strides (modification 20).
                self.lse_stride_h_v = fx.Index(self.lse_strides[1])
                self.lse_stride_n_v = fx.Index(self.lse_strides[2])
                self.lse_per_batch_elems = (
                    (self.num_heads_q_v - fx.Index(1)) * self.lse_stride_h_v
                    + (self.seq_len_v - fx.Index(1)) * self.lse_stride_n_v
                    + fx.Index(1)
                )
                _lse_batch_elems = fx.Index(self.lse_strides[0])
            # OOB sentinel (== num records) the buffer bound drops; predicates the store.
            self.lse_oob_off = self.lse_per_batch_elems
            _lse_nrec_bytes = as_mlir_value(
                _lse_records(self.lse_on, self.lse_per_batch_elems * fx.Index(4))
            )
            _lse_batch_elem_off = self.batch_idx * _lse_batch_elems
            if self.packed_tok is not None:
                # A packed LSE's sequence starts at its first query token, its batch
                # stride passed as 0 (modification 34); stores past the sequence's rows
                # are dropped (rows_b).
                _lse_batch_elem_off = (
                    _lse_batch_elem_off + self.packed_tok[0] * self.lse_stride_n_v
                )
            _lse_batch_byte_off = as_mlir_value(_lse_batch_elem_off * fx.Index(4))
            self.lse_rsrc = buffer_ops.create_buffer_resource(
                LSE,
                max_size=False,
                num_records_bytes=_lse_nrec_bytes,
                base_byte_offset=_lse_batch_byte_off,
            )

    def init_bias(
        self, Bias, bias_stride0, stride_bias_b, stride_bias_h, kstrided=None
    ):
        """The additive bias's descriptor (modification 12): one (batch, query head) slice,
        rebased at batch * stride_bias_b + head * stride_bias_h and ending at its last
        element, (seq_len - 1) * bias_stride0 + seq_len_kv, so rows and columns past the
        scores read as zero; a stride of 0 broadcasts that axis.

        A strided bias build (@p kstrided, its key stride and f32 switch; modification
        31) reads one key at a time: its slice ends at (seq_len - 1) * bias_stride0 +
        (seq_len_kv - 1) * stride_k + 1 elements of 2 bytes, or 4 for an f32 bias."""
        self.bias_stride0_v = fx.Index(bias_stride0)
        self.bias_kstrided = kstrided is not None
        if kstrided is None:
            span = (
                self.seqlen_q_b - fx.Index(1)
            ) * self.bias_stride0_v + self.seqlen_kv_b
            elem_bytes = fx.Index(2)
        else:
            stride_k, f32_on = kstrided
            self.bias_stride_k_i32 = fx.Int32(stride_k)
            self.bias_f32_i32 = fx.Int32(f32_on)
            span = (
                (self.seqlen_q_b - fx.Index(1)) * self.bias_stride0_v
                + (self.seqlen_kv_b - fx.Index(1)) * fx.Index(stride_k)
                + fx.Index(1)
            )
            elem_bytes = fx.Index(2) + fx.Index(2) * fx.Index(f32_on)
        if self.varlen_flags is not None:
            # A padded batch with no query rows or no keys reads no bias (modification
            # 32); the span above would go negative.
            span = _rows_bytes(
                fx.Index(
                    (self.seqlen_q_b < self.seqlen_kv_b).select(
                        self.seqlen_q_b, self.seqlen_kv_b
                    )
                ),
                span,
            )
        base = fx.Index(self.batch_idx) * fx.Index(
            stride_bias_b
        ) + self.q_head_idx * fx.Index(stride_bias_h)
        self.bias_rsrc = buffer_ops.create_buffer_resource(
            Bias,
            max_size=False,
            num_records_bytes=as_mlir_value(span * elem_bytes),
            base_byte_offset=as_mlir_value(base * elem_bytes),
        )

    def _init_strided_descriptors(self, Q, O):  # noqa: E741
        """Per-(batch, head) descriptors under runtime strides (modification 6).

        Each starts at its head's first row and ends at that head's last element,
        (seq_len - 1) * stride_n + head_dim, so any stride order reads rows past the
        sequence as zero and drops stores to them, as the per-batch BSHD
        descriptors did; the head offset lives in the base, not in each index.
        """
        traits = self.traits
        strides = self.view_strides
        head_dim = fx.Index(traits.HEAD_DIM)
        # Q ends at its actual head dim, O at V's (modification 13).
        q_width = self.head_dim_qk_v if const_expr(traits.QK_MASK) else head_dim
        o_width = (
            self.head_dim_qk_v
            if const_expr(traits.V_FOLLOWS_QK)
            else fx.Index(traits.HEAD_DIM_V)
        )
        batch = fx.Index(self.batch_idx)

        per_batch = self.varlen_flags is not None
        eb = traits.ELEM_BYTES  # f32 (modification 35)

        def span_bytes(rows, stride_n, width=head_dim, may_be_empty=False):
            span = ((rows - fx.Index(1)) * stride_n + width) * fx.Index(eb)
            # A padded batch's length may be 0 (modification 32).
            return as_mlir_value(_rows_bytes(rows, span) if may_be_empty else span)

        def base_bytes(stride_b, head, stride_h, tok=None, stride_n=None):
            base = batch * fx.Index(stride_b) + head * fx.Index(stride_h)
            if tok is not None:
                # A packed sequence's first token (modification 34); the host passes a
                # packed operand's batch stride as 0.
                base = base + tok * stride_n
            return as_mlir_value(base * fx.Index(eb))

        tok_q, tok_kv = (None, None) if self.packed_tok is None else self.packed_tok
        self.kv_nrec_bytes = span_bytes(
            self.seqlen_kv_b, self.stride_token_kv_v, may_be_empty=per_batch
        )
        self.kv_batch_byte_off = base_bytes(
            strides["kv_b"],
            self.kv_head_idx,
            strides["kv_h"],
            tok_kv,
            self.stride_token_kv_v,
        )
        self.stride_token_o_v = fx.Index(strides["o_n"])
        q_records = span_bytes(
            self.seqlen_q_b, self.stride_token_q_v, q_width, may_be_empty=per_batch
        )
        # O's view spans the launch's rows, padded ones included: a padded row is
        # stored as 0 (modification 32). Q's spans the batch's own, so a padded row
        # reads zeros. A packed sequence's O spans its own rows (modification 34).
        o_records = span_bytes(
            self.view_q_b,
            self.stride_token_o_v,
            o_width,
            may_be_empty=self.packed_tok is not None,
        )
        if const_expr(traits.PACK_GQA):
            # The group's heads too (modification 15).
            group_m1 = fx.Index(self.gqa_group_i32) - fx.Index(1)
            q_span = (
                group_m1 * fx.Index(strides["q_h"])
                + (self.seqlen_q_b - fx.Index(1)) * self.stride_token_q_v
                + q_width
            ) * fx.Index(eb)
            q_records = as_mlir_value(
                _rows_bytes(self.seqlen_q_b, q_span) if per_batch else q_span
            )
            o_records = as_mlir_value(
                (
                    group_m1 * fx.Index(strides["o_h"])
                    + (self.seq_len_v - fx.Index(1)) * self.stride_token_o_v
                    + o_width
                )
                * fx.Index(eb)
            )
        self.q_rsrc = buffer_ops.create_buffer_resource(
            Q,
            max_size=False,
            num_records_bytes=q_records,
            base_byte_offset=base_bytes(
                strides["q_b"],
                self.q_head_idx,
                strides["q_h"],
                tok_q,
                self.stride_token_q_v,
            ),
        )
        self.o_rsrc = buffer_ops.create_buffer_resource(
            O,
            max_size=False,
            num_records_bytes=o_records,
            base_byte_offset=base_bytes(
                strides["o_b"],
                self.q_head_idx,
                strides["o_h"],
                tok_q,
                self.stride_token_o_v,
            ),
        )

    def _init_bshd_descriptors(self, Q, O):  # noqa: E741
        self.stride_token_o_v = self.stride_token_q_v
        self.kv_nrec_bytes = as_mlir_value(
            self.seqlen_kv_b
            * (self.stride_token_kv_v * fx.Index(self.traits.ELEM_BYTES))
        )
        self.kv_batch_byte_off = as_mlir_value(
            self.kv_tok_base
            * (self.stride_token_kv_v * fx.Index(self.traits.ELEM_BYTES))
        )
        q_nrec_bytes = as_mlir_value(
            self.seqlen_q_b * (self.stride_token_q_v * fx.Index(self.traits.ELEM_BYTES))
        )
        q_batch_byte_off = as_mlir_value(
            self.q_tok_base * (self.stride_token_q_v * fx.Index(self.traits.ELEM_BYTES))
        )
        self.q_rsrc = buffer_ops.create_buffer_resource(
            Q,
            max_size=False,
            num_records_bytes=q_nrec_bytes,
            base_byte_offset=q_batch_byte_off,
        )
        self.o_rsrc = buffer_ops.create_buffer_resource(
            O,
            max_size=False,
            num_records_bytes=q_nrec_bytes,
            base_byte_offset=q_batch_byte_off,
        )

    def init_q_row(self):
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        self.q_row = self.q_start + self.wave_q_offset + self.lane_mod_32
        self.q_row_i32 = fx.Int32(self.q_row)
        # The rows a store may write: the query positions, or the packed group's rows.
        # The launch's, padded rows included (modification 32); a packed sequence's own
        # (modification 34), the rows after it being the next sequence's.
        self.rows_b = self.view_q_b
        if const_expr(self.traits.PACK_GQA):
            self.rows_b = self.pack_rows
            # The mask compares each row's position, not its packed index.
            self.q_row_i32 = fx.Int32(self.q_row % self.seq_len_v)
        # Padded rows are stored as O = 0, LSE = -inf (modification 32); padded_rows()
        # says which, at the stores, so nothing of it is live across the KV loop.

    def padded_rows(self):
        """(whether this lane's row is a real query position of its batch rather than
        padding, whether every row of the tile is -- always so on a dense launch), or
        None for a build without per-batch lengths (modification 32)."""
        if self.varlen_flags is None:
            return None
        len_q = fx.Int32(self.seqlen_q_b)
        row_live = self.q_row_i32 < len_q
        tile_live = len_q >= fx.Int32(self.seq_len)
        if const_expr(not self.traits.PACK_GQA):
            tile_live = tile_live | (
                fx.Int32(self.q_start) + fx.Int32(self.traits.BLOCK_M) <= len_q
            )
        return row_live, tile_live

    def init_dropout(
        self, seed, offset, offset2_lo, offset2_hi, threshold, scale
    ):  # noqa: PLR0913
        """A dropout build's Philox state (modification 33), AOTriton's scheme: the key
        is the 8-byte seed at @p seed's base; each run of 4 keys of a row draws one
        counter, offset1 + offset2 + ((b * Hq + h) * Sq + i) * cdiv(Skv, 4) + j / 4, with
        offset1 the 8 bytes at @p offset's base, offset2 the two words, and Sq, Skv the
        launch's lengths (a padded batch's mask is the dense one's); key j takes word
        j % 4 and is kept iff it, read as an int32, exceeds @p threshold. Kept
        probabilities are scaled by @p scale, 1 / (1 - p), folded into 1 / l."""
        traits = self.traits
        assert not (
            traits.PACK_GQA or traits.SPLIT_KV
        ), "a dropout build is a prefill build"
        self.dropout = True
        self.ph_key = _load_u64_words(seed)
        off_lo, off_hi = _load_u64_words(offset)
        offset_base = (
            (fx.Int64(off_hi) << fx.Int64(32))
            | (fx.Int64(off_lo) & fx.Int64(0xFFFFFFFF))
        ) + (
            (fx.Int64(offset2_hi) << fx.Int64(32))
            | (fx.Int64(offset2_lo) & fx.Int64(0xFFFFFFFF))
        )
        row_stride = fx.Int64((fx.Int32(self.seq_len_kv) + fx.Int32(3)) >> fx.Int32(2))
        plane = fx.Int64(self.batch_idx) * fx.Int64(self.num_heads_q_v) + fx.Int64(
            self.q_head_idx
        )
        # This lane's row's first counter: its plane's rows, then its own.
        self.ph_row_base = (
            offset_base
            + (plane * fx.Int64(self.seq_len) + fx.Int64(self.q_row)) * row_stride
        )
        self.ph_threshold = fx.Int32(threshold)
        self.dropout_scale = fx.Float32(scale)

    def init_kv_bounds(self):
        traits = self.traits
        q_end = self.pos_end
        if const_expr(traits.CAUSAL):
            # Query row i sees keys j <= i + delta: bottom-right unless the caller
            # passes its own offset (modification 5).
            if self.diag_offset is None:
                self.delta_i32 = fx.Int32(self.seqlen_kv_b) - fx.Int32(self.seqlen_q_b)
            else:
                self.delta_i32 = fx.Int32(self.diag_offset)
                if self.diag_shift_i32 is not None:
                    # A bottom-right diagonal at this batch's own lengths (modification
                    # 32); 0 for a dense launch or a top-left one.
                    self.delta_i32 = self.delta_i32 + self.diag_shift_i32
            self.causal_end_raw_i32 = fx.Int32(q_end) + self.delta_i32
            causal_end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(
                    self.causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_end = fx.Index(causal_end_i32)
            self.kv_upper = fx.Index(
                (causal_end < self.seqlen_kv_b).select(causal_end, self.seqlen_kv_b)
            )
            # The first KV tile any row of this block sees (modification 11): rows
            # see no key j < i + window_lo, so the loop starts at the KV-tile boundary
            # at or below q_start + window_lo.
            if const_expr(self.window_lo is not None):
                self.win_lo_i32 = fx.Int32(self.window_lo)
                if self.diag_shift_i32 is not None:
                    # The window's edge moves with the diagonal (modification 32).
                    self.win_lo_i32 = self.win_lo_i32 + self.diag_shift_i32
                lower_raw_i32 = fx.Int32(self.pos_first) + self.win_lo_i32
                lower_i32 = fx.Int32(
                    (lower_raw_i32 > fx.Int32(0)).select(lower_raw_i32, fx.Int32(0))
                )
                self.kv_lower = (
                    fx.Index(lower_i32) // fx.Index(traits.BLOCK_N_OUT)
                ) * fx.Index(traits.BLOCK_N_OUT)
            else:
                self.win_lo_i32 = None
                self.kv_lower = fx.Index(0)
        else:
            self.kv_upper = self.seqlen_kv_b
            self.win_lo_i32 = None
            self.kv_lower = fx.Index(0)
        if self.varlen_flags is not None and not traits.PACK_GQA:
            # A query tile wholly past its batch's length runs no KV iteration
            # (modification 32): its rows are padding, stored as O = 0, LSE = -inf.
            self.kv_upper = fx.Index(
                (self.q_start < self.seqlen_q_b).select(self.kv_upper, self.kv_lower)
            )
        if const_expr(traits.SPLIT_KV):
            # This split's share of the block's KV tiles, whole tiles each, so the
            # masks see the same tile boundaries as an unsplit launch (modification
            # 14). A split past the last tile runs no iteration and writes l = 0.
            bn = fx.Index(traits.BLOCK_N_OUT)
            span = (self.kv_upper > self.kv_lower).select(
                self.kv_upper - self.kv_lower, fx.Index(0)
            )
            tiles = (span + bn - fx.Index(1)) // bn
            per = (tiles + self.num_splits_v - fx.Index(1)) // self.num_splits_v
            lo = self.kv_lower + self.split_idx * per * bn
            hi = lo + per * bn
            self.kv_upper = fx.Index((hi < self.kv_upper).select(hi, self.kv_upper))
            self.kv_lower = lo

    def _packed_row_offset(self, row, stride_n, stride_h):
        """A packed row's element offset within the group's slice: its head's and its
        position's, or past any slice for a row beyond the group (modification 15)."""
        pos = row % self.seq_len_v
        head = row // self.seq_len_v
        off = head * fx.Index(stride_h) + pos * stride_n
        return (row < self.pack_rows).select(off, fx.Index(0x3FFFFFF0))

    def global_idx_q(self, token_idx, col):
        traits = self.traits
        if const_expr(traits.PACK_GQA):
            return (
                self._packed_row_offset(
                    token_idx, self.stride_token_q_v, self.view_strides["q_h"]
                )
                + col
            )
        if self.view_strides is not None:
            return token_idx * self.stride_token_q_v + col
        return (
            token_idx * self.stride_token_q_v + self.q_head_idx * traits.HEAD_DIM + col
        )

    def global_idx_o(self, token_idx, col):
        """O's element index: its own token stride under runtime strides. A column at or
        past a runtime V width maps past every slice, so its store is dropped
        (modification 17)."""
        if const_expr(self.traits.V_FOLLOWS_QK):
            idx = self._global_idx_o(token_idx, col)
            return (col < self.head_dim_qk_v).select(idx, fx.Index(0x3FFFFFF0))
        return self._global_idx_o(token_idx, col)

    def _global_idx_o(self, token_idx, col):
        if const_expr(self.traits.DV_SPLIT > 1):
            # This workgroup's slice of O's columns (modification 16).
            col = self.dv_col_base + col
        if const_expr(self.traits.PACK_GQA):
            return (
                self._packed_row_offset(
                    token_idx, self.stride_token_o_v, self.view_strides["o_h"]
                )
                + col
            )
        if self.view_strides is not None:
            return token_idx * self.stride_token_o_v + col
        return self.global_idx_q(token_idx, col)

    def global_idx_kv(self, token_idx, col):
        traits = self.traits
        if self.view_strides is not None:
            return token_idx * self.stride_token_kv_v + col
        return (
            token_idx * self.stride_token_kv_v
            + self.kv_head_idx * traits.HEAD_DIM
            + col
        )

    def global_idx_v(self, token_idx, col):
        """V's element index: its own token stride (modification 13)."""
        traits = self.traits
        if self.view_strides is not None:
            return token_idx * self.stride_token_v_v + col
        return (
            token_idx * self.stride_token_v_v
            + self.kv_head_idx * traits.HEAD_DIM_V
            + col
        )

    def kv_row_clamp(self, row_idx):
        last = self.seqlen_kv_b - fx.Index(1)
        return fx.Index((row_idx < self.seqlen_kv_b).select(row_idx, last))

    def load_global_half_vec(self, ptr, base_idx, vec_elems: int):
        gep = buffer_ops.get_element_ptr(
            ptr, fx.Int64(base_idx), elem_type=self.elem_type
        )
        return _pointer_load(Vec.make_type(vec_elems, self.elem_dtype), gep)

    def store_global_half(self, ptr, base_idx, val):
        gep = buffer_ops.get_element_ptr(
            ptr, fx.Int64(base_idx), elem_type=self.elem_type
        )
        _pointer_store(val, gep)

    def load_global_f16x4(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, 4)

    def load_global_mfma_pack(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, self.traits.MFMA_LANE_K)

    def load_global_f16xN(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, self.traits.VEC_WIDTH)

    def k_buf_base(self, buf_id):
        if const_expr(isinstance(buf_id, int)):
            return fx.Index(buf_id * self.traits.LDS_K_TILE_SIZE)
        return buf_id * fx.Index(self.traits.LDS_K_TILE_SIZE)

    def v_buf_base(self, buf_id):
        return fx.Index(self.traits.LDS_V_BASE + buf_id * self.traits.LDS_V_TILE_SIZE)

    def k_swizzle(self, row_idx, col_idx):
        mask = (row_idx & fx.Index(self.traits.K_SWZ_ROWMASK)) << fx.Index(4)
        return col_idx ^ mask

    def v_swizzle(self, row_idx, col_idx):
        mask = (row_idx & fx.Index(0x3)) << fx.Index(4)
        return col_idx ^ mask

    def sigma_kv(self, n):
        return (
            (n & fx.Index(3))
            | ((n & fx.Index(8)) >> fx.Index(1))
            | ((n & fx.Index(4)) << fx.Index(1))
            | (n & fx.Index(-16))
        )

    def k_vec_elem_idx(self, n, col_f16):
        traits = self.traits
        d_group = col_f16 // fx.Index(traits.K_VEC_SIZE)
        d_local = col_f16 % fx.Index(traits.K_VEC_SIZE)
        return (
            d_group * fx.Index(traits.K_VEC_N_STRIDE)
            + n * fx.Index(traits.K_VEC_SIZE)
            + d_local
        )


class GenericPageIdLoader:
    def __init__(self, ctx, BlockTable, block_table_stride):
        self.ctx = ctx
        self.BlockTable = BlockTable
        self.block_table_stride = block_table_stride
        if const_expr(ctx.traits.PAGED):
            self.bt_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(BlockTable), fx.make_layout(1, 1)
            )
            self.bt_stride_v = fx.Index(block_table_stride)

    def page_id(self, tile_start):
        ctx = self.ctx
        tile_idx = tile_start // fx.Index(ctx.traits.PAGE_SIZE)
        bt_off = ctx.batch_idx * self.bt_stride_v + tile_idx
        v = fly.copy_atom_call_ssa(
            [ctx.v1i32_type],
            ctx.load_atom_32,
            fx.slice(self.bt_div, (None, fx.Int32(bt_off))),
        )
        return fx.Index(Vec(v, (1,), fx.Int32)[0])


class GenericKvGmemToLdsLoader:
    """KV address/load leaf helper; the caller keeps pipeline scheduling local."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.page_ids = None

    def global_idx(self, token_idx, col):
        return self.ctx.global_idx_kv(token_idx, col)

    def load_k_cols(self, row_idx, col):
        """One lane's VEC_WIDTH K columns of a row. A QK_MASK build reads columns at or
        past the runtime head dim as zero, from column 0's address so nothing past the
        row is touched (modification 13)."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.QK_MASK):
            # In 8-column halves, so any multiple of 8 is served (modification 17).
            return self._masked_halves(
                ctx.k_ptr, lambda c: self.global_idx(row_idx, c), col, zero=True
            )
        if const_expr(traits.K_LOAD_GUARD):
            # A lane past the row's end reads column 0 instead; it stores nothing
            # (modification 26).
            ok = col < fx.Index(traits.K_COLS)
            col = ok.select(col, fx.Index(0))
        return self.load_f16xN(ctx.k_ptr, self.global_idx(row_idx, col))

    def _masked_halves(self, ptr, idx_fn, col, zero):
        """VEC_WIDTH columns of a row as 8-column halves, each read from column 0's
        address when it lies at or past the runtime head dim, and zeroed if @p zero."""
        ctx = self.ctx
        traits = ctx.traits
        halves = []
        for h in range_constexpr(traits.VEC_WIDTH // 8):
            c = col + fx.Index(h * 8)
            ok = c < ctx.head_dim_qk_v
            half = ctx.load_global_half_vec(ptr, idx_fn(ok.select(c, fx.Index(0))), 8)
            if const_expr(zero):
                half = ok.select(half, Vec.filled(8, 0.0, ctx.elem_dtype))
            halves.append(Vec(half))
        if const_expr(len(halves) == 1):
            return halves[0]
        return Vec.from_elements(
            [halves[h][i] for h in range(len(halves)) for i in range(8)], ctx.elem_dtype
        )

    def load_v_cols(self, row_idx, col):
        """One lane's VEC_WIDTH V columns of a row, through V's own strides. Columns at
        or past a narrower V's head dim are read from column 0's address instead: the PV
        product never consumes them (modification 13)."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.DV_SPLIT > 1):
            col = ctx.dv_col_base + col
        if const_expr(traits.V_FOLLOWS_QK):
            # V is the runtime head dim wide (modification 17), in 8-column halves.
            return self._masked_halves(
                ctx.v_ptr, lambda c: ctx.global_idx_v(row_idx, c), col, zero=False
            )
        if const_expr(traits.HEAD_DIM_V != traits.V_WIDTH * traits.DV_SPLIT):
            ok = col < fx.Index(traits.HEAD_DIM_V)
            col = ok.select(col, fx.Index(0))
        if const_expr(traits.V_LOAD_GUARD):
            # As K's (modification 26).
            ok = col < fx.Index(traits.V_WIDTH)
            col = ok.select(col, fx.Index(0))
        return self.load_f16xN(ctx.v_ptr, ctx.global_idx_v(row_idx, col))

    def row_clamp(self, row_idx):
        return self.ctx.kv_row_clamp(row_idx)

    def load_half_vec(self, ptr, base_idx, vec_elems: int):
        return self.ctx.load_global_half_vec(ptr, base_idx, vec_elems)

    def load_f16xN(self, rsrc, base_idx):
        return self.ctx.load_global_f16xN(rsrc, base_idx)

    def load_vectorized_k(self, k_ptr, page_id, kv_row, col):
        ctx = self.ctx
        traits = ctx.traits
        base = (
            page_id * fx.Index(traits.PAGE_STRIDE_VEC)
            + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            + kv_row * fx.Index(traits.KV_VEC_SIZE)
        )
        dg = col // fx.Index(traits.KV_VEC_SIZE)
        lo = ctx.load_global_half_vec(
            k_ptr,
            base + dg * fx.Index(traits.PAGE_SIZE * traits.KV_VEC_SIZE),
            traits.KV_VEC_SIZE,
        )
        hi = ctx.load_global_half_vec(
            k_ptr,
            base + (dg + fx.Index(1)) * fx.Index(traits.PAGE_SIZE * traits.KV_VEC_SIZE),
            traits.KV_VEC_SIZE,
        )
        return Vec(lo).shuffle(Vec(hi), list(range(traits.VEC_WIDTH))).ir_value()

    def load_vectorized_v(self, v_ptr, page_id, kv_row, col):
        ctx = self.ctx
        traits = ctx.traits
        kg = kv_row // fx.Index(traits.KV_VEC_SIZE)
        kr = kv_row % fx.Index(traits.KV_VEC_SIZE)
        base = (
            page_id * fx.Index(traits.PAGE_STRIDE_VEC)
            + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            + kg * fx.Index(traits.HEAD_DIM * traits.KV_VEC_SIZE)
            + kr
        )
        elems = []
        for i in range_constexpr(traits.VEC_WIDTH):
            v1 = ctx.load_global_half_vec(
                v_ptr, base + (col + fx.Index(i)) * fx.Index(traits.KV_VEC_SIZE), 1
            )
            elems.append(Vec(v1)[0])
        return Vec.from_elements(elems, ctx.elem_dtype).ir_value()

    def k_buf_base(self, buf_id):
        return self.ctx.k_buf_base(buf_id)

    def v_buf_base(self, buf_id):
        return self.ctx.v_buf_base(buf_id)

    def k_swizzle(self, row_idx, col_idx):
        return self.ctx.k_swizzle(row_idx, col_idx)

    def v_swizzle(self, row_idx, col_idx):
        return self.ctx.v_swizzle(row_idx, col_idx)

    def sigma_kv(self, n):
        return self.ctx.sigma_kv(n)

    def _tile_page_id(self, tile_start):
        return self.page_ids.page_id(tile_start)

    def coop_load_k(self, tile_start, buf_id=0, col_off=0):
        """Cooperative K load (row-major, XOR-swizzled); @p col_off is a K chunk's first
        column (modification 18), the tile's columns being the chunk's."""
        ctx = self.ctx
        traits = ctx.traits
        lds_kv = ctx.lds_kv
        row = ctx.load_row_in_batch
        col = ctx.load_col_base
        k_base = self.k_buf_base(buf_id)
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + row + row_offset
            else:
                row_idx = self.row_clamp(tile_start + row + row_offset)
            if const_expr(traits.KV_NEEDS_GUARD):

                def _store_row():
                    lds_row = row + row_offset
                    lds_idx = (
                        k_base
                        + lds_row * traits.K_STRIDE
                        + self.k_swizzle(lds_row, col)
                    )
                    self._k_lds_store(
                        self.load_k_cols(row_idx, col + fx.Index(col_off)), lds_idx
                    )

                _if_then(row < fx.Index(traits.BLOCK_N), _store_row)
            else:
                lds_row = row + row_offset
                lds_idx = (
                    k_base + lds_row * traits.K_STRIDE + self.k_swizzle(lds_row, col)
                )
                if const_expr(traits.KV_VECTORIZED):
                    vec = self.load_vectorized_k(
                        ctx.k_ptr, pid, self.sigma_kv(lds_row), col
                    )
                else:
                    vec = self.load_k_cols(row_idx, col + fx.Index(col_off))
                self._k_lds_store(vec, lds_idx)

    def _k_lds_store(self, vec, lds_idx):
        """Store one lane's K vector to the tile; a lane past a guarded row's end stores
        nothing (modification 26)."""
        ctx = self.ctx
        traits = ctx.traits

        def _store():
            fx.ptr_store(vec, ctx.lds_kv + fx.Int64(lds_idx))

        if const_expr(traits.K_LOAD_GUARD):
            _if_then(ctx.load_col_base < fx.Index(traits.K_COLS), _store)
        else:
            _store()

    def coop_load_k_global(self, tile_start):
        ctx = self.ctx
        traits = ctx.traits
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = (
                    pid * fx.Index(traits.PAGE_SIZE)
                    + ctx.load_row_in_batch
                    + row_offset
                )
            else:
                row_idx = self.row_clamp(
                    tile_start + ctx.load_row_in_batch + row_offset
                )
            vecs.append(self.load_k_cols(row_idx, ctx.load_col_base))
        return vecs

    def coop_store_k_lds(self, vecs, buf_id=0):
        ctx = self.ctx
        traits = ctx.traits
        k_base = self.k_buf_base(buf_id)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.KV_NEEDS_GUARD):

                def _store_row():
                    lds_row = ctx.load_row_in_batch + row_offset
                    lds_idx = (
                        k_base
                        + lds_row * traits.K_STRIDE
                        + self.k_swizzle(lds_row, ctx.load_col_base)
                    )
                    self._k_lds_store(vecs[batch], lds_idx)

                _if_then(ctx.load_row_in_batch < fx.Index(traits.BLOCK_N), _store_row)
            else:
                lds_row = ctx.load_row_in_batch + row_offset
                lds_idx = (
                    k_base
                    + lds_row * traits.K_STRIDE
                    + self.k_swizzle(lds_row, ctx.load_col_base)
                )
                self._k_lds_store(vecs[batch], lds_idx)

    def _v_store_to_lds(self, v_base, lds_row, vec):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.V_LOAD_GUARD):
            # A lane past a guarded row's end stores nothing (modification 26).
            _if_then(
                ctx.load_col_base_v < fx.Index(traits.V_WIDTH),
                lambda: self._v_store_to_lds_body(v_base, lds_row, vec),
            )
        else:
            self._v_store_to_lds_body(v_base, lds_row, vec)

    def _v_store_to_lds_body(self, v_base, lds_row, vec):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_HW_TR):
            lds_idx = v_base + lds_row * traits.V_STRIDE + ctx.load_col_base_v
            fx.ptr_store(vec, ctx.lds_kv + fx.Int64(lds_idx))
        else:
            for _e in range_constexpr(traits.VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = ctx.load_col_base_v + _e
                vt_idx = v_base + vt_d * traits.VT_STRIDE + lds_row
                fx.ptr_store(
                    Vec.from_elements([elem], ctx.elem_dtype),
                    ctx.lds_kv + fx.Int64(vt_idx),
                )

    def coop_load_v(self, tile_start, buf_id=0):
        """Cooperative V load, storing row-major or transposed per USE_HW_TR."""
        ctx = self.ctx
        traits = ctx.traits
        row = ctx.load_row_in_batch_v
        col = ctx.load_col_base_v
        v_base = self.v_buf_base(buf_id)
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_V):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD_V
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + row + row_offset
            else:
                row_idx = self.row_clamp(tile_start + row + row_offset)
            if const_expr(traits.V_NEEDS_GUARD):

                def _store_row():
                    self._v_store_to_lds(
                        v_base, row + row_offset, self.load_v_cols(row_idx, col)
                    )

                _if_then(row < fx.Index(traits.BLOCK_N), _store_row)
            else:
                lds_row = row + row_offset
                if const_expr(traits.KV_VECTORIZED):
                    vec = self.load_vectorized_v(
                        ctx.v_ptr, pid, self.sigma_kv(lds_row), col
                    )
                else:
                    vec = self.load_v_cols(row_idx, col)
                self._v_store_to_lds(v_base, lds_row, vec)

    def coop_load_v_global(self, tile_start):
        """Issue global V loads; vectorized mode returns no-major v8 rows."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.KV_VECTORIZED):
            pid = self._tile_page_id(tile_start)
            base_v = pid * fx.Index(
                traits.PAGE_STRIDE_VEC
            ) + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            vecs = []
            for j in range_constexpr(traits.NV8_PER_THREAD):
                flat = ctx.tid + fx.Index(j * traits.BLOCK_SIZE)
                d = flat // fx.Index(traits.VEC_V_NGROUPS)
                ng = flat % fx.Index(traits.VEC_V_NGROUPS)
                src = (
                    base_v
                    + ng * fx.Index(traits.HEAD_DIM * traits.KV_VEC_SIZE)
                    + d * fx.Index(traits.KV_VEC_SIZE)
                )
                vecs.append(self.load_half_vec(ctx.v_ptr, src, traits.KV_VEC_SIZE))
            return vecs
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_V):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD_V
            if const_expr(traits.PAGED):
                row_idx = (
                    pid * fx.Index(traits.PAGE_SIZE)
                    + ctx.load_row_in_batch_v
                    + row_offset
                )
            else:
                row_idx = self.row_clamp(
                    tile_start + ctx.load_row_in_batch_v + row_offset
                )
            vecs.append(self.load_v_cols(row_idx, ctx.load_col_base_v))
        return vecs

    def coop_store_v_lds(self, vecs, buf_id=0):
        """Write V vectors to LDS; vectorized mode uses no-major rows."""
        ctx = self.ctx
        traits = ctx.traits
        v_base = self.v_buf_base(buf_id)
        if const_expr(traits.KV_VECTORIZED):
            for j in range_constexpr(traits.NV8_PER_THREAD):
                flat = ctx.tid + fx.Index(j * traits.BLOCK_SIZE)
                d = flat // fx.Index(traits.VEC_V_NGROUPS)
                ng = flat % fx.Index(traits.VEC_V_NGROUPS)
                dst = (
                    v_base
                    + (d // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                    + ng * fx.Index(traits.VEC_V_D128)
                    + (d % fx.Index(8)) * fx.Index(8)
                )
                fx.ptr_store(vecs[j], ctx.lds_kv + fx.Int64(dst))
            return
        for batch in range_constexpr(traits.NUM_BATCHES_V):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD_V
            if const_expr(traits.V_NEEDS_GUARD):
                _if_then(
                    ctx.load_row_in_batch_v < fx.Index(traits.BLOCK_N),
                    lambda: self._v_store_to_lds(
                        v_base, ctx.load_row_in_batch_v + row_offset, vecs[batch]
                    ),
                )
            else:
                self._v_store_to_lds(
                    v_base, ctx.load_row_in_batch_v + row_offset, vecs[batch]
                )

    def coop_load_v_global_perm(self, tile_start):
        ctx = self.ctx
        traits = ctx.traits
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for r in range_constexpr(ctx.vp_rows_per_thread):
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + ctx.vp_row_base + r
            else:
                row_idx = self.row_clamp(tile_start + ctx.vp_row_base + r)
            vecs.append(self.load_v_cols(row_idx, ctx.vp_col_base))
        return vecs

    @flyc.jit
    def coop_store_v_lds_perm(self, vecs, buf_id=0):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(ctx.vp_active_threads < traits.BLOCK_SIZE):
            # only active lanes store
            active = ctx.tid < fx.Index(ctx.vp_active_threads)
            if active:
                self._coop_store_v_lds_perm_body(vecs, buf_id)
        else:
            self._coop_store_v_lds_perm_body(vecs, buf_id)

    def _coop_store_v_lds_perm_body(self, vecs, buf_id):
        ctx = self.ctx
        traits = ctx.traits
        v_base = self.v_buf_base(buf_id)
        dwords = [
            Vec(vecs[r]).bitcast(fx.Int32)
            for r in range_constexpr(ctx.vp_rows_per_thread)
        ]
        ndw = traits.VEC_WIDTH // 2
        for w in range_constexpr(ndw):
            a = dwords[0][w]
            b = dwords[1][w]
            dl = ctx.vp_col_base + fx.Index(2 * w)
            dh = dl + fx.Index(1)
            lo_01 = rocdl.perm_b32(b, a, ctx.vp_sel_lo)
            v_lo = Vec.from_elements([lo_01], fx.Int32).bitcast(ctx.elem_dtype)
            fx.ptr_store(
                v_lo,
                ctx.lds_kv
                + fx.Int64(v_base + dl * fx.Index(traits.VT_STRIDE) + ctx.vp_row_base),
            )
            hi_01 = rocdl.perm_b32(b, a, ctx.vp_sel_hi)
            v_hi = Vec.from_elements([hi_01], fx.Int32).bitcast(ctx.elem_dtype)
            fx.ptr_store(
                v_hi,
                ctx.lds_kv
                + fx.Int64(v_base + dh * fx.Index(traits.VT_STRIDE) + ctx.vp_row_base),
            )

    def init_dma_nomajor(self):
        # KV_VECTORIZED V: no-major GM->LDS DMA constants (one aligned v8 per lane).
        ctx = self.ctx
        traits = ctx.traits
        self._v_dma_base_i64 = fx.Int64(
            buffer_ops.extract_base_index(ctx.V, address_space=1)
        )
        self._v_dma_page_bytes = fx.Int64(traits.PAGE_STRIDE_VEC * 2)
        self._v_dma_lds_base = ctx.lds_kv_offset
        self._v_dma_sz = fx.Int32(16)
        self._v_dma_z = fx.Int32(0)
        self._v_dma_aux = fx.Int32(1)
        self._v_dma_no = ctx.lane // fx.Index(8)
        self._v_dma_dloc = ctx.lane % fx.Index(8)

    def coop_dma_v_nomajor(self, tile_start, buf_id=0):
        # Each lane DMAs one contiguous v8 into the no-major V line.
        ctx = self.ctx
        traits = ctx.traits
        pid = self._tile_page_id(tile_start)
        paddr = as_mlir_value(
            self._v_dma_base_i64 + fx.Int64(pid) * self._v_dma_page_bytes
        )
        rsrc = fx.rocdl.get_buffer_rsrc(
            buffer_ops.create_buffer_resource_from_addr(
                paddr, num_records_bytes=as_mlir_value(self._v_dma_page_bytes)
            )
        )
        if const_expr(isinstance(buf_id, int)):
            vb = self._v_dma_lds_base + fx.Index(
                (traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2
            )
        else:
            vb = (
                self._v_dma_lds_base
                + fx.Index(traits.LDS_V_BASE * 2)
                + buf_id * fx.Index(traits.LDS_V_TILE_SIZE * 2)
            )
        for d in range_constexpr(traits.NV8_PER_THREAD):
            row = ctx.wave_id * fx.Index(traits.NV8_PER_THREAD) + fx.Index(d)
            lds_b = vb + row * fx.Index(traits.VEC_V_LINE * 2)
            lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_b))
            lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
            dcol = row * fx.Index(8) + self._v_dma_dloc
            voff_e = (
                ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
                + self._v_dma_no * fx.Index(traits.HEAD_DIM * 8)
                + dcol * fx.Index(8)
            )
            voff = fx.Int32(voff_e * fx.Index(2))
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                self._v_dma_sz,
                voff,
                self._v_dma_z,
                self._v_dma_z,
                self._v_dma_aux,
            )

    def init_dma(self):
        # buffer_load_dwordx4 GM->LDS DMA constants + K/V per-batch resources.
        ctx = self.ctx
        traits = ctx.traits
        self.DMA_BYTES = 4 if traits.ENABLE_GFX942_DMA else 16
        self.DMA_BATCH_BYTES = traits.BLOCK_SIZE * self.DMA_BYTES
        self.lds_kv_base_idx = ctx.lds_kv_offset
        self._dma_size = fx.Int32(self.DMA_BYTES)
        self._dma_soff = fx.Int32(0)
        self._dma_off = fx.Int32(0)
        self._dma_aux = fx.Int32(1)
        self.k_rsrc = fx.rocdl.get_buffer_rsrc(
            buffer_ops.create_buffer_resource(
                ctx.K,
                max_size=False,
                num_records_bytes=ctx.kv_nrec_bytes,
                base_byte_offset=ctx.kv_batch_byte_off,
            )
        )
        self.NUM_DMA_K = (traits.BLOCK_N * traits.K_STRIDE * 2) // self.DMA_BATCH_BYTES
        self.LANES_PER_K_ROW = traits.HEAD_DIM * 2 // self.DMA_BYTES
        self.ROWS_PER_DMA_BATCH = self.DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)
        self.v_rsrc = fx.rocdl.get_buffer_rsrc(
            buffer_ops.create_buffer_resource(
                ctx.V,
                max_size=False,
                num_records_bytes=ctx.kv_nrec_bytes,
                base_byte_offset=ctx.kv_batch_byte_off,
            )
        )
        self.NUM_DMA_V = (traits.BLOCK_N * traits.V_STRIDE * 2) // self.DMA_BATCH_BYTES
        self.LANES_PER_V_ROW = traits.HEAD_DIM * 2 // self.DMA_BYTES
        self.ROWS_PER_DMA_BATCH_V = self.DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)

    def _coop_dma_row(
        self,
        rsrc,
        lds_byte_base,
        num_dma,
        lanes_per_row,
        rows_per_batch,
        tile_start,
        xor_and,
    ):
        ctx = self.ctx
        traits = ctx.traits
        for d in range_constexpr(num_dma):
            lds_addr = (
                lds_byte_base
                + ctx.wave_id * fx.Index(traits.WARP_SIZE * self.DMA_BYTES)
                + fx.Index(d * self.DMA_BATCH_BYTES)
            )
            lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
            lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
            row_in_tile = ctx.tid // lanes_per_row + fx.Index(d * rows_per_batch)
            swiz_col_f16 = (ctx.tid % lanes_per_row) * (self.DMA_BYTES // 2)
            xor_mask = (row_in_tile & fx.Index(xor_and)) << fx.Index(4)
            col_byte = (swiz_col_f16 ^ xor_mask) * 2
            global_row = (
                tile_start + row_in_tile
            )  # 0-based: batch base folded into k/v_rsrc
            global_byte = (
                global_row * (ctx.stride_token_kv_v * fx.Index(2))
                + ctx.kv_head_byte_offset
                + col_byte
            )
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                self._dma_size,
                fx.Int32(global_byte),
                self._dma_soff,
                self._dma_off,
                self._dma_aux,
            )

    def coop_dma_k(self, tile_start, buf_id=0):
        """Load K tile via DMA with XOR-swizzled global fetch."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(isinstance(buf_id, int)):
            k_lds_byte_base = self.lds_kv_base_idx + fx.Index(
                buf_id * traits.LDS_K_TILE_SIZE * 2
            )
        else:
            k_lds_byte_base = self.lds_kv_base_idx + buf_id * fx.Index(
                traits.LDS_K_TILE_SIZE * 2
            )
        if const_expr(traits.ENABLE_GFX942_VEC_K):
            for d in range_constexpr(self.NUM_DMA_K):
                row_in_tile = ctx.tid // self.LANES_PER_K_ROW + fx.Index(
                    d * self.ROWS_PER_DMA_BATCH
                )
                col_f16 = (ctx.tid % self.LANES_PER_K_ROW) * (self.DMA_BYTES // 2)
                lds_elem = ctx.k_vec_elem_idx(row_in_tile, col_f16)
                lds_addr = k_lds_byte_base + lds_elem * 2
                global_row = tile_start + ctx.sigma_kv(row_in_tile)
                col_byte = col_f16 * 2
                lds_ptr = buffer_ops.create_llvm_ptr(
                    fx.Int64(lds_addr), address_space=3
                )
                global_byte = (
                    global_row * (ctx.stride_token_kv_v * fx.Index(2))
                    + ctx.kv_head_byte_offset
                    + col_byte
                )
                rocdl.raw_ptr_buffer_load_lds(
                    self.k_rsrc,
                    lds_ptr,
                    self._dma_size,
                    fx.Int32(global_byte),
                    self._dma_soff,
                    self._dma_off,
                    self._dma_aux,
                )
        else:
            self._coop_dma_row(
                self.k_rsrc,
                k_lds_byte_base,
                self.NUM_DMA_K,
                self.LANES_PER_K_ROW,
                self.ROWS_PER_DMA_BATCH,
                tile_start,
                0x7,
            )

    def coop_dma_v(self, tile_start, buf_id=0):
        """Load V tile via DMA with XOR-swizzled global fetch."""
        traits = self.ctx.traits
        v_lds_byte_base = self.lds_kv_base_idx + fx.Index(
            (traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2
        )
        self._coop_dma_row(
            self.v_rsrc,
            v_lds_byte_base,
            self.NUM_DMA_V,
            self.LANES_PER_V_ROW,
            self.ROWS_PER_DMA_BATCH_V,
            tile_start,
            0x3,
        )


class GenericKvLdsToVgprLoader:
    def __init__(self, ctx):
        self.ctx = ctx
        self._pv_steps = [
            (dc, pks)
            for dc in range(ctx.traits.D_CHUNKS)
            for pks in range(ctx.traits.PV_K_STEPS)
        ]
        self.total_pv = len(self._pv_steps)

    def ds_read_tr_v4f16(self, lds_elem_idx):
        ctx = self.ctx
        byte_offset = lds_elem_idx * 2 + ctx.lds_kv_offset
        ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
        return rocdl.ds_read_tr16_b64(ctx.v4f16_type, ptr).result

    def load_k_packs(self, k_base):
        ctx = self.ctx
        traits = ctx.traits
        k_hi_offset = traits.K_SUB_N * traits.K_STRIDE
        k_swz_mask = (ctx.lane_mod_32 & fx.Index(traits.K_SWZ_ROWMASK)) << fx.Index(4)

        def _idx(ks, hi):
            if const_expr(traits.ENABLE_GFX942_VEC_K):
                base = k_base + (
                    fx.Index(traits.K_VEC_HI_N_OFFSET) if hi else fx.Index(0)
                )
                return (
                    base
                    + fx.Index(ks) * fx.Index(traits.K_VEC_N_STRIDE)
                    + ctx.lane_mod_32 * fx.Index(traits.K_VEC_SIZE)
                    + ctx.lane_div_32 * ctx.MFMA_LANE_K
                )
            col = fx.Index(ks * traits.K_STEP_QK) + ctx.lane_div_32 * ctx.MFMA_LANE_K
            base = (
                k_base
                + (k_hi_offset if hi else fx.Index(0))
                + ctx.lane_mod_32 * traits.K_STRIDE
            )
            return base + (col ^ k_swz_mask)

        depth = traits.QK_PREFETCH_DEPTH
        lo = [None] * traits.K_STEPS_QK
        hi = [None] * traits.K_STEPS_QK
        for p in range_constexpr(depth):
            lo[p] = fx.ptr_load(
                ctx.lds_kv + fx.Int64(_idx(p, False)), result_type=ctx.mfma_pack_type
            ).ir_value()
            hi[p] = fx.ptr_load(
                ctx.lds_kv + fx.Int64(_idx(p, True)), result_type=ctx.mfma_pack_type
            ).ir_value()
        if const_expr(traits.ENABLE_GFX942_VEC_K or traits.ENABLE_GFX942_KV_GPFETCH):
            rocdl.sched_group_barrier(rocdl.mask_dsrd, depth * 2, 0)
        self._k_idx = _idx
        self._k_depth = depth
        return lo, hi

    def load_k_pack_at(self, ks):
        ctx = self.ctx
        lo = fx.ptr_load(
            ctx.lds_kv + fx.Int64(self._k_idx(ks, False)),
            result_type=ctx.mfma_pack_type,
        ).ir_value()
        hi = fx.ptr_load(
            ctx.lds_kv + fx.Int64(self._k_idx(ks, True)), result_type=ctx.mfma_pack_type
        ).ir_value()
        return lo, hi

    def read_v_pack(self, step_idx, v_base):
        ctx = self.ctx
        traits = ctx.traits
        dc, pks = self._pv_steps[step_idx]
        if const_expr(traits.KV_VECTORIZED):
            # No-major V: one aligned v8 per (dc,pks) half. Lane l reads
            # V[d=dc*32+l%32, n=pks*16+(l//32)*8+0..7] (lo) / +32 (hi).
            lm = ctx.lane_mod_32
            v_lane_base = (
                v_base
                + (lm // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                + ctx.lane_div_32 * fx.Index(traits.VEC_V_D128)
                + (lm % fx.Index(8)) * fx.Index(8)
            )
            lo_off = (
                dc * (traits.D_CHUNK // 8) * traits.VEC_V_LINE
                + pks * (traits.PV_K_STEP // 8) * traits.VEC_V_D128
            )
            hi_off = lo_off + (traits.K_SUB_N // 8) * traits.VEC_V_D128
            vl = fx.ptr_load(
                ctx.lds_kv + fx.Int64(v_lane_base + fx.Index(lo_off)),
                result_type=ctx.mfma_pack_type,
            )
            vh = fx.ptr_load(
                ctx.lds_kv + fx.Int64(v_lane_base + fx.Index(hi_off)),
                result_type=ctx.mfma_pack_type,
            )
            return vl, vh
        if const_expr(traits.USE_HW_TR):
            d_col = (
                fx.Index(dc * traits.D_CHUNK)
                + ctx.tr_col_half * 16
                + ctx.tr_col_sub * 4
            )
            k_row = (
                fx.Index(pks * traits.PV_K_STEP) + ctx.lane_div_32 * 4 + ctx.tr_k_group
            )
            d_col_eff = ctx.v_swizzle(k_row, d_col) if traits.ENABLE_DMA else d_col
            lds_lo = v_base + k_row * traits.V_STRIDE + d_col_eff
            lds_hi = lds_lo + fx.Index(traits.K_SUB_N * traits.V_STRIDE)
            if const_expr(traits.USE_K16):
                vl_a = self.ds_read_tr_v4f16(lds_lo)
                vl_b = self.ds_read_tr_v4f16(lds_lo + fx.Index(8 * traits.V_STRIDE))
                vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                vh_a = self.ds_read_tr_v4f16(lds_hi)
                vh_b = self.ds_read_tr_v4f16(lds_hi + fx.Index(8 * traits.V_STRIDE))
                vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            else:
                vl = self.ds_read_tr_v4f16(lds_lo)
                vh = self.ds_read_tr_v4f16(lds_hi)
            return vl, vh
        d_pos = fx.Index(dc * traits.D_CHUNK) + ctx.lane_mod_32
        k_col = fx.Index(pks * traits.PV_K_STEP) + ctx.lane_div_32 * 4
        v_lo_idx = v_base + d_pos * traits.VT_STRIDE + k_col
        v_hi_idx = v_lo_idx + fx.Index(traits.K_SUB_N)
        vl = fx.ptr_load(ctx.lds_kv + fx.Int64(v_lo_idx), result_type=ctx.v4f16_type)
        vh = fx.ptr_load(ctx.lds_kv + fx.Int64(v_hi_idx), result_type=ctx.v4f16_type)
        return vl, vh


class GenericQLoader:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_pack(self, q_rsrc, q_row, ks):
        ctx = self.ctx
        traits = ctx.traits
        q_col = fx.Index(ks * traits.K_STEP_QK) + ctx.lane_div_32 * traits.MFMA_LANE_K
        return buffer_ops.buffer_load(
            q_rsrc,
            ctx.global_idx_q(q_row, q_col),
            vec_width=traits.MFMA_LANE_K,
            dtype=ctx.elem_dtype,
            # Past the actual head dim, read as zero (modification 13).
            mask=(q_col < ctx.head_dim_qk_v) if traits.QK_MASK else None,
        )

    def load_all(self, q_rsrc, q_row):
        traits = self.ctx.traits
        if const_expr(traits.Q_STREAM):
            # Past head_dim 256 the packs are read where each K-step uses them
            # (modification 16).
            return _QPackStream(self, q_rsrc, q_row)
        # f32 (modification 35): a K-chunked build with Q resident holds every chunk's packs
        # (gemm1_accumulate indexes them by ks + chunk * K_STEPS_QK).
        return [
            self.load_pack(q_rsrc, q_row, ks)
            for ks in range_constexpr(traits.K_STEPS_QK * traits.K_CHUNKS)
        ]


class _QPackStream:
    """Q's packs re-read from global on each use, in the region that uses them, rather
    than held in registers through the KV loop (modification 16)."""

    def __init__(self, loader, q_rsrc, q_row):
        self.loader = loader
        self.q_rsrc = q_rsrc
        self.q_row = q_row

    def __getitem__(self, ks):
        return self.loader.load_pack(self.q_rsrc, self.q_row, ks)


class GenericGemmHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def _mfma(self, mfma_fn, a, b, c):
        return mfma_fn(self.ctx.v16f32_type, [a, b, c])

    def mfma_acc(self, a, b, c):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_K16):
            return fly.mma_atom_call_ssa([ctx.v16f32_type], ctx.mma_atom_k16, a, b, c)
        if const_expr(traits.DTYPE_STR == "f32"):
            # f32 (modification 35): one 16-bit K-step (a 4-element pack per lane along K) is four
            # exact-f32 32x32x2 MFMAs, the j-th taking element j of the same pack from A
            # and B; both name the same K index on every lane, so the contraction is
            # exact (DESIGN §1.1).
            av = Vec(a)
            bv = Vec(b)
            acc = c
            for j in range_constexpr(4):
                aj, bj, cj, cbsz, abid, blgp = rocdl._split_mfma_operands(
                    [av[j], bv[j], acc]
                )
                acc = rocdl.mfma_f32_32x32x2f32(
                    ctx.v16f32_type, aj, bj, cj, cbsz, abid, blgp
                ).result
            return acc
        if const_expr(traits.DTYPE_STR == "bf16"):
            a = Vec(a).bitcast(fx.Int16)
            b = Vec(b).bitcast(fx.Int16)
            return self._mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
        return self._mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

    def gemm1_accumulate(
        self,
        kv_lds_to_vgpr,
        k_lo,
        k_hi,
        q_b_packs,
        kv_gmem_to_lds=None,
        kv_start=None,
        s_init=None,
        ks_off=0,
    ):
        ctx = self.ctx
        traits = ctx.traits
        depth = traits.QK_PREFETCH_DEPTH
        # A K chunk's pass continues the previous chunks' scores and reads Q from its
        # first step on (modification 18).
        s_lo, s_hi = (
            (ctx.c_zero_v16f32, ctx.c_zero_v16f32) if s_init is None else s_init
        )
        for ks in range_constexpr(traits.K_STEPS_QK):
            if const_expr(
                traits.ENABLE_DMA
                and traits.USE_HW_TR
                and not traits.ENABLE_PREFETCH_3BUF
                and ks == traits.K_STEPS_QK // 2
            ):
                kv_gmem_to_lds.coop_dma_v(kv_start, 0)
                rocdl.sched_barrier(0)
            q_pack = q_b_packs[ks + ks_off]
            s_lo = self.mfma_acc(k_lo[ks], q_pack, s_lo)
            s_hi = self.mfma_acc(k_hi[ks], q_pack, s_hi)
            if const_expr(ks + depth < traits.K_STEPS_QK):
                k_lo[ks + depth], k_hi[ks + depth] = kv_lds_to_vgpr.load_k_pack_at(
                    ks + depth
                )
        return s_lo, s_hi

    def gemm2_pv(
        self, kv_lds_to_vgpr, o_accs, p_packs_lo, p_packs_hi, v_base, corr_vec
    ):
        # O += V^T_lo @ P_lo + V^T_hi @ P_hi with interleaved V prefetch.
        traits = self.ctx.traits
        steps = kv_lds_to_vgpr._pv_steps
        total = kv_lds_to_vgpr.total_pv
        if const_expr(not traits.USE_HW_TR):
            # The loop below rescales chunks 1..PV_K_STEPS lazily, one per dc == 0 step;
            # any past those (head_dim 256: 8 chunks, 4 steps) are rescaled here, before
            # their first accumulate (modification 10).
            for dc in range_constexpr(traits.PV_K_STEPS + 1, traits.D_CHUNKS):
                o_accs[dc] = Vec(o_accs[dc]) * corr_vec
        v_lo_cur, v_hi_cur = kv_lds_to_vgpr.read_v_pack(0, v_base)
        for si in range_constexpr(total):
            dc, pks = steps[si]
            if const_expr(si + 1 < total):
                v_lo_nxt, v_hi_nxt = kv_lds_to_vgpr.read_v_pack(si + 1, v_base)
            o_accs[dc] = self.mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
            o_accs[dc] = self.mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
            if const_expr(
                not traits.USE_HW_TR and dc == 0 and pks < traits.D_CHUNKS - 1
            ):
                o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
            if const_expr(si + 1 < total):
                v_lo_cur = v_lo_nxt
                v_hi_cur = v_hi_nxt
        return o_accs


class GenericSoftmaxHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def reduction_peer(self, v_f32):
        ctx = self.ctx
        if const_expr(ctx.traits.REDUCE_MODE == "ds_bpermute"):
            v_i32 = fx.Float32(v_f32).bitcast(fx.Int32)
            peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, ctx.lane_xor_32_byte, v_i32)
            return fx.Int32(peer_i32).bitcast(ctx.compute_type)
        return fx.Float32(v_f32).shuffle_xor(ctx.shuf_32_i32, ctx.width_i32)

    def split_scores(self, s_acc_lo, s_acc_hi):
        return _score_pair_to_lists((s_acc_lo, s_acc_hi))

    def _kv_mask_lane_off(self, kv_start_i32):
        # Physical KV column base per lane; KV_VECTORIZED applies sigma(kv) in the K load.
        ctx = self.ctx
        if const_expr(ctx.traits.KV_VECTORIZED):
            lane_off = fx.Int32(ctx.lane_div_32) * fx.Int32(8)
            moff = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
        else:
            lane_off = fx.Int32(ctx.lane_div_32) * fx.Int32(4)
            moff = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
        return kv_start_i32 + lane_off, moff

    def add_bias(self, s_raw_lo, s_raw_hi, kv_start):
        """Scale the scores and add the bias (modification 12): score * scale + bias, the
        scaled domain the rest of a bias build's softmax works in (modification 28).
        Each lane's 32 scores are 8 runs of 4 consecutive keys, one 8-byte load each."""
        ctx = self.ctx
        traits = ctx.traits
        col_base_i32, moff = self._kv_mask_lane_off(fx.Int32(kv_start))
        row_off = ctx.q_row * ctx.bias_stride0_v
        out_lo = list(s_raw_lo)
        out_hi = list(s_raw_hi)
        if const_expr(ctx.bias_kstrided):
            # One key at a time, at a runtime key stride (0 broadcasts the keys), in f32
            # or the query's dtype (modification 31): each lane's first key's offset in a
            # register, every other key's distance from it a scalar.
            base_i32 = fx.Int32(row_off) + col_base_i32 * ctx.bias_stride_k_i32
            cols = [moff[r] + extra for extra in (0, traits.K_SUB_N) for r in range(16)]
            return self._add_bias_kstrided(out_lo, out_hi, base_i32, cols)
        for scores, extra in ((out_lo, 0), (out_hi, traits.K_SUB_N)):
            for g in range_constexpr(4):
                idx = row_off + fx.Index(col_base_i32 + fx.Int32(moff[4 * g] + extra))
                raw = buffer_ops.buffer_load(
                    ctx.bias_rsrc, idx, vec_width=4, dtype=ctx.elem_dtype
                )
                vals = Vec(Vec(raw, (4,), ctx.elem_dtype).to(fx.Float32))
                for e in range_constexpr(4):
                    scores[4 * g + e] = fx.fma(
                        scores[4 * g + e],
                        ctx.c_score_scale,
                        vals[e],
                        fastmath=ctx.fm_fast,
                    )
        return out_lo, out_hi

    def _kstrided_bias(self, base_i32, col, elem_bytes, dtype):
        """The bias of the key @p col keys past a lane's first, widened to f32: its
        distance a scalar, so the load takes it as its scalar offset (modification 31).
        """
        ctx = self.ctx
        raw = buffer_ops.buffer_load(
            ctx.bias_rsrc,
            as_mlir_value(base_i32),
            vec_width=1,
            dtype=dtype,
            soffset_bytes=ctx.bias_stride_k_i32 * fx.Int32(col * elem_bytes),
        )
        return fx.Float32(fx.Float32(raw))

    def _add_bias_kstrided(self, s_lo, s_hi, base_i32, cols):
        """score * scale + bias over a lane's 32 scores, the bias read one key at a time,
        as f32 or in the query's dtype by the launch's `bias_f32` (modification 31): any
        key stride, key length or row pitch is read exactly, and only the branch the
        launch's dtype takes runs."""
        ctx = self.ctx
        # The scores are halved in this build (`_BIAS_HALVED_NEG_FLOOR`): the scale in
        # c_score_scale, the bias here, both exactly.
        c = ctx.c_score_scale
        half = fx.Float32(0.5)
        fm = ctx.fm_fast
        f32_on = ctx.bias_f32_i32 != ctx.c_zero_i32

        def biased(scores, first, elem_bytes, dtype):
            return [
                fx.fma(
                    scores[r],
                    c,
                    self._kstrided_bias(base_i32, cols[first + r], elem_bytes, dtype)
                    * half,
                    fastmath=fm,
                )
                for r in range(16)
            ]

        @flyc.jit
        def _add(s_lo, s_hi):
            if f32_on:
                s_lo = biased(s_lo, 0, 4, T.f32)
                s_hi = biased(s_hi, 16, 4, T.f32)
            else:
                s_lo = biased(s_lo, 0, 2, ctx.elem_dtype)
                s_hi = biased(s_hi, 16, 2, ctx.elem_dtype)
            return s_lo, s_hi

        return _add(s_lo, s_hi)

    @flyc.jit
    def apply_kv_mask(self, s_raw_lo, s_raw_hi, kv_start):
        ctx = self.ctx
        traits = ctx.traits
        # A masked score: -inf, or in a bias build -FLT_MAX, which fast math cannot
        # fold out of the bias build's (s - m) exponent (`_BIAS_MASKED_SCORE`).
        c_masked = ctx.c_neg_inf
        if const_expr(getattr(ctx, "bias_rsrc", None) is not None):
            s_raw_lo, s_raw_hi = self.add_bias(s_raw_lo, s_raw_hi, kv_start)
            c_masked = ctx.c_masked_score
        kv_start_i32 = fx.Int32(kv_start)
        if const_expr(traits.CAUSAL):
            # below-diagonal tiles skip masking; guard carries the two score lists
            # A row whose diagonal passes the last key (top-left with more queries than
            # keys, a band past Skv - Sq) sees every key, and the last KV tile's padding
            # stays masked (modification 30): in a tile that holds padding, each row's
            # limit is clamped to the last key there and then (a scalar select and a
            # vector min per tile, nothing kept across the loop), and the block's first
            # row's diagonal is clamped too, so that tile is masked at all. Clamping the
            # limit once per workgroup instead, or adding a compare per score, took
            # registers the 256-row gfx942 tile spills for. A split (decode) build keeps
            # the plain compare: the provider runs it only on diagonals inside the keys,
            # which prefill builds serve at any query length.
            if const_expr(traits.SPLIT_KV):
                q_start_i32 = fx.Int32(ctx.pos_first) + ctx.delta_i32
                q_mask_limit_i32 = ctx.q_row_i32 + ctx.delta_i32
                max_kv_col_i32 = kv_start_i32 + fx.Int32(traits.BLOCK_N - 1)
            else:
                last_key_i32 = fx.Int32(ctx.seqlen_kv_b) - fx.Int32(1)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(traits.BLOCK_N - 1)
                q_start_raw_i32 = fx.Int32(ctx.pos_first) + ctx.delta_i32
                q_start_i32 = fx.Int32(
                    (q_start_raw_i32 < last_key_i32).select(
                        q_start_raw_i32, last_key_i32
                    )
                )
                key_cap_i32 = fx.Int32(
                    (max_kv_col_i32 > last_key_i32).select(
                        last_key_i32, fx.Int32(0x7FFFFFFF)
                    )
                )
                q_limit_raw_i32 = ctx.q_row_i32 + ctx.delta_i32
                q_mask_limit_i32 = fx.Int32(
                    (q_limit_raw_i32 < key_cap_i32).select(q_limit_raw_i32, key_cap_i32)
                )
            tile_needs_mask = max_kv_col_i32 > q_start_i32
            col_base_i32, moff = self._kv_mask_lane_off(kv_start_i32)
            c_neg_inf = c_masked
            if const_expr(ctx.win_lo_i32 is not None):
                # The window's lower edge (modification 11): a tile below the block's
                # last row's edge needs the mask too.
                q_window_lo_i32 = ctx.q_row_i32 + ctx.win_lo_i32
                last_row_lo_i32 = fx.Int32(ctx.pos_end) - fx.Int32(1) + ctx.win_lo_i32
                tile_needs_mask = tile_needs_mask | (kv_start_i32 < last_row_lo_i32)

            masked_lo = list(s_raw_lo)
            masked_hi = list(s_raw_hi)
            if tile_needs_mask:
                if const_expr(ctx.win_lo_i32 is None):
                    masked_lo = [
                        (col_base_i32 + fx.Int32(moff[r]) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_lo[r]
                        )
                        for r in range(16)
                    ]
                    masked_hi = [
                        (
                            col_base_i32 + fx.Int32(moff[r]) + fx.Int32(traits.K_SUB_N)
                            > q_mask_limit_i32
                        ).select(c_neg_inf, s_raw_hi[r])
                        for r in range(16)
                    ]
                else:
                    masked_lo = []
                    masked_hi = []
                    for r in range_constexpr(16):
                        col_lo = col_base_i32 + fx.Int32(moff[r])
                        col_hi = col_lo + fx.Int32(traits.K_SUB_N)
                        out_lo = (col_lo > q_mask_limit_i32) | (
                            col_lo < q_window_lo_i32
                        )
                        out_hi = (col_hi > q_mask_limit_i32) | (
                            col_hi < q_window_lo_i32
                        )
                        masked_lo.append(out_lo.select(c_neg_inf, s_raw_lo[r]))
                        masked_hi.append(out_hi.select(c_neg_inf, s_raw_hi[r]))
            return masked_lo, masked_hi

        # Non-causal: mask physical KV columns outside seqlen so tail rows stay out of softmax.
        seq_len_i32 = fx.Int32(ctx.seqlen_kv_b)
        if const_expr(not traits.SKIP_KV_PAD_MASK):
            col_base_i32, moff = self._kv_mask_lane_off(kv_start_i32)
            kv_tile_end = kv_start + fx.Index(traits.BLOCK_N)
            needs_pad_mask = fx.Int32(kv_tile_end) > seq_len_i32
            for r in range_constexpr(16):
                kv_col = col_base_i32 + fx.Int32(moff[r])
                masked_lo = (kv_col >= seq_len_i32).select(c_masked, s_raw_lo[r])
                masked_hi = (kv_col + fx.Int32(traits.K_SUB_N) >= seq_len_i32).select(
                    c_masked, s_raw_hi[r]
                )
                s_raw_lo[r] = (needs_pad_mask).select(masked_lo, s_raw_lo[r])
                s_raw_hi[r] = (needs_pad_mask).select(masked_hi, s_raw_hi[r])
        return s_raw_lo, s_raw_hi

    def _exp2(self, x):
        ctx = self.ctx
        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_NATIVE_EXP2", "1") == "1"):
            return rocdl.exp2(T.f32, as_mlir_value(x))
        return fx.Float32(x).exp2(fastmath=ctx.fm_fast)

    def online_softmax_stats(self, m_running, s_raw_lo, s_raw_hi):
        ctx = self.ctx
        traits = ctx.traits

        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_TREE_REDUCE", "0") == "1"):

            def _max_pair(a, b):
                return fx.maxnumf(a, b)

            local_max = _tree_reduce(list(s_raw_lo) + list(s_raw_hi), _max_pair)
        else:
            local_max = s_raw_lo[0]
            for r in range_constexpr(15):
                local_max = fx.maxnumf(local_max, s_raw_lo[r + 1])
            for r in range_constexpr(16):
                local_max = fx.maxnumf(local_max, s_raw_hi[r])
        row_max = fx.maxnumf(local_max, self.reduction_peer(local_max))
        m_new_raw = fx.maxnumf(m_running, row_max)
        # A bias can mask a whole row as causal masking does (modification 28).
        if const_expr(traits.CAUSAL or getattr(ctx, "bias_rsrc", None) is not None):
            m_new_raw = fx.maxnumf(m_new_raw, ctx.c_neg_floor)

        diff_m_scaled = (m_running - m_new_raw) * ctx.c_sm_scale_log2e
        corr = self._exp2(diff_m_scaled)
        neg_scaled_max = ctx.c_zero_f - ctx.c_sm_scale_log2e * m_new_raw
        return m_new_raw, corr, neg_scaled_max

    def _exp2_arg(self, s, m_new_raw, neg_scaled_max):
        """A score's exponent, (s - m) * scale * log2(e). A bias build takes the
        difference first (modification 28): near -FLT_MAX, the fused form's s * log2(e)
        and -m * log2(e) overflow to -inf + inf."""
        ctx = self.ctx
        if const_expr(getattr(ctx, "bias_rsrc", None) is not None):
            return (s - m_new_raw) * ctx.c_sm_scale_log2e
        return fx.fma(s, ctx.c_sm_scale_log2e, neg_scaled_max, fastmath=ctx.fm_fast)

    def online_softmax(self, m_running, l_running, s_raw_lo, s_raw_hi):
        ctx = self.ctx
        m_new_raw, corr, neg_scaled_max = self.online_softmax_stats(
            m_running, s_raw_lo, s_raw_hi
        )

        p_vals_lo = []
        p_vals_hi = []
        local_sum = ctx.c_zero_f
        for r in range_constexpr(16):
            diff_lo = self._exp2_arg(s_raw_lo[r], m_new_raw, neg_scaled_max)
            p_lo = self._exp2(diff_lo)
            p_vals_lo.append(p_lo)
            local_sum = local_sum + p_lo
        for r in range_constexpr(16):
            diff_hi = self._exp2_arg(s_raw_hi[r], m_new_raw, neg_scaled_max)
            p_hi = self._exp2(diff_hi)
            p_vals_hi.append(p_hi)
            local_sum = local_sum + p_hi

        tile_sum = local_sum + self.reduction_peer(local_sum)
        l_new = corr * l_running + tile_sum
        return m_new_raw, l_new, corr, p_vals_lo, p_vals_hi

    def dropout_group(self, p_vals, first, kv_start, extra, score=None):
        """Drop the run of 4 probabilities @p p_vals[first:first + 4] (modification 33),
        the lane's scores @p score.. (default @p first) of its 16 in a half, @p extra
        keys right of the first half's: their keys are 4 consecutive ones starting at a
        multiple of 4 (each lane's column base is one, and so is each run's offset in
        `_kv_mask_lane_off`), so one Philox draw covers them, key j taking word j % 4. A
        dropped probability is 0; the row's sum `l` was formed before, so it and the
        LSE stay undropped."""
        ctx = self.ctx
        score = first if score is None else score
        col_base_i32, moff = self._kv_mask_lane_off(fx.Int32(kv_start))
        assert all(moff[score + e] == moff[score] + e for e in range(4))
        assert (moff[score] + extra) % 4 == 0
        counter = (
            ctx.ph_row_base
            + fx.Int64(col_base_i32 >> fx.Int32(2))
            + fx.Int64((moff[score] + extra) // 4)
        )
        words = _philox4x32(
            fx.Int32(counter),
            fx.Int32(counter >> fx.Int64(32)),
            ctx.ph_key[0],
            ctx.ph_key[1],
        )
        out = list(p_vals)
        for e in range_constexpr(4):
            keep = fx.Int32(words[e]) > ctx.ph_threshold
            out[first + e] = keep.select(fx.Float32(p_vals[first + e]), ctx.c_zero_f)
        return out

    def apply_dropout(self, p_vals_lo, p_vals_hi, kv_start):
        """Every run of 4 of a lane's 32 probabilities through `dropout_group`."""
        traits = self.ctx.traits
        for first in range_constexpr(0, 16, 4):
            p_vals_lo = self.dropout_group(p_vals_lo, first, kv_start, 0)
            p_vals_hi = self.dropout_group(p_vals_hi, first, kv_start, traits.K_SUB_N)
        return p_vals_lo, p_vals_hi

    def rescale_o_accs(self, o_accs, corr):
        ctx = self.ctx
        traits = ctx.traits
        corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
        if const_expr(not traits.USE_HW_TR):
            o_accs[0] = Vec(o_accs[0]) * corr_vec
        else:
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = Vec(o_accs[dc]) * corr_vec
        return o_accs, corr_vec

    def build_p_packs(self, p_vals):
        ctx = self.ctx
        traits = ctx.traits
        packs = []
        if const_expr(traits.DTYPE_STR == "bf16"):
            for pks in range_constexpr(traits.PV_K_STEPS):
                p_base = pks * traits.MFMA_LANE_K
                chunk = p_vals[p_base : p_base + traits.MFMA_LANE_K]
                if const_expr(traits.USE_K16):
                    packs.append(self.bf16_trunc_pack_v8(chunk))
                else:
                    packs.append(self.bf16_trunc_pack_v4(chunk))
            return packs

        p_f16 = []
        for r in range_constexpr(16):
            if const_expr(traits.DTYPE_STR == "f32"):
                # f32 (modification 35): P stays f32 (exactness; AOTriton keeps P in V's dtype).
                p_f16.append(fx.Float32(p_vals[r]))
            else:
                p_f16.append(fx.Float32(p_vals[r]).to(ctx.elem_dtype))
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * traits.MFMA_LANE_K
            packs.append(
                Vec.from_elements(
                    p_f16[p_base : p_base + traits.MFMA_LANE_K], ctx.elem_dtype
                ).ir_value()
            )
        return packs

    def _bitcast_i32(self, value):
        return fx.Float32(value).bitcast(fx.Int32)

    def _pack_bf16_pair(self, lo, hi, shift, mask):
        lo_i32 = self._bitcast_i32(lo)
        hi_i32 = self._bitcast_i32(hi)
        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_PERM_PACK", "1") == "1"):
            return fx.Int32(rocdl.perm_b32(hi_i32, lo_i32, fx.Int32(0x07060302)))
        return (hi_i32 & mask) | lo_i32.shrui(shift)

    def bf16_trunc_pack_v4(self, f32_vals):
        c16 = fx.Int32(16)
        cmask = fx.Int32(0xFFFF0000)
        packed = [
            self._pack_bf16_pair(f32_vals[0], f32_vals[1], c16, cmask),
            self._pack_bf16_pair(f32_vals[2], f32_vals[3], c16, cmask),
        ]
        return (
            Vec.from_elements(packed, fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()
        )

    def bf16_trunc_pack_v8(self, f32_vals):
        c16 = fx.Int32(16)
        cmask = fx.Int32(0xFFFF0000)
        pairs = []
        for j in range_constexpr(4):
            pairs.append(
                self._pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], c16, cmask)
            )
        return (
            Vec.from_elements(pairs, fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()
        )

    def gemm2_gpfetch_fused(
        self,
        gemm_helper,
        kv_lds_to_vgpr,
        o_accs,
        corr_vec,
        corr,
        l_running,
        s_raw_lo,
        s_raw_hi,
        neg_scaled_max,
        v_base,
        m_new_raw,
        kv_start=None,
    ):
        ctx = self.ctx
        traits = ctx.traits
        local_sum = ctx.c_zero_f
        if const_expr(not traits.USE_HW_TR):
            for dc in range_constexpr(1, traits.D_CHUNKS):
                o_accs[dc] = Vec(o_accs[dc]) * corr_vec
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * traits.MFMA_LANE_K
            p_exp_lo = []
            p_exp_hi = []
            for j in range_constexpr(traits.MFMA_LANE_K):
                diff_lo = self._exp2_arg(
                    s_raw_lo[p_base + j], m_new_raw, neg_scaled_max
                )
                p_exp_lo.append(self._exp2(diff_lo))
                diff_hi = self._exp2_arg(
                    s_raw_hi[p_base + j], m_new_raw, neg_scaled_max
                )
                p_exp_hi.append(self._exp2(diff_hi))
            for j in range_constexpr(traits.MFMA_LANE_K):
                local_sum = local_sum + p_exp_lo[j]
                local_sum = local_sum + p_exp_hi[j]
            if const_expr(ctx.dropout):
                # Dropped after the sum, so l stays undropped (modification 33).
                for first in range_constexpr(0, traits.MFMA_LANE_K, 4):
                    p_exp_lo = self.dropout_group(
                        p_exp_lo, first, kv_start, 0, score=p_base + first
                    )
                    p_exp_hi = self.dropout_group(
                        p_exp_hi,
                        first,
                        kv_start,
                        traits.K_SUB_N,
                        score=p_base + first,
                    )
            p_lo = self.bf16_trunc_pack_v4(p_exp_lo)
            p_hi = self.bf16_trunc_pack_v4(p_exp_hi)
            v_lo = [None] * traits.D_CHUNKS
            v_hi = [None] * traits.D_CHUNKS
            for dc in range_constexpr(traits.D_CHUNKS):
                v_lo[dc], v_hi[dc] = kv_lds_to_vgpr.read_v_pack(
                    dc * traits.PV_K_STEPS + pks, v_base
                )
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = gemm_helper.mfma_acc(v_lo[dc], p_lo, o_accs[dc])
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = gemm_helper.mfma_acc(v_hi[dc], p_hi, o_accs[dc])
        tile_sum = local_sum + self.reduction_peer(local_sum)
        l_new = corr * l_running + tile_sum
        return o_accs, l_new


class GenericStoreHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def zero_o_block(self, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_PERMLANE_OSTORE):
            zero_pack = Vec.filled(4, 0, fx.Int32)
            for dc in range_constexpr(traits.D_CHUNKS):
                for g in range_constexpr(2):
                    d_col = fx.Index(dc * traits.D_CHUNK) + (
                        fx.Index(2 * g) + ctx.lane_div_32
                    ) * fx.Index(8)
                    o_global = ctx.global_idx_o(q_row, d_col)
                    buffer_ops.buffer_store(
                        zero_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True
                    )
        else:
            # f32 (modification 35): 4 elements are 4 dwords at 4 bytes each.
            zero_pack = Vec.filled(traits.ELEM_BYTES, 0, fx.Int32)
            for dc in range_constexpr(traits.D_CHUNKS):
                for grp in range_constexpr(4):
                    d_col = (
                        fx.Index(dc * traits.D_CHUNK)
                        + ctx.lane_div_32 * fx.Index(4)
                        + fx.Index(grp * 8)
                    )
                    o_global = ctx.global_idx_o(q_row, d_col)
                    buffer_ops.buffer_store(
                        zero_pack,
                        o_rsrc,
                        o_global * fx.Index(traits.ELEM_BYTES),
                        offset_is_bytes=True,
                    )

    def with_sink(self, m_final, l_final, q_row):
        """The row's softmax denominator, natural-log LSE and O factor, an attention
        sink's logit included when one is bound (modification 19): a row that reaches
        keys takes the max over its scores and the sink, m_top = max(m, sink), so its
        denominator is l * exp(m - m_top) + exp(sink - m_top) and O is scaled by
        exp(m - m_top) (modification 29); one that reaches none keeps O = 0 and gets
        LSE = sink. Without a sink, LSE = -inf for such a row, as before. A row whose
        sum is NaN keeps a NaN LSE (modification 29): "no keys" is l == +0, tested on
        its bits (`_no_keys`)."""
        ctx = self.ctx
        if ctx.sink is None:
            # A row no key reached has m = -inf and l = 0, so the sum is -inf + -inf by
            # design. Under fast math (no infinities) LLVM may drop it as poison and
            # store a stale register; select -inf instead and take the log without the
            # fast flags (modification 23).
            lse = _no_keys(l_final).select(
                ctx.c_neg_inf,
                fx.Float32(
                    m_final * ctx.c_sm_scale
                    + fx.log(l_final, fastmath=fx.arith.FastMathFlags.none)
                ),
            )
            return l_final, lse, None
        head = ctx.q_head_idx
        if const_expr(ctx.traits.PACK_GQA):
            # A packed row's own head; a row past the group reads the group's first.
            head = head + (q_row < ctx.rows_b).select(
                q_row // ctx.seq_len_v, fx.Index(0)
            )
        # In f32 or the query's dtype (modification 36).
        sink_val = load_sink_logit(
            ctx.sink,
            ctx.sink_stride,
            ctx.sink_on,
            head,
            ctx.elem_dtype,
            ctx.c_neg_inf,
        )
        no_keys = _no_keys(l_final)
        m_scaled = fx.Float32(m_final * ctx.c_sm_scale)
        # The max over the scores and the sink (modification 29): exp(sink - m) alone
        # overflows to +inf once the sink leads the row's max by more than ~88.7.
        m_top = fx.Float32(m_scaled.maximumf(sink_val))
        keep = fx.Float32(fx.math.exp(m_scaled - m_top))
        l_adj = no_keys.select(
            ctx.c_zero_f,
            fx.Float32(l_final) * keep + fx.math.exp(sink_val - m_top),
        )
        lse = no_keys.select(
            sink_val, fx.Float32(m_top + fx.log(l_adj, fastmath=ctx.fm_fast))
        )
        return l_adj, lse, keep

    def store_lse(self, m_final, l_final, q_row):
        # LSE = sm_scale * m_raw + ln(l); natural log, softmax scale folded in
        # (fully-masked row has l == 0 -> -inf, or the sink's logit).
        ctx = self.ctx
        _, lse_val, _ = self.with_sink(m_final, l_final, q_row)
        head, pos = ctx.q_head_idx, q_row
        if const_expr(ctx.traits.PACK_GQA):
            # A packed row's own head and position (modification 15).
            head = head + q_row // ctx.seq_len_v
            pos = q_row % ctx.seq_len_v
        lse_local = head * ctx.lse_stride_h_v + pos * ctx.lse_stride_n_v
        # One writer per row: low half-wave + in-bounds q_row; else redirect to the
        # dropped OOB sentinel.
        off_row = (q_row < ctx.rows_b).select(lse_local, ctx.lse_oob_off)
        if const_expr(ctx.traits.DV_SPLIT > 1):
            # Every column slice computes the same LSE; the first writes it.
            off_row = (ctx.dv_idx == fx.Index(0)).select(off_row, ctx.lse_oob_off)
        off = fx.Index(
            (ctx.lane_div_32 == fx.Index(0)).select(off_row, ctx.lse_oob_off)
        )
        padded = ctx.padded_rows()
        if padded is not None:
            # A padded row's LSE is -inf, on a sink build too: it is padding, not a row
            # no key reached (modification 32). Stored by its bits through a second,
            # redirected store rather than selected into the value: fast math merges such
            # a select with the no-keys test and, assuming no NaN, drops a NaN row's NaN.
            row_live, _ = padded
            padded_off = row_live.select(ctx.lse_oob_off, off)
            off = row_live.select(off, ctx.lse_oob_off)
            buffer_ops.buffer_store(
                as_mlir_value(fx.Int32(-8388608)),  # 0xFF800000, -inf
                ctx.lse_rsrc,
                as_mlir_value(fx.Int32(padded_off)),
            )
        buffer_ops.buffer_store(
            as_mlir_value(fx.Float32(lse_val)),
            ctx.lse_rsrc,
            as_mlir_value(fx.Int32(off)),
        )

    def store_partial(self, loop_results, q_row):
        """One split's normalized f32 O and its LSE, to the workspace (modification 14):
        O at (((b * Hq + h) * Sq + row) * splits + split) * Dv + col, the LSE at the same
        row-and-split index. A split no key reached writes O = 0 and LSE = -inf; one
        whose sum is NaN writes NaN (modification 29)."""
        ctx = self.ctx
        traits = ctx.traits
        m_final = loop_results[0]
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(traits.D_CHUNKS)]
        # "No keys" is l == +0, tested on its bits (`_no_keys`), so a split
        # whose sum is NaN (a NaN or +inf score) keeps a NaN LSE and O, and the merge
        # propagates them, rather than storing the empty split's O = 0, LSE = -inf
        # that the merge skips (modification 29).
        no_keys = _no_keys(l_final)
        lse_val = no_keys.select(
            ctx.c_neg_inf,
            fx.Float32(
                m_final * ctx.c_sm_scale + fx.log(l_final, fastmath=ctx.fm_fast)
            ),
        )
        inv_l = no_keys.select(ctx.c_zero_f, rocdl.rcp(T.f32, l_final))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        row_ok = q_row < ctx.rows_b
        # Each descriptor spans this (batch, head)'s rows exactly -- or, packed, the
        # group's heads' rows, contiguous in the head-major order -- so a row past
        # them, masked to the out-of-range sentinel, is dropped.
        bh = fx.Index(ctx.batch_idx) * ctx.num_heads_q_v + ctx.q_head_idx
        rows_ns = ctx.rows_b * ctx.num_splits_v
        head_rows_ns = ctx.seq_len_v * ctx.num_splits_v
        part_row = q_row * ctx.num_splits_v + ctx.split_idx
        ws_l = buffer_ops.create_buffer_resource(
            ctx.ws_lse,
            max_size=False,
            num_records_bytes=as_mlir_value(rows_ns * fx.Index(4)),
            base_byte_offset=as_mlir_value(bh * head_rows_ns * fx.Index(4)),
        )
        # One writer per row: the low half-wave, and of the column slices the first.
        lse_mask = row_ok & (ctx.lane_div_32 == fx.Index(0))
        if const_expr(traits.DV_SPLIT > 1):
            lse_mask = lse_mask & (ctx.dv_idx == fx.Index(0))
        buffer_ops.buffer_store(
            as_mlir_value(fx.Float32(lse_val)),
            ws_l,
            as_mlir_value(fx.Int32(part_row)),
            mask=lse_mask,
        )
        dv = fx.Index(traits.HEAD_DIM_V)
        ws_o = buffer_ops.create_buffer_resource(
            ctx.ws_o,
            max_size=False,
            num_records_bytes=as_mlir_value(rows_ns * dv * fx.Index(4)),
            base_byte_offset=as_mlir_value(bh * head_rows_ns * dv * fx.Index(4)),
        )
        o_row = part_row * dv + ctx.dv_col_base
        for dc in range_constexpr(traits.D_CHUNKS):
            v_o = Vec(o_finals[dc]) * inv_l_vec
            for grp in range_constexpr(4):
                d_col = (
                    fx.Index(dc * traits.D_CHUNK)
                    + ctx.lane_div_32 * fx.Index(4)
                    + fx.Index(grp * 8)
                )
                # A row no key of this split reached stores 0, selected rather than
                # multiplied: a fully masked q-block's accumulators are not valid, and
                # garbage * 0 can be NaN (modification 27; the direct store zeroes such
                # blocks the same way).
                vals = Vec.from_elements(
                    [
                        no_keys.select(ctx.c_zero_f, fx.Float32(v_o[grp * 4 + i]))
                        for i in range_constexpr(4)
                    ],
                    fx.Float32,
                )
                buffer_ops.buffer_store(
                    vals, ws_o, as_mlir_value(fx.Int32(o_row + d_col)), mask=row_ok
                )

    def finalize_o(self, loop_results):
        # Normalize and store O (128-bit buffer_store_dwordx4); cross-seqlen zeroes
        # fully-masked q-blocks (causal_end_raw <= 0) instead of dividing by l==0.
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.SPLIT_KV):
            # More than one split: partials to the workspace, which the merge kernel
            # combines; one split writes O and LSE as an unsplit launch does.

            @flyc.jit
            def _store_split():
                if ctx.num_splits_v > fx.Index(1):
                    self.store_partial(loop_results, ctx.q_row)
                else:
                    self._finalize_o_direct(loop_results)

            _store_split()
            return
        self._finalize_o_direct(loop_results)

    def _finalize_o_direct(self, loop_results):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):

            @flyc.jit
            def _store_cross_o():
                if ctx.causal_end_raw_i32 <= ctx.c_zero_i32:
                    self.zero_o_block(ctx.o_rsrc, ctx.q_row)
                    if const_expr(traits.RETURN_LSE):
                        # No valid keys for this q-tile: LSE = ln(0) = -inf per row.
                        self.store_lse(loop_results[0], loop_results[1], ctx.q_row)
                else:
                    self.normalize_and_store_o(loop_results, ctx.o_rsrc, ctx.q_row)

            _store_cross_o()
        else:
            self.normalize_and_store_o(loop_results, ctx.o_rsrc, ctx.q_row)

    def normalize_and_store_o(self, loop_results, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(traits.D_CHUNKS)]

        if const_expr(traits.RETURN_LSE):
            self.store_lse(loop_results[0], l_final, q_row)
        if ctx.sink is not None:
            # The sink's share of the denominator (modification 19), against the max
            # over the scores and the sink, which also scales O by keep
            # (modification 29). A row with l == 0 gets O = 0; a NaN row stays NaN.
            l_final, _, keep = self.with_sink(loop_results[0], l_final, q_row)
            inv_l = _no_keys(l_final).select(
                ctx.c_zero_f, keep * rocdl.rcp(T.f32, l_final)
            )
        else:
            inv_l_rcp = rocdl.rcp(T.f32, l_final)
            if const_expr(traits.CAUSAL or getattr(ctx, "bias_rsrc", None) is not None):
                inv_l = (fx.Float32(l_final) > ctx.c_zero_f).select(
                    inv_l_rcp, ctx.c_zero_f
                )
            elif ctx.varlen_flags is not None:
                # A padded batch may have no keys, so even with no mask a row can sum
                # to 0: O = 0 rather than 0 * inf (modification 32). "No keys" is
                # tested on the bits, so a NaN row stays NaN (`_no_keys`).
                inv_l = _no_keys(l_final).select(ctx.c_zero_f, fx.Float32(inv_l_rcp))
            else:
                inv_l = inv_l_rcp
        if ctx.dropout:
            # Kept probabilities scaled by 1 / (1 - p), O only: the LSE above is the
            # undropped one (modification 33).
            inv_l = fx.Float32(inv_l) * ctx.dropout_scale
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(traits.D_CHUNKS)]

        def store(row_ok=None):
            if const_expr(traits.USE_PERMLANE_OSTORE):
                self._store_permlane_o(v_o, o_rsrc, q_row, row_ok)
            else:
                self._store_fallback_o(v_o, o_rsrc, q_row, row_ok)

        padded = ctx.padded_rows()
        if padded is None:
            store()
            return

        # A padded row is stored as 0 (modification 32): its words zeroed after they are
        # packed to the output dtype, never by a select on O's f32 values, which fast math
        # would fold into 1 / l's and so lose a NaN row's NaN.
        row_live, _ = padded
        store(row_live)

    @staticmethod
    def _row_words(words, row_ok):
        """Packed output words, or zeros for a row @p row_ok excludes (modification 32)."""
        if row_ok is None:
            return words
        return [row_ok.select(word, fx.Int32(0)) for word in words]

    def _store_permlane_o(self, v_o, o_rsrc, q_row, row_ok=None):
        ctx = self.ctx
        traits = ctx.traits
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        is_hi_half = ctx.lane_div_32 != fx.Index(0)

        def _o_pack_2dw(dc, store_group):
            r_base = store_group * 4
            if const_expr(traits.DTYPE_STR == "bf16"):
                lo = rocdl.cvt_pk_bf16_f32(
                    Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1]
                )
                hi = rocdl.cvt_pk_bf16_f32(
                    Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3]
                )
                return lo, hi
            o_f16 = [
                fx.Float32(Vec(v_o[dc])[r_base + i]).to(ctx.elem_dtype)
                for i in range_constexpr(4)
            ]
            pack = Vec.from_elements(o_f16, ctx.elem_dtype).bitcast(fx.Int32)
            return as_mlir_value(pack[0]), as_mlir_value(pack[1])

        def _swap_halves(dw):
            swapped = rocdl.permlane32_swap(
                pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
            )
            lo_res = llvm.extractvalue(T.i32, swapped, [0])
            hi_res = llvm.extractvalue(T.i32, swapped, [1])
            return is_hi_half.select(lo_res, hi_res)

        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
                w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
                w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
                w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
                words = [fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)]
                # A lane's words are its own row's (its xor-32 partner holds the same
                # row), so a row @p row_ok excludes stores zeros (modification 32).
                o_pack = Vec.from_elements(self._row_words(words, row_ok), fx.Int32)
                d_col = fx.Index(dc * traits.D_CHUNK) + (
                    fx.Index(2 * g) + ctx.lane_div_32
                ) * fx.Index(8)
                o_global = ctx.global_idx_o(q_row, d_col)
                buffer_ops.buffer_store(
                    o_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True
                )

    def _store_fallback_o(self, v_o, o_rsrc, q_row, row_ok=None):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.DTYPE_STR == "f32"):
            # f32 (modification 35): a lane's 4 consecutive head-dim values are one
            # dwordx4, stored as their bits so a padded row's zeros are an integer select
            # (modification 32), never an FP one.
            for dc in range_constexpr(traits.D_CHUNKS):
                for grp in range_constexpr(4):
                    r0 = grp * 4
                    o4 = Vec.from_elements(
                        self._row_words(
                            [
                                fx.Float32(Vec(v_o[dc])[r0 + i]).bitcast(fx.Int32)
                                for i in range_constexpr(4)
                            ],
                            row_ok,
                        ),
                        fx.Int32,
                    )
                    d_col = (
                        fx.Index(dc * traits.D_CHUNK)
                        + ctx.lane_div_32 * fx.Index(4)
                        + fx.Index(grp * 8)
                    )
                    o_global = ctx.global_idx_o(q_row, d_col)
                    buffer_ops.buffer_store(
                        o4, o_rsrc, o_global * fx.Index(4), offset_is_bytes=True
                    )
            return
        for dc in range_constexpr(traits.D_CHUNKS):
            for grp in range_constexpr(4):
                r0 = grp * 4
                o_f16 = [
                    fx.Float32(Vec(v_o[dc])[r0 + i]).to(ctx.elem_dtype)
                    for i in range_constexpr(4)
                ]
                pack = Vec.from_elements(o_f16, ctx.elem_dtype).bitcast(fx.Int32)
                o2 = Vec.from_elements(
                    self._row_words([fx.Int32(pack[0]), fx.Int32(pack[1])], row_ok),
                    fx.Int32,
                )
                d_col = (
                    fx.Index(dc * traits.D_CHUNK)
                    + ctx.lane_div_32 * fx.Index(4)
                    + fx.Index(grp * 8)
                )
                o_global = ctx.global_idx_o(q_row, d_col)
                buffer_ops.buffer_store(
                    o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True
                )


class DualwaveKernelContext:
    """Shared per-kernel state for the gfx950 dualwave attention helpers."""

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,  # noqa: E741
        DebugCounts=None,
        CuSeqQ=None,
        CuSeqKv=None,
        BlockTable=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        block_table_stride=None,
        LSE=None,
        sm_scale=None,
        num_heads_q=None,
        num_heads_kv=None,
        lse_on=None,
        diag_offset=None,
        view_strides=None,
    ):
        if isinstance(traits_or_ctx, DualwaveKernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.LSE = LSE
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.BlockTable = BlockTable
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.block_table_stride = block_table_stride
        self.sm_scale = sm_scale
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.lse_on = lse_on
        self.diag_offset = diag_offset
        self.view_strides = view_strides

    def init_types_and_constants(self, head_dim_runtime=None, scaled_scores=False):
        if head_dim_runtime is None:
            head_dim_runtime = self.head_dim_runtime
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.q_load_i32x4_type = Vec.make_type(4, fx.Int32)
        self.v_lds_read_vec4_type = Vec.make_type(4, self.elem_dtype)
        self.kv_mfma_pack_type = Vec.make_type(8, self.elem_dtype)
        self.mfma_acc_vec_type = Vec.make_type(16, fx.Float32)

        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        # LSE store folds the log2->ln conversion (m_row is sm_scale*log2e-scaled).
        self.c_ln2_f = fx.Float32(1.0 / _LOG2E)
        if self.sm_scale is None:
            self.c_sm_scale_log2e = (
                fx.rsqrt(head_dim_f32, fastmath=self.fm_fast) * c_log2e_f
            )
        else:
            # Runtime softmax scale (modification 2).
            self.c_sm_scale_log2e = fx.Float32(self.sm_scale) * c_log2e_f
        self.scaled_scores = scaled_scores
        if scaled_scores:
            # A bias build scales its scores and adds the bias before the softmax, as
            # the reference does (modification 28): the running max is a scaled score,
            # the exponent takes only log2(e), and the LSE store takes the max in
            # natural units, since times log2(e) a row the bias masks at the dtype's
            # lowest value would overflow.
            if self.sm_scale is None:
                self.c_score_scale = fx.rsqrt(head_dim_f32, fastmath=self.fm_fast)
            else:
                self.c_score_scale = fx.Float32(self.sm_scale)
            self.c_sm_scale_log2e = c_log2e_f
            self.c_ln2_f = fx.Float32(1.0)

    def init_runtime_indices(
        self, seq_len=None, seq_len_kv=None, stride_q_n=None, stride_kv_n=None
    ):
        if seq_len is None:
            seq_len = self.seq_len
        if seq_len_kv is None:
            seq_len_kv = self.seq_len_kv
        if stride_q_n is None:
            stride_q_n = self.stride_q_n
        if stride_kv_n is None:
            stride_kv_n = self.stride_kv_n
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.seq_len_v = fx.Index(seq_len)
        self.seq_len_kv_v = fx.Index(seq_len_kv)
        self.stride_q_n_v = fx.Index(stride_q_n)
        self.stride_kv_n_v = fx.Index(stride_kv_n)

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr
        if const_expr(self.traits.PAGED):
            self.lds_bt_base_idx = fx.Index(fx.ptrtoint(lds.bt.ptr))
            self.lds_bt_base_ptr = lds.bt.ptr.llvm_ptr
        else:
            self.lds_bt_base_ptr = None

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_sequence_lengths(self, CuSeqQ=None, CuSeqKv=None):
        if CuSeqQ is None:
            CuSeqQ = self.CuSeqQ
        if CuSeqKv is None:
            CuSeqKv = self.CuSeqKv
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1)
            )
            _cuk_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(
                _cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(
                _cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv

    def init_descriptors(
        self,
        q_tensor=None,
        k_tensor=None,
        v_tensor=None,
        o_tensor=None,
        block_table=None,
        block_table_stride=None,
    ):
        if q_tensor is None:
            q_tensor = self.Q
        if k_tensor is None:
            k_tensor = self.K
        if v_tensor is None:
            v_tensor = self.V
        if o_tensor is None:
            o_tensor = self.O
        if block_table is None:
            block_table = self.BlockTable
        if block_table_stride is None:
            block_table_stride = self.block_table_stride
        traits = self.traits
        if const_expr(traits.PAGED):
            self.block_table_stride_v = fx.Index(block_table_stride)
            self.bt_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(block_table), fx.make_layout(1, 1)
            )
            self.bt_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            self.bt_v1i32 = Vec.make_type(1, fx.Int32)
            self.kv_head_elem_offset = self.kv_head_idx * traits.HEAD_DIM
        else:
            self.block_table_stride_v = None
            self.bt_div = None
            self.bt_atom = None
            self.bt_v1i32 = None
            self.kv_head_elem_offset = None

        # The causal diagonal: query row i sees keys j <= i + delta. Bottom-right
        # alignment unless the caller passes its own offset (modification 5).
        if getattr(self, "diag_offset", None) is None:
            self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        else:
            self.delta_i32 = fx.Int32(self.diag_offset)
        self.buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
        self.elem_ir = self.elem_dtype.ir_type
        view_strides = getattr(self, "view_strides", None)
        if view_strides is not None:
            self._init_strided_views(
                q_tensor, k_tensor, v_tensor, o_tensor, view_strides
            )
            return
        self.o_stride_n_v = self.stride_q_n_v
        self.o_head_elem_offset = self.q_head_idx * traits.HEAD_DIM
        self.q_gmem_elem_offset = (
            self.q_start * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        )
        self.kv_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM
        qo_per_batch_elems = self.seqlen_q_v * self.stride_q_n_v
        qo_nrec_bytes = qo_per_batch_elems * fx.Index(traits.BF16_BYTES)
        qo_layout = fx.make_layout(fx.Int32(qo_per_batch_elems), fx.Int32(1))
        q_batch_byte_off = (
            self.q_tok_base * self.stride_q_n_v * fx.Index(traits.BF16_BYTES)
        )
        self.q_div = _make_rebased_view(
            fx.get_iter(q_tensor),
            q_batch_byte_off,
            qo_nrec_bytes,
            qo_layout,
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )
        self.o_div = _make_rebased_view(
            fx.get_iter(o_tensor),
            q_batch_byte_off,
            qo_nrec_bytes,
            qo_layout,
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

        if const_expr(traits.PAGED):
            self.k_div = None
            self.v_div = None
            page_elems = fx.Index(traits.BLOCK_N) * self.stride_kv_n_v
            self.page_byte_stride = page_elems * fx.Index(traits.BF16_BYTES)
            self.page_nrec_bytes = fx.Int64(self.page_byte_stride)
            self.page_layout = fx.make_layout(fx.Int32(page_elems), fx.Int32(1))
        else:
            kv_per_batch_elems = self.seqlen_kv_v * self.stride_kv_n_v
            kv_nrec_bytes = kv_per_batch_elems * fx.Index(traits.BF16_BYTES)
            kv_layout = fx.make_layout(fx.Int32(kv_per_batch_elems), fx.Int32(1))
            kv_batch_byte_off = (
                self.kv_tok_base * self.stride_kv_n_v * fx.Index(traits.BF16_BYTES)
            )
            self.k_div = _make_rebased_view(
                fx.get_iter(k_tensor),
                kv_batch_byte_off,
                kv_nrec_bytes,
                kv_layout,
                _buf_flags_i32=self.buf_flags_i32,
                _elem_ir=self.elem_ir,
            )
            self.v_div = _make_rebased_view(
                fx.get_iter(v_tensor),
                kv_batch_byte_off,
                kv_nrec_bytes,
                kv_layout,
                _buf_flags_i32=self.buf_flags_i32,
                _elem_ir=self.elem_ir,
            )
            self.page_byte_stride = None
            self.page_nrec_bytes = None
            self.page_layout = None
        self.debug_counts_rsrc = (
            _make_raw_buffer_rsrc(self.DebugCounts)
            if traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS
            else None
        )

    def _init_strided_views(self, q_tensor, k_tensor, v_tensor, o_tensor, strides):
        """Per-(batch, head) views of Q, O, K and V under runtime strides
        (modification 6).

        Each view starts at its head's first row and ends at that head's last
        element, (seq_len - 1) * stride_n + head_dim, so any (batch, sequence,
        head) stride order -- BSHD, BHSD, a packed-QKV view -- reads rows past the
        sequence as zero and drops stores to them, as the per-batch BSHD views
        did. The head offset lives in the view's base, not in each lane's index.
        """
        traits = self.traits
        if const_expr(traits.PAGED or traits.VARLEN or traits.SPLITK):
            raise ValueError("runtime strides serve the dense, unsplit kernel only")
        elem_bytes = fx.Index(traits.BF16_BYTES)
        head_dim = fx.Index(traits.HEAD_DIM)
        batch = fx.Index(self.batch_idx)

        def view(tensor, stride_b, stride_n, stride_h, head, rows):
            base_bytes = (
                batch * fx.Index(stride_b) + head * fx.Index(stride_h)
            ) * elem_bytes
            elems = (rows - fx.Index(1)) * fx.Index(stride_n) + head_dim
            return _make_rebased_view(
                fx.get_iter(tensor),
                base_bytes,
                elems * elem_bytes,
                fx.make_layout(fx.Int32(elems), fx.Int32(1)),
                _buf_flags_i32=self.buf_flags_i32,
                _elem_ir=self.elem_ir,
            )

        self.q_div = view(
            q_tensor,
            strides["q_b"],
            self.stride_q_n_v,
            strides["q_h"],
            self.q_head_idx,
            self.seqlen_q_v,
        )
        self.o_div = view(
            o_tensor,
            strides["o_b"],
            strides["o_n"],
            strides["o_h"],
            self.q_head_idx,
            self.seqlen_q_v,
        )
        self.k_div = view(
            k_tensor,
            strides["kv_b"],
            self.stride_kv_n_v,
            strides["kv_h"],
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.v_div = view(
            v_tensor,
            strides["kv_b"],
            self.stride_kv_n_v,
            strides["kv_h"],
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.o_stride_n_v = fx.Index(strides["o_n"])
        self.o_head_elem_offset = fx.Index(0)
        self.q_gmem_elem_offset = self.q_start * self.stride_q_n_v
        self.kv_gmem_elem_offset = fx.Index(0)
        self.page_byte_stride = None
        self.page_nrec_bytes = None
        self.page_layout = None
        self.debug_counts_rsrc = None

    def init_workspace(self, DebugCounts=None):
        if DebugCounts is None:
            DebugCounts = self.DebugCounts
        traits = self.traits
        if const_expr(traits.SPLITK):
            self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(DebugCounts)))
            self.ws_opart_per_split_elems = (
                fx.Index(traits.NUM_HEADS_Q)
                * self.seq_len_v
                * fx.Index(traits.HEAD_DIM // 2)
            )
            self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
            self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
            self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
            self.ws_grid_z = fx.Index(gpu.grid_dim.z)
            self.ws_mrow_abs_bytes = self.ws_grid_z * self.ws_opart_per_split_bytes
            self.ws_lrow_abs_bytes = (
                self.ws_mrow_abs_bytes + self.ws_grid_z * self.ws_ml_per_split_bytes
            )
        else:
            self.ws_base_i64 = None
            self.ws_opart_per_split_bytes = None
            self.ws_ml_per_split_bytes = None
            self.ws_mrow_abs_bytes = None
            self.ws_lrow_abs_bytes = None

    def init_atoms_and_lds_ptrs(self):
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.lds_ptr_ty = fx.PointerType.get(
            self.elem_dtype.ir_type, 2, self.traits.DMA_BYTES
        )

    def init_dma_thread_offsets(self):
        self.lane_in_warp = self.tid % self.traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // self.traits.VEC_KV
        self.d_bucket = self.lane_in_warp % self.traits.VEC_KV

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_dma_base(self, buf_id, d):
        return _k_dma_m0_base(
            self.traits,
            buf_id,
            d,
            lane_in_warp=self.lane_in_warp,
            lds_kv_base_idx=self.lds_kv_base_idx,
            wave_id_uni=self.wave_id_uni,
        )

    def v_dma_base(self, buf_id, d):
        return _v_dma_m0_base(
            self.traits,
            buf_id,
            d,
            lane_in_warp=self.lane_in_warp,
            lds_kv_base_idx=self.lds_kv_base_idx,
            wave_id_uni=self.wave_id_uni,
        )

    def dma_m0_table(self, base_fn, count):
        return tuple(tuple(base_fn(buf, d) for d in range(count)) for buf in range(2))

    def init_dma_m0_tables(self):
        self.k_dma_m0 = self.dma_m0_table(self.k_dma_base, self.NUM_DMA_K)
        self.v_dma_m0 = self.dma_m0_table(self.v_dma_base, self.NUM_DMA_V)

    def init_tile_bounds(self):
        traits = self.traits
        self.kv_tile_size = traits.BLOCK_N
        self.num_kv_tiles = (
            self.seqlen_kv_v + self.kv_tile_size - 1
        ) // self.kv_tile_size
        if const_expr(traits.CAUSAL):
            self.causal_end_raw_i32 = (
                fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            )
            causal_end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(
                    self.causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_num_tiles = (
                fx.Index(causal_end_i32) + self.kv_tile_size - 1
            ) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                (causal_num_tiles < self.num_kv_tiles).select(
                    causal_num_tiles, self.num_kv_tiles
                )
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        self.max_num_tiles = (
            (self.max_num_tiles + fx.Index(1)) // fx.Index(2)
        ) * fx.Index(2)
        self.max_num_tiles = fx.Index(
            (self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles)
        )

        if const_expr(traits.SPLITK):
            chunk = (
                (
                    (self.max_num_tiles + (traits.NUM_KV_SPLITS - 1))
                    // traits.NUM_KV_SPLITS
                    + 1
                )
                // 2
                * 2
            )
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            self.split_t0 = self.split_idx * chunk
            self.split_t_end = self.split_t0 + chunk
            self.split_t_end = fx.Index(
                (self.split_t_end < self.max_num_tiles).select(
                    self.split_t_end, self.max_num_tiles
                )
            )
            self.split_t_end = fx.Index(
                (self.max_num_tiles - self.split_t_end < fx.Index(4)).select(
                    self.max_num_tiles, self.split_t_end
                )
            )
            self.split_nonempty = self.split_t0 + fx.Index(4) <= self.max_num_tiles
        else:
            self.split_t0 = 0
            self.split_t_end = self.max_num_tiles

    def compute_active_guard(self):
        traits = self.traits
        if const_expr(traits.SPLITK):
            return self.split_nonempty
        if const_expr(traits.VARLEN):
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                return (self.q_start < self.seqlen_q_v) & (
                    self.causal_end_raw_i32 > fx.Int32(0)
                )
            return self.q_start < self.seqlen_q_v
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            return self.causal_end_raw_i32 > fx.Int32(0)
        return None

    def init_active_guard(self):
        self.active = self.compute_active_guard()

    def init_lds_read_bases(self):
        self.k_lds_read_base_per_lane = _k_lds_read_base_per_lane(
            self.traits, self.lane_mod_32, self.lane_div_32
        )
        self.v_lds_read_base_per_lane = _v_lds_read_base_per_lane(
            self.traits, self.lane, self.lane_div_32
        )

    def split_tile(self, offset_tiles=0):
        return self.split_t0 + fx.Index(offset_tiles)

    def tile_start(self, tile_idx):
        return tile_idx * self.traits.BLOCK_N


# Pipeline helpers


class DualwavePageIdLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_block_table_to_lds(self):
        traits = self.traits
        tid = self.tid
        split_t0 = self.split_t0
        split_t_end = self.split_t_end
        num_kv_tiles = self.num_kv_tiles
        batch_idx = self.batch_idx
        block_table_stride_v = self.block_table_stride_v
        lds_bt_base_ptr = self.lds_bt_base_ptr
        bt_div = self.bt_div
        bt_atom = self.bt_atom
        bt_v1i32 = self.bt_v1i32

        @flyc.jit
        def _load_block_table_to_lds():
            segment_tiles = split_t_end - split_t0
            for pass_id in range_constexpr(
                traits.PAGED_BT_LDS_SIZE // traits.BLOCK_SIZE
            ):
                local_tile = tid + fx.Index(pass_id * traits.BLOCK_SIZE)
                if local_tile < segment_tiles:
                    tile_idx = split_t0 + local_tile
                    byte_off = as_mlir_value(fx.Int32(local_tile * fx.Index(4)))
                    dst = buffer_ops.get_element_ptr(
                        lds_bt_base_ptr, byte_offset=byte_off, elem_type=T.i8
                    )
                    llvm.StoreOp(as_mlir_value(fx.Int32(0)), dst)
                    if tile_idx < num_kv_tiles:
                        row_idx = batch_idx * block_table_stride_v + tile_idx
                        v = fly.copy_atom_call_ssa(
                            [bt_v1i32],
                            bt_atom,
                            fx.slice(bt_div, (None, fx.Int32(row_idx))),
                        )
                        page_id_i32 = as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0]))
                        llvm.StoreOp(page_id_i32, dst)

        _load_block_table_to_lds()

    def load_page_id_lds(self, tile_idx):
        src = buffer_ops.get_element_ptr(
            self.lds_bt_base_ptr,
            byte_offset=as_mlir_value(
                _paged_bt_byte_offset(tile_idx, split_t0=self.split_t0)
            ),
            elem_type=T.i8,
        )
        return llvm.LoadOp(T.i32, src).result

    def finish_page_id(self, v):
        rocdl.s_waitcnt(self.traits.LGKMCNT_0_ONLY)
        v = rocdl.readfirstlane(T.i32, v)
        return fx.Index(fx.Int32(v))

    def async_load_tile_page_id(self, tile_idx, page_id_override=None):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id_override is not None):
                return page_id_override
            page_id = self.load_page_id_lds(tile_idx)
            return self.finish_page_id(page_id)
        return fx.Index(0)

    def async_load_split_page(self, offset_tiles=0, page_id_override=None):
        return self.async_load_tile_page_id(
            self.split_tile(offset_tiles), page_id_override=page_id_override
        )

    def async_load_page_id(self, tile_start, page_id_override=None):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id_override is not None):
                return page_id_override
            return self.async_load_tile_page_id(
                tile_start // fx.Index(self.traits.BLOCK_N)
            )
        return fx.Index(0)


class DualwaveQLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_pack(self, q_row_in_block, ks):
        q_i32_pack = _buffer_load_128(
            self.q_gmem_elem_offset
            + _q_pack_global_idx(
                self.traits,
                q_row_in_block,
                ks,
                lane_div_32=self.lane_div_32,
                stride_q_n_v=self.stride_q_n_v,
            ),
            _load_atom_128=self.load_atom_128,
            q_div=self.q_div,
            q_load_i32x4_type=self.q_load_i32x4_type,
        )
        return Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()

    def load_all(self):
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()

        q_raw_packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            q_raw_packs.append(self.load_pack(ctx.q_row_in_block, ks))
        q_16_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 2):
            q_16_packs.append(
                _concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1])
            )

        q_32_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 4):
            q_32_packs.append(
                _concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1])
            )

        q_all = (
            q_32_packs[0]
            if const_expr(traits.K_STEPS_QK == 4)
            else _concat_vectors(q_32_packs[0], q_32_packs[1])
        )
        return Vec(q_all, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)

    def scale_all(self, q_all_bf16):
        traits = self.traits
        fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
        v64bf16_type = Vec.make_type(
            traits.K_STEPS_QK * traits.MFMA_LANE_K, self.elem_dtype
        )
        v64f32_type = Vec.make_type(traits.K_STEPS_QK * traits.MFMA_LANE_K, fx.Float32)
        q_all_f32_op = llvm.FPExtOp(v64f32_type, as_mlir_value(q_all_bf16))
        q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_f32 = q_all_f32_op.result
        scale_vec = Vec.from_elements([self.c_sm_scale_log2e], fx.Float32).broadcast_to(
            traits.K_STEPS_QK * traits.MFMA_LANE_K
        )
        q_all_scaled_f32 = as_mlir_value(scale_vec * Vec(q_all_f32))
        q_all_scaled_bf16_op = llvm.FPTruncOp(v64bf16_type, q_all_scaled_f32)
        q_all_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_scaled_bf16 = q_all_scaled_bf16_op.result
        return Vec(
            q_all_scaled_bf16,
            (traits.K_STEPS_QK * traits.MFMA_LANE_K,),
            self.elem_dtype,
        )


class DualwaveGemmHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def qk(self, v_k, q_all_scaled_bf16):
        k_lo, k_hi = v_k
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            q_pack = _get_q_pack(self.traits, q_all_scaled_bf16, ks)
            v_s_lo = _mfma_acc(
                k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type
            )
            v_s_hi = _mfma_acc(
                k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type
            )
        # Raw scores: DualwaveSoftmaxHelper applies the softmax scale (modification 21).
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o):
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _mfma_acc(
                v_pk[dc], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type
            )
        return v_o

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o


class DualwaveSoftmaxHelper(DualwaveKernelContext):
    """Softmax over raw scores (modification 21).

    The QK MFMA emits unscaled scores and the running max m_row is kept over them;
    ``sm_scale * log2e`` enters where the max is subtracted (one FMA, as the fp8
    path's ``_scale_sub_score_pair`` and the generic kernel do) and where two maxima
    are compared. Upstream instead rounded ``Q * sm_scale * log2e`` to the element
    type before the MFMA. The scale must be positive, which the engine checks;
    ``scaled_max`` turns m_row into the scaled domain the sink and stores expect.
    A bias build (``scaled_scores``) scales its scores before the bias instead, and
    keeps the max over them (modification 28).
    """

    # The floor for a row with no key yet. Finite after scaling by any scale below
    # 3e8, unlike the context's -3e38, so ``fma(-inf, scale, -scale * floor)`` stays
    # -inf rather than NaN.
    _RAW_NEG_FLOOR = -1.0e30

    def __init__(self, ctx):
        super().__init__(ctx)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def floor_masked_max(self, row_max):
        if const_expr(getattr(self, "scaled_scores", False)):
            # A bias build's scores are scaled already, and its floor lifts only a row
            # of -inf, not one the bias masks at the dtype's lowest value (modification 28).
            return fx.maxnumf(row_max, fx.Float32(_BIAS_NEG_FLOOR))
        return fx.maxnumf(row_max, fx.Float32(self._RAW_NEG_FLOOR))

    def scaled_max(self, m_row):
        if const_expr(getattr(self, "scaled_scores", False)):
            # Already scaled, and kept in natural units (modification 28).
            return m_row
        return m_row * self.c_sm_scale_log2e

    def _corr(self, m_row, m_new):
        return rocdl.exp2(T.f32, as_mlir_value((m_row - m_new) * self.c_sm_scale_log2e))

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        return row_max, self._corr(m_row, row_max)

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def reduce_sum(self, l_row, v_p):
        return l_row + _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def sub_m(self, v_s, row_max):
        if isinstance(v_s[0], list):  # masked tiles arrive as per-element lists
            v_s = _score_lists_to_vecs(v_s)
        if const_expr(getattr(self, "scaled_scores", False)):
            # (s - m) * log2(e), the difference first: near -FLT_MAX the fused form's
            # s * log2(e) and -m * log2(e) overflow to -inf + inf (modification 28).
            log2e_v = Vec.from_elements(
                [self.c_sm_scale_log2e], fx.Float32
            ).broadcast_to(16)
            s_lo, s_hi = v_s
            m_v = Vec.from_elements([row_max], fx.Float32).broadcast_to(16)
            return (
                as_mlir_value((Vec(s_lo) - m_v) * log2e_v),
                as_mlir_value((Vec(s_hi) - m_v) * log2e_v),
            )
        return _scale_sub_score_pair(
            v_s, row_max, self.c_sm_scale_log2e, self.c_zero_f, self.fm_fast
        )

    def cast_p(self, v_p):
        return _pack_p_v8_slices(
            self.traits,
            v_p,
            lambda vals: _bf16_trunc_pack_v8(
                self.traits, vals, elem_dtype=self.elem_dtype
            ),
        )

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new = fx.maxnumf(m_row, m_tile_max)
        corr = self._corr(m_row, m_new)
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        v_p = _scale_v_p(
            self.traits,
            v_p,
            corr,
            elem_dtype=self.elem_dtype,
            fm_fast=self.fm_fast,
        )
        l_row = l_row * corr
        return v_o, m_new, l_row, v_p

    def fold_sink(self, v_o, m_row, l_row, sink_log2):
        m_new = fx.maxnumf(m_row, sink_log2)
        corr = rocdl.exp2(T.f32, as_mlir_value(m_row - m_new))
        self.scale_o(v_o, corr)
        sink_w = rocdl.exp2(T.f32, as_mlir_value(sink_log2 - m_new))
        l_row = l_row * corr + sink_w
        return m_new, l_row

    def _lazy_rescale_o_rescale(self, v_o, m_row, l_row, m_tile_max, v_p):
        # Lanes below their running max are dragged into this wave-uniform branch and
        # m_tile_max alone would overflow them. nnan, not the ambient: -inf is live here.
        m_new = fx.maxnumf(m_row, m_tile_max, fastmath=arith.FastMathFlags.nnan)
        corr = self._corr(m_row, m_new)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        out = [as_mlir_value(scaled_accs[dc]) for dc in range(self.traits.D_CHUNKS)]
        scaled_p = _scale_v_p(
            self.traits,
            v_p,
            corr,
            elem_dtype=self.elem_dtype,
            fm_fast=self.fm_fast,
        )
        out.append(_v_p_to_vec32(scaled_p))
        out.append(as_mlir_value(l_row * corr))
        out.append(_anchor_scalar_f32(m_new))
        return out

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        traits = self.traits
        lane = self.lane
        debug_counts_rsrc = self.debug_counts_rsrc

        @flyc.jit
        def _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p):
            c_eight_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
            m_diff = (m_tile_max - m_row) * self.c_sm_scale_log2e
            below = fx.Float32(m_diff) <= c_eight_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())
            _debug_count_lazy_branch(
                traits,
                all_below,
                debug_counts_rsrc=debug_counts_rsrc,
                lane=lane,
            )

            _res = [as_mlir_value(v_o[dc]) for dc in range(traits.D_CHUNKS)]
            _res += [_v_p_to_vec32(v_p), as_mlir_value(l_row), as_mlir_value(m_row)]
            if fx.Boolean(all_below):
                pass
            else:
                _res = self._lazy_rescale_o_rescale(v_o, m_row, l_row, m_tile_max, v_p)

            o_out = list(_res[0 : traits.D_CHUNKS])
            vp_out = _res[traits.D_CHUNKS]
            l_out = _res[traits.D_CHUNKS + 1]
            m_out = _res[traits.D_CHUNKS + 2]
            return (
                o_out,
                m_out,
                l_out,
                _v_vec32_to_p(traits, vp_out, elem_dtype=self.elem_dtype),
            )

        return _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p)

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        """Apply causal mask using DUALWAVE_SWP inline-asm attn_mask_vec2_imm."""
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        # lane>=32 has a larger n offset in the K-permuted P layout.
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(lane_n_off)
        rel_lo_i32 = fx.Int32(q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

        pair_thresholds = _causal_pair_thresholds(traits.KV_VECTORIZED)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

    def causal_mask_prologue_if_needed(
        self,
        v_s,
        tile_idx=None,
        kv_end_pos=None,
        q_start_pos_i32=None,
        q_row_i32=None,
        *,
        kv_end_tile=None,
    ):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            end_tile = tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            kv_end_pos = self.tile_start(end_tile)
        if q_start_pos_i32 is None:
            q_start_pos_i32 = self.ctx_ref.q_start_pos_i32
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32

        @flyc.jit
        def _causal_mask_prologue_if_needed(
            v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32
        ):
            s_lo, s_hi = v_s
            if q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace(
                    (lo_list, hi_list), tile_idx, q_row_i32=q_row_i32
                )
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _causal_mask_prologue_if_needed(
            v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32
        )

    def causal_mask_split_prologue_if_needed(
        self, v_s, offset_tiles=0, end_offset_tiles=1
    ):
        return self.causal_mask_prologue_if_needed(
            v_s,
            self.split_tile(offset_tiles),
            kv_end_tile=self.split_tile(end_offset_tiles),
        )

    def seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        s_lo, s_hi = v_s_lists
        col_base = _seq_pad_col_base(
            self.traits, tile_idx, lane_div_32=self.lane_div_32
        )
        for r in range_constexpr(16):
            col_lo = col_base + fx.Int32(_seq_pad_score_threshold(self.traits, r))
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        traits = self.traits
        seqlen_kv_i32 = self.seqlen_kv_i32

        @flyc.jit
        def _seq_pad_mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * traits.BLOCK_N
            if fx.Int32(kv_tile_end) > seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self.seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)


class DualwaveKvGmemToLdsLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.num_dma_k = self.NUM_DMA_K
        self.num_dma_v = self.NUM_DMA_V

    def _issue_kv_dma(self, src_div, lds_addr, src_elem, soffset):
        _buffer_load_lds_128(
            src_div,
            lds_addr,
            src_elem,
            soffset,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def _kv_src_div(self, tensor, dense_div, page_id, name):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id is None):
                raise ValueError(f"{name} requires page_id when PAGED=True")
            base_iter = fx.get_iter(tensor)
            return _make_page_view(
                base_iter,
                base_iter.type,
                base_iter.alignment,
                page_id,
                self.page_byte_stride,
                self.page_nrec_bytes,
                self.page_layout,
                self.elem_ir,
                self.buf_flags_i32,
            )
        return dense_div

    def _async_load_kv_linear(
        self, dma_m0, buf_id, src_div, src_base, soffset, num_dma
    ):
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(
                src_div,
                dma_m0[buf_id][d],
                _linear_kv_src_elem(
                    self.traits,
                    src_base,
                    d,
                    n_in_warp=self.n_in_warp,
                    wave_id=self.wave_id,
                    d_bucket=self.d_bucket,
                    stride_kv_n_v=self.stride_kv_n_v,
                ),
                soffset,
            )

    def load_k(self, tile_start, buf_id, page_id=None):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            self.traits,
            tile_start,
            kv_gmem_elem_offset=self.kv_gmem_elem_offset,
            kv_head_elem_offset=self.kv_head_elem_offset,
            stride_kv_n_v=self.stride_kv_n_v,
        )
        src_div = self._kv_src_div(
            self.K, self.k_div, page_id, "DualwaveKvGmemToLdsLoader.load_k"
        )
        if const_expr(self.traits.KV_VECTORIZED):
            for d in range_constexpr(self.num_dma_k):
                self._issue_kv_dma(
                    src_div,
                    ctx.k_dma_m0[buf_id][d],
                    _vec_k_src_elem(
                        self.traits,
                        d,
                        wave_id_uni=self.wave_id_uni,
                        lane_in_warp=self.lane_in_warp,
                        kv_head_idx=self.kv_head_idx,
                    ),
                    soffset,
                )
        else:
            self._async_load_kv_linear(
                ctx.k_dma_m0, buf_id, src_div, src_base, soffset, self.num_dma_k
            )

    def load_k_tile(self, tile_idx, buf_id, page_id=None):
        self.load_k(self.tile_start(tile_idx), buf_id, page_id=page_id)

    def load_k_split(self, offset_tiles, buf_id, page_id=None):
        self.load_k_tile(self.split_tile(offset_tiles), buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            self.traits,
            tile_start,
            kv_gmem_elem_offset=self.kv_gmem_elem_offset,
            kv_head_elem_offset=self.kv_head_elem_offset,
            stride_kv_n_v=self.stride_kv_n_v,
        )
        src_div = self._kv_src_div(
            self.V, self.v_div, page_id, "DualwaveKvGmemToLdsLoader.load_v"
        )
        if const_expr(self.traits.KV_VECTORIZED):
            for d in range_constexpr(self.num_dma_v):
                self._issue_kv_dma(
                    src_div,
                    ctx.v_dma_m0[buf_id][d],
                    _vec_v_src_elem(
                        self.traits,
                        d,
                        wave_id_uni=self.wave_id_uni,
                        lane_in_warp=self.lane_in_warp,
                        kv_head_idx=self.kv_head_idx,
                    ),
                    soffset,
                )
        else:
            self._async_load_kv_linear(
                ctx.v_dma_m0, buf_id, src_div, src_base, soffset, self.num_dma_v
            )

    def load_v_tile(self, tile_idx, buf_id, page_id=None):
        self.load_v(self.tile_start(tile_idx), buf_id, page_id=page_id)

    def load_v_split(self, offset_tiles, buf_id, page_id=None):
        self.load_v_tile(self.split_tile(offset_tiles), buf_id, page_id=page_id)


class DualwaveKvLdsToVgprLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _load_k_pair(self, buf_id, idx_lo):
        lo = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo,
            buf_id,
            self.kv_mfma_pack_type,
        )
        hi = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo + self.traits.K_LDS_TO_REG_N_STRIP_STRIDE,
            buf_id,
            self.kv_mfma_pack_type,
        )
        return lo, hi

    def load_k(self, buf_id, urk_base=None):
        if urk_base is None:
            urk_base = self.k_lds_read_base_per_lane
        k_base = _k_buf_base(self.traits, buf_id)
        k_lo = [None] * self.traits.K_STEPS_QK
        k_hi = [None] * self.traits.K_STEPS_QK

        if const_expr(self.traits.KV_VECTORIZED):
            for ks in range_constexpr(self.traits.K_STEPS_QK):
                k_lo[ks], k_hi[ks] = self._load_k_pair(
                    buf_id,
                    _vec_k_lds_idx_lo(
                        self.traits,
                        k_base,
                        ks,
                        lane_div_32=self.lane_div_32,
                        lane_mod_32=self.lane_mod_32,
                    ),
                )
            return (k_lo, k_hi)

        for ks in range_constexpr(self.traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(
                buf_id, k_base + urk_base + _swizzled_ks_offset(self.traits, ks)
            )
        return (k_lo, k_hi)

    def load_v(self, buf_id, urv_base=None):
        if urv_base is None:
            urv_base = self.v_lds_read_base_per_lane
        v_base = _v_buf_base(self.traits, buf_id)
        packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        if const_expr(self.traits.KV_VECTORIZED):
            v_base_ptr = buffer_ops.get_element_ptr(
                self.lds_kv_base_ptr,
                byte_offset=as_mlir_value(
                    fx.Int32(
                        _vec_v_lds_addr_base(
                            self.traits,
                            v_base,
                            lane_div_32=self.lane_div_32,
                            lane_mod_32=self.lane_mod_32,
                        )
                        * self.traits.BF16_BYTES
                    )
                ),
                elem_type=T.i8,
            )
            for dc in range_constexpr(self.traits.D_CHUNKS):
                for k_substep in range_constexpr(4):
                    packs[k_substep][dc] = _read_v8f16_off(
                        self.traits,
                        v_base_ptr,
                        _vec_v_const_off(self.traits, dc, k_substep),
                        self.kv_mfma_pack_type,
                    )
            return packs

        lds_base = v_base + urv_base
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _swizzled_v_imm_lo(self.traits, dc, k_substep)
                a = _ds_read_tr_v4f16_imm(
                    lds_base,
                    imm_lo,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                )
                b = _ds_read_tr_v4f16_imm(
                    lds_base,
                    imm_lo
                    + self.traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE
                    * self.traits.BF16_BYTES,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                )
                packs[k_substep][dc] = (
                    Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                )
        return packs


class DualwaveStoreHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _final_o_base(self, q_row):
        return (
            q_row * self.o_stride_n_v + self.o_head_elem_offset + self.lane_div_32 * 8
        )

    def _final_o_global(self, o_base, dc, g):
        return o_base + (dc * self.traits.D_CHUNK + 2 * g * 8)

    def _store_final_o_128(self, v_o, dc, g, o_base):
        _buffer_store_128(
            _packed_o_128_vec(
                self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype
            ),
            self._final_o_global(o_base, dc, g),
            _o_store_reg_128=self.o_store_reg_128,
            _store_atom_128=self.store_atom_128,
            o_div=self.o_div,
        )

    def _store_final_o_row(self, v_o, q_row):
        o_base = self._final_o_base(q_row)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                self._store_final_o_128(v_o, dc, g, o_base)

    def _store_splitk_partial_o_quad(
        self, v_o, dc, g, local_opart_row_base, opart_rsrc
    ):
        w0, w1, w2, w3 = _packed_o_128_dwords(
            self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype
        )
        _ws_store_quad_i32(
            [w0, w1, w2, w3],
            local_opart_row_base
            + _splitk_o_partial_dword_col(self.traits, dc, g, self.lane_div_32),
            opart_rsrc,
        )

    def _store_splitk_partial_o_row(self, v_o, local_opart_row_base, opart_rsrc):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                self._store_splitk_partial_o_quad(
                    v_o, dc, g, local_opart_row_base, opart_rsrc
                )

    def zero_o_block_if_needed(self, causal_end_raw_i32=None, sink_log2=None):
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        traits = self.traits
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        seq_len_v = self.seq_len_v

        lse_m_z = self.c_zero_f if sink_log2 is None else sink_log2
        lse_l_z = self.c_zero_f if sink_log2 is None else fx.Float32(1.0)

        @flyc.jit
        def _zero_o_block_if_needed():
            if causal_end_raw_i32 <= fx.Int32(0):
                q_row_z = q_start + wave_q_offset + lane_mod_32
                c_zero_i = fx.Int32(0)
                zero_pack = Vec.from_elements(
                    [c_zero_i, c_zero_i, c_zero_i, c_zero_i], fx.Int32
                )
                if q_row_z < seq_len_v:
                    o_base_z = self._final_o_base(q_row_z)
                    for dc in range_constexpr(traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            _buffer_store_128(
                                zero_pack,
                                self._final_o_global(o_base_z, dc, g),
                                _o_store_reg_128=self.o_store_reg_128,
                                _store_atom_128=self.store_atom_128,
                                o_div=self.o_div,
                            )
                    if const_expr(traits.RETURN_LSE):
                        self._store_lse_row(lse_m_z, lse_l_z, q_row_z)

        _zero_o_block_if_needed()

    def _splitk_workspace_resources(self):
        split_z = _splitk_workspace_split_z(self.traits, self.batch_idx, self.split_idx)
        return _splitk_workspace_resources(
            self.ws_base_i64,
            split_z,
            self.ws_opart_per_split_bytes,
            self.ws_ml_per_split_bytes,
            self.ws_mrow_abs_bytes,
            self.ws_lrow_abs_bytes,
        )

    def store_empty_split(self):
        traits = self.traits
        batch_idx = self.batch_idx
        split_idx = self.split_idx
        seq_len_v = self.seq_len_v
        q_start = self.q_start
        q_head_idx = self.q_head_idx
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        lane_div_32 = self.lane_div_32
        lane = self.lane
        max_num_tiles = self.max_num_tiles
        split_t0 = self.split_t0
        c_zero_f = self.c_zero_f
        ws_base_i64 = self.ws_base_i64
        ws_opart_per_split_bytes = self.ws_opart_per_split_bytes
        ws_ml_per_split_bytes = self.ws_ml_per_split_bytes
        ws_mrow_abs_bytes = self.ws_mrow_abs_bytes
        ws_lrow_abs_bytes = self.ws_lrow_abs_bytes

        @flyc.jit
        def _store_empty_split():
            if max_num_tiles < split_t0 + fx.Index(4):
                q_row_e = q_start + wave_q_offset + lane_mod_32
                split_z_e = _splitk_workspace_split_z(traits, batch_idx, split_idx)
                _opart_rsrc_e, _mrow_rsrc_e, _lrow_rsrc_e = _splitk_workspace_resources(
                    ws_base_i64,
                    split_z_e,
                    ws_opart_per_split_bytes,
                    ws_ml_per_split_bytes,
                    ws_mrow_abs_bytes,
                    ws_lrow_abs_bytes,
                )
                local_opart_base_e = _splitk_local_opart_row_base(
                    traits, q_head_idx, seq_len_v, q_row_e
                )
                local_ml_e = _splitk_local_ml_idx(q_head_idx, seq_len_v, q_row_e)
                if q_row_e < seq_len_v:
                    _store_empty_splitk_o_partial_row(
                        traits, local_opart_base_e, lane_div_32, _opart_rsrc_e
                    )
                    if lane < fx.Index(32):
                        _store_splitk_ml_row(
                            fx.Float32(-1e30),
                            c_zero_f,
                            local_ml_e,
                            _mrow_rsrc_e,
                            _lrow_rsrc_e,
                        )

        _store_empty_split()

    def _store_lse_row(self, m_row, l_row, q_row):
        # LSE = m_row * ln2 + ln(l_row); natural log, scale folded (m_row is
        # sm_scale*log2e-scaled). Fully-masked row has l_row == 0 -> -inf.
        traits = self.traits
        lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE)))
        num_heads_q = getattr(self, "num_heads_q", None)
        lse_per_batch_elems = (
            fx.Index(traits.NUM_HEADS_Q if num_heads_q is None else num_heads_q)
            * self.seq_len_v
        )
        lse_per_batch_bytes = lse_per_batch_elems * fx.Index(4)
        lse_records = _lse_records(getattr(self, "lse_on", None), lse_per_batch_bytes)
        lse_rsrc = _make_ws_rsrc(
            lse_base_i64, self.batch_idx * lse_per_batch_bytes, lse_records
        )
        # -inf for a row no key reached, selected rather than left to a fast-math
        # -inf + log(0) that LLVM may drop as poison (modification 23).
        # "No keys" is l == +0 (`_no_keys`), so a NaN row keeps a NaN LSE
        # (modification 29).
        lse_val = _no_keys(l_row).select(
            self.c_neg_inf,
            fx.Float32(
                m_row * self.c_ln2_f
                + fx.log(l_row, fastmath=fx.arith.FastMathFlags.none)
            ),
        )
        lse_local = self.q_head_idx * self.seq_len_v + q_row
        # One writer per row: low half-wave + in-bounds q_row; else the dropped OOB sentinel.
        lse_off_row = (q_row < self.seqlen_q_v).select(lse_local, lse_per_batch_elems)
        lse_off = fx.Index(
            (self.lane < fx.Index(32)).select(lse_off_row, lse_per_batch_elems)
        )
        _ws_store_f32(lse_val, lse_off, lse_rsrc)

    def store_final_o(self, v_o, q_row, m_row=None, l_row=None):
        self._store_final_o_row(v_o, q_row)
        if const_expr(self.traits.RETURN_LSE):
            self._store_lse_row(m_row, l_row, q_row)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        _opart_rsrc, _mrow_rsrc, _lrow_rsrc = self._splitk_workspace_resources()
        local_opart_row_base = _splitk_local_opart_row_base(
            self.traits, self.q_head_idx, self.seq_len_v, q_row
        )
        local_ml_idx = _splitk_local_ml_idx(self.q_head_idx, self.seq_len_v, q_row)
        seq_len_v = self.seq_len_v
        lane = self.lane

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < seq_len_v:
                self._store_splitk_partial_o_row(v_o, local_opart_row_base, _opart_rsrc)
                if lane < fx.Index(32):
                    _store_splitk_ml_row(
                        m_row, l_row, local_ml_idx, _mrow_rsrc, _lrow_rsrc
                    )

        _store_splitk_partial_if_qrow()


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Q/K/V remain byte-typed for DMA. Descales apply to accumulated QK scores
    and the final P*V output; the ``vt`` allocation holds packed FP8 V bytes.
    """

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,  # noqa: E741
        DebugCounts=None,
        CuSeqQ=None,
        CuSeqKv=None,
        QDescale=None,
        KDescale=None,
        VDescale=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        stride_o_n=None,
        BlockTable=None,
        block_table_stride=None,
    ):
        if isinstance(traits_or_ctx, DualwaveFp8KernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return
        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_o_n = stride_q_n if stride_o_n is None else stride_o_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.BlockTable = BlockTable
        self.block_table_stride = block_table_stride

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.p_elem = fx.BFloat16
        self.v4bf16_type = Vec.make_type(4, fx.BFloat16)
        if const_expr(traits.PAGED):
            self.NUM_DMA_K = traits.SMEM_D_RPT
        else:
            self.NUM_DMA_K = len(traits.K_BAND_CHUNK)
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_rescale_thr_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        traits = self.traits
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_o_n_v = fx.Int64(self.stride_o_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)
        if const_expr(traits.PAGED):
            self.stride_v_n_v = fx.Int64(traits.NUM_HEADS_KV * traits.HEAD_DIM_V)
        elif traits.HEAD_DIM_V == traits.HEAD_DIM:
            self.stride_v_n_v = self.stride_kv_n_v
            self.stride_o_n_v = self.stride_q_n_v
        else:
            self.stride_v_n_v = fx.Index(traits.DEFAULT_STRIDE_V_N)
            self.stride_o_n_v = fx.Index(traits.DEFAULT_STRIDE_O_N)

    def init_causal_lpt_order(self):
        """Issue causal q-blocks longest-first by reversing the q-block grid axis.

        Causal work per q-block grows with the block index and workgroups dispatch in
        flattened-id order, so the natural order issues the heaviest block last and the
        makespan carries its tail. Must run after init_thread_mapping and before
        init_sequence_lengths / init_tile_bounds / init_q_row read q_start.
        """
        traits = self.traits
        num_q_blocks = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_block_idx = num_q_blocks - fx.Index(1) - self.q_block_idx
        self.q_start = self.q_block_idx * traits.BLOCK_M

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        self.lds_vt_base_ptr = lds.vt.ptr.llvm_ptr
        if const_expr(self.traits.PAGED):
            # The 16-byte tile stride preserves wide LDS loads; i64-element
            # views can instead lower to slower paired 64-bit reads.
            self.k_lds_i32_tiles = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, lds.kv.ptr),
                    fx.make_layout(self.traits.LDS_KV_TOTAL_SIZE // 4, 1),
                ),
                fx.make_layout(4, 1),
            )
            self.v_lds_i32_tiles = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, lds.vt.ptr),
                    fx.make_layout(self.traits.VT_BF16_TOTAL // 2, 1),
                ),
                fx.make_layout(4, 1),
            )
        if hasattr(lds, "q"):
            self.lds_q_base_idx = fx.Int64(fx.ptrtoint(lds.q.ptr))
            self.lds_q_base_ptr = lds.q.ptr.llvm_ptr
        else:
            self.lds_q_base_idx = None
            self.lds_q_base_ptr = None

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_dma_thread_offsets(self):
        # Emitted after descriptors/atoms (matching the original schedule) so the
        # d_bucket ``v_and`` lands at the same ISA position.
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.LANE_SPLIT_KV
        self.d_bucket = self.lane_in_warp % traits.LANE_SPLIT_KV

    def init_sequence_lengths(self):
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cuk_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqKv), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)
            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(
                _cuk_div, self.batch_idx + 1, _cu_atom, _cu_v1i32
            )
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = (
            self.kv_tok_base * self.stride_kv_n_v + self.kv_head_idx * traits.HEAD_DIM
        )
        self.v_gmem_elem_offset = (
            self.kv_tok_base * self.stride_v_n_v + self.kv_head_idx * traits.HEAD_DIM_V
        )

    def init_varlen_causal_lpt_order(self):
        """Reverse active query blocks without making padded workgroups valid."""
        num_q_blocks = (
            self.seqlen_q_v + self.traits.BLOCK_M - 1
        ) // self.traits.BLOCK_M
        active_q_block = self.q_block_idx < num_q_blocks
        reversed_q_block = num_q_blocks - 1 - self.q_block_idx
        self.q_block_idx = active_q_block.select(reversed_q_block, self.q_block_idx)
        self.q_start = self.q_block_idx * self.traits.BLOCK_M
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        o_nrec_bytes = as_mlir_value(
            self.q_tok_end * self.stride_o_n_v * traits.OUT_ELEM_BYTES
        )

        def _make_buf_div(tensor, nrec_bytes):
            # fp8 Q/K/V buffer views are i8-typed so DMA and register loads share one
            # byte view.
            bt = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=nrec_bytes)
            it = fx.get_iter(bt)
            i8_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.PointerType(it.type).address_space,
                alignment=fx.PointerType(it.type).alignment,
            )
            bt = fx.Tensor(
                fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt))
            )
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes),
            fx.make_layout(1, 1),
        )
        if const_expr(traits.PAGED):
            # K/V are one-byte OCP FP8 pages.  Rebase one descriptor per tile
            # with 64-bit pointer arithmetic so serving-sized pools beyond the
            # 4-GiB boundary retain the exact physical page address.
            self.k_base_iter = fx.recast_iter(fx.Int8, fx.get_iter(self.K))
            self.v_base_iter = fx.recast_iter(fx.Int8, fx.get_iter(self.V))
            self.k_page_bytes = traits.PAGE_SIZE * traits.NUM_HEADS_KV * traits.HEAD_DIM
            self.v_page_bytes = (
                traits.PAGE_SIZE * traits.NUM_HEADS_KV * traits.HEAD_DIM_V
            )
            self.init_page_table()
            self.k_div = None
            self.v_div = None
            if const_expr(traits.CACHE_BUFFERED):

                def _cache_div(tensor, base_iter):
                    num_bytes = fx.Int64(
                        fx.get_scalar(fx.cosize(fx.get_layout(tensor)))
                    )
                    flat = fx.make_view(base_iter, fx.make_layout(num_bytes, 1))
                    # make_buffer_tensor defaults to an unbounded resource;
                    # supply the exact byte count for zero-filling OOB copies.
                    bounded = fx.rocdl.make_buffer_tensor(
                        flat, num_records_bytes=num_bytes.ir_value()
                    )
                    return fx.logical_divide(bounded, fx.make_layout(1, 1))

                self.k_div = _cache_div(self.K, self.k_base_iter)
                self.v_div = _cache_div(self.V, self.v_base_iter)
        else:
            kv_nrec_bytes = (self.kv_tok_end * self.stride_kv_n_v * eb).ir_value()
            v_nrec_bytes = (self.kv_tok_end * self.stride_v_n_v * eb).ir_value()
            self.k_div = _make_buf_div(self.K, kv_nrec_bytes)
            self.v_div = _make_buf_div(self.V, v_nrec_bytes)

    def init_atoms_and_lds_ptrs(self):
        traits = self.traits
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.o_store_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # fp8 global->LDS DMA uses i8 destination typing; K/V LDS reads are byte-addressed.
        self.lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, traits.DMA_BYTES)

    def _load_scale_scalar(self, tensor):
        _div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1)
        )
        _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        _v = fly.copy_atom_call_ssa(
            [Vec.make_type(1, fx.Float32)], _atom, fx.slice(_div, (None, fx.Int32(0)))
        )
        return Vec(_v, (1,), fx.Float32)[0]

    def init_descale(self):
        head_dim_f32 = fx.Float32(fx.Int32(self.head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.rsqrt(head_dim_f32, fastmath=self.fm_fast) * c_log2e_f
        _qd = self._load_scale_scalar(self.QDescale)
        _kd = self._load_scale_scalar(self.KDescale)
        if const_expr(not self.traits.PAGED):
            self.vd_fp8 = self._load_scale_scalar(self.VDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = c_sm_scale_log2e * (_qd * _kd)

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        self.num_kv_tiles = num_kv_tiles
        if const_expr(traits.CAUSAL):
            causal_end_raw_i32 = (
                fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            )
            causal_end_i32 = fx.Int32(
                (causal_end_raw_i32 > fx.Int32(0)).select(
                    causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_num_tiles = (
                fx.Index(causal_end_i32) + kv_tile_size - 1
            ) // kv_tile_size
            max_num_tiles = fx.Index(
                (causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles)
            )
        else:
            causal_end_raw_i32 = None
            max_num_tiles = num_kv_tiles
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        max_num_tiles = fx.Index(
            (max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles)
        )
        self.max_num_tiles = max_num_tiles
        if const_expr(traits.SPLITK):
            chunk = (
                (
                    (max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS
                    + 1
                )
                // 2
                * 2
            )
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index(
                (split_t_end < max_num_tiles).select(split_t_end, max_num_tiles)
            )
            split_t_end = fx.Index(
                (max_num_tiles - split_t_end < fx.Index(4)).select(
                    max_num_tiles, split_t_end
                )
            )
            self.split_nonempty = split_t0 + fx.Index(4) <= max_num_tiles
        else:
            split_t0 = 0
            split_t_end = max_num_tiles
            self.split_nonempty = None

        if const_expr(traits.VARLEN or (traits.CAUSAL and traits.CROSS_SEQLEN)):
            active = None
            if const_expr(traits.VARLEN):
                active = self.q_start < self.seqlen_q_v
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                in_mask = causal_end_raw_i32 > fx.Int32(0)
                active = in_mask if active is None else (active & in_mask)
            split_t_end = fx.Index(active.select(split_t_end, split_t0))

        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def init_workspace_io(self):
        if const_expr(self.traits.SPLITK):
            self.ws_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.DebugCounts), fx.make_layout(1, 1)
            )
            self.ws_store_atom_32 = fx.make_copy_atom(
                fx.rocdl.BufferCopy32b(), fx.Int32
            )
            self.ws_store_reg_32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            self.ws_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def ws_store_f32(self, f32_val, elem_index):
        pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_32)
        fx.copy(
            self.ws_store_atom_32,
            self.ws_store_reg_32,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def ws_store_quad_i32(self, dwords, elem_index):
        pack = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_128)
        fx.copy(
            self.store_atom_128,
            self.ws_store_reg_128,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
        return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
        return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_pair_to_vec32(self, v):
        return _v_pair_to_vec32(v)

    def v_vec32_to_pair(self, v):
        return _v_vec32_to_pair(v)

    def bf16_trunc_pack_v8(self, f32_vals):
        # The dtype-dispatched pack would incorrectly select FP16 for FP8 input.
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(
            elem_index, self.load_atom_128, self.q_div, self.v4i32_type
        )

    def buffer_load_fp8x16(self, src_div, elem_index):
        return fly.copy_atom_call_ssa(
            [self.v4i32_type],
            self.load_atom_128,
            fx.slice(src_div, (None, fx.Int32(elem_index))),
        )

    def global_load_fp8x16(self, base_iter, byte_offset, valid, *, is_value=False):
        """Load a native FP8 vector through a bounded cache or a 64-bit pointer."""
        if const_expr(self.traits.CACHE_BUFFERED):
            offset = valid.select(
                fx.Int32(byte_offset), fx.Int32(PAGED_FP8_BUFFER_LIMIT_BYTES)
            )
            return self.buffer_load_fp8x16(
                self.v_div if is_value else self.k_div, offset
            )
        # Do not issue an out-of-allocation load before selecting zero padding.
        safe_offset = valid.select(fx.Int64(byte_offset), fx.Int64(0))
        loaded = fx.generic_load(
            fx.add_offset(base_iter, safe_offset), dtype=fx.Int32, count=4
        )
        return valid.select(loaded, fx.Vector.filled(4, 0, fx.Int32)).ir_value()

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div,
            lds_byte_addr,
            src_elem,
            soffset_elems,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(
            pack_i32_vec,
            elem_index,
            self.o_store_reg_128,
            self.store_atom_128,
            self.o_div,
        )

    def global_idx_q(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_q_n_v
            + self.q_head_idx * self.traits.HEAD_DIM
            + col
        )

    def global_idx_o(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + col
        )

    def init_page_table(self):
        self.page_indices_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.BlockTable), fx.make_layout(1, 1)
        )
        self.page_i32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.page_v1i32 = Vec.make_type(1, fx.Int32)
        self.page_i32x2_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.page_v2i32 = Vec.make_type(2, fx.Int32)
        self.page_base = fx.Int64(self.batch_idx) * fx.Int64(self.block_table_stride)
        if const_expr(self.traits.PAIRED_PAGE_IDS):
            request = fx.make_view(
                fx.add_offset(fx.get_iter(self.BlockTable), self.page_base),
                fx.make_layout(fx.Int64(self.block_table_stride), 1),
            )
            num_pages = (
                self.seqlen_kv_v + self.traits.PAGE_SIZE - 1
            ) // self.traits.PAGE_SIZE
            bounded = fx.rocdl.make_buffer_tensor(
                request, num_records_bytes=(num_pages * 4).ir_value()
            )
            self.page_pair_indices_div = fx.logical_divide(
                bounded, fx.make_layout(1, 1)
            )

    def load_page_id(self, tile_start, *, uniform=True):
        """Load a physical page ID, zero-filling lookups past this request's pages."""
        local_page = fx.Int64(tile_start) // self.traits.PAGE_SIZE
        page_idx = self.page_base + local_page
        if const_expr(self.traits.PAGE_SIZE == self.traits.BLOCK_N):
            num_pages = self.num_kv_tiles
        else:
            num_pages = (
                self.seqlen_kv_v + self.traits.PAGE_SIZE - 1
            ) // self.traits.PAGE_SIZE
        valid = local_page < num_pages
        safe_idx = valid.select(page_idx, 0)
        v = fly.copy_atom_call_ssa(
            [self.page_v1i32],
            self.page_i32_atom,
            fx.slice(self.page_indices_div, (None, fx.Int32(safe_idx))),
        )
        page = valid.select(Vec(v, (1,), fx.Int32)[0], fx.Int32(0))
        if uniform:
            return fx.Int64(rocdl.readfirstlane(T.i32, page.ir_value()))
        return fx.Int64(page)

    def load_page_id_pair(self, tile_start):
        """Return reusable page IDs for a compute-tile pair, or defer to producers."""
        if const_expr(self.traits.PAGE_SIZE < self.traits.BLOCK_N):
            layout = self.traits.SHARED_PAGE_ID_LAYOUT
            if const_expr(layout is None):
                return None, None
            if const_expr(layout == "tokens"):
                token = fx.Int64(tile_start) + self.lane_in_warp
            elif const_expr(layout == "page_waves"):
                token = fx.Int64(tile_start) + (
                    self.wave_id_uni % fx.Int64(4)
                ) * fx.Int64(16)
            else:
                token = fx.Int64(tile_start) + (
                    self.lane_in_warp // fx.Int64(16)
                ) * fx.Int64(16)
            uniform = layout == "page_waves"
            return self.load_page_id(token, uniform=uniform), self.load_page_id(
                token + 64, uniform=uniform
            )
        if const_expr(not self.traits.PAIRED_PAGE_IDS):
            return self.load_page_id(tile_start), self.load_page_id(
                tile_start + self.traits.BLOCK_N
            )
        local_page = fx.Int32(fx.Int64(tile_start) // self.traits.PAGE_SIZE)
        # gfx950 zero-fills each out-of-range dword, including an odd pair tail.
        v = fly.copy_atom_call_ssa(
            [self.page_v2i32],
            self.page_i32x2_atom,
            fx.slice(self.page_pair_indices_div, (None, local_page)),
        )
        pages = Vec(v, (2,), fx.Int32)
        return (
            fx.Int64(rocdl.readfirstlane(T.i32, pages[0].ir_value())),
            fx.Int64(rocdl.readfirstlane(T.i32, pages[1].ir_value())),
        )

    def make_page_view(self, tensor_iter, page_id, *, is_value=False):
        page_bytes = self.v_page_bytes if is_value else self.k_page_bytes
        # Rebase in 64 bits before constructing the bounded per-page resource.
        page_ptr = fx.add_offset(tensor_iter, page_id * page_bytes)
        page = fx.make_view(page_ptr, fx.make_layout(page_bytes, 1))
        return fx.logical_divide(
            fx.rocdl.make_buffer_tensor(page, num_records_bytes=page_bytes),
            fx.make_layout(1, 1),
        )

    def read_i32x8_lds(self, base_ptr, byte_row):
        halves = []
        for h in range_constexpr(2):
            p = buffer_ops.get_element_ptr(
                base_ptr, byte_offset=fx.Int32(byte_row + h * 16), elem_type=T.i8
            )
            halves.append(
                Vec(llvm.LoadOp(Vec.make_type(4, fx.Int32), p, alignment=16).result)
            )
        return halves[0].shuffle(halves[1], [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

    def read_i32x4_lds(self, byte_row):
        tile = fx.slice(self.k_lds_i32_tiles, (None, fx.Uint32(byte_row) // 16))
        return fx.generic_load(fx.get_iter(tile), dtype=fx.Int32, count=4).ir_value()

    def preserve_accumulators(self, v_o):
        if const_expr(self.traits.PAGED):
            # Preserve FP expression boundaries without inline-assembly pins.
            return [llvm.intr_arithmetic_fence(fx.as_ir_value(acc)) for acc in v_o]
        return _anchor_v_o(self.traits, v_o)


class DualwaveFp8QLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def stage_q_to_lds(self):
        traits = self.traits
        chunks_per_row = traits.HEAD_DIM // 16  # 16-byte DMA chunks per Q row
        total_chunks = (traits.BLOCK_M * traits.HEAD_DIM) // 16
        for p in range_constexpr(total_chunks // traits.BLOCK_SIZE):
            c = self.tid + fx.Index(p * traits.BLOCK_SIZE)
            row = c // fx.Index(chunks_per_row)
            dchunk = c % fx.Index(chunks_per_row)
            src_elem = (
                self.q_gmem_elem_offset
                + row * self.stride_q_n_v
                + dchunk * fx.Index(16)
            )
            lds_addr = self.lds_q_base_idx + c * fx.Index(16)
            self.buffer_load_lds_128(self.q_div, lds_addr, src_elem, 0)

    def load_all_wide(self, q_row_in_block):
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = (
                q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            )
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)
        if const_expr(self.traits.PAGED):
            self.fp8_mma = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN)
            )

    def _pack_p_fp8(self, f32):
        # P words must follow vectorized K's four-token groups for paged P*V.
        packed = self._pack_fp8_i32x8(f32)
        if const_expr(self.traits.PAGED):
            words = Vec(packed, (8,), fx.Int32)
            return Vec.from_elements(
                [words[i] for i in (0, 2, 1, 3, 4, 6, 5, 7)],
                fx.Int32,
            ).ir_value()
        return packed

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        if const_expr(self.traits.PAGED):
            a = fx.make_rmem_tensor(8, fx.Int32)
            b = fx.make_rmem_tensor(8, fx.Int32)
            c = fx.make_rmem_tensor(16, fx.Float32)
            a.store(Vec(a_i32x8))
            b.store(Vec(b_i32x8))
            c.store(Vec(c_v16))
            fx.gemm(
                self.fp8_mma,
                c,
                a,
                b,
                c,
                scale_a=fx.Int32(0x7F7F7F7F),
                scale_b=fx.Int32(0x7F7F7F7F),
            )
            return c.load().ir_value()
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            [
                as_mlir_value(a_i32x8),
                as_mlir_value(b_i32x8),
                as_mlir_value(c_v16),
                0,
                0,
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
            ],
        )

    def _pack_fp8_i32x8(self, f32_vals):
        c0 = llvm.mlir_poison(T.i32)
        words = []
        for g in range_constexpr(8):
            base = g * 4
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base]),
                as_mlir_value(f32_vals[base + 1]),
                c0,
                0,
            )
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base + 2]),
                as_mlir_value(f32_vals[base + 3]),
                w,
                1,
            )
            words.append(fx.Int32(w))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _v_concat_i32x8(self, v_v, dc):
        words = []
        for ks in range_constexpr(4):
            v2 = Vec.from_elements([fx.Int64(v_v[ks][dc])], fx.Int64).bitcast(fx.Int32)
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = (
                q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            )
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def _load_q_wide_global(self):
        """Pull this lane's Q operands straight from global into VGPRs."""
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            elem = self.global_idx_q(self.ctx_ref.q_row, fx.Index(ws * 64) + d_base)
            lo = self.buffer_load_128(elem)
            hi = self.buffer_load_128(elem + fx.Index(16))
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs

    def load_q_wide(self):
        if const_expr(self.traits.QLDS):
            return self._load_q_wide_lds()
        return self._load_q_wide_global()

    def qk(self, v_k, q_wide=None):
        traits = self.traits
        k_lo, k_hi = v_k
        q_all_wide = self._load_q_wide_lds() if q_wide is None else q_wide
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_w = q_all_wide[ws]
            v_s_lo = self._mfma_acc_fp8_wide(k_lo[ws], q_w, v_s_lo)
            v_s_hi = self._mfma_acc_fp8_wide(k_hi[ws], q_w, v_s_hi)
        n_ds = const_expr(traits.HEAD_DIM // 64 * 4)
        n_mfma = const_expr(traits.HEAD_DIM // 64 * 2)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, 12)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, n_mfma - 1, 12)
        return (v_s_lo, v_s_hi)

    def cast_p_fp8_direct(self, v_p):
        lo_partial_list, hi_full = v_p
        f32 = []
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [hi_full[p_base + s] for s in range_constexpr(8)]
        return self._pack_p_fp8(f32)

    def _pv_fp8_direct(self, p_fp8, v_v, v_o):
        v_o = self.preserve_accumulators(v_o)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        return self._pv_fp8_direct(v_p, v_v, v_o)


def _page1_k_page_ids(page_ids):
    """Convert token-major page IDs to K's sigma lanes without another lookup."""
    # Swap lane bits 2/3: move lanes 4..7 forward and 8..11 backward by four.
    value = fx.Int32(page_ids).ir_value()
    swapped = rocdl.update_dpp(T.i32, value, value, 0x104, 15, 2, False)
    result = rocdl.update_dpp(T.i32, swapped, value, 0x114, 15, 4, False)
    return fx.Int64(fx.Int32(result))


def _transpose_v_fp8_16x16(source, lane, stages=4):
    """Transpose byte quads (two stages) or 16-lane tiles (four), with full EXEC."""
    assert stages in (2, 4)
    words = [fx.Uint32(Vec(source)[i]) for i in range_constexpr(4)]
    lane = fx.Int32(lane)
    for bit in range_constexpr(stages):
        lane_bit = (lane & (1 << bit)) != 0
        if const_expr(bit < 2):
            # Select alternating bytes, then halfwords, from self/peer words.
            low_selector = 0x06020400 if bit == 0 else 0x05040100
            high_selector = 0x03070105 if bit == 0 else 0x03020706
            selector = lane_bit.select(fx.Int32(high_selector), fx.Int32(low_selector))
        updated = []
        for word in range_constexpr(4):
            peer_word = word if bit < 2 else word ^ (1 << (bit - 2))
            peer_value = words[peer_word].ir_value()
            if const_expr(bit < 2):
                dpp_ctrl = 0xB1 if bit == 0 else 0x4E
                peer = rocdl.update_dpp(
                    T.i32, peer_value, peer_value, dpp_ctrl, 15, 15, False
                )
                updated.append(fx.Uint32(rocdl.perm_b32(peer, words[word], selector)))
            else:
                peer = fx.Uint32(
                    rocdl.ds_swizzle(
                        T.i32, peer_value, fx.Int32(31 | (1 << (bit + 10))).ir_value()
                    )
                )
                if const_expr(word & (1 << (bit - 2))):
                    updated.append(lane_bit.select(words[word], peer))
                else:
                    updated.append(lane_bit.select(peer, words[word]))
        words = updated
    return Vec.from_elements([fx.Int32(word) for word in words], fx.Int32)


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id, page_id=None):
        """DMA K using the vectorized page layout or dense head-dimension bands.

        Dense bands are row-contiguous within each wave. Their D192 tail
        uses the low 32 lanes; paged tiles retain the physical D-group mapping.
        """
        traits = self.traits
        if const_expr(traits.PAGED and traits.K_LDS_PAGE_GROUPED):
            self._load_k_page16(tile_start, buf_id, page_id=page_id)
            return
        eb = traits.ELEM_BYTES
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        if const_expr(traits.KV_VECTORIZED):
            src_div = None
            if const_expr(traits.PAGE_SIZE == 1):
                if page_id is not None:
                    page_id = _page1_k_page_ids(page_id)
            if const_expr(traits.PAGE_SIZE >= traits.BLOCK_N):
                if page_id is None:
                    page_id = self.load_page_id(tile_start)
                src_div = self.make_page_view(self.k_base_iter, page_id)

            def _load_vectorized():
                for d in range_constexpr(self.NUM_DMA_K):
                    if const_expr(traits.PAGE_SIZE < traits.BLOCK_N):
                        oct_idx = _vec_k_dma_oct_idx(
                            traits, d, self.wave_id_uni, self.lane_in_warp
                        )
                        token = fx.Int64(tile_start) + _sigma_k_tile_n(
                            oct_idx % traits.BLOCK_N
                        )
                        dim_group = oct_idx // traits.BLOCK_N
                        physical_page = (
                            self.load_page_id(token, uniform=False)
                            if page_id is None
                            else page_id
                        )
                        src_elem = (
                            physical_page * self.k_page_bytes
                            + self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                            + (dim_group * traits.PAGE_SIZE + token % traits.PAGE_SIZE)
                            * traits.KV_VEC_SIZE
                        )
                        valid = (token < self.seqlen_kv_v) & (
                            dim_group * traits.KV_VEC_SIZE < traits.HEAD_DIM
                        )
                        if const_expr(traits.CACHE_BUFFERED):
                            offset = valid.select(
                                fx.Int32(src_elem),
                                fx.Int32(PAGED_FP8_BUFFER_LIMIT_BYTES),
                            )
                            dst_byte = (
                                k_lds_byte_base
                                + (self.wave_id_uni * traits.SMEM_D_RPT + d)
                                * traits.WARP_SIZE
                                * traits.KV_VEC_SIZE
                            )
                            dst_byte = rocdl.readfirstlane(
                                T.i32, fx.Int32(dst_byte).ir_value()
                            )
                            self.buffer_load_lds_128(self.k_div, dst_byte, offset, 0)
                        else:
                            source = self.global_load_fp8x16(
                                self.k_base_iter, src_elem, valid
                            )
                            dst_byte = (
                                self.k_buf_base(buf_id) + oct_idx * traits.KV_VEC_SIZE
                            )
                            dst = fx.slice(
                                self.k_lds_i32_tiles, (None, fx.Uint32(dst_byte) // 16)
                            )
                            fx.generic_store(fx.get_iter(dst), fx.Vector(source))
                    else:
                        oct_idx = _vec_k_dma_oct_idx(
                            traits, d, self.wave_id_uni, self.lane_in_warp
                        )
                        token = _sigma_k_tile_n(oct_idx % traits.BLOCK_N)
                        if const_expr(traits.PAGE_SIZE != traits.BLOCK_N):
                            token = fx.Int64(tile_start) % traits.PAGE_SIZE + token
                        dim_group = oct_idx // traits.BLOCK_N
                        src_elem = (
                            self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                            + (dim_group * traits.PAGE_SIZE + token)
                            * traits.KV_VEC_SIZE
                        )
                    if const_expr(traits.PAGE_SIZE >= traits.BLOCK_N):
                        lds_addr = (
                            k_lds_byte_base
                            + (self.wave_id_uni * traits.SMEM_D_RPT + d)
                            * traits.WARP_SIZE
                            * traits.KV_VEC_SIZE
                        )
                        lds_addr = rocdl.readfirstlane(
                            T.i32, fx.Int32(lds_addr).ir_value()
                        )
                        self.buffer_load_lds_128(src_div, lds_addr, src_elem, 0)

            if const_expr(traits.HEAD_DIM_V == 192):

                @flyc.jit
                def _load_compact():
                    copy_waves = (traits.BLOCK_N * traits.HEAD_DIM) // (
                        self.NUM_DMA_K * traits.WARP_SIZE * traits.KV_VEC_SIZE
                    )
                    if self.wave_id_uni < fx.Int64(copy_waves):
                        _load_vectorized()

                _load_compact()
            else:
                _load_vectorized()
            return

        rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
        for d in range_constexpr(self.NUM_DMA_K):
            lanes_per_row = traits.K_BAND_CHUNK[d] // traits.VEC_KV
            slots = rows_per_wave * lanes_per_row
            band_base = (
                k_lds_byte_base
                + fx.Index(traits.K_BAND_BASE[d] * eb)
                + self.wave_id_uni * (traits.K_BAND_LINE_STRIDE[d] * eb)
            )
            for pas in range_constexpr(-(-slots // traits.WARP_SIZE)):
                slot = self.lane_in_warp + fx.Index(pas * traits.WARP_SIZE)
                n_in_tile = (slot // lanes_per_row) * traits.NUM_WAVES + self.wave_id
                global_d = (
                    slot % lanes_per_row
                ) * traits.VEC_KV + traits.K_BAND_GLOBAL_D[d]
                src_elem = (
                    self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                )
                lds_addr = band_base + fx.Index(
                    pas * traits.WARP_SIZE * traits.VEC_KV * eb
                )
                active = min(slots - pas * traits.WARP_SIZE, traits.WARP_SIZE)
                if const_expr(active == traits.WARP_SIZE):
                    self.buffer_load_lds_128(
                        self.k_div, lds_addr, src_elem, tile_start * self.stride_kv_n_v
                    )
                else:
                    self._load_k_band_partial_wave(
                        lds_addr, src_elem, tile_start, active
                    )

    def _load_k_page16(self, tile_start, buf_id, page_id=None):
        """Each wave copies four D-groups from one physical page directly to LDS."""
        traits = self.traits
        page_in_tile = self.wave_id_uni % fx.Int64(4)
        if page_id is None:
            page_id = self.load_page_id(
                fx.Int64(tile_start) + page_in_tile * fx.Int64(16)
            )
        if const_expr(traits.CACHE_BUFFERED):
            # The byte bound makes the uniform displacement fit i32; the
            # shared resource retains the full 64-bit cache base pointer.
            src_div = self.k_div
            page_offset = fx.Int32(page_id * self.k_page_bytes)
        else:
            src_div = self.make_page_view(self.k_base_iter, page_id)
            page_offset = fx.Int32(0)
        num_dma = traits.HEAD_DIM // traits.KV_VEC_SIZE
        for part in range_constexpr(
            (num_dma + traits.NUM_WAVES - 1) // traits.NUM_WAVES
        ):
            dma_id = self.wave_id_uni + part * traits.NUM_WAVES

            def _copy_page_groups():
                d_group = (dma_id // fx.Int64(4)) * fx.Int64(
                    4
                ) + self.lane_in_warp // fx.Int64(16)
                token = _sigma_k_tile_n(self.lane_in_warp % fx.Int64(16))
                src_elem = (
                    self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                    + (d_group * traits.PAGE_SIZE + token) * traits.KV_VEC_SIZE
                )
                dst_byte = (
                    self.lds_kv_base_idx
                    + self.k_buf_base(buf_id)
                    + dma_id * fx.Int64(1024)
                )
                dst_byte = rocdl.readfirstlane(T.i32, fx.Int32(dst_byte).ir_value())
                self.buffer_load_lds_128(src_div, dst_byte, src_elem, page_offset)

            if const_expr(
                num_dma % traits.NUM_WAVES == 0 or part < num_dma // traits.NUM_WAVES
            ):
                _copy_page_groups()
            else:

                @flyc.jit
                def _copy_tail_waves():
                    if dma_id < fx.Int64(num_dma):
                        _copy_page_groups()

                _copy_tail_waves()

    def _load_k_band_partial_wave(self, lds_addr, src_elem, tile_start, active_lanes):
        soffset = tile_start * self.stride_kv_n_v
        k_div = self.k_div

        @flyc.jit
        def _run():
            if self.lane_in_warp < fx.Index(active_lanes):
                self.buffer_load_lds_128(k_div, lds_addr, src_elem, soffset)

        _run()

    def load_v(self, tile_start, buf_id, page_id=None, *, mask_padding=True):
        if const_expr(self.traits.PAGED):
            self._stage_v_fp8_vectorized_bankpad(
                tile_start, buf_id, page_id=page_id, mask_padding=mask_padding
            )
        else:
            self._stage_v_fp8_block_dma(tile_start, buf_id)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id):
        traits = self.traits
        nbands = traits.HEAD_DIM_V // 16
        v_tile_bytes = (traits.BLOCK_N // 8) * nbands * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
        # The tile is BLOCK_N * nbands 16-byte slots, and one DMA instruction moves a
        # whole wave of them. Hand out instructions, not row-groups: a wave's LDS
        # destination is then always a full WARP_SIZE*16 span, so nothing has to be
        # masked off inside a wave. buffer_load...lds strides the LDS write by lane
        # regardless of exec, so an intra-wave mask would still write past the span.
        per_dma = traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
        slots_per_group = 8 * nbands
        num_dma = (traits.BLOCK_N * nbands * 16) // per_dma
        passes = -(-num_dma // traits.NUM_WAVES)
        for pas in range_constexpr(passes):
            dma_id = self.wave_id_uni + fx.Index(pas * traits.NUM_WAVES)
            slot = dma_id * fx.Index(traits.WARP_SIZE) + self.lane
            lds_addr = aligned_base + fx.Index(buf_off) + dma_id * fx.Index(per_dma)
            grp = slot // fx.Index(slots_per_group)
            rem = slot % fx.Index(slots_per_group)
            dest_n = fx.Int32(grp * fx.Index(8) + rem % fx.Index(8))
            w16 = dest_n % fx.Int32(16)
            c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
            c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
            n = (
                dest_n
                + c_add.select(fx.Int32(4), fx.Int32(0))
                - c_sub.select(fx.Int32(4), fx.Int32(0))
            )
            d_block = rem // fx.Index(8)
            src_elem = (
                self.v_gmem_elem_offset
                + fx.Index(n) * self.stride_v_n_v
                + d_block * fx.Index(16)
            )
            if const_expr(num_dma % traits.NUM_WAVES == 0 or pas < passes - 1):
                self.buffer_load_lds_128(
                    self.v_div, lds_addr, src_elem, tile_start * self.stride_v_n_v
                )
            else:
                self._load_v_group_if_in_tile(
                    lds_addr, src_elem, tile_start, dma_id, num_dma
                )

    def _load_v_group_if_in_tile(self, lds_addr, src_elem, tile_start, grp, groups):
        soffset = tile_start * self.stride_v_n_v
        v_div = self.v_div

        @flyc.jit
        def _run():
            if grp < fx.Index(groups):
                self.buffer_load_lds_128(v_div, lds_addr, src_elem, soffset)

        _run()

    def _permute_v_fp8_vectorized(self, src_i32x4):
        src_words = Vec(src_i32x4, (4,), fx.Int32)
        # permlane16_swap returns an LLVM pair; unpack it at this intrinsic boundary.
        pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        pair_lo = rocdl.permlane16_swap(
            pair_ty,
            src_words[0].ir_value(),
            src_words[1].ir_value(),
            False,
            False,
        )
        pair_hi = rocdl.permlane16_swap(
            pair_ty,
            src_words[2].ir_value(),
            src_words[3].ir_value(),
            False,
            False,
        )
        return Vec.from_elements(
            [
                fx.Int32(llvm.extractvalue(T.i32, pair_lo, [0])),
                fx.Int32(llvm.extractvalue(T.i32, pair_lo, [1])),
                fx.Int32(llvm.extractvalue(T.i32, pair_hi, [0])),
                fx.Int32(llvm.extractvalue(T.i32, pair_hi, [1])),
            ],
            fx.Int32,
        )

    def _stage_v_fp8_vectorized_bankpad(
        self, tile_start, buf_id, page_id=None, *, mask_padding=True
    ):
        """Stage vectorized paged V in the bank-padded direct-FP8 layout."""
        if const_expr(self.traits.K_LDS_PAGE_GROUPED and self.traits.CACHE_BUFFERED):
            self._stage_v_fp8_page16_segments(
                tile_start, buf_id, page_id=page_id, mask_padding=mask_padding
            )
            return
        source = self.load_v_source(tile_start, page_id=page_id)
        self.store_v_source(source, buf_id, tile_start, mask_padding=mask_padding)

    def _mask_v_fp8_group(self, source, token_start):
        """Zero inactive tokens before PV: a zero probability cannot mask a NaN V."""
        packed = Vec(source)
        full_words = fx.Int32(self.seqlen_kv_v) >> 2
        tail_bits = (fx.Uint32(self.seqlen_kv_v) & 3) * 8
        tail_mask = (fx.Uint32(1) << tail_bits) - 1
        group_word = fx.Int32(fx.Int64(token_start) >> 2)
        words = []
        for word in range_constexpr(packed.numel):
            index = group_word + word % 4
            last_word = (index == full_words).select(tail_mask, fx.Uint32(0))
            mask = (index < full_words).select(fx.Uint32(0xFFFFFFFF), last_word)
            words.append(fx.Int32(fx.Uint32(packed[word]) & mask))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _page16_v_source(self, tile_start, page_id):
        """Coordinates and bounded resource for the wave-grouped V producer."""
        traits = self.traits
        token = fx.Int64(tile_start) + (self.wave_id_uni % fx.Int64(4)) * fx.Int64(16)
        d_col = (self.wave_id // fx.Int64(4)) * fx.Int64(64) + self.lane_in_warp
        if page_id is None:
            page_id = self.load_page_id(token)
        src_div = self.make_page_view(self.v_base_iter, page_id, is_value=True)
        src_elem = (
            self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
            + d_col * traits.KV_VEC_SIZE
        )
        return token, d_col, src_div, src_elem

    def _stage_v_fp8_page16_segments(
        self, tile_start, buf_id, page_id=None, *, mask_padding=True
    ):
        """Consume each V192 segment before loading the next segment's words."""
        traits = self.traits
        token, _, src_div, src_elem = self._page16_v_source(tile_start, page_id)
        src_elem = (token < self.seqlen_kv_v).select(
            src_elem, fx.Int64(self.v_page_bytes)
        )
        prefix = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(mask_padding):
            prefix = self._mask_v_fp8_group(prefix, token)
        self._store_v_fp8_page16(prefix, buf_id, include_tail=False)

        @flyc.jit
        def _stage_tail():
            if self.wave_id_uni < fx.Int64(4):
                tail = self.buffer_load_fp8x16(
                    src_div, src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE)
                )
                if const_expr(mask_padding):
                    tail = self._mask_v_fp8_group(tail, token)
                self._store_v_fp8_page16(
                    tail, buf_id, segment_offset=traits.FP8_V_H1, include_tail=False
                )

        _stage_tail()

    def _store_v_fp8_lds(self, data, byte_offset):
        dst = fx.slice(self.v_lds_i32_tiles, (None, fx.Uint32(byte_offset) // 16))
        fx.generic_store(fx.get_iter(dst), data)

    def load_v_source(self, tile_start, page_id=None):
        """Prefetch a native V packet; store_v_source consumes its selected layout."""
        traits = self.traits
        if const_expr(traits.PAGE_SIZE < traits.BLOCK_N):
            return self._load_v_fp8_small_pages_source(tile_start, page_id=page_id)
        if page_id is None:
            page_id = self.load_page_id(tile_start)
        src_div = self.make_page_view(self.v_base_iter, page_id, is_value=True)
        token_groups16 = traits.BLOCK_N // traits.KV_VEC_SIZE
        # permlane16 requires a full EXEC mask. The first segment has exactly
        # one vector per CTA thread, so no per-lane bounds branch is needed.
        prefix_dim = traits.FP8_V_H1 if traits.FP8_PV_SEGMENTED else traits.HEAD_DIM_V
        assert prefix_dim * token_groups16 == traits.BLOCK_SIZE
        n_group16 = self.lane_in_warp // fx.Int64(16)
        d_col = self.wave_id * fx.Int64(16) + self.lane_in_warp % fx.Int64(16)
        src_elem = (
            self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
            + n_group16 * traits.HEAD_DIM_V * traits.KV_VEC_SIZE
            + d_col * traits.KV_VEC_SIZE
        )
        if const_expr(traits.PAGE_SIZE != traits.BLOCK_N):
            src_elem += (fx.Int64(tile_start) % traits.PAGE_SIZE) * traits.HEAD_DIM_V
        source = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail = self.buffer_load_fp8x16(
                src_div, src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE)
            )
            source = Vec(source).shuffle(Vec(tail), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return source

    def _load_v_fp8_small_pages_source(self, tile_start, page_id=None):
        """Load native small-page V in the selected lane- or wave-grouped layout."""
        traits = self.traits
        if const_expr(traits.PAGE_SIZE == 1):
            return self._load_v_fp8_page1_source(tile_start, page_id=page_id)
        if const_expr(traits.V_LOAD_LAYOUT == "lane_groups"):
            token = fx.Int64(tile_start) + (
                self.lane_in_warp // fx.Int64(16)
            ) * fx.Int64(16)
            d_col = self.wave_id * fx.Int64(16) + self.lane_in_warp % fx.Int64(16)
            if page_id is None:
                page_id = self.load_page_id(token, uniform=False)
            src_elem = (
                page_id * self.v_page_bytes
                + self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
                + d_col * traits.KV_VEC_SIZE
            )
            return self.global_load_fp8x16(
                self.v_base_iter, src_elem, token < self.seqlen_kv_v, is_value=True
            )
        token, d_col, src_div, src_elem = self._page16_v_source(tile_start, page_id)
        valid = token < self.seqlen_kv_v
        src_elem = valid.select(src_elem, fx.Int64(self.v_page_bytes))
        source = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail_elem = (valid & (d_col + traits.FP8_V_H1 < traits.HEAD_DIM_V)).select(
                src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE),
                fx.Int64(self.v_page_bytes),
            )
            tail = self.buffer_load_fp8x16(src_div, tail_elem)
            source = Vec(source).shuffle(Vec(tail), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return source

    def _load_v_fp8_page1_source(self, tile_start, page_id=None):
        """Load token-major vectors for the selected full/scattered V transpose."""
        traits = self.traits
        token = fx.Int64(tile_start) + self.lane_in_warp
        d_base = self.wave_id * fx.Int64(16)
        if page_id is None:
            page_id = self.load_page_id(token, uniform=False)
        byte_offset = (
            page_id * self.v_page_bytes + self.kv_head_idx * traits.HEAD_DIM_V + d_base
        )
        valid = token < self.seqlen_kv_v
        prefix = self.global_load_fp8x16(
            self.v_base_iter, byte_offset, valid, is_value=True
        )
        stages = 2 if traits.V_LOAD_LAYOUT == "token_words" else 4
        source = _transpose_v_fp8_16x16(prefix, self.lane_in_warp, stages=stages)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail = self.global_load_fp8x16(
                self.v_base_iter,
                byte_offset + traits.FP8_V_H1,
                valid & (d_base + traits.FP8_V_H1 < traits.HEAD_DIM_V),
                is_value=True,
            )
            tail_source = _transpose_v_fp8_16x16(tail, self.lane_in_warp, stages=2)
            source = source.shuffle(tail_source, [0, 1, 2, 3, 4, 5, 6, 7])
        return source.ir_value()

    def store_v_source(self, src_i32x4, buf_id, tile_start, *, mask_padding=True):
        """Store a load_v_source packet in the common MFMA byte order."""
        if const_expr(self.traits.V_LOAD_LAYOUT == "token_words"):
            self._store_v_fp8_page1(src_i32x4, buf_id)
            return
        page16_scatter = self.traits.V_LOAD_LAYOUT == "page_waves"

        if const_expr(mask_padding and self.traits.PAGE_SIZE != 1):
            group = self.wave_id_uni % 4 if page16_scatter else self.lane_in_warp // 16
            token = fx.Int64(tile_start) + fx.Int64(group) * 16
            src_i32x4 = self._mask_v_fp8_group(src_i32x4, token)
        if const_expr(page16_scatter):
            self._store_v_fp8_page16(src_i32x4, buf_id)
            return
        if const_expr(self.traits.FP8_PV_SEGMENTED):
            source = Vec(src_i32x4)
            prefix = source.shuffle(source, [0, 1, 2, 3]).ir_value()
            tail = source.shuffle(source, [4, 5, 6, 7]).ir_value()
            self._store_v_fp8_vectorized_bankpad_segment(prefix, buf_id, 0)

            @flyc.jit
            def _store_tail():
                if self.wave_id_uni < fx.Int64(
                    self.traits.FP8_V_H2 // self.traits.KV_VEC_SIZE
                ):
                    self._store_v_fp8_vectorized_bankpad_segment(
                        tail, buf_id, self.traits.FP8_V_H1
                    )

            _store_tail()
        else:
            self._store_v_fp8_vectorized_bankpad_segment(src_i32x4, buf_id, 0)

    def _store_v_fp8_page1(self, source, buf_id):
        """Complete the byte transpose by scattering four-token words to D rows."""
        traits = self.traits
        words = Vec(source)
        lane = self.lane_in_warp
        # Retained V token-bit order is [0, 1, 4, 3, 5, 2]; low bits stay in each word.
        token_word = (
            ((lane & 16) >> 2) + ((lane & 32) >> 1) + ((lane & 4) << 3) + (lane & 8)
        )
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = (
            aligned_base
            - self.lds_vt_base_idx
            + buf_id * (traits.HEAD_DIM_V * traits.FP8_V_ROW_STRIDE)
        )
        d_col = self.wave_id * fx.Int64(16) + (lane & 3)

        def _store_segment(first_word, d_offset):
            for word in range_constexpr(4):
                row = d_col + fx.Int64(d_offset + word * 4)
                token_offset = token_word
                if const_expr(traits.HEAD_DIM_V == 192):
                    token_offset = token_offset ^ (
                        ((row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                    )
                dst = tile_base + row * traits.FP8_V_ROW_STRIDE + token_offset
                ptr = fx.add_offset(
                    fx.get_iter(self.v_lds_i32_tiles), fx.Uint32(dst) // 4
                )
                fx.generic_store(ptr, fx.Int32(words[first_word + word]))

        _store_segment(0, 0)
        if const_expr(traits.FP8_PV_SEGMENTED):

            @flyc.jit
            def _store_tail():
                if self.wave_id_uni < fx.Int64(traits.FP8_V_H2 // traits.KV_VEC_SIZE):
                    _store_segment(4, traits.FP8_V_H1)

            _store_tail()

    def _store_v_fp8_page16(self, source, buf_id, segment_offset=0, include_tail=True):
        """Scatter native page vectors into the retained MFMA byte order."""
        traits = self.traits
        words = Vec(source)
        page = self.wave_id % fx.Int64(4)
        group_offset = (page % fx.Int64(2)) * fx.Int64(4) + (
            page // fx.Int64(2)
        ) * fx.Int64(16)
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = (
            aligned_base
            - self.lds_vt_base_idx
            + buf_id * (traits.HEAD_DIM_V * traits.FP8_V_ROW_STRIDE)
        )
        d_col = (self.wave_id // fx.Int64(4)) * fx.Int64(64) + self.lane_in_warp

        def _store_segment(first_word, d_offset):
            row = d_col + fx.Int64(d_offset)
            row_base = tile_base + row * traits.FP8_V_ROW_STRIDE
            for word in range_constexpr(4):
                token_offset = group_offset + (word % 2) * 32 + (word // 2) * 8
                if const_expr(traits.HEAD_DIM_V == 192):
                    token_offset = token_offset ^ (
                        ((row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                    )
                ptr = fx.add_offset(
                    fx.get_iter(self.v_lds_i32_tiles),
                    fx.Uint32(row_base + token_offset) // 4,
                )
                fx.generic_store(ptr, fx.Int32(words[first_word + word]))

        _store_segment(0, segment_offset)
        if const_expr(traits.FP8_PV_SEGMENTED and include_tail):

            @flyc.jit
            def _store_tail_pages():
                if self.wave_id_uni < fx.Int64(4):
                    _store_segment(4, traits.FP8_V_H1)

            _store_tail_pages()

    def _store_v_fp8_vectorized_bankpad_segment(self, src_i32x4, buf_id, d_offset):
        traits = self.traits
        row_stride = traits.FP8_V_ROW_STRIDE
        v_tile_bytes = traits.HEAD_DIM_V * row_stride
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        dst_base = aligned_base + fx.Int64(buf_id * v_tile_bytes)
        n_group16 = self.lane_in_warp // fx.Int64(16)
        d_col = (
            fx.Int64(d_offset)
            + self.wave_id * fx.Int64(16)
            + self.lane_in_warp % fx.Int64(16)
        )
        reordered = self._permute_v_fp8_vectorized(src_i32x4)
        dest_group16 = (n_group16 % fx.Int64(2)) * fx.Int64(2) + n_group16 // fx.Int64(
            2
        )
        if const_expr(traits.HEAD_DIM_V == 192):
            # XOR 16-byte groups instead of padding each row in the six-slot ring.
            dest_group16 = dest_group16 ^ ((d_col // fx.Int64(4)) % fx.Int64(4))
        dst_byte = dst_base + d_col * fx.Int64(row_stride) + dest_group16 * fx.Int64(16)
        self._store_v_fp8_lds(reordered, dst_byte - self.lds_vt_base_idx)


class DualwaveFp8KvLdsToVgprLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, buf_id):
        # Read K in the wide 32x32x64 QK operand layout (32 contiguous head-dim/lane,
        # two N-strips, two head-dim halves).
        traits = self.traits
        k_base = self.k_buf_base(buf_id)
        d_base = self.lane_div_32 * 32
        n_lo = self.lane_mod_32
        n_hi = self.lane_mod_32 + 32

        if const_expr(traits.KV_VECTORIZED):

            def _read_vec_key(key):
                key_sigma = _sigma_k_tile_n(key)
                packs = []
                for ws in range_constexpr(traits.HEAD_DIM // 64):
                    dg0 = ws * 4 + self.lane_div_32 * 2
                    if const_expr(traits.PAGED and traits.K_LDS_PAGE_GROUPED):
                        # [D-group/4, page, D-group%4, token%16, byte]. The
                        # offset delta from page-64 layout is divisible by 256,
                        # preserving the gfx950 LDS bank index.
                        oct_idx = (
                            (ws * 4 + key_sigma // 16) * 64
                            + self.lane_div_32 * 32
                            + key_sigma % 16
                        )
                        off0 = k_base + oct_idx * traits.KV_VEC_SIZE
                        off1 = off0 + 16 * traits.KV_VEC_SIZE
                    else:
                        off0 = (
                            k_base
                            + (dg0 * traits.BLOCK_N + key_sigma) * traits.KV_VEC_SIZE
                        ) * traits.ELEM_BYTES
                        off1 = (
                            k_base
                            + ((dg0 + 1) * traits.BLOCK_N + key_sigma)
                            * traits.KV_VEC_SIZE
                        ) * traits.ELEM_BYTES
                    lo = Vec(self.read_i32x4_lds(off0), (4,), fx.Int32)
                    hi = Vec(self.read_i32x4_lds(off1), (4,), fx.Int32)
                    packs.append(lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
                return packs

            return (_read_vec_key(n_lo), _read_vec_key(n_hi))

        rows_per_line = traits.NUM_WAVES

        def _read_strip(key):
            out = []
            for ws in range_constexpr(traits.HEAD_DIM // 64):
                b = traits.K_WS_BAND[ws]
                line = (key % rows_per_line) * traits.K_BAND_LINE_STRIDE[b]
                row = line + (key // rows_per_line) * traits.K_BAND_CHUNK[b]
                addr = (
                    k_base + traits.K_BAND_BASE[b] + row + traits.K_WS_OFF[ws] + d_base
                )
                out.append(self.read_i32x8_lds(self.lds_kv_base_ptr, addr))
            return out

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id):
        if const_expr(self.traits.PAGED):
            return self._load_v_fp8_vectorized_bankpad(buf_id)
        return self._load_v_fp8_block(buf_id)

    def _load_v_fp8_vectorized_bankpad(self, buf_id):
        """Read the D-major V tile into the FP8 MFMA A fragment."""
        packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            fragment = Vec(self.load_v_fragment(buf_id, dc)).bitcast(fx.Int64)
            for ks in range_constexpr(4):
                packs[ks][dc] = fragment[ks].ir_value()
        return packs

    def load_v_fragment(self, buf_id, dc):
        """Read one 32-channel V fragment in the retained MFMA byte order."""
        traits = self.traits
        row_stride = traits.FP8_V_ROW_STRIDE
        v_tile_bytes = traits.HEAD_DIM_V * row_stride
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = aligned_base + fx.Int64(buf_id * v_tile_bytes)
        token_base = self.lane_div_32 * fx.Int64(32)
        d_row = fx.Int64(dc * traits.D_CHUNK) + self.lane_mod_32
        row_base = tile_base + d_row * fx.Int64(row_stride)
        halves = []
        for half in range_constexpr(2):
            token_offset = token_base + fx.Int64(half * 16)
            if const_expr(traits.HEAD_DIM_V == 192):
                token_offset = token_offset ^ (
                    ((d_row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                )
            view = fx.slice(
                self.v_lds_i32_tiles,
                (None, fx.Uint32(row_base + token_offset - self.lds_vt_base_idx) // 16),
            )
            halves.append(fx.generic_load(fx.get_iter(view), dtype=fx.Int32, count=4))
        return halves[0].shuffle(halves[1], [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM_V // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM_V // 16
        rh = (self.lane % fx.Index(32)) // fx.Index(16)
        l16 = self.lane % fx.Index(16)
        lane_hi = self.lane // fx.Index(32)
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
        base = fx.Int32(
            aligned_base
            + buf_off
            + rh * fx.Index(128)
            + l16 * fx.Index(8)
            + lane_hi * fx.Index(nbands * 128)
        )

        def _tr8(imm):
            r = _ds_read_tr8_b64_imm(self.v2i32_type, base, imm)
            return Vec(r).bitcast(fx.Int64)[0].ir_value()

        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for ks in range_constexpr(4):
                imm0 = (2 * ks * nbands + dc * 2) * 128
                packs[ks][dc] = _tr8(imm0)
        return packs


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def reduce_max_pair(self, v_s_a, v_s_b):
        values = [self.c_neg_inf]
        for scores in (*v_s_a, *v_s_b):
            values += [Vec(scores)[i] for i in range_constexpr(16)]
        local_max = _tree_reduce(values, fx.maxnumf)
        lhs, rhs = _reduction_pair(local_max)
        return fx.maxnumf(lhs, rhs)

    def _attn_mask_vec2_imm(
        self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
    ):
        return _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
        )

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        # Vectorized-K sigma arranges wide FP8 scores in four-token groups
        # (0..3, 8..11, ...), matching dense FP8 but not vectorized BF16.
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        pair_thresholds = _causal_pair_thresholds(False)
        if const_expr(self.traits.PAGED):
            for r in range_constexpr(16):
                threshold = (r // 4) * 8 + r % 4
                s_lo[r] = (rel_lo_i32 < threshold).select(self.c_neg_inf, s_lo[r])
                s_hi[r] = (rel_hi_i32 < threshold).select(self.c_neg_inf, s_hi[r])
        else:
            _apply_dualwave_causal_mask_pair(
                s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds
            )
            _apply_dualwave_causal_mask_pair(
                s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds
            )

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def causal_mask_pair_if_needed(self, v_s_a, v_s_b, tile_a):
        """Causal-mask a BN128 tile pair under one scalar-uniform branch.

        Branching per sub-tile would put two scf.if regions in the body on every
        iteration, splitting it into five basic blocks and blocking the QK/softmax/PV
        interleave the sched_group_barriers ask for. Branching once on the pair's end
        position keeps every strictly-below-diagonal pair a single straight-line block.
        Masking sub-tile `a` inside the taken branch even when only `b` needs it is a
        no-op beyond the VALU compares, and happens in at most one pair per q-block.

        This replaces seq_pad_mask_if_needed: with delta = seqlen_kv - seqlen_q the
        largest key any row may attend to is seqlen_kv - 1, so every padding column is
        already strictly above the diagonal.
        """
        traits = self.traits
        kv_end_pos = (tile_a + fx.Index(2)) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                a_l, a_h = self.v_s_vec_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = self.v_s_vec_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + fx.Index(1))
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        col_base = fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(
            self.lane_div_32
        ) * fx.Int32(4)
        for r in range_constexpr(16):
            thr = (r // 4) * 8 + (r % 4)
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * self.traits.BLOCK_N
            if fx.Int32(kv_tile_end) > self.seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def max2(self, a, b):
        return fx.maxnumf(a, b)

    def floor_masked_max(self, row_max):
        return fx.maxnumf(row_max, self.c_neg_floor)

    # log2 of e4m3's largest finite value, 448.
    _P_HEADROOM_LOG2 = 8.807354922057604

    def sub_m(self, v_s, row_max):
        # P is cast to e4m3, whose smallest subnormal is 2**-9, so a softmax
        # over thousands of keys loses its tail to flush-to-zero -- while l_row,
        # summed before the cast, still counts it. Scaling P up first uses the
        # format's whole range; l_row scales with it, so the output is unchanged
        # apart from the tail that survives. Free: it rides the FMA's addend.
        #
        # Available headroom is bounded by how large exp2 gets: the lazy path
        # holds the running max until a tile exceeds it by RESCALE_THRESHOLD, so
        # exp2 <= 2**THRESHOLD there; the eager path rebases every tile.
        headroom = self._P_HEADROOM_LOG2
        if const_expr(self.traits.DUALWAVE_SWP_LAZY_RESCALE):
            headroom -= self.traits.DUALWAVE_SWP_RESCALE_THRESHOLD
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s,
            row_max,
            self.c_logit_scale,
            self.c_zero_f,
            self.fm_fast,
            bias,
            explicit_rounding=self.traits.PAGED,
        )

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return l_row + self.tile_sum(v_p)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(
                    llvm.FPExtOp(
                        Vec.make_type(8, fx.Float32), as_mlir_value(pk)
                    ).result,
                    (8,),
                    fx.Float32,
                )
                scaled = [fx.Float32(f32[i]) * scale_scalar for i in range(8)]
                dst.append(self.bf16_trunc_pack_v8(scaled))
        return out_lo, out_hi

    def anchor_v_p(self, v_p):
        return _anchor_v_p(self.traits, v_p, elem_dtype=self.p_elem)

    def anchor_v_o(self, v_o):
        return self.preserve_accumulators(v_o)

    def anchor_scalar_f32(self, x):
        if const_expr(self.traits.PAGED):
            return llvm.intr_arithmetic_fence(fx.as_ir_value(x))
        return _anchor_scalar_f32(x)

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = rocdl.exp2(T.f32, as_mlir_value(diff_scaled))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        self.scale_o(v_o, corr)
        v_o = self.anchor_v_o(v_o)
        v_p = self.scale_v_p(v_p, corr)
        l_row = self.apply_l_rescale(l_row, corr)
        return v_o, m_new, l_row, v_p

    def v_p_to_vec32(self, v_p):
        # P packs are (p_lo[0..1], p_hi[0..1]) v8 bf16; concat into one v32 SSA value
        # for the scf.if loop-carry.
        return _v_p_to_vec32(v_p)

    def v_vec32_to_p(self, v_p_all):
        return _v_vec32_to_p(self.traits, v_p_all, elem_dtype=self.p_elem)

    def _lazy_correction(self, v_o, m_row, m_tile_max):
        """Monotonic: a downward rebase gives corr > 1 and repeated ones overflow."""
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            m_new,
            corr,
            [as_mlir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = (fx.Int64(ballot) == fx.Int64(_read_exec_i64())).ir_value()
            all_below = llvm.intr_expect(all_below, fx.Boolean(True).ir_value())

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out)

        return _run(v_o, m_row, l_row, m_tile_max)


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        if const_expr(self.traits.PAGED):
            values = Vec(v_o[dc])
            packed = (
                values.shuffle(values, list(range(r_base, r_base + 4)))
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            return packed[0].ir_value(), packed[1].ir_value()
        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(
            pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
        )
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != fx.Index(0)).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != fx.Index(0)
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(
            [fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32
        )

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        m_row = fx.Float32(m_row) * self.c_logit_scale
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = (
            (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v
            + q_row
        ) * (self.traits.HEAD_DIM_V // 2)
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = (
            grid_z
            * self.traits.NUM_HEADS_Q
            * self.seq_len_v
            * (self.traits.HEAD_DIM_V // 2)
        )
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (
            split_z * self.traits.NUM_HEADS_Q + self.q_head_idx
        ) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = (
                            dc * (self.traits.D_CHUNK // 2)
                            + (2 * g + self.lane_div_32) * 4
                        )
                        self.ws_store_quad_i32(
                            self._packed_o_128_dwords(v_o, dc, g),
                            o_part_row_base + dw_col,
                        )
                if self.lane < fx.Index(32):
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + fx.Index(4):
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = (
                    (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx)
                    * self.seq_len_v
                    + q_row_e
                ) * (self.traits.HEAD_DIM_V // 2)
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = (
                    grid_z_e
                    * self.traits.NUM_HEADS_Q
                    * self.seq_len_v
                    * (self.traits.HEAD_DIM_V // 2)
                )
                lrow_base_e = (
                    mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                )
                ml_row_e = (
                    split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx
                ) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = (
                                dc * (self.traits.D_CHUNK // 2)
                                + (2 * g + self.lane_div_32) * 4
                            )
                            self.ws_store_quad_i32(
                                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                                o_row_base_e + dw_col,
                            )
                    if self.lane < fx.Index(32):
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()


class DualwaveSplitKCombineContext:
    """Shared per-kernel state for the split-K combine pass."""

    def __init__(
        self,
        traits_or_ctx,
        O=None,  # noqa: E741
        WS=None,
        batch_size=None,
        seq_len=None,
        stride_o_n=None,
        LSE=None,
        Sink=None,
        CuSeqQ=None,
    ):
        if isinstance(traits_or_ctx, DualwaveSplitKCombineContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.O = O
        self.WS = WS
        self.LSE = LSE
        self.Sink = Sink
        self.CuSeqQ = CuSeqQ
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride_o_n = stride_o_n

    def init_types_and_constants(self):
        self.out_dtype_str = (
            "bf16" if self.traits.DTYPE_STR == "fp8" else self.traits.DTYPE_STR
        )
        self.elem_dtype = dtype_to_elem_type(self.out_dtype_str)
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.c_zero_f = fx.Float32(0.0)
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        # LSE store folds the log2->ln conversion (m_max is sm_scale*log2e-scaled).
        self.c_ln2_f = fx.Float32(1.0 / _LOG2E)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.stride_o_n_v = fx.Index(self.stride_o_n)
        self.batch_size_v = fx.Index(self.batch_size)

    def init_thread_mapping(self, combine_rows_per_block, combine_lanes_per_row):
        traits = self.traits
        self.tid = fx.Index(gpu.thread_idx.x)
        self.blk = fx.Index(gpu.block_idx.x)
        self.batch_idx = fx.Index(gpu.block_idx.y)
        self.col = (self.tid % combine_lanes_per_row) * 4
        rows_per_batch = self.seq_len_v * traits.NUM_HEADS_Q
        row_raw = self.blk * combine_rows_per_block + self.tid // combine_lanes_per_row
        threads_in_use = fx.Index(combine_rows_per_block * combine_lanes_per_row)
        self.row = (self.tid < threads_in_use).select(row_raw, rows_per_batch)
        self.row_valid = self.row < rows_per_batch
        self.q_head_idx = self.row // self.seq_len_v
        self.seq_idx = self.row % self.seq_len_v

    def init_workspace(self):
        traits = self.traits
        z_total = self.batch_size_v * traits.NUM_KV_SPLITS
        self.ws_opart_per_split_elems = (
            fx.Index(traits.NUM_HEADS_Q)
            * self.seq_len_v
            * fx.Index(traits.HEAD_DIM_V // 2)
        )
        self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
        self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
        self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
        self.ws_mrow_abs_bytes = z_total * self.ws_opart_per_split_bytes
        self.ws_lrow_abs_bytes = (
            self.ws_mrow_abs_bytes + z_total * self.ws_ml_per_split_bytes
        )
        self.local_ml_idx = self.q_head_idx * self.seq_len_v + self.seq_idx
        self.local_o_base = (
            self.q_head_idx * self.seq_len_v + self.seq_idx
        ) * fx.Index(traits.HEAD_DIM_V // 2)
        self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.WS)))

    def init_descriptors(self):
        if const_expr(self.traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)
            q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            q_tok_end = _cu_load(
                _cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            batch_byte_off = q_tok_base * self.stride_o_n_v * fx.Index(2)
            nrec_bytes = (q_tok_end - q_tok_base) * self.stride_o_n_v * fx.Index(2)
        else:
            per_batch_elems = self.seq_len_v * self.stride_o_n_v
            batch_byte_off = self.batch_idx * per_batch_elems * fx.Index(2)
            nrec_bytes = per_batch_elems * fx.Index(2)
        self.o_nrec_bytes = nrec_bytes
        self.o_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(
                fx.Int64(fx.ptrtoint(fx.get_iter(self.O))) + fx.Int64(batch_byte_off)
            ),
            num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)),
        )
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def workspace_resource(self, byte_offset, nrec_bytes):
        return _make_ws_rsrc(self.ws_base_i64, byte_offset, nrec_bytes)

    def split_z(self, split_i):
        return self.batch_idx * self.traits.NUM_KV_SPLITS + split_i

    def opart_resource(self, split_z):
        return self.workspace_resource(
            split_z * self.ws_opart_per_split_bytes, self.ws_opart_per_split_bytes
        )

    def mrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_mrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )

    def lrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_lrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )


class DualwaveSplitKCombineHelper(DualwaveSplitKCombineContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_ml_rows(self):
        m_s = []
        l_s = []
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            split_z_i = self.split_z(i)
            m_f32 = buffer_ops.buffer_load(
                self.mrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            l_f32 = buffer_ops.buffer_load(
                self.lrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            m_s.append(m_f32)
            l_s.append(l_f32)
        return m_s, l_s

    def reduce_m_max(self, m_s):
        m_max = m_s[0]
        for i in range_constexpr(self.traits.NUM_KV_SPLITS - 1):
            m_max = fx.maxnumf(m_max, m_s[i + 1])
        return m_max

    def fold_sink(self, m_max, bias_log2e):
        sink_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(fx.Int64(fx.ptrtoint(fx.get_iter(self.Sink)))),
            num_records_bytes=as_mlir_value(fx.Int64(self.traits.NUM_HEADS_Q * 4)),
        )
        sink_f32 = buffer_ops.buffer_load(
            sink_rsrc,
            as_mlir_value(fx.Int32(self.q_head_idx)),
            vec_width=1,
            dtype=T.f32,
        )

        sink_log2 = sink_f32 * fx.Float32(bias_log2e)
        m_new = fx.maxnumf(m_max, sink_log2)
        sink_w = rocdl.exp2(T.f32, as_mlir_value(sink_log2 - m_new))
        return m_new, sink_w

    def add_sink_den(self, den, sink_w):
        return den + sink_w

    def init_accumulators(self):
        return as_mlir_value(self.c_zero_v4f32), as_mlir_value(self.c_zero_f)

    def accumulate_split(self, acc, den, split_i, m_i, l_i, m_max):
        orsrc_i = self.opart_resource(self.split_z(split_i))
        local_o_idx_i = self.local_o_base + self.col // 2

        @flyc.jit
        def _accum_split(acc, den):
            if fx.Float32(l_i) > fx.Float32(0.0):
                w = rocdl.exp2(T.f32, as_mlir_value(m_i - m_max))
                wl = w * l_i
                den = den + wl
                o2_raw = buffer_ops.buffer_load(
                    orsrc_i,
                    as_mlir_value(fx.Int32(local_o_idx_i)),
                    vec_width=2,
                    dtype=T.i32,
                )
                o4 = Vec(o2_raw, (2,), fx.Int32).bitcast(self.elem_dtype).to(fx.Float32)
                w4 = Vec.from_elements([fx.Float32(wl)], fx.Float32).broadcast_to(4)
                acc = acc + w4 * o4
            return acc, den

        return _accum_split(acc, den)

    def accumulate_splits(self, m_s, l_s, m_max):
        acc, den = self.init_accumulators()
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            acc, den = self.accumulate_split(acc, den, i, m_s[i], l_s[i], m_max)
        return acc, den

    def pack_output(self, acc, den):
        inv_rcp = rocdl.rcp(T.f32, den)
        inv = (fx.Float32(den) > self.c_zero_f).select(inv_rcp, self.c_zero_f)
        inv4 = Vec.from_elements([fx.Float32(inv)], fx.Float32).broadcast_to(4)
        out4 = Vec(acc * inv4, (4,), fx.Float32)
        if const_expr(self.out_dtype_str == "bf16"):
            lo = rocdl.cvt_pk_bf16_f32(out4[0], out4[1])
            hi = rocdl.cvt_pk_bf16_f32(out4[2], out4[3])
        else:
            o_f16 = []
            for i in range_constexpr(4):
                o_f16.append(fx.Float32(out4[i]).to(self.elem_dtype))
            pack = Vec.from_elements(o_f16, self.elem_dtype).bitcast(fx.Int32)
            lo, hi = as_mlir_value(pack[0]), as_mlir_value(pack[1])
        return Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)

    def store_lse(self, m_max, den):
        # Combined LSE = m_max * ln2 + ln(den); den = sum_s 2^(m_s - m_max) * l_s
        # completes the natural-log, scale-folded LSE. One lane (col == 0) writes.
        lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE)))
        lse_per_batch_elems = fx.Index(self.traits.NUM_HEADS_Q) * self.seq_len_v
        lse_per_batch_bytes = lse_per_batch_elems * fx.Index(4)
        lse_rsrc = _make_ws_rsrc(
            lse_base_i64, self.batch_idx * lse_per_batch_bytes, lse_per_batch_bytes
        )
        # As the unsplit store: -inf selected for a row no split reached (modification 23).
        # "No keys" is den == +0 (`_no_keys`), so a NaN row keeps a NaN LSE
        # (modification 29).
        lse_val = _no_keys(den).select(
            self.c_neg_inf,
            fx.Float32(
                m_max * self.c_ln2_f + fx.log(den, fastmath=fx.arith.FastMathFlags.none)
            ),
        )
        lse_in_range = self.row_valid.select(self.local_ml_idx, lse_per_batch_elems)
        lse_off = fx.Index(
            (self.col == fx.Index(0)).select(lse_in_range, lse_per_batch_elems)
        )
        buffer_ops.buffer_store(
            as_mlir_value(fx.Float32(lse_val)),
            lse_rsrc,
            as_mlir_value(fx.Int32(lse_off)),
        )

    def store_output(self, o_pack):
        o_global = (
            self.seq_idx * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + self.col
        )
        # Out-of-range rows aim past num_records, which the buffer drops.
        o_off = self.row_valid.select(o_global * fx.Index(2), self.o_nrec_bytes)
        buffer_ops.buffer_store(
            o_pack.ir_value(),
            self.o_rsrc,
            as_mlir_value(fx.Int32(o_off)),
            offset_is_bytes=True,
        )


# Misc debug and public helpers


def _scale_sched_pairs(pairs, head_dim):
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs


def _sched_barrier_pairs(traits, pairs, valu_cnt, group):
    """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, valu_cnt, group)


def _sched_barrier_exp_pairs(traits, pairs, exp_cnt, group):
    """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, exp_cnt, group)


def _stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier;` only when stagger == 0."""
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s",
        has_side_effects=True,
    )


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


def _debug_atomic_inc_lazy_count(byte_offset, debug_counts_rsrc):
    rocdl.raw_buffer_atomic(as_mlir_value(fx.Float32(1.0))) + debug_counts_rsrc


@flyc.jit
def _debug_count_lazy_branch(traits, all_below, debug_counts_rsrc, lane):
    if const_expr(traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS):
        if fx.Int32(lane) == fx.Int32(0):
            if fx.Boolean(all_below):
                _debug_atomic_inc_lazy_count(0, debug_counts_rsrc=debug_counts_rsrc)
            else:
                _debug_atomic_inc_lazy_count(4, debug_counts_rsrc=debug_counts_rsrc)


def dualwave_splitk_workspace_elems(
    batch_size, num_heads, seq_len, num_kv_splits, head_dim=128
):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows


def daz_denormal_attr():
    """Denormals-are-zero for f32, as an attribute the backend honours.

    Attach under the key ``llvm.denormal_fpenv``: the prefix is what carries it
    through GPU-to-ROCDL lowering onto the inherent attribute upstream reads.
    The old ``denormal-fp-math-f32`` passthrough string reaches the IR but no
    longer moves the denorm mode -- so verify via
    ``.amdhsa_float_denorm_mode_32`` in ``*_final_isa.s``, never by grepping
    the IR.
    """
    return ir.Attribute.parse(
        "#llvm.denormal_fpenv<"
        "default_output_mode = ieee, default_input_mode = ieee, "
        "float_output_mode = preservesign, float_input_mode = preservesign>"
    )
