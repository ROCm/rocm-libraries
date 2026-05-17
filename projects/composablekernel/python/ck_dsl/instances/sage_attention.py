# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sage attention forward (CK Tile ``49_sageattention`` parity).

Extends FMHA-fwd-fp8 with per-block / per-head Q and K scales that
compensate for the dynamic range loss when Q and K live in
fp8 / int8 / int4 storage. The pipeline (per K-block)::

    K_dequant = dequant_codebook[K_quant[k_block, :]] * k_block_scale
    score_log2 = (Q[q_token, :] . K_dequant) * Q_scale * K_scale
    ... (online softmax as in fmha_fwd) ...
    V_dequant = dequant_codebook[V_quant[k_block, :]] * v_block_scale
    acc += p * V_dequant

The four CK Tile Sage variants share one builder, parameterised by
``quant_mode``:

* ``"fp16_bf16"`` -- baseline (no QK quant; pipeline validation).
* ``"fp8_bf16"``  -- Q in activation dtype; K/V in fp8e4m3 with per-block scales.
* ``"i8_fp8_bf16"`` -- K/V stored as i8; codebook re-materialises fp8 mantissa.
* ``"i4_fp8_bf16"`` -- K/V as packed i4; 16-entry codebook + per-block scale.

The kernel runs with one warp per CTA -- lanes distribute the head_dim
axis, the QK dot product reduces via the warp-shuffle butterfly, the
online-softmax state and per-lane accumulator slice live in registers
across the K-loop via ``scf.for`` iter_args. No LDS state, no
thread-redundant work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import (
    BF8E5M2,
    FP8E4M3,
    I32,
    I8,
    IRBuilder,
    KernelDef,
)
from ..helpers.attention import apply_attention_mask, warp_xor_reduce_sum
from ..helpers.codebook import (
    codebook_lookup_i4_pair_to_fp8,
    codebook_lookup_i8_to_fp8,
)
from ..helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ..helpers.qk_scale import (
    QkScaleSpec,
    apply_qk_scales,
    load_k_scale_for_block,
    load_q_scale_for_block,
)
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE


__all__ = [
    "SageAttentionSpec",
    "SageQuantMode",
    "build_sage_attention",
    "is_valid_spec",
    "sage_attention_grid",
    "sage_attention_signature",
]


SageQuantMode = Literal["fp16_bf16", "fp8_bf16", "i8_fp8_bf16", "i4_fp8_bf16"]


@dataclass(frozen=True)
class SageAttentionSpec:
    """One concrete Sage attention configuration."""

    common: FmhaCommonSpec
    quant_mode: SageQuantMode
    q_scale: QkScaleSpec
    k_scale: QkScaleSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "ck_dsl_sage_attention"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            self.quant_mode,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
        )


def is_valid_spec(spec: SageAttentionSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.quant_mode not in (
        "fp16_bf16",
        "fp8_bf16",
        "i8_fp8_bf16",
        "i4_fp8_bf16",
    ):
        return False, f"unknown quant_mode {spec.quant_mode!r}"
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.quant_mode == "i4_fp8_bf16" and (spec.common.shape.head_size % 2) != 0:
        return False, (
            f"i4 sage requires head_size divisible by 2 "
            f"(got {spec.common.shape.head_size})"
        )
    if spec.common.shape.head_size % WARP_SIZE != 0:
        return False, (
            f"sage warp body needs head_size % {WARP_SIZE} == 0 "
            f"(got {spec.common.shape.head_size})"
        )
    return True, "ok"


def _kv_pointee_for_quant_mode(quant_mode: str, dtype: str):
    if quant_mode == "fp16_bf16":
        return io_ir_type(dtype)
    if quant_mode == "fp8_bf16":
        return FP8E4M3
    return I8


def _kv_dtype_str(quant_mode: str, dtype: str) -> str:
    """Map ``quant_mode`` to the ABI dtype string for K / V tensors."""
    if quant_mode == "fp16_bf16":
        return dtype
    if quant_mode == "fp8_bf16":
        return "fp8e4m3"
    return "i8"


def _declare_params(kb: FmhaKernelBuilder, spec) -> None:
    """Declare the sage attention kernel ABI (shared between build + sig)."""
    kv_dtype = _kv_dtype_str(spec.quant_mode, kb.common.dtype)
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", dtype=kv_dtype, readonly=True, align=8)
    kb.add_tensor("V", dtype=kv_dtype, readonly=True, align=8)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("q_scale", dtype="f32", readonly=True)
    kb.add_ptr("k_scale", dtype="f32", readonly=True)
    if spec.quant_mode in ("i8_fp8_bf16", "i4_fp8_bf16"):
        kb.add_ptr("codebook_k", dtype="f32", readonly=True)
        kb.add_ptr("codebook_v", dtype="f32", readonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def _load_k_slot_f32(
    b: IRBuilder,
    *,
    K,
    base,
    d_addr,
    dtype: str,
    quant_mode: str,
    cb_k,
    kv_ty,
):
    """Load one head-dim K element and dequant to f32.

    For the ``i4`` quant mode the byte at ``base + d_addr/2`` holds
    two nibbles; the caller's lane is responsible for the correct
    pair indexing. Sage's i4 path requires ``head_size`` to be even
    AND the lane's ``EPT`` to be even so the two nibbles always
    belong to the same lane. v1 enforces ``EPT == 1`` by limiting to
    head_size == warp_size = 64; that means the i4 path is currently
    *not* directly representable with one element per lane (one byte
    = two elements). The i4 build below splits the work so two
    lanes share one byte load -- each lane uses one nibble.
    """
    addr = b.add(base, d_addr)
    if quant_mode == "fp16_bf16":
        v_raw = b.global_load(K, addr, kv_ty)
        return b.cast_to_f32(v_raw)
    if quant_mode == "fp8_bf16":
        v_raw = b.global_load(K, addr, kv_ty)
        return b.cvt_fp8_to_f32(v_raw)
    if quant_mode == "i8_fp8_bf16":
        i8_byte = b.global_load(K, addr, I8)
        i32 = b.sext(i8_byte, I32)
        kd_fp8 = codebook_lookup_i8_to_fp8(b, cb_k, i32)
        return b.cvt_fp8_to_f32(kd_fp8)
    raise ValueError(f"unsupported quant_mode {quant_mode!r}")


def build_sage_attention(spec: SageAttentionSpec) -> KernelDef:
    """Sage attention forward kernel; dispatches on quant_mode.

    The i4 path uses one packed-byte load per lane (each byte holds
    two nibbles); the lane's low nibble feeds the ``2*tid``-th head
    dim element and the high nibble feeds the ``2*tid+1``-th. This
    keeps the byte load alignment natural without requiring a cross-
    lane shuffle to share the byte.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid sage_attention spec: {why}")
    s = spec.common
    dtype = s.dtype
    H = s.shape.head_size
    block_size = WARP_SIZE
    kv_ty = _kv_pointee_for_quant_mode(spec.quant_mode, dtype)

    if spec.quant_mode == "i4_fp8_bf16":
        # Two head-dim slots per lane (one byte = two nibbles); requires
        # head_size = 2 * warp_size i.e. >= 128.
        if H < 2 * WARP_SIZE:
            raise ValueError(
                f"i4 sage requires head_size >= {2 * WARP_SIZE} so each "
                f"lane owns one packed byte (two nibbles); got {H}"
            )
        if H % (2 * WARP_SIZE) != 0:
            raise ValueError(
                f"i4 sage requires head_size % {2 * WARP_SIZE} == 0; got {H}"
            )
        ept_pairs = H // (2 * WARP_SIZE)  # bytes per lane
        # We don't currently support ept_pairs > 1; bail out if so.
        if ept_pairs != 1:
            raise ValueError(
                "i4 sage v1 supports head_size == 128 (one byte per lane); "
                f"got head_size={H} which would need {ept_pairs} bytes/lane"
            )
    ept = H // WARP_SIZE

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(block_size)
    _declare_params(kb, spec)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    q_scale_ptr = kb.ptr("q_scale")
    k_scale_ptr = kb.ptr("k_scale")
    cb_k = (
        kb.ptr("codebook_k")
        if spec.quant_mode in ("i8_fp8_bf16", "i4_fp8_bf16")
        else None
    )
    cb_v = (
        kb.ptr("codebook_v")
        if spec.quant_mode in ("i8_fp8_bf16", "i4_fp8_bf16")
        else None
    )
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k = kb.scalar("seqlen_k")
    q_token = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = b.const_i32(0)
    tid = b.thread_id_x()

    # Per-lane head-dim slice. For most quant modes EPT slots per lane.
    # For i4 mode EPT is 2 (one byte = two nibbles per lane).
    if spec.quant_mode == "i4_fp8_bf16":
        lane_d_base = b.mul(tid, b.const_i32(2))  # head_dim cursor
        n_slots = 2
    else:
        lane_d_base = b.mul(tid, b.const_i32(ept))
        n_slots = ept

    q_row = kb.q_row_base()
    o_row = kb.o_row_base()

    # Pre-load this lane's Q slice (activation dtype, f32-promoted).
    q_lane = []
    for k in range(n_slots):
        d = b.add(lane_d_base, b.const_i32(k))
        q_lane.append(load_scalar_as_f32(b, Q, b.add(q_row, d), dtype=dtype))

    # Per-Q-block scale (loaded once for the whole CTA).
    q_block_idx = (
        b.div(q_token, b.const_i32(spec.q_scale.scale_block))
        if spec.q_scale.layout == "per_block"
        else b.const_i32(0)
    )
    q_scale_v = load_q_scale_for_block(
        b,
        q_scale_ptr,
        spec=spec.q_scale,
        batch_idx=batch_idx,
        head_idx=head_idx,
        q_block_idx=q_block_idx,
    )

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    iter_args = [("m", neg_inf), ("l", zero_f)]
    for k in range(n_slots):
        iter_args.append((f"a{k}", zero_f))

    k_loop = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="k_idx",
    )
    with k_loop as (k_idx, state_vals):
        m, l = state_vals[0], state_vals[1]  # noqa: E741 - online-softmax (m,l) state
        acc_iter = state_vals[2:]

        k_block_idx = (
            b.div(k_idx, b.const_i32(spec.k_scale.scale_block))
            if spec.k_scale.layout == "per_block"
            else b.const_i32(0)
        )
        k_scale_v = load_k_scale_for_block(
            b,
            k_scale_ptr,
            spec=spec.k_scale,
            batch_idx=batch_idx,
            head_idx=kv_head_idx,
            k_block_idx=k_block_idx,
        )

        k_row_base = kb.k_row_base(k_idx)
        v_row_base = kb.v_row_base(k_idx)

        partial = b.const_f32(0.0)
        v_lane = []
        if spec.quant_mode == "i4_fp8_bf16":
            # Each lane reads one packed byte = two nibbles -> two
            # head_dim slots [2*tid, 2*tid+1].
            byte_off = b.div(lane_d_base, b.const_i32(2))  # = tid
            packed_k = b.global_load(K, b.add(k_row_base, byte_off), I8)
            lo_fp8, hi_fp8 = codebook_lookup_i4_pair_to_fp8(b, cb_k, packed_k)
            lo_k = b.cvt_fp8_to_f32(lo_fp8)
            hi_k = b.cvt_fp8_to_f32(hi_fp8)
            partial = b.fadd(partial, b.fmul(q_lane[0], lo_k))
            partial = b.fadd(partial, b.fmul(q_lane[1], hi_k))
            packed_v = b.global_load(V, b.add(v_row_base, byte_off), I8)
            v_lo_fp8, v_hi_fp8 = codebook_lookup_i4_pair_to_fp8(b, cb_v, packed_v)
            v_lane.append(b.cvt_fp8_to_f32(v_lo_fp8))
            v_lane.append(b.cvt_fp8_to_f32(v_hi_fp8))
        else:
            for k in range(n_slots):
                d_addr = b.add(lane_d_base, b.const_i32(k))
                k_v = _load_k_slot_f32(
                    b,
                    K=K,
                    base=k_row_base,
                    d_addr=d_addr,
                    dtype=dtype,
                    quant_mode=spec.quant_mode,
                    cb_k=cb_k,
                    kv_ty=kv_ty,
                )
                partial = b.fadd(partial, b.fmul(q_lane[k], k_v))
                v_v = _load_k_slot_f32(
                    b,
                    K=V,
                    base=v_row_base,
                    d_addr=d_addr,
                    dtype=dtype,
                    quant_mode=spec.quant_mode,
                    cb_k=cb_v,
                    kv_ty=kv_ty,
                )
                v_lane.append(v_v)

        dot = warp_xor_reduce_sum(b, partial, stages=6)
        score_log2 = b.fmul(dot, scale_log2)
        score_log2 = apply_qk_scales(
            b,
            score_log2,
            q_scale=q_scale_v,
            k_scale=k_scale_v,
        )
        score_log2 = apply_attention_mask(
            b,
            score_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )

        m_new = b.fmax(m, score_log2)
        alpha = b.exp2(b.fsub(m, m_new))
        p = b.exp2(b.fsub(score_log2, m_new))
        l_new = b.fadd(b.fmul(l, alpha), p)

        new_yields = [m_new, l_new]
        for k in range(n_slots):
            new_yields.append(b.fadd(b.fmul(acc_iter[k], alpha), b.fmul(p, v_lane[k])))
        b.scf_yield(*new_yields)

    l_final = k_loop.results[1]
    acc_final = list(k_loop.results[2:])
    inv_l = b.rcp(l_final)
    for k in range(n_slots):
        d = b.add(lane_d_base, b.const_i32(k))
        store_scalar_from_f32(
            b,
            O,
            b.add(o_row, d),
            b.fmul(acc_final[k], inv_l),
            dtype=dtype,
        )

    b.ret()
    return kb.kernel


def sage_attention_grid(spec: SageAttentionSpec) -> Tuple[int, int, int]:
    return (spec.seqlen_q, spec.common.shape.num_query_heads, 1)


def sage_attention_signature(spec: SageAttentionSpec):
    kb = FmhaKernelBuilder("ck_dsl_sage_attention_sig_probe", spec.common)
    _declare_params(kb, spec)
    return kb.signature()


_BF8E5M2 = BF8E5M2  # noqa: F841 - re-export anchor
