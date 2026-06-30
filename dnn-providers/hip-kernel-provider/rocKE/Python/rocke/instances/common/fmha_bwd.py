# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA backward pass (CK Tile ``01_fmha`` bwd parity).

Computes::

 dQ[i, d] = sum_j P[i, j] * scale * (dS[i, j] - rowsum) * K[j, d]
 dK[j, d] = sum_i P[i, j] * scale * (dS[i, j] - rowsum) * Q[i, d]
 dV[j, d] = sum_i P[i, j] * dO[i, d]

where ``P = softmax(Q @ K^T / sqrt(d))`` (recomputed from saved
``(M, L)``) and ``dS[i, j] = dO[i, :] . V[j, :]``.

Warp-distributed body: one CTA per ``(q_token, head)``, one warp
per CTA, lane ``t`` owns head-dim slot ``t * EPT + k``. The first
K-loop computes ``rowsum_dp`` via a loop-carried iter_arg; the
second loop atomic-adds dV / dK to global and accumulates dQ per
lane (single store per lane at the end). Both loops apply the
spec's ``mask_mode`` consistently with the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.attention import apply_attention_mask, warp_xor_reduce_sum
from ...helpers.io import load_lane_slice_f32
from ...helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE


__all__ = [
    "FmhaBwdSpec",
    "build_fmha_bwd",
    "fmha_bwd_grid",
    "fmha_bwd_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaBwdSpec:
    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "rocke_fmha_bwd"
    # P69: when True, the kernel uses the new
    # :mod:`rocke.helpers.mfma_attention_bwd` MFMA-tiled body
    # (``mfma_attention_bwd_dq_dk_dv_inner_body``) instead of the
    # warp-distributed scalar inner. Same parity contract; ~32× density
    # for the QK / dP MFMAs once the kernel body wires it up.
    use_mfma_body: bool = False
    # P70: ``output_grad_dtype`` selects the gradient atomic accumulator
    # dtype. ``"f32"`` (default) routes through
    # ``global_atomic_add_f32``; ``"bf16"`` routes through
    # ``global_atomic_add_pk_bf16`` for halved atomic engine pressure.
    # The bf16 path is a real numerical change so callers gate on
    # parity; H=256 workloads that are atomic-engine-bound see the
    # biggest relative improvement.
    output_grad_dtype: str = "f32"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaBwdSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one FMHA backward config on ``arch``.

    The backward body is warp-distributed scalar (one wave64 warp per
    CTA, head-dim distributed across lanes): it issues no MFMA atoms and
    accumulates dQ / dK / dV via f32 global atomics, both of which are
    shared CDNA3 / CDNA4 hardware. The only architecture facts consulted
    are the per-WG thread cap and the wave size (the warp body needs
    ``head_size % wave_size == 0``), sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch or an
    impossible head-size / wave-size combination is rejected with a
    structured reason rather than failing later at lower time. gfx950
    behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.common.shape.head_size % target.wave_size != 0:
        return False, (
            f"fmha_bwd warp body needs head_size % {target.wave_size} == 0 "
            f"(got {spec.common.shape.head_size}) on {arch}"
        )
    if target.wave_size > target.max_threads_per_block:
        return False, (
            f"wave_size {target.wave_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the bwd kernel ABI (shared between build + sig).

    dQ / dK / dV are f32 accumulator tensors targeted by atomic_add;
    their head stride is implicit (== K / V head stride for dK / dV,
    == Q head stride for dQ) so the ABI only carries the *token*
    stride for each gradient tensor.
    """
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("M_saved", dtype="f32", readonly=True)
    kb.add_ptr("L_saved", dtype="f32", readonly=True)
    kb.add_ptr("dQ", dtype="f32", readonly=False)
    kb.add_ptr("dK", dtype="f32", readonly=False)
    kb.add_ptr("dV", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dq_token", "i32")
    kb.add_scalar("stride_dk_token", "i32")
    kb.add_scalar("stride_dv_token", "i32")


def build_fmha_bwd(spec: FmhaBwdSpec, arch: str = "gfx950") -> KernelDef:
    """FMHA backward kernel (warp-distributed, with mask support).

    Per CTA (one warp): lane ``t`` owns head-dim slot ``t * EPT + k``.

    1. Pre-load Q[q_token, head, lane_slice] / dO[..., lane_slice].
    2. Load saved m, l for this row.
    3. **First K-loop**: per-lane partial Q.K + dO.V, warp-reduce,
       apply mask, accumulate rowsum_dp via iter_arg.
    4. **Second K-loop**: recompute p, dp = p * (dO.V - rowsum_dp);
       atomic-add dV[k, head, lane_d] += p * dO; atomic-add dK[k, head,
       lane_d] += dp * scale * Q; per-lane register dq slice updates.
    5. Final: atomic-add per-lane dq into dQ[q_token, head, lane_d].

    The body is warp-distributed scalar (f32 global atomics + warp
    butterfly reductions) and issues **no** MFMA atoms, so it is
    arch-portable: ``arch`` is threaded through :func:`is_valid_spec`
    only to validate the per-WG thread cap / wave-size facts against
    :class:`rocke.core.arch.ArchTarget`. The emitted IR is identical
    on gfx942 and gfx950 (default ``arch="gfx950"`` is byte-for-byte
    backward-compatible).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_bwd spec: {why}")
    s = spec.common
    dtype = s.dtype
    H = s.shape.head_size
    if H % WARP_SIZE != 0:
        raise ValueError(
            f"fmha_bwd warp body needs head_size % {WARP_SIZE} == 0; got {H}"
        )
    ept = H // WARP_SIZE

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(WARP_SIZE)
    _declare_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    dO = kb.tensor("dO")
    M_saved = kb.ptr("M_saved")
    L_saved = kb.ptr("L_saved")
    dQ = kb.ptr("dQ")
    dK = kb.ptr("dK")
    dV = kb.ptr("dV")
    scale_log2 = kb.scalar("scale_log2")
    scale_inv = kb.scalar("scale_inv")
    seqlen_k = kb.scalar("seqlen_k")
    q_token = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx

    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))

    q_row = kb.q_row_base()
    do_row = kb.row_base("do", q_token, head_idx)

    # Saved M / L are indexed by (q_token, head) -- linear (q * HQ + h).
    # Hoist the ``q * HQ + h`` row index so we don't rematerialise the
    # ``HQ`` constant twice.
    ml_row = b.add(b.mul(q_token, b.const_i32(s.shape.num_query_heads)), head_idx)
    m_qt = b.global_load_f32(M_saved, ml_row)
    l_qt = b.global_load_f32(L_saved, ml_row)
    inv_l = b.rcp(l_qt)

    # Pre-load Q + dO lane slices as one vectorised global load per
    # tensor (``global_load_vN`` when EPT in {2, 4, 8}, scalar fallback
    # otherwise). Mirrors CK Tile's ``load_tile`` over a distributed
    # ``rt<bf16, 16, D>`` register tile -- the per-CTA Q / dO buffers
    # are constant over the K-loop, so the loads happen once per CTA
    # and the results live in registers.
    q_lane = load_lane_slice_f32(b, Q, q_row, lane_d_base, dtype=dtype, ept=ept)
    do_lane = load_lane_slice_f32(b, dO, do_row, lane_d_base, dtype=dtype, ept=ept)

    zero_f = b.const_f32(0.0)

    # First K-loop: rowsum_dp accumulated via loop-carried iter_arg.
    k_loop_1 = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=[("rs", zero_f)],
        iv_name="k_idx",
    )
    with k_loop_1 as (k_idx, (rowsum_carry,)):
        k_row = kb.k_row_base(k_idx)
        v_row = kb.v_row_base(k_idx)
        # Vectorised per-lane K / V loads (one transaction each when
        # EPT in {2, 4, 8}); the QK + dO.V partial accumulators stay
        # scalar f32 because the warp-distributed body uses a
        # ``warp_xor_reduce_sum`` butterfly to fold across the warp.
        k_lane = load_lane_slice_f32(b, K, k_row, lane_d_base, dtype=dtype, ept=ept)
        v_lane = load_lane_slice_f32(b, V, v_row, lane_d_base, dtype=dtype, ept=ept)
        partial_qk = zero_f
        partial_dov = zero_f
        for k in range(ept):
            # ``fma`` guarantees ``v_fma_f32`` (single VALU + single
            # rounding) for the per-lane dot partials; the bare
            # fadd(fmul()) only fuses opportunistically. Strictly more
            # accurate (one rounding) and one fewer issue slot per term.
            partial_qk = b.fma(q_lane[k], k_lane[k], partial_qk)
            partial_dov = b.fma(do_lane[k], v_lane[k], partial_dov)
        dot_qk = warp_xor_reduce_sum(b, partial_qk, stages=6)
        dot_dov = warp_xor_reduce_sum(b, partial_dov, stages=6)
        s_log2 = b.fmul(dot_qk, scale_log2)
        s_log2 = apply_attention_mask(
            b,
            s_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )
        p = b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l)
        b.scf_yield(b.fma(p, dot_dov, rowsum_carry))
    rowsum_dp = k_loop_1.results[0]

    # Second K-loop: atomic-add dV / dK; accumulate dQ in per-lane regs.
    iter_args2 = [(f"dq{k}", zero_f) for k in range(ept)]
    k_loop_2 = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=iter_args2,
        iv_name="k_idx2",
    )
    with k_loop_2 as (k_idx, dq_state):
        k_row = kb.k_row_base(k_idx)
        v_row = kb.v_row_base(k_idx)
        dk_row = b.add(
            b.mul(k_idx, kb.scalar("stride_dk_token")),
            b.mul(kv_head_idx, kb.stride_head("k")),
        )
        dv_row = b.add(
            b.mul(k_idx, kb.scalar("stride_dv_token")),
            b.mul(kv_head_idx, kb.stride_head("v")),
        )
        # Vectorised K / V loads. The QK and dO.V partials reuse the
        # same lane slice the first K-loop's pattern uses; the second
        # K-loop additionally feeds dK = dp * scale * Q (no extra K
        # touch) and the dQ accumulator update (re-uses K once per
        # K-step). One ``global_load_vN`` per tensor keeps the inner
        # loop's address-arithmetic / MEM-channel pressure as low as
        # the CK Tile ``BlockFmhaBwd*`` register-tile loaders.
        k_lane = load_lane_slice_f32(b, K, k_row, lane_d_base, dtype=dtype, ept=ept)
        v_lane = load_lane_slice_f32(b, V, v_row, lane_d_base, dtype=dtype, ept=ept)
        partial_qk = zero_f
        partial_dov = zero_f
        for k in range(ept):
            partial_qk = b.fma(q_lane[k], k_lane[k], partial_qk)
            partial_dov = b.fma(do_lane[k], v_lane[k], partial_dov)
        dot_qk = warp_xor_reduce_sum(b, partial_qk, stages=6)
        dot_dov = warp_xor_reduce_sum(b, partial_dov, stages=6)
        s_log2 = b.fmul(dot_qk, scale_log2)
        s_log2 = apply_attention_mask(
            b,
            s_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )
        p = b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l)
        dp = b.fmul(p, b.fsub(dot_dov, rowsum_dp))
        dp_scale = b.fmul(dp, scale_inv)

        # f32 atomic adds remain scalar (no packed-f32 atomic exists
        # on gfx950; ``global_atomic_add_pk_bf16`` would help only if
        # the ABI lowered dV/dK/dQ from f32 to bf16). One scalar
        # atomic per (lane, head-dim slot) matches what CK Tile's
        # bwd kernel emits when its accumulator dtype is f32.
        new_dq = []
        for k in range(ept):
            d = b.add(lane_d_base, b.const_i32(k))
            b.global_atomic_add(dV, b.add(dv_row, d), b.fmul(p, do_lane[k]))
            b.global_atomic_add(dK, b.add(dk_row, d), b.fmul(dp_scale, q_lane[k]))
            # fma for the per-lane dQ register accumulator (loop-carried):
            # guaranteed single-rounding v_fma_f32, one fewer issue slot.
            new_dq.append(b.fma(dp_scale, k_lane[k], dq_state[k]))
        b.scf_yield(*new_dq)

    # Final: write dQ. Only this CTA writes dQ[q_token, head, :].
    dq_row = b.add(
        b.mul(q_token, kb.scalar("stride_dq_token")),
        b.mul(head_idx, kb.stride_head("q")),
    )
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        b.global_atomic_add(dQ, b.add(dq_row, d), k_loop_2.results[k])

    b.ret()
    return kb.kernel


def fmha_bwd_grid(spec: FmhaBwdSpec) -> Tuple[int, int, int]:
    return (spec.seqlen_q, spec.common.shape.num_query_heads, 1)


def fmha_bwd_signature(spec: FmhaBwdSpec):
    kb = FmhaKernelBuilder("rocke_fmha_bwd_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()
