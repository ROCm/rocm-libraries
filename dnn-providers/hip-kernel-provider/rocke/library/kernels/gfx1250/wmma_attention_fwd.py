# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dense WMMA FMHA forward for gfx1250 (gfx1250-class CDNA, GFX12 programming model).

This is the gfx1250 analogue of ``instances/gfx1151/wmma_fmha_fwd.py``, but it is
a **standalone** kernel rather than an adapter over the shared
``mfma_attention_fwd_inner_body`` — because the gfx1250 WMMA atom is **16x16x32**
(K=32), the attention tiling is structurally different from the gfx1151 K=16 body:

  * The QK^T score tile spanning one K-loop iteration is **16 q-rows x 32 k**
    (``BLOCK_K = 32``), built from **two** WMMA N-sub-tiles (each op.n=16 wide)
    accumulated over ``head_size // 32`` d-tiles (op.k=32 each).
  * Online softmax runs over the 32 k-columns: the per-row reduction is the
    wave32 16-lane XOR butterfly applied to each N-sub-tile, then the two
    sub-tile partials are combined.
  * **PV** contracts the full 32 k-positions with one K=32 WMMA per d-tile
    (``head_size // 16`` d-tiles). The A-operand (P) is read from a 16x32 LDS
    tile in the gfx1250 A-fragment layout (lane ``l`` holds k = ``(l//16)*16 +
    j``), so lane-half 0 reads P cols 0..15 and half 1 reads cols 16..31.

Fragment physical layouts come from the verified ``wmma_gfx1250_f32_16x16x32_f16``
``MmaOp`` maps (``examples/gfx1250/wmma_probe.py``). Correctness-first: one wave32
per ``(q_tile, head, batch)``, no async DMA / multi-tile pipelining (Phase 2).

Algorithm (BLOCK_M = 16 Q rows, BLOCK_K = 32 K positions per wave32 per CTA):

  * Grid ``(seqlen_q // 16, num_query_heads, batch)``.
  * QK^T: ``S[q,k] = sum_d Q[q,d] * K[k,d]`` via ``A @ B^T`` (A=Q rows x d,
    B=K rows x d). ``head_size`` must be a multiple of 32.
  * Online softmax over 32 k columns, running ``m``/``l`` per q-row carried as
    ``scf.for`` iter-args.
  * PV: ``O[q,d] = sum_k P[q,k] * V[k,d]`` with ``B = V`` in ``d x k`` layout
    (strided V column gather), one K=32 WMMA per ``head_size // 16`` d-tile.
  * Epilogue: ``O = acc / l`` (zero-denominator guarded), f16 store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType

# Experimental direct-LLVM WMMA spacing knob. Default off: fixed v_nop counts did
# not robustly fix the causal NaN across shapes; Phase-1 causal verification uses
# the HIP/clang path instead (see the gfx1250 attention verify example).
_WMMA_SPACING = int(os.environ.get("ROCKE_GFX1250_WMMA_SPACING", "0"))

__all__ = [
    "WmmaAttentionFwdSpec",
    "build_wmma_attention_fwd",
    "wmma_attention_fwd_grid",
    "is_valid_spec",
]

_WMMA_OP_ID = "wmma_gfx1250_f32_16x16x32_f16"
_BLOCK_M = 16  # Q rows per wave per CTA (WMMA M)
_WMMA_N = 16  # k positions per WMMA N-sub-tile
_WMMA_K = 32  # contraction per WMMA step (head-dim for QK, k-positions for PV)
_BLOCK_K = 32  # K positions per K-loop iteration (== two N-sub-tiles == PV K)


@dataclass(frozen=True)
class WmmaAttentionFwdSpec:
    """One gfx1250 WMMA FMHA forward configuration.

    ``head_size`` must be a multiple of 32 (the K=32 WMMA tile); standard FMHA
    head sizes 64 / 128 / 256 qualify. ``seqlen_q`` / ``seqlen_k`` are runtime
    kernel args.
    """

    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0  # 0 -> equal to num_query_heads (MHA)
    dtype: str = "fp16"
    mask_mode: str = "none"  # "none" | "causal"
    sliding_window: int = 0
    name: str = "rocke_wmma_attention_fwd_gfx1250"

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "f16"):
            raise ValueError(
                f"WmmaAttentionFwdSpec currently supports fp16 only, got {self.dtype!r}"
            )
        if self.head_size % _WMMA_K != 0:
            raise ValueError(
                f"head_size must be a multiple of {_WMMA_K}, got {self.head_size}"
            )
        if self.mask_mode not in ("none", "causal"):
            raise ValueError(
                f"WMMA FMHA supports mask_mode 'none'/'causal', got {self.mask_mode!r}"
            )

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32  # one wave32 per block

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            "wmma16x16x32",
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            "fp16",
            self.mask_mode,
        )


def is_valid_spec(
    spec: WmmaAttentionFwdSpec, arch: str = "gfx1250"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)``. The gfx1250 WMMA 16x16x32 f16 atom must exist."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    op = target.mma.by_op_id(_WMMA_OP_ID)
    if op is None or op.family != "wmma":
        return False, (
            f"WMMA {_WMMA_OP_ID} atom absent on {arch} "
            f"(this kernel targets the gfx1250 16x16x32 WMMA)"
        )
    if target.wave_size != op.wave_size:
        return False, (
            f"arch wave size {target.wave_size} != WMMA atom wave size "
            f"{op.wave_size} on {arch}"
        )
    if spec.head_size % _WMMA_K != 0:
        return False, f"head_size must be a multiple of {_WMMA_K}"
    # LDS: one 16x32 f16 P-staging tile.
    bytes_lds = _BLOCK_M * _BLOCK_K * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(b: IRBuilder):
    Q = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    K = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    V = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    O = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )  # noqa: E741
    scale_log2 = b.param("scale_log2", F32)
    seqlen_q = b.param("seqlen_q", I32)
    seqlen_k = b.param("seqlen_k", I32)
    stride_q_token = b.param("stride_q_token", I32)
    stride_q_head = b.param("stride_q_head", I32)
    stride_k_token = b.param("stride_k_token", I32)
    stride_k_head = b.param("stride_k_head", I32)
    stride_v_token = b.param("stride_v_token", I32)
    stride_v_head = b.param("stride_v_head", I32)
    stride_o_token = b.param("stride_o_token", I32)
    stride_o_head = b.param("stride_o_head", I32)
    return locals()


def build_wmma_attention_fwd(
    spec: WmmaAttentionFwdSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build the gfx1250 WMMA FMHA forward ``KernelDef`` (BLOCK_K=32)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wmma_attention_fwd spec: {why}")

    from ...core.arch import ArchTarget
    from ...helpers.attention import (
        apply_attention_mask,
        wave_reduce_max,
        wave_reduce_sum,
    )

    target = ArchTarget.from_gfx(arch)
    op = target.mma.by_op_id(_WMMA_OP_ID)
    wave = op.wave_size  # 32
    a_map = op.a_layout()  # (row, k): lane l, slot j -> (l%16, (l//16)*16 + j)
    c_map = op.c_layout()  # (row, col): lane l, slot i -> ((l//16)*8 + i, l%16)
    a_frag = op.a_frag_len  # 16
    c_frag = op.c_frag_len  # 8

    H = spec.head_size
    n_dk = H // _WMMA_K  # QK^T d-tiles (op.k = 32 each)
    n_pv = H // _WMMA_N  # PV d-tiles (op.n = 16 each)
    dtype_ir = F16

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    p = _declare_params(b)

    c16 = b.const_i32(16)
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    q_tile = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()

    qh, kvh = spec.num_query_heads, spec.kv_heads
    kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    stride_q_token = p["stride_q_token"]
    stride_q_head = p["stride_q_head"]
    stride_k_token = p["stride_k_token"]
    stride_k_head = p["stride_k_head"]
    stride_v_token = p["stride_v_token"]
    stride_v_head = p["stride_v_head"]
    stride_o_token = p["stride_o_token"]
    stride_o_head = p["stride_o_head"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741
    scale_log2 = p["scale_log2"]

    def causal_wmma_spacing():
        if spec.mask_mode == "causal" and _WMMA_SPACING > 0:
            # gfx1250 WMMA co-execution hazard experiment. Direct-LLVM causal
            # attention still NaNs with simple fixed spacing; leave this as an
            # opt-in knob until the backend grows a proper gfx1250 WMMA
            # scheduler. The HIP/clang path is the Phase-1 causal correctness
            # path.
            b.inline_asm("\n".join(["v_nop"] * _WMMA_SPACING), "", result_type=None)

    lane = b.mod(b.thread_id_x(), b.const_i32(wave))
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16 (Q/K/P fragment row)
    col = b.mod(lane, c16)  # lane % 16 (accumulator column / d-column index)
    half_k = a_map.coord(b, lane, 0)[1]  # (l//16)*16 -- this lane's K base

    q_row0 = b.mul(q_tile, c16)
    batch_row_q = b.mul(batch, seqlen_q)
    q_row = b.add(b.add(q_row0, batch_row_q), a_row)
    q_addr_row_base = b.add(b.mul(q_row, stride_q_token), b.mul(head, stride_q_head))

    # Pre-load Q fragments (K-loop invariant): for each QK d-tile, this lane's
    # 16 contiguous d-elements starting at d*32 + half_k.
    q_frags = []
    for d in range(n_dk):
        q_addr = b.add(b.add(q_addr_row_base, b.const_i32(d * _WMMA_K)), half_k)
        q_frags.append(b.global_load_vN(Q, q_addr, dtype_ir, a_frag, align=a_frag * 2))

    P_lds = b.smem_alloc(dtype_ir, [_BLOCK_M, _BLOCK_K], name_hint="Pgfx1250")

    batch_off_k = b.mul(b.mul(batch, seqlen_k), stride_k_token)
    batch_off_v = b.mul(b.mul(batch, seqlen_k), stride_v_token)

    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_pv):
        iter_args.append((f"acc{d}", b.zero_vec_f32(c_frag)))

    c_block_k = b.const_i32(_BLOCK_K)
    loop_stop = b.div(seqlen_k, c_block_k)
    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms = [state[2 * r] for r in range(c_frag)]
        ls = [state[2 * r + 1] for r in range(c_frag)]
        accs = list(state[2 * c_frag :])

        k_tile_base = b.mul(kt, c_block_k)

        # ---- QK^T: two N-sub-tiles (k 0..15 and 16..31), each summed over d ----
        scores = []
        for nsub in range(2):
            score = b.zero_vec_f32(c_frag)
            # This lane's K row for sub-tile nsub: k_tile_base + nsub*16 + (l%16).
            k_row = b.add(b.add(k_tile_base, b.const_i32(nsub * _WMMA_N)), a_row)
            k_addr_row_base = b.add(
                b.add(b.mul(k_row, stride_k_token), b.mul(kv_head, stride_k_head)),
                batch_off_k,
            )
            for d in range(n_dk):
                k_addr = b.add(b.add(k_addr_row_base, b.const_i32(d * _WMMA_K)), half_k)
                k_frag = b.global_load_vN(K, k_addr, dtype_ir, a_frag, align=a_frag * 2)
                score = b.mma(op, q_frags[d], k_frag, score)
                causal_wmma_spacing()
            scores.append(score)

        # ---- scale + mask + online softmax over the 32 k columns ----
        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = [[], []]  # ps[nsub][r] scaled probability (acc layout)
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            row_q_pos = b.add(q_row0, row_rel)
            srs = []
            for nsub in range(2):
                s_r = b.fmul(b.vec_extract(scores[nsub], r), scale_log2)
                k_col_pos = b.add(
                    b.add(k_tile_base, b.const_i32(nsub * _WMMA_N)), col_k
                )
                s_r = apply_attention_mask(
                    b,
                    s_r,
                    mask_mode=spec.mask_mode,
                    k_idx=k_col_pos,
                    query_pos=row_q_pos,
                    sliding_window=spec.sliding_window,
                    context_len=b.const_i32(0),
                )
                srs.append(s_r)
            rm0 = wave_reduce_max(b, srs[0], wave_size=wave, lanes_per_row=16)
            rm1 = wave_reduce_max(b, srs[1], wave_size=wave, lanes_per_row=16)
            row_max = b.fmax(rm0, rm1)
            m_new = b.fmax(ms[r], row_max)
            alpha = b.exp2(b.fsub(ms[r], m_new))
            p0 = b.exp2(b.fsub(srs[0], m_new))
            p1 = b.exp2(b.fsub(srs[1], m_new))
            rs0 = wave_reduce_sum(b, p0, wave_size=wave, lanes_per_row=16)
            rs1 = wave_reduce_sum(b, p1, wave_size=wave, lanes_per_row=16)
            row_sum = b.fadd(rs0, rs1)
            l_new = b.fadd(b.fmul(ls[r], alpha), row_sum)
            new_ms.append(m_new)
            new_ls.append(l_new)
            ps[0].append(p0)
            ps[1].append(p1)
            for d in range(n_pv):
                old = b.vec_extract(new_accs[d], r)
                new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

        # ---- P staging: acc layout -> 16x32 LDS tile ----
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            b.smem_store_vN(
                P_lds, [row_rel, col_k], b.cast_f32_to(ps[0][r], dtype_ir), 1
            )
            b.smem_store_vN(
                P_lds,
                [row_rel, b.add(col_k, c16)],
                b.cast_f32_to(ps[1][r], dtype_ir),
                1,
            )
        b.sync()

        # ---- PV: A=P (gfx1250 a-layout over 32 k), B=V (d x k gather) ----
        p_a = b.zero_vec(dtype_ir, a_frag)
        for j in range(a_frag):
            a_k = a_map.coord(b, lane, j)[1]  # half_k + j (P column to read)
            p_v = b.vec_extract(
                b.smem_load_vN(P_lds, a_row, a_k, dtype=dtype_ir, n=1), 0
            )
            p_a = b.vec_insert(p_a, p_v, j)

        for d in range(n_pv):
            d_col = b.add(b.const_i32(d * _WMMA_N), col)
            v_b = b.zero_vec(dtype_ir, a_frag)
            for j in range(a_frag):
                # B-operand for d-column d_col: V[k = k_tile_base + half_k + j, d_col].
                v_k = a_map.coord(b, lane, j)[1]  # half_k + j
                v_row = b.add(k_tile_base, v_k)
                v_row_base = b.add(
                    b.add(b.mul(v_row, stride_v_token), b.mul(kv_head, stride_v_head)),
                    batch_off_v,
                )
                v_elem = b.global_load(V, b.add(v_row_base, d_col), dtype_ir, align=2)
                v_b = b.vec_insert(v_b, v_elem, j)
            new_accs[d] = b.mma(op, p_a, v_b, new_accs[d])
            causal_wmma_spacing()

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)

    final = kloop.results
    ls_final = [final[2 * r + 1] for r in range(c_frag)]
    accs_final = list(final[2 * c_frag :])

    # ---- Epilogue: O[q,d] = acc / l (zero-denominator guarded) ----
    for d in range(n_pv):
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)
            l_safe = ls_final[r]
            zero_mask = b.fcmp("oeq", l_safe, zero_f)
            inv_l = b.select(zero_mask, zero_f, b.rcp(l_safe))
            v_f32 = b.fmul(b.vec_extract(accs_final[d], r), inv_l)
            o_row = b.add(b.add(q_row0, batch_row_q), row_rel)
            o_col = b.add(b.const_i32(d * _WMMA_N), col_n)
            o_addr = b.add(
                b.add(b.mul(o_row, stride_o_token), b.mul(head, stride_o_head)),
                o_col,
            )
            b.global_store(O, o_addr, b.cast_f32_to(v_f32, dtype_ir), align=2)

    b.ret()
    return b.kernel


def wmma_attention_fwd_grid(spec: WmmaAttentionFwdSpec, *, seqlen_q: int, batch: int):
    """Launch grid ``(seqlen_q // 16, num_query_heads, batch)``."""
    if seqlen_q % _BLOCK_M != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {_BLOCK_M}")
    return (seqlen_q // _BLOCK_M, spec.num_query_heads, batch)
