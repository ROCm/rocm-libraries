# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Transposed-QK WMMA FMHA-forward inner body for gfx1151 (CK gfx11 ``qr_ks_vs``
design), FP16-only, dense, original (row-major ``[B, S, H, D]``) V layout.

Computes the scores **transposed**: ``S^T = K @ Q^T`` instead of ``S = Q @
K^T``. That puts the query on the lane (``col = lane % 16``) and the kv
position on the accumulator slots, so the online-softmax reduction over kv is
an in-lane reduce over the 8 accumulator slots plus one ``permlanex16``
cross-half exchange (no 16-lane butterfly), and the running ``m``/``l`` are
plain per-lane scalars. The C-operand transpose needed to feed P into the PV
step becomes CK's ``PermuteWarpGemmCToA``: one ``permlanex16`` + two
``v_perm_b32`` per packed dword -- a register transpose with no LDS round trip
and no barrier (it works here, unlike on ``S = Q @ K^T``, precisely because
the query is already on the lane).

PV is computed transposed too: ``O^T = V @ P`` (V is the WMMA A operand, P the
B operand), which keeps the PV output ``O^T[d, query]`` in the same
per-lane distribution as the softmax stats, so the online rescale is a
trivial in-lane vector multiply. The V gather stays column-major (this
module's V is the ``[B, S, H, D]`` row-major layout, not the pre-transposed
``[B, H, D, S]`` form): each 16-key A-fragment is filled via 16
buffer-descriptor D16 half-return loads, and the dual-subtile broadcast
(``permlanex16`` + select) fills both WMMA lane-halves from one such gather so
adjacent head-dim subtiles share the load traffic.

This is a narrow specialization -- only the single fixed configuration that
was validated end-to-end: ``head_size`` in ``{64, 128}``, one query tile per
wave (no query-blocking), ``mask_mode`` in ``{"none", "causal"}``, the
d-outer QK loop, lazy online-softmax rescale, and the raw (fast) ``exp2``.
``block_n`` (kv tile size) and ``n_waves`` are the only tunables; every other
lever from the wider exploration (V pre-transpose, LDS-staged K/V, software
pipelining, f16 O-carry, persistent launch, ...) is deliberately not exposed.

Ref: CK ``ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp::
PermuteWarpGemmCToA`` and ``block_fmha_pipeline_qr_ks_vs.hpp``.
"""

from __future__ import annotations

from rocke.core.ir import F16, F32, I16, I32, IRBuilder, Value, VectorType
from rocke.helpers import (
    WmmaAtom,
    WmmaTensor,
    load_wmma_tile,
    make_global_view,
    make_tile_window,
    store_wmma_tile,
    wmma_mma,
)
from rocke.helpers.attention import apply_attention_mask

__all__ = ["wmma_swapqk_fwd_inner_body"]

# Lazy-rescale re-anchor threshold in the log2 domain: skip the O/l rescale
# when every lane's (tile_max - m_i) <= this. exp2(8)=256 bounds the
# unnormalized probabilities before their FP16 operand cast.
_LAZY_RESCALE_THRESHOLD = 8.0


def wmma_swapqk_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_tile_base: Value,
    q_pos_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    k_token_offset_elems: Value,
    v_token_offset_elems: Value,
    mask_mode: str,
    block_n: int,
    n_waves: int,
    arch: str = "gfx1151",
) -> None:
    """Emit one transposed-QK WMMA FMHA-forward wave body.

    The caller has already declared the kernel parameters, decoded the
    ``(q_group, head, batch)`` grid ids, and resolved GQA head mapping; this
    function only emits the QK/softmax/PV/epilogue body (including the O
    stores) for the current wave. It does not declare params, decode
    ``block_id_{x,y,z}``, or emit ``ret()``.

    ``q_tile_base`` is this CTA's GLOBAL Q/O group start (batch offset
    included); ``q_pos_base`` is the corresponding sequence-local group start
    (no batch offset, used only for the causal-mask position). This helper
    adds ``wave_id * 16`` to both to get the per-wave row bases.

    ``head_idx`` is the Q head; ``kv_head_idx`` is the already GQA-mapped KV
    head. ``k_token_offset_elems`` / ``v_token_offset_elems`` are the batch's
    element offset into the K / V token axis; this helper adds the KV head
    offset and the per-iteration / per-lane token and head-dim offsets on top.
    All strides are in elements.

    ``mask_mode`` is ``"none"`` or ``"causal"`` (top-left, zero context
    offset). ``block_n`` (32 or 64) and ``n_waves`` (1 or 2) are the only tunables.
    """
    if arch != "gfx1151":
        raise ValueError(f"wmma_swapqk is a gfx1151 (RDNA3.5) kernel; got arch={arch!r}")
    if head_size not in (64, 128):
        raise ValueError(f"wmma_swapqk head_size must be 64 or 128 (got {head_size})")
    if block_n not in (32, 64):
        raise ValueError(f"block_n must be 32 or 64 (got {block_n})")
    if n_waves not in (1, 2):
        raise ValueError(f"n_waves must be 1 or 2 (got {n_waves})")
    if mask_mode not in ("none", "causal"):
        raise ValueError(f"mask_mode must be 'none' or 'causal' (got {mask_mode!r})")

    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    c_frag = atom.c_per_lane  # 8
    a_frag = atom.a_per_lane  # 16
    hs = head_size
    n_dk = hs // 16

    # ---- thread -> (wave, lane, query column) ----
    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    col = b.mod(lane, c16)  # lane % 16 == query row within the 16-tile
    lane_lt16 = b.cmp_lt(lane, c16)

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # Q/K: (head, token, dim), dim contiguous -> WMMA operands are contiguous
    # d-slices. O^T view: (head, dim, token) so store_wmma_tile's (row=d,
    # col=query) lands on O[query, d]; dim is contiguous, token strided.
    Q_view = make_global_view(
        Q, shape=(1, 1, hs), dtype=F16, strides=(stride_q_head, stride_q_token, 1)
    )
    # The leading K-view coordinate is an element offset, not a head index.
    K_view = make_global_view(
        K, shape=(1, 1, hs), dtype=F16, strides=(1, stride_k_token, 1)
    )
    O_T_view = make_global_view(
        O, shape=(1, hs, 1), dtype=F16, strides=(stride_o_head, 1, stride_o_token)
    )

    # This wave's 16-row query tile: sequence-local (for masking) and global
    # (for addressing) row bases.
    wave_row = b.mul(wave_id, c16)
    q_token_base = b.add(q_tile_base, wave_row)
    q_pos_local = b.add(q_pos_base, wave_row)
    qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head_idx, q_token_base, c0))

    # ---- CK PermuteWarpGemmCToA: C-dist P (8 f16) -> operand fragment (16 f16). ----
    # Query already sits on lane%16, so only kv needs the lane^16 reshuffle +
    # 2x2 f16 interleave. Byte selectors are swapped for the upper 16 lanes
    # (the CK trick).
    sel0 = b.select(lane_lt16, b.const_i32(0x05040100), b.const_i32(0x01000504))
    sel1 = b.select(lane_lt16, b.const_i32(0x07060302), b.const_i32(0x03020706))

    # ---- iter-args: m (scalar) | l (scalar) | acc (n_dk O^T f32 tiles) ----
    iter_args = [("m", neg_inf), ("l", zero_f)]
    for d in range(n_dk):
        iter_args.append((f"acc{d}", atom.zero_acc(b)))

    def unpack(state):
        m_i = state[0]
        l_i = state[1]
        acc_raw = list(state[2 : 2 + n_dk])
        return m_i, l_i, acc_raw

    n_kv_sub = block_n // 16  # 16-wide kv WMMA sub-tiles per K-loop iteration
    c_block_n = b.const_i32(block_n)
    loop_stop = b.div(seqlen_k, c_block_n)
    if mask_mode == "causal":
        # CTA owns q rows up to q_pos_base + 16*n_waves - 1; a kv block kt is
        # needed iff kt*block_n <= max q pos. Round up + 1 (over-inclusion is
        # masked, safe).
        causal_stop = b.add(
            b.div(b.add(q_pos_base, b.const_i32(16 * n_waves)), c_block_n), b.const_i32(1)
        )
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    # ---- buffer-descriptor D16 V-gather (address in the memory unit, no VALU) ----
    c2 = b.const_i32(2)
    v_rsrc = b.buffer_rsrc(V, b.const_i32(0x7FFFFFFF))
    sv2 = b.mul(stride_v_token, c2)  # bytes per kv step (loop-invariant)
    soff_list = [b.mul(b.const_i32(j), sv2) for j in range(a_frag)]  # hoisted
    k_base_elems = b.add(k_token_offset_elems, b.mul(kv_head_idx, stride_k_head))
    kvh_off = b.add(v_token_offset_elems, b.mul(kv_head_idx, stride_v_head))

    def k_window(k_tile_base):
        return make_tile_window(
            K_view,
            (1, 16, hs),
            origin=(k_base_elems, k_tile_base, c0),
        )

    def permx16_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def p_transpose_reg(ps):
        outs = []
        for m in range(c_frag // 2):
            lo = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m], F16), I16), I32)
            hi = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m + 1], F16), I16), I32)
            v = b.lor(lo, b.shl(hi, c16))  # {kv 4m | kv 4m+2} (own parity)
            w = b.permlanex16(v)  # partner (lane^16): other kv parity
            outs.append(b.perm_b32(w, v, sel0))  # {kv 4m,   4m+1}
            outs.append(b.perm_b32(w, v, sel1))  # {kv 4m+2, 4m+3}
        packed = b.vec_pack(outs, I32)
        return b.vec_bitcast(packed, VectorType(F16, a_frag))

    def _load_col(k_base, d_col):
        """Gather V[kv=0..15, d_col] (per-lane d_col) via the buffer-descriptor
        D16 half-return load; row-major ``[B, S, H, D]`` V (no host transpose)."""
        elem0 = b.add(b.add(kvh_off, b.mul(k_base, stride_v_token)), d_col)
        voff = b.mul(elem0, c2)
        v_a = b.undef_vec(F16, a_frag)  # fully overwritten by the 16 loads
        for j in range(a_frag):
            v_a = b.vec_insert(v_a, b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j]), j)
        return v_a

    def dual_gather_issue(k_base, d):
        """Issue only the loads for subtiles (d, d+1): lanes 0-15 fetch subtile
        d, lanes 16-31 subtile d+1. Halves the V load count versus a
        lane%16-only gather (the WMMA A-operand's lane^16 duplication would
        otherwise re-issue the same load twice)."""
        d_col = b.add(b.const_i32(d * 16), b.add(b.mul(b.div(lane, c16), c16), col))
        return _load_col(k_base, d_col)

    n_i32 = a_frag // 2  # 8 dwords per <16 x f16> fragment

    def dual_gather_finish(loaded):
        """permlanex16 + select broadcast each subtile into both lane-halves
        (the layout the WMMA A-operand's lane^16 duplication requires)."""
        li = b.vec_bitcast(loaded, VectorType(I32, n_i32))
        fd, fd1 = [], []
        for i in range(n_i32):
            e = b.vec_extract(li, i)
            p = b.permlanex16(e)  # value held by lane^16 (the other subtile)
            fd.append(b.select(lane_lt16, e, p))  # subtile d, both halves
            fd1.append(b.select(lane_lt16, p, e))  # subtile d+1, both halves
        frag_d = b.vec_bitcast(b.vec_pack(fd, I32), VectorType(F16, a_frag))
        frag_d1 = b.vec_bitcast(b.vec_pack(fd1, I32), VectorType(F16, a_frag))
        return frag_d, frag_d1

    def dual_gather(k_base, d):
        return dual_gather_finish(dual_gather_issue(k_base, d))

    def _tree(vals, op):
        # log-depth reduction (shorter loop-carried m/l critical path).
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    def compute_qk(k_block_base):
        """S^T = K @ Q^T for all n_kv_sub sub-tiles -> list of score
        WmmaTensors. d-outer / kv-inner: Q[d] is invariant in kv, so the
        n_kv_sub accumulator chains stay mutually independent (their own ILP)
        and only n_dk Q loads are issued per K-tile."""
        b.s_setprio(1)
        kwins = [k_window(b.add(k_block_base, b.const_i32(ns * 16))) for ns in range(n_kv_sub)]
        subs = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(n_kv_sub)]
        for d in range(n_dk):
            q_tile = load_wmma_tile(b, qwin, atom, lane, role="b", k_offset=d * 16, lead=[c0])
            for ns in range(n_kv_sub):
                k_frag = load_wmma_tile(
                    b, kwins[ns], atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )
                subs[ns] = wmma_mma(b, k_frag, q_tile, subs[ns])
        b.s_setprio(0)
        return subs

    kloop = b.scf_for_iter(c0, loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt")
    with kloop as (kt, state):
        m_i, l_i, accs = unpack(state)
        k_block_base = b.mul(kt, c_block_n)

        # ---- QK: S^T = K @ Q^T ----
        sub_scores = compute_qk(k_block_base)
        b.s_setprio(0)

        k_bases = [b.add(k_block_base, b.const_i32(ns * 16)) for ns in range(n_kv_sub)]

        # ---- online softmax over ALL block_n keys (n_kv_sub*8 in-lane slots
        # + 1 permlanex16) ----
        s_sub = []  # s_sub[ns] = list of c_frag scaled+masked scores
        for ns in range(n_kv_sub):
            kv_base = b.add(k_block_base, b.const_i32(ns * 16))
            row = []
            for i in range(c_frag):
                kv_rel, q_rel = sub_scores[ns].coord(b, lane, i)  # (row=kv, col=query)
                s_i = sub_scores[ns].slot(b, i)
                s_i = b.fmul(s_i, scale_log2)
                s_i = apply_attention_mask(
                    b,
                    s_i,
                    mask_mode=mask_mode,
                    k_idx=b.add(kv_base, kv_rel),
                    query_pos=b.add(q_pos_local, q_rel),
                    sliding_window=0,
                )
                row.append(s_i)
            s_sub.append(row)

        all_s = [v for row in s_sub for v in row]
        local_max = _tree(list(all_s), b.fmax)
        tile_max = b.fmax(local_max, permx16_f32(local_max))
        # lazy: if every lane's tile_max is within threshold of m_i, don't
        # re-anchor (m_new = m_i -> alpha = 1) and skip the O rescale below.
        below = b.select(
            b.fcmp("ole", b.fsub(tile_max, m_i), b.const_f32(_LAZY_RESCALE_THRESHOLD)),
            b.const_i32(1),
            c0,
        )
        skip_rescale = b.cmp_ne(b.wave_all(below), c0)  # wave-uniform i1
        m_new = b.select(skip_rescale, m_i, b.fmax(m_i, tile_max))

        alpha = b.exp2_fast(b.fsub(m_i, m_new))
        ps_sub = [
            [b.exp2_fast(b.fsub(s_sub[ns][i], m_new)) for i in range(c_frag)]
            for ns in range(n_kv_sub)
        ]
        all_p = [v for row in ps_sub for v in row]
        local_sum = _tree(list(all_p), b.fadd)
        tile_sum = b.fadd(local_sum, permx16_f32(local_sum))
        l_new = b.fadd(b.fmul(l_i, alpha), tile_sum)

        # ---- alpha (rescale factor) + P operand tiles ----
        alpha_vec = b.zero_vec_f32(c_frag)
        for i in range(c_frag):
            alpha_vec = b.vec_insert(alpha_vec, alpha, i)
        p_tiles = [
            WmmaTensor(atom, "b", p_transpose_reg(ps_sub[ns]), arch) for ns in range(n_kv_sub)
        ]

        # ---- rescale the O^T accumulators by alpha ONCE per block_n keys ----
        accs_wt = [WmmaTensor(atom, "c", v, arch) for v in accs]
        # wave-uniform 0/1-trip loop: run the n_dk rescale muls only when the
        # max re-anchored (skip_rescale False) -> 0-trip skips them.
        n_res = b.select(skip_rescale, c0, b.const_i32(1))
        rloop = b.scf_for_iter(
            c0,
            n_res,
            b.const_i32(1),
            iter_args=[(f"ra{d}", accs_wt[d].value) for d in range(n_dk)],
            iv_name="rsc",
        )
        with rloop as (_rsc, rstate):
            out = [
                WmmaTensor(atom, "c", rstate[d], arch).scale(b, alpha_vec).value
                for d in range(n_dk)
            ]
            b.scf_yield(*out)
        new_accs = [WmmaTensor(atom, "c", v, arch) for v in rloop.results]

        # ---- PV: O^T += V @ P per kv sub-tile (register P-transpose, no LDS) ----
        b.s_setprio(1)
        for ns in range(n_kv_sub):
            for dp in range(0, n_dk, 2):
                frag_d, frag_d1 = dual_gather(k_bases[ns], dp)
                new_accs[dp] = wmma_mma(
                    b, WmmaTensor(atom, "a", frag_d, arch), p_tiles[ns], new_accs[dp]
                )
                new_accs[dp + 1] = wmma_mma(
                    b, WmmaTensor(atom, "a", frag_d1, arch), p_tiles[ns], new_accs[dp + 1]
                )
        b.s_setprio(0)
        new_acc_vals = [a.value for a in new_accs]

        b.scf_yield(m_new, l_new, *new_acc_vals)

    _m_f, l_f, accs_f = unpack(kloop.results)

    # ---- Epilogue: O^T[d, query] -> O[query, d], rescaled by 1/l. ----
    zmask = b.fcmp("oeq", l_f, zero_f)
    inv_l = b.select(zmask, zero_f, b.rcp(l_f))

    def _rescale(bld, val, slot, row, colv):
        return bld.fmul(val, inv_l)

    for d in range(n_dk):
        owin = make_tile_window(
            O_T_view, (1, 16, 16), origin=(head_idx, b.const_i32(d * 16), q_token_base)
        )
        acc_wt = WmmaTensor(atom, "c", accs_f[d], arch)
        store_wmma_tile(b, owin, acc_wt, lane, col_offset=0, lead=[c0], align=2, transform=_rescale)
