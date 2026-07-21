# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Transposed-QK WMMA FMHA-forward kernel for gfx1151 (CK gfx11 `qr_ks_vs` design).

This is the structural change the register-transpose investigation pointed at. The
gather winner (``fmha_multiwave``) computes ``S = Q*K^T`` (query on the accumulator
slots, kv on the lane), which forces (a) a cross-lane 16-lane butterfly softmax and
(b) an LDS round-trip P-transpose every K-tile (the fixed WMMA ``a_map`` needs a full
16-lane gather that ``permlanex16`` cannot do -- the documented ``p_xpose="shuffle"``
dead-end).

Computing the scores **transposed** ``S^T = K*Q^T`` flips both:

  * **query lands on the lane** (``col = lane%16``), kv on the 8 accumulator slots
    (+ the ``lane^16`` half). So the online softmax reduction over kv is an **in-lane
    reduce over 8 slots + ONE ``permlanex16`` cross-half exchange** -- no 16-lane
    butterfly. The running ``m``/``l`` are scalars per lane (one query per lane).
  * the C->operand P-transpose becomes CK's exact ``PermuteWarpGemmCToA``: one
    ``permlanex16`` + two ``v_perm_b32`` per u32, **NO LDS round-trip, no barrier**.
    It works here (unlike on ``S=Q*K^T``) precisely because query is already on the
    lane -- only kv needs the ``lane^16`` reshuffle.

PV is computed **transposed too**: ``O^T = V*P`` (V is the A operand, P the B
operand). That keeps the PV output ``O^T[d, query]`` with query on the lane -- the
SAME distribution as the softmax stats -- so the online rescale ``O *= alpha`` is a
trivial in-lane vector-mul (no cross-lane alpha redistribution). The cost is a
transposed O store in the epilogue (strided in d), paid once per q-block.

V is still the cache-resident column **gather** (gfx1151 has no ``ds_read_tr``); the
transpose here is on P (registers), not V. Ref: CK
``ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp::PermuteWarpGemmCToA`` and
``block_fmha_pipeline_qr_ks_vs.hpp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import (
    F16,
    F32,
    I16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
)
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

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"
_BLOCK_K = 16


@dataclass(frozen=True)
class SwapQKCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    # Swept defaults on gfx1151 (H24 B1 D128): block_n=32, qk_ilp=2, pingpong.
    # n_waves=2 is the ROBUST choice: best at L512 (18.8 TF) and dominant in the
    # memory-bound regime (L4096 14.1 vs w1 7.3; L8192 7.3 vs w1 2.3 -- more waves
    # = more DRAM MLP), only ~3% behind w1 at L2048 (19.0 vs 19.5). +11-15% over
    # the gather winner in the compute-bound regime.
    n_waves: int = 2
    waves_per_eu: Optional[int] = None
    sched_mode: str = "pingpong"  # "none" | "pingpong"
    qk_ilp: int = 2
    # block_n: kv tile processed per K-loop iteration (multiple of 16). Larger
    # block_n does block_n/16 QK+PV WMMA sub-steps per iteration and rescales the
    # O accumulator only ONCE per block_n keys (vs once per 16) -- amortizes the
    # online-softmax fixed cost + adds WMMA ILP, at the price of block_n/16 live
    # score/P fragments. seqlen_k must be a multiple of block_n.
    block_n: int = 32
    # prefetch_v: software-pipeline the PV V-gather so the strided loads are
    # hidden behind compute. The first fragment's gather is issued BEFORE the
    # softmax (overlaps the softmax VALU); every subsequent (ns,d) fragment is
    # gathered ONE PV-step ahead of the WMMA that consumes it, so its 16 scalar
    # loads stay in flight while the previous step's WMMA runs (the vmcnt wait
    # lands a full step later, when the data is already home). Costs +1 live V
    # fragment.
    #
    # MEASURED DEAD-END (gfx1151, H24 B1 D128): prefetch_v=True REGRESSES at every
    # shape. Compute-bound (L<=2048): the gather is L1-resident, so the AMDGPU
    # backend already hides it; the extra live fragment only adds spills (8->12)
    # -> -8..-10%. Memory-bound (L>=4096): the bottleneck is DRAM MLP, which is
    # hidden by MORE WAVES (n_waves=2 >> n_waves=1 at long L), not by manual
    # prefetch (~neutral to worse). Kept as a lever for the A/B record; OFF by
    # default.
    prefetch_v: bool = False
    # static_shape: bake the Q/K/V/O strides (all pure functions of heads/head_size,
    # known at build time) as compile-time CONSTANTS instead of runtime params.
    # The gather address math is base + kv*stride_v_token repeated 16x per fragment;
    # with a runtime stride the backend emits an s_mul_i32/v_mul_lo_u32 + 64-bit
    # v_lshlrev/v_add_co per element (the ISA showed ~40 s_mul_i32 by the stride reg
    # + 36x v_lshlrev_b64/v_add_co in the K-loop). Constant strides fold those into
    # scaled immediates / a single shift. The params are still declared (ABI/driver
    # packing unchanged) -- just ignored for addressing.
    #
    # MEASURED DEAD-END: static_shape=True REGRESSES (-9%, 18.97->17.26 @ L2048).
    # The per-kv offset kv*stride_v*2 (~6 KB steps) exceeds the global_load
    # immediate-offset range (+/-4 KB), so constant strides don't fold into scaled
    # immediates -- LLVM instead materializes each address with an explicit 64-bit
    # v_add_co (163 vs 36). The runtime-stride path shares the stride multiply in
    # SALU and is cheaper. Kept as a lever; OFF by default.
    static_shape: bool = False
    # buffer_gather: issue the PV V-gather via a buffer descriptor + the D16
    # half-return load (buffer_load_f16_d16 -> raw.ptr.buffer.load.f16). Address =
    # base + voffset + soffset is computed in the MEMORY UNIT (no 64-bit address
    # VALU), and returning `half` directly (not i16+bitcast) keeps the load clause
    # batched (s_clause 121->38) so the loads pipeline like the flat d16 path.
    # MEASURED +2..12% (avg ~+7%, ~19.2-19.5 TF, peak 21.7) over flat @ L2048,
    # bit-identical (1.53e-5).
    #   FRAGILE: only wins at n_waves=2 AND block_n>=32. At block_n=16 or
    #   n_waves=1 the backend stops batching the buffer loads and it collapses to
    #   ~8 TF. Enabled by default because the shipped defaults ARE w2/bn32; the
    #   sweep will still expose the bad off-sweet-spot points.
    buffer_gather: bool = True
    name: str = "wmma_fmha_swapqk"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32 * self.n_waves

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.n_waves

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"w{self.n_waves}",
            f"vpe{self.waves_per_eu}" if self.waves_per_eu is not None else "vpedef",
            self.sched_mode,
            f"ilp{self.qk_ilp}",
            f"bn{self.block_n}",
            "pfv" if self.prefetch_v else "npf",
            "stat" if self.static_shape else "dyn",
            "buf" if self.buffer_gather else "flat",
        )


def swapqk_grid(cfg: SwapQKCfg, *, seqlen_q: int, batch: int):
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per, cfg.num_query_heads, batch)


def _declare_params(b: IRBuilder):
    P = {}
    P["Q"] = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["K"] = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["V"] = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["O"] = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )
    P["scale_log2"] = b.param("scale_log2", F32)
    P["seqlen_q"] = b.param("seqlen_q", I32)
    P["seqlen_k"] = b.param("seqlen_k", I32)
    for nm in (
        "stride_q_token",
        "stride_q_head",
        "stride_k_token",
        "stride_k_head",
        "stride_v_token",
        "stride_v_head",
        "stride_o_token",
        "stride_o_head",
    ):
        P[nm] = b.param(nm, I32)
    return P


def build_wmma_fmha_swapqk(cfg: SwapQKCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    c_map = atom.c_layout(arch)
    c_frag = atom.c_per_lane  # 8
    a_frag = atom.a_per_lane  # 16
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.n_waves
    dtype_ir = F16

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.block_size
    if cfg.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = cfg.waves_per_eu
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    col = b.mod(lane, c16)  # lane % 16  == query row within the 16-tile
    lane_lt16 = b.cmp_lt(lane, c16)

    q_group = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()

    qh, kvh = cfg.num_query_heads, cfg.kv_heads
    kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    if cfg.static_shape:
        # strides are pure functions of the (build-time) head config -> constants.
        # Layout [.., token, head, dim]: token stride = n_heads*D, head stride = D.
        sq, sqh = b.const_i32(qh * hs), b.const_i32(hs)
        sk, skh = b.const_i32(kvh * hs), b.const_i32(hs)
        sv, svh = b.const_i32(kvh * hs), b.const_i32(hs)
        so, soh = b.const_i32(qh * hs), b.const_i32(hs)
    else:
        sq, sqh = p["stride_q_token"], p["stride_q_head"]
        sk, skh = p["stride_k_token"], p["stride_k_head"]
        sv, svh = p["stride_v_token"], p["stride_v_head"]
        so, soh = p["stride_o_token"], p["stride_o_head"]
    scale_log2 = p["scale_log2"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # Q/K: (head, token, dim), dim contiguous -> WMMA operands are contiguous d-slices.
    Q_view = make_global_view(
        Q, shape=(qh, 1, hs), dtype=dtype_ir, strides=(sqh, sq, 1)
    )
    K_view = make_global_view(
        K, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(skh, sk, 1)
    )
    V_view = make_global_view(
        V, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(svh, sv, 1)
    )
    # O^T view: (head, dim, token) so store_wmma_tile's (row=d, col=query) lands on
    # O[query, d]. dim is contiguous (stride 1), token strided (so).
    O_T_view = make_global_view(
        O, shape=(qh, hs, 1), dtype=dtype_ir, strides=(soh, 1, so)
    )

    q_rows_per_cta = b.const_i32(cfg.q_rows_per_cta)
    cta_row0 = b.mul(q_group, q_rows_per_cta)
    wave_row0 = b.add(cta_row0, b.mul(wave_id, c16))
    batch_tok_q = b.mul(batch, seqlen_q)
    batch_tok_k = b.mul(batch, seqlen_k)
    q_pos_base = wave_row0
    q_token_base = b.add(wave_row0, batch_tok_q)

    # Q window is loop-invariant (this wave's 16 query rows); loaded as the B operand.
    qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base, c0))

    pingpong = cfg.sched_mode == "pingpong"

    # ---- CK PermuteWarpGemmCToA: C-dist P (8 f16) -> operand fragment (16 f16). ----
    # query already sits on lane%16, so only kv needs the lane^16 reshuffle + 2x2
    # f16 interleave. Byte selectors are swapped for the upper 16 lanes (CK trick).
    sel0 = b.select(lane_lt16, b.const_i32(0x05040100), b.const_i32(0x01000504))
    sel1 = b.select(lane_lt16, b.const_i32(0x07060302), b.const_i32(0x03020706))

    def permx16_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def p_transpose_reg(ps):
        outs = []
        for m in range(c_frag // 2):
            lo = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m], dtype_ir), I16), I32)
            hi = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m + 1], dtype_ir), I16), I32)
            v = b.lor(lo, b.shl(hi, c16))  # {kv 4m | kv 4m+2}  (own parity)
            w = b.permlanex16(v)  # partner (lane^16): other kv parity
            outs.append(b.perm_b32(w, v, sel0))  # {kv 4m,   4m+1}
            outs.append(b.perm_b32(w, v, sel1))  # {kv 4m+2, 4m+3}
        packed = b.vec_pack(outs, I32)
        return b.vec_bitcast(packed, VectorType(dtype_ir, a_frag))

    # ---- iter-args: m (scalar) | l (scalar) | acc (n_dk O^T tiles) ----
    iter_args = [("m", neg_inf), ("l", zero_f)]
    for d in range(n_dk):
        iter_args.append((f"acc{d}", atom.zero_acc(b)))

    def unpack(state):
        m_i = state[0]
        l_i = state[1]
        accs = [WmmaTensor(atom, "c", v, arch) for v in state[2 : 2 + n_dk]]
        return m_i, l_i, accs

    block_n = cfg.block_n
    n_kv_sub = block_n // 16  # 16-wide kv WMMA sub-tiles per K-loop iteration
    c_block_n = b.const_i32(block_n)
    loop_stop = b.div(seqlen_k, c_block_n)
    if cfg.mask_mode == "causal":
        # CTA owns q rows up to cta_row0 + 16*W - 1; a kv block kt is needed iff
        # kt*block_n <= max q pos. Round up + 1 (over-inclusion is masked, safe).
        causal_stop = b.add(
            b.div(b.add(cta_row0, b.const_i32(16 * W)), c_block_n), b.const_i32(1)
        )
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    def k_window(k_tile_base):
        return make_tile_window(
            K_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def v_window(k_tile_base):
        return make_tile_window(
            V_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def gather_v_a_frag(vwin, d):
        # PV A-operand V[kv=0..15, d_col] (column gather, cache-resident).
        d_col = b.add(b.const_i32(d * 16), col)
        v_a = b.zero_vec(dtype_ir, a_frag)
        for j in range(a_frag):
            v_a = b.vec_insert(v_a, vwin.load_scalar(b, c0, b.const_i32(j), d_col), j)
        return v_a

    # ---- buffer-descriptor gather (address in the memory unit, no VALU) ----
    c2 = b.const_i32(2)
    if cfg.buffer_gather:
        v_rsrc = b.buffer_rsrc(V, b.const_i32(0x7FFFFFFF))
        sv2 = b.mul(sv, c2)  # bytes per kv step (loop-invariant)
        soff_list = [b.mul(b.const_i32(j), sv2) for j in range(a_frag)]  # hoisted
        kvh_off = b.mul(kv_head, svh)  # element base for this (kv) head, per-CTA

    def gather_v_a_frag_buf(k_base, d):
        # k_base = batch_tok_k + k_block_base + ns*16 (uniform i32). Per-lane
        # voffset selects V[kv=0, d_col]; the buffer HW adds soffset = kv*stride_v.
        d_col = b.add(b.const_i32(d * 16), col)
        elem0 = b.add(b.add(kvh_off, b.mul(k_base, sv)), d_col)
        voff = b.mul(elem0, c2)  # bytes, per-lane
        v_a = b.zero_vec(dtype_ir, a_frag)
        for j in range(a_frag):
            # D16 half-return load -> backend packs lo/hi (buffer_load_short_d16
            # /_d16_hi) like the flat global_load_d16 path, no v_mov_b16 pack.
            v_a = b.vec_insert(
                v_a, b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j]), j
            )
        return v_a

    def _tree(vals, op):
        # log-depth reduction (shorter loop-carried m/l critical path).
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        m_i, l_i, accs = unpack(state)
        k_block_base = b.mul(kt, c_block_n)
        ilp = max(1, cfg.qk_ilp)

        # ---- QK for each 16-kv sub-tile: S^T = K @ Q^T (A=K rows=kv, B=Q cols=query) ----
        if pingpong:
            b.s_setprio(1)
        sub_scores = []
        for ns in range(n_kv_sub):
            kwin = k_window(b.add(k_block_base, b.const_i32(ns * 16)))
            acc_ilp = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
            for d in range(n_dk):
                k_tile = load_wmma_tile(
                    b, kwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )
                q_tile = load_wmma_tile(
                    b, qwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                )
                acc_ilp[d % ilp] = wmma_mma(b, k_tile, q_tile, acc_ilp[d % ilp])
            sc = acc_ilp[0]
            for si in range(1, ilp):
                sc = WmmaTensor(
                    atom, "c", b.vector_add(sc.value, acc_ilp[si].value), arch
                )
            sub_scores.append(sc)
        if pingpong:
            b.s_setprio(0)

        # ---- flattened PV step order + V gather dispatch (flat window | buffer SRD) ----
        vwins = [
            v_window(b.add(k_block_base, b.const_i32(ns * 16)))
            for ns in range(n_kv_sub)
        ]
        k_bases = [
            b.add(b.add(batch_tok_k, k_block_base), b.const_i32(ns * 16))
            for ns in range(n_kv_sub)
        ]

        def do_gather(ns, d):
            if cfg.buffer_gather:
                return gather_v_a_frag_buf(k_bases[ns], d)
            return gather_v_a_frag(vwins[ns], d)

        pv_steps = [(ns, d) for ns in range(n_kv_sub) for d in range(n_dk)]

        # PREFETCH: issue the first V gather BEFORE the softmax so its strided
        # loads overlap the softmax VALU (memory<->VALU overlap).
        v_next = None
        if cfg.prefetch_v:
            ns0, d0 = pv_steps[0]
            v_next = do_gather(ns0, d0)

        # ---- online softmax over ALL block_n keys (n_kv_sub*8 in-lane slots + 1 permlanex16) ----
        s_sub = []  # s_sub[ns] = list of c_frag scaled+masked scores
        for ns in range(n_kv_sub):
            kv_base = b.add(k_block_base, b.const_i32(ns * 16))
            row = []
            for i in range(c_frag):
                kv_rel, q_rel = sub_scores[ns].coord(b, lane, i)  # (row=kv, col=query)
                s_i = b.fmul(sub_scores[ns].slot(b, i), scale_log2)
                s_i = apply_attention_mask(
                    b,
                    s_i,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(kv_base, kv_rel),
                    query_pos=b.add(q_pos_base, q_rel),
                    sliding_window=0,
                )
                row.append(s_i)
            s_sub.append(row)

        all_s = [v for row in s_sub for v in row]
        local_max = _tree(list(all_s), b.fmax)
        tile_max = b.fmax(local_max, permx16_f32(local_max))
        m_new = b.fmax(m_i, tile_max)
        alpha = b.exp2(b.fsub(m_i, m_new))
        ps_sub = [
            [b.exp2(b.fsub(s_sub[ns][i], m_new)) for i in range(c_frag)]
            for ns in range(n_kv_sub)
        ]
        all_p = [v for row in ps_sub for v in row]
        local_sum = _tree(list(all_p), b.fadd)
        tile_sum = b.fadd(local_sum, permx16_f32(local_sum))
        l_new = b.fadd(b.fmul(l_i, alpha), tile_sum)

        # rescale the O^T accumulators by alpha ONCE per block_n keys.
        alpha_vec = b.zero_vec_f32(c_frag)
        for i in range(c_frag):
            alpha_vec = b.vec_insert(alpha_vec, alpha, i)
        new_accs = [accs[d].scale(b, alpha_vec) for d in range(n_dk)]

        # ---- PV: O^T += V @ P for each kv sub-tile (register P-transpose, no LDS) ----
        p_tiles = [
            WmmaTensor(atom, "b", p_transpose_reg(ps_sub[ns]), arch)
            for ns in range(n_kv_sub)
        ]
        if pingpong:
            b.s_setprio(1)
        if cfg.prefetch_v:
            # one-step-ahead prefetch: gather (idx+1) while WMMA(idx) runs.
            for idx, (ns, d) in enumerate(pv_steps):
                v_cur = v_next
                if idx + 1 < len(pv_steps):
                    n1, d1 = pv_steps[idx + 1]
                    v_next = do_gather(n1, d1)
                v_tile = WmmaTensor(atom, "a", v_cur, arch)
                new_accs[d] = wmma_mma(b, v_tile, p_tiles[ns], new_accs[d])
        else:
            for ns, d in pv_steps:
                v_tile = WmmaTensor(atom, "a", do_gather(ns, d), arch)
                new_accs[d] = wmma_mma(b, v_tile, p_tiles[ns], new_accs[d])
        if pingpong:
            b.s_setprio(0)

        b.scf_yield(m_new, l_new, *[a.value for a in new_accs])

    m_f, l_f, accs_f = unpack(kloop.results)

    # ---- Epilogue: O^T[d, query] -> O[query, d], rescaled by 1/l. ----
    l_safe = l_f
    zmask = b.fcmp("oeq", l_safe, zero_f)
    inv_l = b.select(zmask, zero_f, b.rcp(l_safe))

    def _rescale(bld, val, slot, row, colv, _inv=inv_l):
        return bld.fmul(val, _inv)

    for d in range(n_dk):
        owin = make_tile_window(
            O_T_view, (1, 16, 16), origin=(head, b.const_i32(d * 16), q_token_base)
        )
        store_wmma_tile(
            b,
            owin,
            accs_f[d],
            lane,
            col_offset=0,
            lead=[c0],
            align=2,
            transform=_rescale,
        )
    b.ret()
    return b.kernel
