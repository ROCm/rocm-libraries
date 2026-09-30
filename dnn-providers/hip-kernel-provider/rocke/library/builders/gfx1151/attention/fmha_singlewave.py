# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Heavily-parameterized WMMA FMHA-forward kernel for the gfx1151 case study.

This is the single-wave (block_size=32) winner of the optimization campaign in
this directory. It is built **entirely on the CK Tile helper layer**,
demonstrating that the helpers can drive RDNA/WMMA at a minimal instruction
budget (the kernel is issue-bound: WMMA is ~1% of the static instructions, so any
added address-arithmetic op costs TFLOP/s). The helper primitives it reuses:

  * :func:`~rocke.helpers.make_global_view` + :func:`~rocke.helpers.make_tile_window`
    for the Q/K/V/O addressing — 3D ``(head, token, dim)`` views whose head axis
    carries the per-head stride (which cannot fold into the token stride) and
    whose token axis folds the batch offset.
  * :class:`~rocke.helpers.WmmaAtom` (``wmma_f32_16x16x16_f16``) for the WMMA
    contract, with its hardware-verified ``a_layout``/``b_layout``/``c_layout``
    maps driving every lane decode.
  * :class:`~rocke.helpers.WmmaTensor` — a packed distributed tensor carrying
    one lane's fragment/accumulator as a single SSA vector — together with
    :func:`~rocke.helpers.load_wmma_tile` (one packed ``global_load_dwordx8``
    per ``<16 x half>`` fragment, no f32 cast) and :func:`~rocke.helpers.wmma_mma`
    (one ``b.mma``). The score/accumulator carry as ``WmmaTensor`` tiles, so the
    online-softmax rescale is one ``tile.scale`` (a single ``v_mul``) and the
    per-slot lane decode is ``tile.coord`` off the atom's verified layout map.
  * :func:`~rocke.helpers.make_lds_view` for the P-transpose (and the optional
    V-transpose) LDS staging.
  * :func:`~rocke.helpers.store_wmma_tile` for the O epilogue.
  * :mod:`rocke.helpers.attention` (``wave_reduce_max``/``wave_reduce_sum`` /
    ``apply_attention_mask``) for the online softmax.

Swept levers (unchanged from the campaign):

  * ``bm_tiles`` -- number of 16-row Q-tiles a single wave owns. K and V are
    independent of the query rows, so loading them ONCE per K-tile and feeding
    ``bm_tiles`` QK/PV matmuls amortizes the load/gather traffic. (BM amplification.)
  * ``p_mode`` -- ``"lds"`` round-trips P through LDS to transpose the score
    accumulator layout into the PV A-operand layout.
  * ``v_mode`` -- ``"gather"`` reads the PV V B-operand straight from global;
    ``"lds_t"`` stages V *transposed* in LDS so the column gather becomes a
    vectorized contiguous read.
  * ``prefetch_k`` -- hoist the next K-tile's global loads above the compute.

Build with ``compile_kernel(build_wmma_fmha_singlewave(cfg), arch="gfx1151")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers import (
    WmmaAtom,
    WmmaTensor,
    load_wmma_tile,
    make_global_view,
    make_lds_view,
    make_tile_window,
    store_wmma_tile,
    wmma_mma,
)
from rocke.helpers.attention import (
    apply_attention_mask,
    wave_reduce_max,
    wave_reduce_sum,
)

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"
_BLOCK_K = 16


@dataclass(frozen=True)
class SingleWaveCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    bm_tiles: int = 1  # 16-row Q-tiles per wave
    p_mode: str = "lds"  # "lds" | "shuffle"
    v_mode: str = "gather"  # "gather" | "lds_t"
    prefetch_k: bool = False
    q_preload: bool = False  # hoist Q frags out of K-loop (no win: Q is L1-resident)
    # fuse_k: load each K-frag inside the QK matmul (1 live vs n_dk) -- cuts VGPR
    # spills. Net win ONLY when the kernel spills (head_size>=128); at head_size=64
    # there's no spill so the extra reloads just add latency. None = auto.
    fuse_k: Optional[bool] = None
    name: str = "wmma_fmha_singlewave"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.bm_tiles

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"bm{self.bm_tiles}",
            f"p{self.p_mode}",
            f"v{self.v_mode}",
            "pf" if self.prefetch_k else "npf",
        )


def singlewave_grid(cfg: SingleWaveCfg, *, seqlen_q: int, batch: int):
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


def build_wmma_fmha_singlewave(cfg: SingleWaveCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    a_frag = atom.a_per_lane  # 16
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
    BM = cfg.bm_tiles
    dtype_ir = F16
    # fuse_k auto: only a win when the PV accumulator spills (head_size>=128);
    # at D64 there's no spill so the extra K reloads just add latency.
    fuse_k = cfg.fuse_k if cfg.fuse_k is not None else (cfg.head_size >= 128)

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    lane = b.mod(b.thread_id_x(), b.const_i32(wave))
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16
    col = b.mod(lane, c16)

    q_group = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()

    qh, kvh = cfg.num_query_heads, cfg.kv_heads
    kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    sq = p["stride_q_token"]
    sqh = p["stride_q_head"]
    sk = p["stride_k_token"]
    skh = p["stride_k_head"]
    sv = p["stride_v_token"]
    svh = p["stride_v_head"]
    so = p["stride_o_token"]
    soh = p["stride_o_head"]
    scale_log2 = p["scale_log2"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # ---- CK Tile tensor views: 3D (head, token, dim). The head axis carries the
    # per-head stride (which cannot fold into the token stride); the token axis
    # stride is the per-token stride and the batch offset folds into the token
    # ORIGIN (batch and token share that stride). dim is contiguous (stride 1). ----
    hs = cfg.head_size
    Q_view = make_global_view(
        Q, shape=(qh, 1, hs), dtype=dtype_ir, strides=(sqh, sq, 1)
    )
    K_view = make_global_view(
        K, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(skh, sk, 1)
    )
    V_view = make_global_view(
        V, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(svh, sv, 1)
    )
    O_view = make_global_view(
        O, shape=(qh, 1, hs), dtype=dtype_ir, strides=(soh, so, 1)
    )

    q_rows_per_cta = b.const_i32(cfg.q_rows_per_cta)
    group_row0 = b.mul(q_group, q_rows_per_cta)  # within-batch first q row
    batch_tok_q = b.mul(batch, seqlen_q)  # batch offset folded into token index
    batch_tok_k = b.mul(batch, seqlen_k)

    # Per-tile within-batch q position base (for mask) and global q-token base.
    def q_pos_base(t):
        return b.add(group_row0, b.const_i32(t * 16))

    def q_token_base(t):
        return b.add(b.add(group_row0, b.const_i32(t * 16)), batch_tok_q)

    def q_window(t):
        return make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base(t), c0))

    def o_window(t):
        return make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base(t), c0))

    # ---- LDS staging (CK Tile LDS views) ----
    P_lds = None
    V_lds_t = None
    if cfg.p_mode == "lds":
        P_lds = make_lds_view(b, dtype=dtype_ir, shape=(BM, 16, 16), name_hint="Pwmma")
    if cfg.v_mode == "lds_t":
        # transposed: [d, k] so the B-operand column gather is contiguous in k.
        V_lds_t = make_lds_view(b, dtype=dtype_ir, shape=(hs, 16), name_hint="VwmmaT")

    # ---- iter-args: per-tile m/l (c_frag each) then per-tile acc (n_dk vecs) ----
    iter_args = []
    for t in range(BM):
        for r in range(c_frag):
            iter_args.append((f"m{t}_{r}", neg_inf))
        for r in range(c_frag):
            iter_args.append((f"l{t}_{r}", zero_f))
    for t in range(BM):
        for d in range(n_dk):
            iter_args.append((f"acc{t}_{d}", atom.zero_acc(b)))

    def unpack(state):
        idx = 0
        ms = []
        ls = []
        for _ in range(BM):
            ms.append(list(state[idx : idx + c_frag]))
            idx += c_frag
            ls.append(list(state[idx : idx + c_frag]))
            idx += c_frag
        accs = []
        for _ in range(BM):
            accs.append(
                [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
            )
            idx += n_dk
        return ms, ls, accs

    c_block_k = b.const_i32(_BLOCK_K)
    loop_stop = b.div(seqlen_k, c_block_k)
    if cfg.mask_mode == "causal":
        # Causal early-exit: this CTA owns q rows [group_row0, group_row0+BM*16-1].
        # K-tile kt covers keys [kt*16, kt*16+15], needed iff kt*16 <= max query
        # pos, i.e. kt < (group_row0/16)+BM. Skipping the fully-masked upper
        # triangle roughly halves the causal K-loop.
        causal_stop = b.add(b.div(group_row0, c_block_k), b.const_i32(BM))
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    # ---- Q is loop-invariant across K-tiles: optionally preload all frags once.
    # Trades VGPR (BM*n_dk live frags) for fewer dynamic loads -- but Q is already
    # L1-resident, so the register pressure can backfire via spills. ----
    q_frags = [[None] * n_dk for _ in range(BM)]
    if cfg.q_preload:
        for t in range(BM):
            qwin = q_window(t)
            for d in range(n_dk):
                q_frags[t][d] = load_wmma_tile(
                    b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )

    def k_window(k_tile_base):
        return make_tile_window(
            K_view,
            (1, 16, hs),
            origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
        )

    def v_window(k_tile_base):
        return make_tile_window(
            V_view,
            (1, 16, hs),
            origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
        )

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        k_tile_base = b.mul(kt, c_block_k)
        kwin = k_window(k_tile_base)

        # ---- K frags (shared across all BM Q-tiles). With fuse_k (BM==1) the
        # frags are loaded inside the QK matmul so only one is live at a time. ----
        k_frags = None
        if not fuse_k:
            k_frags = [
                load_wmma_tile(
                    b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                )
                for d in range(n_dk)
            ]

        new_ms = [list(ms[t]) for t in range(BM)]
        new_ls = [list(ls[t]) for t in range(BM)]
        new_accs = [list(accs[t]) for t in range(BM)]
        ps = [[None] * c_frag for _ in range(BM)]

        # ---- QK + online softmax for each Q-tile ----
        for t in range(BM):
            qwin = None if cfg.q_preload else q_window(t)
            score = WmmaTensor.zero_acc(b, atom, arch=arch)
            for d in range(n_dk):
                if cfg.q_preload:
                    q_tile = q_frags[t][d]
                else:
                    q_tile = load_wmma_tile(
                        b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                    )
                if fuse_k:
                    k_tile = load_wmma_tile(
                        b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                    )
                else:
                    k_tile = k_frags[d]
                score = wmma_mma(b, q_tile, k_tile, score)
            alpha_vec = atom.zero_acc(b)
            for r in range(c_frag):
                row_rel, col_k = score.coord(b, lane, r)
                s_r = b.fmul(score.slot(b, r), scale_log2)
                s_r = apply_attention_mask(
                    b,
                    s_r,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(k_tile_base, col_k),
                    query_pos=b.add(q_pos_base(t), row_rel),
                    sliding_window=0,
                )
                row_max = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
                m_new = b.fmax(ms[t][r], row_max)
                alpha = b.exp2(b.fsub(ms[t][r], m_new))
                p_r = b.exp2(b.fsub(s_r, m_new))
                row_sum = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
                new_ms[t][r] = m_new
                new_ls[t][r] = b.fadd(b.fmul(ls[t][r], alpha), row_sum)
                ps[t][r] = p_r
                alpha_vec = b.vec_insert(alpha_vec, alpha, r)
            # Vectorized online-softmax rescale: acc[d] *= alpha (one vmul/d).
            for d in range(n_dk):
                new_accs[t][d] = new_accs[t][d].scale(b, alpha_vec)

        # ---- transpose P (acc layout -> PV A-operand layout) ----
        p_a = _transpose_p(
            b, cfg, ps, lane, a_row, c_map, a_frag, c_frag, dtype_ir, P_lds
        )
        p_tiles = [WmmaTensor(atom, "a", pa, arch) for pa in p_a]

        # ---- V staging (transposed) once, shared across tiles ----
        if cfg.v_mode == "lds_t":
            _stage_v_transposed(
                b,
                cfg,
                V_view,
                V_lds_t,
                k_tile_base,
                kv_head,
                a_row,
                batch_tok_k,
                dtype_ir,
            )
            b.sync()

        # ---- PV: load V B-operand once per d, reuse across BM tiles ----
        vwin = v_window(k_tile_base)
        for d in range(n_dk):
            d_col = b.add(b.const_i32(d * 16), col)
            v_b = _load_v_b(b, cfg, vwin, V_lds_t, d, d_col, col, a_frag, dtype_ir)
            v_tile = WmmaTensor(atom, "b", v_b, arch)
            for t in range(BM):
                new_accs[t][d] = wmma_mma(b, p_tiles[t], v_tile, new_accs[t][d])

        yields = []
        for t in range(BM):
            for r in range(c_frag):
                yields.append(new_ms[t][r])
            for r in range(c_frag):
                yields.append(new_ls[t][r])
        for t in range(BM):
            yields.extend(a.value for a in new_accs[t])
        b.scf_yield(*yields)

    final = kloop.results
    ms_f, ls_f, accs_f = unpack(final)

    # ---- Epilogue per tile (CK Tile store_wmma_tile + TileWindow) ----
    for t in range(BM):
        owin = o_window(t)
        # inv_l depends only on (t,r); compute once instead of per-d (n_dk reloads).
        inv_l = []
        for r in range(c_frag):
            l_safe = ls_f[t][r]
            zmask = b.fcmp("oeq", l_safe, zero_f)
            inv_l.append(b.select(zmask, zero_f, b.rcp(l_safe)))

        def _rescale(bld, val, slot, row, colv, _inv=inv_l):
            return bld.fmul(val, _inv[slot])

        for d in range(n_dk):
            store_wmma_tile(
                b,
                owin,
                accs_f[t][d],
                lane,
                col_offset=d * 16,
                lead=[c0],
                align=2,
                transform=_rescale,
            )
    b.ret()
    return b.kernel


def _transpose_p(b, cfg, ps, lane, a_row, c_map, a_frag, c_frag, dtype_ir, P_lds):
    """Return a list of BM PV A-operand fragments (one per Q-tile).

    Uses the CK Tile :func:`make_lds_view` ``P_lds`` view: scatter each score
    slot to its ``(t, row, col)`` LDS cell, barrier, then read back row ``a_row``
    as the contiguous ``<16 x half>`` A fragment (two ``ds_read_b128`` halves +
    a concat -- an instruction cut over 16 scalar ds_loads on this issue-bound
    kernel)."""
    BM = cfg.bm_tiles
    if cfg.p_mode == "lds":
        for t in range(BM):
            ct = b.const_i32(t)
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                P_lds.store_scalar(
                    b, [ct, row_rel, col_k], b.cast_f32_to(ps[t][r], dtype_ir)
                )
        b.sync()
        out = []
        for t in range(BM):
            ct = b.const_i32(t)
            # WMMA a-operand slot j -> k=j, so the 16-wide A fragment is row
            # a_row's contiguous columns 0..15 of P_lds. f16 LDS loads cap at
            # <8 x f16> (ds_read_b128), so read the two halves and concat.
            lo = P_lds.load_vec(b, [ct, a_row, b.const_i32(0)], n=8)
            hi = P_lds.load_vec(b, [ct, a_row, b.const_i32(8)], n=8)
            out.append(b.vec_concat(lo, hi))
        return out
    raise ValueError(f"unknown p_mode {cfg.p_mode!r}")


def _stage_v_transposed(
    b, cfg, V_view, V_lds_t, k_tile_base, kv_head, a_row, batch_tok_k, dtype_ir
):
    """Stage the K-tile's V rows into LDS *transposed* (V_lds_t[d, k]) so the PV
    B-operand column gather becomes a contiguous read. Each lane owns k=a_row and
    scatters its head_size d-slice down the d axis."""
    tok = b.add(b.add(batch_tok_k, k_tile_base), a_row)
    hs = cfg.head_size
    for e in range(hs // 8):
        v_g = V_view.load_vec(b, [kv_head, tok, b.const_i32(e * 8)], n=8)
        for u in range(8):
            V_lds_t.store_scalar(
                b, [b.const_i32(e * 8 + u), a_row], b.vec_extract(v_g, u)
            )


def _load_v_b(b, cfg, vwin, V_lds_t, d, d_col, col, a_frag, dtype_ir):
    """PV B-operand for this lane's d-column: V[k, d_col] for k=0..15."""
    if cfg.v_mode == "lds_t":
        # contiguous in k: V_lds_t[d_col, 0..15] -- smem vec width caps at 8 for
        # f16, so two 8-wide contiguous reads concatenated into the 16-frag.
        lo = V_lds_t.load_vec(b, [d_col, b.const_i32(0)], n=8)
        hi = V_lds_t.load_vec(b, [d_col, b.const_i32(8)], n=8)
        return b.vec_concat(lo, hi)
    # gather from global through the V TileWindow (origin already at the K-tile).
    v_b = b.zero_vec(dtype_ir, a_frag)
    for j in range(a_frag):
        v_elem = vwin.load_scalar(b, b.const_i32(0), b.const_i32(j), d_col)
        v_b = b.vec_insert(v_b, v_elem, j)
    return v_b
