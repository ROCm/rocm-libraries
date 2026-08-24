# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Production-CK register-blocked multi-wave WMMA FMHA-forward kernel (gfx1151).

This is the full structural rewrite the single-wave campaign (``fmha_singlewave`` /
``fmha_pipelined``) named as the only path past the ~11 TF wall. It combines the TWO
levers that each failed *alone*:

  * **Multi-wave, shared K/V in LDS** (like ``fmha_multiwave``) -- amortizes the
    cooperative-staging ``s_barrier`` and the global K/V traffic across
    ``num_warps`` query-row tiles. ``fmha_multiwave`` did only this and lost (6.3 TF):
    each wave still issued only ``n_dk`` QK + ``n_dk`` PV WMMAs per tile, so the
    WMMA:overhead ratio never rose -- the barrier was pure cost.

  * **Register-blocked larger output tiles per wave** (like ``fmha_blockn``) --
    each wave owns ``m_repeat`` M-atoms x ``n_repeat = block_n/16`` N-atoms, so
    it issues ``m_repeat*n_repeat*n_dk`` QK + ``m_repeat*n_dk*n_repeat`` PV
    WMMAs per K-tile (e.g. m1 n4 D128 = 64 vs single-wave 16). ``fmha_blockn`` did
    only this and lost: spills exploded (10->130) with no second wave to hide
    behind.

Doing both at once is the bet: the density rise pays for the staging barrier,
and the extra waves give the barrier (and the spills) latency to hide behind.
Honest target ~25-35 TF (40-60% of the ~59 TF f16 WMMA peak), NOT 200.

Warps partition along **M only** (warp_n = 1), so each wave owns COMPLETE score
rows and the online softmax stays intra-warp (``wave_reduce_*`` over 16 lanes +
an in-register combine across the ``n_repeat`` score fragments) -- no cross-wave
reduction. K is staged ``[n][d]`` (QK B-operand = contiguous d-slice); V is
staged transposed ``[d][n]`` (PV B-operand = contiguous n-slice). The P->A
handoff uses an LDS transpose (NOT ds_bpermute -- proven structural dead-end on
this DSL); the barrier is now amortized across waves.

Like ``fmha_singlewave`` this kernel is built on the CK Tile WMMA helper layer:
every score/accumulator fragment carries as a packed
:class:`~rocke.helpers.WmmaTensor` (one SSA vector, no f32 cast), each
``m_repeat * n_repeat`` matmul is one :func:`~rocke.helpers.wmma_mma`, the
online-softmax rescale is one ``tile.scale``, the per-slot lane decode is
``tile.coord`` off the atom's verified ``c_layout``, and the O epilogue is
:func:`~rocke.helpers.store_wmma_tile`. The cooperative K/V LDS staging and the
P->A LDS transpose are NOT WMMA-tile ops, so they stay on the raw LDS-view path.
Each tile op lowers to the exact same single instruction the raw path emitted
(issue-bound: zero static-instruction regression).
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class RegBlockedCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    num_warps: int = 4  # wave32s/CTA, partition along M (warp_n = 1)
    m_repeat: int = 1  # 16-row M-atoms owned per wave
    block_n: int = 32  # keys consumed per K-loop step (n_repeat = /16)
    double_buffer: bool = False  # prefetch next K/V tile into a 2nd LDS buffer
    name: str = "wmma_fmha_regblocked"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def n_repeat(self) -> int:
        return self.block_n // 16

    @property
    def block_size(self) -> int:
        return 32 * self.num_warps

    @property
    def block_m(self) -> int:
        return self.num_warps * self.m_repeat * 16

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"w{self.num_warps}",
            f"m{self.m_repeat}",
            f"n{self.block_n}",
            "db" if self.double_buffer else "sb",
        )


def regblocked_grid(cfg: RegBlockedCfg, *, seqlen_q: int, batch: int):
    bm = cfg.block_m
    if seqlen_q % bm != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of block_m {bm}")
    return (seqlen_q // bm, cfg.num_query_heads, batch)


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


def build_wmma_fmha_regblocked(cfg: RegBlockedCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    c_frag = atom.c_per_lane  # 8
    if cfg.double_buffer:
        # Prefetch path is gated behind the base single-buffer result; only
        # build it if single-buffer beats the single-wave wall (see plan).
        raise NotImplementedError("double_buffer prefetch not implemented yet")
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.num_warps
    MR = cfg.m_repeat
    NR = cfg.n_repeat
    bn = cfg.block_n
    dtype_ir = F16

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.block_size
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)  # 0..W-1
    lane = b.mod(tid, c_wave)  # 0..wave-1
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16
    col = b.mod(lane, c16)  # lane % 16

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

    # ---- CK Tile 3D (head, token, dim) views (see fmha_singlewave for the rationale). ----
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

    block_m = b.const_i32(cfg.block_m)
    cta_row0 = b.mul(q_group, block_m)  # within-batch first q row of CTA
    batch_row_q = b.mul(batch, seqlen_q)
    batch_tok_k = b.mul(batch, seqlen_k)

    # this wave's first M-atom row base (within batch); + mr*16 per atom
    wave_m_base = b.add(cta_row0, b.mul(wave_id, b.const_i32(MR * 16)))

    def m_pos_base(mr):  # within-batch row base of this wave's mr-th M-atom
        return b.add(wave_m_base, b.const_i32(mr * 16))

    # per-(mr) Q/O windows (loop-invariant across K-tiles).
    qwins, owins = [], []
    for mr in range(MR):
        m_tok = b.add(m_pos_base(mr), batch_row_q)
        qwins.append(make_tile_window(Q_view, (1, 16, hs), origin=(head, m_tok, c0)))
        owins.append(make_tile_window(O_view, (1, 16, hs), origin=(head, m_tok, c0)))

    # ---- LDS: shared K [bn][d], transposed V [d][bn], per-wave P slab ----
    K_lds = make_lds_view(b, dtype=dtype_ir, shape=(bn, hs), name_hint="Ksh")
    V_lds_t = make_lds_view(b, dtype=dtype_ir, shape=(hs, bn), name_hint="VshT")
    P_lds = make_lds_view(b, dtype=dtype_ir, shape=(W, 16, 16), name_hint="Psh")

    # ---- cooperative vec8 loaders (all block_size threads participate) ----
    n_threads = cfg.block_size
    elems = bn * hs
    if elems % (n_threads * 8) != 0:
        raise ValueError(
            f"tile {bn}x{hs} ({elems} elems) not divisible by {n_threads}*8; "
            f"pick block_n/head_size/num_warps so it is"
        )
    chunks_per_thread = (elems // 8) // n_threads

    def coop_load_k(k_tile_base):
        # K_lds[n][d] = K[k_tile_base+n, d]; contiguous vec8 copy.
        for i in range(chunks_per_thread):
            c = b.add(tid, b.const_i32(i * n_threads))
            base = b.mul(c, b.const_i32(8))
            row = b.div(base, b.const_i32(hs))
            colc = b.mod(base, b.const_i32(hs))
            tok = b.add(b.add(batch_tok_k, k_tile_base), row)
            v8 = K_view.load_vec(b, [kv_head, tok, colc], n=8)
            K_lds.store_vec(b, [row, colc], v8, 8)

    def coop_load_v_t(k_tile_base):
        # V_lds_t[d][n] = V[k_tile_base+n, d]; contiguous load in d, transposed
        # scatter into LDS (the only strided write, paid once per CTA).
        for i in range(chunks_per_thread):
            c = b.add(tid, b.const_i32(i * n_threads))
            base = b.mul(c, b.const_i32(8))
            row = b.div(base, b.const_i32(hs))  # n (kv row)
            colc = b.mod(base, b.const_i32(hs))  # d base
            tok = b.add(b.add(batch_tok_k, k_tile_base), row)
            v8 = V_view.load_vec(b, [kv_head, tok, colc], n=8)
            for u in range(8):
                V_lds_t.store_scalar(
                    b, [b.add(colc, b.const_i32(u)), row], b.vec_extract(v8, u)
                )

    def load_k_b_frag(nr, d):
        # QK B-operand from LDS: K_lds[nr*16 + lane%16][d*16 .. +16]. f16 smem
        # loads cap at <8 x f16> (ds_read_b128), so read two halves + concat into
        # the 16-frag (2 loads vs 16 scalar ds_loads + 16 inserts).
        krow = b.add(b.const_i32(nr * 16), col)
        lo = K_lds.load_vec(b, [krow, b.const_i32(d * 16)], n=8)
        hi = K_lds.load_vec(b, [krow, b.const_i32(d * 16 + 8)], n=8)
        return b.vec_concat(lo, hi)

    def load_v_b_frag(nr, d):
        # PV B-operand from LDS: V_lds_t[d*16 + lane%16][nr*16 .. +16].
        d_col = b.add(b.const_i32(d * 16), col)
        lo = V_lds_t.load_vec(b, [d_col, b.const_i32(nr * 16)], n=8)
        hi = V_lds_t.load_vec(b, [d_col, b.const_i32(nr * 16 + 8)], n=8)
        return b.vec_concat(lo, hi)

    # ---- iter-args: per mr -> m[c_frag], l[c_frag], acc[n_dk] ----
    iter_args = []
    for mr in range(MR):
        for r in range(c_frag):
            iter_args.append((f"m{mr}_{r}", neg_inf))
        for r in range(c_frag):
            iter_args.append((f"l{mr}_{r}", zero_f))
        for d in range(n_dk):
            iter_args.append(
                (f"acc{mr}_{d}", WmmaTensor.zero_acc(b, atom, arch=arch).value)
            )

    per_mr = 2 * c_frag + n_dk

    def unpack(state):
        ms, ls, accs = [], [], []
        for mr in range(MR):
            base = mr * per_mr
            ms.append(list(state[base : base + c_frag]))
            ls.append(list(state[base + c_frag : base + 2 * c_frag]))
            accs.append(
                [
                    WmmaTensor(atom, "c", v, arch)
                    for v in state[base + 2 * c_frag : base + per_mr]
                ]
            )
        return ms, ls, accs

    c_block_n = b.const_i32(bn)
    loop_stop = b.div(seqlen_k, c_block_n)

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        k_tile_base = b.mul(kt, c_block_n)

        # ---- cooperative K/V staging (all waves) ----
        coop_load_k(k_tile_base)
        coop_load_v_t(k_tile_base)
        b.sync()

        new_ms = [list(ms[mr]) for mr in range(MR)]
        new_ls = [list(ls[mr]) for mr in range(MR)]
        new_accs = [list(accs[mr]) for mr in range(MR)]

        # ---- QK: A = this wave's Q rows (global), B = shared K (LDS) ----
        # score[mr][nr] is a c_frag (<8 f32>) C-distribution fragment. Q frags
        # are loaded ONCE per (mr,d) and each K B-operand ONCE per (nr,d), then
        # reused across the other axis -- the register-blocking amortization
        # (1 LDS K-read feeds m_repeat matmuls) that makes the staging pay.
        q_frags = [[None] * n_dk for _ in range(MR)]
        for mr in range(MR):
            for d in range(n_dk):
                q_frags[mr][d] = load_wmma_tile(
                    b, qwins[mr], atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )
        scores = [
            [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(NR)]
            for _ in range(MR)
        ]
        for nr in range(NR):
            for d in range(n_dk):
                k_b = load_k_b_frag(nr, d)
                k_tile = WmmaTensor(atom, "b", k_b, arch)
                for mr in range(MR):
                    scores[mr][nr] = wmma_mma(b, q_frags[mr][d], k_tile, scores[mr][nr])

        # ---- intra-warp online softmax over the full block_n row, per mr ----
        ps = [[None] * NR for _ in range(MR)]  # probabilities, C-dist
        alpha_vecs = [b.zero_vec_f32(c_frag) for _ in range(MR)]
        for mr in range(MR):
            for r in range(c_frag):
                row_rel = scores[mr][0].coord(b, lane, r)[0]
                q_pos = b.add(m_pos_base(mr), row_rel)
                # per-nr masked/scaled score and its 16-lane row-max
                s_nr = []
                atom_max = None
                for nr in range(NR):
                    col_k = scores[mr][nr].coord(b, lane, r)[1]
                    s_r = b.fmul(scores[mr][nr].slot(b, r), scale_log2)
                    s_r = apply_attention_mask(
                        b,
                        s_r,
                        mask_mode=cfg.mask_mode,
                        k_idx=b.add(k_tile_base, b.add(b.const_i32(nr * 16), col_k)),
                        query_pos=q_pos,
                        sliding_window=0,
                    )
                    rmax = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
                    s_nr.append(s_r)
                    atom_max = rmax if atom_max is None else b.fmax(atom_max, rmax)
                m_new = b.fmax(ms[mr][r], atom_max)
                alpha = b.exp2(b.fsub(ms[mr][r], m_new))
                total_sum = None
                for nr in range(NR):
                    p_r = b.exp2(b.fsub(s_nr[nr], m_new))
                    rsum = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
                    total_sum = rsum if total_sum is None else b.fadd(total_sum, rsum)
                    # stash p_r into the c_frag for this (mr,nr)
                    if ps[mr][nr] is None:
                        ps[mr][nr] = b.zero_vec_f32(c_frag)
                    ps[mr][nr] = b.vec_insert(ps[mr][nr], p_r, r)
                new_ms[mr][r] = m_new
                new_ls[mr][r] = b.fadd(b.fmul(ls[mr][r], alpha), total_sum)
                alpha_vecs[mr] = b.vec_insert(alpha_vecs[mr], alpha, r)
            for d in range(n_dk):
                new_accs[mr][d] = new_accs[mr][d].scale(b, alpha_vecs[mr])

        # ---- transpose all P fragments (C-dist -> PV A-operand) via LDS ----
        # P_lds[wave_id] is reused per (mr,nr); the read-back must complete before
        # the next store overwrites it, so transpose all MR*NR frags up front into
        # registers, then run PV with each V B-operand loaded ONCE per (nr,d).
        p_a = [[None] * NR for _ in range(MR)]
        for mr in range(MR):
            for nr in range(NR):
                for r in range(c_frag):
                    row_rel, col_k = c_map.coord(b, lane, r)
                    P_lds.store_scalar(
                        b,
                        [wave_id, row_rel, col_k],
                        b.cast_f32_to(b.vec_extract(ps[mr][nr], r), dtype_ir),
                    )
                b.s_waitcnt(lgkmcnt=0)  # intra-wave: lockstep on own P_lds slab
                # A-operand slot j -> k=j, so row a_row's contiguous cols 0..15;
                # two <8 x f16> reads + concat (vs 16 scalar ds_loads + inserts).
                lo = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(0)], n=8)
                hi = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(8)], n=8)
                p_a[mr][nr] = b.vec_concat(lo, hi)
                b.s_waitcnt(lgkmcnt=0)  # P_lds reused by next (mr,nr)

        # ---- PV: V B-operand loaded once per (nr,d), reused across all M-atoms ----
        for nr in range(NR):
            for d in range(n_dk):
                v_b = load_v_b_frag(nr, d)
                v_tile = WmmaTensor(atom, "b", v_b, arch)
                for mr in range(MR):
                    p_tile = WmmaTensor(atom, "a", p_a[mr][nr], arch)
                    new_accs[mr][d] = wmma_mma(b, p_tile, v_tile, new_accs[mr][d])

        # barrier before next iteration overwrites the shared K/V tiles
        b.sync()

        yields = []
        for mr in range(MR):
            yields.extend(new_ms[mr])
            yields.extend(new_ls[mr])
            yields.extend(a.value for a in new_accs[mr])
        b.scf_yield(*yields)

    final = kloop.results
    ms_f, ls_f, accs_f = unpack(final)

    # ---- Epilogue (CK Tile store_wmma_tile + TileWindow), per M-atom ----
    for mr in range(MR):
        inv_l = []
        for r in range(c_frag):
            l_safe = ls_f[mr][r]
            zmask = b.fcmp("oeq", l_safe, zero_f)
            inv_l.append(b.select(zmask, zero_f, b.rcp(l_safe)))

        def _rescale(bld, val, slot, row, colv, _inv=inv_l):
            return bld.fmul(val, _inv[slot])

        for d in range(n_dk):
            store_wmma_tile(
                b,
                owins[mr],
                accs_f[mr][d],
                lane,
                col_offset=d * 16,
                lead=[c0],
                align=2,
                transform=_rescale,
            )
    b.ret()
    return b.kernel
