# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Persistent-grid WMMA FMHA-forward kernel for gfx1151.

This is the multi-wave gather winner (``fmha_multiwave.py``, ``kv_source="gather"``
+ ``sched_mode="pingpong"`` + ``batch_softmax``) with ONE structural change: instead
of a ``(Sq/q_per, heads, batch)`` grid that launches one CTA per (query-block, head,
batch) work-item, we launch a **fixed 1-D grid of ``num_persistent`` long-lived CTAs**
that pull work-items from a global atomic counter until the ``num_tiles`` work-items
are exhausted (:mod:`rocke.helpers.persistent`).

Why persistency here
--------------------
Every one-shot CTA pays a fixed prologue -- Q-window setup, LDS/iter-arg init,
address arithmetic, the WMMA warm-up -- that is pure overhead against the K-loop.
At the small per-CTA tile (16*n_waves query rows) the gfx1151 attention grid is
launch-heavy: hundreds-to-thousands of tiny CTAs whose launch + prologue is not
hidden by anything. A persistent grid sized to the machine (``~CU * occupancy``
CTAs) hits steady state once and amortizes that prologue across every tile the CTA
drains, and (with a work-item order that keeps a CTA on the same ``(head, batch)``)
keeps K/V hot in L2 across successive query blocks.

The math/dataflow inside the K-loop is BYTE-FOR-BYTE the gather winner -- only the
grid/work-dispatch wrapper differs. The V "scatter" (cache-resident column gather)
is unchanged: gfx1151 has no ``ds_read_tr``, so the transposed-PV / LDS-staging path
that would remove it loses to the gather on this APU (see ``CK_PARITY_CASE_STUDY``).
"""

from __future__ import annotations

import math
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
class PersistentCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    n_waves: int = 4  # wave32s per CTA; each owns a 16-row Q-tile
    waves_per_eu: Optional[int] = None
    sched_mode: str = "none"  # "none" | "pingpong"
    batch_softmax: bool = False
    qk_ilp: int = 1
    # num_persistent: size of the fixed 1-D launch grid (long-lived CTAs). This is
    # the COMPILE-TIME loop budget the ``scf.for`` trip count is derived from, so it
    # is baked into the kernel; the driver must launch exactly this many CTAs. A
    # sweet spot on the 40-CU mini is deep oversubscription (~960, i.e. ~24x the
    # CU count): enough queue depth to hide the cooperative-fetch latency and keep
    # K/V hot in L2. 0 is rejected at build time.
    num_persistent: int = 960
    # persist_decode: work-item -> (q_group, head, batch) unpack order.
    #   "qb_major"  (default): tile = (batch*Hq + head)*NQB + q_group  -> q_group is
    #               the fastest axis, so a CTA that happens to drain adjacent tile
    #               ids stays on one (head,batch) and reuses its K/V in L2.
    #   "batch_major": tile = (q_group*Hq + head)*B + batch -> spreads batch, used
    #               only for the reproducible A/B.
    persist_decode: str = "qb_major"
    name: str = "wmma_fmha_persistent"

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
            "bsm" if self.batch_softmax else "ssm",
            f"ilp{self.qk_ilp}",
            f"pers{self.num_persistent}",
            self.persist_decode,
        )


def num_work_items(cfg: PersistentCfg, *, seqlen_q: int, batch: int) -> int:
    """Total (query-block, head, batch) work-items for this shape."""
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per) * cfg.num_query_heads * batch


def persistent_grid(cfg: PersistentCfg):
    """Fixed 1-D launch grid (independent of the problem size)."""
    return (cfg.num_persistent, 1, 1)


def _declare_params(b: IRBuilder):
    P = {}
    P["Q"] = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["K"] = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["V"] = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["O"] = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )
    # Persistent work-queue counter: a single i32 global slot the host pre-clears
    # to 0 before EACH launch. CTAs atomic-add(1) to fetch the next work-item.
    P["Counter"] = b.param("Counter", PtrType(I32, "global"), noalias=True, align=16)
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


def build_wmma_fmha_persistent(
    cfg: PersistentCfg,
    arch: str = "gfx1151",
    *,
    num_q_blocks: int,
    batch: int,
) -> KernelDef:
    """Build the persistent FMHA kernel.

    ``num_q_blocks`` (= seqlen_q // q_rows_per_cta) and ``batch`` are baked at build
    time so the tile decode uses compile-time constant divs/mods and the ``scf.for``
    trip count (``max_iters``) is exact. The driver rebuilds per shape.
    """
    if cfg.num_persistent <= 0:
        raise ValueError(f"num_persistent must be > 0, got {cfg.num_persistent}")
    if cfg.persist_decode not in ("qb_major", "batch_major"):
        raise ValueError(f"bad persist_decode {cfg.persist_decode!r}")

    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.n_waves
    dtype_ir = F16

    qh = cfg.num_query_heads
    num_tiles = num_q_blocks * qh * batch
    # worst-case per-CTA drain count; the in_range guard makes any over-count safe.
    max_iters = (num_tiles + cfg.num_persistent - 1) // cfg.num_persistent

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.block_size
    if cfg.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = cfg.waves_per_eu
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)  # 0..W-1
    lane = b.mod(tid, c_wave)  # 0..wave-1
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16
    col = b.mod(lane, c16)  # lane % 16

    kvh = cfg.kv_heads

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
    Counter = p["Counter"]

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # ---- tile-independent global views (pure pointer + stride) ----
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

    # per-wave P transpose slab (scratch; reused every K-tile and every work-item).
    P_lds = make_lds_view(b, dtype=dtype_ir, shape=(W, 16, 16), name_hint="Psh")

    q_rows_per_cta = b.const_i32(cfg.q_rows_per_cta)
    c_block_k = b.const_i32(_BLOCK_K)
    c_nqb = b.const_i32(num_q_blocks)
    c_qh = b.const_i32(qh)
    pingpong = cfg.sched_mode == "pingpong"

    def decode(tile_idx):
        """work-item id -> (q_group, head, batch), compile-time constant divisors."""
        if cfg.persist_decode == "qb_major":
            q_group = b.mod(tile_idx, c_nqb)
            hb = b.div(tile_idx, c_nqb)
            head = b.mod(hb, c_qh)
            batch_i = b.div(hb, c_qh)
        else:  # batch_major
            batch_i = b.mod(tile_idx, b.const_i32(batch))
            hq = b.div(tile_idx, b.const_i32(batch))
            head = b.mod(hq, c_qh)
            q_group = b.div(hq, c_qh)
        return q_group, head, batch_i

    def do_work(q_group, head, batch_i):
        kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

        cta_row0 = b.mul(q_group, q_rows_per_cta)
        wave_row0 = b.add(cta_row0, b.mul(wave_id, c16))
        batch_tok_q = b.mul(batch_i, seqlen_q)
        batch_tok_k = b.mul(batch_i, seqlen_k)
        q_pos_base = wave_row0
        q_token_base = b.add(wave_row0, batch_tok_q)

        qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base, c0))
        owin = make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base, c0))

        iter_args = []
        for r in range(c_frag):
            iter_args.append((f"m{r}", neg_inf))
        for r in range(c_frag):
            iter_args.append((f"l{r}", zero_f))
        for d in range(n_dk):
            iter_args.append((f"acc{d}", atom.zero_acc(b)))

        def unpack(state):
            idx = 0
            ms = list(state[idx : idx + c_frag])
            idx += c_frag
            ls = list(state[idx : idx + c_frag])
            idx += c_frag
            accs = [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
            idx += n_dk
            return ms, ls, accs

        loop_stop = b.div(seqlen_k, c_block_k)
        if cfg.mask_mode == "causal":
            causal_stop = b.add(b.div(cta_row0, c_block_k), b.const_i32(W))
            loop_stop = b.select(
                b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop
            )

        def v_window(k_tile_base):
            return make_tile_window(
                V_view,
                (1, 16, hs),
                origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
            )

        def k_window(k_tile_base):
            return make_tile_window(
                K_view,
                (1, 16, hs),
                origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
            )

        def gather_v_b_frag(vwin, d):
            d_col = b.add(b.const_i32(d * 16), col)
            v_b = b.zero_vec(dtype_ir, atom.a_per_lane)
            for j in range(atom.a_per_lane):
                v_elem = vwin.load_scalar(b, c0, b.const_i32(j), d_col)
                v_b = b.vec_insert(v_b, v_elem, j)
            return v_b

        kloop = b.scf_for_iter(
            b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
        )
        with kloop as (kt, state):
            ms, ls, accs = unpack(state)
            k_tile_base = b.mul(kt, c_block_k)
            kwin = k_window(k_tile_base)
            vwin = v_window(k_tile_base)

            new_ms = list(ms)
            new_ls = list(ls)
            new_accs = list(accs)
            ps = [None] * c_frag

            # ---- QK ----
            if pingpong:
                b.s_setprio(1)
            ilp = max(1, cfg.qk_ilp)
            scores = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
            for d in range(n_dk):
                q_tile = load_wmma_tile(
                    b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )
                k_tile = load_wmma_tile(
                    b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                )
                si = d % ilp
                scores[si] = wmma_mma(b, q_tile, k_tile, scores[si])
            score = scores[0]
            for si in range(1, ilp):
                score = WmmaTensor(
                    atom, "c", b.vector_add(score.value, scores[si].value), arch
                )
            if pingpong:
                b.s_setprio(0)

            # ---- online softmax ----
            def _scaled_masked_row(r):
                row_rel, col_k = score.coord(b, lane, r)
                s_r = b.fmul(score.slot(b, r), scale_log2)
                return apply_attention_mask(
                    b,
                    s_r,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(k_tile_base, col_k),
                    query_pos=b.add(q_pos_base, row_rel),
                    sliding_window=0,
                )

            alpha_vec = b.zero_vec_f32(c_frag)
            if cfg.batch_softmax:
                s_rows = [_scaled_masked_row(r) for r in range(c_frag)]
                row_maxs = [
                    wave_reduce_max(b, s_rows[r], wave_size=wave, lanes_per_row=16)
                    for r in range(c_frag)
                ]
                m_news = [b.fmax(ms[r], row_maxs[r]) for r in range(c_frag)]
                alphas = [b.exp2(b.fsub(ms[r], m_news[r])) for r in range(c_frag)]
                p_batch = [b.exp2(b.fsub(s_rows[r], m_news[r])) for r in range(c_frag)]
                row_sums = [
                    wave_reduce_sum(b, p_batch[r], wave_size=wave, lanes_per_row=16)
                    for r in range(c_frag)
                ]
                for r in range(c_frag):
                    new_ms[r] = m_news[r]
                    new_ls[r] = b.fadd(b.fmul(ls[r], alphas[r]), row_sums[r])
                    ps[r] = p_batch[r]
                    alpha_vec = b.vec_insert(alpha_vec, alphas[r], r)
            else:
                for r in range(c_frag):
                    s_r = _scaled_masked_row(r)
                    row_max = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
                    m_new = b.fmax(ms[r], row_max)
                    alpha = b.exp2(b.fsub(ms[r], m_new))
                    p_r = b.exp2(b.fsub(s_r, m_new))
                    row_sum = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
                    new_ms[r] = m_new
                    new_ls[r] = b.fadd(b.fmul(ls[r], alpha), row_sum)
                    ps[r] = p_r
                    alpha_vec = b.vec_insert(alpha_vec, alpha, r)
            for d in range(n_dk):
                new_accs[d] = new_accs[d].scale(b, alpha_vec)

            # ---- transpose P (acc layout -> PV A-operand) via this wave's LDS slab ----
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                P_lds.store_scalar(
                    b, [wave_id, row_rel, col_k], b.cast_f32_to(ps[r], dtype_ir)
                )
            b.s_waitcnt(lgkmcnt=0)
            lo = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(0)], n=8)
            hi = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(8)], n=8)
            p_a = b.vec_concat(lo, hi)

            # ---- PV ----
            if pingpong:
                b.s_setprio(1)
            p_tile = WmmaTensor(atom, "a", p_a, arch)
            for d in range(n_dk):
                v_b = gather_v_b_frag(vwin, d)
                v_tile = WmmaTensor(atom, "b", v_b, arch)
                new_accs[d] = wmma_mma(b, p_tile, v_tile, new_accs[d])
            if pingpong:
                b.s_setprio(0)

            yields = []
            for r in range(c_frag):
                yields.append(new_ms[r])
            for r in range(c_frag):
                yields.append(new_ls[r])
            yields.extend(a.value for a in new_accs)
            b.scf_yield(*yields)

        final = kloop.results
        ms_f, ls_f, accs_f = unpack(final)

        # ---- Epilogue ----
        inv_l = []
        for r in range(c_frag):
            l_safe = ls_f[r]
            zmask = b.fcmp("oeq", l_safe, zero_f)
            inv_l.append(b.select(zmask, zero_f, b.rcp(l_safe)))

        def _rescale(bld, val, slot, row, colv, _inv=inv_l):
            return bld.fmul(val, _inv[slot])

        for d in range(n_dk):
            store_wmma_tile(
                b,
                owin,
                accs_f[d],
                lane,
                col_offset=d * 16,
                lead=[c0],
                align=2,
                transform=_rescale,
            )

    # ---- persistent work-queue ----
    # TOP-FETCH pattern (correctness-critical): each of the ``max_iters``
    # iterations does exactly ONE cooperative ``atomic_add(1)`` at the TOP and
    # processes the fetched tile iff it is in range. Because every CTA fetches
    # exactly ``max_iters`` times, total fetches = ``num_persistent * max_iters``
    # >= ``num_tiles`` (by the ceil), and the counter is monotonic, so every tile
    # id in ``[0, num_tiles)`` is handed out to some iteration and processed
    # exactly once -- REGARDLESS of load imbalance. (The shared
    # ``persistent_tile_loop`` helper fetches the NEXT tile at the BOTTOM, so its
    # final fetch per CTA is consumed-but-never-processed; under imbalance that
    # steals a still-valid tile id and silently drops that output tile.)
    c_num_tiles = b.const_i32(num_tiles)
    multiwave = cfg.block_size > wave
    brd = b.smem_alloc(I32, [1], name_hint="pers_brd") if multiwave else None
    c_one = b.const_i32(1)

    def fetch_tile():
        """Cooperative atomic tile fetch broadcast to every thread in the CTA."""
        is_lead = b.cmp_eq(tid, c0)
        if not multiwave:
            # single wave: every lane issues the atomic (only lane 0 increments),
            # then ds_bpermute(0) broadcasts lane 0's result wave-internally --
            # no LDS / barrier, race-free (matches helpers.persistent P35 fix).
            inc = b.select(is_lead, c_one, c0)
            fetched = b.global_atomic_add(Counter, c0, inc)
            return b.ds_bpermute(c0, fetched)
        # multi-wave: lead thread does the atomic + LDS store, then ONE trailing
        # barrier publishes the value to every wave before any reads it. No
        # leading barrier is needed: the only reader of ``brd`` is the load right
        # after this barrier, and each thread latches its tile_idx into registers
        # there -- a straggler never re-reads ``brd``, so lead overwriting it early
        # next iteration is a harmless WAR (no thread observes the stale value).
        with b.scf_if(is_lead):
            v = b.global_atomic_add(Counter, c0, c_one)
            b.smem_store_vN(brd, [c0], v, 1)
        # LDS-only barrier (lgkmcnt + s_barrier) publishes ``brd`` without a
        # full vmcnt drain, so the previous work-item's outstanding V-gathers /
        # O-stores keep flowing across the work-item boundary (avoids the
        # per-tile memory-pipeline stall a full ``b.sync()`` would impose).
        b.sync_lds_only()
        return b.vec_extract(b.smem_load_vN(brd, c0, dtype=I32, n=1), 0)

    ploop = b.scf_for_iter(
        b.const_i32(0), b.const_i32(max_iters), c_one, iter_args=[], iv_name="pers_iter"
    )
    with ploop as _pi:
        tile_idx = fetch_tile()
        in_range = b.cmp_lt(tile_idx, c_num_tiles)
        with b.scf_if(in_range):
            q_group, head, batch_i = decode(tile_idx)
            do_work(q_group, head, batch_i)

    b.ret()
    return b.kernel
