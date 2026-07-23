# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Persistent-grid + query-blocked transposed-QK WMMA FMHA-forward (gfx1151).

This fuses the two independent large-L levers the campaign found for D128:

  * **MQ=2 query-blocking with f16 O-carry** (``wmma_fmha_swapqk`` ``q_block=2``
    ``o_f16=True``): each wave processes MQ 16-row query tiles per K-loop
    iteration, loading each K (QK) and V (PV) fragment ONCE and reusing it across
    all MQ groups. That is *register-level* KV reuse -- it halves the KV DRAM read
    per query and pushes the DRAM-roofline knee at D128 from ~L2K out to ~L4K
    (measured 24 TF @ L4096 vs the 17 TF MQ=1 baseline). It is the ONLY MQ that
    fits D128's 256-VGPR budget (f16 C-frags halve the accumulator peak); it still
    spills 38 but the reuse dominates.

  * **Persistent work-queue** (``wmma_fmha_swapqk_persistent``): a fixed grid of
    ``num_persistent`` long-lived CTAs drains (q_block, head, batch) work-items
    from a global atomic counter in ``qb_major`` order, so a CTA that drains
    adjacent tile ids stays on ONE (head, batch) and its K/V stay hot in cache.
    At D128 L16K the per-head K+V working set is ~8 MB, which FITS Strix Halo's
    32 MB MALL -- so a head-consecutive traversal converts the L>4K DRAM re-reads
    (which crater MQ=2 to ~10 TF @ L8K) into MALL hits. That is *cache-level* KV
    reuse, and it is what MQ=2 alone cannot buy past its ~L4K register-reuse knee.

The two reuse axes are orthogonal (registers within a work-item, MALL across
work-items), so stacking them is the path to sustaining the compute-bound ~25 TF
D128 number out into the L8K-16K regime.

LDS barrier / pipeline note
---------------------------
The MQ body itself uses NO LDS and NO barriers (the P-transpose is the register
``PermuteWarpGemmCToA``). The ONLY barrier is the multi-wave work-item broadcast
in ``fetch_tile``, and it is a ``sync_lds_only`` (lgkmcnt + ``s_barrier``), NOT a
full ``b.sync()``: it publishes the atomically-fetched tile id through LDS without
draining ``vmcnt``, so the previous work-item's outstanding V-gathers / O-stores
keep flowing across the work-item boundary instead of stalling the memory pipe at
every tile. This is the "don't let the LDS barrier drain the pipeline completely"
requirement -- the barrier orders LDS, the VMEM pipeline stays live.

The math/dataflow inside the K-loop is byte-for-byte the validated swapqk MQ path
(``build_wmma_fmha_swapqk`` with ``q_block>1``); only the grid + work-dispatch
wrapper differs. See ``wmma_fmha_swapqk.py`` for the transposed-QK derivation.
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
    make_lds_view,
    make_tile_window,
    store_wmma_tile,
    wmma_mma,
)
from rocke.helpers.attention import apply_attention_mask

__all__ = [
    "SwapQKPersistentCfg",
    "build_wmma_fmha_swapqk_pmq",
    "swapqk_pmq_grid",
    "num_work_items",
]

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"


@dataclass(frozen=True)
class SwapQKPersistentCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    # --- swapqk MQ knobs (defaults = the validated d128_mq2_of16 winner) ---
    n_waves: int = 2
    q_block: int = 2  # MQ; register KV reuse across MQ query groups
    o_f16: bool = True  # f16 O-carry -- required for MQ=2 to fit D128
    block_n: int = 32
    qk_ilp: int = 2
    sched_mode: str = "pingpong"  # "none" | "pingpong"
    buffer_gather: bool = True
    dual_gather: bool = True
    fast_exp2: bool = True
    waves_per_eu: Optional[int] = None
    # --- persistent-grid knobs ---
    # num_persistent: fixed 1-D launch grid size (long-lived CTAs). Baked at build
    # time (the scf.for trip budget derives from it); the driver must launch
    # exactly this many CTAs.
    #
    # For this MALL-reuse kernel the sweet spot is NON-monotonic and much SMALLER
    # than the untransposed persistent kernel's deep oversubscription: too few
    # CTAs underutilize the 40 CUs, too many make the aggregate KV working set
    # (8 MB/head at L16K/D128) of the concurrently-active heads blow the 32 MB
    # MALL back to DRAM. Measured L16K D128 (H24 B1, MQ2/bn32/ilp2/of16): pers40
    # 6.7 -> pers144 12.65 (PEAK) -> pers208 8.0 -> pers256 6.8 TF. ~3.5x the CU
    # count (140-160) balances occupancy against MALL residency; larger L wants
    # the low end of that, shorter L the high end.
    num_persistent: int = 144
    # persist_decode: work-item -> (q_group, head, batch) unpack order.
    #   "qb_major" (default): tile = (batch*Hq + head)*NQB + q_group  -> q_group is
    #       the fastest axis, so a CTA draining adjacent tile ids stays on one
    #       (head, batch) and its K/V stay hot in MALL across query blocks. THIS is
    #       the cache-reuse lever; use it.
    #   "batch_major": tile = (q_group*Hq + head)*B + batch -> spreads (head,batch),
    #       defeats the reuse. For the A/B only.
    persist_decode: str = "qb_major"
    # head_blocked: replace the atomic work-queue with a STATIC cohort partition.
    # The grid is split into ``num_cohorts`` cohorts of ``num_persistent //
    # num_cohorts`` CTAs each; a cohort is pinned to ONE (head, batch) at a time
    # (its CTAs split that head's q_blocks grid-stride) and strides across
    # head-units by ``num_cohorts``. So at any instant EXACTLY ``num_cohorts``
    # heads are active -> the concurrent KV working set is bounded to
    # ``num_cohorts * (per-head KV)`` (8 MB/head at L16K/D128), which lets us pin
    # it under the 32 MB MALL regardless of how CTAs drift. Unlike the atomic
    # counter (whose contiguous id-window can smear across head boundaries), this
    # is exact -- and it needs NO atomics, NO broadcast, NO barrier (each CTA is
    # independent), so there is no pipeline drain at all. Counter param is unused.
    #
    # MEASURED DEAD-END (hardware, gfx1151 stx-halo, D128 dense H24 B1, MQ2/bn32/
    # ilp2/of16, pers144; correct 1.07e-4): LOSES to the atomic work-queue at
    # every L / cohort-count / npers --
    #     L16K: wq 12.65 vs hb2 11.09 / hb3 10.10 / hb4 11.08 / hb6 10.52
    #     L32K: wq 11.34 vs hb2 10.17 / hb3 9.76
    # The premise (recover MALL residency by bounding concurrent heads) was WRONG:
    # the work-queue ALREADY keeps KV MALL-resident -- with num_q_blocks (256 @
    # L16K) >> num_persistent, its monotonic contiguous id-window naturally sits
    # within ~1 head -- so there is no residency to recover, and PMQ is MALL-
    # bandwidth / latency bound, not residency bound. Static cohorts only ADD
    # load imbalance (fixed q_block split -> stragglers) vs the dynamic queue.
    # Kept OFF for the A/B record.
    head_blocked: bool = False
    num_cohorts: int = 2
    # kv_lds: stage the current K-loop KV tile in LDS (shared by all W waves)
    # instead of each wave re-gathering it. K is stored row-major (its
    # d-consecutive WMMA-a read is already coalesced); V is stored TRANSPOSED
    # (V^T[d, kv]) + row-padded so the PV a-operand read -- 16 consecutive kv at
    # a fixed d per lane -- is 2 coalesced vec8 ds_reads (NOT 16 scattered scalar
    # loads). O stays in registers (MQ2), so there is NO O-LDS traffic (the thing
    # that pinned kvstat). This is the untested "PMQ + LDS KV" path: the reuse is
    # the W-wave share within a work-item (the coop MALL read is amortized over
    # W*MQ query tiles) done through fast LDS instead of repeated MALL gathers.
    #
    # MEASURED DEAD-END (hardware, gfx1151 stx-halo, D128 dense H24 B1, pers144;
    # correct 1.07e-4): ~2x BELOW plain PMQ at every config, and the loss is NOT
    # the things this path was built to fix --
    #     L16K PMQ(MALL) 12.65  vs  kvlds ilp2 4.08 / ilp1 6.70 / ilp1+bn16 6.93
    #     L8K  PMQ(MALL) 15.83  vs  kvlds ilp2 6.22
    # We fixed the V read (transposed+pad -> 2 coalesced vec8, NOT 16 scalar
    # loads), kept O in registers (zero O-LDS traffic, unlike kvstat), cut spill
    # to 52 (bn16), and pad6 == pad2 (bank conflicts are not the dominant cost).
    # It STILL loses ~2x. Root cause = the 2 sync_lds_only barriers PER K-tile the
    # coop-load requires (produce-KV -> consume): with only W=2 waves + pingpong,
    # that barrier re-synchronizes the two waves EVERY tile, destroying the
    # pingpong out-of-phase overlap (the 2.25x scheduling lever). The MALL gather
    # path needs NO barrier -- each wave gathers independently and the hardware
    # dedups the cross-wave re-reads in MALL for free. So cross-wave LDS sharing
    # fundamentally fights pingpong on this 2-wave APU; the halved-MALL-read it
    # buys never offsets the lost wave overlap + the transposed scatter store.
    # 4th confirmation of the gfx1151 LDS lesson, now in the O-in-registers /
    # coalesced-V / persistent structure. Kept OFF for the record.
    #
    # FOLLOW-UP (double-buffered, 1 barrier/tile): converting to 2 KV buffers
    # (write kt%2, no WAR barrier -- only the RAW barrier remains) made it WORSE,
    # not better: L16K bn32/ilp2 4.08->3.23, bn16/ilp1 6.93->5.68. Two reasons:
    # (1) the dynamic buf=kt%2 index defeats static LDS-offset folding (+addr ALU);
    # (2) the barrier COUNT was never the bottleneck -- the single remaining RAW
    # barrier still forces a per-tile CROSS-WAVE RENDEZVOUS (both waves must
    # arrive), and THAT is what breaks pingpong's deliberate out-of-phase overlap.
    # s_waitcnt cannot replace it (per-wave, no cross-wave visibility). Register
    # reuse doesn't help either: bn16/ilp1 already spills only 52-72 yet still
    # ~half of PMQ. The wall is the rendezvous<->pingpong conflict, irreducible
    # for shared LDS on a 2-wave CTA. Definitively closed.
    kv_lds: bool = False
    # kv_pad: f16 pad on the transposed-V kv row. Chosen so (block_n+kv_pad)/2 is
    # ODD -> the 16 per-lane d-strided reads start on distinct LDS banks
    # (swizzle-by-pad). Measure LDSBankConflict and tune.
    kv_pad: int = 2
    name: str = "wmma_fmha_swapqk_pmq"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32 * self.n_waves

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.n_waves * self.q_block

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"w{self.n_waves}",
            f"qb{self.q_block}",
            "of16" if self.o_f16 else "of32",
            f"bn{self.block_n}",
            f"ilp{self.qk_ilp}",
            self.sched_mode,
            "buf" if self.buffer_gather else "flat",
            "dual" if self.dual_gather else "single",
            "fexp" if self.fast_exp2 else "iexp",
            f"pers{self.num_persistent}",
            f"hb{self.num_cohorts}" if self.head_blocked else self.persist_decode,
            f"kvlds{self.kv_pad}" if self.kv_lds else "kvglob",
        )


def num_work_items(cfg: SwapQKPersistentCfg, *, seqlen_q: int, batch: int) -> int:
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per) * cfg.num_query_heads * batch


def swapqk_pmq_grid(cfg: SwapQKPersistentCfg):
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


def build_wmma_fmha_swapqk_pmq(
    cfg: SwapQKPersistentCfg,
    arch: str = "gfx1151",
    *,
    num_q_blocks: int,
    batch: int,
) -> KernelDef:
    """Build the persistent + MQ swapqk kernel.

    ``num_q_blocks`` (= seqlen_q // q_rows_per_cta) and ``batch`` are baked at build
    time so the tile decode uses compile-time constant divs/mods and the scf.for
    trip count (``max_iters``) is exact. The driver rebuilds per shape.
    """
    if cfg.num_persistent <= 0:
        raise ValueError(f"num_persistent must be > 0, got {cfg.num_persistent}")
    if cfg.persist_decode not in ("qb_major", "batch_major"):
        raise ValueError(f"bad persist_decode {cfg.persist_decode!r}")
    if cfg.q_block < 1:
        raise ValueError(f"q_block must be >= 1, got {cfg.q_block}")

    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    c_frag = atom.c_per_lane  # 8
    a_frag = atom.a_per_lane  # 16
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.n_waves
    MQ = cfg.q_block
    dtype_ir = F16

    qh, kvh = cfg.num_query_heads, cfg.kv_heads
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
    c2 = b.const_i32(2)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    col = b.mod(lane, c16)  # lane % 16  == query row within the 16-tile
    lane_lt16 = b.cmp_lt(lane, c16)

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    sq, sqh = p["stride_q_token"], p["stride_q_head"]
    sk, skh = p["stride_k_token"], p["stride_k_head"]
    sv, svh = p["stride_v_token"], p["stride_v_head"]
    so, soh = p["stride_o_token"], p["stride_o_head"]
    scale_log2 = p["scale_log2"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741
    Counter = p["Counter"]

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)
    _exp2 = b.exp2_fast if cfg.fast_exp2 else b.exp2
    pingpong = cfg.sched_mode == "pingpong"

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
    # O^T view: (head, dim, token) so store_wmma_tile's (row=d, col=query) lands on
    # O[query, d]. dim is contiguous (stride 1), token strided (so).
    O_T_view = make_global_view(
        O, shape=(qh, hs, 1), dtype=dtype_ir, strides=(soh, 1, so)
    )

    # ---- CK PermuteWarpGemmCToA: C-dist P (8 f16) -> operand fragment (16 f16). ----
    sel0 = b.select(lane_lt16, b.const_i32(0x05040100), b.const_i32(0x01000504))
    sel1 = b.select(lane_lt16, b.const_i32(0x07060302), b.const_i32(0x03020706))

    def permx16_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def p_transpose_reg(ps):
        outs = []
        for m in range(c_frag // 2):
            lo = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m], dtype_ir), I16), I32)
            hi = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m + 1], dtype_ir), I16), I32)
            v = b.lor(lo, b.shl(hi, c16))
            w = b.permlanex16(v)
            outs.append(b.perm_b32(w, v, sel0))
            outs.append(b.perm_b32(w, v, sel1))
        packed = b.vec_pack(outs, I32)
        return b.vec_bitcast(packed, VectorType(dtype_ir, a_frag))

    def _tree(vals, op):
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    block_n = cfg.block_n
    n_kv_sub = block_n // 16  # 16-wide kv WMMA sub-tiles per K-loop iteration
    c_block_n = b.const_i32(block_n)
    n_i32 = a_frag // 2  # 8 dwords per <16 x f16> fragment
    ilp = max(1, cfg.qk_ilp)
    gs = 2 + n_dk  # iter-arg stride per group: m, l, n_dk O tiles

    # ---- buffer-descriptor gather (tile-independent bits) ----
    if cfg.buffer_gather and not cfg.kv_lds:
        v_rsrc = b.buffer_rsrc(V, b.const_i32(0x7FFFFFFF))
        sv2 = b.mul(sv, c2)  # bytes per kv step (loop-invariant)
        soff_list = [b.mul(b.const_i32(j), sv2) for j in range(a_frag)]  # hoisted

    # ---- kv_lds: shared LDS KV tile (K row-major, V transposed + pad-swizzled) ----
    # DOUBLE-BUFFERED: 2 KV buffers indexed by kt%2. Writing buffer (kt%2) never
    # collides with the buffer read this iter or last, so the WAR barrier is
    # eliminated -- only ONE RAW barrier/tile remains (cross-wave visibility of
    # the cooperative write, which s_waitcnt cannot provide).
    if cfg.kv_lds:
        _KPAD = 8  # K row-major bank pad (d-consecutive reads)
        c8 = b.const_i32(8)
        c_hs = b.const_i32(hs)
        K_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(2, block_n, hs),
            strides=(block_n * (hs + _KPAD), hs + _KPAD, 1),
            name_hint="Ksh",
        )
        # V^T[d, kv]: row=d, col=kv. (block_n+kv_pad) chosen so /2 is odd -> the
        # 16 per-lane d-strided row starts land on distinct banks.
        _vstride = block_n + cfg.kv_pad
        V_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(2, hs, block_n),
            strides=(hs * _vstride, _vstride, 1),
            name_hint="Vsh",
        )
        _nthreads = wave * W
        if (block_n * hs) % (_nthreads * 8) != 0:
            raise ValueError(
                f"kv_lds coop loader needs block_n*hs ({block_n * hs}) divisible "
                f"by n_threads*8 ({_nthreads * 8})"
            )
        _kv_chunks = (block_n * hs) // (_nthreads * 8)

        def k_lds_read(buf, ns, d):
            row = b.add(b.const_i32(ns * 16), col)
            lo = K_lds.load_vec(b, [buf, row, b.const_i32(d * 16)], n=8)
            hi = K_lds.load_vec(b, [buf, row, b.const_i32(d * 16 + 8)], n=8)
            return WmmaTensor(atom, "a", b.vec_concat(lo, hi), arch)

        def v_lds_read(buf, ns, d):
            # V^T[buf, d_col, ns*16 + 0..15]: 16 consecutive kv (stride 1) -> 2 vec8.
            d_col = b.add(b.const_i32(d * 16), col)
            kv0 = b.const_i32(ns * 16)
            lo = V_lds.load_vec(b, [buf, d_col, kv0], n=8)
            hi = V_lds.load_vec(b, [buf, d_col, b.add(kv0, c8)], n=8)
            return WmmaTensor(atom, "a", b.vec_concat(lo, hi), arch)

    # ======================================================================
    # do_work: process ONE (q_group, head, batch) work-item = a CTA-tile of
    # W waves x MQ query 16-row groups. Byte-for-byte the swapqk MQ body.
    # ======================================================================
    def do_work(q_group, head, batch_i):
        kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))
        batch_tok_q = b.mul(batch_i, seqlen_q)
        batch_tok_k = b.mul(batch_i, seqlen_k)
        kvh_off = b.mul(kv_head, svh) if cfg.buffer_gather else None

        cta_row0 = b.mul(q_group, b.const_i32(cfg.q_rows_per_cta))
        wave_base = b.add(cta_row0, b.mul(wave_id, b.const_i32(16 * MQ)))
        q_pos_base_g = [b.add(wave_base, b.const_i32(g * 16)) for g in range(MQ)]
        q_token_base_g = [b.add(qpb, batch_tok_q) for qpb in q_pos_base_g]
        qwin_g = [
            make_tile_window(Q_view, (1, 16, hs), origin=(head, qtb, c0))
            for qtb in q_token_base_g
        ]

        loop_stop = b.div(seqlen_k, c_block_n)
        if cfg.mask_mode == "causal":
            causal_stop = b.add(
                b.div(b.add(cta_row0, b.const_i32(16 * W)), c_block_n), b.const_i32(1)
            )
            loop_stop = b.select(
                b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop
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

        def _load_col(k_base, vwin, d_col):
            if cfg.buffer_gather:
                elem0 = b.add(b.add(kvh_off, b.mul(k_base, sv)), d_col)
                voff = b.mul(elem0, c2)

                def _load(j):
                    return b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j])

            else:

                def _load(j):
                    return vwin.load_scalar(b, c0, b.const_i32(j), d_col)

            v_a = b.undef_vec(dtype_ir, a_frag)
            for j in range(a_frag):
                v_a = b.vec_insert(v_a, _load(j), j)
            return v_a

        def dual_gather(k_base, vwin, d):
            d_col = b.add(b.const_i32(d * 16), b.add(b.mul(b.div(lane, c16), c16), col))
            loaded = _load_col(k_base, vwin, d_col)
            li = b.vec_bitcast(loaded, VectorType(I32, n_i32))
            fd, fd1 = [], []
            for i in range(n_i32):
                e = b.vec_extract(li, i)
                pp = b.permlanex16(e)
                fd.append(b.select(lane_lt16, e, pp))
                fd1.append(b.select(lane_lt16, pp, e))
            frag_d = b.vec_bitcast(b.vec_pack(fd, I32), VectorType(dtype_ir, a_frag))
            frag_d1 = b.vec_bitcast(b.vec_pack(fd1, I32), VectorType(dtype_ir, a_frag))
            return frag_d, frag_d1

        def coop_load_kv(buf, k_block_base):
            # all W waves cooperatively stream this KV tile global -> LDS[buf] once.
            # NO leading WAR barrier: double-buffering means buf (=kt%2) was last
            # touched 2 iters ago, provably drained by the intervening RAW barrier.
            kbase_tok = b.add(batch_tok_k, k_block_base)
            for i in range(_kv_chunks):
                cc = b.add(tid, b.const_i32(i * _nthreads))
                base = b.mul(cc, c8)
                row = b.div(base, c_hs)  # kv row within tile
                colc = b.mod(base, c_hs)  # d start (8 consecutive)
                gtok = b.add(kbase_tok, row)
                k8 = K_view.load_vec(b, [kv_head, gtok, colc], n=8)
                K_lds.store_vec(b, [buf, row, colc], k8, 8)  # row-major (coalesced)
                v8 = V_view.load_vec(b, [kv_head, gtok, colc], n=8)
                # transposed store: v8[j] = V[row, colc+j] -> V^T[colc+j, row].
                for j in range(8):
                    V_lds.store_scalar(
                        b, [buf, b.add(colc, b.const_i32(j)), row], b.vec_extract(v8, j)
                    )
            b.sync_lds_only()  # single RAW barrier: staged tile visible to all waves

        # ---- iter-args: per group (m, l, n_dk O tiles) ----
        qb_iter = []
        for g in range(MQ):
            qb_iter.append((f"qm{g}", neg_inf))
            qb_iter.append((f"ql{g}", zero_f))
            for d in range(n_dk):
                o0 = b.zero_vec(dtype_ir, c_frag) if cfg.o_f16 else atom.zero_acc(b)
                qb_iter.append((f"qo{g}_{d}", o0))

        kloop = b.scf_for_iter(
            b.const_i32(0), loop_stop, b.const_i32(1), iter_args=qb_iter, iv_name="kt"
        )
        with kloop as (kt, state):
            m_i = [state[g * gs] for g in range(MQ)]
            l_i = [state[g * gs + 1] for g in range(MQ)]
            accs = [list(state[g * gs + 2 : g * gs + 2 + n_dk]) for g in range(MQ)]
            k_block_base = b.mul(kt, c_block_n)
            k_bases = [
                b.add(b.add(batch_tok_k, k_block_base), b.const_i32(ns * 16))
                for ns in range(n_kv_sub)
            ]
            vwins = [
                v_window(b.add(k_block_base, b.const_i32(ns * 16)))
                for ns in range(n_kv_sub)
            ]

            # kv_lds: cooperatively stage this KV tile into shared LDS[buf] first.
            buf = b.mod(kt, c2) if cfg.kv_lds else None
            if cfg.kv_lds:
                coop_load_kv(buf, k_block_base)

            # ---- QK: load each K fragment ONCE, reuse across MQ query groups ----
            if pingpong:
                b.s_setprio(1)
            subs = [[None] * n_kv_sub for _ in range(MQ)]
            for ns in range(n_kv_sub):
                kwin = (
                    None
                    if cfg.kv_lds
                    else k_window(b.add(k_block_base, b.const_i32(ns * 16)))
                )
                acc = [
                    [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
                    for _ in range(MQ)
                ]
                for d in range(n_dk):
                    k_tile = (
                        k_lds_read(buf, ns, d)
                        if cfg.kv_lds
                        else load_wmma_tile(
                            b, kwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                        )
                    )
                    for g in range(MQ):
                        q_tile = load_wmma_tile(
                            b,
                            qwin_g[g],
                            atom,
                            lane,
                            role="b",
                            k_offset=d * 16,
                            lead=[c0],
                        )
                        acc[g][d % ilp] = wmma_mma(b, k_tile, q_tile, acc[g][d % ilp])
                for g in range(MQ):
                    sc = acc[g][0]
                    for si in range(1, ilp):
                        sc = WmmaTensor(
                            atom, "c", b.vector_add(sc.value, acc[g][si].value), arch
                        )
                    subs[g][ns] = sc
            if pingpong:
                b.s_setprio(0)

            # ---- per-group online softmax (eager) + register P-transpose ----
            p_tiles = [None] * MQ
            alpha_vec = [None] * MQ
            m_new = [None] * MQ
            l_new = [None] * MQ
            for g in range(MQ):
                s_sub = []
                for ns in range(n_kv_sub):
                    kv_base = b.add(k_block_base, b.const_i32(ns * 16))
                    row = []
                    for i in range(c_frag):
                        kv_rel, q_rel = subs[g][ns].coord(b, lane, i)
                        s_i = b.fmul(subs[g][ns].slot(b, i), scale_log2)
                        s_i = apply_attention_mask(
                            b,
                            s_i,
                            mask_mode=cfg.mask_mode,
                            k_idx=b.add(kv_base, kv_rel),
                            query_pos=b.add(q_pos_base_g[g], q_rel),
                            sliding_window=0,
                        )
                        row.append(s_i)
                    s_sub.append(row)
                all_s = [v for r in s_sub for v in r]
                local_max = _tree(list(all_s), b.fmax)
                tile_max = b.fmax(local_max, permx16_f32(local_max))
                mn = b.fmax(m_i[g], tile_max)
                al = _exp2(b.fsub(m_i[g], mn))
                ps = [
                    [_exp2(b.fsub(s_sub[ns][i], mn)) for i in range(c_frag)]
                    for ns in range(n_kv_sub)
                ]
                all_p = [v for r in ps for v in r]
                local_sum = _tree(list(all_p), b.fadd)
                tile_sum = b.fadd(local_sum, permx16_f32(local_sum))
                m_new[g] = mn
                l_new[g] = b.fadd(b.fmul(l_i[g], al), tile_sum)
                av = b.zero_vec_f32(c_frag)
                for i in range(c_frag):
                    av = b.vec_insert(av, al, i)
                alpha_vec[g] = av
                p_tiles[g] = [
                    WmmaTensor(atom, "b", p_transpose_reg(ps[ns]), arch)
                    for ns in range(n_kv_sub)
                ]

            # ---- PV: rescale O[g] by alpha[g], share each V fragment across groups ----
            if pingpong:
                b.s_setprio(1)
            new_accs = [[None] * n_dk for _ in range(MQ)]
            if cfg.o_f16:
                for dp in range(0, n_dk, 2):
                    t0 = [
                        WmmaTensor(
                            atom,
                            "c",
                            b.vector_mul(b.vec_ext_to_f32(accs[g][dp]), alpha_vec[g]),
                            arch,
                        )
                        for g in range(MQ)
                    ]
                    t1 = [
                        WmmaTensor(
                            atom,
                            "c",
                            b.vector_mul(
                                b.vec_ext_to_f32(accs[g][dp + 1]), alpha_vec[g]
                            ),
                            arch,
                        )
                        for g in range(MQ)
                    ]
                    for ns in range(n_kv_sub):
                        if cfg.kv_lds:
                            a0 = v_lds_read(buf, ns, dp)
                            a1 = v_lds_read(buf, ns, dp + 1)
                        else:
                            frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                            a0 = WmmaTensor(atom, "a", frag_d, arch)
                            a1 = WmmaTensor(atom, "a", frag_d1, arch)
                        for g in range(MQ):
                            t0[g] = wmma_mma(b, a0, p_tiles[g][ns], t0[g])
                            t1[g] = wmma_mma(b, a1, p_tiles[g][ns], t1[g])
                    for g in range(MQ):
                        new_accs[g][dp] = b.vec_trunc_f32_to_f16(t0[g].value)
                        new_accs[g][dp + 1] = b.vec_trunc_f32_to_f16(t1[g].value)
            else:
                new_accs = [
                    [
                        WmmaTensor(atom, "c", accs[g][d], arch).scale(b, alpha_vec[g])
                        for d in range(n_dk)
                    ]
                    for g in range(MQ)
                ]
                for ns in range(n_kv_sub):
                    for dp in range(0, n_dk, 2):
                        if cfg.kv_lds:
                            a0 = v_lds_read(buf, ns, dp)
                            a1 = v_lds_read(buf, ns, dp + 1)
                        else:
                            frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                            a0 = WmmaTensor(atom, "a", frag_d, arch)
                            a1 = WmmaTensor(atom, "a", frag_d1, arch)
                        for g in range(MQ):
                            new_accs[g][dp] = wmma_mma(
                                b, a0, p_tiles[g][ns], new_accs[g][dp]
                            )
                            new_accs[g][dp + 1] = wmma_mma(
                                b, a1, p_tiles[g][ns], new_accs[g][dp + 1]
                            )
            if pingpong:
                b.s_setprio(0)

            yields = []
            for g in range(MQ):
                yields.append(m_new[g])
                yields.append(l_new[g])
                if cfg.o_f16:
                    yields.extend(new_accs[g])
                else:
                    yields.extend(a.value for a in new_accs[g])
            b.scf_yield(*yields)

        res = kloop.results
        for g in range(MQ):
            l_f = res[g * gs + 1]
            inv_l = b.select(b.fcmp("oeq", l_f, zero_f), zero_f, b.rcp(l_f))

            def _rescale(bld, val, slot, row, colv, _inv=inv_l):
                return bld.fmul(val, _inv)

            accs_g = res[g * gs + 2 : g * gs + 2 + n_dk]
            for d in range(n_dk):
                owin = make_tile_window(
                    O_T_view,
                    (1, 16, 16),
                    origin=(head, b.const_i32(d * 16), q_token_base_g[g]),
                )
                ov = b.vec_ext_to_f32(accs_g[d]) if cfg.o_f16 else accs_g[d]
                store_wmma_tile(
                    b,
                    owin,
                    WmmaTensor(atom, "c", ov, arch),
                    lane,
                    col_offset=0,
                    lead=[c0],
                    align=2,
                    transform=_rescale,
                )

    # ======================================================================
    # scheduling
    # ======================================================================
    c_num_tiles = b.const_i32(num_tiles)
    c_nqb = b.const_i32(num_q_blocks)
    c_qh = b.const_i32(qh)
    c_one = b.const_i32(1)

    if cfg.head_blocked:
        # ---- static cohort partition (no atomics, no barrier) ----
        G = cfg.num_cohorts
        if cfg.num_persistent % G != 0:
            raise ValueError(
                f"num_persistent ({cfg.num_persistent}) must be divisible by "
                f"num_cohorts ({G})"
            )
        S = cfg.num_persistent // G  # CTAs per cohort
        HB = qh * batch  # head-batch units
        c_G = b.const_i32(G)
        c_S = b.const_i32(S)
        c_HB = b.const_i32(HB)
        bid = b.block_id_x()
        cohort = b.div(bid, c_S)  # 0..G-1
        lane_c = b.mod(bid, c_S)  # 0..S-1  (CTA within cohort)

        # hb = cohort, cohort+G, ...  < HB   (this cohort's head-batch units)
        hbloop = b.scf_for_iter(cohort, c_HB, c_G, iter_args=[], iv_name="hb")
        with hbloop as _hb:
            head = b.mod(_hb, c_qh)
            batch_i = b.div(_hb, c_qh)
            # qg = lane_c, lane_c+S, ... < NQB   (CTAs split this head's q_blocks)
            qgloop = b.scf_for_iter(lane_c, c_nqb, c_S, iter_args=[], iv_name="qg")
            with qgloop as _qg:
                do_work(_qg, head, batch_i)
        b.ret()
        return b.kernel

    # ---- atomic persistent work-queue (default) ----
    multiwave = cfg.block_size > wave
    brd = b.smem_alloc(I32, [1], name_hint="pmq_brd") if multiwave else None

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

    def fetch_tile():
        """Cooperative atomic tile fetch broadcast to every thread in the CTA."""
        is_lead = b.cmp_eq(tid, c0)
        if not multiwave:
            # single wave: every lane issues the atomic (only lane 0 increments),
            # then ds_bpermute(0) broadcasts lane 0's result wave-internally.
            inc = b.select(is_lead, c_one, c0)
            fetched = b.global_atomic_add(Counter, c0, inc)
            return b.ds_bpermute(c0, fetched)
        # multi-wave: lead thread does the atomic + LDS store, then ONE trailing
        # LDS-only barrier publishes the value to every wave. Crucially this is
        # sync_lds_only (lgkmcnt + s_barrier), NOT a full vmcnt drain -- the
        # previous work-item's V-gathers / O-stores keep flowing across the
        # boundary (the barrier only orders the LDS broadcast).
        with b.scf_if(is_lead):
            v = b.global_atomic_add(Counter, c0, c_one)
            b.smem_store_vN(brd, [c0], v, 1)
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
