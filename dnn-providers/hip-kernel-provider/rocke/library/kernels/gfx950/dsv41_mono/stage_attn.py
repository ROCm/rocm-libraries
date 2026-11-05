# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2a's decode attention of ATOM's DeepSeek-V4.1 mono as rocKE stage functions (W15 PORT K2-attn-gemv).

Bit-identical re-implementations of ATOM ``attention.py`` / ``attn_post.py`` (origin/main df86c482), TP2 (32 heads a
rank, two 16-head tiles, not ragged):

* ``score`` = ``stage_attn_score(c, task, per)``: task -> (token t, head tile, task-in-tile sg); ``await_selection``
  (an indexer build: the token's ``irdy`` pair when ``iact``); the token's key count (``kmeta`` first / count / base,
  ``sbound`` at device scope -> n_sel = min(max(bound, 0), 512)); each of the wave's splits ``(sg + per k) 8 + wave``:
  the 16 key rows (``_key_row``: the selection's rows then the window ring, oldest first) by one 16-B LDS DMA a lane a
  row into the wave's LDS (row stride 260 words), the next split's rows in flight under a split's MFMAs; 16
  ``v_mfma_f32_16x16x32_bf16`` (keys x the q of the tile's 16 heads); scale, the split's max (butterfly 32, 16), exp2,
  the sum ``(p0 + p2) + (p1 + p3)`` + xor 32 + xor 16; ``am`` / ``al`` puts (lanes 0..15), the bf16 p pair puts (``ap``).
* ``irq`` = ``stage_attn_irq(c, task, groups)``: the RoPE table loads of the task's column groups first; then
  ``stage_attn_pv``: the 8 splits' key-row quarter loads (64-bit addresses), the sink, wave 0 polls the split stats,
  barrier, one 12-pair poll (stats + the 8 splits' p), the stats (m_max butterfly, exp2 alpha, the ``_l_tree`` row-shift
  / bpermute sum of l x alpha), barrier, then per column group: the split tiles through LDS, ``v_mfma_f32_16x16x16_bf16``
  (keys x p), the alpha-weighted pair fma tree, barrier, the 8-wave sum, the sink combine (``m_final``, ``alpha_kv`` /
  ``alpha_sink``, ``o = tot alpha_kv / max(l, 1e-30)``), bf16 -> ``out``; barrier; ``irq_group``: inverse RoPE, the
  16-thread amax (floor 1e-8), the ceil E8M0 code, x pow2(254 - code) clamped, RNE E4M3, word / code puts (``xo`` /
  ``xos``).

ABI (``lds="arg"``; LDS operands are word offsets from ``lds_base``)::

    score: rocke_dsv41_attn_p<per>x<ix>n<S>
      void (i32 tag, i32 task, i32 iact, i64 q, i64 pool, i64 sel, i32 topk, i64 kmeta, i64 sbound, i32 ring_off,
            i32 ring_slots, i32 qk_scale, i64 irdy, i64 am, i64 al, i64 ap, i32 kv, i32 lds_base)
    irq:   rocke_dsv41_irq_g<groups>x<ix>n<S>
      void (i32 tag, i32 task, i32 iact, i64 pool, i64 sel, i32 topk, i64 kmeta, i64 sbound, i32 ring_off,
            i32 ring_slots, i64 sink, i64 pos, i64 cos, i64 sin, i64 irdy, i64 am, i64 al, i64 ap, i64 xo, i64 xos,
            i32 tree, i32 alpha, i32 stat, i32 out, i32 kv, i32 lds_base)

Knobs (:class:`AttnKnobs`): :class:`seam_common.SeamKnobs` plus ``askip`` (score: no DMA for a split past the token's
keys, ATOM's W13 bitwise lever), ``stat_w0``
(1 = ATOM: wave 0 polls the stats before the CTA; 0: every wave polls at once). Mutants: ``put_bit`` (score: ap word;
irq: xo word), ``key`` (score: the split's keys shifted by one row), ``alpha`` (irq: alpha = 1), ``sink`` (irq: no sink).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rocke.core.ir import BF16, F32, I16, I32, I64, IRBuilder, PtrType, VectorType

from . import seam_common as S

HEAD_DIM = 512
HEADS = 32  # Dims(2).heads
HEAD_TILE = 16
TILES = HEADS // HEAD_TILE  # 2
SPLITS = 64
BK = 16
SCORE_WAVES = 8
SCORE_LDS_ROW = HEAD_DIM // 2 + 4  # 260 words
SCORE_LDS_WORDS = BK * SCORE_LDS_ROW
COLS = 32
GROUPS = HEAD_DIM // COLS  # 16
ROW_BYTES = HEAD_DIM * 2
TOPK_SEL = 512  # index_plan.TOPK
ROPE = 64
HALF = ROPE // 2
KMETA = 4
NEG = -3.4028234663852886e38
IRQ_FLOOR = 1e-8
XO_WORDS = HEADS * HEAD_DIM // 4
MUTANTS = ("put_bit", "key", "alpha", "sink")
CPOL_DEV = 16

ABI_SCORE = (
    ("tag", "i32"), ("task", "i32"), ("iact", "i32"), ("q", "i64"), ("pool", "i64"), ("sel", "i64"), ("topk", "i32"),
    ("kmeta", "i64"), ("sbound", "i64"), ("ring_off", "i32"), ("ring_slots", "i32"), ("qk_scale", "i32"),
    ("irdy", "i64"), ("am", "i64"), ("al", "i64"), ("ap", "i64"), ("kv", "i32"),
)  # fmt: skip
ABI_IRQ = (
    ("tag", "i32"), ("task", "i32"), ("iact", "i32"), ("pool", "i64"), ("sel", "i64"), ("topk", "i32"),
    ("kmeta", "i64"), ("sbound", "i64"), ("ring_off", "i32"), ("ring_slots", "i32"), ("sink", "i64"), ("pos", "i64"),
    ("cos", "i64"), ("sin", "i64"), ("irdy", "i64"), ("am", "i64"), ("al", "i64"), ("ap", "i64"), ("xo", "i64"),
    ("xos", "i64"), ("tree", "i32"), ("alpha", "i32"), ("stat", "i32"), ("out", "i32"), ("kv", "i32"),
)  # fmt: skip


@dataclass(frozen=True)
class AttnKnobs(S.SeamKnobs):
    askip: int = 0
    stat_w0: int = 1

    def __post_init__(self) -> None:
        self.check(MUTANTS)


def score_tasks(s: int) -> int:
    per = SPLITS // 8
    while s * TILES * per > S.BLOCKS and per > 1:
        per //= 2
    return per


def pv_groups(s: int) -> int:
    g = 1
    while s * TILES * GROUPS // g > S.BLOCKS and g < GROUPS:
        g *= 2
    return g


class _A(S.SeamEnv):
    """The attention stages' shared pieces (``_keys``, ``_key_row``, ``await_selection``)."""

    def setup(self, p, ix):
        self.p, self.ix = p, ix

    def tile_task(self, task, per):
        b, c = self.b, self.c
        return b.div(task, c(per * TILES)), b.mod(b.div(task, c(per)), c(TILES)), b.mod(task, c(per))

    def ld_i32(self, addr, i, aux=0):
        b = self.b
        return b.buffer_load_cp(self.rsrc(addr), b.mul(i, self.c(4)), self.c(0), dwords=1, aux=aux)

    def await_selection(self, tag, t):
        if not self.ix:
            return
        b, c = self.b, self.c
        with b.scf_if(b.cmp_ne(b.readfirstlane(self.p["iact"]), c(0))):
            self.poll(tag, [(self.rsrc(self.p["irdy"]), b.readfirstlane(self.p["irdy"]), t)])

    def keys(self, t):
        b, c, p = self.b, self.c, self.p
        first, count, base = (self.ld_i32(p["kmeta"], b.add(b.mul(t, c(KMETA)), c(i))) for i in (0, 1, 3))
        bound = self.ld_i32(p["sbound"], t, aux=CPOL_DEV)
        n_sel = b.smin(b.smax(bound, c(0)), c(TOPK_SEL))
        return b.add(n_sel, count), (n_sel, first, base)

    def key_row(self, t, meta, key):
        b, c, p = self.b, self.c, self.p
        n_sel, first, base = meta
        key = b.smax(key, c(0))
        picked = b.cmp_lt(key, n_sel)
        topk = b.readfirstlane(p["topk"])
        sel_i = b.add(b.mul(t, topk), b.smin(key, b.sub(topk, c(1))))
        selected = b.select(picked, self.ld_i32(p["sel"], sel_i, aux=CPOL_DEV), c(0))
        ring = b.readfirstlane(p["ring_slots"])
        window = b.add(b.add(base, b.readfirstlane(p["ring_off"])), b.mod(b.sub(b.add(first, key), n_sel), ring))
        return b.select(picked, selected, window)

    def written(self, kv_len):
        b, c = self.b, self.c
        return b.smax(b.div(b.add(kv_len, c(BK - 1)), c(BK)), c(1))



def build_score_kernel(rows: int, per: int, ix: int, knobs: AttnKnobs = AttnKnobs()):
    s = rows
    b = IRBuilder(f"rocke_dsv41_attn_p{per}x{ix}n{s}")
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if ty == "i32" else I64) for n, ty in ABI_SCORE}
    e = _A(b, knobs)
    e.setup(p, ix)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    kvw = b.readfirstlane(p["kv"])
    lane, wave = e.lane, e.wave
    g = b.lshr(lane, c(4))
    head = b.land(lane, c(15))
    t, tile, sg = e.tile_task(task, per)
    qh = b.add(b.mul(tile, c(HEAD_TILE)), head)
    e.await_selection(tag, t)
    kv_len, meta = e.keys(t)
    nsp = SPLITS // SCORE_WAVES // per
    ss = [b.add(b.mul(b.add(sg, c(per * k)), c(SCORE_WAVES)), wave) for k in range(nsp)]
    koff = 1 if knobs.mutate == "key" else 0
    slots = [e.key_row(t, meta, b.smin(b.add(b.add(b.mul(sp, c(BK)), head), c(koff)), b.sub(kv_len, c(1)))) for sp in ss]
    r_q = e.rsrc(p["q"])
    qws = [b.buffer_load_cp(r_q, b.mul(b.lshr(b.add(b.add(b.mul(b.add(b.mul(t, c(HEADS)), qh), c(HEAD_DIM)), c(ch * 32)),
                                                     b.mul(g, c(8))), c(1)), c(4)), c(0), dwords=4)
           for ch in range(HEAD_DIM // 32)]  # fmt: skip
    kv = b.add(kvw, b.mul(wave, c(SCORE_LDS_WORDS)))
    pool_lds = b.smem_addr_of(e.pool)
    nw = e.written(kv_len)

    def rows_to_lds(slot):
        for r in range(BK):
            row = b.readlane(slot, c(r))
            rs = b.buffer_rsrc(b.int_to_ptr(b.add(p["pool"], b.mul(b.sext(row, I64), b.const_i64(ROW_BYTES))),
                                            PtrType(I32, "global")), c(0x7FFFFFF0))  # fmt: skip
            at = b.smem_ptr_add(pool_lds, b.sext(b.mul(b.add(kv, c(r * SCORE_LDS_ROW)), c(4)), I64))
            b.async_buffer_load_lds_addr(rs, at, b.mul(lane, c(16)), c(0), dwords=4)

    def rows_from_lds():
        at = b.add(b.add(kv, b.mul(head, c(SCORE_LDS_ROW))), b.mul(g, c(4)))
        return [b.smem_load_vN(e.pool, b.add(at, c(ch * 16)), dtype=I32, n=4) for ch in range(HEAD_DIM // 32)]

    def score_split(sp, kws):
        with b.scf_if(b.cmp_lt(sp, nw)):
            key = b.add(b.mul(sp, c(BK)), head)
            ok = b.cmp_lt(key, kv_len)
            sc = b.vec_pack([e.f(0.0)] * 4, F32)
            for ch in range(HEAD_DIM // 32):
                kw = b.vec_pack([b.select(ok, b.vec_extract(kws[ch], q), c(0)) for q in range(4)], I32)
                sc = b.mfma_f32_16x16x32_bf16(b.vec_bitcast(kw, VectorType(BF16, 8)), b.vec_bitcast(qws[ch], VectorType(BF16, 8)), sc)
            scale = b.bitcast(b.readfirstlane(p["qk_scale"]), F32)
            sv = [b.select(b.cmp_lt(b.add(b.add(b.mul(sp, c(BK)), b.mul(g, c(4))), c(i)), kv_len),
                           b.fmul(b.vec_extract(sc, i), scale), e.f(NEG)) for i in range(4)]  # fmt: skip
            m = e.butterfly(b.fmax(b.fmax(sv[0], sv[1]), b.fmax(sv[2], sv[3])), (32, 16), b.fmax)
            pv = [b.exp2_fast(b.fsub(sv[i], m)) for i in range(4)]
            ssum = b.fadd(b.fadd(pv[0], pv[2]), b.fadd(pv[1], pv[3]))
            ssum = e.xred(ssum, 32, b.fadd)
            ssum = e.xred(ssum, 16, b.fadd)
            with b.scf_if(b.cmp_eq(g, c(0))):
                ms = b.add(b.mul(b.add(b.mul(t, c(SPLITS)), sp), c(HEADS)), qh)
                e.put(tag, p["am"], ms, b.bitcast(m, I32))
                e.put(tag, p["al"], ms, b.bitcast(ssum, I32))
            w0 = b.lor(e.bf16_bits(pv[0]), b.shl(e.bf16_bits(pv[1]), c(16)))
            w1 = b.lor(e.bf16_bits(pv[2]), b.shl(e.bf16_bits(pv[3]), c(16)))
            if knobs.mutate == "put_bit":
                w0 = b.xor(w0, c(1))
            api = b.add(b.mul(b.add(b.mul(b.add(b.mul(t, c(TILES)), tile), c(SPLITS)), sp), c(64)), lane)
            b.buffer_store_cp(b.vec_pack([w0, tag, w1, tag], I32), e.rsrc(p["ap"]), b.mul(api, c(16)), c(0), aux=S.CPOL_SC1)

    def dma(k):
        if knobs.askip:
            with b.scf_if(b.cmp_lt(ss[k], nw)):
                rows_to_lds(slots[k])
        else:
            rows_to_lds(slots[k])

    dma(0)
    for k in range(nsp):
        b.s_waitcnt(vmcnt=0)
        kws = rows_from_lds()
        if k + 1 < nsp:
            b.s_waitcnt(lgkmcnt=0)
            dma(k + 1)
        score_split(ss[k], kws)
    b.ret()
    return b.kernel


def build_irq_kernel(rows: int, groups: int, ix: int, knobs: AttnKnobs = AttnKnobs()):
    s = rows
    b = IRBuilder(f"rocke_dsv41_irq_g{groups}x{ix}n{s}")
    b.kernel.attrs["max_workgroup_size"] = S.THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if ty == "i32" else I64) for n, ty in ABI_IRQ}
    e = _A(b, knobs)
    e.setup(p, ix)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    tree, alpha_l, stat, out_l, kvw = (b.readfirstlane(p[n]) for n in ("tree", "alpha", "stat", "out", "kv"))
    tid, lane, wave = e.tid, e.lane, e.wave
    g = b.lshr(lane, c(4))
    head16 = b.land(lane, c(15))
    t, tile, g0 = e.tile_task(task, GROUPS // groups)
    dcs = [b.add(b.mul(g0, c(groups)), c(i)) for i in range(groups)]
    # ---- irq_rope_loads
    tt16 = b.land(tid, c(COLS // 2 - 1))
    pos = e.ld_i32(p["pos"], b.mul(t, c(2)))
    ropes = []
    for dc in dcs:
        d = b.add(b.mul(dc, c(COLS)), b.mul(tt16, c(2)))
        live = b.cmp_ge(d, c(HEAD_DIM - ROPE))
        jj = b.smax(b.div(b.sub(d, c(HEAD_DIM - ROPE)), c(2)), c(0))
        at = b.add(b.mul(pos, c(HALF)), jj)
        ropes.append((b.select(live, e.ld_bf(p["cos"], at), e.f(1.0)), b.select(live, e.ld_bf(p["sin"], at), e.f(0.0))))
    # ---- stage_attn_pv
    e.await_selection(tag, t)
    kv_len, meta = e.keys(t)
    act = b.div(b.add(kv_len, c(BK - 1)), c(BK))
    last = b.sub(e.written(kv_len), c(1))
    splits = [b.add(wave, c(8 * r)) for r in range(8)]
    krows = [e.key_row(t, meta, b.smin(b.add(b.mul(sp, c(BK)), b.lshr(lane, c(2))), b.sub(kv_len, c(1)))) for sp in splits]

    def ld_row4(row, byte):
        addr = b.add(b.add(p["pool"], b.mul(b.sext(row, I64), b.const_i64(ROW_BYTES))), b.sext(byte, I64))
        ptr = b.int_to_ptr(addr, PtrType(I32, "global"))
        return b.global_load_vN(ptr, c(0), I32, 4, align=16)

    def group_tiles(i):
        byte = b.mul(b.add(b.mul(dcs[i], c(COLS)), b.mul(b.land(lane, c(3)), c(8))), c(2))
        return [ld_row4(krows[r], byte) for r in range(8)]

    tiles = group_tiles(0)
    sink = e.ld_f32(p["sink"], b.add(b.mul(tile, c(HEAD_TILE)), b.lshr(tid, c(5))))
    src = b.smin(lane, last)
    hs = [b.add(b.mul(tile, c(HEAD_TILE)), b.add(b.mul(wave, c(2)), c(k))) for k in range(2)]

    def ms(sp, h):
        return b.add(b.mul(b.add(b.mul(t, c(SPLITS)), sp), c(HEADS)), h)

    r_am, r_al, r_ap = e.rsrc(p["am"]), e.rsrc(p["al"]), e.rsrc(p["ap"])
    a_am, a_al = b.readfirstlane(p["am"]), b.readfirstlane(p["al"])
    stats = [(r_am, a_am, ms(src, hs[k])) for k in range(2)] + [(r_al, a_al, ms(src, hs[k])) for k in range(2)]
    if knobs.stat_w0:
        with b.scf_if(b.cmp_eq(wave, c(0))):
            e.poll(tag, stats)
    e.barrier()
    # one 12-pair batch: the 4 stats (a pair each) and the 8 splits' p (two pairs each, one 16-B load)
    sp_specs = []
    for sp in splits:
        psp = b.smin(sp, last)
        api = b.add(b.mul(b.add(b.mul(b.add(b.mul(t, c(TILES)), tile), c(SPLITS)), psp), c(64)), lane)
        sp_specs.append((r_ap, b.mul(api, c(16)), 4))
    tspecs = [(rs, b.mul(i, c(8)), 2) for rs, _, i in stats] + sp_specs
    vecs = b.tagged_poll(tag, tspecs, aux=S.CPOL_SC1, pause=e.pause_text(), reload="launder")
    st_w = [b.vec_extract(v, 0) for v in vecs[:4]]
    pws = [(b.vec_extract(v, 0), b.vec_extract(v, 2)) for v in vecs[4:]]
    # ---- _pv_stats
    livel = b.cmp_lt(lane, act)

    def lane_gather(v_f32, src_lane):
        return b.bitcast(b.ds_bpermute(b.mul(src_lane, c(4)), b.bitcast(v_f32, I32)), F32)

    def row_shr(v, k):
        moved = lane_gather(v, b.smax(b.sub(lane, c(k)), c(0)))
        return b.select(b.cmp_ge(head16, c(k)), moved, e.f(0.0))

    def l_tree(l, al):
        x = b.fmul(l, al)
        v = e.fma(l, al, row_shr(x, 8))
        for k in (4, 2, 1):
            v = b.fadd(row_shr(v, k), v)
        row = b.lshr(lane, c(4))
        b15 = lane_gather(v, b.smax(b.sub(b.mul(row, c(16)), c(1)), c(0)))
        v = b.select(b.cmp_eq(b.land(row, c(1)), c(1)), b.fadd(b15, v), v)
        b31 = lane_gather(v, c(31))
        v = b.select(b.cmp_ge(row, c(2)), b.fadd(b31, v), v)
        return lane_gather(v, c(63))

    for k in range(2):
        h = b.add(b.mul(wave, c(2)), c(k))
        m = b.select(livel, b.bitcast(st_w[k], F32), e.f(NEG))
        l = b.select(livel, b.bitcast(st_w[2 + k], F32), e.f(0.0))
        m_max = e.butterfly(m, (32, 16, 8, 4, 2, 1), b.fmax)
        al = b.exp2_fast(b.fsub(m, m_max))
        if knobs.mutate == "alpha":
            al = e.f(1.0)
        l_comb = l_tree(l, al)
        e.lds_st_f32(b.add(alpha_l, b.add(b.mul(lane, c(HEAD_TILE)), h)), al)
        with b.scf_if(b.cmp_eq(lane, c(0))):
            e.lds_st_f32(b.add(stat, h), m_max)
            e.lds_st_f32(b.add(stat, b.add(c(HEAD_TILE), h)), l_comb)
    e.barrier()
    als = [e.lds_f32(b.add(alpha_l, b.add(b.mul(sp, c(HEAD_TILE)), head16))) for sp in splits]

    def pv_group(i, tl8):
        kt = b.add(kvw, b.mul(wave, c(BK * COLS // 2)))
        accs = []
        for r in range(8):
            sp = splits[r]
            pb = b.vec_bitcast(b.vec_pack([pws[r][0], pws[r][1]], I32), VectorType(BF16, 4))
            ok = b.cmp_lt(b.add(b.mul(sp, c(BK)), b.lshr(lane, c(2))), kv_len)
            vv = b.vec_pack([b.select(ok, b.vec_extract(tl8[r], q), c(0)) for q in range(4)], I32)
            b.smem_store_vN(e.pool, [b.add(kt, b.add(b.mul(b.lshr(lane, c(2)), c(COLS // 2)), b.mul(b.land(lane, c(3)), c(4))))], vv, 4)
            for tt in range(2):
                col = b.add(c(tt * 16), head16)
                half = []
                for i4 in range(4):
                    w = b.vec_extract(b.smem_load_vN(e.pool, b.add(kt, b.add(b.mul(b.add(b.mul(g, c(4)), c(i4)), c(COLS // 2)),
                                                                             b.lshr(col, c(1)))), dtype=I32, n=1), 0)  # fmt: skip
                    half.append(b.land(b.lshr(w, b.mul(b.land(col, c(1)), c(16))), c(0xFFFF)))
                av = b.vec_pack([b.lor(half[0], b.shl(half[1], c(16))), b.lor(half[2], b.shl(half[3], c(16)))], I32)
                acc = b.mfma_f32_16x16x16_bf16(b.vec_bitcast(av, VectorType(BF16, 4)), pb, b.vec_pack([e.f(0.0)] * 4, F32))
                live = b.cmp_lt(sp, act)
                accs.append([b.select(live, b.vec_extract(acc, j), e.f(0.0)) for j in range(4)])
        for k in range(8):
            tt, j = k // 4, k % 4
            av = [accs[2 * r + tt][j] for r in range(8)]
            pr = [e.fma(av[2 * m_], als[2 * m_], b.fmul(av[2 * m_ + 1], als[2 * m_ + 1])) for m_ in range(4)]
            node = b.fadd(b.fadd(pr[0], pr[1]), b.fadd(pr[2], pr[3]))
            e.lds_st_f32(b.add(tree, b.add(b.mul(b.add(b.mul(wave, c(64)), lane), c(8)), c(k))), node)
        e.barrier()
        h = b.lshr(tid, c(5))
        col = b.land(tid, c(COLS - 1))
        owner = b.add(b.mul(b.lshr(b.land(col, c(15)), c(2)), c(16)), h)
        kk = b.add(b.mul(b.lshr(col, c(4)), c(4)), b.land(col, c(3)))
        w = [e.lds_f32(b.add(tree, b.add(b.mul(b.add(c(ww * 64), owner), c(8)), kk))) for ww in range(8)]
        x = [b.fadd(w[2 * m_], w[2 * m_ + 1]) for m_ in range(4)]
        tot = b.fadd(b.fadd(x[0], x[2]), b.fadd(x[1], x[3]))
        m_max = e.lds_f32(b.add(stat, h))
        l_comb = e.lds_f32(b.add(stat, b.add(c(HEAD_TILE), h)))
        log2e = e.f(S.LOG2E)
        sk = e.f(NEG) if knobs.mutate == "sink" else sink
        m_final = b.fmax(m_max, b.fmul(sk, log2e))
        alpha_kv = b.exp2_fast(b.fsub(m_max, m_final))
        alpha_sink = b.exp2_fast(e.fma(sk, log2e, b.fneg(m_final)))
        l_final = e.fma(l_comb, alpha_kv, alpha_sink)
        o = b.fdiv(b.fmul(tot, alpha_kv), b.fmax(l_final, e.f(1e-30)))
        o = b.select(b.fcmp("ogt", l_final, e.f(0.0)), o, e.f(0.0))
        e.lds_st_f32(b.add(out_l, b.add(b.mul(b.add(c(i * HEAD_TILE), h), c(COLS)), col)), e.bf16r(o))
        e.barrier()

    for i in range(groups):
        nxt = group_tiles(i + 1) if i + 1 < groups else None
        pv_group(i, tiles)
        tiles = nxt
    # ---- irq_group (inverse RoPE + group-32 FP8 quant) for each column group
    for i in range(groups):
        mine = b.cmp_lt(tid, c(HEAD_TILE * COLS // 2))
        h = b.lshr(b.smin(tid, c(HEAD_TILE * COLS // 2 - 1)), c(4))
        headg = b.add(b.mul(tile, c(HEAD_TILE)), h)
        d = b.add(b.mul(dcs[i], c(COLS)), b.mul(tt16, c(2)))
        ob = b.add(b.add(out_l, c(i * HEAD_TILE * COLS)), b.mul(h, c(COLS)))
        ev = e.lds_f32(b.add(ob, b.mul(tt16, c(2))))
        od = e.lds_f32(b.add(ob, b.add(b.mul(tt16, c(2)), c(1))))
        cs, sn = ropes[i]
        e1 = b.fadd(b.fmul(ev, cs), b.fmul(od, sn))
        o1 = b.fsub(b.fmul(od, cs), b.fmul(ev, sn))
        amax = e.butterfly(b.fmax(b.fabs(e1), b.fabs(o1)), (1, 2, 4, 8), b.fmax)
        code = e.code_ceil(b.fmul(b.fmax(amax, e.f(IRQ_FLOOR)), e.f(1.0 / S.FP8_MAX)))
        inv = e.pow2(b.sub(c(254), code))
        q0 = e.clamp_fp8(b.fmul(e1, inv))
        q1 = e.clamp_fp8(b.fmul(o1, inv))
        n0, n1 = e.xchg(q0, 1), e.xchg(q1, 1)
        w = e.fp8_pack4([q0, q1, n0, n1])
        if knobs.mutate == "put_bit":
            w = b.xor(w, c(1))
        with b.scf_if(b.land(mine, b.cmp_eq(b.land(tt16, c(1)), c(0)))):
            e.put(tag, p["xo"], b.add(b.mul(t, c(XO_WORDS)), b.lshr(b.add(b.mul(headg, c(HEAD_DIM)), d), c(2))), w)
        with b.scf_if(b.land(mine, b.cmp_eq(tt16, c(0)))):
            e.put(tag, p["xos"], b.add(b.mul(t, c(HEADS * HEAD_DIM // 32)), b.lshr(b.add(b.mul(headg, c(HEAD_DIM)), d), c(5))), code)
    b.ret()
    return b.kernel


def parse_variant(stage: str, v: str):
    """score ``p<per>x<ix>n<S>`` / irq ``g<groups>x<ix>n<S>`` -> (k, ix, S)."""
    a, _, rest = v[1:].partition("x")
    ix, _, n = rest.partition("n")
    return int(a), int(ix), int(n)


def write_attn_functions(out: str, stage: str, variants, knobs: AttnKnobs = AttnKnobs(), **kw) -> list:
    def build(v):
        k, ix, n = parse_variant(stage, v)
        return build_score_kernel(n, k, ix, knobs) if stage == "attn" else build_irq_kernel(n, k, ix, knobs)

    return S.write_functions(build, out, list(variants), stage, ABI_SCORE if stage == "attn" else ABI_IRQ, knobs, **kw)


def default_variants(stage: str, rows=range(1, 49), ixs=(0, 1)):
    out = []
    for n in rows:
        for ix in ixs:
            out.append(f"p{score_tasks(n)}x{ix}n{n}" if stage == "attn" else f"g{pv_groups(n)}x{ix}n{n}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 attention (score / irq) stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("attn", "irq", "both"), default="both")
    ap.add_argument("--variants", default="")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    kn = S.knobs_from(AttnKnobs, a.knobs)
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    res = []
    for st in (("attn", "irq") if a.stage == "both" else (a.stage,)):
        vs = [v for v in a.variants.split(",") if v and v[0] == ("p" if st == "attn" else "g")] or default_variants(st)
        res += write_attn_functions(a.out, st, vs, kn, arch=a.arch, flavor=a.flavor, bitcode=bc)
    print(json.dumps({"knobs": S.spec_of(kn), "written": res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
