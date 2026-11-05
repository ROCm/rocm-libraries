# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W14 T1: routed-expert MALL prefetch for ATOM's DeepSeek-V4.1 mono decode (gfx950), a side-stream rocKE kernel.

``dsv41_rpf`` ("record + prefetch"), one launch per layer on a side stream of ATOM's FULL graph
(``ATOM_DSV41_RK_PREFETCH``, ``atom/models/deepseek_v41/rk/prefetch.py``):

1. record (``rec >= 0``): the routing of the K2 launch that just ended, read from its ``route`` mailbox region
   (copy 0: ``S x topk`` (id, tag) pairs per token, then the weights; a pair counts only with the launch's tag),
   becomes the 0/1 expert flags of layer ``rec`` in ``table`` (``E`` int32 per layer; CTA 0 writes, ``sc0 sc1``).
2. predict: the expert flags of layer ``pf`` from ``table`` (what layer ``pf`` routed the previous step: the temporal
   predictor ``T``), as an ascending expert list in LDS (every CTA the same list).
3. prefetch: the four MXFP4 regions (``w13``, its scales, ``w2``, its scales) of the first ``budget`` experts of the
   list, read with a NON-nt policy (``aux``; W14 mb1: only loads without ``nt`` allocate into the MALL, and ATOM's
   ``nt`` expert stream then hits there) and dropped (an XOR fold stored on an impossible value).

With ``MODE_EA`` the prediction is (or adds) the early router's: ``dsv41_rpr`` (``build_router_kernel``) puts layer
``pf``'s router logits of the previous layer's FFN input rows into ``logits``; each row's top ``kp`` of
sqrt(softplus(logit)) + bias join the flags (the selection rule of ATOM's ``topk_gating``, fp32).

It never writes ATOM memory (the ``write`` mutant does: the bitwise gate's must-fail twin). ``mode`` bit 2 shifts
every predicted id by one (the ``ids`` mutant: still bitwise, must show no MALL gain). Small LDS and few VGPRs, so it
runs beside K1 / K2, which hold a CTA per CU (W14 mb1 ``co``: a 0-LDS streamer co-resides with a K2-footprint
occupant and keeps streaming, in streams and in a captured graph).

Optional stamps (``stamps != 0``): per CTA ``(t_start, t_end, xcc, n_experts)`` (s_memrealtime, 100 MHz), the same
clock as ATOM's ``ATOM_MONO_TIMELINE`` stamps.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D

__all__ = ["RpfKnobs", "build_rpf_kernel", "rpf_name", "build_router_kernel", "router_name", "ShadowKnobs",
           "build_shadow_kernel", "shadow_name", "MAX_PICKS", "MAX_E",
           "STAMP_WORDS", "MODE_T", "MODE_EA", "MODE_IDS"]

MAX_PICKS = 512  # S x topk
MAX_E = 512
EA_CHUNKS = 6  # the early router's experts a lane holds (E <= 384)
STAMP_WORDS = 4
OOB = 0x7FFFFFF0
MODE_T = 1  # predict from the table: what layer pf routed the previous step
MODE_EA = 2  # predict from ``logits`` (the early router: layer pf's router on layer pf - 1's FFN input), top-``kp``
MODE_IDS = 4  # mutant: every predicted id + 1 (mod E)


def _c(b, v):
    return b.const_i32(int(v))


def _ptr(b, addr, t=I32):
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b, addr, nbytes):
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b, v):
    return b.zext(v, I64)


def _mbcnt(b, mask):
    """Set bits of the i64 ballot ``mask`` below this lane (``s_nop 4``: the mask's SGPRs come from a VALU compare;
    LLVM's hazard recognizer does not look into inline asm, alola rule 185)."""
    lo = b.trunc(mask, I32)
    hi = b.trunc(b.lshr(mask, b.const_i64(32)), I32)
    return b.inline_asm("s_nop 4\nv_mbcnt_lo_u32_b32 $0, $1, 0\nv_mbcnt_hi_u32_b32 $0, $2, $0", "=&v,s,s", [lo, hi],
                        I32, sideeffect=False, result_name_hint="mbcnt")


@dataclass(frozen=True)
class RpfKnobs:
    aux: int = 0  # prefetch load policy (no nt: allocates in MALL)
    threads: int = 256
    depth: int = 4
    unroll: int = 2
    mutant: str = "none"  # none | write
    tag: str = ""

    def check(self) -> None:
        if self.aux & 2:
            raise ValueError("aux with nt does not allocate in the MALL")
        if self.threads not in (64, 128, 256, 512) or not 1 <= self.depth <= 8 or self.unroll not in (1, 2, 4):
            raise ValueError("threads 64..512, depth 1..8, unroll 1 | 2 | 4")
        if self.mutant not in ("none", "write"):
            raise ValueError("mutant none | write")
        if self.threads * 2 < 384:
            raise ValueError("threads x 2 must cover the experts")


def rpf_name(k: RpfKnobs) -> str:
    return (f"dsv41_rpf_a{k.aux}_t{k.threads}_d{k.depth}u{k.unroll}" + (f"_m{k.mutant}" if k.mutant != "none" else "")
            + (f"_{k.tag}" if k.tag else ""))


def build_rpf_kernel(k: RpfKnobs) -> KernelDef:
    """Params (ABI, in order): i64 ``route`` (route pairs of the K2 that ended), ``table`` (int32 [layers][E]),
    ``w13, s13, w2, s2`` (layer ``pf``'s expert regions), ``sink``, ``stamps`` (0: none), ``logits`` (f32
    [rows_ea][E], MODE_EA), ``bias`` (f32 [E], layer pf's selection bias); i32 ``S``, ``topk``, ``E``, ``tag`` (the K2
    launch's MoE tag + 1), ``rec`` (-1: none), ``pf`` (-1: none), ``sz0..sz3`` (bytes per expert of each region),
    ``budget`` (max experts prefetched), ``mode``, ``grid``, ``rows_ea``, ``kp``."""
    k.check()
    b = IRBuilder(rpf_name(k))
    b.kernel.attrs["max_workgroup_size"] = k.threads
    route = b.param("route", I64)
    table = b.param("table", I64)
    bases = [b.param(n, I64) for n in ("w13", "s13", "w2", "s2")]
    sink = b.param("sink", I64)
    stamps = b.param("stamps", I64)
    logits = b.param("logits", I64)
    bias = b.param("bias", I64)
    S = b.readfirstlane(b.param("S", I32))
    topk = b.readfirstlane(b.param("topk", I32))
    E = b.readfirstlane(b.param("E", I32))
    tag = b.readfirstlane(b.param("tag", I32))
    rec = b.readfirstlane(b.param("rec", I32))
    pf = b.readfirstlane(b.param("pf", I32))
    sizes = [b.readfirstlane(b.param(f"sz{i}", I32)) for i in range(4)]
    budget = b.readfirstlane(b.param("budget", I32))
    mode = b.readfirstlane(b.param("mode", I32))
    grid = b.readfirstlane(b.param("grid", I32))
    rows_ea = b.readfirstlane(b.param("rows_ea", I32))  # logits rows (the early router's)
    kp = b.readfirstlane(b.param("kp", I32))
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    zero = _c(b, 0)
    one = _c(b, 1)
    lane = b.land(tid, _c(b, 63))
    t_start = D.memrealtime(b)
    flg = b.smem_alloc(I32, [MAX_E], name_hint="flg")  # recorded flags, then the predicted ones
    lst = b.smem_alloc(I32, [MAX_E + 64], name_hint="lst")  # the predicted list, [MAX_E] = its length
    for j in range(MAX_E // k.threads):
        b.smem_store_vN(flg, [b.add(tid, _c(b, j * k.threads))], zero, 1)
    b.sync_lds_only()
    tab_rs = _rsrc(b, b.readfirstlane(table), b.mul(b.mul(_c(b, 64), E), _c(b, 4)))
    # 1. record: this K2's routing -> flags -> table[rec] (CTA 0)
    with b.scf_if(b.cmp_ge(rec, zero)):
        n = b.mul(S, topk)
        r_rs = _rsrc(b, b.readfirstlane(route), b.mul(b.mul(S, b.mul(topk, _c(b, 2))), _c(b, 8)))
        for j in range(MAX_PICKS // k.threads):
            i = b.add(tid, _c(b, j * k.threads))
            with b.scf_if(b.cmp_lt(i, n)):
                tok = b.div(i, topk)
                kk = b.sub(i, b.mul(tok, topk))
                pair = b.add(b.mul(tok, b.mul(topk, _c(b, 2))), kk)
                v = b.buffer_load_cp(r_rs, b.mul(pair, _c(b, 8)), zero, dwords=2, aux=16)
                e = b.vec_extract(v, 0)
                ok = b.land(b.cmp_eq(b.vec_extract(v, 1), tag), b.land(b.cmp_ge(e, zero), b.cmp_lt(e, E)))
                with b.scf_if(ok):
                    b.smem_store_vN(flg, [e], one, 1)
        b.sync_lds_only()
        with b.scf_if(b.cmp_eq(cta, zero)):
            for j in range(MAX_E // k.threads):
                e = b.add(tid, _c(b, j * k.threads))
                with b.scf_if(b.cmp_lt(e, E)):
                    f = b.vec_extract(b.smem_load_vN(flg, e, dtype=I32, n=1), 0)
                    b.buffer_store_cp(f, tab_rs, b.mul(b.add(b.mul(rec, E), e), _c(b, 4)), zero, aux=17)
        b.sync_lds_only()
    # 2. predict: flags of layer pf from the table (previous step), the ids mutant shifts them
    want = b.land(b.cmp_ge(pf, zero), b.cmp_ne(b.land(mode, _c(b, 1)), zero))
    mut = b.cmp_ne(b.land(mode, _c(b, 4)), zero)
    for j in range(MAX_E // k.threads):
        e = b.add(tid, _c(b, j * k.threads))
        src = b.select(mut, b.select(b.cmp_eq(e, zero), b.sub(E, one), b.sub(e, one)), e)  # flag[e] = T[e - 1]
        inb = b.land(want, b.cmp_lt(e, E))
        f = b.buffer_load_cp(tab_rs, b.select(inb, b.mul(b.add(b.mul(pf, E), src), _c(b, 4)), _c(b, OOB)), zero,
                             dwords=1, aux=16)
        b.smem_store_vN(flg, [e], b.select(inb, f, zero), 1)
    b.sync_lds_only()
    # early router: each row's top kp of sqrt(softplus(logit)) + bias -> flags (one wave a row)
    nwaves = k.threads // 64
    wv = D.wave_id(b)
    ea = b.land(b.cmp_ge(pf, zero), b.cmp_ne(b.land(mode, _c(b, MODE_EA)), zero))
    with b.scf_if(ea):
        lg_rs = _rsrc(b, b.readfirstlane(logits), b.mul(b.mul(rows_ea, E), _c(b, 4)))
        bs_rs = _rsrc(b, b.readfirstlane(bias), b.mul(E, _c(b, 4)))
        ninf = b.const_f32(float("-inf"))
        log2e, ln2 = 1.4426950408889634, 0.6931471805599453
        with b.scf_for(wv, rows_ea, _c(b, nwaves), iv_name="earow") as row:
            vals = []
            for j in range(EA_CHUNKS):
                e = b.add(lane, _c(b, 64 * j))
                ok = b.cmp_lt(e, E)
                xi = b.buffer_load_cp(lg_rs, b.select(ok, b.mul(b.add(b.mul(row, E), e), _c(b, 4)), _c(b, OOB)), zero,
                                      dwords=1, aux=0)
                x = b.bitcast(xi, F32)
                ax = b.fmax(x, b.fmul(x, b.const_f32(-1.0)))
                tail = b.fmul(b.const_f32(ln2), b.log2(b.fadd(b.const_f32(1.0), b.exp2(b.fmul(ax, b.const_f32(-log2e))))))
                sp = b.fadd(b.fmax(x, b.const_f32(0.0)), tail)
                bv = b.buffer_load_cp(bs_rs, b.select(ok, b.mul(e, _c(b, 4)), _c(b, OOB)), zero, dwords=1, aux=0)
                sc = b.fadd(b.sqrt(sp), b.bitcast(bv, F32))
                vals.append(b.select(ok, sc, ninf))
            iters = [(f"ev{j}", v) for j, v in enumerate(vals)]
            loop = b.scf_for_iter(zero, kp, one, iters, iv_name="eak")
            with loop as (_i, vs):
                vs = list(vs)
                m = vs[0]
                for v in vs[1:]:
                    m = b.fmax(m, v)
                for msk in (32, 16, 8, 4, 2, 1):
                    m = b.fmax(m, b.warp_shuffle_xor(m, msk))
                live = b.fcmp("ogt", m, ninf)
                nv = []
                for j, v in enumerate(vs):
                    hit = b.land(live, b.fcmp("oeq", v, m))
                    with b.scf_if(hit):
                        b.smem_store_vN(flg, [b.add(lane, _c(b, 64 * j))], one, 1)
                    nv.append(b.select(hit, ninf, v))
                b.scf_yield(*nv)
    b.sync_lds_only()
    # (lane defined at entry)
    with b.scf_if(b.cmp_lt(tid, _c(b, 64))):
        base = zero
        for c0 in range(0, MAX_E, 64):
            e = b.add(lane, _c(b, c0))
            f = b.vec_extract(b.smem_load_vN(flg, e, dtype=I32, n=1), 0)
            hit = b.cmp_ne(f, zero)
            m = b.wave_ballot(b.zext(hit, I32))
            rank = b.add(base, _mbcnt(b, m))
            with b.scf_if(hit):
                b.smem_store_vN(lst, [rank], e, 1)
            cnt = b.inline_asm("s_nop 4\ns_bcnt1_i32_b64 $0, $1", "=s,s", [m], I32, sideeffect=False,
                               result_name_hint="bc")
            base = b.add(base, cnt)
        with b.scf_if(b.cmp_eq(lane, zero)):
            b.smem_store_vN(lst, [_c(b, MAX_E)], base, 1)
    b.sync_lds_only()
    nlist = b.readfirstlane(b.vec_extract(b.smem_load_vN(lst, _c(b, MAX_E), dtype=I32, n=1), 0))
    n_ids = b.readfirstlane(b.smin(nlist, budget))
    # 3. prefetch: the regions of the first n_ids experts of the list, CTA units round robin
    per = k.threads * 16 * k.unroll
    acc = zero
    for r in range(4):
        sz = sizes[r]
        steps_e = b.div(b.add(sz, _c(b, per - 1)), _c(b, per))
        total = b.mul(steps_e, n_ids)
        my = b.readfirstlane(b.smax(b.div(b.add(b.sub(total, cta), b.sub(grid, one)), grid), zero))
        base_r = b.readfirstlane(bases[r])

        def issue(i, base_r=base_r, sz=sz, steps_e=steps_e, my=my):
            idx = b.add(cta, b.mul(i, grid))
            ok = b.cmp_lt(i, my)
            idx = b.select(ok, idx, zero)
            slot = b.readfirstlane(b.div(idx, steps_e))
            st = b.sub(idx, b.mul(slot, steps_e))
            e = b.readfirstlane(b.vec_extract(b.smem_load_vN(lst, b.select(ok, slot, zero), dtype=I32, n=1), 0))
            rs = _rsrc(b, b.add(base_r, b.mul(_i64(b, e), _i64(b, sz))), sz)
            out = []
            for j in range(k.unroll):
                off = b.add(b.mul(st, _c(b, per)), b.add(_c(b, j * k.threads * 16), b.mul(tid, _c(b, 16))))
                off = b.select(ok, off, _c(b, OOB))
                v = b.buffer_load_cp(rs, off, zero, dwords=4, aux=k.aux)
                if k.mutant == "write":
                    b.buffer_store_cp(b.vec_insert(v, b.xor(b.vec_extract(v, 0), one), 0), rs, off, zero, aux=0)
                out.append(v)
            return out

        u = k.unroll
        q = [v for i in range(k.depth) for v in issue(_c(b, i))]
        iters = [(f"r{r}s{i}", v) for i, v in enumerate(q)] + [(f"acc{r}", acc)]
        loop = b.scf_for_iter(zero, my, one, iters, iv_name=f"pf{r}")
        with loop as (i, vals):
            vals = list(vals)
            ring, a_ = vals[:-1], vals[-1]
            nxt = issue(b.add(i, _c(b, k.depth)))
            for v in ring[:u]:
                a_ = b.xor(a_, b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
            b.scf_yield(*(ring[u:] + nxt), a_)
        acc = loop.results[-1]
    with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
        b.store(acc, _ptr(b, b.readfirstlane(sink)), tid)
    with b.scf_if(b.cmp_ne(stamps, b.const_i64(0))):
        b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)
        b.sync_lds_only()
        t_end = D.memrealtime(b)
        with b.scf_if(D.is_thread0(b)):
            sp = _ptr(b, b.readfirstlane(stamps), I64)
            o = b.mul(cta, _c(b, STAMP_WORDS))
            b.store(t_start, sp, o)
            b.store(t_end, sp, b.add(o, one))
            b.store(_i64(b, D.xcc_id(b)), sp, b.add(o, _c(b, 2)))
            b.store(_i64(b, n_ids), sp, b.add(o, _c(b, 3)))
    return b.kernel


def router_name(threads: int = 256) -> str:
    return f"dsv41_rpr_t{threads}"


def build_router_kernel(threads: int = 256) -> KernelDef:
    """``dsv41_rpr``: f32 logits [S][E] = x [S][H] (bf16) . W [E][H] (bf16), one expert a wave (``threads / 64``
    experts a CTA, grid E / that), H split over the lanes in 16-byte chunks, fixed-order f32 sums (a prediction only:
    not ATOM's bf16-rounded router). Params: i64 ``x``, ``w``, ``out``; i32 ``S``, ``E``, ``H``."""
    b = IRBuilder(router_name(threads))
    b.kernel.attrs["max_workgroup_size"] = threads
    x = b.param("x", I64)
    w = b.param("w", I64)
    out = b.param("out", I64)
    S = b.readfirstlane(b.param("S", I32))
    E = b.readfirstlane(b.param("E", I32))
    H = b.readfirstlane(b.param("H", I32))
    zero = _c(b, 0)
    tid = b.thread_id_x()
    lane = b.land(tid, _c(b, 63))
    wv = D.wave_id(b)
    e = b.readfirstlane(b.add(b.mul(b.readfirstlane(b.block_id_x()), _c(b, threads // 64)), wv))
    x_rs = _rsrc(b, b.readfirstlane(x), b.mul(b.mul(S, H), _c(b, 2)))
    w_rs = _rsrc(b, b.add(b.readfirstlane(w), b.mul(_i64(b, e), _i64(b, b.mul(H, _c(b, 2))))), b.mul(H, _c(b, 2)))
    o_rs = _rsrc(b, b.readfirstlane(out), b.mul(b.mul(S, E), _c(b, 4)))
    chunks = b.div(H, _c(b, 512))  # 64 lanes x 8 bf16
    with b.scf_if(b.cmp_lt(e, E)):
        with b.scf_for(zero, S, _c(b, 1), iv_name="rrow") as row:
            loop = b.scf_for_iter(zero, chunks, _c(b, 1), [("racc", b.const_f32(0.0))], iv_name="rch")
            with loop as (ch, vals):
                acc = list(vals)[0]
                off = b.mul(b.add(b.mul(ch, _c(b, 64)), lane), _c(b, 16))
                wv4 = b.buffer_load_cp(w_rs, off, zero, dwords=4, aux=0)
                xv4 = b.buffer_load_cp(x_rs, b.add(b.mul(b.mul(row, H), _c(b, 2)), off), zero, dwords=4, aux=0)
                for d in range(4):
                    ww, xx = b.vec_extract(wv4, d), b.vec_extract(xv4, d)
                    for half in (0, 1):
                        if half == 0:
                            wf = b.bitcast(b.shl(ww, _c(b, 16)), F32)
                            xf = b.bitcast(b.shl(xx, _c(b, 16)), F32)
                        else:
                            wf = b.bitcast(b.land(ww, _c(b, -65536)), F32)
                            xf = b.bitcast(b.land(xx, _c(b, -65536)), F32)
                        acc = b.fma(wf, xf, acc)
                b.scf_yield(acc)
            tot = loop.results[0]
            for msk in (32, 16, 8, 4, 2, 1):
                tot = b.fadd(tot, b.warp_shuffle_xor(tot, msk))
            with b.scf_if(b.cmp_eq(lane, zero)):
                b.buffer_store_cp(b.bitcast(tot, I32), o_rs, b.mul(b.add(b.mul(row, E), e), _c(b, 4)), zero, aux=0)
    return b.kernel


# ------------------------------------------------------------------------------------------------ the shadow prefetcher
@dataclass(frozen=True)
class ShadowKnobs:
    aux: int = 0  # prefetch load policy (no nt)
    threads: int = 256
    depth: int = 4
    unroll: int = 1
    pause: str = "s_sleep 1"  # between polls
    mutant: str = "none"  # none | write (stores the prefetched words back, low bit flipped) | ids (every id + 1)
    path: str = "vgpr"  # vgpr (dwordx4 loads into registers) | lds (buffer_load ... lds DMA: no data registers)
    relay: bool = False  # only CTA 0 polls ATOM's tag word; the others poll the shadow's own relay word (sink + 8 KiB)
    tag: str = ""

    def check(self) -> None:
        dmax = 32 if self.path == "lds" else 8
        if self.aux & 2 or self.threads not in (64, 128, 256) or not 1 <= self.depth <= dmax or self.unroll not in (1, 2):
            raise ValueError("aux without nt; threads 64..256; depth 1..8 (lds: 1..32); unroll 1 | 2")
        if self.path not in ("vgpr", "lds") or (self.path == "lds" and (self.mutant == "write" or self.aux > 1)):
            raise ValueError("path vgpr | lds (lds: aux 0 / 1, no write mutant)")
        if self.mutant not in ("none", "write", "ids"):
            raise ValueError("mutant none | write | ids")


def shadow_name(k: ShadowKnobs) -> str:
    return (f"dsv41_rpfs{'l' if k.path == 'lds' else ''}{'r' if k.relay else ''}_a{k.aux}_t{k.threads}_d{k.depth}u{k.unroll}"
            + (f"_m{k.mutant}" if k.mutant != "none" else "")
            + (f"_{k.tag}" if k.tag else ""))


def _poll_tag(b, base, voff, want, maxit, ge: bool, pause: str):
    """Thread-level asm poll of the 32-bit word at byte ``voff`` of ``base`` (``sc1``, a 64-bit SGPR address) until it is ``== want``
    (``ge`` False) or ``>= want`` unsigned (``ge`` True), at most ``maxit`` retries. Returns the iterations
    (``>= maxit``: timed out). ``s_nop 4`` first: the SGPR base is a ``readfirstlane`` (VALU) result and LLVM's hazard
    recognizer does not look into inline asm (alola rules 176 / 185)."""
    top = "${:private}t1p${:uid}_top"
    done = "${:private}t1p${:uid}_done"
    cmp = "v_cmp_lt_u32_e64 $1, $0, $4" if ge else "v_cmp_ne_u32_e64 $1, $0, $4"
    lines = ["s_nop 4", "s_mov_b32 $2, 0", f"{top}:", "global_load_dword $0, $3, $5 sc1", "s_waitcnt vmcnt(0)", cmp,
             "s_and_b64 $1, $1, exec", f"s_cbranch_scc0 {done}", "s_add_u32 $2, $2, 1", "s_cmp_ge_u32 $2, $6",
             f"s_cbranch_scc1 {done}", pause, f"s_branch {top}", f"{done}:"]
    _seen, _mask, it = b.inline_asm_multi(
        "\n".join(x for x in lines if x), "=&v,=&s,=&s,v,v,s,s,~{vcc},~{scc}", [voff, want, base, maxit],
        result_types=[I32, I64, I32], result_name_hint="t1p")
    return it


def build_shadow_kernel(k: ShadowKnobs) -> KernelDef:
    """``dsv41_rpfs``: ONE launch a decode step, on its own (high-priority) stream beside ATOM's graph replay; every
    CTA (one wave a SIMD, small LDS: it fits beside K1 / K2 when K2's VGPRs leave room) walks the layers:

    1. waits until K1 of layer L has published (the tag word of K1's ``x8`` pair 0 is ``L + 1``; layer 0: exactly 1,
       so the previous step's last tag cannot pass);
    2. CTA 0 records what K2 of layer L - 1 routed (its ``route`` pairs, tag ``moe_tag0 + L``) into the table;
    3. prefetches its share of the first ``budget`` experts of table row L (the previous step's routing of layer L)
       with non-nt loads (the window: the rest of K1 and K2's attention half, HBM idle there);

    then records layer ``nl - 1`` once K2's routing of it is published. Every wait is bounded (``maxit`` polls): a
    timeout ends the CTA (it never blocks ATOM). Never writes ATOM memory.

    Params: i64 ``x8tag`` (address of K1 x8 pair 0's tag word), ``route`` (K2 route pair 0; its tag word at + 4), ``table`` (int32
    [64][E]), ``bases`` (i64 [nl][4]: each layer's four expert regions), ``sink``, ``stamps`` (0: none; i64 [nl][grid][4]);
    i32 ``nl``, ``S``, ``topk``, ``E``, ``moe_tag0`` (+1 = K2's tag of layer 0), ``sz0..3``, ``budget``, ``maxit``,
    ``grid``, ``token`` (a per-launch value, increasing, < 2^30: the relay's step id).

    ``relay``: CTA 0 alone polls ATOM's tag word, then stores ``2 token + aborted`` into relay word L (``sink`` + 8 KiB
    + 4 L); every other CTA polls that word for ``>= 2 token`` (a stale value from an earlier launch is smaller) and
    aborts on an odd value: one poller on ATOM's mailbox line instead of ``grid``."""
    k.check()
    b = IRBuilder(shadow_name(k))
    b.kernel.attrs["max_workgroup_size"] = k.threads
    x8tag = b.param("x8tag", I64)
    route = b.param("route", I64)
    table = b.param("table", I64)
    bases_a = b.param("bases", I64)
    sink = b.param("sink", I64)
    stamps = b.param("stamps", I64)
    nl = b.readfirstlane(b.param("nl", I32))
    S = b.readfirstlane(b.param("S", I32))
    topk = b.readfirstlane(b.param("topk", I32))
    E = b.readfirstlane(b.param("E", I32))
    mtag0 = b.readfirstlane(b.param("moe_tag0", I32))
    sizes = [b.readfirstlane(b.param(f"sz{i}", I32)) for i in range(4)]
    budget = b.readfirstlane(b.param("budget", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    grid = b.readfirstlane(b.param("grid", I32))
    token = b.readfirstlane(b.param("token", I32))
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    zero, one = _c(b, 0), _c(b, 1)
    lane = b.land(tid, _c(b, 63))
    flg = b.smem_alloc(I32, [MAX_E], name_hint="sflg")
    lst = b.smem_alloc(I32, [MAX_E + 64], name_hint="slst")
    ctl = b.smem_alloc(I32, [4], name_hint="sctl")  # [0] abort
    dma = b.smem_alloc(I32, [k.threads * 4], name_hint="sdma") if k.path == "lds" else None
    with b.scf_if(D.is_thread0(b)):
        b.smem_store_vN(ctl, [zero], zero, 1)
    b.sync_lds_only()
    xbase = b.readfirstlane(x8tag)
    rbase = b.readfirstlane(route)
    tab_rs = _rsrc(b, b.readfirstlane(table), b.mul(b.mul(_c(b, 64), E), _c(b, 4)))
    bs_rs = _rsrc(b, b.readfirstlane(bases_a), b.mul(nl, _c(b, 32)))
    r_rs = _rsrc(b, rbase, b.mul(b.mul(S, b.mul(topk, _c(b, 2))), _c(b, 8)))
    per = k.threads * 16 * k.unroll

    def aborted():
        return b.cmp_ne(b.readfirstlane(b.vec_extract(b.smem_load_vN(ctl, zero, dtype=I32, n=1), 0)), zero)

    def wait(base, want, ge):
        with b.scf_if(D.is_thread0(b)):
            it = _poll_tag(b, base, zero, want, maxit, ge, k.pause)
            with b.scf_if(b.cmp_ge(it, maxit)):
                b.smem_store_vN(ctl, [zero], one, 1)
        b.sync_lds_only()

    def record(layer):
        """CTA 0: K2(layer)'s route pairs (tag moe_tag0 + layer) -> table row ``layer``."""
        with b.scf_if(b.cmp_eq(cta, zero)):
            for j in range(MAX_E // k.threads):
                b.smem_store_vN(flg, [b.add(tid, _c(b, j * k.threads))], zero, 1)
            b.sync_lds_only()
            n = b.mul(S, topk)
            want = b.add(mtag0, layer)
            for j in range(MAX_PICKS // k.threads):
                i = b.add(tid, _c(b, j * k.threads))
                with b.scf_if(b.cmp_lt(i, n)):
                    tok = b.div(i, topk)
                    pair = b.add(b.mul(tok, b.mul(topk, _c(b, 2))), b.sub(i, b.mul(tok, topk)))
                    v = b.buffer_load_cp(r_rs, b.mul(pair, _c(b, 8)), zero, dwords=2, aux=16)
                    e = b.vec_extract(v, 0)
                    ok = b.land(b.cmp_eq(b.vec_extract(v, 1), want), b.land(b.cmp_ge(e, zero), b.cmp_lt(e, E)))
                    with b.scf_if(ok):
                        b.smem_store_vN(flg, [e], one, 1)
            b.sync_lds_only()
            for j in range(MAX_E // k.threads):
                e = b.add(tid, _c(b, j * k.threads))
                with b.scf_if(b.cmp_lt(e, E)):
                    f = b.vec_extract(b.smem_load_vN(flg, e, dtype=I32, n=1), 0)
                    b.buffer_store_cp(f, tab_rs, b.mul(b.add(b.mul(layer, E), e), _c(b, 4)), zero, aux=17)
            b.sync_lds_only()

    sinkp = _ptr(b, b.readfirstlane(sink))
    with b.scf_for(zero, nl, one, iv_name="shl") as L:
        # 1. K1(L) has started: its x8 pair 0 tag is L + 1 (layer 0: exactly 1, so the previous step's last tag
        #    cannot pass; later layers: >= L + 1, a layer this CTA arrived late for is not waited on)
        def k1_wait():
            with b.scf_if(b.cmp_eq(aborted(), b.cmp_ne(zero, zero))):
                with b.scf_if(b.cmp_eq(L, zero)):
                    wait(xbase, one, False)
                with b.scf_if(b.cmp_gt(L, zero)):
                    wait(xbase, b.add(L, one), True)

        if not k.relay:
            k1_wait()
        else:
            rel_rs = _rsrc(b, b.add(b.readfirstlane(sink), b.const_i64(8192)), _c(b, 256))
            with b.scf_if(b.cmp_eq(cta, zero)):
                k1_wait()
                with b.scf_if(D.is_thread0(b)):
                    ab = b.vec_extract(b.smem_load_vN(ctl, zero, dtype=I32, n=1), 0)
                    val = b.add(b.mul(token, _c(b, 2)), b.select(b.cmp_ne(ab, zero), one, zero))
                    b.buffer_store_cp(val, rel_rs, b.mul(L, _c(b, 4)), zero, aux=17)
            with b.scf_if(b.cmp_ne(cta, zero)):
                with b.scf_if(b.cmp_eq(aborted(), b.cmp_ne(zero, zero))):
                    rel_word = b.add(b.add(b.readfirstlane(sink), b.const_i64(8192)), _i64(b, b.mul(L, _c(b, 4))))
                    wait(b.readfirstlane(rel_word), b.mul(token, _c(b, 2)), True)
                    with b.scf_if(D.is_thread0(b)):
                        v = b.buffer_load_cp(rel_rs, b.mul(L, _c(b, 4)), zero, dwords=1, aux=16)
                        with b.scf_if(b.cmp_ne(b.land(v, one), zero)):
                            b.smem_store_vN(ctl, [zero], one, 1)
                    b.sync_lds_only()
        live = b.cmp_eq(b.readfirstlane(b.vec_extract(b.smem_load_vN(ctl, zero, dtype=I32, n=1), 0)), zero)
        with b.scf_if(live):
            t0 = D.memrealtime(b)
            # 2. record layer L - 1
            with b.scf_if(b.cmp_gt(L, zero)):
                record(b.sub(L, one))
            # 3. predict layer L (table row L, the previous step's routing) -> ascending list
            for j in range(MAX_E // k.threads):
                e = b.add(tid, _c(b, j * k.threads))
                inb = b.cmp_lt(e, E)
                src = e
                if k.mutant == "ids":  # flag[e] = T[e - 1]: every predicted id + 1
                    src = b.select(b.cmp_eq(e, zero), b.sub(E, one), b.sub(e, one))
                f = b.buffer_load_cp(tab_rs, b.select(inb, b.mul(b.add(b.mul(L, E), src), _c(b, 4)), _c(b, OOB)), zero,
                                     dwords=1, aux=16)
                b.smem_store_vN(flg, [e], b.select(inb, f, zero), 1)
            b.sync_lds_only()
            with b.scf_if(b.cmp_lt(tid, _c(b, 64))):
                base = zero
                for c0 in range(0, MAX_E, 64):
                    e = b.add(lane, _c(b, c0))
                    f = b.vec_extract(b.smem_load_vN(flg, e, dtype=I32, n=1), 0)
                    hit = b.cmp_ne(f, zero)
                    m = b.wave_ballot(b.zext(hit, I32))
                    rank = b.add(base, _mbcnt(b, m))
                    with b.scf_if(hit):
                        b.smem_store_vN(lst, [rank], e, 1)
                    cnt = b.inline_asm("s_nop 4\ns_bcnt1_i32_b64 $0, $1", "=s,s", [m], I32, sideeffect=False,
                                       result_name_hint="bc")
                    base = b.add(base, cnt)
                with b.scf_if(b.cmp_eq(lane, zero)):
                    b.smem_store_vN(lst, [_c(b, MAX_E)], base, 1)
            b.sync_lds_only()
            n_ids = b.readfirstlane(b.smin(b.vec_extract(b.smem_load_vN(lst, _c(b, MAX_E), dtype=I32, n=1), 0), budget))
            acc = zero
            for r in range(4):
                sz = sizes[r]
                steps_e = b.div(b.add(sz, _c(b, per - 1)), _c(b, per))
                total = b.mul(steps_e, n_ids)
                my = b.readfirstlane(b.smax(b.div(b.add(b.sub(total, cta), b.sub(grid, one)), grid), zero))
                bv = b.buffer_load_cp(bs_rs, b.mul(b.add(b.mul(L, _c(b, 4)), _c(b, r)), _c(b, 8)), zero, dwords=2, aux=0)
                base_r = b.readfirstlane(b.lor(b.shl(_i64(b, b.vec_extract(bv, 1)), b.const_i64(32)),
                                               _i64(b, b.vec_extract(bv, 0))))

                def issue(i, base_r=base_r, sz=sz, steps_e=steps_e, my=my):
                    idx = b.add(cta, b.mul(i, grid))
                    ok = b.cmp_lt(i, my)
                    idx = b.select(ok, idx, zero)
                    slot = b.readfirstlane(b.div(idx, steps_e))
                    st = b.sub(idx, b.mul(slot, steps_e))
                    e = b.readfirstlane(b.vec_extract(b.smem_load_vN(lst, b.select(ok, slot, zero), dtype=I32, n=1), 0))
                    rs = _rsrc(b, b.add(base_r, b.mul(_i64(b, e), _i64(b, sz))), sz)
                    out = []
                    for j in range(k.unroll):
                        off = b.add(b.mul(st, _c(b, per)), b.add(_c(b, j * k.threads * 16), b.mul(tid, _c(b, 16))))
                        off = b.select(ok, off, _c(b, OOB))
                        v = b.buffer_load_cp(rs, off, zero, dwords=4, aux=k.aux)
                        if k.mutant == "write":
                            b.buffer_store_cp(b.vec_insert(v, b.xor(b.vec_extract(v, 0), one), 0), rs, off, zero, aux=0)
                        out.append(v)
                    return out

                if k.path == "lds":  # DMA into a per-wave scratch row of LDS (overwritten): no data registers
                    wv = D.wave_id(b)
                    lp = b.smem_ptr_add(b.smem_addr_of(dma), b.zext(b.mul(wv, _c(b, 1024)), I64))
                    n_wait = min(k.depth * k.unroll, 48)
                    with b.scf_for(zero, my, one, iv_name=f"sd{r}") as i:
                        idx = b.add(cta, b.mul(i, grid))
                        slot = b.readfirstlane(b.div(idx, steps_e))
                        st = b.sub(idx, b.mul(slot, steps_e))
                        e = b.readfirstlane(b.vec_extract(b.smem_load_vN(lst, slot, dtype=I32, n=1), 0))
                        rs = _rsrc(b, b.add(base_r, b.mul(_i64(b, e), _i64(b, sz))), sz)
                        for j2 in range(k.unroll):
                            off = b.add(b.mul(st, _c(b, per)), b.add(_c(b, j2 * k.threads * 16), b.mul(tid, _c(b, 16))))
                            b.async_buffer_load_lds(rs, lp, off, zero, dwords=4, coherency=k.aux)
                        b.inline_asm(f"s_waitcnt vmcnt({n_wait})", "~{memory}", [], None)
                    continue
                u = k.unroll
                q = [v for i in range(k.depth) for v in issue(_c(b, i))]
                iters = [(f"s{r}q{i}", v) for i, v in enumerate(q)] + [(f"sacc{r}", acc)]
                loop = b.scf_for_iter(zero, my, one, iters, iv_name=f"sp{r}")
                with loop as (i, vals):
                    vals = list(vals)
                    ring, a_ = vals[:-1], vals[-1]
                    nxt = issue(b.add(i, _c(b, k.depth)))
                    for v in ring[:u]:
                        a_ = b.xor(a_, b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
                    b.scf_yield(*(ring[u:] + nxt), a_)
                acc = loop.results[-1]
            with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
                b.store(acc, sinkp, tid)
            b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)
            b.sync_lds_only()
            with b.scf_if(b.cmp_ne(stamps, b.const_i64(0))):
                t1 = D.memrealtime(b)
                with b.scf_if(D.is_thread0(b)):
                    sp = _ptr(b, b.readfirstlane(stamps), I64)
                    o = b.mul(b.add(b.mul(L, grid), cta), _c(b, STAMP_WORDS))
                    b.store(t0, sp, o)
                    b.store(t1, sp, b.add(o, one))
                    b.store(_i64(b, D.xcc_id(b)), sp, b.add(o, _c(b, 2)))
                    b.store(_i64(b, n_ids), sp, b.add(o, _c(b, 3)))
    # the last layer's routing (once K2 published it)
    live = b.cmp_eq(b.readfirstlane(b.vec_extract(b.smem_load_vN(ctl, zero, dtype=I32, n=1), 0)), zero)
    with b.scf_if(b.land(live, b.cmp_eq(cta, zero))):
        wait(b.add(rbase, b.const_i64(4)), b.add(mtag0, b.sub(nl, one)), False)
        record(b.sub(nl, one))
    return b.kernel
