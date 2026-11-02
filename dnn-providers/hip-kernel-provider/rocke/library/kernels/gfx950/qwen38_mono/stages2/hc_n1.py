# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""N1 (DESIGN 12.3, probe P3x variants iv-v): tolerance-class forms of the HC down-projection (A1 / M1) and the
matching A2 / M2 reducers, as rocKE stage functions linked like ``stages2.hc``'s (``ATOM_Q38_EXTERN_STAGES``).

The FlyDSL / WP2a HC splits the 80 k_iters of the down product into 4 partial chains ``kit = p + 4 i``: every partial
reads all 4 streams, so every down task normalizes the WHOLE row (4 x 2560 square sums, 4 inverse RMS, the full MXFP8
pack) before its first MFMA (P3 section 5: that norm, not the weight stream, is A1's time). N1 re-associates the sum
so that a partial reads ONE stream: partial ``s`` = k_iters ``20 s .. 20 s + 19`` (stream ``s``'s 2560 columns), so a
down unit (n-block, stream) needs only its own stream's statistics:

* ``form="ks"`` (N1a, stream-aligned K-split): the unit normalizes its stream locally (the same square-sum tree,
  the same f64 inverse RMS, the same bf16 values and MXFP8 codes as the mono: the pack is bit for bit FlyDSL's),
  runs the 20-step chain over its stream's k_iters and publishes 16 f32 partials; A2 / M2 sum the 4 stream partials
  in fixed order ``((p0 + p1) + p2) + p3``. Only the f32 summation order differs from the mono (tolerance class).
* ``form="na"`` (N1b, norm after the GEMV): the unit packs the UN-normalized ``bf16(x w')`` (no statistics, no inverse
  RMS, no barrier before the pack) and A2 / M2 scale each stream's partial by that stream's inverse RMS, which they
  already compute for their own read epilogue: ``low = ((i0 p0 + i1 p1) + i2 p2) + i3 p3``. The MXFP8 codes of the
  un-normalized streams differ from the normalized ones where ``inv`` is not a power of two (same relative error).
* ``form="nb"`` (N1b proper, norm after the GEMV in the SERVED layout: no spread change, the up stage untouched):
  the down task keeps the WP2a task map and its interleaved partial chains, packs the un-normalized ``bf16(x w')``
  without waiting for statistics, splits each 20-step chain into 4 per-stream accumulators (steps ``5 s .. 5 s + 4``
  read stream ``s``: dependency depth 5), computes the row statistics AFTER issuing the chains (they overlap the
  MFMA latency) and combines ``((i0 a0 + i1 a1) + i2 a2) + i3 a3`` per partial; the partial sum, silu and pack
  (layout 0) or the n-block activation (layout 2) are the WP2a code's. ``g`` / ``kh`` are unused (0 / 1).
* ``form="nbz"`` (TIMING ONLY): ``nb``'s layout with an empty down task (publishes silu(0)): the bound of any
  down-task restructure in the served layout.
* ``form="null"`` (TIMING ONLY, never served, numerically wrong): the units publish zeros at once. The layer time
  with it bounds what ANY restructure of the down unit can still buy (N1c, the partial-down in the M5 / C1
  epilogues, P3x variant vi, included).

Layout (the ATOM build's P3 ``spread`` lever gives the task counts; the sidecar names it, ATOM refuses a mismatch):
``g`` n-blocks per down task, waves ``0 .. g * kh - 1`` active, ``80 / g`` down tasks per request row, then the 4
injection tasks per row (the WP2a code, unchanged). ``g = 1`` needs a units spread value (1 or 10: 80 tasks per row,
rows <= 3) or the deduplicated 4 (80 units for all rows: rows >= 2, each unit normalizes its stream of every row and
loads its weights once); ``g = 4`` the n-block value 2 (or 3 at one row: 20 tasks per row). ``kh`` splits each chain over ``kh``
waves (``20 / kh`` steps each) summed in the CTA in fixed order (another order: tolerance class). Task ``k`` of a
row: stream ``k % 4``, n-blocks ``g (k // 4) .. g (k // 4) + g - 1``. The partial of (n-block ``nb``, stream ``s``)
goes to the spread mailbox pair ``(4 nb + s) 16 + j`` (the P3 units' slots, so A2 / M2 poll the units' pairs).

Mutants (the T0 gate's must-fails, DESIGN 12.4): ``drop`` (the reducer drops stream 3's partial), ``dup`` (the reducer
adds stream 0's partial in place of stream 1's), ``rrms`` (``ks``: the unit's inverse RMS times 1 + 2^-5; ``na`` / ``nb``:
partial ``q`` scaled by stream ``q + 1``'s inverse RMS). In ``nb`` the reducer mutants act in the down task's combine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from rocke.core.ir import F32, I32, I64, IRBuilder

from . import hc as H
from .hc import (
    DOWN_KI,
    HC_COUNT,
    HC_WORDS,
    INJ_TASK,
    LOWRANK,
    OOB_BYTES,
    RECORDS,
    SCALE_ONE,
    STREAM_CHUNKS,
    STREAM_VECS,
    THREADS,
    WAVES,
    DOWN_GROUPS,
    HcKnobs,
    HcSpec,
)
from ..device_function import lower_device_function, write_device_function

__all__ = ["N1Spec", "N1_FORMS", "N1_MUTANTS", "build_n1_kernel", "lower_n1_function", "n1_supported",
           "write_n1_functions", "n1_spec_from_str", "n1_spec_str"]  # fmt: skip

N1_FORMS = ("ks", "na", "null", "nb", "nbz")
N1_MUTANTS = ("drop", "dup", "rrms")
UNITS = 4 * 2 * DOWN_GROUPS  # 80 (n-block, stream) units per request row
NB = 2 * DOWN_GROUPS  # 20 n-blocks of 16 low-rank columns
STREAM_KI = DOWN_KI // HC_COUNT  # 20 k_iters per stream
assert STREAM_KI * HC_COUNT == DOWN_KI and STREAM_VECS * HC_COUNT == H.VECS
LAYOUT_UNITS = (1, 10)  # ATOM spread values with 80 down tasks per row
LAYOUT_NB = (2, 3)  # 20 down tasks per row (3 only at one row: deduplicated over rows otherwise)


@dataclass(frozen=True)
class N1Spec:
    """One N1 stage function. ``knobs``: the WP2a ``HcKnobs`` of the body (mailbox, numa, cache bits, chain knobs ra /
    sgb / scale, amax, shfl4, pre, wait, stats, inv, hbar, anchor); ``knobs.spread`` = the ATOM build's spread value of
    this HC (the task layout). ``form`` / ``g`` / ``kh`` / ``nab`` / ``mutate``: module docstring; ``nab`` (``na``
    only): 1 = round ``x w'`` to bf16 before the pack (the mono's rounding point), 0 = pack the f32 product."""

    stage: str
    kind: str = "delta"
    rows: int = 1
    seq: bool = False
    knobs: HcKnobs = field(default_factory=HcKnobs)
    form: str = "ks"
    g: int = 1
    kh: int = 1
    nab: int = 1
    mutate: str = ""

    @property
    def variant(self) -> str:
        return f"r{self.rows}" + ("s" if self.seq else "")

    @property
    def symbol(self) -> str:
        return f"{H.SYMBOL_PREFIX}{self.stage}_{self.variant}"

    @property
    def hc_spec(self) -> HcSpec:
        """The WP2a spec of the same stage (``spread`` left at the layout value, for the sidecar)."""
        return HcSpec(self.stage, self.kind, self.rows, self.seq, self.knobs)

    @property
    def down(self) -> str:
        return "m1" if self.stage in ("m1", "m2") else "a1"


def n1_spec_str(s: N1Spec) -> str:
    """``form=..,g=..,kh=..[,nab=0][,mutate=..]`` (the N1 part; the body knobs are ``hc.knobs_spec``)."""
    out = f"form={s.form},g={s.g},kh={s.kh}"
    if s.nab != 1:
        out += f",nab={s.nab}"
    if s.mutate:
        out += f",mutate={s.mutate}"
    return out


def n1_spec_from_str(stage, kind, rows, seq, knobs: HcKnobs, n1: str) -> N1Spec:
    kv = dict(x.split("=", 1) for x in n1.split(",") if x)
    return N1Spec(stage, kind, rows, seq, knobs, form=kv.get("form", "ks"), g=int(kv.get("g", 1)),
                  kh=int(kv.get("kh", 1)), nab=int(kv.get("nab", 1)), mutate=kv.get("mutate", ""))  # fmt: skip


def n1_supported(s: N1Spec) -> str | None:
    if s.stage not in H.HC_STAGES:
        return f"stage {s.stage!r}"
    if s.kind not in ("delta", "delta_ple", "qsa"):
        return f"kind {s.kind!r}"
    if not 1 <= s.rows <= H.A.MAX_ROWS:
        return "rows 1..4 (narrow ungrouped builds: ATOM builds the spread there only)"
    if s.seq and s.rows < 2:
        return "a seq build has 2 rows at least"
    if s.form not in N1_FORMS:
        return f"form must be one of {N1_FORMS}"
    if s.form in ("nb", "nbz"):
        k = s.knobs
        if s.stage not in ("a1", "m1"):
            return "nb: the down stage only (the served A2 / M2 stays)"
        if s.g != 0 or s.kh != 1:
            return "nb: g=0 (the served layout), kh=1"
        if not (k.spread == 0 or k.spread == 2 or (k.spread == 3 and s.rows == 1)):
            return "nb: layout 0 (10 tasks per row) or 2 (20 n-block tasks per row; 3 at one row)"
        if k.acc != 1 or k.mutate or getattr(k, "src", "mem") != "mem":
            return "nb: acc / WP2a mutants / src are not combined"
        if k.pre == "args" and (s.stage != "a1" or s.kind == "delta_ple"):
            return "pre=args: A1 of a memory-read attention HC only"
        if s.mutate and s.mutate not in N1_MUTANTS:
            return f"mutate must be one of {N1_MUTANTS}"
        if s.nab not in (0, 1):
            return "nab 0 / 1"
        return None
    if s.g not in (1, 4):
        return "g: 1 (80 tasks per row) or 4 (20 tasks per row): the task layouts ATOM's spread values give"
    if s.kh not in (1, 2, 4, 5) or s.g * s.kh > WAVES:
        return "kh in 1, 2, 4, 5 with g * kh <= 8 waves"
    if s.nab not in (0, 1) or (s.nab == 0 and s.form != "na"):
        return "nab=0: the na form only"
    if s.mutate and s.mutate not in N1_MUTANTS:
        return f"mutate must be one of {N1_MUTANTS}"
    k = s.knobs
    if k.acc != 1:
        return "acc=2 (WP2a's two-accumulator hook) is not combined with N1"
    if getattr(k, "src", "mem") != "mem" or getattr(k, "xhold", "none") != "none" or getattr(k, "order", "weights") != "weights":
        return "src / xhold / order: the memory-read or polled forms of the served path only"
    if k.mutate:
        return "WP2a mutants: use N1Spec.mutate"
    sv = k.spread
    if s.g == 1 and sv == 4:
        if s.rows < 2:
            return "spread 4 (deduplicated units) needs rows >= 2"
        n = UNITS + s.rows * HC_COUNT
        return None if n <= H.GRID else f"{n} tasks > {H.GRID}"
    if s.g == 1 and sv not in LAYOUT_UNITS:
        return f"g=1 needs spread in {LAYOUT_UNITS} (80 down tasks per row) or 4 (deduplicated over rows)"
    if s.g == 4 and not (sv == 2 or (sv == 3 and s.rows == 1)):
        return "g=4 needs spread 2 (or 3 at one row): 20 down tasks per row"
    n = (UNITS // s.g) * s.rows + s.rows * HC_COUNT
    if n > H.GRID:
        return f"{n} tasks > {H.GRID}"
    if k.pre == "args" and (s.stage != "a1" or s.kind == "delta_ple"):
        return "pre=args: A1 of a memory-read attention HC only"
    return None


# ------------------------------------------------------------------ the down unit (A1 / M1)
def _w_own(e, hc, s):
    """``w'`` of vector ``320 s + tid`` (clamped): the polled path loads it ahead of the producer wait."""
    v = e.add(e.mul(s, STREAM_VECS), e.imin(e.tid, STREAM_VECS - 1))
    return e.vec(e.load(e.rsrc(hc.norm_w), e.mul(e.mul(v, 4), 4), 4, e.k.nw_cpol))


def _own(e, hc, s, rr, pre, ws_pre=None):
    """This thread's vector of stream ``s`` of request row ``rr``: (owner predicate, global vector index, h words,
    w' words). Memory-read with ``pre=args``: the preload's round whose vector lies in the stream (at most one: the
    rounds are 512 vectors apart, a stream is 320); otherwise vector ``320 s + tid`` (``tid < 320``). Polled: the
    stream words from the mailbox (``ws_pre``: :func:`_w_own`'s)."""
    b = e.b
    tid = e.tid
    lo = e.mul(s, STREAM_VECS)
    if hc.polled:
        v = e.add(lo, e.imin(tid, STREAM_VECS - 1))
        w = e.poll([(hc.h, e.mb_ix(e.mul(v, 4), rr), 2), (hc.h, e.mb_ix(e.add(e.mul(v, 4), 2), rr), 2)])
        hs = w[0] + w[1]
        return e.lt(tid, STREAM_VECS), v, hs, ws_pre
    if pre is not None:
        hp, wp = pre
        own, v, hsel, wsel = None, None, None, None
        for r in range(3):
            vr = e.add(tid, THREADS * r)
            c = b.land(e.ge(vr, lo), b.land(e.lt(vr, e.add(lo, STREAM_VECS)), e.lt(vr, H.VECS)))
            if own is None:
                own, v, hsel, wsel = c, vr, list(hp[r]), list(wp[r])
            else:
                own = b.lor(own, c)
                v = b.select(c, vr, v)
                hsel = [b.select(c, x, y) for x, y in zip(hp[r], hsel)]
                wsel = [b.select(c, x, y) for x, y in zip(wp[r], wsel)]
        return own, v, hsel, wsel
    own = e.lt(tid, STREAM_VECS)
    v = e.add(lo, e.imin(tid, STREAM_VECS - 1))
    off_h = e.mul(e.row_ix(e.mul(v, 4), rr, HC_WORDS), 4)
    hs = e.vec(e.load(e.rsrc(hc.h), off_h, 4, e.k.nw_cpol))
    ws = e.vec(e.load(e.rsrc(hc.norm_w), e.mul(e.mul(v, 4), 4), 4, e.k.nw_cpol))
    return own, v, hs, ws


def _scales_b(e, hc, active, nb, s, i0, steps, kh):
    """The weights' E8M0 scale of every chain step, as a dword whose byte 0 is the step's scale (op_sel 0). Step ``i``
    reads k_iter ``kit = 20 s + i0 + i``: chain block ``(nb, kit % 4)`` (``p2.scales_chain``), index ``kit // 4``.
    ``kh == 1``: two dword loads per part (the 5 indices ``5 s .. 5 s + 4`` span two dwords), then a select and a
    shift per step; else one dword load per step."""
    b = e.b
    rs = e.rsrc(hc.s_down, RECORDS["s_down"])
    cp = e.k.w_cpol
    lane = e.lane

    def dword(p, dd):  # dword ``dd`` (0..4: sblock dd // 4, word dd % 4) of chain block (nb, p)
        sb, dw = e.udiv(dd, 4), e.umod(dd, 4)
        off = e.add(e.mul(e.add(e.mul(e.add(e.mul(nb, 4), p), 2 * 64), e.mul(sb, 64), lane), 16), e.mul(dw, 4))
        return e.load(rs, e.sel(active, off, OOB_BYTES), 1, cp)

    s5 = e.mul(s, 5)
    if kh == 1:
        d0 = e.udiv(s5, 4)
        r5 = e.umod(s5, 4)
        words = {p: (dword(p, d0), dword(p, e.add(d0, 1))) for p in range(4)}

        def at(i):
            p, q = i % 4, i // 4
            w0, w1 = words[p]
            dw = b.select(e.ge(e.add(r5, q), 4), w1, w0)
            return b.lshr(dw, e.mul(e.umod(e.add(r5, q), 4), 8))

        return [at(i) for i in range(steps)]
    out = []
    for i in range(steps):
        x = e.add(i0, i)  # step's index past 20 s
        p = e.umod(x, 4)
        ip = e.add(s5, e.udiv(x, 4))
        dw = dword(p, e.udiv(ip, 4))
        out.append(b.lshr(dw, e.mul(e.umod(ip, 4), 8)))
    return out


def _scales_a(e, ne_base, kit0, steps):
    """The activations' scales of the chain over ``kit0 + i``: per-step words (``ldsstep`` / ``hoist``: op_sel 0) or
    ``opsel``-packed (4 per dword, op_sel ``i % 4``)."""
    b = e.b
    g = e.udiv(e.lane, 16)

    def word(i):
        return e.ld(e.add(ne_base, e.mul(e.add(kit0, i), 4), g))

    if e.k.scale != "opsel":
        return lambda i: (word(i), 0)
    words = [word(i) for i in range(steps)]
    packed = []
    for j in range(0, steps, 4):
        p = words[j]
        for q in range(1, min(4, steps - j)):
            p = b.lor(p, b.shl(words[j + q], e.c(8 * q)))
        packed.append(p)
    return lambda i: (packed[i // 4], i % 4)


def _frag(e, n8, kit):
    b = e.b
    off = e.add(n8, e.mul(kit, 32), e.mul(e.udiv(e.lane, 16), 4))
    a0 = e.ldn(off, 4)
    a1 = e.ldn(e.add(off, 16), 4)
    return b.vec_pack(e.vec(a0) + e.vec(a1), I32)


def _n1_unit(e, hc, n1: N1Spec, k, rr, pre):
    """Down task ``k`` (< 80 / g) of request row ``rr`` (``e``: a one-row view; LDS row 0)."""
    b = e.b
    tid, lane, wave = e.tid, e.lane, e.wave
    g, kh = n1.g, n1.kh
    steps = STREAM_KI // kh
    s = e.umod(k, 4)
    nb = e.add(e.mul(e.udiv(k, 4), g), e.umod(wave, g)) if g > 1 else e.udiv(k, 4)
    hh = e.udiv(wave, g) if g > 1 else wave
    active = e.lt(wave, g * kh)
    i0 = e.mul(hh, steps)
    kit0 = e.add(e.mul(s, STREAM_KI), i0)

    def box(i):
        return e.mb_ix(H._spread_box_pair(e, hc, i), rr)

    if n1.form == "null":
        with b.scf_if(b.land(e.lt(wave, g), e.lt(lane, 16))):
            e.put("isc", box(e.add(e.mul(e.add(e.mul(nb, 4), s), 16), lane)), e.f(0.0))
        return

    def weights():
        rw = e.rsrc(hc.w_down, RECORDS["w_down"])
        ws = []
        for i in range(steps):
            off = e.mul(e.add(e.mul(e.add(e.mul(nb, DOWN_KI), e.add(kit0, i)), 64), lane), 16)
            ws.append(e.load(rw, e.sel(active, off, OOB_BYTES), 4, e.k.w_cpol))
        return ws, _scales_b(e, hc, active, nb, s, i0, steps, kh)

    if hc.polled:
        ws_pre = _w_own(e, hc, s)
        units = weights()
        e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.add(e.mul(s, 2 * STREAM_CHUNKS), e.mul(m, hc.hint_stride)), rr)),
                         hc.producers)  # fmt: skip
        own, v, hs, ws = _own(e, hc, s, rr, None, ws_pre)
    else:
        own, v, hs, ws = _own(e, hc, s, rr, pre if hc.down == "a1" else None)
        b.sched_barrier(0)
        if e.k.hbar:
            e.barrier()
        units = weights()
    xs, wf = e.bf16x8(hs), e.bf16x8(ws)
    if n1.form == "ks":
        if hc.polled and e.k.stats == "halves":
            # chunk c of the stream: the producer's 16-column sums 2 (80 s + c), + 1 (the norm's own tree)
            with b.scf_if(e.lt(tid, STREAM_CHUNKS)):
                pr = e.add(e.mul(s, 2 * STREAM_CHUNKS), e.mul(tid, 2))
                (vv,) = e.poll([(hc.hs, e.mb_ix(pr, rr), 2)])
                e.st(e.lix("hsum", e.add(e.mul(s, STREAM_CHUNKS), tid)), b.fadd(b.bitcast(vv[0], F32), b.bitcast(vv[1], F32)))
        else:
            cs = H._chunk_sum(e, hs)
            with b.scf_if(b.land(own, e.eq(e.umod(lane, 4), 0))):
                e.st(e.lix("hsum", e.udiv(v, 4)), cs)
        e.barrier()
        inv = H._inv_of(e, s, 0)  # every wave: the stream's inverse RMS (FlyDSL's tree, f64 sqrt / divide)
        if n1.mutate == "rrms":
            inv = b.fmul(inv, e.f(1.0 + 2.0**-5))
        y = [e.bf16r(b.fmul(b.fmul(xs[i], inv), wf[i])) for i in range(8)]
    else:  # na: the un-normalized product (the reducer applies the inverse RMS)
        y = [b.fmul(xs[i], wf[i]) for i in range(8)]
        if n1.nab:
            y = [e.bf16r(x) for x in y]
    ex, words = H._pack8(e, y)
    with b.scf_if(own):
        e.st(e.lix("n8", e.mul(v, 2)), words, 2)
        with b.scf_if(e.eq(e.umod(lane, 4), 0)):
            e.st(e.lix("ne", e.udiv(v, 4)), ex)
    e.barrier()
    n8, ne = e.lbase("n8"), e.lbase("ne")
    ws_, ss_ = units
    with b.scf_if(active):
        acc = H._chain(
            e, steps,
            frag=lambda i: _frag(e, n8, e.add(kit0, i)),
            scale_a=_scales_a(e, ne, kit0, steps),
            weight=lambda i: ws_[i],
            scale_b=lambda i: ss_[i],
            opsel_b=lambda i: 0,
        )  # fmt: skip
        a0 = b.vec_extract(acc, 0)
        if kh == 1:
            with b.scf_if(e.lt(lane, 16)):
                e.put("isc", box(e.add(e.mul(e.add(e.mul(nb, 4), s), 16), lane)), a0)
        else:
            with b.scf_if(e.lt(lane, 16)):
                e.st(e.lix("red", e.add(e.mul(wave, 16), lane)), a0)
    if kh > 1:
        e.barrier()
        with b.scf_if(b.land(e.lt(wave, g), e.lt(lane, 16))):
            tot = e.ld(e.lix("red", e.add(e.mul(wave, 16), lane)), F32)
            for q in range(1, kh):
                tot = b.fadd(tot, e.ld(e.lix("red", e.add(e.mul(e.add(wave, g * q), 16), lane)), F32))
            e.put("isc", box(e.add(e.mul(e.add(e.mul(nb, 4), s), 16), lane)), tot)


def _pack_z(e, hs, ws, nab):
    """``norm_pack`` of the UN-normalized ``x w'`` (one row, the 3 rounds of the full row): LDS ``n8`` / ``ne``."""
    b = e.b
    for r in range(3):
        xs, wf = e.bf16x8(hs[r]), e.bf16x8(ws[r])
        y = [b.fmul(xs[i], wf[i]) for i in range(8)]
        if nab:
            y = [e.bf16r(x) for x in y]
        ex, words = H._pack8(e, y)
        v = e.add(e.tid, THREADS * r)
        with b.scf_if(e.lt(v, H.VECS)):
            e.st(e.lix("n8", e.mul(v, 2)), words, 2)
            with b.scf_if(e.eq(e.umod(e.lane, 4), 0)):
                e.st(e.lix("ne", e.udiv(v, 4)), ex)
    e.barrier()


def _nb_task(e, hc, n1: N1Spec, nb, part, active, rr, pre, publish):
    """One N1b down task (one-row view): the WP2a loads, the z pack, 4 per-stream sub-chains, the statistics, the
    per-stream scaling into this wave's partial (LDS ``red``), then ``publish()`` (the WP2a reduce / pack)."""
    b = e.b
    lane, wave = e.lane, e.wave
    if n1.form == "nbz":  # TIMING ONLY: the served layout's down task with nothing to do (the N1c / any-form bound)
        with b.scf_if(b.land(active, e.lt(lane, 16))):
            e.st(e.lix("red", e.add(e.mul(wave, 16), lane)), e.f(0.0))
        e.barrier()
        publish()
        return
    if hc.polled:
        ws = H._norm_w(e, hc)
        units = H._a1_units(e, hc, active, nb, part)
        e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(m, hc.hint_stride), rr)), hc.producers)
        (hs,) = H._poll_streams(e, hc, [rr])
    else:
        if pre is not None:
            hs, ws = pre
        else:
            hs = H._norm_load(e, hc, rr)
            ws = H._norm_w(e, hc)
        b.sched_barrier(0)
        if e.k.hbar:
            e.barrier()
        units = H._a1_units(e, hc, active, nb, part)
    _pack_z(e, hs, ws, n1.nab)
    n8, ne = e.lbase("n8"), e.lbase("ne")
    wts, ss = units
    sa = H._down_scales(e, ne, part)
    accs = []
    # the 4 per-stream sub-chains (an inactive wave reads out-of-range weights: zeros; it publishes nothing)
    for st in range(HC_COUNT):
        accs.append(H._chain(
            e, 5,
            frag=lambda j, st=st: H._a_frag_dyn(e, n8, part, 5 * st + j),
            scale_a=lambda j, st=st: sa(5 * st + j),
            weight=lambda j, st=st: wts[5 * st + j],
            scale_b=lambda j, st=st: ss[(5 * st + j) // 4],
            opsel_b=lambda j, st=st: (5 * st + j) % 4,
        ))  # fmt: skip
    # the statistics after the chains' issue (FlyDSL's tree and f64 inverse RMS: the mono's inv, bit for bit)
    if hc.polled and e.k.stats == "halves":
        H._halves_sums(e, hc, [rr])
    else:
        H._norm_sums(e, hs)
    e.barrier()
    invs = [H._inv_of(e, e.c(q), 0) for q in range(HC_COUNT)]
    if n1.mutate == "rrms":  # a wrong per-stream inverse RMS: stream q scaled by stream q + 1's
        invs = invs[1:] + invs[:1]
    a = [b.vec_extract(x, 0) for x in accs]
    t = [b.fmul(a[q], invs[q]) for q in range(HC_COUNT)]
    if n1.mutate == "dup":
        t[1] = t[0]
    low = b.fadd(b.fadd(t[0], t[1]), t[2])
    if n1.mutate != "drop":
        low = b.fadd(low, t[3])
    with b.scf_if(b.land(active, e.lt(lane, 16))):
        e.st(e.lix("red", e.add(e.mul(wave, 16), lane)), low)
    e.barrier()
    publish()


def _emit_nb_a1(e, hc, n1: N1Spec):
    b = e.b
    t = e.a.task
    rows = e.rows
    sv = e.k.spread
    o = H._one_row(e)
    pre = H._pre_values(e) if e.k.pre == "args" else None
    per = DOWN_GROUPS if sv == 0 else NB
    n_down = per * rows
    with b.scf_if(e.lt(t, n_down)):
        k = t if rows == 1 else e.umod(t, per)
        rr = 0 if rows == 1 else e.udiv(t, per)
        part = e.umod(o.wave, 4)
        if sv == 0:
            nb = e.add(e.mul(k, 2), e.udiv(o.wave, 4))

            def publish():  # ``_a1_down``'s wave-0 reduce, silu and 32-column pack
                with b.scf_if(e.eq(o.wave, 0)):
                    lane = o.lane
                    col = e.add(e.mul(e.udiv(e.umod(lane, 32), 16), 64), e.umod(lane, 16))
                    low = o.ld(o.lix("red", col), F32)
                    for q in range(1, 4):
                        low = b.fadd(low, o.ld(o.lix("red", e.add(col, 16 * q)), F32))
                    act = o.silu_q(low)
                    amax = o.absmax_start(act)
                    for off in (1, 2, 4, 8, 16):
                        amax = o.xred(amax, off, o.absmax2)
                    ex, scale = o.e8m0(o.absmax_end(amax))
                    v = o.lane_gather(act, e.mul(e.umod(lane, 4), 8), 8)
                    lo, hi = o.fp8_word(*v[0:4], scale), o.fp8_word(*v[4:8], scale)
                    with b.scf_if(e.lt(lane, 4)):
                        o.put2(hc.lr8, o.mb_ix(e.add(e.mul(k, 8), e.mul(lane, 2)), rr), lo, hi)
                    with b.scf_if(e.eq(lane, 0)):
                        o.put(hc.lre, o.mb_ix(k, rr), ex)

            _nb_task(o, hc, n1, nb, part, e.lt(o.wave, WAVES), rr, pre, publish)
        else:
            nb = k

            def publish():  # ``_spread_down``'s in-CTA ordered sum, silu, the n-block's activations
                with b.scf_if(e.eq(o.wave, 0)):
                    col = e.umod(o.lane, 16)
                    low = o.ld(o.lix("red", col), F32)
                    for q in range(1, 4):
                        low = b.fadd(low, o.ld(o.lix("red", e.add(col, 16 * q)), F32))
                    act = o.silu_q(low)
                    with b.scf_if(e.lt(o.lane, 16)):
                        o.put("isc", o.mb_ix(H._spread_box_pair(o, hc, e.add(e.mul(nb, 16), o.lane)), rr), act)

            _nb_task(o, hc, n1, nb, part, e.lt(o.wave, 4), rr, pre, publish)
    with b.scf_if(e.ge(t, n_down)):
        inj = e.imax(b.sub(t, e.c(n_down)), 0)
        H._a1_task(o, hc, e.add(INJ_TASK, e.umod(inj, HC_COUNT)), e.udiv(inj, HC_COUNT),
                   e.lt(t, n_down + rows * HC_COUNT), pre)  # fmt: skip


def _n1_unit_dd(e, hc, n1: N1Spec, k):
    """Deduplicated down unit ``k`` (< 80; ATOM spread value 4): (n-block ``k // 4``, stream ``k % 4``) for EVERY
    request row: each row's stream statistics and pack in its own LDS row, one weight stream for all rows (the
    MFMA's A row ``r`` = request row ``r``), each row's partial to its own mailbox row. ``e``: the multi-row view."""
    b = e.b
    tid, lane, wave = e.tid, e.lane, e.wave
    rows = e.rows
    kh = n1.kh
    steps = STREAM_KI // kh
    s = e.umod(k, 4)
    nb = e.udiv(k, 4)
    hh = wave
    active = e.lt(wave, kh)
    i0 = e.mul(hh, steps)
    kit0 = e.add(e.mul(s, STREAM_KI), i0)

    def box(i, r):
        return e.mb_ix(H._spread_box_pair(e, hc, i), r)

    if n1.form == "null":
        with b.scf_if(b.land(e.eq(wave, 0), e.lt(lane, 16))):
            for r in range(rows):
                e.put("isc", box(e.add(e.mul(k, 16), lane), r), e.f(0.0))
        return

    def weights():
        rw = e.rsrc(hc.w_down, RECORDS["w_down"])
        ws = []
        for i in range(steps):
            off = e.mul(e.add(e.mul(e.add(e.mul(nb, DOWN_KI), e.add(kit0, i)), 64), lane), 16)
            ws.append(e.load(rw, e.sel(active, off, OOB_BYTES), 4, e.k.w_cpol))
        return ws, _scales_b(e, hc, active, nb, s, i0, steps, kh)

    lo = e.mul(s, STREAM_VECS)
    own = e.lt(tid, STREAM_VECS)
    v = e.add(lo, e.imin(tid, STREAM_VECS - 1))
    if hc.polled:
        ws_ = _w_own(e, hc, s)
        units = weights()
        e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.add(e.mul(s, 2 * STREAM_CHUNKS), e.mul(e.umod(m, hc.producers), hc.hint_stride)),
                                                   e.udiv(m, hc.producers))), hc.producers * rows)  # fmt: skip
        hss = []
        for r in range(rows):
            w = e.poll([(hc.h, e.mb_ix(e.mul(v, 4), r), 2), (hc.h, e.mb_ix(e.add(e.mul(v, 4), 2), r), 2)])
            hss.append(w[0] + w[1])
    else:
        rh = e.rsrc(hc.h)
        hss = [e.vec(e.load(rh, e.mul(e.row_ix(e.mul(v, 4), r, HC_WORDS), 4), 4, e.k.nw_cpol)) for r in range(rows)]
        ws_ = e.vec(e.load(e.rsrc(hc.norm_w), e.mul(e.mul(v, 4), 4), 4, e.k.nw_cpol))
        b.sched_barrier(0)
        if e.k.hbar:
            e.barrier()
        units = weights()
    wf = e.bf16x8(ws_)
    if n1.form == "ks":
        for r in range(rows):
            cs = H._chunk_sum(e, hss[r])
            with b.scf_if(b.land(own, e.eq(e.umod(lane, 4), 0))):
                e.st(e.lix("hsum", e.udiv(v, 4), r), cs)
        e.barrier()
    for r in range(rows):
        xs = e.bf16x8(hss[r])
        if n1.form == "ks":
            inv = H._inv_of(e, s, r)
            if n1.mutate == "rrms":
                inv = b.fmul(inv, e.f(1.0 + 2.0**-5))
            y = [e.bf16r(b.fmul(b.fmul(xs[i], inv), wf[i])) for i in range(8)]
        else:
            y = [b.fmul(xs[i], wf[i]) for i in range(8)]
            if n1.nab:
                y = [e.bf16r(x) for x in y]
        ex, words = H._pack8(e, y)
        with b.scf_if(own):
            e.st(e.lix("n8", e.mul(v, 2), r), words, 2)
            with b.scf_if(e.eq(e.umod(lane, 4), 0)):
                e.st(e.lix("ne", e.udiv(v, 4), r), ex)
    e.barrier()
    mr = e.imin(e.umod(lane, 16), rows - 1)
    n8 = e.add(e.lbase("n8"), e.mul(mr, e.L.stride["n8"]))
    ne = e.add(e.lbase("ne"), e.mul(mr, e.L.stride["ne"]))
    ws2, ss2 = units
    with b.scf_if(active):
        acc = H._chain(
            e, steps,
            frag=lambda i: _frag(e, n8, e.add(kit0, i)),
            scale_a=_scales_a(e, ne, kit0, steps),
            weight=lambda i: ws2[i],
            scale_b=lambda i: ss2[i],
            opsel_b=lambda i: 0,
        )  # fmt: skip
        with b.scf_if(e.lt(lane, 16)):
            for r in range(rows):
                if kh == 1:
                    e.put("isc", box(e.add(e.mul(k, 16), lane), r), b.vec_extract(acc, r))
                else:
                    e.st(e.lix("red", e.add(e.mul(wave, 16), lane), r), b.vec_extract(acc, r))
    if kh > 1:
        e.barrier()
        with b.scf_if(b.land(e.eq(wave, 0), e.lt(lane, 16))):
            for r in range(rows):
                tot = e.ld(e.lix("red", lane, r), F32)
                for q in range(1, kh):
                    tot = b.fadd(tot, e.ld(e.lix("red", e.add(16 * q, lane), r), F32))
                e.put("isc", box(e.add(e.mul(k, 16), lane), r), tot)


def _emit_n1_a1(e, hc, n1: N1Spec):
    b = e.b
    t = e.a.task
    rows = e.rows
    dedupe = n1.g == 1 and e.k.spread == 4
    per = UNITS // n1.g
    n_down = per if dedupe else per * rows
    one = H._one_row(e)
    pre = H._pre_values(e) if e.k.pre == "args" else None
    if dedupe:
        with b.scf_if(e.lt(t, n_down)):
            _n1_unit_dd(e, hc, n1, t)
        with b.scf_if(e.ge(t, n_down)):
            inj = e.imax(b.sub(t, e.c(n_down)), 0)
            H._a1_task(one, hc, e.add(INJ_TASK, e.umod(inj, HC_COUNT)), e.udiv(inj, HC_COUNT),
                       e.lt(t, n_down + rows * HC_COUNT), pre)  # fmt: skip
        return
    with b.scf_if(e.lt(t, n_down)):
        if rows == 1:
            _n1_unit(one, hc, n1, t, 0, pre)
        else:
            _n1_unit(one, hc, n1, e.umod(t, per), e.udiv(t, per), pre)
    with b.scf_if(e.ge(t, n_down)):
        inj = e.imax(b.sub(t, e.c(n_down)), 0)
        H._a1_task(one, hc, e.add(INJ_TASK, e.umod(inj, HC_COUNT)), e.udiv(inj, HC_COUNT),
                   e.lt(t, n_down + rows * HC_COUNT), pre)  # fmt: skip


# ------------------------------------------------------------------ the reducer (A2 / M2)
def _n1_pack(e, hc, n1: N1Spec):
    """``hc._spread_pack`` of the units (4 stream partials per column), the stream sum in fixed order; ``na``: each
    partial times its stream's inverse RMS first. Item ``(row, group)`` on a 32-lane half of a wave."""
    b = e.b
    lane, wave, rows = e.lane, e.wave, e.rows
    items = DOWN_GROUPS * rows
    slots = 2 * WAVES
    passes = -(-items // slots)
    half = e.udiv(lane, 32)
    c32 = e.umod(lane, 32)
    specs, its = [], []
    for ps in range(passes):
        raw = e.add(e.mul(wave, 2), half, ps * slots)
        it = e.imin(raw, items - 1)
        r = 0 if rows == 1 else e.udiv(it, DOWN_GROUPS)
        g = e.umod(it, DOWN_GROUPS)
        col = e.add(e.mul(g, 32), c32)
        nb, j = e.udiv(col, 16), e.umod(col, 16)
        specs += [("isc", e.mb_ix(H._spread_box_pair(e, hc, e.add(e.mul(e.add(e.mul(nb, 4), q), 16), j)), r), 1)
                  for q in range(4)]  # fmt: skip
        its.append((raw, r, g))
    vals = e.poll(specs)
    for ps in range(passes):
        raw, r, g = its[ps]
        v = [b.bitcast(vals[ps * 4 + q][0], F32) for q in range(4)]
        if n1.form == "na":
            sq = (lambda q: (q + 1) % 4) if n1.mutate == "rrms" else (lambda q: q)
            v = [b.fmul(v[q], H._inv_for(e, e.c(sq(q)), r)) for q in range(4)]
        if n1.mutate == "dup":
            v = [v[0], v[0], v[2], v[3]]
        low = b.fadd(v[0], v[1])
        low = b.fadd(low, v[2])
        if n1.mutate != "drop":
            low = b.fadd(low, v[3])
        act = e.silu_q(low)
        amax = e.absmax_start(act)
        for off in (1, 2, 4, 8, 16):
            amax = e.xred(amax, off, e.absmax2)
        ex, scale = e.e8m0(e.absmax_end(amax))
        w8 = e.lane_gather(act, e.add(e.mul(half, 32), e.mul(e.umod(lane, 4), 8)), 8)
        lo, hi = e.fp8_word(*w8[0:4], scale), e.fp8_word(*w8[4:8], scale)
        ok = e.lt(raw, items)
        with b.scf_if(b.land(e.lt(c32, 4), ok)):
            e.st(e.lix("l8", e.add(e.mul(g, 8), e.mul(c32, 2)), r), b.vec_pack([lo, hi], I32), 2)
        with b.scf_if(b.land(e.eq(c32, 0), ok)):
            e.st(e.lix("le", g, r), ex)


def _emit_n1_a2(e, hc, n1: N1Spec):
    """``hc._emit_a2`` (narrow builds) with the N1 units as the low-rank producers (``_n1_pack``)."""
    b = e.b
    tid, lane, wave, rows = e.tid, e.lane, e.wave, e.rows
    t = e.a.task
    stream = e.udiv(wave, 2)
    atom = e.umod(wave, 2)
    nb = e.add(e.mul(t, 2), e.mul(stream, H.UP_NB), atom)
    j = e.umod(lane, 16)
    idx = e.add(e.mul(stream, H.HIDDEN), e.mul(t, 32), e.mul(atom, 16), j)
    with b.scf_if(e.lt(t, H.TASKS_UP)):
        rwu, rsu, rnw = e.rsrc(hc.w_up), e.rsrc(hc.s_up), e.rsrc(hc.norm_w)
        cp = e.k.w_cpol

        def weights():
            wu = [e.load(rwu, e.mul(e.add(e.mul(e.add(e.mul(nb, 4), kit), 64), lane), 16), 4, cp) for kit in range(H.UP_KI)]
            su = e.load(rsu, e.mul(e.add(e.mul(nb, 64), lane), 4), 1, cp)
            ww = e.load(rnw, e.mul(e.udiv(idx, 2), 4), 1, e.k.nw_cpol)
            return wu, su, ww

        h_first = e.k.a2order == "h_first" and not hc.polled
        if not h_first:
            wu, su, ww = weights()
        if hc.polled:
            e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(e.umod(m, hc.producers), hc.hint_stride),
                                                       e.udiv(m, hc.producers))), hc.producers * rows)  # fmt: skip
            hws = [v[0] for v in e.poll([(hc.h, e.mb_ix(e.udiv(idx, 2), r), 1) for r in range(rows)])]
            H._halves_sums(e, hc)
        else:
            rh = e.rsrc(hc.h)
            hws = [e.load(rh, e.mul(e.row_ix(e.udiv(idx, 2), r, HC_WORDS), 4), 1, e.k.nw_cpol) for r in range(rows)]
            nl = [H._norm_load(e, hc, r) for r in range(rows)] if h_first else None
            if h_first:
                wu, su, ww = weights()
            for r in range(rows):
                H._norm_sums(e, nl[r] if h_first else H._norm_load(e, hc, r), r)
        H._norm_inv(e, rows)
        # the units' last pair (j = 15) of every (n-block, stream) of every row is the hint
        e.wait_producers(lambda m: ("isc", e.mb_ix(H._spread_box_pair(e, hc, e.add(e.mul(e.umod(m, UNITS), 16), 15)),
                                                   e.udiv(m, UNITS))), UNITS * rows, after_barrier=True)  # fmt: skip
        _n1_pack(e, hc, n1)
        with b.scf_if(b.land(e.ge(tid, 128), e.lt(tid, 128 + 16))):
            for r in range(rows):
                e.st(e.lix("l8", e.add(b.sub(tid, e.c(128)), LOWRANK // 4), r), e.c(0))
        with b.scf_if(b.land(e.ge(tid, 192), e.lt(tid, 192 + 2))):
            for r in range(rows):
                e.st(e.lix("le", e.add(b.sub(tid, e.c(192)), DOWN_GROUPS), r), e.c(SCALE_ONE))
        e.barrier()
        if rows == 1:
            xl8, xle = e.lbase("l8"), e.lbase("le")
        else:
            mr = e.imin(e.umod(lane, 16), rows - 1)
            xl8 = e.add(e.lbase("l8"), e.mul(mr, e.L.stride["l8"]))
            xle = e.add(e.lbase("le"), e.mul(mr, e.L.stride["le"]))
        g = e.udiv(lane, 16)
        sa = (lambda i: (e.ld(e.add(xle, e.mul(i, 4), g)), 0))
        if e.k.scale == "opsel":
            w3 = [e.ld(e.add(xle, 4 * i, g)) for i in range(H.UP_KI)]
            p = b.lor(b.lor(w3[0], b.shl(w3[1], e.c(8))), b.shl(w3[2], e.c(16)))
            sa = lambda i: (p, i)  # noqa: E731
        elif e.k.scale == "hoist":
            hs_ = [e.ld(e.add(xle, e.mul(i, 4), g)) for i in range(H.UP_KI)]
            sa = lambda i: (hs_[i], 0)  # noqa: E731
        acc = H._chain(e, H.UP_KI, frag=lambda i: H._a_frag(e, xl8, i), scale_a=sa, weight=lambda i: wu[i],
                       scale_b=lambda i: su, opsel_b=lambda i: i)  # fmt: skip
        for r in range(rows):
            hw = hws[r]
            rw = e.bf16r(e.sigmoid_acc(e.bf16r(b.vec_extract(acc, r))))
            even = e.eq(e.umod(j, 2), 0)
            x = b.bitcast(b.select(even, b.shl(hw, e.c(16)), b.land(hw, e.c(-65536))), F32)
            w = b.bitcast(b.select(even, b.shl(ww, e.c(16)), b.land(ww, e.c(-65536))), F32)
            inv = H._inv_for(e, stream, r)
            y = e.bf16r(b.fmul(b.fmul(x, inv), w))
            with b.scf_if(e.lt(lane, 16)):
                e.st(e.lix("prod", e.add(e.mul(stream, 32), e.mul(atom, 16), j), r), e.bf16r(b.fmul(rw, y)))
        e.barrier()
        cond = e.eq(wave, 0) if rows == 1 else e.lt(wave, rows)
        _mix(e, hc, t, cond)


def _mix(e, hc, t, cond):
    """A2 / M2's epilogue (``hc._emit_a2``'s, the same operations): wave ``r`` (row ``r``; wave 0 at one row) averages
    the 4 streams' products and publishes the mixed row's fp8 words, bf16 words and exponent."""
    b = e.b
    lane = e.lane
    with b.scf_if(cond):
        rq = 0 if e.rows == 1 else e.imin(e.wave, e.rows - 1)
        col = e.umod(lane, 32)
        s = e.f(0.0)
        for bb in range(HC_COUNT):
            s = b.fadd(s, e.ld(e.lix("prod", e.add(bb * 32, col), rq), F32))
        mixed = e.bf16r(b.fmul(s, e.f(0.25)))
        amax = e.absmax_start(mixed)
        for off in (1, 2, 4, 8, 16):
            amax = e.xred(amax, off, e.absmax2)
        ex, scale = e.e8m0(e.absmax_end(amax))
        v8 = e.lane_gather(mixed, e.mul(e.umod(lane, 4), 8), 8)
        lo, hi = e.fp8_word(*v8[0:4], scale), e.fp8_word(*v8[4:8], scale)
        v4 = e.lane_gather(mixed, e.mul(e.umod(lane, 8), 4), 4)
        with b.scf_if(e.lt(lane, 4)):
            e.put2(hc.x8, e.mb_ix(e.add(e.mul(t, 8), e.mul(lane, 2)), rq), lo, hi)
        with b.scf_if(e.lt(lane, 8)):
            e.put2(hc.xbf, e.mb_ix(e.add(e.mul(t, 16), e.mul(lane, 2)), rq),
                   e.bf16_pair(v4[0], v4[1]), e.bf16_pair(v4[2], v4[3]))  # fmt: skip
        with b.scf_if(e.eq(lane, 0)):
            e.put(hc.xe, e.mb_ix(t, rq), ex)


# ------------------------------------------------------------------ build / write
def build_n1_kernel(n1: N1Spec):
    why = n1_supported(n1)
    if why:
        raise ValueError(f"N1 {n1.stage} {n1.kind} {n1.variant}: {why}")
    spec = n1.hc_spec
    b = IRBuilder(n1.symbol)
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in H.abi_params(spec):
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))))  # fmt: skip
    if spec.knobs.pre == "args":
        a.pre = [vals[f"pre{i}"] for i in range(H.PRE_WORDS)]
    # (``hc.spec_lds`` is newer than the served stack's ``hc.py``: narrow ungrouped builds use ``hc_lds``)
    e = H._Hc(b, a, spec, H.hc_lds(spec.rows, spec.seq))
    hc = H._hc(spec)
    if n1.form in ("nb", "nbz"):
        _emit_nb_a1(e, hc, n1)
    elif n1.stage in ("a1", "m1"):
        _emit_n1_a1(e, hc, n1)
    else:
        _emit_n1_a2(e, hc, n1)
    b.ret()
    return b.kernel


def lower_n1_function(n1: N1Spec, *, arch="gfx950", flavor=None):
    return lower_device_function(build_n1_kernel(n1), arch=arch, symbol=n1.symbol, llvm_flavor=flavor,
                                 knobs=n1.knobs.fn)  # fmt: skip


def sidecar(n1: N1Spec) -> dict:
    meta = H.sidecar(n1.hc_spec)
    meta["hc"]["tolerance"] = n1.form not in ("null", "nbz")
    meta["hc"]["n1"] = n1_spec_str(n1)
    if n1.form in ("null", "nbz"):
        meta["hc"]["timing_only"] = True
    return meta


def write_n1_functions(out_dir: str, specs, flavor=None) -> list:
    """``<out_dir>/<kind>/<stage>_<variant>.{ll,json}`` per N1 spec (as ``hc.write_hc_functions``)."""
    written = []
    for n1 in specs:
        fn = lower_n1_function(n1, flavor=flavor)
        d = os.path.join(out_dir, n1.kind)
        paths = write_device_function(fn, d, stem=f"{n1.stage}_{n1.variant}", bitcode=False)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(sidecar(n1))
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append(paths["ll"])
    return written


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="emit the N1 HC stage functions (P3x / WP15)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="a1,a2")
    ap.add_argument("--kinds", default="delta,delta_ple,qsa")
    ap.add_argument("--variants", default="r1,r4s")
    ap.add_argument("--knobs", default="numa=1", help="HcKnobs spec of the body (spread = the ATOM layout value)")
    ap.add_argument("--n1", default="form=ks,g=1,kh=1")
    ap.add_argument("--flavor", default=None)
    a = ap.parse_args(argv)
    base = H.knobs_from_spec(a.knobs)
    specs = []
    for kind in a.kinds.split(","):
        for v in a.variants.split(","):
            rows, seq = int(v[1:].rstrip("s")), v.endswith("s")
            for st in a.stages.split(","):
                k = base
                if k.pre == "args" and (st != "a1" or kind == "delta_ple"):
                    k = replace(k, pre="none")
                sp = n1_spec_from_str(st, kind, rows, seq, k, a.n1)
                if n1_supported(sp) is None:
                    specs.append(sp)
    print(json.dumps(write_n1_functions(a.out, specs, flavor=a.flavor), indent=1))
    return 0


if __name__ == "__main__":
    import importlib
    import sys

    sys.exit(importlib.import_module(__spec__.name).main())
