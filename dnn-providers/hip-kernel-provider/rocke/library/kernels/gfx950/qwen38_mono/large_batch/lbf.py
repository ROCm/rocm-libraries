# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1: the large-batch (LB) framework kernel on W5's ``persist/`` platform (DESIGN-LB 4.1, 5, 6 E13, 15.6).

One persistent launch (256 CTAs x 512 threads, one CTA per CU) runs ``nl`` layers ``l0 .. l0 + nl - 1`` of a stage list
in a runtime layer loop (``walk="loop"``: one launch per step), or the host launches it once per layer (``nl = 1``,
``walk="launch"``: the launch boundary is the first stage's edge). The binary is the same; the payload bits are the
same in both walks (tags are per layer, :func:`.lbf_plan.layer_tag`).

Per layer, per stage ``i`` (static code per stage, so every stage can have its own emitter, :class:`StageSpec`), every
CTA:

1. **Grid edge** (stage flag bit 0, and not the launch's first stage): every wave drains its stores, CTA barrier, one
   agent-scope arrival (``persist.sync.GridSync``: 8 stripes per XCD, cumulative counters, abort on timeout, the
   launch-fatal word); the wait comes after the first unit's prefetch (FlyTape's prefetch before the sync). A CTA without
   a unit in the stage still arrives and waits (cumulative counters must never run an epoch ahead).
2. **Units** ``u = bid, bid + 256, ...`` of the stage (FlyTape's worker order). Per unit: the emitter's ``prefetch``
   (weights before the wait), then the **polled edge** (stage flag bit 2): the flags of the unit's read set
   (``sync="tile"``) or of every unit of the previous stage (``"a2a"``), all 512 lanes, ``sc1`` loads, a 100 MHz
   deadline (``poll_timeout`` is launch-fatal), producers that are masked skipped; then the emitter's ``body``; then the
   **publish**: ``drain`` (every wave ``vmcnt(0)``, CTA barrier, one flag) or ``wave`` (every wave drains and stores its
   own flag, 8 per unit), ``lazy``: only when the next stage polls (stage flag bit 3).
3. **Masking**: a unit whose first token tile is past the live tiles (runtime ``ntiles``) does no work and publishes
   nothing; its consumers skip it (DESIGN-LB 4.1).

Sync arms (compile time ``sync``; the per-stage plan is host data, :func:`.lbf_plan.tables`): ``mix`` (the measured
default, P-L2 5.1: grid on the stages in the plan's gmask, per-wave lazy flags elsewhere), ``grid``, ``tile``, ``a2a``,
``none`` (the timing floor; racy). Publish ``counted`` is NOT ported (it lost in every P-L2 cell; it stays a knob of
:mod:`.sync_chain`).

Tags and replays: ``ctl`` words ``[E, done, K, B]`` = step epoch, cumulative CTA-done counter, launch count, grid
barriers completed. Layer ``l`` publishes ``tag = E * ltot + l + 1``. The last CTA of a launch zeroes the ticket,
adds the launch's barriers to ``B``, advances ``K``, and advances ``E`` when the launch ends the step
(``l0 + nl == ltot``). Graph replays need no host reset.

Slots: payload / flags of layer ``l`` live in slot ``l mod slots`` (``slots = ltot``: one per layer; ``2``: layer
parity, legal when :func:`.lbf_plan.validate` passes).

Mutants (must fail): ``early`` (grid: release one arrival early; flags: skip the last entry of the wait set), ``tag``
(stale-read twin: poll for the slot's previous occupant ``tag - slots``: a stale payload or a poll timeout), ``noacq``
(coherence twin, informational: plain payload loads with the lines pre-touched before the wait).

Kernel arguments: i64 ``wts, stab, utab, deps, dly, pay, flg, gs, ctl, err, info, sink`` then i32 ``nunits, gpl, dl,
maxit, blk_layer, gridn, l0, nl, ltot, ntiles, slots``. Python engine only (``buffer_*_cp``, ``int_to_ptr``); names
``q38lbf_*``; nothing served imports it. ``hz`` (default on) pads every inline-asm VMEM with an SGPR input
(:mod:`.lbf_hazard`), and :func:`.lbf_hazard.compile_gated` scans the ISA as a compile gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields

from rocke.core.ir import CPOL_NT, CPOL_SC1, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers import device_ops as D

from ..ccl import wire as W
from ..persist import sync as PS
from ..persist.knobs import D0_DEFAULTS, PersistKnobs
from .lbf_plan import GRID, SCOLS, TT_SHIFT, UCOLS
from .sync_plan import BLOCK

__all__ = ["LBKnobs", "StageSpec", "StreamEmitter", "Fx", "build_lb_kernel", "kernel_name", "lb_params", "SYNCS",
           "PUBS", "MUTANTS", "THREADS", "PAY_WORDS"]  # fmt: skip

THREADS = 512
WAVES = THREADS // 64
SYNCS = ("grid", "tile", "a2a", "mix", "none")
PUBS = ("drain", "wave")
MUTANTS = ("none", "early", "tag", "noacq")
PAY_WORDS = 512
RD_UNROLL = 8
SLOTS = 8  # LDS word index of the first checksum slot (pad_lds words 0 / 1: abort, timeout; 2: arrive counter)
MAGIC = 0x5EED1234
TTMASK = (1 << TT_SHIFT) - 1
_I32 = ("nunits", "gpl", "dl", "maxit", "blk_layer", "gridn", "l0", "nl", "ltot", "ntiles", "slots")
_I64 = ("wts", "stab", "utab", "deps", "dly", "pay", "flg", "gs", "ctl", "err", "info", "sink")
# PersistKnobs fields the LB framework applies; every other field must keep its default (refused otherwise)
_PERSIST_APPLIED = ("machine_licm", "arrivals", "stripes", "ticket", "fatal_word")


@dataclass(frozen=True)
class LBKnobs:
    sync: str = "mix"
    pub: str = "wave"
    lazy: bool = True
    pause: str = "s_nop"
    depth: int = 8  # stream emitter: 16-byte loads a lane keeps in flight
    ticket: bool = True
    mutant: str = "none"
    hz: bool = True  # pad inline-asm VMEM with an SGPR input (the gfx9 VALU-SGPR -> VMEM hazard, alola 176 / 182 / 191)
    persist: PersistKnobs | None = None  # None = persist.knobs.D0_DEFAULTS
    stagecode: str = "static"  # static: code per stage | loop: one body in a runtime stage loop (one shared emitter)
    tag: str = ""  # free suffix: a separately compiled copy (the recompiled control)

    @property
    def pk(self) -> PersistKnobs:
        return self.persist or D0_DEFAULTS

    def check(self) -> None:
        if self.sync not in SYNCS:
            raise ValueError(f"sync {self.sync!r}")
        if self.pub not in PUBS:
            raise ValueError(f"pub {self.pub!r} (counted is not ported: it stays a sync_chain knob)")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant {self.mutant!r}")
        if self.mutant != "none" and self.sync == "none":
            raise ValueError("mutants need a sync model")
        if self.mutant == "tag" and not self.flags:
            raise ValueError("mutant tag needs flags (tile / a2a / mix)")
        if self.lazy and self.sync != "mix":
            raise ValueError("lazy publish is a mix knob")
        if not 1 <= self.depth <= 16:
            raise ValueError("depth 1..16")
        if self.pause not in ("s_nop", "s_sleep1", "none"):
            raise ValueError(self.pause)
        if self.stagecode not in ("static", "loop"):
            raise ValueError(f"stagecode {self.stagecode!r}")
        if self.stagecode == "loop" and self.sync == "a2a":
            # the a2a wait set reads the previous stage's row through a runtime modulo that no gate exercises
            # (WP-L1 hardware mutation run): refused rather than shipped untested
            raise NotImplementedError("stagecode='loop' with sync='a2a' is not built (no gate covers it)")
        pk = self.pk
        pk.check()
        d = PersistKnobs()
        bad = [f.name for f in fields(pk) if f.name not in _PERSIST_APPLIED and getattr(pk, f.name) != getattr(d, f.name)]
        if bad:
            raise NotImplementedError(f"PersistKnobs fields not applied by the LB framework: {bad} (refused, not ignored)")
        if pk.arrivals == "xcd":
            raise NotImplementedError("arrivals='xcd' is not built in persist.sync.GridSync")
        if pk.ticket == "one":
            raise NotImplementedError("ticket='one' (FlyTape's single START word) is not built; use 'xcd' or 'off'")
        if not pk.fatal_word:
            raise NotImplementedError("the LB framework always sets the launch-fatal word (D0.8 amendment)")
        if (pk.ticket == "off") == self.ticket:
            raise ValueError("LBKnobs.ticket and PersistKnobs.ticket disagree")

    @property
    def flags(self) -> bool:
        return self.sync in ("tile", "a2a", "mix")

    @property
    def gridsync(self) -> bool:
        return self.sync in ("grid", "mix")

    @property
    def pf(self) -> int:
        return WAVES if self.flags and self.pub == "wave" else 1


def kernel_name(k: LBKnobs, nstages: int, emit: str = "st") -> str:
    d = LBKnobs()
    parts = ["q38lbf", emit, f"s{nstages}", k.sync]
    if k.flags:
        parts.append(k.pub)
    for f in fields(k):
        if f.name in ("sync", "pub", "tag", "persist"):
            continue
        v, dv = getattr(k, f.name), getattr(d, f.name)
        if v != dv:
            parts.append(f"{f.name}{int(v) if isinstance(v, bool) else v}")
    ps = k.pk.suffix()
    if ps:
        parts.append(ps)
    if k.tag:
        parts.append(k.tag)
    return "_".join(parts)


def lb_params() -> list[str]:
    return list(_I64) + list(_I32)


def _c(b: IRBuilder, v: int) -> Value:
    v = int(v) & 0xFFFFFFFF
    return b.const_i32(v - (1 << 32) if v >= 1 << 31 else v)


def _ptr(b: IRBuilder, addr: Value) -> Value:
    return b.int_to_ptr(addr, PtrType(I32, "global"))


def _lds_add(b: IRBuilder, addr: Value, v: Value) -> None:
    b.inline_asm("ds_add_u32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, v], None)


class _TicketSpec:
    def __init__(self, pause: str):
        self.pause = pause


# ----------------------------------------------------------------------------------------------- emitter interface


@dataclass
class Task:
    """Uniform fields of one unit, as the framework read them (``t.i``: the stage index, an int for static stage code
    or a uniform i32 in a runtime stage loop; ``t.nm``: its name suffix for loop / value names)."""

    i: object
    lay: Value
    src: Value  # the producer layer of this stage's edge (lay - 1 for stage 0)
    tag: Value  # this layer's tag
    stag: Value  # the producer's tag
    u: Value  # layer-relative global unit id
    blk0: Value
    nblk: Value
    rw: Value
    doff: Value
    dcnt: Value
    tt: Value
    live: Value  # i1: not masked
    reads: Value  # i1: the unit reads its read set (live and not layer 0 / stage 0)
    pu0: Value
    pn: Value
    j: Value  # the CTA's running read-set counter (LDS checksum slot rotation)
    dly: Value  # injected skew (s_sleep 1 units), read with the other table words
    nm: str = ""


@dataclass
class StageSpec:
    """One stage of the static stage list: its name (for the record) and its emitter."""

    name: str
    emitter: object


class Fx:
    """What an emitter may use: the builder, the kernel arguments, ids, and the edge-protocol helpers."""

    def __init__(self, b: IRBuilder, k: LBKnobs, a: dict, bid: Value, tid: Value, wave: Value, lane: Value, lds0: Value):
        self.b, self.k, self.a, self.bid, self.tid, self.wave, self.lane, self.lds0 = b, k, a, bid, tid, wave, lane, lds0

    def ld(self, base: Value, idx: Value) -> Value:
        b = self.b
        return b.readfirstlane(b.global_load_i32(_ptr(b, base), idx))

    def slot_of(self, layer: Value) -> Value:
        return self.b.mod(layer, self.a["slots"])

    def alive(self, ttp1: Value) -> Value:
        """Producer not masked: its ``tt + 1`` (deps word high bits, 0 = not token-tiled) <= ``ntiles``."""
        return self.b.cmp_le(ttp1, self.a["ntiles"])

    # -------------------------------------------------------------------- read set (the edge's load protocol)
    def read_set(self, t: Task, *, pretouch: bool = False, sfx: str = "") -> Value:
        """u32 sum of 64 words of every LIVE producer of the read set (words ``(d & 7) * 64 + i``, the ones producer
        wave ``d & 7`` stored), ``sc1`` loads (P5 ``bulk=sc1``; ``noacq``: plain); 16 lanes x dwordx4 per producer, 8
        loads in flight a lane; uniform result through the 3-slot LDS rotation. ``pretouch``: the same loads plain,
        folded into a throw-away value (the ``noacq`` twin)."""
        b, k = self.b, self.k
        src_slot = self.slot_of(t.src)
        cnt = b.select(t.reads, t.dcnt, _c(b, 0))
        sub = b.land(self.lane, _c(b, 15))
        slot = b.add(b.lshr(self.lane, _c(b, 4)), b.mul(self.wave, _c(b, 4)))
        per = 32 * RD_UNROLL
        ng = b.div(b.add(cnt, _c(b, per - 1)), _c(b, per))
        pay_rs = b.buffer_rsrc(_ptr(b, self.a["pay"]), _c(b, 0x7FFFFFF0))
        aux = 0 if (pretouch or k.mutant == "noacq") else CPOL_SC1
        nm = f"lbrs{t.nm}{sfx}{'p' if pretouch else ''}"
        loop = b.scf_for_iter(_c(b, 0), ng, _c(b, 1), [(nm + "a", _c(b, 0))], iv_name=nm + "g")
        with loop as (g, vals):
            acc = vals[0]
            lds = []
            # every deps word of the group first (8 loads in flight; interleaving them with the payload loads, which
            # depend on them through the masking test, serializes the group: WP-L1 WaveScope / ISA, one vmcnt(0) each)
            kks = [b.add(b.mul(b.add(b.mul(g, _c(b, RD_UNROLL)), _c(b, q)), _c(b, 32)), slot) for q in range(RD_UNROLL)]
            oks = [b.cmp_lt(kk, cnt) for kk in kks]
            ws = [b.global_load_i32(_ptr(b, self.a["deps"]), b.add(t.doff, b.select(ok, kk, _c(b, 0))))
                  for kk, ok in zip(kks, oks)]
            for q in range(RD_UNROLL):
                ok, w = oks[q], ws[q]
                d = b.smin(b.land(w, _c(b, TTMASK)), b.sub(self.a["nunits"], _c(b, 1)))  # clamp: provable index width
                ok = b.land(ok, self.alive(b.lshr(w, _c(b, TT_SHIFT))))
                word = b.add(b.mul(b.land(d, _c(b, 7)), _c(b, 64)), b.mul(sub, _c(b, 4)))
                off = b.mul(b.add(b.mul(b.add(b.mul(src_slot, self.a["nunits"]), d), _c(b, PAY_WORDS)), word), _c(b, 4))
                lds.append(b.buffer_load_cp(pay_rs, b.select(ok, off, _c(b, 0x7FFFFFF0)), _c(b, 0), dwords=4, aux=aux))
            for v in lds:
                for e in range(4):
                    acc = b.add(acc, b.vec_extract(v, e))
            b.scf_yield(acc)
        s = loop.results[0]
        if pretouch:
            return s
        tot = b.wave_reduce(s, "add")
        sl = b.mod(t.j, _c(b, 3))
        with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
            _lds_add(b, b.add(self.lds0, b.mul(b.add(sl, _c(b, SLOTS)), _c(b, 4))), tot)
        b.s_barrier_bare(fence="compiler")
        chk = PS.lds_read(b, b.add(self.lds0, b.mul(b.add(sl, _c(b, SLOTS)), _c(b, 4))))
        with b.scf_if(D.is_thread0(b)):
            nx = b.mod(b.add(t.j, _c(b, 2)), _c(b, 3))
            PS.lds_write(b, b.add(self.lds0, b.mul(b.add(nx, _c(b, SLOTS)), _c(b, 4))), _c(b, 0))
        return chk

    # -------------------------------------------------------------------- payload (the edge's store protocol)
    def payload_word(self, t: Task, chk: Value) -> Value:
        b = self.b
        h = b.add(b.add(b.mul(t.tag, _c(b, 0x01000193)), b.mul(t.lay, _c(b, 0x9E3779B1))),
                  b.add(b.mul(_c(b, t.i) if isinstance(t.i, int) else t.i, _c(b, 0x85EBCA77)),
                        b.mul(t.u, _c(b, 0xC2B2AE3D))))
        return b.add(b.add(h, b.mul(chk, _c(b, 3))), b.mul(self.tid, _c(b, 0x9E37)))

    def store_payload(self, t: Task, val: Value) -> None:
        """Word ``tid`` of the unit's 2 KiB payload, ``sc1`` write-through (P5)."""
        b = self.b
        slot = self.slot_of(t.lay)
        poff = b.mul(b.add(b.mul(b.add(b.mul(slot, self.a["nunits"]), t.u), _c(b, PAY_WORDS)), self.tid), _c(b, 4))
        PS.st_sc1(b, self.a["pay"], poff, val)


class StreamEmitter:
    """P-L2's synthetic stage body (``sync_chain``'s task body): read the read set, stream ``nblk`` 8 KiB weight blocks
    (``rw`` units write every block back, NT: the GDN state), inject the per-unit delay, store the payload. ``prefetch``
    issues the first ``depth`` blocks (weights before the wait)."""

    name = "st"

    def __init__(self, depth: int = 8):
        self.depth = depth

    def _wrs(self, fx: Fx, t: Task):
        b, A = fx.b, fx.a
        wbase = b.add(A["wts"], b.mul(b.zext(b.add(b.mul(t.lay, A["blk_layer"]), t.blk0), I64), b.const_i64(BLOCK)))
        return (b.buffer_rsrc(_ptr(b, b.readfirstlane(wbase)), b.mul(t.nblk, _c(b, BLOCK))), b.mul(fx.tid, _c(b, 16)))

    def prefetch(self, fx: Fx, t: Task) -> list[Value]:
        b = fx.b
        w_rs, lane_off = self._wrs(fx, t)
        return [b.buffer_load_cp(w_rs, b.add(lane_off, _c(b, q * BLOCK)), _c(b, 0), dwords=4) for q in range(self.depth)]

    def body(self, fx: Fx, t: Task, ring: list[Value], acc: Value) -> Value:
        b, D_ = fx.b, self.depth
        zero = _c(b, 0)
        chk = fx.read_set(t)
        w_rs, lane_off = self._wrs(fx, t)
        ngr = b.div(b.add(t.nblk, _c(b, D_ - 1)), _c(b, D_))
        isrw = b.cmp_ne(t.rw, zero)
        ngm1 = b.smax(b.sub(ngr, _c(b, 1)), zero)
        sl = b.scf_for_iter(zero, ngm1, _c(b, 1), [(f"lbr{t.nm}_{q}", v) for q, v in enumerate(ring)] + [(f"lbsa{t.nm}", acc)],
                            iv_name=f"lbbk{t.nm}")
        with sl as (g, sv):
            sv = list(sv)
            rg, a2 = sv[:-1], sv[-1]
            nxt = []
            for q in range(D_):
                v = rg[q]
                a2 = b.add(b.mul(a2, _c(b, 33)), b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
                cur = b.add(lane_off, b.mul(b.add(b.mul(g, _c(b, D_)), _c(b, q)), _c(b, BLOCK)))
                b.buffer_store_cp(v, w_rs, b.select(isrw, cur, _c(b, 0x7FFFFFF0)), zero, aux=CPOL_NT)
                b.sched_barrier(0)
                nxt.append(b.buffer_load_cp(w_rs, b.add(cur, _c(b, D_ * BLOCK)), zero, dwords=4))
            b.scf_yield(*nxt, a2)
        res = list(sl.results)
        acc = res[-1]
        for q in range(D_):
            v = res[q]
            acc = b.add(b.mul(acc, _c(b, 33)), b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
            cur = b.add(lane_off, b.mul(b.add(b.mul(ngm1, _c(b, D_)), _c(b, q)), _c(b, BLOCK)))
            b.buffer_store_cp(v, w_rs, b.select(isrw, cur, _c(b, 0x7FFFFFF0)), zero, aux=CPOL_NT)
        with b.scf_if(t.live):
            with b.scf_for(zero, t.dly, _c(b, 1), iv_name=f"lbdl{t.nm}"):
                D.s_sleep(b, 1)
            fx.store_payload(t, fx.payload_word(t, chk))
        return acc


# ----------------------------------------------------------------------------------------------- the kernel


class _LB:
    def __init__(self, k: LBKnobs, specs: Sequence[StageSpec], name: str):
        k.check()
        if not specs:
            raise ValueError("empty stage list")
        if k.stagecode == "loop" and any(s.emitter is not specs[0].emitter for s in specs):
            raise ValueError("stagecode='loop' needs one emitter shared by every stage (runtime stage loop)")
        self.k, self.specs = k, list(specs)
        b = self.b = IRBuilder(name)
        b.kernel.attrs["max_workgroup_size"] = THREADS
        if k.pk.machine_licm == "off":
            b.kernel.attrs["machine_licm"] = "off"
        self.a = {}
        for n in lb_params():
            self.a[n] = b.param(n, I64 if n in _I64 else I32)
        for n in lb_params():
            self.a[n] = b.readfirstlane(self.a[n])
        self.bid = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.wave = D.wave_id(b)
        self.lane = b.lane_id()
        pk = k.pk
        self.gs = PS.GridSync(grid=GRID, stripes=pk.stripes, pause="s_nop",
                              arrivals={"cta": "cta", "wave": "wave", "last": "last"}.get(pk.arrivals, "cta"))

    def timeout(self, cond: Value, reason: str) -> None:
        b = self.b
        with b.scf_if(cond):
            D.atomic_add_scoped(b, self.a["err"], _c(b, 4 * PS.ERR_TIMEOUT), _c(b, 1))
            PS.raise_fatal(b, self.a["err"], reason)

    def poll(self, t: Task) -> None:
        """Every lane polls one flag per round of 512 wait-set entries (entry = producer x flag); masked producers are
        satisfied at once."""
        b, k, A = self.b, self.k, self.a
        PF = k.pf
        cnt = t.pn if k.sync == "a2a" else t.dcnt
        if k.mutant == "early":
            cnt = b.sub(cnt, _c(b, 1))
        ent = b.mul(cnt, _c(b, PF))
        want = b.sub(t.stag, A["slots"]) if k.mutant == "tag" else t.stag
        src_slot = self.fx.slot_of(t.src)
        abort_a = b.add(A["err"], b.const_i64(12))
        with b.scf_for(_c(b, 0), ent, _c(b, THREADS), iv_name=f"lbpc{t.nm}") as c0:
            e = b.add(c0, self.tid)
            act = b.cmp_lt(e, ent)
            kk = b.lshr(e, _c(b, 3)) if PF == 8 else e
            sub = b.land(e, _c(b, 7)) if PF == 8 else _c(b, 0)
            if k.sync != "a2a":
                w = b.global_load_i32(_ptr(b, A["deps"]), b.add(t.doff, b.select(act, kk, _c(b, 0))))
                d = b.smin(b.land(w, _c(b, TTMASK)), b.sub(A["nunits"], _c(b, 1)))
                ttp1 = b.lshr(w, _c(b, TT_SHIFT))
            else:
                d = b.add(t.pu0, kk)
                ttp1 = b.add(b.global_load_i32(_ptr(b, A["utab"]), b.add(b.mul(b.select(act, d, _c(b, 0)), _c(b, UCOLS)),
                                                                      _c(b, 5))), _c(b, 1))
            act = b.land(act, self.fx.alive(ttp1))
            with b.scf_if(act):
                woff = b.add(b.mul(b.add(b.mul(src_slot, A["nunits"]), d), _c(b, PF)), sub)
                addr = b.add(A["flg"], b.zext(b.mul(woff, _c(b, 4)), I64))
                dl = b.add(D.memrealtime(b), b.zext(A["dl"], I64))
                _seen, _mask, st = W.poll_flags(b, addr, want, dl, abort_a, _c(b, 0), pause=k.pause, poll_cpol="sc1")
                self.timeout(b.cmp_ne(st, _c(b, 0)), "poll_timeout")

    def publish(self, t: Task, pubc: Value) -> None:
        b, k, A = self.b, self.k, self.a
        zero = _c(b, 0)
        fbase = b.mul(b.add(b.mul(self.fx.slot_of(t.lay), A["nunits"]), t.u), _c(b, k.pf))
        with b.scf_if(pubc):
            b.s_waitcnt(vmcnt=0)
            if k.pub == "drain":
                b.s_barrier_bare(fence="compiler")
                with b.scf_if(D.is_thread0(b)):
                    PS.st_sc1(b, A["flg"], b.mul(fbase, _c(b, 4)), t.tag)
            else:
                with b.scf_if(b.cmp_eq(self.lane, zero)):
                    PS.st_sc1(b, A["flg"], b.mul(b.add(fbase, self.wave), _c(b, 4)), t.tag)

    def grid_wait(self, sidx: Value) -> None:
        b, A, k = self.b, self.a, self.k
        live = b.cmp_eq(PS.lds_read(b, self.lds0), _c(b, 0))
        epoch1 = b.add(self.B, sidx)
        with b.scf_if(live):
            PS.grid_sync_wait(b, self.gs, A["gs"], epoch1, self.lds0, A["err"], maxit=A["maxit"],
                              bias=1 if k.mutant == "early" and k.sync == "grid" else 0)
        b.s_barrier_bare(fence="compiler")

    def stage(self, i, spec: StageSpec, li: Value, lay: Value, tag: Value, acc: Value, j: Value):
        """Stage ``i`` (an int: static stage code; a uniform i32: the body of the runtime stage loop)."""
        b, k, A = self.b, self.k, self.a
        zero = _c(b, 0)
        S = len(self.specs)
        rt = not isinstance(i, int)
        nm = "r" if rt else str(i)
        if rt:
            srow = b.mul(i, _c(b, SCOLS))
            prow = b.mul(b.mod(b.add(i, _c(b, S - 1)), _c(b, S)), _c(b, SCOLS))
        else:
            srow = _c(b, i * SCOLS)
        u0 = self.fx.ld(A["stab"], srow)
        units = self.fx.ld(A["stab"], b.add(srow, _c(b, 1)))
        fl = self.fx.ld(A["stab"], b.add(srow, _c(b, 2)))
        pref = self.fx.ld(A["stab"], b.add(srow, _c(b, 3)))
        if rt:
            pu0 = self.fx.ld(A["stab"], prow)
            pn = self.fx.ld(A["stab"], b.add(prow, _c(b, 1)))
        else:
            pu0 = self.fx.ld(A["stab"], _c(b, ((i - 1) % S) * SCOLS))
            pn = self.fx.ld(A["stab"], _c(b, ((i - 1) % S) * SCOLS + 1))
        yes = b.cmp_eq(zero, zero)
        if rt:
            nz = b.cmp_ne(i, zero)
            notfirst = b.lor(b.cmp_ne(li, zero), nz)
            reads = b.lor(b.cmp_ne(lay, zero), nz)
            one0 = b.select(nz, zero, _c(b, 1))
        else:
            notfirst = b.cmp_ne(li, zero) if i == 0 else yes  # the launch's first stage waits for nothing
            reads = b.cmp_ne(lay, zero) if i == 0 else yes  # layer 0's first stage reads nothing
        isgrid = b.land(notfirst, b.cmp_ne(b.land(fl, _c(b, 1)), zero)) if k.gridsync else None
        if k.sync == "mix":
            ispoll = b.land(notfirst, b.cmp_ne(b.land(fl, _c(b, 4)), zero))
        else:
            ispoll = notfirst
        pubb = b.cmp_ne(b.land(fl, _c(b, 8)), zero) if k.lazy else yes
        sidx = b.sub(b.add(b.add(b.mul(li, A["gpl"]), pref), _c(b, 1)), self.s0g)
        src = b.sub(lay, one0 if rt else _c(b, 1 if i == 0 else 0))
        stag = b.sub(tag, one0 if rt else _c(b, 1 if i == 0 else 0))
        if k.gridsync:
            with b.scf_if(isgrid):
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare(fence="compiler")
                live = b.cmp_eq(PS.lds_read(b, self.lds0), zero)
                PS.grid_sync_arrive(b, self.gs, A["gs"], self.bid, live=live, lds0=self.lds0)
        mine = b.select(b.cmp_gt(units, self.bid), b.div(b.add(b.sub(units, self.bid), _c(b, GRID - 1)), _c(b, GRID)), zero)
        loop = b.scf_for_iter(zero, mine, _c(b, 1), [(f"lbacc{nm}", acc), (f"lbj{nm}", j)], iv_name=f"lbu{nm}")
        with loop as (kq, vals):
            acc_, j_ = vals
            u = b.add(u0, b.add(self.bid, b.mul(kq, _c(b, GRID))))
            ub = b.mul(u, _c(b, UCOLS))
            tt = self.fx.ld(A["utab"], b.add(ub, _c(b, 5)))
            live = b.cmp_lt(tt, A["ntiles"])  # signed: tt = -1 (not token-tiled) is always live
            t = Task(i=i, lay=lay, src=src, tag=tag, stag=stag, u=u,
                     blk0=self.fx.ld(A["utab"], ub),
                     nblk=b.select(live, self.fx.ld(A["utab"], b.add(ub, _c(b, 1))), zero),
                     rw=self.fx.ld(A["utab"], b.add(ub, _c(b, 2))),
                     doff=self.fx.ld(A["utab"], b.add(ub, _c(b, 3))),
                     dcnt=self.fx.ld(A["utab"], b.add(ub, _c(b, 4))),
                     tt=tt, live=live, reads=b.land(live, reads), pu0=pu0, pn=pn, j=j_,
                     dly=self.fx.ld(A["dly"], b.add(b.mul(lay, A["nunits"]), u)), nm=nm)
            ring = spec.emitter.prefetch(self.fx, t)
            pre = self.fx.read_set(t, pretouch=True) if k.mutant == "noacq" else None
            if k.gridsync:
                with b.scf_if(b.land(isgrid, b.cmp_eq(kq, zero))):
                    self.grid_wait(sidx)
            if k.flags:
                with b.scf_if(b.land(ispoll, live)):
                    self.poll(t)
                    b.s_barrier_bare(fence="compiler")
            acc_ = spec.emitter.body(self.fx, t, ring, acc_)
            if pre is not None:
                acc_ = b.xor(acc_, pre)
            if k.flags:
                self.publish(t, b.land(live, pubb))
            b.scf_yield(acc_, b.add(j_, _c(b, 1)))
        acc, j = loop.results
        if k.gridsync:
            with b.scf_if(b.land(isgrid, b.cmp_eq(mine, zero))):
                self.grid_wait(sidx)
        return acc, j

    def build(self) -> KernelDef:
        b, k, A = self.b, self.k, self.a
        zero = _c(b, 0)
        self.lds0 = PS.pad_lds(b, name_hint="lbfpad", arrive_word=k.pk.arrivals == "last")
        self.fx = Fx(b, k, A, self.bid, self.tid, self.wave, self.lane, self.lds0)
        with b.scf_if(D.is_thread0(b)):
            for s in range(3):
                PS.lds_write(b, b.add(self.lds0, _c(b, 4 * (SLOTS + s))), zero)
        b.s_barrier_bare(fence="compiler")
        E = self.fx.ld(A["ctl"], zero)
        K = self.fx.ld(A["ctl"], _c(b, 2))
        self.B = self.fx.ld(A["ctl"], _c(b, 3))
        tk_a = b.add(A["ctl"], b.const_i64(1024))
        if k.ticket:
            row = b.mul(self.bid, _c(b, 8))
            info = b.int_to_ptr(A["info"], PtrType(I64, "global"))

            def st(col, v):
                b.store(v if v.type == I64 else b.zext(v, I64), info, b.add(row, _c(b, col)))

            PS.ticket_xcd(b, _TicketSpec("s_nop"), tk_a, row, st, A["gridn"], A["maxit"], _c(b, 1), fatal_err=A["err"])
            b.s_barrier_bare(fence="compiler")
        self.s0g = b.land(self.fx.ld(A["stab"], _c(b, 2)), _c(b, 1)) if k.gridsync else zero
        lp = b.scf_for_iter(zero, A["nl"], _c(b, 1), [("lbacc", zero), ("lbj", zero)], iv_name="lbli")
        with lp as (li, vals):
            acc, j = vals
            lay = b.add(A["l0"], li)
            tag = b.add(b.add(b.mul(E, A["ltot"]), lay), _c(b, 1))
            if k.stagecode == "loop":
                sl = b.scf_for_iter(zero, _c(b, len(self.specs)), _c(b, 1), [("lbsacc", acc), ("lbsj", j)], iv_name="lbsi")
                with sl as (si, sv):
                    acc2, j2 = self.stage(si, self.specs[0], li, lay, tag, sv[0], sv[1])
                    b.scf_yield(acc2, j2)
                acc, j = sl.results
            else:
                for i, spec in enumerate(self.specs):
                    acc, j = self.stage(i, spec, li, lay, tag, acc, j)
            b.scf_yield(acc, j)
        acc = lp.results[0]
        with b.scf_if(b.cmp_eq(acc, _c(b, MAGIC))):
            b.store(acc, _ptr(b, A["sink"]), self.tid)
        # launch end: the last CTA zeroes the ticket, advances B by this launch's barriers, K, and E at the step end
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, A["ctl"], _c(b, 4), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(b.add(old, _c(b, 1)), b.mul(b.add(K, _c(b, 1)), _c(b, GRID)))):
                for w in range(9):
                    PS.st_sc1(b, tk_a, _c(b, w * PS.LINE), zero)
                nsync = b.sub(b.mul(A["nl"], A["gpl"]), self.s0g) if k.gridsync else zero
                PS.st_sc1(b, A["ctl"], _c(b, 12), b.add(self.B, nsync))
                PS.st_sc1(b, A["ctl"], _c(b, 8), b.add(K, _c(b, 1)))
                with b.scf_if(b.cmp_eq(b.add(A["l0"], A["nl"]), A["ltot"])):
                    PS.st_sc1(b, A["ctl"], zero, b.add(E, _c(b, 1)))
        b.ret()
        return b.kernel


def build_lb_kernel(k: LBKnobs, specs: Sequence[StageSpec] | int, *, emit: str = "st") -> KernelDef:
    """The LB framework kernel over ``specs`` (or ``S`` stages of the stream emitter). ``k.hz`` pads every asm VMEM
    with an SGPR input (:func:`.lbf_hazard.pad_asm_vmem`)."""
    if isinstance(specs, int):
        em = StreamEmitter(k.depth)
        specs = [StageSpec(f"s{i}", em) for i in range(specs)]
    kd = _LB(k, specs, kernel_name(k, len(specs), emit)).build()
    if k.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd
