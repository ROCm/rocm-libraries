# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L2: the sync-model chain of large-batch decoder layers in one persistent launch (DESIGN-LB 5, 12), gfx950.

One launch (256 CTAs x 512 threads, one CTA per CU) walks ``L`` layers of a stage graph built by :mod:`.sync_plan`
(FlyTape's phase list ``ft`` or DESIGN-LB's ``lb``; host tables, so one binary serves every plan, rows bucket and TP
shape). A CTA runs its task list in (layer, stage) order; a task is one unit of a stage:

1. **wait** for its inputs by the arm's sync model (compile-time ``sync``):

   - ``grid``: before the first task of every stage, P4's striped counter barrier (8 counter words per XCD, one
     agent-scope arrival per CTA after every wave's stores, wave 0 polls the 64 words lane-parallel, cumulative
     counters with abort-on-timeout and the launch-fatal word: ``persist.sync.GridSync``); the unit's first weight
     loads are issued between its arrival and its wait (FlyTape's prefetch before the sync). A CTA with no unit in a
     stage runs a dummy task so every boundary sees every CTA.
   - ``tile``: the flags of its read set (per-token-tile / per-head / per-expert partial edges), all 512 lanes
     polling one flag each per round (``ccl.wire.poll_flags``: ``sc1`` loads, a 2 s ``s_memrealtime`` deadline,
     timeout -> error count + launch-fatal bit); weights issued before the poll.
   - ``a2a``: the flags of EVERY unit of the previous stage (the mailbox all-to-all), same poll.
   - ``mix``: per stage, the grid barrier or the ``tile`` poll (task flag bit 0 / bit 2 from the host plan:
     ``sync_plan.task_table(gmask=...)``): the per-edge decision of DESIGN-LB 5; producers publish flags (``lazy``:
     only units whose consumer stage polls them, task flag bit 3; the next grid barrier covers the rest).
   - ``none``: nothing (timing floor: no dependency is honoured, numerics racy by construction).

2. **read** its read set (the same in every arm): 64 words of every producer ``d`` (the 64 words producer wave
   ``d & 7`` stored), ``sc1`` loads (P5's ``bulk=sc1`` protocol), 16 lanes x dwordx4 per producer, 8 loads in flight a
   lane; the CTA's checksum ``chk`` = the u32 sum (wave reduce + LDS add, three rotating LDS slots).
3. **stream** its weight bytes: ``nblk`` 8 KiB blocks of the layer's weight region (512 lanes x 16 B, ``depth``
   loads in flight, the ring carried through the loop); ``rw`` units (the GDN state) store every block back (NT).
4. **delay** ``dly[layer][unit]`` x ``s_sleep 1`` (injected skew).
5. **publish**: every thread stores word ``tid`` of the unit's 2 KiB payload, ``H(tag, layer, stage, unit) + 3 chk +
   tid * 0x9E37`` (``sc1``); then (``tile`` / ``a2a``) ``pub``:

   - ``drain`` (``publish=unit_end``): every wave ``vmcnt(0)``, CTA barrier, thread 0 stores the unit's flag = tag;
   - ``wave`` (``publish=comm_wave``): every wave ``vmcnt(0)`` and stores ITS flag (8 flags a unit, no barrier;
     consumers poll all 8);
   - ``counted`` (``publish=counted``): the next task's first weight loads are issued before the publish, which then
     waits ``vmcnt(depth)`` (vector memory operations retire in issue order: the payload stores are done, the next
     unit's loads stay in flight), CTA barrier, flag; the loads are carried into the next task.

Tags and replays: word 0 of ``ctl`` is the launch epoch ``E`` (tag = ``E + 1``, never 0); the last CTA to finish (a
cumulative done counter, ``ctl`` word 1) zeroes the residency ticket's words and advances ``E``, so graph replays
need no host reset. Grid counters are cumulative: sync ``k`` of launch ``E`` waits for ``E * nsync + k``.

Residency: P4's per-XCD-start ticket at launch start (``ticket``), launch-fatal on timeout.

Mutants (must fail): ``early`` (grid: release one arrival early, ``bias = 1``; tile / a2a: skip the last flag of the
wait set), ``tag`` (stale-read twin: poll for the previous launch's tag, so a consumer reads last launch's payload),
``noacq`` (coherence twin: plain payload loads, with the same lines pre-touched before the wait).

Kernel arguments: i64 ``wts, tasks, utab, deps, dly, pay, flg, gs, ctl, err, info, stm`` then i32 ``tmax`` (task
row stride), ``nunits`` (N, units per layer), ``nsync`` (grid syncs per launch), ``dl`` (poll deadline, 100 MHz
ticks), ``maxit`` (grid wait and ticket bound, retries), ``blk_layer`` (8 KiB weight blocks per layer), ``gridn`` (the
launched grid: the ticket's expected count; the oversize-grid control launches more CTAs than CUs). Layouts in
:mod:`.sync_plan` (``unit_table``, ``task_table``); ``pay`` i32 ``[L][N][512]``, ``flg`` i32 ``[L][N][PF]`` (PF = 8 for
``pub="wave"`` else 1), ``err`` i32 ``[mismatch, timeouts, fatal, abort(0)]``, ``info`` i64 ``[256][8]`` (the ticket
row), ``stm`` i64 ``[256][tmax][4]`` (``stamp`` builds: task start, before the wait, after the wait, after publish;
otherwise the sink of the stream checksum).

Python engine only (``buffer_*_cp``, ``int_to_ptr``); names ``q38lbs_*``; nothing served imports it.
Measurements: ``experiments/qwen38-true-mono/large-batch/P-L2.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_NT, CPOL_SC1, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers import device_ops as D

from ..ccl import wire as W
from ..persist import sync as PS
from .sync_plan import BLOCK, GRID, TCOLS, UCOLS

__all__ = ["ChainKnobs", "build_chain_kernel", "chain_params", "kernel_name", "SYNCS", "PUBS", "MUTANTS", "THREADS"]

THREADS = 512
WAVES = THREADS // 64
SYNCS = ("grid", "tile", "a2a", "mix", "none")
PUBS = ("drain", "wave", "counted")
MUTANTS = ("none", "early", "tag", "noacq")
PAY_WORDS = 512  # payload words per unit (one per thread)
RD_UNROLL = 8  # producers' payload loads in flight per lane
SLOTS = 8  # LDS word index of the first checksum slot (pad_lds words 0 / 1 are the abort and timeout words)
MAGIC = 0x5EED1234


@dataclass(frozen=True)
class ChainKnobs:
    sync: str = "tile"
    pub: str = "drain"  # tile / a2a only
    depth: int = 8  # 16-byte loads a lane keeps in flight in the weight stream
    stamp: bool = False
    ticket: bool = True
    mutant: str = "none"
    pause: str = "s_nop"  # flag-poll pause (grid: the barrier's own)
    lazy: bool = False  # mix: publish a unit's flag only when the next stage polls it (task flag bit 3)
    tag: str = ""  # free suffix: a separately compiled copy (the recompiled control)

    def check(self) -> None:
        if self.sync not in SYNCS:
            raise ValueError(f"sync {self.sync!r}")
        if self.pub not in PUBS:
            raise ValueError(f"pub {self.pub!r}")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant {self.mutant!r}")
        if self.mutant != "none" and self.sync == "none":
            raise ValueError("mutants need a sync model")
        if self.mutant == "tag" and not self.flags:
            raise ValueError("mutant tag needs flags (tile / a2a / mix)")
        if self.pub == "counted" and self.sync not in ("tile", "a2a", "mix"):
            raise ValueError("pub counted needs flags")
        if self.lazy and self.sync != "mix":
            raise ValueError("lazy publish is a mix knob")
        if not 1 <= self.depth <= 16:
            raise ValueError("depth 1..16")
        if self.pause not in ("s_nop", "s_sleep1", "none"):
            raise ValueError(self.pause)

    @property
    def pf(self) -> int:
        """Flags per unit."""
        return WAVES if self.flags and self.pub == "wave" else 1

    @property
    def flags(self) -> bool:
        return self.sync in ("tile", "a2a", "mix")

    @property
    def gridsync(self) -> bool:
        return self.sync in ("grid", "mix")


def kernel_name(k: ChainKnobs) -> str:
    d = ChainKnobs()
    parts = ["q38lbs", k.sync]
    if k.flags:
        parts.append(k.pub)
    for f in fields(k):
        if f.name in ("sync", "pub", "tag"):
            continue
        v, dv = getattr(k, f.name), getattr(d, f.name)
        if v != dv:
            parts.append(f"{f.name}{int(v) if isinstance(v, bool) else v}")
    if k.tag:
        parts.append(k.tag)
    return "_".join(parts)


def chain_params() -> list[str]:
    return ["wts", "tasks", "utab", "deps", "dly", "pay", "flg", "gs", "ctl", "err", "info", "stm",
            "tmax", "nunits", "nsync", "dl", "maxit", "blk_layer", "gridn"]  # fmt: skip


def _c(b: IRBuilder, v: int) -> Value:
    """i32 constant; u32 values >= 2**31 wrap to their two's-complement i32."""
    v = int(v) & 0xFFFFFFFF
    return b.const_i32(v - (1 << 32) if v >= 1 << 31 else v)


def _ptr(b: IRBuilder, addr: Value) -> Value:
    return b.int_to_ptr(addr, PtrType(I32, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _u(b: IRBuilder, v: Value) -> Value:
    return b.readfirstlane(v)


def _lds_add(b: IRBuilder, addr: Value, v: Value) -> None:
    b.inline_asm("ds_add_u32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, v], None)


_I32 = ("tmax", "nunits", "nsync", "dl", "maxit", "blk_layer", "gridn")


class _TicketSpec:
    def __init__(self, pause: str):
        self.pause = pause


class _Chain:
    def __init__(self, k: ChainKnobs):
        k.check()
        self.k = k
        b = self.b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = THREADS
        self.a = {}
        for n in chain_params():
            self.a[n] = b.param(n, I64 if n not in _I32 else I32)
        for n in _I32:
            self.a[n] = b.readfirstlane(self.a[n])
        for n in ("wts", "tasks", "utab", "deps", "dly", "pay", "flg", "gs", "ctl", "err", "info", "stm"):
            self.a[n] = b.readfirstlane(self.a[n])
        self.bid = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.wave = D.wave_id(b)
        self.lane = b.lane_id()
        self.gs = PS.GridSync(grid=GRID, stripes=8, pause="s_nop", arrivals="cta")

    # ---------------------------------------------------------------- small helpers
    def ld(self, base: Value, idx: Value) -> Value:
        """Uniform i32 of a read-only table."""
        b = self.b
        return _u(b, b.global_load_i32(_ptr(b, base), idx))

    def stamp(self, j: Value, col: int) -> None:
        if not self.k.stamp:
            return
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            row = b.mul(b.add(b.mul(self.bid, self.a["tmax"]), j), _c(b, 4))
            b.store(D.memrealtime(b), b.int_to_ptr(self.a["stm"], PtrType(I64, "global")), b.add(row, _c(b, col)))

    def timeout(self, cond: Value, reason: str) -> None:
        b = self.b
        with b.scf_if(cond):
            D.atomic_add_scoped(b, self.a["err"], _c(b, 4 * PS.ERR_TIMEOUT), _c(b, 1))
            PS.raise_fatal(b, self.a["err"], reason)

    # ---------------------------------------------------------------- the flag wait
    def poll(self, lay: Value, stg: Value, doff: Value, dcnt: Value, pu0: Value, pn: Value, tag: Value) -> None:
        """Every lane polls one flag per round of 512 wait-set entries (entry = producer x flag of the producer)."""
        b, k = self.b, self.k
        PF = k.pf
        src = b.sub(lay, b.select(b.cmp_eq(stg, _c(b, 0)), _c(b, 1), _c(b, 0)))
        cnt = pn if k.sync == "a2a" else dcnt
        if k.mutant == "early":
            cnt = b.sub(cnt, _c(b, 1))
        ent = b.mul(cnt, _c(b, PF))
        want = b.sub(tag, _c(b, 1)) if k.mutant == "tag" else tag
        abort_a = b.add(self.a["err"], b.const_i64(12))
        with b.scf_for(_c(b, 0), ent, _c(b, THREADS), iv_name="pl2pc") as c0:
            e = b.add(c0, self.tid)
            act = b.cmp_lt(e, ent)
            with b.scf_if(act):
                kk = b.lshr(e, _c(b, 3)) if PF == 8 else e
                sub = b.land(e, _c(b, 7)) if PF == 8 else _c(b, 0)
                if k.sync != "a2a":
                    d = b.global_load_i32(_ptr(b, self.a["deps"]), b.add(doff, kk))
                else:
                    d = b.add(pu0, kk)
                woff = b.add(b.mul(b.add(b.mul(src, self.a["nunits"]), d), _c(b, PF)), sub)
                addr = b.add(self.a["flg"], b.zext(b.mul(woff, _c(b, 4)), I64))
                dl = b.add(D.memrealtime(b), b.zext(self.a["dl"], I64))
                _seen, _mask, st = W.poll_flags(b, addr, want, dl, abort_a, _c(b, 0), pause=k.pause, poll_cpol="sc1")
                self.timeout(b.cmp_ne(st, _c(b, 0)), "poll_timeout")

    # ---------------------------------------------------------------- the read set
    def read_set(self, lay: Value, stg: Value, doff: Value, dcnt: Value, j: Value, pretouch: bool = False) -> Value:
        """Sum of the read set's payload words (uniform). ``pretouch``: issue the same loads plain and fold them into
        a throw-away value (the ``noacq`` twin), returns that value instead."""
        b, k = self.b, self.k
        src = b.sub(lay, b.select(b.cmp_eq(stg, _c(b, 0)), _c(b, 1), _c(b, 0)))
        sub = b.land(self.lane, _c(b, 15))
        slot = b.add(b.lshr(self.lane, _c(b, 4)), b.mul(self.wave, _c(b, 4)))  # 0..31
        per = 32 * RD_UNROLL
        ng = b.div(b.add(dcnt, _c(b, per - 1)), _c(b, per))
        pay_rs = b.buffer_rsrc(_ptr(b, self.a["pay"]), _c(b, 0x7FFFFFF0))
        aux = 0 if (pretouch or k.mutant == "noacq") else CPOL_SC1
        sfx = "p" if pretouch else ""
        loop = b.scf_for_iter(_c(b, 0), ng, _c(b, 1), [("pl2rs" + sfx, _c(b, 0))], iv_name="pl2rg" + sfx)
        with loop as (g, vals):
            acc = vals[0]
            lds = []
            for i in range(RD_UNROLL):
                kk = b.add(b.mul(b.add(b.mul(g, _c(b, RD_UNROLL)), _c(b, i)), _c(b, 32)), slot)
                ok = b.cmp_lt(kk, dcnt)
                d = b.global_load_i32(_ptr(b, self.a["deps"]), b.add(doff, b.select(ok, kk, _c(b, 0))))
                word = b.add(b.mul(b.land(d, _c(b, 7)), _c(b, 64)), b.mul(sub, _c(b, 4)))
                off = b.mul(b.add(b.mul(b.add(b.mul(src, self.a["nunits"]), d), _c(b, PAY_WORDS)), word), _c(b, 4))
                lds.append(b.buffer_load_cp(pay_rs, b.select(ok, off, _c(b, 0x7FFFFFF0)), _c(b, 0), dwords=4, aux=aux))
            for v in lds:
                for e in range(4):
                    acc = b.add(acc, b.vec_extract(v, e))
            b.scf_yield(acc)
        s = loop.results[0]
        if pretouch:
            return s
        tot = b.wave_reduce(s, "add")
        lds0 = self.lds0
        sl = b.mod(j, _c(b, 3))
        with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
            _lds_add(b, b.add(lds0, b.mul(b.add(sl, _c(b, SLOTS)), _c(b, 4))), tot)
        b.s_barrier_bare(fence="compiler")
        chk = PS.lds_read(b, b.add(lds0, b.mul(b.add(sl, _c(b, SLOTS)), _c(b, 4))))
        with b.scf_if(D.is_thread0(b)):
            nx = b.mod(b.add(j, _c(b, 2)), _c(b, 3))
            PS.lds_write(b, b.add(lds0, b.mul(b.add(nx, _c(b, SLOTS)), _c(b, 4))), _c(b, 0))
        return chk

    # ---------------------------------------------------------------- build
    def task(self, j: Value, trow: Value, ntask: Value):
        """Uniform fields of task ``j`` (clamped to the CTA's last task; not ``valid`` past it)."""
        b, A = self.b, self.a
        zero = _c(b, 0)
        live = b.cmp_lt(j, ntask)
        jj = b.select(live, j, b.smax(b.sub(ntask, _c(b, 1)), zero))
        tb = b.add(b.add(trow, _c(b, 1)), b.mul(jj, _c(b, TCOLS)))
        t = dict(lay=self.ld(A["tasks"], tb), stg=self.ld(A["tasks"], b.add(tb, _c(b, 1))),
                 u=self.ld(A["tasks"], b.add(tb, _c(b, 2))), fl=self.ld(A["tasks"], b.add(tb, _c(b, 3))))
        t["valid"] = b.land(live, b.cmp_ge(t["u"], zero))
        us = b.select(t["valid"], t["u"], zero)
        ub = b.mul(us, _c(b, UCOLS))
        t["ub"] = ub
        t["blk0"] = self.ld(A["utab"], ub)
        t["nblk"] = b.select(t["valid"], self.ld(A["utab"], b.add(ub, _c(b, 1))), zero)
        return t

    def ring(self, t):
        """``depth`` 16-byte loads a lane of the unit's first blocks (OOB = no traffic for empty / dummy units)."""
        b, A, k = self.b, self.a, self.k
        zero = _c(b, 0)
        wbase = b.add(A["wts"], b.mul(b.zext(b.add(b.mul(t["lay"], A["blk_layer"]), t["blk0"]), I64), b.const_i64(BLOCK)))
        w_rs = _rsrc(b, wbase, b.mul(t["nblk"], _c(b, BLOCK)))
        lane_off = b.mul(self.tid, _c(b, 16))
        ring = [b.buffer_load_cp(w_rs, b.add(lane_off, _c(b, i * BLOCK)), zero, dwords=4) for i in range(k.depth)]
        return ring, w_rs, lane_off

    def build(self) -> KernelDef:
        b, k = self.b, self.k
        A = self.a
        zero = _c(b, 0)
        tid = self.tid
        self.lds0 = PS.pad_lds(b, name_hint="pl2pad")
        with b.scf_if(D.is_thread0(b)):
            for s in range(3):
                PS.lds_write(b, b.add(self.lds0, _c(b, 4 * (SLOTS + s))), zero)
        b.s_barrier_bare(fence="compiler")
        E = _u(b, b.global_load_i32(_ptr(b, A["ctl"]), zero))
        tag = b.add(E, _c(b, 1))
        tk_a = b.add(A["ctl"], b.const_i64(1024))
        if k.ticket:
            row = b.mul(self.bid, _c(b, 8))
            info = b.int_to_ptr(A["info"], PtrType(I64, "global"))

            def st(col, v):
                b.store(v if v.type == I64 else b.zext(v, I64), info, b.add(row, _c(b, col)))

            PS.ticket_xcd(b, _TicketSpec("s_nop"), tk_a, row, st, A["gridn"], A["maxit"], _c(b, 1), fatal_err=A["err"])
            b.s_barrier_bare(fence="compiler")
        trow = b.mul(self.bid, b.add(b.mul(A["tmax"], _c(b, TCOLS)), _c(b, 1)))
        ntask = self.ld(A["tasks"], trow)
        D_ = k.depth
        counted = k.pub == "counted" and k.flags
        iters = [("pl2acc", zero)]
        if counted:
            # pub=counted: task j's first weight loads were issued at the end of task j - 1 (before its publish);
            # the prologue issues task 0's
            r0, _, _ = self.ring(self.task(zero, trow, ntask))
            iters += [(f"pl2c{i}", v) for i, v in enumerate(r0)]
        loop = b.scf_for_iter(zero, ntask, _c(b, 1), iters, iv_name="pl2j")
        with loop as (j, vals):
            vals = list(vals)
            acc = vals[0]
            t = self.task(j, trow, ntask)
            # pub=counted: the next task's table entries are read now (their loads must not sit between the payload
            # store and the counted wait)
            tn = self.task(b.add(j, _c(b, 1)), trow, ntask) if counted else None
            lay, stg, u, fl, valid, ub = t["lay"], t["stg"], t["u"], t["fl"], t["valid"], t["ub"]
            nblk = t["nblk"]
            rw = self.ld(A["utab"], b.add(ub, _c(b, 2)))
            doff = self.ld(A["utab"], b.add(ub, _c(b, 3)))
            dcnt = self.ld(A["utab"], b.add(ub, _c(b, 4)))
            pu0 = self.ld(A["utab"], b.add(ub, _c(b, 5)))
            pn = self.ld(A["utab"], b.add(ub, _c(b, 6)))
            # the injected delay, read with the other table words (a load right before the payload store would make
            # the compiler drain the unit's weight / state streams there)
            dly = self.ld(A["dly"], b.add(b.mul(lay, A["nunits"]), b.select(valid, u, zero)))
            sync = b.cmp_ne(b.land(fl, _c(b, 1)), zero)
            waitc = b.land(valid, b.cmp_eq(b.land(fl, _c(b, 2)), zero))  # a unit that waits for / reads its inputs
            if k.sync == "mix":
                pollc = b.land(waitc, b.cmp_ne(b.land(fl, _c(b, 4)), zero))  # this stage's edge is polled flags
            else:
                pollc = waitc
            sidx = b.lshr(fl, _c(b, 8))
            self.stamp(j, 0)
            if k.gridsync:
                with b.scf_if(sync):
                    b.s_waitcnt(vmcnt=0)
                    b.s_barrier_bare(fence="compiler")
                    live = b.cmp_eq(PS.lds_read(b, self.lds0), zero)
                    PS.grid_sync_arrive(b, self.gs, A["gs"], self.bid, live=live)
            # the unit's first weight loads (before the wait; pub=counted: carried from the previous task)
            if counted:
                w_rs, lane_off = self.wrs(t)
                ring = vals[1:1 + D_]
            else:
                ring, w_rs, lane_off = self.ring(t)
            pre = None
            if k.mutant == "noacq":
                pre = self.read_set(lay, stg, doff, b.select(waitc, dcnt, zero), j, pretouch=True)
            self.stamp(j, 1)
            if k.gridsync:
                with b.scf_if(sync):
                    live = b.cmp_eq(PS.lds_read(b, self.lds0), zero)
                    epoch1 = b.add(b.mul(E, A["nsync"]), sidx)
                    with b.scf_if(live):
                        PS.grid_sync_wait(b, self.gs, A["gs"], epoch1, self.lds0, A["err"], maxit=A["maxit"],
                                          bias=1 if k.mutant == "early" and k.sync == "grid" else 0)
                    b.s_barrier_bare(fence="compiler")
            if k.flags:
                with b.scf_if(pollc):
                    self.poll(lay, stg, doff, dcnt, pu0, pn, tag)
                    b.s_barrier_bare(fence="compiler")
            self.stamp(j, 2)
            chk = self.read_set(lay, stg, doff, b.select(waitc, dcnt, zero), j)
            if pre is not None:
                acc = b.xor(acc, pre)
            # stream the weight blocks: groups of `depth` blocks, ring slot i always holds block g * depth + i (no
            # register rotation, so `depth` loads stay in flight); blocks past nblk load 0 / store nothing (OOB)
            ngr = b.div(b.add(nblk, _c(b, D_ - 1)), _c(b, D_))
            isrw = b.cmp_ne(rw, zero)
            # the last group is peeled (consumed without issuing further loads), so no out-of-range load is in flight
            # after the stream and its registers can be reused without a vmcnt(0) drain
            ngm1 = b.smax(b.sub(ngr, _c(b, 1)), zero)
            sl = b.scf_for_iter(zero, ngm1, _c(b, 1), [(f"pl2r{i}", v) for i, v in enumerate(ring)] + [("pl2sa", acc)],
                                iv_name="pl2bk")
            with sl as (g, sv):
                sv = list(sv)
                rg, a2 = sv[:-1], sv[-1]
                nxt = []
                for i in range(D_):
                    v = rg[i]
                    # an order-dependent checksum (multiply-add chain): an xor chain is reassociated and keeps every
                    # slot live to the end of the group, which forces register copies of the in-flight ring
                    a2 = b.add(b.mul(a2, _c(b, 33)), b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
                    cur = b.add(lane_off, b.mul(b.add(b.mul(g, _c(b, D_)), _c(b, i)), _c(b, BLOCK)))
                    # rw units write the block back; others store to an out-of-range offset (dropped, no branch)
                    b.buffer_store_cp(v, w_rs, b.select(isrw, cur, _c(b, 0x7FFFFFF0)), zero, aux=CPOL_NT)
                    b.sched_barrier(0)  # slot i is consumed before its next load is issued (no ring copies)
                    nxt.append(b.buffer_load_cp(w_rs, b.add(cur, _c(b, D_ * BLOCK)), zero, dwords=4))
                b.scf_yield(*nxt, a2)
            res = list(sl.results)
            acc = res[-1]
            for i in range(D_):
                v = res[i]
                acc = b.add(b.mul(acc, _c(b, 33)), b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
                cur = b.add(lane_off, b.mul(b.add(b.mul(ngm1, _c(b, D_)), _c(b, i)), _c(b, BLOCK)))
                b.buffer_store_cp(v, w_rs, b.select(isrw, cur, _c(b, 0x7FFFFFF0)), zero, aux=CPOL_NT)
            PF = k.pf
            fbase = b.mul(b.add(b.mul(lay, A["nunits"]), u), _c(b, PF))
            pubc = b.land(valid, b.cmp_ne(b.land(fl, _c(b, 8)), zero)) if k.lazy else valid
            with b.scf_if(valid):
                with b.scf_for(zero, dly, _c(b, 1), iv_name="pl2dl"):
                    D.s_sleep(b, 1)
                h = b.add(b.add(b.mul(tag, _c(b, 0x01000193)), b.mul(lay, _c(b, 0x9E3779B1))),
                          b.add(b.mul(stg, _c(b, 0x85EBCA77)), b.mul(u, _c(b, 0xC2B2AE3D))))
                val = b.add(b.add(h, b.mul(chk, _c(b, 3))), b.mul(tid, _c(b, 0x9E37)))
                poff = b.mul(b.add(b.mul(b.add(b.mul(lay, A["nunits"]), u), _c(b, PAY_WORDS)), tid), _c(b, 4))
                if counted:
                    _st_sc1_mem(b, A["pay"], poff, val)
                else:
                    PS.st_sc1(b, A["pay"], poff, val)
            if k.flags and not counted:
                with b.scf_if(pubc):
                    b.s_waitcnt(vmcnt=0)
                    if k.pub == "drain":
                        b.s_barrier_bare(fence="compiler")
                        with b.scf_if(D.is_thread0(b)):
                            PS.st_sc1(b, A["flg"], b.mul(fbase, _c(b, 4)), tag)
                    else:
                        with b.scf_if(b.cmp_eq(self.lane, zero)):
                            PS.st_sc1(b, A["flg"], b.mul(b.add(fbase, self.wave), _c(b, 4)), tag)
            carry = []
            if counted:
                # issue the next task's first weight loads, then wait until only they are in flight (vector memory
                # operations retire in issue order, so this wave's payload stores are done), barrier, flag: the
                # publish does not drain the next unit's weight stream (DESIGN-CCL 4.2.1 publish=counted)
                carry, _, _ = self.ring(tn)
                b.inline_asm(f"s_waitcnt vmcnt({D_})", "~{memory}", [], None)
                with b.scf_if(pubc):
                    b.s_barrier_bare(fence="compiler")
                    with b.scf_if(D.is_thread0(b)):
                        PS.st_sc1(b, A["flg"], b.mul(fbase, _c(b, 4)), tag)
            self.stamp(j, 3)
            b.scf_yield(acc, *carry)
        acc = loop.results[0]
        if not k.stamp:
            with b.scf_if(b.cmp_eq(acc, _c(b, MAGIC))):
                b.store(acc, _ptr(b, A["stm"]), tid)
        # launch end: the last CTA zeroes the ticket words and advances the epoch
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, A["ctl"], _c(b, 4), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(b.add(old, _c(b, 1)), b.mul(tag, _c(b, GRID)))):
                for w in range(9):
                    PS.st_sc1(b, tk_a, _c(b, w * PS.LINE), zero)
                PS.st_sc1(b, A["ctl"], zero, tag)
        b.ret()
        return b.kernel

    def wrs(self, t):
        b, A = self.b, self.a
        wbase = b.add(A["wts"], b.mul(b.zext(b.add(b.mul(t["lay"], A["blk_layer"]), t["blk0"]), I64), b.const_i64(BLOCK)))
        return _rsrc(b, wbase, b.mul(t["nblk"], _c(b, BLOCK))), b.mul(self.tid, _c(b, 16))


def _st_sc1_mem(b: IRBuilder, base, voff, v) -> None:
    """``global_store_dword ... sc1`` with a memory clobber: no load may move across it (pub=counted's
    ``vmcnt(depth)`` relies on the next unit's loads being issued after the payload store)."""
    b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s,~{memory}", [voff, v, base], None)


def build_chain_kernel(k: ChainKnobs) -> KernelDef:
    return _Chain(k).build()
