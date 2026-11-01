# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1: a deterministic stale-read harness for the LB edge protocol (DESIGN-LB 15.4 "correctness infrastructure").

P-L4 F2: the ``noacq`` twin (pretouch + plain loads) of a full kernel is not a reliable must-fail check on MI355X:
whether the consumer still holds the old line when it reads depends on timing. Here the timing is forced: **a producer
cannot store until its consumer has done the thing the twin needs** (pretouched the lines, or read them), without any
traffic between that point and the read that could evict them.

One launch, 256 CTAs x 512 threads (one CTA per CU). CTA ``p < 128`` produces for consumer CTA
``128 + (p + xs) mod 128`` (``xs = 0``: the same XCD, CTA ``c`` and ``c + 128`` share an XCD; ``xs = 1``: the
neighbouring XCD). ``T`` trials in sequence, each on its own 2 KiB payload, flag and hold word, so no reset is needed
and the host pre-fills every payload with OLD words and every flag with the previous tag ``tag - 1``.

Per trial, consumer: (1) pretouch its producer's lines (plain loads, ``vmcnt(0)``); with ``sig="pre"`` (2) signal the
producer (``sc1`` store of the hold word); (3) poll the flag (``ccl.wire.poll_flags`` with ``sc1`` loads, the
framework's poll; ``tag`` twin: polls for ``tag - 1``, which the flag already holds); (4) read the payload with the
protocol under test: ``sc1`` (``sc1`` loads: the framework's ``read_set``), ``inv`` (``buffer_inv sc1`` + plain loads),
or ``plain`` (no acquire: the ``noacq`` twin); ``early`` twin: the read happens before the poll; with ``sig="post"`` the
signal comes after the read. (5) count words that differ from the NEW pattern into ``out[t][c]``.
Producer: wait for the hold word (``hold`` deadline in 100 MHz ticks; a timeout is counted in ``cnt[0]``, not fatal:
with ``sig="post"`` a correct consumer never signals before the flag), then store NEW words (``st_sc1``, the
framework's ``store_payload``), drain, barrier, flag = ``tag`` (the framework's ``drain`` publish).

Determinism: with ``sig="post"`` a read that precedes the hand-off (``early``, ``tag``) happens before the producer may
store, so it reads OLD words every time. With ``sig="pre"`` the producer stores only after the pretouch, so a plain read
can only see NEW words if the pretouched line was dropped between the pretouch and the read (nothing but the signal
store and the flag polls run in between). The hardware run decides whether that is deterministic per placement.

Kernel arguments: i64 ``pay, flg, hw, out, cnt, err, sink``; i32 ``T, tag, xs, dl, hold``. Names ``q38lbf_stale_*``.
The kernel computes the NEW words itself (``tag * 0x01000193 + t * 0x9E3779B1 + p * 0x85EBCA77 + i * 0x9E37``); the host
only pre-fills OLD words (any pattern that differs from every NEW word).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import CPOL_SC1, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers import device_ops as D

from ..ccl import wire as W
from ..persist import sync as PS

__all__ = ["StaleKnobs", "build_stale_kernel", "stale_name", "PROTOS", "MUTANTS", "SIGS", "PRODUCERS", "WORDS"]

PROTOS = ("sc1", "inv", "plain")
MUTANTS = ("none", "early", "tag")
SIGS = ("pre", "post")
PRODUCERS = 128
WORDS = 512
THREADS = 512
M32 = 0xFFFFFFFF


@dataclass(frozen=True)
class StaleKnobs:
    proto: str = "sc1"
    mutant: str = "none"
    sig: str = "pre"
    hz: bool = True

    def check(self) -> None:
        if self.proto not in PROTOS or self.mutant not in MUTANTS or self.sig not in SIGS:
            raise ValueError(self)


def stale_name(k: StaleKnobs) -> str:
    return f"q38lbf_stale_{k.proto}_{k.mutant}_{k.sig}" + ("" if k.hz else "_hz0")


def _c(b: IRBuilder, v: int) -> Value:
    v = int(v) & M32
    return b.const_i32(v - (1 << 32) if v >= 1 << 31 else v)


def build_stale_kernel(k: StaleKnobs) -> KernelDef:
    k.check()
    b = IRBuilder(stale_name(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    a = {n: b.readfirstlane(b.param(n, I64)) for n in ("pay", "flg", "hw", "out", "cnt", "err", "sink")}
    for n in ("T", "tag", "xs", "dl", "hold"):
        a[n] = b.readfirstlane(b.param(n, I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lane = b.lane_id()
    zero = _c(b, 0)
    lds0 = PS.pad_lds(b, name_hint="lbstpad")
    b.s_barrier_bare(fence="compiler")
    isprod = b.cmp_lt(bid, _c(b, PRODUCERS))
    p_of_c = b.mod(bid, _c(b, PRODUCERS))
    # consumer c = 128 + (p + xs) mod 128  =>  p = (c - 128 - xs) mod 128
    pc = b.mod(b.add(b.sub(p_of_c, a["xs"]), _c(b, PRODUCERS)), _c(b, PRODUCERS))
    pay_rs = b.buffer_rsrc(b.int_to_ptr(a["pay"], PtrType(I32, "global")), _c(b, 0x7FFFFFF0))
    abort_a = b.add(a["err"], b.const_i64(12))
    with b.scf_for(zero, a["T"], _c(b, 1), iv_name="stt") as t:
        with b.scf_if(isprod):
            p = bid
            hoff = b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, 4))
            with b.scf_if(D.is_wave0(b)):
                dl = b.add(D.memrealtime(b), b.zext(a["hold"], I64))
                haddr = b.add(a["hw"], b.zext(hoff, I64))
                _s, _m, st = W.poll_flags(b, haddr, _c(b, 1), dl, abort_a, zero, pause="s_nop", poll_cpol="sc1")
                with b.scf_if(b.land(b.cmp_ne(st, zero), b.cmp_eq(lane, zero))):
                    D.atomic_add_scoped(b, a["cnt"], zero, _c(b, 1))
            b.s_barrier_bare(fence="compiler")
            val = b.add(b.add(b.mul(a["tag"], _c(b, 0x01000193)), b.mul(t, _c(b, 0x9E3779B1))),
                        b.add(b.mul(p, _c(b, 0x85EBCA77)), b.mul(tid, _c(b, 0x9E37))))
            poff = b.mul(b.add(b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, WORDS)), tid), _c(b, 4))
            PS.st_sc1(b, a["pay"], poff, val)
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare(fence="compiler")
            with b.scf_if(D.is_thread0(b)):
                PS.st_sc1(b, a["flg"], hoff, a["tag"])
        with b.scf_if(b.cmp_ge(bid, _c(b, PRODUCERS))):
            p = pc
            hoff = b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, 4))
            rd = b.cmp_lt(tid, _c(b, 128))
            woff = b.mul(b.add(b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, WORDS)), b.mul(tid, _c(b, 4))), _c(b, 4))
            woff = b.select(rd, woff, _c(b, 0x7FFFFFF0))

            def load(aux):
                return b.buffer_load_cp(pay_rs, woff, zero, dwords=4, aux=aux)

            def signal():
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare(fence="compiler")
                with b.scf_if(D.is_thread0(b)):
                    PS.st_sc1(b, a["hw"], hoff, _c(b, 1))

            pre = load(0)
            acc = zero
            for e in range(4):
                acc = b.add(acc, b.vec_extract(pre, e))
            b.s_waitcnt(vmcnt=0)
            got = None
            if k.mutant == "early":
                got = load({"sc1": CPOL_SC1, "inv": 0, "plain": 0}[k.proto])
                b.s_waitcnt(vmcnt=0)
            if k.sig == "pre":
                signal()
            if k.mutant == "early" and k.sig == "post":
                signal()
            want = b.sub(a["tag"], _c(b, 1)) if k.mutant == "tag" else a["tag"]
            fdl = b.add(D.memrealtime(b), b.zext(a["dl"], I64))
            faddr = b.add(a["flg"], b.zext(hoff, I64))
            _s, _m, st = W.poll_flags(b, faddr, want, fdl, abort_a, zero, pause="s_nop", poll_cpol="sc1")
            with b.scf_if(b.land(b.cmp_ne(st, zero), b.cmp_eq(tid, zero))):
                D.atomic_add_scoped(b, a["cnt"], _c(b, 4), _c(b, 1))
                PS.raise_fatal(b, a["err"], "poll_timeout")
            b.s_barrier_bare(fence="compiler")
            if got is None:
                if k.proto == "inv":
                    b.inline_asm("buffer_inv sc1", "~{memory}", [], None)
                got = load(CPOL_SC1 if k.proto == "sc1" else 0)
                b.s_waitcnt(vmcnt=0)
            if k.sig == "post" and k.mutant != "early":
                signal()
            base = b.add(b.add(b.mul(a["tag"], _c(b, 0x01000193)), b.mul(t, _c(b, 0x9E3779B1))),
                         b.mul(p, _c(b, 0x85EBCA77)))
            bad = zero
            for e in range(4):
                exp = b.add(base, b.mul(b.add(b.mul(tid, _c(b, 4)), _c(b, e)), _c(b, 0x9E37)))
                bad = b.add(bad, b.select(b.cmp_ne(b.vec_extract(got, e), exp), _c(b, 1), zero))
            bad = b.select(rd, bad, zero)
            with b.scf_if(b.cmp_ne(bad, zero)):
                ooff = b.mul(b.add(b.mul(t, _c(b, 256)), bid), _c(b, 4))
                D.atomic_add_scoped(b, a["out"], ooff, bad)
            acc = b.add(acc, b.vec_extract(got, 0))
            with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):  # keeps the pretouch loads alive
                b.store(acc, b.int_to_ptr(a["sink"], PtrType(I32, "global")), tid)
    b.ret()
    kd = b.kernel
    if k.hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd
