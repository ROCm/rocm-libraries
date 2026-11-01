# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2c: a deterministic stale-read harness for the GDN chain's acquire variants (``gdn_lb.ChainKnobs.acq``).

WP-L1's harness (:mod:`.lbf_stale`) proves a single consumer's acquire (``buffer_inv sc1`` + plain loads). The chain's
``acq="xcd"`` splits it: one leader CTA per real XCD invalidates the XCD's L2 (``buffer_inv sc1``) and releases the
others through an XCD word; they invalidate only their own L1 (``buffer_inv sc0``). This kernel forces the case each
half must cover: every consumer pretouches its producer's lines first (plain loads put them in its CU's L1 and its XCD's
L2), signals, and only then may the producer store the NEW words (write-through ``sc1``) and arrive on the trial's
counter. Then the consumers run the protocol under test and read with plain loads:

* ``load`` (the chain's default ``acq="load"``: plain loads become ``sc1`` loads, no invalidate): positive.
* ``xcd`` (an XCD-leader form, refused by the chain): run as a positive and measured to FAIL (stale in its followers).
  ``wave``: every consumer ``buffer_inv sc1`` (the chain's ``acq="wave"``, the first measured base): positive. ``cta``: wave 0 of each consumer ``buffer_inv sc1`` + wait + CTA barrier, then ``buffer_inv sc0`` (the
  chain's ``acq="cta"``): positive.
* twins (must read stale): ``xcd_nol1`` (followers skip their L1 invalidate: their pretouched L1 lines are hit),
  ``xcd_nol2`` (the leader skips the L2 invalidate, everyone invalidates L1: the XCD's L2 lines are hit), ``plain``
  (no acquire at all).

One launch, 256 CTAs x 512 threads: producers ``p < 128``, consumer ``128 + (p + xs) mod 128``. ``T`` trials, each on
its own 2 KiB payload, hold word, counter and XCD release words; the host pre-fills OLD words. ``out[t][c]`` counts the
consumer's stale words; ``cnt[0]`` producer hold timeouts, ``cnt[1]`` consumer wait timeouts (bounded waits, never a hang).
Kernel arguments: i64 ``pay, hw, ctr, xw, rk, out, cnt, sink``; i32 ``T, xs, maxit``. Names ``q38lb_acqstale_*``.
"""

from __future__ import annotations

from rocke.core.ir import CPOL_SC1, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers import device_ops as D

from ..persist import sync as PS

__all__ = ["PROTOS", "build_acq_stale_kernel", "acq_stale_name", "POSITIVE"]

PROTOS = ("xcd", "wave", "cta", "load", "xcd_nol1", "xcd_nol2", "plain")
POSITIVE = ("xcd", "wave", "cta", "load")
PRODUCERS = 128
WORDS = 512
THREADS = 512
INV1 = "s_waitcnt vmcnt(0)\nbuffer_inv sc1\ns_waitcnt vmcnt(0)"


def acq_stale_name(proto: str) -> str:
    return f"q38lb_acqstale_{proto}"


def _c(b: IRBuilder, v: int) -> Value:
    v = int(v) & 0xFFFFFFFF
    return b.const_i32(v - (1 << 32) if v >= 1 << 31 else v)


def _new(b, t, p, w) -> Value:
    return b.add(b.add(b.mul(t, _c(b, 0x9E3779B1)), b.mul(p, _c(b, 0x85EBCA77))), b.add(b.mul(w, _c(b, 0x9E37)), _c(b, 7)))


def _wait_ge(b, base, voff, target, maxit, cnt, slot) -> None:
    """Wave 0 (lane 0's word) waits until the word >= target; a timeout is counted, never fatal."""
    with b.scf_if(D.is_wave0(b)):
        lane = b.lane_id()
        _s, it = PS.poll_counter(b, voff, base, b.select(b.cmp_eq(lane, _c(b, 0)), target, _c(b, 0)), maxit, "s_nop")
        with b.scf_if(b.land(D.is_thread0(b), b.cmp_ge(it, maxit))):
            D.atomic_add_scoped(b, cnt, _c(b, 4 * slot), _c(b, 1))
    b.s_barrier_bare(fence="compiler")


def build_acq_stale_kernel(proto: str, hz: bool = True) -> KernelDef:
    if proto not in PROTOS:
        raise ValueError(proto)
    b = IRBuilder(acq_stale_name(proto))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    a = {n: b.readfirstlane(b.param(n, I64)) for n in ("pay", "hw", "ctr", "xw", "rk", "out", "cnt", "sink")}
    for n in ("T", "xs", "maxit"):
        a[n] = b.readfirstlane(b.param(n, I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    zero = _c(b, 0)
    lds0 = PS.pad_lds(b, name_hint="acqstpad")
    b.s_barrier_bare(fence="compiler")
    isprod = b.cmp_lt(bid, _c(b, PRODUCERS))
    xcc = b.readfirstlane(D.xcc_id(b))
    # consumers elect one leader per real XCD (the first arrival on the XCD's rank word)
    with b.scf_if(b.land(b.lnot(isprod), D.is_thread0(b))):
        old = D.atomic_add_scoped(b, a["rk"], b.mul(xcc, _c(b, PS.LINE)), _c(b, 1), scope="agent", returns=True)
        PS.lds_write(b, b.add(lds0, _c(b, 16)), old)
    b.s_barrier_bare(fence="compiler")
    leader = b.cmp_eq(PS.lds_read(b, b.add(lds0, _c(b, 16))), zero)
    pc = b.mod(b.add(b.sub(b.mod(bid, _c(b, PRODUCERS)), a["xs"]), _c(b, PRODUCERS)), _c(b, PRODUCERS))
    pay_rs = b.buffer_rsrc(b.int_to_ptr(a["pay"], PtrType(I32, "global")), _c(b, 0x7FFFFFF0))
    with b.scf_for(zero, a["T"], _c(b, 1), iv_name="ast") as t:
        with b.scf_if(isprod):
            p = bid
            hoff = b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, 4))
            _wait_ge(b, a["hw"], hoff, _c(b, 1), a["maxit"], a["cnt"], 0)  # the consumer has pretouched
            poff = b.mul(b.add(b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, WORDS)), tid), _c(b, 4))
            PS.st_sc1(b, a["pay"], poff, _new(b, t, p, tid))
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare(fence="compiler")
            with b.scf_if(D.is_thread0(b)):
                D.atomic_add_scoped(b, a["ctr"], b.mul(t, _c(b, PS.LINE)), _c(b, 1), scope="agent")
        with b.scf_if(b.lnot(isprod)):
            p = pc
            hoff = b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, 4))
            rd = b.cmp_lt(tid, _c(b, 128))
            woff = b.mul(b.add(b.mul(b.add(b.mul(t, _c(b, PRODUCERS)), p), _c(b, WORDS)), b.mul(tid, _c(b, 4))), _c(b, 4))
            woff = b.select(rd, woff, _c(b, 0x7FFFFFF0))
            pre = b.buffer_load_cp(pay_rs, woff, zero, dwords=4)
            acc = b.vec_extract(pre, 0)
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare(fence="compiler")
            with b.scf_if(D.is_thread0(b)):
                PS.st_sc1(b, a["hw"], hoff, _c(b, 1))
            # every producer of the trial has stored (the chain's grid sync)
            _wait_ge(b, a["ctr"], b.mul(t, _c(b, PS.LINE)), _c(b, PRODUCERS), a["maxit"], a["cnt"], 1)
            xoff = b.add(b.mul(t, _c(b, 16 * PS.LINE)), b.mul(xcc, _c(b, PS.LINE)))
            if proto == "wave":
                b.inline_asm("buffer_inv sc1", "~{memory}", [], None)
            elif proto == "cta":
                with b.scf_if(D.is_wave0(b)):
                    b.inline_asm(INV1, "~{memory}", [], None)
                b.s_barrier_bare(fence="compiler")
                b.inline_asm("buffer_inv sc0", "~{memory}", [], None)
            elif proto.startswith("xcd"):
                with b.scf_if(leader):
                    if proto != "xcd_nol2":
                        b.inline_asm(INV1, "~{memory}", [], None)
                    else:
                        b.inline_asm("buffer_inv sc0", "~{memory}", [], None)
                    b.s_barrier_bare(fence="compiler")
                    with b.scf_if(D.is_thread0(b)):
                        PS.st_sc1(b, a["xw"], xoff, _c(b, 1))
                with b.scf_if(b.lnot(leader)):
                    _wait_ge(b, a["xw"], xoff, _c(b, 1), a["maxit"], a["cnt"], 1)
                    if proto != "xcd_nol1":
                        b.inline_asm("buffer_inv sc0", "~{memory}", [], None)
            got = b.buffer_load_cp(pay_rs, woff, zero, dwords=4, aux=CPOL_SC1 if proto == "load" else 0)
            b.s_waitcnt(vmcnt=0)
            bad = zero
            for e in range(4):
                exp = _new(b, t, p, b.add(b.mul(tid, _c(b, 4)), _c(b, e)))
                bad = b.add(bad, b.select(b.cmp_ne(b.vec_extract(got, e), exp), _c(b, 1), zero))
            bad = b.select(rd, bad, zero)
            with b.scf_if(b.cmp_ne(bad, zero)):
                D.atomic_add_scoped(b, a["out"], b.mul(b.add(b.mul(t, _c(b, 256)), bid), _c(b, 4)), bad)
            acc = b.add(acc, b.vec_extract(got, 0))
            with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):  # keeps the pretouch loads alive
                b.store(acc, b.int_to_ptr(a["sink"], PtrType(I32, "global")), tid)
    b.ret()
    kd = b.kernel
    if hz:
        from .lbf_hazard import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd
