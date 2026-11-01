# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Microbench kernels of the platform's own primitives (WP1 hardware gate).

* :func:`build_gs_kernel`: :func:`.sync.grid_sync` (in-kernel stripe layout,
  per-CTA (thread 0 or last wave) or per-wave arrivals, abort rule, launch-fatal word) with exactly
  the parameters and output columns of P4's counter kernel
  (``probes/p4_sync.build_counter_kernel``), so P4's bench drives it and
  times it against P4's ``s8L`` arm. The counter table's per-CTA dropped
  epoch (``tab[TAB_ARR + grid + cta]``) is honoured (the dropped-arrival
  control); the other table words are ignored (the layout is computed).
  Mutant ``early``: every word waits for one arrival less (must fail with a
  late producer). Mutant ``noinit`` (``arrivals="last"`` only): the
  :func:`.sync.cta_arrive` LDS counter is not zeroed (must fail after
  :func:`build_lds_poison_kernel`).
  ``arrivals="last"``: every wave stores its own data word (``dat`` holds
  ``phases x grid x waves`` words) and the checker reads all of them;
  ``stagger`` orders the waves' arrivals.
* :func:`build_lds_poison_kernel`: fills a CTA's whole LDS pool with a
  value, so a following launch on the same CUs starts from non-zero LDS.
* :func:`build_ticket_fatal_kernel`: P4's per-XCD-start ticket with the
  launch-fatal word (``err``): an oversize grid must set ``ticket_timeout``.

Kernel names start with ``q38wp1_``; nothing here is imported by a served
build.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D

from . import sync as PS

__all__ = ["GsSpec", "TicketFatalSpec", "build_gs_kernel", "build_lds_poison_kernel", "build_ticket_fatal_kernel"]

OUT_COLS = 4
TAB_ARR = 192  # p4_sync.TAB_ARR
GS_MUTANTS = ("none", "early", "noinit")


@dataclass(frozen=True)
class GsSpec:
    grid: int = PS.GRID
    threads: int = PS.THREADS
    stripes: int = 8
    arrivals: str = "cta"
    pause: str = "s_nop"
    fatal: bool = True
    mutant: str = "none"
    tag: str = ""
    stagger: int = 0  # arrivals="last": wave w sleeps w * stagger x 64 clocks before its store (orders the arrivals)

    def name(self) -> str:
        parts = ["q38wp1_gs", f"g{self.grid}", f"t{self.threads}", f"s{self.stripes}", f"a{self.arrivals}"]
        if self.stagger:
            parts.append(f"st{self.stagger}")
        if not self.fatal:
            parts.append("nofw")
        if self.pause != "s_nop":
            parts.append(f"p{self.pause}")
        if self.mutant != "none":
            parts.append(f"mut{self.mutant}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def _payload_word(b, salt, e, w):
    v = b.add(b.mul(salt, b.const_i32(1000003)), b.mul(e, b.const_i32(65599)))
    return b.add(v, b.add(b.mul(w, b.const_i32(7)), b.const_i32(1)))


def build_gs_kernel(spec: GsSpec = GsSpec()) -> KernelDef:  # noqa: B008
    """Params as P4's counter kernel: ``sync``, ``tab``, ``dtab``, ``out``,
    ``dat``, ``err`` (i32[4]: mismatch, timeouts, fatal bits), ``info``,
    ``salt``, ``phases``, ``maxit``. Every CTA is a producer: it stores its
    data word (``sc1``) and arrives, then waits; after release every thread
    ``< grid`` checks one producer's word."""
    if spec.mutant not in GS_MUTANTS:
        raise ValueError(spec.mutant)
    if spec.mutant == "noinit" and spec.arrivals != "last":
        raise ValueError("mutant noinit needs arrivals='last'")
    if spec.stagger and spec.arrivals != "last":
        raise ValueError("stagger needs arrivals='last'")
    G, T = spec.grid, spec.threads
    gs = PS.GridSync(grid=G, stripes=spec.stripes, pause=spec.pause, arrivals=spec.arrivals, waves=T // 64)
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    sync = b.param("sync", gp)
    tab = b.param("tab", gp)
    dtab = b.param("dtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    dat = b.param("dat", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lds0 = PS.pad_lds(b, name_hint="wp1pad", arrive_word=spec.arrivals == "last" and spec.mutant != "noinit")
    with b.scf_if(D.is_thread0(b)):
        row = b.mul(bid, b.const_i32(2))
        b.store(b.zext(D.xcc_id(b), I64), info, row)
        b.store(b.zext(D.hw_id(b), I64), info, b.add(row, b.const_i32(1)))
        with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
            b.store(D.memtime(b), info, b.const_i32(2 * G))
            b.store(D.memrealtime(b), info, b.const_i32(2 * G + 1))
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    dat_a = b.readfirstlane(b.global_ptr_to_i64(dat))
    drop = b.readfirstlane(b.global_load_i32(tab, b.add(bid, b.const_i32(TAB_ARR + G))))
    chk = b.cmp_lt(tid, b.const_i32(G))
    tid0 = D.is_thread0(b)
    four = b.const_i32(4)
    bias = 1 if spec.mutant == "early" else 0
    last = spec.arrivals == "last"
    W = T // 64
    if last:
        # every wave is a producer of its own word (dat = phases x grid x waves words): the CTA's arrival must come
        # from its last wave, after every wave's store (cta_arrive's rule, no CTA barrier before it)
        wv = b.readfirstlane(b.lshr(tid, b.const_i32(6)))
        lane0 = b.cmp_eq(b.lane_id(), b.const_i32(0))
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        PS.delay(b, d)
        ok_cta = b.cmp_eq(PS.lds_read(b, lds0), b.const_i32(0))
        arrive = b.cmp_ne(ph, drop)
        if last:
            if spec.stagger:
                # not PS.delay: a second scf_for with its iv_name "p4d" lowers to a duplicate SSA name (COMGR error)
                with b.scf_for(b.const_i32(0), b.mul(wv, b.const_i32(spec.stagger)), b.const_i32(1), iv_name="wst"):
                    D.s_sleep(b, 1)
            with b.scf_if(b.land(b.land(ok_cta, arrive), lane0)):
                k = b.add(b.mul(b.add(eg, bid), b.const_i32(W)), wv)
                PS.st_sc1(b, dat_a, b.mul(k, four), _payload_word(b, salt, ph, b.add(b.mul(bid, b.const_i32(W)), wv)))
        else:
            with b.scf_if(b.land(b.land(ok_cta, arrive), tid0)):
                PS.st_sc1(b, dat_a, b.mul(b.add(eg, bid), four), _payload_word(b, salt, ph, bid))
        b.s_waitcnt(vmcnt=0)
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        with b.scf_if(b.land(b.land(ok_cta, arrive), tid0)):
            PS.stamp(b, out, row, 0)
        released = PS.grid_sync(b, gs, sync_a, bid, b.add(ph, b.const_i32(1)), lds0, err_a, maxit=maxit,
                                arrive=arrive, bias=bias, fatal=spec.fatal)  # fmt: skip
        with b.scf_if(tid0):
            PS.stamp(b, out, row, 1)
            st = b.select(
                ok_cta,
                b.select(released, b.const_i32(PS.ST_OK), b.const_i32(PS.ST_TIMEOUT)),
                b.const_i32(PS.ST_ABORTED),
            )
            b.store(b.zext(st, I64), out, b.add(row, b.const_i32(2)))
            b.store(b.const_i64(0), out, b.add(row, b.const_i32(3)))
        with b.scf_if(b.land(chk, released)):
            if not last:
                v = PS.ld_sc1(b, dat_a, b.mul(b.add(eg, tid), four))
                with b.scf_if(b.cmp_ne(v, _payload_word(b, salt, ph, tid))):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * PS.ERR_MISMATCH), b.const_i32(1))
            for kw in range(W if last else 0):
                idx = b.add(b.mul(b.add(eg, tid), b.const_i32(W)), b.const_i32(kw))
                v = PS.ld_sc1(b, dat_a, b.mul(idx, four))
                want = _payload_word(b, salt, ph, b.add(b.mul(tid, b.const_i32(W)), b.const_i32(kw)))
                with b.scf_if(b.cmp_ne(v, want)):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * PS.ERR_MISMATCH), b.const_i32(1))
    with b.scf_if(b.land(tid0, b.cmp_eq(bid, b.const_i32(0)))):
        b.store(D.memtime(b), info, b.const_i32(2 * G + 2))
        b.store(D.memrealtime(b), info, b.const_i32(2 * G + 3))
    b.ret()
    return b.kernel


def build_lds_poison_kernel(threads: int = PS.THREADS) -> KernelDef:
    """Param ``value`` (i32): every thread writes it to its words of a
    one-CTA-per-CU LDS pool (asm stores, so the pool is kept)."""
    from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

    b = IRBuilder(f"q38wp1_ldspoison_t{threads}")
    b.kernel.attrs["max_workgroup_size"] = threads
    value = b.readfirstlane(b.param("value", I32))
    lds = b.smem_alloc(I32, [ONE_CTA_PER_CU_WORDS], name_hint="poison")
    w0 = b.trunc(b.smem_addr_of(lds), I32)
    tid = b.thread_id_x()
    n = (ONE_CTA_PER_CU_WORDS + threads - 1) // threads
    with b.scf_for(b.const_i32(0), b.const_i32(n), b.const_i32(1), iv_name="i") as i:
        w = b.add(b.mul(i, b.const_i32(threads)), tid)
        with b.scf_if(b.cmp_lt(w, b.const_i32(ONE_CTA_PER_CU_WORDS))):
            PS.lds_write(b, b.add(w0, b.mul(w, b.const_i32(4))), value)
    b.s_barrier_bare(fence="compiler")
    b.ret()
    return b.kernel


@dataclass(frozen=True)
class TicketFatalSpec:
    threads: int = PS.THREADS
    pause: str = "s_nop"

    def name(self) -> str:
        return f"q38wp1_tkf_t{self.threads}"


def build_ticket_fatal_kernel(spec: TicketFatalSpec = TicketFatalSpec()) -> KernelDef:  # noqa: B008
    """P4's ``start="xcd"`` ticket with the launch-fatal word. Params:
    ``sync`` (zeroed per launch), ``out`` i64[grid * 8] (P4's ticket row),
    ``err`` i32[4], ``expected``, ``maxit``, ``ticket``."""
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    sync = b.param("sync", PtrType(I32, "global"))
    out = b.param("out", PtrType(I64, "global"))
    err = b.param("err", PtrType(I32, "global"))
    expected = b.readfirstlane(b.param("expected", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    ticket = b.readfirstlane(b.param("ticket", I32))
    bid = b.readfirstlane(b.block_id_x())
    PS.pad_lds(b, name_hint="wp1pad")
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    row = b.mul(bid, b.const_i32(8))

    def st(col, v):
        b.store(v if v.type == I64 else b.zext(v, I64), out, b.add(row, b.const_i32(col)))

    PS.ticket_xcd(b, spec, sync_a, row, st, expected, maxit, ticket, fatal_err=err_a)
    b.s_barrier_bare(fence="compiler")
    b.ret()
    return b.kernel
