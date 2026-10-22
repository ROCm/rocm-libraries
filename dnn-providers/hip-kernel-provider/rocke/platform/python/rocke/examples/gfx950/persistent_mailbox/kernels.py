# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mailbox microkernels: day-1 device-op gate, ping-pong, all-to-all, launch
gap and tag wrap.

All timings are taken on the device with ``s_memrealtime`` (a constant-rate
clock shared by every XCD), so the host only divides tick differences.

Tags: the ping-pong and all-to-all kernels take a ``base`` i32 and use
``base + it`` (``it`` from 1); the host advances ``base`` past every tag a
launch used, so no stale pair from an earlier launch ever matches. The tag
wrap kernel uses the real ``launch_tag(step, layer)`` rule instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rocke.core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

__all__ = [
    "A2A_BARRIERS",
    "A2A_COLS",
    "A2A_CONSUME",
    "BULK_FENCES",
    "BULK_ROW",
    "DOPS_COLS",
    "DOPS_SLEEP",
    "GATE_FORMS",
    "PP_COLS",
    "AllToAllSpec",
    "BulkSpec",
    "PingPongSpec",
    "a2a_expected_thread_sum",
    "build_all_to_all",
    "build_bulk_kernel",
    "build_device_ops_kernel",
    "build_gate_kernel",
    "build_pingpong",
    "build_stamp_kernel",
    "build_tag_wrap_kernel",
    "load_ir_impl",
]

GATE_FORMS = ("asm_s", "asm_v", "ir", "rsrc_vec")
PP_COLS = 8
A2A_COLS = 4
IR_IMPL_MODULE = "kernels.gfx950.qwen38_mono.mailbox_ir"


def load_ir_impl() -> bool:
    """Register the Python-engine-only ``ir`` mailbox implementation (it lives
    in the rocKE library family, not in ``rocke.helpers``; the library root
    must be on ``sys.path``). Returns False when it is not importable."""
    import importlib

    try:
        importlib.import_module(IR_IMPL_MODULE)
    except ImportError:
        return False
    return "ir" in M.registered_impls()


def _name(base: str, knobs: M.MailboxKnobs, *extra: str) -> str:
    parts = [base, *extra]
    sfx = knobs.suffix()
    if sfx:
        parts.append(sfx)
    return "_".join(parts)


def build_gate_kernel(form: str) -> KernelDef:
    """Day-1 gate: ``dst[t] = src[t] * 2 + 1`` for the 64 lanes of one wave,
    through pointers rebuilt from i64 params (``src_addr``, ``dst_addr``;
    ``n`` = source bytes).

    Forms: ``asm_s`` ``ptr_from_i64`` (``s_mov_b64``); ``asm_v`` a per-lane
    address ``src + 4t`` through ``v_mov_b64``; ``ir`` ``IRBuilder.int_to_ptr``
    (Python engine only: this example form is excluded from cpp parity);
    ``rsrc_vec`` an asm-built descriptor fed to an asm
    ``buffer_load_dword ... sc1``.
    """
    if form not in GATE_FORMS:
        raise ValueError(form)
    b = IRBuilder(f"mbx_gate_{form}")
    b.kernel.attrs["max_workgroup_size"] = 64
    s_addr = b.param("src_addr", I64)
    d_addr = b.param("dst_addr", I64)
    n = b.param("n", I32)
    tid = b.thread_id_x()
    dst = D.ptr_from_i64(b, b.readfirstlane(d_addr))
    if form == "asm_s":
        src = D.ptr_from_i64(b, b.readfirstlane(s_addr))
        v = b.global_load_i32(src, tid)
    elif form == "ir":
        src = b.int_to_ptr(s_addr, PtrType(I32, "global"))
        v = b.global_load_i32(src, tid)
    elif form == "asm_v":
        lane_addr = b.add(s_addr, b.zext(b.mul(tid, b.const_i32(4)), I64))
        src = D.ptr_from_i64(b, lane_addr, uniform=False)
        v = b.global_load_i32(src, b.const_i32(0))
    else:
        rs = D.rsrc_vec(b, b.readfirstlane(s_addr), b.readfirstlane(n))
        v = b.inline_asm(
            "buffer_load_dword $0, $1, $2, 0 offen sc1\ns_waitcnt vmcnt(0)",
            "=&v,v,s",
            [b.mul(tid, b.const_i32(4)), rs],
            I32,
        )
    b.global_store(dst, tid, b.add(b.mul(v, b.const_i32(2)), b.const_i32(1)))
    b.ret()
    return b.kernel


def _lane0_store(b: IRBuilder, out, idx, v64) -> None:
    with b.scf_if(b.cmp_eq(b.thread_id_x(), b.const_i32(0))):
        b.store(v64, out, idx)


@dataclass(frozen=True)
class PingPongSpec:
    """CTA ``a`` and CTA ``bb`` bounce one mailbox hand-off ``iters`` times.

    ``pairs`` 1 (``put``, 8 B) or 2 (``put2``, 16 B). One wave per CTA; the
    whole wave polls (same address), lane 0 stores.
    """

    knobs: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    pairs: int = 1

    def name(self) -> str:
        return _name("mbx_pingpong", self.knobs, f"p{self.pairs}")


def build_pingpong(spec: PingPongSpec) -> KernelDef:
    """Params: ``box`` i32* (>= 64 B, or 16 KiB 16 KiB-aligned with
    ``numa_copies=2``; ping at pair 0, pong at pair 2),
    ``out`` i64[PP_COLS * grid], ``a``, ``bb`` (CTA ids), ``base``, ``iters``.

    ``out`` row of A: [t0, t1, -, last pong payload, xcc, hw_id]; of B:
    [-, -, t_first_ping_seen, last ping payload, xcc, hw_id].
    A: t0, then ``iters`` x (put ping it, poll pong it), t1.
    B: ``iters`` x (poll ping it, put pong ping + 1000).
    """
    kn = spec.knobs
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = 64
    box = b.param("box", PtrType(I32, "global"))
    out = b.param("out", PtrType(I64, "global"))
    a_id = b.readfirstlane(b.param("a", I32))
    b_id = b.readfirstlane(b.param("bb", I32))
    base = b.readfirstlane(b.param("base", I32))
    iters = b.readfirstlane(b.param("iters", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    numa = kn.numa_copies == 2
    mb = M.mailbox_from_ptr(b, box, b.const_i32(16384 if numa else 64), kn)
    # numa_copies=2: puts write both copies, polls read this CTA's IOD-local
    # copy (uncalibrated: flip 0)
    mbp = M.local_copy(b, mb, 0) if numa else mb
    ping, pong = 0, 2
    one = b.const_i32(1)
    lane0 = b.cmp_eq(tid, b.const_i32(0))
    row = b.mul(bid, b.const_i32(PP_COLS))

    def put(i, val, tag):
        with b.scf_if(lane0):
            if spec.pairs == 1:
                M.put(b, mb, i, val, tag, kn)
            else:
                M.put2(b, mb, i, val, val, tag, kn)

    def poll(i, tag):
        return M.poll(b, [(mbp, i, spec.pairs)], tag, kn)[0][-1]

    def wr(col, v64):
        _lane0_store(b, out, b.add(row, b.const_i32(col)), v64)

    with b.scf_if(b.cmp_eq(bid, a_id)):
        t0 = D.memrealtime(b)
        loop = b.scf_for_iter(
            one, b.add(iters, one), one, [("a_last", b.const_i32(0))], iv_name="a_it"
        )
        with loop as (it, _acc):
            tag = b.add(base, it)
            put(ping, it, tag)
            b.scf_yield(poll(pong, tag))
        t1 = D.memrealtime(b)
        wr(0, t0)
        wr(1, t1)
        wr(3, b.zext(loop.results[0], I64))
        wr(4, b.zext(D.xcc_id(b), I64))
        wr(5, b.zext(D.hw_id(b), I64))
    with b.scf_if(b.cmp_eq(bid, b_id)):
        tag1 = b.add(base, one)
        first = poll(ping, tag1)
        t_seen = D.memrealtime(b)
        put(pong, b.add(first, b.const_i32(1000)), tag1)
        loop = b.scf_for_iter(
            b.const_i32(2),
            b.add(iters, one),
            one,
            [("b_last", first)],
            iv_name="b_it",
        )
        with loop as (it, _acc):
            tag = b.add(base, it)
            got = poll(ping, tag)
            put(pong, b.add(got, b.const_i32(1000)), tag)
            b.scf_yield(got)
        wr(2, t_seen)
        wr(3, b.zext(loop.results[0], I64))
        wr(4, b.zext(D.xcc_id(b), I64))
        wr(5, b.zext(D.hw_id(b), I64))
    b.ret()
    return b.kernel


@dataclass(frozen=True)
class AllToAllSpec:
    """``phases`` rounds; in each, thread 0 of every CTA ``< producers``
    publishes one pair and every CTA of the grid consumes all of them, then
    barriers (``s_barrier``, no vmcnt drain) and starts the next round.

    consume (who polls what; every mode reads every producer's pair):
      ``"wave0"``   wave 0, lane l polls pairs l, l+64, ... (8 B loads) -- the
                    ``pre_poll`` throttle shape;
      ``"wave0x4"`` wave 0, lane l polls 16 B chunks l, l+64, ... (two adjacent
                    producers' pairs per load, both tags checked);
      ``"waves"``   wave w polls pair l + 64 w (one 8 B load per lane, one wave
                    per 64 producers);
      ``"all"``     every wave polls every pair (the unthrottled flood).
    ``check`` stores each round's per-thread payload sum (correctness runs;
    off for timing). Pairs are never reused across rounds.

    ``barrier`` ends each round: ``"bare"`` (default)
    ``s_barrier_bare(fence="compiler")``, no ``vmcnt`` drain; ``"sync"``
    ``b.sync()`` (``s_waitcnt`` drain + ``s_barrier``), the analogue of the
    ``gpu.barrier()`` the FlyDSL reference all-to-all uses, for
    like-for-like comparisons.

    ``knobs.numa_copies`` 2: every round is a 16 KiB block (``box`` 16 KiB
    aligned) holding the two copies 8 KiB apart; producers publish both
    (``mailbox.put``) and a consumer polls ``mailbox.local_copy(flip[round])``.
    The host calibrates ``flip`` per 2 MiB page
    (:func:`harness.calibrate_iod_home`); all-zero is the uncalibrated variant.
    """

    knobs: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    producers: int = 160
    threads: int = 512
    consume: str = "wave0"
    check: bool = False
    barrier: str = "bare"

    @property
    def numa_copies(self) -> int:
        return self.knobs.numa_copies

    @property
    def round_pairs(self) -> int:
        """Pair-index stride of one round in ``box``."""
        return 2048 if self.numa_copies == 2 else self.producers

    def name(self) -> str:
        extra = [f"p{self.producers}", f"t{self.threads}", self.consume]
        if self.check:
            extra.append("chk")
        if self.barrier != "bare":
            extra.append(f"bar{self.barrier}")
        return _name("mbx_a2a", self.knobs, *extra)


A2A_CONSUME = ("wave0", "wave0x4", "waves", "all")
A2A_BARRIERS = ("bare", "sync")


def a2a_expected_thread_sum(spec: AllToAllSpec, phase: int, tid: int) -> int:
    """Host model of one thread's payload sum in round ``phase`` (0 for a
    thread that does not poll). Payload of producer m = m + 7 * phase."""
    P = spec.producers
    lane, wave = tid % 64, tid // 64

    def pay(m):
        return m + 7 * phase

    if spec.consume in ("wave0", "all"):
        if spec.consume == "wave0" and wave:
            return 0
        return sum(pay(min(lane + 64 * r, P - 1)) for r in range((P + 63) // 64))
    if spec.consume == "wave0x4":
        if wave:
            return 0
        chunks = P // 2
        tot = 0
        for r in range((chunks + 63) // 64):
            c = min(lane + 64 * r, chunks - 1)
            tot += pay(2 * c) + pay(2 * c + 1)
        return tot
    # waves
    if wave >= (P + 63) // 64:
        return 0
    return pay(min(lane + 64 * wave, P - 1))


def build_all_to_all(spec: AllToAllSpec) -> KernelDef:
    """Params: ``box`` i32* (phases x producers pairs), ``out`` i64[A2A_COLS *
    grid], ``chk`` i32[phases * grid * threads], [``flip`` i32[phases] when
    ``numa_copies == 2``], ``base``, ``phases``, ``grid``.

    ``out`` row per CTA: [t_start, t_end, xcc, 0]. ``chk[(ph * grid + bid) *
    threads + tid]`` = that round's payload sum of the thread (see
    :func:`a2a_expected_thread_sum`)."""
    kn = spec.knobs
    P = spec.producers
    if spec.consume not in A2A_CONSUME:
        raise ValueError(spec.consume)
    if spec.barrier not in A2A_BARRIERS:
        raise ValueError(spec.barrier)
    if spec.consume == "wave0x4" and P % 2:
        raise ValueError("wave0x4 needs an even producer count")
    if spec.consume == "waves" and spec.threads < 64 * ((P + 63) // 64):
        raise ValueError("waves needs one wave per 64 producers")
    if spec.numa_copies not in (1, 2) or (spec.numa_copies == 2 and P > 1024):
        raise ValueError("numa_copies must be 1, or 2 with <= 1024 producers")
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    box = b.param("box", PtrType(I32, "global"))
    out = b.param("out", PtrType(I64, "global"))
    chk = b.param("chk", PtrType(I32, "global"))
    flip = b.param("flip", PtrType(I32, "global")) if spec.numa_copies == 2 else None
    base = b.readfirstlane(b.param("base", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    grid = b.readfirstlane(b.param("grid", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lane = b.lane_id()
    one = b.const_i32(1)
    mb = M.mailbox_from_ptr(b, box, b.const_i32(0x7FFFFFF0), kn)
    tid0 = b.cmp_eq(tid, b.const_i32(0))
    is_prod = b.cmp_lt(bid, b.const_i32(P))

    def clamp(x, hi):
        last = b.const_i32(hi)
        return b.select(b.cmp_lt(x, last), x, last)

    def consume(ph, first, tag, mb):
        if spec.consume in ("wave0", "all"):
            specs = [
                (mb, b.add(first, clamp(b.add(lane, b.const_i32(64 * r)), P - 1)), 1)
                for r in range((P + 63) // 64)
            ]
        elif spec.consume == "wave0x4":
            chunks = P // 2
            specs = [
                (
                    mb,
                    b.add(
                        first,
                        b.mul(
                            clamp(b.add(lane, b.const_i32(64 * r)), chunks - 1),
                            b.const_i32(2),
                        ),
                    ),
                    2,
                )
                for r in range((chunks + 63) // 64)
            ]
        else:
            w = D.wave_id(b)
            specs = [
                (
                    mb,
                    b.add(first, clamp(b.add(lane, b.mul(w, b.const_i32(64))), P - 1)),
                    1,
                )
            ]
        vals = M.poll(b, specs, tag, kn)
        if spec.check:
            flat = [v for vs in vals for v in vs]
            s = flat[0]
            for v in flat[1:]:
                s = b.add(s, v)
            idx = b.add(
                b.mul(b.add(b.mul(ph, grid), bid), b.const_i32(spec.threads)), tid
            )
            b.global_store(chk, idx, s)

    b.s_barrier_bare(fence="compiler")
    t0 = D.memrealtime(b)
    with b.scf_for(b.const_i32(0), phases, one, iv_name="ph") as ph:
        tag = b.add(base, b.add(ph, one))
        first = b.mul(ph, b.const_i32(spec.round_pairs))
        with b.scf_if(b.land(is_prod, tid0)):
            val = b.add(bid, b.mul(ph, b.const_i32(7)))
            M.put(b, mb, b.add(first, bid), val, tag, kn)  # both copies if numa
        mbc = mb
        if spec.numa_copies == 2:
            mbc = M.local_copy(b, mb, b.readfirstlane(b.global_load_i32(flip, ph)))
        if spec.consume == "all":
            consume(ph, first, tag, mbc)
        elif spec.consume == "waves":
            nw = (P + 63) // 64
            with b.scf_if(b.cmp_lt(D.wave_id(b), b.const_i32(nw))):
                consume(ph, first, tag, mbc)
        else:
            with b.scf_if(D.is_wave0(b)):
                consume(ph, first, tag, mbc)
        if spec.barrier == "sync":
            b.sync()
        else:
            b.s_barrier_bare(fence="compiler")
    t1 = D.memrealtime(b)
    row = b.mul(bid, b.const_i32(A2A_COLS))
    _lane0_store(b, out, row, t0)
    _lane0_store(b, out, b.add(row, one), t1)
    _lane0_store(b, out, b.add(row, b.const_i32(2)), b.zext(D.xcc_id(b), I64))
    b.ret()
    return b.kernel


def build_stamp_kernel(threads: int = 512) -> KernelDef:
    """Launch-gap probe. Thread 0 of each CTA stamps entry, then (after a CTA
    barrier) exit into ``out[(launch * grid + bid) * 2 + {0, 1}]``.
    Params: ``out`` i64*, ``launch`` i32, ``grid`` i32."""
    b = IRBuilder(f"mbx_stamp_t{threads}")
    b.kernel.attrs["max_workgroup_size"] = threads
    out = b.param("out", PtrType(I64, "global"))
    launch = b.readfirstlane(b.param("launch", I32))
    grid = b.readfirstlane(b.param("grid", I32))
    bid = b.readfirstlane(b.block_id_x())
    t0 = D.memrealtime(b)
    b.s_barrier_bare(fence="compiler")
    t1 = D.memrealtime(b)
    row = b.mul(b.add(b.mul(launch, grid), bid), b.const_i32(2))
    _lane0_store(b, out, row, t0)
    _lane0_store(b, out, b.add(row, b.const_i32(1)), t1)
    b.ret()
    return b.kernel


def build_tag_wrap_kernel(knobs: M.MailboxKnobs = M.DEFAULT_KNOBS) -> KernelDef:
    """One "layer launch" with the real tag rule. Params: ``step`` i32* (the
    device step counter), ``box`` i32* (persists across launches, never
    cleared), ``out`` i32[4], ``layer`` i32, ``delay`` i32 (s_sleep units the
    producer waits before publishing).

    CTA 0 (producer): sleeps ``delay`` x 64 clocks x 127, then
    ``put(pair 0, step * 3 + layer, tag)``. CTA 1 (consumer): polls pair 0 for
    ``tag`` and writes [payload, tag, step, layer] to ``out``. A consumer that
    matched a stale tag returns the previous launch's payload."""
    b = IRBuilder(_name("mbx_tagwrap", knobs))
    b.kernel.attrs["max_workgroup_size"] = 64
    step_p = b.param("step", PtrType(I32, "global"))
    box = b.param("box", PtrType(I32, "global"))
    out = b.param("out", PtrType(I32, "global"))
    layer = b.readfirstlane(b.param("layer", I32))
    delay = b.readfirstlane(b.param("delay", I32))
    step = b.readfirstlane(b.global_load_i32(step_p, b.const_i32(0)))
    tag = M.launch_tag(b, step, layer)
    mb = M.mailbox_from_ptr(b, box, b.const_i32(64), knobs)
    bid = b.readfirstlane(b.block_id_x())
    lane0 = b.cmp_eq(b.thread_id_x(), b.const_i32(0))
    with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
        with b.scf_for(b.const_i32(0), delay, b.const_i32(1), iv_name="tw_d"):
            D.s_sleep(b, 127)
        with b.scf_if(lane0):
            val = b.add(b.mul(step, b.const_i32(3)), layer)
            M.put(b, mb, 0, val, tag, knobs)
    with b.scf_if(b.cmp_eq(bid, b.const_i32(1))):
        got = M.get(b, mb, 0, tag, knobs)
        with b.scf_if(lane0):
            for k, v in enumerate((got, tag, step, layer)):
                b.global_store(out, b.const_i32(k), v)
    b.ret()
    return b.kernel


# ------------------------------------------------- bulk hand-off + bf16


BULK_ROW = 1024
BULK_FENCES = ("helper", "norelease")


@dataclass(frozen=True)
class BulkSpec:
    """``release_bulk`` / ``acquire_bulk`` + ``put_bf`` + ``payload_wait`` +
    ``atomic_add_scoped`` across CTAs (on different XCDs for most shifts).

    Each of ``grid`` CTAs (``threads`` threads) plain-stores a
    ``BULK_ROW``-int row ``data[p][k] = step * 7919 + p * BULK_ROW + k``,
    releases it with ready pair ``p`` and publishes four f32 values
    ``(step * 13 + 4 p + j) * 0.3701`` as two packed bf16 pairs; it then
    acquires CTA ``q = (p + shift) % grid``'s row, counts mismatching words
    (stale data from an earlier step shows up as mismatches), waits with
    ``payload_wait(rows)`` and decodes q's bf16 pairs. Counters (all through
    ``device_ops.atomic_add_scoped``): ``cnt[0] += mismatches`` (agent,
    no return), ``cnt[1] += 1`` (agent, no return), ``cnt[2] += 1``
    (system, returning; the returned ticket is stored per thread).

    ``fence``: ``"helper"`` (the mailbox helpers) or ``"norelease"`` (a
    mutation: the producer skips the L2 write-back before its ready pair; a
    consumer on another XCD may then read stale rows).
    """

    knobs: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    grid: int = 256
    threads: int = 256
    rows: int = 1
    fence: str = "helper"

    def name(self) -> str:
        extra = [f"g{self.grid}", f"t{self.threads}", f"r{self.rows}"]
        if self.fence != "helper":
            extra.append(self.fence)
        return _name("mbx_bulk", self.knobs, *extra)


def build_bulk_kernel(spec: BulkSpec) -> KernelDef:
    """Params: ``data`` i32[grid * BULK_ROW], ``box`` i32* (ready pairs,
    >= grid * 8 B), ``bfbox`` i32* (>= grid * 16 B), ``cnt`` i32[4],
    ``tickets`` i32[grid * threads], ``out`` f32[grid * 4], ``step``,
    ``shift``. ``grid`` must be a power of two."""
    kn = spec.knobs
    G, T = spec.grid, spec.threads
    if G & (G - 1) or BULK_ROW % T:
        raise ValueError("grid must be a power of two and threads divide BULK_ROW")
    if spec.fence not in BULK_FENCES:
        raise ValueError(spec.fence)
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    data = b.param("data", PtrType(I32, "global"))
    box = b.param("box", PtrType(I32, "global"))
    bfbox = b.param("bfbox", PtrType(I32, "global"))
    cnt = b.param("cnt", PtrType(I32, "global"))
    tickets = b.param("tickets", PtrType(I32, "global"))
    out = b.param("out", PtrType(F32, "global"))
    step = b.readfirstlane(b.param("step", I32))
    shift = b.readfirstlane(b.param("shift", I32))
    tag = M.launch_tag(b, step, 5)
    mb = M.mailbox_from_ptr(b, box, b.const_i32(G * 8), kn)
    bmb = M.mailbox_from_ptr(b, bfbox, b.const_i32(G * 16), kn)
    cnt_a = b.readfirstlane(b.global_ptr_to_i64(cnt))
    p = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    tid0 = b.cmp_eq(tid, b.const_i32(0))
    base = b.mul(step, b.const_i32(7919))
    for k in range(BULK_ROW // T):
        idx = b.add(b.mul(p, b.const_i32(BULK_ROW)), b.add(b.const_i32(k * T), tid))
        b.global_store(data, idx, b.add(base, idx))
    if spec.fence == "helper":
        M.release_bulk(b, mb, p, tag, kn)
    else:
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(tid0):
            M.put(b, mb, p, b.const_i32(0), tag, kn)
    with b.scf_if(tid0):
        vs = [
            b.fmul(
                b.sitofp_f32(
                    b.add(
                        b.mul(step, b.const_i32(13)),
                        b.add(b.mul(p, b.const_i32(4)), b.const_i32(j)),
                    )
                ),
                b.const_f32(0.3701),
            )
            for j in range(4)
        ]
        M.put_bf(b, bmb, b.mul(p, b.const_i32(4)), vs, tag, kn)
    q = b.readfirstlane(b.land(b.add(p, shift), b.const_i32(G - 1)))
    M.acquire_bulk(b, mb, q, tag, kn)
    bad = b.const_i32(0)
    for k in range(BULK_ROW // T):
        idx = b.add(b.mul(q, b.const_i32(BULK_ROW)), b.add(b.const_i32(k * T), tid))
        v = b.global_load_i32(data, idx)
        bad = b.add(
            bad,
            b.select(b.cmp_ne(v, b.add(base, idx)), b.const_i32(1), b.const_i32(0)),
        )
    D.atomic_add_scoped(b, cnt_a, b.const_i32(0), bad, scope="agent")
    D.atomic_add_scoped(b, cnt_a, b.const_i32(4), b.const_i32(1), scope="agent")
    ticket = D.atomic_add_scoped(
        b, cnt_a, b.const_i32(8), b.const_i32(1), scope="system", returns=True
    )
    b.global_store(tickets, b.add(b.mul(p, b.const_i32(T)), tid), ticket)
    # wait for every CTA's bf16 pairs (hint = its ready pair): rows <= 4 only
    # barriers, wider builds pre-poll all ``G`` ready pairs first
    M.payload_wait(b, lambda m: (mb, m), G, tag, spec.rows, kn)
    with b.scf_if(tid0):
        (w,) = M.poll(b, [(bmb, b.mul(q, b.const_i32(2)), 2)], tag, kn)
        for j, word in enumerate(w):
            lo, hi = M.bf2_f32(b, word)
            o = b.add(b.mul(p, b.const_i32(4)), b.const_i32(2 * j))
            b.global_store(out, o, lo)
            b.global_store(out, b.add(o, b.const_i32(1)), hi)
    b.ret()
    return b.kernel


# ------------------------------------------------------- device-op probe


DOPS_COLS = 8
DOPS_SLEEP = 200


def build_device_ops_kernel() -> KernelDef:
    """One wave per CTA. Params: ``inp`` i32[grid * 64], ``echo``
    i32[grid * 64 * 2], ``out`` i64[grid * DOPS_COLS].

    ``echo[(bid * 64 + lane) * 2 + {0, 1}]`` = ``fresh_v(x)``,
    ``fresh_s(readfirstlane(x))``. ``out`` row: [wave_umax_dpp(x),
    memtime delta across ``s_sleep(DOPS_SLEEP)``, memrealtime delta across
    the same, xcc_id, hw_id, 0, 0, 0]; ``buffer_wbl2`` and ``buffer_inv``
    (agent and system) run once in between."""
    b = IRBuilder("mbx_device_ops")
    b.kernel.attrs["max_workgroup_size"] = 64
    inp = b.param("inp", PtrType(I32, "global"))
    echo = b.param("echo", PtrType(I32, "global"))
    out = b.param("out", PtrType(I64, "global"))
    bid = b.readfirstlane(b.block_id_x())
    lane = b.thread_id_x()
    gi = b.add(b.mul(bid, b.const_i32(64)), lane)
    x = b.global_load_i32(inp, gi)
    mx = D.wave_umax_dpp(b, x)
    e0 = b.mul(gi, b.const_i32(2))
    b.global_store(echo, e0, D.fresh_v(b, x))
    b.global_store(echo, b.add(e0, b.const_i32(1)), D.fresh_s(b, b.readfirstlane(x)))
    t0 = D.memtime(b)
    r0 = D.memrealtime(b)
    D.s_sleep(b, DOPS_SLEEP)
    t1 = D.memtime(b)
    r1 = D.memrealtime(b)
    for scope in ("agent", "system"):
        D.buffer_wbl2(b, scope)
        D.buffer_inv(b, scope)
    row = b.mul(bid, b.const_i32(DOPS_COLS))
    cols = (
        b.zext(mx, I64),
        b.sub(t1, t0),
        b.sub(r1, r0),
        b.zext(D.xcc_id(b), I64),
        b.zext(D.hw_id(b), I64),
    )
    for k, v in enumerate(cols):
        _lane0_store(b, out, b.add(row, b.const_i32(k)), v)
    b.ret()
    return b.kernel
