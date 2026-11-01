# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P4: synchronization primitives of a persistent decode kernel (gfx950).

``experiments/qwen38-true-mono/DESIGN.md`` section 8.2, P4: striped grid sync,
residency ticket and per-XCD counters vs the NUMA tagged-pair all-to-all, at 160
and 256 producers, idle and skewed arrivals, plus a payload-carrying ``xl`` hop
(the layer output handed to the next layer's readers) at 1 / 4 / 16 rows against
the graph-launch boundary it would replace. Negative controls: an oversize grid
must time out, a dropped arrival must not release, mutants must fail.

Kernels (``grid`` x ``threads``, default 256 x 512; an LDS pad above half a CU's
LDS keeps one CTA per CU, as FlyTape's residency rule):

* :func:`build_counter_kernel` (``ctr``): counter barriers. Producers (a host
  table names them) ``global_atomic_add`` one counter word, every CTA's wave 0
  polls up to 64 words lane-parallel against ``epoch * count - bias`` (a ballot
  over ``v_cmp``, FlyTape's ``grid_wait``). The word layout is host data, so ONE
  binary covers one device word, one word per XCD (the per-XCD counter) and 2 /
  4 / 8 stripes per XCD (FlyTape's striped grid sync), each word IOD-local or
  IOD-remote, any producer set, the early-release mutant (``bias`` 1) and the
  dropped-arrival control. Each producer stores a data word (``sc1``) before it
  arrives; after release every thread ``< grid`` reads one producer's word back
  (``sc1``) and counts mismatches: the timed binary checks its own release.
* :func:`build_a2a_kernel` (``a2a``): the tagged-pair all-to-all of the
  mailbox layer (one pair per producer per epoch, consume ``waves``), with
  ``rocke.helpers.mailbox`` polls (``asm`` or the family's ``ir``) or a
  deadline poll (``poll="deadline"``: the asm loop with an iteration bound, so
  a dropped producer times out instead of hanging). Payloads are checked.
* :func:`build_ticket_kernel` (``tk``): FlyTape's residency ticket (per-XCD
  rank, start count, bounded wait, XCD-contiguous worker ids), runtime on/off.
* :func:`build_xl_kernel` (``xl``): producers ``0 .. producers-1`` (M5's 160
  tasks, 16 hidden columns x 4 HC streams x ``rows`` bf16 each) publish their
  slice, ``ncons`` consumer CTAs (A1-like readers of the whole layer input)
  read all of it. ``mode="pairs"``: tagged bf16 pairs (the payload is its own
  flag; NUMA dual copies optional). ``mode="bulk"``: plain stores + one ready
  pair per producer (``release_bulk``), consumers poll the ready pairs,
  ``buffer_inv sc1``, plain loads (``xl_late=bulk`` / ``xl=fence`` shape).
  Epochs are separated by an untimed closing counter sync.
* :func:`build_boundary_kernel` (``bd``): the reference: one launch per epoch
  (captured in a HIP graph by the host): consumers plain-load the previous
  launch's payload, producers plain-store this launch's.

Stamps: thread 0 of each CTA records ``s_memrealtime`` (100 MHz, one clock for
every XCD) into ``out[(epoch * grid + cta) * OUT_COLS + col]``: col 0 arrival /
publish start, 1 release / consumer done, 2 status, 3 poll iterations. CTA 0
records ``s_memtime`` / ``s_memrealtime`` at its start and end (``info``) so the
host derives the shader clock of every launch.

Python engine only by policy (the ``ir`` mailbox opcodes); kernel names start
with ``q38p4_`` (the family README's cpp-parity exemption). Nothing here is
imported by a served build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "A2ASpec",
    "BoundarySpec",
    "CounterSpec",
    "ERR_MISMATCH",
    "ERR_TIMEOUT",
    "GRID",
    "LINE",
    "OUT_COLS",
    "ST_ABORTED",
    "ST_OK",
    "ST_TIMEOUT",
    "TAB_ARR",
    "THREADS",
    "TicketSpec",
    "XlSpec",
    "build_a2a_kernel",
    "build_boundary_kernel",
    "build_counter_kernel",
    "build_ticket_kernel",
    "build_xl_kernel",
    "info_words",
    "numa_phys_pair",
    "payload_word",
    "poll_counter",
    "poll_pair_deadline",
    "tab_words",
    "xl_chunks_per_thread",
]

GRID = 256
THREADS = 512
LINE = 256  # bytes between counter words (one per 256-byte line, FlyTape SYNC_WORD_STRIDE)
OUT_COLS = 4
ST_OK = 1
ST_TIMEOUT = 2
ST_ABORTED = 3  # a CTA that timed out earlier neither arrives nor waits again (FlyTape ABORTED)
ERR_MISMATCH = 0  # err[] word indices
ERR_TIMEOUT = 1
# counter table layout (i32 words): poll byte offsets [0, 64), counts [64, 128),
# bias [128, 192), per-CTA arrival byte offset (-1 = not a producer) [192, 192 + G),
# per-CTA dropped epoch (-1 = none) [192 + G, 192 + 2 G)
TAB_POFF, TAB_CNT, TAB_BIAS, TAB_ARR = 0, 64, 128, 192


def tab_words(grid: int) -> int:
    return TAB_ARR + 2 * grid


def info_words(grid: int) -> int:
    """i64 words of ``info``: [2 cta] = xcc, [2 cta + 1] = HW_ID, then CTA 0's
    memtime / memrealtime at start and end."""
    return 2 * grid + 4


def payload_word(b: IRBuilder, salt, e, w):
    """``salt * 1000003 + e * 65599 + w * 7 + 1`` (i32, wrapping): the value of
    payload / data word ``w`` in epoch ``e``. The host twin is in the bench."""
    v = b.add(b.mul(salt, b.const_i32(1000003)), b.mul(e, b.const_i32(65599)))
    return b.add(v, b.add(b.mul(w, b.const_i32(7)), b.const_i32(1)))


def _lds_write(b: IRBuilder, addr, v) -> None:
    """``ds_write_b32`` + ``lgkmcnt(0)`` (complete before a later barrier)."""
    b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, v], None)


def _lds_read(b: IRBuilder, addr):
    """Uniform ``ds_read_b32`` of a CTA control word."""
    v = b.inline_asm(
        "ds_read_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "=v,v,~{memory}", [addr], I32,
        result_name_hint="ldsw",
    )  # fmt: skip
    return b.readfirstlane(v)


def _pad_lds(b: IRBuilder):
    """One CTA per CU: allocate more than half a CU's LDS, touched through asm
    so the lowering keeps it (a plain LDS store that is never read is dropped
    with its allocation: seen, group segment 0). Words 0 and 1 are CTA control
    words (abort flag, epoch-tagged timeout), zeroed here; the caller barriers
    before using them. Returns the LDS byte address of word 0."""
    lds = b.smem_alloc(I32, [ONE_CTA_PER_CU_WORDS], name_hint="p4pad")
    w0 = b.trunc(b.smem_addr_of(lds), I32)
    with b.scf_if(D.is_thread0(b)):
        zero = b.const_i32(0)
        _lds_write(b, w0, zero)
        _lds_write(b, b.add(w0, b.const_i32(4)), zero)
        _lds_write(b, b.add(w0, b.const_i32(4 * (ONE_CTA_PER_CU_WORDS - 1))), zero)
    return w0


def _info_start(b: IRBuilder, info, bid, grid: int) -> None:
    with b.scf_if(D.is_thread0(b)):
        row = b.mul(bid, b.const_i32(2))
        b.store(b.zext(D.xcc_id(b), I64), info, row)
        b.store(b.zext(D.hw_id(b), I64), info, b.add(row, b.const_i32(1)))
        with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
            b.store(D.memtime(b), info, b.const_i32(2 * grid))
            b.store(D.memrealtime(b), info, b.const_i32(2 * grid + 1))


def _info_end(b: IRBuilder, info, bid, grid: int) -> None:
    with b.scf_if(b.land(D.is_thread0(b), b.cmp_eq(bid, b.const_i32(0)))):
        b.store(D.memtime(b), info, b.const_i32(2 * grid + 2))
        b.store(D.memrealtime(b), info, b.const_i32(2 * grid + 3))


def _delay(b: IRBuilder, d) -> None:
    """``d`` (uniform i32) x ``s_sleep 1`` (64 clocks each): the skew."""
    with b.scf_for(b.const_i32(0), d, b.const_i32(1), iv_name="p4d"):
        D.s_sleep(b, 1)


def _st_sc1(b: IRBuilder, base, voff, v) -> None:
    """``global_store_dword ... sc1`` (write-through, device scope)."""
    b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s", [voff, v, base], None)


def _ld_sc1(b: IRBuilder, base, voff):
    """``global_load_dword ... sc1`` + ``vmcnt(0)``."""
    return b.inline_asm(
        "global_load_dword $0, $1, $2 sc1\ns_waitcnt vmcnt(0)",
        "=&v,v,s",
        [voff, base],
        I32,
        result_name_hint="ld1",
    )


def _stamp(b: IRBuilder, out, row, col: int) -> None:
    b.store(D.memrealtime(b), out, b.add(row, b.const_i32(col)))


def poll_counter(b: IRBuilder, voff, base, expected, maxit, pause: str = "s_nop"):
    """Spin until every active lane's counter word (``sc1`` load at byte
    ``voff`` of ``base``) is ``>= expected`` (unsigned; a lane with
    ``expected == 0`` is satisfied at once), or ``maxit`` retries ran.
    Returns ``(seen, iterations)``; ``iterations >= maxit`` means timed out.
    The lanes' conditions combine as a ballot (``v_cmp`` + ``exec``), one load
    round trip per retry for all lanes (FlyTape ``grid_wait``)."""
    top = "${:private}p4c${:uid}_top"
    done = "${:private}p4c${:uid}_done"
    lines = [
        "s_mov_b32 $2, 0",
        f"{top}:",
        "global_load_dword $0, $3, $4 sc1",
        "s_waitcnt vmcnt(0)",
        "v_cmp_lt_u32_e64 $1, $0, $5",
        "s_and_b64 $1, $1, exec",
        f"s_cbranch_scc0 {done}",
        "s_add_u32 $2, $2, 1",
        "s_cmp_ge_u32 $2, $6",
        f"s_cbranch_scc1 {done}",
        D.PAUSES[pause],
        f"s_branch {top}",
        f"{done}:",
    ]
    seen, _mask, it = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,v,s,v,s,~{vcc},~{scc}",
        [voff, base, expected, maxit],
        result_types=[I32, I64, I32],
        result_name_hint="p4c",
    )
    return seen, it


def poll_pair_deadline(b: IRBuilder, voff, base, tag, maxit, pause: str = "s_nop"):
    """The mailbox asm poll of one ``(payload, tag)`` pair per lane with an
    iteration bound. Returns ``(payload, iterations)``."""
    top = "${:private}p4p${:uid}_top"
    done = "${:private}p4p${:uid}_done"
    lines = [
        "s_mov_b32 $3, 0",
        f"{top}:",
        "global_load_dwordx2 $0, $4, $5 sc1",
        "s_waitcnt vmcnt(0)",
        "v_lshrrev_b64 $1, 32, $0",
        "v_cmp_ne_u64_e64 $2, $1, $6",
        "s_and_b64 $2, $2, exec",
        f"s_cbranch_scc0 {done}",
        "s_add_u32 $3, $3, 1",
        "s_cmp_ge_u32 $3, $7",
        f"s_cbranch_scc1 {done}",
        D.PAUSES[pause],
        f"s_branch {top}",
        f"{done}:",
    ]
    pair, _tmp, _mask, it = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&v,=&s,=&s,v,s,s,s,~{vcc},~{scc}",
        [voff, base, b.zext(tag, I64), maxit],
        result_types=[VectorType(I32, 2), I64, I64, I32],
        result_name_hint="p4p",
    )
    return b.vec_extract(pair, 0), it


def _counter_sync(b: IRBuilder, sync_a, aoff, arrive, poff, pcnt, pbias, epoch1, maxit, err_a, pause):
    """Arrive (thread 0, when ``arrive``) and wait (wave 0) on the counter
    words; ``epoch1`` = 1-based epoch. Records a timeout in ``err``. Returns
    nothing; the caller barriers."""
    tid0 = D.is_thread0(b)
    with b.scf_if(b.land(arrive, tid0)):
        D.atomic_add_scoped(b, sync_a, aoff, b.const_i32(1), scope="agent")
    with b.scf_if(D.is_wave0(b)):
        exp = b.select(
            b.cmp_gt(pcnt, b.const_i32(0)),
            b.sub(b.mul(epoch1, pcnt), pbias),
            b.const_i32(0),
        )
        _seen, it = poll_counter(b, poff, sync_a, exp, maxit, pause)
        with b.scf_if(b.land(tid0, b.cmp_ge(it, maxit))):
            D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))


# ------------------------------------------------------------------ counters


@dataclass(frozen=True)
class CounterSpec:
    grid: int = GRID
    threads: int = THREADS
    pause: str = "s_nop"
    tag: str = ""  # name suffix of a separately compiled control copy ("rc")

    def name(self) -> str:
        parts = ["q38p4_ctr", f"g{self.grid}", f"t{self.threads}"]
        if self.pause != "s_nop":
            parts.append(f"p{self.pause}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def build_counter_kernel(spec: CounterSpec = CounterSpec()) -> KernelDef:
    """Params: ``sync`` i32* (counter words; zeroed by the host per launch),
    ``tab`` i32[tab_words(grid)], ``dtab`` i32[phases * grid] (per-epoch
    pre-arrival delay of each CTA, ``s_sleep 1`` units), ``out``
    i64[phases * grid * OUT_COLS], ``dat`` i32[phases * grid] (data words),
    ``err`` i32[4], ``info`` i64[info_words(grid)], ``salt``, ``phases``,
    ``maxit``. Producer p of epoch e stores ``dat[e * grid + p] =
    payload_word(salt, e, p)`` before it arrives (unless ``e`` is its dropped
    epoch: then it neither stores nor arrives)."""
    G, T = spec.grid, spec.threads
    if G > 512 or T < 64 or T % 64:
        raise ValueError("grid <= 512, threads a multiple of 64")
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
    lane = b.lane_id()
    lds0 = _pad_lds(b)
    _info_start(b, info, bid, G)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    dat_a = b.readfirstlane(b.global_ptr_to_i64(dat))
    poff = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_POFF)))
    pcnt = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_CNT)))
    pbias = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_BIAS)))
    aoff = b.readfirstlane(b.global_load_i32(tab, b.add(bid, b.const_i32(TAB_ARR))))
    drop = b.readfirstlane(b.global_load_i32(tab, b.add(bid, b.const_i32(TAB_ARR + G))))
    is_prod = b.cmp_ge(aoff, b.const_i32(0))
    tchk = b.select(b.cmp_lt(tid, b.const_i32(G)), tid, b.const_i32(G - 1))
    tprod = b.global_load_i32(tab, b.add(tchk, b.const_i32(TAB_ARR)))
    chk = b.land(b.cmp_lt(tid, b.const_i32(G)), b.cmp_ge(tprod, b.const_i32(0)))
    tid0 = D.is_thread0(b)
    four = b.const_i32(4)
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        _delay(b, d)
        # a CTA that timed out stops arriving and waiting (cumulative counters:
        # its later arrivals would otherwise release CTAs still in the epoch
        # that timed out; seen on hardware)
        ok_cta = b.cmp_eq(_lds_read(b, lds0), b.const_i32(0))
        live = b.land(b.land(is_prod, b.cmp_ne(ph, drop)), ok_cta)
        with b.scf_if(b.land(live, tid0)):
            _st_sc1(b, dat_a, b.mul(b.add(eg, bid), four), payload_word(b, salt, ph, bid))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        with b.scf_if(b.land(live, tid0)):
            _stamp(b, out, row, 0)
            D.atomic_add_scoped(b, sync_a, aoff, b.const_i32(1), scope="agent")
        with b.scf_if(b.land(D.is_wave0(b), ok_cta)):
            exp = b.select(
                b.cmp_gt(pcnt, b.const_i32(0)),
                b.sub(b.mul(b.add(ph, b.const_i32(1)), pcnt), pbias),
                b.const_i32(0),
            )
            _seen, it = poll_counter(b, poff, sync_a, exp, maxit, spec.pause)
            timed = b.cmp_ge(it, maxit)
            with b.scf_if(tid0):
                st = b.select(timed, b.const_i32(ST_TIMEOUT), b.const_i32(ST_OK))
                b.store(b.zext(st, I64), out, b.add(row, b.const_i32(2)))
                b.store(b.zext(it, I64), out, b.add(row, b.const_i32(3)))
                with b.scf_if(timed):
                    _lds_write(b, lds0, b.const_i32(1))
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
        with b.scf_if(b.land(tid0, b.lnot(ok_cta))):
            b.store(b.const_i64(ST_ABORTED), out, b.add(row, b.const_i32(2)))
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(tid0):
            _stamp(b, out, row, 1)
        released = b.cmp_eq(_lds_read(b, lds0), b.const_i32(0))
        with b.scf_if(b.land(chk, released)):
            v = _ld_sc1(b, dat_a, b.mul(b.add(eg, tid), four))
            with b.scf_if(b.cmp_ne(v, payload_word(b, salt, ph, tid))):
                D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), b.const_i32(1))
    _info_end(b, info, bid, G)
    b.ret()
    return b.kernel


# ----------------------------------------------------------------------- a2a


A2A_POLLS = ("mailbox", "deadline")
A2A_MUTANTS = ("none", "skiplast")


@dataclass(frozen=True)
class A2ASpec:
    """Tagged-pair all-to-all: producers ``0 .. producers - 1`` put one pair
    per epoch (fresh slots per epoch), every CTA polls all of them with one
    wave per 64 producers (``waves``), then barriers.

    ``poll``: ``"mailbox"`` (``rocke.helpers.mailbox.poll``, impl from
    ``knobs``) or ``"deadline"`` (:func:`poll_pair_deadline`, asm, bounded).
    ``mutant="skiplast"``: consumers never wait for producer ``producers - 1``
    (one unpolled ``sc1`` load of its pair instead): must fail when it is late.
    """

    knobs: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    producers: int = 160
    grid: int = GRID
    threads: int = THREADS
    poll: str = "mailbox"
    mutant: str = "none"
    tag: str = ""

    @property
    def round_pairs(self) -> int:
        return 2048 if self.knobs.numa_copies == 2 else self.producers

    def name(self) -> str:
        parts = ["q38p4_a2a", f"p{self.producers}", f"g{self.grid}", f"t{self.threads}"]
        if self.poll != "mailbox":
            parts.append("dl")
        if self.mutant != "none":
            parts.append(f"mut{self.mutant}")
        sfx = self.knobs.suffix()
        if sfx:
            parts.append(sfx)
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def build_a2a_kernel(spec: A2ASpec) -> KernelDef:
    """Params: ``box`` i32* (phases x round_pairs pairs; 16 KiB aligned with
    NUMA copies), ``flip`` i32[phases], ``dtab``, ``out``, ``err`` i32[4],
    ``info``, ``base`` (tag of epoch e = base + e + 1), ``salt``, ``phases``,
    ``maxit`` (deadline poll), ``drop_cta``, ``drop_ep`` (that producer skips
    its put in that epoch; -1 = none). Payload of producer p in epoch e:
    ``payload_word(salt, e, p)``; every polled payload is checked."""
    kn = spec.knobs
    P, G, T = spec.producers, spec.grid, spec.threads
    if spec.poll not in A2A_POLLS or spec.mutant not in A2A_MUTANTS:
        raise ValueError((spec.poll, spec.mutant))
    nw = (P + 63) // 64
    if T < 64 * nw or P > G or (kn.numa_copies == 2 and P > 1024):
        raise ValueError("one wave per 64 producers; producers <= grid")
    if spec.poll == "deadline" and kn.impl != "asm":
        raise ValueError("the deadline poll is asm")
    numa = kn.numa_copies == 2
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    box = b.param("box", gp)
    flip = b.param("flip", gp)
    dtab = b.param("dtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    base = b.readfirstlane(b.param("base", I32))
    salt = b.readfirstlane(b.param("salt", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    drop_cta = b.readfirstlane(b.param("drop_cta", I32))
    drop_ep = b.readfirstlane(b.param("drop_ep", I32))
    bid = b.readfirstlane(b.block_id_x())
    lane = b.lane_id()
    lds0 = _pad_lds(b)
    lds1 = b.add(lds0, b.const_i32(4))
    _info_start(b, info, bid, G)
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    mb = M.mailbox_from_ptr(b, box, b.const_i32(0x7FFFFFF0), kn)
    tid0 = D.is_thread0(b)
    is_prod = b.cmp_lt(bid, b.const_i32(P))
    wave = D.wave_id(b)
    m = b.add(lane, b.mul(wave, b.const_i32(64)))
    last = P - 2 if spec.mutant == "skiplast" else P - 1
    mm = b.select(b.cmp_lt(m, b.const_i32(last)), m, b.const_i32(last))
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        _delay(b, d)
        tag = b.add(base, b.add(ph, b.const_i32(1)))
        first = b.mul(ph, b.const_i32(spec.round_pairs))
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        dropped = b.land(b.cmp_eq(bid, drop_cta), b.cmp_eq(ph, drop_ep))
        with b.scf_if(b.land(b.land(is_prod, b.lnot(dropped)), tid0)):
            _stamp(b, out, row, 0)
            M.put(b, mb, b.add(first, bid), payload_word(b, salt, ph, bid), tag, kn)
        mbc = M.local_copy(b, mb, b.readfirstlane(b.global_load_i32(flip, ph))) if numa else mb
        with b.scf_if(b.cmp_lt(wave, b.const_i32(nw))):
            idx = b.add(first, mm)
            if spec.poll == "mailbox":
                (vs,) = M.poll(b, [(mbc, idx, 1)], tag, kn)
                v = vs[0]
                it = None
            else:
                voff = M.pair_offset(b, idx)
                if mbc.copy_off is not None:
                    voff = b.add(voff, mbc.copy_off)
                v, it = poll_pair_deadline(b, voff, mbc.addr, tag, maxit, kn.pause)
            src = mm
            if spec.mutant == "skiplast":
                voff = M.pair_offset(b, b.add(first, b.const_i32(P - 1)))
                if mbc.copy_off is not None:
                    voff = b.add(voff, mbc.copy_off)
                raw = b.inline_asm(
                    "global_load_dwordx2 $0, $1, $2 sc1\ns_waitcnt vmcnt(0)",
                    "=&v,v,s",
                    [voff, mbc.addr],
                    VectorType(I32, 2),
                    result_name_hint="p4raw",
                )
                is_last = b.cmp_eq(m, b.const_i32(P - 1))
                v = b.select(is_last, b.vec_extract(raw, 0), v)
                src = b.select(is_last, b.const_i32(P - 1), mm)
            bad = b.land(b.cmp_lt(m, b.const_i32(P)), b.cmp_ne(v, payload_word(b, salt, ph, src)))
            if it is not None:
                # any wave's timeout marks the CTA's epoch (epoch-tagged word,
                # no reset race); thread 0 reads it after the barrier
                timed = b.cmp_ge(it, maxit)
                with b.scf_if(b.land(timed, b.cmp_eq(lane, b.const_i32(0)))):
                    _lds_write(b, lds1, b.add(ph, b.const_i32(1)))
                bad = b.land(bad, b.cmp_lt(it, maxit))
                with b.scf_if(tid0):
                    b.store(b.zext(it, I64), out, b.add(row, b.const_i32(3)))
            with b.scf_if(bad):
                D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), b.const_i32(1))
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(tid0):
            _stamp(b, out, row, 1)
            if spec.poll == "deadline":
                tcta = b.cmp_eq(_lds_read(b, lds1), b.add(ph, b.const_i32(1)))
                st = b.select(tcta, b.const_i32(ST_TIMEOUT), b.const_i32(ST_OK))
                b.store(b.zext(st, I64), out, b.add(row, b.const_i32(2)))
                with b.scf_if(tcta):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
            else:
                b.store(b.const_i64(ST_OK), out, b.add(row, b.const_i32(2)))
    _info_end(b, info, bid, G)
    b.ret()
    return b.kernel


# -------------------------------------------------------------------- ticket


TICKET_COLS = 8


TICKET_STARTS = ("one", "xcd")


@dataclass(frozen=True)
class TicketSpec:
    """``start="one"``: FlyTape's ticket (rank word per XCD, one START word
    every CTA adds to and polls). ``start="xcd"``: the rank words ARE the
    start count: one returning atomic per CTA, wave 0 polls the 8 rank words
    lane-parallel and sums them (``v_readlane``) against ``expected``."""

    threads: int = THREADS
    pause: str = "s_nop"
    start: str = "one"
    tag: str = ""

    def name(self) -> str:
        parts = ["q38p4_tk", f"t{self.threads}"]
        if self.start != "one":
            parts.append(f"st{self.start}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def _poll_rank_sum(b: IRBuilder, voff, base, expected, maxit, pause: str):
    """Wave 0: lane l < 8 loads rank word l (``sc1``); the 8 values are summed
    through ``v_readlane`` until the sum is ``>= expected`` or ``maxit``
    retries ran. Returns (per-lane word VGPR, iterations)."""
    top = "${:private}p4t${:uid}_top"
    done = "${:private}p4t${:uid}_done"
    lines = ["s_mov_b32 $3, 0", f"{top}:", "global_load_dword $0, $4, $5 sc1", "s_waitcnt vmcnt(0)",
             "v_readlane_b32 $1, $0, 0", "s_mov_b32 $2, $1"]
    for k in range(1, 8):
        lines += [f"v_readlane_b32 $1, $0, {k}", "s_add_u32 $2, $2, $1"]
    lines += ["s_cmp_ge_u32 $2, $6", f"s_cbranch_scc1 {done}", "s_add_u32 $3, $3, 1",
              "s_cmp_ge_u32 $3, $7", f"s_cbranch_scc1 {done}", D.PAUSES[pause],
              f"s_branch {top}", f"{done}:"]
    v, _t, _sum, it = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,v,s,s,s,~{scc}",
        [voff, base, expected, maxit],
        result_types=[I32, I32, I32, I32],
        result_name_hint="p4t",
    )
    return v, it


def build_ticket_kernel(spec: TicketSpec = TicketSpec()) -> KernelDef:
    """FlyTape's residency ticket (``runtime.py:1046-1114``). Params: ``sync``
    i32* (zeroed per launch: START at byte 0, the per-XCD rank words at
    ``LINE * (1 + xcc)``), ``out`` i64[grid * TICKET_COLS], ``expected``,
    ``maxit``, ``ticket`` (0 = off: stamps only).

    ``out`` row: [t_entry, t_after, xcc, rank, worker, hw_id, status, iters].
    Thread 0: rank = atomic add (returning) on its XCD's word, START += 1, wait
    (bounded) for START >= expected, then worker = sum of earlier XCDs' counts
    + rank (XCD-contiguous). A grid that cannot be resident times out (status
    ST_TIMEOUT) instead of hanging."""
    T = spec.threads
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    sync = b.param("sync", PtrType(I32, "global"))
    out = b.param("out", PtrType(I64, "global"))
    expected = b.readfirstlane(b.param("expected", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    ticket = b.readfirstlane(b.param("ticket", I32))
    bid = b.readfirstlane(b.block_id_x())
    _pad_lds(b)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    row = b.mul(bid, b.const_i32(TICKET_COLS))

    def st(col, v):
        b.store(v if v.type == I64 else b.zext(v, I64), out, b.add(row, b.const_i32(col)))

    if spec.start == "xcd":
        _ticket_xcd(b, spec, sync_a, row, st, expected, maxit, ticket)
        b.s_barrier_bare(fence="compiler")
        b.ret()
        return b.kernel
    if spec.start != "one":
        raise ValueError(spec.start)
    with b.scf_if(D.is_thread0(b)):
        t0 = D.memrealtime(b)
        xcc = D.xcc_id(b)
        with b.scf_if(b.cmp_ne(ticket, b.const_i32(0))):
            rank = D.atomic_add_scoped(
                b,
                sync_a,
                b.mul(b.add(xcc, b.const_i32(1)), b.const_i32(LINE)),
                b.const_i32(1),
                scope="agent",
                returns=True,
            )
            D.atomic_add_scoped(b, sync_a, b.const_i32(0), b.const_i32(1), scope="agent")
            _seen, it = poll_counter(b, b.const_i32(0), sync_a, expected, maxit, spec.pause)
            loads = "\n".join(
                f"global_load_dword ${k}, $8, $9 offset:{LINE * (1 + k)} sc1" for k in range(8)
            )
            cnts = b.inline_asm_multi(
                loads + "\ns_waitcnt vmcnt(0)",
                ",".join(["=&v"] * 8 + ["v", "s"]),
                [b.const_i32(0), sync_a],
                result_types=[I32] * 8,
                result_name_hint="p4cnt",
            )
            prefix = b.const_i32(0)
            for k in range(8):
                prefix = b.add(
                    prefix, b.select(b.cmp_lt(b.const_i32(k), xcc), cnts[k], b.const_i32(0))
                )
            t1 = D.memrealtime(b)
            st(1, t1)
            st(3, rank)
            st(4, b.add(prefix, rank))
            st(6, b.select(b.cmp_ge(it, maxit), b.const_i32(ST_TIMEOUT), b.const_i32(ST_OK)))
            st(7, it)
        with b.scf_if(b.cmp_eq(ticket, b.const_i32(0))):
            st(1, D.memrealtime(b))
            st(6, b.const_i32(0))
        st(0, t0)
        st(2, xcc)
        st(5, D.hw_id(b))
    b.s_barrier_bare(fence="compiler")
    b.ret()
    return b.kernel


def _ticket_xcd(b: IRBuilder, spec, sync_a, row, st, expected, maxit, ticket) -> None:
    """The ``start="xcd"`` ticket body, all in wave 0 (lane 0 stores)."""
    with b.scf_if(D.is_wave0(b)):
        lane = b.lane_id()
        lane0 = b.cmp_eq(lane, b.const_i32(0))
        t0 = D.memrealtime(b)
        xcc = D.xcc_id(b)
        with b.scf_if(b.cmp_ne(ticket, b.const_i32(0))):
            # lane 0 alone adds (exec narrowed inside the asm, restored after)
            ret, _sv = b.inline_asm_multi(
                "s_mov_b64 $1, exec\ns_mov_b64 exec, 1\n"
                "global_atomic_add $0, $2, $3, $4 sc0\ns_waitcnt vmcnt(0)\n"
                "s_mov_b64 exec, $1",
                "=&v,=&s,v,v,s,~{memory}",
                [
                    b.mul(b.add(xcc, b.const_i32(1)), b.const_i32(LINE)),
                    b.const_i32(1),
                    sync_a,
                ],
                result_types=[I32, I64],
                result_name_hint="p4rk",
            )
            rank = b.readfirstlane(ret)
            lane8 = b.select(b.cmp_lt(lane, b.const_i32(8)), lane, b.const_i32(0))
            voff = b.mul(b.add(lane8, b.const_i32(1)), b.const_i32(LINE))
            v, it = _poll_rank_sum(b, voff, sync_a, expected, maxit, spec.pause)
            t1 = D.memrealtime(b)
            prefix = b.const_i32(0)
            for k in range(8):
                prefix = b.add(
                    prefix,
                    b.select(
                        b.cmp_lt(b.const_i32(k), xcc),
                        b.readlane(v, b.const_i32(k)),
                        b.const_i32(0),
                    ),
                )
            with b.scf_if(lane0):
                st(1, t1)
                st(3, rank)
                st(4, b.add(prefix, rank))
                st(6, b.select(b.cmp_ge(it, maxit), b.const_i32(ST_TIMEOUT), b.const_i32(ST_OK)))
                st(7, it)
        with b.scf_if(b.land(lane0, b.cmp_eq(ticket, b.const_i32(0)))):
            st(1, D.memrealtime(b))
            st(6, b.const_i32(0))
        with b.scf_if(lane0):
            st(0, t0)
            st(2, xcc)
            st(5, D.hw_id(b))


# ------------------------------------------------------------------- xl hop


XL_MODES = ("pairs", "bulk")
XL_MUTANTS = ("none", "tagoff", "noready")
XL_COLS_PER_PRODUCER = 16  # hidden columns per M5 task (moe.py:27-31)
XL_STREAMS = 4  # HC streams


def xl_words_per_producer(rows: int) -> int:
    """bf16 values packed two per word: 16 columns x 4 streams x rows / 2."""
    return XL_COLS_PER_PRODUCER * XL_STREAMS * rows // 2


def xl_chunks_per_thread(rows: int, producers: int = 160, threads: int = THREADS) -> int:
    """8-byte chunks (two payload words) each consumer thread reads."""
    total = producers * xl_words_per_producer(rows) // 2
    if total % threads:
        raise ValueError("payload chunks must divide over the consumer threads")
    return total // threads


def numa_phys_pair(q: int) -> int:
    """Logical pair q -> its copy-0 pair index in the NUMA dual-copy layout
    (1024 pairs per 8 KiB, the copy 8 KiB after)."""
    return (q >> 10) * 2048 + (q & 1023)


@dataclass(frozen=True)
class XlSpec:
    rows: int = 1
    mode: str = "pairs"
    knobs: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    producers: int = 160
    grid: int = GRID
    threads: int = THREADS
    mutant: str = "none"
    tag: str = ""

    def name(self) -> str:
        parts = ["q38p4_xl", self.mode, f"r{self.rows}", f"p{self.producers}", f"g{self.grid}"]
        if self.mutant != "none":
            parts.append(f"mut{self.mutant}")
        sfx = self.knobs.suffix()
        if sfx:
            parts.append(sfx)
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


SUM_BATCH = 16  # plain 8-byte loads in flight per running-sum batch


def _sum_plain(b: IRBuilder, data, wbase, tid, K: int, T: int):
    """Sum of payload words ``2 c, 2 c + 1`` for ``c = tid + T k``, k < K, of
    the epoch at word ``wbase``: plain 8-byte loads, ``SUM_BATCH`` in flight,
    one running sum (no per-load wait, no spill of all K values)."""
    tf = D.fresh_v(b, tid)
    s = b.const_i32(0)
    for k0 in range(0, K, SUM_BATCH):
        vs = []
        for k in range(k0, min(K, k0 + SUM_BATCH)):
            c = b.add(tf, b.const_i32(T * k))
            vs.append(b.global_load_vN(data, b.add(wbase, b.mul(c, b.const_i32(2))), I32, 2))
        for v2 in vs:
            s = b.add(s, b.add(b.vec_extract(v2, 0), b.vec_extract(v2, 1)))
        s = D.fresh_v(b, s)
    return s


def _phys(b: IRBuilder, q, numa: bool):
    if not numa:
        return q
    return b.add(
        b.mul(b.lshr(q, b.const_i32(10)), b.const_i32(2048)), b.land(q, b.const_i32(1023))
    )


def build_xl_kernel(spec: XlSpec) -> KernelDef:
    """Params (pairs): ``box`` i32* (epoch e at pair ``e * stride``),
    ``flip`` i32[phases]; (bulk): ``data`` i32* (epoch e at word ``e *
    producers * W``), ``rbox`` i32* (ready pairs, epoch e at ``e *
    round_pairs``), ``rflip`` i32[phases]; then (both) ``sync`` i32* (closing
    counter words, zeroed per launch), ``tab`` (counter table of the closing
    sync: every CTA a producer), ``dtab``, ``out``, ``sink`` i32[phases *
    ncons * threads] (each consumer thread's payload-word sum per epoch),
    ``err``, ``info``, ``base``, ``salt``, ``phases``, ``ncons`` (consumers are
    CTAs ``grid - ncons .. grid - 1``), ``maxit``, ``stride`` (pairs per epoch,
    pairs mode).

    Producer p (CTA p) waits its delay, stamps col 0, then threads ``t < W/2``
    publish words ``(2 t, 2 t + 1)`` of its slice: word ``p * W + i`` =
    ``payload_word(salt, e, p * W + i)``. Consumer thread t reads chunks
    ``c = t + threads * k`` (words ``2 c, 2 c + 1``), sums them, stores the sum,
    the CTA barriers and thread 0 stamps col 1 (done).

    Mutants: ``tagoff`` (pairs): epochs >= 1 poll the previous epoch's slots
    and tag (the ``xl`` tag off by one layer); ``noready`` (bulk): consumers
    skip the ready poll and the acquire."""
    kn = spec.knobs
    P, G, T, R = spec.producers, spec.grid, spec.threads, spec.rows
    if spec.mode not in XL_MODES or spec.mutant not in XL_MUTANTS:
        raise ValueError((spec.mode, spec.mutant))
    if spec.mutant != "none" and spec.mutant != {"pairs": "tagoff", "bulk": "noready"}[spec.mode]:
        raise ValueError("tagoff is the pairs mutant, noready the bulk one")
    W = xl_words_per_producer(R)
    K = xl_chunks_per_thread(R, P, T)
    if W // 2 > T:
        raise ValueError("one producer thread per chunk")
    numa = kn.numa_copies == 2
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    if spec.mode == "pairs":
        box = b.param("box", gp)
        flip = b.param("flip", gp)
    else:
        data = b.param("data", gp)
        rbox = b.param("rbox", gp)
        flip = b.param("rflip", gp)
    sync = b.param("sync", gp)
    tab = b.param("tab", gp)
    dtab = b.param("dtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    sink = b.param("sink", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    base = b.readfirstlane(b.param("base", I32))
    salt = b.readfirstlane(b.param("salt", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    ncons = b.readfirstlane(b.param("ncons", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    stride = b.readfirstlane(b.param("stride", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lane = b.lane_id()
    _pad_lds(b)
    _info_start(b, info, bid, G)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    poff = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_POFF)))
    pcnt = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_CNT)))
    pbias = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_BIAS)))
    aoff = b.readfirstlane(b.global_load_i32(tab, b.add(bid, b.const_i32(TAB_ARR))))
    tid0 = D.is_thread0(b)
    is_prod = b.cmp_lt(bid, b.const_i32(P))
    first_cons = b.sub(b.const_i32(G), ncons)
    is_cons = b.cmp_ge(bid, first_cons)
    ci = b.sub(bid, first_cons)
    pub = b.cmp_lt(tid, b.const_i32(W // 2))
    if spec.mode == "pairs":
        mb = M.mailbox_from_ptr(b, box, b.const_i32(0x7FFFFFF0), kn)
    else:
        rmb = M.mailbox_from_ptr(b, rbox, b.const_i32(0x7FFFFFF0), kn)
        rpairs = 2048 if numa else P
    nw = (P + 63) // 64
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        tag = b.add(base, b.add(ph, b.const_i32(1)))
        # ---- producers
        with b.scf_if(is_prod):
            d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
            _delay(b, d)
            with b.scf_if(tid0):
                _stamp(b, out, row, 0)
            c = b.add(b.mul(bid, b.const_i32(W // 2)), tid)
            w0 = b.mul(c, b.const_i32(2))
            v0 = payload_word(b, salt, ph, w0)
            v1 = payload_word(b, salt, ph, b.add(w0, b.const_i32(1)))
            if spec.mode == "pairs":
                with b.scf_if(pub):
                    idx = b.add(b.mul(ph, stride), _phys(b, w0, numa))
                    M.put2(b, mb, idx, v0, v1, tag, kn)
            else:
                with b.scf_if(pub):
                    widx = b.add(b.mul(ph, b.const_i32(P * W)), w0)
                    b.global_store_vN(data, widx, b.vec_pack([v0, v1], I32), 2)
                M.release_bulk(b, rmb, b.add(b.mul(ph, b.const_i32(rpairs)), bid), tag, kn)
        # ---- consumers
        with b.scf_if(is_cons):
            if spec.mode == "pairs":
                eslot = ph
                if spec.mutant == "tagoff":
                    eslot = b.select(
                        b.cmp_gt(ph, b.const_i32(0)), b.sub(ph, b.const_i32(1)), ph
                    )
                ctag = b.add(base, b.add(eslot, b.const_i32(1)))
                mbc = mb
                if numa:
                    mbc = M.local_copy(b, mb, b.readfirstlane(b.global_load_i32(flip, eslot)))
                # per epoch copy of tid (no hoisted, spilled offsets) and one
                # running sum per poll batch (few live payload registers)
                tf = D.fresh_v(b, tid)
                eoff = b.mul(eslot, stride)
                s = b.const_i32(0)
                for k0 in range(0, K, kn.poll_max):
                    specs = []
                    for k in range(k0, min(K, k0 + kn.poll_max)):
                        q = b.mul(b.add(tf, b.const_i32(T * k)), b.const_i32(2))
                        specs.append((mbc, b.add(eoff, _phys(b, q, numa)), 2))
                    for vs in M.poll(b, specs, ctag, kn):
                        for w in vs:
                            s = b.add(s, w)
                    # pin the running sum here: LLVM otherwise sinks the adds
                    # past the later polls and keeps every payload live (spills)
                    s = D.fresh_v(b, s)
            else:
                if spec.mutant != "noready":
                    rmbc = rmb
                    if numa:
                        rmbc = M.local_copy(
                            b, rmb, b.readfirstlane(b.global_load_i32(flip, ph))
                        )
                    wave = D.wave_id(b)
                    with b.scf_if(b.cmp_lt(wave, b.const_i32(nw))):
                        m = b.add(lane, b.mul(wave, b.const_i32(64)))
                        m = b.select(b.cmp_lt(m, b.const_i32(P - 1)), m, b.const_i32(P - 1))
                        M.poll(b, [(rmbc, b.add(b.mul(ph, b.const_i32(rpairs)), m), 1)], tag, kn)
                    b.s_barrier_bare(fence="compiler")
                    with b.scf_if(tid0):
                        D.acquire_l2(b, kn.scope)
                    b.s_barrier_bare(fence="compiler")
                s = _sum_plain(b, data, b.mul(ph, b.const_i32(P * W)), tid, K, T)
            b.global_store(
                sink, b.add(b.mul(b.add(b.mul(ph, ncons), ci), b.const_i32(T)), tid), s
            )
        b.s_barrier_bare(fence="compiler")
        with b.scf_if(b.land(is_cons, tid0)):
            _stamp(b, out, row, 1)
        # ---- closing sync (untimed): every CTA arrives
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare(fence="compiler")
        _counter_sync(
            b, sync_a, aoff, b.cmp_ge(aoff, b.const_i32(0)), poff, pcnt, pbias,
            b.add(ph, b.const_i32(1)), maxit, err_a, kn.pause,
        )  # fmt: skip
        b.s_barrier_bare(fence="compiler")
    _info_end(b, info, bid, G)
    b.ret()
    return b.kernel


# ---------------------------------------------------------------- boundary


@dataclass(frozen=True)
class BoundarySpec:
    rows: int = 1
    producers: int = 160
    grid: int = GRID
    threads: int = THREADS
    tag: str = ""

    def name(self) -> str:
        parts = ["q38p4_bd", f"r{self.rows}", f"p{self.producers}", f"g{self.grid}"]
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def build_boundary_kernel(spec: BoundarySpec) -> KernelDef:
    """One launch = one hand-off across a launch boundary (the host captures
    ``phases + 1`` launches in a HIP graph). Params: ``data`` i32* (as the
    bulk ``xl`` layout), ``ctl`` i32[1] (``salt``, read at entry like a
    layer kernel's step word), ``dtab``, ``out``, ``sink``, ``info``, ``ep``
    (this launch's epoch), ``phases``, ``ncons``.

    Consumers (``ep >= 1``) plain-load epoch ``ep - 1``'s payload, sum,
    store, barrier, stamp col 1 of epoch ``ep - 1``; producers (``ep <
    phases``) wait their delay, stamp col 0 of epoch ``ep`` and plain-store
    its payload."""
    P, G, T, R = spec.producers, spec.grid, spec.threads, spec.rows
    W = xl_words_per_producer(R)
    K = xl_chunks_per_thread(R, P, T)
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    data = b.param("data", gp)
    ctl = b.param("ctl", gp)
    dtab = b.param("dtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    sink = b.param("sink", gp)
    info = b.param("info", PtrType(I64, "global"))
    ep = b.readfirstlane(b.param("ep", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    ncons = b.readfirstlane(b.param("ncons", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    _pad_lds(b)
    _info_start(b, info, bid, G)
    salt = b.readfirstlane(b.global_load_i32(ctl, b.const_i32(0)))
    tid0 = D.is_thread0(b)
    first_cons = b.sub(b.const_i32(G), ncons)
    is_cons = b.land(b.cmp_ge(bid, first_cons), b.cmp_gt(ep, b.const_i32(0)))
    ci = b.sub(bid, first_cons)
    prev = b.sub(ep, b.const_i32(1))
    with b.scf_if(is_cons):
        s = _sum_plain(b, data, b.mul(prev, b.const_i32(P * W)), tid, K, T)
        b.global_store(sink, b.add(b.mul(b.add(b.mul(prev, ncons), ci), b.const_i32(T)), tid), s)
    b.s_barrier_bare(fence="compiler")
    with b.scf_if(b.land(is_cons, tid0)):
        _stamp(b, out, b.mul(b.add(b.mul(prev, b.const_i32(G)), bid), b.const_i32(OUT_COLS)), 1)
    with b.scf_if(b.land(b.cmp_lt(bid, b.const_i32(P)), b.cmp_lt(ep, phases))):
        eg = b.mul(ep, b.const_i32(G))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        _delay(b, d)
        with b.scf_if(tid0):
            _stamp(b, out, b.mul(b.add(eg, bid), b.const_i32(OUT_COLS)), 0)
        with b.scf_if(b.cmp_lt(tid, b.const_i32(W // 2))):
            c = b.add(b.mul(bid, b.const_i32(W // 2)), tid)
            w0 = b.mul(c, b.const_i32(2))
            v0 = payload_word(b, salt, ep, w0)
            v1 = payload_word(b, salt, ep, b.add(w0, b.const_i32(1)))
            b.global_store_vN(
                data, b.add(b.mul(ep, b.const_i32(P * W)), w0), b.vec_pack([v0, v1], I32), 2
            )
    _info_end(b, info, bid, G)
    b.ret()
    return b.kernel
