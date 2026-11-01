# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Synchronization primitives of the true-mono persistent platform (WP1).

What P4 measured, as reusable emitters (``experiments/qwen38-true-mono/probes/
P4.md`` sections 1.4-1.5, P9 finding 2, P2 finding 3; ``D0.md`` D0.8):

* CTA control words in LDS (:func:`pad_lds`): an LDS pad above half a CU keeps
  one CTA per CU (FlyTape's residency rule); words 0 / 1 are the CTA's abort
  flag and its epoch-tagged timeout word.
* :func:`poll_counter`: wave-0 lane-parallel poll of up to 64 counter words
  (FlyTape ``grid_wait``: a ballot per retry), with an iteration bound.
* :func:`poll_pair_deadline`: the mailbox asm poll of one pair per lane with an
  iteration bound (a dropped producer times out instead of hanging).
* :func:`poll_rank_sum` / :func:`ticket_xcd`: the per-XCD-start residency
  ticket (0.25-0.40 of FlyTape's single-START ticket).
* :class:`GridSync` / :func:`grid_sync`: the striped counter barrier (8 words
  per XCD by default; the word layout is computed in the kernel), with P4's
  safety rules built in: a CTA whose wait timed out aborts (it never arrives
  or waits again: a later arrival would release CTAs still waiting in the
  timed-out epoch), and every timeout sets the launch-fatal error word.
* :func:`cta_arrive`: one epoch arrival per CTA from its last wave (P9
  ``ectr=cta``; every wave adding to one word was a launch-start hot spot).
* :func:`raise_fatal` / :data:`FATAL_BITS`: the launch-fatal error word
  (``err[ERR_FATAL]``, an atomic OR of :data:`FATAL_BITS`); the host checks it
  after every launch (:func:`fatal_reasons`), never per CTA.

The first group (``pad_lds`` .. ``poll_rank_sum``, ``delay``, ``stamp``,
``st_sc1`` / ``ld_sc1``) emits exactly what ``probes/p4_sync.py`` emits:
:mod:`.parity` rebuilds the P4 / P5 kernels on them and the tests check the
LLVM IR is byte-identical to the probes', so the probes' hardware verdicts
carry over and the hardware parity run re-checks them.

Mirrored ops only (inline asm, ``readfirstlane``, ``scf``, LDS).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, IRBuilder, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "ERR_FATAL",
    "ERR_MISMATCH",
    "ERR_TIMEOUT",
    "ERR_WORDS",
    "FATAL_BITS",
    "GRID",
    "LINE",
    "LaunchFatal",
    "ST_ABORTED",
    "ST_OK",
    "ST_TIMEOUT",
    "THREADS",
    "GridSync",
    "aborted",
    "check_launch",
    "cta_arrive",
    "delay",
    "fatal_reasons",
    "grid_sync",
    "grid_sync_arrive",
    "grid_sync_wait",
    "ld_sc1",
    "lds_read",
    "lds_write",
    "pad_lds",
    "poll_counter",
    "poll_pair_deadline",
    "poll_rank_sum",
    "raise_fatal",
    "st_sc1",
    "stamp",
    "ticket_xcd",
]

GRID = 256
THREADS = 512
LINE = 256  # bytes between counter words (one per 256-byte line, FlyTape SYNC_WORD_STRIDE)
ST_OK = 1
ST_TIMEOUT = 2
ST_ABORTED = 3  # a CTA that timed out earlier neither arrives nor waits again (FlyTape ABORTED)
# err[] word indices (i32 words)
ERR_MISMATCH = 0
ERR_TIMEOUT = 1
ERR_FATAL = 2  # launch-fatal bits (atomic OR), checked by the host after the launch
ERR_WORDS = 4
FATAL_BITS = {
    "poll_timeout": 1,  # a bounded mailbox poll ran out (dropped / late producer, broken tag invariant)
    "sync_timeout": 2,  # a grid-sync wait ran out (dropped arrival): every later epoch aborts
    "ticket_timeout": 4,  # the residency ticket ran out: the grid is not co-resident
    "aborted": 8,  # a CTA skipped work because it had aborted earlier
}
# CTA control words (LDS word index, from pad_lds's base)
LDS_ABORT = 0
LDS_TIMEOUT = 1
LDS_ARRIVE = 2  # the per-CTA arrival counter of cta_arrive


def fatal_reasons(word: int) -> list[str]:
    """Names of the bits set in a launch-fatal error word (host side)."""
    return [k for k, v in FATAL_BITS.items() if int(word) & v]


class LaunchFatal(RuntimeError):
    """A launch set the launch-fatal word: its outputs must not be used."""

    def __init__(self, what: str, word: int, timeouts: int):
        self.word, self.timeouts, self.reasons = int(word), int(timeouts), fatal_reasons(word)
        super().__init__(f"{what or 'launch'}: launch-fatal {self.reasons} (word {self.word:#x}), timeouts {self.timeouts}")


def check_launch(err, what: str = "") -> None:
    """Host check of one launch's error words (``err[ERR_*]``, i32): raise
    :class:`LaunchFatal` when the launch-fatal word is set.

    Mandatory after every Path B (``layers=1``) launch, or after every graph
    replay of such launches (the word is a sticky OR, so one check covers
    every launch that wrote it since the host cleared it). A broken tag
    invariant on an edge whose producer ran in an earlier launch can be
    invisible to numerics: a timed-out deadline poll returns the earlier
    launch's payload, which can be the right value (WP1 verification,
    ``wave1/WP1.md``); only this word tells."""
    word = int(err[ERR_FATAL])
    if word:
        raise LaunchFatal(what, word, int(err[ERR_TIMEOUT]))


# ---------------------------------------------------------- P4's primitives


def lds_write(b: IRBuilder, addr, v) -> None:
    """``ds_write_b32`` + ``lgkmcnt(0)`` (complete before a later barrier)."""
    b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, v], None)


def lds_read(b: IRBuilder, addr):
    """Uniform ``ds_read_b32`` of a CTA control word."""
    v = b.inline_asm(
        "ds_read_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "=v,v,~{memory}", [addr], I32,
        result_name_hint="ldsw",
    )  # fmt: skip
    return b.readfirstlane(v)


def pad_lds(b: IRBuilder, name_hint: str = "p4pad", *, arrive_word: bool = False):
    """One CTA per CU: allocate more than half a CU's LDS, touched through asm
    so the lowering keeps it (a plain LDS store that is never read is dropped
    with its allocation: seen, group segment 0). Words 0 and 1 are CTA control
    words (abort flag, epoch-tagged timeout), zeroed here; the caller barriers
    before using them. ``arrive_word``: also zero word ``LDS_ARRIVE``, the
    counter :func:`cta_arrive` needs at 0 (LDS is not zeroed at launch; the
    default keeps P4's bytes). Returns the LDS byte address of word 0."""
    lds = b.smem_alloc(I32, [ONE_CTA_PER_CU_WORDS], name_hint=name_hint)
    w0 = b.trunc(b.smem_addr_of(lds), I32)
    with b.scf_if(D.is_thread0(b)):
        zero = b.const_i32(0)
        lds_write(b, w0, zero)
        lds_write(b, b.add(w0, b.const_i32(4)), zero)
        if arrive_word:
            lds_write(b, b.add(w0, b.const_i32(4 * LDS_ARRIVE)), zero)
        lds_write(b, b.add(w0, b.const_i32(4 * (ONE_CTA_PER_CU_WORDS - 1))), zero)
    return w0


def delay(b: IRBuilder, d) -> None:
    """``d`` (uniform i32) x ``s_sleep 1`` (64 clocks each): an injected skew
    (P4 timing, the delay-injection stress harness)."""
    with b.scf_for(b.const_i32(0), d, b.const_i32(1), iv_name="p4d"):
        D.s_sleep(b, 1)


def st_sc1(b: IRBuilder, base, voff, v) -> None:
    """``global_store_dword ... sc1`` (write-through, device scope)."""
    b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s", [voff, v, base], None)


def ld_sc1(b: IRBuilder, base, voff):
    """``global_load_dword ... sc1`` + ``vmcnt(0)``."""
    return b.inline_asm(
        "global_load_dword $0, $1, $2 sc1\ns_waitcnt vmcnt(0)",
        "=&v,v,s",
        [voff, base],
        I32,
        result_name_hint="ld1",
    )


def stamp(b: IRBuilder, out, row, col: int) -> None:
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


def poll_rank_sum(b: IRBuilder, voff, base, expected, maxit, pause: str):
    """Wave 0: lane l < 8 loads rank word l (``sc1``); the 8 values are summed
    through ``v_readlane`` until the sum is ``>= expected`` or ``maxit``
    retries ran. Returns (per-lane word VGPR, iterations)."""
    top = "${:private}p4t${:uid}_top"
    done = "${:private}p4t${:uid}_done"
    lines = ["s_mov_b32 $3, 0", f"{top}:", "global_load_dword $0, $4, $5 sc1", "s_waitcnt vmcnt(0)",
             "v_readlane_b32 $1, $0, 0", "s_mov_b32 $2, $1"]  # fmt: skip
    for k in range(1, 8):
        lines += [f"v_readlane_b32 $1, $0, {k}", "s_add_u32 $2, $2, $1"]
    lines += ["s_cmp_ge_u32 $2, $6", f"s_cbranch_scc1 {done}", "s_add_u32 $3, $3, 1",
              "s_cmp_ge_u32 $3, $7", f"s_cbranch_scc1 {done}", D.PAUSES[pause],
              f"s_branch {top}", f"{done}:"]  # fmt: skip
    v, _t, _sum, it = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,v,s,s,s,~{scc}",
        [voff, base, expected, maxit],
        result_types=[I32, I32, I32, I32],
        result_name_hint="p4t",
    )
    return v, it


def ticket_xcd(b: IRBuilder, spec, sync_a, row, st, expected, maxit, ticket, *, fatal_err=None) -> None:
    """The per-XCD-start residency ticket (P4 ``start="xcd"``), all in wave 0
    (lane 0 stores through ``st(col, value)``): one returning atomic on the
    CTA's XCD rank word (the rank words ARE the start count), wave 0 polls the
    8 words lane-parallel and sums them against ``expected``; worker id =
    earlier XCDs' counts + rank (XCD-contiguous). ``spec.pause``: the retry
    pause. A grid that cannot be co-resident times out (status ST_TIMEOUT);
    with ``fatal_err`` (the err buffer's i64 address) the timeout also sets
    ``ticket_timeout`` in the launch-fatal word (P4: a residency timeout is
    not all-or-nothing, so it must be launch-fatal, checked by the host)."""
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
            v, it = poll_rank_sum(b, voff, sync_a, expected, maxit, spec.pause)
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
                if fatal_err is not None:
                    with b.scf_if(b.cmp_ge(it, maxit)):
                        raise_fatal(b, fatal_err, "ticket_timeout")
        with b.scf_if(b.land(lane0, b.cmp_eq(ticket, b.const_i32(0)))):
            st(1, D.memrealtime(b))
            st(6, b.const_i32(0))
        with b.scf_if(lane0):
            st(0, t0)
            st(2, xcc)
            st(5, D.hw_id(b))


# ------------------------------------------------------ platform additions


def raise_fatal(b: IRBuilder, err_a: Value, reason: str) -> None:
    """Set ``reason``'s bit in the launch-fatal word ``err[ERR_FATAL]``
    (``global_atomic_or``, agent scope; no return, no wait). Call from one
    lane (the caller guards)."""
    bit = FATAL_BITS[reason]
    b.inline_asm(
        "global_atomic_or $0, $1, $2",
        "v,v,s,~{memory}",
        [b.const_i32(4 * ERR_FATAL), b.const_i32(bit), err_a],
        None,
    )


def aborted(b: IRBuilder, lds0) -> Value:
    """Uniform i1: this CTA aborted earlier (its LDS abort word is set)."""
    return b.cmp_ne(lds_read(b, lds0), b.const_i32(0))


def cta_arrive(b: IRBuilder, lds0, sync_a: Value, aoff: Value, waves: int, *, value: int = 1) -> None:
    """One epoch arrival per CTA from its last wave (P9 ``ectr=cta``).

    Every wave calls this once per epoch after its own work: lane 0 waits for
    the wave's stores (``vmcnt(0)``), adds 1 to the CTA's LDS arrival counter
    (``ds_add_rtn_u32``), and the wave that brings the count to a multiple of
    ``waves`` (the CTA's last) adds ``value`` to the global word at byte
    ``aoff`` of ``sync_a`` (agent scope). The LDS counter is never reset (the
    multiple-of rule), so no barrier is needed between epochs; it must start
    at 0 (``pad_lds(arrive_word=True)`` and a CTA barrier before the first
    call). Mirrored ops (asm). Hardware-gated through ``GridSync(arrivals=
    "last")`` (WP1 fix round)."""
    if waves < 1:
        raise ValueError("waves must be >= 1")
    lane0 = b.cmp_eq(b.lane_id(), b.const_i32(0))
    with b.scf_if(lane0):
        addr = b.add(lds0, b.const_i32(4 * LDS_ARRIVE))
        old = b.inline_asm(
            "s_waitcnt vmcnt(0)\nds_add_rtn_u32 $0, $1, $2\ns_waitcnt lgkmcnt(0)",
            "=v,v,v,~{memory}",
            [addr, b.const_i32(1)],
            I32,
            result_name_hint="arr",
        )
        last = b.cmp_eq(b.mod(b.add(old, b.const_i32(1)), b.const_i32(waves)), b.const_i32(0))
        with b.scf_if(last):
            D.atomic_add_scoped(b, sync_a, aoff, b.const_i32(value), scope="agent")


@dataclass(frozen=True)
class GridSync:
    """A striped grid sync over ``grid`` CTAs: ``stripes`` counter words per
    XCD, one per 256-byte line (word ``k`` at byte ``k * LINE``; P4: the IOD
    home of counter words is noise). CTA ``c`` arrives at word
    ``(c % 8) * stripes + (c // 8) % stripes`` (P4's ``ctr_table`` map; with
    round-robin dispatch ``c % 8`` is the CTA's XCD), so every word counts
    ``grid / (8 * stripes)`` arrivals per epoch and the counters are
    cumulative (never reset inside a launch: epoch ``e`` (1-based) waits for
    ``e * count``).

    ``maxit``: wait bound (0 = unbounded, a hang on a dropped arrival);
    ``pause``: retry pause; ``arrivals``: ``cta`` (thread 0 after the CTA
    barrier) | ``wave`` (every wave's lane 0, the losing P9 form; the word
    then counts ``waves`` per CTA) | ``last`` (:func:`cta_arrive`: every wave
    counts itself in LDS and the CTA's last wave adds 1; needs
    ``pad_lds(arrive_word=True)``)."""

    grid: int = GRID
    stripes: int = 8
    maxit: int = 0
    pause: str = "s_nop"
    arrivals: str = "cta"
    waves: int = 8

    def __post_init__(self) -> None:
        if self.stripes not in (1, 2, 4, 8):
            raise ValueError("stripes must be 1, 2, 4 or 8")
        if self.grid % (8 * self.stripes):
            raise ValueError("grid must be a multiple of 8 * stripes")
        if self.arrivals not in ("cta", "wave", "last"):
            raise ValueError("arrivals must be 'cta', 'wave' or 'last'")

    @property
    def words(self) -> int:
        return 8 * self.stripes

    @property
    def count(self) -> int:
        """Arrivals per word per epoch."""
        per = self.grid // self.words
        return per * (self.waves if self.arrivals == "wave" else 1)

    @property
    def nbytes(self) -> int:
        return self.words * LINE

    def word_of(self, b: IRBuilder, bid: Value) -> Value:
        x = b.land(bid, b.const_i32(7))
        s = b.land(b.lshr(bid, b.const_i32(3)), b.const_i32(self.stripes - 1))
        return b.add(b.mul(x, b.const_i32(self.stripes)), s)


def grid_sync_arrive(b: IRBuilder, gs: GridSync, sync_a: Value, bid: Value, *, live=None, lds0=None) -> None:
    """The arrival half: ``cta`` = thread 0 adds 1 (call after a CTA barrier
    that follows the CTA's work); ``wave`` = every wave's lane 0 adds 1 (call
    from every wave); ``last`` = :func:`cta_arrive` from every wave (``lds0``
    required). ``live`` (uniform i1, optional): arrive only if set (an
    aborted CTA must not arrive)."""
    aoff = b.mul(gs.word_of(b, bid), b.const_i32(LINE))
    if gs.arrivals == "last":
        if lds0 is None:
            raise ValueError("arrivals='last' needs lds0 (the pad_lds word 0 address)")
        if live is None:
            cta_arrive(b, lds0, sync_a, aoff, gs.waves)
        else:
            with b.scf_if(live):
                cta_arrive(b, lds0, sync_a, aoff, gs.waves)
        return
    who = D.is_thread0(b) if gs.arrivals == "cta" else b.cmp_eq(b.lane_id(), b.const_i32(0))
    cond = who if live is None else b.land(live, who)
    with b.scf_if(cond):
        D.atomic_add_scoped(b, sync_a, aoff, b.const_i32(1), scope="agent")


def grid_sync_wait(b: IRBuilder, gs: GridSync, sync_a: Value, epoch1: Value, lds0, err_a: Value | None,
                   *, maxit: Value | None = None, bias: int = 0, fatal: bool = True) -> None:  # fmt: skip
    """The wait half (wave 0, lane-parallel over the words): wait until
    every word reached ``epoch1 * count - bias``. ``maxit`` (uniform i32;
    default ``gs.maxit``, 0 = unbounded): on a timeout thread 0 sets the CTA's
    abort word, counts it in ``err[ERR_TIMEOUT]`` and (``fatal``) sets
    ``sync_timeout`` in the launch-fatal word. The caller barriers after it;
    every wave then reads the abort word (:func:`aborted`). ``bias`` > 0 is
    the early-release mutant (must fail)."""
    lane = b.lane_id()
    nw = gs.words
    inw = b.cmp_lt(lane, b.const_i32(nw))
    lw = b.select(inw, lane, b.const_i32(0))
    voff = b.mul(lw, b.const_i32(LINE))
    exp = b.select(inw, b.sub(b.mul(epoch1, b.const_i32(gs.count)), b.const_i32(bias)), b.const_i32(0))
    bounded = maxit is not None or gs.maxit > 0
    if maxit is None:
        maxit = b.const_i32(gs.maxit if gs.maxit > 0 else 0x7FFFFFFF)
    with b.scf_if(D.is_wave0(b)):
        _seen, it = poll_counter(b, voff, sync_a, exp, maxit, gs.pause)
        if bounded:
            t = b.cmp_ge(it, maxit)
            with b.scf_if(b.land(D.is_thread0(b), t)):
                lds_write(b, lds0, b.const_i32(1))
                if err_a is not None:
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
                    if fatal:
                        raise_fatal(b, err_a, "sync_timeout")


def grid_sync(b: IRBuilder, gs: GridSync, sync_a: Value, bid: Value, epoch1: Value, lds0,
              err_a: Value | None, *, maxit: Value | None = None, arrive=None, bias: int = 0,
              fatal: bool = True) -> Value:  # fmt: skip
    """A full grid sync with P4's abort rule: an aborted CTA neither arrives
    nor waits; the others arrive (thread 0, after a CTA barrier; ``arrive``:
    an extra uniform i1 condition, e.g. the dropped-arrival control), wait,
    and barrier. Returns the uniform i1 ``ok`` (this CTA has not aborted).
    ``arrivals="last"``: no barrier before the arrival (every wave counts
    itself after its own stores, :func:`cta_arrive`); the abort word read is
    ordered by the previous sync's closing barrier."""
    if gs.arrivals != "last":
        b.s_barrier_bare(fence="compiler")
    live = b.cmp_eq(lds_read(b, lds0), b.const_i32(0))
    grid_sync_arrive(b, gs, sync_a, bid, live=live if arrive is None else b.land(live, arrive), lds0=lds0)
    with b.scf_if(live):
        grid_sync_wait(b, gs, sync_a, epoch1, lds0, err_a, maxit=maxit, bias=bias, fatal=fatal)
    b.s_barrier_bare(fence="compiler")
    return b.cmp_eq(lds_read(b, lds0), b.const_i32(0))
