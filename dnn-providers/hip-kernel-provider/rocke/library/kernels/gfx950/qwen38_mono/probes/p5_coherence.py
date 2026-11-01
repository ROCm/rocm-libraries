# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P5: cross-CTA / cross-XCD coherence inside one persistent launch (gfx950).

``experiments/qwen38-true-mono/DESIGN.md`` section 8.2, P5: a KV row rewritten
on one XCD after other XCDs read it, re-read in the same launch (the MTP draft
forwards of a ``round`` launch rewrite rows the verify read); and a table word
written in-launch read back through the scalar path (draft positions, ``L``,
slots: the mono reads its tables with scalar loads, ``stages/common.py``
``table_entry``). Section 7.5 / 4 rules under test, and their negative controls:

* bulk rule: plain stores, ``vmcnt(0)``, barrier, one thread ``buffer_wbl2 sc1``
  plus a ready pair; consumer polls the pair (``sc1``), ``buffer_inv sc1``,
  barrier, plain loads. ``bulk=sc1``: ``sc1`` stores and ``sc1`` loads, no
  write-back or invalidate.
* in-launch words are carried in registers / the mailbox payload, never
  re-read through scalar (or plain) loads; a table that must be re-read goes
  through ``s_dcache_inv`` (and the L2 acquire).

Kernels (``grid`` x ``threads``, an LDS pad keeps one CTA per CU, P4's rule):

* :func:`build_kv_kernel` (``kv``): per epoch one writer CTA (host table
  ``wtab``, so the writer's XCD rotates) rewrites the ``tail`` KV rows
  (``ROW_WORDS`` words = a 1 KiB K row and a 1 KiB V row of 2 kv heads x 256
  bf16; thread ``t`` owns word ``t``) and publishes them by ``wmode``; ``ncons``
  reader CTAs (the last ones, all 8 XCDs) acquire by ``rmode`` and read the
  tail rows plus a slice of ``slice_rows`` stable rows (Q5's 32-row tiles;
  ``cold=1`` walks the slices through a buffer larger than L2, ``cold=0``
  re-reads the same slice), checking every word. Epochs are separated by a
  closing counter sync (the dataflow that orders a rewrite after the reads).
  Mutant ``early``: CTAs that do not read skip the closing wait, so the next
  epoch's writer rewrites the rows while delayed readers still read them (the
  rewrite-after-read order itself): must fail.
* :func:`build_scalar_kernel` (``sc``): per epoch one writer CTA stores a
  table word (``wmode``) and a ready pair carrying the same value; every CTA's
  wave 0 polls the pair and reads the word by ``rmode`` (``reg``: the payload;
  ``vec_*``: vector loads; ``smem*`` / ``sbuf*``: ``s_load_dword`` /
  ``s_buffer_load_dword``, optionally after ``s_dcache_inv`` and / or the L2
  acquire, or with ``glc``). Every CTA read the word in the previous epoch, so
  a stale scalar-cache (or L2) line is there to be hit.
* :func:`build_host_kernel` (``host``, loop-mode rule of 7.5): the host
  rewrites a table word and a 2 KiB data block (hipMemcpyAsync H2D on another
  stream) between rounds of one persistent launch and bumps a round word in
  pinned host memory; every CTA polls it (system scope), reads by ``rmode``,
  and acknowledges (a system-scope store) so the host never rewrites before
  every CTA has read (rewrite-after-read order across the host boundary).

Stamps (``stamps=True``): thread 0 records ``s_memrealtime`` into ``out[(epoch *
grid + cta) * OUT_COLS + col]``: col 0 writer publish start, col 1 reader done.
Every build (also ``stamps=False``) records each CTA's start / end time and
xcc / HW_ID in ``info`` (outside the epoch loop), and CTA 0's ``s_memtime`` /
``s_memrealtime`` pairs (shader clock).

Python engine only (``buffer_load_cp`` / ``buffer_store_cp`` carry the cache
bits); kernel names start with ``q38p5_`` (the family README's cpp-parity
exemption). Nothing here is imported by a served build.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import CPOL_SC0, CPOL_SC1, I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from kernels.gfx950.qwen38_mono.probes.p4_sync import (
    ERR_MISMATCH,
    ERR_TIMEOUT,
    GRID,
    OUT_COLS,
    TAB_ARR,
    TAB_BIAS,
    TAB_CNT,
    TAB_POFF,
    THREADS,
    _lds_read,
    _lds_write,
    _pad_lds,
    _stamp,
    payload_word,
    poll_counter,
    poll_pair_deadline,
)

__all__ = [
    "HOST_RMODES",
    "HostSpec",
    "HostWriterSpec",
    "INFO_COLS",
    "KV_MUTANTS",
    "KV_RMODES",
    "KV_WMODES",
    "KvSpec",
    "ROW_WORDS",
    "SALT0",
    "SC_RMODES",
    "SC_WMODES",
    "SINK_COLS",
    "ScSpec",
    "build_host_kernel",
    "build_host_writer_kernel",
    "build_kv_kernel",
    "build_scalar_kernel",
    "info_words",
]

ROW_WORDS = 512  # a K row + a V row (2 kv heads x 256 bf16 each), one i32 word per thread
SALT0 = 0x5EED  # salt of the stable (host-written) KV rows; their epoch is -1
SINK_COLS = 2  # per (epoch, reader): words wrong, of which stale (= previous epoch's value)
INFO_COLS = 4  # per CTA: xcc, HW_ID, start, end (s_memrealtime)
HOST_DATA_WORDS = 512  # the host-rewritten data block (2 KiB, a block-table row)
HOST_DATA_OFF = 64  # its first word in the table buffer (word 0 is the table word)

KV_WMODES = ("wbl2", "sc1", "plain")
KV_RMODES = ("inv", "inv0", "sc1", "none")
KV_MUTANTS = ("none", "early")
SC_WMODES = ("wbl2", "sc1")
SC_RMODES = (
    "reg",
    "vec_sc1",
    "vec_inv",
    "vec_inv0",
    "vec_plain",
    "smem",
    "smem_dinv",
    "smem_inv",
    "smem_dinv_inv",
    "smem_glc",
    "smem_glc_inv",
    "sbuf",
    "sbuf_dinv_inv",
)
HOST_RMODES = (
    "vec_plain",
    "vec_inv",
    "vec_inv0",
    "vec_invsys",
    "vec_sc1",
    "vec_sys",
    "smem",
    "smem_pre",
    "smem_glc",
    "smem_dinv",
    "smem_dinv_inv",
    "smem_dinv_invsys",
)

# the memory-model-endorsed protocols (the design's positives); every other
# combination is a negative control or an informative arm
KV_POSITIVE = {("wbl2", "inv"), ("sc1", "sc1"), ("sc1", "inv"), ("wbl2", "sc1")}
SC_POSITIVE = {"reg", "vec_sc1", "vec_inv", "smem_dinv_inv", "sbuf_dinv_inv"}
HOST_POSITIVE = {"vec_invsys", "vec_sys", "smem_dinv_invsys"}


def info_words(grid: int) -> int:
    """i64 words of ``info``: ``INFO_COLS`` per CTA, then CTA 0's memtime /
    memrealtime at start and end."""
    return INFO_COLS * grid + 4


def _info_begin(b: IRBuilder, info, bid, grid: int) -> None:
    with b.scf_if(D.is_thread0(b)):
        row = b.mul(bid, b.const_i32(INFO_COLS))
        b.store(b.zext(D.xcc_id(b), I64), info, row)
        b.store(b.zext(D.hw_id(b), I64), info, b.add(row, b.const_i32(1)))
        b.store(D.memrealtime(b), info, b.add(row, b.const_i32(2)))
        with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
            b.store(D.memtime(b), info, b.const_i32(INFO_COLS * grid))
            b.store(D.memrealtime(b), info, b.const_i32(INFO_COLS * grid + 1))


def _info_finish(b: IRBuilder, info, bid, grid: int) -> None:
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare(fence="compiler")
    with b.scf_if(D.is_thread0(b)):
        row = b.mul(bid, b.const_i32(INFO_COLS))
        b.store(D.memrealtime(b), info, b.add(row, b.const_i32(3)))
        with b.scf_if(b.cmp_eq(bid, b.const_i32(0))):
            b.store(D.memtime(b), info, b.const_i32(INFO_COLS * grid + 2))
            b.store(D.memrealtime(b), info, b.const_i32(INFO_COLS * grid + 3))


_DELAYS = [0]


def _delay(b: IRBuilder, d) -> None:
    """``d`` (uniform i32) x ``s_sleep 1``; a fresh loop-variable name per call
    (two delay loops in one kernel)."""
    _DELAYS[0] += 1
    with b.scf_for(b.const_i32(0), d, b.const_i32(1), iv_name=f"p5d{_DELAYS[0]}"):
        D.s_sleep(b, 1)


def _check_arch(arch: str) -> None:
    """The builder contract is ``(spec, arch)``; these probes are gfx950-only."""
    if arch != "gfx950":
        raise ValueError(f"P5 probes target gfx950, got {arch!r}")


def _true(b: IRBuilder):
    return b.cmp_eq(b.const_i32(0), b.const_i32(0))


def _b2i(b: IRBuilder, c):
    return b.select(c, b.const_i32(1), b.const_i32(0))


def _close_sync(b: IRBuilder, sync_a, aoff, poff, pcnt, pbias, epoch1, maxit, err_a, wait) -> None:
    """Closing counter sync of an epoch: thread 0 of every CTA arrives
    (agent-scope atomic, after the CTA's ``vmcnt(0)`` + barrier); wave 0 waits
    when ``wait`` (uniform i1) until every counter word reached ``epoch1 *
    count`` (bounded; a timeout is counted in ``err``). P4's counter barrier."""
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare(fence="compiler")
    tid0 = D.is_thread0(b)
    with b.scf_if(b.land(b.cmp_ge(aoff, b.const_i32(0)), tid0)):
        D.atomic_add_scoped(b, sync_a, aoff, b.const_i32(1), scope="agent")
    with b.scf_if(b.land(D.is_wave0(b), wait)):
        exp = b.select(
            b.cmp_gt(pcnt, b.const_i32(0)),
            b.sub(b.mul(epoch1, pcnt), pbias),
            b.const_i32(0),
        )
        _seen, it = poll_counter(b, poff, sync_a, exp, maxit)
        with b.scf_if(b.land(tid0, b.cmp_ge(it, maxit))):
            D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
    b.s_barrier_bare(fence="compiler")


def _counter_params(b: IRBuilder, tab, bid):
    lane = b.lane_id()
    poff = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_POFF)))
    pcnt = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_CNT)))
    pbias = b.global_load_i32(tab, b.add(lane, b.const_i32(TAB_BIAS)))
    aoff = b.readfirstlane(b.global_load_i32(tab, b.add(bid, b.const_i32(TAB_ARR))))
    return poff, pcnt, pbias, aoff


ACQ_ASM = {
    "inv": "s_waitcnt vmcnt(0)\nbuffer_inv sc1",
    "inv0": "s_waitcnt vmcnt(0)\nbuffer_inv sc0",
    "invsys": "s_waitcnt vmcnt(0)\nbuffer_inv sc0 sc1",
}


def _acquire(b: IRBuilder, kind: str) -> None:
    b.inline_asm(ACQ_ASM[kind], "~{memory}", [], None)


# ------------------------------------------------------------------ KV rows


@dataclass(frozen=True)
class KvSpec:
    """``wmode``: ``wbl2`` (plain stores + ``buffer_wbl2 sc1`` release, the bulk
    rule), ``sc1`` (``sc1`` stores, ``vmcnt(0)``, no write-back), ``plain``
    (plain stores, ``vmcnt(0)``, no write-back: negative control).
    ``rmode``: ``inv`` (``buffer_inv sc1``, the bulk rule), ``inv0``
    (``buffer_inv sc0``: vL1D only), ``sc1`` (``sc1`` loads of the tail rows,
    no invalidate), ``none`` (plain loads, no acquire: negative control)."""

    wmode: str = "wbl2"
    rmode: str = "inv"
    tail: int = 1
    slice_rows: int = 32
    grid: int = GRID
    mutant: str = "none"
    stamps: bool = True
    tag: str = ""

    @property
    def threads(self) -> int:
        return ROW_WORDS

    @property
    def positive(self) -> bool:
        return self.mutant == "none" and (self.wmode, self.rmode) in KV_POSITIVE

    def name(self) -> str:
        parts = ["q38p5_kv", f"w{self.wmode}", f"r{self.rmode}", f"t{self.tail}",
                 f"s{self.slice_rows}", f"g{self.grid}"]  # fmt: skip
        if self.mutant != "none":
            parts.append(f"mut{self.mutant}")
        if not self.stamps:
            parts.append("ns")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


KN = M.MailboxKnobs()  # ready pairs: asm puts (sc1 dwordx2); polls are the bounded asm loop


def build_kv_kernel(spec: KvSpec, arch: str = "gfx950") -> KernelDef:
    """Params: ``kv`` i32* (``nstable + tail`` rows of ``ROW_WORDS`` words;
    rows ``< nstable`` hold ``payload_word(SALT0, -1, w)``, written by the
    host), ``rbox`` i32* (ready pair of epoch e at pair e), ``sync`` i32*
    (closing counter words, zeroed per launch), ``tab`` (counter table, every
    CTA a producer), ``dtab`` i32[phases * grid] (per-epoch delay of each CTA,
    ``s_sleep 1`` units), ``wtab`` i32[phases] (writer CTA per epoch), ``out``
    i64[phases * grid * OUT_COLS], ``sink`` i32[phases * ncons * SINK_COLS],
    ``err`` i32[4], ``info`` i64[info_words(grid)], ``salt``, ``base`` (tag of
    epoch e = base + e + 1), ``phases``, ``ncons`` (readers = CTAs ``grid -
    ncons ..``), ``nstable``, ``cold``, ``kvbytes``, ``maxit``.

    Word ``w`` (global word index) of tail row ``j`` in epoch ``e`` =
    ``payload_word(salt, e, w)``. A reader thread counts the words it read
    that differ from the expected value (``bad``) and, of those, the ones
    equal to the previous epoch's value (``stale``)."""
    _check_arch(arch)
    if spec.wmode not in KV_WMODES or spec.rmode not in KV_RMODES or spec.mutant not in KV_MUTANTS:
        raise ValueError((spec.wmode, spec.rmode, spec.mutant))
    if not 1 <= spec.tail <= 8 or not 1 <= spec.slice_rows <= 64:
        raise ValueError("tail 1..8 rows, slice_rows 1..64")
    G, T, W, S = spec.grid, spec.threads, spec.tail, spec.slice_rows
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    kv = b.param("kv", gp)
    rbox = b.param("rbox", gp)
    sync = b.param("sync", gp)
    tab = b.param("tab", gp)
    dtab = b.param("dtab", gp)
    wtab = b.param("wtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    sink = b.param("sink", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    base = b.readfirstlane(b.param("base", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    ncons = b.readfirstlane(b.param("ncons", I32))
    nstable = b.readfirstlane(b.param("nstable", I32))
    cold = b.readfirstlane(b.param("cold", I32))
    kvbytes = b.readfirstlane(b.param("kvbytes", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    _pad_lds(b)
    _info_begin(b, info, bid, G)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    sink_a = b.readfirstlane(b.global_ptr_to_i64(sink))
    poff, pcnt, pbias, aoff = _counter_params(b, tab, bid)
    rs = b.buffer_rsrc(kv, kvbytes)
    rmb = M.mailbox_from_ptr(b, rbox, b.const_i32(0x7FFFFFF0), KN)
    tid0 = D.is_thread0(b)
    first_cons = b.sub(b.const_i32(G), ncons)
    is_cons = b.cmp_ge(bid, first_cons)
    ci = b.sub(bid, first_cons)
    salt0 = b.const_i32(SALT0)
    m1 = b.const_i32(-1)
    four = b.const_i32(4)
    rw = b.const_i32(ROW_WORDS)
    waux = CPOL_SC1 if spec.wmode == "sc1" else 0
    raux = CPOL_SC1 if spec.rmode == "sc1" else 0
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        tag = b.add(base, b.add(ph, b.const_i32(1)))
        writer = b.readfirstlane(b.global_load_i32(wtab, ph))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        # ---- the writer rewrites the tail rows
        with b.scf_if(b.cmp_eq(bid, writer)):
            _delay(b, d)
            if spec.stamps:
                with b.scf_if(tid0):
                    _stamp(b, out, row, 0)
            tf = D.fresh_v(b, tid)
            for j in range(W):
                w = b.add(b.mul(b.add(nstable, b.const_i32(j)), rw), tf)
                b.buffer_store_cp(payload_word(b, salt, ph, w), rs, b.mul(w, four), b.const_i32(0), aux=waux)
            if spec.wmode == "wbl2":
                M.release_bulk(b, rmb, ph, tag, KN)
            else:
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare(fence="compiler")
                with b.scf_if(tid0):
                    M.put(b, rmb, ph, b.const_i32(0), tag, KN)
        # ---- readers: acquire, read the tail rows and a slice of stable rows, check
        with b.scf_if(is_cons):
            _delay(b, d)
            with b.scf_if(tid0):
                _v, it = poll_pair_deadline(b, M.pair_offset(b, ph), rmb.addr, tag, maxit)
                with b.scf_if(b.cmp_ge(it, maxit)):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
                if spec.rmode in ("inv", "inv0"):
                    _acquire(b, spec.rmode)
            D.compiler_fence(b)
            b.s_barrier_bare(fence="compiler")
            tf = D.fresh_v(b, tid)
            hot0 = b.mul(ci, b.const_i32(S))
            cold0 = b.mod(b.mul(b.add(b.mul(ph, ncons), ci), b.const_i32(S)), nstable)
            sb = b.readfirstlane(b.select(b.cmp_ne(cold, b.const_i32(0)), cold0, hot0))
            tw, tv = [], []
            for j in range(W):
                w = b.add(b.mul(b.add(nstable, b.const_i32(j)), rw), tf)
                tw.append(w)
                tv.append(b.buffer_load_cp(rs, b.mul(w, four), b.const_i32(0), dwords=1, aux=raux))
            sw, sv = [], []
            for k in range(S):
                w = b.add(b.mul(b.add(sb, b.const_i32(k)), rw), tf)
                sw.append(w)
                sv.append(b.buffer_load_cp(rs, b.mul(w, four), b.const_i32(0), dwords=1, aux=0))
            bad = b.const_i32(0)
            stale = b.const_i32(0)
            prev = b.sub(ph, b.const_i32(1))
            has_prev = b.cmp_gt(ph, b.const_i32(0))
            for w, v in zip(tw, tv):
                ne = b.cmp_ne(v, payload_word(b, salt, ph, w))
                bad = b.add(bad, _b2i(b, ne))
                old = b.land(has_prev, b.cmp_eq(v, payload_word(b, salt, prev, w)))
                stale = b.add(stale, _b2i(b, b.land(ne, old)))
            for w, v in zip(sw, sv):
                bad = b.add(bad, _b2i(b, b.cmp_ne(v, payload_word(b, salt0, m1, w))))
            srow = b.mul(b.add(b.mul(ph, ncons), ci), b.const_i32(4 * SINK_COLS))
            with b.scf_if(b.cmp_gt(bad, b.const_i32(0))):
                D.atomic_add_scoped(b, sink_a, srow, bad)
                D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), bad)
            with b.scf_if(b.cmp_gt(stale, b.const_i32(0))):
                D.atomic_add_scoped(b, sink_a, b.add(srow, four), stale)
            b.s_barrier_bare(fence="compiler")
            if spec.stamps:
                with b.scf_if(tid0):
                    _stamp(b, out, row, 1)
        # ---- closing sync: orders the next epoch's rewrite after these reads
        wait = _true(b) if spec.mutant == "none" else is_cons
        _close_sync(b, sync_a, aoff, poff, pcnt, pbias, b.add(ph, b.const_i32(1)), maxit, err_a, wait)
    _info_finish(b, info, bid, G)
    b.ret()
    return b.kernel


# ------------------------------------------------------------ scalar path


def _scalar_read(b: IRBuilder, mode: str, addr, nbytes, rs, p):
    """The table word (byte 0 of the table) as this wave reads it under
    ``mode``; ``p`` is the polled ready-pair payload (``reg``)."""
    if mode == "reg":
        return b.readfirstlane(p)
    if mode.startswith("vec_"):
        if mode in ("vec_inv", "vec_inv0", "vec_invsys"):
            _acquire(b, mode[4:])
        aux = {"vec_sc1": CPOL_SC1, "vec_sys": CPOL_SC0 | CPOL_SC1}.get(mode, 0)
        return b.readfirstlane(b.buffer_load_cp(rs, b.const_i32(0), b.const_i32(0), dwords=1, aux=aux))
    pre = []
    if mode.endswith("_invsys"):
        pre += ["s_waitcnt vmcnt(0)", "buffer_inv sc0 sc1", "s_waitcnt vmcnt(0)"]
    elif mode.endswith("_inv"):
        pre += ["s_waitcnt vmcnt(0)", "buffer_inv sc1", "s_waitcnt vmcnt(0)"]
    if "_dinv" in mode:
        pre += ["s_dcache_inv", "s_waitcnt lgkmcnt(0)"]
    glc = " glc" if "_glc" in mode else ""
    if mode.startswith("sbuf"):
        ld = f"s_buffer_load_dword $0, $1, 0x0{glc}"
        src = D.rsrc_vec(b, addr, nbytes)
    else:
        ld = f"s_load_dword $0, $1, 0x0{glc}"
        src = addr
    return b.inline_asm(
        "\n".join([*pre, ld, "s_waitcnt lgkmcnt(0)"]),
        "=s,s,~{memory}",
        [src],
        I32,
        result_name_hint="p5s",
    )


@dataclass(frozen=True)
class ScSpec:
    """``wmode``: ``wbl2`` (plain store + ``buffer_wbl2 sc1``) or ``sc1``
    (``sc1`` store); both then publish a ready pair whose payload is the
    value. ``rmode``: see ``SC_RMODES`` / :func:`_scalar_read`."""

    wmode: str = "wbl2"
    rmode: str = "reg"
    grid: int = GRID
    threads: int = 256
    stamps: bool = True
    tag: str = ""

    @property
    def positive(self) -> bool:
        return self.rmode in SC_POSITIVE

    def name(self) -> str:
        parts = ["q38p5_sc", f"w{self.wmode}", f"r{self.rmode}", f"g{self.grid}"]
        if not self.stamps:
            parts.append("ns")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def build_scalar_kernel(spec: ScSpec, arch: str = "gfx950") -> KernelDef:
    """Params: ``tbl`` i32* (the table word at byte 0; 64 KiB buffer),
    ``rbox``, ``sync``, ``tab``, ``dtab``, ``wtab``, ``out``, ``sink``
    i32[phases * grid * SINK_COLS] (per CTA-epoch: wrong, stale), ``err``,
    ``info``, ``salt``, ``base``, ``phases``, ``maxit``. Value of epoch e:
    ``payload_word(salt, e, 0)``."""
    _check_arch(arch)
    if spec.wmode not in SC_WMODES or spec.rmode not in SC_RMODES:
        raise ValueError((spec.wmode, spec.rmode))
    G, T = spec.grid, spec.threads
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    tbl = b.param("tbl", gp)
    rbox = b.param("rbox", gp)
    sync = b.param("sync", gp)
    tab = b.param("tab", gp)
    dtab = b.param("dtab", gp)
    wtab = b.param("wtab", gp)
    out = b.param("out", PtrType(I64, "global"))
    sink = b.param("sink", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    base = b.readfirstlane(b.param("base", I32))
    phases = b.readfirstlane(b.param("phases", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    bid = b.readfirstlane(b.block_id_x())
    _pad_lds(b)
    _info_begin(b, info, bid, G)
    sync_a = b.readfirstlane(b.global_ptr_to_i64(sync))
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    tbl_a = b.readfirstlane(b.global_ptr_to_i64(tbl))
    poff, pcnt, pbias, aoff = _counter_params(b, tab, bid)
    nbytes = b.const_i32(1 << 16)
    rs = b.buffer_rsrc(tbl, nbytes)
    rmb = M.mailbox_from_ptr(b, rbox, b.const_i32(0x7FFFFFF0), KN)
    tid0 = D.is_thread0(b)
    zero = b.const_i32(0)
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(b.const_i32(0), phases, b.const_i32(1), iv_name="ph") as ph:
        eg = b.mul(ph, b.const_i32(G))
        row = b.mul(b.add(eg, bid), b.const_i32(OUT_COLS))
        tag = b.add(base, b.add(ph, b.const_i32(1)))
        writer = b.readfirstlane(b.global_load_i32(wtab, ph))
        d = b.readfirstlane(b.global_load_i32(dtab, b.add(eg, bid)))
        v = payload_word(b, salt, ph, zero)
        with b.scf_if(b.land(b.cmp_eq(bid, writer), tid0)):
            _delay(b, d)
            if spec.stamps:
                _stamp(b, out, row, 0)
            b.buffer_store_cp(v, rs, zero, zero, aux=CPOL_SC1 if spec.wmode == "sc1" else 0)
            if spec.wmode == "wbl2":
                D.release_l2(b, "agent")
            else:
                b.s_waitcnt(vmcnt=0)
            M.put(b, rmb, ph, v, tag, KN)
        with b.scf_if(D.is_wave0(b)):
            p, it = poll_pair_deadline(b, M.pair_offset(b, ph), rmb.addr, tag, maxit)
            D.compiler_fence(b)
            got = _scalar_read(b, spec.rmode, tbl_a, nbytes, rs, p)
            ne = b.cmp_ne(got, v)
            old = b.land(
                b.cmp_gt(ph, zero), b.cmp_eq(got, payload_word(b, salt, b.sub(ph, b.const_i32(1)), zero))
            )
            with b.scf_if(tid0):
                srow = b.mul(b.add(eg, bid), b.const_i32(SINK_COLS))
                b.global_store(sink, srow, _b2i(b, ne))
                b.global_store(sink, b.add(srow, b.const_i32(1)), _b2i(b, b.land(ne, old)))
                with b.scf_if(ne):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), b.const_i32(1))
                with b.scf_if(b.cmp_ge(it, maxit)):
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
                if spec.stamps:
                    _stamp(b, out, row, 1)
        _close_sync(b, sync_a, aoff, poff, pcnt, pbias, b.add(ph, b.const_i32(1)), maxit, err_a, _true(b))
    _info_finish(b, info, bid, G)
    b.ret()
    return b.kernel


# ------------------------------------------------- host rewrites (loop mode)


@dataclass(frozen=True)
class HostSpec:
    """``rmode``: ``vec_plain`` (plain loads), ``vec_inv`` (``buffer_inv
    sc1``), ``vec_invsys`` (``buffer_inv sc0 sc1``), ``vec_sc1`` / ``vec_sys``
    (``sc1`` / ``sc0 sc1`` loads), ``vec_inv0`` (``buffer_inv sc0``: vL1D
    only), ``smem*`` (scalar load of the table word, with ``s_dcache_inv``
    and / or the agent / system acquire, or ``glc``; ``smem_pre`` also
    scalar-loads the word just before polling the round word). Vector modes
    check the table word and the data block, scalar modes the table word."""

    rmode: str = "vec_invsys"
    grid: int = 128
    threads: int = HOST_DATA_WORDS
    tag: str = ""

    @property
    def positive(self) -> bool:
        return self.rmode in HOST_POSITIVE

    def name(self) -> str:
        parts = ["q38p5_host", f"r{self.rmode}", f"g{self.grid}"]
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def _poll_host_round(b: IRBuilder, addr, want, maxit):
    """Spin (thread 0) until the host round word (``sc0 sc1`` load of pinned
    host memory at ``addr``) is ``>= want``, or ``maxit`` retries (``s_sleep
    1`` between retries). Returns ``(seen, iterations)``."""
    top = "${:private}p5h${:uid}_top"
    done = "${:private}p5h${:uid}_done"
    lines = [
        "s_mov_b32 $1, 0",
        f"{top}:",
        "global_load_dword $0, $3, $4 sc0 sc1",
        "s_waitcnt vmcnt(0)",
        "v_readfirstlane_b32 $2, $0",
        "s_cmp_ge_u32 $2, $5",
        f"s_cbranch_scc1 {done}",
        "s_add_u32 $1, $1, 1",
        "s_cmp_ge_u32 $1, $6",
        f"s_cbranch_scc1 {done}",
        "s_sleep 1",
        f"s_branch {top}",
        f"{done}:",
    ]
    seen, it, _tmp = b.inline_asm_multi(
        "\n".join(lines),
        "=&v,=&s,=&s,v,s,s,s,~{scc},~{memory}",
        [b.const_i32(0), addr, want, maxit],
        result_types=[I32, I32, I32],
        result_name_hint="p5h",
    )
    return seen, it


def build_host_kernel(spec: HostSpec, arch: str = "gfx950") -> KernelDef:
    """Params: ``tbl`` i32* (device; word 0 = table word, words
    ``HOST_DATA_OFF ..`` = data block), ``hround`` i32* (pinned host: round
    word), ``hack`` i32[grid] (pinned host: CTA c stores ``r + 1`` at word c,
    system scope, after reading round r), ``sink``
    i32[rounds * grid * SINK_COLS], ``err``, ``info``, ``salt``, ``rounds``,
    ``maxit``. Round r's values: table word ``payload_word(salt, r, 0)``,
    data word i ``payload_word(salt, r, 1 + i)``. A CTA that timed out
    stops (no ack): the host sees the missing acks."""
    _check_arch(arch)
    if spec.rmode not in HOST_RMODES:
        raise ValueError(spec.rmode)
    G, T = spec.grid, spec.threads
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = T
    gp = PtrType(I32, "global")
    tbl = b.param("tbl", gp)
    hround = b.param("hround", gp)
    hack = b.param("hack", gp)
    sink = b.param("sink", gp)
    err = b.param("err", gp)
    info = b.param("info", PtrType(I64, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    rounds = b.readfirstlane(b.param("rounds", I32))
    maxit = b.readfirstlane(b.param("maxit", I32))
    bid = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lds0 = _pad_lds(b)
    _info_begin(b, info, bid, G)
    err_a = b.readfirstlane(b.global_ptr_to_i64(err))
    tbl_a = b.readfirstlane(b.global_ptr_to_i64(tbl))
    hr_a = b.readfirstlane(b.global_ptr_to_i64(hround))
    ha_a = b.readfirstlane(b.global_ptr_to_i64(hack))
    sink_a = b.readfirstlane(b.global_ptr_to_i64(sink))
    nbytes = b.const_i32(4 * (HOST_DATA_OFF + HOST_DATA_WORDS))
    rs = b.buffer_rsrc(tbl, nbytes)
    tid0 = D.is_thread0(b)
    zero = b.const_i32(0)
    four = b.const_i32(4)
    vec = spec.rmode.startswith("vec_")
    b.s_barrier_bare(fence="compiler")
    with b.scf_for(zero, rounds, b.const_i32(1), iv_name="rd") as rd:
        alive = b.cmp_eq(_lds_read(b, lds0), zero)
        with b.scf_if(alive):
            if spec.rmode == "smem_pre":
                # refill this CU group's scalar-cache line with the old value just
                # before the host's rewrite (the read after the poll must then hit it)
                with b.scf_if(D.is_wave0(b)):
                    _scalar_read(b, "smem", tbl_a, nbytes, rs, zero)
            with b.scf_if(tid0):
                _seen, it = _poll_host_round(b, hr_a, b.add(rd, b.const_i32(1)), maxit)
                with b.scf_if(b.cmp_ge(it, maxit)):
                    _lds_write(b, lds0, b.const_i32(1))
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_TIMEOUT), b.const_i32(1))
            D.compiler_fence(b)
            b.s_barrier_bare(fence="compiler")
        alive = b.cmp_eq(_lds_read(b, lds0), zero)
        with b.scf_if(alive):
            srow = b.mul(b.add(b.mul(rd, b.const_i32(G)), bid), b.const_i32(4 * SINK_COLS))
            if vec:
                with b.scf_if(tid0):
                    if spec.rmode in ("vec_inv", "vec_inv0", "vec_invsys"):
                        _acquire(b, spec.rmode[4:])
                D.compiler_fence(b)
                b.s_barrier_bare(fence="compiler")
                aux = {"vec_sc1": CPOL_SC1, "vec_sys": CPOL_SC0 | CPOL_SC1}.get(spec.rmode, 0)
                w = b.add(tid, b.const_i32(HOST_DATA_OFF))
                dv = b.buffer_load_cp(rs, b.mul(w, four), zero, dwords=1, aux=aux)
                tv = b.buffer_load_cp(rs, zero, zero, dwords=1, aux=aux)
                ne_d = b.cmp_ne(dv, payload_word(b, salt, rd, b.add(tid, b.const_i32(1))))
                ne_t = b.land(tid0, b.cmp_ne(tv, payload_word(b, salt, rd, zero)))
                prev = b.sub(rd, b.const_i32(1))
                hp = b.cmp_gt(rd, zero)
                st_d = b.land(ne_d, b.land(hp, b.cmp_eq(dv, payload_word(b, salt, prev, b.add(tid, b.const_i32(1))))))
                st_t = b.land(ne_t, b.land(hp, b.cmp_eq(tv, payload_word(b, salt, prev, zero))))
                bad = b.add(_b2i(b, ne_d), _b2i(b, ne_t))
                stale = b.add(_b2i(b, st_d), _b2i(b, st_t))
                with b.scf_if(b.cmp_gt(bad, zero)):
                    D.atomic_add_scoped(b, sink_a, srow, bad)
                    D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), bad)
                with b.scf_if(b.cmp_gt(stale, zero)):
                    D.atomic_add_scoped(b, sink_a, b.add(srow, four), stale)
            else:
                with b.scf_if(D.is_wave0(b)):
                    got = _scalar_read(b, spec.rmode, tbl_a, nbytes, rs, zero)
                    want = payload_word(b, salt, rd, zero)
                    ne = b.cmp_ne(got, want)
                    old = b.land(b.cmp_gt(rd, zero),
                                 b.cmp_eq(got, payload_word(b, salt, b.sub(rd, b.const_i32(1)), zero)))  # fmt: skip
                    with b.scf_if(b.land(tid0, ne)):
                        D.atomic_add_scoped(b, sink_a, srow, b.const_i32(1))
                        D.atomic_add_scoped(b, err_a, b.const_i32(4 * ERR_MISMATCH), b.const_i32(1))
                        with b.scf_if(old):
                            D.atomic_add_scoped(b, sink_a, b.add(srow, four), b.const_i32(1))
            # ---- acknowledge (system scope): the host rewrites only after every ack
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare(fence="compiler")
            with b.scf_if(tid0):
                b.inline_asm(
                    "global_store_dword $0, $1, $2 sc0 sc1\ns_waitcnt vmcnt(0)",
                    "v,v,s,~{memory}",
                    [b.mul(bid, four), b.add(rd, b.const_i32(1)), ha_a],
                    None,
                )
    _info_finish(b, info, bid, G)
    b.ret()
    return b.kernel


@dataclass(frozen=True)
class HostWriterSpec:
    """The out-of-launch writer kernel; ``tag`` names a separately compiled copy."""

    tag: str = ""

    def name(self) -> str:
        return "q38p5_hostw" + (f"_{self.tag}" if self.tag else "")


def build_host_writer_kernel(spec: HostWriterSpec = HostWriterSpec(), arch: str = "gfx950") -> KernelDef:
    """The out-of-launch writer of the host-rewrite section, as a kernel (ATOM
    also fills per-step buffers with small kernels, e.g.
    ``_fill_deferred_decode_ids_kernel``): one CTA of ``HOST_DATA_WORDS``
    threads plain-stores round ``r``'s table word and data block. Params:
    ``tbl``, ``salt``, ``r``. Launched on another stream while the persistent
    kernel runs (it occupies ``HostSpec.grid`` CUs; this CTA takes a free one)."""
    _check_arch(arch)
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = HOST_DATA_WORDS
    tbl = b.param("tbl", PtrType(I32, "global"))
    salt = b.readfirstlane(b.param("salt", I32))
    r = b.readfirstlane(b.param("r", I32))
    tid = b.thread_id_x()
    one = b.const_i32(1)
    b.global_store(tbl, b.add(tid, b.const_i32(HOST_DATA_OFF)), payload_word(b, salt, r, b.add(tid, one)))
    with b.scf_if(D.is_thread0(b)):
        b.global_store(tbl, b.const_i32(0), payload_word(b, salt, r, b.const_i32(0)))
    b.ret()
    return b.kernel
