# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""In-kernel collectives of the CCL (gfx950): kernel builders.

Push model (DESIGN-CCL section 0): a producer wave pushes its unit (1 KiB:
one 16-byte store per lane) straight into every rank's receive arena (its own
included) through 64-bit global addresses; the consumer polls only its own
arena and adds the ranks' parts in rank order in f32, so every rank gets the
same bits and a numpy f32 loop reproduces them (``ref``).

Work split: unit ``u`` of an exchange belongs to global wave ``u mod (G W)``
on every rank (all ranks run the same grid), so the wave that pushes unit
``u`` on rank P is the one that consumes it there: push before poll, no
deadlock, and the slot ring needs only S = 2 (:mod:`.war` checks the rule).
The grid is at most 256 CTAs, every CTA resident (the waves of different
ranks depend on each other).

Tags: ``tag = epoch (+) use``, ``slot = (slotbase + use) mod S``; the epoch
and the slot base live in the arena's control words and the last CTA of a
launch advances them (no host reset between launches; graph-replay safe).
With ``header`` (the default) use 0 of a launch is the launch header.

Failure containment: a wave that records an error (a failed poll, or the
abort test hook) marks itself dead in LDS and never pushes or publishes again
in that launch (:meth:`_Ctx.alive`). A wave's later inputs derive only from
units it consumed itself, so every valid-tagged word a receiver can accept was
pushed by a wave whose inputs were good; the abort word a poll checks every
round makes peers fail fast but does not guard correctness.

Builders (each kernel runs ``iters`` chained collectives per launch):

* :func:`build_allreduce_kernel` (one-shot / two-shot x line / pair);
* :func:`build_allgather_kernel` (one-shot, line / pair);
* :func:`build_argmax_kernel` (vocab-parallel argmax merge, pairs).

Kernel parameters (in order): ``peer0 .. peer{N-1}`` (i64 arena bases as
mapped in this process), ``ctrl`` (i64 device address of the mapped error
record), ``arena`` (i64, this rank's own arena = ``peer{rank}``), the collective's pointers, ``stamps`` (i64[G x 3]: CTA start, end of
the first collective, end), [``htab`` (i32[8]) with ``hooks``], then i32
``rank``, ``nctas``, ``iters``, ``flips`` (bit p = the IOD-local flag copy of
arena page p is copy 1), ``dl_first`` / ``dl`` (poll deadlines in 100 MHz
ticks: first collective of a launch / the rest), and f32 ``scale`` (the
all-reduce chain's per-rank factor).
"""

from __future__ import annotations

import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rocke.core.ir import BF16, F32, I16, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from . import wire as W
from .layout import (
    DIAG_FROM,
    DIAG_N,
    DIAG_PH,
    ERR,
    GSB_WORDS,
    GSYNC_STRIPES,
    HDR_OFF,
    HDR_SRC_BYTES,
    OFF,
    STAGES,
    AllGatherSpec,
    AllReduceSpec,
    ArgmaxSpec,
    Exchange,
    Layout,
    make_layout,
    stamp_cols,
)

__all__ = [
    "HOOK_WORDS",
    "STAMP_COLS",
    "backend_for",
    "allgather_kernel_in",
    "allreduce_kernel_in",
    "argmax_kernel_in",
    "build_allgather_kernel",
    "build_allreduce_kernel",
    "build_argmax_kernel",
    "pack_kernargs",
]

STAMP_COLS = 3
HOOK_WORDS = 8  # htab: [0] drop_use, [1] abort_use, [2] sleep salt (0 = off), [3] sleep max (s_sleep 1 units),
# [4] late CTA, [5] its delay before every striped grid-sync arrival (s_sleep 1 units; 0 = off)
V4 = VectorType(I32, 4)


def backend_for(kernel: KernelDef) -> str:
    """The CCL builds use mirrored ops only (inline asm + plain IR); they are
    compiled with the Python engine explicitly (the family's choice, no
    silent fallback) and are outside the C++ byte-identity gate like the rest
    of this family's Python-engine builds."""
    return "python"


def pack_kernargs(kernel: KernelDef, values: Dict[str, object]) -> bytes:
    """Kernarg bytes in ``kernel.params`` order (natural alignment)."""
    out = bytearray()
    for p in kernel.params:
        if p.name not in values:
            raise KeyError(f"missing kernel arg {p.name!r}")
        v = values[p.name]
        t = p.type
        if isinstance(t, PtrType) or t == I64:
            fmt, size = "<Q", 8
            v = int(v) & 0xFFFFFFFFFFFFFFFF
        elif t == I32:
            fmt, size = "<I", 4
            v = int(v) & 0xFFFFFFFF
        elif t == F32:
            fmt, size = "<f", 4
            v = float(v)
        else:
            raise TypeError(f"unsupported param type {t}")
        out += b"\0" * ((-len(out)) % size)
        out += struct.pack(fmt, v)
    return bytes(out)


# ------------------------------------------------------------------ context


class _Ctx:
    """Common prologue values of a CCL kernel."""

    def __init__(self, b: IRBuilder, ranks: int, knobs, ptrs: Sequence[tuple], lay: Layout, scale: bool):
        self.b, self.N, self.kn, self.lay = b, ranks, knobs, lay
        self.peers = [b.param(f"peer{r}", I64) for r in range(ranks)]
        self.ctrl = b.param("ctrl", I64)
        # this rank's own arena (= peer{rank}); a separate kernarg so that the
        # peer-address lint taints peer*/ctrl only: local atomics (exit counter,
        # error claim) and polls go through ``arena``
        self.own = b.param("arena", I64)
        self.ptr = {name: b.param(name, PtrType(t, "global")) for name, t in ptrs}
        self.stamps = b.param("stamps", PtrType(I64, "global"))
        self.htab = b.param("htab", PtrType(I32, "global")) if knobs.hooks else None
        # CCL-LM: striped grid-sync words (device memory, GSB_WORDS i32, zero between launches)
        self.gsb = b.param("gsb", PtrType(I32, "global")) if knobs.gsync == "stripe" else None
        self.utab = b.param("utab", PtrType(I32, "global")) if knobs.place == "table" else None
        self.scols = stamp_cols(knobs)
        self.it = None  # the current collective's index (diag stamps; set by the chain loop)
        rd = b.readfirstlane
        self.rank = rd(b.param("rank", I32))
        self.nctas = rd(b.param("nctas", I32))
        self.iters = rd(b.param("iters", I32))
        self.flips = rd(b.param("flips", I32))
        self.dl_first = rd(b.param("dl_first", I32))
        self.dl = rd(b.param("dl", I32))
        self.scale = b.param("scale", F32) if scale else None
        self.bid = rd(b.block_id_x())
        self.wave = D.wave_id(b)
        self.lane = b.lane_id()
        self.W = knobs.waves
        self.gw = b.add(b.mul(self.bid, b.const_i32(self.W)), self.wave)
        self.tw = b.mul(self.nctas, b.const_i32(self.W))
        if self.utab is not None:  # place="table": this wave's unit range
            self.ubeg = rd(b.global_load_i32(self.utab, self.gw))
            self.uend = rd(b.global_load_i32(self.utab, b.add(self.gw, b.const_i32(1))))
        self.zero = b.const_i32(0)
        self.abort_addr = b.add(self.own, b.const_i64(OFF["abort"]))
        self.epoch = rd(W.load_word_sys(b, self.own, b.const_i32(OFF["epoch"])))
        self.launches = rd(W.load_word_sys(b, self.own, b.const_i32(OFF["launches"])))
        self.slotbase = rd(W.load_word_sys(b, self.own, b.const_i32(OFF["slotbase"])))
        lmod = b.mod(self.lane, b.const_i32(ranks)) if ranks > 1 else self.zero
        self.lane_src = lmod
        self.lane_peer = self.peers[0]
        for r in range(1, ranks):
            self.lane_peer = b.select(b.cmp_eq(lmod, b.const_i32(r)), self.peers[r], self.lane_peer)
        self._ivs = 0
        self.hooks = None
        if knobs.hooks:
            self.hooks = [rd(b.global_load_i32(self.htab, b.const_i32(i))) for i in range(4)]
        # per-wave "recorded an error" flag (LDS, written and read by its own wave only)
        self.dead = b.smem_alloc(I32, [self.W], name_hint="ccldead")
        with b.scf_if(b.cmp_eq(self.lane, self.zero)):
            b.smem_store_vN(self.dead, [self.wave], self.zero, 1)

    # -- helpers
    def iv(self, stem: str) -> str:
        """A fresh loop induction-variable name (LLVM value names are unique)."""
        self._ivs += 1
        return f"{stem}{self._ivs}"

    def peer_of(self, r: Value) -> Value:
        b = self.b
        p = self.peers[0]
        for k in range(1, self.N):
            p = b.select(b.cmp_eq(r, b.const_i32(k)), self.peers[k], p)
        return p

    def i64(self, v: Value) -> Value:
        return self.b.zext(v, I64)

    def stamp(self, col: int) -> None:
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            b.store(D.memrealtime(b), self.stamps, b.add(b.mul(self.bid, b.const_i32(self.scols)), b.const_i32(col)))

    def dstamp(self, ph: int) -> None:
        """``diag`` builds: thread 0 of every CTA stamps phase ``ph`` of
        collectives ``DIAG_FROM .. DIAG_FROM + DIAG_N - 1`` (column
        ``3 + (it - DIAG_FROM) DIAG_PH + ph``); no-op otherwise."""
        if not self.kn.diag or self.it is None:
            return
        b = self.b
        rel = b.sub(self.it, b.const_i32(DIAG_FROM))
        inwin = b.land(b.cmp_ge(rel, self.zero), b.cmp_lt(rel, b.const_i32(DIAG_N)))
        with b.scf_if(b.land(D.is_thread0(b), inwin)):
            col = b.add(b.const_i32(3 + ph), b.mul(rel, b.const_i32(DIAG_PH)))
            b.store(D.memrealtime(b), self.stamps, b.add(b.mul(self.bid, b.const_i32(self.scols)), col))

    def local_copy(self, ex: Exchange) -> Value:
        """IOD-local flag copy index (0 / 1) of ``ex`` (dual homing)."""
        b = self.b
        if ex.copies == 1:
            return self.zero
        page = self.lay.flag_page(ex.name)
        flip = b.land(b.lshr(self.flips, b.const_i32(page)), b.const_i32(1))
        return b.xor(M.iod_id(b), flip)

    def flag_off(self, ex: Exchange, slot: Value, src: Value, unit: Value, copy: Value) -> Value:
        b = self.b
        i = b.add(b.mul(b.add(b.mul(slot, b.const_i32(ex.srcs)), src), b.const_i32(ex.units)), unit)
        if ex.copies == 1:
            return b.add(b.const_i32(ex.flags), b.shl(i, b.const_i32(2)))
        blk = b.shl(b.lshr(i, b.const_i32(11)), b.const_i32(14))
        within = b.shl(b.land(i, b.const_i32(2047)), b.const_i32(2))
        return b.add(b.add(b.const_i32(ex.flags), b.add(blk, within)), b.mul(copy, b.const_i32(8192)))

    def payload_off(self, ex: Exchange, slot: Value, src: Value, unit: Value) -> Value:
        b = self.b
        i = b.add(b.mul(b.add(b.mul(slot, b.const_i32(ex.srcs)), src), b.const_i32(ex.units)), unit)
        return b.add(b.const_i32(ex.payload), b.add(b.shl(i, b.const_i32(10)), b.shl(self.lane, b.const_i32(4))))

    def deadline(self, it: Value) -> Value:
        b = self.b
        d = b.select(b.cmp_eq(it, self.zero), self.dl_first, self.dl)
        return b.add(D.memrealtime(b), self.i64(d))

    def units_loop(self, units: int):
        """``scf_for`` over this wave's units ``u = gw + k * tw < units``;
        yields the loop builder, the caller computes ``u``."""
        b = self.b
        left = b.sub(b.const_i32(units), self.gw)
        nk = b.select(
            b.cmp_gt(left, self.zero),
            b.div(b.add(left, b.sub(self.tw, b.const_i32(1))), self.tw),
            self.zero,
        )
        return b.scf_for(self.zero, nk, b.const_i32(1), iv_name=self.iv("cclk"))

    def unit(self, k: Value) -> Value:
        return self.b.add(self.gw, self.b.mul(k, self.tw))

    def slot(self, use: Value) -> Value:
        b = self.b
        return b.mod(b.add(self.slotbase, use), b.const_i32(self.kn.slots))

    # -- failure containment
    def alive(self) -> Value:
        """Uniform i1: this wave has not recorded an error in this launch.
        Every push and flag store is guarded by it (mutant ``deadpush``: not
        guarded, the must-fail control of the containment test)."""
        b = self.b
        if self.kn.mutant == "deadpush":
            return b.cmp_eq(self.zero, self.zero)
        v = b.vec_extract(b.smem_load_vN(self.dead, self.wave, dtype=I32, n=1), 0)
        return b.cmp_eq(b.readfirstlane(v), self.zero)

    def mark_dead(self) -> None:
        b = self.b
        with b.scf_if(b.cmp_eq(self.lane, self.zero)):
            b.smem_store_vN(self.dead, [self.wave], b.const_i32(1), 1)

    def guard(self, drop: Value) -> Value:
        """Push / publish condition: not a dropped use (test hook) and alive."""
        return self.b.land(self.b.lnot(drop), self.alive())

    # -- hooks
    def delay(self, use: Value, point: int) -> None:
        if self.hooks is None:
            return
        b = self.b
        salt, mx = self.hooks[2], self.hooks[3]
        with b.scf_if(b.cmp_ne(salt, self.zero)):
            # xorshift32 over (salt, wave, use, point): shifts and xors only (no
            # signed-overflow arithmetic); host twin ``ref.delay_units``
            h = b.xor(salt, b.xor(b.shl(self.gw, b.const_i32(12)), b.shl(use, b.const_i32(2))))
            h = b.xor(h, b.const_i32(point))
            for sh, left in ((13, True), (17, False), (5, True), (13, True), (17, False), (5, True)):
                h = b.xor(h, b.shl(h, b.const_i32(sh)) if left else b.lshr(h, b.const_i32(sh)))
            n = b.mod(b.land(h, b.const_i32(0x7FFFFFFF)), b.add(mx, b.const_i32(1)))
            W.sleep_units(b, b.readfirstlane(n), self.iv("cclsl"))

    def dropped(self, use: Value) -> Value:
        b = self.b
        if self.hooks is None:
            return b.cmp_ne(self.zero, self.zero)
        return b.cmp_eq(use, self.hooks[0])

    def abort_hook(self, use: Value) -> None:
        if self.hooks is None:
            return
        b = self.b
        hit = b.land(b.cmp_eq(use, self.hooks[1]), b.land(b.cmp_eq(self.bid, self.zero), b.cmp_eq(self.wave, self.zero)))
        with b.scf_if(hit):
            self.mark_dead()
            with b.scf_if(b.cmp_eq(self.lane, self.zero)):
                self.record(b.const_i32(ERR["PEER_TIMEOUT"]), STAGES["hook"], use, b.const_i32(-1), b.const_i32(0), b.const_i32(0))

    # -- errors
    def record(self, code: Value, stage: int, use: Value, peer: Value, tag: Value, seen: Value) -> None:
        """Lane-level (call under a one-lane guard): claim the rank's first
        error, write the record to ``ctrl``, push ABORT to every arena."""
        b = self.b
        old = D.atomic_add_scoped(b, self.own, b.const_i32(OFF["claim"]), b.const_i32(1), scope="agent", returns=True)
        with b.scf_if(b.cmp_eq(old, self.zero)):
            rt = D.memrealtime(b)
            words = [
                (1, code),
                (2, b.const_i32(stage)),
                (3, use),
                (4, peer),
                (5, tag),
                (6, seen),
                (7, self.bid),
                (8, b.trunc(rt, I32)),
                (9, b.trunc(b.lshr(rt, b.const_i64(32)), I32)),
                (10, self.rank),
            ]
            for i, v in words:
                W.store_word_sys(b, self.ctrl, b.const_i32(4 * i), v)
            W.drain(b)
            W.store_word_sys(b, self.ctrl, self.zero, b.const_i32(1))
        for p in self.peers:
            W.store_word_sys(b, p, b.const_i32(OFF["abort"]), b.const_i32(1))
        W.drain(b)

    def check(self, st: Value, mask: Value, seen: Value, stage: int, use: Value, tag: Value, src_from_lane: bool) -> None:
        """Record a failed poll (``st`` uniform)."""
        b = self.b
        with b.scf_if(b.cmp_ne(st, self.zero)):
            fl = b.inline_asm("s_ff1_i32_b64 $0, $1", "=s,s", [mask], I32, result_name_hint="ff1")
            fl = b.select(b.cmp_lt(fl, self.zero), self.zero, fl)
            seen_l = b.readlane(seen, fl)
            peer = b.mod(fl, b.const_i32(self.N)) if src_from_lane and self.N > 1 else fl
            code = b.select(
                b.cmp_eq(st, b.const_i32(W.ST_TIMEOUT)),
                b.const_i32(ERR["PEER_TIMEOUT"]),
                b.select(b.cmp_eq(st, b.const_i32(W.ST_ABORT)), b.const_i32(ERR["PEER_ABORT"]), b.const_i32(ERR["TAG_AHEAD"])),
            )
            self.mark_dead()
            with b.scf_if(b.cmp_eq(self.lane, self.zero)):
                self.record(code, stage, use, peer, tag, seen_l)

    # -- epilogue
    def finish(self, uses: Value) -> None:
        """Barrier, stamp, and (last CTA) advance epoch / slot base / launch
        count for the next launch."""
        b = self.b
        b.sync()
        self.stamp(2)
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, self.own, b.const_i32(OFF["exit"]), b.const_i32(1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, b.sub(self.nctas, b.const_i32(1)))):
                nxt = W.tag_add(b, self.epoch, uses)
                sb = b.mod(b.add(self.slotbase, uses), b.const_i32(self.kn.slots))
                W.store_word_sys(b, self.own, b.const_i32(OFF["exit"]), self.zero)
                if self.kn.sync == "grid":
                    W.store_word_sys(b, self.own, b.const_i32(OFF["gsync"]), self.zero)
                if self.gsb is not None:
                    gsa = b.readfirstlane(b.global_ptr_to_i64(self.gsb))
                    with b.scf_for(self.zero, b.const_i32(GSB_WORDS), b.const_i32(1), iv_name="cclgz") as k:
                        _st_dev(b, gsa, b.shl(k, b.const_i32(2)), self.zero)
                W.store_word_sys(b, self.own, b.const_i32(OFF["epoch"]), nxt)
                W.store_word_sys(b, self.own, b.const_i32(OFF["slotbase"]), sb)
                W.store_word_sys(b, self.own, b.const_i32(OFF["launches"]), b.add(self.launches, b.const_i32(1)))
                W.drain(b)

    def grid_sync(self, it: Value, use: Value) -> None:
        """Grid-wide barrier after collective ``it`` (``sync="grid"``): CTA
        barrier, thread 0 adds one to the arena's ``gsync`` counter (system
        scope) and waits for ``(it + 1) x nctas`` (deadline; GRID_SYNC_TIMEOUT),
        CTA barrier. The last CTA of the launch zeroes the counter."""
        b = self.b
        if self.kn.gsync == "stripe":
            self._grid_sync_stripe(it, use)
            return
        # uniform operands computed outside the one-lane region (inside it, LLVM moves the loop
        # counter's chain to VALU and the SGPR asm operands of the chain stop compiling)
        target = b.readfirstlane(b.mul(b.add(it, b.const_i32(1)), self.nctas))
        b.sync()
        self.dstamp(5)
        dl = self.deadline(it)
        with b.scf_if(D.is_thread0(b)):
            D.atomic_add_scoped(b, self.own, b.const_i32(OFF["gsync"]), b.const_i32(1), scope="system", returns=True)
            st = W.poll_count_ge(b, self.own, b.const_i32(OFF["gsync"]), target, dl, pause=self.kn.pause)
            with b.scf_if(b.cmp_ne(st, self.zero)):
                self.record(b.const_i32(ERR["GRID_SYNC_TIMEOUT"]), STAGES["gsync"], use, b.const_i32(-1), target, self.zero)
        self.dstamp(6)
        b.sync()
        self.dstamp(7)

    def _grid_sync_stripe(self, it: Value, use: Value) -> None:
        """``gsync="stripe"``: CTA barrier; thread 0 adds one (agent scope) to
        stripe ``bid mod 64`` of ``gsb`` (device memory); wave 0 waits until
        every stripe ``l`` reaches ``(it + 1) x`` (its CTA count), one lane per
        stripe (P4's striped sync). Mutant ``gsearly`` waits for one arrival
        less on stripe 0. ``hooks`` builds check the release: every CTA stores
        ``it + 1`` into its word ``gsb[64 + bid]`` before it arrives and, once
        released, reads its successor's word: a smaller value (the successor
        has not arrived) records GRID_SYNC_TIMEOUT with ``tag_seen`` = that value;
        ``htab[4]`` / ``htab[5]`` make one CTA arrive late (so an early release
        is caught deterministically)."""
        b = self.b
        S = GSYNC_STRIPES
        gsa = b.readfirstlane(b.global_ptr_to_i64(self.gsb))
        itp = b.readfirstlane(b.add(it, b.const_i32(1)))
        lane = self.lane
        cnt = b.select(
            b.cmp_lt(lane, self.nctas),
            b.add(b.lshr(b.sub(b.sub(self.nctas, b.const_i32(1)), lane), b.const_i32(6)), b.const_i32(1)),
            self.zero,
        )
        tgt = b.mul(itp, cnt)
        if self.kn.mutant == "gsearly":
            tgt = b.select(b.land(b.cmp_eq(lane, self.zero), b.cmp_gt(tgt, self.zero)), b.sub(tgt, b.const_i32(1)), tgt)
        sidx = b.readfirstlane(b.land(self.bid, b.const_i32(S - 1)))
        b.sync()
        self.dstamp(5)
        dl = self.deadline(it)
        if self.kn.hooks:
            late = b.readfirstlane(b.global_load_i32(self.htab, b.const_i32(4)))
            lnum = b.readfirstlane(b.global_load_i32(self.htab, b.const_i32(5)))
            lnum = b.readfirstlane(b.select(b.cmp_eq(self.bid, late), lnum, self.zero))
        with b.scf_if(D.is_thread0(b)):
            if self.kn.hooks:
                W.sleep_units(b, lnum, self.iv("cclgl"))
                _st_dev(b, gsa, b.shl(b.add(self.bid, b.const_i32(S)), b.const_i32(2)), itp)
                W.drain(b)
            D.atomic_add_scoped(b, gsa, b.shl(sidx, b.const_i32(2)), b.const_i32(1), scope="agent")
        with b.scf_if(b.cmp_eq(self.wave, self.zero)):
            st = W.poll_stripes(b, gsa, b.shl(lane, b.const_i32(2)), tgt, dl, pause=self.kn.pause)
            with b.scf_if(b.land(b.cmp_ne(st, self.zero), b.cmp_eq(lane, self.zero))):
                self.record(b.const_i32(ERR["GRID_SYNC_TIMEOUT"]), STAGES["gsync"], use, b.const_i32(-1), itp, self.zero)
        self.dstamp(6)
        if self.kn.hooks:
            nxt = b.readfirstlane(b.mod(b.add(self.bid, b.const_i32(1)), self.nctas))
            with b.scf_if(D.is_thread0(b)):
                seen = _ld_dev(b, gsa, b.shl(b.add(nxt, b.const_i32(S)), b.const_i32(2)))
                with b.scf_if(b.cmp_lt(seen, itp)):
                    self.record(b.const_i32(ERR["GRID_SYNC_TIMEOUT"]), STAGES["gsync"], use, nxt, itp, seen)
        b.sync()
        self.dstamp(7)

    # -- pair-poll operands
    def pair_ops(self, tag: Value):
        b = self.b
        t64 = self.i64(tag)
        neg = b.sub(b.const_i64(0), b.add(t64, b.const_i64(1)))
        return t64, neg, b.const_i64(self.kn.slots)

    def lane_off16(self) -> Value:
        return self.b.shl(self.lane, self.b.const_i32(4))


# ------------------------------------------------------------- data helpers


def _st_dev(b: IRBuilder, base: Value, voff: Value, v: Value) -> None:
    """``global_store_dword ... sc1`` to local device memory (``gsb``), not waited."""
    b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s,~{memory}", [voff, v, base], None)


def _ld_dev(b: IRBuilder, base: Value, voff: Value) -> Value:
    """``global_load_dword ... sc1`` of local device memory, waited."""
    return b.inline_asm("global_load_dword $0, $1, $2 sc1\ns_waitcnt vmcnt(0)", "=&v,v,s,~{memory}", [voff, base], I32, result_name_hint="lddev")


def _f32_of_dwords(b: IRBuilder, dwords: List[Value], dtype: str) -> List[Value]:
    """Wire dwords -> f32 values (bf16: two per dword, low half first)."""
    if dtype == "f32":
        return [b.bitcast(d, F32) for d in dwords]
    out = []
    for d in dwords:
        out.append(b.bitcast(b.shl(d, b.const_i32(16)), F32))
        out.append(b.bitcast(b.shl(b.lshr(d, b.const_i32(16)), b.const_i32(16)), F32))
    return out


def _bf16_bits(b: IRBuilder, x: Value) -> Value:
    return b.zext(b.bitcast(b.trunc_f32_to_bf16(x), I16), I32)


def _dwords_of_f32(b: IRBuilder, vals: List[Value], dtype: str) -> List[Value]:
    """f32 values -> wire dwords (bf16: round to nearest even, pack pairs)."""
    if dtype == "f32":
        return [b.bitcast(v, I32) for v in vals]
    return [b.lor(_bf16_bits(b, vals[2 * i]), b.shl(_bf16_bits(b, vals[2 * i + 1]), b.const_i32(16))) for i in range(len(vals) // 2)]


def _rank_sum(b: IRBuilder, parts: List[List[Value]], reverse: bool = False) -> List[Value]:
    order = list(range(len(parts)))
    if reverse:
        order.reverse()
    acc = list(parts[order[0]])
    for r in order[1:]:
        acc = [b.fadd(a, x) for a, x in zip(acc, parts[r])]
    return acc


def _vec_dwords(b: IRBuilder, v: Value, n: int) -> List[Value]:
    return [b.vec_extract(v, i) for i in range(n)]


# ---------------------------------------------------------------- all-reduce


def allreduce_kernel_in(spec: AllReduceSpec, lay: Optional[Layout] = None, ex_index: int = 0) -> KernelDef:
    """Chained all-reduce. Params (after the common prefix): ``src`` i32*
    (rows x cols wire dtype, this rank's partial of the first collective),
    ``dst`` f32* (rows x cols, the latest result), ``xbuf`` i32* (scratch:
    the next collective's input). Collective i + 1's input on rank r is
    ``wire(out_i * scale)`` (host passes ``scale_r``).

    ``knobs.carry="auto"``: when every wave owns at most one unit (``units <=
    grid() x waves``; launch with ``nctas = spec.grid()``) the chain carries
    the next input in registers (as a producer epilogue would) instead of
    storing and reloading ``xbuf``; ``"mem"`` always goes through ``xbuf``."""
    kn = spec.knobs
    lay = lay or make_layout([spec])
    if kn.lm_path:
        return _allreduce_lm_kernel(spec, lay, ex_index)
    N = spec.ranks
    if kn.mutant == "slot" and kn.form != "line":
        raise ValueError("mutant='slot' is a line-form mutant")
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    c = _Ctx(b, N, kn, [("src", I32), ("dst", F32), ("xbuf", I32)], lay, scale=True)
    src_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["src"]))
    xbuf_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["xbuf"]))
    ld = spec.lane_dwords  # 4 (line) / 2 (pair)
    hdr = 1 if kn.header else 0
    reg = spec.reg_carry
    c.stamp(0)
    go = b.const_i32(1)
    if kn.header:
        _header(c, spec, b.add(c.iters, b.const_i32(hdr)))
        go = b.readfirstlane(b.select(b.cmp_eq(W.load_word_sys(b, c.own, b.const_i32(OFF["abort"])), c.zero), b.const_i32(1), c.zero))
    if kn.ar == "oneshot":
        exs = (lay.ex(f"{ex_index}.ar"),)
    else:
        exs = (lay.ex(f"{ex_index}.rs"), lay.ex(f"{ex_index}.ag"))
    body = _ar_oneshot if kn.ar == "oneshot" else _ar_twoshot

    def iteration(it, xv):
        c.it = it
        c.dstamp(0)
        use = b.add(it, b.const_i32(hdr))
        tag = W.tag_add(b, c.epoch, use)
        slot = c.slot(use)
        dl = c.deadline(it)
        first = b.cmp_eq(it, c.zero)
        c.abort_hook(use)
        if reg:

            def src(u):
                return xv

        else:
            xin = b.select(first, src_a, xbuf_a)

            def src(u):
                return _load_x(c, xin, u, ld, spec.unit_dwords)

        nxt = body(c, spec, *exs, use, tag, slot, dl, src, ld, reg)
        with b.scf_if(first):
            c.stamp(1)
        return nxt

    if reg:
        with b.scf_if(b.land(b.cmp_ne(go, c.zero), b.cmp_lt(c.gw, b.const_i32(spec.units)))):
            x0 = _load_x(c, src_a, c.gw, ld, spec.unit_dwords)
            loop = b.scf_for_iter(c.zero, c.iters, b.const_i32(1), [("cclxv", x0)], iv_name="cclit")
            with loop as (it, carried):
                b.scf_yield(iteration(it, carried[0]))
    else:
        with b.scf_if(b.cmp_ne(go, c.zero)):
            with b.scf_for(c.zero, c.iters, b.const_i32(1), iv_name="cclit") as it:
                iteration(it, None)
                if kn.sync == "grid":
                    c.grid_sync(it, b.add(it, b.const_i32(hdr)))
    c.finish(b.add(c.iters, b.const_i32(hdr)))
    b.ret()
    return b.kernel


@contextmanager
def _units(c: _Ctx, units: int, reg: bool):
    """The units of this wave: ``gw`` alone in register-carry mode (the
    caller guards ``gw < units``), else the strided ``units_loop``."""
    if reg:
        yield c.gw
    else:
        with c.units_loop(units) as k:
            yield c.unit(k)


def _load_x(c: _Ctx, xin: Value, u: Value, ld: int, ud: int) -> Value:
    """This lane's ``ld`` input dwords of unit ``u`` as a vector."""
    b = c.b
    voff = b.shl(b.add(b.mul(u, b.const_i32(ud)), b.mul(c.lane, b.const_i32(ld))), b.const_i32(2))
    return W.load_v4_local(b, xin, voff) if ld == 4 else W.load_v2_local(b, xin, voff)


def _push_data(c: _Ctx, x: Value, tag: Value, ld: int, form: str) -> Value:
    """The 16 bytes a lane stores: the line payload, or two tagged pairs."""
    b = c.b
    if form == "line":
        return x
    return b.vec_pack([b.vec_extract(x, 0), tag, b.vec_extract(x, 1), tag], I32)


def _consume_store(c: _Ctx, spec, u: Value, out_f32: List[Value], ld: int, ud: int, reg: bool) -> Optional[Value]:
    """Store this lane's f32 results (``dst``); the next input goes to
    ``xbuf`` (memory carry) or is returned (register carry)."""
    b = c.b
    d0 = b.add(b.mul(u, b.const_i32(ud)), b.mul(c.lane, b.const_i32(ld)))  # first wire dword
    nf = len(out_f32)
    f0 = d0 if spec.dtype == "f32" else b.shl(d0, b.const_i32(1))
    b.global_store_vN(c.ptr["dst"], f0, b.vec_pack(out_f32, F32), nf)
    nxt = b.vec_pack(_dwords_of_f32(b, [b.fmul(v, c.scale) for v in out_f32], spec.dtype), I32)
    if reg:
        return nxt
    b.global_store_vN(c.ptr["xbuf"], d0, nxt, ld)
    return None


def _parts_from_lines(c: _Ctx, vecs: List[Value], ld: int, dtype: str) -> List[List[Value]]:
    return [_f32_of_dwords(c.b, _vec_dwords(c.b, v, ld), dtype) for v in vecs]


def _parts_from_pairs(c: _Ctx, pairs: List[Value], dtype: str) -> List[List[Value]]:
    b = c.b
    n = len(pairs) // 2
    return [_f32_of_dwords(b, [b.trunc(pairs[2 * r], I32), b.trunc(pairs[2 * r + 1], I32)], dtype) for r in range(n)]


def _publish(c: _Ctx, ex: Exchange, slot: Value, src: Value, u: Value, tag: Value, nlanes: int, peer_addr: Value, copy_of_lane: Value) -> None:
    """Flag store under a lane guard: lanes < ``nlanes`` store ``tag`` at
    ``peer_addr + flag_off(slot, src, u, copy_of_lane)``."""
    b = c.b
    with b.scf_if(b.cmp_lt(c.lane, b.const_i32(nlanes))):
        off = c.flag_off(ex, slot, src, u, copy_of_lane)
        W.store_flags_lanes(b, b.add(peer_addr, c.i64(off)), tag)


def _ar_oneshot(c: _Ctx, spec: AllReduceSpec, ex: Exchange, use, tag, slot, dl, src, ld, reg) -> Optional[Value]:
    b, kn, N = c.b, spec.knobs, spec.ranks
    ud = spec.unit_dwords
    line = kn.form == "line"
    drop = c.dropped(use)
    # push
    with _units(c, spec.units, reg) as u:
        c.delay(use, 0)
        x = src(u)
        with b.scf_if(c.guard(drop)):
            W.push_v4(b, c.peers, c.payload_off(ex, slot, c.rank, u), _push_data(c, x, tag, ld, kn.form), kn.push_cpol)
    c.dstamp(1)
    if line:
        W.drain(b)
        c.dstamp(2)
        copies = ex.copies
        cl = b.div(c.lane, b.const_i32(N)) if copies == 2 else c.zero
        with b.scf_if(c.guard(drop)):
            with _units(c, spec.units, reg) as u:
                _publish(c, ex, slot, c.rank, u, tag, N * copies, c.lane_peer, cl)
        c.dstamp(3)
    # consume
    pslot = b.mod(b.add(slot, b.const_i32(1)), b.const_i32(kn.slots)) if kn.mutant == "slot" else slot
    nxt = None
    with _units(c, spec.units, reg) as u:
        c.delay(use, 1)
        if line:
            lc = c.local_copy(ex)
            faddr = b.add(c.own, c.i64(c.flag_off(ex, slot, c.lane_src, u, lc)))
            seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol)
            c.check(st, mask, seen, STAGES["ar"], use, tag, True)
            vecs = W.load_lines(b, c.own, [c.payload_off(ex, pslot, b.const_i32(r), u) for r in range(N)], poll_cpol=kn.poll_cpol)
            parts = _parts_from_lines(c, vecs, ld, spec.dtype)
        else:
            t64, neg, win = c.pair_ops(tag)
            pairs, mask, st = W.poll_pairs(
                b, c.own, [c.payload_off(ex, slot, b.const_i32(r), u) for r in range(N)], t64, neg, win, dl, c.abort_addr,
                pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol,
            )  # fmt: skip
            c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["ar"], use, tag, False)
            parts = _parts_from_pairs(c, pairs, spec.dtype)
        s = _rank_sum(b, parts)
        if kn.mutant == "revsum":
            rv = _rank_sum(b, parts, reverse=True)
            r0 = b.cmp_eq(c.rank, c.zero)
            s = [b.select(r0, x, y) for x, y in zip(rv, s)]
        nxt = _consume_store(c, spec, u, s, ld, ud, reg)
    c.dstamp(4)
    return nxt


def _ar_twoshot(c: _Ctx, spec: AllReduceSpec, rs: Exchange, ag: Exchange, use, tag, slot, dl, src, ld, reg) -> Optional[Value]:
    b, kn, N = c.b, spec.knobs, spec.ranks
    ud = spec.unit_dwords
    line = kn.form == "line"
    drop = c.dropped(use)
    Nc = b.const_i32(N)
    # A: partial of unit u -> its owner (u mod N), rs[slot][rank][u]
    with _units(c, spec.units, reg) as u:
        c.delay(use, 0)
        x = src(u)
        owner = b.readfirstlane(b.mod(u, Nc))
        with b.scf_if(c.guard(drop)):
            W.push_v4(b, [c.peer_of(owner)], c.payload_off(rs, slot, c.rank, u), _push_data(c, x, tag, ld, kn.form), kn.push_cpol)
    c.dstamp(1)
    if line:
        W.drain(b)
        with b.scf_if(c.guard(drop)):
            with _units(c, spec.units, reg) as u:
                owner = b.readfirstlane(b.mod(u, Nc))
                _publish(c, rs, slot, c.rank, u, tag, rs.copies, c.peer_of(owner), c.lane)
    c.dstamp(2)
    # B: owner sums its units in rank order, rounds to the wire dtype, pushes to all (ag[slot][0][u])
    with _units(c, spec.units, reg) as u:
        mine = b.cmp_eq(b.mod(u, Nc), c.rank)
        with b.scf_if(mine):
            if line:
                lc = c.local_copy(rs)
                faddr = b.add(c.own, c.i64(c.flag_off(rs, slot, c.lane_src, u, lc)))
                seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol)
                c.check(st, mask, seen, STAGES["rs"], use, tag, True)
                vecs = W.load_lines(b, c.own, [c.payload_off(rs, slot, b.const_i32(r), u) for r in range(N)], poll_cpol=kn.poll_cpol)
                parts = _parts_from_lines(c, vecs, ld, spec.dtype)
            else:
                t64, neg, win = c.pair_ops(tag)
                pairs, mask, st = W.poll_pairs(
                    b, c.own, [c.payload_off(rs, slot, b.const_i32(r), u) for r in range(N)], t64, neg, win, dl,
                    c.abort_addr, pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol,
                )  # fmt: skip
                c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["rs"], use, tag, False)
                parts = _parts_from_pairs(c, pairs, spec.dtype)
            y = _dwords_of_f32(b, _rank_sum(b, parts), spec.dtype)
            yv = b.vec_pack(y, I32)
            # an owner whose reduce-scatter poll failed must not publish its (garbage) sum
            with b.scf_if(c.alive()):
                W.push_v4(b, c.peers, c.payload_off(ag, slot, c.zero, u), _push_data(c, yv, tag, ld, kn.form), kn.push_cpol)
    if line:
        W.drain(b)
        cl = b.div(c.lane, Nc) if ag.copies == 2 else c.zero
        with _units(c, spec.units, reg) as u:
            with b.scf_if(b.land(b.cmp_eq(b.mod(u, Nc), c.rank), c.alive())):
                _publish(c, ag, slot, c.zero, u, tag, N * ag.copies, c.lane_peer, cl)
    c.dstamp(3)
    # C: everyone reads the reduced unit
    nxt = None
    with _units(c, spec.units, reg) as u:
        c.delay(use, 1)
        if line:
            lc = c.local_copy(ag)
            faddr = b.add(c.own, c.i64(c.flag_off(ag, slot, c.zero, u, lc)))
            seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol)
            c.check(st, mask, seen, STAGES["ag"], use, tag, False)
            (vec,) = W.load_lines(b, c.own, [c.payload_off(ag, slot, c.zero, u)], poll_cpol=kn.poll_cpol)
            out = _f32_of_dwords(b, _vec_dwords(b, vec, ld), spec.dtype)
        else:
            t64, neg, win = c.pair_ops(tag)
            pairs, mask, st = W.poll_pairs(
                b, c.own, [c.payload_off(ag, slot, c.zero, u)], t64, neg, win, dl, c.abort_addr, pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol
            )
            c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["ag"], use, tag, False)
            out = _parts_from_pairs(c, pairs, spec.dtype)[0]
        nxt = _consume_store(c, spec, u, out, ld, ud, reg)
    c.dstamp(4)
    return nxt


def _header(c: _Ctx, spec, uses: Value) -> None:
    """Launch header (use 0): CTA 0 wave 0 broadcasts (rows, cols | dtype,
    uses, schedule hash) as 4 tagged pairs; wave 0 of every CTA polls every
    rank's header and records ``CCL_MISMATCH`` on a difference. Every
    collective (all-reduce, all-gather, argmax) runs it unless ``header`` is
    off; the caller skips its work when the abort word is set afterwards."""
    b, kn, N = c.b, spec.knobs, spec.ranks
    tag = W.tag_add(b, c.epoch, c.zero)
    hslot = b.land(c.launches, b.const_i32(1))
    cols, dtype = getattr(spec, "cols", 1), getattr(spec, "dtype", "f32")
    vals = [
        b.const_i32(spec.rows),
        b.const_i32(cols | ((1 if dtype == "bf16" else 2) << 24)),
        uses,
        b.const_i32(spec.header_hash()),
    ]
    half = b.land(c.lane, b.const_i32(1))
    mine0 = b.select(b.cmp_eq(half, c.zero), vals[0], vals[2])
    mine1 = b.select(b.cmp_eq(half, c.zero), vals[1], vals[3])

    def off(src):
        return b.add(b.const_i32(HDR_OFF), b.add(b.mul(b.add(b.mul(hslot, b.const_i32(8)), src), b.const_i32(HDR_SRC_BYTES)), b.shl(half, b.const_i32(4))))

    w0 = b.cmp_eq(c.wave, c.zero)
    with b.scf_if(b.land(b.land(b.cmp_eq(c.bid, c.zero), w0), b.cmp_lt(c.lane, b.const_i32(2)))):
        W.push_v4(b, c.peers, off(c.rank), b.vec_pack([mine0, tag, mine1, tag], I32), kn.push_cpol)
    with b.scf_if(w0):
        dl = b.add(D.memrealtime(b), c.i64(c.dl_first))
        t64, neg, win = c.pair_ops(tag)
        pairs, mask, st = W.poll_pairs(b, c.own, [off(b.const_i32(r)) for r in range(N)], t64, neg, win, dl, c.abort_addr, pause=kn.pause, poll_cpol=kn.poll_cpol)
        c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["header"], c.zero, tag, False)
        bad = b.cmp_ne(c.zero, c.zero)
        for r in range(N):
            bad = b.lor(bad, b.cmp_ne(b.trunc(pairs[2 * r], I32), mine0))
            bad = b.lor(bad, b.cmp_ne(b.trunc(pairs[2 * r + 1], I32), mine1))
        anybad = b.wave_any(b.zext(b.land(bad, b.cmp_lt(c.lane, b.const_i32(2))), I32))
        with b.scf_if(b.land(b.cmp_eq(st, c.zero), b.cmp_ne(anybad, c.zero))):
            with b.scf_if(b.cmp_eq(c.lane, c.zero)):
                c.record(b.const_i32(ERR["CCL_MISMATCH"]), STAGES["header"], c.zero, b.const_i32(-1), tag, c.zero)
    b.sync()


# ------------------------------------------------------- CCL-LM all-reduce
#
# The large-message levers of ``layout.CclKnobs`` (lamport, unit_kib, ar_dtype,
# publish, self_local). Builds with none of them take the code above unchanged.

# flag-store targets of the lanes that publish nothing (counted publish): one word per (wave slot, lane) in the unused
# arena bytes between the launch-header pairs and the first exchange ([1536, 16384)), so the dummy stores of
# different waves never hit one address (a single shared word serialised them: measured)
JUNK_BASE = 2048
JUNK_WAVES = 32


@contextmanager
def _lm_units(c: _Ctx, spec: AllReduceSpec, reg: bool):
    """The units of this wave: the ``utab`` range (``place="table"``) or :func:`_units`."""
    if c.utab is not None:
        with c.b.scf_for(c.ubeg, c.uend, c.b.const_i32(1), iv_name=c.iv("cclt")) as u:
            yield u
    else:
        with _units(c, spec.units, reg) as u:
            yield u


def _lm_payload_off(c: _Ctx, ex: Exchange, slot: Value, src: Value, unit: Value) -> Value:
    """Byte offset of this lane's first vector of ``unit`` from ``src`` in
    exchange ``ex`` (vector ``v`` of the unit at ``+ 1024 v``)."""
    b = c.b
    i = b.add(b.mul(b.add(b.mul(slot, b.const_i32(ex.srcs)), src), b.const_i32(ex.units)), unit)
    sh = {1024: 10, 2048: 11, 4096: 12}[ex.unit_bytes]
    return b.add(b.const_i32(ex.payload), b.add(b.shl(i, b.const_i32(sh)), b.shl(c.lane, b.const_i32(4))))


class _LM:
    """Geometry of a CCL-LM all-reduce build: ``V`` = ``unit_kib`` 16-byte
    wire vectors per lane per unit (vector ``v`` covers wire dwords
    ``u ud + 256 v + 4 lane ..``), ``ratio`` input vectors per wire vector
    (2: f32 data with bf16 partials)."""

    def __init__(self, c: _Ctx, spec: AllReduceSpec):
        self.c, self.spec = c, spec
        self.V = spec.knobs.unit_kib
        self.wdt = spec.wire_dtype
        self.ratio = 2 if (spec.dtype == "f32" and self.wdt == "bf16") else 1
        self.ud = spec.unit_dwords

    def wire_d0(self, u: Value, v: int) -> Value:
        b, c = self.c.b, self.c
        return b.add(b.mul(u, b.const_i32(self.ud)), b.add(b.const_i32(256 * v), b.shl(c.lane, b.const_i32(2))))

    def in_dw(self, u: Value) -> List[Value]:
        """Dword indices (data dtype) of this lane's input vectors of unit ``u``."""
        b = self.c.b
        out = []
        for v in range(self.V):
            d = self.wire_d0(u, v)
            if self.ratio == 1:
                out.append(d)
            else:
                d2 = b.shl(d, b.const_i32(1))
                out += [d2, b.add(d2, b.const_i32(4))]
        return out

    def load_x(self, xin: Value, u: Value) -> List[Value]:
        b = self.c.b
        return W.load_vecs_local(b, xin, [b.shl(d, b.const_i32(2)) for d in self.in_dw(u)])

    def wire(self, xs: Sequence[Value]) -> List[Value]:
        """Input vectors -> the ``V`` wire vectors (f32 data with bf16 partials:
        round to nearest even, pack pairs low half first)."""
        b = self.c.b
        if self.ratio == 1:
            return list(xs)
        out = []
        for v in range(self.V):
            f = [b.bitcast(b.vec_extract(xs[2 * v + h], i), F32) for h in range(2) for i in range(4)]
            out.append(b.vec_pack(_dwords_of_f32(b, f, "bf16"), I32))
        return out

    def to_f32(self, vec: Value) -> List[Value]:
        b = self.c.b
        return _f32_of_dwords(b, _vec_dwords(b, vec, 4), self.wdt)

    def encode(self, vec: Value) -> Value:
        """Lamport: map the pattern that would read as empty (bf16 ``0xFFFF``,
        f32 ``0xFFFFFFFF``) to the quiet NaN, then complement (mutant
        ``nocanon`` skips the map)."""
        b, kn = self.c.b, self.spec.knobs
        ones = b.const_i32(-1)
        out = []
        for i in range(4):
            d = b.vec_extract(vec, i)
            if kn.mutant != "nocanon":
                if self.wdt == "bf16":
                    m16 = b.const_i32(0xFFFF)
                    lo = b.land(d, m16)
                    hi = b.lshr(d, b.const_i32(16))
                    lo = b.select(b.cmp_eq(lo, m16), b.const_i32(0x7FC0), lo)
                    hi = b.select(b.cmp_eq(hi, m16), b.const_i32(0x7FC0), hi)
                    d = b.lor(lo, b.shl(hi, b.const_i32(16)))
                else:
                    d = b.select(b.cmp_eq(d, ones), b.const_i32(0x7FC00000), d)
            out.append(b.xor(d, ones))
        return b.vec_pack(out, I32)

    def decode(self, vec: Value) -> Value:
        b = self.c.b
        ones = b.const_i32(-1)
        return b.vec_pack([b.xor(b.vec_extract(vec, i), ones) for i in range(4)], I32)

    def store(self, u: Value, outs: List[List[Value]], reg: bool) -> Optional[List[Value]]:
        """``dst`` (f32) of every lane vector of unit ``u``; the next input
        (``scale x``, data dtype) to ``xbuf`` or returned (register carry).
        ``outs[v]`` = the f32 values of wire vector ``v``."""
        b, c, spec = self.c.b, self.c, self.spec
        k = 8 if self.wdt == "bf16" else 4
        nxt = []
        for v in range(self.V):
            e0 = self.wire_d0(u, v)
            if self.wdt == "bf16":
                e0 = b.shl(e0, b.const_i32(1))
            b.global_store_vN(c.ptr["dst"], e0, b.vec_pack(outs[v], F32), k)
            sc = [b.fmul(x, c.scale) for x in outs[v]]
            if spec.dtype == "bf16":
                nxt.append(b.vec_pack(_dwords_of_f32(b, sc, "bf16"), I32))
            else:
                bits = [b.bitcast(x, I32) for x in sc]
                nxt += [b.vec_pack(bits[i:i + 4], I32) for i in range(0, len(bits), 4)]
        if reg:
            return nxt
        for d, vec in zip(self.in_dw(u), nxt):
            b.global_store_vN(c.ptr["xbuf"], d, vec, 4)
        return None


def _lm_rank_parts(c: _Ctx, N: int, own: List[List[Value]], loaded: List[List[List[Value]]]) -> List[List[List[Value]]]:
    """``self_local``: the parts in rank order (``[r][v]`` = f32 values):
    this rank's own registers at ``r == rank``, else peer ``loaded[i]`` with
    ``i = (r - rank - 1) mod N`` (the order the peers were polled in)."""
    b = c.b
    parts = []
    for r in range(N):
        k = b.readfirstlane(b.mod(b.add(b.sub(b.const_i32(r + N - 1), c.rank), b.const_i32(0)), b.const_i32(N)))
        pr = []
        for v in range(len(own)):
            vals = list(own[v])
            for i in range(N - 1):
                hit = b.cmp_eq(k, b.const_i32(i))
                vals = [b.select(hit, x, y) for x, y in zip(loaded[i][v], vals)]
            pr.append(vals)
        parts.append(pr)
    return parts


def _lm_sum(c: _Ctx, spec: AllReduceSpec, parts: List[List[List[Value]]]) -> List[List[Value]]:
    b, N = c.b, spec.ranks
    outs = []
    for v in range(len(parts[0])):
        pv = [parts[r][v] for r in range(N)]
        s = _rank_sum(b, pv)
        if spec.knobs.mutant == "revsum":
            rv = _rank_sum(b, pv, reverse=True)
            r0 = b.cmp_eq(c.rank, c.zero)
            s = [b.select(r0, x, y) for x, y in zip(rv, s)]
        outs.append(s)
    return outs


def _lm_oneshot(c: _Ctx, spec: AllReduceSpec, lm: _LM, ex: Exchange, use, tag, slot, dl, src, reg, xin, cw_lds) -> Optional[List[Value]]:
    b, kn, N = c.b, spec.knobs, spec.ranks
    V = lm.V
    lam = kn.form == "lamport"
    drop = c.dropped(use)
    zero4 = b.vec_pack([c.zero] * 4, I32)
    if kn.self_local:
        others = [b.readfirstlane(b.mod(b.add(c.rank, b.const_i32(i)), b.const_i32(N))) for i in range(1, N)]
        push_bases = [c.peer_of(r) for r in others]
    else:
        others = None
        push_bases = c.peers
    # ---- push (+ publish)
    if kn.form == "pull" and kn.publish == "counted" and not reg and c.utab is None:
        # stage and publish unit by unit (the flag of a unit follows ITS local stores only)
        _lm_counted(c, spec, lm, ex, slot, tag, xin, drop, bases=[c.own])
        c.dstamp(2)
    elif kn.form == "pull":
        # stage this rank's partial in its OWN arena, drain (local), publish the unit's flag to every rank
        with _lm_units(c, spec, reg) as u:
            c.delay(use, 0)
            w = lm.wire(src(u))
            with b.scf_if(c.guard(drop)):
                W.push_vecs(b, [c.own], _lm_payload_off(c, ex, slot, c.rank, u), w, kn.push_cpol)
        c.dstamp(1)
        W.drain(b)
        c.dstamp(2)
        with b.scf_if(c.guard(drop)):
            with _lm_units(c, spec, reg) as u:
                _publish(c, ex, slot, c.rank, u, tag, N, c.lane_peer, c.zero)
    elif kn.publish == "comm_wave":
        _lm_comm_wave(c, spec, lm, ex, slot, tag, src, drop, cw_lds)
    elif kn.publish == "counted" and not reg:
        _lm_counted(c, spec, lm, ex, slot, tag, xin, drop)
    else:
        if lam:
            W.drain(b)  # this wave's lamport resets of the previous use are performed before it pushes again
        with _lm_units(c, spec, reg) as u:
            c.delay(use, 0)
            w = lm.wire(src(u))
            if lam:
                w = [lm.encode(x) for x in w]
            with b.scf_if(c.guard(drop)):
                W.push_vecs(b, push_bases, _lm_payload_off(c, ex, slot, c.rank, u), w, kn.push_cpol)
        c.dstamp(1)
        if not lam:
            W.drain(b)
            c.dstamp(2)
            copies = ex.copies
            cl = b.div(c.lane, b.const_i32(N)) if copies == 2 else c.zero
            with b.scf_if(c.guard(drop)):
                with _lm_units(c, spec, reg) as u:
                    _publish(c, ex, slot, c.rank, u, tag, N * copies, c.lane_peer, cl)
    c.dstamp(3)
    # ---- consume
    nxt = None
    with _lm_units(c, spec, reg) as u:
        c.delay(use, 1)
        if kn.form == "pull":
            faddr = b.add(c.own, c.i64(c.flag_off(ex, slot, c.lane_src, u, c.zero)))
            seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, poll_cpol=kn.poll_cpol)
            c.check(st, mask, seen, STAGES["ar"], use, tag, True)
            rslot = b.mod(b.add(slot, b.const_i32(1)), b.const_i32(kn.slots)) if kn.mutant == "slot" else slot
            vecs = W.load_lines_peers(b, c.peers, [_lm_payload_off(c, ex, rslot, b.const_i32(r), u) for r in range(N)], V)
            parts = [[lm.to_f32(vecs[r * V + v]) for v in range(V)] for r in range(N)]
        elif lam:
            srcs = others if kn.self_local else [b.const_i32(r) for r in range(N)]
            offs = [_lm_payload_off(c, ex, slot, r, u) for r in srcs]
            vecs, mask, st = W.poll_lamport(b, c.own, offs, V, lm.wdt, dl, c.abort_addr, pause=kn.pause)
            c.check(st, mask, c.zero, STAGES["ar"], use, tag, False)
            if kn.mutant != "noreset":
                W.store_zero_lines(b, c.own, offs, V, zero4)
            loaded = [[lm.to_f32(lm.decode(vecs[i * V + v])) for v in range(V)] for i in range(len(srcs))]
            if kn.self_local:
                own = [lm.to_f32(x) for x in lm.wire(src(u))]
                parts = _lm_rank_parts(c, N, own, loaded)
            else:
                parts = loaded
        else:
            lc = c.local_copy(ex)
            faddr = b.add(c.own, c.i64(c.flag_off(ex, slot, c.lane_src, u, lc)))
            seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, poll_cpol=kn.poll_cpol)
            c.check(st, mask, seen, STAGES["ar"], use, tag, True)
            vecs = W.load_lines_v(b, c.own, [_lm_payload_off(c, ex, slot, b.const_i32(r), u) for r in range(N)], V, poll_cpol=kn.poll_cpol)
            parts = [[lm.to_f32(vecs[r * V + v]) for v in range(V)] for r in range(N)]
        nxt = lm.store(u, _lm_sum(c, spec, parts), reg)
    c.dstamp(4)
    return nxt


def _lm_comm_wave(c: _Ctx, spec: AllReduceSpec, lm: _LM, ex: Exchange, slot, tag, src, drop, lds) -> None:
    """``publish="comm_wave"``: every wave stages its unit (``gw``) in LDS;
    after a CTA barrier wave 0 pushes the CTA's units to every rank, drains
    ITS stores only and publishes the flags (lane ``k`` < ``W N``: unit
    ``bid W + k / N`` to rank ``k mod N``); the other waves never drain."""
    b, kn, N = c.b, spec.knobs, spec.ranks
    V, Wv = lm.V, kn.waves
    units = b.const_i32(spec.units)

    def idx(w, v):
        return b.add(b.mul(b.add(b.mul(w, b.const_i32(V)), b.const_i32(v)), b.const_i32(256)), b.shl(c.lane, b.const_i32(2)))

    with b.scf_if(b.cmp_lt(c.gw, units)):
        w = lm.wire(src(c.gw))
        for v in range(V):
            b.smem_store_vN(lds, [idx(c.wave, v)], w[v], 4)
    b.sync()
    with b.scf_if(b.land(b.cmp_eq(c.wave, c.zero), c.guard(drop))):
        for ww in range(Wv):
            uu = b.add(b.mul(c.bid, b.const_i32(Wv)), b.const_i32(ww))
            with b.scf_if(b.cmp_lt(uu, units)):
                datas = [b.smem_load_vN(lds, idx(b.const_i32(ww), v), dtype=I32, n=4) for v in range(V)]
                W.push_vecs(b, c.peers, _lm_payload_off(c, ex, slot, c.rank, uu), datas, kn.push_cpol)
        c.dstamp(1)
        W.drain(b)
        c.dstamp(2)
        k = c.lane
        uu = b.add(b.mul(c.bid, b.const_i32(Wv)), b.div(k, b.const_i32(N)))
        r = b.mod(k, b.const_i32(N))
        with b.scf_if(b.land(b.cmp_lt(k, b.const_i32(Wv * N)), b.cmp_lt(uu, units))):
            off = c.flag_off(ex, slot, c.rank, uu, c.zero)
            W.store_flags_lanes(b, b.add(c.peer_of(r), c.i64(off)), tag)


def _lm_counted(c: _Ctx, spec: AllReduceSpec, lm: _LM, ex: Exchange, slot, tag, xin, drop, bases=None) -> None:
    """``publish="counted"`` (memory carry): per unit of this wave, push it,
    then issue the next unit's input loads and publish the flag once only
    those loads are outstanding (:func:`.wire.counted_publish`); the last
    unit reloads itself (a harmless load) so the loop body is uniform."""
    b, kn, N = c.b, spec.knobs, spec.ranks
    left = b.sub(b.const_i32(spec.units), c.gw)
    nk = b.readfirstlane(b.select(b.cmp_gt(left, c.zero), b.div(b.add(left, b.sub(c.tw, b.const_i32(1))), c.tw), c.zero))
    with b.scf_if(b.cmp_gt(nk, c.zero)):
        x0 = lm.load_x(xin, c.gw)
        loop = b.scf_for_iter(c.zero, nk, b.const_i32(1), [(f"cclcx{i}", x) for i, x in enumerate(x0)], iv_name=c.iv("cclck"))
        with loop as (k, carried):
            u = c.unit(k)
            w = lm.wire(list(carried))
            with b.scf_if(c.guard(drop)):
                W.push_vecs(b, bases or c.peers, _lm_payload_off(c, ex, slot, c.rank, u), w, kn.push_cpol)
            k1 = b.select(b.cmp_lt(b.add(k, b.const_i32(1)), nk), b.add(k, b.const_i32(1)), k)
            un = c.unit(k1)
            real = b.land(b.cmp_lt(c.lane, b.const_i32(N)), c.guard(drop))
            fo = c.i64(c.flag_off(ex, slot, c.rank, u, c.zero))
            junk = b.add(b.const_i32(JUNK_BASE), b.shl(b.add(b.mul(b.land(c.gw, b.const_i32(JUNK_WAVES - 1)), b.const_i32(64)), c.lane), b.const_i32(2)))
            faddr = b.select(real, b.add(c.lane_peer, fo), b.add(c.own, c.i64(junk)))
            nxt = W.counted_publish(b, xin, [b.shl(d, b.const_i32(2)) for d in lm.in_dw(un)], faddr, tag)
            b.scf_yield(*nxt)
    c.dstamp(1)


def _lm_twoshot(c: _Ctx, spec: AllReduceSpec, lm: _LM, rs: Exchange, ag: Exchange, use, tag, slot, dl, src, reg) -> Optional[List[Value]]:
    b, kn, N = c.b, spec.knobs, spec.ranks
    V = lm.V
    lam = kn.form == "lamport"
    drop = c.dropped(use)
    Nc = b.const_i32(N)
    zero4 = b.vec_pack([c.zero] * 4, I32)
    if lam:
        W.drain(b)
    # A: partial of unit u -> its owner (u mod N)
    with _lm_units(c, spec, reg) as u:
        c.delay(use, 0)
        w = lm.wire(src(u))
        if lam:
            w = [lm.encode(x) for x in w]
        owner = b.readfirstlane(b.mod(u, Nc))
        with b.scf_if(c.guard(drop)):
            W.push_vecs(b, [c.peer_of(owner)], _lm_payload_off(c, rs, slot, c.rank, u), w, kn.push_cpol)
    c.dstamp(1)
    if not lam:
        W.drain(b)
        with b.scf_if(c.guard(drop)):
            with _lm_units(c, spec, reg) as u:
                owner = b.readfirstlane(b.mod(u, Nc))
                _publish(c, rs, slot, c.rank, u, tag, rs.copies, c.peer_of(owner), c.lane)
    c.dstamp(2)
    # B: the owner sums in rank order, rounds to the wire dtype, pushes to every rank
    with _lm_units(c, spec, reg) as u:
        with b.scf_if(b.cmp_eq(b.mod(u, Nc), c.rank)):
            offs = [_lm_payload_off(c, rs, slot, b.const_i32(r), u) for r in range(N)]
            if lam:
                vecs, mask, st = W.poll_lamport(b, c.own, offs, V, lm.wdt, dl, c.abort_addr, pause=kn.pause)
                c.check(st, mask, c.zero, STAGES["rs"], use, tag, False)
                if kn.mutant != "noreset":
                    W.store_zero_lines(b, c.own, offs, V, zero4)
                vecs = [lm.decode(x) for x in vecs]
            else:
                lc = c.local_copy(rs)
                faddr = b.add(c.own, c.i64(c.flag_off(rs, slot, c.lane_src, u, lc)))
                seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, poll_cpol=kn.poll_cpol)
                c.check(st, mask, seen, STAGES["rs"], use, tag, True)
                vecs = W.load_lines_v(b, c.own, offs, V, poll_cpol=kn.poll_cpol)
            sums = _lm_sum(c, spec, [[lm.to_f32(vecs[r * V + v]) for v in range(V)] for r in range(N)])
            ys = [b.vec_pack(_dwords_of_f32(b, s, lm.wdt), I32) for s in sums]
            if lam:
                ys = [lm.encode(y) for y in ys]
            with b.scf_if(c.alive()):
                W.push_vecs(b, c.peers, _lm_payload_off(c, ag, slot, c.zero, u), ys, kn.push_cpol)
    if not lam:
        W.drain(b)
        cl = b.div(c.lane, Nc) if ag.copies == 2 else c.zero
        with _lm_units(c, spec, reg) as u:
            with b.scf_if(b.land(b.cmp_eq(b.mod(u, Nc), c.rank), c.alive())):
                _publish(c, ag, slot, c.zero, u, tag, N * ag.copies, c.lane_peer, cl)
    c.dstamp(3)
    # C: everyone reads the reduced unit
    nxt = None
    with _lm_units(c, spec, reg) as u:
        c.delay(use, 1)
        off = _lm_payload_off(c, ag, slot, c.zero, u)
        if lam:
            vecs, mask, st = W.poll_lamport(b, c.own, [off], V, lm.wdt, dl, c.abort_addr, pause=kn.pause)
            c.check(st, mask, c.zero, STAGES["ag"], use, tag, False)
            if kn.mutant != "noreset":
                W.store_zero_lines(b, c.own, [off], V, zero4)
            vecs = [lm.decode(x) for x in vecs]
        else:
            lc = c.local_copy(ag)
            faddr = b.add(c.own, c.i64(c.flag_off(ag, slot, c.zero, u, lc)))
            seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, poll_cpol=kn.poll_cpol)
            c.check(st, mask, seen, STAGES["ag"], use, tag, False)
            vecs = W.load_lines_v(b, c.own, [off], V, poll_cpol=kn.poll_cpol)
        nxt = lm.store(u, [lm.to_f32(vecs[v]) for v in range(V)], reg)
    c.dstamp(4)
    return nxt


def _allreduce_lm_kernel(spec: AllReduceSpec, lay: Layout, ex_index: int) -> KernelDef:
    """The chained all-reduce of a CCL-LM build (same parameters and chain
    contract as :func:`allreduce_kernel_in`; ``src`` / ``xbuf`` hold the
    data dtype, the wire carries ``spec.wire_dtype``)."""
    kn = spec.knobs
    N = spec.ranks
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    c = _Ctx(b, N, kn, [("src", I32), ("dst", F32), ("xbuf", I32)], lay, scale=True)
    lm = _LM(c, spec)
    src_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["src"]))
    xbuf_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["xbuf"]))
    hdr = 1 if kn.header else 0
    reg = spec.reg_carry
    cw_lds = b.smem_alloc(I32, [kn.waves * lm.V * 256], name_hint="cclcw") if kn.publish == "comm_wave" else None
    c.stamp(0)
    go = b.const_i32(1)
    if kn.header:
        _header(c, spec, b.add(c.iters, b.const_i32(hdr)))
        go = b.readfirstlane(b.select(b.cmp_eq(W.load_word_sys(b, c.own, b.const_i32(OFF["abort"])), c.zero), b.const_i32(1), c.zero))
    if kn.ar == "oneshot":
        exs = (lay.ex(f"{ex_index}.ar"),)
    else:
        exs = (lay.ex(f"{ex_index}.rs"), lay.ex(f"{ex_index}.ag"))

    def iteration(it, xv):
        c.it = it
        c.dstamp(0)
        use = b.add(it, b.const_i32(hdr))
        tag = W.tag_add(b, c.epoch, use)
        slot = c.slot(use)
        dl = c.deadline(it)
        first = b.cmp_eq(it, c.zero)
        c.abort_hook(use)
        xin = b.select(first, src_a, xbuf_a)
        if reg:

            def src(u):
                return xv

        else:

            def src(u):
                return lm.load_x(xin, u)

        if kn.ar == "oneshot":
            nxt = _lm_oneshot(c, spec, lm, exs[0], use, tag, slot, dl, src, reg, xin, cw_lds)
        else:
            nxt = _lm_twoshot(c, spec, lm, exs[0], exs[1], use, tag, slot, dl, src, reg)
        with b.scf_if(first):
            c.stamp(1)
        return nxt

    if reg:
        with b.scf_if(b.land(b.cmp_ne(go, c.zero), b.cmp_lt(c.gw, b.const_i32(spec.units)))):
            x0 = lm.load_x(src_a, c.gw)
            loop = b.scf_for_iter(c.zero, c.iters, b.const_i32(1), [(f"cclxv{i}", x) for i, x in enumerate(x0)], iv_name="cclit")
            with loop as (it, carried):
                b.scf_yield(*iteration(it, list(carried)))
    else:
        with b.scf_if(b.cmp_ne(go, c.zero)):
            with b.scf_for(c.zero, c.iters, b.const_i32(1), iv_name="cclit") as it:
                iteration(it, None)
                if kn.sync == "grid":
                    c.grid_sync(it, b.add(it, b.const_i32(hdr)))
    c.finish(b.add(c.iters, b.const_i32(hdr)))
    b.ret()
    return b.kernel


# ---------------------------------------------------------------- all-gather


def allgather_kernel_in(spec: AllGatherSpec, lay: Optional[Layout] = None, ex_index: int = 0) -> KernelDef:
    """Chained all-gather. Params: ``src`` i32* (this rank's rows x cols
    slice, wire dtype), ``dst`` i32* (rows x (ranks cols)), ``xbuf`` i32*.
    Collective i + 1's input on rank r is the gathered slice of rank
    ``(r + 1) mod ranks`` (a rotation: exact)."""
    kn = spec.knobs
    lay = lay or make_layout([spec])
    ex = lay.ex(f"{ex_index}.gather")
    N = spec.ranks
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    c = _Ctx(b, N, kn, [("src", I32), ("dst", I32), ("xbuf", I32)], lay, scale=False)
    src_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["src"]))
    xbuf_a = b.readfirstlane(b.global_ptr_to_i64(c.ptr["xbuf"]))
    ld, ud, Ds = spec.lane_dwords, spec.unit_dwords, spec.row_dwords
    line = kn.form == "line"
    hdr = 1 if kn.header else 0
    c.stamp(0)
    go = b.const_i32(1)
    if kn.header:
        _header(c, spec, b.add(c.iters, b.const_i32(hdr)))
        go = b.readfirstlane(b.select(b.cmp_eq(W.load_word_sys(b, c.own, b.const_i32(OFF["abort"])), c.zero), b.const_i32(1), c.zero))
    nxt_src = b.mod(b.add(c.rank, b.const_i32(1)), b.const_i32(N))
    with b.scf_if(b.cmp_ne(go, c.zero)), b.scf_for(c.zero, c.iters, b.const_i32(1), iv_name="cclit") as it:
        use = b.add(it, b.const_i32(hdr))
        tag = W.tag_add(b, c.epoch, use)
        slot = c.slot(use)
        dl = c.deadline(it)
        first = b.cmp_eq(it, c.zero)
        xin = b.select(first, src_a, xbuf_a)
        c.abort_hook(use)
        drop = c.dropped(use)
        with c.units_loop(spec.units) as k:
            u = c.unit(k)
            c.delay(use, 0)
            x = _load_x(c, xin, u, ld, ud)
            with b.scf_if(c.guard(drop)):
                W.push_v4(b, c.peers, c.payload_off(ex, slot, c.rank, u), _push_data(c, x, tag, ld, kn.form), kn.push_cpol)
        if line:
            W.drain(b)
            cl = b.div(c.lane, b.const_i32(N)) if ex.copies == 2 else c.zero
            with b.scf_if(c.guard(drop)):
                with c.units_loop(spec.units) as k:
                    _publish(c, ex, slot, c.rank, c.unit(k), tag, N * ex.copies, c.lane_peer, cl)
        with c.units_loop(spec.units) as k:
            u = c.unit(k)
            c.delay(use, 1)
            if line:
                lc = c.local_copy(ex)
                faddr = b.add(c.own, c.i64(c.flag_off(ex, slot, c.lane_src, u, lc)))
                seen, mask, st = W.poll_flags(b, faddr, tag, dl, c.abort_addr, b.const_i32(kn.slots), pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol)
                c.check(st, mask, seen, STAGES["allgather"], use, tag, True)
                vecs = W.load_lines(b, c.own, [c.payload_off(ex, slot, b.const_i32(r), u) for r in range(N)], poll_cpol=kn.poll_cpol)
                dws = [_vec_dwords(b, v, ld) for v in vecs]
            else:
                t64, neg, win = c.pair_ops(tag)
                pairs, mask, st = W.poll_pairs(
                    b, c.own, [c.payload_off(ex, slot, b.const_i32(r), u) for r in range(N)], t64, neg, win, dl, c.abort_addr,
                    pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol,
                )  # fmt: skip
                c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["allgather"], use, tag, False)
                dws = [[b.trunc(pairs[2 * r], I32), b.trunc(pairs[2 * r + 1], I32)] for r in range(N)]
            d = b.add(b.mul(u, b.const_i32(ud)), b.mul(c.lane, b.const_i32(ld)))
            row = b.div(d, b.const_i32(Ds))
            col = b.sub(d, b.mul(row, b.const_i32(Ds)))
            for r in range(N):
                o = b.add(b.mul(row, b.const_i32(N * Ds)), b.add(b.const_i32(r * Ds), col))
                b.global_store_vN(c.ptr["dst"], o, b.vec_pack(dws[r], I32), ld)
            nx = dws[0]
            for r in range(1, N):
                sel = b.cmp_eq(nxt_src, b.const_i32(r))
                nx = [b.select(sel, a, z) for a, z in zip(dws[r], nx)]
            b.global_store_vN(c.ptr["xbuf"], d, b.vec_pack(nx, I32), ld)
        with b.scf_if(first):
            c.stamp(1)
    c.finish(b.add(c.iters, b.const_i32(hdr)))
    b.ret()
    return b.kernel


# -------------------------------------------------------------------- argmax


def argmax_kernel_in(spec: ArgmaxSpec, lay: Optional[Layout] = None, ex_index: int = 0) -> KernelDef:
    """Vocab-parallel argmax merge over ranks, one CTA (``nctas`` = 1).
    Params: ``vals`` f32[iters x rows], ``ids`` i32[iters x rows] (this
    rank's per-row max and its GLOBAL token id per collective), ``ovals`` /
    ``oids`` (merged). Merge in rank order: a larger value wins, an equal
    value takes the lower global id (ATOM's ``tie="low"``)."""
    kn = spec.knobs
    lay = lay or make_layout([spec])
    ex = lay.ex(f"{ex_index}.amax")
    N = spec.ranks
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    c = _Ctx(b, N, kn, [("vals", F32), ("ids", I32), ("ovals", F32), ("oids", I32)], lay, scale=False)
    c.stamp(0)
    units = spec.units
    hdr = 1 if kn.header else 0
    go = b.const_i32(1)
    if kn.header:
        _header(c, spec, b.add(c.iters, b.const_i32(hdr)))
        go = b.readfirstlane(b.select(b.cmp_eq(W.load_word_sys(b, c.own, b.const_i32(OFF["abort"])), c.zero), b.const_i32(1), c.zero))
    rowl = b.add(b.mul(c.wave, b.const_i32(64)), c.lane)
    live = b.cmp_lt(rowl, b.const_i32(spec.rows))
    rowc = b.select(live, rowl, b.const_i32(spec.rows - 1))
    with b.scf_if(b.land(b.cmp_ne(go, c.zero), b.cmp_lt(c.wave, b.const_i32(units)))):
        with b.scf_for(c.zero, c.iters, b.const_i32(1), iv_name="cclit") as it:
            use = b.add(it, b.const_i32(hdr))
            tag = W.tag_add(b, c.epoch, use)
            slot = c.slot(use)
            dl = c.deadline(it)
            c.abort_hook(use)
            idx = b.add(b.mul(it, b.const_i32(spec.rows)), rowc)
            v = b.global_load(c.ptr["vals"], idx, F32)
            i = b.global_load_i32(c.ptr["ids"], idx)
            with b.scf_if(c.guard(c.dropped(use))):
                W.push_v4(b, c.peers, c.payload_off(ex, slot, c.rank, c.wave), b.vec_pack([b.bitcast(v, I32), tag, i, tag], I32), kn.push_cpol)
            t64, neg, win = c.pair_ops(tag)
            pairs, mask, st = W.poll_pairs(
                b, c.own, [c.payload_off(ex, slot, b.const_i32(r), c.wave) for r in range(N)], t64, neg, win, dl, c.abort_addr,
                pause=kn.pause, mutant=kn.mutant, poll_cpol=kn.poll_cpol,
            )  # fmt: skip
            c.check(st, mask, b.trunc(b.lshr(pairs[0], b.const_i64(32)), I32), STAGES["argmax"], use, tag, False)
            bv = b.bitcast(b.trunc(pairs[0], I32), F32)
            bi = b.trunc(pairs[1], I32)
            for r in range(1, N):
                vr = b.bitcast(b.trunc(pairs[2 * r], I32), F32)
                ir = b.trunc(pairs[2 * r + 1], I32)
                better = b.lor(b.fcmp("ogt", vr, bv), b.land(b.fcmp("oeq", vr, bv), b.cmp_lt(ir, bi)))
                bv = b.select(better, vr, bv)
                bi = b.select(better, ir, bi)
            with b.scf_if(live):
                b.store(bv, c.ptr["ovals"], idx)
                b.store(bi, c.ptr["oids"], idx)
            with b.scf_if(b.cmp_eq(it, c.zero)):
                c.stamp(1)
    c.finish(b.add(c.iters, b.const_i32(hdr)))
    b.ret()
    return b.kernel


# ------------------------------------------------- builders: (spec, *, arch)


def _check_arch(arch: str) -> None:
    if arch != "gfx950":
        raise ValueError(f"the CCL kernels target gfx950 (got {arch!r})")


def build_allreduce_kernel(spec: AllReduceSpec, *, arch: str = "gfx950") -> KernelDef:
    """The all-reduce of ``spec`` on its own arena map (``make_layout([spec])``);
    :func:`allreduce_kernel_in` places it in a shared multi-collective map."""
    _check_arch(arch)
    return allreduce_kernel_in(spec)


def build_allgather_kernel(spec: AllGatherSpec, *, arch: str = "gfx950") -> KernelDef:
    _check_arch(arch)
    return allgather_kernel_in(spec)


def build_argmax_kernel(spec: ArgmaxSpec, *, arch: str = "gfx950") -> KernelDef:
    _check_arch(arch)
    return argmax_kernel_in(spec)
