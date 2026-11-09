# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1-Flash's routed experts (gate / up and down) of monofly's TP decode layer as rocKE stage functions
(hybrid path: monofly links them into its FlyDSL layer kernel in place of the bodies of the ``moe.ug`` / ``moe.down``
stages, ``MoEExperts(rk="v41[:knobs]")``, off by default; study: ``monofly-study/dsv41/ports/moe``).

Bit-identical re-implementation of monofly ``blocks/gfx950/moe_experts.py`` ``_ug_body`` / ``_down_body`` /
``_gemv`` for V4.1's recipe (``act="silu"``, ``limit``, the route weight in gate / up, route ``"cta"`` (the slot table
in the CTA's LDS, built by ``moe.rt``), MXFP4 weights x MXFP8 activations, TP ``intra`` / ``expert`` or TP1), generic
over the build: rows (1..16: one MFMA tile a slot), the rank's expert width (TP4: 640 padded / 576 live; TP2: 1152;
TP1: 2304), ``ug_g`` / ``down_g``, the prefetch depths, and each mailbox region's dual-copy layout. Per (slot, row
group) the arithmetic is monofly's:

* gate / up: K split over ``wk`` = ``_wk(hidden / 128, ug_g)`` waves (wave ``kw`` takes K steps kw, kw + wk, ..., one
  ``v_mfma_scale_f32_16x16x128_f8f6f4`` chain a row group from a zero accumulator), the K-wave partials summed in
  wave order; ``bf16`` of gate / up, ``min(g, L)``, ``clamp(u, -L, L)``, ``bf16(w * (silu(g) * u))`` with
  ``silu(g) = g / (1 + exp2(g * -log2 e))`` (``v_exp_f32``, IEEE ``fdiv``), the MX group's amax over 8 lanes, the
  E8M0 code of ``mx.e8m0_code``, ``x 2^-(code-127)`` saturated, RNE E4M3 -> ``moe.mid``;
* down: K over ``_wk(inter / 128, down_g)`` waves likewise, the padded K tail (``live``) read as zeros; ``bf16(y)``
  -> ``moe.y`` (segment kp of the pick's row).

The task layout (``Knobs.map``) is free (a row group's arithmetic does not depend on the CTA computing it):
``"parity"`` = monofly's own: ug task t = (slot t // ug_parts, part t % ug_parts), down task t = (slot t //
down_parts, part t % down_parts), one task a call.

Mailboxes are monofly's: ``(payload, tag)`` 8-byte pairs, device scope (sc1), polled until every tag matches; a region
of a dual-copy program (``Program.dual``) is read at the CTA's copy (``l_<reg>``: 0 or 8192 bytes) and written to
both; a spread one (> 8 KiB) places logical byte o at ``o + (o & ~8191)``. Both are build-time facts of the region
(``Shape.regions``), the copy a runtime argument.

ABI (``lds="arg"``; LDS operands are word offsets from ``lds_base`` = the byte address of monofly's LDS arena)::

    ug:   void (i32 tag, i32 task, i64 w, i64 ws, i64 xq, i64 xs, i64 mq, i64 ms,
                i32 l_xq, i32 l_xs, i32 l_mq, i32 l_ms, i32 lds_base)
    down: void (i32 tag, i32 task, i64 w, i64 ws, i64 mq, i64 ms, i64 y, i32 l_mq, i32 l_ms, i32 l_y, i32 lds_base)

``w`` / ``ws``: every expert's packed MXFP4 units / their scale codes (monofly ``pack_experts``, the expert at a
64-bit base: TP2's gate / up weights pass 2 GiB). The slot table is monofly's (``_Cfg.rt``: [rt] count; [rt + 1 + s]
expert; [rt + 1 + S + s] its first pick's place; [rt + 2 + 2S + i] pick (row * K + kp); [rt + 2 + 3S + i] its route
weight bits).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass, fields

from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType

WAVES, THREADS, TILE, GRID = 8, 512, 16, 256
HALF = 8192  # dual copies: copy 1 HALF bytes after copy 0 (monofly ``device/gfx950/dual.py``)
CM_NT, CM_DEV = 2, 16
OOB_BYTE = 0x7FFFFFF0  # a byte offset no buffer reaches: the load returns 0
ZERO_CODE = 127
FP8_MAX = 448.0
NEG_LOG2E = -1.4426950408889634
POOL_WORDS = 160 * 1024 // 4
SYMBOL_PREFIX = "rocke_v41mf_"
REGION_NAMES = ("xq", "xs", "mq", "ms", "y", "shq", "shs")
# Test-only mutants (each must make the in-situ bitwise A/B differ, or hang-free fail the layer gate):
#   mid_bit:  every mid word's low bit flipped
#   y_bit:    every y pair's low bit flipped
#   ug_chain: gate / up K step 1 left out of its chain (wave 1's first unit)
#   dn_chain: down's K step 1 left out
#   code:     the mid scale code + 1 (halving an E4M3 value is exact unless it goes subnormal: may be bit-equal)
#   ug_order: the gate / up K-wave partials summed in order 1, 0, 2, .. (an fp32 reassociation: may be bit-equal)
MUTANTS = ("mid_bit", "y_bit", "ug_chain", "dn_chain", "code", "ug_order", "comb_order")
REPORT_ONLY = ("code", "ug_order", "comb_order")


def wk_of(steps: int, g: int) -> int:
    """monofly ``moe_experts._wk``: waves along K."""
    return next((wk for wk in (8, 4, 2, 1) if steps % wk == 0 and g % (WAVES // wk) == 0), WAVES)


def gather_n(kk: int) -> int:
    """monofly ``moe_experts._gather_n``: rows a gather batch of ``kk``-element MXFP8 rows."""
    return max(2, min(TILE, THREADS // max(kk // 8, 1)))


@dataclass(frozen=True)
class Shape:
    """The build (monofly's ``_Cfg`` fields this stage reads)."""

    rows: int
    k: int
    hidden: int
    inter: int  # the rank's expert width (padded to 128)
    live: int  # its real width when padded (0: all of it)
    ug_g: int
    down_g: int
    prefetch: int = 8
    pre_b: int = 4
    down_pre_b: int = 4
    limit: float = 0.0
    rt: int = 0  # LDS word of the slot table
    gemv_lds: int = 0  # monofly ``_Cfg.gemv_lds`` (the scale codes at gemv_lds + 32)
    y_k: int = 7  # y segments a row (K picks + the folded shared expert's)
    regions: tuple = ()  # ((name, dual, spread), ...) for REGION_NAMES
    experts: int = 384  # experts in the packed weight arrays (TP intra: all; expert layout: the rank's)
    # map=bal: the folded shared expert's stages on their CTAs (first CTA, tasks, a task's weight bytes in half KiB)
    sh_ug_base: int = 0
    sh_ug_tasks: int = 0
    sh_ug_cost: int = 0
    sh_dn_base: int = 0
    sh_dn_tasks: int = 0
    sh_dn_cost: int = 0
    sh_inter: int = 0  # map=cols: the folded shared expert's width on this rank (padded) and its live width
    sh_live: int = 0

    @property
    def slots(self) -> int:
        return self.rows * self.k

    @property
    def ug_parts(self) -> int:
        return 2 * self.inter // 16 // self.ug_g

    @property
    def down_parts(self) -> int:
        return self.hidden // 16 // self.down_g

    def region(self, name: str) -> tuple[bool, bool]:
        for n, d, s in self.regions:
            if n == name:
                return bool(d), bool(s)
        return False, False

    def check(self) -> None:
        if not 1 <= self.rows <= TILE:
            raise ValueError(f"rows 1..{TILE} (one MFMA tile a slot), got {self.rows}")
        if self.hidden % 128 or self.inter % 128 or (self.live and self.live % 32):
            raise ValueError("hidden / inter multiples of 128, live of 32")
        if self.ug_g % 4 or (2 * self.inter // 16) % self.ug_g or (self.hidden // 16) % self.down_g:
            raise ValueError("ug_g a multiple of 4 dividing the gate / up groups; down_g dividing hidden / 16")
        for name, need in (("ug", self.ug_lds_words), ("down", self.down_lds_words)):
            if need > self.rt:
                raise ValueError(f"{name}: LDS {need} words meets the slot table at {self.rt}")

    # LDS (words from lds_base): the B rows at 0 (TILE rows x pitch, then TILE x spitch codes), the K-wave partials
    # overlaying them after the stream, the task's scale codes at gemv_lds + 32 (monofly's layout)
    @property
    def ws_at(self) -> int:
        return self.gemv_lds + 2 * TILE

    @property
    def ug_lds_words(self) -> int:
        return self.ws_at + self.ug_g * (self.hidden // 128) * 16

    @property
    def down_lds_words(self) -> int:
        return self.ws_at + self.down_g * (self.inter // 128) * 16


@dataclass(frozen=True)
class Knobs:
    map: str = "parity"
    stage: str = "split"  # (monofly side) "split": the ug / down stages' bodies; "fused": one stage of 256 tasks
    pause: int = 0  # poll retry: 0 ``s_nop 0`` (monofly's), N ``s_sleep N``
    poll_max: int = 12  # pair specs a poll round trip (monofly POLL_MAX)
    bar: str = "fence"  # fence: FlyDSL gpu.barrier (release / acquire fences + s_barrier); lds: lgkmcnt(0) + s_barrier
    sbar: int = 0  # 1: a sched_barrier before every unit (monofly's MoE stream has none)
    dpre: int = 0  # (bal) 1: the first down row group's weights issued before its slots' mid rows are polled
    xpre: int = 0  # (bal) 1: the next gate / up block's first units issued before this block's epilogue (carried)
    nop: int = 0  # inert: another symbol, the same code (a recompiled control of an A/B: the compile lottery)
    bsh: int = 1  # (bal) 1: no gate / up blocks on the folded shared expert's gate / up CTAs (they start late: their
    #               blocks' slots held every down of them back, ATT g2); 0: every CTA
    gall: int = 0  # 1: a task's / CTA's mid (and x) row gathers in one poll (rows up to the step's), not batches
    lb: int = 0  # 1: weight scale codes read as LDS bytes (ds_read_u8, offsets folded: monofly's ds_read_i8);
    #              0: the code's word + a bit-field extract (ATT g4: +37 v_bfe / +45 v_and / +34 v_add in down)
    mx: int = 0  # 1: llvm.maximum / minimum natively (v_maximum3 / v_minimum3, monofly's); 0: select-emulated
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.map not in ("parity", "bal", "cols", "blk"):
            raise ValueError("map: parity | bal | cols | blk")
        if self.stage not in ("split", "fused"):
            raise ValueError("stage: split | fused")
        if self.bar not in ("fence", "lds"):
            raise ValueError("bar: fence | lds")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max: 1..16")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def spec(self) -> str:
        d = Knobs()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if getattr(self, f.name) != getattr(d, f.name))


def knobs_from_spec(spec: str) -> Knobs:
    kw, types = {}, {f.name: f.type for f in fields(Knobs)}
    for part in (spec or "").replace(";", ",").replace("/", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown knob {k!r}")
        kw[k] = int(v) if types[k] in ("int", int) else v
    return Knobs(**kw)


def f32c(v: float) -> float:
    """The f32 value of a Python float (as FlyDSL's f32 constants hold it)."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


class _E:
    def __init__(self, b: IRBuilder, k: Knobs):
        self.b, self.k = b, k
        self.pool = b.smem_alloc(I32, [POOL_WORDS], name_hint="mono_lds")

    def c(self, v):
        return self.b.const_i32(int(v))

    def f(self, v):
        return self.b.const_f32(f32c(v))

    def rsrc(self, addr, nbytes=0x7FFFFFF0):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(addr) if addr.type == I64 else addr, PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def barrier(self):
        if self.k.bar == "lds":
            self.b.sync_lds_only(fence="compiler")
        else:
            self.b.s_barrier_bare(fence="workgroup")  # FlyDSL gpu.barrier: fences + s_barrier

    def pause_text(self) -> str:
        return "s_nop 0" if self.k.pause == 0 else f"s_sleep {self.k.pause}"

    # ---- floats with monofly's (FlyDSL's) semantics
    def fmax(self, a, x):  # llvm.maximum.f32
        b = self.b
        if self.k.mx:
            return b.fmaximum(a, x)
        return b.select(b.fcmp("uno", a, x), b.fadd(a, x), b.fmax(a, x))

    def fmin(self, a, x):  # llvm.minimum.f32
        b = self.b
        if self.k.mx:
            return b.fminimum(a, x)
        return b.select(b.fcmp("uno", a, x), b.fadd(a, x), b.fmin(a, x))

    def bfr(self, v):
        return self.b.cast_to_f32(self.b.trunc_f32_to_bf16(v))

    def bf_pair(self, v0, v1):
        b = self.b
        lo = b.zext(b.bitcast(b.trunc_f32_to_bf16(v0), I16), I32)
        hi = b.zext(b.bitcast(b.trunc_f32_to_bf16(v1), I16), I32)
        return b.lor(lo, b.shl(hi, self.c(16)))

    def absf(self, v):
        b = self.b
        return b.bitcast(b.land(b.bitcast(v, I32), self.c(0x7FFFFFFF)), F32)

    def e8m0(self, amax):
        """monofly ``mx.e8m0_code``."""
        b, c = self.b, self.c
        bits = b.bitcast(amax, I32)
        ex = b.land(b.lshr(bits, c(23)), c(255))
        man = b.land(bits, c(0x7FFFFF))
        code = b.add(b.sub(ex, c(8)), b.select(b.cmp_gt(man, c(0x600000)), c(1), c(0)))
        code = b.smax(b.smin(code, c(254)), c(1))
        return b.select(b.fcmp("oeq", amax, b.const_f32(0.0)), c(ZERO_CODE), code)

    # ---- LDS (word offsets from lds_base)
    def ld(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=I32, n=1), 0)

    def ldf(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=F32, n=1), 0)

    def ld4(self, at):
        return self.b.smem_load_vN(self.pool, at, dtype=I32, n=4)

    def ld4f(self, at):
        return self.b.smem_load_vN(self.pool, at, dtype=F32, n=4)

    def st(self, at, v, n=1):
        self.b.smem_store_vN(self.pool, [at], v, n)

    def ld_byte(self, byte):
        """The byte at LDS byte offset ``byte`` (from lds_base), zero-extended."""
        b, c = self.b, self.c
        if self.k.lb:
            return b.smem_load_u8(self.pool, byte)
        word = self.ld(b.lshr(byte, c(2)))
        return b.land(b.lshr(word, b.mul(b.land(byte, c(3)), c(8))), c(0xFF))

    # ---- mailbox regions: (rsrc, dual, spread, local)
    def off(self, reg, byte, load):
        b = self.b
        _, dual, spread, local = reg
        o = b.add(b.land(byte, self.c(~(HALF - 1))), byte) if spread else byte
        return b.add(o, local) if (load and dual) else o

    def put(self, reg, pair, words):
        """(payload, tag) pairs from pair ``pair``: ``words`` = [w0, tag] or [w0, tag, w1, tag]; both copies."""
        b = self.b
        data = b.vec_pack(words, I32)
        o = self.off(reg, b.mul(pair, self.c(8)), load=False)
        b.buffer_store_cp(data, reg[0], o, self.c(0), aux=CM_DEV)
        if reg[1]:
            b.buffer_store_cp(data, reg[0], b.add(o, self.c(HALF)), self.c(0), aux=CM_DEV)

    def poll(self, tag, specs):
        """[(reg, pair, npairs)] -> payload word lists; ``poll_max`` specs a round trip."""
        b, out = self.b, []
        for b0 in range(0, len(specs), self.k.poll_max):
            batch = specs[b0 : b0 + self.k.poll_max]
            vecs = b.tagged_poll(tag, [(reg[0], self.off(reg, b.mul(pr, self.c(8)), load=True), 2 * n)
                                       for reg, pr, n in batch], aux=CM_DEV, pause=self.pause_text(),
                                 reload="launder")  # fmt: skip
            for (_, _, n), v in zip(batch, vecs):
                out.append([b.vec_extract(v, 2 * q) for q in range(n)])
        return out


class _Ctx:
    """Per-function values: thread ids, the slot table reader."""

    def __init__(self, e: _E, sh: Shape, tag):
        b, c = e.b, e.c
        self.e, self.sh, self.tag = e, sh, tag
        self.tid = b.thread_id_x()
        self.lane = b.land(self.tid, c(63))
        self.wave = b.readfirstlane(b.lshr(self.tid, c(6)))
        self.j = b.lshr(self.lane, c(4))
        self.row = b.land(self.lane, c(15))
        self._uid = 0

    def uid(self, stem: str) -> str:
        """A fresh name (loop induction variables are named SSA values: one name a loop)."""
        self._uid += 1
        return f"{stem}{self._uid}"

    def slot(self, s):
        """(count, expert, first pick's place, pick count) of slot ``s`` (clamped) from the CTA's table."""
        e, b, c, sh = self.e, self.e.b, self.e.c, self.sh
        S, rt = sh.slots, sh.rt
        sc = b.smin(s, c(S - 1))
        count = b.readfirstlane(e.ld(c(rt)))
        eid = b.readfirstlane(e.ld(b.add(c(rt + 1), sc)))
        a = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), sc)))
        z = b.readfirstlane(e.ld(b.add(c(rt + 2 + S), sc)))
        return count, eid, a, b.sub(z, a)


def _gather(cx: _Ctx, src_q, src_s, width, seg_w, kk, pitch, spitch, xs_at, pick_lds, npick, at=0):
    """The slot's ``npick`` picks' MXFP8 rows (channel row r = pick // K, segment (pick % K) * seg_w, ``kk``
    elements) into LDS rows 0 .. npick - 1 (words at j * pitch, codes at xs_at + j * spitch), in a dynamic loop of
    ``gather_n(kk)``-row batches, each one poll (monofly ``_gather_rows`` / ``_gather_batch``)."""
    e, b, c, sh = cx.e, cx.e.b, cx.e.c, cx.sh
    K = sh.k
    words, codes = kk // 4, kk // 32
    gr = gather_n(kk)
    nb = b.readfirstlane(b.div(b.add(npick, c(gr - 1)), c(gr)))
    jmax = b.sub(npick, c(1))
    tid = cx.tid
    with b.scf_for(c(0), nb, c(1), iv_name=cx.uid("gbat")) as bi:
        j0 = b.mul(bi, c(gr))
        specs, dests = [], []
        nq = gr * words // 2
        for it in range(-(-nq // THREADS)):
            i = b.smin(b.add(tid, c(it * THREADS)), c(nq - 1))
            jj = b.smin(b.add(j0, b.div(i, c(words // 2))), jmax)
            w2 = b.mod(i, c(words // 2))
            pick = e.ld(b.add(pick_lds, jj))
            r = b.div(pick, c(K))
            base = b.mul(r, c(width // 4))
            if seg_w:
                base = b.add(base, b.mul(b.mod(pick, c(K)), c(seg_w // 4)))
            specs.append((src_q, b.add(base, b.mul(w2, c(2))), 2))
            dests.append(b.add(c(at), b.add(b.mul(jj, c(pitch)), b.mul(w2, c(2)))))
        nc = gr * codes
        for it in range(-(-nc // THREADS)):
            i = b.smin(b.add(tid, c(it * THREADS)), c(nc - 1))
            jj = b.smin(b.add(j0, b.div(i, c(codes))), jmax)
            g = b.mod(i, c(codes))
            pick = e.ld(b.add(pick_lds, jj))
            r = b.div(pick, c(K))
            base = b.mul(r, c(width // 32))
            if seg_w:
                base = b.add(base, b.mul(b.mod(pick, c(K)), c(seg_w // 32)))
            specs.append((src_s, b.add(base, g), 1))
            dests.append(b.add(c(at + xs_at), b.add(b.mul(jj, c(spitch)), g)))
        got = e.poll(cx.tag, specs)
        for words_, d in zip(got, dests):
            for q, w in enumerate(words_):
                e.st(b.add(d, c(q)), w)


def _gather_static(cx: _Ctx, slots, cap: int):
    """``_gather`` of several slots at once (``gall=1``): every slot's rows 0 .. ``cap`` - 1 (rows past its npick
    repeat its last) in ONE poll (``poll_max`` specs a round trip): one round trip a thread instead of a dynamic loop
    of batches a slot. ``slots``: [(src_q, src_s, width, seg_w, kk, pitch, spitch, xs_at, pick_lds, npick, at)]."""
    e, b, c, sh = cx.e, cx.e.b, cx.e.c, cx.sh
    K = sh.k
    tid = cx.tid
    specs, dests = [], []
    for sl in slots:
        src_q, src_s, width, seg_w, kk, pitch, spitch, xs_at, pick_lds, npick, at = sl[:11]
        cap_ = sl[11] if len(sl) > 11 else cap  # (a slot's own row cap)
        words, codes = kk // 4, kk // 32
        jmax = b.sub(npick, c(1))
        for kind, per, n in (("q", words // 2, cap_ * (words // 2)), ("s", codes, cap_ * codes)):
            for it in range(-(-n // THREADS)):
                i = b.smin(b.add(tid, c(it * THREADS)), c(n - 1))
                jj = b.smin(b.div(i, c(per)), jmax)
                x = b.mod(i, c(per))
                pick = e.ld(b.add(pick_lds, jj))
                r = b.div(pick, c(K))
                if kind == "q":
                    base = b.mul(r, c(width // 4))
                    if seg_w:
                        base = b.add(base, b.mul(b.mod(pick, c(K)), c(seg_w // 4)))
                    specs.append((src_q, b.add(base, b.mul(x, c(2))), 2))
                    dests.append(b.add(c(at), b.add(b.mul(jj, c(pitch)), b.mul(x, c(2)))))
                else:
                    base = b.mul(r, c(width // 32))
                    if seg_w:
                        base = b.add(base, b.mul(b.mod(pick, c(K)), c(seg_w // 32)))
                    specs.append((src_s, b.add(base, x), 1))
                    dests.append(b.add(c(at + xs_at), b.add(b.mul(jj, c(spitch)), x)))
    got = e.poll(cx.tag, specs)
    for words_, d in zip(got, dests):
        for q, w in enumerate(words_):
            e.st(b.add(d, c(q)), w)


def _gemv(cx: _Ctx, *, eid, g, kk, n_rows, part, w_addr, s_addr, pitch, spitch, stage_b, pre_b, live_rg=0,
          live_k=0, skip=None):  # fmt: skip
    """monofly ``_gemv`` (MXFP4 weights): the ``g`` row groups ``part * g ..`` of expert ``eid``'s ``n_rows`` x
    ``kk`` weight against the 16 B columns staged by ``stage_b``; partials [K wave][g][16][16] f32 into LDS at word 0.
    -> wk. ``skip(i, kw)`` (mutant): a Value that drops K unit i's MFMA, or None."""
    e, b, c, sh = cx.e, cx.e.b, cx.e.c, cx.sh
    lane, wave, tid = cx.lane, cx.wave, cx.tid
    steps = kk // 128
    w_bytes, s_bytes = n_rows * kk // 2, n_rows * kk // 32
    ex64 = b.zext(eid, I64)
    r_w = e.rsrc(b.add(w_addr, b.mul(ex64, b.const_i64(w_bytes))), w_bytes)
    r_s = e.rsrc(b.add(s_addr, b.mul(ex64, b.const_i64(s_bytes))), s_bytes)
    wk = wk_of(steps, g)
    gw = g // (WAVES // wk)
    sh_wk = wk.bit_length() - 1
    kw = b.land(wave, c(wk - 1))
    g0 = b.mul(b.lshr(wave, c(sh_wk)), c(gw))
    wr = b.lshr(wave, c(sh_wk))
    per = -(-steps // wk)
    units = [(i, jg) for i in range(per) for jg in range(gw)]
    prefetch = sh.prefetch

    def unit_of(i, jg):
        st = b.add(kw, c(wk * i))
        valid = None if steps % wk == 0 else b.cmp_lt(st, c(steps))
        stc = st if valid is None else b.smin(st, c(steps - 1))
        rg = b.add(b.mul(part, c(g)), b.add(g0, c(jg)))
        if live_rg:
            lv = b.cmp_lt(rg, c(live_rg))
            valid = lv if valid is None else b.land(valid, lv)
        return b.add(b.mul(rg, c(steps)), stc), valid, stc

    def issue(u):
        i, jg = units[u]
        unit, valid, stc = unit_of(i, jg)
        if live_k:
            kl = b.sub(c(live_k), b.mul(stc, c(128)))
            lv = b.cmp_lt(b.mul(b.lshr(lane, c(4)), c(32)), kl)
            valid = lv if valid is None else b.land(valid, lv)
        voff = b.mul(b.add(b.mul(unit, c(64)), lane), c(16))
        if valid is not None:
            voff = b.select(valid, voff, c(OOB_BYTE))
        return b.buffer_load_cp(r_w, voff, c(0), dwords=4, aux=CM_NT)

    # the task's scale codes (g row groups x every K step, 64 B a unit) with 16 B loads, issued first, staged in LDS
    ws_at = sh.ws_at
    chunks = g * steps * 4
    codes = []
    for r in range(-(-chunks // THREADS)):
        ch = b.smin(b.add(tid, c(r * THREADS)), c(chunks - 1))
        src = b.add(b.mul(part, c(chunks)), ch)
        codes.append((ch, b.buffer_load_cp(r_s, b.mul(src, c(16)), c(0), dwords=4, aux=CM_NT)))
    q = [issue(u) for u in range(min(pre_b, len(units)))]
    for ch, v in codes:
        e.st(b.add(c(ws_at), b.mul(ch, c(4))), v, 4)
    stage_b()
    e.barrier()
    q += [issue(u) for u in range(len(q), min(prefetch, len(units)))]
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    acc = [zero for _ in range(gw)]
    xs_at = TILE * pitch
    brow = cx.row
    for u, (i, jg) in enumerate(units):
        if sh_sbar(e):
            b.sched_barrier(0)
        if u + prefetch < len(units):
            q.append(issue(u + prefetch))
        st = b.smin(b.add(kw, c(wk * i)), c(steps - 1))
        off = b.add(b.add(b.mul(brow, c(pitch)), b.mul(st, c(32))), b.mul(cx.j, c(4)))
        xv = b.vec_concat(e.ld4(off), e.ld4(b.add(off, c(16))))
        xs = e.ld(b.add(c(xs_at), b.add(b.add(b.mul(brow, c(spitch)), b.mul(st, c(4))), cx.j)))
        byte = b.add(c(ws_at * 4), b.add(b.mul(b.add(b.mul(b.add(g0, c(jg)), c(steps)), st), c(64)), lane))
        ws = e.ld_byte(byte)
        if steps % wk:
            ws = b.select(b.cmp_lt(b.add(kw, c(wk * i)), c(steps)), ws, c(0))
        new = b.mfma_scale_mx(q[u], xv, acc[jg], ws, xs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
        if skip is not None:
            sk = skip(i, jg, kw)
            if sk is not None:
                new = b.select(sk, acc[jg], new)
        acc[jg] = new
    e.barrier()  # every wave done with the gathered rows: the partials overlay them
    n = cx.row
    for gl in range(gw):
        off = b.add(b.mul(b.add(b.mul(b.add(b.add(b.mul(kw, c(g)), b.mul(wr, c(gw))), c(gl)), c(16)), n), c(16)),
                    b.mul(cx.j, c(4)))  # fmt: skip
        e.st(off, b.vec_pack([b.bitcast(b.vec_extract(acc[gl], q), I32) for q in range(4)], I32), 4)
    e.barrier()
    return wk


def sh_sbar(e: _E) -> bool:
    return bool(e.k.sbar)


def _wave_sum(cx: _Ctx, g, gl, n, f4, wk, order=None):
    """Sum over the K waves (in order) of the partials of row group ``gl``, pick ``n``, features f4 * 4 .. + 3."""
    e, b, c = cx.e, cx.e.b, cx.e.c
    stride = g * TILE * 16
    base = b.add(b.mul(b.add(b.mul(gl, c(TILE)), n), c(16)), b.mul(f4, c(4)))
    tot = None
    for w in order or range(wk):
        v = e.ld4f(b.add(base, c(w * stride)))
        vals = [b.vec_extract(v, q) for q in range(4)]
        tot = vals if tot is None else [b.fadd(tot[q], vals[q]) for q in range(4)]
    return tot


def _regions(e: _E, p, sh: Shape, names):
    out = {}
    for n in names:
        dual, spread = sh.region(n)
        local = p[f"l_{n}"] if dual else None
        out[n] = (e.rsrc(p[n]), dual, spread, local)
    return out


def _ug_task(cx: _Ctx, k: Knobs, p, regs, task):
    """Gate / up task ``task`` (parity: slot task // ug_parts, part task % ug_parts; a slot past the step's count does
    nothing): monofly ``_ug`` / ``_ug_body``."""
    e, b, c, sh, tag = cx.e, cx.e.b, cx.e.c, cx.sh, cx.tag
    S, K, H, I = sh.slots, sh.k, sh.hidden, sh.inter
    G = sh.ug_g
    UGP = sh.ug_parts
    s = b.div(task, c(UGP))
    part = b.sub(task, b.mul(s, c(UGP)))
    count, eid, first, npick = cx.slot(s)
    pitch, spitch = H // 4 + 4, H // 32 + 1
    live_rg = 2 * sh.live // 16 if sh.live and sh.live < I else 0
    with b.scf_if(b.cmp_lt(s, count)):
        pick_lds = b.add(c(sh.rt + 2 + 2 * S), first)
        wt_lds = b.add(c(sh.rt + 2 + 3 * S), first)

        def stage_b():
            if k.gall:
                _gather_static(cx, [(regs["xq"], regs["xs"], H, 0, H, pitch, spitch, TILE * pitch, pick_lds, npick, 0)],
                               min(sh.rows, TILE))  # fmt: skip
            else:
                _gather(cx, regs["xq"], regs["xs"], H, 0, H, pitch, spitch, TILE * pitch, pick_lds, npick)

        skip = None
        if k.mutate == "ug_chain":
            def skip(i, jg, kw):  # noqa: E306 - K step 1 = wave 1's unit 0 (wk 8) / unit 0 of kw 1
                return b.cmp_eq(kw, c(1)) if i == 0 else None
        wk = _gemv(cx, eid=eid, g=G, kk=H, n_rows=2 * I, part=part, w_addr=p["ugw"], s_addr=p["ugws"], pitch=pitch,
                   spitch=spitch, stage_b=stage_b, pre_b=sh.pre_b, live_rg=live_rg, skip=skip)  # fmt: skip
        # epilogue item = (pick n, pair p, f4): 32 features of a pick in 8 adjacent lanes (an MX group)
        pairs = G // 2
        items = TILE * pairs * 4
        order = [1, 0] + list(range(2, wk)) if (k.mutate == "ug_order" and wk > 1) else None
        for ps in range(-(-items // THREADS)):
            tps = b.add(cx.tid, c(ps * THREADS))
            i = b.smin(tps, c(items - 1))
            n = b.div(i, c(pairs * 4))
            pp = b.mod(b.lshr(i, c(2)), c(pairs))
            f4 = b.land(i, c(3))
            ok = b.land(b.cmp_lt(tps, c(items)), b.cmp_lt(n, npick))
            gate = _wave_sum(cx, G, b.mul(pp, c(2)), n, f4, wk, order)
            up = _wave_sum(cx, G, b.add(b.mul(pp, c(2)), c(1)), n, f4, wk, order)
            pick = e.ld(b.add(pick_lds, n))
            w = b.bitcast(e.ld(b.add(wt_lds, n)), F32)
            mid = []
            for qq in range(4):
                gq, uq = e.bfr(gate[qq]), e.bfr(up[qq])
                if sh.limit > 0:
                    gq = e.fmin(gq, e.f(sh.limit))
                    uq = e.fmax(e.fmin(uq, e.f(sh.limit)), e.f(-sh.limit))
                ex = b.exp2_fast(b.fmul(gq, e.f(NEG_LOG2E)))
                silu = b.fdiv(gq, b.fadd(e.f(1.0), ex))
                mid.append(e.bfr(b.fmul(w, b.fmul(silu, uq))))
            m = e.fmax(e.fmax(e.absf(mid[0]), e.absf(mid[1])), e.fmax(e.absf(mid[2]), e.absf(mid[3])))
            for off in (1, 2, 4):
                m = e.fmax(m, b.bitcast(b.ds_swizzle_xor(b.bitcast(m, I32), off), F32))
            code = e.e8m0(m)
            if k.mutate == "code":
                code = b.add(code, c(1))
            inv = b.bitcast(b.shl(b.sub(c(254), code), c(23)), F32)
            qv = [e.fmax(e.fmin(b.fmul(v, inv), e.f(FP8_MAX)), e.f(-FP8_MAX)) for v in mid]
            word = b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(qv, F32)), I32)
            if k.mutate == "mid_bit":
                word = b.xor(word, c(1))
            r = b.div(pick, c(K))
            kp = b.sub(pick, b.mul(r, c(K)))
            el = b.add(b.add(b.mul(r, c(K * I)), b.mul(kp, c(I))),
                       b.add(b.mul(b.add(b.mul(part, c(pairs)), pp), c(16)), b.mul(f4, c(4))))  # fmt: skip
            lead = b.land(b.cmp_eq(b.land(pp, c(1)), c(0)), b.cmp_eq(f4, c(0)))
            with b.scf_if(ok):
                e.put(regs["mq"], b.lshr(el, c(2)), [word, tag])
                with b.scf_if(lead):
                    e.put(regs["ms"], b.lshr(el, c(5)), [code, tag])
        e.barrier()


def _down_task(cx: _Ctx, k: Knobs, p, regs, task):
    """Down task ``task`` (parity: slot task // down_parts, part task % down_parts): monofly ``_down`` /
    ``_down_body``."""
    e, b, c, sh, tag = cx.e, cx.e.b, cx.e.c, cx.sh, cx.tag
    S, K, H, I = sh.slots, sh.k, sh.hidden, sh.inter
    G = sh.down_g
    DP = sh.down_parts
    s = b.div(task, c(DP))
    part = b.sub(task, b.mul(s, c(DP)))
    count, eid, first, npick = cx.slot(s)
    pitch, spitch = I // 4 + 4, I // 32 + 1
    live_k = sh.live if sh.live and sh.live < I else 0
    with b.scf_if(b.cmp_lt(s, count)):
        pick_lds = b.add(c(sh.rt + 2 + 2 * S), first)

        def stage_b():
            if k.gall:
                _gather_static(cx, [(regs["mq"], regs["ms"], K * I, I, I, pitch, spitch, TILE * pitch, pick_lds, npick,
                                     0)], min(sh.rows, TILE))  # fmt: skip
            else:
                _gather(cx, regs["mq"], regs["ms"], K * I, I, I, pitch, spitch, TILE * pitch, pick_lds, npick)

        skip = None
        if k.mutate == "dn_chain":
            def skip(i, jg, kw):  # noqa: E306 - K step 1 (wk 1: unit row i = 1)
                return b.cmp_eq(kw, c(0)) if i == 1 else None
        wk = _gemv(cx, eid=eid, g=G, kk=I, n_rows=H, part=part, w_addr=p["dnw"], s_addr=p["dnws"], pitch=pitch,
                   spitch=spitch, stage_b=stage_b, pre_b=sh.down_pre_b, live_k=live_k, skip=skip)  # fmt: skip
        items = TILE * G * 4
        for ps in range(-(-items // THREADS)):
            tps = b.add(cx.tid, c(ps * THREADS))
            i = b.smin(tps, c(items - 1))
            n = b.div(i, c(G * 4))
            gl = b.mod(b.lshr(i, c(2)), c(G))
            f4 = b.land(i, c(3))
            ok = b.land(b.cmp_lt(tps, c(items)), b.cmp_lt(n, npick))
            y = _wave_sum(cx, G, gl, n, f4, wk)
            pick = e.ld(b.add(pick_lds, n))
            r = b.div(pick, c(K))
            kp = b.sub(pick, b.mul(r, c(K)))
            col = b.add(b.mul(kp, c(H)), b.add(b.mul(b.add(b.mul(part, c(G)), gl), c(16)), b.mul(f4, c(4))))
            el = b.add(b.mul(r, c(sh.y_k * H)), col)
            w01, w23 = e.bf_pair(y[0], y[1]), e.bf_pair(y[2], y[3])
            if k.mutate == "y_bit":
                w01 = b.xor(w01, c(1))
            with b.scf_if(ok):
                e.put(regs["y"], b.lshr(el, c(1)), [w01, tag, w23, tag])
        e.barrier()


def _block_sums(cx: _Ctx, vals, at: int):
    """The CTA-wide sums of the i32 ``vals`` (every thread's), uniform: a wave reduction each, the waves' totals
    through LDS words ``at ..`` (``len(vals) * WAVES``), behind one barrier (and one more before the words are
    reused)."""
    e, b, c = cx.e, cx.e.b, cx.e.c
    sums = [b.wave_reduce(v, "add") for v in vals]
    with b.scf_if(b.cmp_eq(cx.lane, c(0))):
        for i, v in enumerate(sums):
            e.st(b.add(c(at + i * WAVES), cx.wave), v)
    e.barrier()
    out = []
    for i in range(len(vals)):
        tot = None
        for w in range(WAVES):
            x = e.ld(c(at + i * WAVES + w))
            tot = x if tot is None else b.add(tot, x)
        out.append(b.readfirstlane(tot))
    e.barrier()
    return out


def bal_layout(sh: Shape) -> dict:
    """``map=bal``'s LDS words (from lds_base): the x rows (``rows`` x rows), the K-wave partials of a gate / up block
    (after the x rows when they fit below the slot table, else past it), the down phase's two slots' mid rows (over
    the x rows), the budget sums' scratch (past the table)."""
    H, I, R = sh.hidden, sh.inter, sh.rows
    px, sx = H // 4 + 4, H // 32 + 1
    pm, sm = I // 4 + 4, I // 32 + 1
    xw = R * (px + sx)
    red = WAVES * 4 * TILE * 16
    tab_end = sh.rt + 2 + 4 * sh.slots
    scr = -(-tab_end // 16) * 16
    red_at = -(-xw // 16) * 16
    end = scr + 4 * WAVES
    if red_at + red > sh.rt:
        red_at = -(-end // 16) * 16
        end = red_at + red
    mid_b = TILE * (pm + sm)
    if 2 * mid_b > sh.rt:
        raise ValueError("bal: two slots' mid rows meet the slot table")
    return {"x": 0, "px": px, "sx": sx, "red": red_at, "mid": (0, mid_b), "pm": pm, "sm": sm, "scr": scr,
            "words": end}  # fmt: skip


def cols_layout(sh: Shape) -> dict:
    """``map=cols``'s down-phase LDS words (over the gate / up phase's x rows and partials, ``bal_layout``): every
    pick's mid rows (capacity rows x K), the folded shared expert's mid rows, the picks' and the shared rows' y
    contributions ([row][2 row groups][16] f32), the combined sums ([row][2][16] f32, read by monofly's push), the
    picks' inverse map, the shared rows' pick list."""
    S, R, I = sh.slots, sh.rows, sh.inter
    pm, sm = I // 4 + 4, I // 32 + 1
    spm, ssm = sh.sh_inter // 4 + 4, sh.sh_inter // 32 + 1
    at = 0
    out = {"pm": pm, "sm": sm, "spm": spm, "ssm": ssm}
    for name, words in (("mid", S * (pm + sm)), ("shm", R * (spm + ssm)), ("con", S * 32), ("cons", R * 32),
                        ("sum", R * 32), ("inv", S), ("shp", R)):  # fmt: skip
        at = -(-at // 4) * 4
        out[name] = at
        at += words
    out["end"] = at
    if at > sh.rt:
        raise ValueError(f"cols: the down phase's LDS ({at} words) meets the slot table at {sh.rt}: rows {R}, inter {I}")
    return out


def _cols_budget(cx: _Ctx, task, count, bt, cu, cd, L):
    """``map=cols``: (b0, b1, c0, c1): this CTA's gate / up blocks and its down row groups (16 output columns of every
    slot and the shared expert: ``(c * hidden / 16) / 256`` .. , 1 or 2), the blocks sized so every CTA's bytes are
    the mean (its columns' down, the shared gate / up on its CTAs, its blocks; rounded to whole blocks, the rest
    dealt from CTA 0)."""
    e, b, c, sh = cx.e, cx.e.b, cx.e.c, cx.sh
    DR = sh.hidden // 16
    shc = (sh.sh_live or sh.sh_inter) // 32  # the shared down of one row group: 16 rows x its live K, half KiB
    t = cx.tid
    ncol_t = b.sub(b.lshr(b.mul(b.add(t, c(1)), c(DR)), c(8)), b.lshr(b.mul(t, c(DR)), c(8)))
    per_col = b.add(b.mul(count, c(cd)), c(shc))
    e_t = c(0)
    if sh.sh_ug_tasks:
        on = b.land(b.cmp_ge(t, c(sh.sh_ug_base)), b.cmp_lt(t, c(sh.sh_ug_base + sh.sh_ug_tasks)))
        e_t = b.select(on, c(sh.sh_ug_cost), c(0))
    sum_e = sh.sh_ug_cost * sh.sh_ug_tasks
    tot256 = b.add(b.add(b.mul(bt, c(cu)), b.mul(c(DR), per_col)), c(sum_e))
    bud = b.sub(tot256, b.mul(c(GRID), b.add(e_t, b.mul(ncol_t, per_col))))
    nb_t = b.smax(b.div(b.add(bud, c(GRID * cu // 2)), c(GRID * cu)), c(0))
    nb_t = b.select(b.cmp_lt(t, c(GRID)), nb_t, c(0))
    lo, hi, tot = _block_sums(cx, [b.select(b.cmp_lt(t, task), nb_t, c(0)), b.select(b.cmp_le(t, task), nb_t, c(0)),
                                   nb_t], L["scr"])  # fmt: skip
    deficit = b.smax(b.sub(bt, tot), c(0))
    b0 = b.readfirstlane(b.smin(bt, b.add(lo, b.smin(task, deficit))))
    b1 = b.readfirstlane(b.smin(bt, b.add(hi, b.smin(b.add(task, c(1)), deficit))))
    c0 = b.readfirstlane(b.lshr(b.mul(task, c(DR)), c(8)))
    c1 = b.readfirstlane(b.lshr(b.mul(b.add(task, c(1)), c(DR)), c(8)))
    return b0, b1, c0, c1


def _cols_down(cx: _Ctx, k: Knobs, p, regs, count, c0, c1):
    """``map=cols``: row groups c0 .. c1 - 1 (16 output columns each) of every slot's down and the folded shared
    expert's, then the combine of those columns: per row ``0 + y_0 + .. + y_{K-1} + y_shared`` in f32 (the picks in K
    order, the shared last: monofly's ``_combine``), each y bf16 as monofly's ``y`` channel holds it -> LDS ``sum``
    ([row][2][16] f32), which monofly pushes (``MoEExperts`` cols: ``push_rs`` / ``push_partial``). Slots ascending by
    expert id, a row's picks ascending by expert id (the router's order): the K order is the slots' order."""
    e, b, c, sh, tag = cx.e, cx.e.b, cx.e.c, cx.sh, cx.tag
    S, K, H, I, R = sh.slots, sh.k, sh.hidden, sh.inter, sh.rows
    IS, SS = I // 128, sh.sh_inter // 128
    DR = H // 16
    tid, lane, wave, j4, row = cx.tid, cx.lane, cx.wave, cx.j, cx.row
    rt = sh.rt
    L = cols_layout(sh)
    pm, sm, spm, ssm = L["pm"], L["sm"], L["spm"], L["ssm"]
    padded = bool(sh.live and sh.live < I)
    sh_live = sh.sh_live if sh.sh_live and sh.sh_live < sh.sh_inter else 0
    ncol = b.sub(c1, c0)
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    npk = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), count)))  # the step's picks (the table's run end)
    # the inverse map (pick value -> its place in the list) and the shared rows' pick list (row r = pick r K)
    with b.scf_if(b.cmp_lt(tid, npk)):
        e.st(b.add(c(L["inv"]), e.ld(b.add(c(rt + 2 + 2 * S), tid))), tid)
    with b.scf_if(b.cmp_lt(tid, c(R))):
        e.st(b.add(c(L["shp"]), tid), b.mul(tid, c(K)))
    e.barrier()
    # every pick's mid rows and the shared rows in ONE poll (monofly's batches of gather_n rows took a round trip
    # each: 12 at 36 picks of TP2)
    _gather_static(cx, [(regs["mq"], regs["ms"], K * I, I, I, pm, sm, S * pm, c(rt + 2 + 2 * S), npk, L["mid"]),
                        (regs["shq"], regs["shs"], sh.sh_inter, 0, sh.sh_inter, spm, ssm, R * spm, c(L["shp"]), c(R),
                         L["shm"], R)], S)  # fmt: skip
    e.barrier()
    dn_all = I * H // 2 * sh.experts
    r_dw = e.rsrc(p["dnw"], dn_all)
    r_ds = e.rsrc(p["dnws"], dn_all // 16)
    nitem = b.mul(ncol, count)
    nmine = b.readfirstlane(b.div(b.add(b.sub(nitem, wave), c(WAVES - 1)), c(WAVES)))

    def item(i):
        it = b.smin(b.add(wave, b.mul(i, c(WAVES))), b.sub(nitem, c(1)))
        g = b.readfirstlane(b.div(it, count))
        return g, b.readfirstlane(b.sub(it, b.mul(g, count)))

    def dn_issue(i):
        g, s_ = item(i)
        run = b.cmp_lt(i, nmine)
        eid_ = b.readfirstlane(e.ld(b.add(c(rt + 1), s_)))
        u0 = b.mul(b.add(b.mul(eid_, c(DR)), b.add(c0, g)), c(IS))
        out = []
        for st in range(IS):
            sbx = b.add(b.mul(b.add(u0, c(st)), c(64)), lane)
            lv = run
            if padded:
                lv = b.land(lv, b.cmp_lt(b.mul(j4, c(32)), c(sh.live - st * 128)))
            w = b.buffer_load_cp(r_dw, b.select(lv, b.mul(sbx, c(16)), c(OOB_BYTE)), c(0), dwords=4, aux=CM_NT)
            sw = b.buffer_load_cp(r_ds, b.select(run, b.land(sbx, c(~3)), c(OOB_BYTE)), c(0), dwords=1, aux=CM_NT)
            out += [w, b.land(b.lshr(sw, b.mul(b.land(sbx, c(3)), c(8))), c(0xFF))]
        return out

    loop = b.scf_for_iter(c(0), nmine, c(1), [(cx.uid("cq"), v) for v in dn_issue(c(0))], iv_name=cx.uid("ci"))
    with loop as (i, cur):
        cur = list(cur)
        nxt = dn_issue(b.add(i, c(1)))
        g, s_ = item(i)
        fs = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), s_)))
        ns = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), s_))), fs)
        brow = b.add(fs, b.smin(row, b.sub(ns, c(1))))
        acc = zero
        for st in range(IS):
            if k.mutate == "dn_chain" and st == 1:
                continue
            off = b.add(c(L["mid"]), b.add(b.add(b.mul(brow, c(pm)), c(st * 32)), b.mul(j4, c(4))))
            xv = b.vec_concat(e.ld4(off), e.ld4(b.add(off, c(16))))
            xs = e.ld(b.add(c(L["mid"] + S * pm), b.add(b.add(b.mul(brow, c(sm)), c(st * 4)), j4)))
            acc = b.mfma_scale_mx(cur[2 * st], xv, acc, cur[2 * st + 1], xs, shape="16x16x128", fmt_a="fp4",
                                  fmt_b="fp8")  # fmt: skip
        y = [e.bfr(b.vec_extract(acc, q_)) for q_ in range(4)]
        if k.mutate == "y_bit":
            y[0] = b.bitcast(b.xor(b.bitcast(y[0], I32), c(1 << 16)), F32)
        with b.scf_if(b.cmp_lt(row, ns)):
            e.st(b.add(c(L["con"]), b.add(b.mul(b.add(b.mul(b.add(fs, row), c(2)), g), c(16)), b.mul(j4, c(4)))),
                 b.vec_pack([b.bitcast(v, I32) for v in y], I32), 4)  # fmt: skip
        b.scf_yield(*nxt)
    # the folded shared expert's down of these columns (MXFP8 weights, every row a pick), waves 0 .. ncol - 1
    shw = sh.hidden * sh.sh_inter
    r_sw = e.rsrc(p["shdw"], shw)
    r_ss = e.rsrc(p["shdws"], shw // 32)
    with b.scf_if(b.cmp_lt(wave, ncol)):
        g = wave
        wq = []
        for st in range(SS):
            u = b.add(b.mul(b.add(c0, g), c(SS)), c(st))
            ws_h = []
            for h in range(2):
                vo = b.mul(b.add(b.mul(b.add(b.mul(u, c(2)), c(h)), c(64)), lane), c(16))
                if sh_live:
                    lv = b.cmp_lt(b.add(b.mul(j4, c(16)), c(64 * h)), c(sh_live - st * 128))
                    vo = b.select(lv, vo, c(OOB_BYTE))
                ws_h.append(b.buffer_load_cp(r_sw, vo, c(0), dwords=4, aux=CM_NT))
            sbx = b.add(b.mul(u, c(64)), lane)
            sw = b.buffer_load_cp(r_ss, b.land(sbx, c(~3)), c(0), dwords=1, aux=CM_NT)
            wq.append((b.vec_concat(ws_h[0], ws_h[1]), b.land(b.lshr(sw, b.mul(b.land(sbx, c(3)), c(8))), c(0xFF))))
        brow = b.smin(row, c(R - 1))
        acc = zero
        for st in range(SS):
            off = b.add(c(L["shm"]), b.add(b.add(b.mul(brow, c(spm)), c(st * 32)), b.mul(j4, c(4))))
            xv = b.vec_concat(e.ld4(off), e.ld4(b.add(off, c(16))))
            xs = e.ld(b.add(c(L["shm"] + R * spm), b.add(b.add(b.mul(brow, c(ssm)), c(st * 4)), j4)))
            acc = b.mfma_scale_mx(wq[st][0], xv, acc, wq[st][1], xs, shape="16x16x128", fmt_a="fp8", fmt_b="fp8")
        y = [e.bfr(b.vec_extract(acc, q_)) for q_ in range(4)]
        with b.scf_if(b.cmp_lt(row, c(R))):
            e.st(b.add(c(L["cons"]), b.add(b.mul(b.add(b.mul(row, c(2)), g), c(16)), b.mul(j4, c(4)))),
                 b.vec_pack([b.bitcast(v, I32) for v in y], I32), 4)  # fmt: skip
    e.barrier()
    # the combine: thread (row, row group, 4 columns) -> 0 + y_0 + .. + y_{K-1} + y_shared (f32, in order)
    r_ = b.lshr(tid, c(3))
    g = b.land(b.lshr(tid, c(2)), c(1))
    f4 = b.land(tid, c(3))
    ok = b.land(b.cmp_lt(tid, c(R * 8)), b.cmp_lt(g, ncol))
    rr = b.smin(r_, c(R - 1))
    acc = [b.const_f32(0.0)] * 4
    order = list(range(K))
    if k.mutate == "comb_order":
        order = [1, 0] + list(range(2, K))
    for kp in order:
        j = e.ld(b.add(c(L["inv"]), b.add(b.mul(rr, c(K)), c(kp))))
        v = e.ld4f(b.add(c(L["con"]), b.add(b.mul(b.add(b.mul(j, c(2)), g), c(16)), b.mul(f4, c(4)))))
        acc = [b.fadd(acc[q_], b.vec_extract(v, q_)) for q_ in range(4)]
    v = e.ld4f(b.add(c(L["cons"]), b.add(b.mul(b.add(b.mul(rr, c(2)), g), c(16)), b.mul(f4, c(4)))))
    acc = [b.fadd(acc[q_], b.vec_extract(v, q_)) for q_ in range(4)]
    with b.scf_if(ok):
        e.st(b.add(c(L["sum"]), b.add(b.mul(b.add(b.mul(rr, c(2)), g), c(16)), b.mul(f4, c(4)))),
             b.vec_pack([b.bitcast(x, I32) for x in acc], I32), 4)  # fmt: skip
    e.barrier()


def _bal_task(cx: _Ctx, k: Knobs, p, regs, task):
    """``map=bal``: CTA ``task``'s even share of the step's routed MoE in one call. Gate / up in MX blocks (4 row
    groups: one mid MX group of every pick of a slot, all 8 waves splitting K as monofly's ``_wk``), the step's
    ``count x blocks`` dealt in contiguous runs, ``floor`` / ``ceil`` of the mean a CTA; x staged once a CTA (every
    row: a block reads its picks' rows there). Down in row groups (16 output columns of a slot, all K in one wave,
    the ``y`` put straight from the accumulators), contiguous runs sized so every CTA's bytes (its gate / up blocks,
    its down row groups and, on the folded shared expert's CTAs, the shared stages') are the mean: the budgets of all
    256 CTAs computed by every CTA, their prefix by a CTA sum. Per (slot, row group) the arithmetic is monofly's, so
    the bits are the parity port's."""
    e, b, c, sh, tag = cx.e, cx.e.b, cx.e.c, cx.sh, cx.tag
    S, K, H, I, R = sh.slots, sh.k, sh.hidden, sh.inter, sh.rows
    XS, IS = H // 128, I // 128
    padded = bool(sh.live and sh.live < I)
    lrg = 2 * sh.live // 16 if padded else 2 * I // 16
    NBK, NBP = lrg // 4, (2 * I // 16 - lrg) // 4
    DR = H // 16
    live_k = sh.live if padded else I
    cu = 4 * XS * 2  # a gate / up block's weight bytes, in half KiB (MXFP4 units of 1 KiB)
    cd = live_k * 2 // 128  # a down row group's, its live K
    if wk_of(XS, 4) != WAVES:
        raise ValueError("bal: the gate / up K split must be 8 waves")
    L = bal_layout(sh)
    tid, lane, wave, j4, row = cx.tid, cx.lane, cx.wave, cx.j, cx.row
    rt = sh.rt
    count = b.readfirstlane(e.ld(c(rt)))
    bt = b.mul(count, c(NBK))
    dt = b.mul(count, c(DR))

    nug = sh.sh_ug_base if (k.bsh and sh.sh_ug_tasks and k.map == "bal") else GRID  # (bsh: no blocks on those CTAs)

    def ub(x):
        """The first gate / up block of CTA x: Bt blocks dealt over CTAs 0 .. nug - 1."""
        if nug == GRID:
            return b.lshr(b.mul(x, bt), c(8))
        return b.div(b.mul(b.smin(x, c(nug)), bt), c(nug))

    if k.map == "cols":
        b0, b1, c0, c1 = _cols_budget(cx, task, count, bt, cu, cd, L)
    elif k.map == "bal":
        b0, b1 = b.readfirstlane(ub(task)), b.readfirstlane(ub(b.add(task, c(1))))
    if k.map == "bal":
        # ---- every CTA's down budget (thread t < 256: CTA t), and this CTA's prefix
        t = tid
        nb_t = b.sub(ub(b.add(t, c(1))), ub(t))
        e_t = c(0)
        if sh.sh_ug_tasks:
            on = b.land(b.cmp_ge(t, c(sh.sh_ug_base)), b.cmp_lt(t, c(sh.sh_ug_base + sh.sh_ug_tasks)))
            e_t = b.add(e_t, b.select(on, c(sh.sh_ug_cost), c(0)))
        if sh.sh_dn_tasks:
            on = b.land(b.cmp_ge(t, c(sh.sh_dn_base)), b.cmp_lt(t, c(sh.sh_dn_base + sh.sh_dn_tasks)))
            e_t = b.add(e_t, b.select(on, c(sh.sh_dn_cost), c(0)))
        sum_e = sh.sh_ug_cost * sh.sh_ug_tasks + sh.sh_dn_cost * sh.sh_dn_tasks
        num = b.sub(b.add(b.add(b.mul(bt, c(cu)), b.mul(dt, c(cd))), c(sum_e)),
                    b.mul(c(GRID), b.add(e_t, b.mul(nb_t, c(cu)))))  # fmt: skip
        d_t = b.smax(b.div(num, c(GRID * cd)), c(0))
        d_t = b.select(b.cmp_lt(t, c(GRID)), d_t, c(0))
        lo, hi, tot = _block_sums(cx, [b.select(b.cmp_lt(t, task), d_t, c(0)), b.select(b.cmp_le(t, task), d_t, c(0)), d_t],
                                  L["scr"])  # fmt: skip
        deficit = b.smax(b.sub(dt, tot), c(0))
        d0 = b.readfirstlane(b.smin(dt, b.add(lo, b.smin(task, deficit))))
        d1 = b.readfirstlane(b.smin(dt, b.add(hi, b.smin(b.add(task, c(1)), deficit))))
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)

    # ---- the pieces: x into LDS, one gate / up block, a run of down row groups
    px, sx = L["px"], L["sx"]
    pm, sm = L["pm"], L["sm"]
    ug_w, ug_s = 2 * I * H // 2, 2 * I * H // 32
    kw = wave
    units = [(i, jg) for i in range(XS // WAVES) for jg in range(4)]
    pf = min(sh.prefetch, len(units))
    red_at = blk_layout(sh)["red"] if k.map == "blk" else L["red"]

    def stage_x():
        """Every row of x into LDS (two-pair specs of the FP8 words, then the codes), one poll a batch."""
        nq, ncd = R * (H // 8), R * (H // 32)
        specs, dests = [], []
        for it in range(-(-nq // THREADS)):
            i = b.smin(b.add(tid, c(it * THREADS)), c(nq - 1))
            r, w2 = b.div(i, c(H // 8)), b.mod(i, c(H // 8))
            specs.append((regs["xq"], b.add(b.mul(r, c(H // 4)), b.mul(w2, c(2))), 2))
            dests.append(b.add(b.mul(r, c(px)), b.mul(w2, c(2))))
        for it in range(-(-ncd // THREADS)):
            i = b.smin(b.add(tid, c(it * THREADS)), c(ncd - 1))
            r, g = b.div(i, c(H // 32)), b.mod(i, c(H // 32))
            specs.append((regs["xs"], b.add(b.mul(r, c(H // 32)), g), 1))
            dests.append(b.add(c(R * px), b.add(b.mul(r, c(sx)), g)))
        got = e.poll(tag, specs)
        for ws_, d in zip(got, dests):
            for q, w in enumerate(ws_):
                e.st(b.add(d, c(q)), w)
        e.barrier()

    def blk_of(bv):
        s_ = b.readfirstlane(b.div(bv, c(NBK)))
        ex64_ = b.zext(b.readfirstlane(e.ld(b.add(c(rt + 1), s_))), I64)
        return (s_, b.sub(bv, b.mul(s_, c(NBK))), e.rsrc(b.add(p["ugw"], b.mul(ex64_, b.const_i64(ug_w))), ug_w),
                e.rsrc(b.add(p["ugws"], b.mul(ex64_, b.const_i64(ug_s))), ug_s))  # fmt: skip

    def issue_in(jb_, r_w_, r_s_, u, valid=None):
        i, jg = units[u]
        unit = b.add(b.mul(b.add(b.mul(jb_, c(4)), c(jg)), c(XS)), b.add(kw, c(WAVES * i)))
        sb = b.add(b.mul(unit, c(64)), lane)
        vo, so = b.mul(sb, c(16)), b.land(sb, c(~3))
        if valid is not None:
            vo, so = b.select(valid, vo, c(OOB_BYTE)), b.select(valid, so, c(OOB_BYTE))
        w = b.buffer_load_cp(r_w_, vo, c(0), dwords=4, aux=CM_NT)
        sw = b.buffer_load_cp(r_s_, so, c(0), dwords=1, aux=CM_NT)
        return w, b.land(b.lshr(sw, b.mul(b.land(sb, c(3)), c(8))), c(0xFF))

    def ug_one(blk, q=None, nxt_of=None):
        """Gate / up block ``blk`` (slot blk // NBK, its MX block blk % NBK): the weight stream (``q``: its first
        units already issued), the K-wave partials, the mid epilogue. ``nxt_of``: (block, valid) whose first units are
        issued after this block's MFMAs (returned)."""
        s, jb, r_w, r_s = blk_of(blk)
        first = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), s)))
        npick = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), s))), first)
        pick_lds = b.add(c(rt + 2 + 2 * S), first)
        wt_lds = b.add(c(rt + 2 + 3 * S), first)
        pn = e.ld(b.add(pick_lds, b.smin(row, b.sub(npick, c(1)))))
        brow = b.div(pn, c(K))
        if q is None:
            q = [issue_in(jb, r_w, r_s, u) for u in range(pf)]
        q = list(q)
        acc = [zero] * 4
        for u, (i, jg) in enumerate(units):
            if u + pf < len(units):
                q.append(issue_in(jb, r_w, r_s, u + pf))
            st = b.add(kw, c(WAVES * i))
            off = b.add(b.add(b.mul(brow, c(px)), b.mul(st, c(32))), b.mul(j4, c(4)))
            xv = b.vec_concat(e.ld4(off), e.ld4(b.add(off, c(16))))
            xs = e.ld(b.add(c(R * px), b.add(b.add(b.mul(brow, c(sx)), b.mul(st, c(4))), j4)))
            w, ws = q[u]
            new = b.mfma_scale_mx(w, xv, acc[jg], ws, xs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
            if k.mutate == "ug_chain" and i == 0:
                new = b.select(b.cmp_eq(kw, c(1)), acc[jg], new)
            acc[jg] = new
        nxt = None
        if nxt_of is not None:
            nb_v, valid = nxt_of
            _, jb2, rw2, rs2 = blk_of(nb_v)
            nxt = [issue_in(jb2, rw2, rs2, u, valid) for u in range(pf)]
        for jg in range(4):
            off = b.add(c(red_at), b.add(b.mul(b.add(b.mul(b.add(b.mul(kw, c(4)), c(jg)), c(16)), row), c(16)),
                                          b.mul(j4, c(4))))  # fmt: skip
            e.st(off, b.vec_pack([b.bitcast(b.vec_extract(acc[jg], q_), I32) for q_ in range(4)], I32), 4)
        e.barrier()
        # epilogue: thread (pick n, pair pp, f4) < 128; 8 adjacent lanes an MX group
        n = b.lshr(tid, c(3))
        pp = b.land(b.lshr(tid, c(2)), c(1))
        f4 = b.land(tid, c(3))
        nn = b.smin(n, c(TILE - 1))
        ok = b.land(b.cmp_lt(tid, c(TILE * 8)), b.cmp_lt(n, npick))
        order = [1, 0] + list(range(2, WAVES)) if k.mutate == "ug_order" else list(range(WAVES))

        def wsum(gl):
            base = b.add(c(red_at), b.add(b.mul(b.add(b.mul(gl, c(TILE)), nn), c(16)), b.mul(f4, c(4))))
            tot_ = None
            for w_ in order:
                v = e.ld4f(b.add(base, c(w_ * 4 * TILE * 16)))
                vals = [b.vec_extract(v, q_) for q_ in range(4)]
                tot_ = vals if tot_ is None else [b.fadd(tot_[q_], vals[q_]) for q_ in range(4)]
            return tot_

        gate, up = wsum(b.mul(pp, c(2))), wsum(b.add(b.mul(pp, c(2)), c(1)))
        pick = e.ld(b.add(pick_lds, nn))
        w = b.bitcast(e.ld(b.add(wt_lds, nn)), F32)
        mid = []
        for qq in range(4):
            gq, uq = e.bfr(gate[qq]), e.bfr(up[qq])
            if sh.limit > 0:
                gq = e.fmin(gq, e.f(sh.limit))
                uq = e.fmax(e.fmin(uq, e.f(sh.limit)), e.f(-sh.limit))
            ex = b.exp2_fast(b.fmul(gq, e.f(NEG_LOG2E)))
            silu = b.fdiv(gq, b.fadd(e.f(1.0), ex))
            mid.append(e.bfr(b.fmul(w, b.fmul(silu, uq))))
        m = e.fmax(e.fmax(e.absf(mid[0]), e.absf(mid[1])), e.fmax(e.absf(mid[2]), e.absf(mid[3])))
        for off_ in (1, 2, 4):
            m = e.fmax(m, b.bitcast(b.ds_swizzle_xor(b.bitcast(m, I32), off_), F32))
        code = e.e8m0(m)
        if k.mutate == "code":
            code = b.add(code, c(1))
        inv = b.bitcast(b.shl(b.sub(c(254), code), c(23)), F32)
        qv = [e.fmax(e.fmin(b.fmul(v, inv), e.f(FP8_MAX)), e.f(-FP8_MAX)) for v in mid]
        word = b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(qv, F32)), I32)
        if k.mutate == "mid_bit":
            word = b.xor(word, c(1))
        r = b.div(pick, c(K))
        kp = b.sub(pick, b.mul(r, c(K)))
        el = b.add(b.add(b.mul(r, c(K * I)), b.mul(kp, c(I))),
                   b.add(b.mul(b.add(b.mul(jb, c(2)), pp), c(16)), b.mul(f4, c(4))))  # fmt: skip
        with b.scf_if(ok):
            e.put(regs["mq"], b.lshr(el, c(2)), [word, tag])
            with b.scf_if(b.land(b.cmp_eq(pp, c(0)), b.cmp_eq(f4, c(0)))):
                e.put(regs["ms"], b.lshr(el, c(5)), [code, tag])
        if NBP:  # the slot's zero-padding tail of mid (features live .. inter), with its last live block
            tw = (I - sh.live) // 4
            pz = b.sub(tid, c(TILE * 8))
            zn = b.div(pz, c(tw))
            zw = b.sub(pz, b.mul(zn, c(tw)))
            zok = b.land(b.land(b.cmp_ge(pz, c(0)), b.cmp_lt(zn, npick)), b.cmp_eq(jb, c(NBK - 1)))
            zpick = e.ld(b.add(pick_lds, b.smin(b.smax(zn, c(0)), c(TILE - 1))))
            zr = b.div(zpick, c(K))
            zel = b.add(b.add(b.mul(zr, c(K * I)), b.mul(b.sub(zpick, b.mul(zr, c(K))), c(I))), c(sh.live))
            with b.scf_if(zok):
                e.put(regs["mq"], b.add(b.lshr(zel, c(2)), zw), [c(0), tag])
                with b.scf_if(b.cmp_lt(zw, c((I - sh.live) // 32))):
                    e.put(regs["ms"], b.add(b.lshr(zel, c(5)), zw), [c(ZERO_CODE), tag])
        e.barrier()
        return nxt

    dn_all = I * H // 2 * sh.experts

    def down_run(d0, d1, areas, one_slot=False):
        """Down row groups d0 .. d1 - 1 (slot q // DR, row group q % DR; at most two slots, their mid rows at LDS
        ``areas``; ``one_slot``: the run lies in one slot), wave w taking d0 + w, + 8, ..: every K step of a row group
        in one MFMA chain, its picks' y put from the accumulators (no LDS reduction: one K wave)."""
        r_dw = e.rsrc(p["dnw"], dn_all)
        r_ds = e.rsrc(p["dnws"], dn_all // 16)
        nmine = b.readfirstlane(b.div(b.smax(b.add(b.sub(b.sub(d1, d0), wave), c(WAVES - 1)), c(0)), c(WAVES)))

        def rg_of(i):
            qd = b.add(b.add(d0, wave), b.mul(i, c(WAVES)))
            qc = b.smin(qd, b.sub(dt, c(1)))
            s_ = b.readfirstlane(b.div(qc, c(DR)))
            return s_, b.sub(qc, b.mul(s_, c(DR)))

        def dn_issue(i):
            """Row group i of this wave's run: its K units and their scale codes (a unit past the K live, or past
            the run: 0, no memory)."""
            s_, rg = rg_of(i)
            run = b.cmp_lt(i, nmine)
            eid_ = b.readfirstlane(e.ld(b.add(c(rt + 1), s_)))
            u0 = b.mul(b.add(b.mul(eid_, c(DR)), rg), c(IS))
            out = []
            for st in range(IS):
                sbx = b.add(b.mul(b.add(u0, c(st)), c(64)), lane)
                voff = b.mul(sbx, c(16))
                lv = run
                if padded:
                    lv = b.land(lv, b.cmp_lt(b.mul(j4, c(32)), c(sh.live - st * 128)))
                voff = b.select(lv, voff, c(OOB_BYTE))
                w = b.buffer_load_cp(r_dw, voff, c(0), dwords=4, aux=CM_NT)
                sw = b.buffer_load_cp(r_ds, b.select(run, b.land(sbx, c(~3)), c(OOB_BYTE)), c(0), dwords=1, aux=CM_NT)
                out += [w, b.land(b.lshr(sw, b.mul(b.land(sbx, c(3)), c(8))), c(0xFF))]
            return out

        first_q = dn_issue(c(0)) if k.dpre else None  # (dpre: the first row group's units fly while mid is polled)
        sa = b.readfirstlane(b.div(d0, c(DR)))
        sb_ = b.readfirstlane(b.div(b.sub(d1, c(1)), c(DR)))
        fa = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), sa)))
        na = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), sa))), fa)
        if one_slot:
            if k.gall:
                _gather_static(cx, [(regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S),
                                                                                                    fa), na, areas[0])],
                               min(R, TILE))  # fmt: skip
            else:
                _gather(cx, regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S), fa), na,
                        at=areas[0])  # fmt: skip
        elif k.gall:  # (slot b = slot a when the run is in one slot: its rows polled twice)
            fb = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), sb_)))
            nb_ = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), sb_))), fb)
            _gather_static(cx, [(regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S), fa),
                                 na, areas[0]),
                                (regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S), fb),
                                 nb_, areas[1])], min(R, TILE))  # fmt: skip
        else:
            _gather(cx, regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S), fa), na,
                    at=areas[0])  # fmt: skip
            with b.scf_if(b.cmp_ne(sb_, sa)):
                fb = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), sb_)))
                nb_ = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), sb_))), fb)
                _gather(cx, regs["mq"], regs["ms"], K * I, I, I, pm, sm, TILE * pm, b.add(c(rt + 2 + 2 * S), fb), nb_,
                        at=areas[1])  # fmt: skip
        e.barrier()
        if first_q is None:
            first_q = dn_issue(c(0))
        loop = b.scf_for_iter(c(0), nmine, c(1), [(cx.uid("dq"), v) for v in first_q], iv_name=cx.uid("dn"))
        with loop as (i, cur):
            cur = list(cur)
            nxt = dn_issue(b.add(i, c(1)))
            s_, rg = rg_of(i)
            fs = b.readfirstlane(e.ld(b.add(c(rt + 1 + S), s_)))
            ns = b.sub(b.readfirstlane(e.ld(b.add(c(rt + 2 + S), s_))), fs)
            area = c(areas[0]) if one_slot else b.select(b.cmp_eq(s_, sa), c(areas[0]), c(areas[1]))
            brow = b.smin(row, b.sub(ns, c(1)))
            acc = zero
            for st in range(IS):
                if k.mutate == "dn_chain" and st == 1:
                    continue
                off = b.add(area, b.add(b.add(b.mul(brow, c(pm)), c(st * 32)), b.mul(j4, c(4))))
                xv = b.vec_concat(e.ld4(off), e.ld4(b.add(off, c(16))))
                xs = e.ld(b.add(area, b.add(c(TILE * pm), b.add(b.add(b.mul(brow, c(sm)), c(st * 4)), j4))))
                acc = b.mfma_scale_mx(cur[2 * st], xv, acc, cur[2 * st + 1], xs, shape="16x16x128", fmt_a="fp4",
                                      fmt_b="fp8")  # fmt: skip
            pick = e.ld(b.add(c(rt + 2 + 2 * S), b.add(fs, brow)))
            r = b.div(pick, c(K))
            kp = b.sub(pick, b.mul(r, c(K)))
            el = b.add(b.add(b.mul(r, c(sh.y_k * H)), b.mul(kp, c(H))), b.add(b.mul(rg, c(16)), b.mul(j4, c(4))))
            y = [b.vec_extract(acc, q_) for q_ in range(4)]
            w01, w23 = e.bf_pair(y[0], y[1]), e.bf_pair(y[2], y[3])
            if k.mutate == "y_bit":
                w01 = b.xor(w01, c(1))
            with b.scf_if(b.cmp_lt(row, ns)):
                e.put(regs["y"], b.lshr(el, c(1)), [w01, tag, w23, tag])
            b.scf_yield(*nxt)
        e.barrier()  # (the mid rows' LDS is reused by the next run)

    if k.map == "blk":
        # ---- block-coupled: block t + 256 r in round r, its down row groups (its slot's
        # [j DR / NBK, (j + 1) DR / NBK)) one round later, after the next block's gate / up
        LB = blk_layout(sh)
        nr = b.readfirstlane(b.div(b.add(b.sub(bt, task), c(GRID - 1)), c(GRID)))  # my blocks
        with b.scf_if(b.cmp_gt(nr, c(0))):
            stage_x()
            with b.scf_for(c(0), b.add(nr, c(1)), c(1), iv_name=cx.uid("rnd")) as rr:
                with b.scf_if(b.cmp_lt(rr, nr)):
                    ug_one(b.add(task, b.mul(rr, c(GRID))))
                with b.scf_if(b.cmp_gt(rr, c(0))):
                    pb = b.add(task, b.mul(b.sub(rr, c(1)), c(GRID)))
                    ps = b.readfirstlane(b.div(pb, c(NBK)))
                    pj = b.sub(pb, b.mul(ps, c(NBK)))
                    d0_ = b.add(b.mul(ps, c(DR)), b.div(b.mul(pj, c(DR)), c(NBK)))
                    d1_ = b.add(b.mul(ps, c(DR)), b.div(b.mul(b.add(pj, c(1)), c(DR)), c(NBK)))
                    down_run(b.readfirstlane(d0_), b.readfirstlane(d1_), (LB["mid"], LB["mid"]), one_slot=True)
        return

    # ---- gate / up blocks b0 .. b1 - 1
    with b.scf_if(b.cmp_gt(b1, b0)):
        stage_x()
        if k.xpre:  # the next block's first units issued before this block's partials are summed (carried)
            _, jb0, rw0, rs0 = blk_of(b0)
            q0 = [issue_in(jb0, rw0, rs0, u) for u in range(pf)]
            with b.scf_for_iter(b0, b1, c(1), [(cx.uid("xq"), v) for pr in q0 for v in pr],
                                iv_name=cx.uid("blk")) as (blk, cur):  # fmt: skip
                cur = list(cur)
                nb_v = b.add(blk, c(1))
                nxt = ug_one(blk, [(cur[2 * u], cur[2 * u + 1]) for u in range(pf)],
                             (b.smin(nb_v, b.sub(b1, c(1))), b.cmp_lt(nb_v, b1)))  # fmt: skip
                b.scf_yield(*[v for pr in nxt for v in pr])
        else:
            with b.scf_for(b0, b1, c(1), iv_name=cx.uid("blk")) as blk:
                ug_one(blk)

    if k.map == "cols":
        _cols_down(cx, k, p, regs, count, c0, c1)
        return
    # ---- down row groups d0 .. d1 - 1
    with b.scf_if(b.cmp_gt(d1, d0)):
        down_run(d0, d1, L["mid"])


def blk_layout(sh: Shape) -> dict:
    """``map=blk``'s LDS words: the x rows (resident for every round), the gate / up partials after them, one slot's
    mid rows after those (a block's down row groups lie in its slot)."""
    H, I, R = sh.hidden, sh.inter, sh.rows
    xw = R * (H // 4 + 4 + H // 32 + 1)
    red = -(-xw // 16) * 16
    mid = -(-(red + WAVES * 4 * TILE * 16) // 16) * 16
    end = mid + TILE * (I // 4 + 4 + I // 32 + 1)
    if end > sh.rt:
        raise ValueError(f"blk: x rows, partials and a slot's mid rows ({end} words) meet the slot table at {sh.rt}")
    return {"red": red, "mid": mid, "words": end}


# the functions' ABIs (lds_base appended by the device-function lowering); a fused / standalone build takes every
# operand, the stage functions their own
ABI = {
    "ug": ("tag", "task", "ugw", "ugws", "xq", "xs", "mq", "ms", "l_xq", "l_xs", "l_mq", "l_ms"),
    "down": ("tag", "task", "dnw", "dnws", "mq", "ms", "y", "l_mq", "l_ms", "l_y"),
    "fused": ("tag", "task", "ugw", "ugws", "dnw", "dnws", "xq", "xs", "mq", "ms", "y", "l_xq", "l_xs", "l_mq", "l_ms",
              "l_y"),
}
ABI["cols"] = ("tag", "task", "ugw", "ugws", "dnw", "dnws", "shdw", "shdws", "xq", "xs", "mq", "ms", "shq", "shs",
               "l_xq", "l_xs", "l_mq", "l_ms", "l_shq", "l_shs")
_I32 = ("tag", "task", "l_xq", "l_xs", "l_mq", "l_ms", "l_y", "l_shq", "l_shs")


def build_kernel(sh: Shape, phase: str, k: Knobs = Knobs(), symbol: str = "", standalone: bool = False):
    """``phase``: "ug" / "down" (one task a call: monofly's stage of that name), "fused" (one call a CTA: the CTA's
    ug tasks t, t + 256, .., then its down tasks: one stage of 256 tasks). ``standalone``: a kernel of its own for
    the rig (the task is the workgroup id, the slot table copied into LDS from one more ``i64`` argument ``tabg``
    holding monofly's table words; LDS its own pool)."""
    sh.check()
    b = IRBuilder(symbol or f"v41mf_{phase}")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    names = ABI["cols" if (phase == "fused" and k.map == "cols") else phase] + (("tabg",) if standalone else ())
    p = {n: b.param(n, I32 if n in _I32 else I64) for n in names}
    e = _E(b, k)
    c = e.c
    tag = b.readfirstlane(p["tag"])
    task = b.readfirstlane(p["task"])
    for n in names:
        if n.startswith("l_"):
            p[n] = b.readfirstlane(p[n])
    cx = _Ctx(e, sh, tag)
    if standalone:
        task = b.block_id_x()
        r_tab = e.rsrc(p["tabg"])
        nw = 2 + 4 * sh.slots
        for it in range(-(-nw // THREADS)):
            i = b.add(cx.tid, c(it * THREADS))
            with b.scf_if(b.cmp_lt(i, c(nw))):
                e.st(b.add(c(sh.rt), i), b.buffer_load_cp(r_tab, b.mul(i, c(4)), c(0), dwords=1, aux=0))
        e.barrier()
    regs = _regions(e, p, sh, [n for n in REGION_NAMES if n in names])
    if phase == "ug":
        _ug_task(cx, k, p, regs, task)
    elif phase == "down":
        _down_task(cx, k, p, regs, task)
    elif k.map in ("bal", "cols", "blk"):
        _bal_task(cx, k, p, regs, task)
    else:
        with b.scf_for(task, c(sh.slots * sh.ug_parts), c(GRID), iv_name="tug") as t:
            _ug_task(cx, k, p, regs, t)
        with b.scf_for(task, c(sh.slots * sh.down_parts), c(GRID), iv_name="tdn") as t:
            _down_task(cx, k, p, regs, t)
    b.ret()
    return b.kernel


def build_ug_kernel(sh: Shape, k: Knobs = Knobs(), symbol: str = "v41mf_ug"):
    return build_kernel(sh, "ug", k, symbol)


def build_down_kernel(sh: Shape, k: Knobs = Knobs(), symbol: str = "v41mf_down"):
    return build_kernel(sh, "down", k, symbol)


def build_fused_kernel(sh: Shape, k: Knobs = Knobs(), symbol: str = "v41mf_fused"):
    return build_kernel(sh, "fused", k, symbol)


PHASES = {"ug": build_ug_kernel, "down": build_down_kernel, "fused": build_fused_kernel}


def source_digest() -> str:
    """sha256 of this module's bytes (a generated function's identity includes its generator)."""
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def symbol_of(sh: Shape, phase: str, k: Knobs) -> str:
    key = json.dumps({"shape": asdict(sh), "phase": phase, "knobs": k.spec(), "src": source_digest(),
                      "dbg": os.environ.get("ROCKE_DEBUG_LOC", "")}, sort_keys=True)  # fmt: skip
    return f"{SYMBOL_PREFIX}{phase}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def generate(sh: Shape, phase: str, k: Knobs, out_dir: str, *, arch: str = "gfx950", flavor=None) -> dict:
    """The device function of ``phase`` ("ug" | "down") for build ``sh`` -> its record (``symbol``, ``ll``, ``abi``
    with ``ffi_types``, ``lds_words``). Cached: an existing ``<symbol>.json`` is returned as is (the symbol carries the
    shape, knobs and generator digest). Written through a temporary name, then renamed (concurrent ranks)."""
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    sym = symbol_of(sh, phase, k)
    js = os.path.join(out_dir, sym + ".json")
    if os.path.exists(js):
        with open(js) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    fn = lower_device_function(PHASES[phase](sh, k, symbol=sym), arch=arch, symbol=sym, llvm_flavor=flavor,
                               knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
    tmp = f".{os.getpid()}"
    ll = os.path.join(out_dir, sym + ".ll")
    with open(ll + tmp, "w", newline="\n") as f:
        f.write(fn.ir if fn.ir.endswith("\n") else fn.ir + "\n")
    os.replace(ll + tmp, ll)
    meta = fn.meta()
    meta.update(stage=f"v41mf_{phase}", phase=phase, shape=asdict(sh), stage_knobs=k.spec(), ll=ll,
                lds_words=bal_layout(sh)["words"] if k.map in ("bal", "cols", "blk") else max(sh.ug_lds_words,
                                                                                      sh.down_lds_words),
                cols=cols_layout(sh) if k.map == "cols" else None)  # fmt: skip
    with open(js + tmp, "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    os.replace(js + tmp, js)
    return meta


def shape_from_json(d: dict) -> Shape:
    d = dict(d)
    d["regions"] = tuple(tuple(r) for r in d.get("regions", ()))
    return Shape(**d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 MoE stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", required=True, help="JSON of Shape")
    ap.add_argument("--phase", choices=("ug", "down", "both"), default="both")
    ap.add_argument("--knobs", action="append", default=[], help="k=v,.. (repeatable: one function set each)")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    a = ap.parse_args(argv)
    sh = shape_from_json(json.loads(a.shape))
    out = []
    for spec in a.knobs or [""]:
        kn = knobs_from_spec(spec)
        for ph in (("ug", "down") if a.phase == "both" else (a.phase,)):
            m = generate(sh, ph, kn, a.out, arch=a.arch, flavor=a.flavor)
            out.append({"knobs": kn.spec(), "phase": ph, "symbol": m["symbol"], "ll": m["ll"]})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
