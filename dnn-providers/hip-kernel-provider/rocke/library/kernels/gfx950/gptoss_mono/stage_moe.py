# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPT-OSS-120B's routed experts (gate / up + down) of monofly's TP1 decode layer as ONE rocKE stage function
(hybrid path: monofly links it into its FlyDSL layer kernel in place of the ``moe.ug`` / ``moe.down`` stages,
``MoEExperts(rk=...)``, off by default).

Bit-identical re-implementation of monofly ``blocks/gfx950/moe_experts.py`` (``_ug_body`` / ``_down_body`` /
``_gemv`` with the GPT-OSS recipe: ``act="gptoss"``, ``bias``, ``weight_at="down"``, ``combine="bf16"``, route
``"cta"`` with ``route_from``, MXFP4 weights x MXFP8 activations) for a 1-row step (B1S1: 4 picks, 4 slots):

* gate / up, a row group (16 gate or up rows) per wave over all 23 K steps of 128 (monofly's ``ug_g`` 8 has
  ``_wk`` = 1: one ``v_mfma_scale_f32_16x16x128_f8f6f4`` chain a row group, steps ascending), x the pick's MXFP8 row
  (polled from ``xm``) as B; the epilogue (``bf16`` rounds where HF rounds, + bias, ``min(g, 7)``,
  ``clamp(u, -7, 7)``, ``bf16(bf16(u + 1) * bf16(g * bf16(sigmoid(bf16(1.702 g)))))``, the MX group's amax over 8
  lanes, the E8M0 code of ``mx.e8m0_code``, ``x 2^-(code-127)`` saturated, RNE E4M3) -> ``moe.mid``;
* down, a row group (16 output features) per CTA's 8 waves (monofly's ``down_g`` 4 has ``_wk`` = 8: wave ``w`` takes
  K steps w, w + 8, w + 16, the last wave's third step a zero unit with scale code 0, as monofly's; the 8 partials
  summed in wave order), the pick's ``moe.mid`` row as B; ``bf16(bf16(bf16(y) + b) * bf16(w))`` -> ``moe.y``.

Every floating-point op is monofly's, in its order (``llvm.exp.f32``, plain ``fdiv``, ``fptrunc`` RNE,
``llvm.maximum`` / ``minimum`` semantics: ``maxnum`` + a NaN select), the MFMA operands the same integers. Each row
group's arithmetic does not depend on which CTA or task computes it, so the task layout (``FusedKnobs.map``) is free:

* ``map="parity"``: monofly's own layout (ug: 184 tasks of 8 row groups on CTAs 0..183; down: 184 tasks of 4 row
  groups from CTA 184, wrapping), in one stage;
* ``map="bal"`` (structural): the 368 four-row-group MX blocks of gate / up one or two a CTA (CTAs 0..111 two,
  112..255 one), the 736 down row groups only on the light CTAs (5 or 6 each), whose down weights are issued at the
  stage's start (``dpre="early"``: VGPR-resident, streaming while the CTA's gate / up runs) instead of after its
  mid rows land. Bytes a CTA: ~188 KB heavy, ~214 KB light, against monofly's 282 KB on CTAs 0..111.

Mailboxes are monofly's: ``(payload, tag)`` 8-byte pairs, device scope (sc1), polled until every tag matches; every
region of this stage is a dual-copy region of the program (``Program.dual``): loads read the CTA's copy (``local``
0 or 8192 bytes), stores write both; ``moe.mid.q`` / ``moe.y`` are spread over runs (logical byte o at
``o + (o & ~8191)``).

ABI (``rocke_gptoss_moe_<variant>``, ``lds="arg"``; LDS operands are word offsets from ``lds_base`` = the byte
address of monofly's LDS arena)::

    void (i32 tag, i32 task, i32 rt, i64 ugw, i64 ugws, i64 ugb, i64 dnw, i64 dnws, i64 dnb,
          i64 xq, i64 xs, i64 mq, i64 ms, i64 y, i32 local, i32 lds_base)

``rt``: the word offset of monofly's route-"cta" slot table (``_Cfg.rt``: [0] count, [1 + s] expert,
[1 + S + s] first pick, [2 + 2S + i] pick, [2 + 3S + i] its route weight bits); ``task``: this CTA's task (0..255).

Knobs (:class:`FusedKnobs`): ``map``, ``dpre`` (``late``: down weights issued when the down task starts, monofly's
order; ``early``), ``upf`` / ``upre`` (gate / up units in flight a wave, of which issued before x is polled: monofly's
``prefetch`` 8 / ``pre_b`` 4), ``dpre_b`` (down units issued before mid is polled, ``late`` only: monofly's 4),
``pause`` (poll retry: ``s_sleep N``, 0: ``s_nop 0``), ``poll_max``, ``mutate`` (test only, :data:`MUTANTS`).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, fields

from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType

# ---- GPT-OSS-120B at TP1 (monofly ``models/gpt_oss_120b/config.py``: hidden / inter padded to 2944)
H = 2944  # hidden (padded): x's K, down's N
INTER = 2944  # expert width (padded): ug's N / 2, down's K
STEPS = H // 128  # 23 K steps of 128 (ug and down alike)
UG_RG = 2 * INTER // 16  # 368 gate / up row groups an expert (gate 2i, up 2i + 1)
UG_BLOCKS = UG_RG // 4  # 92 MX blocks of 4 row groups (32 mid features) an expert
DN_RG = H // 16  # 184 down row groups an expert
K_TOP = 4  # picks a row
SLOTS = K_TOP  # 1 row: a slot a pick
XW, XC = H // 4, H // 32  # 736 MXFP8 words, 92 codes a row
MW, MC = INTER // 4, INTER // 32
LIMIT, ALPHA, FP8_MAX = 7.0, 1.702, 448.0
ZERO_CODE = 127
HALF = 8192  # dual copies: copy 1 HALF bytes after copy 0 (monofly ``device/gfx950/dual.py``)
WAVES, THREADS, TILE = 8, 512, 16
GRID = 256
CM_NT, CM_DEV = 2, 16
UG_W_BYTES = 2 * INTER * H // 2  # an expert's gate / up MXFP4 bytes
UG_S_BYTES = 2 * INTER * H // 32
DN_W_BYTES = H * INTER // 2
DN_S_BYTES = H * INTER // 32
# monofly parity layout
UG_G, DN_G = 8, 4
UG_PARTS, DN_PARTS = UG_RG // UG_G, DN_RG // DN_G  # 46, 46
UG_TASKS = SLOTS * UG_PARTS  # 184
DN_TASKS = SLOTS * DN_PARTS  # 184
DN_BASE = 184  # monofly's place of moe.down (the base program)
# bal layout
HEAVY = SLOTS * UG_BLOCKS - GRID  # 112 CTAs with two MX blocks
LIGHT = GRID - HEAVY  # 144
DN_TOTAL = SLOTS * DN_RG  # 736
DN_PER, DN_EXTRA = DN_TOTAL // LIGHT, DN_TOTAL % LIGHT  # 5, 16: 16 CTAs take 6
DN_MAX = DN_PER + (1 if DN_EXTRA else 0)
SYMBOL_PREFIX = "rocke_gptoss_"
STAGE = "moe"
POOL_WORDS = 160 * 1024 // 4
# LDS words (from lds_base) this function uses: x row, two mid rows, the partials (below the slot table, checked by
# the monofly side against _Cfg.rt)
PITCH, SPITCH = XW + 4, XC + 1
L_X = 0
L_XS = L_X + PITCH
L_M = L_XS + SPITCH  # mid rows: 2 x (PITCH + SPITCH)
L_RED = L_M + 2 * (PITCH + SPITCH)
RED_WORDS = WAVES * DN_MAX * 16
SC_WORDS = STEPS * 64 // 4  # a row group's scale codes (23 units x 64 B) in words
L_US = L_RED + max(RED_WORDS, WAVES * 16)  # gate / up scale codes: a wave's row group each (scl="lds")
L_DS = L_US + WAVES * SC_WORDS  # down scale codes: a group each
LDS_WORDS = L_DS + DN_MAX * SC_WORDS

ABI = (
    ("tag", "i32"), ("task", "i32"), ("rt", "i32"), ("ugw", "i64"), ("ugws", "i64"), ("ugb", "i64"),
    ("dnw", "i64"), ("dnws", "i64"), ("dnb", "i64"), ("xq", "i64"), ("xs", "i64"), ("mq", "i64"), ("ms", "i64"),
    ("y", "i64"), ("local", "i32"),
)  # fmt: skip
# the regions: (ABI name, monofly region, spread)
REGIONS = (("xq", "xm.q", False), ("xs", "xm.s", False), ("mq", "moe.mid.q", True), ("ms", "moe.mid.s", False),
           ("y", "moe.y", True))  # fmt: skip
# Test-only mutants (each must make the in-situ bitwise A/B differ, or hang-free fail the layer gate):
#   mid_bit:    every mid word's low bit flipped
#   y_bit:      every y pair's low bit flipped
#   ug_chain:   each gate / up chain leaves out K step 1
#   dn_order:   the down partials summed in wave order 1, 0, 2, .. 7 (an fp32 reassociation: may be bit-equal)
#   code:       the mid scale code + 1
MUTANTS = ("mid_bit", "y_bit", "ug_chain", "dn_order", "code")
# report-only: an fp32 reassociation, and code + 1 (halving an E4M3 value is exact unless it goes subnormal: equal
# bits on most rows; layer 0's fixture rows showed none)
REPORT_ONLY = ("dn_order", "code")


_STANDALONE = []


class standalone:
    """Builds made inside are STANDALONE kernels (rocKE-only rig: COMGR vs the Rust direct compiler, and the cheap
    rung of the layout search): the task is the workgroup id, the slot table comes from a global image (one more
    ``i64`` argument, ``tabg``: the same words as monofly's LDS table, ``rt`` = 0), the LDS pool is the kernel's own.
    Not for linking (the ABI differs)."""

    def __enter__(self):
        _STANDALONE.append(1)

    def __exit__(self, *exc):
        _STANDALONE.pop()


@dataclass(frozen=True)
class FusedKnobs:
    map: str = "parity"
    dpre: str = "late"
    upf: int = 8
    upre: int = 4
    dpre_b: int = 4
    dpre_bd: int = 4
    bar: str = "fence"
    pause: int = 1
    poll_max: int = 12
    scl: str = "lds"
    bpre: int = 1
    stamps: int = 0
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.map not in ("parity", "bal"):
            raise ValueError("map: parity | bal")
        if self.dpre not in ("late", "early"):
            raise ValueError("dpre: late | early")
        if not 1 <= self.upre <= self.upf <= STEPS:
            raise ValueError("1 <= upre <= upf <= 23")
        if not 0 <= self.dpre_b <= 3 * DN_MAX or not -1 <= self.dpre_bd <= 3 * DN_MAX:
            raise ValueError("dpre_b / dpre_bd out of range (dpre_bd -1: every unit)")
        if self.bar not in ("fence", "lds"):
            raise ValueError("bar: fence (FlyDSL gpu.barrier: release / acquire fences) | lds (lgkmcnt(0) + s_barrier)")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max: 1..16")
        if self.scl not in ("lds", "reg"):
            raise ValueError("scl: lds (a task's scale codes staged in LDS, monofly's) | reg (a dword a unit a lane)")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def spec(self) -> str:
        d = FusedKnobs()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self) if getattr(self, f.name) != getattr(d, f.name))

    def variant(self) -> str:
        s = self.spec()
        return "base" if not s else s.replace(",", "_").replace("=", "").replace("-", "m")


def knobs_from_spec(spec: str) -> FusedKnobs:
    kw, types = {}, {f.name: f.type for f in fields(FusedKnobs)}
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown knob {k!r}")
        kw[k] = int(v) if types[k] in ("int", int) else v
    return FusedKnobs(**kw)


def symbol(knobs: FusedKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}_{knobs.variant()}"


def f32c(v: float) -> float:
    """The f32 value of a Python float (as FlyDSL's f32 constants hold it)."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


class _E:
    def __init__(self, b: IRBuilder, k: FusedKnobs):
        self.b, self.k = b, k
        self.pool = b.smem_alloc(I32, [POOL_WORDS], name_hint="mono_lds")

    def c(self, v):
        return self.b.const_i32(int(v))

    def f(self, v):
        return self.b.const_f32(f32c(v))

    def rsrc(self, addr, nbytes=0x7FFFFFF0):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def barrier(self):
        if self.k.bar == "lds":  # every barrier of this function orders LDS only (mailbox stores need no fence)
            self.b.sync_lds_only(fence="compiler")
        else:
            self.b.s_barrier_bare(fence="workgroup")  # FlyDSL gpu.barrier: fences + s_barrier

    def pause_text(self) -> str:
        return "s_nop 0" if self.k.pause == 0 else f"s_sleep {self.k.pause}"

    # ---- floats with monofly's (FlyDSL's) semantics
    def fmax(self, a, x):  # llvm.maximum.f32
        b = self.b
        return b.select(b.fcmp("uno", a, x), b.fadd(a, x), b.fmax(a, x))

    def fmin(self, a, x):  # llvm.minimum.f32
        b = self.b
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

    # ---- LDS
    def ld(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=I32, n=1), 0)

    def ldf(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=F32, n=1), 0)

    def ld4(self, at):
        return self.b.smem_load_vN(self.pool, at, dtype=I32, n=4)

    def st(self, at, v, n=1):
        self.b.smem_store_vN(self.pool, [at], v, n)

    # ---- dual-copy regions: (rsrc, spread, local)
    def off(self, reg, byte, load):
        b = self.b
        rs, spread, local = reg
        o = b.add(b.land(byte, self.c(~(HALF - 1))), byte) if spread else byte
        return b.add(o, local) if load else o

    def put(self, reg, pair, words):
        """(payload, tag) pairs from pair ``pair``: ``words`` = [w0, tag] or [w0, tag, w1, tag]; both copies."""
        b = self.b
        data = b.vec_pack(words, I32)
        o = self.off(reg, b.mul(pair, self.c(8)), load=False)
        b.buffer_store_cp(data, reg[0], o, self.c(0), aux=CM_DEV)
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


def _row_gather(e: _E, tag, tid, regq, regs, q0, s0, nw, nc, dst_w, dst_c):
    """One MXFP8 row (``nw`` words from pair ``q0`` of ``regq``, ``nc`` codes from pair ``s0`` of ``regs``) into LDS
    words ``dst_w ..`` / ``dst_c ..``: two-pair specs a thread (monofly ``_gather_batch``'s shape), one batch."""
    b, c = e.b, e.c
    nq = nw // 2
    specs, dests = [], []
    for it in range(-(-nq // THREADS)):
        i = b.smin(b.add(tid, c(it * THREADS)), c(nq - 1))
        specs.append((regq, b.add(q0, b.mul(i, c(2))), 2))
        dests.append(b.add(dst_w, b.mul(i, c(2))))
    for it in range(-(-nc // THREADS)):
        i = b.smin(b.add(tid, c(it * THREADS)), c(nc - 1))
        specs.append((regs, b.add(s0, i), 1))
        dests.append(b.add(dst_c, i))
    got = e.poll(tag, specs)
    for g, d in zip(got, dests):
        for q, w in enumerate(g):
            e.st(b.add(d, c(q)), w)


def _b_operand(e: _E, xw, xc, st, j):
    """The lane's FP8 B operand (8 words) and scale code of K step ``st`` (monofly ``gemv.b_operand``, row 0)."""
    b, c = e.b, e.c
    off = b.add(xw, b.add(b.mul(st, c(32)), b.mul(j, c(4))))
    lo = e.ld4(off)
    hi = e.ld4(b.add(off, c(16)))
    xs = e.ld(b.add(xc, b.add(b.mul(st, c(4)), j)))
    return b.vec_concat(lo, hi), xs


def build_moe_kernel(knobs: FusedKnobs = FusedKnobs()):
    k = knobs
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    sa = bool(_STANDALONE)
    extra = ((("tabg", "i64"),) if sa else ()) + ((("dbg", "i64"),) if sa and k.stamps else ())
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI + extra}
    e = _E(b, k)
    c = e.c
    tag, task, rt, local = (b.readfirstlane(p[n]) for n in ("tag", "task", "rt", "local"))
    if sa:
        task = b.block_id_x()
        r_tab = e.rsrc(p["tabg"])

        def tword(at):
            return b.buffer_load_cp(r_tab, b.mul(at, c(4)), c(0), dwords=1, aux=0)
    else:

        def tword(at):
            return e.ld(at)
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = b.readfirstlane(b.lshr(tid, c(6)))
    j = b.lshr(lane, c(4))
    r_dbg = e.rsrc(p["dbg"]) if "dbg" in p else None

    def stamp(point):
        """(standalone, stamps=1) thread 0: s_memrealtime's low word -> dbg[task 8 + point] (after a barrier)."""
        if r_dbg is None:
            return
        with b.scf_if(b.cmp_eq(tid, c(0))):
            t_ = b.trunc(b.memrealtime(), I32)
            b.buffer_store_cp(t_, r_dbg, b.mul(b.add(b.mul(task, c(8)), c(point)), c(4)), c(0), aux=CM_DEV)

    regs = {n: (e.rsrc(p[n]), spread, local) for n, _, spread in REGIONS}

    # ---- the slot table (1 row: 4 slots, one pick each), read before this stage overwrites LDS
    ex, kp, wt = [], [], []
    for s in range(SLOTS):
        ex.append(b.readfirstlane(tword(b.add(rt, c(1 + s)))))
        first = b.readfirstlane(tword(b.add(rt, c(1 + SLOTS + s))))
        pick = b.readfirstlane(tword(b.add(rt, b.add(c(2 + 2 * SLOTS), first))))
        kp.append(b.land(pick, c(K_TOP - 1)))  # pick = row * K + kp, row 0
        wt.append(b.bitcast(b.readfirstlane(tword(b.add(rt, b.add(c(2 + 3 * SLOTS), first)))), F32))
    e.barrier()
    stamp(0)

    def sel(vals, s):
        """vals[s] for a uniform slot index ``s`` (4 values)."""
        out = vals[0]
        for q in range(1, SLOTS):
            out = b.select(b.cmp_eq(s, c(q)), vals[q], out)
        return out

    def ebase(addr, eid, nbytes):
        return b.add(addr, b.mul(b.zext(eid, I64), b.const_i64(nbytes)))

    # one buffer resource a weight tensor (all 128 experts lie below 2^31 bytes: 1.1 GB gate / up), the expert's
    # offset in the voffset (fewer SGPRs than a resource an expert)
    r_all = {n: e.rsrc(p[n], nb * 128) for n, nb in (("ugw", UG_W_BYTES), ("ugws", UG_S_BYTES), ("dnw", DN_W_BYTES),
                                                    ("dnws", DN_S_BYTES), ("ugb", 2 * INTER * 4),
                                                    ("dnb", H * 4))}  # fmt: skip

    # ---- the task layout: ug (slot, block) per wave half, down groups
    if k.map == "parity":
        ug_on = b.cmp_lt(task, c(UG_TASKS))  # task: slot task / 46, part task % 46, row groups part 8 + wave
        ug_slot = b.div(b.smin(task, c(UG_TASKS - 1)), c(UG_PARTS))
        ug_rg = b.add(b.mul(b.sub(b.smin(task, c(UG_TASKS - 1)), b.mul(ug_slot, c(UG_PARTS))), c(UG_G)), wave)
        ug_wave_on = ug_on
        dt = b.land(b.add(b.sub(task, c(DN_BASE)), c(GRID)), c(GRID - 1))  # (task - 184) mod 256
        dn_on = b.cmp_lt(dt, c(DN_TASKS))
        dts = b.smin(dt, c(DN_TASKS - 1))
        dn_q0 = b.mul(dts, c(DN_G))  # global down group of item 0 (slot = q // 184)
        dn_n = DN_G
        dn_count = b.select(dn_on, c(DN_G), c(0))
    else:
        blk = b.lshr(wave, c(2))  # wave half: block 0 = MX block ``task``, block 1 = ``task + 256`` (task < 112)
        u = b.add(task, b.mul(blk, c(GRID)))
        ug_wave_on = b.cmp_lt(u, c(SLOTS * UG_BLOCKS))
        uc = b.smin(u, c(SLOTS * UG_BLOCKS - 1))
        ug_slot = b.div(uc, c(UG_BLOCKS))
        ug_rg = b.add(b.mul(b.sub(uc, b.mul(ug_slot, c(UG_BLOCKS))), c(4)), b.land(wave, c(3)))
        ug_on = c(1)
        d = b.sub(task, c(HEAVY))
        dn_on = b.cmp_ge(d, c(0))
        dd = b.smax(d, c(0))
        dn_q0 = b.add(b.mul(dd, c(DN_PER)), b.smin(dd, c(DN_EXTRA)))
        dn_n = DN_MAX
        dn_count = b.select(dn_on, b.select(b.cmp_lt(dd, c(DN_EXTRA)), c(DN_PER + 1), c(DN_PER)), c(0))
    ug_slot = b.readfirstlane(ug_slot)
    ug_e = sel(ex, ug_slot)
    ug_kp = sel(kp, ug_slot)

    # ---- down weights (issued early: before gate / up; late: when the down task starts)
    dn_groups = []
    for gi in range(dn_n):
        q = b.smin(b.add(dn_q0, c(gi)), c(DN_TOTAL - 1))
        s = b.readfirstlane(b.div(q, c(DN_RG)))
        rg = b.sub(q, b.mul(s, c(DN_RG)))
        live = b.cmp_lt(c(gi), dn_count)
        dn_groups.append((s, rg, live))
    # the CTA's distinct slots lie at its first and last group (parity: one slot a task)
    dn_slots = [0] if k.map == "parity" else sorted({0, dn_n - 1})

    def dn_unit_loads(gi, i):
        s, rg, live = dn_groups[gi]
        eid = sel(ex, s)
        st = b.add(wave, c(WAVES * i))
        valid = b.cmp_lt(st, c(STEPS))
        unit = b.add(b.add(b.mul(eid, c(DN_RG * STEPS)), b.mul(rg, c(STEPS))), b.smin(st, c(STEPS - 1)))
        # a group past the CTA's count (bal: 5-group CTAs) reads nothing; its MFMAs run on zeros, never stored
        voff = b.select(b.land(valid, live), b.mul(b.add(b.mul(unit, c(64)), lane), c(16)), c(0x7FFFFFF0))
        w = b.buffer_load_cp(r_all["dnw"], voff, c(0), dwords=4, aux=CM_NT)
        if k.scl == "lds":  # the code is read from the staged copy when the MFMA needs it (``dn_code``)
            return w, (gi, i)
        sb = b.add(b.mul(unit, c(64)), lane)
        sw = b.buffer_load_cp(r_all["dnws"], b.land(sb, c(~3)), c(0), dwords=1, aux=CM_NT)
        code = b.land(b.lshr(sw, b.mul(b.land(sb, c(3)), c(8))), c(0xFF))
        return w, b.select(valid, code, c(0))

    def dn_code(gi, i):
        """(scl lds) the A scale code of group ``gi``'s unit ``i`` of this wave from the staged copy (0 past K)."""
        st = b.add(wave, c(WAVES * i))
        byte = b.add(b.add(c(gi * STEPS * 64), b.mul(b.smin(st, c(STEPS - 1)), c(64))), lane)
        word = e.ld(b.add(c(L_DS), b.lshr(byte, c(2))))
        code = b.land(b.lshr(word, b.mul(b.land(byte, c(3)), c(8))), c(0xFF))
        return b.select(b.cmp_lt(st, c(STEPS)), code, c(0))

    def stage_codes(r, at, nchunks, src_of):
        """(scl lds) 16 B chunks 0 .. nchunks - 1 (chunk -> byte offset in ``r``) into LDS words ``at + 4 chunk``:
        loads issued now, stored by the returned ``store()`` (after the caller's other loads)."""
        got = []
        for it in range(-(-nchunks // THREADS)):
            ch = b.add(tid, c(it * THREADS))
            chc = b.smin(ch, c(nchunks - 1))
            got.append((ch, b.buffer_load_cp(r, src_of(chc), c(0), dwords=4, aux=CM_NT)))

        def store():
            for ch, v in got:
                with b.scf_if(b.cmp_lt(ch, c(nchunks))):
                    e.st(b.add(c(at), b.mul(ch, c(4))), v, 4)

        return store

    def dn_codes_issue():
        def src(ch):
            gi_ = b.div(ch, c(SC_WORDS // 4))
            within = b.sub(ch, b.mul(gi_, c(SC_WORDS // 4)))
            q = b.smin(b.add(dn_q0, gi_), c(DN_TOTAL - 1))
            s_ = b.div(q, c(DN_RG))
            rg_ = b.sub(q, b.mul(s_, c(DN_RG)))
            u0 = b.add(b.mul(sel(ex, s_), c(DN_RG * STEPS)), b.mul(rg_, c(STEPS)))
            live = b.cmp_lt(gi_, dn_count)
            return b.select(live, b.add(b.mul(u0, c(64)), b.mul(within, c(16))), c(0x7FFFFFF0))

        return stage_codes(r_all["dnws"], L_DS, dn_n * (SC_WORDS // 4), src)

    dn_units = [(gi, i) for i in range(3) for gi in range(dn_n)]  # monofly's issue order: i-major, group-minor
    dn_q = {}
    dn_st = {}
    if k.dpre == "early":
        for u in dn_units:
            dn_q[u] = dn_unit_loads(*u)
        if k.scl == "lds":
            dn_st["store"] = dn_codes_issue()

    # ---- gate / up: a row group a wave, 23 steps
    ug_u0 = b.add(b.mul(ug_e, c(UG_RG * STEPS)), b.mul(ug_rg, c(STEPS)))  # the wave's row group's first unit

    def ug_unit(st):
        unit = b.add(ug_u0, c(st))
        voff = b.select(ug_wave_on, b.mul(b.add(b.mul(unit, c(64)), lane), c(16)), c(0x7FFFFFF0))
        w = b.buffer_load_cp(r_all["ugw"], voff, c(0), dwords=4, aux=CM_NT)
        if k.scl == "lds":
            return w, st
        sb = b.add(b.mul(unit, c(64)), lane)
        sw = b.buffer_load_cp(r_all["ugws"], b.land(sb, c(~3)), c(0), dwords=1, aux=CM_NT)
        return w, b.land(b.lshr(sw, b.mul(b.land(sb, c(3)), c(8))), c(0xFF))

    def ug_code(st):
        byte = b.add(b.add(b.mul(wave, c(STEPS * 64)), c(st * 64)), lane)
        word = e.ld(b.add(c(L_US), b.lshr(byte, c(2))))
        return b.land(b.lshr(word, b.mul(b.land(byte, c(3)), c(8))), c(0xFF))

    def ug_first_unit(wv):
        """(on, first unit) of wave ``wv``'s row group (any wave of the CTA)."""
        if k.map == "parity":
            tc = b.smin(task, c(UG_TASKS - 1))
            sl = b.div(tc, c(UG_PARTS))
            rg_ = b.add(b.mul(b.sub(tc, b.mul(sl, c(UG_PARTS))), c(UG_G)), wv)
            on_ = ug_on
        else:
            u_ = b.add(task, b.mul(b.lshr(wv, c(2)), c(GRID)))
            on_ = b.cmp_lt(u_, c(SLOTS * UG_BLOCKS))
            uc_ = b.smin(u_, c(SLOTS * UG_BLOCKS - 1))
            sl = b.div(uc_, c(UG_BLOCKS))
            rg_ = b.add(b.mul(b.sub(uc_, b.mul(sl, c(UG_BLOCKS))), c(4)), b.land(wv, c(3)))
        return on_, b.add(b.mul(sel(ex, sl), c(UG_RG * STEPS)), b.mul(rg_, c(STEPS)))

    def ug_codes_issue():
        def src(ch):
            wv = b.div(ch, c(SC_WORDS // 4))
            within = b.sub(ch, b.mul(wv, c(SC_WORDS // 4)))
            on_, u0 = ug_first_unit(wv)
            return b.select(on_, b.add(b.mul(u0, c(64)), b.mul(within, c(16))), c(0x7FFFFFF0))

        return stage_codes(r_all["ugws"], L_US, WAVES * (SC_WORDS // 4), src)

    def ug_ep_issue():
        """The gate / up epilogue's per-thread indices and its bias loads (tid < 16 use them)."""
        hb = b.land(b.lshr(tid, c(3)), c(1))
        pp = b.land(b.lshr(tid, c(2)), c(1))
        f4 = b.land(tid, c(3))
        w0 = b.mul(hb, c(4))  # the block's first wave
        if k.map == "parity":
            on = ug_on
            bi = b.add(b.mul(b.sub(b.smin(task, c(UG_TASKS - 1)), b.mul(ug_slot, c(UG_PARTS))), c(2)), hb)
            slot = ug_slot
        else:
            uu = b.add(task, b.mul(hb, c(GRID)))
            on = b.cmp_lt(uu, c(SLOTS * UG_BLOCKS))
            uu = b.smin(uu, c(SLOTS * UG_BLOCKS - 1))
            slot = b.div(uu, c(UG_BLOCKS))
            bi = b.sub(uu, b.mul(slot, c(UG_BLOCKS)))
        eid, kpx = sel(ex, slot), sel(kp, slot)
        feat = b.add(b.mul(b.add(b.mul(bi, c(2)), pp), c(16)), b.mul(f4, c(4)))  # mid feature of value 0
        # the expert can differ lane to lane (bal: the two blocks' slots): one resource, the expert in the voffset
        fb = b.add(b.mul(eid, c(2 * INTER)), feat)
        bg = b.buffer_load_cp(r_all["ugb"], b.mul(fb, c(4)), c(0), dwords=4, aux=0)
        bu = b.buffer_load_cp(r_all["ugb"], b.mul(b.add(fb, c(INTER)), c(4)), c(0), dwords=4, aux=0)
        return hb, pp, f4, w0, on, kpx, feat, bg, bu

    def dn_ep_issue():
        """The down epilogue's per-thread indices and its bias load (tid < 4 count use them)."""
        gi_t = b.smin(b.lshr(tid, c(2)), c(DN_MAX - 1))
        f4 = b.land(tid, c(3))
        q_t = b.smin(b.add(dn_q0, gi_t), c(DN_TOTAL - 1))
        s_t = b.div(q_t, c(DN_RG))
        rg_t = b.sub(q_t, b.mul(s_t, c(DN_RG)))
        eid, kpx, wtx = sel(ex, s_t), sel(kp, s_t), sel(wt, s_t)
        col = b.add(b.mul(rg_t, c(16)), b.mul(f4, c(4)))
        bias = b.buffer_load_cp(r_all["dnb"], b.mul(b.add(b.mul(eid, c(H)), col), c(4)), c(0), dwords=4, aux=0)
        return gi_t, f4, kpx, wtx, col, bias

    def ug_section():
        uq = [ug_unit(st) for st in range(k.upre)]
        st_codes = ug_codes_issue() if k.scl == "lds" else None
        # x: the pick's MXFP8 row (row 0 of xm) into LDS
        _row_gather(e, tag, tid, regs["xq"], regs["xs"], c(0), c(0), XW, XC, c(L_X), c(L_XS))
        if st_codes is not None:
            st_codes()
        e.barrier()
        stamp(1)
        uq += [ug_unit(st) for st in range(k.upre, k.upf)]
        acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        for st in range(STEPS):
            b.sched_barrier(0)
            if st + k.upf < STEPS and st + k.upf >= len(uq):
                uq.append(ug_unit(st + k.upf))
            w, ws = uq[st]
            if k.mutate == "ug_chain" and st == 1:
                continue
            if k.scl == "lds":
                ws = ug_code(ws)
            xv, xs = _b_operand(e, c(L_X), c(L_XS), c(st), j)
            acc = b.mfma_scale_mx(w, xv, acc, ws, xs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
        b.sched_barrier(0)
        e.barrier()  # every wave done with x
        stamp(2)
        with b.scf_if(b.cmp_eq(b.land(lane, c(15)), c(0))):  # column 0 (the pick): 4 features a lane
            e.st(b.add(c(L_RED), b.add(b.mul(wave, c(16)), b.mul(j, c(4)))),
                 b.vec_pack([b.bitcast(b.vec_extract(acc, q), I32) for q in range(4)], I32), 4)  # fmt: skip
        e.barrier()

        # ---- gate / up epilogue: per MX block (4 row groups = waves 4 hb .. 4 hb + 3), 8 lanes: (pair p, f4)
        nblk = 2  # wave halves (parity: one 8-group task = 2 blocks, waves 0-3 and 4-7)
        with b.scf_if(b.cmp_lt(tid, c(8 * nblk))):
            hb, pp, f4, w0, on, kpx, feat, bg, bu = ug_ep if ug_ep is not None else ug_ep_issue()
            gat = b.add(c(L_RED), b.add(b.mul(b.add(w0, b.mul(pp, c(2))), c(16)), b.mul(f4, c(4))))
            mid = []
            for q in range(4):
                gate = e.ldf(b.add(gat, c(q)))
                up = e.ldf(b.add(gat, c(16 + q)))
                gq, uqv = e.bfr(gate), e.bfr(up)
                gq = e.bfr(b.fadd(gq, b.bitcast(b.vec_extract(bg, q), F32)))
                uqv = e.bfr(b.fadd(uqv, b.bitcast(b.vec_extract(bu, q), F32)))
                gq = e.fmin(gq, e.f(LIMIT))
                uqv = e.fmax(e.fmin(uqv, e.f(LIMIT)), e.f(-LIMIT))
                sg = e.bfr(b.fdiv(e.f(1.0), b.fadd(e.f(1.0), b.exp(b.fneg(e.bfr(b.fmul(gq, e.f(ALPHA))))))))
                mid.append(e.bfr(b.fmul(e.bfr(b.fadd(uqv, e.f(1.0))), e.bfr(b.fmul(gq, sg)))))
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
            el = b.add(b.mul(kpx, c(INTER)), feat)  # mid element (row 0)
            with b.scf_if(on):
                e.put(regs["mq"], b.lshr(el, c(2)), [word, tag])
                with b.scf_if(b.cmp_eq(b.land(tid, c(7)), c(0))):
                    e.put(regs["ms"], b.lshr(el, c(5)), [code, tag])
        e.barrier()
        stamp(3)


    ug_ep = dn_ep = None
    if k.bpre:  # the epilogues' biases requested at entry (monofly's order: before the weight stream)
        ug_ep = ug_ep_issue()
        dn_ep = dn_ep_issue()
    if k.map == "parity":  # CTAs 184.. have no gate / up task (they go straight to down)
        with b.scf_if(ug_on):
            ug_section()
    else:
        ug_section()

    # ---- down: the CTA's groups (none on heavy / ug-only CTAs)
    def down_section(pre_n):
        """The down groups of the CTA; ``pre_n`` of its units (i-major) issued before mid is polled (late)."""
        if k.dpre == "late":
            for u in dn_units[:pre_n]:
                dn_q[u] = dn_unit_loads(*u)
            if k.scl == "lds":
                dn_st["store"] = dn_codes_issue()
        # the mid rows of the CTA's slots (at most two: its first and last group's)
        mrows = {}
        for r_i, gi in enumerate(dn_slots):
            s = dn_groups[gi][0]
            kps = sel(kp, s)
            mw, mc = c(L_M + r_i * (PITCH + SPITCH)), c(L_M + r_i * (PITCH + SPITCH) + PITCH)
            _row_gather(e, tag, tid, regs["mq"], regs["ms"], b.mul(kps, c(MW)), b.mul(kps, c(MC)), MW, MC, mw, mc)
            mrows[gi] = (s, mw, mc)
        if "store" in dn_st:
            dn_st["store"]()
        e.barrier()
        stamp(4)
        if k.dpre == "late":
            for u in dn_units[pre_n:]:
                dn_q[u] = dn_unit_loads(*u)
        s0, mw0, mc0 = mrows[0]
        _, mwl, mcl = mrows.get(dn_n - 1, mrows[0])
        accs = [b.vec_pack([b.const_f32(0.0)] * 4, F32) for _ in range(dn_n)]
        for i in range(3):
            st = b.smin(b.add(wave, c(WAVES * i)), c(STEPS - 1))
            x0 = _b_operand(e, mw0, mc0, st, j)
            xl = _b_operand(e, mwl, mcl, st, j) if len(mrows) > 1 else x0
            for gi in range(dn_n):
                b.sched_barrier(0)
                s, _rg, _live = dn_groups[gi]
                same = b.cmp_eq(s, s0)
                xv = b.select(same, x0[0], xl[0])
                xs = b.select(same, x0[1], xl[1])
                w, ws = dn_q[(gi, i)]
                if k.scl == "lds":
                    ws = dn_code(*ws)
                accs[gi] = b.mfma_scale_mx(w, xv, accs[gi], ws, xs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
        b.sched_barrier(0)
        e.barrier()
        stamp(5)
        with b.scf_if(b.cmp_eq(b.land(lane, c(15)), c(0))):
            for gi in range(dn_n):
                at = b.add(c(L_RED), b.add(b.mul(b.add(b.mul(wave, c(dn_n)), c(gi)), c(16)), b.mul(j, c(4))))
                e.st(at, b.vec_pack([b.bitcast(b.vec_extract(accs[gi], q), I32) for q in range(4)], I32), 4)
        e.barrier()
        with b.scf_if(b.cmp_lt(tid, b.mul(dn_count, c(4)))):
            gi_t, f4, kpx, wtx, col, bias = dn_ep if dn_ep is not None else dn_ep_issue()
            order = [1, 0, 2, 3, 4, 5, 6, 7] if k.mutate == "dn_order" else list(range(WAVES))
            ys = []
            for q in range(4):
                tot = None
                for w_i in order:
                    v = e.ldf(b.add(c(L_RED), b.add(b.mul(b.add(b.mul(c(w_i), c(dn_n)), gi_t), c(16)),
                                                     b.add(b.mul(f4, c(4)), c(q)))))  # fmt: skip
                    tot = v if tot is None else b.fadd(tot, v)
                y = e.bfr(b.fadd(e.bfr(tot), b.bitcast(b.vec_extract(bias, q), F32)))
                y = e.bfr(b.fmul(e.bfr(y), wtx))
                ys.append(y)
            w01, w23 = e.bf_pair(ys[0], ys[1]), e.bf_pair(ys[2], ys[3])
            if k.mutate == "y_bit":
                w01 = b.xor(w01, c(1))
            el = b.add(b.mul(kpx, c(H)), col)  # y element (row 0): pairs el / 2, el / 2 + 1
            e.put(regs["y"], b.lshr(el, c(1)), [w01, tag, w23, tag])
        e.barrier()
        stamp(6)

    if k.map == "parity" and k.dpre == "late":  # down-only CTAs (no gate / up) request every unit at once
        with b.scf_if(b.land(b.cmp_gt(dn_count, c(0)), b.cmp_ge(task, c(UG_TASKS)))):
            down_section(len(dn_units) if k.dpre_bd < 0 else k.dpre_bd)
        with b.scf_if(b.land(b.cmp_gt(dn_count, c(0)), b.cmp_lt(task, c(UG_TASKS)))):
            down_section(k.dpre_b)
    else:
        with b.scf_if(b.cmp_gt(dn_count, c(0))):
            down_section(k.dpre_b)
    b.ret()
    return b.kernel


def lower_moe_function(knobs: FusedKnobs = FusedKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_moe_kernel(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_moe_function(out: str, knobs: FusedKnobs = FusedKnobs(), *, arch="gfx950", flavor=None, bitcode="auto"):
    from ..qwen38_mono.device_function import write_device_function

    fn = lower_moe_function(knobs, arch=arch, flavor=flavor)
    paths = write_device_function(fn, out, stem=f"{STAGE}_{knobs.variant()}", bitcode=bitcode)
    with open(paths["json"]) as f:
        meta = json.load(f)
    meta.update(stage=STAGE, stage_knobs=knobs.spec(), abi_gptoss=[list(x) for x in ABI], lds_words=LDS_WORDS,
                regions=[list(r) for r in REGIONS])  # fmt: skip
    with open(paths["json"], "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE GPT-OSS MoE stage function")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knobs", action="append", default=[], help="k=v,.. (repeatable: one function each)")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    out = []
    for spec in a.knobs or [""]:
        kn = knobs_from_spec(spec)
        out.append({"knobs": kn.spec(), "symbol": symbol(kn), **write_moe_function(a.out, kn, arch=a.arch,
                                                                                    flavor=a.flavor, bitcode=bc)})  # fmt: skip
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
