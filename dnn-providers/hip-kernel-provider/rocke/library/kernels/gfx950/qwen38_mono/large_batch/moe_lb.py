# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L4: the large-batch expert-major MXFP4 MoE (DESIGN-LB S9 / S10, E7 / E8), gfx950.

``experiments/qwen38-true-mono/DESIGN-LB.md`` 6.2 and 12 (P-L4). One persistent launch (256 CTAs x 512 threads, one
CTA per CU) runs a Qwen3.8-Flash-Next MoE layer for ``rows <= bucket`` tokens:

1. **route** (S8's last step): token ``r`` on wave 0 of CTA ``r``: top-10 of its 512 router logits (iterative wave
   argmax, ties to the lower expert id), softmax over the 10, shared gate ``sigmoid(logit[512])``; ids and weights
   are stored ``sc1`` and published with one agent-scope add on the route counter.
2. **sort** (every CTA, redundant, no hop): thread 0 polls the route counter, then the CTA builds per-expert token
   bitmaps in LDS (``ds_add`` of one bit per (expert, token): a token routes to distinct experts), counts, one
   packed block scan (token offsets and 16-token tile prefixes), the tile table (tile ``tau`` -> (expert, tile of the
   expert)) and the token lists (sorted by token id: rank = popcount of the bitmap below the token). The shared
   expert is expert 512 with every row. The last CTA to finish its sort resets the route counter.
3. **gate_up units** (``GU``): (tile ``tau``, 32-column block ``cb`` of the intermediate) on one wave: the two 16-row
   gate tiles and the two matching up tiles over K = 2560 (20 chunks of 128, ``depth`` chunks of weight loads in
   flight), B operand = the tile's 16 gathered token rows of the MoE input (MXFP8 for ``act=fp8``: ATOM's W4A8;
   MXFP4 for ``act=fp4``: FlyTape's W4A4); epilogue ``h = silu(g) u`` (f32), quantized per (token, 32 columns) with
   ATOM's MXFP8 rule (``rocke.helpers.mx_quant``) or FlyTape's RCEIL MXFP4 rule, stored ``sc1`` at the token's list
   position; then one agent-scope add on the (expert, tile) counter.
4. **down units** (``D``): (tile, ``dt`` 16-row tiles of the hidden output) on one wave: all its weight loads issued,
   then lane 0 polls the (expert, tile) gate_up counter (``>= I / 32``), then the ``h`` rows (``sc1``), MFMAs over
   K = I, and the epilogue stores ``w_slot * part`` f32 at ``part[token, slot, columns]`` (``sc1``).
5. **combine** (``combine=phase``): every down lane adds one (no return) to the counter of its (token, column block,
   lane group); after its down units a wave runs combine units (16 tokens x one down column block): all 64 lanes poll
   their own counter in one spin until 11 arrivals (10 routed slots + the shared expert), then sum the 11 slot
   partials in slot order (bitwise deterministic) and write bf16 ``y``. ``combine=last`` (v1, a measured loser):
   the down lane whose returning add sees 10 earlier arrivals combines at once.

Unit placement (``map``): units are ordered by tile (experts in id order, the shared expert last) and dealt to CTAs:
``split`` gives CTA ``t`` the gate_up units ``[t G / 256, (t + 1) G / 256)`` and the down units ``[t D / 256, ..)``;
``line`` (default; ``moe_fused``'s byte line) keeps the gate_up deal and gives the down units so that each CTA's
weight bytes are equal (a gate_up unit weighs ``gu_weight`` down units): ``D(t)`` = running max of
``t W / 256 - r floor(t G / 256)``. Within a CTA, wave ``w`` runs the CTA's gate_up units ``w, w + 8, ..`` then its down
units ``w, w + 8, ..``: every gate_up unit is run before any down unit of the same wave, all CTAs are resident, so
every poll terminates (combine units come after a wave's down units and wait only on down units).

Counters: two sets; layer ``l`` of a launch with epoch ``E`` (layers walked by earlier launches, word ``W_EP``) uses set
``(E + l) & 1`` and zeroes the other one (``sc1`` stores), which the next layer (after the grid barrier) or launch
(after the kernel boundary) uses; the launch's last CTA adds ``layers`` to the epoch. No counter is ever reset by a
returning atomic, and the host zeroes the area once.

``null="oob"``: every expert weight load (codes and scales) is issued against a zero-length buffer resource at its
real offset: zero bytes move, the instruction stream, routing, sort, polls, ``h``, partials and combine are unchanged (the fixed overhead;
wrong outputs by design).

W6 additions (every one a default-off knob except ``hz``; measurements in
``experiments/qwen38-true-mono/large-batch/P-L4.md``): ``hz`` (default on) pads every asm atomic / poll with ``s_nop 4``
(a ``v_readlane`` / ``v_readfirstlane`` write of their SGPR base needs 5 wait states before the asm VMEM reads it; LLVM
does not pad inside inline asm: without it grid-barrier arrivals landed at wrong addresses or faulted);
``combine=gbar`` / ``hand=gbar`` (grid barriers instead of counters, plain loads after one acquire, the first down
unit's weights issued across the hand-off barrier); ``sop`` / ``dsop`` (op_sel scale bytes); ``gcb``, ``dt=8``,
``wbal`` (per-wave byte balance), ``xu`` (cross-unit prefetch), ``gap`` (expert gap repack); stale-read twins
(``pretouch`` + ``drot``) with the must-fail ``noacq`` mutant; diagnosis-only ``diag`` / ``dbgerr`` / ``ustamp``. P-L4 fix
round (twin-only, default off, existing builds' IR unchanged): ``ptall`` (the XCD's CTAs pretouch all of h / hs) and
``tdelay`` (odd CTAs sleep before their gate_up and down units); neither made ``noacq`` detectable in the builds where
the targeted twin misses it (P-L4.md, Fix round). ``layers > 1``: one launch walks the layers with a striped grid barrier between them (the
in-kernel per-layer cost); weights, logits and ``y`` are per layer, the MoE input is shared.

Expert weights (per layer, per expert ``e`` of 513, the shared expert last): gate_up ``[2I, 2560]`` and down
``[2560, I]`` MXFP4 in the *tile* layout ``[M / 16][K chunk][16 rows][chunk bytes]`` (chunk bytes 64, the ragged last
down chunk of TP2 32: K = I = 320), E2M1 codes with the even K in the low nibble; scales ``[M / 16][K chunk][16 rows]
[4]`` E8M0 bytes (one per 32 K). Kernel names start ``q38lb_moe``. Python engine only. Nothing served imports it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_NT, CPOL_SC1, F32, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mx_quant as Q
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "H",
    "NEXP",
    "MoeKnobs",
    "build_moe_kernel",
    "build_stream_kernel",
    "sop_layout_gus",
    "sop_layout_xs",
    "counter_words",
    "dims",
    "kernel_name",
    "moe_params",
    "stream_name",
    "unit_plan",
]

H = 2560  # hidden
E = 512  # routed experts
SH = 512  # the shared expert's index
NEXP = 513
TOPK = 10
SLOTS = 11  # 10 routed + shared
TILE = 16
CHUNK = 128
NQG = H // CHUNK  # 20 gate_up K chunks
LST = 528  # router logit row stride (f32): 513 used
RST = 16  # rid / rw row stride (dwords)
GRID = 256
WAVES = 8
THREADS = 512
XCDS = 8
L2E = 1.4426950408889634
OOB = 0x7FFFFFF0
NRB_MAX = H // 32  # combine counter column blocks (dt = 2)
# counter words (i32): error count, route counter, sort-seen counter, then per (expert, tile) gate_up / down
# counters, then per (token, column block, lane group) combine counters
W_ERR, W_EP, W_END = 0, 16, 32  # error count, launch epoch (layers done so far), end-of-launch arrivals
W_SET = 64  # two counter sets (parity of epoch + layer); a layer uses one and zeroes the other
MAX_TPE = 8  # tiles per expert (bucket 128)
S_RT = 0  # within a set: route counter, then per (expert, tile) gate_up arrivals, then per (token, column block,
S_GU = 16  # lane group) down arrivals for the combine
S_CMB = S_GU + NEXP * MAX_TPE
STAMP_PTS = 8
MUTANTS = ("none", "slot", "wscale", "sort", "race", "cnt", "noacq", "sopsel")


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _f(b: IRBuilder, v: float) -> Value:
    return b.const_f32(float(v))


@dataclass(frozen=True)
class MoeKnobs:
    """One build. Defaults = the P-L4 default arm; the kernel name lists only deviations (and ``tp`` / ``act`` /
    ``bucket``)."""

    tp: int = 1  # 1: I = 640 | 2: the rank's I = 320 (intermediate split)
    act: str = "fp8"  # "fp8" (W4A8, ATOM) | "fp4" (W4A4, FlyTape)
    bucket: int = 128  # rows bucket (16 | 32 | 64 | 128)
    null: str = "none"  # "none" | "oob" (expert weight loads move zero bytes: the fixed overhead)
    depth: int = 3  # gate_up chunks of weight loads in flight (1..6)
    dt: int = 4  # 16-row hidden tiles per down unit (2 | 4 | 8)
    map: str = "line"  # "line" | "split"
    wcp: str = "nt"  # expert weight cache policy: "nt" | "default"
    xcp: str = "acq"  # MoE input loads (written in the launch by S7): "acq" (one ``buffer_inv sc1`` per CTA after the
    #   route poll, then plain loads: P5's acquire) | "sc1" (every load sc1) | "default" (plain, no acquire: only valid
    #   when the input is not produced in the launch)
    combine: str = "phase"  # "phase": down units add one per (token, column block, lane group) with no return, then
    #   combine units (token tile x column block) poll their 64 counters lane-parallel and sum the 11 slots | "last":
    #   the down lane whose returning add sees 10 earlier arrivals combines (v1) | "gbar" (v3): down units add nothing;
    #   after its down units every CTA passes a striped grid barrier, acquires (``buffer_inv sc1``), and the waves of
    #   the grid run combine units (16 tokens x 16 columns) with plain partial loads
    hand: str = "flow"  # gate_up -> down hand-off: "flow" (per (expert, tile) counter, down units poll it, sc1 ``h``
    #   loads) | "gbar" (v3: one striped grid barrier between the gate_up and the down units; every wave issues its first
    #   down unit's weight loads between its CTA's arrival and the poll, then the CTA acquires and ``h`` loads are plain)
    xu: int = 0  # cross-unit prefetch (P-L3's lever): bit 1 gate_up units (needs hand=gbar), bit 2 down units (needs
    #   hand=gbar and combine=gbar): a wave's units form one load stream, no drain between units
    gap: int = 0  # bytes between consecutive experts' weight / scale blocks (FlyTape's EXPERT_GAP: experts at a large
    #   power-of-two stride put the same chunk of every expert on the same HBM channels; a host repack, same bytes read)
    wbal: bool = False  # per-wave byte balance: down units dealt to a CTA's waves as contiguous ranges that even out
    #   each wave's gate_up + down bytes (default: round robin, wave w takes d0 + w, + 8, ..)
    gcb: int = 1  # 32-column blocks per gate_up unit (1 | 2): 2 shares the gathered x rows (B operand) of 8 MFMA tiles
    sop: bool = False  # op_sel scales (P-L3's lever: a scaled MFMA whose scale VGPR was just rewritten is ~8x slower):
    #   gate_up weight and MoE-input scales in the 4-chunk layout (dword (row tile, chunk / 4, row, g) holds the scale
    #   bytes of chunks 4j..4j+3 of the lane's 32-K group g: ``sop_layout``), loaded once per 4 chunks; down weight
    #   and h scales packed in registers once per 4 chunks; the MFMA picks the chunk's byte with op_sel
    dsop: bool = False  # (needs sop) down weight scales in the 4-chunk layout: one scale load per (down tile, 4 chunks)
    pretouch: bool = False  # stale-read twin (numerics only): right before its CTA arrives at a gbar hand-off /
    #   combine barrier every wave loads (plain) the h / partial lines it reads after it, so lines not produced yet sit
    #   in the CU's vL1D / the XCD's L2 with the pre-launch bytes; with the acquire the build must still pass,
    #   ``mutant=noacq`` (no acquire) must fail
    hz: bool = True  # every asm atomic / poll starts with ``s_nop 4``: a v_readlane / v_readfirstlane (VALU) write of an
    #   SGPR needs 5 wait states before a VMEM instruction reads it as its base, and LLVM does not pad VMEM inside inline
    #   asm (False = the v2 emission, which faulted / lost barrier arrivals when the base was a restored spill)
    ustamp: bool = False  # debug (needs stamp): per-wave summed region times at the stamp points instead of phase
    #   stamps: 0 gate_up loop, 1 gate_up epilogue, 2 down issue, 3 down poll, 4 down h + MFMA, 5 down epilogue,
    #   6 combine (+ its barrier), 7 hand-off barrier (s_memtime shader-clock cycles)
    dbgerr: bool = False  # debug: a poll timeout also adds 1 at word W_ERR + site (1 route, 2 gate_up poll, 3 counted
    #   combine, 4 inter-layer barrier, 5 hand-off barrier, 6 combine barrier, 7 pretouch)
    ptall: bool = False  # stale-read twin only (needs pretouch): before the hand-off arrival the 32 CTAs of each XCD
    #   also load (plain) the whole h and hs regions between them, so every XCD's L2 holds a pre-production copy of
    #   every h line (the targeted pretouch only covers the lines the CTA's own down units read; P-L4 fix round)
    tdelay: int = 0  # stale-read twin only (needs pretouch): the odd CTAs (odd XCDs) spin ``tdelay`` x ``s_sleep 127``
    #   before their gate_up units and again after the hand-off barrier before their down units, so the even CTAs
    #   pretouch h / partial lines the odd ones have not written yet (P-L4 fix round: without it the twin of the
    #   non-op_sel gbar build at TP1 fp8 r16 / r32 had too little skew for ``noacq`` to read a stale line)
    drot: int = 0  # stale-read twin only: CTA t runs the down units of CTA (t + drot) mod 256, so the h it reads after
    #   the hand-off barrier was produced by far CTAs (with the line deal a CTA mostly reads its own gate_up output)
    sb: bool = True  # sched_barrier fences around each gate_up chunk's issue block
    stamp: bool = False  # per-wave s_memrealtime at the phase points (last layer): entry, route seen, sorted, gate_up
    #   loop done, down loop done, end (``stamps[(cta * 8 + wave) * 8 + point]``)
    diag: str = "none"  # timing diagnosis only (wrong numerics): "+"-joined subset of nogu (no gate_up units; implies
    #   nopoll), nodn (no down units), nopoll (down units do not wait for gate_up), nocmb (no combine arrival / combine)
    mutant: str = "none"
    tag: str = ""  # recompiled-control name suffix

    # ---------------------------------------------------------------- derived
    @property
    def inter(self) -> int:
        return 640 // self.tp

    @property
    def ngu(self) -> int:  # 32-column blocks of the intermediate per tile (gate_up arrivals per (expert, tile))
        return self.inter // 32

    @property
    def ngu_u(self) -> int:  # gate_up units per tile
        return self.ngu // self.gcb

    @property
    def nqd(self) -> int:  # down K chunks
        return -(-self.inter // CHUNK)

    @property
    def ngrp(self) -> int:  # down K chunk groups of 4 (dsop)
        return -(-self.nqd // 4)

    @property
    def nrb(self) -> int:  # down units per tile
        return H // (TILE * self.dt)

    @property
    def nw(self) -> int:  # bitmap words per expert
        return -(-self.bucket // 32)

    @property
    def tpe(self) -> int:  # max tiles per expert
        return -(-self.bucket // TILE)

    @property
    def tt_max(self) -> int:  # tile table entries
        return (TOPK * self.bucket) // TILE + min(E, TOPK * self.bucket) + self.tpe + 1

    @property
    def gu_weight(self) -> int:  # a gate_up unit's bytes in down units (rounded)
        gu = 4 * NQG * (1024 + 64) * self.gcb
        dn = self.dt * sum(16 * self.chunk_bytes(c) + 64 for c in range(self.nqd))
        return max(1, round(gu / dn))

    def chunk_bytes(self, c: int) -> int:
        """Bytes per row of down K chunk ``c`` (64; the ragged last chunk of I = 320: 32)."""
        return min(64, (self.inter - c * CHUNK) // 2)

    def check(self) -> None:
        if self.tp not in (1, 2):
            raise ValueError("tp 1 | 2")
        if self.act not in ("fp8", "fp4"):
            raise ValueError("act fp8 | fp4")
        if self.bucket not in (16, 32, 64, 128):
            raise ValueError("bucket 16 | 32 | 64 | 128")
        if self.null not in ("none", "oob"):
            raise ValueError("null none | oob")
        if not 1 <= self.depth <= 6:
            raise ValueError("depth 1..6")
        if self.dt not in (2, 4, 8) or (self.dt == 8 and self.tp == 1):
            raise ValueError("dt 2 | 4 | 8 (8 only at tp 2: tp 1 spills)")
        if self.map not in ("line", "split"):
            raise ValueError("map line | split")
        if self.wcp not in ("nt", "default") or self.xcp not in ("acq", "sc1", "default"):
            raise ValueError("wcp nt | default, xcp acq | sc1 | default")
        if self.combine not in ("phase", "last", "gbar"):
            raise ValueError("combine phase | last | gbar")
        if self.hand not in ("flow", "gbar"):
            raise ValueError("hand flow | gbar")
        if self.gcb not in (1, 2):
            raise ValueError("gcb 1 | 2")
        if self.gap < 0 or self.gap % 256:
            raise ValueError("gap: a multiple of 256 bytes")
        if self.xu not in (0, 1, 2, 3) or (self.xu and self.hand != "gbar") or (self.xu & 2 and self.combine != "gbar"):
            raise ValueError("xu 0..3; bit 1 needs hand=gbar, bit 2 needs hand=gbar and combine=gbar")
        if self.xu & 2 and self.pretouch:
            raise ValueError("the stale-read twin has no cross-barrier prefetch: xu bit 2 off")
        if self.ptall and (not self.pretouch or self.hand != "gbar"):
            raise ValueError("ptall needs pretouch and hand=gbar (stale-read twin only)")
        if self.tdelay and (not self.pretouch or not 0 < self.tdelay <= 256):
            raise ValueError("tdelay 1..256 needs pretouch (stale-read twin only)")
        if self.dsop and not self.sop:
            raise ValueError("dsop needs sop")
        if self.mutant == "sopsel" and not self.sop:
            raise ValueError("mutant sopsel needs sop")
        if self.mutant == "noacq" and self.hand != "gbar" and self.combine != "gbar":
            raise ValueError("mutant noacq needs hand=gbar or combine=gbar")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.diag != "none" and not set(self.diag.split("+")) <= {"nogu", "nodn", "nopoll", "nocmb", "nogst", "nodst", "nox", "noh"}:
            raise ValueError("diag none | '+'-joined nogu, nodn, nopoll, nocmb, nogst, nodst, nox, noh")

    def suffix(self) -> str:
        parts = []
        d = MoeKnobs()
        for f in fields(self):
            if f.name in ("tp", "act", "bucket", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}".replace("+", "x"))
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kernel_name(k: MoeKnobs) -> str:
    s = k.suffix()
    return f"q38lb_moe_tp{k.tp}_{k.act}_b{k.bucket}" + (f"_{s}" if s else "")


def moe_params(k: MoeKnobs) -> list[str]:
    """i64 addresses then i32 ``rows, layers, gsync`` (host packs ``<Q...iii``)."""
    p = ["lg", "x", "xs", "wgu", "sgu", "wdn", "sdn", "h", "hs", "rid", "rw", "part", "y", "cnt", "gs"]
    if k.stamp:
        p.append("stamps")
    return p + ["rows", "layers", "gsync"]


def dims(k: MoeKnobs) -> dict:
    """Byte sizes of one expert's blocks and of the per-launch buffers (host allocation; ``unit tests``)."""
    i = k.inter
    gub = 2 * i * H // 2
    gus = (2 * i // TILE) * NQG * 64
    dnb = H * i // 2
    dns = (H // TILE) * (k.ngrp * 256 if k.dsop else k.nqd * 64)
    return dict(
        inter=i, gub=gub, gus=gus, dnb=dnb, dns=dns, expert=gub + gus + dnb + dns,
        hrow=i if k.act == "fp8" else i // 2, hsrow=4 * k.ngu, xrow=H if k.act == "fp8" else H // 2, xsrow=H // 32,
    )


def set_words(k: MoeKnobs) -> int:
    return -(-(S_CMB + k.bucket * NRB_MAX * 4) // 64) * 64


def sop_layout_gus(gus_rows):
    """Host: gate_up weight scales of one expert from the tile layout ``[R/16][NQG][16][4 g]`` (bytes, any leading dims
    flattened by the caller to ``[R/16, NQG, 16, 4]``) to the 4-chunk layout ``[R/16][NQG/4][16][4 g][4 j]``
    (works on numpy / torch arrays with ``reshape`` / ``permute``-style ``transpose``)."""
    a = gus_rows.reshape(gus_rows.shape[0], NQG // 4, 4, 16, 4)  # rt, qg, j, row, g
    return a.permute(0, 1, 3, 4, 2) if hasattr(a, "permute") else a.transpose(0, 1, 3, 4, 2)


def sop_layout_xs(xs):
    """Host: MoE-input scales ``[rows][H/32]`` (byte 4q + g) to ``[rows][NQG/4][4 g][4 j]`` (byte 16 qg + 4 g + j)."""
    a = xs.reshape(xs.shape[0], NQG // 4, 4, 4)  # row, qg, j, g
    return a.permute(0, 1, 3, 2) if hasattr(a, "permute") else a.transpose(0, 1, 3, 2)


def counter_words(k: MoeKnobs) -> int:
    return W_SET + 2 * set_words(k)


def unit_plan(k: MoeKnobs, nt: int, grid: int = GRID) -> list[tuple[int, int, int, int]]:
    """Host model of the in-kernel deal: per CTA ``(g0, g1, d0, d1)`` for ``nt`` tiles (unit tests, analysis)."""
    G, Dn = nt * k.ngu_u, nt * k.nrb
    r = k.gu_weight
    W = r * G + Dn
    out = []

    def dl(t: int) -> int:
        if t >= grid:
            return Dn
        if k.map == "split":
            return t * Dn // grid
        best = 0
        K = ((r + 1) * grid + Dn - 1) // max(Dn, 1)
        for j in range(K + 1):
            tt = t - j
            if tt < 0:
                break
            best = max(best, tt * W // grid - r * (tt * G // grid))
        return min(max(best, 0), Dn)

    for t in range(grid):
        out.append((t * G // grid, (t + 1) * G // grid, dl(t), dl(t + 1)))
    return out


# ============================================================ kernel


def _ptr(b: IRBuilder, addr: Value, t=I32) -> Value:
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b: IRBuilder, v: Value) -> Value:
    return b.zext(v, I64)


def _wave_all(b: IRBuilder, v: Value, op: str) -> Value:
    """Wave-wide max (f32) / signed min (i32) on registers: DPP xor 1, 2, 7, 15 inside each 16-lane row, then the
    permlane16 / permlane32 swaps (``llvm.amdgcn.wave.reduce`` is not in every comgr's LLVM)."""
    f = b.fmax if op == "fmax" else b.smin
    for m in (1, 2, 7, 15):
        v = f(v, b.warp_shuffle_xor_dpp(v, m))
    for sw in (b.permlane16_swap, b.permlane32_swap):
        x = b.bitcast(v, I32) if op == "fmax" else v
        lo, hi = sw(x, x)
        if op == "fmax":
            lo, hi = b.bitcast(lo, F32), b.bitcast(hi, F32)
        v = f(lo, hi)
    return v


def _aadd(k: "MoeKnobs", b: IRBuilder, base: Value, voff: Value, value: Value, *, returns: bool = False, scope: str = "agent"):
    """``device_ops.atomic_add_scoped`` (agent scope), with the SGPR-base hazard pad when ``k.hz``."""
    if not k.hz:
        return D.atomic_add_scoped(b, base, voff, value, scope=scope, returns=returns)
    assert scope == "agent"
    if not returns:
        b.inline_asm("s_nop 4\nglobal_atomic_add $0, $1, $2", "v,v,s,~{memory}", [voff, value, base], None)
        return None
    return b.inline_asm("s_nop 4\nglobal_atomic_add $0, $1, $2, $3 sc0\ns_waitcnt vmcnt(0)", "=&v,v,v,s,~{memory}",
                        [voff, value, base], I32, result_name_hint="aold")


def _pollge(k: "MoeKnobs", b: IRBuilder, base: Value, voff: Value, target: Value, deadline: Value, *, pause: str = "s_sleep1") -> Value:
    """``ccl.wire.poll_count_ge`` with the SGPR-base hazard pad when ``k.hz`` (same spin otherwise)."""
    from kernels.gfx950.qwen38_mono.ccl import wire as W

    if not k.hz:
        return W.poll_count_ge(b, base, voff, target, deadline, pause=pause)
    s_ = "g"
    lines = [
        "s_nop 4",
        "s_mov_b32 $2, 0",
        W._lbl(s_, "top") + ":",
        "global_load_dword $0, $5, $6 sc0 sc1",
        "s_waitcnt vmcnt(0)",
        "v_cmp_lt_u32_e64 $1, $0, $7",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 " + W._lbl(s_, "ok"),
        *W._tail(s_, pause, "$8", "$1", "$4", "$2", "$3"),
        W._lbl(s_, "ok") + ":",
        "s_mov_b32 $3, 0",
        "s_branch " + W._lbl(s_, "end"),
        W._lbl(s_, "timeout") + ":",
        "s_mov_b32 $3, 1",
        W._lbl(s_, "end") + ":",
    ]
    _v, _m, _it, st, _now = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,=&s,v,s,s,v,~{scc},~{memory}",
        [voff, base, target, deadline],
        result_types=[I32, I64, I32, I32, I64],
        result_name_hint="pollg",
    )
    return st


def _flat(tree):
    """Nested lists / tuples of Values (``None`` leaves allowed) -> (values, shape)."""
    if tree is None:
        return [], None
    if isinstance(tree, (list, tuple)):
        vals, shape = [], []
        for t in tree:
            v, s_ = _flat(t)
            vals += v
            shape.append(s_)
        return vals, (type(tree), shape)
    return [tree], "v"


def _unflat(vals, shape, pos=0):
    if shape is None:
        return None, pos
    if shape == "v":
        return vals[pos], pos + 1
    typ, subs = shape
    out = []
    for s_ in subs:
        v, pos = _unflat(vals, s_, pos)
        out.append(v)
    return (tuple(out) if typ is tuple else out), pos


def _bcnt(b: IRBuilder, v: Value) -> Value:
    return b.inline_asm("v_bcnt_u32_b32 $0, $1, 0", "=v,v", [v], I32, sideeffect=False, result_name_hint="bcnt")


class _Moe:
    def __init__(self, k: MoeKnobs):
        k.check()
        self.k = k
        self.dm = dims(k)
        self.b = b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = THREADS
        names = moe_params(k)
        self.a = {n: b.param(n, I64) for n in names[:-3]}
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.L = b.readfirstlane(b.param("layers", I32))
        gs_ = b.readfirstlane(b.param("gsync", I32))
        self.gsync = b.cmp_ne(gs_, _c(b, 0))
        self.cta = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))
        self.l16 = b.land(self.lane, _c(b, 15))
        self.sh8g = b.mul(self.g, _c(b, 8))
        self.lane0 = b.cmp_eq(self.lane, _c(b, 0))
        self.wcp = CPOL_NT if k.wcp == "nt" else 0
        self.xcp = CPOL_SC1 if k.xcp == "sc1" else 0
        self.setw = set_words(k)
        self.cnt_a = b.readfirstlane(self.a["cnt"])
        self.cnt_rs = _rsrc(b, self.cnt_a, _c(b, 4 * counter_words(k)))
        # launch epoch: written by the previous launch's last CTA (kernel boundary: fresh); a vector sc1 load, never
        # a scalar load of a word the device writes (P5)
        self.ep = b.readfirstlane(b.buffer_load_cp(self.cnt_rs, _c(b, 4 * W_EP), _c(b, 0), dwords=1, aux=CPOL_SC1))
        # LDS: the one-CTA-per-CU pad, then the sort tables
        pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
        a0 = b.trunc(b.smem_addr_of(pad), I32)
        with b.scf_if(D.is_thread0(b)):
            b.inline_asm(
                "ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
                "v,v,~{memory}",
                [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)],
                None,
            )
        self.bm = b.smem_alloc(I32, [NEXP * k.nw], name_hint="bm")
        self.eo = b.smem_alloc(I32, [NEXP], name_hint="eo")  # token offset | count << 16
        self.tt = b.smem_alloc(I32, [k.tt_max], name_hint="tt")  # expert | tile << 10
        self.lst = b.smem_alloc(I32, [SLOTS * k.bucket], name_hint="lst")  # token | slot << 8
        self.lw = b.smem_alloc(F32, [SLOTS * k.bucket], name_hint="lw")
        self.misc = b.smem_alloc(I32, [16], name_hint="misc")  # [0] tiles, [8..15] scan wave totals
        # per-layer strides
        dm = self.dm
        g_ = k.gap
        self.lgu = b.const_i64(NEXP * (dm["gub"] + g_))
        self.lgs = b.const_i64(NEXP * (dm["gus"] + g_))
        self.ldn = b.const_i64(NEXP * (dm["dnb"] + g_))
        self.lds_ = b.const_i64(NEXP * (dm["dns"] + g_))
        self.llg = b.const_i64(k.bucket * LST * 4)
        self.ly = b.const_i64(k.bucket * H * 2)
        rl = b.smax(self.rows, _c(b, 1))
        nox = "nox" in k.diag.split("+")  # diag: MoE-input loads move zero bytes
        self.x_rs = _rsrc(b, self.a["x"], _c(b, 0) if nox else b.mul(rl, _c(b, dm["xrow"])))
        self.xs_rs = _rsrc(b, self.a["xs"], _c(b, 0) if nox else b.mul(rl, _c(b, dm["xsrow"])))
        self.h_rs = _rsrc(b, self.a["h"], _c(b, SLOTS * k.bucket * dm["hrow"]))
        self.hs_rs = _rsrc(b, self.a["hs"], _c(b, SLOTS * k.bucket * dm["hsrow"]))
        if "noh" in k.diag.split("+"):  # diag: the down units' h loads move zero bytes (stores unchanged)
            self.hl_rs, self.hsl_rs = _rsrc(b, self.a["h"], _c(b, 0)), _rsrc(b, self.a["hs"], _c(b, 0))
        else:
            self.hl_rs, self.hsl_rs = self.h_rs, self.hs_rs
        self.rid_rs = _rsrc(b, self.a["rid"], _c(b, k.bucket * RST * 4))
        self.rw_rs = _rsrc(b, self.a["rw"], _c(b, k.bucket * RST * 4))
        self.part_rs = _rsrc(b, self.a["part"], _c(b, SLOTS * k.bucket * H * 4))
        self.gsw = b.smem_alloc(F32, [1], name_hint="gsw")
        if k.ustamp:
            self.ust = b.smem_alloc(I32, [WAVES * 8], name_hint="ust")
            with b.scf_if(b.cmp_lt(self.lane, _c(b, 8))):
                self.lds_put(self.ust, b.add(b.mul(self.wave, _c(b, 8)), self.lane), _c(b, 0))

    # ------------------------------------------------------------------ small helpers
    def lds_i(self, smem, idx) -> Value:
        b = self.b
        return b.vec_extract(b.smem_load_vN(smem, idx, dtype=I32, n=1), 0)

    def lds_f(self, smem, idx) -> Value:
        b = self.b
        return b.vec_extract(b.smem_load_vN_f32(smem, idx, n=1), 0)

    def lds_put(self, smem, idx, v) -> None:
        self.b.smem_store_vN(smem, [idx], v, 1)

    def err(self, site: int = 0) -> None:
        _aadd(self.k, self.b, self.cnt_a, _c(self.b, 4 * W_ERR), _c(self.b, 1), scope="agent")
        if self.k.dbgerr and site:
            _aadd(self.k, self.b, self.cnt_a, _c(self.b, 4 * (W_ERR + site)), _c(self.b, 1), scope="agent")

    def set_base(self, layer: Value, other: bool = False) -> Value:
        """First word of the counter set of ``layer`` (or of the other set)."""
        b = self.b
        par = b.land(b.add(b.add(self.ep, layer), _c(b, 1 if other else 0)), _c(b, 1))
        return b.readfirstlane(b.add(_c(b, W_SET), b.mul(par, _c(b, self.setw))))

    def zero_other_set(self, layer: Value) -> None:
        """Zero the set the next layer (or launch) uses: thread g of the grid zeroes words g, g + 131072, .. (sc1)."""
        b = self.b
        ob = self.set_base(layer, other=True)
        gt = b.add(b.mul(self.cta, _c(b, THREADS)), self.tid)
        for i in range(-(-self.setw // (GRID * THREADS))):
            w = b.add(gt, _c(b, i * GRID * THREADS))
            off = b.mul(b.add(ob, w), _c(b, 4))
            b.buffer_store_cp(_c(b, 0), self.cnt_rs, b.select(b.cmp_lt(w, _c(b, self.setw)), off, _c(b, OOB)), _c(b, 0),
                              aux=CPOL_SC1)

    def deadline(self) -> Value:
        b = self.b
        return b.add(D.memrealtime(b), b.const_i64(200_000_000))  # 2 s

    def poll_ge(self, word: Value, target: Value) -> None:
        """Lane 0 of the calling wave spins until counter ``word`` >= ``target`` (2 s deadline: error count)."""
        b = self.b
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        with b.scf_if(self.lane0):
            st = _pollge(self.k, b, self.cnt_a, b.mul(word, _c(b, 4)), target, self.deadline(), pause="s_sleep1")
            with b.scf_if(b.cmp_ne(st, _c(b, 0))):
                self.err(2)

    def dg(self, what: str) -> bool:
        return what in self.k.diag.split("+") or (what == "nopoll" and "nogu" in self.k.diag.split("+"))

    def ut(self):
        return D.memtime(self.b) if self.k.ustamp else None

    def uacc(self, region: int, t0):
        """ustamp: add now - t0 to this wave's region counter; returns now."""
        if not self.k.ustamp or t0 is None:
            return None
        b = self.b
        t1 = D.memtime(b)
        d = b.trunc(b.sub(t1, t0), I32)
        with b.scf_if(self.lane0):
            b.lds_atomic_add(self.ust, [b.add(b.mul(self.wave, _c(b, 8)), _c(b, region))], d)
        return t1

    def stamp(self, layer: Value, p: int) -> None:
        if not self.k.stamp or self.k.ustamp:
            return
        b = self.b
        last = b.cmp_eq(layer, b.sub(self.L, _c(b, 1)))
        with b.scf_if(b.land(self.lane0, last)):
            sp = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            b.store(D.memrealtime(b), sp, b.add(b.mul(b.add(b.mul(self.cta, _c(b, WAVES)), self.wave), _c(b, STAMP_PTS)), _c(b, p)))

    # ------------------------------------------------------------------ route (token r = cta on wave 0)
    def route(self, layer: Value) -> None:
        b, k = self.b, self.k
        r = self.cta
        mine = b.land(b.cmp_lt(r, self.rows), b.cmp_eq(self.wave, _c(b, 0)))
        with b.scf_if(mine):
            lg_a = b.add(self.a["lg"], b.mul(_i64(b, layer), self.llg))
            lg_rs = _rsrc(b, lg_a, _c(b, k.bucket * LST * 4))
            base = b.mul(r, _c(b, LST * 4))
            vals = []
            for h in range(2):
                v = b.buffer_load_cp(lg_rs, b.add(base, b.mul(b.add(b.mul(self.lane, _c(b, 2)), _c(b, h)), _c(b, 16))), _c(b, 0), dwords=4)
                vf = b.vec_bitcast(v, VectorType(F32, 4))
                vals += [b.vec_extract(vf, e) for e in range(4)]
            sgl = b.bitcast(b.buffer_load_cp(lg_rs, b.add(base, _c(b, 4 * E)), _c(b, 0), dwords=1), F32)
            ninf = _f(b, float("-inf"))
            ids, lgs = [], []
            for _ in range(TOPK):
                m = vals[0]
                for v in vals[1:]:
                    m = b.fmax(m, v)
                mx = _wave_all(b, m, "fmax")
                cand = _c(b, 1 << 20)
                for j in reversed(range(8)):
                    cand = b.select(b.fcmp("oeq", vals[j], mx), b.add(b.mul(self.lane, _c(b, 8)), _c(b, j)), cand)
                best = b.readfirstlane(_wave_all(b, cand, "smin"))
                ids.append(best)
                lgs.append(b.readfirstlane(b.bitcast(mx, I32)))
                own = b.cmp_eq(b.lshr(best, _c(b, 3)), self.lane)
                bj = b.land(best, _c(b, 7))
                vals = [b.select(b.land(own, b.cmp_eq(bj, _c(b, j))), ninf, vals[j]) for j in range(8)]
            lgf = [b.bitcast(v, F32) for v in lgs]
            ex = [b.exp2(b.fmul(b.fsub(lv, lgf[0]), _f(b, L2E))) for lv in lgf]
            tot = ex[0]
            for e_ in ex[1:]:
                tot = b.fadd(tot, e_)
            wts = [b.fdiv(e_, tot) for e_ in ex]
            sg = b.fdiv(_f(b, 1.0), b.fadd(_f(b, 1.0), b.exp2(b.fmul(b.fneg(sgl), _f(b, L2E)))))
            my_id = _c(b, 0)
            my_w = sg
            for j in reversed(range(TOPK)):
                sel = b.cmp_eq(self.lane, _c(b, j))
                my_id = b.select(sel, ids[j], my_id)
                my_w = b.select(sel, wts[j], my_w)
            off = b.mul(b.add(b.mul(r, _c(b, RST)), self.lane), _c(b, 4))
            okid = b.cmp_lt(self.lane, _c(b, TOPK))
            okw = b.cmp_le(self.lane, _c(b, TOPK))
            b.buffer_store_cp(my_id, self.rid_rs, b.select(okid, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            b.buffer_store_cp(b.bitcast(my_w, I32), self.rw_rs, b.select(okw, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            b.s_waitcnt(vmcnt=0)
            with b.scf_if(self.lane0):
                _aadd(self.k, b, self.cnt_a, b.mul(b.add(self.sb_, _c(b, S_RT)), _c(b, 4)), _c(b, 1), scope="agent")

    # ------------------------------------------------------------------ sort (whole CTA)
    def sort(self) -> Value:
        """Bitmaps, counts, scan, tile table, token lists; returns the tile count (uniform)."""
        b, k = self.b, self.k
        nw = k.nw
        tid = self.tid
        nbm = NEXP * nw
        for i in range(-(-nbm // THREADS)):
            idx = b.add(tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(idx, _c(b, nbm))):
                self.lds_put(self.bm, idx, _c(b, 0))
        b.sync_lds_only()
        npairs = b.mul(self.rows, _c(b, TOPK))
        pairs = []
        for i in range(-(-(TOPK * k.bucket) // THREADS)):
            p = b.add(tid, _c(b, i * THREADS))
            ok = b.cmp_lt(p, npairs)
            tok = b.div(p, _c(b, TOPK))
            kk = b.sub(p, b.mul(tok, _c(b, TOPK)))
            off = b.mul(b.add(b.mul(tok, _c(b, RST)), kk), _c(b, 4))
            off = b.select(ok, off, _c(b, OOB))
            e = b.buffer_load_cp(self.rid_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1)
            w = b.buffer_load_cp(self.rw_rs, off, _c(b, 0), dwords=1, aux=CPOL_SC1)
            e = b.select(ok, b.land(e, _c(b, 511)), _c(b, 0))
            pairs.append((ok, tok, kk, e, w))
            with b.scf_if(ok):
                b.lds_atomic_add(self.bm, [b.add(b.mul(e, _c(b, nw)), b.lshr(tok, _c(b, 5)))],
                                 b.shl(_c(b, 1), b.land(tok, _c(b, 31))))
        # the shared expert: every row
        for i in range(-(-k.bucket // THREADS)):
            t_ = b.add(tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(t_, self.rows)):
                b.lds_atomic_add(self.bm, [b.add(_c(b, SH * nw), b.lshr(t_, _c(b, 5)))], b.shl(_c(b, 1), b.land(t_, _c(b, 31))))
        b.sync_lds_only()
        # counts and one packed scan over the 512 routed experts (thread e = expert e): cnt | tiles << 16
        cnt = _c(b, 0)
        for w_ in range(nw):
            cnt = b.add(cnt, _bcnt(b, self.lds_i(self.bm, b.add(b.mul(tid, _c(b, nw)), _c(b, w_)))))
        tiles = b.lshr(b.add(cnt, _c(b, 15)), _c(b, 4))
        val = b.add(cnt, b.shl(tiles, _c(b, 16)))
        inc = val
        for s in (1, 2, 4, 8, 16, 32):
            src = b.mul(b.smax(b.sub(self.lane, _c(b, s)), _c(b, 0)), _c(b, 4))
            o = b.ds_bpermute(src, inc)
            inc = b.add(inc, b.select(b.cmp_ge(self.lane, _c(b, s)), o, _c(b, 0)))
        with b.scf_if(b.cmp_eq(self.lane, _c(b, 63))):
            self.lds_put(self.misc, b.add(self.wave, _c(b, 8)), inc)
        b.sync_lds_only()
        pre = _c(b, 0)
        tot = _c(b, 0)
        for w_ in range(WAVES):
            t_ = self.lds_i(self.misc, _c(b, 8 + w_))
            pre = b.add(pre, b.select(b.cmp_lt(_c(b, w_), self.wave), t_, _c(b, 0)))
            tot = b.add(tot, t_)
        exc = b.add(pre, b.sub(inc, val))
        eoff = b.land(exc, _c(b, 0xFFFF))
        tpre = b.lshr(exc, _c(b, 16))
        self.lds_put(self.eo, tid, b.add(eoff, b.shl(cnt, _c(b, 16))))
        for j in range(k.tpe):
            with b.scf_if(b.cmp_lt(_c(b, j), tiles)):
                self.lds_put(self.tt, b.add(tpre, _c(b, j)), b.add(tid, _c(b, j << 10)))
        tot = b.readfirstlane(tot)
        off_sh = b.land(tot, _c(b, 0xFFFF))  # = 10 rows
        tsh0 = b.lshr(tot, _c(b, 16))
        tsh = b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4))
        with b.scf_if(b.cmp_lt(tid, tsh)):
            self.lds_put(self.tt, b.add(tsh0, tid), b.add(_c(b, SH), b.shl(tid, _c(b, 10))))
        with b.scf_if(D.is_thread0(b)):
            self.lds_put(self.eo, _c(b, SH), b.add(off_sh, b.shl(self.rows, _c(b, 16))))
            self.lds_put(self.misc, _c(b, 0), b.add(tsh0, tsh))
        b.sync_lds_only()
        # token lists: rank = popcount of the expert's bitmap below the token
        for ok, tok, kk, e, w in pairs:
            with b.scf_if(ok):
                wi = b.lshr(tok, _c(b, 5))
                below = _c(b, 0)
                for w_ in range(nw - 1):
                    word = self.lds_i(self.bm, b.add(b.mul(e, _c(b, nw)), _c(b, w_)))
                    below = b.add(below, b.select(b.cmp_lt(_c(b, w_), wi), _bcnt(b, word), _c(b, 0)))
                word = self.lds_i(self.bm, b.add(b.mul(e, _c(b, nw)), wi))
                bit = b.land(tok, _c(b, 31))
                msk = b.sub(b.shl(_c(b, 1), bit), _c(b, 1))
                if k.mutant == "sort":
                    msk = b.lor(msk, b.shl(_c(b, 1), bit))  # counts the token itself: every rank off by one
                below = b.add(below, _bcnt(b, b.land(word, msk)))
                pos = b.add(b.land(self.lds_i(self.eo, e), _c(b, 0xFFFF)), below)
                self.lds_put(self.lst, pos, b.add(tok, b.shl(kk, _c(b, 8))))
                self.lds_put(self.lw, pos, b.bitcast(w, F32))
        for i in range(-(-k.bucket // THREADS)):
            t_ = b.add(tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(t_, self.rows)):
                wsh = b.buffer_load_cp(self.rw_rs, b.mul(b.add(b.mul(t_, _c(b, RST)), _c(b, TOPK)), _c(b, 4)), _c(b, 0),
                                       dwords=1, aux=CPOL_SC1)
                pos = b.add(off_sh, t_)
                self.lds_put(self.lst, pos, b.add(t_, _c(b, TOPK << 8)))
                self.lds_put(self.lw, pos, b.bitcast(wsh, F32))
        b.sync_lds_only()
        return b.readfirstlane(self.lds_i(self.misc, _c(b, 0)))

    # ------------------------------------------------------------------ deal
    def deal(self, nt: Value):
        b, k = self.b, self.k
        G = b.mul(nt, _c(b, k.ngu_u))
        Dn = b.mul(nt, _c(b, k.nrb))
        c = self.cta
        g0 = b.div(b.mul(c, G), _c(b, GRID))
        g1 = b.div(b.mul(b.add(c, _c(b, 1)), G), _c(b, GRID))
        if k.drot:  # stale-read twin: the down units of CTA (cta + drot) mod 256
            c = b.land(b.add(c, _c(b, k.drot)), _c(b, GRID - 1))
        if k.map == "split":
            d0 = b.div(b.mul(c, Dn), _c(b, GRID))
            d1 = b.div(b.mul(b.add(c, _c(b, 1)), Dn), _c(b, GRID))
            if k.wbal:
                return g0, g1, *self.wave_deal(g0, g1, d0, d1)
            return g0, g1, d0, d1
        r = k.gu_weight
        W = b.add(b.mul(G, _c(b, r)), Dn)

        # f(t) in (t Dn / N - 1, t Dn / N + r): the running maximum needs only the last K = ceil((r + 1) N / Dn) values
        K = b.readfirstlane(b.div(b.add(_c(b, (r + 1) * GRID), b.sub(Dn, _c(b, 1))), b.smax(Dn, _c(b, 1))))

        def dl(t: Value, nm: str) -> Value:
            loop = b.scf_for_iter(_c(b, 0), b.add(K, _c(b, 1)), _c(b, 1), [(f"best{nm}", _c(b, 0))], iv_name=f"dlw{nm}")
            with loop as (j, vals):
                best = vals[0]
                tt = b.sub(t, j)
                ttc = b.smax(tt, _c(b, 0))
                v = b.sub(b.div(b.mul(ttc, W), _c(b, GRID)), b.mul(_c(b, r), b.div(b.mul(ttc, G), _c(b, GRID))))
                b.scf_yield(b.select(b.cmp_ge(tt, _c(b, 0)), b.smax(best, v), best))
            best = b.smin(b.smax(loop.results[0], _c(b, 0)), Dn)
            return b.select(b.cmp_ge(t, _c(b, GRID)), Dn, best)

        d0, d1 = dl(c, "a"), dl(b.add(c, _c(b, 1)), "b")
        if k.wbal:
            return g0, g1, *self.wave_deal(g0, g1, d0, d1)
        return g0, g1, d0, d1

    def wave_deal(self, g0: Value, g1: Value, d0: Value, d1: Value):
        """``wbal``: this wave's contiguous down-unit range [lo, hi) of the CTA's [d0, d1)."""
        b, k = self.b, self.k
        r = k.gu_weight
        ng = b.sub(g1, g0)
        nd = b.sub(d1, d0)
        W = b.add(b.mul(ng, _c(b, r)), nd)
        gq = b.lshr(ng, _c(b, 3))
        ex = b.land(ng, _c(b, 7))
        m = [_c(b, 0)]
        for v in range(1, WAVES):
            G = b.add(b.mul(_c(b, v), gq), b.smin(_c(b, v), ex))
            f = b.sub(b.div(b.mul(_c(b, v), W), _c(b, WAVES)), b.mul(_c(b, r), G))
            m.append(b.smax(m[-1], b.smin(b.smax(f, _c(b, 0)), nd)))
        m.append(nd)
        lo = m[0]
        hi = m[1]
        for v in range(1, WAVES):
            sel = b.cmp_eq(self.wave, _c(b, v))
            lo = b.select(sel, m[v], lo)
            hi = b.select(sel, m[v + 1], hi)
        return b.readfirstlane(b.add(d0, lo)), b.readfirstlane(b.add(d0, hi))

    def dfirst(self, d0: Value) -> Value:
        return d0 if self.k.wbal else self.b.add(d0, self.wave)

    def dstep(self) -> Value:
        return _c(self.b, 1 if self.k.wbal else WAVES)

    def tile_info(self, tau: Value):
        """(expert, tile of the expert, list offset, token count) of tile ``tau`` (uniform)."""
        b = self.b
        te = b.readfirstlane(self.lds_i(self.tt, tau))
        e = b.land(te, _c(b, 1023))
        ti = b.lshr(te, _c(b, 10))
        eo = b.readfirstlane(self.lds_i(self.eo, e))
        return e, ti, b.land(eo, _c(b, 0xFFFF)), b.lshr(eo, _c(b, 16))

    # ------------------------------------------------------------------ gate_up unit
    def gu_unit(self, layer: Value, j: Value, head=None, nxt=None, has: Value | None = None, ctx_only: bool = False):
        """One gate_up unit. ``ctx_only``: return its chunk-issue function (``has`` false: unit 0's addresses with
        zero-length weight resources). ``head``: its first ``depth`` chunks already issued (xu); ``nxt``: the next
        unit's issue function, called for the chunks past this unit's end; returns those (the next head)."""
        b, k, dm = self.b, self.k, self.dm
        t0 = None if ctx_only else self.ut()
        if has is not None:
            j = b.select(has, j, _c(b, 0))
        tau = b.div(j, _c(b, k.ngu_u))
        cbi = b.sub(j, b.mul(tau, _c(b, k.ngu_u)))
        cbs = [cbi] if k.gcb == 1 else [b.add(b.mul(cbi, _c(b, k.gcb)), _c(b, c_)) for c_ in range(k.gcb)]
        e, ti, off, n = self.tile_info(tau)
        p = b.add(b.mul(ti, _c(b, TILE)), self.l16)
        valid = b.cmp_lt(p, n)
        pos = b.add(off, b.smin(p, b.sub(b.smax(n, _c(b, 1)), _c(b, 1))))
        tok = b.land(self.lds_i(self.lst, pos), _c(b, 255))
        el = _i64(b, e)
        wa = b.add(b.add(self.a["wgu"], b.mul(_i64(b, layer), self.lgu)), b.mul(el, b.const_i64(dm["gub"] + k.gap)))
        sa = b.add(b.add(self.a["sgu"], b.mul(_i64(b, layer), self.lgs)), b.mul(el, b.const_i64(dm["gus"] + k.gap)))
        # null=oob: zero-length resources, so every weight load (same offsets, same instructions) moves no bytes
        nul = k.null == "oob"
        if has is None or nul:
            w_rs = _rsrc(b, wa, _c(b, 0 if nul else dm["gub"]))
            s_rs = _rsrc(b, sa, _c(b, 0 if nul else dm["gus"]))
        else:
            w_rs = _rsrc(b, wa, b.select(has, _c(b, dm["gub"]), _c(b, 0)))
            s_rs = _rsrc(b, sa, b.select(has, _c(b, dm["gus"]), _c(b, 0)))
        nrt = k.inter // TILE  # up tiles start here
        rts = []
        for cb in cbs:  # per 32-column block: gate tiles 2 cb, 2 cb + 1, then the matching up tiles
            rts += [b.add(b.mul(cb, _c(b, 2)), _c(b, t)) for t in range(2)] + [b.add(b.mul(cb, _c(b, 2)), _c(b, nrt + t)) for t in range(2)]
        nwt = len(rts)
        xrow = dm["xrow"]

        sop = k.sop

        def issue(q: int) -> list:
            out = []
            for rt in rts:
                blk = b.add(b.mul(rt, _c(b, NQG)), _c(b, q))
                coff = b.add(b.add(b.mul(blk, _c(b, 1024)), b.mul(self.l16, _c(b, 64))), b.mul(self.g, _c(b, 16)))
                if not sop:
                    soff = b.add(b.mul(blk, _c(b, 64)), b.mul(self.l16, _c(b, 4)))
                    out.append(b.buffer_load_cp(w_rs, coff, _c(b, 0), dwords=4, aux=self.wcp))
                    out.append(b.buffer_load_cp(s_rs, soff, _c(b, 0), dwords=1, aux=self.wcp))
                    continue
                out.append(b.buffer_load_cp(w_rs, coff, _c(b, 0), dwords=4, aux=self.wcp))
                if q % 4 == 0:  # 4-chunk layout: ((rt * NQG / 4 + q / 4) * 16 + row) * 16 + 4 g
                    sblk = b.add(b.mul(rt, _c(b, NQG // 4)), _c(b, q // 4))
                    soff = b.add(b.mul(b.add(b.mul(sblk, _c(b, 16)), self.l16), _c(b, 16)), b.mul(self.g, _c(b, 4)))
                    out.append(b.buffer_load_cp(s_rs, soff, _c(b, 0), dwords=1, aux=self.wcp))
                else:
                    out.append(None)
            if k.act == "fp8":
                for h_ in range(2):
                    xo = b.add(b.mul(tok, _c(b, xrow)), _c(b, q * CHUNK))
                    xo = b.add(xo, b.add(b.mul(self.g, _c(b, 16)), _c(b, 64 * h_)))
                    out.append(b.buffer_load_cp(self.x_rs, b.select(valid, xo, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.xcp))
            else:
                xo = b.add(b.mul(tok, _c(b, xrow)), b.add(_c(b, q * 64), b.mul(self.g, _c(b, 16))))
                out.append(b.buffer_load_cp(self.x_rs, b.select(valid, xo, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.xcp))
            if not sop:
                xso = b.add(b.mul(tok, _c(b, dm["xsrow"])), _c(b, 4 * q))
                out.append(b.buffer_load_cp(self.xs_rs, b.select(valid, xso, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.xcp))
            elif q % 4 == 0:  # 4-chunk layout: token row * 80 + (q / 4) * 16 + 4 g
                xso = b.add(b.mul(tok, _c(b, dm["xsrow"])), b.add(_c(b, 16 * (q // 4)), b.mul(self.g, _c(b, 4))))
                out.append(b.buffer_load_cp(self.xs_rs, b.select(valid, xso, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.xcp))
            else:
                out.append(None)
            return out

        if ctx_only:
            return issue
        acc = [b.zero_vec_f32(4) for _ in range(nwt)]
        dep = min(k.depth, NQG)
        q_ = [issue(q) for q in range(dep)] if head is None else list(head)
        nhead = []
        fb = "fp8" if k.act == "fp8" else "fp4"
        for q in range(NQG):
            if k.sb:
                b.sched_barrier(0)
            if q + dep < NQG:
                q_.append(issue(q + dep))
            elif nxt is not None:
                nhead.append(nxt(q + dep - NQG))
            if k.sb:
                b.sched_barrier(0)
            cur = q_.pop(0)
            xi = 2 * nwt
            if k.act == "fp8":
                bv = b.vec_concat(cur[xi], cur[xi + 1])
                xs_v = cur[xi + 2]
            else:
                bv = cur[xi]
                xs_v = cur[xi + 1]
            if sop:
                if q % 4 == 0:
                    gsc = [cur[2 * t + 1] for t in range(nwt)]
                    gxs = xs_v
                sel = (q + 1) % 4 if k.mutant == "sopsel" else q % 4
                for t in range(nwt):
                    acc[t] = b.mfma_scale_mx(cur[2 * t], bv, acc[t], gsc[t], gxs, shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                             opsel_a=sel, opsel_b=q % 4)
                continue
            bs = b.lshr(xs_v, self.sh8g)
            for t in range(nwt):
                sw = cur[2 * t + 1]
                sa_ = sw if k.mutant == "wscale" else b.lshr(sw, self.sh8g)
                acc[t] = b.mfma_scale_mx(cur[2 * t], bv, acc[t], sa_, bs, shape="16x16x128", fmt_a="fp4", fmt_b=fb)
        if k.sb:
            b.sched_barrier(0)
        t1 = self.uacc(0, t0)
        for ci, cb in enumerate(cbs):
            self.gu_epilogue(acc[4 * ci : 4 * ci + 4], cb, off, p, valid)
        if k.hand == "gbar":
            self.uacc(1, t1)
            return nhead  # the hand-off barrier waits for the stores
        b.s_waitcnt(vmcnt=0)
        with b.scf_if(self.lane0):
            _aadd(self.k, b, self.cnt_a, b.mul(b.add(self.sb_, b.add(_c(b, S_GU), b.add(b.mul(e, _c(b, MAX_TPE)), ti))), _c(b, 4)),
                                _c(b, k.gcb), scope="agent")
        self.uacc(1, t1)

    def gu_epilogue(self, acc, cb: Value, off: Value, p: Value, valid: Value) -> None:
        """h = silu(g) u of one 32-column block, quantized per (token, 32 columns), stored ``sc1``."""
        b, k, dm = self.b, self.k, self.dm
        if self.dg("nogst"):  # diag: the h / scale stores never execute (values still computed)
            with b.scf_if(b.cmp_eq(self.rows, _c(b, -1))):
                self._gu_epilogue(acc, cb, off, p, valid)
            return
        self._gu_epilogue(acc, cb, off, p, valid)

    def _gu_epilogue(self, acc, cb: Value, off: Value, p: Value, valid: Value) -> None:
        b, k, dm = self.b, self.k, self.dm
        # lane holds columns 4g + i (tile 0) and 16 + 4g + i (tile 1) of token p
        hv = []
        for t in range(2):
            for i in range(4):
                gv = b.vec_extract(acc[t], i)
                uv = b.vec_extract(acc[2 + t], i)
                den = b.fadd(_f(b, 1.0), b.exp2(b.fmul(b.fneg(gv), _f(b, L2E))))
                hv.append(b.fmul(b.fdiv(gv, den), uv))
        amax = b.fabs(hv[0])
        for v in hv[1:]:
            amax = b.fmax(amax, b.fabs(v))
        for sw in ("p16", "p32"):
            x_ = b.bitcast(amax, I32)
            lo, hi = b.permlane16_swap(x_, x_) if sw == "p16" else b.permlane32_swap(x_, x_)
            amax = b.fmax(b.bitcast(lo, F32), b.bitcast(hi, F32))
        hrow = dm["hrow"]
        hp = b.add(off, p)
        if k.act == "fp8":
            ex, scale = Q.e8m0_fp8(b, amax)
            w0 = Q.pack_fp8x4(b, *hv[0:4], scale)
            w1 = Q.pack_fp8x4(b, *hv[4:8], scale)
            base = b.add(b.mul(hp, _c(b, hrow)), b.add(b.mul(cb, _c(b, 32)), b.mul(self.g, _c(b, 4))))
            b.buffer_store_cp(w0, self.h_rs, b.select(valid, base, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            b.buffer_store_cp(w1, self.h_rs, b.select(valid, b.add(base, _c(b, 16)), _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        else:
            ex, scale = _rceil_fp4(b, amax)
            halves = [_pack_fp4x4(b, hv[4 * t : 4 * t + 4], scale) for t in range(2)]
            even = b.cmp_eq(b.land(self.g, _c(b, 1)), _c(b, 0))
            ok = b.land(valid, even)
            for t in range(2):
                x_ = halves[t]
                lo, hi = b.permlane16_swap(x_, x_)  # even rows: hi = the odd neighbour's 16 bits
                wd = b.lor(x_, b.shl(hi, _c(b, 16)))
                o = b.add(b.mul(hp, _c(b, hrow)), b.add(b.mul(cb, _c(b, 16)), b.add(_c(b, 8 * t), b.mul(b.lshr(self.g, _c(b, 1)), _c(b, 4)))))
                b.buffer_store_cp(wd, self.h_rs, b.select(ok, o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        so = b.mul(b.add(b.mul(hp, _c(b, k.ngu)), cb), _c(b, 4))
        b.buffer_store_cp(ex, self.hs_rs, b.select(b.land(valid, b.cmp_eq(self.g, _c(b, 0))), so, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)

    # ------------------------------------------------------------------ down unit (+ last-arriver combine)
    def dn_unit(self, layer: Value, kk: Value) -> None:
        self.dn_rest(layer, self.dn_issue(layer, kk))

    def dn_issue(self, layer: Value, kk: Value, has: Value | None = None, loads: bool = True) -> dict:
        """The down unit's list lookups and weight loads. ``has`` (prefetch across the hand-off barrier): when false,
        the unit index is clamped to 0 and the weight resources are zero-length (no bytes, harmless loads)."""
        b, k, dm = self.b, self.k, self.dm
        t0 = self.ut()
        if has is not None:
            kk = b.select(has, kk, _c(b, 0))
        tau = b.div(kk, _c(b, k.nrb))
        rb = b.sub(kk, b.mul(tau, _c(b, k.nrb)))
        e, ti, off, n = self.tile_info(tau)
        p = b.add(b.mul(ti, _c(b, TILE)), self.l16)
        valid = b.cmp_lt(p, n)
        pos = b.add(off, b.smin(p, b.sub(b.smax(n, _c(b, 1)), _c(b, 1))))
        ent = self.lds_i(self.lst, pos)
        tok = b.land(ent, _c(b, 255))
        slot = b.lshr(ent, _c(b, 8))
        wgt = self.lds_f(self.lw, pos)
        if not loads:  # xu: the weights were issued by the previous unit
            return dict(e=e, ti=ti, off=off, p=p, valid=valid, tok=tok, slot=slot, wgt=wgt, rb=rb, wl=None, t=self.uacc(2, t0))
        el = _i64(b, e)
        wa = b.add(b.add(self.a["wdn"], b.mul(_i64(b, layer), self.ldn)), b.mul(el, b.const_i64(dm["dnb"] + k.gap)))
        sa = b.add(b.add(self.a["sdn"], b.mul(_i64(b, layer), self.lds_)), b.mul(el, b.const_i64(dm["dns"] + k.gap)))
        nul = k.null == "oob"
        if has is None or nul:
            w_rs = _rsrc(b, wa, _c(b, 0 if nul else dm["dnb"]))
            s_rs = _rsrc(b, sa, _c(b, 0 if nul else dm["dns"]))
        else:
            w_rs = _rsrc(b, wa, b.select(has, _c(b, dm["dnb"]), _c(b, 0)))
            s_rs = _rsrc(b, sa, b.select(has, _c(b, dm["dns"]), _c(b, 0)))
        rtb = k.inter * TILE // 2  # bytes of one 16-row tile over all K
        wl = []
        for d in range(k.dt):
            rt = b.add(b.mul(rb, _c(b, k.dt)), _c(b, d))
            row = []
            for c in range(k.nqd):
                cbt = k.chunk_bytes(c)
                coff = b.add(b.add(b.mul(rt, _c(b, rtb)), _c(b, c * 1024)), b.add(b.mul(self.l16, _c(b, cbt)), b.mul(self.g, _c(b, 16))))
                if cbt < 64:
                    coff = b.select(b.cmp_lt(b.mul(self.g, _c(b, 16)), _c(b, cbt)), coff, _c(b, OOB))
                if k.dsop:  # ((rt * ngrp + c / 4) * 16 + row) * 16 + 4 g, at the group's first chunk only
                    wv = b.buffer_load_cp(w_rs, coff, _c(b, 0), dwords=4, aux=self.wcp)
                    sv = None
                    if c % 4 == 0:
                        sblk = b.add(b.mul(rt, _c(b, k.ngrp)), _c(b, c // 4))
                        soff = b.add(b.mul(b.add(b.mul(sblk, _c(b, 16)), self.l16), _c(b, 16)), b.mul(self.g, _c(b, 4)))
                        sv = b.buffer_load_cp(s_rs, soff, _c(b, 0), dwords=1, aux=self.wcp)
                    row.append((wv, sv))
                    continue
                soff = b.mul(b.add(b.mul(b.add(b.mul(rt, _c(b, k.nqd)), _c(b, c)), _c(b, TILE)), self.l16), _c(b, 4))
                row.append((b.buffer_load_cp(w_rs, coff, _c(b, 0), dwords=4, aux=self.wcp),
                            b.buffer_load_cp(s_rs, soff, _c(b, 0), dwords=1, aux=self.wcp)))
            wl.append(row)
        return dict(e=e, ti=ti, off=off, p=p, valid=valid, tok=tok, slot=slot, wgt=wgt, rb=rb, wl=wl, t=self.uacc(2, t0))

    def h_loads(self, hp: Value, valid: Value, aux: int) -> list:
        """The B operand of a down unit: per K chunk (h, h scale) of token row ``hp``."""
        b, k, dm = self.b, self.k, self.dm
        hrow = dm["hrow"]
        bl = []
        for c in range(k.nqd):
            if k.act == "fp8":
                parts = []
                for h_ in range(2):
                    kk_ = c * CHUNK + 64 * h_
                    o = b.add(b.mul(hp, _c(b, hrow)), b.add(_c(b, kk_), b.mul(self.g, _c(b, 16))))
                    okh = b.land(valid, b.cmp_lt(b.add(_c(b, kk_), b.mul(self.g, _c(b, 16))), _c(b, k.inter)))
                    parts.append(b.buffer_load_cp(self.hl_rs, b.select(okh, o, _c(b, OOB)), _c(b, 0), dwords=4, aux=aux))
                bv = b.vec_concat(parts[0], parts[1])
            else:
                kb = c * 64
                o = b.add(b.mul(hp, _c(b, hrow)), b.add(_c(b, kb), b.mul(self.g, _c(b, 16))))
                okh = b.land(valid, b.cmp_lt(b.add(_c(b, kb), b.mul(self.g, _c(b, 16))), _c(b, k.inter // 2)))
                bv = b.buffer_load_cp(self.hl_rs, b.select(okh, o, _c(b, OOB)), _c(b, 0), dwords=4, aux=aux)
            blk = b.add(_c(b, 4 * c), self.g)
            so = b.mul(b.add(b.mul(hp, _c(b, k.ngu)), blk), _c(b, 4))
            oks = b.land(valid, b.cmp_lt(blk, _c(b, k.ngu)))
            bs = b.buffer_load_cp(self.hsl_rs, b.select(oks, so, _c(b, OOB)), _c(b, 0), dwords=1, aux=aux)
            bl.append((bv, bs))
        return bl

    def dn_rest(self, layer: Value, st: dict) -> None:
        b, k = self.b, self.k
        e, ti, off, p, valid = st["e"], st["ti"], st["off"], st["p"], st["valid"]
        tok, slot, wgt, rb, wl = st["tok"], st["slot"], st["wgt"], st["rb"], st["wl"]
        if k.hand == "flow":
            # the tile's h is complete once all I / 32 gate_up blocks of (expert, tile) arrived
            gu_word = b.add(self.sb_, b.add(_c(b, S_GU), b.add(b.mul(e, _c(b, MAX_TPE)), ti)))
            if k.mutant != "race" and not self.dg("nopoll"):
                self.poll_ge(gu_word, _c(b, k.ngu))
        t2 = self.ut()
        if k.ustamp and st["t"] is not None:
            self.uacc(3, st["t"])
        hp = b.add(off, p)
        bl = self.h_loads(hp, valid, CPOL_SC1 if k.hand == "flow" else 0)
        fb = "fp8" if k.act == "fp8" else "fp4"
        acc = [b.zero_vec_f32(4) for _ in range(k.dt)]
        if k.sop:
            # one packed scale dword per 4 chunks: byte c % 4 = the chunk's scale (h: block 4c + g; weights: byte g of
            # the row's chunk dword)
            ff = _c(b, 0xFF)
            for c0 in range(0, k.nqd, 4):
                cs = list(range(c0, min(c0 + 4, k.nqd)))
                hp_ = None
                for c in cs:
                    v = b.land(bl[c][1], ff)
                    hp_ = v if hp_ is None else b.lor(hp_, b.shl(v, _c(b, 8 * (c % 4))))
                wp = []
                for d in range(k.dt):
                    if k.dsop:
                        wp.append(wl[d][c0][1])
                        continue
                    w_ = None
                    for c in cs:
                        v = b.land(b.lshr(wl[d][c][1], self.sh8g), ff)
                        w_ = v if w_ is None else b.lor(w_, b.shl(v, _c(b, 8 * (c % 4))))
                    wp.append(w_)
                for c in cs:
                    sel = ((c + 1) % 4 if (c + 1) % 4 < len(cs) else 0) if k.mutant == "sopsel" else c % 4
                    for d in range(k.dt):
                        acc[d] = b.mfma_scale_mx(wl[d][c][0], bl[c][0], acc[d], wp[d], hp_, shape="16x16x128", fmt_a="fp4",
                                                 fmt_b=fb, opsel_a=sel, opsel_b=c % 4)
        for c in range(k.nqd if not k.sop else 0):
            bv, bs = bl[c]
            for d in range(k.dt):
                a_, sw = wl[d][c]
                acc[d] = b.mfma_scale_mx(a_, bv, acc[d], b.lshr(sw, self.sh8g), bs, shape="16x16x128", fmt_a="fp4", fmt_b=fb)
        t3 = self.uacc(4, t2)
        # partials: w_slot * acc at part[token, slot, columns]
        cols0 = b.mul(rb, _c(b, k.dt * TILE))
        prow = b.add(b.mul(tok, _c(b, SLOTS)), slot)
        stg = b.scf_if(b.cmp_eq(self.rows, _c(b, -1))) if self.dg("nodst") else contextlib.nullcontext()
        with stg:  # diag nodst: the partial stores never execute
            for d in range(k.dt):
                v = b.vec_pack([b.fmul(wgt, b.vec_extract(acc[d], i)) for i in range(4)], F32)
                o = b.mul(b.add(b.mul(prow, _c(b, H)), b.add(cols0, b.add(_c(b, d * TILE), b.mul(self.g, _c(b, 4))))), _c(b, 4))
                b.buffer_store_cp(b.vec_bitcast(v, VectorType(I32, 4)), self.part_rs, b.select(valid, o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if k.combine == "gbar":
            self.uacc(5, t3)
            return  # the combine barrier waits for the stores
        b.s_waitcnt(vmcnt=0)
        if self.dg("nocmb"):
            self.uacc(5, t3)
            return
        # combine arrival on (token, column block, lane group): one per down lane
        cw = self.cmb_word(tok, rb)
        if k.combine == "phase":
            with b.scf_if(valid):
                _aadd(self.k, b, self.cnt_a, b.mul(cw, _c(b, 4)), _c(b, 1), scope="agent")
            self.uacc(5, t3)
            return
        need = 9 if k.mutant == "cnt" else SLOTS - 1
        with b.scf_if(valid):
            old = _aadd(self.k, b, self.cnt_a, b.mul(cw, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, need))):
                self.combine_cols(layer, tok, cols0)

    def cmb_word(self, tok: Value, rb: Value) -> Value:
        b = self.b
        return b.add(self.sb_, b.add(_c(b, S_CMB), b.add(b.mul(b.add(b.mul(tok, _c(b, NRB_MAX)), rb), _c(b, 4)), self.g)))

    def combine_cols(self, layer: Value, tok: Value, cols0: Value, ok: Value | None = None, ndt: int | None = None,
                     aux: int = CPOL_SC1) -> None:
        """y[tok, this lane's columns of ``ndt`` (default ``dt``) 16-column tiles from ``cols0``] = the 11 slot partials
        summed in slot order."""
        b, k = self.b, self.k
        y_a = b.add(self.a["y"], b.mul(_i64(b, layer), self.ly))
        y_rs = _rsrc(b, y_a, _c(b, k.bucket * H * 2))
        for d in range(k.dt if ndt is None else ndt):
            col = b.add(cols0, b.add(_c(b, d * TILE), b.mul(self.g, _c(b, 4))))
            tot = None
            for s_ in range(SLOTS):
                if k.mutant == "slot" and s_ == TOPK - 1:
                    continue
                o = b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, SLOTS)), _c(b, s_)), _c(b, H)), col), _c(b, 4))
                v = b.vec_bitcast(b.buffer_load_cp(self.part_rs, o, _c(b, 0), dwords=4, aux=aux), VectorType(F32, 4))
                vs = [b.vec_extract(v, i) for i in range(4)]
                tot = vs if tot is None else [b.fadd(tot[i], vs[i]) for i in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            yo = b.mul(b.add(b.mul(tok, _c(b, H)), col), _c(b, 2))
            b.buffer_store_cp(yv, y_rs, yo if ok is None else b.select(ok, yo, _c(b, OOB)), _c(b, 0))

    def combine_phase(self, layer: Value) -> None:
        """Combine units (token tile, column block) after the down units: every lane polls its own (token, column
        block, lane group) counter (all 64 lanes in one asm spin), then sums its columns' 11 slots."""
        b, k = self.b, self.k
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        ntt = b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4))
        ncu = b.readfirstlane(b.mul(ntt, _c(b, k.nrb)))
        gw = b.add(b.mul(self.cta, _c(b, WAVES)), self.wave)
        need = SLOTS - 1 if k.mutant == "cnt" else SLOTS
        with b.scf_for(gw, ncu, _c(b, GRID * WAVES), iv_name="cu") as u:
            u = b.readfirstlane(u)
            tt = b.div(u, _c(b, k.nrb))
            rb = b.sub(u, b.mul(tt, _c(b, k.nrb)))
            tok = b.add(b.mul(tt, _c(b, TILE)), self.l16)
            ok = b.cmp_lt(tok, self.rows)
            tok = b.smin(tok, b.sub(self.rows, _c(b, 1)))
            st = _pollge(self.k, b, self.cnt_a, b.mul(self.cmb_word(tok, rb), _c(b, 4)), _c(b, need), self.deadline(),
                                 pause="s_sleep1")
            with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), self.lane0)):
                self.err(3)
            self.combine_cols(layer, tok, b.mul(rb, _c(b, k.dt * TILE)), ok)

    def combine_gbar(self, layer: Value) -> None:
        """``combine=gbar``: after the barrier and the acquire, combine units (16 tokens x 16 columns) over all the
        grid's waves, plain partial loads, slot order (bitwise equal to the counted combines)."""
        b = self.b
        ntt = b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4))
        nct = H // TILE
        ncu = b.readfirstlane(b.mul(ntt, _c(b, nct)))
        gw = b.add(b.mul(self.cta, _c(b, WAVES)), self.wave)
        with b.scf_for(gw, ncu, _c(b, GRID * WAVES), iv_name="cg") as u:
            u = b.readfirstlane(u)
            tt = b.div(u, _c(b, nct))
            ct = b.sub(u, b.mul(tt, _c(b, nct)))
            tok = b.add(b.mul(tt, _c(b, TILE)), self.l16)
            ok = b.cmp_lt(tok, self.rows)
            tok = b.smin(tok, b.sub(self.rows, _c(b, 1)))
            self.combine_cols(layer, tok, b.mul(ct, _c(b, TILE)), ok, ndt=1, aux=0)

    def pretouch_lines(self, layer: Value, d0: Value, d1: Value, which: str) -> None:
        """``pretouch``: plain loads of every h line (``which="h"``, before the hand-off barrier) / partial line
        (``"part"``, before the combine barrier) this wave reads after the barrier, issued right before its CTA
        arrives: lines other CTAs have not produced yet are cached (vL1D / L2) with the pre-launch bytes, so only the
        acquire makes the later reads see the producers' stores."""
        b, k = self.b, self.k
        loop_acc = _c(b, 0)
        if which == "h":
            loop = b.scf_for_iter(self.dfirst(d0), d1, self.dstep(), [("pth_acc", loop_acc)], iv_name="ptd")
            with loop as (kk, vals):
                kk = b.readfirstlane(kk)
                tau = b.div(kk, _c(b, k.nrb))
                e, ti, off, n = self.tile_info(tau)
                p = b.add(b.mul(ti, _c(b, TILE)), self.l16)
                valid = b.cmp_lt(p, n)
                acc = vals[0]
                for bv, bs in self.h_loads(b.add(off, p), valid, 0):
                    acc = b.xor(acc, b.xor(b.vec_extract(b.vec_bitcast(bv, VectorType(I32, 8 if k.act == "fp8" else 4)), 0), bs))
                b.scf_yield(acc)
            loop_acc = loop.results[0]
            if k.ptall:  # the XCD's 32 CTAs sweep all of h / hs (16 B per thread per step)
                dm = dims(k)
                per = (GRID // XCDS) * THREADS * 16
                t0 = b.mul(b.add(b.mul(b.lshr(self.cta, _c(b, 3)), _c(b, THREADS)), self.tid), _c(b, 16))
                for nm, rs, nb in (("pta", self.h_rs, SLOTS * k.bucket * dm["hrow"]), ("ptb", self.hs_rs, SLOTS * k.bucket * dm["hsrow"])):
                    loop = b.scf_for_iter(_c(b, 0), _c(b, -(-nb // per)), _c(b, 1), [(f"{nm}_acc", loop_acc)], iv_name=nm)
                    with loop as (i, vals):
                        v = b.buffer_load_cp(rs, b.add(t0, b.mul(i, _c(b, per))), _c(b, 0), dwords=4)
                        b.scf_yield(b.xor(vals[0], b.vec_extract(v, 0)))
                    loop_acc = loop.results[0]
        if which == "part":
            ntt = b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4))
            nct = H // TILE
            ncu = b.readfirstlane(b.mul(ntt, _c(b, nct)))
            gw = b.add(b.mul(self.cta, _c(b, WAVES)), self.wave)
            loop = b.scf_for_iter(gw, ncu, _c(b, GRID * WAVES), [("ptc_acc", loop_acc)], iv_name="ptc")
            with loop as (u, vals):
                u = b.readfirstlane(u)
                tt = b.div(u, _c(b, nct))
                ct = b.sub(u, b.mul(tt, _c(b, nct)))
                tok = b.smin(b.add(b.mul(tt, _c(b, TILE)), self.l16), b.sub(self.rows, _c(b, 1)))
                col = b.add(b.mul(ct, _c(b, TILE)), b.mul(self.g, _c(b, 4)))
                acc = vals[0]
                for s_ in range(SLOTS):
                    o = b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, SLOTS)), _c(b, s_)), _c(b, H)), col), _c(b, 4))
                    v = b.buffer_load_cp(self.part_rs, o, _c(b, 0), dwords=4)
                    acc = b.xor(acc, b.vec_extract(v, 0))
                b.scf_yield(acc)
            loop_acc = loop.results[0]
        b.s_waitcnt(vmcnt=0)
        with b.scf_if(b.land(self.lane0, b.cmp_eq(loop_acc, _c(b, 0x5EED1234)))):
            self.err(7)  # keeps the loads; a 2^-32 chance per wave of a false error word

    def twin_delay(self, site: int) -> None:
        """``tdelay`` (stale-read twin): the odd CTAs sleep ``tdelay`` x ``s_sleep 127`` (about 8K clocks each)."""
        b = self.b
        with b.scf_if(b.cmp_ne(b.land(self.cta, _c(b, 1)), _c(b, 0))):
            with b.scf_for(_c(b, 0), _c(b, self.k.tdelay), _c(b, 1), iv_name=f"tdl{site}"):
                b.inline_asm("s_sleep 127", "", [], None)

    # ------------------------------------------------------------------ grid barriers
    def gbar(self, bnd: Value, acquire: bool = False, site: int = 4, pre=None) -> None:
        """Striped grid barrier (gemm.py's): one agent-scope add per CTA on stripe ``cta mod 8``, 8 lanes poll."""
        self.gbar_arrive(bnd, pre)
        self.gbar_wait(bnd, acquire, site)

    def gbar_arrive(self, bnd: Value, pre=None) -> None:
        """``pre``: emitted after the CTA barrier, before the arrival (the stale-read twin's pretouch: no other
        traffic of this CU between it and the reads after the grid barrier)."""
        b = self.b
        per = GRID // XCDS
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        if pre is not None:
            pre()
            b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(D.is_thread0(b)):
            mine = b.mul(b.add(row, b.land(self.cta, _c(b, XCDS - 1))), _c(b, 4))
            old = _aadd(self.k, b, gsa, mine, _c(b, 1), scope="agent", returns=True)
            target = b.mul(b.add(b.div(old, _c(b, per)), _c(b, 1)), _c(b, per))
            b.smem_store_vN(self.gsw, [_c(b, 0)], b.bitcast(target, F32), 1)
            if self.k.dbgerr and self.k.stamp:  # debug: (mine, old, gs address) at stamp points 6, 7 of waves 0 / 1
                sp = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
                base = b.mul(self.cta, _c(b, WAVES * STAMP_PTS))
                b.store(_i64(b, mine), sp, b.add(base, _c(b, 6)))
                b.store(_i64(b, old), sp, b.add(base, _c(b, 7)))
                b.store(gsa, sp, b.add(base, _c(b, STAMP_PTS + 6)))
                b.store(_i64(b, target), sp, b.add(base, _c(b, STAMP_PTS + 7)))

    def gbar_wait(self, bnd: Value, acquire: bool = False, site: int = 4) -> None:
        """8 lanes of wave 0 poll the 8 stripes; ``acquire``: then one ``buffer_inv sc1`` for the CTA (P5) before the
        CTA barrier releases the other waves."""
        b = self.b
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(b.cmp_lt(self.tid, _c(b, XCDS))):
            b.s_waitcnt(lgkmcnt=0)
            target = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.gsw, _c(b, 0), n=1), 0), I32)
            st = _pollge(self.k, b, gsa, b.mul(b.add(row, self.tid), _c(b, 4)), target, self.deadline(), pause="s_sleep1")
            with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), D.is_thread0(b))):
                self.err(site)
            if acquire and self.k.mutant != "noacq":
                D.acquire_l2(b, "agent")
        b.sync_lds_only()

    # ------------------------------------------------------------------ main
    def build(self) -> KernelDef:
        b = self.b
        zero = _c(b, 0)
        with b.scf_for(zero, self.L, _c(b, 1), iv_name="layer") as layer:
            layer = b.readfirstlane(layer)
            with b.scf_if(b.land(self.gsync, b.cmp_gt(layer, zero))):
                self.gbar(b.sub(layer, _c(b, 1)))
            b.sync_lds_only()
            self.stamp(layer, 0)
            self.sb_ = self.set_base(layer)
            self.zero_other_set(layer)
            self.route(layer)
            with b.scf_if(D.is_thread0(b)):
                from kernels.gfx950.qwen38_mono.ccl import wire as W

                st = _pollge(self.k, b, self.cnt_a, b.mul(b.add(self.sb_, _c(b, S_RT)), _c(b, 4)), self.rows, self.deadline(),
                                     pause="s_sleep1")
                with b.scf_if(b.cmp_ne(st, zero)):
                    self.err(1)
                if self.k.xcp == "acq":
                    D.acquire_l2(b, "agent")
            b.sync_lds_only()
            self.stamp(layer, 1)
            nt = self.sort()
            self.stamp(layer, 2)
            g0, g1, d0, d1 = self.deal(nt)
            k = self.k
            if k.tdelay:
                self.twin_delay(0)
            if not self.dg("nogu") and k.xu & 1:
                j0 = b.add(g0, self.wave)
                iss0 = self.gu_unit(layer, j0, has=b.cmp_lt(j0, g1), ctx_only=True)
                head0 = [iss0(q) for q in range(min(k.depth, NQG))]
                vals0, shp = _flat(head0)
                loop = b.scf_for_iter(j0, g1, _c(b, WAVES), [(f"guh{i}", v) for i, v in enumerate(vals0)], iv_name="gux")
                with loop as (j, vals):
                    j = b.readfirstlane(j)
                    head, _ = _unflat(list(vals), shp)
                    nj = b.add(j, _c(b, WAVES))
                    nxt = self.gu_unit(layer, nj, has=b.cmp_lt(nj, g1), ctx_only=True)
                    nhead = self.gu_unit(layer, j, head=head, nxt=nxt)
                    b.scf_yield(*_flat(nhead)[0])
            elif not self.dg("nogu"):
                with b.scf_for(b.add(g0, self.wave), g1, _c(b, WAVES), iv_name="gu") as j:
                    self.gu_unit(layer, b.readfirstlane(j))
            self.stamp(layer, 3)
            if k.hand == "gbar":
                ib = b.add(self.L, b.mul(layer, _c(b, 2)))  # intra-layer barrier rows: layers + 2 layer (+ 1)
                if k.pretouch:  # stale-read twin: right before this CTA's arrival (inside the arrival measured worse)
                    self.pretouch_lines(layer, d0, d1, "h")
                th = self.ut()
                if k.mutant != "race":
                    self.gbar_arrive(ib)
                k0 = self.dfirst(d0)
                has = b.cmp_lt(k0, d1)
                # the stale-read twin issues no weight prefetch across the barrier: nothing evicts the pretouched lines
                # before the first down unit reads them
                pre = None if self.dg("nodn") or k.pretouch else self.dn_issue(layer, k0, has)
                if k.mutant != "race":
                    self.gbar_wait(ib, acquire=True, site=5)
                self.uacc(7, th)
                if k.tdelay:
                    self.twin_delay(1)
                if k.pretouch and not self.dg("nodn"):
                    with b.scf_for(k0, d1, self.dstep(), iv_name="dn") as kk:
                        self.dn_unit(layer, b.readfirstlane(kk))
                if pre is not None and k.xu & 2:
                    vals0, shp = _flat(pre["wl"])
                    loop = b.scf_for_iter(k0, d1, self.dstep(), [(f"dnw{i}", v) for i, v in enumerate(vals0)], iv_name="dnx")
                    with loop as (kk, vals):
                        kk = b.readfirstlane(kk)
                        st = self.dn_issue(layer, kk, loads=False)
                        st["wl"], _ = _unflat(list(vals), shp)
                        nk = b.add(kk, self.dstep())
                        nst = self.dn_issue(layer, nk, has=b.cmp_lt(nk, d1))
                        self.dn_rest(layer, st)
                        b.scf_yield(*_flat(nst["wl"])[0])
                elif pre is not None:
                    with b.scf_if(has):
                        self.dn_rest(layer, pre)
                    with b.scf_for(b.add(k0, self.dstep()), d1, self.dstep(), iv_name="dn") as kk:
                        self.dn_unit(layer, b.readfirstlane(kk))
            elif not self.dg("nodn"):
                with b.scf_for(self.dfirst(d0), d1, self.dstep(), iv_name="dn") as kk:
                    self.dn_unit(layer, b.readfirstlane(kk))
            tc = self.ut()
            if self.k.combine == "phase" and not self.dg("nocmb") and not self.dg("nodn"):
                self.combine_phase(layer)
            if k.combine == "gbar" and not self.dg("nocmb") and not self.dg("nodn"):
                if k.pretouch:
                    self.pretouch_lines(layer, d0, d1, "part")
                if k.mutant != "cnt":
                    self.gbar(b.add(b.add(self.L, b.mul(layer, _c(b, 2))), _c(b, 1)), acquire=True, site=6)
                self.combine_gbar(layer)
            self.uacc(6, tc)
            self.stamp(layer, 4)
            b.sync_lds_only()
            self.stamp(layer, 5)
        if self.k.ustamp:
            b.sync_lds_only()
            with b.scf_if(b.cmp_lt(self.lane, _c(b, 8))):
                sp = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
                v = self.lds_i(self.ust, b.add(b.mul(self.wave, _c(b, 8)), self.lane))
                b.store(_i64(b, v), sp, b.add(b.mul(b.add(b.mul(self.cta, _c(b, WAVES)), self.wave), _c(b, STAMP_PTS)), self.lane))
        # the launch's last CTA advances the epoch by the layers walked (the next launch reads it)
        b.s_waitcnt(vmcnt=0)
        with b.scf_if(D.is_thread0(b)):
            old = _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_END), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, GRID - 1))):
                _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_EP), self.L, scope="agent")
                _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_END), _c(b, -GRID), scope="agent")
        return b.kernel


def _rceil_fp4(b: IRBuilder, amax: Value):
    """FlyTape's RCEIL E8M0 code of a 32-value group (``flytape.blocks.mx.scale_code``): exponent of ``amax / 6``
    rounded up, clamped to 1..253; returns (code i32, f32 2^(code - 127))."""
    bits = b.bitcast(b.fdiv(amax, _f(b, 6.0)), I32)
    code = b.add(b.land(b.lshr(bits, _c(b, 23)), _c(b, 0xFF)),
                 b.select(b.cmp_ne(b.land(bits, _c(b, 0x7FFFFF)), _c(b, 0)), _c(b, 1), _c(b, 0)))
    code = b.smin(b.smax(code, _c(b, 1)), _c(b, 253))
    return code, b.bitcast(b.shl(code, _c(b, 23)), F32)


def _pack_fp4x4(b: IRBuilder, vals, scale: Value) -> Value:
    """Four f32 / ``scale`` rounded to E2M1 (``v_cvt_scalef32_pk_fp4_f32``: nearest even, saturating) as 16 bits
    (value 0 in the low nibble)."""
    out = []
    for i in range(2):
        r = b.inline_asm("v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3", "=v,v,v,v", [vals[2 * i], vals[2 * i + 1], scale],
                         I32, sideeffect=False, result_name_hint="q4")
        out.append(b.land(r, _c(b, 0xFF)))
    return b.add(out[0], b.shl(out[1], _c(b, 8)))


def build_moe_kernel(k: MoeKnobs) -> KernelDef:
    return _Moe(k).build()


# ------------------------------------------------------------------ the bytes floor


@dataclass(frozen=True)
class StreamKnobs:
    """``q38lb_mst``: read ``nbytes`` per layer (16 bytes a lane, ``depth`` loads in flight), no math: the floor the
    MoE's expert bytes are compared with. 256 CTAs x 512 threads."""

    depth: int = 4  # CTA steps in flight
    wcp: str = "nt"
    unroll: int = 1  # 16-byte loads per lane per step (each instruction of the CTA covers 8 KiB contiguous)
    tag: str = ""


def stream_name(k: StreamKnobs) -> str:
    return f"q38lb_mst_d{k.depth}_{k.wcp}" + (f"_u{k.unroll}" if k.unroll != 1 else "") + (f"_{k.tag}" if k.tag else "")


def build_stream_kernel(k: StreamKnobs) -> KernelDef:
    b = IRBuilder(stream_name(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    src = b.param("src", I64)
    sink = b.param("sink", I64)
    nbytes = b.readfirstlane(b.param("nbytes", I32))  # per layer, a multiple of 16
    L = b.readfirstlane(b.param("layers", I32))
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
    a0 = b.trunc(b.smem_addr_of(pad), I32)
    with b.scf_if(D.is_thread0(b)):
        b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}",
                     [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)], None)
    per = THREADS * 16 * k.unroll
    steps_l = b.div(b.add(nbytes, _c(b, per - 1)), _c(b, per))
    total = b.mul(steps_l, L)
    my = b.smax(b.div(b.add(b.sub(total, cta), _c(b, GRID - 1)), _c(b, GRID)), _c(b, 0))
    aux = CPOL_NT if k.wcp == "nt" else 0
    zero = _c(b, 0)

    def issue(i: Value):
        idx = b.add(cta, b.mul(i, _c(b, GRID)))
        ok = b.cmp_lt(i, my)
        idx = b.select(ok, idx, zero)
        layer = b.readfirstlane(b.div(idx, steps_l))
        st = b.sub(idx, b.mul(layer, steps_l))
        base = b.add(src, b.mul(_i64(b, layer), _i64(b, nbytes)))
        rs = _rsrc(b, base, nbytes)
        out = []
        for j in range(k.unroll):
            off = b.add(b.mul(st, _c(b, per)), b.add(_c(b, j * THREADS * 16), b.mul(tid, _c(b, 16))))
            out.append(b.buffer_load_cp(rs, b.select(ok, off, _c(b, OOB)), zero, dwords=4, aux=aux))
        return out

    u = k.unroll
    q = [v for i in range(k.depth) for v in issue(_c(b, i))]
    iters = [(f"s{i}", v) for i, v in enumerate(q)] + [("acc", _c(b, 0))]
    loop = b.scf_for_iter(zero, my, _c(b, 1), iters, iv_name="st")
    with loop as (i, vals):
        vals = list(vals)
        ring, acc = vals[:-1], vals[-1]
        nxt = issue(b.add(i, _c(b, k.depth)))
        for v in ring[:u]:
            acc = b.xor(acc, b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
        b.scf_yield(*(ring[u:] + nxt), acc)
    acc = loop.results[-1]
    with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
        b.store(acc, _ptr(b, b.readfirstlane(sink)), tid)
    return b.kernel
