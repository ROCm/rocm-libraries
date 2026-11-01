# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2d: the large-batch MXFP4 MoE rebuilt from P-L4's WaveScope findings (gfx950), rows <= 128, TP1 / TP2.

``experiments/qwen38-true-mono/large-batch/P-L4b.md``. P-L4 (``moe_lb.py``, wave units, a global ``h`` hand-off through
per-(expert, tile) counters, per-wave gathers of the MoE input) was structure bound (ATT: vector-memory issue stalls on
weights and gathers, ``vmcnt(0)`` at the hand-off / combine polls, CTA barriers at the route poll). This kernel keeps
no global intermediate at all:

1. **CTA-level fused units on an I-split byte line.** Every active (expert, pass) item (a pass = ``tpp`` 16-token tiles
   of the expert's token list) is ``NA = I / 32`` *atoms* of 32 intermediate columns; an atom is the gate_up rows of
   its 32 columns (2 gate + 2 up 16-row tiles over K = 2560) and the down weights of its 32 K columns (all 2560
   outputs). Items are laid on one line (experts by id, the shared expert's passes last) and the line is cut into
   ``C`` equal atom ranges (``C`` = 256 unless the line is short); line position ``L`` takes range ``L``. A CTA runs
   each item piece in its range as one *segment*: gate_up of the piece's columns, SiLU x up and the MX requantization
   in LDS, then the down projection over exactly those K columns, so ``h`` never leaves the CU and no CTA waits for
   another before the combine. An item cut by range ends becomes 2-4 segments; each writes its own f32 partial.
2. **MoE input staged once per segment in LDS** (``xst=dma``: ``buffer_load ... lds`` DMA of the item's token rows,
   padded rows so the MFMA B reads are conflict free); every wave's gate_up reads its B operand from LDS.
3. **K split canonically.** gate_up accumulates K halves (chunks 0-9, 10-19) separately and adds them once
   (``half0 + half1``): an atom pair runs 8 waves x (tile of 4, atom of 2) over full K, a lone atom 8 waves x (tile,
   K half) with an LDS sum; both give the same bits, so ``h`` does not depend on the deal.
4. **Routes are the sort input.** ``entry=route``: the router epilogue's (id | tag << 16, weight) pairs are read
   directly by every CTA's sort (no counter, no route poll barrier). ``entry=logits``: P-L4's in-kernel top-10, but
   published as tagged pairs that the sort threads poll themselves (``tag`` = launch epoch + layer).
5. **Combine:** one striped grid barrier + acquire, then every wave sums its (token, 256 columns) units over the 11
   slots in slot order and over each slot's segments in segment order (``desc[token, slot]`` = segment count).
   Deterministic; bitwise invariant over every order-preserving knob (``xh``, ``depth``, ``ddepth``, ``wcp``, ``xst``,
   ``lds1``, ``sop`` variants); the line (``tpp``, ``lw``) defines the segment boundaries and therefore the rounding.

Levers (default-off knobs): ``xh`` (XCD-homed line: XCD x runs line positions 32 x .. 32 x + 31, so an expert's
segments share one L2; XCD-SCHEDULING lever 2 / 12), ``tpp`` (token tiles per pass), ``depth`` / ``ddepth``,
``wcp``, ``xst=vgpr``. Diagnosis: ``null=oob`` (zero-byte expert weight loads), ``stamp``, ``hdump`` (numerics twin:
also stores ``h`` / scales at P-L4's global layout), mutants. Python engine only; names ``q38lb2_moe*``.

Weights per layer, per expert (513, the shared expert last), MXFP4: gate_up codes in P-L4's tile layout
``[2I/16][20][16][64]``, gate_up scales in P-L4's 4-chunk (``sop``) layout, down codes ``[H/16][nqd][16][<=64]``,
down scales ``dsx`` ``[H/16][16 rows][4 g][ngrp dwords]`` (byte ``c % 4`` of dword ``c / 4`` = chunk ``c``'s scale
of K group ``g``; pad 127). MoE input codes ``[rows][xrow]``, scales in P-L4's ``sop`` layout ``[rows][80]``.
"""

from __future__ import annotations

import bisect
import contextlib
from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_NT, CPOL_SC1, F32, I8, I16, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mx_quant as Q
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

from .moe_lb import _aadd, _bcnt, _c, _f, _i64, _pack_fp4x4, _pollge, _rceil_fp4, _rsrc, _wave_all

__all__ = [
    "H",
    "NEXP",
    "MAXSEG",
    "Moe2Knobs",
    "build_moe2_kernel",
    "dims2",
    "kernel_name2",
    "moe2_params",
    "line_plan",
    "counter_words2",
]

H = 2560
E = 512
SH = 512
NEXP = 513
TOPK = 10
SLOTS = 11
TILE = 16
CHUNK = 128
NQG = H // CHUNK  # 20 gate_up K chunks
LST = 528  # router logit row stride (f32)
PRS = 16  # route pairs per token row (11 used): 8 bytes each (id | tag << 16, weight)
GRID = 256
WAVES = 8
THREADS = 512
XCDS = 8
L2E = 1.4426950408889634
OOB = 0x7FFFFFF0
MAXSEG = 6  # segments per item (the line's minimum share keeps every item within it)
DSTR = 16  # desc words per token
W_ERR, W_EP, W_END = 0, 16, 32
W_TICK = 48  # deal=dyn: two ticket counters (layer parity)
CNT_WORDS = 64 + 2 * 128  # + cmb=flow: per-token slot counters, two parities x 128 tokens
W_TOKC = 64
STAMP_PTS = 8
MUTANTS = ("none", "slot", "sort", "sopsel", "seg", "half", "lone", "mask", "bar", "noacq", "hsc", "desc", "tag", "tick", "flow", "flow0")


@dataclass(frozen=True)
class Moe2Knobs:
    """One build; the kernel name lists only deviations from these defaults (and ``tp`` / ``act`` / ``bucket``)."""

    tp: int = 1  # 1: I = 640 | 2: the rank's I = 320
    act: str = "fp8"  # "fp8" (W4A8, ATOM) | "fp4" (W4A4, FlyTape)
    bucket: int = 128  # rows bucket (16 | 32 | 64 | 128)
    pipe: str = "stream"  # "stream" (v2: one weight-load stream per wave across atoms, the down phase and segments; K
    #   halves interleaved by scale group) | "unit" (v1: per-unit rings, atom pairs + a lone K-half atom; other rounding)
    entry: str = "route"  # "route" (route pairs are inputs) | "logits" (in-kernel top-10, tagged pairs polled by the sort)
    null: str = "none"  # "none" | "oob" (expert weight loads move zero bytes: the fixed overhead)
    tpp: int = 1  # 16-token tiles per pass (1 | 2): x / h LDS and MFMAs per weight load; changes the line (order)
    depth: int = 4  # gate_up chunks of weight loads in flight per wave
    ddepth: int = 8  # down (tile, chunk) weight loads in flight per wave (stream: topped up from the gate_up handoff)
    lka: str = "tail"  # stream: next segment's head issued as the down stream's last refills ("tail") or before the down
    #   variant dispatch ("early": LLVM then drains the stream at every segment boundary)
    prio: str = "none"  # stream: wave priority balance of a SIMD's two waves ("none" | "dn" | "all" | "alt")
    tlbp: bool = False  # stream: touch the next regions' pages at a segment start (address-translation prefetch)
    xv: bool = False  # stage only the valid token rows of each pass (order-preserving)
    xsync: str = "bar"  # stream: segment-start sync on the staged x: "bar" (CTA barrier) | "cnt" (LDS count, DMA by waves 0-3)
    deal: str = "line"  # stream: "line" (atom line) | "rr" (units round-robin) | "dyn" (unit L, then global tickets) |
    #   "hy" (whole-item rounds + atom-line remainder)
    pc: int = 1  # rr / dyn: fixed pieces per item (unit = item piece; changes the rounding for pc > 1); 0: auto,
    #   2 when 2 x items <= 256 (few items: the pieces fill more CTAs), else 1
    uord: str = "id"  # rr / dyn unit order: "id" | "heavy" (most tokens first)
    gu: str = "atom"  # stream gate_up: "atom" (K split over waves, per-atom LDS reduction) | "wt" (wave-owned tile pairs, full K)
    wdep: int = 2  # gu=wt: chunk steps of loads in flight ahead of the MFMAs
    wg: int = 0  # gu=wt: pairs per full-K sweep (0: all the wave's pairs in one sweep)
    word: str = "chunk"  # gu=wt step order: "chunk" (a chunk over the sweep's tiles) | "grp" (per scale group, tile-major)
    wpin: bool = True  # gu=wt: pin each step's loads / LDS reads at their issue point (asm-copied offsets)
    pal: bool = True  # gu=wt: the panel aliases the x rows when they fit (one sweep)
    wk: int = 0  # gu=wt: K parts (0 auto: 1 if the gate tiles divide over the waves, else 2)
    dt: int = 1  # down: output tiles sharing each h fragment (own accumulators; stores after the group)
    doff: str = "mul"  # down load offsets: "mul" (uniform tile x chunk product) | "imm" (tile base SGPR + chunk immediate)
    dlev: str = ""  # direct compiler levers ('+'-joined ROCKE_DIRECT_LEVERS names; kernel attr, no IR change)
    w2d: int = 3  # pipe=w2: chunk steps of loads in flight (gate_up and down)
    w2t: int = 4  # pipe=w2: down output tiles per unit
    dsk: int = 0  # down skew: waves 0-3 own 20 + dsk output tiles, waves 4-7 20 - dsk (the upper waves lag on a shared SIMD)
    pcal: bool = False  # pieces on down-chunk (4-atom) boundaries
    sskip: bool = False  # pstg: skip the staged store instructions whose rows are all past the item's tokens
    dsq: bool = False  # down scales in the dsq layout (4 tiles per lane contiguous: one dwordx4 per 4 / 2 tiles)
    sdl: bool = False  # down scales loaded after the gate_up (h complete) instead of at the segment start
    shd: bool = False  # shared-expert items split by down columns over two CTAs (when the units fit the last round)
    cmb: str = "gbar"  # combine hand-off: "gbar" (grid barrier) | "flow" (per-token counters, no barrier)
    gpc: str = "sc0sc1"  # grid-barrier poll loads: "sc0sc1" (system scope) | "sc1" (agent scope)
    gpause: str = "s_sleep1"  # grid-barrier poll pause per miss (ccl.wire pauses: s_sleep1 | s_nop | ...)
    cob: str = ""  # code-object compiler of the build: "" (process default, COMGR) | "direct" (Rust; harness-side, no IR change)
    xb: bool = False  # deal=rr: XCD blocks per round (consecutive units, e.g. one expert's passes, share an XCD's L2;
    #   every XCD busy also in a partial round)
    lpf: int = 1  # stream atom loop: x B operands read this many chunks ahead (LDS latency hiding)
    shs: int = 0  # tokens per shared-expert item (0: 16 x tpp, as routed passes; 4 | 8: more, lighter shared items)
    lrs: bool = False  # buffer resources built at each use (fewer SGPRs held across the kernel)
    pstg: int = 0  # partial stores through a per-wave LDS stage: 0 direct (16 rows x 64 B) | 2 | 4 tiles (rows of 128 | 256 B)
    pst: str = "sc1"  # partial stores' cache policy: "sc1" | "nt" | "def" (nt / def: an L2 write-back at the barrier arrival)
    hdbg: int = 0  # heavy-order debug bits: 1 extra barriers, 2 s_nop before mbcnt, 4 perm check (err), 8 check without err
    snap: int = 0  # stream + wl: line cuts within ``snap`` units of an item boundary move onto it
    wl: str = ""  # stream: cost-weighted line "wa.wc.ws.wt" (units per atom, chunk start, item start, token); "" = atoms
    ov: int = 0  # stream: virtual atoms at each item start on the line (segment-overhead weight; changes the line)
    ab: int = 2  # stream: atoms per LDS-reduction barrier (the K halves of ``ab`` atoms are summed after one barrier)
    lds1: bool = True  # gate_up B operand read from LDS one chunk ahead
    xst: str = "dma"  # MoE-input staging: "dma" (buffer_load ... lds) | "vgpr" (loads + ds_write)
    xh: bool = False  # XCD-homed line: XCD x (CTAs x, x + 8, ..) takes line positions 32 x .. 32 x + 31 (order-preserving)
    wcp: str = "nt"  # expert weight cache policy: "nt" | "default"
    hz: bool = True  # s_nop 4 before every asm atomic / poll (the P-L4 SGPR-base hazard)
    stamp: bool = False  # per-wave s_memrealtime at the phase points (last layer)
    hinit: str = "zero"  # LDS h at kernel start: "zero" | "dirty" (numerics twin: finite non-zero codes, so a down lane that
    #   does not mask the K columns its segment does not own reads them; the default must stay bitwise equal)
    hdump: bool = False  # numerics twin: also store h / h scales to ``hd`` / ``hds`` at the item's list positions
    pretouch: int = 0  # stale-read twin: load the partial / desc lines the combine reads, 1: before the combine barrier,
    #   2: at the layer start (before any partial of the layer is written: every pretouched line is stale)
    diag: str = "none"  # timing diagnosis only (wrong numerics): "+"-joined nox (x staging moves no bytes), nogu, nodn, nocmb
    mutant: str = "none"
    tag: str = ""

    @property
    def inter(self) -> int:
        return 640 // self.tp

    @property
    def na(self) -> int:  # atoms (32 intermediate columns) per item
        return self.inter // 32

    @property
    def wkx(self) -> int:  # gu=wt K parts
        return self.wk or (1 if self.nrt % WAVES == 0 else 2)

    @property
    def nrt(self) -> int:  # gate (or up) 16-row tiles
        return self.inter // TILE

    @property
    def nqd(self) -> int:  # down K chunks
        return -(-self.inter // CHUNK)

    @property
    def ngrp(self) -> int:  # down scale dwords per (tile, row, g)
        return -(-self.nqd // 4)

    @property
    def nw(self) -> int:
        return self.bucket // 32 if self.bucket >= 32 else 1

    @property
    def maxpass(self) -> int:
        return -(-(-(-self.bucket // TILE)) // self.tpp)

    @property
    def shsx(self) -> int:  # tokens per shared-expert item
        return self.shs or TILE * self.tpp

    @property
    def it_max(self) -> int:
        return min(E, TOPK * self.bucket) + (TOPK * self.bucket) // (TILE * self.tpp) + -(-self.bucket // self.shsx) + 1

    @property
    def m0(self) -> int:  # minimum line units per range (keeps every item within MAXSEG segments)
        return -(-(self.na + self.ov - 1) // (MAXSEG - 1))

    @property
    def unit(self) -> int:  # line units per item: ov virtual + NA atoms
        return self.na + self.ov

    @property
    def wlw(self) -> tuple:
        return tuple(int(x) for x in self.wl.split(".")) if self.wl else ()

    def wl_item(self, tokens: int) -> int:
        """Units of an item with ``tokens`` tokens under ``wl``."""
        wa, wc, ws, wt = self.wlw
        return ws + tokens * wt + self.na * wa + self.nqd * wc

    def wl_atom(self, tokens: int, a: int) -> int:
        """Unit offset (within its item) of atom ``a``'s start under ``wl``."""
        wa, wc, ws, wt = self.wlw
        return ws + tokens * wt + wc + a * wa + (a // 4) * wc

    def chunk_bytes(self, c: int) -> int:
        return min(64, (self.inter - c * CHUNK) // 2)

    def check(self) -> None:
        if self.tp not in (1, 2) or self.act not in ("fp8", "fp4") or self.bucket not in (16, 32, 64, 128):
            raise ValueError("tp 1 | 2, act fp8 | fp4, bucket 16 | 32 | 64 | 128")
        if self.pipe == "w2" and (self.tpp != 1 or self.mutant in ("lone", "half", "mask", "seg", "tick") or self.diag != "none"):
            raise ValueError("pipe=w2: tpp 1, no diag; mutants slot sort sopsel hsc desc bar noacq")
        if self.pipe not in ("stream", "unit", "w2"):
            raise ValueError("pipe stream | unit")
        if self.pipe == "stream" and self.mutant == "lone":
            raise ValueError("mutant lone is a pipe=unit mutant")
        if self.entry not in ("route", "logits") or self.null not in ("none", "oob"):
            raise ValueError("entry route | logits, null none | oob")
        if self.tpp not in (1, 2) or not 1 <= self.depth <= 10 or not 1 <= self.ddepth <= 16:
            raise ValueError("tpp 1 | 2, depth 1..10, ddepth 1..16")
        if self.ab not in (1, 2, 4):
            raise ValueError("ab 1 | 2 | 4")
        if self.prio not in ("none", "dn", "all", "alt"):
            raise ValueError("prio none | dn | all | alt")
        if self.xsync not in ("bar", "cnt") or (self.xsync == "cnt" and (self.pipe != "stream" or self.lka != "tail")):
            raise ValueError("xsync bar | cnt (cnt: pipe=stream, lka=tail)")
        if self.lka not in ("tail", "early"):
            raise ValueError("lka tail | early")
        if self.deal not in ("line", "rr", "hy", "dyn") or (self.deal != "line" and (self.pipe != "stream" or self.wl or self.ov)):
            raise ValueError("deal line | rr | dyn | hy (not line: pipe=stream, no wl / ov)")
        if self.pc not in (0, 1, 2, 3, 4) or (self.pc != 1 and self.deal not in ("rr", "dyn")) or self.uord not in ("id", "heavy", "hid", "rev", "hchk") \
                or (self.uord != "id" and self.deal not in ("rr", "dyn")):
            raise ValueError("pc 1..4, uord id | heavy (rr / dyn)")
        if self.gu not in ("atom", "wt") or not 1 <= self.wdep <= 32 or self.word not in ("chunk", "grp") or not 0 <= self.wg <= 5 or self.wk not in (0, 1, 2):
            raise ValueError("gu atom | wt, wdep 1..8")
        if self.gu == "wt" and (self.nrt * self.wkx) % WAVES:
            raise ValueError("gu=wt: tiles x wk must divide over the waves")
        if self.gu == "wt" and (self.deal not in ("rr", "dyn") or self.pc != 1 or self.tpp != 1 or self.xsync != "bar"
                                or self.tlbp or self.diag != "none"):
            raise ValueError("gu=wt: deal rr | dyn, pc 1, tpp 1, xsync bar, no tlbp / diag")
        if self.snap and (not self.wl or not 0 < self.snap <= 64):
            raise ValueError("snap 1..64 needs wl")
        if self.wl and (self.pipe != "stream" or self.ov or len(self.wlw) != 4 or min(self.wlw) < 0 or self.wlw[0] < 1):
            raise ValueError("wl 'wa.wc.ws.wt' (ints, wa >= 1; pipe=stream, ov 0)")
        if not 0 <= self.ov <= 8 or (self.ov and self.pipe != "stream"):
            raise ValueError("ov 0..8 (pipe=stream)")
        if self.xst not in ("dma", "vgpr") or self.wcp not in ("nt", "default"):
            raise ValueError("xst dma | vgpr, wcp nt | default")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "tick" and self.deal != "dyn":
            raise ValueError("mutant tick needs deal=dyn")
        if self.mutant == "tag" and self.entry != "logits":
            raise ValueError("mutant tag needs entry=logits")
        if self.mutant == "noacq" and not self.pretouch:
            raise ValueError("mutant noacq needs the pretouch twin")
        if self.hinit not in ("zero", "dirty"):
            raise ValueError("hinit zero | dirty")
        if self.diag != "none" and not set(self.diag.split("+")) <= {"nox", "nogu", "nodn", "nocmb", "gsink", "dsink", "noep", "nored", "nost"}:
            raise ValueError("diag: + list of nox nogu nodn nocmb gsink dsink noep nored nost")
        if self.pstg not in (0, 2, 4):
            raise ValueError("pstg 0 | 2 | 4")
        if self.xb and (self.deal != "rr" or self.xh):
            raise ValueError("xb needs deal=rr, no xh")
        if self.lpf not in (1, 2, 3):
            raise ValueError("lpf 1 | 2 | 3")
        if self.shs not in (0, 4, 8) or (self.shs and ((TILE * self.tpp) % self.shs or self.wl or self.pipe != "stream")):
            raise ValueError("shs 0 | 4 | 8 (dividing 16 x tpp; pipe=stream, no wl)")
        if self.sskip:  # failed numerics on MI350X (tp2 fp4 r8, tp1 fp8 r64; a tie in timing): kept for the record, refused
            raise ValueError("sskip: refused (numerics failure, lb2-n49)")
        if self.dsq and (self.pipe != "stream" or (self.dsk and self.dsk % 4) or self.shd):
            raise ValueError("dsq: pipe=stream, dsk multiple of 4, no shd")
        if self.shd and (self.deal not in ("rr", "dyn") or self.uord != "id" or self.dsk or self.shs or self.gu != "atom"
                         or self.dt > 2 or self.pstg > 2):
            raise ValueError("shd: deal rr | dyn, uord id, no dsk / shs, gu atom, dt / pstg <= 2 (10 tiles a wave per half)")
        if self.cmb not in ("gbar", "flow") or (self.cmb == "flow" and (self.pipe != "stream" or self.tpp != 1 or self.pretouch)):
            raise ValueError("cmb gbar | flow (flow: pipe=stream, tpp 1, no pretouch twin)")
        if self.dsk not in (0, 2, 4, 5) or (H // TILE // WAVES + self.dsk) % self.dt or (self.pstg and self.dsk % self.pstg)                 or (self.dsk and self.pipe != "stream"):
            raise ValueError("dsk 0 | 2 | 4 | 5, dt dividing 20 + dsk")
        if self.dt not in (1, 2, 4, 5) or self.doff not in ("mul", "imm"):
            raise ValueError("dt 1 | 2 | 4 | 5, doff mul | imm")

    def suffix(self) -> str:
        parts = []
        d = Moe2Knobs()
        for f in fields(self):
            if f.name in ("tp", "act", "bucket", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}".replace("+", "x").replace(".", "p"))
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kernel_name2(k: Moe2Knobs) -> str:
    s = k.suffix()
    return f"q38lb2_moe_tp{k.tp}_{k.act}_b{k.bucket}" + (f"_{s}" if s else "")


def moe2_params(k: Moe2Knobs) -> list[str]:
    """i64 addresses then i32 ``rows, layers, gsync`` (host packs ``<Q...iii``)."""
    p = ["lg", "pr", "x", "xs", "wgu", "sgu", "wdn", "sdn", "part", "desc", "y", "cnt", "gs", "hd", "hds"]
    if k.stamp:
        p.append("stamps")
    return p + ["rows", "layers", "gsync"]


def dims2(k: Moe2Knobs) -> dict:
    """Byte sizes of one expert's blocks (``dns`` = the dsx down-scale layout) and the per-row strides."""
    i = k.inter
    gub = 2 * i * H // 2
    gus = (2 * i // TILE) * NQG * 64
    dnb = H * i // 2
    dns = H * 16 * k.ngrp
    dns_min = (H // TILE) * k.nqd * 64  # P-L4's (unpadded) down-scale bytes: the floor's expert size
    return dict(inter=i, gub=gub, gus=gus, dnb=dnb, dns=dns, expert=gub + gus + dnb + dns, expert_min=gub + gus + dnb + dns_min,
                hrow=i if k.act == "fp8" else i // 2, xrow=H if k.act == "fp8" else H // 2, xsrow=H // 32)


def counter_words2(k: Moe2Knobs) -> int:
    return CNT_WORDS


def line_plan(k: Moe2Knobs, n_items: int, grid: int = GRID, tokens=None):
    """Host model of the deal: (C, [(line pos, item, a0, a1, seg, nseg), ...]) per CTA index (``xh`` mapping applied);
    ``tokens`` (per item) is needed for ``wl``."""
    if k.wl:
        return _line_plan_wl(k, tokens, grid)
    if k.deal in ("rr", "dyn"):  # dyn: the rr schedule stands in (the dynamic one is unknown on the host)
        NA = k.na
        U = unit_split(k, n_items, grid)[2]
        out = {}
        for cta in range(grid):
            L = (cta % XCDS) * (grid // XCDS) + cta // XCDS if k.xh else cta
            if k.xb:
                us = [xb_unit(U, r, cta, grid) for r in range(-(-U // grid))]
                us = [u for u in us if u is not None]
            else:
                us = list(range(L, U, grid))
            out[cta] = [(L,) + unit_seg(k, u, n_items, grid) for u in us]
        return grid, out
    if k.deal == "hy":
        return _line_plan_rr(k, n_items, grid)
    NA, U, ov = k.na, k.unit, k.ov
    A = n_items * U
    C = max(1, min(grid, A // k.m0))

    def lpos(x):  # line position holding unit x
        return ((x + 1) * C - 1) // A

    out = {}
    for cta in range(grid):
        L = (cta % XCDS) * (grid // XCDS) + cta // XCDS if k.xh else cta
        segs = []
        if L < C:
            lo, hi = L * A // C, (L + 1) * A // C
            if hi > lo:
                for it in range(lo // U, (hi - 1) // U + 1):
                    a0, a1 = max(lo - it * U - ov, 0), min(hi - it * U - ov, NA)
                    if a1 <= a0:
                        continue
                    lf, ll = lpos(it * U + ov), lpos(it * U + ov + NA - 1)
                    segs.append((L, it, a0, a1, L - lf, ll - lf + 1))
        out[cta] = segs
    return C, out


def _line_plan_rr(k: Moe2Knobs, n_items: int, grid: int = GRID):
    """rr / hy: whole items k on CTA (line position) k mod 256 (hy: the first N // 256 rounds), then (hy) the remainder
    on the atom line (segment numbers within the remainder line)."""
    NA = k.na
    R = n_items // grid if k.deal == "hy" else -(-n_items // grid)
    k0 = min(R * grid, n_items)
    A = (n_items - k0) * NA
    C = max(1, min(grid, A // k.m0)) if A else 1

    def lpos(x):
        return ((x + 1) * C - 1) // A

    out = {}
    for cta in range(grid):
        L = (cta % XCDS) * (grid // XCDS) + cta // XCDS if k.xh else cta
        segs = [(L, it, 0, NA, 0, 1) for it in range(L, k0, grid)]
        if A and L < C:
            lo, hi = L * A // C, (L + 1) * A // C
            if hi > lo:
                for j in range(lo // NA, (hi - 1) // NA + 1):
                    a0, a1 = max(lo - j * NA, 0), min(hi - j * NA, NA)
                    lf, ll = lpos(j * NA), lpos(j * NA + NA - 1)
                    segs.append((L, k0 + j, a0, a1, L - lf, ll - lf + 1))
        out[cta] = segs
    return C, out


def xb_unit(U: int, r: int, cta: int, grid: int = GRID):
    """xb: CTA ``cta``'s unit of round ``r`` (None if none): XCD x = cta % 8 takes the round's block x of S consecutive
    units, S = 32 in a full round, ceil(left / 8) in the last one."""
    x, j = cta % XCDS, cta // XCDS
    left = U - grid * r
    S = grid // XCDS if left >= grid else -(-left // XCDS)
    u = grid * r + x * S + j
    return u if j < S and u < U else None


def pc_of(k: Moe2Knobs, n_items: int, grid: int = GRID) -> int:
    """Pieces per item of the last (partial) round: ``k.pc``, or (pc 0) as many (up to 4) as fit the grid."""
    if k.pc:
        return k.pc
    rem = n_items % grid
    cap = min(4, -(-k.na // 4)) if k.pcal else 4  # aligned pieces: at most one per down chunk
    return max(1, min(cap, grid // rem)) if rem else 1


def unit_split(k: Moe2Knobs, n_items: int, grid: int = GRID):
    """(whole units, pieces per remaining item, total units): pc 0 keeps the full rounds whole and splits only the
    last round's items; pc > 0 splits every item."""
    if k.pc:
        return 0, k.pc, n_items * k.pc
    F = (n_items // grid) * grid
    p = pc_of(k, n_items, grid)
    return F, p, F + (n_items - F) * p


def unit_seg(k: Moe2Knobs, u: int, n_items: int, grid: int = GRID):
    """(item, a0, a1, jseg, nseg) of unit ``u``."""
    F, p, _ = unit_split(k, n_items, grid)
    NA = k.na
    if u < F:
        return u, 0, NA, 0, 1
    v = u - F
    it, j = F + v // p, v % p
    if k.pcal:  # pieces on down-chunk boundaries (4 atoms)
        NC = -(-NA // 4)
        return it, min(NA, 4 * (j * NC // p)), min(NA, 4 * ((j + 1) * NC // p)), j, p
    return it, j * NA // p, (j + 1) * NA // p, j, p


def wl_c(k: Moe2Knobs, W: int) -> int:
    umax = k.wl_item(TILE * k.tpp)
    return max(1, min(GRID, (MAXSEG - 1) * W // (umax + 1 + 2 * (MAXSEG - 1) * k.snap)))


def wl_cuts(k: Moe2Knobs, pre: list, C: int) -> list:
    """The C + 1 line cuts (units), snapped onto item boundaries within ``k.snap`` units."""
    W = pre[-1]
    out = []
    for t in range(C + 1):
        x = t * W // C
        if k.snap and 0 < x < W:
            it = bisect.bisect_right(pre, x) - 1
            b0, b1 = pre[it], pre[it + 1]
            if x - b0 <= k.snap:
                x = b0
            elif b1 - x <= k.snap:
                x = b1
        out.append(x)
    return out


def _line_plan_wl(k: Moe2Knobs, tokens, grid: int = GRID):
    NA = k.na
    pre = [0]
    for t in tokens:
        pre.append(pre[-1] + k.wl_item(t))
    W = pre[-1]
    C = wl_c(k, W)
    cuts = wl_cuts(k, pre, C)

    def lpos(x):
        return min(bisect.bisect_right(cuts, x) - 1, C - 1)

    out = {}
    for cta in range(grid):
        L = (cta % XCDS) * (grid // XCDS) + cta // XCDS if k.xh else cta
        segs = []
        if L < C:
            lo, hi = cuts[L], cuts[L + 1]
            for it in range(max(0, bisect.bisect_right(pre, lo) - 2), min(len(tokens), bisect.bisect_right(pre, hi) + 1)):
                us = [pre[it] + k.wl_atom(tokens[it], a) for a in range(NA)]
                a0 = sum(1 for u in us if u < lo)
                a1 = sum(1 for u in us if u < hi)
                if a1 > a0:
                    lf, ll = lpos(us[0]), lpos(us[-1])
                    segs.append((L, it, a0, a1, L - lf, ll - lf + 1))
        out[cta] = segs
    return C, out


# ============================================================ kernel


class _Moe2:
    def __init__(self, k: Moe2Knobs):
        k.check()
        self.k = k
        self.dm = dims2(k)
        self.b = b = IRBuilder(kernel_name2(k))
        b.kernel.attrs["max_workgroup_size"] = THREADS
        if k.dlev:
            b.kernel.attrs["direct_levers"] = k.dlev
        names = moe2_params(k)
        self.a = {n: b.param(n, I64) for n in names[:-3]}
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.L = b.readfirstlane(b.param("layers", I32))
        self.gsync = b.cmp_ne(b.readfirstlane(b.param("gsync", I32)), _c(b, 0))
        self.cta = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))
        self.l16 = b.land(self.lane, _c(b, 15))
        self.wt = b.readfirstlane(b.land(self.wave, _c(b, 3)))  # stream: gate_up 16-row tile of the atom (0, 1 gate; 2, 3 up)
        self.wh = b.readfirstlane(b.lshr(self.wave, _c(b, 2)))  # stream: K half
        self.gvo = b.add(b.mul(self.l16, _c(b, 64)), b.mul(self.g, _c(b, 16)))  # lane part of a 1 KiB tile-chunk load
        self.gso = b.add(b.mul(self.l16, _c(b, 16)), b.mul(self.g, _c(b, 4)))  # lane part of a sop scale dword
        self.lane0 = b.cmp_eq(self.lane, _c(b, 0))
        self.wcp = CPOL_NT if k.wcp == "nt" else 0
        self.cnt_a = b.readfirstlane(self.a["cnt"])
        self.cnt_rs = _rsrc(b, self.cnt_a, _c(b, 4 * CNT_WORDS))
        self.ep = b.readfirstlane(b.buffer_load_cp(self.cnt_rs, _c(b, 4 * W_EP), _c(b, 0), dwords=1, aux=CPOL_SC1))
        dm = self.dm
        P = k.tpp
        # ---- LDS
        self.XST = dm["xrow"] + 16  # padded token row (bytes): conflict-free B reads
        self.HST = dm["hrow"] + 16
        self.XSS = 80  # x scale row (bytes, sop layout): 20 dwords, conflict-free for 16 rows
        self.xl = b.smem_alloc(I32, [P * TILE * self.XST // 4], name_hint="xl")
        self.xsl = b.smem_alloc(I8, [P * TILE * self.XSS], name_hint="xsl")
        self.hl = b.smem_alloc(I32, [P * TILE * self.HST // 4], name_hint="hl")
        self.hsl = b.smem_alloc(I8, [P * TILE * 4 * 8], name_hint="hsl")  # [p][tok][g][chunk] E8M0 bytes
        nred = 2 * (k.ab if k.pipe == "stream" else 1)
        if k.gu == "wt":
            nred = 0
            self.PST = k.inter * (2 if k.wkx == 2 else 1) + 4  # panel row (f32): 16 B skew per row
            npw_ = k.nrt * k.wkx // WAVES
            nsw = -(-npw_ // (k.wg or npw_))  # sweeps: an early sweep's panel would overwrite x
            self.RG = P * TILE * self.PST  # wk 2: one g | u partial region per K part
            if P * TILE * self.PST * 4 <= P * TILE * self.XST and nsw == 1 and k.pal and k.wkx == 1:
                self.pnl = self.xl
            else:
                self.pnl = b.smem_alloc(I32, [self.RG * k.wkx], name_hint="pnl")
        self.red = b.smem_alloc(I32, [max(nred * 8 * P * 256, 4)], name_hint="red")  # [buf][entry][p][lane][4] f32
        if k.pstg:
            self.SGS = k.pstg * TILE + 4  # stage row (f32)
            sw = WAVES * P * TILE * self.SGS
            if k.gu == "wt" and self.pnl is not self.xl and self.RG * k.wkx >= sw:
                self.stg = self.pnl  # the panel is free during the down
            else:
                self.stg = b.smem_alloc(I32, [sw], name_hint="stg")
                nred += -(-sw // (8 * P * 256))
        self.bm = b.smem_alloc(I32, [NEXP * k.nw], name_hint="bm")
        self.eo = b.smem_alloc(I32, [NEXP], name_hint="eo")  # list offset | count << 16
        self.itl = b.smem_alloc(I32, [k.it_max], name_hint="itl")  # expert | pass << 10
        self.lst = b.smem_alloc(I32, [SLOTS * k.bucket], name_hint="lst")  # token | slot << 8
        self.lw = b.smem_alloc(F32, [SLOTS * k.bucket], name_hint="lw")
        self.misc = b.smem_alloc(I32, [16], name_hint="misc")
        self.xcnt = b.smem_alloc(I32, [4], name_hint="xcnt")
        if k.deal in ("rr", "dyn"):
            self.tick = b.smem_alloc(I32, [4], name_hint="tick")  # dyn: the next unit, by segment parity
            if k.uord != "id":
                self.perm = b.smem_alloc(I32, [k.it_max], name_hint="perm")  # rank -> item
                self.bins = b.smem_alloc(I32, [2 * (TILE * k.tpp + 2)], name_hint="bins")
                self.cntw = b.smem_alloc(I32, [-(-k.it_max // THREADS) * WAVES * (TILE * k.tpp + 1)], name_hint="cntw")
        if k.wl:
            self.ipre = b.smem_alloc(I32, [k.it_max + 1], name_hint="ipre")  # exclusive prefix of the item units
            self.cuts = b.smem_alloc(I32, [GRID + 1], name_hint="cuts")  # the line cuts (units)
            self.wsc = b.smem_alloc(I32, [16], name_hint="wsc")
        self.gsw = b.smem_alloc(F32, [1], name_hint="gsw")
        if k.gu == "wt" and self.pnl is not self.xl:
            nred = self.RG * k.wkx // (8 * P * 256) + 1  # the panel in the budget
        used = 4 * (P * TILE * (self.XST + self.HST) // 4 + P * TILE * self.XSS // 4 + nred * 8 * P * 256 + NEXP * (k.nw + 1)
                    + k.it_max + 2 * SLOTS * k.bucket + 17) + P * TILE * 32
        if used > 160 * 1024 - 256:
            raise ValueError(f"{kernel_name2(k)}: LDS {used} bytes over the CU's 160 KiB (tpp 2 needs ab 1 at fp8)")
        padw = ONE_CTA_PER_CU_WORDS - used // 4
        if padw > 64:  # one CTA per CU: keep the group segment above half the CU's LDS
            pad = b.smem_alloc(F32, [padw], name_hint="occpad")
            a0 = b.trunc(b.smem_addr_of(pad), I32)
            with b.scf_if(D.is_thread0(b)):
                b.inline_asm("ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)", "v,v,~{memory}",
                             [b.add(a0, _c(b, 4 * (padw - 1))), _c(b, 0)], None)
        self.xl_a = b.smem_addr_of(self.xl)
        self.xsl_a = b.smem_addr_of(self.xsl)
        # ---- per-layer strides and resources
        self.lgu = b.const_i64(NEXP * dm["gub"])
        self.lgs = b.const_i64(NEXP * dm["gus"])
        self.ldn = b.const_i64(NEXP * dm["dnb"])
        self.lds_ = b.const_i64(NEXP * dm["dns"])
        self.llg = b.const_i64(k.bucket * LST * 4)
        self.lpr = b.const_i64(k.bucket * PRS * 8)
        self.ly = b.const_i64(k.bucket * H * 2)
        rl = b.smax(self.rows, _c(b, 1))
        nox = self.dg("nox")
        # buffer resources: (base, bytes); built at every use when k.lrs (no resource held for the whole kernel)
        self._rsa = dict(x=(self.a["x"], _c(b, 0) if nox else b.mul(rl, _c(b, dm["xrow"]))),
                         xs=(self.a["xs"], _c(b, 0) if nox else b.mul(rl, _c(b, dm["xsrow"]))),
                         part=(self.a["part"], _c(b, k.bucket * SLOTS * MAXSEG * H * 4)),
                         desc=(self.a["desc"], _c(b, k.bucket * DSTR * 4)),
                         hd=(self.a["hd"], _c(b, SLOTS * k.bucket * dm["hrow"])),
                         hds=(self.a["hds"], _c(b, SLOTS * k.bucket * 4 * k.na)))
        if not k.lrs:
            self._rsh = {n: _rsrc(b, a_, n_) for n, (a_, n_) in self._rsa.items()}
        # h / h scales start finite (stale LDS may hold NaN codes; masked down lanes multiply them by zero weights)
        nh = P * TILE * self.HST // 4
        for i in range(-(-nh // THREADS)):
            idx = b.add(self.tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(idx, _c(b, nh))):
                self.lds_put(self.hl, idx, _c(b, 0x38383838 if k.hinit == "dirty" else 0))
        nhs = P * TILE * 32
        for i in range(-(-nhs // THREADS)):
            idx = b.add(self.tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(idx, _c(b, nhs))):
                b.smem_store_vN(self.hsl, [idx], b.trunc(_c(b, 0x7F), I8), 1)

    # ------------------------------------------------------------------ helpers
    def _rs(self, n: str) -> Value:
        if not self.k.lrs:
            return self._rsh[n]
        a_, n_ = self._rsa[n]
        return _rsrc(self.b, a_, n_)

    x_rs = property(lambda self: self._rs("x"))
    xs_rs = property(lambda self: self._rs("xs"))
    part_rs = property(lambda self: self._rs("part"))
    desc_rs = property(lambda self: self._rs("desc"))
    hd_rs = property(lambda self: self._rs("hd"))
    hds_rs = property(lambda self: self._rs("hds"))

    def dg(self, what: str) -> bool:
        return what in self.k.diag.split("+")

    def lds_i(self, smem, idx) -> Value:
        b = self.b
        return b.vec_extract(b.smem_load_vN(smem, idx, dtype=I32, n=1), 0)

    def lds_f(self, smem, idx) -> Value:
        b = self.b
        return b.vec_extract(b.smem_load_vN_f32(smem, idx, n=1), 0)

    def lds_put(self, smem, idx, v) -> None:
        self.b.smem_store_vN(smem, [idx], v, 1)

    def err(self) -> None:
        _aadd(self.k, self.b, self.cnt_a, _c(self.b, 4 * W_ERR), _c(self.b, 1), scope="agent")

    def deadline(self) -> Value:
        b = self.b
        return b.add(D.memrealtime(b), b.const_i64(200_000_000))

    def stamp(self, layer: Value, p: int) -> None:
        if not self.k.stamp:
            return
        b = self.b
        with b.scf_if(b.land(self.lane0, b.cmp_eq(layer, b.sub(self.L, _c(b, 1))))):
            sp = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            b.store(D.memrealtime(b), sp, b.add(b.mul(b.add(b.mul(self.cta, _c(b, WAVES)), self.wave), _c(b, STAMP_PTS)), _c(b, p)))

    def sstamp(self, layer: Value, seg: Value, p: int) -> None:
        """Per-segment stamp (``stamp`` builds, last layer): wave w of the CTA, segment ``seg`` (< 4), point ``p`` (0 start,
        1 x staged, 2 atom loop entry, 3 first atom reduced, 4 atom loop exit, 5 h complete, 6 down operands + next DMA
        issued, 7 down done) at ``stamps[GRID * 8 * STAMP_PTS + ((cta * 8 + w) * 4 + seg) * 8 + p]``."""
        if not self.k.stamp:
            return
        b = self.b
        with b.scf_if(b.land(b.land(self.lane0, b.cmp_eq(layer, b.sub(self.L, _c(b, 1)))), b.cmp_lt(seg, _c(b, 4)))):
            sp = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            i = b.add(_c(b, GRID * WAVES * STAMP_PTS + p), b.mul(b.add(b.mul(b.add(b.mul(self.cta, _c(b, WAVES)), self.wave), _c(b, 4)), seg), _c(b, 8)))
            b.store(D.memrealtime(b), sp, i)

    def tag_of(self, layer: Value) -> Value:
        b = self.b
        return b.readfirstlane(b.add(b.land(b.add(self.ep, layer), _c(b, 0x7FFF)), _c(b, 1)))

    # ------------------------------------------------------------------ route (entry=logits): P-L4's top-10, tagged pairs
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
            my_id = _c(b, SH)
            my_w = sg
            for j in reversed(range(TOPK)):
                sel = b.cmp_eq(self.lane, _c(b, j))
                my_id = b.select(sel, ids[j], my_id)
                my_w = b.select(sel, wts[j], my_w)
            tg = self.tag_of(layer)
            word = b.lor(my_id, b.shl(tg, _c(b, 16)))
            pr_a = b.add(self.a["pr"], b.mul(_i64(b, layer), self.lpr))
            pr_rs = _rsrc(b, pr_a, _c(b, k.bucket * PRS * 8))
            off = b.mul(b.add(b.mul(r, _c(b, PRS)), self.lane), _c(b, 8))
            ok = b.cmp_le(self.lane, _c(b, TOPK))
            b.buffer_store_cp(b.vec_pack([word, b.bitcast(my_w, I32)], I32), pr_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)

    def poll_tag(self, pr_a: Value, voff: Value, tag: Value) -> Value:
        """Every active lane spins until the id word at ``pr_a + voff`` carries ``tag`` (bits 16..30); 2 s deadline.
        Returns the status (uniform): 0 ok, 1 timeout."""
        b = self.b
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        s_ = "t"
        lines = [
            "s_nop 4" if self.k.hz else "",
            "s_mov_b32 $2, 0",
            W._lbl(s_, "top") + ":",
            "global_load_dword $0, $5, $6 sc0 sc1",
            "s_waitcnt vmcnt(0)",
            "v_lshrrev_b32 $0, 16, $0",
            "v_cmp_ne_u32_e64 $1, $0, $7",
            "s_and_b64 $1, $1, exec",
            "s_cbranch_scc0 " + W._lbl(s_, "ok"),
            *W._tail(s_, "s_sleep1", "$8", "$1", "$4", "$2", "$3"),
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
            [voff, pr_a, tag, self.deadline()],
            result_types=[I32, I64, I32, I32, I64],
            result_name_hint="pollt",
        )
        return st

    # ------------------------------------------------------------------ sort (whole CTA)
    def sort(self, layer: Value) -> Value:
        """Bitmaps, counts, one packed scan (list offsets | item offsets), the item table and the token lists; returns
        the item count (uniform)."""
        b, k = self.b, self.k
        nw, tid, P = k.nw, self.tid, k.tpp
        nbm = NEXP * nw
        for i in range(-(-nbm // THREADS)):
            idx = b.add(tid, _c(b, i * THREADS))
            with b.scf_if(b.cmp_lt(idx, _c(b, nbm))):
                self.lds_put(self.bm, idx, _c(b, 0))
        pr_a = b.readfirstlane(b.add(self.a["pr"], b.mul(_i64(b, layer), self.lpr)))
        pr_rs = _rsrc(b, pr_a, _c(b, k.bucket * PRS * 8))
        npairs = b.mul(self.rows, _c(b, TOPK))
        tg = self.tag_of(layer) if k.entry == "logits" else None
        aux = CPOL_SC1 if k.entry == "logits" else 0
        pairs = []
        for i in range(-(-(TOPK * k.bucket) // THREADS)):
            p = b.add(tid, _c(b, i * THREADS))
            ok = b.cmp_lt(p, npairs)
            tok = b.div(p, _c(b, TOPK))
            kk = b.sub(p, b.mul(tok, _c(b, TOPK)))
            off = b.select(ok, b.mul(b.add(b.mul(tok, _c(b, PRS)), kk), _c(b, 8)), _c(b, OOB))
            if k.entry == "logits" and k.mutant != "tag":
                with b.scf_if(ok):
                    st = self.poll_tag(pr_a, off, tg)
                    with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), self.lane0)):
                        self.err()
            v = b.buffer_load_cp(pr_rs, off, _c(b, 0), dwords=2, aux=aux)
            e = b.select(ok, b.land(b.vec_extract(v, 0), _c(b, 511)), _c(b, 0))
            w = b.vec_extract(v, 1)
            pairs.append((ok, tok, kk, e, w))
        b.sync_lds_only()
        for ok, tok, kk, e, w in pairs:
            with b.scf_if(ok):
                b.lds_atomic_add(self.bm, [b.add(b.mul(e, _c(b, nw)), b.lshr(tok, _c(b, 5)))], b.shl(_c(b, 1), b.land(tok, _c(b, 31))))
        # the shared expert's gate (slot 10) of token tid
        if k.entry == "logits" and k.mutant != "tag":
            with b.scf_if(b.cmp_lt(tid, self.rows)):
                st = self.poll_tag(pr_a, b.mul(b.add(b.mul(tid, _c(b, PRS)), _c(b, TOPK)), _c(b, 8)), tg)
                with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), self.lane0)):
                    self.err()
        sgv = b.vec_extract(b.buffer_load_cp(pr_rs, b.select(b.cmp_lt(tid, self.rows), b.mul(b.add(b.mul(tid, _c(b, PRS)), _c(b, TOPK)), _c(b, 8)),
                                                             _c(b, OOB)), _c(b, 0), dwords=2, aux=aux), 1)
        b.sync_lds_only()
        # thread e = expert e: count, tiles, passes; packed scan of count | passes << 16
        cnt = _c(b, 0)
        for w_ in range(nw):
            cnt = b.add(cnt, _bcnt(b, self.lds_i(self.bm, b.add(b.mul(tid, _c(b, nw)), _c(b, w_)))))
        tiles = b.lshr(b.add(cnt, _c(b, 15)), _c(b, 4))
        passes = b.div(b.add(tiles, _c(b, P - 1)), _c(b, P)) if P > 1 else tiles
        val = b.add(cnt, b.shl(passes, _c(b, 16)))
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
        ioff = b.lshr(exc, _c(b, 16))
        self.lds_put(self.eo, tid, b.add(eoff, b.shl(cnt, _c(b, 16))))
        for j in range(k.maxpass):
            with b.scf_if(b.cmp_lt(_c(b, j), passes)):
                self.lds_put(self.itl, b.add(ioff, _c(b, j)), b.add(tid, _c(b, j << 10)))
        tot = b.readfirstlane(tot)
        off_sh = b.land(tot, _c(b, 0xFFFF))
        it_sh = b.lshr(tot, _c(b, 16))
        if k.shs:  # shared-expert items of shs tokens
            psh = b.div(b.add(self.rows, _c(b, k.shs - 1)), _c(b, k.shs))
        else:
            tsh = b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4))
            psh = b.div(b.add(tsh, _c(b, P - 1)), _c(b, P)) if P > 1 else tsh
        with b.scf_if(b.cmp_lt(tid, psh)):
            self.lds_put(self.itl, b.add(it_sh, tid), b.add(_c(b, SH), b.shl(tid, _c(b, 10))))
        with b.scf_if(D.is_thread0(b)):
            self.lds_put(self.eo, _c(b, SH), b.add(off_sh, b.shl(self.rows, _c(b, 16))))
            self.lds_put(self.misc, _c(b, 0), b.add(it_sh, psh))
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
                    msk = b.lor(msk, b.shl(_c(b, 1), bit))
                below = b.add(below, _bcnt(b, b.land(word, msk)))
                pos = b.add(b.land(self.lds_i(self.eo, e), _c(b, 0xFFFF)), below)
                self.lds_put(self.lst, pos, b.add(tok, b.shl(kk, _c(b, 8))))
                self.lds_put(self.lw, pos, b.bitcast(w, F32))
        with b.scf_if(b.cmp_lt(tid, self.rows)):
            pos = b.add(off_sh, tid)
            self.lds_put(self.lst, pos, b.add(tid, _c(b, TOPK << 8)))
            self.lds_put(self.lw, pos, b.bitcast(sgv, F32))
        b.sync_lds_only()
        nit = b.readfirstlane(self.lds_i(self.misc, _c(b, 0)))
        if k.wl:
            self.wl_scan(nit)
        if k.deal in ("rr", "dyn") and k.uord != "id":
            self.heavy_order(nit)
        return nit

    def heavy_order(self, nit: Value) -> None:
        """perm[rank] = item: items by token count, most first, ties by item index -- a stable counting sort, so every
        CTA computes the same order (an atomic-order sort would deal different units on different CTAs). Chunks of 64
        items (one wave, one round) count each key by ballot; chunk-prefix per key; key prefix; position = key offset +
        chunk offset + rank in the chunk."""
        b, k = self.b, self.k
        TP = TILE * k.tpp
        NK = TP + 1  # keys 0 (TP tokens) .. TP - 1 (1 token); NK - 1 unused
        NB = NK + 1
        R = -(-k.it_max // THREADS)
        lane = self.lane
        items = []
        for r in range(R):
            i = b.add(self.tid, _c(b, r * THREADS))
            ok = b.cmp_lt(i, nit)
            key = b.sub(_c(b, TP), self.wl_tokens_v(b.select(ok, i, _c(b, 0))))
            rank = _c(b, 0)
            cntv = _c(b, 0)
            for kv in range(NK):
                m = b.wave_ballot(b.select(b.land(ok, b.cmp_eq(key, _c(b, kv))), _c(b, 1), _c(b, 0)))
                lo = b.trunc(m, I32)
                hi = b.trunc(b.lshr(m, b.const_i64(32)), I32)
                # the ballot's SGPR pair is written by a VALU compare right before: an asm VALU read of it needs the pad
                # (observed on MI350X: wrong mbcnt ranks without it; the hazard recognizer does not cover the asm)
                pad = "s_nop 4\n" if (k.hz or k.hdbg & 2) else ""
                rk = b.inline_asm(pad + "v_mbcnt_lo_u32_b32 $0, $1, 0\nv_mbcnt_hi_u32_b32 $0, $2, $0",
                                  "=&v,s,s", [lo, hi], I32,
                                  sideeffect=False, result_name_hint="mbc")
                rank = b.select(b.cmp_eq(key, _c(b, kv)), rk, rank)
                nb_ = b.inline_asm(pad + "v_bcnt_u32_b32 $0, $1, 0\nv_bcnt_u32_b32 $0, $2, $0", "=&v,s,s", [lo, hi], I32,
                                   sideeffect=False, result_name_hint="bcm")
                cntv = b.select(b.cmp_eq(lane, _c(b, kv)), nb_, cntv)
            c = b.add(_c(b, r * WAVES), self.wave)  # chunk index (items in index order)
            with b.scf_if(b.cmp_lt(lane, _c(b, NK))):
                self.lds_put(self.cntw, b.add(b.mul(c, _c(b, NK)), lane), cntv)
            items.append((ok, i, key, rank, c))
        b.sync_lds_only()
        with b.scf_if(b.land(b.cmp_eq(self.wave, _c(b, 0)), b.cmp_lt(lane, _c(b, NK)))):  # lane kv: chunk prefix, total
            run = _c(b, 0)
            for c in range(R * WAVES):
                ix = b.add(_c(b, c * NK), lane)
                t_ = self.lds_i(self.cntw, ix)
                self.lds_put(self.cntw, ix, run)
                run = b.add(run, t_)
            self.lds_put(self.bins, lane, run)
        b.sync_lds_only()
        with b.scf_if(D.is_thread0(b)):  # exclusive key offsets in bins[NB ..]
            acc = _c(b, 0)
            for j in range(NK):
                self.lds_put(self.bins, _c(b, NB + j), acc)
                acc = b.add(acc, self.lds_i(self.bins, _c(b, j)))
        b.sync_lds_only()
        for ok, i, key, rank, c in items:
            with b.scf_if(ok):
                pos = b.add(b.add(self.lds_i(self.bins, b.add(key, _c(b, NB))), self.lds_i(self.cntw, b.add(b.mul(c, _c(b, NK)), key))), rank)
                tgt = {"heavy": pos, "hchk": pos, "hid": i}.get(k.uord, b.sub(b.sub(nit, _c(b, 1)), i))  # hid / rev: debug
                self.lds_put(self.perm, tgt, i)
        b.sync_lds_only()
        if k.hdbg & 1:
            b.sync_lds_only()
            b.sync_lds_only()
        if k.uord == "hchk" or k.hdbg & 12:  # debug: perm is a permutation of [0, nit) (sums of v and v^2), else err
            with b.scf_if(D.is_thread0(b)):
                self.lds_put(self.misc, _c(b, 2), _c(b, 0))
                self.lds_put(self.misc, _c(b, 3), _c(b, 0))
            b.sync_lds_only()
            for ok, i, key, rank, c in items:
                with b.scf_if(ok):
                    v = self.lds_i(self.perm, i)
                    b.lds_atomic_add(self.misc, [_c(b, 2)], b.sub(v, i))
                    b.lds_atomic_add(self.misc, [_c(b, 3)], b.sub(b.mul(v, v), b.mul(i, i)))
            b.sync_lds_only()
            if not k.hdbg & 8:
                with b.scf_if(b.land(D.is_thread0(b), b.cmp_ne(b.lor(self.lds_i(self.misc, _c(b, 2)), self.lds_i(self.misc, _c(b, 3))), _c(b, 0)))):
                    self.err()
            b.sync_lds_only()

    def wl_tokens_v(self, it: Value) -> Value:
        """Token count of item ``it`` (per lane)."""
        b, k = self.b, self.k
        v = self.lds_i(self.itl, it)
        e = b.land(v, _c(b, 1023))
        ps = b.lshr(v, _c(b, 10))
        cnt = b.lshr(self.lds_i(self.eo, e), _c(b, 16))
        tp_ = b.select(b.cmp_eq(e, _c(b, SH)), _c(b, k.shsx), _c(b, TILE * k.tpp))
        return b.smin(b.sub(cnt, b.mul(ps, tp_)), tp_)

    def wl_scan(self, nit: Value) -> None:
        """``wl``: units of every item (by its token count) and their exclusive prefix ``ipre`` (``ipre[nit]`` = total)."""
        b, k = self.b, self.k
        wa, wc, ws, wt = k.wlw
        P = k.tpp
        base = _c(b, 0)
        for r in range(-(-k.it_max // THREADS)):
            i = b.add(self.tid, _c(b, r * THREADS))
            ok = b.cmp_lt(i, nit)
            v = self.lds_i(self.itl, b.select(ok, i, _c(b, 0)))
            e = b.land(v, _c(b, 1023))
            ps = b.lshr(v, _c(b, 10))
            cnt = b.lshr(self.lds_i(self.eo, e), _c(b, 16))
            tk = b.smin(b.sub(cnt, b.mul(ps, _c(b, TILE * P))), _c(b, TILE * P))
            val = b.select(ok, b.add(_c(b, ws + k.na * wa + k.nqd * wc), b.mul(tk, _c(b, wt))), _c(b, 0))
            inc = val
            for s_ in (1, 2, 4, 8, 16, 32):
                src = b.mul(b.smax(b.sub(self.lane, _c(b, s_)), _c(b, 0)), _c(b, 4))
                o = b.ds_bpermute(src, inc)
                inc = b.add(inc, b.select(b.cmp_ge(self.lane, _c(b, s_)), o, _c(b, 0)))
            with b.scf_if(b.cmp_eq(self.lane, _c(b, 63))):
                self.lds_put(self.wsc, self.wave, inc)
            b.sync_lds_only()
            pre = base
            tot = base
            for w_ in range(WAVES):
                t_ = self.lds_i(self.wsc, _c(b, w_))
                pre = b.add(pre, b.select(b.cmp_lt(_c(b, w_), self.wave), t_, _c(b, 0)))
                tot = b.add(tot, t_)
            with b.scf_if(b.cmp_le(i, nit)):
                self.lds_put(self.ipre, i, b.add(pre, b.sub(inc, val)))
            base = b.readfirstlane(tot)
            b.sync_lds_only()
        # the cuts: thread t < C + 1 computes cut t (its own binary search over the item prefix when snapping)
        W = b.readfirstlane(self.lds_i(self.ipre, nit))
        C = self.wl_cnum(W)
        t = self.tid
        with b.scf_if(b.cmp_le(t, C)):
            x = b.div(b.mul(t, W), C)
            if k.snap:
                lo = _c(b, 0)
                hi = nit
                for _ in range(max(1, k.it_max.bit_length())):
                    mid = b.lshr(b.add(lo, hi), _c(b, 1))
                    v = self.lds_i(self.ipre, mid)
                    big = b.cmp_gt(b.sub(hi, lo), _c(b, 1))
                    nlo = b.select(b.land(big, b.cmp_le(v, x)), mid, lo)
                    hi = b.select(b.land(big, b.cmp_gt(v, x)), mid, hi)
                    lo = nlo
                b0 = self.lds_i(self.ipre, lo)
                b1 = self.lds_i(self.ipre, b.add(lo, _c(b, 1)))
                inner = b.land(b.cmp_gt(x, _c(b, 0)), b.cmp_lt(x, W))
                x = b.select(b.land(inner, b.cmp_le(b.sub(x, b0), _c(b, k.snap))), b0,
                             b.select(b.land(inner, b.cmp_le(b.sub(b1, x), _c(b, k.snap))), b1, x))
            self.lds_put(self.cuts, t, x)
        b.sync_lds_only()

    # ------------------------------------------------------------------ segment pieces
    def item_info(self, it: Value):
        """(expert, pass, list offset, count) of item ``it`` (uniform)."""
        b = self.b
        v = b.readfirstlane(self.lds_i(self.itl, it))
        e = b.land(v, _c(b, 1023))
        ps = b.lshr(v, _c(b, 10))
        eo = b.readfirstlane(self.lds_i(self.eo, e))
        off, cnt = b.land(eo, _c(b, 0xFFFF)), b.lshr(eo, _c(b, 16))
        if self.k.shs:  # a shared-expert item: pass 0 of its own shs-token slice of the list
            sh = b.cmp_eq(e, _c(b, SH))
            base = b.mul(ps, _c(b, self.k.shs))
            off = b.readfirstlane(b.select(sh, b.add(off, base), off))
            cnt = b.readfirstlane(b.select(sh, b.smin(b.sub(cnt, base), _c(b, self.k.shs)), cnt))
            ps = b.readfirstlane(b.select(sh, _c(b, 0), ps))
        return e, ps, off, cnt

    def stage_x(self, off: Value, cnt: Value, ps: Value) -> None:
        """The pass's token rows of the MoE input to LDS (codes by DMA or loads + ds_write, scales by loads)."""
        b, k, dm = self.b, self.k, self.dm
        P = k.tpp
        xrow = dm["xrow"]
        npc = -(-xrow // 1024)
        last = xrow - (npc - 1) * 1024  # bytes of the last piece
        items = [(p, r, pc) for p in range(P) for r in range(TILE) for pc in range(npc)]
        tb = b.mul(ps, _c(b, TILE * P))
        cm1 = b.sub(cnt, _c(b, 1))
        WD = 4 if k.xsync == "cnt" else WAVES  # DMA-issuing waves
        nslot = -(-len(items) // WD)
        nvr = [b.smax(b.smin(b.sub(cnt, b.add(tb, _c(b, p * TILE))), _c(b, TILE)), _c(b, 0)) for p in range(P)]  # valid rows
        # every token id first: a ds_read after an LDS DMA waits for the DMA (LLVM's alias rule)
        plan = []
        for s in range(nslot):
            ii = b.add(_c(b, s * WD), self.wave)  # item index = s * WD + wave (uniform per wave)
            iic = b.smin(ii, _c(b, len(items) - 1))
            p = b.div(iic, _c(b, TILE * npc))
            rem = b.sub(iic, b.mul(p, _c(b, TILE * npc)))
            r = b.div(rem, _c(b, npc))
            pc = b.sub(rem, b.mul(r, _c(b, npc)))
            li = b.smax(b.smin(b.add(tb, b.add(b.mul(p, _c(b, TILE)), r)), cm1), _c(b, 0))
            tok = b.readfirstlane(b.land(self.lds_i(self.lst, b.add(off, li)), _c(b, 255)))
            ok = b.cmp_lt(ii, _c(b, len(items)))
            if WD < WAVES:
                ok = b.land(ok, b.cmp_lt(self.wave, _c(b, WD)))
            if k.xv:
                nv = nvr[0]
                for p_ in range(1, P):
                    nv = b.select(b.cmp_eq(p, _c(b, p_)), nvr[p_], nv)
                ok = b.land(ok, b.cmp_lt(r, nv))
            plan.append((ok, p, r, pc, tok))
        xs_plan = []
        nsd = P * TILE * 20
        if k.xst == "dma":
            nins = -(-nsd // 64)
            for s in range(-(-nins // WD)):
                ii = b.add(_c(b, s * WD), self.wave)
                di = b.add(b.mul(ii, _c(b, 64)), self.lane)
                pr_ = b.div(di, _c(b, 20))
                d = b.sub(di, b.mul(pr_, _c(b, 20)))
                p = b.div(pr_, _c(b, TILE))
                r = b.sub(pr_, b.mul(p, _c(b, TILE)))
                li = b.smax(b.smin(b.add(tb, b.add(b.mul(p, _c(b, TILE)), r)), cm1), _c(b, 0))
                tok = b.land(self.lds_i(self.lst, b.add(off, b.smin(li, _c(b, SLOTS * k.bucket - 1)))), _c(b, 255))
                ok = b.cmp_lt(ii, _c(b, nins))
                if WD < WAVES:
                    ok = b.land(ok, b.cmp_lt(self.wave, _c(b, WD)))
                lok = b.cmp_lt(di, _c(b, nsd))
                if k.xv:
                    nv = nvr[0]
                    for p_ in range(1, P):
                        nv = b.select(b.cmp_eq(p, _c(b, p_)), nvr[p_], nv)
                    lok = b.land(lok, b.cmp_lt(r, nv))
                xs_plan.append((ok, di, d, tok, lok))
        for ok, p, r, pc, tok in plan:
            with b.scf_if(ok):
                src = b.add(b.mul(tok, _c(b, xrow)), b.add(b.mul(pc, _c(b, 1024)), b.mul(self.lane, _c(b, 16))))
                dst = b.add(b.mul(b.add(b.mul(p, _c(b, TILE)), r), _c(b, self.XST)), b.mul(pc, _c(b, 1024)))
                full = b.cmp_lt(pc, _c(b, npc - 1))
                lane_ok = b.lor(full, b.cmp_lt(self.lane, _c(b, last // 16)))
                if k.xst == "dma":
                    la = b.smem_ptr_add(self.xl_a, _i64(b, b.readfirstlane(dst)))
                    with b.scf_if(lane_ok):
                        b.async_buffer_load_lds_addr(self.x_rs, la, src, _c(b, 0), dwords=4, coherency=0)
                else:
                    with b.scf_if(lane_ok):
                        v = b.buffer_load_cp(self.x_rs, src, _c(b, 0), dwords=4)
                        b.smem_store_vN(self.xl, [b.lshr(b.add(dst, b.mul(self.lane, _c(b, 16))), _c(b, 2))], v, 4)
        # scales: the pass's 16 P rows of 80 bytes (20 dwords each), contiguous in LDS: dword i of the block is row
        # i / 20, dword i % 20 (a gather by DMA, 64 dwords per instruction, or loads + ds_write)
        if k.xst == "dma":
            for ok, di, d, tok, lok in xs_plan:
                with b.scf_if(ok):
                    src = b.select(lok, b.add(b.mul(tok, _c(b, dm["xsrow"])), b.mul(d, _c(b, 4))), _c(b, OOB))
                    la = b.smem_ptr_add(self.xsl_a, _i64(b, b.readfirstlane(b.mul(b.div(di, _c(b, 64)), _c(b, 256)))))
                    with b.scf_if(lok):
                        b.async_buffer_load_lds_addr(self.xs_rs, la, src, _c(b, 0), dwords=1, coherency=0)
        else:
            with b.scf_if(b.cmp_lt(self.tid, _c(b, nsd))):
                pr_ = b.div(self.tid, _c(b, 20))
                d = b.sub(self.tid, b.mul(pr_, _c(b, 20)))
                p = b.div(pr_, _c(b, TILE))
                r = b.sub(pr_, b.mul(p, _c(b, TILE)))
                li = b.smin(b.add(tb, b.add(b.mul(p, _c(b, TILE)), r)), cm1)
                tok = b.land(self.lds_i(self.lst, b.add(off, li)), _c(b, 255))
                v = b.buffer_load_cp(self.xs_rs, b.add(b.mul(tok, _c(b, dm["xsrow"])), b.mul(d, _c(b, 4))), _c(b, 0), dwords=1)
                b.smem_store_vN(self.xsl, [b.add(b.mul(pr_, _c(b, self.XSS)), b.mul(d, _c(b, 4)))], b.vec_bitcast(b.vec_pack([v], I32), VectorType(I8, 4)), 4)

    def xs_dw(self, row: Value, dw: Value | int) -> Value:
        """Dword ``dw`` of staged x-scale row ``row`` (sop layout: byte 16 grp + 4 g + j)."""
        b = self.b
        d = _c(b, dw) if isinstance(dw, int) else dw
        v = b.smem_load_vN(self.xsl, b.add(b.mul(row, _c(b, self.XSS)), b.mul(d, _c(b, 4))), dtype=I8, n=4)
        return b.vec_extract(b.vec_bitcast(v, VectorType(I32, 1)), 0)

    def xs_u16(self, row: Value, byte: Value) -> Value:
        """The 16 bits at byte ``byte`` (even) of staged x-scale row ``row``, zero-extended."""
        b = self.b
        v = b.smem_load_vN(self.xsl, b.add(b.mul(row, _c(b, self.XSS)), byte), dtype=I8, n=2)
        if self.k.cob == "direct":  # the direct compiler has no i16 vector extract: two bytes
            lo, hi = (b.land(b.zext(b.vec_extract(v, i), I32), _c(b, 255)) for i in range(2))
            return b.lor(lo, b.shl(hi, _c(b, 8)))
        return b.zext(b.vec_extract(b.vec_bitcast(v, VectorType(I16, 1)), 0), I32)

    def gu_rt(self, a: Value, t: int | Value) -> Value:
        """gate_up 16-row tile of atom ``a``: t 0, 1 gate rows 32 a .. ; t 2, 3 up rows I + 32 a .."""
        b, k = self.b, self.k
        if isinstance(t, int):
            return b.add(b.mul(a, _c(b, 2)), _c(b, (t % 2) + (k.nrt if t >= 2 else 0)))
        up = b.cmp_ge(t, _c(b, 2))
        return b.add(b.add(b.mul(a, _c(b, 2)), b.land(t, _c(b, 1))), b.select(up, _c(b, k.nrt), _c(b, 0)))

    def gu_stream(self, w_rs, s_rs, rt: Value, q0: int, nq: int):
        """One wave's gate_up chunk stream for 16-row tile ``rt``, chunks ``q0 .. q0 + nq - 1``: returns the per-pass
        accumulators of the K halves (chunks < 10 and >= 10)."""
        b, k = self.b, self.k
        P = k.tpp
        fb = k.act
        dep = min(k.depth, nq)

        def issue(q: int):
            coff = b.add(b.add(b.mul(b.add(b.mul(rt, _c(b, NQG)), _c(b, q)), _c(b, 1024)), b.mul(self.l16, _c(b, 64))), b.mul(self.g, _c(b, 16)))
            wv = b.buffer_load_cp(w_rs, coff, _c(b, 0), dwords=4, aux=self.wcp)
            sv = None
            if q % 4 == 0 or q == q0:
                sblk = b.add(b.mul(rt, _c(b, NQG // 4)), _c(b, q // 4))
                soff = b.add(b.mul(b.add(b.mul(sblk, _c(b, 16)), self.l16), _c(b, 16)), b.mul(self.g, _c(b, 4)))
                sv = b.buffer_load_cp(s_rs, soff, _c(b, 0), dwords=1, aux=self.wcp)
            return wv, sv

        def bread(q: int):
            out = []
            for p in range(P):
                row = b.add(_c(b, p * TILE), self.l16)
                rb = b.mul(row, _c(b, self.XST))
                if fb == "fp8":
                    parts = [b.smem_load_vN(self.xl, b.lshr(b.add(b.add(rb, _c(b, q * 128 + 64 * hh)), b.mul(self.g, _c(b, 16))), _c(b, 2)),
                                            dtype=I32, n=4) for hh in range(2)]
                    bv = b.vec_concat(parts[0], parts[1])
                else:
                    bv = b.smem_load_vN(self.xl, b.lshr(b.add(b.add(rb, _c(b, q * 64)), b.mul(self.g, _c(b, 16))), _c(b, 2)), dtype=I32, n=4)
                xs = None
                if q % 4 == 0 or q == q0:
                    xs = self.xs_dw(row, b.add(_c(b, 4 * (q // 4)), self.g))
                out.append((bv, xs))
            return out

        accs = [[b.zero_vec_f32(4) for _ in range(P)] for _ in range(2)]
        ring = [issue(q0 + i) for i in range(dep)]
        bnext = bread(q0) if k.lds1 else None
        sc_w = None
        sc_x = [None] * P
        for i in range(nq):
            q = q0 + i
            b.sched_barrier(0)
            if i + dep < nq:
                ring.append(issue(q + dep))
            b.sched_barrier(0)
            wv, sv = ring.pop(0)
            if k.lds1:
                bcur = bnext
                if i + 1 < nq:
                    bnext = bread(q + 1)
            else:
                bcur = bread(q)
            if sv is not None:
                sc_w = sv
            hh = 0 if q < NQG // 2 else 1
            for p in range(P):
                bv, xs = bcur[p]
                if xs is not None:
                    sc_x[p] = xs
                sel = (q + 1) % 4 if k.mutant == "sopsel" else q % 4
                if self.dg("nogu"):
                    continue
                accs[hh][p] = b.mfma_scale_mx(wv, bv, accs[hh][p], sc_w, sc_x[p], shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                              opsel_a=sel, opsel_b=q % 4)
        b.sched_barrier(0)
        return accs

    def red_put(self, buf: Value, entry: Value, p: int, v: Value) -> None:
        b = self.b
        idx = b.add(b.mul(b.add(b.mul(b.add(b.mul(buf, _c(b, 8)), entry), _c(b, self.k.tpp)), _c(b, p)), _c(b, 256)), b.mul(self.lane, _c(b, 4)))
        b.smem_store_vN(self.red, [idx], b.vec_bitcast(v, VectorType(I32, 4)), 4)

    def red_get(self, buf: Value, entry: int | Value, p: int) -> Value:
        b = self.b
        ent = _c(b, entry) if isinstance(entry, int) else entry
        idx = b.add(b.mul(b.add(b.mul(b.add(b.mul(buf, _c(b, 8)), ent), _c(b, self.k.tpp)), _c(b, p)), _c(b, 256)), b.mul(self.lane, _c(b, 4)))
        return b.vec_bitcast(b.smem_load_vN(self.red, idx, dtype=I32, n=4), VectorType(F32, 4))

    def vadd(self, x: Value, y: Value) -> Value:
        b = self.b
        return b.vec_pack([b.fadd(b.vec_extract(x, i), b.vec_extract(y, i)) for i in range(4)], F32)

    def epilogue(self, acc, a: Value, p: int, hp_base: Value, valid: Value, glob: bool = False) -> None:
        """h = silu(g) u of atom ``a`` (32 columns) for pass tile ``p``: requantized per (token, 32 columns) into the
        LDS h rows / scale bytes (and, ``hdump``, to the global numerics twin at list position ``hp_base + l16``)."""
        b, k, dm = self.b, self.k, self.dm
        hv = []
        for t in range(2):
            for i in range(4):
                gv = b.vec_extract(acc[t], i)
                uv = b.vec_extract(acc[2 + t], i)
                den = b.fadd(_f(b, 1.0), b.exp2(b.fmul(b.fneg(gv), _f(b, L2E))))
                hv.append(b.select(valid, b.fmul(b.fdiv(gv, den), uv), _f(b, 0.0)))  # invalid token columns: 0 (staged x may be stale garbage)
        amax = b.fabs(hv[0])
        for v in hv[1:]:
            amax = b.fmax(amax, b.fabs(v))
        for sw in ("p16", "p32"):
            x_ = b.bitcast(amax, I32)
            lo, hi = b.permlane16_swap(x_, x_) if sw == "p16" else b.permlane32_swap(x_, x_)
            amax = b.fmax(b.bitcast(lo, F32), b.bitcast(hi, F32))
        row = b.add(_c(b, p * TILE), self.l16)
        hrow = dm["hrow"]
        if k.act == "fp8":
            ex, scale = Q.e8m0_fp8(b, amax)
            w0 = Q.pack_fp8x4(b, *hv[0:4], scale)
            w1 = Q.pack_fp8x4(b, *hv[4:8], scale)
            if not glob:
                base = b.add(b.mul(row, _c(b, self.HST)), b.add(b.mul(a, _c(b, 32)), b.mul(self.g, _c(b, 4))))
                self.lds_put(self.hl, b.lshr(base, _c(b, 2)), w0)
                self.lds_put(self.hl, b.lshr(b.add(base, _c(b, 16)), _c(b, 2)), w1)
            if k.hdump or glob:
                gx = CPOL_SC1 if glob else 0
                gb = b.add(b.mul(b.add(hp_base, self.l16), _c(b, hrow)), b.add(b.mul(a, _c(b, 32)), b.mul(self.g, _c(b, 4))))
                b.buffer_store_cp(w0, self.hd_rs, b.select(valid, gb, _c(b, OOB)), _c(b, 0), aux=gx)
                b.buffer_store_cp(w1, self.hd_rs, b.select(valid, b.add(gb, _c(b, 16)), _c(b, OOB)), _c(b, 0), aux=gx)
        else:
            ex, scale = _rceil_fp4(b, amax)
            halves = [_pack_fp4x4(b, hv[4 * t : 4 * t + 4], scale) for t in range(2)]
            even = b.cmp_eq(b.land(self.g, _c(b, 1)), _c(b, 0))
            for t in range(2):
                x_ = halves[t]
                lo, hi = b.permlane16_swap(x_, x_)
                wd = b.lor(x_, b.shl(hi, _c(b, 16)))
                o = b.add(b.mul(a, _c(b, 16)), b.add(_c(b, 8 * t), b.mul(b.lshr(self.g, _c(b, 1)), _c(b, 4))))
                if not glob:
                    with b.scf_if(even):
                        self.lds_put(self.hl, b.lshr(b.add(b.mul(row, _c(b, self.HST)), o), _c(b, 2)), wd)
                if k.hdump or glob:
                    gb = b.add(b.mul(b.add(hp_base, self.l16), _c(b, hrow)), o)
                    b.buffer_store_cp(wd, self.hd_rs, b.select(b.land(valid, even), gb, _c(b, OOB)), _c(b, 0),
                                      aux=CPOL_SC1 if glob else 0)
        # scale byte [p][tok][g = a % 4][chunk = a / 4]
        g0 = b.cmp_eq(self.g, _c(b, 0))
        sidx = b.add(b.mul(b.add(b.mul(row, _c(b, 4)), b.land(a, _c(b, 3))), _c(b, 8)), b.lshr(a, _c(b, 2)))
        if not glob:
            with b.scf_if(g0):
                b.smem_store_vN(self.hsl, [sidx], b.trunc(ex, I8), 1)
        if k.hdump or glob:
            so = b.mul(b.add(b.mul(b.add(hp_base, self.l16), _c(b, k.na)), a), _c(b, 4))
            b.buffer_store_cp(ex, self.hds_rs, b.select(b.land(valid, g0), so, _c(b, OOB)), _c(b, 0),
                              aux=CPOL_SC1 if glob else 0)

    # ------------------------------------------------------------------ one segment
    def segment(self, layer: Value, it: Value, a0: Value, a1: Value, jseg: Value, nseg: Value) -> None:
        b, k, dm = self.b, self.k, self.dm
        P = k.tpp
        e, ps, off, cnt = self.item_info(it)
        el = _i64(b, e)
        nul = k.null == "oob"
        wa = b.add(b.add(self.a["wgu"], b.mul(_i64(b, layer), self.lgu)), b.mul(el, b.const_i64(dm["gub"])))
        sa = b.add(b.add(self.a["sgu"], b.mul(_i64(b, layer), self.lgs)), b.mul(el, b.const_i64(dm["gus"])))
        gw_rs = _rsrc(b, wa, _c(b, 0 if nul else dm["gub"]))
        gs_rs = _rsrc(b, sa, _c(b, 0 if nul else dm["gus"]))
        da = b.add(b.add(self.a["wdn"], b.mul(_i64(b, layer), self.ldn)), b.mul(el, b.const_i64(dm["dnb"])))
        dsa = b.add(b.add(self.a["sdn"], b.mul(_i64(b, layer), self.lds_)), b.mul(el, b.const_i64(dm["dns"])))
        dw_rs = _rsrc(b, da, _c(b, 0 if nul else dm["dnb"]))
        ds_rs = _rsrc(b, dsa, _c(b, 0 if nul else dm["dns"]))
        # tokens of this pass: per tile p, lane l16 -> list index 16 (ps P + p) + l16
        tb = b.mul(ps, _c(b, TILE * P))
        cm1 = b.sub(cnt, _c(b, 1))
        toks = []
        for p in range(P):
            li = b.add(b.add(tb, _c(b, p * TILE)), self.l16)
            valid = b.cmp_lt(li, cnt)
            pos = b.add(off, b.smin(li, cm1))
            ent = self.lds_i(self.lst, pos)
            toks.append(dict(valid=valid, tok=b.land(ent, _c(b, 255)), slot=b.lshr(ent, _c(b, 8)), wgt=self.lds_f(self.lw, pos),
                             hp=b.add(off, b.add(tb, _c(b, p * TILE)))))
        self.stage_x(off, cnt, ps)
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        n = b.sub(a1, a0)
        npair = b.lshr(n, _c(b, 1))
        sidx = b.lshr(self.wave, _c(b, 2))  # pair mode: atom of the pair; lone mode: K half
        t = b.land(self.wave, _c(b, 3))
        if not self.dg("nogu"):
            with b.scf_for(_c(b, 0), npair, _c(b, 1), iv_name="pairi") as i:
                i = b.readfirstlane(i)
                buf = b.land(i, _c(b, 1))
                a = b.readfirstlane(b.add(b.add(a0, b.mul(i, _c(b, 2))), sidx))
                accs = self.gu_stream(gw_rs, gs_rs, b.readfirstlane(self.gu_rt(a, t)), 0, NQG)
                for p in range(P):
                    v = accs[0][p] if k.mutant == "half" else self.vadd(accs[0][p], accs[1][p])
                    self.red_put(buf, self.wave, p, v)
                b.sync_lds_only()
                for s_ in range(2):
                    for p in range(P):
                        jw = s_ * P + p
                        with b.scf_if(b.cmp_eq(self.wave, _c(b, jw))):
                            aa = b.readfirstlane(b.add(b.add(a0, b.mul(i, _c(b, 2))), _c(b, s_)))
                            acc = [self.red_get(buf, 4 * s_ + tt, p) for tt in range(4)]
                            self.epilogue(acc, aa, p, toks[p]["hp"], toks[p]["valid"])
            # lone atom: 8 waves x (tile, K half), summed in the epilogue in half order
            lone = b.cmp_ne(b.land(n, _c(b, 1)), _c(b, 0))
            with b.scf_if(lone):
                buf = b.land(npair, _c(b, 1))
                a = b.readfirstlane(b.sub(a1, _c(b, 1)))
                rt = b.readfirstlane(self.gu_rt(a, t))
                # two static streams (half 0 / half 1), one taken per wave (uniform)
                with b.scf_if(b.cmp_eq(sidx, _c(b, 0))):
                    accs = self.gu_stream(gw_rs, gs_rs, rt, 0, NQG // 2)
                    for p in range(P):
                        self.red_put(buf, self.wave, p, accs[0][p])
                with b.scf_if(b.cmp_ne(sidx, _c(b, 0))):
                    accs = self.gu_stream(gw_rs, gs_rs, rt, NQG // 2, NQG // 2)
                    for p in range(P):
                        self.red_put(buf, self.wave, p, accs[1][p])
                b.sync_lds_only()
                for p in range(P):
                    with b.scf_if(b.cmp_eq(self.wave, _c(b, p))):
                        acc = []
                        for tt in range(4):
                            v0 = self.red_get(buf, tt, p)
                            acc.append(v0 if k.mutant == "lone" else self.vadd(v0, self.red_get(buf, 4 + tt, p)))
                        self.epilogue(acc, a, p, toks[p]["hp"], toks[p]["valid"])
        b.sync_lds_only()  # h complete
        if not self.dg("nodn"):
            self.down(dw_rs, ds_rs, a0, a1, toks, jseg)
        # desc[token, slot] = segments of this item (written by segment 0)
        with b.scf_if(b.land(b.cmp_eq(jseg, _c(b, 0)), b.land(b.cmp_eq(self.wave, _c(b, 0)), b.cmp_eq(self.g, _c(b, 0))))):
            for p in range(P):
                tk = toks[p]
                o = b.mul(b.add(b.mul(tk["tok"], _c(b, DSTR)), tk["slot"]), _c(b, 4))
                b.buffer_store_cp(nseg, self.desc_rs, b.select(tk["valid"], o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        b.sync_lds_only()  # x / h / red reuse by the next segment

    def down(self, dw_rs, ds_rs, a0: Value, a1: Value, toks, jseg: Value) -> None:
        """Down projection over the segment's K columns [32 a0, 32 a1): wave w owns the 20 output tiles 20 w .. 20 w + 19;
        one static (tile, chunk) stream per chunk count ``n_c`` (dispatched on the uniform ``n_c``)."""
        b, k = self.b, self.k
        c_lo = b.readfirstlane(b.lshr(a0, _c(b, 2)))
        c_hi = b.readfirstlane(b.lshr(b.add(a1, _c(b, 3)), _c(b, 2)))
        nc = b.sub(c_hi, c_lo)
        for v in range(1, k.nqd + 1):
            with b.scf_if(b.cmp_eq(nc, _c(b, v))):
                self.down_v(dw_rs, ds_rs, a0, a1, c_lo, v, toks, jseg)

    def down_v(self, dw_rs, ds_rs, a0, a1, c_lo, nc: int, toks, jseg) -> None:
        b, k, dm = self.b, self.k, self.dm
        P = k.tpp
        fb = k.act
        rtb = k.inter * TILE // 2
        ntl = H // TILE // WAVES  # 20 output tiles per wave
        rt0 = b.mul(self.wave, _c(b, ntl))
        # B operand (h) and h scales: the same for every tile of the wave
        sh = b.mul(c_lo, _c(b, 8))
        bh, hsc = [], []
        for p in range(P):
            row = b.add(_c(b, p * TILE), self.l16)
            rb = b.mul(row, _c(b, self.HST))
            per = []
            for kk in range(nc):
                c = b.add(c_lo, _c(b, kk))
                # K bytes past I (the ragged last chunk of TP2) read zeros, not the next LDS row
                z4 = b.vec_pack([_c(b, 0)] * 4, I32)
                if fb == "fp8":
                    parts = []
                    for hh in range(2):
                        kb = b.add(b.add(b.mul(c, _c(b, 128)), _c(b, 64 * hh)), b.mul(self.g, _c(b, 16)))
                        v_ = b.smem_load_vN(self.hl, b.lshr(b.add(rb, b.smin(kb, _c(b, dm["hrow"]))), _c(b, 2)), dtype=I32, n=4)
                        parts.append(b.select(b.cmp_lt(kb, _c(b, dm["hrow"])), v_, z4) if k.inter % CHUNK else v_)
                    per.append(b.vec_concat(parts[0], parts[1]))
                else:
                    kb = b.add(b.mul(c, _c(b, 64)), b.mul(self.g, _c(b, 16)))
                    v_ = b.smem_load_vN(self.hl, b.lshr(b.add(rb, b.smin(kb, _c(b, dm["hrow"]))), _c(b, 2)), dtype=I32, n=4)
                    per.append(b.select(b.cmp_lt(kb, _c(b, dm["hrow"])), v_, z4) if k.inter % CHUNK else v_)
            bh.append(per)
            q = b.vec_bitcast(b.smem_load_vN(self.hsl, b.mul(b.add(b.mul(row, _c(b, 4)), self.g), _c(b, 8)), dtype=I8, n=8), VectorType(I32, 2))
            q64 = b.lor(_i64(b, b.vec_extract(q, 0)), b.shl(_i64(b, b.vec_extract(q, 1)), b.const_i64(32)))
            hsh = b.lshr(q64, _i64(b, sh))
            if k.mutant == "hsc":
                hsh = q64
            hsc.append((b.trunc(hsh, I32), b.trunc(b.lshr(hsh, b.const_i64(32)), I32)))
        steps = [(j, kk) for j in range(ntl) for kk in range(nc)]
        dep = min(k.ddepth, len(steps))

        def issue(si: int):
            j, kk = steps[si]
            rt = b.add(rt0, _c(b, j))
            c = b.add(c_lo, _c(b, kk))
            blk = b.add(b.mul(c, _c(b, 4)), self.g)
            own = b.land(b.cmp_ge(blk, a0), b.cmp_lt(blk, a1))
            if k.mutant == "mask":
                own = b.cmp_lt(blk, _c(b, k.na))
            if k.inter % CHUNK:
                last = b.cmp_eq(c, _c(b, k.nqd - 1))
                cbt = b.select(last, _c(b, k.chunk_bytes(k.nqd - 1)), _c(b, 64))
            else:
                cbt = _c(b, 64)
            coff = b.add(b.add(b.mul(rt, _c(b, rtb)), b.mul(c, _c(b, 1024))), b.add(b.mul(self.l16, cbt), b.mul(self.g, _c(b, 16))))
            wv = b.buffer_load_cp(dw_rs, b.select(own, coff, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.wcp)
            sv = None
            if kk == 0:
                soff = b.mul(b.add(b.mul(b.add(b.mul(rt, _c(b, TILE)), self.l16), _c(b, 4)), self.g), _c(b, 4 * k.ngrp))
                sv = b.buffer_load_cp(ds_rs, soff, _c(b, 0), dwords=k.ngrp, aux=self.wcp)
            return wv, sv

        ring = [issue(i) for i in range(dep)]
        acc = None
        sw = None
        for si, (j, kk) in enumerate(steps):
            b.sched_barrier(0)
            if si + dep < len(steps):
                ring.append(issue(si + dep))
            b.sched_barrier(0)
            wv, sv = ring.pop(0)
            if kk == 0:
                acc = [b.zero_vec_f32(4) for _ in range(P)]
                if k.ngrp == 1:
                    sw = (b.lshr(sv, sh), None)
                else:
                    s64 = b.lor(_i64(b, b.vec_extract(sv, 0)), b.shl(_i64(b, b.vec_extract(sv, 1)), b.const_i64(32)))
                    s64 = b.lshr(s64, _i64(b, sh))
                    sw = (b.trunc(s64, I32), b.trunc(b.lshr(s64, b.const_i64(32)), I32))
            sa_, oa = (sw[0], kk) if kk < 4 else (sw[1], kk - 4)
            for p in range(P):
                sb_ = hsc[p][0] if kk < 4 else hsc[p][1]
                acc[p] = b.mfma_scale_mx(wv, bh[p][kk], acc[p], sa_, sb_, shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                         opsel_a=oa, opsel_b=oa)
            if kk == nc - 1:
                rt = b.add(rt0, _c(b, j))
                for p in range(P):
                    tk = toks[p]
                    v = acc[p]
                    prow = b.add(b.mul(b.add(b.mul(tk["tok"], _c(b, SLOTS)), tk["slot"]), _c(b, MAXSEG)), jseg)
                    o = b.mul(b.add(b.mul(prow, _c(b, H)), b.add(b.mul(rt, _c(b, TILE)), b.mul(self.g, _c(b, 4)))), _c(b, 4))
                    b.buffer_store_cp(b.vec_bitcast(v, VectorType(I32, 4)), self.part_rs, b.select(tk["valid"], o, _c(b, OOB)), _c(b, 0),
                                      aux=CPOL_SC1)
        b.sched_barrier(0)

    # ------------------------------------------------------------------ deal + segments
    # ------------------------------------------------------------------ pipe=stream (v2)
    def wl_find(self, nit: Value, x: Value) -> Value:
        """Largest item k in [0, nit) with ipre[k] <= x (binary search, uniform)."""
        b = self.b
        lo = _c(b, 0)
        hi = nit
        steps = max(1, (self.k.it_max).bit_length())
        for _ in range(steps):
            mid = b.readfirstlane(b.lshr(b.add(lo, hi), _c(b, 1)))
            v = b.readfirstlane(self.lds_i(self.ipre, mid))
            big = b.cmp_gt(b.sub(hi, lo), _c(b, 1))
            le = b.cmp_le(v, x)
            gt = b.cmp_gt(v, x)
            nlo = b.readfirstlane(b.select(b.land(big, le), mid, lo))
            hi = b.readfirstlane(b.select(b.land(big, gt), mid, hi))
            lo = nlo
        return lo

    def wl_atoms(self, it: Value, pk: Value, x: Value) -> Value:
        """Atoms of item ``it`` (line start ``pk``) whose start unit is below ``x`` (uniform)."""
        b, k = self.b, self.k
        wa, wc, ws, wt = k.wlw
        tk = self.wl_tokens(it)
        o = b.sub(b.sub(x, pk), b.add(_c(b, ws + wc), b.mul(tk, _c(b, wt))))  # units past atom 0's start
        n = _c(b, 0)
        for a in range(k.na):
            n = b.add(n, b.select(b.cmp_lt(_c(b, a * wa + (a // 4) * wc), o), _c(b, 1), _c(b, 0)))
        return b.readfirstlane(n)

    def wl_tokens(self, it: Value) -> Value:
        b, k = self.b, self.k
        v = b.readfirstlane(self.lds_i(self.itl, it))
        e = b.land(v, _c(b, 1023))
        ps = b.lshr(v, _c(b, 10))
        cnt = b.lshr(b.readfirstlane(self.lds_i(self.eo, e)), _c(b, 16))
        tp_ = b.readfirstlane(b.select(b.cmp_eq(e, _c(b, SH)), _c(b, k.shsx), _c(b, TILE * k.tpp)))
        return b.readfirstlane(b.smin(b.sub(cnt, b.mul(ps, tp_)), tp_))

    def wl_cnum(self, W: Value) -> Value:
        b, k = self.b, self.k
        umax = k.wl_item(TILE * k.tpp)
        return b.readfirstlane(b.smax(b.smin(_c(b, GRID), b.div(b.mul(W, _c(b, MAXSEG - 1)),
                                                               _c(b, umax + 1 + 2 * (MAXSEG - 1) * k.snap))), _c(b, 1)))

    def wl_lpos(self, C: Value, x: Value) -> Value:
        """The line position whose range holds unit ``x``: the largest L < C with cuts[L] <= x (uniform)."""
        b = self.b
        lo = _c(b, 0)
        hi = C
        for _ in range(GRID.bit_length()):
            mid = b.readfirstlane(b.lshr(b.add(lo, hi), _c(b, 1)))
            v = b.readfirstlane(self.lds_i(self.cuts, mid))
            big = b.cmp_gt(b.sub(hi, lo), _c(b, 1))
            nlo = b.readfirstlane(b.select(b.land(big, b.cmp_le(v, x)), mid, lo))
            hi = b.readfirstlane(b.select(b.land(big, b.cmp_gt(v, x)), mid, hi))
            lo = nlo
        return lo

    def deal_line_wl(self, nit: Value):
        b, k = self.b, self.k
        W = b.readfirstlane(self.lds_i(self.ipre, nit))
        C = self.wl_cnum(W)
        if k.xh:
            L = b.add(b.mul(b.land(self.cta, _c(b, XCDS - 1)), _c(b, GRID // XCDS)), b.lshr(self.cta, _c(b, 3)))
        else:
            L = self.cta
        L = b.readfirstlane(L)
        act = b.cmp_lt(L, C)
        Lc = b.readfirstlane(b.smin(L, C))
        lo = b.readfirstlane(self.lds_i(self.cuts, Lc))
        hi = b.readfirstlane(b.select(act, self.lds_i(self.cuts, b.add(Lc, _c(b, 1))), lo))
        kf = self.wl_find(nit, lo)
        # the range may start past kf's last atom start
        pf = b.readfirstlane(self.lds_i(self.ipre, kf))
        kf = b.readfirstlane(b.select(b.cmp_ge(self.wl_atoms(kf, pf, lo), _c(b, k.na)), b.add(kf, _c(b, 1)), kf))
        kl = b.readfirstlane(b.add(self.wl_find(nit, b.sub(hi, _c(b, 1))), _c(b, 1)))
        pl = b.readfirstlane(self.lds_i(self.ipre, b.sub(kl, _c(b, 1))))
        kl = b.readfirstlane(b.select(b.cmp_eq(self.wl_atoms(b.sub(kl, _c(b, 1)), pl, hi), _c(b, 0)), b.sub(kl, _c(b, 1)), kl))
        kl = b.readfirstlane(b.select(b.cmp_gt(hi, lo), b.smax(kl, kf), kf))
        return dict(A=W, C=C, L=L, lo=lo, hi=hi, kf=kf, kl=kl, wl=True)

    def deal_line_rr(self, nit: Value):
        """rr / hy: this CTA's whole items L, L + 256, .. below k0, then (hy) its atom-line range over [k0, nit)."""
        b, k = self.b, self.k
        NA = k.na
        if k.xh:
            L = b.add(b.mul(b.land(self.cta, _c(b, XCDS - 1)), _c(b, GRID // XCDS)), b.lshr(self.cta, _c(b, 3)))
        else:
            L = self.cta
        L = b.readfirstlane(L)
        if k.deal == "hy":
            R = b.readfirstlane(b.div(nit, _c(b, GRID)))
        else:
            R = b.readfirstlane(b.div(b.add(nit, _c(b, GRID - 1)), _c(b, GRID)))
        k0 = b.readfirstlane(b.smin(b.mul(R, _c(b, GRID)), nit))
        nw = b.readfirstlane(b.div(b.add(b.sub(k0, L), _c(b, GRID - 1)), _c(b, GRID)))  # whole items L + j 256 < k0
        nw = b.readfirstlane(b.select(b.cmp_lt(L, k0), nw, _c(b, 0)))
        A = b.readfirstlane(b.mul(b.sub(nit, k0), _c(b, NA)))
        C = b.readfirstlane(b.smax(b.smin(_c(b, GRID), b.div(A, _c(b, k.m0))), _c(b, 1)))
        act = b.land(b.cmp_lt(L, C), b.cmp_gt(A, _c(b, 0)))
        Ad = b.smax(A, _c(b, 1))
        lo = b.readfirstlane(b.select(act, b.div(b.mul(L, A), C), _c(b, 0)))
        hi = b.readfirstlane(b.select(act, b.div(b.mul(b.add(L, _c(b, 1)), A), C), _c(b, 0)))
        kf = b.readfirstlane(b.add(k0, b.div(lo, _c(b, NA))))
        kl = b.readfirstlane(b.select(b.cmp_gt(hi, lo), b.add(k0, b.add(b.div(b.sub(hi, _c(b, 1)), _c(b, NA)), _c(b, 1))), kf))
        return dict(A=Ad, C=C, L=L, lo=lo, hi=hi, kf=kf, kl=kl, k0=k0, nw=nw, nv=b.readfirstlane(b.add(nw, b.sub(kl, kf))), rr=True)

    def div_small(self, x: Value, d: Value) -> Value:
        """x / d for a uniform d in 1 .. 4 (constant divisions selected; no runtime integer division)."""
        b = self.b
        if not isinstance(d, Value):
            return b.div(x, _c(b, d))
        q3 = b.div(x, _c(b, 3))
        return b.select(b.cmp_eq(d, _c(b, 1)), x, b.select(b.cmp_eq(d, _c(b, 2)), b.lshr(x, _c(b, 1)),
                                                            b.select(b.cmp_eq(d, _c(b, 3)), q3, b.lshr(x, _c(b, 2)))))

    def useg(self, layer: Value, u: Value, ln: dict) -> dict:
        """Segment of unit ``u`` (rr / dyn): item piece u % pc of item u // pc (or of rank u // pc, uord=heavy)."""
        b, k = self.b, self.k
        NA = k.na
        pcv = ln["pc"]  # int or (pc 0) a uniform value
        has = b.cmp_lt(u, ln["U"])
        u_in = u
        if k.shd:  # units past U0: the second halves of the shared-expert items
            sec = b.cmp_ge(u, ln["U0"])
            u = b.readfirstlane(b.select(sec, _c(b, 0), u))
        uc = b.readfirstlane(b.select(has, u, _c(b, 0)))
        if pcv == 1:
            r, pj, pcc = uc, _c(b, 0), _c(b, 1)
        elif isinstance(pcv, int):
            pcc = _c(b, pcv)
            r = b.readfirstlane(b.div(uc, pcc))
            pj = b.readfirstlane(b.sub(uc, b.mul(r, pcc)))
        else:  # pc 0: units below F whole, then pieces
            F = ln["F"]
            whole = b.cmp_lt(uc, F)
            v = b.sub(uc, F)
            q = self.div_small(b.smax(v, _c(b, 0)), pcv)
            r = b.readfirstlane(b.select(whole, uc, b.add(F, q)))
            pj = b.readfirstlane(b.select(whole, _c(b, 0), b.sub(v, b.mul(q, pcv))))
            pcc = b.readfirstlane(b.select(whole, _c(b, 1), pcv))
        it = b.readfirstlane(self.lds_i(self.perm, r)) if k.uord != "id" else r
        if k.pcal:  # pieces on down-chunk boundaries
            NC = -(-NA // 4)
            a0 = b.readfirstlane(b.smin(b.mul(self.div_small(b.mul(pj, _c(b, NC)), pcc), _c(b, 4)), _c(b, NA)))
            a1 = b.readfirstlane(b.smin(b.mul(self.div_small(b.mul(b.add(pj, _c(b, 1)), _c(b, NC)), pcc), _c(b, 4)), _c(b, NA)))
        else:
            a0 = b.readfirstlane(self.div_small(b.mul(pj, _c(b, NA)), pcc))
            a1 = b.readfirstlane(self.div_small(b.mul(b.add(pj, _c(b, 1)), _c(b, NA)), pcc))
        if k.shd:
            it2 = b.readfirstlane(b.add(ln["itsh"], b.sub(u_in, ln["U0"])))
            it = b.readfirstlane(b.select(sec, it2, it))
            a0 = b.readfirstlane(b.select(sec, _c(b, 0), a0))
            a1 = b.readfirstlane(b.select(sec, _c(b, NA), a1))
            pj = b.readfirstlane(b.select(sec, _c(b, 0), pj))
            pcc = b.readfirstlane(b.select(sec, _c(b, 1), pcc))
            first = b.land(ln["split"], b.cmp_ge(it, ln["itsh"]))
            dh = b.readfirstlane(b.select(sec, _c(b, 2), b.select(first, _c(b, 1), _c(b, 0))))
            S = self.seg_info(layer, it, has, ln, unit=(a0, a1, pj, pcc))
            S["dh"] = dh
            return S
        return self.seg_info(layer, it, has, ln, unit=(a0, a1, pj, pcc))

    def vseg(self, layer: Value, v: Value, ln: dict) -> dict:
        """Segment ``v`` of this CTA's sequence (rr / hy: whole items first)."""
        b = self.b
        if not ln.get("rr"):
            it = b.add(ln["kf"], v)
            return self.seg_info(layer, it, b.cmp_lt(it, ln["kl"]), ln)
        whole = b.cmp_lt(v, ln["nw"])
        it = b.readfirstlane(b.select(whole, b.add(ln["L"], b.mul(v, _c(b, GRID))), b.add(ln["kf"], b.sub(v, ln["nw"]))))
        return self.seg_info(layer, it, b.cmp_lt(v, ln["nv"]), ln, whole=whole)

    def deal_line(self, nit: Value):
        """The line: atoms A = items x NA cut into C ranges; this CTA's line position, range and item span (uniform)."""
        b, k = self.b, self.k
        if k.deal in ("rr", "hy"):
            return self.deal_line_rr(nit)
        if k.wl:
            return self.deal_line_wl(nit)
        U, ov = k.unit, k.ov
        A = b.readfirstlane(b.mul(nit, _c(b, U)))
        C = b.readfirstlane(b.smax(b.smin(_c(b, GRID), b.div(A, _c(b, k.m0))), _c(b, 1)))
        if k.xh:
            L = b.add(b.mul(b.land(self.cta, _c(b, XCDS - 1)), _c(b, GRID // XCDS)), b.lshr(self.cta, _c(b, 3)))
        else:
            L = self.cta
        L = b.readfirstlane(L)
        act = b.cmp_lt(L, C)
        lo = b.readfirstlane(b.div(b.mul(L, A), C))
        hi = b.readfirstlane(b.select(act, b.div(b.mul(b.add(L, _c(b, 1)), A), C), lo))
        kf = b.readfirstlane(b.div(lo, _c(b, U)))
        kl = b.readfirstlane(b.select(b.cmp_gt(hi, lo), b.add(b.div(b.sub(hi, _c(b, 1)), _c(b, U)), _c(b, 1)), kf))
        if ov:  # a range that ends inside the last item's virtual units holds no atom of it
            kl = b.readfirstlane(b.select(b.cmp_le(b.sub(b.sub(hi, b.mul(b.sub(kl, _c(b, 1)), _c(b, U))), _c(b, ov)), _c(b, 0)),
                                          b.smax(b.sub(kl, _c(b, 1)), kf), kl))
        return dict(A=A, C=C, L=L, lo=lo, hi=hi, kf=kf, kl=kl)

    def seg_info(self, layer: Value, it: Value, has: Value, ln: dict, whole: Value | None = None, unit=None) -> dict:
        """Segment of item ``it`` in this CTA's range; ``has`` false: a dummy (item 0, zero-length weights)."""
        b, k, dm = self.b, self.k, self.dm
        NA = k.na
        itc = b.readfirstlane(b.select(has, it, _c(b, 0)))
        e, ps, off, cnt = self.item_info(itc)
        A, C = ln["A"], ln["C"]
        if unit is not None:  # rr / dyn: the unit's fixed piece
            a0 = b.readfirstlane(b.select(has, unit[0], _c(b, 0)))
            a1 = b.readfirstlane(b.select(has, unit[1], _c(b, 1)))
            lf, ll = _c(b, 0), _c(b, 0)
        elif ln.get("wl"):
            wa, wc, ws, wt = k.wlw
            pk = b.readfirstlane(self.lds_i(self.ipre, itc))
            a0 = b.readfirstlane(b.select(has, self.wl_atoms(itc, pk, ln["lo"]), _c(b, 0)))
            a1 = b.readfirstlane(b.select(has, self.wl_atoms(itc, pk, ln["hi"]), _c(b, 1)))
            u0 = b.add(pk, b.add(_c(b, ws + wc), b.mul(self.wl_tokens(itc), _c(b, wt))))  # atom 0's start unit
            uN = b.add(u0, _c(b, (NA - 1) * wa + ((NA - 1) // 4) * wc))
            lf = self.wl_lpos(C, u0)
            ll = self.wl_lpos(C, uN)
        else:
            base = b.add(b.mul(b.sub(itc, ln.get("k0", _c(b, 0))), _c(b, k.unit)), _c(b, k.ov))  # the item's first atom on the line
            a0 = b.readfirstlane(b.select(has, b.smax(b.sub(ln["lo"], base), _c(b, 0)), _c(b, 0)))
            a1 = b.readfirstlane(b.select(has, b.smin(b.sub(ln["hi"], base), _c(b, NA)), _c(b, 1)))
            lf = b.div(b.sub(b.mul(b.add(base, _c(b, 1)), C), _c(b, 1)), A)
            ll = b.div(b.sub(b.mul(b.add(base, _c(b, NA)), C), _c(b, 1)), A)
        el = _i64(b, e)
        nul = k.null == "oob"

        def nb(n):
            return b.readfirstlane(b.select(has, _c(b, 0 if nul else n), _c(b, 0)))

        jseg = b.readfirstlane(b.sub(ln["L"], lf))
        nseg = b.readfirstlane(b.add(b.sub(ll, lf), _c(b, 1)))
        if unit is not None:
            jseg, nseg = unit[2], unit[3]
        if whole is not None:  # a whole item: all atoms, one segment
            a0 = b.readfirstlane(b.select(whole, _c(b, 0), a0))
            a1 = b.readfirstlane(b.select(b.land(whole, has), _c(b, NA), a1))
            jseg = b.readfirstlane(b.select(whole, _c(b, 0), jseg))
            nseg = b.readfirstlane(b.select(whole, _c(b, 1), nseg))
        S = dict(e=e, ps=ps, off=off, cnt=cnt, a0=a0, a1=a1, has=has, jseg=jseg, nseg=nseg,
                 gw_a=b.add(b.add(self.a["wgu"], b.mul(_i64(b, layer), self.lgu)), b.mul(el, b.const_i64(dm["gub"]))),
                 gs_a=b.add(b.add(self.a["sgu"], b.mul(_i64(b, layer), self.lgs)), b.mul(el, b.const_i64(dm["gus"]))),
                 dw_a=b.add(b.add(self.a["wdn"], b.mul(_i64(b, layer), self.ldn)), b.mul(el, b.const_i64(dm["dnb"]))),
                 ds_a=b.add(b.add(self.a["sdn"], b.mul(_i64(b, layer), self.lds_)), b.mul(el, b.const_i64(dm["dns"]))),
                 gw_n=nb(dm["gub"]), gs_n=nb(dm["gus"]), dw_n=nb(dm["dnb"]), ds_n=nb(dm["dns"]))
        S["c_lo"] = b.readfirstlane(b.lshr(a0, _c(b, 2)))
        S["nc"] = b.readfirstlane(b.sub(b.lshr(b.add(a1, _c(b, 3)), _c(b, 2)), S["c_lo"]))
        S["gs_rs"] = _rsrc(b, S["gs_a"], S["gs_n"])
        return S

    def gq(self, c: int) -> Value:
        """Chunk of this wave's K half for relative chunk ``c`` (interleaved by scale group: half 0 = chunks 0-3, 8-11,
        16-17; half 1 = 4-7, 12-15, 18-19)."""
        b = self.b
        h = self.wh
        if c < 4:
            return b.add(b.mul(h, _c(b, 4)), _c(b, c))
        if c < 8:
            return b.add(b.mul(h, _c(b, 4)), _c(b, 8 + c - 4))
        return b.add(b.mul(h, _c(b, 2)), _c(b, 16 + c - 8))

    def gu_woff(self, a: Value, c: int) -> Value:
        """Uniform (SGPR) part of the gate_up weight offset of atom ``a``, relative chunk ``c``; the lane part is
        ``self.gvo`` (row l16 x 64 + g x 16)."""
        b = self.b
        rt = self.gu_rt(a, self.wt)
        return b.readfirstlane(b.mul(b.add(b.mul(rt, _c(b, NQG)), self.gq(c)), _c(b, 1024)))

    def gu_sload(self, S: dict, a: Value, c: int, live: Value | None = None) -> Value:
        """The gate_up weight-scale register of relative chunk ``c`` (0, 4: a group dword of P-L4's sop layout; 8: the
        16 bits of group 4 holding this half's chunks 16 + 2 h, 17 + 2 h)."""
        b = self.b
        rt = self.gu_rt(a, self.wt)
        grp = self.wh if c == 0 else (b.add(self.wh, _c(b, 2)) if c == 4 else _c(b, 4))
        su = b.mul(b.add(b.mul(rt, _c(b, NQG // 4)), grp), _c(b, 256))
        if c == 8:
            su = b.add(su, b.mul(self.wh, _c(b, 2)))
        su = b.readfirstlane(su)
        vo = self.gso if live is None else b.select(live, self.gso, _c(b, OOB))
        if c == 8:
            v = b.buffer_load_f16(S["gs_rs"], vo, su)
            return b.zext(b.bitcast(v, I16), I32)
        return b.buffer_load_cp(S["gs_rs"], vo, su, dwords=1, aux=self.wcp)

    def dn_vo(self, S: dict, c: Value) -> Value:
        """Lane part of a down weight offset at K chunk ``c``: row x chunk bytes + g x 16, OOB unless the lane's K group
        (atom 4 c + g) is in the segment."""
        b, k = self.b, self.k
        blk = b.add(b.mul(c, _c(b, 4)), self.g)
        own = b.land(b.cmp_ge(blk, S["a0"]), b.cmp_lt(blk, S["a1"]))
        if k.mutant == "mask":
            own = b.cmp_lt(blk, _c(b, k.na))
        if k.inter % CHUNK:
            cbt = b.select(b.cmp_eq(c, _c(b, k.nqd - 1)), _c(b, k.chunk_bytes(k.nqd - 1)), _c(b, 64))
        else:
            cbt = _c(b, 64)
        return b.select(own, b.add(b.mul(self.l16, cbt), b.mul(self.g, _c(b, 16))), _c(b, OOB))

    def dn_base(self) -> Value:
        """This wave's first output tile (dsk: waves 0-3 own 20 + dsk, waves 4-7 20 - dsk)."""
        b, d = self.b, self.k.dsk
        n0 = H // TILE // WAVES
        dh = getattr(self, "_dh", None)
        if dh is not None:  # shd: a half unit owns tiles [80 (dh - 1), + 80): 10 per wave
            full = b.mul(self.wave, _c(b, n0))
            half = b.add(b.mul(self.wave, _c(b, n0 // 2)), b.mul(b.sub(dh, _c(b, 1)), _c(b, 4 * n0)))
            return b.readfirstlane(b.select(b.cmp_eq(dh, _c(b, 0)), full, half))
        if not d:
            return b.readfirstlane(b.mul(self.wave, _c(b, n0)))
        lo = b.mul(self.wave, _c(b, n0 + d))
        hi = b.add(_c(b, 4 * (n0 + d)), b.mul(b.sub(self.wave, _c(b, 4)), _c(b, n0 - d)))
        return b.readfirstlane(b.select(b.cmp_lt(self.wave, _c(b, 4)), lo, hi))

    def dn_live(self, tile: int):
        """None if tile ``tile`` of this wave always exists, else its (uniform) existence."""
        b, d = self.b, self.k.dsk
        dh = getattr(self, "_dh", None)
        if dh is not None:
            return None if tile < (H // TILE // WAVES) // 2 else b.cmp_eq(dh, _c(b, 0))
        if tile < H // TILE // WAVES - d:
            return None
        return b.cmp_lt(self.wave, _c(b, 4))

    def dn_steps(self, nc: int) -> list:
        """The down steps (tile, chunk index kk) of a wave: tile groups of ``dt``, chunk, tile."""
        dt = self.k.dt
        ntl = H // TILE // WAVES + self.k.dsk
        return [(G * dt + t, kk) for G in range(ntl // dt) for kk in range(nc) for t in range(dt)]

    def dn_step_v(self, e: Value, nc: Value):
        """(tile, kk) of step ``e`` for a runtime chunk count ``nc`` (the atom loop's hand-off)."""
        b, dt = self.b, self.k.dt
        if dt == 1:
            tile = b.div(e, nc)
            return tile, b.sub(e, b.mul(tile, nc))
        per = b.mul(nc, _c(b, dt))
        G = b.div(e, per)
        r = b.sub(e, b.mul(G, per))
        kk = b.div(r, _c(b, dt))
        return b.add(b.mul(G, _c(b, dt)), b.sub(r, b.mul(kk, _c(b, dt)))), kk

    def dn_load(self, dw_rs, dvo_c: Value, tile: int, c: int, tb: dict):
        """A down weight load of (this wave's tile ``tile``, absolute chunk ``c``); ``dvo_c``: the chunk's lane offset
        (``dn_vo``); ``tb``: per-tile base cache (doff=imm)."""
        b, k = self.b, self.k
        lv = self.dn_live(tile)
        if lv is not None:
            dvo_c = b.select(lv, dvo_c, _c(b, OOB))
        if k.doff == "mul":
            return b.buffer_load_cp(dw_rs, dvo_c, self.dn_so(_c(b, tile), _c(b, c)), dwords=4, aux=self.wcp)
        if tile not in tb:
            rt = b.add(self.dn_base(), _c(b, tile))
            tb[tile] = b.readfirstlane(b.mul(rt, _c(b, k.inter * TILE // 2)))
        off = b.add(dvo_c, _c(b, c * 1024)) if lv is None else b.select(lv, b.add(dvo_c, _c(b, c * 1024)), _c(b, OOB))
        return b.buffer_load_cp(dw_rs, off, tb[tile], dwords=4, aux=self.wcp)

    def dn_so(self, tile: Value, c: Value) -> Value:
        """Uniform part of a down weight offset: (this wave's output tile, K chunk)."""
        b, k = self.b, self.k
        rt = b.add(self.dn_base(), tile)
        return b.readfirstlane(b.add(b.mul(rt, _c(b, k.inter * TILE // 2)), b.mul(c, _c(b, 1024))))

    def gu_head(self, S: dict, D: int) -> list:
        """The first ``D`` (weight, scale) entries of this wave's stream for the segment's first atom."""
        b = self.b
        w_rs = _rsrc(b, S["gw_a"], S["gw_n"])
        out = []
        for c in range(D):
            w = b.buffer_load_cp(w_rs, self.gvo, self.gu_woff(S["a0"], c), dwords=4, aux=self.wcp)
            sv = self.gu_sload(S, S["a0"], c) if c in (0, 4, 8) else _c(b, 0)
            out.append((w, sv))
        return out

    def stream_segment(self, layer: Value, S: dict, ring: list, ln: dict, it: Value) -> list:
        """One segment on the stream; ``ring``: the D entries of its first atom (issued by the previous segment or the
        prologue). Returns the next segment's head (the lookahead)."""
        b, k = self.b, self.k
        P, D = k.tpp, k.depth
        if k.shd:
            self._dh = S["dh"]
        tb = b.mul(S["ps"], _c(b, TILE * P))
        cm1 = b.sub(S["cnt"], _c(b, 1))
        toks = []
        for p in range(P):
            li = b.add(b.add(tb, _c(b, p * TILE)), self.l16)
            pos = b.add(S["off"], b.smin(li, cm1))
            ent = self.lds_i(self.lst, pos)
            toks.append(dict(valid=b.cmp_lt(li, S["cnt"]), tok=b.land(ent, _c(b, 255)), slot=b.lshr(ent, _c(b, 8)),
                             hp=b.add(S["off"], b.add(tb, _c(b, p * TILE)))))
        # the x rows are in LDS once at most ``after_dma`` younger VMEM ops are outstanding (first segment: the head;
        # later ones: the previous segment's down ops issued after its DMA, the minimum over the variants)
        sg_ = it  # the segment's index in this CTA's sequence
        self.sstamp(layer, sg_, 0)
        if k.xsync == "cnt":
            with b.scf_if(b.cmp_gt(sg_, _c(b, 0))):  # segment 0 was counted in the prologue
                self.x_arrive(self.after_dma())
            self.x_wait(b.mul(b.add(sg_, _c(b, 1)), _c(b, 4)))
        else:
            b.s_waitcnt(vmcnt=min(self.after_dma(), 63))
            b.sync_lds_only()
        self.sstamp(layer, sg_, 1)
        sdn = self.load_sdn(S) if k.gu == "atom" and not k.sdl else None
        a0, a1 = S["a0"], S["a1"]
        if ln.get("units") and k.deal == "dyn":  # the next unit's ticket (read after the h-complete barrier)
            with b.scf_if(b.land(b.cmp_eq(self.wave, _c(b, 0)), self.lane0)):
                more = b.cmp_lt(b.add(it, _c(b, 1)), _c(b, ln["nmax"]))
                t_ = _aadd(k, b, self.cnt_a, ln["q"], b.select(more, _c(b, 2 if k.mutant == "tick" else 1), _c(b, 0)), scope="agent", returns=True)
                nx = b.select(more, b.add(t_, _c(b, GRID)), ln["U"])  # more false: stop (adds 0)
                self.lds_put(self.tick, b.land(it, _c(b, 1)), nx)
        tacc = self.tlb_touch(layer, S, ln, it) if k.tlbp else None
        self.sstamp(layer, sg_, 2)
        if k.gu == "wt":
            return self.wt_rest(layer, S, ln, it, toks, sg_, tacc)
        n = b.readfirstlane(b.sub(a1, a0))
        w_rs = _rsrc(b, S["gw_a"], S["gw_n"])
        fb = k.act
        vals0, shp = _flat2(ring)
        loop = b.scf_for_iter(_c(b, 0), n, _c(b, 1), [(f"gr{i}", v) for i, v in enumerate(vals0)], iv_name="atomi")
        with loop as (i, vals):
            i = b.readfirstlane(i)
            ring = _unflat2(list(vals), shp)
            a = b.readfirstlane(b.add(a0, i))
            nxt = b.add(a, _c(b, 1))
            last = b.cmp_eq(b.add(i, _c(b, 1)), n)
            notlast = b.cmp_ne(b.add(i, _c(b, 1)), n)
            mix_rs = _rsrc(b, b.select(last, S["dw_a"], S["gw_a"]), b.select(last, S["dw_n"], S["gw_n"]))
            acc = [b.zero_vec_f32(4) for _ in range(P)]
            sc_w = None
            sc_x = [None] * P

            def bread(c: int):
                q = self.gq(c)
                out = []
                for p in range(P):
                    row = b.add(_c(b, p * TILE), self.l16)
                    rb = b.mul(row, _c(b, self.XST))
                    if fb == "fp8":
                        parts = [b.smem_load_vN(self.xl, b.lshr(b.add(b.add(rb, b.add(b.mul(q, _c(b, 128)), _c(b, 64 * hh))), b.mul(self.g, _c(b, 16))), _c(b, 2)),
                                                dtype=I32, n=4) for hh in range(2)]
                        bv = b.vec_concat(parts[0], parts[1])
                    else:
                        bv = b.smem_load_vN(self.xl, b.lshr(b.add(b.add(rb, b.mul(q, _c(b, 64))), b.mul(self.g, _c(b, 16))), _c(b, 2)), dtype=I32, n=4)
                    xs = None
                    if c in (0, 4):
                        grp = self.wh if c == 0 else b.add(self.wh, _c(b, 2))
                        xs = self.xs_dw(row, b.add(b.mul(grp, _c(b, 4)), self.g))
                    elif c == 8:
                        xs = self.xs_u16(row, b.add(_c(b, 64), b.add(b.mul(self.g, _c(b, 4)), b.mul(self.wh, _c(b, 2)))))
                    out.append((bv, xs))
                return out

            bq = [bread(c_) for c_ in range(min(k.lpf, 10))]  # x B operands read lpf chunks ahead
            for c in range(10):
                b.sched_barrier(0)
                r = c + D
                if r < 10:
                    w = b.buffer_load_cp(w_rs, self.gvo, self.gu_woff(a, r), dwords=4, aux=self.wcp)
                    sv = self.gu_sload(S, a, r) if r in (0, 4, 8) else _c(b, 0)
                else:
                    j = r - 10  # the next atom's chunk j, or (last atom) the down head's entry j
                    ej = _c(b, j)
                    tile, kk_ = self.dn_step_v(ej, S["nc"])
                    cd = b.add(S["c_lo"], kk_)
                    vo = b.select(last, self.dn_vo(S, cd), self.gvo)
                    so = b.readfirstlane(b.select(last, self.dn_so(tile, cd), self.gu_woff(nxt, j)))
                    w = b.buffer_load_cp(mix_rs, vo, so, dwords=4, aux=self.wcp)
                    sv = self.gu_sload(S, nxt, j, live=notlast) if j in (0, 4, 8) else _c(b, 0)
                ring.append((w, sv))
                b.sched_barrier(0)
                wv, s_ = ring.pop(0)
                bcur = bq.pop(0)
                if c + k.lpf < 10:
                    bq.append(bread(c + k.lpf))
                if c in (0, 4, 8):
                    sc_w = s_
                for p in range(P):
                    bv, xs = bcur[p]
                    if xs is not None:
                        sc_x[p] = xs
                    op = c % 4 if c < 8 else c - 8
                    opa = ((c + 1) % 4 if c < 8 else 9 - c) if k.mutant == "sopsel" else op
                    if self.dg("nogu"):
                        continue
                    if self.dg("gsink"):
                        acc[p] = b.vec_bitcast(b.vec_pack([b.xor(b.vec_extract(b.vec_bitcast(acc[p], VectorType(I32, 4)), e_),
                                                                 b.vec_extract(wv, e_)) for e_ in range(4)], I32), VectorType(F32, 4))
                        continue
                    acc[p] = b.mfma_scale_mx(wv, bv, acc[p], sc_w, sc_x[p], shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                             opsel_a=opa, opsel_b=op)
            b.sched_barrier(0)
            AB = k.ab
            slot = b.readfirstlane(b.land(i, _c(b, AB - 1)))
            par = b.readfirstlane(b.land(b.lshr(i, _c(b, AB.bit_length() - 1)), _c(b, 1)))
            for p in range(P):
                if self.dg("nored"):  # keep the accumulator live: one LDS word per wave
                    with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
                        self.lds_put(self.misc, _c(b, 1), b.bitcast(b.vec_extract(acc[p], 0), I32))
                    continue
                self.red_put(b.add(b.mul(par, _c(b, AB)), slot), self.wave, p, acc[p])
            fire = b.lor(b.cmp_eq(slot, _c(b, AB - 1)), last) if AB > 1 else None
            if self.dg("nored"):
                fire = b.cmp_eq(self.rows, _c(b, -1))
            ctx = b.scf_if(fire) if fire is not None else contextlib.nullcontext()
            with ctx:
                b.sync_lds_only()
                for js in range(AB):
                    aj = b.readfirstlane(b.add(b.sub(i, slot), _c(b, js)))  # the batch's atom js (atom index within the segment)
                    for p in range(P):
                        ew = b.land(b.add(b.mul(aj, _c(b, P)), _c(b, p)), _c(b, WAVES - 1))
                        cond = b.cmp_eq(self.wave, ew)
                        if AB > 1:
                            cond = b.land(cond, b.cmp_le(_c(b, js), slot))
                        with b.scf_if(cond):
                            bs = b.add(b.mul(par, _c(b, AB)), _c(b, js))
                            accs = []
                            for tt in range(4):
                                v0 = self.red_get(bs, tt, p)
                                accs.append(v0 if k.mutant == "half" else self.vadd(v0, self.red_get(bs, 4 + tt, p)))
                            if self.dg("noep"):
                                with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
                                    self.lds_put(self.misc, _c(b, 1), b.bitcast(b.vec_extract(accs[0], 0), I32))
                                continue
                            self.epilogue(accs, b.add(a0, aj), p, toks[p]["hp"], toks[p]["valid"])
            if k.stamp:
                with b.scf_if(b.cmp_eq(i, _c(b, 0))):
                    self.sstamp(layer, sg_, 3)
            b.scf_yield(*_flat2(ring)[0])
        ring = _unflat2(list(loop.results), shp)
        self.sstamp(layer, sg_, 4)
        b.sync_lds_only()  # h complete; the x buffer is free
        if k.sdl:  # the down scales now (not held through the gate_up, no VMEM issue ahead of the first atoms)
            sdn = self.load_sdn(S)
        self.sstamp(layer, sg_, 5)
        # the next segment's head (the lookahead) first; its x rows by DMA inside the down variant, after the variant's
        # LDS reads (a ds_read after an LDS DMA waits for the DMA)
        if ln.get("units"):
            un = ln["unext"] if k.deal == "rr" else b.readfirstlane(self.lds_i(self.tick, b.land(it, _c(b, 1))))
            Sn = self.useg(layer, un, ln)
        else:
            Sn = self.vseg(layer, b.add(it, _c(b, 1)), ln)
        hasn = Sn["has"]
        early = k.lka == "early" or self.dg("nodn")
        look = self.gu_head(Sn, D) if early else None
        if self.dg("nodn"):
            with b.scf_if(hasn):
                self.stage_x(Sn["off"], Sn["cnt"], Sn["ps"])
        else:
            # one static variant per (first chunk, chunk count); a 0/1-trip loop carries the lookahead out of it
            # the variant loops carry the ring itself: the taken variant consumes the down head and yields the next
            # segment's head in the same slots (one register set, no copy at the segment loop's back edge)
            res = None if early else _flat2(ring)[0]
            for cl in range(k.nqd):
                for ncv in range(1, k.nqd - cl + 1):
                    cond = b.land(b.cmp_eq(S["c_lo"], _c(b, cl)), b.cmp_eq(S["nc"], _c(b, ncv)))
                    if early:
                        with b.scf_if(cond):
                            dB = self.down_operands(S, cl, ncv)
                            with b.scf_if(hasn):
                                self.stage_x(Sn["off"], Sn["cnt"], Sn["ps"])
                            self.down_stream(S, cl, ncv, ring, sdn, toks, dB)
                        continue
                    vl = b.scf_for_iter(_c(b, 0), b.select(cond, _c(b, 1), _c(b, 0)), _c(b, 1),
                                        [(f"lk{cl}{ncv}_{i}", v) for i, v in enumerate(res)], iv_name=f"dv{cl}{ncv}")
                    with vl as (_vi, _vals):
                        dB = self.down_operands(S, cl, ncv)
                        with b.scf_if(hasn):
                            self.stage_x(Sn["off"], Sn["cnt"], Sn["ps"])
                        self.sstamp(layer, sg_, 6)
                        lk = self.down_stream(S, cl, ncv, _unflat2(list(_vals), D), sdn, toks, dB, Sn=Sn)
                        b.scf_yield(*_flat2(lk)[0])
                    res = list(vl.results)
            if not early:
                look = _unflat2(res, D)
        with b.scf_if(b.land(b.cmp_eq(S["jseg"], _c(b, 0)), b.land(b.cmp_eq(self.wave, _c(b, 0)), b.cmp_eq(self.g, _c(b, 0))))):
            for p in range(P):
                tk = toks[p]
                o = b.mul(b.add(b.mul(tk["tok"], _c(b, DSTR)), tk["slot"]), _c(b, 4))
                b.buffer_store_cp(S["nseg"], self.desc_rs, b.select(tk["valid"], o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if k.cmb == "flow":
            self.flow_bump(layer, S, toks[0])
        if tacc is not None:
            with b.scf_if(b.land(self.lane0, b.cmp_eq(tacc, _c(b, 0x5EED1234)))):
                self.err()
        self.sstamp(layer, sg_, 7)
        return look

    def load_sdn(self, S: dict) -> list:
        """Down scales: one register per output tile for the whole down phase (op_sel picks the chunk's byte)."""
        b, k = self.b, self.k
        ntl = H // TILE // WAVES + k.dsk
        ds_rs = _rsrc(b, S["ds_a"], S["ds_n"])
        sdn = []
        base = self.dn_base()
        if k.dsq:  # the dsq layout: one dwordx4 per 4 tiles (ngrp 1) / per 2 tiles (ngrp 2)
            per = 4 // k.ngrp
            for j0 in range(0, ntl, per):
                rtq = b.lshr(b.add(base, _c(b, j0 - j0 % 4)), _c(b, 2))
                soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(rtq, _c(b, TILE)), self.l16), _c(b, 4)), self.g), _c(b, 16 * k.ngrp)),
                             _c(b, 16 * ((j0 % 4) // per)))
                lv = self.dn_live(j0 + per - 1)
                if lv is not None:
                    soff = b.select(lv, soff, _c(b, OOB))
                v = b.buffer_load_cp(ds_rs, soff, _c(b, 0), dwords=4, aux=self.wcp)
                for t in range(per):
                    sdn.append((b.vec_extract(v, t), None) if k.ngrp == 1 else (b.vec_extract(v, 2 * t), b.vec_extract(v, 2 * t + 1)))
            return sdn
        for j in range(ntl):
            rt = b.add(base, _c(b, j))
            soff = b.mul(b.add(b.mul(b.add(b.mul(rt, _c(b, TILE)), self.l16), _c(b, 4)), self.g), _c(b, 4 * k.ngrp))
            if self.dn_live(j) is not None:
                soff = b.select(self.dn_live(j), soff, _c(b, OOB))
            v = b.buffer_load_cp(ds_rs, soff, _c(b, 0), dwords=k.ngrp, aux=self.wcp)
            sdn.append((v, None) if k.ngrp == 1 else (b.vec_extract(v, 0), b.vec_extract(v, 1)))
        return sdn

    def pin(self, v: Value) -> Value:
        """``v`` through a side-effect asm copy: values computed from it cannot move above this point."""
        return self.b.inline_asm("v_mov_b32 $0, $1", "=v,v", [v], I32, sideeffect=True, result_name_hint="pin")

    def wt_rest(self, layer: Value, S: dict, ln: dict, it: Value, toks: list, sg_: Value, tacc) -> list:
        """gu=wt: the gate_up of the whole unit, the panel requant, the down; returns the (empty) lookahead."""
        b, k, dm = self.b, self.k, self.dm
        fb = k.act
        ntl = H // TILE // WAVES
        wk = k.wkx
        npw = k.nrt * wk // WAVES
        nck = NQG // wk  # chunk steps per sweep
        part = b.readfirstlane(b.land(self.wave, _c(b, wk - 1))) if wk == 2 else _c(b, 0)
        st = b.readfirstlane(b.mul(b.lshr(self.wave, _c(b, wk - 1)), _c(b, npw)))
        w_rs = _rsrc(b, S["gw_a"], S["gw_n"])
        s_rs = S["gs_rs"]
        tl = []  # row tiles: pair m gate, pair m up
        for m in range(npw):
            j = b.readfirstlane(b.add(st, _c(b, m)))
            tl.append(j)
            tl.append(b.readfirstlane(b.add(j, _c(b, k.nrt))))
        wb = [b.readfirstlane(b.mul(rt, _c(b, NQG * 1024))) for rt in tl]  # tile bases (bytes)
        sb_ = [b.readfirstlane(b.mul(rt, _c(b, (NQG // 4) * 256))) for rt in tl]

        def chunk(c: int):
            """Relative chunk c of this wave's K part: (absolute chunk (int or uniform), scale kind, op_sel): kind 0 a
            group dword (group: int or uniform), 1 the 16 bits at byte 2 part of group 4, None no scale load."""
            if wk == 1:
                return c, ((0, c // 4) if c % 4 == 0 else None), c % 4
            if c < 4:
                q = b.add(b.mul(part, _c(b, 4)), _c(b, c))
                return q, ((0, part) if c == 0 else None), c
            if c < 8:
                q = b.add(b.mul(part, _c(b, 4)), _c(b, 8 + c - 4))
                return q, ((0, b.add(part, _c(b, 2))) if c == 4 else None), c - 4
            q = b.add(b.mul(part, _c(b, 2)), _c(b, 16 + c - 8))
            return q, ((1, 4) if c == 8 else None), c - 8

        def issue_f(e: int):
            si, c = flat[e]
            q, sk, _ = chunk(c)
            ts = [i for m in sweeps[si] for i in (2 * m, 2 * m + 1)]
            vo = self.pin(self.gvo) if k.wpin else self.gvo
            if isinstance(q, int):
                ws = [b.buffer_load_cp(w_rs, b.add(vo, _c(b, q * 1024)), wb[i], dwords=4, aux=self.wcp) for i in ts]
            else:
                qo = b.readfirstlane(b.mul(q, _c(b, 1024)))
                ws = [b.buffer_load_cp(w_rs, vo, b.readfirstlane(b.add(wb[i], qo)), dwords=4, aux=self.wcp) for i in ts]
            ss = None
            if sk is not None:
                kind, grp = sk
                so = self.pin(self.gso) if k.wpin else self.gso
                if kind == 0:
                    if isinstance(grp, int):
                        ss = [b.buffer_load_cp(s_rs, b.add(so, _c(b, grp * 256)), sb_[i], dwords=1, aux=self.wcp) for i in ts]
                    else:
                        go = b.readfirstlane(b.mul(grp, _c(b, 256)))
                        ss = [b.buffer_load_cp(s_rs, so, b.readfirstlane(b.add(sb_[i], go)), dwords=1, aux=self.wcp) for i in ts]
                else:  # group 4's 16 bits of this part's chunks 16 + 2 part, 17 + 2 part
                    go = b.readfirstlane(b.add(_c(b, 4 * 256), b.mul(part, _c(b, 2))))
                    ss = [b.zext(b.bitcast(b.buffer_load_f16(s_rs, so, b.readfirstlane(b.add(sb_[i], go))), I16), I32) for i in ts]
            return ws, ss

        row = self.l16
        rb0 = b.mul(row, _c(b, self.XST))

        def bread(c: int):
            q, sk, _ = chunk(c)
            rb = self.pin(rb0) if k.wpin else rb0
            qb = _c(b, q * (128 if fb == "fp8" else 64)) if isinstance(q, int) else b.mul(q, _c(b, 128 if fb == "fp8" else 64))
            base = b.add(b.add(rb, qb), b.mul(self.g, _c(b, 16)))
            if fb == "fp8":
                parts = [b.smem_load_vN(self.xl, b.lshr(b.add(base, _c(b, 64 * hh)), _c(b, 2)), dtype=I32, n=4) for hh in range(2)]
                bv = b.vec_concat(parts[0], parts[1])
            else:
                bv = b.smem_load_vN(self.xl, b.lshr(base, _c(b, 2)), dtype=I32, n=4)
            xs = None
            if sk is not None:
                kind, grp = sk
                if kind == 0:
                    g4 = _c(b, 4 * grp) if isinstance(grp, int) else b.mul(grp, _c(b, 4))
                    xs = self.xs_dw(row, b.add(g4, self.g))
                else:
                    xs = self.xs_u16(row, b.add(_c(b, 64), b.add(b.mul(self.g, _c(b, 4)), b.mul(part, _c(b, 2)))))
            return bv, xs

        wg = k.wg or npw
        sweeps = [list(range(m0, min(m0 + wg, npw))) for m0 in range(0, npw, wg)]
        # the step list: (sweep, chunk, tiles loaded); word "chunk": one step per chunk over the sweep's tiles; "grp":
        # per 4-chunk scale group, tile-major (each tile's chunks of the group contiguous, 4 KiB runs)
        groups = [list(range(g0, min(g0 + 4, nck))) for g0 in range(0, nck, 4)]
        flat = []
        for si in range(len(sweeps)):
            nt = 2 * len(sweeps[si])
            if k.word == "chunk":
                flat += [(si, c, list(range(nt))) for c in range(nck)]
            else:
                flat += [(si, c, [ti]) for grp in groups for ti in range(nt) for c in grp]
        last_of_sweep = {max(e for e, f in enumerate(flat) if f[0] == si) for si in range(len(sweeps))}
        dep = min(k.wdep, len(flat))
        valid = toks[0]["valid"]
        dw_rs = _rsrc(b, S["dw_a"], S["dw_n"])
        nc = k.nqd
        steps = self.dn_steps(nc)  # down_stream's step order (the prefetched head)

        def issue_s(e: int):
            si, c, tis = flat[e]
            q, sk, _ = chunk(c)
            sw = [i for m in sweeps[si] for i in (2 * m, 2 * m + 1)]
            ts = [sw[t] for t in tis]
            vo = self.pin(self.gvo) if k.wpin else self.gvo
            if isinstance(q, int):
                ws = [b.buffer_load_cp(w_rs, b.add(vo, _c(b, q * 1024)), wb[i], dwords=4, aux=self.wcp) for i in ts]
            else:
                qo = b.readfirstlane(b.mul(q, _c(b, 1024)))
                ws = [b.buffer_load_cp(w_rs, vo, b.readfirstlane(b.add(wb[i], qo)), dwords=4, aux=self.wcp) for i in ts]
            ss = None
            if sk is not None:
                kind, grp = sk
                so = self.pin(self.gso) if k.wpin else self.gso
                if kind == 0:
                    if isinstance(grp, int):
                        ss = [b.buffer_load_cp(s_rs, b.add(so, _c(b, grp * 256)), sb_[i], dwords=1, aux=self.wcp) for i in ts]
                    else:
                        go = b.readfirstlane(b.mul(grp, _c(b, 256)))
                        ss = [b.buffer_load_cp(s_rs, so, b.readfirstlane(b.add(sb_[i], go)), dwords=1, aux=self.wcp) for i in ts]
                else:
                    go = b.readfirstlane(b.add(_c(b, 4 * 256), b.mul(part, _c(b, 2))))
                    ss = [b.zext(b.bitcast(b.buffer_load_f16(s_rs, so, b.readfirstlane(b.add(sb_[i], go))), I16), I32) for i in ts]
            return ws, ss

        ring = [issue_s(e) for e in range(dep)]
        sc_w = {}
        sc_x = None
        bq = {}  # chunk -> B operand (and its x scale)
        if k.word == "chunk":
            bq[0] = bread(0)
        acc = None
        dring = sdn = None
        for e, (si, c, tis) in enumerate(flat):
            if e == 0 or flat[e - 1][0] != si:
                acc = [b.zero_vec_f32(4) for _ in range(2 * len(sweeps[si]))]
            b.sched_barrier(0)
            if e + dep < len(flat):
                ring.append(issue_s(e + dep))
            b.sched_barrier(0)
            ws, ss = ring.pop(0)
            if ss is not None:
                for t, v in zip(tis, ss):
                    sc_w[t] = v
            if k.word == "chunk":
                bv, xs = bq.pop(c)
                if e + 1 < len(flat):
                    bq[flat[e + 1][1]] = bread(flat[e + 1][1])
            else:
                if c not in bq or tis[0] == 0 and c == [g for g in groups if c in g][0][0]:
                    for c2 in [g for g in groups if c in g][0]:
                        bq[c2] = bread(c2)
                bv, xs = bq[c]
            if xs is not None:
                sc_x = xs
            op = chunk(c)[2]
            sel = (op + 1) % 4 if k.mutant == "sopsel" else op
            for t, wv in zip(tis, ws):
                acc[t] = b.mfma_scale_mx(wv, bv, acc[t], sc_w[t], sc_x, shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                         opsel_a=sel, opsel_b=op)
            if e not in last_of_sweep:
                continue
            b.sched_barrier(0)
            if si == len(sweeps) - 1:
                self.sstamp(layer, sg_, 3)
                # the down head (no dependence on h) and the down scales, before the last panel work
                dvo = [self.dn_vo(S, _c(b, kk)) for kk in range(nc)]
                tbd = {}  # the down head in down_stream's step order
                dring = [(self.dn_load(dw_rs, dvo[kk], j, kk, tbd), None) for (j, kk) in steps[: min(k.ddepth, len(steps))]]
                sdn = self.load_sdn(S)
            for mi, m in enumerate(sweeps[si]):
                j = tl[2 * m]
                gacc, uacc = acc[2 * mi], acc[2 * mi + 1]
                col = b.add(b.mul(j, _c(b, TILE)), b.mul(self.g, _c(b, 4)))
                idx = b.add(b.mul(row, _c(b, self.PST)), col)
                if wk == 1:
                    self.wt_h_store(gacc, uacc, idx, valid)
                    continue
                # wk 2: g | u partials to this K part's region; the requant sums the parts (part 0 + part 1)
                ip = b.add(b.mul(part, _c(b, self.RG)), idx)
                b.smem_store_vN(self.pnl, [ip], b.vec_bitcast(gacc, VectorType(I32, 4)), 4)
                b.smem_store_vN(self.pnl, [b.add(ip, _c(b, k.inter))], b.vec_bitcast(uacc, VectorType(I32, 4)), 4)
        self.sstamp(layer, sg_, 4)
        b.sync_lds_only()
        self.wt_requant(toks[0]["hp"], b.sub(S["cnt"], b.mul(S["ps"], _c(b, TILE))))
        b.sync_lds_only()  # h complete; the x buffer (and panel) free
        self.sstamp(layer, sg_, 5)
        if ln.get("units"):
            un = ln["unext"] if k.deal == "rr" else b.readfirstlane(self.lds_i(self.tick, b.land(it, _c(b, 1))))
            Sn = self.useg(layer, un, ln)
        else:
            Sn = self.vseg(layer, b.add(it, _c(b, 1)), ln)
        dB = self.down_operands(S, 0, nc)
        with b.scf_if(Sn["has"]):
            self.stage_x(Sn["off"], Sn["cnt"], Sn["ps"])
        self.sstamp(layer, sg_, 6)
        self.down_stream(S, 0, nc, dring, sdn, toks, dB, Sn=None)
        with b.scf_if(b.land(b.cmp_eq(S["jseg"], _c(b, 0)), b.land(b.cmp_eq(self.wave, _c(b, 0)), b.cmp_eq(self.g, _c(b, 0))))):
            tk = toks[0]
            o = b.mul(b.add(b.mul(tk["tok"], _c(b, DSTR)), tk["slot"]), _c(b, 4))
            b.buffer_store_cp(S["nseg"], self.desc_rs, b.select(tk["valid"], o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if k.cmb == "flow":
            self.flow_bump(layer, S, toks[0])
        self.sstamp(layer, sg_, 7)
        return []

    def wt_h_store(self, gacc: Value, uacc: Value, idx: Value, valid: Value) -> None:
        """h = silu(g) u of a lane's 4 columns to the panel (f32); invalid tokens 0."""
        b = self.b
        hv = []
        for i in range(4):
            gv = b.vec_extract(gacc, i)
            uv = b.vec_extract(uacc, i)
            den = b.fadd(_f(b, 1.0), b.exp2(b.fmul(b.fneg(gv), _f(b, L2E))))
            hv.append(b.bitcast(b.select(valid, b.fmul(b.fdiv(gv, den), uv), _f(b, 0.0)), I32))
        b.smem_store_vN(self.pnl, [idx], b.vec_pack(hv, I32), 4)

    def wt_requant(self, hp: Value, nv: Value) -> None:
        """Every thread: (row, 32-column group) items of the panel to the h rows (fp8: 32 bytes, fp4: 16) and the
        scale byte; hdump: the numerics twin."""
        b, k, dm = self.b, self.k, self.dm
        NA = k.na
        nit = TILE * NA
        hrow = dm["hrow"]
        for r_ in range(-(-nit // THREADS)):
            t = b.add(self.tid, _c(b, r_ * THREADS))
            ok = b.cmp_lt(t, _c(b, nit))
            tc = b.select(ok, t, _c(b, 0))
            row = b.div(tc, _c(b, NA))
            a = b.sub(tc, b.mul(row, _c(b, NA)))
            base = b.add(b.mul(row, _c(b, self.PST)), b.mul(a, _c(b, 32)))
            vals = []
            if k.wkx == 1:
                for v4 in range(8):
                    w = b.smem_load_vN(self.pnl, b.add(base, _c(b, 4 * v4)), dtype=I32, n=4)
                    vals += [b.bitcast(b.vec_extract(w, e), F32) for e in range(4)]
            else:  # h = silu(g0 + g1) (u0 + u1) from the two parts' regions; rows past the item's tokens 0
                vrow = b.cmp_lt(row, nv)

                def ld(off):
                    w = b.smem_load_vN(self.pnl, off, dtype=I32, n=4)
                    return [b.bitcast(b.vec_extract(w, e), F32) for e in range(4)]
                for v4 in range(8):
                    o = b.add(base, _c(b, 4 * v4))
                    g0, g1 = ld(o), ld(b.add(o, _c(b, self.RG)))
                    u0, u1 = ld(b.add(o, _c(b, k.inter))), ld(b.add(o, _c(b, self.RG + k.inter)))
                    for e in range(4):
                        gv = g0[e] if k.mutant == "half" else b.fadd(g0[e], g1[e])
                        uv = b.fadd(u0[e], u1[e])
                        den = b.fadd(_f(b, 1.0), b.exp2(b.fmul(b.fneg(gv), _f(b, L2E))))
                        vals.append(b.select(vrow, b.fmul(b.fdiv(gv, den), uv), _f(b, 0.0)))
            amax = b.fabs(vals[0])
            for v in vals[1:]:
                amax = b.fmax(amax, b.fabs(v))
            words = []
            if k.act == "fp8":
                ex, scale = Q.e8m0_fp8(b, amax)
                words = [Q.pack_fp8x4(b, *vals[4 * w_: 4 * w_ + 4], scale) for w_ in range(8)]
                hb = b.add(b.mul(row, _c(b, self.HST)), b.mul(a, _c(b, 32)))
            else:
                ex, scale = _rceil_fp4(b, amax)
                for w_ in range(4):
                    lo = _pack_fp4x4(b, vals[8 * w_: 8 * w_ + 4], scale)
                    hi = _pack_fp4x4(b, vals[8 * w_ + 4: 8 * w_ + 8], scale)
                    words.append(b.lor(lo, b.shl(hi, _c(b, 16))))
                hb = b.add(b.mul(row, _c(b, self.HST)), b.mul(a, _c(b, 16)))
            with b.scf_if(ok):
                for w4 in range(len(words) // 4):
                    b.smem_store_vN(self.hl, [b.lshr(b.add(hb, _c(b, 16 * w4)), _c(b, 2))], b.vec_pack(words[4 * w4: 4 * w4 + 4], I32), 4)
                sidx = b.add(b.mul(b.add(b.mul(row, _c(b, 4)), b.land(a, _c(b, 3))), _c(b, 8)), b.lshr(a, _c(b, 2)))
                b.smem_store_vN(self.hsl, [sidx], b.trunc(ex, I8), 1)
            if k.hdump:  # rows of this item's tokens only (the next rows are another item's)
                ok = b.land(ok, b.cmp_lt(row, nv))
                gb = b.add(b.mul(b.add(hp, row), _c(b, hrow)), b.mul(a, _c(b, 32 if k.act == "fp8" else 16)))
                for w_, wd in enumerate(words):
                    b.buffer_store_cp(wd, self.hd_rs, b.select(ok, b.add(gb, _c(b, 4 * w_)), _c(b, OOB)), _c(b, 0))
                so = b.mul(b.add(b.mul(b.add(hp, row), _c(b, NA)), a), _c(b, 4))
                b.buffer_store_cp(ex, self.hds_rs, b.select(ok, so, _c(b, OOB)), _c(b, 0))

    def after_dma(self) -> int:
        """VMEM instructions a wave issues after a segment's x DMA, at least: the first segment's head (prologue) or
        the smallest down variant's ops after its DMA (top-up and refill loads, partial stores)."""
        k = self.k
        D = k.depth
        ntl = H // TILE // WAVES
        if k.gu == "wt":
            return max(0, ntl * k.nqd - min(k.ddepth, ntl * k.nqd) + (0 if k.sskip else k.tpp * ntl))
        head = D + sum(1 for c in (0, 4, 8) if c < D)
        nst = 0 if k.sskip else k.tpp * ntl  # sskip: the stores may all be skipped
        down_min = min(ntl * nc - D + nst for nc in range(1, k.nqd + 1))  # loads past the handoff + stores
        if k.lka == "tail":
            down_min += head
            return max(0, down_min)  # the first segment's DMA is waited in the prologue (stream_layer)
        return max(0, min(head, down_min))

    def down_operands(self, S: dict, cl: int, nc: int):
        """The down variant's B operands (h of K chunks cl .. cl + nc - 1, zero past I) and h-scale qwords, per pass
        tile (LDS reads, before the variant's DMA)."""
        b, k, dm = self.b, self.k, self.dm
        z4 = b.vec_pack([_c(b, 0)] * 4, I32)
        bh, hsq = [], []
        hr = dm["hrow"]
        for p in range(k.tpp):
            row = b.add(_c(b, p * TILE), self.l16)
            rb = b.mul(row, _c(b, self.HST))
            per = []
            for kk in range(nc):
                c = cl + kk
                if k.act == "fp8":
                    parts = []
                    for hh in range(2):
                        kb0 = c * 128 + 64 * hh
                        kb = b.add(_c(b, kb0), b.mul(self.g, _c(b, 16)))
                        v_ = b.smem_load_vN(self.hl, b.lshr(b.add(rb, b.smin(kb, _c(b, hr))), _c(b, 2)), dtype=I32, n=4)
                        parts.append(b.select(b.cmp_lt(kb, _c(b, hr)), v_, z4) if kb0 + 48 >= hr else v_)
                    per.append(b.vec_concat(parts[0], parts[1]))
                else:
                    kb0 = c * 64
                    kb = b.add(_c(b, kb0), b.mul(self.g, _c(b, 16)))
                    v_ = b.smem_load_vN(self.hl, b.lshr(b.add(rb, b.smin(kb, _c(b, hr))), _c(b, 2)), dtype=I32, n=4)
                    per.append(b.select(b.cmp_lt(kb, _c(b, hr)), v_, z4) if kb0 + 48 >= hr else v_)
            bh.append(per)
            q = b.vec_bitcast(b.smem_load_vN(self.hsl, b.mul(b.add(b.mul(row, _c(b, 4)), self.g), _c(b, 8)), dtype=I8, n=8), VectorType(I32, 2))
            hsq.append((b.vec_extract(q, 0), b.vec_extract(q, 1)))
        return bh, hsq

    def tlb_touch(self, layer: Value, S: dict, ln: dict, it: Value) -> None:
        """Wave 0, lane 0: one dword of each region the segment / the next segment will stream (results sunk)."""
        b, dm = self.b, self.dm
        if ln.get("units"):  # tlbp: the next unit is not known yet under dyn; touch the static successor
            Sn = self.useg(layer, b.add(self.cta, b.mul(b.add(it, _c(b, 1)), _c(b, GRID))), ln)
        else:
            Sn = self.vseg(layer, b.add(it, _c(b, 1)), ln)
        regs = [(S["dw_a"], S["dw_n"], 0), (S["dw_a"], S["dw_n"], dm["dnb"] - 4), (S["ds_a"], S["ds_n"], 0),
                (Sn["gw_a"], Sn["gw_n"], 0), (Sn["gw_a"], Sn["gw_n"], dm["gub"] - 4), (Sn["gs_a"], Sn["gs_n"], 0)]
        w0 = b.cmp_eq(self.wave, _c(b, 0))
        acc = _c(b, 0)
        for a_, n_, o_ in regs:  # zero-length for waves 1-7: one VMEM op each, no bytes
            rs = _rsrc(b, a_, b.select(w0, n_, _c(b, 0)))
            acc = b.xor(acc, b.buffer_load_cp(rs, b.select(self.lane0, _c(b, o_), _c(b, OOB)), _c(b, 0), dwords=1))
        return acc  # sunk at the segment end (a test here would wait for the walks)

    def x_arrive(self, n_after: int) -> None:
        """xsync=cnt: the DMA waves (0-3) wait until their x DMA landed (at most ``n_after`` younger VMEM ops pending),
        then count one arrival in LDS."""
        b = self.b
        with b.scf_if(b.cmp_lt(self.wave, _c(b, 4))):
            b.s_waitcnt(vmcnt=min(n_after, 63))
            with b.scf_if(self.lane0):
                b.lds_atomic_add(self.xcnt, [_c(b, 0)], _c(b, 1))

    def x_wait(self, target: Value) -> None:
        """xsync=cnt: spin (LDS reads) until the arrival count reaches ``target`` (uniform)."""
        b = self.b
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        addr = b.trunc(b.smem_addr_of(self.xcnt), I32)
        s_ = "x"
        lines = [
            "s_mov_b32 $2, 0",
            W._lbl(s_, "top") + ":",
            "ds_read_b32 $0, $3",
            "s_waitcnt lgkmcnt(0)",
            "v_cmp_lt_i32_e64 $1, $0, $4",
            "s_and_b64 $1, $1, exec",
            "s_cbranch_scc0 " + W._lbl(s_, "ok"),
            "s_add_u32 $2, $2, 1",
            "s_cmp_lt_u32 $2, 0x400000",
            "s_cbranch_scc0 " + W._lbl(s_, "ok"),
            "s_sleep 1",
            "s_branch " + W._lbl(s_, "top"),
            W._lbl(s_, "ok") + ":",
        ]
        b.inline_asm_multi("\n".join(lines), "=&v,=&s,=&s,v,s,~{scc},~{memory}", [b.add(addr, _c(b, 0)), b.readfirstlane(target)],
                           result_types=[I32, I64, I32], result_name_hint="xpoll")

    def setprio_h1(self, level: int, level0: int = 0) -> None:
        """K-half-1 waves (4..7) to ``level``, the others to ``level0`` (uniform branches)."""
        b = self.b
        with b.scf_if(b.cmp_ne(self.wh, _c(b, 0))):
            b.s_setprio(level)
        with b.scf_if(b.cmp_eq(self.wh, _c(b, 0))):
            b.s_setprio(level0)

    def stage_rows(self, S: dict, toks: list) -> list:
        """pstg: per pass tile, the staged-store instructions' (row, lane offset): lane L stores row
        i (64 / (4 ST)) + L / (4 ST), 16 bytes at (L % (4 ST)) x 16 of the row's ST x 64 bytes (OOB past the item's tokens)."""
        b, k = self.b, self.k
        ST = k.pstg
        lpr = ST * 4  # lanes per row
        rpi = 64 // lpr  # rows per instruction
        tb = b.mul(S["ps"], _c(b, TILE * k.tpp))
        cm1 = b.sub(S["cnt"], _c(b, 1))
        out = []
        for p in range(k.tpp):
            rows = []
            for i_ in range(TILE // rpi):
                r_ = b.add(_c(b, i_ * rpi), b.div(self.lane, _c(b, lpr)))
                li = b.add(b.add(tb, _c(b, p * TILE)), r_)
                ent = self.lds_i(self.lst, b.add(S["off"], b.smin(li, cm1)))
                tok, slot = b.land(ent, _c(b, 255)), b.lshr(ent, _c(b, 8))
                prow = b.add(b.mul(b.add(b.mul(tok, _c(b, SLOTS)), slot), _c(b, MAXSEG)), S["jseg"])
                vo_ = b.mul(b.add(b.mul(prow, _c(b, H)), b.mul(b.land(self.lane, _c(b, lpr - 1)), _c(b, 4))), _c(b, 4))
                first = b.add(b.add(tb, _c(b, p * TILE)), _c(b, i_ * rpi))  # the instruction's first row (uniform)
                rows.append((r_, b.select(b.cmp_lt(li, S["cnt"]), vo_, _c(b, OOB)), b.cmp_lt(first, S["cnt"])))
            out.append(rows)
        return out

    def down_stream(self, S: dict, cl: int, nc: int, ring: list, sdn: list, toks: list, dB, Sn: dict | None = None):
        """Down over K chunks ``cl .. cl + nc - 1`` (static): wave w's 20 output tiles; ``ring``: its first D entries;
        ``dB``: the B operands / h scales (``down_operands``; index kk = chunk cl + kk)."""
        b, k, dm = self.b, self.k, self.dm
        P, D = k.tpp, k.depth
        fb = k.act
        ntl = H // TILE // WAVES
        dw_rs = _rsrc(b, S["dw_a"], S["dw_n"])
        bh, hsq = dB
        steps = self.dn_steps(nc)
        dvo = [self.dn_vo(S, _c(b, cl + kk)) for kk in range(nc)]
        tbc = {}
        Dd = max(k.ddepth, len(ring))
        pvo = []
        for p in range(P):
            tk = toks[p]
            prow = b.add(b.mul(b.add(b.mul(tk["tok"], _c(b, SLOTS)), tk["slot"]), _c(b, MAXSEG)), S["jseg"])
            pvo.append(b.select(tk["valid"], b.mul(b.add(b.mul(prow, _c(b, H)), b.mul(self.g, _c(b, 4))), _c(b, 4)), _c(b, OOB)))
        stg_rows = self.stage_rows(S, toks) if k.pstg else None
        ring = list(ring)
        for e in range(len(ring), min(Dd, len(steps))):  # top the handoff's entries up to the down depth
            j2, k2 = steps[e]
            ring.append((self.dn_load(dw_rs, dvo[k2], j2, cl + k2, tbc), None))
        acc = {}
        look = []
        nlook = 0 if Sn is None else D
        if k.prio == "dn":
            self.setprio_h1(1)
        gw_rs = _rsrc(b, Sn["gw_a"], Sn["gw_n"]) if Sn is not None else None
        for e, (j, kk) in enumerate(steps):
            b.sched_barrier(0)
            if e + Dd < len(steps):
                j2, k2 = steps[e + Dd]
                ring.append((self.dn_load(dw_rs, dvo[k2], j2, cl + k2, tbc), None))
            else:
                c_ = e + Dd - len(steps)  # the next segment's head entry c_
                if c_ < nlook:
                    w = b.buffer_load_cp(gw_rs, self.gvo, self.gu_woff(Sn["a0"], c_), dwords=4, aux=self.wcp)
                    look.append((w, self.gu_sload(Sn, Sn["a0"], c_) if c_ in (0, 4, 8) else _c(b, 0)))
            b.sched_barrier(0)
            wv = ring.pop(0)[0]
            if kk == 0:
                acc[j] = [b.zero_vec_f32(4) for _ in range(P)]
                if k.prio == "alt":
                    self.setprio_h1(j % 2, 1 - j % 2)
            c = cl + kk
            sa = sdn[j][0] if c < 4 else sdn[j][1]
            oa = c % 4 if c < 4 else c - 4
            ob = oa
            if k.mutant == "hsc":
                ob = (c + 1) % 4 if c < 4 else 1
            for p in range(P):
                sb = hsq[p][0] if c < 4 else hsq[p][1]
                if self.dg("dsink"):
                    acc[j][p] = b.vec_bitcast(b.vec_pack([b.xor(b.vec_extract(b.vec_bitcast(acc[j][p], VectorType(I32, 4)), e_),
                                                                b.vec_extract(wv, e_)) for e_ in range(4)], I32), VectorType(F32, 4))
                    continue
                acc[j][p] = b.mfma_scale_mx(wv, bh[p][kk], acc[j][p], sa, sb, shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                            opsel_a=oa, opsel_b=ob)
            if kk == nc - 1 and j % k.dt == k.dt - 1:  # the group's tiles, stored after its last MFMA
                for jj in range(j - k.dt + 1, j + 1):
                    self.dn_store_tile(jj, acc.pop(jj), pvo, stg_rows)
        b.sched_barrier(0)
        if k.prio in ("dn", "alt"):
            b.s_setprio(0)
        for c_ in range(len(look), nlook):
            w = b.buffer_load_cp(gw_rs, self.gvo, self.gu_woff(Sn["a0"], c_), dwords=4, aux=self.wcp)
            look.append((w, self.gu_sload(Sn, Sn["a0"], c_) if c_ in (0, 4, 8) else _c(b, 0)))
        return look

    def dn_store_tile(self, j: int, accj: list, pvo: list, stg_rows) -> None:
        """Tile ``j``'s partials (per pass tile): direct (16 rows x 64 B) or into the stage, flushed per ``pstg`` tiles
        (rows of pstg x 64 B)."""
        b, k = self.b, self.k
        P = k.tpp
        ntl = H // TILE // WAVES
        aux = {"sc1": CPOL_SC1, "nt": CPOL_NT, "def": 0}[k.pst]
        if not k.pstg:
            if self.dg("nost"):
                return
            rt = b.add(self.dn_base(), _c(b, j))
            for p in range(P):
                vo = pvo[p] if self.dn_live(j) is None else b.select(self.dn_live(j), pvo[p], _c(b, OOB))
                b.buffer_store_cp(b.vec_bitcast(accj[p], VectorType(I32, 4)), self.part_rs, vo,
                                  b.readfirstlane(b.mul(rt, _c(b, TILE * 4))), aux=aux)
            return
        ST = k.pstg
        for p in range(P):  # the tile into the wave's stage
            sidx = b.add(b.mul(b.add(b.mul(b.add(b.mul(self.wave, _c(b, P)), _c(b, p)), _c(b, TILE)), self.l16), _c(b, self.SGS)),
                         b.add(_c(b, (j % ST) * TILE), b.mul(self.g, _c(b, 4))))
            b.smem_store_vN(self.stg, [sidx], b.vec_bitcast(accj[p], VectorType(I32, 4)), 4)
        if j % ST == ST - 1 and not self.dg("nost"):  # the stage's rows, ST x 64 B each, contiguous
            col0 = b.readfirstlane(b.mul(b.add(self.dn_base(), _c(b, j - ST + 1)), _c(b, TILE * 4)))
            glv = self.dn_live(j)  # the group's last tile: a group is live or dead as a whole (dsk multiple of pstg)
            for p in range(P):
                for r_, vo_, any_ in stg_rows[p]:
                    if glv is not None:
                        vo_ = b.select(glv, vo_, _c(b, OOB))
                    ridx = b.add(b.mul(b.add(b.mul(b.add(b.mul(self.wave, _c(b, P)), _c(b, p)), _c(b, TILE)), r_), _c(b, self.SGS)),
                                 b.mul(b.land(self.lane, _c(b, ST * 4 - 1)), _c(b, 4)))
                    ctx = b.scf_if(any_) if k.sskip else contextlib.nullcontext()
                    with ctx:  # sskip: no store instruction for rows past the item's tokens
                        b.buffer_store_cp(b.smem_load_vN(self.stg, ridx, dtype=I32, n=4), self.part_rs, vo_, col0, aux=aux)

    def xb_unit_v(self, U: Value, r: Value) -> Value:
        """xb: this CTA's unit of round ``r``, or U when it has none (uniform)."""
        b = self.b
        x = b.land(self.cta, _c(b, XCDS - 1))
        j = b.lshr(self.cta, _c(b, 3))
        left = b.sub(U, b.mul(r, _c(b, GRID)))
        S = b.select(b.cmp_ge(left, _c(b, GRID)), _c(b, GRID // XCDS), b.div(b.add(left, _c(b, XCDS - 1)), _c(b, XCDS)))
        u = b.add(b.mul(r, _c(b, GRID)), b.add(b.mul(x, S), j))
        ok = b.land(b.cmp_lt(j, S), b.cmp_lt(u, U))
        return b.readfirstlane(b.select(ok, u, U))

    def xb_rounds(self, U: Value) -> Value:
        b = self.b
        F = b.div(U, _c(b, GRID))
        last = self.xb_unit_v(U, F)
        return b.readfirstlane(b.add(F, b.select(b.cmp_lt(last, U), _c(b, 1), _c(b, 0))))

    def rr_units(self, L: Value, U: Value, v: Value, nv: Value):
        """deal=rr: (this round's unit, the next round's unit) of iteration ``v``."""
        b, k = self.b, self.k
        if not k.xb:
            u = b.readfirstlane(b.add(L, b.mul(v, _c(b, GRID))))
            return u, b.add(u, _c(b, GRID))
        u = self.xb_unit_v(U, v)
        un = self.xb_unit_v(U, b.add(v, _c(b, 1)))
        return u, b.readfirstlane(b.select(b.cmp_lt(b.add(v, _c(b, 1)), nv), un, U))

    def unit_layer(self, layer: Value, nit: Value) -> None:
        """deal=rr / dyn: the CTA's units (rr: L, L + 256, ..; dyn: L, then tickets) on one stream."""
        b, k = self.b, self.k
        Dp = k.depth
        if k.xh:
            L = b.add(b.mul(b.land(self.cta, _c(b, XCDS - 1)), _c(b, GRID // XCDS)), b.lshr(self.cta, _c(b, 3)))
        else:
            L = self.cta
        L = b.readfirstlane(L)
        if k.pc:
            pcv, F = k.pc, _c(b, 0)
            U = b.readfirstlane(b.mul(nit, _c(b, k.pc)))
        else:  # whole full rounds; the last round's items in up to 4 pieces
            F = b.readfirstlane(b.mul(b.div(nit, _c(b, GRID)), _c(b, GRID)))
            rem = b.sub(nit, F)
            pcv = b.readfirstlane(b.select(b.cmp_gt(rem, _c(b, 0)),
                                           b.smax(b.smin(b.div(_c(b, GRID), b.smax(rem, _c(b, 1))),
                                                         _c(b, min(4, -(-k.na // 4)) if k.pcal else 4)), _c(b, 1)), _c(b, 1)))
            U = b.readfirstlane(b.add(F, b.mul(rem, pcv)))
        ln = dict(L=L, U=U, A=_c(b, 1), C=_c(b, 1), units=True, layer=layer, pc=pcv, F=F)
        if k.shd:  # the shared items (the last psh items) split when psh more units fit the last round, all whole
            psh = b.readfirstlane(b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4)))
            room = b.sub(b.mul(b.div(b.add(U, _c(b, GRID - 1)), _c(b, GRID)), _c(b, GRID)), U)
            whole = b.cmp_eq(pcv, _c(b, 1)) if not isinstance(pcv, int) else b.cmp_eq(_c(b, pcv), _c(b, 1))
            split = b.land(b.cmp_le(psh, room), whole)
            ln["U0"] = U
            ln["itsh"] = b.readfirstlane(b.sub(nit, psh))
            ln["split"] = split
            U = b.readfirstlane(b.add(U, b.select(split, psh, _c(b, 0))))
            ln["U"] = U
        if k.deal == "dyn":
            par = b.land(b.add(self.ep, layer), _c(b, 1))
            ln["q"] = b.readfirstlane(b.mul(b.add(_c(b, W_TICK), par), _c(b, 4)))
            with b.scf_if(b.land(b.cmp_eq(self.cta, _c(b, 0)), D.is_thread0(b))):  # the next layer's ticket counter
                b.buffer_store_cp(_c(b, 0), self.cnt_rs, b.mul(b.sub(_c(b, W_TICK + 1), par), _c(b, 4)), _c(b, 0), aux=CPOL_SC1)
            nmax = -(-(k.it_max * (k.pc or 4)) // GRID) * 2 + 2
            ln["nmax"] = nmax
        if k.xb:  # round 0's unit (U: none)
            L = self.xb_unit_v(U, _c(b, 0))
            ln["L"] = L
        S0 = self.useg(layer, L, ln)
        with b.scf_if(S0["has"]):
            self.stage_x(S0["off"], S0["cnt"], S0["ps"])
        head = self.gu_head(S0, Dp) if k.gu == "atom" else []
        if k.gu == "wt":
            b.s_waitcnt(vmcnt=0)
            b.sync_lds_only()
        elif k.lka == "tail" and k.xsync == "cnt":
            self.x_arrive(Dp + sum(1 for c in (0, 4, 8) if c < Dp))
        elif k.lka == "tail":
            b.s_waitcnt(vmcnt=min(Dp + sum(1 for c in (0, 4, 8) if c < Dp), 63))
            b.sync_lds_only()
        vals0, shp = _flat2(head)
        if k.deal == "rr" and k.xb:  # the CTA's rounds: full ones, plus the last if its block holds this CTA
            nv = self.xb_rounds(U)
        if k.deal == "rr":
            if not k.xb:
                nv = b.readfirstlane(b.select(b.cmp_lt(L, U), b.div(b.add(b.sub(U, L), _c(b, GRID - 1)), _c(b, GRID)), _c(b, 0)))
            if not vals0:  # no carried head (gu=wt)
                with b.scf_for(_c(b, 0), nv, _c(b, 1), iv_name="segi") as v:
                    v = b.readfirstlane(v)
                    u, ln["unext"] = self.rr_units(L, U, v, nv)
                    self.stream_segment(layer, self.useg(layer, u, ln), [], ln, v)
                return
            loop = b.scf_for_iter(_c(b, 0), nv, _c(b, 1), [(f"hd{i}", v) for i, v in enumerate(vals0)], iv_name="segi")
            with loop as (v, vals):
                v = b.readfirstlane(v)
                u, ln["unext"] = self.rr_units(L, U, v, nv)
                S = self.useg(layer, u, ln)
                look = self.stream_segment(layer, S, _unflat2(list(vals), shp), ln, v)
                b.scf_yield(*_flat2(look)[0])
            return
        # dyn: up to nmax segments; a skipped iteration costs a few scalar ops (0/1-trip body)
        loop = b.scf_for_iter(_c(b, 0), _c(b, ln["nmax"]), _c(b, 1), [("ucur", L)] + [(f"hd{i}", v) for i, v in enumerate(vals0)],
                              iv_name="segi")
        with loop as (v, vals):
            v = b.readfirstlane(v)
            u = b.readfirstlane(vals[0])
            inner = b.scf_for_iter(_c(b, 0), b.select(b.cmp_lt(u, U), _c(b, 1), _c(b, 0)), _c(b, 1),
                                   [("uin", u)] + [(f"hi{i}", x) for i, x in enumerate(vals[1:])], iv_name="segb")
            with inner as (_z, iv):
                ln["v"] = v
                S = self.useg(layer, u, ln)
                look = self.stream_segment(layer, S, _unflat2(list(iv[1:]), shp), ln, v)
                nxt = b.readfirstlane(self.lds_i(self.tick, b.land(v, _c(b, 1))))
                b.scf_yield(nxt, *_flat2(look)[0])
            b.scf_yield(*inner.results)

    def stream_layer(self, layer: Value, nit: Value) -> None:
        b, k = self.b, self.k
        D = k.depth
        if k.prio == "all":
            self.setprio_h1(1)
        if k.deal in ("rr", "dyn"):
            self.unit_layer(layer, nit)
            return
        ln = self.deal_line(nit)
        if not ln.get("rr"):
            ln["nv"] = b.readfirstlane(b.sub(ln["kl"], ln["kf"]))
        S0 = self.vseg(layer, _c(b, 0), ln)
        has0 = S0["has"]
        with b.scf_if(has0):
            self.stage_x(S0["off"], S0["cnt"], S0["ps"])
        head = self.gu_head(S0, D)
        if k.lka == "tail" and k.xsync == "cnt":
            self.x_arrive(D + sum(1 for c in (0, 4, 8) if c < D))
        elif k.lka == "tail":  # the first segment's x rows: in LDS once only its head's loads remain outstanding
            b.s_waitcnt(vmcnt=min(D + sum(1 for c in (0, 4, 8) if c < D), 63))
            b.sync_lds_only()
        vals0, shp = _flat2(head)
        loop = b.scf_for_iter(_c(b, 0), ln["nv"], _c(b, 1), [(f"hd{i}", v) for i, v in enumerate(vals0)], iv_name="segi")
        with loop as (v, vals):
            v = b.readfirstlane(v)
            ring = _unflat2(list(vals), shp)
            S = self.vseg(layer, v, ln)
            look = self.stream_segment(layer, S, ring, ln, v)
            b.scf_yield(*_flat2(look)[0])


    def w2_tokens(self, it: Value):
        """(e, list offset, per-lane (valid, token, slot, list position)) of item ``it`` (lane row l16)."""
        b = self.b
        e, ps, off, cnt = self.item_info(it)
        li = b.add(b.mul(ps, _c(b, TILE)), self.l16)
        valid = b.cmp_lt(li, cnt)
        ent = self.lds_i(self.lst, b.add(off, b.smin(li, b.sub(cnt, _c(b, 1)))))
        return e, valid, b.land(ent, _c(b, 255)), b.lshr(ent, _c(b, 8)), b.add(off, li)

    def w2_units(self, n_units: Value):
        """(trip count, unit of trip ``r`` fn) of this wave: unit = r 2048 + wave 256 + cta."""
        b = self.b
        w0 = b.readfirstlane(b.add(b.mul(self.wave, _c(b, GRID)), self.cta))
        trips = b.readfirstlane(b.select(b.cmp_lt(w0, n_units),
                                         b.div(b.add(b.sub(n_units, w0), _c(b, GRID * WAVES - 1)), _c(b, GRID * WAVES)), _c(b, 0)))
        return trips, (lambda r: b.readfirstlane(b.add(w0, b.mul(r, _c(b, GRID * WAVES)))))

    def w2_gate_up(self, layer: Value, it: Value, a: Value) -> None:
        b, k, dm = self.b, self.k, self.dm
        fb = k.act
        e, valid, tok, slot, hp = self.w2_tokens(it)
        el = _i64(b, e)
        nul = k.null == "oob"
        gw_a = b.add(b.add(self.a["wgu"], b.mul(_i64(b, layer), self.lgu)), b.mul(el, b.const_i64(dm["gub"])))
        gs_a = b.add(b.add(self.a["sgu"], b.mul(_i64(b, layer), self.lgs)), b.mul(el, b.const_i64(dm["gus"])))
        w_rs = _rsrc(b, gw_a, _c(b, 0 if nul else dm["gub"]))
        s_rs = _rsrc(b, gs_a, _c(b, 0 if nul else dm["gus"]))
        rts = [b.readfirstlane(b.add(b.mul(a, _c(b, 2)), _c(b, (t % 2) + (k.nrt if t >= 2 else 0)))) for t in range(4)]
        wb = [b.readfirstlane(b.mul(rt, _c(b, NQG * 1024))) for rt in rts]
        sb_ = [b.readfirstlane(b.mul(rt, _c(b, (NQG // 4) * 256))) for rt in rts]
        xrow, xsrow = dm["xrow"], dm["xsrow"]
        xo = b.add(b.mul(tok, _c(b, xrow)), b.mul(self.g, _c(b, 16)))
        xso = b.add(b.mul(tok, _c(b, xsrow)), b.mul(self.g, _c(b, 4)))
        x_rs, xs_rs = self.x_rs, self.xs_rs

        def issue(q: int):
            ws = [b.buffer_load_cp(w_rs, b.add(self.gvo, _c(b, q * 1024)), wb[t], dwords=4, aux=self.wcp) for t in range(4)]
            if fb == "fp8":
                xv = b.vec_concat(*[b.buffer_load_cp(x_rs, b.add(xo, _c(b, q * 128 + 64 * hh)), _c(b, 0), dwords=4) for hh in range(2)])
            else:
                xv = b.buffer_load_cp(x_rs, b.add(xo, _c(b, q * 64)), _c(b, 0), dwords=4)
            ss = xs = None
            if q % 4 == 0:
                ss = [b.buffer_load_cp(s_rs, b.add(self.gso, _c(b, (q // 4) * 256)), sb_[t], dwords=1, aux=self.wcp) for t in range(4)]
                xs = b.buffer_load_cp(xs_rs, b.add(xso, _c(b, (q // 4) * 16)), _c(b, 0), dwords=1)
            return ws, xv, ss, xs

        dep = min(k.w2d, NQG)
        ring = [issue(q) for q in range(dep)]
        acc = [b.zero_vec_f32(4) for _ in range(4)]
        sw = sx = None
        for q in range(NQG):
            b.sched_barrier(0)
            if q + dep < NQG:
                ring.append(issue(q + dep))
            b.sched_barrier(0)
            ws, xv, ss, xs = ring.pop(0)
            if ss is not None:
                sw, sx = ss, xs
            sel = (q + 1) % 4 if k.mutant == "sopsel" else q % 4
            for t in range(4):
                acc[t] = b.mfma_scale_mx(ws[t], xv, acc[t], sw[t], sx, shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                         opsel_a=sel, opsel_b=q % 4)
        b.sched_barrier(0)
        hp0 = b.sub(hp, self.l16)  # the item's first list position (the epilogue adds the lane row)
        self.epilogue(acc, a, 0, hp0, valid, glob=True)

    def w2_down(self, layer: Value, it: Value, G: Value) -> None:
        b, k, dm = self.b, self.k, self.dm
        fb = k.act
        DT = k.w2t
        nc = k.nqd
        e, valid, tok, slot, hp = self.w2_tokens(it)
        el = _i64(b, e)
        nul = k.null == "oob"
        da = b.add(b.add(self.a["wdn"], b.mul(_i64(b, layer), self.ldn)), b.mul(el, b.const_i64(dm["dnb"])))
        dsa = b.add(b.add(self.a["sdn"], b.mul(_i64(b, layer), self.lds_)), b.mul(el, b.const_i64(dm["dns"])))
        dw_rs = _rsrc(b, da, _c(b, 0 if nul else dm["dnb"]))
        ds_rs = _rsrc(b, dsa, _c(b, 0 if nul else dm["dns"]))
        S = dict(a0=_c(b, 0), a1=_c(b, k.na))
        rts = [b.readfirstlane(b.add(b.mul(G, _c(b, DT)), _c(b, t))) for t in range(DT)]
        TB = k.inter * TILE // 2
        tb = [b.readfirstlane(b.mul(rt, _c(b, TB))) for rt in rts]
        dvo = [self.dn_vo(S, _c(b, c)) for c in range(nc)]
        hrow = dm["hrow"]
        hb = b.mul(hp, _c(b, hrow))
        z4 = b.vec_pack([_c(b, 0)] * 4, I32)
        # h B operands and scales of this lane's token (from L2: other waves wrote them before the barrier)
        bh = []
        for c in range(nc):
            if fb == "fp8":
                parts = []
                for hh in range(2):
                    kb0 = c * 128 + 64 * hh
                    kb = b.add(_c(b, kb0), b.mul(self.g, _c(b, 16)))
                    v_ = b.buffer_load_cp(self.hd_rs, b.add(hb, kb), _c(b, 0), dwords=4)
                    parts.append(b.select(b.cmp_lt(kb, _c(b, hrow)), v_, z4) if kb0 + 48 >= hrow else v_)
                bh.append(b.vec_concat(parts[0], parts[1]))
            else:
                kb0 = c * 64
                kb = b.add(_c(b, kb0), b.mul(self.g, _c(b, 16)))
                v_ = b.buffer_load_cp(self.hd_rs, b.add(hb, kb), _c(b, 0), dwords=4)
                bh.append(b.select(b.cmp_lt(kb, _c(b, hrow)), v_, z4) if kb0 + 48 >= hrow else v_)
        hsq = []
        for dw in range(-(-nc // 4)):
            word = _c(b, 0)
            for cc in range(4):
                c = dw * 4 + cc
                if c >= nc:
                    break
                blk = b.add(_c(b, 4 * c), self.g)
                v_ = b.buffer_load_cp(self.hds_rs, b.mul(b.add(b.mul(hp, _c(b, k.na)), b.smin(blk, _c(b, k.na - 1))), _c(b, 4)),
                                      _c(b, 0), dwords=1)
                word = b.lor(word, b.shl(b.land(v_, _c(b, 255)), _c(b, 8 * cc)))
            hsq.append(word)
        if k.mutant == "hsc":
            hsq = hsq[::-1] if len(hsq) > 1 else [b.lshr(hsq[0], _c(b, 8))]
        sdn = []
        for rt in rts:
            soff = b.mul(b.add(b.mul(b.add(b.mul(rt, _c(b, TILE)), self.l16), _c(b, 4)), self.g), _c(b, 4 * k.ngrp))
            v = b.buffer_load_cp(ds_rs, soff, _c(b, 0), dwords=k.ngrp, aux=self.wcp)
            sdn.append((v, None) if k.ngrp == 1 else (b.vec_extract(v, 0), b.vec_extract(v, 1)))
        steps = [(c, t) for c in range(nc) for t in range(DT)]
        dep = min(k.w2d * DT, len(steps))
        ring = [b.buffer_load_cp(dw_rs, b.add(dvo[c], _c(b, c * 1024)), tb[t], dwords=4, aux=self.wcp) for c, t in steps[:dep]]
        acc = [b.zero_vec_f32(4) for _ in range(DT)]
        for i, (c, t) in enumerate(steps):
            b.sched_barrier(0)
            if i + dep < len(steps):
                c2, t2 = steps[i + dep]
                ring.append(b.buffer_load_cp(dw_rs, b.add(dvo[c2], _c(b, c2 * 1024)), tb[t2], dwords=4, aux=self.wcp))
            b.sched_barrier(0)
            wv = ring.pop(0)
            sa = sdn[t][0] if c < 4 else sdn[t][1]
            o = c % 4 if c < 4 else c - 4
            acc[t] = b.mfma_scale_mx(wv, bh[c], acc[t], sa, hsq[c // 4], shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                                     opsel_a=o, opsel_b=o)
        b.sched_barrier(0)
        slot_ = slot
        if k.mutant == "slot":
            slot_ = b.select(b.cmp_eq(slot, _c(b, TOPK)), _c(b, TOPK - 1), slot)
        prow = b.mul(b.add(b.mul(tok, _c(b, SLOTS)), slot_), _c(b, MAXSEG))
        pvo = b.select(valid, b.mul(b.add(b.mul(prow, _c(b, H)), b.mul(self.g, _c(b, 4))), _c(b, 4)), _c(b, OOB))
        for t in range(DT):
            b.buffer_store_cp(b.vec_bitcast(acc[t], VectorType(I32, 4)), self.part_rs, pvo, b.readfirstlane(b.mul(rts[t], _c(b, TILE * 4))),
                              aux=CPOL_SC1)
        with b.scf_if(b.land(b.cmp_eq(G, _c(b, 0)), b.cmp_eq(self.g, _c(b, 0)))):
            o = b.mul(b.add(b.mul(tok, _c(b, DSTR)), slot), _c(b, 4))
            b.buffer_store_cp(_c(b, 2 if k.mutant == "desc" else 1), self.desc_rs, b.select(valid, o, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)

    def w2_layer(self, layer: Value, nit: Value) -> None:
        b, k = self.b, self.k
        NA = k.na
        nu = b.readfirstlane(b.mul(nit, _c(b, NA)))
        trips, unit = self.w2_units(nu)
        with b.scf_for(_c(b, 0), trips, _c(b, 1), iv_name="w2u") as r:
            u = unit(b.readfirstlane(r))
            it = b.readfirstlane(b.div(u, _c(b, NA)))
            self.w2_gate_up(layer, it, b.readfirstlane(b.sub(u, b.mul(it, _c(b, NA)))))
        self.gbar(b.add(b.add(b.mul(self.L, _c(b, 2)), _c(b, 1)), layer), acquire=True)  # h complete (all XCDs)
        NG = H // TILE // k.w2t
        nd = b.readfirstlane(b.mul(nit, _c(b, NG)))
        trips, unit = self.w2_units(nd)
        with b.scf_for(_c(b, 0), trips, _c(b, 1), iv_name="w2d") as r:
            v = unit(b.readfirstlane(r))
            it = b.readfirstlane(b.div(v, _c(b, NG)))
            self.w2_down(layer, it, b.readfirstlane(b.sub(v, b.mul(it, _c(b, NG)))))

    def run_layer_experts(self, layer: Value, nit: Value) -> None:
        b, k = self.b, self.k
        if k.pipe == "stream":
            self.stream_layer(layer, nit)
            return
        if k.pipe == "w2":
            self.w2_layer(layer, nit)
            return
        NA = k.na
        A = b.readfirstlane(b.mul(nit, _c(b, NA)))
        C = b.readfirstlane(b.smax(b.smin(_c(b, GRID), b.div(A, _c(b, k.m0))), _c(b, 1)))
        if k.xh:
            L = b.add(b.mul(b.land(self.cta, _c(b, XCDS - 1)), _c(b, GRID // XCDS)), b.lshr(self.cta, _c(b, 3)))
        else:
            L = self.cta
        L = b.readfirstlane(L)
        act = b.cmp_lt(L, C)
        lo = b.readfirstlane(b.div(b.mul(L, A), C))
        hi = b.readfirstlane(b.div(b.mul(b.add(L, _c(b, 1)), A), C))
        hi = b.select(act, hi, lo)
        kf = b.readfirstlane(b.div(lo, _c(b, NA)))
        kl = b.readfirstlane(b.select(b.cmp_gt(hi, lo), b.add(b.div(b.sub(hi, _c(b, 1)), _c(b, NA)), _c(b, 1)), kf))
        with b.scf_for(kf, kl, _c(b, 1), iv_name="segi") as it:
            it = b.readfirstlane(it)
            base = b.mul(it, _c(b, NA))
            a0 = b.readfirstlane(b.smax(b.sub(lo, base), _c(b, 0)))
            a1 = b.readfirstlane(b.smin(b.sub(hi, base), _c(b, NA)))
            lf = b.div(b.sub(b.mul(b.add(base, _c(b, 1)), C), _c(b, 1)), A)
            ll = b.div(b.sub(b.mul(b.add(base, _c(b, NA)), C), _c(b, 1)), A)
            jseg = b.readfirstlane(b.sub(L, lf))
            nseg = b.readfirstlane(b.add(b.sub(ll, lf), _c(b, 1)))
            self.segment(layer, it, a0, a1, jseg, nseg)

    # ------------------------------------------------------------------ grid barriers (gemm.py / moe_lb striped)
    def gbar_arrive(self, bnd: Value, pre=None) -> None:
        b = self.b
        per = GRID // XCDS
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        if pre is not None:
            pre()
            b.sync_lds_only()
        if self.k.pst != "sc1":  # partials written back from this XCD's L2 before the arrival is counted
            with b.scf_if(b.cmp_eq(self.wave, _c(b, 0))):
                D.release_l2(b, "agent")
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(D.is_thread0(b)):
            mine = b.mul(b.add(row, b.land(self.cta, _c(b, XCDS - 1))), _c(b, 4))
            old = _aadd(self.k, b, gsa, mine, _c(b, 1), scope="agent", returns=True)
            target = b.mul(b.add(b.div(old, _c(b, per)), _c(b, 1)), _c(b, per))
            b.smem_store_vN(self.gsw, [_c(b, 0)], b.bitcast(target, F32), 1)

    def gbar_wait(self, bnd: Value, acquire: bool = False) -> None:
        b = self.b
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(b.cmp_lt(self.tid, _c(b, XCDS))):
            b.s_waitcnt(lgkmcnt=0)
            target = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.gsw, _c(b, 0), n=1), 0), I32)
            st = self.gpoll(gsa, b.mul(b.add(row, self.tid), _c(b, 4)), target)
            with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), D.is_thread0(b))):
                self.err()
            if acquire and self.k.mutant != "noacq":
                D.acquire_l2(b, "agent")
        b.sync_lds_only()

    def gpoll(self, base: Value, voff: Value, target: Value) -> Value:
        """The grid barrier's counter spin: ``gpc`` the poll loads' cache bits, ``gpause`` the pause per miss."""
        b, k = self.b, self.k
        if k.gpc == "sc0sc1" and k.gpause == "s_sleep1":
            return _pollge(k, b, base, voff, target, self.deadline(), pause="s_sleep1")
        from kernels.gfx950.qwen38_mono.ccl import wire as W
        s_ = "g"
        cp = {"sc0sc1": "sc0 sc1", "sc1": "sc1"}[k.gpc]
        lines = [
            "s_nop 4" if k.hz else "",
            "s_mov_b32 $2, 0",
            W._lbl(s_, "top") + ":",
            f"global_load_dword $0, $5, $6 {cp}",
            "s_waitcnt vmcnt(0)",
            "v_cmp_lt_u32_e64 $1, $0, $7",
            "s_and_b64 $1, $1, exec",
            "s_cbranch_scc0 " + W._lbl(s_, "ok"),
            *W._tail(s_, k.gpause, "$8", "$1", "$4", "$2", "$3"),
            W._lbl(s_, "ok") + ":",
            "s_mov_b32 $3, 0",
            "s_branch " + W._lbl(s_, "end"),
            W._lbl(s_, "timeout") + ":",
            "s_mov_b32 $3, 1",
            W._lbl(s_, "end") + ":",
        ]
        _v, _m, _it, st, _now = b.inline_asm_multi(
            "\n".join(x for x in lines if x), "=&v,=&s,=&s,=&s,=&s,v,s,s,v,~{scc},~{memory}",
            [voff, base, target, self.deadline()], result_types=[I32, I64, I32, I32, I64], result_name_hint="pollg")
        return st

    def gbar(self, bnd: Value, acquire: bool = False, pre=None) -> None:
        self.gbar_arrive(bnd, pre)
        self.gbar_wait(bnd, acquire)

    # ------------------------------------------------------------------ combine
    def flow_par(self, layer: Value) -> Value:
        b = self.b
        return b.readfirstlane(b.land(b.add(self.ep, layer), _c(b, 1)))

    def flow_bump(self, layer: Value, S: dict, tk: dict) -> None:
        """cmb=flow: this segment's partials and desc stored (every wave), then + 12 / nseg on each token's counter."""
        b, k = self.b, self.k
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        inc = b.readfirstlane(b.div(_c(b, 12), S["nseg"]))
        if k.shd:
            inc = b.readfirstlane(b.select(b.cmp_eq(S["dh"], _c(b, 0)), inc, b.lshr(inc, _c(b, 1))))
        with b.scf_if(b.land(b.land(b.cmp_eq(self.wave, _c(b, 0)), b.cmp_eq(self.g, _c(b, 0))), b.land(tk["valid"], S["has"]))):
            o = b.mul(b.add(_c(b, W_TOKC), b.add(b.mul(self.flow_par(layer), _c(b, 128)), tk["tok"])), _c(b, 4))
            _aadd(k, b, self.cnt_a, o, inc, scope="agent")

    def combine(self, layer: Value) -> None:
        """Units (token, 256 columns) over all waves: y = sum over slots (order) of the slot's segment partials (order)."""
        b, k = self.b, self.k
        y_a = b.add(self.a["y"], b.mul(_i64(b, layer), self.ly))
        y_rs = _rsrc(b, y_a, _c(b, k.bucket * H * 2))
        ncu = b.readfirstlane(b.mul(self.rows, _c(b, H // 256)))
        gw = b.add(b.mul(self.cta, _c(b, WAVES)), self.wave)
        with b.scf_for(gw, ncu, _c(b, GRID * WAVES), iv_name="cu") as u:
            u = b.readfirstlane(u)
            tok = b.div(u, _c(b, H // 256))
            cb = b.sub(u, b.mul(tok, _c(b, H // 256)))
            col = b.add(b.mul(cb, _c(b, 256)), b.mul(self.lane, _c(b, 4)))
            fl = k.cmb == "flow"
            if fl:  # wait for the token's 11 slots (132 = 11 x 12), then read coherently
                par = self.flow_par(layer)
                cw = b.mul(b.add(_c(b, W_TOKC), b.add(b.mul(par, _c(b, 128)), tok)), _c(b, 4))
                with b.scf_if(self.lane0):
                    tgt_ = {"flow": 10 * 12, "flow0": 0}.get(k.mutant, 11 * 12)  # mutants: one slot early / no wait
                    st = self.gpoll(self.cnt_a, cw, _c(b, tgt_))
                    with b.scf_if(b.cmp_ne(st, _c(b, 0))):
                        self.err()
                with b.scf_if(b.land(self.lane0, b.cmp_eq(cb, _c(b, 0)))):  # the next layer's counter of this token
                    nw = b.mul(b.add(_c(b, W_TOKC), b.add(b.mul(b.sub(_c(b, 1), par), _c(b, 128)), tok)), _c(b, 4))
                    b.buffer_store_cp(_c(b, 0), self.cnt_rs, nw, _c(b, 0), aux=CPOL_SC1)
            rdx = CPOL_SC1 if fl else 0
            dv = b.buffer_load_cp(self.desc_rs, b.mul(b.add(b.mul(tok, _c(b, DSTR)), b.smin(self.lane, _c(b, SLOTS - 1))), _c(b, 4)), _c(b, 0),
                                  dwords=1, aux=rdx)
            ns = [b.readlane(dv, _c(b, s_)) for s_ in range(SLOTS)]
            if k.mutant == "desc":
                ns = [_c(b, 1)] * SLOTS
            # the slot weights (route pairs: slot 10 = the shared gate), applied once per slot after its segment sum
            pr_a = b.readfirstlane(b.add(self.a["pr"], b.mul(_i64(b, layer), self.lpr)))
            pr_rs = _rsrc(b, pr_a, _c(b, k.bucket * PRS * 8))
            wv = b.buffer_load_cp(pr_rs, b.add(b.mul(b.add(b.mul(tok, _c(b, PRS)), b.smin(self.lane, _c(b, SLOTS - 1))), _c(b, 8)), _c(b, 4)),
                                  _c(b, 0), dwords=1, aux=CPOL_SC1 if k.entry == "logits" else 0)
            ws = [b.bitcast(b.readlane(wv, _c(b, s_)), F32) for s_ in range(SLOTS)]
            mx = ns[0]
            for v in ns[1:]:
                mx = b.smax(mx, v)

            def ld(s_: int, j: int, n_s: Value, colv: Value, wd: int):
                o = b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, SLOTS)), _c(b, s_)), _c(b, MAXSEG)), _c(b, j)), _c(b, H)), colv), _c(b, 4))
                live = b.cmp_lt(_c(b, j), n_s)
                if k.mutant == "seg" and j > 0:
                    live = b.land(live, b.cmp_lt(_c(b, j), b.sub(n_s, _c(b, 1))))
                v = b.buffer_load_cp(self.part_rs, b.select(live, o, _c(b, OOB)), _c(b, 0), dwords=wd, aux=rdx)
                return b.vec_bitcast(v, VectorType(F32, wd)), live

            def total(nj: int, colv: Value, wd: int, groups=(range(SLOTS),)):
                """Slot order kept across the load groups (fewer live registers per group)."""
                tot = None
                for grp in groups:
                    slots = [s_ for s_ in grp if not (k.mutant == "slot" and s_ == TOPK - 1)]
                    tot = total_group(slots, nj, colv, wd, tot)
                return tot

            def total_group(slots, nj: int, colv: Value, wd: int, tot):
                parts = {s_: [ld(s_, j, ns[s_], colv, wd) for j in range(nj)] for s_ in slots}
                for s_ in slots:
                    ssum = None
                    for v, live in parts[s_]:
                        vs = [b.vec_extract(v, i) for i in range(wd)]
                        ssum = vs if ssum is None else [b.select(live, b.fadd(ssum[i], vs[i]), ssum[i]) for i in range(wd)]
                    term = [b.fmul(ws[s_], ssum[i]) for i in range(wd)]
                    tot = term if tot is None else [b.fadd(tot[i], term[i]) for i in range(wd)]
                return tot

            G2 = (range(0, 6), range(6, SLOTS))

            def store(tot, colv: Value, wd: int):
                yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, wd // 2))
                if wd == 2:
                    yv = b.vec_extract(yv, 0)
                b.buffer_store_cp(yv, y_rs, b.mul(b.add(b.mul(tok, _c(b, H)), colv), _c(b, 2)), _c(b, 0))

            with b.scf_if_else(b.cmp_le(mx, _c(b, 2))) as (fast, slow):
                with fast:
                    store(total(2, col, 4), col, 4)
                with slow:
                    with b.scf_if_else(b.cmp_le(mx, _c(b, 3))) as (mid, deep):
                        with mid:
                            store(total(3, col, 4, G2), col, 4)
                        with deep:  # two passes of two-column lanes (128 columns each), every segment slot
                            for hh in range(2):
                                c2 = b.add(b.mul(cb, _c(b, 256)), b.add(_c(b, 128 * hh), b.mul(self.lane, _c(b, 2))))
                                store(total(MAXSEG, c2, 2, G2), c2, 2)

    def pretouch_parts(self, layer: Value) -> None:
        """Stale-read twin: the lines this wave's combine units read after the barrier, loaded (plain) before it."""
        b = self.b
        ncu = b.readfirstlane(b.mul(self.rows, _c(b, H // 256)))
        gw = b.add(b.mul(self.cta, _c(b, WAVES)), self.wave)
        loop = b.scf_for_iter(gw, ncu, _c(b, GRID * WAVES), [("pta", _c(b, 0))], iv_name="ptc")
        with loop as (u, vals):
            u = b.readfirstlane(u)
            tok = b.div(u, _c(b, H // 256))
            cb = b.sub(u, b.mul(tok, _c(b, H // 256)))
            col = b.add(b.mul(cb, _c(b, 256)), b.mul(self.lane, _c(b, 4)))
            acc = vals[0]
            for s_ in range(SLOTS):
                for j in range(2):
                    o = b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, SLOTS)), _c(b, s_)), _c(b, MAXSEG)), _c(b, j)), _c(b, H)), col), _c(b, 4))
                    acc = b.xor(acc, b.vec_extract(b.buffer_load_cp(self.part_rs, o, _c(b, 0), dwords=4), 0))
            acc = b.xor(acc, b.buffer_load_cp(self.desc_rs, b.mul(b.add(b.mul(tok, _c(b, DSTR)), b.smin(self.lane, _c(b, SLOTS - 1))), _c(b, 4)), _c(b, 0), dwords=1))
            b.scf_yield(acc)
        b.s_waitcnt(vmcnt=0)
        with b.scf_if(b.land(self.lane0, b.cmp_eq(loop.results[0], _c(b, 0x5EED1234)))):
            self.err()

    # ------------------------------------------------------------------ main
    def build(self) -> KernelDef:
        b, k = self.b, self.k
        zero = _c(b, 0)
        with b.scf_for(zero, self.L, _c(b, 1), iv_name="layer") as layer:
            layer = b.readfirstlane(layer)
            with b.scf_if(b.land(self.gsync, b.cmp_gt(layer, zero))):
                self.gbar(b.mul(b.sub(layer, _c(b, 1)), _c(b, 2)), acquire=False)
            if k.xsync == "cnt":
                with b.scf_if(D.is_thread0(b)):
                    self.lds_put(self.xcnt, _c(b, 0), _c(b, 0))
            b.sync_lds_only()
            self.stamp(layer, 0)
            if k.entry == "logits":
                self.route(layer)
            nit = self.sort(layer)
            if k.pretouch == 2:
                self.pretouch_parts(layer)
            self.stamp(layer, 1)
            self.run_layer_experts(layer, nit)
            if k.prio == "all":
                b.s_setprio(0)
            self.stamp(layer, 2)
            if not self.dg("nocmb"):
                pre = (lambda: self.pretouch_parts(layer)) if k.pretouch == 1 else None
                if k.cmb == "flow":
                    b.s_waitcnt(vmcnt=0)
                elif k.mutant != "bar":
                    bnd_ = b.add(b.mul(layer, _c(b, 2)), _c(b, 1))
                    self.gbar_arrive(bnd_, pre)
                    self.stamp(layer, 5)  # the partial stores drained, the arrival counted
                    self.gbar_wait(bnd_, True)
                self.stamp(layer, 3)
                self.combine(layer)
            self.stamp(layer, 4)
            b.sync_lds_only()
        b.s_waitcnt(vmcnt=0)
        with b.scf_if(D.is_thread0(b)):
            old = _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_END), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, GRID - 1))):
                _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_EP), self.L, scope="agent")
                _aadd(self.k, b, self.cnt_a, _c(b, 4 * W_END), _c(b, -GRID), scope="agent")
                if k.deal == "dyn":  # both ticket counters zero for the next launch (whatever its epoch parity)
                    for w_ in (W_TICK, W_TICK + 1):
                        b.buffer_store_cp(_c(b, 0), self.cnt_rs, _c(b, 4 * w_), _c(b, 0), aux=CPOL_SC1)
        return b.kernel


def _flat2(ring):
    """[(w, s), ..] -> (values, n)."""
    vals = []
    for w, s_ in ring:
        vals += [w, s_]
    return vals, len(ring)


def _unflat2(vals, n):
    return [(vals[2 * i], vals[2 * i + 1]) for i in range(n)]


def build_moe2_kernel(k: Moe2Knobs) -> KernelDef:
    return _Moe2(k).build()
