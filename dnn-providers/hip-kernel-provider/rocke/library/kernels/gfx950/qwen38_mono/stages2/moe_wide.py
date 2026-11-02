# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP2d: the fused rocKE MoE (gate_up + down + combine, expert-union dedupe) for the grouped / wide builds.

The served hybrid links the fused MoE (:mod:`..moe_fused`) only at rows <= 4 and only for ungrouped builds; at the
grouped / wide builds (``r4g2``, ``r6g3``, ``r8g4``, ``r9g3``, ``r12g3``, ``r12g4``, ``r16g4``, ...: rows / S requests
of S verify rows) it runs FlyDSL's ``moe_wide`` M3 / M4 / M5, whose M4 loads every (row, rank)'s gate_up weights
separately (161 expert loads at 16 rows, no union), splits each into 2 x 64-column tasks in rounds behind CTA
barriers, and whose M5 streams every (row, rank) chain's down weights again per hidden tile.

This module is the true-mono port for those builds (PLAN decision 16; ``wave2/WP2d.md``):

* ``rows <= 4`` grouped builds (``r4g2``): the fused body of :mod:`..moe_fused` itself (the MoE does not depend on
  the grouping: every row is independent), emitted under the grouped variant's symbol (:func:`write_functions`).
* ``rows > 4`` (wide): :func:`emit_wide`, the fused design generalized to 16 rows. Every request row is an MFMA A row
  (16 x 16 x 128 scaled MFMA: lane ``l`` supplies row ``l % 16``, holds output rows ``4 (l // 16) .. + 3``), so one
  weight tile serves every row routed to its expert; the routed slots are the union of the rows' top-10 by expert id
  (wave 0, an LDS bitmap in passes of 64 picks). Per CTA the gate_up units come in *rounds* of ``2 gu_round`` units
  (``gu_round`` chains a wave: the narrow body holds every unit at once, 2 at most, which 16 rows exceed), the next
  round's weights issued right after this round's MFMAs (``gu_next``); the epilogue runs only the live (unit, row)
  entries (``epi=compact``: a per-CTA list built after routing) instead of every row of every unit. The down units come
  in rounds of ``dn_round`` units a wave, the next round issued before this round's MFMAs; the exchange reads each
  slot's live rows once. The combine is the narrow one (32 threads a row, 16 rows = 512 threads).

Numerics are the mono's bit for bit (and FlyDSL ``moe_wide``'s, whose per-row math is ``moe.py``'s): the top-10, the
router weights, each n-block's 20-step gate_up chain, the routed / shared SiLU, the MXFP8 rule, the 5-step routed and
merge-ordered shared down chains, the rank-ordered combine. The router (M3) stays FlyDSL's ``moe_wide.m3`` in this
form: the body runs in the M4 slot (``m4``), ``m5`` is an empty function (``router`` knob: the M3 slot is not built
yet at rows > 4).

Exchange and outputs: as :mod:`..moe_fused` (the mono's ``dn`` mailbox for the gate_up -> down hand-off, the private
``ys`` pairs in ``ss`` under ``tag ^ 2^31``, ``h_out`` stored by the combine). The wide scratch holds every row's
regions (ATOM ``layout.mailbox_bytes(rows)``).

Knobs: :class:`WideKnobs`. Generator::

    python -m kernels.gfx950.qwen38_mono.stages2.moe_wide --out DIR --variants r8g4,r16g4 [--knobs '{...}']
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, fields, replace

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.helpers import mailbox as M

from .. import abi as A
from .. import moe as MO
from .. import moe_fused as MF

WIDE_ROWS = 16  # ATOM layout_wide.WIDE_ROWS
RS = 16  # slot_r stride: one word per request row
MAXS = WIDE_ROWS * A.TOP_K + 1  # slots: the shared one, then up to 160 routed
GU_UNITS = MF.GU_UNITS
DN_UNITS = MF.DN_UNITS
STAGES = ("m3", "m4", "m5")

# Test-only mutants (must make the layer differ from the mono):
#   combine_swap: ranks 0 and 1 swapped in the combine order (the narrow mutant)
#   gred_row: the gate_up result of output row 4 g + v stored as row 4 g + (v + 1) % 4 (the wide lane-group map)
#   ys_row: a down result put for the row of the next lane group (the wide lane-group map of the down phase)
#   slot_rank: a row's rank in a slot read from the next row's word (the 16-row slot table)
#   epi_drop: the compact epilogue skips the CTA's first live (unit, row) entry (the compaction; with dbg=1 the
#     missing hand-off times out instead of hanging the layer)
#   numa_copy: (knob numa) the area puts store copy 0 only (a consumer homed at the other IOD's copy never sees
#     the hand-off: with dbg=1 its deadline poll times out)
MUTANTS = ("combine_swap", "gred_row", "ys_row", "slot_rank", "epi_drop", "numa_copy")

# The wide scratch's NUMA area (ATOM ``numa.wide_header_offset`` / ``WIDE_LOGICAL``, env ``ATOM_Q38_NUMA_WIDE``):
# the header past the wide mailbox part and every measurement region (the narrow header's formula with the wide
# mailbox part), the area at the next 2 MiB boundary past header + 256. Its logical bytes hold this body's own
# regions only, compact (knob ``numa``): every row's ``dn`` exchange (:data:`NUMA_DN_ROW` pairs a row), then every
# row's ys (:data:`MF.YS_BYTES` a row); byte x at x + (x & ~8191) (copy 0) and + 8192 (copy 1).
WIDE_NUMA_HDR = -(-(MF.NUMA_HDR - A.mailbox_bytes(A.MAX_ROWS) + A.mailbox_bytes(WIDE_ROWS)) // 256) * 256
WIDE_NUMA_LOGICAL = 8 << 20
NUMA_DN_ROW = (A.SC["qp"] - A.SC["dn"]) // A.PAIR
NUMA_YS0 = WIDE_ROWS * NUMA_DN_ROW
NUMA_YS_ROW = MF.YS_BYTES // A.PAIR
assert (NUMA_YS0 + WIDE_ROWS * NUMA_YS_ROW) * A.PAIR <= WIDE_NUMA_LOGICAL
assert A.DN_ROW * (A.TOP_K + 1) <= NUMA_DN_ROW
NUMA_MODES = ("off", "dn", "ys", "both")


@dataclass(frozen=True)
class WideKnobs:
    """Knobs of the wide fused MoE (first value = default; every value keeps the bits).

    map / iod_split / sh_ctas / sh_base / line_impl / line_order / combine_base / w_cpol / mailbox / a_pre /
    gu_apipe / xchg_pre / extras / tw_at / guard: as :class:`..moe_fused.FusedKnobs` (the byte line of gate_up and
    down units per CTA, the shared group, the load cache policy, the mailbox implementation, ...).
    gu_round: gate_up chains a wave per round (a round = ``2 gu_round`` units of the CTA).
    gu_next: when a round issues the next round's gate_up weights: ``mfma`` (right after its MFMAs, before the
      epilogue) | ``epi`` (after its epilogue) | ``first`` (rounds 0 and 1 at once, then each round's next-but-one
      after its MFMAs: two rounds in flight) | ``pipe`` (each k_iter group of the next round right after this round's
      same group's MFMAs, into the registers they free: one round in flight across the epilogue, no extra VGPRs).
    epi: ``compact`` (the epilogue runs only the (unit, row) entries whose row routes to the unit's slot, from a
      per-CTA list) | ``dense`` (every row of every unit, the dead ones masked; the narrow body's form).
    dn_round: down units a wave per round.
    dn_early: down rounds issued before the gate_up rounds (0 | 1: with the first gate_up round).
    order: per wave, ``route`` (route its rows, then fill the MFMA A rows) | ``fill`` (fill first).
    tw_at: when the combine CTAs compute the router weights: ``late`` (from the top-10 keys, before the gate_up
      MFMAs) | ``route`` (with the top-10) | ``combine`` (from the keys, right before the combine: off the gate_up
      issue path).
    elist_at: where wave 0 builds the compact epilogue list: ``issue`` (after the first gate_up round's and the early
      down units' weight issue: the list is needed only at round 0's epilogue) | ``route`` (right after the routing,
      ahead of wave 0's weight issue).
    xchg_pre / combine_base: as FusedKnobs.
    router: the body runs the router (M3) itself (not built at rows > 4: the M4 slot behind FlyDSL's M3).
    phase_stamps: measurement only (FusedKnobs).
    mutate: test only (:data:`MUTANTS`).
    numa: the wide scratch's own NUMA area (:data:`WIDE_NUMA_HDR`; ATOM ``ATOM_Q38_NUMA_WIDE=1``, sidecar
      ``numa_wide_hdr``): ``off`` | ``dn`` (the gate_up -> down exchange: both copies stored, the copy homed at the
      consumer's IOD polled) | ``ys`` (the down outputs for the combine, likewise) | ``both``.
    ballot: how the body's wave ballots are emitted (the slot union, the compact epilogue list, the shared helpers it
      calls): ``ir`` (``tile.wave_ballot``, LLVM's ``amdgcn.ballot``) | ``asm`` (``v_cmp_ne_u32_e64`` into an SGPR
      pair, ``s_nop 4`` after it: the direct code object backend lowers no ``tile.wave_ballot``).
    dbg: diagnostics only (wrong numerics allowed): 1 = every poll is a deadline poll (:data:`DBG_MAXIT` retries);
      a wave whose poll times out writes the poll site's id to its debug word (:func:`dbg_offset`), the CTA's phase
      stamps are on (``phase_stamps``): a hang becomes a finished launch whose scratch says where it waited.
    """

    map: str = "line"
    iod_split: int = 1
    sh_ctas: int = 16
    sh_base: int = 160
    line_impl: str = "const"
    line_order: str = "slot"
    combine_base: int = 0
    w_cpol: int = 18
    mailbox: M.MailboxKnobs = field(default_factory=lambda: M.MailboxKnobs(impl="ir"))
    a_pre: bool = True
    gu_apipe: int = 1
    xchg_pre: str = "hint"
    extras: str = "first"
    tw_at: str = "late"
    guard: str = "none"
    gu_round: int = 1
    gu_next: str = "mfma"
    epi: str = "compact"
    dn_round: int = 4
    dn_early: int = 1
    order: str = "route"
    elist_at: str = "issue"
    router: bool = False
    phase_stamps: bool = False
    mutate: str = ""
    dbg: int = 0
    numa: str = "off"
    ballot: str = "ir"

    def __post_init__(self) -> None:
        if self.dbg not in (0, 1):
            raise ValueError("dbg must be 0 or 1")
        if self.ballot not in ("ir", "asm"):
            raise ValueError(f"ballot must be ir|asm, got {self.ballot!r}")
        if self.numa not in NUMA_MODES:
            raise ValueError(f"numa must be one of {NUMA_MODES}, got {self.numa!r}")
        if self.mutate == "numa_copy" and self.numa not in ("dn", "both"):
            raise ValueError("mutate=numa_copy needs the dn exchange in the area (numa dn | both)")
        if not 1 <= self.gu_round <= 2:
            raise ValueError("gu_round must be 1 or 2")
        if self.gu_next not in ("mfma", "epi", "first", "pipe"):
            raise ValueError(f"gu_next must be mfma|epi|first|pipe, got {self.gu_next!r}")
        if self.gu_next == "first" and self.gu_round != 1:
            raise ValueError("gu_next=first holds two rounds in flight: gu_round 1 only")
        if self.epi not in ("compact", "dense"):
            raise ValueError(f"epi must be compact|dense, got {self.epi!r}")
        if not 1 <= self.dn_round <= 8:
            raise ValueError("dn_round must be in 1..8")
        if self.dn_early not in (0, 1):
            raise ValueError("dn_early must be 0 or 1")
        if self.order not in ("route", "fill"):
            raise ValueError(f"order must be route|fill, got {self.order!r}")
        if self.tw_at not in ("late", "route", "combine"):
            raise ValueError(f"tw_at must be late|route|combine, got {self.tw_at!r}")
        if self.elist_at not in ("issue", "route"):
            raise ValueError(f"elist_at must be issue|route, got {self.elist_at!r}")
        if self.router:
            raise ValueError("router=True (the body in the M3 slot) is not built at rows > 4")
        if self.mutate not in ("",) + MUTANTS:
            raise ValueError(f"mutate must be '' or one of {MUTANTS}, got {self.mutate!r}")
        self.fused()  # FusedKnobs' own checks of the shared knobs

    def fused(self) -> MF.FusedKnobs:
        """The :class:`FusedKnobs` the reused emitters read (no router; the M4 slot)."""
        return MF.FusedKnobs(
            map=self.map, iod_split=self.iod_split, sh_ctas=self.sh_ctas, sh_base=self.sh_base,
            dn_issue="after_epi", dn_inflight=0, combine_base=self.combine_base, extras=self.extras,
            tw_at="route" if self.tw_at == "route" else "late", line_impl=self.line_impl, slots="table",
            w_cpol=self.w_cpol, mailbox=self.mailbox, a_pre=self.a_pre, gu_interleave=True, gu_apipe=self.gu_apipe,
            router=False, phase_stamps=self.phase_stamps or bool(self.dbg), guard=self.guard, xchg_pre=self.xchg_pre,
            combine="task", line_order=self.line_order,
            mutate="combine_swap" if self.mutate == "combine_swap" else "",
            xchg_home="numa2" if self.numa in ("dn", "both") else "single",
            ys_home="numa2" if self.numa in ("ys", "both") else "single",
            numa_hdr=WIDE_NUMA_HDR if self.numa != "off" else 0,
        )  # fmt: skip

    def suffix(self) -> str:
        d = WideKnobs()
        parts = []
        for f in fields(self):
            if f.name == "mailbox":
                continue
            v = getattr(self, f.name)
            if v != getattr(d, f.name):
                parts.append(f"{f.name}{v}".replace("_", ""))
        if self.mailbox != d.mailbox:
            parts.append("mb" + (self.mailbox.suffix() or "default"))
        return "_".join(parts)


def knobs_spec(k: WideKnobs) -> str:
    """``k=v,...`` of the knobs off their defaults (mailbox as ``mb_impl`` / ``mb_pause``)."""
    d = WideKnobs()
    out = []
    for f in fields(k):
        if f.name == "mailbox":
            continue
        v = getattr(k, f.name)
        if v != getattr(d, f.name):
            out.append(f"{f.name}={int(v) if isinstance(v, bool) else v}")
    if k.mailbox.impl != d.mailbox.impl:
        out.append(f"mb_impl={k.mailbox.impl}")
    if k.mailbox.pause != d.mailbox.pause:
        out.append(f"mb_pause={k.mailbox.pause}")
    return ",".join(out)


def knobs_from_spec(spec: str) -> WideKnobs:
    """Inverse of :func:`knobs_spec` (``""``: the defaults)."""
    kw = {}
    mb = {}
    types = {f.name: f.type for f in fields(WideKnobs)}
    for item in filter(None, (x.strip() for x in (spec or "").split(","))):
        name, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"knob spec item {item!r}: name=value")
        if name in ("mb_impl", "mb_pause"):
            mb[name[3:]] = val if name == "mb_impl" else val
            continue
        if name not in types:
            raise ValueError(f"unknown wide MoE knob {name!r}")
        t = str(types[name])
        if t in ("int", "<class 'int'>"):
            kw[name] = int(val)
        elif t in ("bool", "<class 'bool'>"):
            kw[name] = val not in ("0", "False", "false", "")
        else:
            kw[name] = val
    if mb:
        base = WideKnobs().mailbox
        pause = mb.get("pause", base.pause)
        if isinstance(pause, str) and pause.lstrip("-").isdigit():
            pause = int(pause)
        kw["mailbox"] = replace(base, impl=mb.get("impl", base.impl), pause=pause)
    return WideKnobs(**kw)


DBG_MAXIT = 1 << 18  # deadline-poll retries (``dbg=1``)
DBG_OFF = 200000  # byte offset of the debug words in request row 0's ``ss`` (past the ys pairs and phase stamps)


def dbg_offset(bid: int, wave: int) -> int:
    """Scratch byte offset of CTA ``bid`` wave ``wave``'s 2 debug words (``dbg=1``): a 64-bit mask, bit ``s`` set
    when poll site ``s`` (the build's ``sites`` list index; 63: any later) timed out on that wave."""
    return A.SC["ss"] + DBG_OFF + 8 * (bid * A.WAVES + wave)


# ============================================================ plan (host side)


def wide_m3(rows: int):
    """(first CTA, tasks) of FlyDSL's wide M3 (ATOM ``layout_wide.for_wide``): M1 / M2 CTAs past A1's, M3 next."""
    base_m1, m1, m2 = 96, A.HC_LOWRANK // 32 + A.HC_COUNT, A.HIDDEN // 32
    base_m3 = (base_m1 + rows * m1 + m2) % A.GRID
    m3_task_rows = rows // 2 if rows % 2 == 0 else rows
    return base_m3, rows // m3_task_rows * (A.EXPERTS // 8 + 1)


@dataclass(frozen=True)
class WidePlan(MF.FusedPlan):
    """:class:`FusedPlan` at 5 .. 16 rows (the line math is row-agnostic; U up to 10 R)."""

    def rot(self, li: int) -> int:
        if self.knobs.line_order != "router_dn":
            return 0
        base, tasks = wide_m3(self.rows)
        return self._rt_before((base + tasks) % A.GRID, li) % self.lines_n[li]

    @property
    def combine_base(self) -> int:
        if self.knobs.combine_base >= 0:
            return self.knobs.combine_base
        base, tasks = wide_m3(self.rows)
        return (base + tasks) % A.GRID

    def validate(self) -> None:
        if not A.MAX_ROWS < self.rows <= WIDE_ROWS:
            raise ValueError(f"wide rows must be in {A.MAX_ROWS + 1}..{WIDE_ROWS}, got {self.rows}")
        if self.experts % 8 or not 16 <= self.experts <= A.EXPERTS:
            raise ValueError(f"experts must be a multiple of 8 in 16..{A.EXPERTS}")


def lds_layout(plan: WidePlan, wk: WideKnobs):
    """(offsets, strides, words) of the wide body's LDS arena (int32 words)."""
    R = plan.rows
    bd = plan.bounds_cached
    table = [
        ("ids", A.TOP_K, R),
        ("tw", A.TOP_K, R),
        ("keys", A.TOP_K, R),
        ("rg", WIDE_ROWS, 1),
        ("res", 32, R),
        ("inj", 32, R),
        ("first", 4 * A.TOP_K, 1),  # the union bitmap (16 words) and its prefix (16)
        ("slot_e", MAXS + 3, 1),
        ("slot_r", (MAXS + 1) * RS, 1),
        ("nslot", 4, 1),
        ("kst", bd["gu"] + 4, 1),  # epi=compact: entries before unit k
        ("elist", max(bd["gu"], 1) * R + 4, 1),  # epi=compact: (unit, row) entries, 16 k + r
        ("gx8", A.HIDDEN // 4, R),
        ("gxe", A.HIDDEN // 32, R),
        ("gred", 2 * wk.gu_round * 64, R),
        ("dnA", max(bd["sl"], 1) * A.DN_ROW, R),
    ]
    if wk.dbg:
        table.append(("dbgm", 2 * A.WAVES, 1))
    off, stride, w = {}, {}, 0
    for name, words, rows in table:
        off[name] = w
        stride[name] = MO._pad_rows(words, rows)
        w += stride[name] * rows
    return off, stride, w


class WideCtx(MF.FusedCtx):
    """:class:`FusedCtx` over the wide arena (``R`` up to 16)."""

    def __init__(self, b: IRBuilder, plan: WidePlan, wk: WideKnobs, *, tag, layer, addr: dict):
        self.b = b
        self.fplan = plan
        self.wk = wk
        self.fk = plan.knobs
        self.plan = plan.moe
        self.k = self.plan.knobs
        self.R = plan.rows
        self.E = plan.experts
        self.tag = tag
        self.xtag = b.xor(tag, b.const_i32(MF.XTAG_BIT))
        self.layer = layer
        self.addr = addr
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = b.readfirstlane(b.lshr(self.tid, b.const_i32(6)))
        self.bid = b.readfirstlane(b.block_id_x())
        self.off, self.stride, words = lds_layout(plan, wk)
        if words > MO.LDS_LIMIT_WORDS:
            raise ValueError(f"LDS arena {words} words > {MO.LDS_LIMIT_WORDS}")
        self.lds_words = words
        self.lds = b.smem_alloc(I32, [words], name_hint="wmoe_lds")
        self.mb = M.mailbox_from_i64(b, addr["scratch"], b.const_i32(-1), self.k.mailbox)
        self.stamp_lds = None
        self.stage_ix = None
        self.mb_area = None
        self.sites: list = []  # dbg=1: (phase, n specs) per poll site, in emission order
        if wk.ballot == "asm":  # this builder's ballots (also those of the moe / moe_fused helpers the body calls)
            b.wave_ballot = lambda pred, _b=b: _asm_ballot(_b, pred)
        self._phase_now = 0
        if wk.numa != "off":  # the wide area: its base, the header's page mask with this CTA's IOD folded in
            a = b.add(addr["scratch"], b.const_i64(WIDE_NUMA_HDR + 256 + MF.NUMA_PAGE - 1))
            area = b.land(a, b.const_i64(-MF.NUMA_PAGE))
            self.mb_area = M.mailbox_from_i64(b, area, b.const_i32(-1), self.k.mailbox)
            hp = b.int_to_ptr(b.add(addr["scratch"], b.const_i64(WIDE_NUMA_HDR)), PtrType(I32, "global"))
            mask = b.readfirstlane(b.global_load(hp, b.const_i32(0), I32, align=4))
            self.sel_mask = b.readfirstlane(b.xor(mask, b.sub(b.const_i32(0), M.iod_id(b))))

    # ---------------------------------------------------- the wide NUMA area (knob numa): compact logical pairs
    def pix(self, region: str, i, row=0):
        if region == "dn" and self.mb_area is not None and self.fk.xchg_home == "numa2":
            return MF._AreaPix(self.add(self.mul(row, NUMA_DN_ROW), i))
        return super().pix(region, i, row)

    def ys_pix(self, r, rank, cp):
        if self.mb_area is None or self.fk.ys_home != "numa2":
            return super().ys_pix(r, rank, cp)
        return self.add(NUMA_YS0, self.mul(r, NUMA_YS_ROW), self.mul(rank, MF.YS_PAIRS), cp)

    def put(self, i, v):
        if isinstance(i, MF._AreaPix) and self.wk.mutate == "numa_copy":
            return M.put(self.b, self.mb_area, self._copy0(i.p), v, self.tag, self.k.mailbox)
        return super().put(i, v)

    def put2(self, i, v0, v1):
        if isinstance(i, MF._AreaPix) and self.wk.mutate == "numa_copy":
            return M.put2(self.b, self.mb_area, self._copy0(i.p), v0, v1, self.tag, self.k.mailbox)
        return super().put2(i, v0, v1)

    def phase(self, i: int) -> None:
        self._phase_now = i
        super().phase(i)

    def poll(self, specs, route=False):
        if not self.wk.dbg:
            return super().poll(specs, route)
        return self.dbg_poll([(self.mb_area, self._local(i.p), n) if isinstance(i, MF._AreaPix) else (self.mb, i, n)
                              for i, n in specs], self.tag)  # fmt: skip

    def poll_m(self, lst, tag):
        """``M.poll`` of ``[(mailbox, pair, n)]`` (the deadline form with ``dbg=1``)."""
        if not self.wk.dbg:
            return M.poll(self.b, lst, tag, self.k.mailbox)
        return self.dbg_poll(lst, tag)

    def dbg_poll(self, lst, tag):
        b = self.b
        self.sites.append((self._phase_now, len(lst)))
        site = len(self.sites) - 1  # bit index (site id - 1)
        if site >= 64:
            site = 63
        out = []
        worst = None
        for mb, i, n in lst:
            words = []
            for q in range(n):
                voff = b.mul(b.add(i, b.const_i32(q)), b.const_i32(M.PAIR_BYTES))
                w, it = _poll_pair_deadline(b, voff, mb.addr, tag, b.const_i32(DBG_MAXIT))
                words.append(w)
                worst = it if worst is None else b.smax(worst, it)
            out.append(words)
        with b.scf_if(b.land(b.cmp_ge(worst, b.const_i32(DBG_MAXIT)), b.cmp_eq(self.lane, b.const_i32(0)))):
            w = self.add(self.off["dbgm"], self.mul(self.wave, 2), site // 32)
            self.st(w, b.lor(self.ld(w), b.const_i32(1 << (site % 32))))
        return out

    def dbg_begin(self) -> None:
        if self.wk.dbg:
            with self.b.scf_if(self.eq(self.lane, 0)):
                for q in range(2):
                    self.st(self.add(self.off["dbgm"], self.mul(self.wave, 2), q), self.c(0))

    def dbg_end(self) -> None:
        """Lane 0 of every wave stores its 2 timed-out-site mask words (bit s - 1: site s) to :func:`dbg_offset`."""
        if not self.wk.dbg:
            return
        b = self.b
        with b.scf_if(self.eq(self.lane, 0)):
            ptr = b.int_to_ptr(self.addr["scratch"], PtrType(I32, "global"))
            for q in range(2):
                idx = self.add((A.SC["ss"] + DBG_OFF) // 4, self.mul(self.add(self.mul(self.bid, A.WAVES), self.wave), 2), q)
                b.global_store(ptr, idx, self.ld(self.add(self.off["dbgm"], self.mul(self.wave, 2), q)), align=4)


def _poll_pair_deadline(b, voff, base, tag, maxit):
    """``persist.sync.poll_pair_deadline`` with ``s_nop 4`` ahead of the asm VMEM (alola rule 176: a VALU write of
    the SGPR base, e.g. a ``v_readlane`` restoring a spilled SGPR, needs 5 wait states before an asm VMEM reads it)."""
    from rocke.core.ir import VectorType

    top = "${:private}wdp${:uid}_top"
    done = "${:private}wdp${:uid}_done"
    lines = [
        "s_nop 4",
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
        "s_nop 0",
        f"s_branch {top}",
        f"{done}:",
    ]
    pair, _tmp, _mask, it = b.inline_asm_multi(
        chr(10).join(lines), "=&v,=&v,=&s,=&s,v,s,s,s,~{vcc},~{scc}", [voff, base, b.zext(tag, I64), maxit],
        result_types=[VectorType(I32, 2), I64, I64, I32], result_name_hint="wdp",
    )  # fmt: skip
    return b.vec_extract(pair, 0), it


# ============================================================ IR pieces


def _per_wave_rows(e: WideCtx, fn) -> None:
    """``fn(r)`` for request rows ``wave, wave + 8, ..`` (< R) on this wave."""
    b = e.b
    for p in range(-(-e.R // A.WAVES)):
        r = e.add(e.wave, A.WAVES * p)
        if (p + 1) * A.WAVES <= e.R:
            fn(r)
        else:
            with b.scf_if(e.lt(r, e.R)):
                fn(r)


def _route_rows(e: WideCtx, weights: bool, keys: bool) -> None:
    """:func:`.moe._route_rows` with rows ``wave + 8 p``: each row's top 10 into ``ids`` (``tw`` / ``keys``)."""
    b = e.b
    j = e.imin(e.lane, e.E // 8 - 1)

    def one(r):
        got = e.poll([(e.pix("lg", e.mul(j, 4), r), 2), (e.pix("lg", e.add(e.mul(j, 4), 2), r), 2)], route=True)
        picks = MO.top10(e, got[0] + got[1])
        ws = MO.router_weights(e, picks) if weights else None
        with b.scf_if(e.eq(e.lane, 0)):
            for k in range(A.TOP_K):
                e.st(e.ix("ids", k, r), MO.key_expert(e, picks[k]))
                if ws is not None:
                    e.st(e.ix("tw", k, r), ws[k])
                if keys:
                    e.st(e.ix("keys", k, r), picks[k])

    _per_wave_rows(e, one)


def _late_weights(e: WideCtx) -> None:
    """:func:`.moe._late_weights` for rows ``wave + 8 p``."""
    b = e.b

    def one(r):
        picks = [e.ld_u(e.ix("keys", k, r)) for k in range(A.TOP_K)]
        ws = MO.router_weights(e, picks)
        with b.scf_if(e.eq(e.lane, 0)):
            for k in range(A.TOP_K):
                e.st(e.ix("tw", k, r), ws[k])

    _per_wave_rows(e, one)


def _extras(e: WideCtx, t) -> None:
    """:func:`.moe._m5_extras` for rows ``wave + 8 p``."""
    b = e.b

    def one(r):
        lane = e.lane
        bs = e.udiv(e.imin(lane, 31), 8)
        word = e.add(e.mul(bs, A.ROW_WORDS), e.mul(t, 8), e.umod(lane, 8))
        got = e.poll([(e.pix("sg", 0, r), 1), (e.pix("w", word, r), 1), (e.pix("minj", bs, r), 1)], route=True)
        with b.scf_if(e.eq(lane, 0)):
            e.st(e.add(e.off["rg"], r), got[0][0])
        with b.scf_if(e.lt(lane, 32)):
            e.st(e.ix("res", lane, r), got[1][0])
            e.st(e.ix("inj", lane, r), got[2][0])

    _per_wave_rows(e, one)


def _fill(e: WideCtx) -> None:
    _per_wave_rows(e, lambda r: MO._fill_x(e, r))


def _build_slots(e: WideCtx) -> None:
    """Wave 0 (after ``ids`` and a barrier): ``slot_e`` (slot 0 the shared expert, slot s + 1 the s-th routed
    expert by id), ``slot_r`` (slot * 16 + row: the row's rank, -1: none; the shared slot: 10) and ``nslot`` (U + 1);
    then a barrier. The union by id is :func:`.moe._union_by_id`'s, the picks in passes of 64 lanes."""
    b = e.b
    R, K = e.R, A.TOP_K
    n = R * K
    nw = -(-e.E // 32)
    bm = e.off["first"]
    se, sr = e.off["slot_e"], e.off["slot_r"]
    with b.scf_if(e.eq(e.wave, 0)):
        lane = e.lane
        nr = (MAXS + 1) * RS
        for q in range(-(-nr // 64)):
            i = e.add(lane, 64 * q)
            with b.scf_if(e.lt(i, nr)):
                e.st(e.add(sr, i), e.c(-1))
        with b.scf_if(e.lt(lane, nw)):
            e.st(e.add(bm, lane), e.c(0))
        lds0 = b.trunc(b.smem_addr_of(e.lds), I32)
        npass = -(-n // 64)
        mine = []
        for p in range(npass):
            t = e.imin(e.add(lane, 64 * p), n - 1)
            row, rank = e.udiv(t, K), e.umod(t, K)
            my = e.ld(e.ix("ids", rank, row))
            valid = e.lt(e.add(lane, 64 * p), n)
            with b.scf_if(valid):
                addr = e.add(lds0, e.mul(e.add(bm, b.lshr(my, e.c(5))), 4))
                bit = b.shl(e.c(1), b.land(my, e.c(31)))
                b.inline_asm("ds_or_b32 $0, $1\n\ts_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, bit], None)
            mine.append((my, valid, row, rank))
        word = e.ld(e.add(bm, e.imin(lane, nw - 1)))
        cnt = b.select(e.lt(lane, nw), MO._bcnt(b, word), e.c(0))
        inc = cnt
        for d in (1, 2, 4, 8):
            if d >= nw:
                break
            y = b.ds_bpermute(e.mul(e.imax(b.sub(lane, e.c(d)), 0), 4), inc)
            inc = e.add(inc, b.select(e.ge(lane, d), y, e.c(0)))
        with b.scf_if(e.lt(lane, nw)):
            e.st(e.add(bm, nw, lane), b.sub(inc, cnt))
        u = b.readlane(inc, e.c(nw - 1))
        for my, valid, row, rank in mine:
            w = b.lshr(my, e.c(5))
            below = b.land(e.ld(e.add(bm, w)), b.sub(b.shl(e.c(1), b.land(my, e.c(31))), e.c(1)))
            slot = e.add(e.ld(e.add(bm, nw, w)), MO._bcnt(b, below))
            with b.scf_if(valid):
                e.st(e.add(se, slot, 1), my)
                e.st(e.add(sr, e.mul(e.add(slot, 1), RS), row), rank)
        with b.scf_if(e.eq(lane, 0)):
            e.st(se, e.c(MO.SHARED))
            for r in range(R):
                e.st(sr + r, e.c(K))
            e.st(e.off["nslot"], e.add(u, 1))
    e.barrier()


def _slot_rank(e: WideCtx, is_sh, lo, local, r, uniform: bool):
    """Rank request row ``r`` has in line slot ``local`` (-1: none; the shared group: 10)."""
    if e.wk.mutate == "slot_rank":
        r = e.umod(e.add(r, 1), e.R)
    idx = e.add(e.off["slot_r"], e.mul(e.add(lo, local, 1), RS), r)
    rk = e.ld_u(idx) if uniform else e.ld(idx)
    return e.sel(is_sh, A.TOP_K, rk)


def _out_rows(e: WideCtx, acc, put):
    """``put(row, v)`` for every output row of an MFMA accumulator this lane holds: rows ``4 (lane // 16) + v``
    (``v`` < 4) below R. ``row`` is per lane, ``v`` static."""
    b = e.b
    g = e.udiv(e.lane, 16)
    for v in range(4):
        row = e.add(e.mul(g, 4), v)
        if e.wk.mutate == "gred_row":
            row = e.add(e.mul(g, 4), e.umod(e.add(e.c(v), 1), 4))
        with b.scf_if(e.lt(row, e.R)):
            put(row, b.vec_extract(acc, v))


def _mbcnt(b, mask):
    """Set bits of the i64 ballot ``mask`` below this lane. ``.moe._mbcnt`` with ``s_nop 4`` ahead of the asm: the
    mask's SGPRs are written by the VALU compare of the ballot right before, and LLVM's hazard recognizer does not
    look into inline asm (alola rule 185: without the wait states the counts were wrong on gfx950; here the compact
    epilogue list lost entries and the layer hung, WP2d debug job 68245633)."""
    lo = b.trunc(mask, I32)
    hi = b.trunc(b.lshr(mask, b.const_i64(32)), I32)
    return b.inline_asm(
        chr(10).join(["s_nop 4", "v_mbcnt_lo_u32_b32 $0, $1, 0", "v_mbcnt_hi_u32_b32 $0, $2, $0"]), "=&v,s,s",
        [lo, hi], I32, sideeffect=False, result_name_hint="wmbcnt",
    )  # fmt: skip


def _asm_ballot(b, pred):
    """``tile.wave_ballot`` as inline asm (knob ``ballot=asm``): the i64 mask of the active lanes whose ``pred`` (i32)
    is nonzero; ``s_nop 4`` after the VALU compare (its SGPR pair is read by SALU / VMEM / asm next, alola rule 185);
    convergent (the mask depends on EXEC: never moved across control flow)."""
    return b.inline_asm(chr(10).join(["v_cmp_ne_u32_e64 $0, 0, $1", "s_nop 4"]), "=s,v", [pred], I64,
                        sideeffect=True, convergent=True, result_name_hint="wballot")  # fmt: skip


def _bcnt_s(b, v):
    """``v_bcnt`` of an SGPR value with ``s_nop 4`` ahead (``v`` a ``readfirstlane`` result: a VALU write)."""
    return b.inline_asm(chr(10).join(["s_nop 4", "v_bcnt_u32_b32 $0, $1, 0"]), "=v,s", [v], I32, sideeffect=False,
                        result_name_hint="wbcnt")  # fmt: skip


def _build_elist(e: WideCtx, g0, g1, lo, is_sh) -> None:
    """``epi=compact``, wave 0: the live (unit k, row r) entries of the CTA's gate_up units (k < bound, in (k, r)
    order) as words ``16 k + r`` in ``elist``, and ``kst[k]`` = the entries before unit k (``kst[bound]`` = all).
    Ranks per row from the slot table (the shared group: every row); a unit past ``g1`` has no entries. Four units
    per ballot (lane ``16 j + r``: unit ``4 c + j``, row ``r``), so the list costs ``bound / 4`` ballots."""
    b = e.b
    bd = e.fplan.bounds_cached["gu"]
    lane = e.lane
    with b.scf_if(e.eq(e.wave, 0)):
        base = e.c(0)
        gu_last = e.imax(b.sub(g1, e.c(1)), 0)
        j = e.udiv(lane, 16)
        r16 = e.umod(lane, 16)
        r = e.imin(r16, e.R - 1)
        for c in range(-(-bd // 4)):
            k = e.add(e.c(4 * c), j)
            u = e.add(g0, k)
            valid = b.land(e.lt(u, g1), e.lt(k, bd))
            local = e.udiv(e.imin(u, gu_last), GU_UNITS)
            rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
            live = b.land(b.land(valid, e.lt(r16, e.R)), e.ge(rank, 0))
            mask = b.wave_ballot(b.select(live, e.c(1), e.c(0)))
            pos = e.add(base, _mbcnt(b, mask))
            with b.scf_if(live):
                e.st(e.add(e.off["elist"], pos), e.add(e.mul(k, 16), r16))
            lo32 = b.readfirstlane(b.trunc(mask, I32))
            hi32 = b.readfirstlane(b.trunc(b.lshr(mask, b.const_i64(32)), I32))
            # entries of units 4 c .. 4 c + jj - 1: the mask's lanes below 16 jj
            parts = [b.land(lo32, e.c(0xFFFF)), b.lshr(lo32, e.c(16)), b.land(hi32, e.c(0xFFFF)), b.lshr(hi32, e.c(16))]
            before = base
            for jj in range(4):
                kk = 4 * c + jj
                if kk < bd:
                    with b.scf_if(e.eq(lane, 0)):
                        e.st(e.off["kst"] + kk, before)
                before = b.readfirstlane(e.add(before, _bcnt_s(b, parts[jj])))
            base = before
        with b.scf_if(e.eq(lane, 0)):
            e.st(e.off["kst"] + bd, base)


def _ys_puts(e: WideCtx, acc_vals, ranks, tn) -> None:
    """Rows of an accumulator (``acc_vals[v]``: this lane's value of row ``4 (lane // 16) + v``): lanes ``l % 16``
    even put (own column, next column) of their row at its rank (``ranks[r]``: uniform per row r; -1: none)."""
    b = e.b
    g = e.udiv(e.lane, 16)
    ln = e.umod(e.lane, 16)
    cp = e.add(e.mul(tn, 8), e.udiv(ln, 2))
    even = e.eq(e.umod(ln, 2), 0)
    for v in range(4):
        val = acc_vals[v]
        nxt = b.bitcast(b.dpp_xor(b.bitcast(val, I32), 1), F32)
        rg = g
        if e.wk.mutate == "ys_row":
            rg = e.umod(e.add(g, 1), 4)
        row = e.add(e.mul(rg, 4), v)
        rank = e.c(-1)
        for gg in range(4):
            rr = 4 * gg + v
            if rr < e.R:
                rank = e.sel(e.eq(rg, gg), ranks[rr], rank)
        ok = b.land(b.land(even, e.lt(row, e.R)), e.ge(rank, 0))
        with b.scf_if(ok):
            MF._ys_put(e, e.ys_pix(e.imin(row, e.R - 1), e.imax(rank, 0), cp), M.bf16_pair(b, val, nxt))


def _dn_compute(e: WideCtx, dns, lo, is_sh, r_l) -> None:
    """The down units' chains (routed: k_iters 0..4 from zero; shared: the mono's merge order) -> ys."""
    b = e.b
    R = e.R
    for du in dns:
        valid, tn = du["valid"], du["tn"]
        shift = e.mul(e.umod(tn, 2), 8)
        row = e.ix("dnA", e.mul(du["sl"], A.DN_ROW), r_l)
        ranks = [_slot_rank(e, is_sh, lo, du["local"], r, uniform=True) for r in range(R)]
        with b.scf_if(b.land(valid, b.lnot(du["shared"]))):
            unit = du["unit"]
            ops = MO._dn_ops(e, row)
            acc = b.zero_vec_f32(4)
            for kk in range(A.DN_STEPS):
                acc = MO._dn_mma(e, acc, unit, kk, row, shift, ops)
            _ys_puts(e, [MO.bf16r(b, b.vec_extract(acc, v)) for v in range(4)], ranks, tn)
        with b.scf_if(b.land(valid, du["shared"])):
            unit = du["unit"]
            ops = MO._dn_ops(e, row)
            zero = b.zero_vec_f32(4)
            w0 = MO._dn_mma(e, MO._dn_mma(e, zero, unit, 0, row, shift, ops), unit, 4, row, shift, ops)
            w1 = MO._dn_mma(e, zero, unit, 1, row, shift, ops)
            w2 = MO._dn_mma(e, zero, unit, 2, row, shift, ops)
            w3 = MO._dn_mma(e, zero, unit, 3, row, shift, ops)
            vals = []
            for v in range(4):
                x = [b.vec_extract(w, v) for w in (w0, w1, w2, w3)]
                vals.append(MO.bf16r(b, b.fadd(b.fadd(b.fadd(x[0], x[1]), x[2]), x[3])))
            _ys_puts(e, vals, [e.c(A.TOP_K)] * R, tn)


def _exchange(e: WideCtx, sl0, d1, n_dn, ul, lo, is_sh, SL: int) -> None:
    """:func:`.moe_fused._exchange` with the 16-row slot table: the ``dn`` rows of every slot this CTA's down units
    read (every row the slot serves) into LDS ``dnA``."""
    b, wk = e.b, e.wk
    R = e.R
    n_sl = e.sel(e.lt(e.c(0), n_dn), e.add(b.sub(e.udiv(e.imax(b.sub(d1, e.c(1)), 0), DN_UNITS), sl0), 1), 0)
    items = SL * R

    def item_rank(m):
        sl = e.udiv(m, R)
        r = e.umod(m, R)
        local = e.imin(e.add(sl0, sl), e.imax(b.sub(ul, e.c(1)), 0))
        rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
        return b.land(e.lt(sl, n_sl), e.ge(rank, 0)), sl, r, rank

    if wk.xchg_pre == "hint":
        MF._xchg_hint(e, items, item_rank)
    nd = items * (A.DN_ROW // 2)
    for m0 in range(0, nd, A.THREADS):
        m = e.imin(e.add(e.tid, m0), nd - 1)
        it = e.udiv(m, A.DN_ROW // 2)
        jp = e.umod(m, A.DN_ROW // 2)
        ok, sl, r, rank = item_rank(it)
        with b.scf_if(b.land(ok, e.lt(e.add(e.tid, m0), nd))):
            got = e.poll([(e.pix("dn", e.add(e.mul(rank, A.DN_ROW), e.mul(jp, 2)), r), 2)])
            e.st(e.ix("dnA", e.add(e.mul(sl, A.DN_ROW), e.mul(jp, 2)), r), b.vec_pack(got[0], I32), 2)
    e.barrier()


# ============================================================ the body


def emit_wide(e: WideCtx) -> None:
    b, fp, fk, wk = e.b, e.fplan, e.fk, e.wk
    R = e.R
    bd = fp.bounds_cached
    JR = wk.gu_round
    UPR = 2 * JR  # gate_up units a round
    NR = -(-max(bd["gu"], 1) // UPR)
    e.dbg_begin()
    e.phase(0)
    e.phase(1)
    is_sh, q, t_rt, n_rt, iod = MF.emit_role(e)
    ct = e.umod(e.add(e.bid, A.GRID - fp.combine_base), A.GRID)
    has_comb = e.lt(ct, DN_UNITS)
    routed = b.lnot(is_sh)

    def route():
        with b.scf_if(routed):
            _route_rows(e, weights=wk.tw_at == "route", keys=wk.tw_at != "route")

    if wk.order == "fill":
        _fill(e)
        route()
    else:
        route()
        _fill(e)
    if wk.extras == "first":
        with b.scf_if(b.land(has_comb, routed)):
            _extras(e, ct)
    e.barrier()
    with b.scf_if(routed):
        _build_slots(e)
    e.phase(2)
    U = b.sub(e.ld_u(e.off["nslot"]), e.c(1))
    lo, ul, g0, g1, d0, d1 = MF.emit_work(e, U, is_sh, q, t_rt, n_rt, iod)
    n_gu = b.sub(g1, g0)
    n_dn = b.sub(d1, d0)
    if wk.epi == "compact" and wk.elist_at == "route":
        _build_elist(e, g0, g1, lo, is_sh)
    # ---- gate_up rounds: chain j of round n on wave w is CTA unit k = n UPR + 2 j + w // 4, block w % 4
    half = e.udiv(e.wave, 4)
    w4 = e.umod(e.wave, 4)
    pair_in = e.umod(w4, 2)
    part = e.udiv(w4, 2)
    gu_last = e.imax(b.sub(g1, e.c(1)), 0)

    def chains_of(n):
        out = []
        for j in range(JR):
            k = e.add(half, n * UPR + 2 * j)
            u = e.add(g0, k)
            valid = e.lt(u, g1)
            uc = e.imin(u, gu_last)
            local = e.udiv(uc, GU_UNITS)
            cb = e.umod(uc, GU_UNITS)
            ent = MF._slot_ent(e, is_sh, lo, local)
            rs_w, rs_s = MF._addrs(e, ent, "gu")
            pair = e.add(e.mul(cb, 2), pair_in)
            out.append((rs_w, rs_s, pair, MF._off(e, valid), valid))
        return out

    def issue(n):
        if n >= NR:
            return None
        ch = chains_of(n)
        if wk.guard == "loads":
            return [MF._guarded(e, valid, lambda rs_w=rs_w, rs_s=rs_s, pair=pair, off=off: [
                MO._gu_issue(e, rs_w, rs_s, pair, part, grp, off) for grp in range(A.GU_KI // 4)], f"wgu{n}_{j}")
                for j, (rs_w, rs_s, pair, off, valid) in enumerate(ch)]  # fmt: skip
        got = [[None] * (A.GU_KI // 4) for _ in range(JR)]
        for grp in range(A.GU_KI // 4):
            for j in range(JR):
                rs_w, rs_s, pair, off, _v = ch[j]
                got[j][grp] = MO._gu_issue(e, rs_w, rs_s, pair, part, grp, off)
        return got

    # ---- down units: unit i of the CTA on wave (i + rot) % 8 (MF._dn_unit), rounds of dn_round units a wave
    rot = e.mul(e.umod(n_gu, 2), 4)
    dn_last = e.imax(b.sub(e.mul(ul, DN_UNITS), e.c(1)), 0)
    sl0 = e.udiv(d0, DN_UNITS)
    DNJ = -(-(bd["dn"] + 4) // A.WAVES)
    DR = wk.dn_round
    ND = -(-DNJ // DR)

    def dn_issue(m):
        if m >= ND:
            return None
        return [MF._dn_unit(e, kk, d0, n_dn, dn_last, lo, is_sh, rot, sl0, f"wdn{kk}")
                for kk in range(m * DR, min((m + 1) * DR, DNJ))]  # fmt: skip

    issued = {0: issue(0)}
    if wk.gu_next == "first":
        issued[1] = issue(1)
    dns = {}
    if wk.dn_early:
        dns[0] = dn_issue(0)
    if wk.epi == "compact" and wk.elist_at == "issue":
        _build_elist(e, g0, g1, lo, is_sh)
    if wk.tw_at == "late":
        with b.scf_if(b.land(has_comb, routed)):
            _late_weights(e)
    e.phase(3)
    r_l = e.imin(e.umod(e.lane, 16), R - 1)
    x8 = e.ix("gx8", 0, r_l)
    xe = e.ix("gxe", 0, r_l)

    def a_ops(kit):
        return e.ld(e.add(xe, kit * 4, e.udiv(e.lane, 16))), MO._a_operand(e, x8, kit)

    n_ent = UPR * R
    c = e.umod(e.lane, 32)
    for n in range(NR):
        live_round = e.lt(e.c(n * UPR), n_gu)
        cur = issued.pop(n)
        accs = [b.zero_vec_f32(4) for _ in range(JR)]
        # MFMAs of round n (every CTA runs them: zero weights past its units; a round past every unit of the CTA
        # is skipped below)
        def mfma_round(cur=cur, n=n, pipe_next=None):
            """Round n's MFMAs and their gred stores; with ``pipe_next`` (the next round's chains) the next round's
            k_iter group g is issued right after group g's MFMAs, into the registers they free. Returns the issued
            loads (or [])."""
            nxt_issued = [[None] * (A.GU_KI // 4) for _ in range(JR)] if pipe_next is not None else None
            pipe = wk.gu_apipe
            nxt = a_ops(0) if pipe else None
            accs_l = list(accs)
            for kit in range(A.GU_KI):
                grp, qq = divmod(kit, 4)
                if pipe:
                    ea, a_op = nxt
                    if kit + 1 < A.GU_KI:
                        nxt = a_ops(kit + 1)
                    b.sched_barrier(0)
                else:
                    ea, a_op = a_ops(kit)
                for j in range(JR):
                    ws, ss = cur[j][grp]
                    sw = b.lshr(ss[qq // 2], e.mul(part, 8))
                    accs_l[j] = MO._mma(e, accs_l[j], a_op, ws[qq], ea, sw, (qq % 2) * 2)
                if pipe_next is not None and qq == 3:
                    for j in range(JR):
                        rs_w, rs_s, pair, off, _v = pipe_next[j]
                        nxt_issued[j][grp] = MO._gu_issue(e, rs_w, rs_s, pair, part, grp, off)
            for j in range(JR):
                col = e.add(e.mul(e.add(half, 2 * j), 64), e.mul(w4, 16), e.umod(e.lane, 16))

                def put(row, val, col=col):
                    e.st(e.ix("gred", col, row), val)

                _out_rows(e, accs_l[j], put)
            return nxt_issued if nxt_issued is not None else []

        if wk.gu_next == "pipe" and n + 1 < NR:
            ch_next = chains_of(n + 1)
            issued[n + 1] = MF._guarded(e, live_round, lambda f=mfma_round, c=ch_next: f(pipe_next=c), f"wpipe{n}")
        else:
            with b.scf_if(live_round):
                mfma_round()
        nn = n + 2 if wk.gu_next == "first" else n + 1
        if wk.gu_next in ("mfma", "first") and nn < NR:
            issued[nn] = issue(nn)
        with b.scf_if(live_round):
            e.barrier()
            # ---- epilogue of round n
            if wk.epi == "dense":
                for p in range(-(-n_ent // 16)):
                    ent = e.add(p * 16, e.mul(e.wave, 2), e.udiv(e.lane, 32))
                    ent_c = e.imin(ent, n_ent - 1)
                    kk = e.udiv(ent_c, R)
                    r = e.umod(ent_c, R)
                    k = e.add(kk, n * UPR)
                    u = e.add(g0, k)
                    valid = b.land(e.lt(u, g1), e.lt(ent, n_ent))
                    uc = e.imin(u, gu_last)
                    local = e.udiv(uc, GU_UNITS)
                    cb = e.umod(uc, GU_UNITS)
                    rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
                    live = b.land(valid, e.ge(rank, 0))
                    base = e.add(e.mul(kk, 64), c)
                    x = MO.bf16r(b, e.ld(e.ix("gred", base, r), F32))
                    y = MO.bf16r(b, e.ld(e.ix("gred", e.add(base, 32), r), F32))
                    MO.gu_epilogue_entry(e, x, y, is_sh, live, rank, r, c, cb)
            else:
                k_hi = min((n + 1) * UPR, bd["gu"])
                e0 = e.ld_u(e.off["kst"] + n * UPR)
                e1 = e.ld_u(e.off["kst"] + k_hi)
                if wk.mutate == "epi_drop" and n == 0:  # test only: the CTA's first live entry is never run
                    e0 = e.imin(e.add(e0, 1), e1)
                npass = e.udiv(e.add(b.sub(e1, e0), 15), 16)
                with b.scf_for(e.c(0), npass, e.c(1), iv_name=f"wepi{n}") as p:
                    ent = e.add(e0, e.mul(p, 16), e.mul(e.wave, 2), e.udiv(e.lane, 32))
                    valid = e.lt(ent, e1)
                    w = e.ld(e.add(e.off["elist"], e.imin(ent, e.imax(b.sub(e1, e.c(1)), 0))))
                    k = b.lshr(w, e.c(4))
                    r = b.land(w, e.c(15))
                    kk = b.sub(k, e.c(n * UPR))
                    u = e.add(g0, k)
                    uc = e.imin(u, gu_last)
                    local = e.udiv(uc, GU_UNITS)
                    cb = e.umod(uc, GU_UNITS)
                    rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
                    live = b.land(valid, e.ge(rank, 0))
                    base = e.add(e.mul(kk, 64), c)
                    x = MO.bf16r(b, e.ld(e.ix("gred", base, r), F32))
                    y = MO.bf16r(b, e.ld(e.ix("gred", e.add(base, 32), r), F32))
                    MO.gu_epilogue_entry(e, x, y, is_sh, live, rank, r, c, cb)
            e.barrier()
        if wk.gu_next == "epi" and n + 1 < NR:
            issued[n + 1] = issue(n + 1)
    e.phase(4)
    e.phase(5)
    if 0 not in dns:
        dns[0] = dn_issue(0)
    # ---- exchange: every row of every slot this CTA's down units read
    _exchange(e, sl0, d1, n_dn, ul, lo, is_sh, max(bd["sl"], 1))
    e.phase(6)
    for m in range(ND):
        if m + 1 < ND:
            dns[m + 1] = dn_issue(m + 1)
        with b.scf_if(e.lt(e.c(max(A.WAVES * m * DR - 4, 0)), n_dn)):
            _dn_compute(e, dns.pop(m), lo, is_sh, r_l)
    e.phase(7)
    # ---- combine (hidden tile ct)
    with b.scf_if(has_comb):
        if wk.tw_at == "combine":
            with b.scf_if(routed):
                _late_weights(e)
        with b.scf_if(is_sh):  # the shared group routed nothing so far: the router weights now
            _route_rows(e, weights=True, keys=False)
            if wk.extras == "first":
                _extras(e, ct)
        if wk.extras == "late":
            _extras(e, ct)
        e.barrier()
        with b.scf_if(e.lt(e.tid, 32 * R)):
            i = e.umod(e.tid, 32)
            rt = e.udiv(e.tid, 32)
            cp = e.add(e.mul(ct, 8), e.umod(i, 8))
            got = e.poll_m([MF._ys_spec(e, rt, k, cp) for k in range(A.TOP_K + 1)], e.xtag)
            vals = [M.bf2_f32(b, got[k][0]) for k in range(A.TOP_K + 1)]
            order = [1, 0] + list(range(2, A.TOP_K)) if wk.mutate == "combine_swap" else None
            MO.combine_tile(e, ct, lambda k, h, jj, rt_: vals[k][h], order=order)
    e.phase(8)
    e.dbg_end()


def build_wide_suffix(rows: int, group: int, wk: WideKnobs, experts: int = A.EXPERTS):
    """A standalone kernel of the wide body with the mono kernel's signature (behind a FlyDSL ``a1 .. m3`` prefix):
    for compile / ISA / resource checks. Python engine only."""
    plan = WidePlan(rows, experts, wk.fused())
    plan.validate()
    b = IRBuilder(f"qwen38_wmoe_e{experts}_r{rows}g{group}" + (f"_{wk.suffix()}" if wk.suffix() else "") + "_pyonly")
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    step = b.param("step", I64)
    layer = b.param("layer", I32)
    addr = {n: b.param(n, I64) for n in A.LAYER_ARGS}
    step_v = b.global_load(b.int_to_ptr(step, PtrType(I32, "global")), b.const_i32(0), I32, align=4)
    tag = M.launch_tag(b, step_v, b.readfirstlane(layer))
    e = WideCtx(b, plan, wk, tag=tag, layer=b.readfirstlane(layer), addr=addr)
    emit_wide(e)
    b.ret()
    return b.kernel


# ============================================================ stage functions


def arena_words(rows: int, group: int, atom_root: str | None = None) -> int:
    """The FlyDSL layer arena of a grouped build (ATOM ``layout_wide.for_wide(rows, group).lds_words``; the
    ``atom`` package, else the checkout at ``Q38_ATOM_DIR`` / ``atom_root``)."""
    import os

    try:
        from atom.models.qwen4_exp_mono.kernels import layout_wide as LW
    except Exception as ex:  # noqa: BLE001
        root = atom_root or os.environ.get("Q38_ATOM_DIR")
        if not root:
            raise RuntimeError("the wide LDS arena check needs ATOM (importable, or Q38_ATOM_DIR)") from ex
        from ..mono_layout import _import_atom

        LW = _import_atom("atom.models.qwen4_exp_mono.kernels.layout_wide", root)
    return LW.for_wide(rows, group).lds_words


LAST_SITES: list = []  # dbg=1: the poll sites of the last built m4 body


def build_function(stage: str, rows: int, group: int, wk: WideKnobs, *, experts=A.EXPERTS, arena=None):
    """``m4``: the wide body; ``m5``: empty (FlyDSL's M3 runs the router)."""
    from ..stage_fn import ABI_PARAMS, variant_name

    if stage not in ("m4", "m5"):
        raise ValueError(f"wide functions are m4 (the body) and m5 (empty), not {stage!r}")
    plan = WidePlan(rows, experts, wk.fused())
    plan.validate()
    b = IRBuilder(f"rocke_q38_{stage}_{variant_name(rows, False, group)}")
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    words = None
    if stage == "m4":
        addr = {n: vals[n] for n in A.LAYER_ARGS}
        e = WideCtx(b, plan, wk, tag=b.readfirstlane(vals["tag"]), layer=b.readfirstlane(vals["layer"]), addr=addr)
        words = e.lds_words
        if arena is not None and words > arena:
            raise ValueError(f"wide MoE r{rows}g{group}: LDS arena {words} words > the layer's {arena}")
        emit_wide(e)
        LAST_SITES[:] = list(e.sites)
    b.ret()
    return b.kernel, plan, words


def pair_id(variant: str, wk) -> str:
    return hashlib.sha256(json.dumps(["wide", variant, repr(wk)]).encode()).hexdigest()[:16]


def write_functions(out: str, variants, wk: WideKnobs, *, narrow=None, experts=A.EXPERTS, arena="atom",
                    flavor=None, arch="gfx950", bitcode=False, atom_root=None) -> list:
    """Emit ``<stage>_<variant>.{ll,json}`` per grouped variant ``r<rows>g<S>``: rows > 4 the wide body (``m4``) and
    an empty ``m5`` (FlyDSL's M3 stays); rows <= 4 (``r4g2``) the narrow fused body (:mod:`..stage_moe_fused`:
    ``m3`` fused, ``m4`` / ``m5`` empty) built with ``narrow`` (a ``FusedKnobs``; default the served rows > 1 set
    ``stage_moe_fused.best_knobs(rows)``) under the grouped variant's symbols. ``arena``: ``"atom"`` (check against the
    FlyDSL layer's arena), an int, or None."""
    from ..device_function import DeviceFunctionKnobs, lower_device_function, write_device_function
    from ..stage_fn import parse_variant
    from .. import stage_moe_fused as SMF

    fk = DeviceFunctionKnobs(lds="arg", inline="always")
    written = []
    for v in variants:
        rows, seq, group = parse_variant(v)
        if seq or not group:
            raise ValueError(f"{v}: grouped variants only (r<rows>g<S>)")
        have = arena_words(rows, group, atom_root) if arena == "atom" else arena
        if rows <= A.MAX_ROWS:
            nk = narrow if narrow is not None else SMF.best_knobs(rows)
            pair = "g" + SMF.pair_id(v, nk)
            for stage in STAGES:
                kern, _plan = SMF.build_function(stage, rows, True, nk, experts=experts,
                                                 arena_words=have if stage == "m3" else None)
                fn = lower_device_function(kern, arch=arch, symbol=f"rocke_q38_{stage}_{v}", llvm_flavor=flavor,
                                           knobs=fk)  # fmt: skip
                paths = write_device_function(fn, out, stem=f"{stage}_{v}", bitcode=bitcode)
                meta = json.load(open(paths["json"]))
                meta.update(SMF.sidecar_extra(stage, pair, nk), moe_knobs="fused:" + (nk.suffix() or "default"))
                with open(paths["json"], "w", newline="\n") as f:
                    json.dump(meta, f, indent=1)
                    f.write("\n")
                written.append({"stage": stage, "variant": v, "lds_bytes": fn.lds_bytes, **paths})
            continue
        pair = pair_id(v, wk)
        for stage in ("m4", "m5"):
            kern, _plan, words = build_function(stage, rows, group, wk, experts=experts,
                                                arena=have if stage == "m4" else None)
            fn = lower_device_function(kern, arch=arch, symbol=f"rocke_q38_{stage}_{v}", llvm_flavor=flavor, knobs=fk)
            paths = write_device_function(fn, out, stem=f"{stage}_{v}", bitcode=bitcode)
            meta = json.load(open(paths["json"]))
            other = "m5" if stage == "m4" else "m4"
            meta.update(pair=pair, requires={other: pair}, wide=True, moe_knobs="wide:" + (knobs_spec(wk) or "default"))
            if stage == "m4" and wk.numa != "off":  # ATOM checks it against numa.wide_header_offset()
                meta["numa_wide_hdr"] = WIDE_NUMA_HDR
            if stage == "m4" and wk.dbg:
                meta["dbg_sites"] = [[ph, n] for ph, n in LAST_SITES]
            with open(paths["json"], "w", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append({"stage": stage, "variant": v, "lds_bytes": fn.lds_bytes, "lds_words": words, **paths})
    return written


def extern_spec(out: str, variants) -> dict:
    """``ATOM_Q38_EXTERN_STAGES`` entries for :func:`write_functions`' files (narrow grouped: m3 / m4 / m5; wide:
    m4 / m5; one template per stage, ``{variant}`` filled by ATOM)."""
    return {s: f"{out}/{s}_{{variant}}.ll" for s in STAGES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the WP2d grouped / wide fused MoE stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r4g2,r6g3,r8g4,r9g3,r12g3,r12g4,r16g4")
    ap.add_argument("--knobs", default="", help="WideKnobs spec k=v,... (wide variants)")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--no-arena-check", action="store_true")
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    wk = knobs_from_spec(a.knobs)
    w = write_functions(a.out, [v for v in a.variants.split(",") if v], wk, flavor=a.flavor,
                        arena=None if a.no_arena_check else "atom", atom_root=a.atom_root)  # fmt: skip
    print(json.dumps({"knobs": knobs_spec(wk), "written": w}, indent=1))
    return 0


__all__ = ["DBG_MAXIT", "DBG_OFF", "MUTANTS", "dbg_offset", "STAGES", "WIDE_ROWS", "WideCtx", "WideKnobs", "WidePlan", "arena_words", "build_function", "build_wide_suffix",
           "emit_wide", "extern_spec", "knobs_from_spec", "knobs_spec", "lds_layout", "main", "pair_id", "wide_m3",
           "write_functions"]  # fmt: skip

if __name__ == "__main__":
    sys.exit(main())
