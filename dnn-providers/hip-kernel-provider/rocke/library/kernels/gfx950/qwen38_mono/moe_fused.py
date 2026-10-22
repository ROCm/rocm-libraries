# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Expert-major fused MoE (M3 + M4 + M5) of the Qwen3.8 mono layer in rocKE.

Design: ``hackathon-prep/reports/qwen38-rocke/MOE_FUSED_DESIGN.md``. One body
runs in the M3 slot of the mono layer kernel (hybrid path: the ``m3`` stage
function; ``m4`` / ``m5`` are empty functions) and places itself from the
block id. Every CTA streams one byte line: the gate_up units of its slots,
then the down units of the same experts, so the weight stream has no hop
between gate_up and down. Numerics are the mono's bit for bit (the router,
top-10, softmax weights, each n-block's sequential 20-step gate_up chain, the
routed / shared SiLU variants, the MXFP8 rule, the routed 5-step and shared
merge-order down chains, the rank-ordered combine): see the design, section 5.

Work model (``R`` request rows, ``U`` routed slots = the union of the rows'
top-10, by expert id; rows 1: the 10 picks in rank order):

* *shared group*: CTAs ``sh_base .. sh_base + sh_ctas - 1`` own the shared
  expert (slot 0) and start as soon as the mixed row lands (no routing).
* *routed lines*: every other CTA. ``iod_split=1``: a line per IOD (CTA ``b``
  runs on XCD ``b % 8``, IOD ``(b % 8) // 4``); routed slots are dealt to
  the lines in proportion to their CTA counts, so the gate_up -> down
  exchange stays inside one IOD. A line of ``N`` CTAs and ``Ul`` slots has
  ``G = 20 Ul`` gate_up units (one slot x one 32-column block: 4 n-block
  chains of 20 k_iters) and ``160 Ul`` down units (one slot x one 16-column
  hidden n-block, 5 k_iters); a gate_up unit weighs 16 down units. CTA ``t``
  takes gate_up units ``[tG/N, (t+1)G/N)`` and down units ``[D(t), D(t+1))``
  with ``D`` the running maximum of ``tW/N - 16 tG/N`` (``W = 24 G``),
  clipped to ``160 Ul``: per-CTA bytes equal to within one down unit unless
  a CTA's gate_up units alone exceed its share (:meth:`FusedPlan.line`).
* exchange: gate_up results go to the mono's own ``dn`` mailbox (same rows,
  ranks, layout and bits); a down unit's CTA pre-polls the slot's exponents
  and reads the slot's rows (``R x 180`` pairs) into LDS.
* down results go to a private ``ys`` mailbox (bf16 pairs) in the ``ss``
  region (dead from Q6's end through the MoE) under the private tag
  ``tag ^ 2^31`` (``ss`` holds Q5's pairs under the launch tag in QSA layers;
  the XOR differs from the launch tag in bit 31 for every tag, wrapped or not,
  so no pair of this launch's Q5 matches it, and two launches' private tags
  are equal only when their launch tags are).
* combine: hidden tile ``t`` on CTA ``(combine_base + t) % 256`` polls the
  ``11 R x 8`` ``ys`` pairs it needs and runs the mono's combine.

Knobs: :class:`FusedKnobs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import moe as MO

GU_UNITS = A.MOE_INTER // 32  # 20 gate_up units (32-column blocks) per slot
DN_UNITS = A.HIDDEN // 16  # 160 down units (16-column n-blocks) per slot
GU_WEIGHT = 16  # one gate_up unit's bytes in down units
YS_PAIRS = A.HIDDEN // 2  # bf16 pairs per (row, rank) of ys
XTAG_BIT = -(2**31)  # private tag: launch tag ^ 2^31 (never the launch tag itself, at any step count)
STAMP_PTS = 16  # phase stamps per CTA (knob phase_stamps)
YS_BYTES = (A.TOP_K + 1) * YS_PAIRS * A.PAIR  # one row's ys
STAMP_OFF = YS_BYTES  # row 0's ss: the phase stamps after its ys
PHASES = ("entry", "router", "route", "issue", "gu", "epi", "xchg", "dn", "combine")

# ATOM's NUMA area (``kernels/numa.py``): the header (word 0: the calibrated page mask) past the mailbox part,
# the stamps, blame and spin regions (``sched.side_delta() + layout.mailbox_bytes(MAX_ROWS)``), the area at the
# next 2 MiB boundary past header + 256; byte x of the mailbox part at x + (x & ~8191) (copy 0) and + 8192
# (copy 1); a CTA polls copy ``mask_page ^ iod``.
NUMA_MAX_STAGES = 15  # sched.MAX_STAGES
NUMA_HDR = (2 * A.mailbox_bytes(A.MAX_ROWS) + A.STAMP_REGION_BYTES
            + A.LAYER_SLOTS * A.GRID * NUMA_MAX_STAGES * A.WAVES * (8 + 4))  # fmt: skip
NUMA_PAGE = 1 << 21
NUMA_CHUNK = 8192

# ``combine=last``: one int32 counter per hidden tile in the unused tail of the ``inv`` region's 256-byte rounding
# (rows 0..3: 40 counters a row, from byte 32 past the region's 4 pairs).
CTR_REGION = "inv"
CTR_PER_ROW = 40


def counter_offset(t: int) -> int:
    """Scratch byte offset of hidden tile ``t``'s ``combine=last`` counter."""
    if not 0 <= t < DN_UNITS:
        raise ValueError(t)
    return A.SC[CTR_REGION] + (t // CTR_PER_ROW) * A.SCRATCH_BYTES + 32 + 4 * (t % CTR_PER_ROW)


# ============================================================ knobs


@dataclass(frozen=True)
class FusedKnobs:
    """Knobs of the fused MoE (first value = default).

    map: ``line`` (gate_up and down units on one byte line per CTA line) |
      ``groups`` (two independent even splits: gate_up units and down units).
    iod_split: 1 (a line per IOD, slots dealt by CTA count) | 0 (one line).
    sh_ctas / sh_base: the shared group (CTAs ``sh_base ..``, no wrap).
    dn_issue: ``with_gu`` (a wave's first ``dn_inflight`` down units issue
      their weights together with the gate_up chains, the rest after the
      gate_up epilogue) | ``after_epi`` (every down unit after the gate_up
      epilogue) | ``after_gu`` (every down unit right after the gate_up
      MFMAs, ahead of the epilogue) | ``pipe`` (one down unit after each of
      the first four gate_up k_iter groups' MFMAs, into the registers that
      group frees; the rest after the MFMAs) | ``lds`` (a wave's first
      ``dn_lds`` down units' weight tiles DMA into LDS with the gate_up
      chains, their scales and the rest after the epilogue; the cache policy
      of the DMA is ``w_cpol & 3``).
    dn_inflight: down units per wave issued with the gate_up chains.
    combine_base: CTA of hidden tile 0's combine task; -1 = the CTA right
      after the router's (M3) tasks, so the combine tasks start on the CTAs
      that ran no router task.
    extras: ``first`` (shared gate, residual words, injection weights polled
      before the routing) | ``late`` (after the down units).
    route_gate / route_pause: as :class:`.moe.MoeKnobs` (routing polls).
    tw_at: ``late`` (router weights from the top-10 keys after the weight
      issue) | ``route`` (with the top-10) | ``epi`` (from the keys after the
      gate_up epilogue).
    line_impl: ``rt`` (the line's divisions by the CTA's line size, a runtime
      value when the shared group's size differs) | ``const`` (the shared and
      routed lines each with constant divisors, then a select).
    slots: ``table`` (the union / rank tables, one pass + barrier) | ``arith``
      (rows 1: slot s = rank s - 1, no table) | ``wave`` (every wave builds its
      own copy of the tables: no barrier after the pass).
    w_cpol, mailbox, a_pre, gu_interleave, gu_apipe, m3_ept: the port's
      (:class:`.moe.MoeKnobs` ``w_cpol``, ``mailbox``, ``a_pre``,
      ``m4_interleave``, ``m4_apipe``, ``m3_ept``); defaults = the port's tuned
      rows > 1 set.
    router: include M3 (False: a split suffix behind a FlyDSL ``a1 .. m3``
      prefix, for injected real routing).
    w_null: measurement only (WaveScope W0): every expert weight load reads
      past its block (zero bytes): the fixed overhead F. Wrong numerics.
    phase_stamps: measurement only: thread 0 of every CTA stamps
      ``s_memrealtime`` at each phase (:data:`PHASES`) into row 0's ``ss``
      after its ``ys``.
    guard: ``none`` (a chain or down unit with no work issues its loads past
      the block: zero bytes, but one TA address pass per load instruction) |
      ``loads`` (it issues none: each chain's / unit's loads in a 0/1-trip
      loop).
    xchg_pre: ``hint`` (wave 0 spins on the exchanged rows' exponent pairs,
      then a barrier, then every thread reads the rows) | ``none`` (every
      thread polls the rows at once: one round trip less).
    router_knobs: ``fused`` (the router (M3) with this build's ``w_cpol`` and
      ``mailbox``) | ``mono`` (with the mono's: default loads, the asm
      mailbox).
    combine: ``task`` (hidden tile t's combine on CTA ``combine_base + t``,
      after its own down units) | ``sh`` (the shared group combines every
      tile once its own work is done: the routed CTAs end with their down
      units) | ``last`` (last arriver: after its down units a CTA adds one per
      unit to the unit's hidden tile counter (:func:`counter_offset`, unused
      region tails; atomic, device scope); the CTA that brings a tile's count
      to its U + 1 units resets it and combines the tile; data still come by
      tagged polls, the counter only picks the CTA. Needs a zeroed scratch
      at the first launch, as every counter region).
    spec: ``off`` | ``prev``: the expert union of this layer's previous launch
      (a 512-bit history per layer < 48, written by one routed CTA after its
      routing, :func:`hist_offset`) is touched into the memory-side cache
      while the routing is pending: every routed CTA's non-routing waves load
      one dword per 128-byte line of those experts' weight blocks (gate_up
      blocks first, then down), ``spec_loads`` 8 KiB instructions a wave,
      no data used (an OR sink). Served routing repeats ~0.4 of the union
      from step to step at the same layer.
    spec_keep: test only: touch only the history experts with index % 10 <
      spec_keep (an oracle bench at partial overlap).
    mutate: test only: ``combine_swap`` (ranks 0 and 1 swapped in the
      combine order: must differ from the mono) | ``xtag_or`` (the private
      tag ``tag | 2^31`` of the first version, equal to the launch tag once
      the step counter reaches 2^25: QSA layers then read Q5's pairs as ys).
    numa: 0 | 1 = the mailbox layout of a mono build with ATOM's ``numa=1``
      schedule lever (``ATOM_Q38_SCHED``): every mailbox the body shares with
      the mono's stages (``mxbf``, ``mx8``, ``mxe``, ``w``, ``minj``) is read
      from the IOD-local copy in ATOM's NUMA area (``numa_hdr``); the
      sidecars say so (``"mailboxes": "numa1"``) and ATOM keeps the lever for
      such a build (and refuses it without the lever). The private ``ys`` /
      stamps / counters stay in the plain scratch (unused under the lever).
    lg_copy: ``canon`` (the router logits and the shared gate, ``lg`` /
      ``sg``, one copy in the plain scratch) | ``numa2`` (both copies in the
      NUMA area, every consumer polls its IOD-local one; the area must exist:
      a ``numa`` lever somewhere in ``ATOM_Q38_SCHED``).
    xchg_home: ``single`` (the gate_up -> down exchange ``dn`` in the plain
      scratch) | ``numa2`` (both copies in the NUMA area, local polls).
    ys_home: ``single`` (the down outputs ``ys`` for the combine in the plain
      ``ss``) | ``numa2`` (both copies in the NUMA area's ``ss``, the combine
      polls its local copy; the private tag keeps Q5's area pairs apart).
    numa_hdr: byte offset of ATOM's NUMA header (``numa.header_offset()``);
      0 = :data:`NUMA_HDR`.
    line_order: ``slot`` (a line's CTAs in block-id order, gate_up units
      dealt evenly) | ``router_dn`` (the CTAs that ran a router (M3) task take
      the line's last positions and the gate_up units are dealt front-first,
      so they get the fewer gate_up units and more down units).
    sh_join: 0 (off) | 1..8: after the shared expert the shared group joins
      its IOD's routed line for down units only (positions after the routed
      CTAs), each with ``sh_join / 8`` of a routed CTA's byte share (the
      water-fill weights).
    dn_lds: ``dn_issue=lds``: down units per wave whose weight tiles go
      HBM -> LDS (buffer-to-LDS DMA, issued with the gate_up chains, no
      VGPRs held); the rest as ``after_epi``. LDS cost 1280 words per unit and
      wave (refused past the mono arena).
    """

    map: str = "line"
    iod_split: int = 1
    sh_ctas: int = 16
    sh_base: int = 156
    dn_issue: str = "with_gu"
    dn_inflight: int = 5
    combine_base: int = 0
    extras: str = "first"
    route_gate: str = "none"
    route_pause: str = ""
    tw_at: str = "late"
    line_impl: str = "rt"
    slots: str = "table"
    w_cpol: int = 18
    mailbox: M.MailboxKnobs = field(default_factory=lambda: M.MailboxKnobs(impl="ir"))
    a_pre: bool = True
    gu_interleave: bool = True
    gu_apipe: int = 1
    m3_ept: int = 8
    router: bool = True
    w_null: bool = False
    phase_stamps: bool = False
    mutate: str = ""
    guard: str = "none"
    xchg_pre: str = "hint"
    router_knobs: str = "fused"
    combine: str = "task"
    spec: str = "off"
    spec_loads: int = 16
    spec_keep: int = 10
    numa: int = 0
    lg_copy: str = "canon"
    xchg_home: str = "single"
    numa_hdr: int = 0
    line_order: str = "slot"
    sh_join: int = 0
    dn_lds: int = 1
    ys_home: str = "single"

    @property
    def uses_area(self) -> bool:
        """Some mailbox in ATOM's NUMA area."""
        return bool(self.numa) or "numa2" in (self.lg_copy, self.xchg_home, self.ys_home)

    def __post_init__(self) -> None:
        if self.map not in ("line", "groups"):
            raise ValueError(f"map must be line|groups, got {self.map!r}")
        if self.iod_split not in (0, 1):
            raise ValueError("iod_split must be 0 or 1")
        if not 1 <= self.sh_ctas <= 64 or not 0 <= self.sh_base or self.sh_base + self.sh_ctas > A.GRID:
            raise ValueError("shared group: 1..64 CTAs inside the grid (no wrap)")
        if self.dn_issue not in ("with_gu", "after_epi", "after_gu", "pipe", "lds"):
            raise ValueError(f"dn_issue must be with_gu|after_epi|after_gu|pipe|lds, got {self.dn_issue!r}")
        if not 1 <= self.dn_lds <= 8:
            raise ValueError("dn_lds must be in 1..8")
        if self.numa not in (0, 1):
            raise ValueError("numa must be 0 or 1")
        if self.lg_copy not in ("canon", "numa2"):
            raise ValueError(f"lg_copy must be canon|numa2, got {self.lg_copy!r}")
        if self.xchg_home not in ("single", "numa2"):
            raise ValueError(f"xchg_home must be single|numa2, got {self.xchg_home!r}")
        if self.ys_home not in ("single", "numa2"):
            raise ValueError(f"ys_home must be single|numa2, got {self.ys_home!r}")
        if self.numa_hdr < 0 or self.numa_hdr % 256:
            raise ValueError("numa_hdr must be a non-negative multiple of 256")
        if self.line_order not in ("slot", "router_dn"):
            raise ValueError(f"line_order must be slot|router_dn, got {self.line_order!r}")
        if not 0 <= self.sh_join <= 8:
            raise ValueError("sh_join must be in 0..8")
        if self.sh_join and self.combine == "sh":
            raise ValueError("sh_join: the shared group joins the down line, not with combine=sh")
        if not 0 <= self.dn_inflight <= 8:
            raise ValueError("dn_inflight must be in 0..8")
        if not -1 <= self.combine_base < A.GRID:
            raise ValueError("combine_base must be a CTA or -1")
        if self.extras not in ("first", "late"):
            raise ValueError(f"extras must be first|late, got {self.extras!r}")
        if self.tw_at not in ("late", "route", "epi"):
            raise ValueError(f"tw_at must be late|route|epi, got {self.tw_at!r}")
        if self.line_impl not in ("rt", "const"):
            raise ValueError(f"line_impl must be rt|const, got {self.line_impl!r}")
        if self.slots not in ("table", "arith", "wave"):
            raise ValueError(f"slots must be table|arith|wave, got {self.slots!r}")
        if self.gu_apipe and not self.gu_interleave:
            raise ValueError("gu_apipe needs gu_interleave")
        if self.combine not in ("task", "sh", "last"):
            raise ValueError(f"combine must be task|sh|last, got {self.combine!r}")
        if self.spec not in ("off", "prev"):
            raise ValueError(f"spec must be off|prev, got {self.spec!r}")
        if not 1 <= self.spec_loads <= 64 or not 1 <= self.spec_keep <= 10:
            raise ValueError("spec_loads in 1..64, spec_keep in 1..10")
        if self.router_knobs not in ("fused", "mono"):
            raise ValueError(f"router_knobs must be fused|mono, got {self.router_knobs!r}")
        if self.xchg_pre not in ("hint", "none"):
            raise ValueError(f"xchg_pre must be hint|none, got {self.xchg_pre!r}")
        if self.guard not in ("none", "loads"):
            raise ValueError(f"guard must be none|loads, got {self.guard!r}")
        if self.mutate not in ("", "combine_swap", "xtag_or"):
            raise ValueError(f"mutate must be ''|combine_swap|xtag_or, got {self.mutate!r}")
        self.moe_knobs(1)  # the port's checks of the shared knobs

    def moe_knobs(self, rows: int) -> MO.MoeKnobs:
        """The port knobs the reused emitters read."""
        return MO.MoeKnobs(
            dedupe=rows > 1, union_order="id", shared_first=True, m4_waves=4, m4_map="balanced",
            m3_ept=self.m3_ept, w_cpol=self.w_cpol, mailbox=self.mailbox, m5_ranks="pre", a_pre=self.a_pre,
            m4_guard="none", m4_interleave=self.gu_interleave, m4_apipe=self.gu_apipe,
            route_pause=self.route_pause, route_gate=self.route_gate,
        )  # fmt: skip

    def suffix(self) -> str:
        d = FusedKnobs()
        parts = []
        for name in ("map", "iod_split", "sh_ctas", "sh_base", "dn_issue", "dn_inflight", "combine_base",
                     "extras", "route_gate", "route_pause", "tw_at", "line_impl", "slots", "w_cpol", "a_pre",
                     "gu_interleave", "gu_apipe", "m3_ept", "router", "w_null", "phase_stamps", "mutate", "guard", "xchg_pre", "router_knobs", "combine", "spec", "spec_loads",
                     "spec_keep", "numa", "lg_copy", "xchg_home", "numa_hdr", "line_order", "sh_join", "dn_lds",
                     "ys_home"):
            v = getattr(self, name)
            if v != getattr(d, name):
                parts.append(f"{name}{v}".replace("_", ""))
        mb = self.mailbox.suffix()
        if self.mailbox != d.mailbox:
            parts.append("mb" + (mb or "default"))
        return "_".join(parts)


# ============================================================ plan (host side)


def _count_iod(x: int, i: int) -> int:
    """CTAs ``b < x`` on IOD ``i`` (``(b % 8) // 4 == i``)."""
    return 4 * (x >> 3) + min(max((x & 7) - 4 * i, 0), 4)


@dataclass(frozen=True)
class FusedPlan:
    """Static work model of a fused build (``rows`` request rows)."""

    rows: int
    experts: int
    knobs: FusedKnobs

    @property
    def moe(self) -> MO.MoePlan:
        return MO.MoePlan(self.rows, self.experts, self.knobs.moe_knobs(self.rows), MO.STAGES)

    @property
    def u_range(self) -> tuple:
        """(min, max) routed slots of a launch."""
        if self.rows == 1:
            return A.TOP_K, A.TOP_K
        return A.TOP_K, min(self.rows * A.TOP_K, self.experts)

    @property
    def combine_base(self) -> int:
        """CTA of hidden tile 0's combine task (``knobs.combine_base``; -1: past the router tasks)."""
        if self.knobs.combine_base >= 0:
            return self.knobs.combine_base
        mp = self.moe
        return (mp.base("m3") + mp.m3_tasks) % A.GRID

    def is_shared(self, bid: int) -> bool:
        k = self.knobs
        return k.sh_base <= bid < k.sh_base + k.sh_ctas

    def _sh_before(self, x: int, i: int) -> int:
        k = self.knobs
        lo, hi = k.sh_base, k.sh_base + k.sh_ctas
        return _count_iod(min(max(x, lo), hi), i) - _count_iod(lo, i)

    def _rt_before(self, x: int, li: int) -> int:
        """Routed CTAs of line ``li`` with block id < ``x`` (``x`` in 0..GRID)."""
        k = self.knobs
        if k.iod_split:
            return _count_iod(x, li) - self._sh_before(x, li)
        lo, hi = k.sh_base, k.sh_base + k.sh_ctas
        return x - (min(max(x, lo), hi) - lo)

    @property
    def lines_n(self) -> tuple:
        """CTAs per routed line."""
        if self.knobs.iod_split:
            return tuple(_count_iod(A.GRID, i) - self._sh_before(A.GRID, i) for i in (0, 1))
        return (A.GRID - self.knobs.sh_ctas,)

    @property
    def lines_s(self) -> tuple:
        """Shared CTAs per line (the ``sh_join`` positions past the routed ones)."""
        if self.knobs.iod_split:
            return tuple(self._sh_before(A.GRID, i) for i in (0, 1))
        return (self.knobs.sh_ctas,)

    def rot(self, li: int) -> int:
        """``line_order=router_dn``: line position of the first CTA past the router (M3) tasks (0: off)."""
        if self.knobs.line_order != "router_dn":
            return 0
        mp = self.moe
        return self._rt_before((mp.base("m3") + mp.m3_tasks) % A.GRID, li) % self.lines_n[li]

    def role(self, bid: int):
        """``("sh", t, N)`` or ``("rt", line, t, N)``."""
        k = self.knobs
        if self.is_shared(bid):
            return ("sh", bid - k.sh_base, k.sh_ctas)
        li = (bid % 8) // 4 if k.iod_split else 0
        n = self.lines_n[li]
        t = self._rt_before(bid, li)
        return ("rt", li, (t - self.rot(li)) % n, n)

    def join_pos(self, bid: int):
        """``sh_join``: (line, position) of shared CTA ``bid`` among its line's down-unit positions."""
        k = self.knobs
        li = (bid % 8) // 4 if k.iod_split else 0
        if k.iod_split:
            j = _count_iod(bid, li) - _count_iod(k.sh_base, li)
        else:
            j = bid - k.sh_base
        return li, self.lines_n[li] + j

    def line_slots(self, U: int) -> list:
        """``[(first routed slot, slots)]`` per line (routed slots 0 .. U - 1)."""
        ns = self.lines_n
        if len(ns) == 1:
            return [(0, U)]
        n = sum(ns)
        u0 = (U * ns[0] + n // 2) // n
        return [(0, u0), (u0, U - u0)]

    def line(self, t: int, n: int, ul: int, s_join: int = 0) -> tuple:
        """(g0, g1, d0, d1): CTA ``t`` of a line of ``n`` routed CTAs and ``ul`` slots; with ``sh_join`` the line's
        ``s_join`` shared CTAs take positions n .. n + s_join - 1 (down units only, weight ``sh_join`` / 8)."""
        k = self.knobs
        g = GU_UNITS * ul
        dt = DN_UNITS * ul
        front = k.line_order == "router_dn"
        S = s_join if k.sh_join else 0

        def g0(s):
            if s >= n:
                return g
            if front:
                q, rem = divmod(g, n)
                return s * q + min(s, rem)
            return s * g // n

        tot = 8 * n + k.sh_join * S

        def cum(s):
            return 8 * min(s, n) + k.sh_join * max(s - n, 0)

        if k.map == "groups":
            if not S:
                return g0(t), g0(t + 1), t * dt // n, (t + 1) * dt // n
            return g0(t), g0(t + 1), dt * cum(t) // tot, dt * cum(t + 1) // tot
        w = GU_WEIGHT * g + dt

        def raw(s):
            if S:
                return w * cum(s) // tot - GU_WEIGHT * g0(s)
            return s * w // n - GU_WEIGHT * g0(s)

        cm = max(raw(s) for s in range(t + 1))
        return g0(t), g0(t + 1), min(cm, dt), min(max(cm, raw(t + 1)), dt)

    def sh_line(self, t: int) -> tuple:
        """The shared group's own line (CTA ``t`` of ``sh_ctas``, the shared expert)."""
        k = self.knobs
        g, dt, n = GU_UNITS, DN_UNITS, k.sh_ctas
        if k.map == "groups":
            return t * g // n, (t + 1) * g // n, t * dt // n, (t + 1) * dt // n
        w = GU_WEIGHT * g + dt

        def raw(s):
            return s * w // n - GU_WEIGHT * (s * g // n)

        cm = max(raw(s) for s in range(t + 1))
        return t * g // n, (t + 1) * g // n, min(cm, dt), min(max(cm, raw(t + 1)), dt)

    def assignment(self, U: int) -> dict:
        """bid -> (kind, gate_up units [(slot, cb)], down units [(slot, n-block)], joined down units); slot -1 is
        the shared expert, routed slots are 0 .. U - 1. Joined units (``sh_join``): a shared CTA's routed down
        units after its own work (empty otherwise)."""
        out = {}
        spans = self.line_slots(U)
        S = self.lines_s
        for bid in range(A.GRID):
            r = self.role(bid)
            if r[0] == "sh":
                lo = -1
                g0, g1, d0, d1 = self.sh_line(r[1])
            else:
                _k, li, t, n = r
                lo, ul = spans[li]
                g0, g1, d0, d1 = self.line(t, n, ul, S[li])

            def slot(x, lo=lo):
                return -1 if lo < 0 else lo + x

            gu = [(slot(u // GU_UNITS), u % GU_UNITS) for u in range(g0, g1)]
            dn = [(slot(d // DN_UNITS), d % DN_UNITS) for d in range(d0, d1)]
            jn = []
            if r[0] == "sh" and self.knobs.sh_join:
                li, tj = self.join_pos(bid)
                jlo, jul = spans[li]
                _a, _b, j0, j1 = self.line(tj, self.lines_n[li], jul, S[li])
                jn = [(jlo + d // DN_UNITS, d % DN_UNITS) for d in range(j0, j1)]
            out[bid] = (r[0], gu, dn, jn)
        return out

    def bounds(self) -> dict:
        """Static per-CTA maxima over every U of the build: gate_up units, down units, slots spanned by the
        down units; joined down units and their slots (``sh_join``)."""
        gu = dn = sl = jn = jsl = 0
        lo, hi = self.u_range
        for U in range(lo, hi + 1):
            for _kind, g, d, j in self.assignment(U).values():
                gu = max(gu, len(g))
                dn = max(dn, len(d))
                if d:
                    sl = max(sl, len({s for s, _ in d}))
                jn = max(jn, len(j))
                if j:
                    jsl = max(jsl, len({s for s, _ in j}))
        return {"gu": gu, "dn": dn, "sl": sl, "jn": jn, "jsl": jsl}

    @property
    def j(self) -> int:
        """gate_up chains per wave: unit k's 4 chains run on waves 4 (k % 2) .. + 3."""
        return -(-self.bounds_cached["gu"] // 2)

    @property
    def dnj(self) -> int:
        """down units per wave (unit i on wave (i + rot) % 8, rot in {0, 4})."""
        return -(-(self.bounds_cached["dn"] + 4) // A.WAVES)

    @property
    def jnj(self) -> int:
        """joined down units per wave (unit i on wave i % 8)."""
        return -(-self.bounds_cached["jn"] // A.WAVES)

    @property
    def bounds_cached(self) -> dict:
        return _bounds(self)

    def validate(self) -> None:
        if not 1 <= self.rows <= A.MAX_ROWS:
            raise ValueError(f"rows must be in 1..{A.MAX_ROWS}")
        if self.experts % 8 or not 16 <= self.experts <= A.EXPERTS:
            raise ValueError(f"experts must be a multiple of 8 in 16..{A.EXPERTS}")
        if self.knobs.slots == "arith" and self.rows != 1:
            raise ValueError("slots=arith is rows 1 only")
        if self.moe.m3_tasks > A.GRID:
            raise ValueError("M3 tasks > grid")
        b = self.bounds_cached
        if self.j > 2:
            raise ValueError(f"{b['gu']} gate_up units on one CTA: more than 2 chains a wave")


_BOUNDS: dict = {}


def _bounds(plan: FusedPlan) -> dict:
    k = plan.knobs
    key = (plan.rows, plan.experts, k.map, k.iod_split, k.sh_ctas, k.sh_base, k.line_order, k.sh_join,
           k.m3_ept if k.line_order == "router_dn" else None)  # fmt: skip
    if key not in _BOUNDS:
        _BOUNDS[key] = plan.bounds()
    return _BOUNDS[key]


# ============================================================ LDS arena

WSLOT_E = MO.MAX_SLOTS + 3  # a wave's slot_e copy (words)
WSLOT_R = 4 * MO.MAX_SLOTS  # a wave's slot_r copy


def lds_layout(plan: FusedPlan):
    """(offsets, strides, words) of the fused body's LDS arena (int32 words)."""
    R = plan.rows
    mp = plan.moe
    bd = plan.bounds_cached
    table = []
    if plan.knobs.router:
        table += [("rw", 8 * A.RW_ROW, 1), ("rx", len(A.ROUTER_PARTS) * A.RX_PART, mp.m3_rows), ("lgl", 8, mp.m3_rows)]
    table += [
        ("ids", A.TOP_K, R),
        ("tw", A.TOP_K, R),
        ("keys", A.TOP_K, R),
        ("rg", 4, 1),
        ("res", 32, R),
        ("inj", 32, R),
        ("first", 4 * A.TOP_K, 1),
        ("slot_e", MO.MAX_SLOTS + 3, 1),
        ("slot_r", 4 * MO.MAX_SLOTS, 1),
        ("nslot", 4, 1),
        ("gx8", A.HIDDEN // 4, R),
        ("gxe", A.HIDDEN // 32, R),
        # gate_up results: unit k (k < 2 J: every wave's chain slots, valid or not) at k * 64
        ("gred", 2 * plan.j * 64, R),
        ("dnA", max(bd["sl"], bd["jsl"], 1) * A.DN_ROW, R),
    ]
    k = plan.knobs
    if k.dn_issue == "lds":  # per wave, dn_lds units of 5 weight tiles (64 lanes x 16 bytes)
        table.append(("dnw", A.WAVES * k.dn_lds * A.DN_STEPS * 256, 1))
    if k.slots == "wave":  # every wave's own slot tables, union bitmap + prefix, slot count
        table += [("wslot_e", A.WAVES * WSLOT_E, 1), ("wslot_r", A.WAVES * WSLOT_R, 1), ("wbm", A.WAVES * 32, 1),
                  ("wn", A.WAVES, 1)]
    if k.combine == "last":  # the tiles this CTA combines (count, then the list)
        table.append(("cl", 4 + bd["dn"] + bd["jn"], 1))
    off, stride, w = {}, {}, 0
    for name, words, rows in table:
        off[name] = w
        stride[name] = MO._pad_rows(words, rows)
        w += stride[name] * rows
    return off, stride, w


# ============================================================ context


class _AreaPix:
    """A logical mailbox pair index whose region lives in the NUMA area (:meth:`FusedCtx.pix`)."""

    __slots__ = ("p",)

    def __init__(self, p: Value):
        self.p = p


class FusedCtx(MO.MoeCtx):
    """:class:`.moe.MoeCtx` over the fused arena (the port's emitters run on it)."""

    def __init__(self, b: IRBuilder, plan: FusedPlan, *, tag: Value, layer: Value, addr: dict):
        self.b = b
        self.fplan = plan
        self.fk = plan.knobs
        self.plan = plan.moe
        self.k = self.plan.knobs
        self.R = plan.rows
        self.E = plan.experts
        self.tag = tag
        if plan.knobs.mutate == "xtag_or":  # test only: the old OR tag (equals the launch tag from step 2^25 on)
            self.xtag = b.lor(tag, b.const_i32(XTAG_BIT))
        else:
            self.xtag = b.xor(tag, b.const_i32(XTAG_BIT))
        self.layer = layer
        self.addr = addr
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = b.readfirstlane(b.lshr(self.tid, b.const_i32(6)))
        self.bid = b.readfirstlane(b.block_id_x())
        self.off, self.stride, words = lds_layout(plan)
        if words > MO.LDS_LIMIT_WORDS:
            raise ValueError(f"LDS arena {words} words > {MO.LDS_LIMIT_WORDS}")
        self.lds = b.smem_alloc(I32, [words], name_hint="fmoe_lds")
        self.mb = M.mailbox_from_i64(b, addr["scratch"], b.const_i32(-1), self.k.mailbox)
        self.stamp_lds = None
        self.stage_ix = None
        self.mb_area = None
        if self.fk.uses_area:  # ATOM's NUMA area: its base, the header's page mask with this CTA's IOD folded in
            hdr = self.fk.numa_hdr or NUMA_HDR
            a = b.add(addr["scratch"], b.const_i64(hdr + 256 + NUMA_PAGE - 1))
            area = b.land(a, b.const_i64(-NUMA_PAGE))
            self.mb_area = M.mailbox_from_i64(b, area, b.const_i32(-1), self.k.mailbox)
            hp = b.int_to_ptr(b.add(addr["scratch"], b.const_i64(hdr)), PtrType(I32, "global"))
            mask = b.readfirstlane(b.global_load(hp, b.const_i32(0), I32, align=4))
            self.sel_mask = b.readfirstlane(b.xor(mask, b.sub(b.const_i32(0), M.iod_id(b))))

    # ---------------------------------------------------- NUMA area (knobs numa / lg_copy / xchg_home)
    def in_area(self, region: str) -> bool:
        fk = self.fk
        if self.mb_area is None:
            return False
        if region in ("lg", "sg"):
            return fk.lg_copy == "numa2"
        if region == "dn":
            return fk.xchg_home == "numa2"
        return bool(fk.numa)

    def _copy0(self, p: Value) -> Value:
        """Area pair index of copy 0 of logical pair ``p`` (ATOM ``numa.phys``)."""
        b = self.b
        x = b.mul(p, b.const_i32(A.PAIR))
        return b.lshr(b.add(x, b.land(x, b.const_i32(-NUMA_CHUNK))), b.const_i32(3))

    def _local(self, p: Value) -> Value:
        """Area pair index of the copy of logical pair ``p`` homed at this CTA's IOD."""
        b = self.b
        x = b.mul(p, b.const_i32(A.PAIR))
        xi = b.add(x, b.land(x, b.const_i32(-NUMA_CHUNK)))
        sel = b.land(b.lshr(self.sel_mask, b.lshr(xi, b.const_i32(21))), b.const_i32(1))
        return b.lshr(b.add(xi, b.shl(sel, b.const_i32(13))), b.const_i32(3))

    def pix(self, region: str, i, row=0):
        p = super().pix(region, i, row)
        return _AreaPix(p) if self.in_area(region) else p

    def poll(self, specs, route=False):
        if not any(isinstance(i, _AreaPix) for i, _n in specs):
            return super().poll(specs, route)
        kn = self.k.mailbox
        if route and self.k.route_pause:
            kn = replace(kn, pause=self.k.route_pause)
        lst = [(self.mb_area, self._local(i.p), n) if isinstance(i, _AreaPix) else (self.mb, i, n) for i, n in specs]
        out = M.poll(self.b, lst, self.tag, kn)
        self.stamp_ready()
        return out

    def put(self, i, v):
        if isinstance(i, _AreaPix):
            M.put(self.b, self.mb_area, self._copy0(i.p), v, self.tag, replace(self.k.mailbox, numa_copies=2))
        else:
            super().put(i, v)

    def put2(self, i, v0, v1):
        if isinstance(i, _AreaPix):
            M.put2(self.b, self.mb_area, self._copy0(i.p), v0, v1, self.tag, replace(self.k.mailbox, numa_copies=2))
        else:
            super().put2(i, v0, v1)

    def ys_pix(self, r, rank, cp):
        """Pair index of ys row ``r``, rank ``rank``, column pair ``cp``."""
        base = A.SC["ss"] // A.PAIR
        if isinstance(r, int):
            return self.add(base + r * A.ROW_PAIRS, self.mul(rank, YS_PAIRS), cp)
        return self.add(self.mul(r, A.ROW_PAIRS), base, self.mul(rank, YS_PAIRS), cp)

    def phase(self, i: int) -> None:
        """Phase stamp ``i`` (knob ``phase_stamps``): thread 0, one 8-byte store."""
        if not self.fk.phase_stamps:
            return
        b = self.b
        with b.scf_if(self.eq(self.tid, 0)):
            t = D.memrealtime(b)
            base = b.add(self.addr["scratch"], b.const_i64(A.SC["ss"] + STAMP_OFF))
            ptr = b.int_to_ptr(base, PtrType(I64, "global"))
            b.global_store(ptr, self.add(self.mul(self.bid, STAMP_PTS), i), t, align=8)


# ============================================================ IR pieces


def _div(e: FusedCtx, x: Value, n) -> Value:
    """``x // n``, ``x >= 0``; ``n`` an int or a uniform i32."""
    if isinstance(n, int):
        return e.udiv(x, n)
    return e.b.div(x, n)


def emit_role(e: FusedCtx):
    """Uniform (is_shared, t, N) of this CTA (``N`` an int when every line has the same count)."""
    b, fk, fp = e.b, e.fk, e.fplan
    lo, hi = fk.sh_base, fk.sh_base + fk.sh_ctas
    bid = e.bid
    is_sh = b.land(e.ge(bid, lo), e.lt(bid, hi))
    q = b.sub(bid, e.c(lo))
    ns = fp.lines_n
    if fk.iod_split:
        iod = e.udiv(e.umod(bid, 8), 4)

        def count(x, i_val):  # _count_iod with a runtime IOD
            low = e.imin(e.imax(b.sub(e.umod(x, 8), e.mul(i_val, 4)), 0), 4)
            return e.add(e.mul(b.lshr(x, e.c(3)), 4), low)

        def sh_before(x):
            return b.sub(count(e.imin(e.imax(x, lo), hi), iod), count(e.c(lo), iod))

        t = b.sub(count(bid, iod), sh_before(bid))
        n = ns[0] if ns[0] == ns[1] else e.sel(e.eq(iod, 0), ns[0], ns[1])
    else:
        iod = e.c(0)
        t = b.sub(bid, e.sel(e.ge(bid, hi), fk.sh_ctas, 0))
        n = ns[0]
    if fk.line_order == "router_dn":  # the router CTAs to the line's end (FusedPlan.rot)
        pes = [fp.rot(i) for i in range(len(ns))]
        pe = pes[0] if len(set(pes)) == 1 else e.sel(e.eq(iod, 0), pes[0], pes[1])
        t = b.sub(t, e.c(pe))
        t = e.sel(e.lt(t, 0), e.add(t, n), t)
    return is_sh, q, t, n, iod


def _line_s(e: FusedCtx, iod):
    """``sh_join``: the shared CTAs of this IOD's line (int when equal on every line)."""
    ss = e.fplan.lines_s
    if len(ss) == 1 or ss[0] == ss[1]:
        return ss[0]
    return e.sel(e.eq(iod, 0), ss[0], ss[1])


def emit_line(e: FusedCtx, t: Value, n, ul: Value):
    """(g0, g1, d0, d1) of CTA ``t`` of a line of ``n`` CTAs and ``ul`` slots (uniform; FusedPlan.line)."""
    b = e.b
    g = e.mul(ul, GU_UNITS)
    dt = e.mul(ul, DN_UNITS)
    t1 = e.add(t, 1)
    g0 = _div(e, e.mul(t, g), n)
    g1 = _div(e, e.mul(t1, g), n)
    if e.fk.map == "groups":
        return g0, g1, _div(e, e.mul(t, dt), n), _div(e, e.mul(t1, dt), n)
    w = e.add(e.mul(g, GU_WEIGHT), dt)

    def raw(s):
        return b.sub(_div(e, e.mul(s, w), n), e.mul(_div(e, e.mul(s, g), n), GU_WEIGHT))

    best = e.c(0)
    for m in range(A.GRID // 64):
        s = e.add(e.lane, 64 * m)
        best = e.imax(best, e.sel(b.cmp_le(s, t), raw(s), 0))
    cm = D.wave_umax_dpp(b, best)
    d0 = e.imin(cm, dt)
    d1 = e.imin(e.imax(cm, raw(t1)), dt)
    return g0, g1, d0, d1


def emit_line_x(e: FusedCtx, t: Value, n, ul: Value, s_join):
    """:meth:`FusedPlan.line` with ``line_order=router_dn`` (gate_up units dealt front-first) and / or
    ``sh_join`` (``s_join`` shared positions past the ``n`` routed ones, weight ``sh_join`` / 8): a routed line's
    CTA ``t`` (routed or joined), uniform."""
    b, fk = e.b, e.fk
    g = e.mul(ul, GU_UNITS)
    dt = e.mul(ul, DN_UNITS)
    t1 = e.add(t, 1)
    front = fk.line_order == "router_dn"
    join = fk.sh_join > 0
    if front:
        qg = _div(e, g, n)
        rem = b.sub(g, e.mul(qg, n))

    def g0(s):
        v = e.add(e.mul(s, qg), e.imin(s, rem)) if front else _div(e, e.mul(s, g), n)
        if join:
            v = e.sel(e.ge(s, n), g, v)
        return v

    if join:
        tot = e.add(e.mul(n, 8), e.mul(s_join, fk.sh_join))
        if isinstance(n, int) and isinstance(s_join, int):
            tot = 8 * n + fk.sh_join * s_join

        def cum(s):
            return e.add(e.mul(e.imin(s, n), 8), e.mul(e.imax(b.sub(s, e.c(n)), 0), fk.sh_join))

    if fk.map == "groups":
        if not join:
            return g0(t), g0(t1), _div(e, e.mul(t, dt), n), _div(e, e.mul(t1, dt), n)
        return g0(t), g0(t1), _div(e, e.mul(dt, cum(t)), tot), _div(e, e.mul(dt, cum(t1)), tot)
    w = e.add(e.mul(g, GU_WEIGHT), dt)

    def raw(s):
        base = _div(e, e.mul(w, cum(s)), tot) if join else _div(e, e.mul(s, w), n)
        return b.sub(base, e.mul(g0(s), GU_WEIGHT))

    best = e.c(0)
    for m in range(A.GRID // 64):
        s = e.add(e.lane, 64 * m)
        best = e.imax(best, e.sel(b.cmp_le(s, t), raw(s), 0))
    cm = D.wave_umax_dpp(b, best)
    d0 = e.imin(cm, dt)
    d1 = e.imin(e.imax(cm, raw(t1)), dt)
    return g0(t), g0(t1), d0, d1


def _tab(e: FusedCtx, name: str):
    """LDS word offset of slot table ``name`` (``slot_e`` / ``slot_r``): this wave's copy with ``slots=wave``."""
    if e.fk.slots == "wave":
        return e.add(e.off["w" + name], e.mul(e.wave, WSLOT_E if name == "slot_e" else WSLOT_R))
    return e.off[name]


def _slot_ent(e: FusedCtx, is_sh, lo, local):
    """Expert id (SHARED for the shared group) of line slot ``local`` (uniform), via the slot table."""
    if e.fk.slots == "arith":
        ent = e.ld_u(e.ix("ids", e.add(lo, local), 0))
    else:
        ent = e.ld_u(e.add(_tab(e, "slot_e"), e.add(lo, local, 1)))
    return e.sel(is_sh, MO.SHARED, ent)


def _slot_rank(e: FusedCtx, is_sh, lo, local, r, uniform: bool):
    """Rank request row ``r`` has in line slot ``local`` (-1: none; the shared group: 10)."""
    if e.fk.slots == "arith":
        rk = e.add(lo, local)
    else:
        idx = e.add(_tab(e, "slot_r"), e.mul(e.add(lo, local, 1), 4), r)
        rk = e.ld_u(idx) if uniform else e.ld(idx)
    return e.sel(is_sh, A.TOP_K, rk)


def _union_by_id_at(e: FusedCtx, my: Value, valid: Value, bm):
    """:func:`.moe._union_by_id` on this wave's LDS words at ``bm`` (16 bitmap words, then their prefix)."""
    b = e.b
    lane = e.lane
    nw = -(-e.E // 32)
    with b.scf_if(e.lt(lane, nw)):
        e.st(e.add(bm, lane), e.c(0))
    lds0 = b.trunc(b.smem_addr_of(e.lds), I32)
    with b.scf_if(valid):
        addr = e.add(lds0, e.mul(e.add(bm, b.lshr(my, e.c(5))), 4))
        bit = b.shl(e.c(1), b.land(my, e.c(31)))
        b.inline_asm("ds_or_b32 $0, $1\n\ts_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, bit], None)
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
    w = b.lshr(my, e.c(5))
    below = b.land(e.ld(e.add(bm, w)), b.sub(b.shl(e.c(1), b.land(my, e.c(31))), e.c(1)))
    slot = e.add(e.ld(e.add(bm, nw, w)), MO._bcnt(b, below))
    u = b.readlane(inc, e.c(nw - 1))
    return slot, u


def _build_wave_slots(e: FusedCtx) -> None:
    """``slots=wave``: this wave's slot tables (the shared slot first, the routed slots by expert id, as
    :func:`.moe._build_slots`), no barrier: every wave reads only its own copy."""
    b = e.b
    R, K = e.R, A.TOP_K
    n = R * K
    lane = e.lane
    se, sr = _tab(e, "slot_e"), _tab(e, "slot_r")
    for qq in range(-(-WSLOT_R // 64)):
        i = e.add(lane, 64 * qq)
        with b.scf_if(e.lt(i, WSLOT_R)):
            e.st(e.add(sr, i), e.c(-1))
    t = e.imin(lane, n - 1)
    row, rank = e.udiv(t, K), e.umod(t, K)
    my = e.ld(e.ix("ids", rank, row))
    valid = e.lt(lane, n)
    slot, u = _union_by_id_at(e, my, valid, e.add(e.off["wbm"], e.mul(e.wave, 32)))
    with b.scf_if(valid):
        e.st(e.add(se, slot, 1), my)
        e.st(e.add(sr, e.mul(e.add(slot, 1), 4), row), rank)
    with b.scf_if(e.eq(lane, 0)):
        e.st(se, e.c(MO.SHARED))
        for r in range(R):
            e.st(e.add(sr, r), e.c(K))
        e.st(e.add(e.off["wn"], e.wave), e.add(u, 1))


def _n_routed(e: FusedCtx) -> Value:
    """U (uniform) after the slot tables (``slots`` table / wave) or 10 (arith)."""
    b = e.b
    if e.fk.slots == "table":
        return b.sub(e.ld_u(e.off["nslot"]), e.c(1))
    if e.fk.slots == "wave":
        return b.sub(e.ld_u(e.add(e.off["wn"], e.wave)), e.c(1))
    return e.c(A.TOP_K)


def _addrs(e: FusedCtx, ent, kind: str):
    """(weight rsrc, scale rsrc) of expert ``ent`` (SHARED: the shared expert), ``kind`` gu | dn."""
    b = e.b
    shared = e.eq(ent, MO.SHARED)
    ex64 = b.zext(e.imax(ent, 0), I64)
    wb, sb = (A.GU_W_BYTES, A.GU_S_BYTES) if kind == "gu" else (A.DN_W_BYTES, A.DN_S_BYTES)
    aw = b.select(shared, e.addr[f"sh_w_{kind}"], b.add(e.addr[f"e_w_{kind}"], b.mul(ex64, b.const_i64(wb))))
    asc = b.select(shared, e.addr[f"sh_s_{kind}"], b.add(e.addr[f"e_s_{kind}"], b.mul(ex64, b.const_i64(sb))))
    return MO._rsrc(e, b.readfirstlane(aw), wb), MO._rsrc(e, b.readfirstlane(asc), sb)


_GUARD_N = [0]


def _flatten(x, shape):
    if isinstance(x, (list, tuple)):
        shape.append((type(x), len(x)))
        out = []
        for y in x:
            out += _flatten(y, shape)
        return out
    shape.append(None)
    return [x]


def _unflatten(vals, shape, it):
    kind = shape[it[0]]
    it[0] += 1
    if kind is None:
        v = vals[it[1]]
        it[1] += 1
        return v
    typ, n = kind
    items = [_unflatten(vals, shape, it) for _ in range(n)]
    return items if typ is list else typ(items)


def _guarded(e: FusedCtx, cond, fn, tag: str):
    """``fn()``'s loads (a nested list / tuple of i32 or <N x i32> values) only when ``cond`` (uniform i1): a
    0/1-trip loop yielding them, zeros otherwise (the attention kernels' guarded first-step loads). ``fn`` is
    traced twice: once into a detached region for the result types, once into the loop."""
    from rocke.core.ir import Region, VectorType

    b = e.b
    _GUARD_N[0] += 1
    uid = f"fg{_GUARD_N[0]}{tag}"
    b.push_region(Region("guard_types"))  # detached: only the result types are used
    try:
        probe = _flatten(fn(), [])
    finally:
        b.pop_region()
    inits = []
    for v in probe:
        if isinstance(v.type, VectorType):
            inits.append(b.vec_pack([b.const_i32(0)] * v.type.count, I32))
        else:
            inits.append(b.const_i32(0))
    zero, one = b.const_i32(0), b.const_i32(1)
    loop = b.scf_for_iter(zero, b.select(cond, one, zero), one, [(f"{uid}_{i}", z) for i, z in enumerate(inits)],
                          iv_name=f"{uid}_it")
    shape = []
    with loop as (_it, _cur):
        b.scf_yield(*_flatten(fn(), shape))
    return _unflatten(list(loop.results), shape, [0, 0])


def _off(e: FusedCtx, valid) -> Value:
    if e.fk.w_null:
        return e.c(MO.OOB)
    return e.sel(valid, 0, MO.OOB)


def _line_span(e: FusedCtx, U: Value, iod):
    """(first routed slot, slots) of this IOD's line for ``U`` routed slots (uniform; FusedPlan.line_slots)."""
    b, fp, fk = e.b, e.fplan, e.fk
    if fk.iod_split:
        ns = fp.lines_n
        n_all = sum(ns)
        u0 = _div(e, e.add(e.mul(U, ns[0]), n_all // 2), n_all)
        return e.sel(e.eq(iod, 0), 0, u0), e.sel(e.eq(iod, 0), u0, b.sub(U, u0))
    return e.c(0), U


def _rt_line(e: FusedCtx, t_rt, n_rt, ul_rt, iod, const: bool):
    """A routed line's (g0, g1, d0, d1) with ``line_order`` / ``sh_join`` (:func:`emit_line_x`); ``const``: every
    line's sizes as constant divisors, then a select by IOD."""
    b, fp = e.b, e.fplan
    if not const:
        return emit_line_x(e, t_rt, n_rt, ul_rt, _line_s(e, iod))
    lines = list(zip(fp.lines_n, fp.lines_s))
    out = emit_line_x(e, t_rt, lines[0][0], ul_rt, lines[0][1])
    if len(lines) > 1 and lines[1] != lines[0]:
        o1 = emit_line_x(e, t_rt, lines[1][0], ul_rt, lines[1][1])
        pick1 = e.eq(iod, 1)
        out = tuple(b.select(pick1, x1, x0) for x0, x1 in zip(out, o1))
    return out


def emit_work(e: FusedCtx, U: Value, is_sh, q, t_rt, n_rt, iod):
    """This CTA's (first line slot, line slots, g0, g1, d0, d1) for ``U`` routed slots (uniform;
    :meth:`FusedPlan.assignment`); ``is_sh .. iod`` from :func:`emit_role`."""
    b, fp, fk = e.b, e.fplan, e.fk
    lo_rt, ul_rt = _line_span(e, U, iod)
    t = e.sel(is_sh, q, t_rt)
    ul = e.sel(is_sh, 1, ul_rt)
    lo = e.sel(is_sh, 0, lo_rt)
    xline = fk.line_order != "slot" or fk.sh_join > 0
    if fk.line_impl == "const":
        # every line size a constant divisor: the shared line, each routed line size; then selects
        sh = emit_line(e, q, fk.sh_ctas, e.c(1))
        if xline:
            rt = _rt_line(e, t_rt, n_rt, ul_rt, iod, True)
        else:
            ns_rt = sorted(set(fp.lines_n))
            rt = emit_line(e, t_rt, ns_rt[0], ul_rt)
            if len(ns_rt) > 1:
                rt1 = emit_line(e, t_rt, ns_rt[1], ul_rt)
                pick1 = e.eq(n_rt, ns_rt[1])
                rt = tuple(b.select(pick1, x1, x0) for x0, x1 in zip(rt, rt1))
        g0, g1, d0, d1 = (b.select(is_sh, x_s, x_r) for x_s, x_r in zip(sh, rt))
    elif xline:
        sh = emit_line(e, q, fk.sh_ctas, e.c(1))
        rt = _rt_line(e, t_rt, n_rt, ul_rt, iod, False)
        g0, g1, d0, d1 = (b.select(is_sh, x_s, x_r) for x_s, x_r in zip(sh, rt))
    else:
        if isinstance(n_rt, int) and n_rt == fk.sh_ctas:
            n = n_rt
        else:
            n = e.sel(is_sh, fk.sh_ctas, n_rt)
        g0, g1, d0, d1 = emit_line(e, t, n, ul)
    return lo, ul, g0, g1, d0, d1


def emit_join_work(e: FusedCtx, U: Value, iod, q):
    """``sh_join``: a shared CTA's (first line slot, line slots, d0, d1) of joined routed down units: position
    ``n + j`` of its IOD's line (``j``: shared CTAs of this IOD before it)."""
    b, fp, fk = e.b, e.fplan, e.fk
    lo, ul = _line_span(e, U, iod)
    ns = fp.lines_n
    n = ns[0] if len(ns) == 1 or ns[0] == ns[1] else e.sel(e.eq(iod, 0), ns[0], ns[1])
    if fk.iod_split:
        bid = e.bid

        def count(x):
            low = e.imin(e.imax(b.sub(e.umod(x, 8), e.mul(iod, 4)), 0), 4)
            return e.add(e.mul(b.lshr(x, e.c(3)), 4), low)

        j = b.sub(count(bid), count(e.c(fk.sh_base)))
    else:
        j = q
    _g0, _g1, d0, d1 = _rt_line(e, e.add(n, j), n, ul, iod, fk.line_impl == "const")
    return lo, ul, d0, d1


def _xchg_hint(e: FusedCtx, items: int, item_rank) -> None:
    """Wave 0 spins on the exponent pairs of every (slot, row) this CTA reads, then a bare barrier."""
    b = e.b
    with b.scf_if(e.eq(e.wave, 0)):
        nh = items * (A.MOE_INTER // 32)
        for m0 in range(0, nh, 64):
            m = e.imin(e.add(e.lane, m0), nh - 1)
            it = e.udiv(m, A.MOE_INTER // 32)
            x = e.umod(m, A.MOE_INTER // 32)
            ok, _sl, r, rank = item_rank(it)
            with b.scf_if(b.land(ok, e.lt(e.add(e.lane, m0), nh))):
                e.poll([(e.pix("dn", e.add(e.mul(rank, A.DN_ROW), A.DN_X8, x), r), 1)])
    b.s_barrier_bare(fence="compiler")


HIST_REGIONS = ("sg", "selh", "inj", "minj")  # regions whose 256-byte rounding leaves >= 192 unused bytes a row
HIST_LAYERS = 48


def hist_offset(layer: int) -> int:
    """Scratch byte offset of layer ``layer``'s 64-byte union history (``spec=prev``): the unused tail of a
    small region's 256-byte rounding (``sg``, ``selh``, ``inj``, ``minj``; rows 0..3), three layers a tail,
    at bytes 64, 128, 192 of it. No stage writes there."""
    if not 0 <= layer < HIST_LAYERS:
        raise ValueError(layer)
    c = layer // 3
    return A.SC[HIST_REGIONS[c // 4]] + (c % 4) * A.SCRATCH_BYTES + 64 * (1 + layer % 3)


def _hist_addr(e: FusedCtx) -> Value:
    """i64 address of this launch's layer history (layer < HIST_LAYERS; else layer 47's, never touched)."""
    b = e.b
    lay = e.imin(e.imax(e.layer, 0), HIST_LAYERS - 1)
    c = e.udiv(lay, 3)
    reg = e.udiv(c, 4)
    base = e.c(A.SC[HIST_REGIONS[0]])
    for i, name in enumerate(HIST_REGIONS[1:], 1):
        base = e.sel(e.eq(reg, i), A.SC[name], base)
    off = e.add(base, e.mul(e.umod(c, 4), A.SCRATCH_BYTES), e.mul(e.add(e.umod(lay, 3), 1), 64))
    return b.add(e.addr["scratch"], b.zext(off, I64))


def _spec_ok(e: FusedCtx) -> Value:
    return e.lt(e.layer, HIST_LAYERS)


def _ffs_ballot(e: FusedCtx, cond) -> Value:
    """Index of the lowest lane where ``cond`` holds (uniform; 64 when none)."""
    b = e.b
    m = b.wave_ballot(b.select(cond, e.c(1), e.c(0)))
    lo = b.readfirstlane(b.trunc(m, I32))
    hi = b.readfirstlane(b.trunc(b.lshr(m, b.const_i64(32)), I32))
    clo = b.inline_asm("s_ff1_i32_b32 $0, $1", "=s,s", [lo], I32, sideeffect=False, result_name_hint="ff1")
    chi = b.inline_asm("s_ff1_i32_b32 $0, $1", "=s,s", [hi], I32, sideeffect=False, result_name_hint="ff1")
    return b.select(b.cmp_ne(lo, e.c(0)), clo, b.select(b.cmp_ne(hi, e.c(0)), e.add(chi, 32), e.c(64)))


def _spec_prefetch(e: FusedCtx, routed):
    """``spec=prev``: waves >= R of a routed CTA touch the history experts' weight lines (FusedKnobs.spec).
    Returns the loaded dwords (sunk after the routing barrier)."""
    b, fk = e.b, e.fk
    R = e.R
    GU_I = 200  # 8 KiB instructions of an expert's used gate_up blocks (80 blocks x 20 KiB)
    DN_I = 120  # ... of its down blocks (160 n-blocks x 6 KiB)
    lane = e.lane
    pw = A.WAVES - R  # touching waves a CTA
    on = b.land(b.land(routed, e.ge(e.wave, R)), _spec_ok(e))
    hp = b.int_to_ptr(_hist_addr(e), PtrType(I32, "global"))
    w = b.select(e.lt(lane, 16), b.global_load(hp, e.imin(lane, 15), I32, align=4), e.c(0))
    cnt = MO._bcnt(b, w)
    excl = e.c(0)  # exclusive prefix of the 16 words' popcounts
    total = e.c(0)
    for l_ in range(16):
        cl = b.readlane(cnt, e.c(l_))
        excl = b.add(excl, b.select(e.lt(e.c(l_), lane), cl, e.c(0)))
        total = b.add(total, cl)
    n_ins = e.mul(total, GU_I + DN_I)
    gu_ins = e.mul(total, GU_I)
    nw = A.GRID * pw
    g = e.add(e.mul(e.bid, pw), b.sub(e.wave, e.c(R)))
    bit_l = e.umod(lane, 32)
    vals = []
    for s_ in range(fk.spec_loads):
        i = e.add(g, s_ * nw)
        in_gu = e.lt(i, gu_ins)
        j = b.sub(i, b.select(in_gu, e.c(0), gu_ins))
        per = b.select(in_gu, e.c(GU_I), e.c(DN_I))
        k = b.div(j, per)
        chunk = b.sub(j, e.mul(k, per))
        kc = e.imin(k, e.imax(b.sub(total, e.c(1)), 0))  # the kc-th set bit: its word, then its bit
        mine = b.land(b.land(e.lt(lane, 16), b.cmp_le(excl, kc)), e.lt(kc, e.add(excl, cnt)))
        wi = e.imin(_ffs_ballot(e, mine), 15)
        word = b.readlane(w, wi)
        kk = b.sub(kc, b.readlane(excl, wi))
        below = MO._bcnt(b, b.land(word, b.sub(b.shl(e.c(1), bit_l), e.c(1))))
        hit = b.land(b.land(e.lt(lane, 32), b.cmp_ne(b.land(b.lshr(word, bit_l), e.c(1)), e.c(0))), e.eq(below, kk))
        bit = e.imin(_ffs_ballot(e, hit), 31)
        ex = e.add(e.mul(wi, 32), bit)
        valid = b.land(on, e.lt(i, n_ins))
        if fk.spec_keep < 10:
            valid = b.land(valid, e.lt(e.umod(k, 10), fk.spec_keep))
        ex64 = b.zext(ex, I64)
        base = b.select(in_gu, b.add(e.addr["e_w_gu"], b.mul(ex64, b.const_i64(A.GU_W_BYTES))),
                        b.add(e.addr["e_w_dn"], b.mul(ex64, b.const_i64(A.DN_W_BYTES))))
        # bounded by the block: an invalid touch's offset past it reads nothing (no fault, no bytes)
        rs = MO._rsrc(e, b.readfirstlane(base), b.select(in_gu, e.c(A.GU_W_BYTES), e.c(A.DN_W_BYTES)))
        off = b.select(valid, e.add(e.mul(chunk, 8192), e.mul(lane, 128)), e.c(MO.OOB))
        vals.append(b.buffer_load(rs, off, e.c(0), I32))
    return vals


def _spec_sink(e: FusedCtx, vals) -> None:
    """Use the touched dwords (an OR, then a store under a never-true condition) so the loads stay."""
    b = e.b
    acc = vals[0]
    for v in vals[1:]:
        acc = b.lor(acc, v)
    with b.scf_if(b.land(e.eq(acc, 0x5EC0FFEE), e.ge(e.tid, A.THREADS))):  # tid < THREADS: never
        hp = b.int_to_ptr(_hist_addr(e), PtrType(I32, "global"))
        b.global_store(hp, e.c(0), acc, align=4)


def _spec_record(e: FusedCtx) -> None:
    """One routed CTA writes this launch's expert union (a 512-bit bitmap of the rows' top-10 ids in LDS)."""
    b, fk = e.b, e.fk
    writer = (fk.sh_base + fk.sh_ctas) % A.GRID
    with b.scf_if(b.land(b.land(e.eq(e.bid, writer), e.eq(e.wave, 0)), _spec_ok(e))):
        lane = e.lane
        word = e.c(0)
        for r in range(e.R):
            for k in range(A.TOP_K):
                x = e.ld_u(e.ix("ids", k, r))
                bit = b.shl(e.c(1), b.land(x, e.c(31)))
                word = b.lor(word, b.select(e.eq(b.lshr(x, e.c(5)), lane), bit, e.c(0)))
        with b.scf_if(e.lt(lane, 16)):
            hp = b.int_to_ptr(_hist_addr(e), PtrType(I32, "global"))
            b.global_store(hp, lane, word, align=4)


def _route_gate_area(e: FusedCtx) -> None:
    """:func:`.moe._route_gate` with the logits in the NUMA area (``lg_copy=numa2``): wave 0 spins on the local
    copies, then the CTA barriers."""
    b = e.b
    per_row = e.E // 8

    def hint(m):
        t, r = (m, 0) if e.R == 1 else (e.umod(m, per_row), e.udiv(m, per_row))
        return e.mb_area, e._local(MO.MoeCtx.pix(e, "lg", e.add(e.mul(t, 4), 2), r))

    kn = e.k.mailbox
    if e.k.route_pause:
        kn = replace(kn, pause=e.k.route_pause)
    M.pre_poll(b, hint, per_row * e.R, e.tag, kn)


def _route(e: FusedCtx, weights: bool, keys: bool = False) -> None:
    """:func:`.moe._route_rows` with this build's route gate (the area's local logits with ``lg_copy=numa2``)."""
    gate = e.fk.route_gate == "pre_poll"
    if gate and e.in_area("lg"):
        _route_gate_area(e)
        MO._route_rows(e, weights=weights, gate=False, keys=keys)
    elif keys:
        MO._route_rows(e, weights=weights, gate=gate, keys=keys)
    else:
        MO._route_rows(e, weights=weights, gate=gate)


def _dn_unit(e: FusedCtx, kk: int, d0, n_dn, dn_last, lo, is_sh, rot, sl0, tag: str, lds_kk=None):
    """Down unit ``kk`` of this wave (CTA unit ``8 kk + wave - rot``): its weight / scale loads (``lds_kk``: the
    weight tiles by DMA into the wave's LDS slot ``lds_kk``, only the scales into registers)."""
    b, fk = e.b, e.fk
    i = b.sub(e.add(e.wave, 8 * kk), rot)
    valid = b.land(e.ge(i, 0), e.lt(i, n_dn))
    d = e.imin(e.add(d0, e.imax(i, 0)), dn_last)
    local = e.udiv(d, DN_UNITS)
    tn = e.umod(d, DN_UNITS)
    ent = _slot_ent(e, is_sh, lo, local)
    rs_w, rs_s = _addrs(e, ent, "dn")
    if lds_kk is not None:
        if fk.guard == "loads":
            with b.scf_if(valid):
                _dn_dma(e, rs_w, tn, _off(e, valid), lds_kk)
        else:
            _dn_dma(e, rs_w, tn, _off(e, valid), lds_kk)
        return {"lds": lds_kk, "rs_s": rs_s, "valid": valid, "local": local, "sl": b.sub(local, sl0), "tn": tn,
                "shared": e.eq(ent, MO.SHARED)}
    if fk.guard == "loads":
        unit = _guarded(e, valid, lambda: MO._dn_issue(e, rs_w, rs_s, tn, _off(e, valid)), tag)
    else:
        unit = MO._dn_issue(e, rs_w, rs_s, tn, _off(e, valid))
    return {"unit": unit, "valid": valid, "local": local, "sl": b.sub(local, sl0), "tn": tn,
            "shared": e.eq(ent, MO.SHARED)}


def _dnw_word(e: FusedCtx, lds_kk: int, kk: int) -> Value:
    """LDS word of this wave's DMA slot ``lds_kk``, weight tile ``kk`` (64 lanes x 4 words)."""
    per_wave = e.fk.dn_lds * A.DN_STEPS * 256
    return e.add(e.mul(e.wave, per_wave), e.off["dnw"] + (lds_kk * A.DN_STEPS + kk) * 256)


def _dn_dma(e: FusedCtx, rs_w, tn, off, lds_kk: int) -> None:
    """``dn_issue=lds``: a down unit's 5 weight tiles HBM -> LDS (lane l's 16 bytes at word 4 l of the tile)."""
    b = e.b
    arena = b.smem_addr_of(e.lds)
    for kk in range(A.DN_STEPS):
        dst = b.smem_ptr_add(arena, b.zext(e.mul(_dnw_word(e, lds_kk, kk), 4), I64))
        voff = e.add(e.mul(e.add(e.mul(e.add(e.mul(tn, A.DN_KI), kk), 64), e.lane), 16), off)
        b.async_buffer_load_lds_addr(rs_w, dst, voff, e.c(0), 4, coherency=e.fk.w_cpol & 3)


def _dn_scales(e: FusedCtx, du: dict) -> None:
    """The scale loads of an LDS-staged down unit (after the epilogue, as the other units' loads)."""
    fk = e.fk
    valid, tn, rs_s = du["valid"], du["tn"], du["rs_s"]

    def ld():
        return [MO._load1(e, rs_s, e.add(e.mul(e.add(e.mul(e.add(e.mul(e.udiv(tn, 2), A.DN_K1), k1), 64), e.lane),
                                               4), _off(e, valid)))
                for k1 in range(A.DN_K1)]  # fmt: skip

    du["ss"] = _guarded(e, valid, ld, f"ds{du['lds']}") if fk.guard == "loads" else ld()


def _dn_weights(e: FusedCtx, du: dict):
    """(weight tiles, scales) of a down unit: its registers, or the LDS slot it was staged in."""
    if "unit" in du:
        return du["unit"]
    ws = [e.ldn(e.add(_dnw_word(e, du["lds"], kk), e.mul(e.lane, 4)), 4) for kk in range(A.DN_STEPS)]
    return ws, du["ss"]


def _exchange(e: FusedCtx, sl0, d1, n_dn, ul, lo, is_sh, SL: int) -> None:
    """The exchange of a down phase (first slot ``sl0``): the ``dn`` rows of every slot this CTA's down units
    read, every row the slot serves, into LDS ``dnA`` (``xchg_pre=hint``: wave 0 pre-polls the exponents)."""
    b, fk = e.b, e.fk
    R = e.R
    n_sl = e.sel(e.lt(e.c(0), n_dn), e.add(b.sub(e.udiv(e.imax(b.sub(d1, e.c(1)), 0), DN_UNITS), sl0), 1), 0)
    items = SL * R

    def item_rank(m):  # per lane: (valid, sl, r, rank) of item m = sl * R + r
        sl = e.udiv(m, R) if R > 1 else m
        r = e.umod(m, R) if R > 1 else e.c(0)
        local = e.imin(e.add(sl0, sl), e.imax(b.sub(ul, e.c(1)), 0))
        rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
        return b.land(e.lt(sl, n_sl), e.ge(rank, 0)), sl, r, rank

    if fk.xchg_pre == "hint":
        _xchg_hint(e, items, item_rank)
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


def _dn_compute(e: FusedCtx, dns, lo, is_sh, r_l) -> None:
    """The down units' MFMA chains (routed: k_iters 0..4 from zero; shared: the mono's merge order) -> ys."""
    b = e.b
    R = e.R
    if any("lds" in du for du in dns):  # the DMA'd weight tiles landed in LDS
        b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)
    for du in dns:
        valid, tn = du["valid"], du["tn"]
        shift = e.mul(e.umod(tn, 2), 8)
        row = e.ix("dnA", e.mul(du["sl"], A.DN_ROW), r_l)
        ranks = [_slot_rank(e, is_sh, lo, du["local"], r, uniform=True) for r in range(R)]
        with b.scf_if(b.land(valid, b.lnot(du["shared"]))):
            unit = _dn_weights(e, du)
            ops = MO._dn_ops(e, row)
            acc = b.zero_vec_f32(4)
            for kk in range(A.DN_STEPS):
                acc = MO._dn_mma(e, acc, unit, kk, row, shift, ops)
            _ys_puts(e, [MO.bf16r(b, b.vec_extract(acc, r)) for r in range(R)], ranks, tn)
        with b.scf_if(b.land(valid, du["shared"])):
            unit = _dn_weights(e, du)
            ops = MO._dn_ops(e, row)
            zero = b.zero_vec_f32(4)
            w0 = MO._dn_mma(e, MO._dn_mma(e, zero, unit, 0, row, shift, ops), unit, 4, row, shift, ops)
            w1 = MO._dn_mma(e, zero, unit, 1, row, shift, ops)
            w2 = MO._dn_mma(e, zero, unit, 2, row, shift, ops)
            w3 = MO._dn_mma(e, zero, unit, 3, row, shift, ops)
            vals = []
            for r in range(R):
                x = [b.vec_extract(w, r) for w in (w0, w1, w2, w3)]
                vals.append(MO.bf16r(b, b.fadd(b.fadd(b.fadd(x[0], x[1]), x[2]), x[3])))
            _ys_puts(e, vals, [e.c(A.TOP_K)] * R, tn)


def _ctr_incr(e: FusedCtx, d0, n_dn, U, bound: int) -> None:
    """``combine=last``: wave 0 adds one to the counter of every down unit's hidden tile; the unit that brings
    a tile to U + 1 (its slots) resets the counter and lists the tile for this CTA (LDS ``cl``)."""
    b = e.b
    if bound <= 0:
        return
    ptr = b.int_to_ptr(e.addr["scratch"], PtrType(I32, "global"))
    with b.scf_if(e.eq(e.wave, 0)):
        for m0 in range(0, bound, 64):
            i = e.add(e.lane, m0)
            with b.scf_if(e.lt(i, n_dn)):
                tn = e.umod(e.add(d0, i), DN_UNITS)
                widx = e.add(e.mul(e.udiv(tn, CTR_PER_ROW), A.SCRATCH_BYTES // 4), (A.SC[CTR_REGION] + 32) // 4,
                             e.umod(tn, CTR_PER_ROW))  # fmt: skip
                old = b.global_atomic_add(ptr, widx, e.c(1))
                with b.scf_if(e.eq(old, U)):
                    b.global_atomic_add(ptr, widx, b.sub(e.c(-1), U))
                    pos = b.lds_atomic_add(e.lds, [e.c(e.off["cl"])], e.c(1))
                    e.st(e.add(e.off["cl"] + 4, pos), tn)


def _combine_listed(e: FusedCtx) -> None:
    """``combine=last``: the tiles listed in LDS ``cl`` (after a barrier), ``THREADS // 32 R`` side by side;
    thread 32 r + i of a group as the mono's, the extras polled per thread (as ``combine=sh``)."""
    b, fk = e.b, e.fk
    R = e.R
    per = 32 * R
    G = A.THREADS // per
    order = [1, 0] + list(range(2, A.TOP_K)) if fk.mutate == "combine_swap" else None
    cnt = e.ld_u(e.c(e.off["cl"]))
    npass = e.udiv(e.add(cnt, G - 1), G) if G & (G - 1) == 0 else b.div(e.add(cnt, G - 1), e.c(G))
    loc = e.umod(e.tid, per)
    grp = e.udiv(e.tid, per)
    i = e.umod(loc, 32)
    rt = e.udiv(loc, 32)
    with b.scf_for(e.c(0), npass, e.c(1), iv_name="fmcl") as p:
        m = e.add(grp, e.mul(p, G))
        valid = b.land(e.lt(e.tid, G * per), e.lt(m, cnt))
        with b.scf_if(valid):
            t = e.ld(e.add(e.off["cl"] + 4, e.imin(m, e.fplan.bounds_cached["dn"] + e.fplan.bounds_cached["jn"] - 1)))
            cp = e.add(e.mul(t, 8), e.umod(i, 8))
            got = M.poll(b, [_ys_spec(e, rt, k, cp) for k in range(A.TOP_K + 1)], e.xtag, e.k.mailbox)
            vals = [M.bf2_f32(b, got[k][0]) for k in range(A.TOP_K + 1)]
            bs = e.udiv(i, 8)
            word = e.add(e.mul(bs, A.ROW_WORDS), e.mul(t, 8), e.umod(i, 8))
            ext = e.poll([(e.pix("sg", 0, rt), 1), (e.pix("w", word, rt), 1), (e.pix("minj", bs, rt), 1)], route=True)
            MO.combine_tile(
                e, t, lambda k, h, jj, rt_, vals=vals: vals[k][h], order=order, ids=(i, rt),
                rg_at=lambda rt_, ext=ext: b.bitcast(ext[0][0], F32),
                inj_at=lambda i_, rt_, ext=ext: b.bitcast(ext[2][0], F32),
                res_at=lambda i_, rt_, ext=ext: ext[1][0],
            )


def emit_fused(e: FusedCtx) -> None:
    b, fp, fk = e.b, e.fplan, e.fk
    R = e.R
    J = fp.j
    DNJ = fp.dnj
    bd = fp.bounds_cached
    last = fk.combine == "last"
    e.phase(0)
    if fk.router:
        if fk.router_knobs == "mono":  # the router with the mono's own load policy and mailbox (the port's defaults)
            k0, mb0 = e.k, e.mb
            e.k = replace(e.k, w_cpol=0, mailbox=M.MailboxKnobs())
            e.mb = M.mailbox_from_i64(b, e.addr["scratch"], b.const_i32(-1), e.k.mailbox)
            MO.emit_m3(e)
            e.k, e.mb = k0, mb0
        else:
            MO.emit_m3(e)
    e.barrier()
    e.phase(1)
    is_sh, q, t_rt, n_rt, iod = emit_role(e)
    ct = e.umod(e.add(e.bid, A.GRID - fp.combine_base), A.GRID)
    # combine=sh / last: no CTA runs a combine task (extras then come per thread at the combine)
    has_comb = e.lt(ct, DN_UNITS) if fk.combine == "task" else e.lt(ct, -1)
    if last:
        with b.scf_if(e.eq(e.tid, 0)):
            e.st(e.off["cl"], e.c(0))
    # ---- fill, extras, routing, slots
    for r in range(R):
        with b.scf_if(e.eq(e.wave, A.WAVES - A.MAX_ROWS + r)):
            MO._fill_x(e, r)
    routed = b.lnot(is_sh)
    if fk.extras == "first":  # the shared group polls them at its combine (it must not wait for M3 first)
        with b.scf_if(b.land(has_comb, routed)):
            MO._m5_extras(e, ct)
    pf = _spec_prefetch(e, routed) if fk.spec == "prev" else None
    with b.scf_if(routed):
        _route(e, weights=fk.tw_at == "route", keys=fk.tw_at in ("late", "epi"))
    e.barrier()
    if pf is not None:
        _spec_sink(e, pf)
        _spec_record(e)
    if fk.slots == "table":
        with b.scf_if(routed):
            MO._build_slots(e)
    elif fk.slots == "wave":
        with b.scf_if(routed):
            _build_wave_slots(e)
    e.phase(2)
    # ---- line
    U = _n_routed(e)
    lo, ul, g0, g1, d0, d1 = emit_work(e, U, is_sh, q, t_rt, n_rt, iod)
    n_gu = b.sub(g1, g0)
    n_dn = b.sub(d1, d0)
    # ---- gate_up chains: chain c = w + 8 j of the CTA is unit k = 2 j + w // 4, block w % 4
    half = e.udiv(e.wave, 4)
    w4 = e.umod(e.wave, 4)
    pair_in = e.umod(w4, 2)
    part = e.udiv(w4, 2)
    gu_last = e.imax(b.sub(g1, e.c(1)), 0)
    chains = []
    for j in range(J):
        k = e.add(half, 2 * j)
        u = e.add(g0, k)
        valid = e.lt(u, g1)
        uc = e.imin(u, gu_last)
        local = e.udiv(uc, GU_UNITS)
        cb = e.umod(uc, GU_UNITS)
        ent = _slot_ent(e, is_sh, lo, local)
        rs_w, rs_s = _addrs(e, ent, "gu")
        pair = e.add(e.mul(cb, 2), pair_in)
        chains.append((rs_w, rs_s, pair, _off(e, valid), valid))
    issued = [[None] * (A.GU_KI // 4) for _ in range(J)]
    if fk.guard == "loads":  # a chain with no unit issues nothing (0/1-trip loop; zeros otherwise)
        for j in range(J):
            rs_w, rs_s, pair, off, valid = chains[j]
            got = _guarded(e, valid, lambda rs_w=rs_w, rs_s=rs_s, pair=pair, off=off: [
                MO._gu_issue(e, rs_w, rs_s, pair, part, grp, off) for grp in range(A.GU_KI // 4)], f"gu{j}")
            issued[j] = got
    else:
        order = [(grp, j) for grp in range(A.GU_KI // 4) for j in range(J)] if fk.gu_interleave else [
            (grp, j) for j in range(J) for grp in range(A.GU_KI // 4)]
        for grp, j in order:
            rs_w, rs_s, pair, off, _valid = chains[j]
            issued[j][grp] = MO._gu_issue(e, rs_w, rs_s, pair, part, grp, off)
    # ---- down units: unit i of the CTA on wave (i + rot) % 8, slot (i + rot) // 8
    rot = e.mul(e.umod(n_gu, 2), 4)
    dn_last = e.imax(b.sub(e.mul(ul, DN_UNITS), e.c(1)), 0)
    sl0 = e.udiv(d0, DN_UNITS)

    def dn_unit(kk, lds_kk=None):
        return _dn_unit(e, kk, d0, n_dn, dn_last, lo, is_sh, rot, sl0, f"dn{kk}", lds_kk)

    early = min(fk.dn_inflight, DNJ) if fk.dn_issue == "with_gu" else 0
    dns = [dn_unit(kk) for kk in range(early)]
    if fk.dn_issue == "lds":  # the first dn_lds units of the wave: weight tiles by DMA into LDS, now
        early = min(fk.dn_lds, DNJ)
        dns = [dn_unit(kk, lds_kk=kk) for kk in range(early)]
    if fk.tw_at == "late":
        with b.scf_if(routed if last else b.land(has_comb, routed)):
            MO._late_weights(e)
    e.phase(3)
    # ---- gate_up MFMAs into gred (row r: unit k's block w4 at k * 64 + w4 * 16)
    r_l = e.imin(e.umod(e.lane, 16), R - 1)
    x8 = e.ix("gx8", 0, r_l)
    xe = e.ix("gxe", 0, r_l)
    accs = [b.zero_vec_f32(4) for _ in range(J)]

    def a_ops(kit):
        return e.ld(e.add(xe, kit * 4, e.udiv(e.lane, 16))), MO._a_operand(e, x8, kit)

    pipe = fk.gu_apipe
    nxt = a_ops(0) if pipe else None
    for kit in range(A.GU_KI):
        grp, qq = divmod(kit, 4)
        if pipe:
            ea, a_op = nxt
            if kit + 1 < A.GU_KI:
                nxt = a_ops(kit + 1)
            b.sched_barrier(0)
        else:
            ea, a_op = a_ops(kit)
        for j in range(J):
            ws, ss = issued[j][grp]
            sw = b.lshr(ss[qq // 2], e.mul(part, 8))
            accs[j] = MO._mma(e, accs[j], a_op, ws[qq], ea, sw, (qq % 2) * 2)
        if fk.dn_issue == "pipe" and qq == 3:  # k_iter group grp consumed: one more down unit's weights
            kk = early + grp
            if kk < DNJ and grp < A.GU_KI // 4 - 1:
                dns.append(dn_unit(kk))
    if fk.dn_issue == "pipe":
        dns += [dn_unit(kk) for kk in range(early + A.GU_KI // 4 - 1, DNJ)]
    if fk.dn_issue == "after_gu":
        dns += [dn_unit(kk) for kk in range(early, DNJ)]
    with b.scf_if(e.lt(e.lane, 16)):
        for j in range(J):
            col = e.add(e.mul(e.add(half, 2 * j), 64), e.mul(w4, 16), e.lane)
            for r in range(R):
                e.st(e.ix("gred", col, r), b.vec_extract(accs[j], r))
    e.barrier()
    e.phase(4)
    # ---- epilogue: entry (unit k, row r), two per wave a pass
    n_ent = bd["gu"] * R
    c = e.umod(e.lane, 32)
    for p in range(-(-n_ent // 16)):
        ent = e.add(p * 16, e.mul(e.wave, 2), e.udiv(e.lane, 32))
        ent_c = e.imin(ent, n_ent - 1)
        k = e.udiv(ent_c, R) if R > 1 else ent_c
        r = e.umod(ent_c, R) if R > 1 else e.c(0)
        u = e.add(g0, k)
        valid = b.land(e.lt(u, g1), e.lt(ent, n_ent))
        uc = e.imin(u, gu_last)
        local = e.udiv(uc, GU_UNITS)
        cb = e.umod(uc, GU_UNITS)
        rank = _slot_rank(e, is_sh, lo, local, r, uniform=False)
        live = b.land(valid, e.ge(rank, 0))
        base = e.add(e.mul(k, 64), c)
        x = MO.bf16r(b, e.ld(e.ix("gred", base, r), F32))
        y = MO.bf16r(b, e.ld(e.ix("gred", e.add(base, 32), r), F32))
        MO.gu_epilogue_entry(e, x, y, is_sh, live, rank, r, c, cb)
    e.phase(5)
    if fk.dn_issue == "lds":
        for du in dns:
            _dn_scales(e, du)
    if fk.dn_issue not in ("after_gu", "pipe"):
        dns += [dn_unit(kk) for kk in range(early, DNJ)]
    if fk.tw_at == "epi":
        with b.scf_if(routed if last else b.land(has_comb, routed)):
            MO._late_weights(e)
    # ---- exchange: the slots of this CTA's down units, every row the slot serves
    _exchange(e, sl0, d1, n_dn, ul, lo, is_sh, max(bd["sl"], 1))
    e.phase(6)
    # ---- down units -> ys
    _dn_compute(e, dns, lo, is_sh, r_l)
    e.phase(7)
    # ---- combine (hidden tile ct)
    if fk.combine == "sh":
        _combine_sh(e, is_sh, q)
        e.phase(8)
        return
    if last:
        with b.scf_if(routed):
            _ctr_incr(e, d0, n_dn, U, bd["dn"])
    if fk.sh_join or last:
        # the shared group routes now (the combine's router weights; the joined down units' slots)
        with b.scf_if(is_sh):
            _route(e, weights=True)
            e.barrier()
            if fk.slots == "table":
                MO._build_slots(e)
            elif fk.slots == "wave":
                _build_wave_slots(e)
            U_sh = _n_routed(e)
            if last:
                _ctr_incr(e, d0, n_dn, U_sh, bd["dn"])
            if fk.sh_join:
                jlo, jul, j0, j1 = emit_join_work(e, U_sh, iod, q)
                n_jn = b.sub(j1, j0)
                j_last = e.imax(b.sub(e.mul(jul, DN_UNITS), e.c(1)), 0)
                jsl0 = e.udiv(j0, DN_UNITS)
                no = b.cmp_ne(e.c(0), e.c(0))  # i1 false: the joined units are routed slots
                jns = [_dn_unit(e, kk, j0, n_jn, j_last, jlo, no, e.c(0), jsl0, f"jn{kk}") for kk in range(fp.jnj)]
                _exchange(e, jsl0, j1, n_jn, jul, jlo, no, max(bd["jsl"], 1))
                _dn_compute(e, jns, jlo, no, r_l)
                if last:
                    _ctr_incr(e, j0, n_jn, U_sh, bd["jn"])
    if last:
        e.barrier()
        _combine_listed(e)
        e.phase(8)
        return
    with b.scf_if(has_comb):
        if not fk.sh_join:
            with b.scf_if(is_sh):  # the shared group routed nothing so far: the router weights now
                _route(e, weights=True)
                if fk.extras == "first":
                    MO._m5_extras(e, ct)
        elif fk.extras == "first":
            with b.scf_if(is_sh):
                MO._m5_extras(e, ct)
        if fk.extras == "late":
            MO._m5_extras(e, ct)
        e.barrier()
        with b.scf_if(e.lt(e.tid, 32 * R)):
            i = e.umod(e.tid, 32)
            rt = e.udiv(e.tid, 32)
            cp = e.add(e.mul(ct, 8), e.umod(i, 8))
            got = M.poll(b, [_ys_spec(e, rt, k, cp) for k in range(A.TOP_K + 1)], e.xtag, e.k.mailbox)
            vals = [M.bf2_f32(b, got[k][0]) for k in range(A.TOP_K + 1)]
            order = None
            if fk.mutate == "combine_swap":
                order = [1, 0] + list(range(2, A.TOP_K))
            MO.combine_tile(e, ct, lambda k, h, jj, rt_: vals[k][h], order=order)
    e.phase(8)


def _combine_sh(e: FusedCtx, is_sh, q) -> None:
    """``combine=sh``: the shared group (done with its own work first) combines every hidden tile: CTA q of
    the group takes tiles q, q + S, ..; a pass runs THREADS // 32 R tiles side by side (thread group g: tile
    ``q + S (pass * G + g)``, thread 32 r + i of the group as the mono's). The extras (shared gate, residual
    word, injection weight) are polled per thread, not staged in LDS: the same words, the same math."""
    b, fk = e.b, e.fk
    R = e.R
    S = fk.sh_ctas
    nt = -(-DN_UNITS // S)
    per = 32 * R
    G = A.THREADS // per
    order = [1, 0] + list(range(2, A.TOP_K)) if fk.mutate == "combine_swap" else None
    with b.scf_if(is_sh):
        MO._route_rows(e, weights=True, gate=fk.route_gate == "pre_poll")
        e.barrier()
        loc = e.umod(e.tid, per)
        grp = e.udiv(e.tid, per)
        i = e.umod(loc, 32)
        rt = e.udiv(loc, 32)
        for p in range(-(-nt // G)):
            m = e.add(grp, p * G)
            t = e.add(q, e.mul(m, S))
            valid = b.land(b.land(e.lt(e.tid, G * per), e.lt(m, nt)), e.lt(t, DN_UNITS))
            with b.scf_if(valid):
                cp = e.add(e.mul(t, 8), e.umod(i, 8))
                got = M.poll(b, [_ys_spec(e, rt, k, cp) for k in range(A.TOP_K + 1)], e.xtag, e.k.mailbox)
                vals = [M.bf2_f32(b, got[k][0]) for k in range(A.TOP_K + 1)]
                bs = e.udiv(i, 8)
                word = e.add(e.mul(bs, A.ROW_WORDS), e.mul(t, 8), e.umod(i, 8))
                ext = e.poll([(e.pix("sg", 0, rt), 1), (e.pix("w", word, rt), 1), (e.pix("minj", bs, rt), 1)], route=True)
                MO.combine_tile(
                    e, t, lambda k, h, jj, rt_, vals=vals: vals[k][h], order=order, ids=(i, rt),
                    rg_at=lambda rt_, ext=ext: b.bitcast(ext[0][0], F32),
                    inj_at=lambda i_, rt_, ext=ext: b.bitcast(ext[2][0], F32),
                    res_at=lambda i_, rt_, ext=ext: ext[1][0],
                )


def _ys_spec(e: FusedCtx, rt, k, cp):
    """Poll spec of ys pair (row ``rt``, rank ``k``, column pair ``cp``): the plain ``ss`` region, or
    (``ys_home=numa2``) the copy homed at this CTA's IOD in the NUMA area's ``ss``."""
    p = e.ys_pix(rt, k, cp)
    if e.fk.ys_home == "numa2":
        return (e.mb_area, e._local(p), 1)
    return (e.mb, p, 1)


def _ys_put(e: FusedCtx, p, v) -> None:
    """Put ys pair ``p`` under the private tag (``ys_home=numa2``: both NUMA copies)."""
    if e.fk.ys_home == "numa2":
        M.put(e.b, e.mb_area, e._copy0(p), v, e.xtag, replace(e.k.mailbox, numa_copies=2))
    else:
        M.put(e.b, e.mb, p, v, e.xtag, e.k.mailbox)


def _ys_puts(e: FusedCtx, vals, ranks, tn) -> None:
    """Lanes 0..15 hold column ``lane`` of n-block ``tn`` per row: even lanes put (own, next) as one bf16 pair."""
    b = e.b
    nxt = [b.bitcast(b.dpp_xor(b.bitcast(v, I32), 1), F32) for v in vals]
    with b.scf_if(b.land(e.lt(e.lane, 16), e.eq(e.umod(e.lane, 2), 0))):
        cp = e.add(e.mul(tn, 8), e.udiv(e.lane, 2))
        for r in range(e.R):
            with b.scf_if(e.ge(ranks[r], 0)):
                _ys_put(e, e.ys_pix(r, ranks[r], cp), M.bf16_pair(b, vals[r], nxt[r]))


# ============================================================ standalone kernel (split suffix)


def signature():
    return MO.signature()


def build_fused_suffix(rows: int, knobs: FusedKnobs, experts: int = A.EXPERTS):
    """A standalone launch of the fused body with the mono kernel's signature (``knobs.router`` False: after a
    FlyDSL ``a1 .. m3`` prefix; True: after ``a1 .. m2``). Python engine only."""
    plan = FusedPlan(rows, experts, knobs)
    plan.validate()
    b = IRBuilder(f"qwen38_fmoe_e{experts}_r{rows}" + (f"_{knobs.suffix()}" if knobs.suffix() else "") + "_pyonly")
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    step = b.param("step", I64)
    layer = b.param("layer", I32)
    addr = {n: b.param(n, I64) for n in A.LAYER_ARGS}
    step_v = b.global_load(b.int_to_ptr(step, PtrType(I32, "global")), b.const_i32(0), I32, align=4)
    tag = M.launch_tag(b, step_v, b.readfirstlane(layer))
    e = FusedCtx(b, plan, tag=tag, layer=b.readfirstlane(layer), addr=addr)
    emit_fused(e)
    b.ret()
    return b.kernel


__all__ = [
    "PHASES",
    "STAMP_OFF",
    "STAMP_PTS",
    "XTAG_BIT",
    "YS_PAIRS",
    "FusedCtx",
    "FusedKnobs",
    "FusedPlan",
    "build_fused_suffix",
    "emit_fused",
    "emit_line",
    "emit_role",
    "lds_layout",
    "signature",
]  # fmt: skip
