# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Qwen3.8 mono MoE stages in rocKE: M3 router, M4 gate_up, M5 down + combine.

Inline stage emitters over the mono layer kernel's ABI (:mod:`.abi`): they
poll the same mailboxes (M2's mixed row, C1's streams, M1's injection weights),
publish the same ones (``lg``, ``sg``, ``dn``) and write ``h_out`` as the
FlyDSL stages do (``atom/models/qwen4_exp_mono/kernels/stages/moe.py``). Three
users:

* :func:`build_moe_suffix`: a standalone persistent kernel (grid 256 x 512)
  running ``m3 m4 m5`` or ``m4 m5`` with the mono kernel's exact launch
  signature, so it replaces the FlyDSL suffix of a split layer launch
  (FlyDSL ``a1..m2`` or ``a1..m3``, then this) for stage A/B and knob sweeps;
* hybrid builds (rocKE device functions linked into the FlyDSL kernel): the
  emitters take a :class:`MoeCtx` the caller builds from its own arguments;
* the future rocKE layer kernel.

Numerics are the mono stages' bit for bit (the router's 16-part fma chains and
lane-ordered sum, the top-10 key sort, the softmax's exp2 rule, the routed /
shared SiLU variants, the MXFP8 rule, the rank-ordered combine). Request rows
packed as MFMA A rows keep each row's result bitwise (every C row depends only
on its A row), so the expert-union dedupe changes no bit.

Work model (``rows`` request rows, R <= 4):

* *slots*: the expert chains of a launch. Without dedupe slot ``s < 10 R`` is
  rank ``s % 10`` of row ``s // 10`` (mono's task order) and slot ``10 R`` the
  shared expert. With ``dedupe`` the routed slots are the union of the rows'
  top-10 (first occurrence in (row, rank) order, or by expert id); a slot keeps
  per row the rank it has there (or none). The shared expert is the last slot.
* M4 *units*: (slot, column block); a unit is ``m4_waves`` waves, each running
  one 16-column n-block chain over the 20 k_iters (gate blocks on the first
  half of the waves, up blocks on the second): 64 columns (8 waves, mono) or
  32 (4 waves, two units per CTA side by side). Each wave runs up to
  ``j_max`` chains (unit slots of its CTA). ``m4_map``: ``mono`` gives CTA task
  ``t`` units ``t J .. t J + J - 1`` (mono's ``m4_slices``); ``balanced`` gives
  it the contiguous range ``[t n / 256, (t + 1) n / 256)``.
* M5: task ``t`` (160) owns hidden columns ``16 t ..``; slot ``k`` is wave
  ``k % 8``'s chain ``k // 8`` (mono's chain map when not deduped), the A rows
  are the request rows the slot serves, each at its own rank's ``dn`` row.

Knobs (:class:`MoeKnobs`, all default to mono's behaviour; the kernel name names
only the ones that deviate) are listed on the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as Q

from kernels.gfx950 import mx_mfma as MX
from kernels.gfx950 import mx_quant as QL
from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import mailbox_ir  # noqa: F401  (registers MailboxKnobs(impl="ir"))

L2E = 1.4426950408889634
TWO_M64 = 2.0**-64
INT_MIN = -(2**31)
OOB = 1 << 30  # byte offset past every weight block: the load returns zeros
SHARED = -1  # slot_e entry of the shared expert

# Batcher's odd-even merge sort of 8 keys (19 comparators), as mono's top10.
SORT8 = (
    (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2), (5, 6),
    (0, 4), (1, 5), (2, 6), (3, 7), (2, 4), (3, 5), (1, 2), (3, 4), (5, 6),
)  # fmt: skip

ROW_CHUNKS = A.HIDDEN // 8  # 320 chunks of 8 bf16 per router row
RW_LOADS = 8 * ROW_CHUNKS // A.THREADS  # 5
RX_LOADS = A.HIDDEN // 4  # 640 two-pair loads of the bf16 mixed row
CHAIN_STEPS = 24
SHORT_STEPS = 16
X8_LOADS = A.HIDDEN // 8 // 64  # 5
DN_LOADS = (A.TOP_K + 1) * A.DN_ROW // 2  # 990
MAX_SLOTS = A.MAX_ROWS * A.TOP_K + 1  # 41
STAGES = ("m3", "m4", "m5")


# ============================================================ knobs


@dataclass(frozen=True)
class MoeKnobs:
    """MoE stage knobs. Every default is mono's behaviour.

    dedupe: routed slots are the union of the rows' top-10 (rows-4 MTP verify:
      each unique expert's tiles loaded once for all rows routed to it).
    union_order: ``id`` (ascending expert id: an LDS bitmap, a few
      instructions) | ``first`` (first occurrence in (row, rank) order: a
      40-step compare per lane). No numerics effect.
    m4_waves: waves per M4 unit: 8 (64 columns, mono) | 4 (32 columns).
    m4_map: ``mono`` (CTA task t: units t J ..) | ``balanced`` (contiguous
      ranges over all 256 CTAs).
    m4_j: unit slots per wave group (chains per wave); 0 = the smallest that
      holds every unit (mono's ``m4_slices`` for ``mono``).
    m4_base / m5_base / m3_base: first CTA of the stage (-1 = mono's).
    m3_rows: request rows per M3 task (0 = mono's ``m3_task_rows``).
    m3_ept: router rows (experts) per M3 task: 8 (mono) | 4 (twice the tasks,
      half the bytes each; waves 4-7 take the second half of the rows) | 2
      (probe P15, the N3b router: two experts a task, the shared gate a third
      router row of the last task, ``experts / 2`` tasks a row set: every CTA
      at 512 experts with ``m3_rows`` = rows; :func:`_emit_m3_n`).
    w_cpol: cache-policy bits of the weight loads (``buffer_load_cp`` aux:
      0 mono; 2 nt, 16 sc1, 1 sc0, or a sum).
    shared_first: the shared expert is slot 0 (its M4 units on the first M4
      CTAs, its M5 chain on wave 0) instead of the last slot (mono).
    m5_issue: when M5 issues its down weights: ``mono`` (after the first M4
      exponent pair, at once on a CTA that ran M4 with rows > 1) | ``early``
      (right after routing) | ``late`` (after every M4 hint).
    compute: the scaled-MFMA route (:mod:`kernels.gfx950.mx_mfma`), a
      16x16x128 value with op_sel.
    cvt: the MXFP8 convert route (:mod:`kernels.gfx950.mx_quant`).
    mailbox: :class:`rocke.helpers.mailbox.MailboxKnobs`.
    stamps: per-CTA stage stamps (the mono ``ATOM_Q38_STAMPS`` format).
    m5_ranks: how an M5 chain finds the rank each request row has in its slot:
      ``lds`` (an LDS lookup per lane and per row at the chain, divergent
      stores) | ``pre`` (looked up once per chain with the weight issue, into
      SGPRs; arithmetic when not deduped; uniform per-row stores).
    a_pre: M5 chains load all their A operands and exponents from LDS before
      their first MFMA, behind a ``sched_barrier(0)`` (without the barrier the
      machine scheduler sinks the reads back next to their MFMA and every MFMA
      waits a full LDS round trip).
    m4_apipe: M4 (with ``m4_interleave``) reads k_iter ``kit + 1``'s A operand
      and exponent before k_iter ``kit``'s MFMAs, behind a ``sched_barrier(0)``
      (software pipelining of the LDS reads; 0 = off).
    m4_guard: ``branch`` (an M4 chain runs under ``if valid``: LLVM sinks the
      chain's weight loads into the branch) | ``none`` (every chain runs; an
      invalid unit loads zeros past the block, ``OOB``, as mono's do, and its
      result is never read).
    slots: ``table`` (wave 0 writes the slot tables to LDS after routing, then
      a barrier) | ``arith`` (not deduped only: a slot's expert and ranks are
      computed from the routed ids: no table pass, no barrier). Needs
      ``m5_ranks=pre``.
    m5_extras: when M5 polls each row's shared gate, residual words and
      injection weights: ``after`` (after M5's routing) | ``first`` (before
      it: the poll no longer follows the top-10) | ``m4`` (a CTA with an M4
      and an M5 task polls them during M4, after its routing; M5 skips them
      when it reuses M4's routing).
    shared_early: an M4 CTA whose units all belong to the shared expert skips
      the routing (the shared expert needs none) and issues its weights at
      once, as mono's shared-expert tasks do, instead of after M3's logits.
      Needs dedupe off, or ``shared_first`` with ``m4_map=mono``.
    route_pause: the poll-retry pause of the routing polls (the logits, and
      M5's extras), which every M4 and M5-only CTA spins on while M3 runs:
      ``""`` (the mailbox knobs' own) | ``s_sleep1`` (fewer sc1 loads in flight
      against the producers' stores).
    route_gate: ``none`` (the routing waves spin on the logits payload) |
      ``pre_poll`` (wave 0 alone spins on one logits pair per (M3 task, row),
      then a barrier, then one payload read per routing wave: the mono's
      ``pre_poll`` throttle; M4 fills its activation rows before the gate).
    m4_weights: when an M4 CTA computes the router weights (only M5's combine
      uses them): ``route`` (with the top-10, ahead of the barrier that gates
      the weight issue) | ``late`` (the top-10 keys go to LDS; the weights are
      computed from them after the weight loads are issued).
    sched_issue: a ``sched_barrier(0)`` right after M4's and M5's weight
      issue, so that the scheduler cannot sink a later chain's loads among
      an earlier chain's MFMAs (memory-level parallelism over VGPR pressure).
    m4_interleave: a wave's M4 chains advance together, k_iter by k_iter:
      one A operand LDS read feeds every chain's MFMA (each accumulator's
      order is unchanged: bitwise). Needs ``m4_guard=none``.
    """

    dedupe: bool = False
    union_order: str = "id"
    m4_waves: int = 8
    m4_map: str = "mono"
    m4_j: int = 0
    m3_base: int = -1
    m4_base: int = -1
    m5_base: int = -1
    m3_rows: int = 0
    m3_ept: int = 8
    w_cpol: int = 0
    shared_first: bool = False
    m5_issue: str = "mono"
    compute: str = "mfma_16x16x128_op"
    cvt: str = "intrinsic"
    mailbox: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    stamps: bool = False
    m5_ranks: str = "lds"
    a_pre: bool = False
    m4_guard: str = "branch"
    sched_issue: bool = False
    m4_interleave: bool = False
    m4_apipe: int = 0
    slots: str = "table"
    m5_extras: str = "after"
    shared_early: bool = False
    route_pause: str = ""
    route_gate: str = "none"
    m4_weights: str = "route"

    def __post_init__(self) -> None:
        if self.union_order not in ("first", "id"):
            raise ValueError(f"union_order must be first|id, got {self.union_order!r}")
        if self.m4_waves not in (4, 8):
            raise ValueError("m4_waves must be 4 or 8")
        if self.m4_map not in ("mono", "balanced"):
            raise ValueError(f"m4_map must be mono|balanced, got {self.m4_map!r}")
        if self.m4_map == "mono" and self.m4_waves != 8 and self.m4_j:
            raise ValueError("m4_map=mono with 4-wave units takes m4_j=0")
        if self.m3_ept not in (2, 4, 8):
            raise ValueError("m3_ept must be 2, 4 or 8")
        if self.w_cpol not in (0, 1, 2, 3, 16, 17, 18, 19):
            raise ValueError(f"w_cpol must be a sum of sc0 1, nt 2, sc1 16, got {self.w_cpol}")
        if self.m5_issue not in ("mono", "early", "late"):
            raise ValueError(f"m5_issue must be mono|early|late, got {self.m5_issue!r}")
        mode = MX.compute_mode(self.compute)
        if mode.shape != "16x16x128" or not mode.opsel:
            raise ValueError(f"compute must be a 16x16x128 op_sel route, got {self.compute}")
        QL.cvt_impl(self.cvt)
        if self.m5_ranks not in ("lds", "pre"):
            raise ValueError(f"m5_ranks must be lds|pre, got {self.m5_ranks!r}")
        if self.m4_guard not in ("branch", "none"):
            raise ValueError(f"m4_guard must be branch|none, got {self.m4_guard!r}")
        if self.m4_interleave and self.m4_guard != "none":
            raise ValueError("m4_interleave needs m4_guard=none")
        if self.m4_apipe not in (0, 1) or (self.m4_apipe and not self.m4_interleave):
            raise ValueError("m4_apipe is 0 or 1 and needs m4_interleave")
        if self.slots not in ("table", "arith"):
            raise ValueError(f"slots must be table|arith, got {self.slots!r}")
        if self.slots == "arith" and (self.dedupe or self.m5_ranks != "pre"):
            raise ValueError("slots=arith needs dedupe off and m5_ranks=pre")
        if self.m5_extras not in ("after", "first", "m4"):
            raise ValueError(f"m5_extras must be after|first|m4, got {self.m5_extras!r}")
        if self.shared_early and self.dedupe and not (self.shared_first and self.m4_map == "mono"):
            raise ValueError("shared_early with dedupe needs shared_first and m4_map=mono")
        if self.m4_weights not in ("route", "late"):
            raise ValueError(f"m4_weights must be route|late, got {self.m4_weights!r}")
        if self.route_gate not in ("none", "pre_poll"):
            raise ValueError(f"route_gate must be none|pre_poll, got {self.route_gate!r}")
        if self.route_pause not in ("", "s_nop", "s_sleep1", "none"):
            raise ValueError(f"route_pause must be ''|s_nop|s_sleep1|none, got {self.route_pause!r}")

    def suffix(self) -> str:
        d = MoeKnobs()
        parts = []
        if self.dedupe:
            parts.append("dd")
        if self.union_order != d.union_order:
            parts.append(f"uo{self.union_order}")
        if self.m4_waves != d.m4_waves:
            parts.append(f"w{self.m4_waves}")
        if self.m4_map != d.m4_map:
            parts.append(f"map{self.m4_map}")
        if self.m4_j:
            parts.append(f"j{self.m4_j}")
        for name in ("m3_base", "m4_base", "m5_base"):
            v = getattr(self, name)
            if v >= 0:
                parts.append(f"{name[:2]}b{v}")
        if self.m3_rows:
            parts.append(f"m3r{self.m3_rows}")
        if self.m3_ept != d.m3_ept:
            parts.append(f"ept{self.m3_ept}")
        if self.w_cpol:
            parts.append(f"wcp{self.w_cpol}")
        if self.shared_first:
            parts.append("shf")
        if self.m5_issue != d.m5_issue:
            parts.append(f"i5{self.m5_issue}")
        if self.compute != d.compute:
            parts.append(self.compute)
        if self.cvt != d.cvt:
            parts.append(f"cvt{self.cvt}")
        mb = self.mailbox.suffix()
        if mb:
            parts.append(mb)
        if self.m5_ranks != d.m5_ranks:
            parts.append(f"rk{self.m5_ranks}")
        if self.a_pre:
            parts.append("apre")
        if self.m4_guard != d.m4_guard:
            parts.append(f"g{self.m4_guard}")
        if self.sched_issue:
            parts.append("sch")
        if self.m4_interleave:
            parts.append("il")
        if self.m4_apipe:
            parts.append(f"ap{self.m4_apipe}")
        if self.slots != d.slots:
            parts.append(f"sl{self.slots}")
        if self.m5_extras != d.m5_extras:
            parts.append(f"x5{self.m5_extras}")
        if self.shared_early:
            parts.append("she")
        if self.route_pause:
            parts.append(f"rp{self.route_pause}")
        if self.route_gate != d.route_gate:
            parts.append(f"rg{self.route_gate}")
        if self.m4_weights != d.m4_weights:
            parts.append(f"w4{self.m4_weights}")
        if self.stamps:
            parts.append("tl")
        return "_".join(parts)


# ============================================================ plan (host side)


@dataclass(frozen=True)
class MoePlan:
    """Static work model of a build (``rows`` request rows)."""

    rows: int
    experts: int
    knobs: MoeKnobs
    stages: tuple

    @property
    def mono(self):
        return A.mono_moe_layout(self.rows, self.experts)

    @property
    def m3_rows(self) -> int:
        r = self.knobs.m3_rows or self.mono.m3_task_rows
        if self.rows % r:
            raise ValueError(f"m3_rows {r} must divide rows {self.rows}")
        return r

    @property
    def m3_tasks(self) -> int:
        ept = self.knobs.m3_ept
        per_set = self.experts // ept if ept == 2 else self.experts // ept + 1  # ept 2: the gate in the last task
        return per_set * (self.rows // self.m3_rows)

    def base(self, stage: str) -> int:
        v = getattr(self.knobs, f"{stage}_base")
        return self.mono.base[stage] if v < 0 else v % A.GRID

    @property
    def max_slots(self) -> int:
        return self.rows * A.TOP_K + 1

    @property
    def groups(self) -> int:  # unit groups side by side in a CTA
        return A.WAVES // self.knobs.m4_waves

    @property
    def unit_cols(self) -> int:
        return self.knobs.m4_waves * 8

    @property
    def units_per_slot(self) -> int:
        return A.MOE_INTER // self.unit_cols

    @property
    def max_units(self) -> int:
        return self.max_slots * self.units_per_slot

    @property
    def j(self) -> int:
        """Unit slots per wave group (chains per wave)."""
        if self.knobs.m4_j:
            return self.knobs.m4_j
        if self.knobs.m4_map == "mono" and self.knobs.m4_waves == 8:
            return self.mono.m4_slices
        per_cta = -(-self.max_units // A.GRID)
        return -(-per_cta // self.groups)

    @property
    def m4_task_units(self) -> int:
        return self.groups * self.j

    @property
    def m4_tasks_max(self) -> int:
        if self.knobs.m4_map == "balanced":
            return A.GRID
        return -(-self.max_units // self.m4_task_units)

    def validate(self) -> None:
        if not 1 <= self.rows <= A.MAX_ROWS:
            raise ValueError(f"rows must be in 1..{A.MAX_ROWS}")
        if self.experts % 8 or not 16 <= self.experts <= A.EXPERTS:
            raise ValueError(f"experts must be a multiple of 8 in 16..{A.EXPERTS}")
        if self.stages not in (("m3", "m4", "m5"), ("m4", "m5")):
            raise ValueError(f"stages must be m3 m4 m5 or m4 m5, got {self.stages}")
        if self.m4_tasks_max > A.GRID:
            raise ValueError(
                f"{self.max_units} M4 units need {self.m4_tasks_max} CTAs at "
                f"{self.m4_task_units} units a CTA: raise m4_j"
            )
        if self.knobs.m4_map == "balanced" and self.m4_task_units * A.GRID < self.max_units:
            raise ValueError("balanced map: m4_j too small for every unit")
        if self.entries > 4 * 16:
            raise ValueError("too many M4 epilogue entries")
        if self.m3_tasks > A.GRID:
            raise ValueError(f"{self.m3_tasks} M3 tasks > {A.GRID} CTAs: raise m3_rows")

    @property
    def entries(self) -> int:
        """M4 epilogue entries per CTA: (unit, row, 32-column block)."""
        return self.m4_task_units * self.rows * (self.knobs.m4_waves // 4)

    @property
    def m5_chains(self) -> int:  # chains per wave in M5
        return -(-self.max_slots // A.WAVES)


# ============================================================ LDS arena


def _pad_rows(words: int, rows: int) -> int:
    """mono's ``layout._place`` row stride: 16-byte multiples, rows 16 words
    apart mod 64 (bank-conflict-free row-parallel reads)."""
    s = (words + 3) // 4 * 4
    if rows > 1:
        s += (16 - s) % 64
    return s


def lds_layout(plan: MoePlan):
    """(offsets, strides, words) of the build's LDS arena (int32 words)."""
    R = plan.rows
    m3r = plan.m3_rows
    table = [
        ("rw", 8 * A.RW_ROW, 1),
        ("rx", len(A.ROUTER_PARTS) * A.RX_PART, m3r),
        ("lgl", 8, m3r),
        ("ids", A.TOP_K, R),  # expert ids by rank
        ("tw", A.TOP_K, R),  # bf16 router weights by rank (f32)
        ("rg", 4, 1),  # shared gate per row (f32)
        ("res", 32, R),  # M5: residual word of lane i
        ("inj", 32, R),  # M5: injection weight of lane i (f32)
        ("first", 4 * A.TOP_K, 1),  # union: first-occurrence flags
        ("slot_e", MAX_SLOTS + 3, 1),  # slot -> expert (SHARED for the shared expert)
        ("slot_r", 4 * MAX_SLOTS, 1),  # slot * 4 + row -> rank or -1
        ("nslot", 4, 1),
        ("gx8", A.HIDDEN // 4, R),
        ("gxe", A.HIDDEN // 32, R),
        ("gred", plan.j * 8 * 16, R),
        ("dn", (A.TOP_K + 1) * A.DN_ROW, R),
        ("ys", (A.TOP_K + 1) * 16, R),
    ]
    if plan.knobs.m4_weights == "late":
        table.append(("keys", A.TOP_K, R))  # M4: the top-10 keys by rank
    off, stride, w = {}, {}, 0
    for name, words, rows in table:
        off[name] = w
        stride[name] = _pad_rows(words, rows)
        w += stride[name] * rows
    return off, stride, w


LDS_LIMIT_WORDS = 40960


# ============================================================ context


class MoeCtx:
    """What the emitters read: builder, ids, tag, LDS arena, mailbox base,
    weight addresses. ``addr`` maps :data:`abi.LAYER_ARGS` names to uniform
    i64 device addresses (``scratch``, ``r_w``, ``e_*``, ``sh_*``, ``h_out``)."""

    def __init__(self, b: IRBuilder, plan: MoePlan, *, tag: Value, layer: Value, addr: dict):
        self.b = b
        self.plan = plan
        self.k = plan.knobs
        self.R = plan.rows
        self.E = plan.experts
        self.tag = tag
        self.layer = layer
        self.addr = addr
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = b.readfirstlane(b.lshr(self.tid, b.const_i32(6)))
        self.bid = b.readfirstlane(b.block_id_x())
        self.off, self.stride, words = lds_layout(plan)
        if words > LDS_LIMIT_WORDS:
            raise ValueError(f"LDS arena {words} words > {LDS_LIMIT_WORDS}")
        self.lds = b.smem_alloc(I32, [words], name_hint="moe_lds")
        self.mb = M.mailbox_from_i64(
            b, addr["scratch"], b.const_i32(-1), plan.knobs.mailbox
        )
        self.stamp_lds = None
        self.stage_ix = None
        if plan.knobs.stamps:
            self.stamp_lds = b.smem_alloc(I32, [2 * A.STAMP_POINTS], name_hint="moe_tl")

    # ---------------------------------------------------- scalar helpers
    def c(self, v):
        if isinstance(v, Value):
            return v
        return self.b.const_i32(int(v))

    def f(self, v: float):
        return self.b.const_f32(float(v))

    def sel(self, cond, a, b_):
        return self.b.select(cond, self.c(a) if not isinstance(a, Value) else a,
                             self.c(b_) if not isinstance(b_, Value) else b_)

    def imin(self, a, b_):
        return self.b.smin(self.c(a), self.c(b_))

    def imax(self, a, b_):
        return self.b.smax(self.c(a), self.c(b_))

    def add(self, *xs):
        acc = self.c(xs[0])
        for x in xs[1:]:
            if isinstance(x, int) and x == 0:
                continue
            acc = self.b.add(acc, self.c(x))
        return acc

    def mul(self, a, b_):
        return self.b.mul(self.c(a), self.c(b_))

    def lt(self, a, b_):
        return self.b.cmp_lt(self.c(a), self.c(b_))

    def eq(self, a, b_):
        return self.b.cmp_eq(self.c(a), self.c(b_))

    def ne(self, a, b_):
        return self.b.cmp_ne(self.c(a), self.c(b_))

    def ge(self, a, b_):
        return self.b.cmp_ge(self.c(a), self.c(b_))

    def band(self, a, b_):
        return self.b.land(a, b_)

    def udiv(self, a, d: int):
        """``a // d`` for a non-negative i32 ``a`` and a positive constant."""
        if d & (d - 1) == 0:
            return self.b.lshr(self.c(a), self.c(d.bit_length() - 1))
        return self.b.div(self.c(a), self.c(d))

    def umod(self, a, d: int):
        if d & (d - 1) == 0:
            return self.b.land(self.c(a), self.c(d - 1))
        return self.b.mod(self.c(a), self.c(d))

    # ---------------------------------------------------- LDS
    def ix(self, name: str, i=0, row=0):
        """Word index of entry ``i`` of row ``row`` of LDS region ``name``."""
        base = self.off[name]
        if isinstance(row, int):
            base += row * self.stride[name]
            return self.add(base, i) if isinstance(i, Value) else self.c(base + i)
        return self.add(self.mul(row, self.stride[name]), base, i)

    def ld(self, idx, dtype=I32):
        v = self.b.smem_load_vN(self.lds, self.c(idx), dtype=dtype, n=1)
        return self.b.vec_extract(v, 0)

    def ldn(self, idx, n, dtype=I32):
        return self.b.smem_load_vN(self.lds, self.c(idx), dtype=dtype, n=n)

    def st(self, idx, v, n=1):
        self.b.smem_store_vN(self.lds, [self.c(idx)], v, n)

    def ld_u(self, idx):
        """A wave-uniform LDS word as an SGPR value."""
        return self.b.readfirstlane(self.ld(idx))

    # ---------------------------------------------------- sync, mailbox
    def barrier(self):
        self.b.s_barrier_bare(fence="workgroup")

    def pix(self, region: str, i, row=0):
        """Pair index (from the scratch base) of pair ``i`` of ``region``, row ``row``."""
        base = A.SC[region] // A.PAIR
        if isinstance(row, int):
            base += row * A.ROW_PAIRS
            return self.add(base, i) if isinstance(i, Value) else self.c(base + i)
        return self.add(self.mul(row, A.ROW_PAIRS), base, i)

    def poll(self, specs, route=False):
        """``route``: a routing poll (``route_pause``)."""
        kn = self.k.mailbox
        if route and self.k.route_pause:
            kn = replace(kn, pause=self.k.route_pause)
        out = M.poll(self.b, [(self.mb, i, n) for i, n in specs], self.tag, kn)
        self.stamp_ready()
        return out

    def put(self, i, v):
        M.put(self.b, self.mb, i, v, self.tag, self.k.mailbox)

    def put2(self, i, v0, v1):
        M.put2(self.b, self.mb, i, v0, v1, self.tag, self.k.mailbox)

    # ---------------------------------------------------- stamps
    def _stamp_at(self, point):
        b = self.b
        t = D.memrealtime(b)
        lo = b.trunc(t, I32)
        hi = b.trunc(b.lshr(t, b.const_i64(32)), I32)
        p2 = self.mul(point, 2)
        b.smem_store_vN(self.stamp_lds, [p2], lo, 1)
        b.smem_store_vN(self.stamp_lds, [self.add(p2, 1)], hi, 1)

    def stamp_begin(self, n_stages: int):
        if self.stamp_lds is None:
            return
        b = self.b
        zero = b.const_i32(0)
        with b.scf_if(self.eq(self.tid, 0)):
            for i in range(n_stages):
                for p in (1 + i * A.STAMP_PER_STAGE, 2 + i * A.STAMP_PER_STAGE):
                    b.smem_store_vN(self.stamp_lds, [self.c(2 * p)], zero, 1)
                    b.smem_store_vN(self.stamp_lds, [self.c(2 * p + 1)], zero, 1)
            self._stamp_at(0)
        with b.scf_if(self.eq(self.lane, 0)):
            for i in range(n_stages):
                p = self.add(3 + i * A.STAMP_PER_STAGE, self.wave)
                b.smem_store_vN(self.stamp_lds, [self.mul(p, 2)], zero, 1)
                b.smem_store_vN(self.stamp_lds, [self.add(self.mul(p, 2), 1)], zero, 1)

    def stamp_enter(self, i: int):
        self.stage_ix = i
        if self.stamp_lds is not None:
            with self.b.scf_if(self.eq(self.tid, 0)):
                self._stamp_at(1 + i * A.STAMP_PER_STAGE)

    def stamp_exit(self, i: int):
        self.stage_ix = None
        if self.stamp_lds is not None:
            with self.b.scf_if(self.eq(self.tid, 0)):
                self._stamp_at(2 + i * A.STAMP_PER_STAGE)

    def stamp_ready(self):
        if self.stamp_lds is None or self.stage_ix is None:
            return
        with self.b.scf_if(self.eq(self.lane, 0)):
            self._stamp_at(self.add(3 + self.stage_ix * A.STAMP_PER_STAGE, self.wave))

    def stamp_flush(self):
        if self.stamp_lds is None:
            return
        b = self.b
        self.barrier()
        with b.scf_if(self.lt(self.tid, A.STAMP_POINTS)):
            lo = b.vec_extract(b.smem_load_vN(self.stamp_lds, self.mul(self.tid, 2), dtype=I32, n=1), 0)
            hi = b.vec_extract(
                b.smem_load_vN(self.stamp_lds, self.add(self.mul(self.tid, 2), 1), dtype=I32, n=1), 0
            )
            v = b.lor(b.zext(lo, I64), b.shl(b.zext(hi, I64), b.const_i64(32)))
            base = b.add(self.addr["scratch"], b.const_i64(A.STAMP_BASE))
            ptr = b.int_to_ptr(base, PtrType(I64, "global"))
            idx = self.add(self.mul(self.add(self.mul(self.layer, A.GRID), self.bid), A.STAMP_POINTS), self.tid)
            b.global_store(ptr, idx, v, align=8)


# ============================================================ math (mono rules)


def bf16r(b: IRBuilder, x: Value) -> Value:
    """Round an f32 to bf16 (RNE) and widen it back."""
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _exp2_scaled(e: MoeCtx, x: Value, fused: bool) -> Value:
    """``exp(x)`` by the hardware exp2 of ``x log2 e``; below 2^-126, of the
    argument plus 64, scaled by 2^-64 (``mexp_fma`` / ``mexp_mul``)."""
    b = e.b
    p = b.fmul(x, e.f(L2E))
    small_arg = b.fma(x, e.f(L2E), e.f(64.0)) if fused else b.fadd(p, e.f(64.0))
    small = b.fmul(b.exp2_fast(small_arg), e.f(TWO_M64))
    return b.select(b.fcmp("olt", p, e.f(-126.0)), small, b.exp2_fast(p))


def sigmoid_m(e: MoeCtx, x: Value) -> Value:
    b = e.b
    return b.fdiv(e.f(1.0), b.fadd(e.f(1.0), _exp2_scaled(e, b.fsub(e.f(0.0), x), False)))


def sigmoid_acc(e: MoeCtx, x: Value) -> Value:
    """``1 / (1 + expf(-x))``: correctly rounded divide, accurate exp."""
    b = e.b
    return b.fdiv(e.f(1.0), b.fadd(e.f(1.0), b.exp(b.fsub(e.f(0.0), x))))


def bf16x8(e: MoeCtx, words) -> list:
    b = e.b
    out = []
    for w in words:
        out += [
            b.bitcast(b.shl(w, e.c(16)), F32),
            b.bitcast(b.land(w, e.c(-65536)), F32),
        ]
    return out


def _ugt(e: MoeCtx, a: Value, b_: Value) -> Value:
    """Unsigned ``a > b`` of i32 bit patterns."""
    m = e.c(INT_MIN)
    return e.b.cmp_gt(e.b.xor(a, m), e.b.xor(b_, m))


def top10(e: MoeCtx, words, extra: int = 0) -> list:
    """The top 10 of a row's logits as wave-uniform keys, in rank order
    (mono ``moe.top10``). ``words``: this lane's 4 logit words (logits
    ``8 l .. 8 l + 7``, low half first). A key is the logit's monotone bits
    with ``511 - idx`` in the 9 low bits: equal logits go to the lower expert.
    ``extra``: that many picks past the 10th (test mutants, probe P15)."""
    b, lane = e.b, e.lane
    ks = []
    for i in range(8):
        w = words[i // 2]
        bits = b.shl(w, e.c(16)) if i % 2 == 0 else b.land(w, e.c(-65536))
        mono = b.select(e.ge(bits, 0), b.xor(bits, e.c(INT_MIN)), b.xor(bits, e.c(-1)))
        key = b.lor(b.land(mono, e.c(-512)), b.sub(e.c(511), e.add(e.mul(lane, 8), i)))
        ks.append(b.select(e.lt(lane, e.E // 8), key, e.c(0)))
    for x, y in SORT8:
        first = _ugt(e, ks[x], ks[y])
        kx, ky = ks[x], ks[y]
        ks[x], ks[y] = b.select(first, kx, ky), b.select(first, ky, kx)
    ks = ks + [e.c(0)]
    picks = []
    for _ in range(A.TOP_K + extra):
        m = D.wave_umax_dpp(b, ks[0])
        hit = b.cmp_eq(ks[0], m)
        ks = [b.select(hit, ks[i + 1], ks[i]) for i in range(8)] + [ks[8]]
        picks.append(m)
    return picks


def key_expert(e: MoeCtx, m: Value) -> Value:
    return e.b.sub(e.c(511), e.b.land(m, e.c(0x1FF)))


def key_logit(e: MoeCtx, m: Value) -> Value:
    b = e.b
    neg = b.xor(b.land(m, e.c(-512)), e.c(INT_MIN))
    pos = b.xor(b.lor(m, e.c(0x1FF)), e.c(-1))
    return b.bitcast(b.select(e.lt(m, 0), neg, pos), F32)


def router_weights(e: MoeCtx, picks) -> list:
    """bf16 router weights of the 10 picks, rank order (mono ``router_weights``)."""
    b = e.b
    v = [key_logit(e, m) for m in picks]
    ex = [_exp2_scaled(e, b.fsub(v[k], v[0]), True) for k in range(A.TOP_K)]
    s = e.f(0.0)
    for k in range(A.TOP_K):
        s = b.fadd(s, ex[k])
    return [bf16r(b, b.fdiv(ex[k], s)) for k in range(A.TOP_K)]


# ============================================================ weights


def _rsrc(e: MoeCtx, addr: Value, nbytes: int) -> Value:
    b = e.b
    ptr = b.int_to_ptr(addr, PtrType(I32, "global"))
    return b.buffer_rsrc(ptr, e.c(nbytes))


def _wload(e: MoeCtx, rs: Value, byte_off: Value, dwords: int) -> Value:
    """A weight load (1 or 4 dwords) with the ``w_cpol`` cache-policy bits."""
    cp = e.k.w_cpol
    if cp:
        return e.b.buffer_load_cp(rs, byte_off, e.c(0), dwords=dwords, aux=cp)
    if dwords == 4:
        return e.b.buffer_load_vN(rs, byte_off, e.c(0), I32, 4)
    return e.b.buffer_load(rs, byte_off, e.c(0), I32)


def _load4(e: MoeCtx, rs: Value, byte_off: Value) -> Value:
    return _wload(e, rs, byte_off, 4)


def _load1(e: MoeCtx, rs: Value, byte_off: Value) -> Value:
    return _wload(e, rs, byte_off, 1)


def _a_operand(e: MoeCtx, base: Value, kit: int) -> Value:
    """This lane's fp8 A operand of k_iter ``kit`` from LDS words at ``base``:
    two 16-byte halves 64 bytes apart (``common.a_operand``)."""
    b = e.b
    off = e.add(base, kit * 32, e.mul(e.udiv(e.lane, 16), 4))
    a0 = e.ldn(off, 4)
    a1 = e.ldn(e.add(off, 16), 4)
    return b.vec_pack([b.vec_extract(a0, i) for i in range(4)] + [b.vec_extract(a1, i) for i in range(4)], I32)


def _mma(e: MoeCtx, acc, a, w, ea, sw, opsel_b):
    return MX.mx_mma(
        e.b, e.k.compute, acc, a, w, ea, sw, fmt_a="fp8", fmt_b="fp4", opsel_a=0, opsel_b=opsel_b
    )


# ============================================================ routing + slots


def _route_gate(e: MoeCtx) -> None:
    """``route_gate=pre_poll``: wave 0 spins on the second logits pair of each
    (M3 router task, request row), then the CTA barriers. Every thread calls it."""
    b = e.b
    per_row = e.E // 8

    def hint(m):
        t, r = (m, 0) if e.R == 1 else (e.umod(m, per_row), e.udiv(m, per_row))
        return e.mb, e.pix("lg", e.add(e.mul(t, 4), 2), r)

    kn = e.k.mailbox
    if e.k.route_pause:
        kn = replace(kn, pause=e.k.route_pause)
    M.pre_poll(b, hint, per_row * e.R, e.tag, kn)


def _route_rows(e: MoeCtx, weights: bool, gate: bool = False, keys: bool = False, k11: bool = False) -> None:
    """Waves ``r < R``: row r's top 10 into LDS ``ids`` (and, ``weights``, the
    bf16 router weights into ``tw``). Every lane runs it; no barrier (``gate``:
    :func:`_route_gate` first, which barriers)."""
    b = e.b
    if gate:
        _route_gate(e)
    with b.scf_if(e.lt(e.wave, e.R)):
        r = e.wave
        j = e.imin(e.lane, e.E // 8 - 1)
        got = e.poll([(e.pix("lg", e.mul(j, 4), r), 2), (e.pix("lg", e.add(e.mul(j, 4), 2), r), 2)], route=True)
        picks = top10(e, got[0] + got[1], extra=1 if k11 else 0)
        if k11:  # test mutant (probe P15): the 11th pick in place of the 10th
            picks = picks[: A.TOP_K - 1] + [picks[A.TOP_K]]
        ws = router_weights(e, picks) if weights else None
        with b.scf_if(e.eq(e.lane, 0)):
            for k in range(A.TOP_K):
                e.st(e.ix("ids", k, r), key_expert(e, picks[k]))
                if ws is not None:
                    e.st(e.ix("tw", k, r), ws[k])
                if keys:
                    e.st(e.ix("keys", k, r), picks[k])


def _late_weights(e: MoeCtx) -> None:
    """``m4_weights=late``: waves ``r < R``: row r's router weights from the
    top-10 keys :func:`_route_rows` left in LDS (bitwise the same values)."""
    b = e.b
    with b.scf_if(e.lt(e.wave, e.R)):
        r = e.wave
        picks = [e.ld_u(e.ix("keys", k, r)) for k in range(A.TOP_K)]
        ws = router_weights(e, picks)
        with b.scf_if(e.eq(e.lane, 0)):
            for k in range(A.TOP_K):
                e.st(e.ix("tw", k, r), ws[k])


def _build_slots(e: MoeCtx) -> None:
    """After ``ids`` are in LDS (and a barrier): the slot tables ``slot_e``,
    ``slot_r`` and ``nslot`` (wave 0), then a barrier."""
    b = e.b
    R, K = e.R, A.TOP_K
    n = R * K
    with b.scf_if(e.eq(e.wave, 0)):
        lane = e.lane
        # every (slot, row) rank starts at -1; the shared slot's rank is 10
        for q in range(-(-4 * MAX_SLOTS // 64)):
            i = e.add(lane, 64 * q)
            with b.scf_if(e.lt(i, 4 * MAX_SLOTS)):
                e.st(e.add(e.off["slot_r"], i), e.c(-1))
        f0 = 1 if e.k.shared_first else 0  # first routed slot
        if not e.k.dedupe:
            with b.scf_if(e.lt(lane, n)):
                row, rank = e.udiv(lane, K), e.umod(lane, K)
                e.st(e.add(e.off["slot_e"], lane, f0), e.ld(e.ix("ids", rank, row)))
                e.st(e.add(e.off["slot_r"], e.mul(e.add(lane, f0), 4), row), rank)
            with b.scf_if(e.eq(lane, 0)):
                sh = 0 if f0 else n
                e.st(e.off["slot_e"] + sh, e.c(SHARED))
                for r in range(R):
                    e.st(e.off["slot_r"] + 4 * sh + r, e.c(K))
                e.st(e.off["nslot"], e.c(n + 1))
        else:
            t = e.imin(lane, n - 1)
            row, rank = e.udiv(t, K), e.umod(t, K)
            my = e.ld(e.ix("ids", rank, row))
            valid = e.lt(lane, n)
            if e.k.union_order == "id":
                slot, u = _union_by_id(e, my, valid)
            else:
                slot, u = _union_first(e, my, valid, n)
            with b.scf_if(valid):
                e.st(e.add(e.off["slot_e"], slot, f0), my)
                e.st(e.add(e.off["slot_r"], e.mul(e.add(slot, f0), 4), row), rank)
            with b.scf_if(e.eq(lane, 0)):
                sh = e.c(0) if f0 else u
                e.st(e.add(e.off["slot_e"], sh), e.c(SHARED))
                for r in range(R):
                    e.st(e.add(e.off["slot_r"], e.mul(sh, 4), r), e.c(K))
                e.st(e.off["nslot"], e.add(u, 1))
    e.barrier()


def _shared_slot(e: MoeCtx) -> int:
    """The shared expert's slot when not deduped."""
    return 0 if e.k.shared_first else e.R * A.TOP_K


def _slot_row_rank(e: MoeCtx, s: Value):
    """Not deduped: (row, rank) of routed slot ``s`` (clamped into range)."""
    b = e.b
    kk = e.imax(b.sub(s, e.c(1)) if e.k.shared_first else s, 0)
    return e.imin(b.div(kk, e.c(A.TOP_K)), e.R - 1), b.mod(kk, e.c(A.TOP_K))


def _n_slots(e: MoeCtx) -> Value:
    if e.k.slots == "arith":
        return e.c(e.R * A.TOP_K + 1)
    return e.ld_u(e.off["nslot"])


def _slot_entry(e: MoeCtx, s: Value) -> Value:
    """Slot ``s``'s expert id (``SHARED`` for the shared expert), wave-uniform ``s``."""
    if e.k.slots == "table":
        return e.ld_u(e.add(e.off["slot_e"], s))
    row, rank = _slot_row_rank(e, s)
    ent = e.ld_u(e.ix("ids", rank, row))
    return e.sel(e.eq(s, _shared_slot(e)), SHARED, ent)


def _slot_is_shared(e: MoeCtx, s: Value) -> Value:
    """Per-lane ``s``."""
    if e.k.slots == "table":
        return e.eq(e.ld(e.add(e.off["slot_e"], s)), SHARED)
    return e.eq(s, _shared_slot(e))


def _slot_rank(e: MoeCtx, s: Value, r: Value) -> Value:
    """Rank row ``r`` has in slot ``s`` (-1: none; the shared slot: 10), per lane."""
    if e.k.slots == "table":
        return e.ld(e.add(e.off["slot_r"], e.mul(s, 4), r))
    row, rank = _slot_row_rank(e, s)
    routed = e.sel(e.eq(row, r), rank, -1)
    return e.sel(e.eq(s, _shared_slot(e)), A.TOP_K, routed)


def _bcnt(b: IRBuilder, v: Value) -> Value:
    return b.inline_asm(
        "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [v], I32, sideeffect=False, result_name_hint="bcnt"
    )


def _mbcnt(b: IRBuilder, mask: Value) -> Value:
    """Set bits of the i64 ``mask`` below this lane."""
    lo = b.trunc(mask, I32)
    hi = b.trunc(b.lshr(mask, b.const_i64(32)), I32)
    return b.inline_asm(
        "v_mbcnt_lo_u32_b32 $0, $1, 0\n\tv_mbcnt_hi_u32_b32 $0, $2, $0",
        "=&v,s,s",
        [lo, hi],
        I32,
        sideeffect=False,
        result_name_hint="mbcnt",
    )


def _union_by_id(e: MoeCtx, my: Value, valid: Value):
    """Wave 0, lanes ``t < 10 R`` holding expert ``my``: (slot of ``my`` in
    the id-sorted union, union size). An LDS bitmap of the routed experts
    (``ds_or_b32``), per-word popcounts and their prefix over the words."""
    b = e.b
    lane = e.lane
    nw = -(-e.E // 32)
    bm = e.off["first"]  # bitmap words, then their exclusive prefix
    assert 2 * nw <= 4 * A.TOP_K
    with b.scf_if(e.lt(lane, nw)):
        e.st(e.add(bm, lane), e.c(0))
    lds0 = b.trunc(b.smem_addr_of(e.lds), I32)
    with b.scf_if(valid):
        addr = e.add(lds0, e.mul(e.add(bm, b.lshr(my, e.c(5))), 4))
        bit = b.shl(e.c(1), b.land(my, e.c(31)))
        b.inline_asm("ds_or_b32 $0, $1\n\ts_waitcnt lgkmcnt(0)", "v,v,~{memory}", [addr, bit], None)
    word = e.ld(e.add(bm, e.imin(lane, nw - 1)))
    cnt = b.select(e.lt(lane, nw), _bcnt(b, word), e.c(0))
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
    slot = e.add(e.ld(e.add(bm, nw, w)), _bcnt(b, below))
    u = b.readlane(inc, e.c(nw - 1))
    return slot, u


def _union_first(e: MoeCtx, my: Value, valid: Value, n: int):
    """(slot of ``my`` in first-occurrence order over (row, rank), union size)."""
    b = e.b
    f = e.c(n)
    for j in range(n):
        idj = b.readlane(my, e.c(j))
        hit = b.land(b.cmp_eq(idj, my), b.cmp_eq(f, e.c(n)))
        f = b.select(hit, e.c(j), f)
    first = b.land(b.cmp_eq(f, e.lane), valid)
    s_lane = _mbcnt(b, b.wave_ballot(b.select(first, e.c(1), e.c(0))))  # ballot takes an i32
    slot = b.ds_bpermute(e.mul(e.imin(f, 63), 4), s_lane)
    u = b.readlane(s_lane, e.c(n))
    return slot, u


# ============================================================ M3


def _part(e: MoeCtx, k: Value):
    """K part of the router's split holding ``k`` and that part's first k."""
    b = e.b
    big = e.lt(k, A.ROUTER_PARTS[8][0])
    p = b.select(big, e.udiv(k, 192), e.udiv(b.sub(k, e.c(512)), 128))
    first = b.select(big, e.mul(p, 192), e.add(e.mul(p, 128), 512))
    return p, first


def _ksum_order(e: MoeCtx) -> list:
    """K-part order of a router logit's lane sum: 0 .. 15 (the mono's). Test mutants of the fused MoE
    (``FusedKnobs.mutate``, probe P15): ``router_ksum`` reversed (a 1-ulp f32 change that the bf16 rounding of the
    logit usually hides), ``router_kdrop`` part 15 left out (the ``lg`` payload must differ)."""
    fk = getattr(e, "fk", None)
    if fk is not None and fk.mutate == "router_ksum":
        return list(range(15, -1, -1))
    if fk is not None and fk.mutate == "router_kdrop":
        return list(range(15))
    return list(range(16))


def _emit_m3_n(e: MoeCtx) -> None:
    """M3 at ``m3_ept=2`` (probe P15, the N3b router): the 513 router rows split over ``experts / 2`` tasks (every
    CTA at 512 experts). Task ``t`` computes router rows ``2 t, 2 t + 1`` and, on the last task, row ``experts``
    (the shared gate) for ``m3_rows`` request rows. Wave ``w``: router row ``w % 4`` of the task (past its 2 or 3
    rows: idle), request rows ``(w // 4) rpw ..`` on lanes ``16 i + q`` (part q). The weight loads, the mixed-row
    poll, the LDS staging, each logit's 16 K-part fma chains and their lane-ordered sum are :func:`emit_m3`'s, so
    the logits and the shared gate are the mono's bit for bit."""
    b = e.b
    plan = e.plan
    rows3 = plan.m3_rows
    NT = e.E // 2
    split = rows3 < e.R
    rpw = -(-rows3 // 2)  # request rows per wave (two wave groups)
    loads = -(-3 * ROW_CHUNKS // A.THREADS)
    task = e.umod(e.add(e.bid, A.GRID - plan.base("m3")), A.GRID)
    with b.scf_if(e.lt(task, plan.m3_tasks)):
        if split:
            t = e.umod(task, NT)
            h0 = e.mul(e.udiv(task, NT), rows3)
        else:
            t, h0 = task, e.c(0)
        last = e.eq(t, NT - 1)
        nch = e.sel(last, 3 * ROW_CHUNKS, 2 * ROW_CHUNKS)
        rs = _rsrc(e, e.addr["r_w"], -1)
        ws = []
        for m in range(loads):
            j = e.imin(e.add(e.tid, m * A.THREADS), b.sub(nch, e.c(1)))
            g = e.add(e.mul(t, 2), e.udiv(j, ROW_CHUNKS))  # the last task's third row: the gate (row experts)
            off = e.add(e.mul(g, A.ROW_WORDS), e.mul(e.umod(j, ROW_CHUNKS), 4))
            ws.append(_wload(e, rs, e.mul(off, 4), 4))
        e.barrier()  # payload_wait: rows <= 4 only barrier (as emit_m3)
        hi = e.imin(e.add(e.tid, A.THREADS), RX_LOADS - 1)
        specs = [(e.pix("mxbf", e.mul(j, 2), e.add(h0, r)), 2) for r in range(rows3) for j in (e.tid, hi)]
        v = e.poll(specs)
        for r in range(rows3):
            for q, j in enumerate((e.tid, hi)):
                words = v[2 * r + q]
                x0, x1 = M.bf2_f32(b, words[0])
                x2, x3 = M.bf2_f32(b, words[1])
                part, first = _part(e, e.mul(j, 4))
                idx = e.ix("rx", b.sub(e.add(e.mul(part, A.RX_PART), e.mul(j, 4)), first), r)
                e.st(idx, b.vec_pack([x0, x1, x2, x3], F32), 4)
        for m in range(loads):
            j = e.add(e.tid, m * A.THREADS)
            jc = e.umod(j, ROW_CHUNKS)
            part, first = _part(e, e.mul(jc, 8))
            off = e.add(
                e.mul(e.udiv(j, ROW_CHUNKS), A.RW_ROW),
                e.mul(part, A.RW_PART),
                b.lshr(b.sub(e.mul(jc, 8), first), e.c(1)),
            )
            if (m + 1) * A.THREADS <= 2 * ROW_CHUNKS:
                e.st(e.ix("rw", off), ws[m], 4)
            else:
                with b.scf_if(e.lt(j, nch)):
                    e.st(e.ix("rw", off), ws[m], 4)
        e.barrier()
        wrow_ix = e.umod(e.wave, 4)
        rblk = e.udiv(e.wave, 4)
        nrow = e.sel(last, 3, 2)
        # lane 16 i + q: part q of weight row ``wrow_ix`` for request row rblk * rpw + i
        with b.scf_if(b.land(e.lt(e.lane, 16 * rpw), e.lt(wrow_ix, nrow))):
            lp = e.umod(e.lane, 16)
            big = e.lt(lp, 8)
            rr = e.add(e.mul(rblk, rpw), e.udiv(e.lane, 16))
            rr_c = e.imin(rr, rows3 - 1)
            wrow = e.ix("rw", e.add(e.mul(wrow_ix, A.RW_ROW), e.mul(lp, A.RW_PART)))
            xrow = e.ix("rx", e.mul(lp, A.RX_PART), rr_c)
            acc = e.f(0.0)
            for st_ in range(CHAIN_STEPS):
                wv = e.ldn(e.add(wrow, st_ * 4), 4)
                w8 = bf16x8(e, [b.vec_extract(wv, i) for i in range(4)])
                xa = e.ldn(e.add(xrow, st_ * 8), 4, F32)
                xb = e.ldn(e.add(xrow, st_ * 8 + 4), 4, F32)
                x8 = [b.vec_extract(xa, i) for i in range(4)] + [b.vec_extract(xb, i) for i in range(4)]
                new = acc
                for i in range(8):
                    new = b.fma(w8[i], x8[i], new)
                acc = new if st_ < SHORT_STEPS else b.select(big, new, acc)
            bits = b.bitcast(acc, I32)
            qs = _ksum_order(e)
            for i in range(rpw):
                total = b.bitcast(b.readlane(bits, e.c(16 * i + qs[0])), F32)
                for q in qs[1:]:
                    total = b.fadd(total, b.bitcast(b.readlane(bits, e.c(16 * i + q)), F32))
                row_i = e.add(e.mul(rblk, rpw), i)
                with b.scf_if(b.land(e.eq(e.lane, 0), e.lt(row_i, rows3))):
                    e.st(e.ix("lgl", wrow_ix, row_i), bf16r(b, total))
        e.barrier()
        # the task's 2 logits of a row = 1 pair (threads 0 ..); the gate (last task, wave 1: threads 64 ..)
        with b.scf_if(e.lt(e.tid, rows3)):
            lg = e.ldn(e.ix("lgl", 0, e.tid), 2, F32)
            e.put(e.pix("lg", t, e.add(h0, e.tid)), M.bf16_pair(b, b.vec_extract(lg, 0), b.vec_extract(lg, 1)))
        rg_ = b.sub(e.tid, e.c(64))
        with b.scf_if(b.land(last, b.land(e.ge(e.tid, 64), e.lt(e.tid, 64 + rows3)))):
            gl = e.ld(e.ix("lgl", 2, rg_), F32)
            e.put(e.pix("sg", 0, e.add(h0, rg_)), bf16r(b, sigmoid_acc(e, gl)))


def emit_m3(e: MoeCtx) -> None:
    """M3 (mono ``moe.m3``): task ``t`` of a row set computes router logits
    ``ept t .. ept t + ept - 1`` (the last task: the shared expert's gate) of
    ``m3_rows`` request rows. Wave ``w``: weight row ``w % ept``, request rows
    ``(w // ept) rpw ..`` on lanes ``16 i + q`` (part q); each logit's 16 K-part
    fma chains and their lane-ordered sum are mono's, whatever the grouping."""
    if e.plan.knobs.m3_ept == 2:
        _emit_m3_n(e)
        return
    b = e.b
    plan = e.plan
    rows3 = plan.m3_rows
    ept = plan.knobs.m3_ept
    NT = e.E // ept + 1
    split = rows3 < e.R
    groups = A.WAVES // ept  # wave groups (row blocks) of a task
    rpw = -(-rows3 // groups)  # request rows per wave
    chunks = ept * ROW_CHUNKS
    loads = -(-chunks // A.THREADS)
    task = e.umod(e.add(e.bid, A.GRID - plan.base("m3")), A.GRID)
    with b.scf_if(e.lt(task, plan.m3_tasks)):
        if split:
            t = e.umod(task, NT)
            h0 = e.mul(e.udiv(task, NT), rows3)
        else:
            t, h0 = task, e.c(0)
        rs = _rsrc(e, e.addr["r_w"], -1)
        ws = []
        for m in range(loads):
            j = e.imin(e.add(e.tid, m * A.THREADS), chunks - 1)
            g = e.imin(e.add(e.mul(t, ept), e.udiv(j, ROW_CHUNKS)), e.E)
            off = e.add(e.mul(g, A.ROW_WORDS), e.mul(e.umod(j, ROW_CHUNKS), 4))
            ws.append(_wload(e, rs, e.mul(off, 4), 4))
        e.barrier()  # payload_wait: rows <= 4 only barrier
        hi = e.imin(e.add(e.tid, A.THREADS), RX_LOADS - 1)
        specs = [(e.pix("mxbf", e.mul(j, 2), e.add(h0, r)), 2) for r in range(rows3) for j in (e.tid, hi)]
        v = e.poll(specs)
        for r in range(rows3):
            for q, j in enumerate((e.tid, hi)):
                words = v[2 * r + q]
                x0, x1 = M.bf2_f32(b, words[0])
                x2, x3 = M.bf2_f32(b, words[1])
                part, first = _part(e, e.mul(j, 4))
                idx = e.ix("rx", b.sub(e.add(e.mul(part, A.RX_PART), e.mul(j, 4)), first), r)
                e.st(idx, b.vec_pack([x0, x1, x2, x3], F32), 4)
        for m in range(loads):
            j = e.add(e.tid, m * A.THREADS)
            jc = e.umod(j, ROW_CHUNKS)
            part, first = _part(e, e.mul(jc, 8))
            off = e.add(
                e.mul(e.udiv(j, ROW_CHUNKS), A.RW_ROW),
                e.mul(part, A.RW_PART),
                b.lshr(b.sub(e.mul(jc, 8), first), e.c(1)),
            )
            if (m + 1) * A.THREADS <= chunks:
                e.st(e.ix("rw", off), ws[m], 4)
            else:
                with b.scf_if(e.lt(j, chunks)):
                    e.st(e.ix("rw", off), ws[m], 4)
        e.barrier()
        wrow_ix = e.umod(e.wave, ept) if ept < A.WAVES else e.wave
        rblk = e.udiv(e.wave, ept) if ept < A.WAVES else e.c(0)
        # lane 16 i + q: part q of weight row ``wrow_ix`` for request row rblk * rpw + i
        with b.scf_if(e.lt(e.lane, 16 * rpw)):
            lp = e.umod(e.lane, 16)
            big = e.lt(lp, 8)
            rr = e.add(e.mul(rblk, rpw), e.udiv(e.lane, 16))
            rr_c = e.imin(rr, rows3 - 1)
            wrow = e.ix("rw", e.add(e.mul(wrow_ix, A.RW_ROW), e.mul(lp, A.RW_PART)))
            xrow = e.ix("rx", e.mul(lp, A.RX_PART), rr_c)
            acc = e.f(0.0)
            for st_ in range(CHAIN_STEPS):
                wv = e.ldn(e.add(wrow, st_ * 4), 4)
                w8 = bf16x8(e, [b.vec_extract(wv, i) for i in range(4)])
                xa = e.ldn(e.add(xrow, st_ * 8), 4, F32)
                xb = e.ldn(e.add(xrow, st_ * 8 + 4), 4, F32)
                x8 = [b.vec_extract(xa, i) for i in range(4)] + [b.vec_extract(xb, i) for i in range(4)]
                new = acc
                for i in range(8):
                    new = b.fma(w8[i], x8[i], new)
                acc = new if st_ < SHORT_STEPS else b.select(big, new, acc)
            bits = b.bitcast(acc, I32)
            qs = _ksum_order(e)
            for i in range(rpw):
                total = b.bitcast(b.readlane(bits, e.c(16 * i + qs[0])), F32)
                for q in qs[1:]:
                    total = b.fadd(total, b.bitcast(b.readlane(bits, e.c(16 * i + q)), F32))
                row_i = e.add(e.mul(rblk, rpw), i)
                with b.scf_if(b.land(e.eq(e.lane, 0), e.lt(row_i, rows3))):
                    e.st(e.ix("lgl", wrow_ix, row_i), bf16r(b, total))
        e.barrier()
        # ept logits of a row = ept / 2 pairs: thread (row, pair half) puts 2 pairs
        per_row = ept // 4  # put2 calls per row
        if per_row == 1 and rows3 == 1:
            li, lr = e.c(0), 0
        elif per_row == 1:
            li, lr = e.c(0), e.tid
        elif rows3 == 1:
            li, lr = e.tid, 0
        else:
            li, lr = e.umod(e.tid, per_row), e.udiv(e.tid, per_row)
        with b.scf_if(b.land(e.lt(t, e.E // ept), e.lt(e.tid, per_row * rows3))):
            lg = e.ldn(e.ix("lgl", e.mul(li, 4), lr), 4, F32)
            e.put2(
                e.pix("lg", e.add(e.mul(t, ept // 2), e.mul(li, 2)), e.add(h0, lr)),
                M.bf16_pair(b, b.vec_extract(lg, 0), b.vec_extract(lg, 1)),
                M.bf16_pair(b, b.vec_extract(lg, 2), b.vec_extract(lg, 3)),
            )
        with b.scf_if(b.land(e.eq(t, e.E // ept), e.lt(e.tid, rows3))):
            gl = e.ld(e.ix("lgl", 0, e.tid), F32)
            e.put(e.pix("sg", 0, e.add(h0, e.tid)), bf16r(b, sigmoid_acc(e, gl)))


# ============================================================ M4


class _Unit:
    """M4 unit ``k`` of this CTA (wave group ``k % G``, chain slot ``k // G``)."""

    def __init__(self, e: MoeCtx, task: Value, n_units: Value, n_slots: Value, k: int):
        b, plan = e.b, e.plan
        G, J = plan.groups, plan.j
        if plan.knobs.m4_map == "mono":
            u = e.add(e.mul(task, G * J), k)
            valid = e.lt(u, n_units)
        else:
            start = b.lshr(e.mul(task, n_units), e.c(8))
            end = b.lshr(e.mul(e.add(task, 1), n_units), e.c(8))
            # group g takes units start + g, start + g + G, ...
            u = e.add(start, k)
            valid = e.lt(u, end)
        u = e.imin(u, b.sub(n_units, e.c(1)))
        ups = plan.units_per_slot
        self.slot = e.udiv(u, ups)
        self.cb = e.umod(u, ups)
        self.valid = valid
        self.j, self.g = k // G, k % G


def _gu_issue(e: MoeCtx, rs_w, rs_s, pair, part, grp, off):
    """Weight (4 tiles) and scale (2 dwords) loads of k_iter group ``grp`` of
    block ``part`` of gate/up block pair ``pair``."""
    blk = e.add(e.mul(pair, 2), part)
    ws = [
        _load4(e, rs_w, e.add(e.mul(e.add(e.mul(e.add(e.mul(blk, A.GU_KI), grp * 4 + q), 64), e.lane), 16), off))
        for q in range(4)
    ]
    ss = [
        _load1(e, rs_s, e.add(e.mul(e.add(e.mul(e.add(e.mul(pair, A.GU_K1), grp * 2 + h), 64), e.lane), 4), off))
        for h in range(2)
    ]
    return ws, ss


def _fill_x(e: MoeCtx, r) -> None:
    """Row ``r``'s MXFP8 mixed row into LDS ``gx8`` / ``gxe`` (this wave)."""
    b, lane = e.b, e.lane
    nexp = A.HIDDEN // 32
    xw = e.poll([(e.pix("mxe", lane, r), 1), (e.pix("mxe", e.imin(e.add(lane, 64), nexp - 1), r), 1)])
    e.st(e.ix("gxe", lane, r), xw[0][0])
    with b.scf_if(e.lt(lane, nexp - 64)):
        e.st(e.ix("gxe", e.add(lane, 64), r), xw[1][0])
    fw = e.poll([(e.pix("mx8", e.mul(e.add(lane, 64 * i), 2), r), 2) for i in range(X8_LOADS)])
    for i in range(X8_LOADS):
        e.st(e.ix("gx8", e.mul(e.add(lane, 64 * i), 2), r), b.vec_pack(fw[i], I32), 2)


def m4_task(e: MoeCtx) -> Value:
    return e.umod(e.add(e.bid, A.GRID - e.plan.base("m4")), A.GRID)


def emit_m4(e: MoeCtx) -> None:
    b, plan = e.b, e.plan
    task = m4_task(e)
    with b.scf_if(e.lt(task, plan.m4_tasks_max)):
        if plan.knobs.shared_early:
            early = _m4_shared_only(e, task)
            with b.scf_if(early):
                _m4_body(e, task, True)
            with b.scf_if(b.lnot(early)):
                _m4_body(e, task, False)
        else:
            _m4_body(e, task, False)


def _m4_units(e: MoeCtx, task, n_units):
    """(first unit, end unit) of M4 task ``task``."""
    b, plan = e.b, e.plan
    if plan.knobs.m4_map == "mono":
        first = e.mul(task, plan.groups * plan.j)
        return first, e.imin(e.add(first, plan.groups * plan.j), n_units)
    return b.lshr(e.mul(task, n_units), e.c(8)), b.lshr(e.mul(e.add(task, 1), n_units), e.c(8))


def _n_slots_early(e: MoeCtx) -> Value:
    """Slot count an early (shared-only) M4 task can assume before routing:
    exact when not deduped; with dedupe (``shared_first``, mono map) only the
    shared slot 0 is known, and the mono map's units do not depend on it."""
    return e.c(e.R * A.TOP_K + 1) if not e.k.dedupe else e.c(1)


def _m4_shared_only(e: MoeCtx, task) -> Value:
    """``shared_early``: i1, every unit of this M4 task is the shared expert's."""
    b, plan = e.b, e.plan
    ups = plan.units_per_slot
    sh = 0 if e.k.shared_first else e.R * A.TOP_K
    if e.k.dedupe:
        n_units = e.c(10**6)  # mono map: the task's units do not depend on the count
    else:
        n_units = e.c((e.R * A.TOP_K + 1) * ups)
    first, end = _m4_units(e, task, n_units)
    return b.land(b.land(e.ge(first, sh * ups), e.b.cmp_le(end, e.c((sh + 1) * ups))), e.lt(first, end))


def _m4_body(e: MoeCtx, task, early: bool) -> None:
    """M4 of task ``task``; ``early``: the task's units are all the shared
    expert's (``shared_early``): no routing, no slot tables."""
    b, plan = e.b, e.plan
    R, W = e.R, plan.knobs.m4_waves
    G, J = plan.groups, plan.j
    e.barrier()  # the previous stage's LDS reads are done
    gated = plan.knobs.route_gate == "pre_poll" and not early
    if gated:  # the fill first: the gate's barrier holds every wave until the logits are out
        for r in range(R):
            with b.scf_if(e.eq(e.wave, A.WAVES - A.MAX_ROWS + r)):
                _fill_x(e, r)
    if not early:
        late = plan.knobs.m4_weights == "late"
        _route_rows(e, weights=not late, gate=gated, keys=late)
    if plan.knobs.m5_extras == "m4" and not early:  # this CTA's M5 task's extras, after the routing
        t5 = m5_task(e)
        with b.scf_if(e.lt(t5, A.HIDDEN // 16)):
            _m5_extras(e, t5)
    # waves 4 .. 4 + R - 1 fill the activation rows meanwhile
    for r in range(R if not gated else 0):
        with b.scf_if(e.eq(e.wave, A.WAVES - A.MAX_ROWS + r)):
            _fill_x(e, r)
    e.barrier()
    if plan.knobs.slots == "table" and not early:
        _build_slots(e)
    n_slots = _n_slots_early(e) if early else _n_slots(e)
    n_units = e.mul(n_slots, plan.units_per_slot)
    g_w = e.udiv(e.wave, W) if G > 1 else e.c(0)
    wg = e.umod(e.wave, W) if G > 1 else e.wave
    half = W // 2
    pair_in = e.umod(wg, half)
    part = e.udiv(wg, half)
    # this wave's chains: unit slot j of group g_w
    units = []
    for j in range(J):
        per_g = [_Unit(e, task, n_units, n_slots, j * G + g) for g in range(G)]
        if G == 1:
            units.append(per_g[0])
        else:
            units.append(per_g)
    loads = []
    for j in range(J):
        if G == 1:
            un = units[j]
            slot, cb, valid = un.slot, un.cb, un.valid
        else:
            us = units[j]
            pick = lambda attr: b.select(e.eq(g_w, 0), getattr(us[0], attr), getattr(us[1], attr))  # noqa: E731
            slot, cb, valid = pick("slot"), pick("cb"), pick("valid")
        ent = e.c(SHARED) if early else _slot_entry(e, slot)
        shared = e.eq(ent, SHARED)
        expert = e.imax(ent, 0)
        ex64 = b.zext(expert, I64)
        aw = b.select(shared, e.addr["sh_w_gu"], b.add(e.addr["e_w_gu"], b.mul(ex64, b.const_i64(A.GU_W_BYTES))))
        asc = b.select(shared, e.addr["sh_s_gu"], b.add(e.addr["e_s_gu"], b.mul(ex64, b.const_i64(A.GU_S_BYTES))))
        rs_w = _rsrc(e, b.readfirstlane(aw), A.GU_W_BYTES)
        rs_s = _rsrc(e, b.readfirstlane(asc), A.GU_S_BYTES)
        pair = e.add(e.mul(cb, half), pair_in)
        off = b.select(valid, e.c(0), e.c(OOB))
        if plan.knobs.m4_interleave:  # issued below, k_iter group major
            loads.append(((rs_w, rs_s, pair, off), valid))
        else:
            loads.append(([_gu_issue(e, rs_w, rs_s, pair, part, grp, off) for grp in range(A.GU_KI // 4)], valid))
    if plan.knobs.m4_interleave:
        issued = [[None] * (A.GU_KI // 4) for _ in range(J)]
        for grp in range(A.GU_KI // 4):
            for j in range(J):
                rs_w, rs_s, pair, off = loads[j][0]
                issued[j][grp] = _gu_issue(e, rs_w, rs_s, pair, part, grp, off)
        loads = [(issued[j], loads[j][1]) for j in range(J)]
    if plan.knobs.sched_issue:
        b.sched_barrier(0)
    if plan.knobs.m4_weights == "late" and not early:
        _late_weights(e)
    # MFMA A rows: lane l carries request row min(l % 16, R - 1)
    r_l = e.imin(e.umod(e.lane, 16), R - 1)
    x8 = e.ix("gx8", 0, r_l)
    xe = e.ix("gxe", 0, r_l)
    if plan.knobs.m4_interleave:
        accs = [b.zero_vec_f32(4) for _ in range(J)]

        def a_ops(kit):
            return e.ld(e.add(xe, kit * 4, e.udiv(e.lane, 16))), _a_operand(e, x8, kit)

        pipe = plan.knobs.m4_apipe
        nxt = a_ops(0) if pipe else None
        for grp in range(A.GU_KI // 4):
            for q in range(4):
                kit = grp * 4 + q
                if pipe:
                    ea, a_op = nxt
                    if kit + 1 < A.GU_KI:
                        nxt = a_ops(kit + 1)
                    b.sched_barrier(0)
                else:
                    ea, a_op = a_ops(kit)
                for j in range(J):
                    ws, ss = loads[j][0][grp]
                    sw = b.lshr(ss[q // 2], e.mul(part, 8))
                    accs[j] = _mma(e, accs[j], a_op, ws[q], ea, sw, (q % 2) * 2)
        with b.scf_if(e.lt(e.lane, 16)):
            for j in range(J):
                for r in range(R):
                    e.st(e.ix("gred", e.add(e.mul(e.wave, 16), j * 128, e.lane), r), b.vec_extract(accs[j], r))
    for j in range(J if not plan.knobs.m4_interleave else 0):
        groups, valid = loads[j]

        def chain(groups=groups, j=j):
            acc = b.zero_vec_f32(4)
            for grp in range(A.GU_KI // 4):
                ws, ss = groups[grp]
                for q in range(4):
                    kit = grp * 4 + q
                    ea = e.ld(e.add(xe, kit * 4, e.udiv(e.lane, 16)))
                    sw = b.lshr(ss[q // 2], e.mul(part, 8))
                    acc = _mma(e, acc, _a_operand(e, x8, kit), ws[q], ea, sw, (q % 2) * 2)
            with b.scf_if(e.lt(e.lane, 16)):
                for r in range(R):
                    e.st(e.ix("gred", e.add(e.mul(e.wave, 16), j * 128, e.lane), r), b.vec_extract(acc, r))

        if plan.knobs.m4_guard == "branch":
            with b.scf_if(valid):
                chain()
        else:  # an invalid unit's loads are OOB (zeros); its gred entries are never read
            chain()
    e.barrier()
    _m4_epilogue(e, task, n_units, n_slots, early)


def _m4_epilogue(e: MoeCtx, task, n_units, n_slots, early: bool = False) -> None:
    b, plan = e.b, e.plan
    R, W, G = e.R, plan.knobs.m4_waves, plan.groups
    QB = W // 4  # 32-column blocks per unit
    n_entries = plan.entries
    c = e.umod(e.lane, 32)
    for p in range(-(-n_entries // 16)):
        ent = e.add(p * 16, e.mul(e.wave, 2), e.udiv(e.lane, 32))
        ent_c = e.imin(ent, n_entries - 1)
        # entry = ((k * R) + r) * QB + qb, k = j * G + g
        qb = e.umod(ent_c, QB) if QB > 1 else e.c(0)
        kr = e.udiv(ent_c, QB) if QB > 1 else ent_c
        r = e.umod(kr, R) if R > 1 else e.c(0)
        k = e.udiv(kr, R) if R > 1 else kr
        j = e.udiv(k, G) if G > 1 else k
        g = e.umod(k, G) if G > 1 else e.c(0)
        # unit k's slot / column block / validity (the per-unit formulas, at runtime k)
        if plan.knobs.m4_map == "mono":
            u = e.add(e.mul(task, G * plan.j), k)
            valid = e.lt(u, n_units)
        else:
            start = b.lshr(e.mul(task, n_units), e.c(8))
            end = b.lshr(e.mul(e.add(task, 1), n_units), e.c(8))
            u = e.add(start, k)
            valid = e.lt(u, end)
        u = e.imin(u, b.sub(n_units, e.c(1)))
        slot = e.udiv(u, plan.units_per_slot)
        cb = e.umod(u, plan.units_per_slot)
        if early:  # every unit is the shared expert's: rank 10 in every row
            rank, shared = e.c(A.TOP_K), e.eq(e.c(0), 0)
        else:
            rank = _slot_rank(e, slot, r)
            shared = _slot_is_shared(e, slot)
        live = b.land(b.land(valid, e.ge(rank, 0)), e.lt(ent, n_entries))
        base = e.add(e.mul(j, 128), e.mul(g, W * 16), e.mul(qb, 32), c)
        x = bf16r(b, e.ld(e.ix("gred", base, r), F32))
        y = bf16r(b, e.ld(e.ix("gred", e.add(base, W * 8), r), F32))
        gu_epilogue_entry(e, x, y, shared, live, rank, r, c, lambda cb=cb, qb=qb: e.add(e.mul(cb, QB), qb))


def gu_epilogue_entry(e: MoeCtx, x, y, shared, live, rank, r, c, cb32) -> None:
    """One M4 epilogue entry (32 lanes, ``c`` = lane % 32 = the column of the
    32-column block): the routed / shared SiLU variant of the bf16 gate ``x``
    and up ``y``, the block's amax and E8M0 exponent, the MXFP8 pack, and the
    puts into ``dn`` row ``r`` at rank ``rank`` (``live``: i1, put at all).
    ``cb32``: the 32-column block index, or a callable emitting it (called at
    the point the M4 epilogue computes it; the default IR's op order)."""
    b = e.b
    h_routed = bf16r(b, b.fmul(bf16r(b, b.fmul(x, y)), bf16r(b, sigmoid_m(e, x))))
    h_shared = bf16r(b, b.fmul(bf16r(b, b.fmul(x, sigmoid_acc(e, x))), y))
    h = b.select(shared, h_shared, h_routed)
    amax = Q.block_amax(b, [h], 32)
    ex, scale = Q.e8m0_fp8(b, amax)
    hb = b.bitcast(h, I32)
    src = e.add(e.mul(e.udiv(e.lane, 32), 32), e.mul(e.umod(e.lane, 4), 8))
    v8 = [b.bitcast(b.ds_bpermute(e.mul(e.add(src, i), 4), hb), F32) for i in range(8)]
    lo = QL.pack_fp8x4(b, v8[0], v8[1], v8[2], v8[3], scale, cvt=e.k.cvt)
    hi = QL.pack_fp8x4(b, v8[4], v8[5], v8[6], v8[7], scale, cvt=e.k.cvt)
    if callable(cb32):
        cb32 = cb32()
    rk = e.imax(rank, 0)
    with b.scf_if(b.land(live, e.lt(c, 4))):
        e.put2(e.pix("dn", e.add(e.mul(rk, A.DN_ROW), e.mul(cb32, 8), e.mul(e.umod(e.lane, 4), 2)), r), lo, hi)
    with b.scf_if(b.land(live, e.eq(c, 0))):
        e.put(e.pix("dn", e.add(e.mul(rk, A.DN_ROW), A.DN_X8, cb32), r), ex)


# ============================================================ M5


def _m5_extras(e: MoeCtx, t: Value) -> None:
    """Waves ``r < R``: row r's shared gate, residual words and injection
    weights (lane i < 32: stream i // 8) into LDS."""
    b = e.b
    with b.scf_if(e.lt(e.wave, e.R)):
        r = e.wave
        lane = e.lane
        bs = e.udiv(e.imin(lane, 31), 8)
        word = e.add(e.mul(bs, A.ROW_WORDS), e.mul(t, 8), e.umod(lane, 8))
        got = e.poll([(e.pix("sg", 0, r), 1), (e.pix("w", word, r), 1), (e.pix("minj", bs, r), 1)], route=True)
        with b.scf_if(e.eq(lane, 0)):
            e.st(e.add(e.off["rg"], r), got[0][0])
        with b.scf_if(e.lt(lane, 32)):
            e.st(e.ix("res", lane, r), got[1][0])
            e.st(e.ix("inj", lane, r), got[2][0])


def _dn_issue(e: MoeCtx, rs_w, rs_s, t, off):
    ws = [
        _load4(e, rs_w, e.add(e.mul(e.add(e.mul(e.add(e.mul(t, A.DN_KI), kk), 64), e.lane), 16), off))
        for kk in range(A.DN_STEPS)
    ]
    ss = [
        _load1(e, rs_s, e.add(e.mul(e.add(e.mul(e.add(e.mul(e.udiv(t, 2), A.DN_K1), k1), 64), e.lane), 4), off))
        for k1 in range(A.DN_K1)
    ]
    return ws, ss


def _dn_operands(e: MoeCtx, row, kk):
    """(A operand, A exponent) of k_iter ``kk`` of the activation row at LDS ``row``."""
    return _a_operand(e, row, kk), e.ld(e.add(row, A.DN_X8 + kk * 4, e.udiv(e.lane, 16)))


def _dn_mma(e: MoeCtx, acc, unit, kk, row, shift, ops=None):
    """``ops``: the preloaded :func:`_dn_operands` of every k_iter (``a_pre``)."""
    ws, ss = unit
    if ops is None:  # the op order of the default build
        ea = e.ld(e.add(row, A.DN_X8 + kk * 4, e.udiv(e.lane, 16)))
        sw = e.b.lshr(ss[kk // 2], shift)
        return _mma(e, acc, _a_operand(e, row, kk), ws[kk], ea, sw, (kk % 2) * 2)
    a, ea = ops[kk]
    sw = e.b.lshr(ss[kk // 2], shift)
    return _mma(e, acc, a, ws[kk], ea, sw, (kk % 2) * 2)


def _dn_ops(e: MoeCtx, row):
    if not e.k.a_pre:
        return None
    ops = [_dn_operands(e, row, kk) for kk in range(A.DN_STEPS)]
    e.b.sched_barrier(0)
    return ops


def _chain_ranks(e: MoeCtx, k: Value, kc: Value) -> list:
    """``m5_ranks=pre``: the rank of chain ``k``'s slot in each request row
    (-1: the row does not route to it), wave-uniform. Not deduped, slot
    ``k`` (past a leading shared slot) is rank ``k % 10`` of row ``k // 10``."""
    b = e.b
    if not e.k.dedupe:
        kk = e.imax(b.sub(k, e.c(1)) if e.k.shared_first else k, 0)
        row, rank = b.div(kk, e.c(A.TOP_K)), b.mod(kk, e.c(A.TOP_K))
        return [e.sel(e.eq(row, r), rank, -1) for r in range(e.R)]
    return [e.ld_u(e.add(e.off["slot_r"], e.mul(kc, 4), r)) for r in range(e.R)]


def m5_routed_by_m4(e: MoeCtx) -> Value:
    """i1: this CTA's M4 task routed the rows and built the slot tables (M5
    reuses them): it ran an M4 task, not an early shared-only one."""
    plan = e.plan
    task = m4_task(e)
    ran = e.lt(task, plan.m4_tasks_max)
    if not plan.knobs.shared_early:
        return ran
    return e.b.land(ran, e.b.lnot(_m4_shared_only(e, task)))


def m5_task(e: MoeCtx) -> Value:
    return e.umod(e.add(e.bid, A.GRID - e.plan.base("m5")), A.GRID)


def emit_m5(e: MoeCtx, routed: Value | None) -> None:
    """``routed``: i1 uniform, true when this CTA's M4 already built the slot
    tables and router weights (``None``: never)."""
    b, plan = e.b, e.plan
    R = e.R
    t = e.umod(e.add(e.bid, A.GRID - plan.base("m5")), A.GRID)
    with b.scf_if(e.lt(t, A.HIDDEN // 16)):
        e.barrier()
        need = e.c(1) if routed is None else b.select(routed, e.c(0), e.c(1))
        x5 = plan.knobs.m5_extras
        if x5 == "first":
            _m5_extras(e, t)
        elif x5 == "m4":  # done in M4 when this CTA ran an M4 task
            with b.scf_if(e.ne(need, 0)):
                _m5_extras(e, t)
        with b.scf_if(e.ne(need, 0)):
            _route_rows(e, weights=True, gate=plan.knobs.route_gate == "pre_poll")
        if x5 == "after":
            _m5_extras(e, t)
        e.barrier()
        if plan.knobs.slots == "table":
            with b.scf_if(e.ne(need, 0)):
                _build_slots(e)
        # the dedupe-off path builds its tables the same way (one slot per (row, rank))
        n_slots = _n_slots(e)
        nhint = (A.TOP_K + 1) * A.GU_TILES

        def hint(m):
            if R == 1:
                k, r = m, 0
            else:
                k, r = b.mod(m, e.c(nhint)), b.div(m, e.c(nhint))
            idx = e.add(e.mul(e.udiv(k, A.GU_TILES), A.DN_ROW), A.DN_X8, e.mul(e.umod(k, A.GU_TILES), 2))
            return e.mb, e.pix("dn", idx, r)

        issue = plan.knobs.m5_issue
        shift = e.mul(e.umod(t, 2), 8)

        def issue_all():
            units = []
            pre = e.k.m5_ranks == "pre"
            ents = []
            if pre:  # every chain's slot lookups first: one LDS round trip, not one per chain
                for s in range(plan.m5_chains):
                    k = e.add(e.wave, A.WAVES * s)
                    kc = e.imin(k, b.sub(n_slots, e.c(1)))
                    ents.append((_slot_entry(e, kc), _chain_ranks(e, k, kc)))
            for s in range(plan.m5_chains):
                k = e.add(e.wave, A.WAVES * s)
                valid = e.lt(k, n_slots)
                last = b.sub(n_slots, e.c(1))
                kc = e.imin(k, last)
                ent = ents[s][0] if pre else e.ld_u(e.add(e.off["slot_e"], kc))
                shared = e.eq(ent, SHARED)
                expert = e.imax(ent, 0)
                ex64 = b.zext(expert, I64)
                aw = b.select(
                    shared, e.addr["sh_w_dn"],
                    b.add(e.addr["e_w_dn"], b.mul(ex64, b.const_i64(A.DN_W_BYTES))),
                )
                asc = b.select(
                    shared, e.addr["sh_s_dn"],
                    b.add(e.addr["e_s_dn"], b.mul(ex64, b.const_i64(A.DN_S_BYTES))),
                )
                rs_w = _rsrc(e, b.readfirstlane(aw), A.DN_W_BYTES)
                rs_s = _rsrc(e, b.readfirstlane(asc), A.DN_S_BYTES)
                unit = _dn_issue(e, rs_w, rs_s, t, b.select(valid, e.c(0), e.c(OOB)))
                ranks = ents[s][1] if pre else None
                units.append((unit, k, b.land(valid, e.ne(ent, SHARED)), b.land(valid, shared), ranks))
            return units

        if issue == "mono":
            if R == 1 or routed is None:
                M.pre_poll(b, hint, 1, e.tag, e.k.mailbox)
            else:
                # rows > 1: at once on a CTA whose M4 task has ended
                with b.scf_if(e.eq(need, 1)):
                    M.pre_poll(b, hint, 1, e.tag, e.k.mailbox)
            units = issue_all()
            M.pre_poll(b, hint, nhint * R, e.tag, e.k.mailbox)
        elif issue == "early":
            units = issue_all()
            M.pre_poll(b, hint, nhint * R, e.tag, e.k.mailbox)
        else:
            M.pre_poll(b, hint, nhint * R, e.tag, e.k.mailbox)
            units = issue_all()
        if plan.knobs.sched_issue:
            b.sched_barrier(0)
        hi = e.imin(e.add(e.tid, A.THREADS), DN_LOADS - 1)
        v = e.poll([(e.pix("dn", e.mul(j, 2), r), 2) for r in range(R) for j in (e.tid, hi)])
        for r in range(R):
            e.st(e.ix("dn", e.mul(e.tid, 2), r), b.vec_pack(v[2 * r], I32), 2)
            with b.scf_if(e.lt(e.add(e.tid, A.THREADS), DN_LOADS)):
                e.st(e.ix("dn", e.mul(hi, 2), r), b.vec_pack(v[2 * r + 1], I32), 2)
        e.barrier()
        r_l = e.imin(e.umod(e.lane, 16), R - 1)
        for unit, k, routed_chain, shared_chain, ranks in units:
            with b.scf_if(routed_chain):
                if ranks is None:
                    rank_l = e.imax(e.ld(e.add(e.off["slot_r"], e.mul(k, 4), r_l)), 0)
                else:
                    rank_l = ranks[0]
                    for r in range(1, R):
                        rank_l = b.select(e.eq(r_l, r), ranks[r], rank_l)
                    rank_l = e.imax(rank_l, 0)
                row = e.ix("dn", e.mul(rank_l, A.DN_ROW), r_l)
                ops = _dn_ops(e, row)
                acc = b.zero_vec_f32(4)
                for kk in range(A.DN_STEPS):
                    acc = _dn_mma(e, acc, unit, kk, row, shift, ops)
                with b.scf_if(e.lt(e.lane, 16)):
                    for r in range(R):
                        rk = e.ld(e.add(e.off["slot_r"], e.mul(k, 4), r)) if ranks is None else ranks[r]
                        with b.scf_if(e.ge(rk, 0)):
                            e.st(e.ix("ys", e.add(e.mul(rk, 16), e.lane), r), bf16r(b, b.vec_extract(acc, r)))
            with b.scf_if(shared_chain):
                row = e.ix("dn", A.TOP_K * A.DN_ROW, r_l)
                ops = _dn_ops(e, row)
                zero = b.zero_vec_f32(4)
                w0 = _dn_mma(e, _dn_mma(e, zero, unit, 0, row, shift, ops), unit, 4, row, shift, ops)
                w1 = _dn_mma(e, zero, unit, 1, row, shift, ops)
                w2 = _dn_mma(e, zero, unit, 2, row, shift, ops)
                w3 = _dn_mma(e, zero, unit, 3, row, shift, ops)
                with b.scf_if(e.lt(e.lane, 16)):
                    for r in range(R):
                        x = [b.vec_extract(w, r) for w in (w0, w1, w2, w3)]
                        sh = bf16r(b, b.fadd(b.fadd(b.fadd(x[0], x[1]), x[2]), x[3]))
                        e.st(e.ix("ys", e.add(A.TOP_K * 16, e.lane), r), sh)
        e.barrier()
        # thread 32 r + i: stream i // 8, columns 16 t + 2 (i % 8) and the next
        with b.scf_if(e.lt(e.tid, 32 * R)):
            combine_tile(e, t, lambda k, h, jj, rt: e.ld(e.ix("ys", e.add(k * 16 + h, jj), rt), F32))


def combine_tile(e: MoeCtx, t, ys_at, order=None, ids=None, tw_at=None, rg_at=None, inj_at=None,
                 res_at=None) -> None:
    """The rank-ordered combine and the MLP HC write of hidden tile ``t`` (16
    columns), thread ``32 r + i`` (callers guard ``tid < 32 R``): stream
    ``i // 8``, columns ``16 t + 2 (i % 8)`` and the next. ``ys_at(k, h, jj,
    rt)``: row ``rt``'s bf16-rounded down output (f32) of rank ``k`` (10: the
    shared expert) at column ``jj + h`` of the tile, called where M5 loads
    it. Router weights, shared gate, injection weights and residual words
    come from LDS ``tw`` / ``rg`` / ``inj`` / ``res`` (or the callables
    ``tw_at(k, rt)``, ``rg_at(rt)``, ``inj_at(i, rt)``, ``res_at(i, rt)``,
    called at the same points). ``ids``: ``(i, rt)`` instead of the thread's
    own. ``order``: the ranks in combine order (default ``0 .. 9``; a
    permutation only for mutants)."""
    b = e.b
    if ids is None:
        i = e.umod(e.tid, 32)
        rt = e.udiv(e.tid, 32)
    else:
        i, rt = ids
    jj = e.mul(e.umod(i, 8), 2)
    acc0, acc1 = e.f(0.0), e.f(0.0)
    for k in order or range(A.TOP_K):
        wk = e.ld(e.ix("tw", k, rt), F32) if tw_at is None else tw_at(k, rt)
        acc0 = bf16r(b, b.fma(ys_at(k, 0, jj, rt), wk, acc0))
        acc1 = bf16r(b, b.fma(ys_at(k, 1, jj, rt), wk, acc1))
    rg = e.ld(e.add(e.off["rg"], rt), F32) if rg_at is None else rg_at(rt)
    sh0 = ys_at(A.TOP_K, 0, jj, rt)
    sh1 = ys_at(A.TOP_K, 1, jj, rt)
    blk0 = bf16r(b, b.fadd(acc0, bf16r(b, b.fmul(rg, sh0))))
    blk1 = bf16r(b, b.fadd(acc1, bf16r(b, b.fmul(rg, sh1))))
    inj = e.ld(e.ix("inj", i, rt), F32) if inj_at is None else inj_at(i, rt)
    r0, r1 = M.bf2_f32(b, e.ld(e.ix("res", i, rt)) if res_at is None else res_at(i, rt))
    out = M.bf16_pair(
        b, b.fadd(r0, bf16r(b, b.fmul(blk0, inj))), b.fadd(r1, bf16r(b, b.fmul(blk1, inj)))
    )
    ptr = b.int_to_ptr(e.addr["h_out"], PtrType(I32, "global"))
    idx = e.add(e.mul(e.udiv(i, 8), A.ROW_WORDS), e.mul(t, 8), e.umod(i, 8), e.mul(rt, A.HC_WORDS))
    b.global_store(ptr, idx, out, align=4)


# ============================================================ standalone kernel


@dataclass(frozen=True)
class MoeSuffixSpec:
    """A split-launch MoE suffix: ``stages`` (``m3 m4 m5`` or ``m4 m5``) of a
    ``rows``-row build for ``experts`` routed experts."""

    rows: int = 1
    experts: int = A.EXPERTS
    stages: tuple = STAGES
    knobs: MoeKnobs = field(default_factory=MoeKnobs)

    @property
    def plan(self) -> MoePlan:
        return MoePlan(self.rows, self.experts, self.knobs, tuple(self.stages))

    def kernel_name(self) -> str:
        suf = self.knobs.suffix()
        py = "_pyonly"  # math.exp, int_to_ptr: Python engine only
        return (
            f"qwen38_moe_{'_'.join(self.stages)}_e{self.experts}_r{self.rows}"
            + (f"_{suf}" if suf else "")
            + py
        )


def build_moe_suffix(spec: MoeSuffixSpec):
    """The standalone suffix kernel: the mono layer kernel's launch signature
    (``step`` i64, ``layer`` i32, the 41 :data:`abi.LAYER_ARGS` addresses),
    grid ``abi.GRID`` x ``abi.THREADS``. Python engine only (``math.exp``,
    ``int_to_ptr``, the op_sel scaled MFMA): compile with ``backend="python"``."""
    plan = spec.plan
    plan.validate()
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    step = b.param("step", I64)
    layer = b.param("layer", I32)
    addr = {}
    for name in A.LAYER_ARGS:
        addr[name] = b.param(name, I64)
    step_ptr = b.int_to_ptr(step, PtrType(I32, "global"))
    step_v = b.global_load(step_ptr, b.const_i32(0), I32, align=4)
    tag = M.launch_tag(b, step_v, b.readfirstlane(layer))
    e = MoeCtx(b, plan, tag=tag, layer=b.readfirstlane(layer), addr=addr)
    e.stamp_begin(len(plan.stages))
    i = 0
    if "m3" in plan.stages:
        e.stamp_enter(i)
        emit_m3(e)
        e.stamp_exit(i)
        i += 1
    e.stamp_enter(i)
    emit_m4(e)
    e.stamp_exit(i)
    i += 1
    e.stamp_enter(i)
    routed = m5_routed_by_m4(e)
    emit_m5(e, routed)
    e.stamp_exit(i)
    e.stamp_flush()
    b.ret()
    return b.kernel


def signature():
    """Launcher signature (``rocke.runtime`` manifest form) of the suffix kernel."""
    return [{"name": "step", "type": "i64"}, {"name": "layer", "type": "i32"}] + [
        {"name": n, "type": "i64"} for n in A.LAYER_ARGS
    ]


def backend_for(kernel=None) -> str:
    return "python"


__all__ = [
    "MoeCtx",
    "MoeKnobs",
    "MoePlan",
    "MoeSuffixSpec",
    "STAGES",
    "build_moe_suffix",
    "emit_m3",
    "emit_m4",
    "emit_m5",
    "lds_layout",
    "signature",
    "top10",
]
