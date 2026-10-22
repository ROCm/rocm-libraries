# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Toy layer chain on the layer-kernel framework (Qwen3.8 roadmap T9).

A chain of ``depth`` GEMV levels with a reduction between levels, built from
stage emitters through :func:`framework.build_layer_kernel`: it measures what
the framework itself costs per hand-off (hop latency x depth) and at the
kernel start (launch-to-first-stage ramp), on the mono's grid (256 x 512, one
CTA per CU, ATOM's 41-argument ABI), and carries the early structural
prototypes (last-arriver hop fold, 2-layer stack, idle-CTA prefetch).

Math (int32, wrapping; exact, so every hand-off is checked bit for bit):

* level ``i``, task ``t`` (``producers`` tasks): ``dot = sum_k W[l][i][t][k]
  * X[k]`` over ``work * threads`` words, then ``y = dot + r_prev * (t + 1) +
  tag``;
* ``r_i = sum_t y_t``; ``r_prev`` of level 0 is the seed
  ``h_in[4 (layers - 1)]`` (:func:`seed_word`: the previous launch's last
  layer output; ``h_in[0]`` for one layer) or, past the first layer of a
  multi-layer launch, the previous layer's last ``r``;
* the layer writes ``h_out[4 li] = r_{depth-1}`` and adds it into
  ``h_out[4 li + 1]`` (a running checksum over launches).

Hop modes (``hop`` knob):

* ``a2a`` (mono style): every level-``i+1`` task polls all ``producers`` y
  pairs of level ``i`` and sums them itself; a final 1-task ``out`` stage
  writes ``h_out``. One hop per level.
* ``reduce_task``: a 1-task stage ``r{i}`` polls the y pairs and publishes
  ``r_i`` in one pair; level ``i+1`` polls that pair. Two hops per level.
* ``last_arriver``: each level-``i`` task, after its put, increments a
  counter (``global_atomic_add``, returning); the task that brings it to
  ``producers`` reads every y, resets the counter and publishes ``r_i``
  (and ``h_out`` at the last level). One hop plus one atomic round trip.

Params (ATOM ABI, :data:`framework.ATOM_LAYER_ARGS`): ``scratch`` (mailboxes,
stamps tail), ``h_in`` (seed), ``h_out``, ``w_down`` (W: ``[layers][depth]
[producers][work * threads]`` i32), ``w_up`` (X: ``[work * threads]`` i32).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers import device_ops as dops
from rocke.helpers.lds_arena import LdsEntry
from rocke.helpers.task_schedule import Dep, Stage, TaskTable

from .framework import (
    FrameworkKnobs,
    LayerCtx,
    Params,
    ScratchLayout,
    StageSpec,
    build_layer_kernel,
)

__all__ = [
    "CONSUMES",
    "HOPS",
    "PREFETCHES",
    "ToyChainSpec",
    "build_step_kernel",
    "best_spec",
    "build_toy_chain",
    "seed_word",
    "toy_layout",
    "toy_reference",
    "toy_stage_names",
    "toy_table",
]

HOPS = ("a2a", "reduce_task", "last_arriver")
CONSUMES = ("wave0", "waves")
PREFETCHES = ("off", "l2_touch")
PLACEMENTS = ("atom", "sequential", "xcd_blocked")
# wave sums: ``swizzle`` = ``warp_shuffle_xor`` (ds_swizzle / ds_bpermute, LDS
# port); ``dpp`` = DPP row ops + ds_swizzle xor 16 + permlane32_swap (VALU,
# the mono's ``xred`` shape)
REDUCES = ("swizzle", "dpp")
# ``entry``: the first layer's g0 weight / X loads are issued at the kernel
# entry (framework ``entry`` hook), before the launch tag's wait on the step
# counter, so the step load and the cold weight stream overlap (the ramp)
EARLIES = ("off", "entry")


@dataclass(frozen=True)
class ToyChainSpec:
    depth: int = 4
    producers: int = 160
    work: int = 0  # i32 words of W per thread per GEMV task
    hop: str = "a2a"
    consume: str = "wave0"
    layers: int = 1
    parity: bool = True
    prefetch: str = "off"
    placement: str = "atom"
    reduce: str = "swizzle"
    early: str = "off"
    knobs: FrameworkKnobs = field(default_factory=FrameworkKnobs)
    grid: int = 256
    threads: int = 512

    def __post_init__(self) -> None:
        if self.hop not in HOPS:
            raise ValueError(f"hop must be one of {HOPS}")
        if self.consume not in CONSUMES:
            raise ValueError(f"consume must be one of {CONSUMES}")
        if self.prefetch not in PREFETCHES:
            raise ValueError(f"prefetch must be one of {PREFETCHES}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        if self.early not in EARLIES:
            raise ValueError(f"early must be one of {EARLIES}")
        if self.reduce not in REDUCES:
            raise ValueError(f"reduce must be one of {REDUCES}")
        if not 1 <= self.producers <= min(self.grid, 64 * (self.threads // 64)):
            raise ValueError("producers must fit the grid and one wave per 64")
        if self.depth < 1 or self.layers < 1 or self.work < 0:
            raise ValueError("depth, layers >= 1; work >= 0")
        if self.prefetch != "off" and (self.work % 4 or self.producers % 8):
            raise ValueError("prefetch needs work % 4 == 0 and producers % 8 == 0")

    @property
    def waves(self) -> int:
        return self.threads // 64

    @property
    def row_words(self) -> int:
        """W words per GEMV task."""
        return self.work * self.threads

    def name(self) -> str:
        parts = [
            f"rk_toy_d{self.depth}_p{self.producers}_w{self.work}", self.hop,
            self.consume,
        ]  # fmt: skip
        if self.layers > 1:
            parts.append(f"l{self.layers}" + ("" if self.parity else "np"))
        if self.prefetch != "off":
            parts.append(f"pf{self.prefetch}")
        if self.placement != "atom":
            parts.append(f"pl{self.placement}")
        if self.early != "off":
            parts.append(f"ea{self.early}")
        if self.reduce != "swizzle":
            parts.append(f"rd{self.reduce}")
        return "_".join(parts)


def toy_stage_names(spec: ToyChainSpec) -> tuple[str, ...]:
    names = []
    for i in range(spec.depth):
        names.append(f"g{i}")
        if spec.hop == "reduce_task":
            names.append(f"r{i}")
    if spec.hop == "a2a":
        names.append("out")
    return tuple(names)


def toy_table(spec: ToyChainSpec) -> TaskTable:
    """The toy's schedule. ``atom`` placement: GEMV tasks from CTA 0, the
    1-task stages on the last CTA (idle during the GEMV levels)."""
    P, last = spec.producers, spec.grid - 1
    stages, parity = [], []
    for i in range(spec.depth):
        deps: list = []
        reads: list[str] = []
        if i == 0:
            if spec.layers > 1:
                prev = {"a2a": "out", "reduce_task": f"r{spec.depth - 1}"}.get(
                    spec.hop, f"g{spec.depth - 1}"
                )
                # previous layer: a2a polls its last y's; others its last r
                src = f"g{spec.depth - 1}" if spec.hop == "a2a" else prev
                deps.append(Dep(src, layer=-1))
                reads.append(f"y{spec.depth - 1}" if spec.hop == "a2a" else f"r{spec.depth - 1}")
            reads.append("seed")
        elif spec.hop == "a2a":
            deps.append(f"g{i - 1}")
            reads.append(f"y{i - 1}")
        elif spec.hop == "reduce_task":
            deps.append(f"r{i - 1}")
            reads.append(f"r{i - 1}")
        else:
            deps.append(f"g{i - 1}")
            reads.append(f"r{i - 1}")
        writes = [f"y{i}"] + ([f"r{i}", f"c{i}"] if spec.hop == "last_arriver" else [])
        if spec.hop == "last_arriver":
            reads.append(f"y{i}")  # the last arriver reads its level's y's
            reads.append(f"c{i}")
        stages.append(
            Stage(f"g{i}", P, base=0, deps=tuple(deps), reads=tuple(reads),
                  writes=tuple(writes), lds_phase="toy")  # fmt: skip
        )
        parity += [f"y{i}", f"r{i}", f"c{i}"]
        if spec.hop == "reduce_task":
            stages.append(
                Stage(f"r{i}", 1, base=last, deps=(f"g{i}",), reads=(f"y{i}",),
                      writes=(f"r{i}",), lds_phase="toy")  # fmt: skip
            )
    if spec.hop == "a2a":
        d = spec.depth - 1
        stages.append(
            Stage("out", 1, base=last, deps=(f"g{d}",), reads=(f"y{d}",),
                  writes=("hout",), lds_phase="toy")  # fmt: skip
        )
    # last_arriver: g{i} reads its own level's y / c (the arriver's
    # intra-stage hand-off)
    inputs = ["seed"]
    t = TaskTable(tuple(stages), grid=spec.grid, inputs=tuple(inputs),
                  parity=tuple(parity), kind="toy")  # fmt: skip
    return t.place(spec.placement)


def toy_layout(spec: ToyChainSpec) -> ScratchLayout:
    regions: dict[str, int] = {}
    for i in range(spec.depth):
        regions[f"y{i}"] = spec.producers
        if spec.hop != "a2a":
            regions[f"r{i}"] = 1
        if spec.hop == "last_arriver":
            regions[f"c{i}"] = 1
    slots = 2 if (spec.layers > 1 and spec.parity) else 1
    return ScratchLayout.of(regions, rows=1, parity_slots=slots,
                            numa_copies=spec.knobs.mailbox.numa_copies)  # fmt: skip


# --------------------------------------------------------------- emitters


def _wave_sum(b: IRBuilder, v: Value, red: str) -> Value:
    """i32 sum over the wave (every lane gets it); ``red`` in :data:`REDUCES`."""
    if red == "dpp":
        # xor 1, xor 2, half-row mirror, row mirror: the 16-lane sum in
        # every lane (``IRBuilder.dpp_row_mirror`` docstring), then the rows
        for m in (1, 2, 7, 15):
            v = b.add(v, b.dpp_xor(v, m))
        v = b.add(v, b.warp_shuffle_xor(v, 16))
        lo, hi = b.permlane32_swap(v, v)
        return b.add(lo, hi)
    for k in (1, 2, 4, 8, 16, 32):
        v = b.add(v, b.warp_shuffle_xor(v, k))
    return v


def _fan_in_sum(b, ctx: LayerCtx, region: str, n: int, slot, tag, consume, red):
    """Poll the ``n`` pairs of ``region`` and sum their payloads.

    ``wave0``: wave 0's lane ``l`` polls ``l, l + 64, ...``; returns the sum
    in every lane of wave 0 (callers stay inside :func:`_in_wave0`).
    ``waves``: wave ``w`` polls pair ``64 w + l``; each wave's sum goes to
    LDS ``fan[w]``; returns ``None`` (callers barrier and read ``fan``)."""
    mb = ctx.mb(region, slot=slot, local=True)
    lane = ctx.lane
    if consume == "wave0":
        specs, valid = [], []
        for r in range((n + 63) // 64):
            idx = b.add(lane, b.const_i32(64 * r))
            valid.append(b.cmp_lt(idx, b.const_i32(n)))
            specs.append((mb, b.select(valid[-1], idx, b.const_i32(n - 1)), 1))
        vals = ctx.poll(specs, tag)
        s = b.const_i32(0)
        for (v,), ok in zip(vals, valid, strict=True):
            s = b.add(s, b.select(ok, v, b.const_i32(0)))
        return _wave_sum(b, s, red)
    nw = (n + 63) // 64
    with b.scf_if(b.cmp_lt(ctx.wave, b.const_i32(nw))):
        idx = b.add(lane, b.mul(ctx.wave, b.const_i32(64)))
        ok = b.cmp_lt(idx, b.const_i32(n))
        (v,), = ctx.poll([(mb, b.select(ok, idx, b.const_i32(n - 1)), 1)], tag)
        s = _wave_sum(b, b.select(ok, v, b.const_i32(0)), red)
        with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
            ctx.lds.st_i32(b, "fan", b.add(ctx.wave, b.const_i32(8 * _pb(ctx))), s)
    return None


def _fan_total(b, ctx: LayerCtx, n: int) -> Value:
    s = b.const_i32(0)
    for w in range((n + 63) // 64):
        s = b.add(s, ctx.lds.ld_i32(b, "fan", 8 * _pb(ctx) + w))
    return s


def _store_word(b: IRBuilder, addr: Value, idx_words: int, v: Value) -> None:
    ptr = dops.ptr_from_i64(b, addr)
    b.global_store(ptr, b.const_i32(idx_words), v, align=4)


def _write_out(b: IRBuilder, ctx: LayerCtx, p: Params, r: Value) -> None:
    """``h_out[4 li] = r``; ``h_out[4 li + 1] += r`` (lane 0 of the caller)."""
    ptr = p.ptr(b, "h_out")
    base = 4 * ctx.li
    b.global_store(ptr, b.const_i32(base), r, align=4)
    old = b.global_load_i32(ptr, b.const_i32(base + 1))
    b.global_store(ptr, b.const_i32(base + 1), b.add(old, r), align=4)


def _prev_tag(ctx: LayerCtx) -> Value:
    return ctx.tags[ctx.li - 1]


def _stage_counter(name: str, li: int, depth: int) -> int:
    """Position of stage ``name`` of layer ``li`` among the launch's stages of
    its kind: a GEMV level ``g{i}`` / reduce stage ``r{i}`` counts levels
    over the unrolled launch (``li * depth + i``), the a2a ``out`` stage
    counts layers."""
    if name == "out":
        return li
    return li * depth + int(name[1:])


def _pb(ctx: LayerCtx) -> int:
    """LDS buffer parity of the current stage: consecutive stages on one CTA
    use different copies of the exchange words, so a wave racing ahead into
    the next stage never overwrites words wave 0 still reads.

    The parity follows the stage's kind counter (:func:`_stage_counter`), not
    the stage's index in the table: with ``atom`` placement a GEMV CTA runs
    every level and nothing else, the last CTA every 1-task stage, so the
    copies alternate on every CTA, across the layers of a multi-layer launch
    too. (The table-index parity repeated a copy whenever a CTA skipped an
    odd number of stages: every GEMV level of ``reduce_task``, the layer
    boundary of a2a stacks. Other waves' ``dot`` stores of the next level do
    not wait for anything, so with L2-hot next-level weights (idle-CTA
    prefetch) they overwrote words thread 0 had not read yet: wrong ``y``
    about once per thousand launches, review-fix stress ``lk-fix-stress2``.)
    Placements where a CTA's consecutive stages still share a parity get a
    CTA barrier at the later stage's entry (:func:`_parity_guards`)."""
    depth = sum(1 for n in ctx.table.order if n.startswith("g"))
    return _stage_counter(ctx.stage_name, ctx.li, depth) & 1


def _parity_guards(spec: ToyChainSpec, table: TaskTable) -> frozenset:
    """``(li, stage)`` whose entry needs a CTA barrier: some CTA runs it
    right after (in launch order) another toy stage with the same LDS
    parity (:func:`_pb`); the barrier keeps every wave out of the stage until
    wave 0 has finished reading the earlier stage's words. Empty for ``atom``
    placement."""
    last: dict[int, tuple[int, str]] = {}
    out = set()
    for li in range(spec.layers):
        for n in table.order:
            st = table.stage(n)
            par = _stage_counter(n, li, spec.depth) & 1
            for t in range(min(st.tasks, table.grid)):
                c = table.cta_of(n, t)
                if c in last and last[c][0] == par:
                    out.add((li, n))
                last[c] = (par, n)
    return frozenset(out)


def _r_prev_wave0(b, ctx: LayerCtx, p: Params, spec: ToyChainSpec, i: int) -> Value:
    """``r_prev`` of level ``i`` in every lane of wave 0 (call inside wave 0).
    Only for the single-pair sources (seed, ``r`` pairs)."""
    if i == 0 and ctx.li == 0:
        # the previous launch's LAST layer output (``h_out[4 (layers - 1)]``;
        # word 0 for a single-layer launch); the host seeds that word
        return b.global_load_i32(p.ptr(b, "h_in"), b.const_i32(seed_word(spec)))
    if i == 0:
        src, slot, tag = f"r{spec.depth - 1}", _slot(spec, ctx.li - 1), _prev_tag(ctx)
    else:
        src, slot, tag = f"r{i - 1}", ctx.slot, ctx.tag
    (v,), = ctx.poll([(ctx.mb(src, slot=slot, local=True), 0, 1)], tag)
    return v


def seed_word(spec: ToyChainSpec) -> int:
    """``h_in`` word a launch seeds from: the last layer's ``r`` of the
    previous launch of a chain (``h_out[4 li]``); the host writes the first
    launch's seed there."""
    return 4 * (spec.layers - 1)


def _slot(spec: ToyChainSpec, li: int) -> int:
    return (li & 1) if (spec.layers > 1 and spec.parity) else 0


def _fan_source(spec: ToyChainSpec, ctx: LayerCtx, i: int):
    """(region, slot, tag) a level-``i`` a2a task fans in from, or None."""
    if spec.hop != "a2a":
        return None
    if i > 0:
        return f"y{i - 1}", ctx.slot, ctx.tag
    if ctx.li > 0:
        return f"y{spec.depth - 1}", _slot(spec, ctx.li - 1), _prev_tag(ctx)
    return None


def _gemv_emit(spec: ToyChainSpec, i: int):
    P, W, T = spec.producers, spec.work, spec.threads

    def emit(b: IRBuilder, ctx: LayerCtx, p: Params, t: Value) -> None:
        tid = ctx.tid
        lane0 = b.cmp_eq(ctx.lane, b.const_i32(0))
        wave0 = b.cmp_eq(ctx.wave, b.const_i32(0))
        # 1. weight and X loads before the poll
        prods = []
        if W and i == 0 and ctx.li == 0 and spec.early == "entry":
            prods = ctx.extra["g0_prods"]  # issued at the kernel entry
        elif W:
            row = b.add(
                b.mul(b.const_i32((ctx.li * spec.depth + i) * P), b.const_i32(1)), t
            )
            wbase = b.add(
                p.addr("w_down"),
                b.mul(b.zext(row, I64), b.const_i64(spec.row_words * 4)),
            )
            wp = dops.ptr_from_i64(b, b.readfirstlane(wbase))
            xp = p.ptr(b, "w_up")
            for j in range(W):
                k = b.add(tid, b.const_i32(j * T))
                prods.append((b.global_load_i32(wp, k), b.global_load_i32(xp, k)))
        fan = _fan_source(spec, ctx, i)
        # 2. r_prev (wave 0, or the fan-in waves), 3. dot, 4. y
        if W == 0 and (fan is None or spec.consume == "wave0"):
            with b.scf_if(wave0):
                if fan is None:
                    r_prev = _r_prev_wave0(b, ctx, p, spec, i)
                else:
                    r_prev = _fan_in_sum(b, ctx, fan[0], P, fan[1], fan[2], "wave0", spec.reduce)
                y = b.add(
                    b.mul(r_prev, b.add(t, b.const_i32(1))), ctx.tag
                )
                with b.scf_if(lane0):
                    _publish(b, ctx, p, spec, i, t, y)
            if spec.hop == "last_arriver":
                _arrive(b, ctx, p, spec, i)
            return
        # general path: LDS exchange
        if fan is None:
            with b.scf_if(wave0):
                r_prev = _r_prev_wave0(b, ctx, p, spec, i)
                with b.scf_if(lane0):
                    ctx.lds.st_i32(b, "rp", 8 * _pb(ctx), r_prev)
        elif spec.consume == "wave0":
            with b.scf_if(wave0):
                r_prev = _fan_in_sum(b, ctx, fan[0], P, fan[1], fan[2], "wave0", spec.reduce)
                with b.scf_if(lane0):
                    ctx.lds.st_i32(b, "rp", 8 * _pb(ctx), r_prev)
        else:
            _fan_in_sum(b, ctx, fan[0], P, fan[1], fan[2], "waves", spec.reduce)
        if W:
            acc = b.const_i32(0)
            for wv, xv in prods:
                acc = b.add(acc, b.mul(wv, xv))
            acc = _wave_sum(b, acc, spec.reduce)
            with b.scf_if(lane0):
                ctx.lds.st_i32(b, "dot", b.add(ctx.wave, b.const_i32(8 * _pb(ctx))), acc)
        b.sync_lds_only(fence="compiler")
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            if fan is not None and spec.consume == "waves":
                r_prev = _fan_total(b, ctx, P)
            else:
                r_prev = ctx.lds.ld_i32(b, "rp", 8 * _pb(ctx))
            dot = b.const_i32(0)
            if W:
                for w in range(spec.waves):
                    dot = b.add(dot, ctx.lds.ld_i32(b, "dot", 8 * _pb(ctx) + w))
            y = b.add(b.add(dot, b.mul(r_prev, b.add(t, b.const_i32(1)))), ctx.tag)
            _publish(b, ctx, p, spec, i, t, y)
        if spec.hop == "last_arriver":
            _arrive(b, ctx, p, spec, i)

    return emit


def _publish(b, ctx: LayerCtx, p: Params, spec: ToyChainSpec, i: int, t, y) -> None:
    """One lane: put ``y`` at ``y{i}[t]``; last_arriver: then count the
    arrival and leave the pre-add count in LDS ``la``."""
    ctx.put(ctx.mb(f"y{i}"), t, y)
    if spec.hop == "last_arriver":
        b.s_waitcnt(vmcnt=0)
        old = dops.atomic_add_scoped(
            b, b.readfirstlane(ctx.region_addr(f"c{i}")), b.const_i32(0),
            b.const_i32(1), scope="agent", returns=True,
        )  # fmt: skip
        ctx.lds.st_i32(b, "la", 8 * _pb(ctx), old)


def _arrive(b, ctx: LayerCtx, p: Params, spec: ToyChainSpec, i: int) -> None:
    """last_arriver: the task whose arrival completed the level reduces."""
    P = spec.producers
    b.sync_lds_only(fence="compiler")
    old = b.readfirstlane(ctx.lds.ld_i32(b, "la", 8 * _pb(ctx)))
    with b.scf_if(b.cmp_eq(old, b.const_i32(P - 1))):
        with b.scf_if(b.cmp_eq(ctx.wave, b.const_i32(0))):
            r = _fan_in_sum(b, ctx, f"y{i}", P, ctx.slot, ctx.tag, "wave0", spec.reduce)
            with b.scf_if(b.cmp_eq(ctx.lane, b.const_i32(0))):
                _store_word(b, b.readfirstlane(ctx.region_addr(f"c{i}")), 0, b.const_i32(0))
                b.s_waitcnt(vmcnt=0)
                ctx.put(ctx.mb(f"r{i}"), 0, r)
                if i == spec.depth - 1:
                    _write_out(b, ctx, p, r)


def _reduce_emit(spec: ToyChainSpec, i: int, *, out: bool):
    """1-task stage: fan in the level's y pairs; ``reduce_task`` publishes
    ``r_i``; the last level (or the a2a ``out`` stage) writes ``h_out``."""
    P = spec.producers

    def emit(b: IRBuilder, ctx: LayerCtx, p: Params, t: Value) -> None:
        lane0 = b.cmp_eq(ctx.lane, b.const_i32(0))
        if spec.consume == "wave0":
            with b.scf_if(b.cmp_eq(ctx.wave, b.const_i32(0))):
                r = _fan_in_sum(b, ctx, f"y{i}", P, ctx.slot, ctx.tag, "wave0", spec.reduce)
                with b.scf_if(lane0):
                    _reduce_tail(b, ctx, p, spec, i, r, out)
            return
        _fan_in_sum(b, ctx, f"y{i}", P, ctx.slot, ctx.tag, "waves", spec.reduce)
        b.sync_lds_only(fence="compiler")
        with b.scf_if(b.cmp_eq(ctx.tid, b.const_i32(0))):
            _reduce_tail(b, ctx, p, spec, i, _fan_total(b, ctx, P), out)

    return emit


def _reduce_tail(b, ctx, p, spec, i, r, out) -> None:
    if spec.hop == "reduce_task":
        ctx.put(ctx.mb(f"r{i}"), 0, r)
    if out:
        _write_out(b, ctx, p, r)


def _prefetch_prologue(spec: ToyChainSpec):
    """Idle CTAs (no GEMV task; ``atom`` placement: CTAs >= producers) touch
    the W rows of levels >= 1 (and of the next layers) owned by CTAs of their
    own XCD (round-robin dispatch: CTA b on XCD b % 8), so the owners'
    loads hit L2.

    The loads are inline asm (the compiler sees no consumer and no
    ``vmcnt`` to track), so their destination VGPRs must stay allocated until
    the data has landed: every :data:`PREFETCH_BATCH` loads end in one
    ``s_waitcnt vmcnt(0)`` asm that takes all of the batch's destinations as
    operands. Without that the allocator reuses a destination register while
    its load is in flight (a hardware fault, seen on gfx950). Branch-free:
    out-of-range rows / chunks are clamped onto the last valid one (a
    duplicate touch), so the batch spans the whole prologue."""
    P, G = spec.producers, spec.grid
    per_xcd = P // 8
    idle = (G - P) // 8
    chunks = spec.row_words // 4  # 16-byte chunks per row

    def prologue(b: IRBuilder, ctx: LayerCtx, p: Params) -> None:
        if ctx.li:
            return  # one prefetch at kernel entry covers every layer
        bid = ctx.bid
        with b.scf_if(b.cmp_ge(bid, b.const_i32(P))):
            q = b.lshr(b.sub(bid, b.const_i32(P)), b.const_i32(3))  # idle index on its XCD
            x = b.land(bid, b.const_i32(7))
            rows = [(l, i) for l in range(spec.layers) for i in range(spec.depth)
                    if (l, i) != (0, 0)]  # fmt: skip
            pending: list[Value] = []
            for l, i in rows:
                for m in range(-(-per_xcd // idle)):
                    tl = b.smin(b.add(q, b.const_i32(m * idle)), b.const_i32(per_xcd - 1))
                    t = b.add(b.mul(tl, b.const_i32(8)), x)
                    row = b.add(b.const_i32((l * spec.depth + i) * P), t)
                    a = b.add(
                        p.addr("w_down"),
                        b.mul(b.zext(row, I64), b.const_i64(spec.row_words * 4)),
                    )
                    a = b.readfirstlane(a)
                    for c in range(-(-chunks // spec.threads)):
                        k = b.add(ctx.tid, b.const_i32(c * spec.threads))
                        if (c + 1) * spec.threads > chunks:
                            k = b.smin(k, b.const_i32(chunks - 1))
                        pending.append(b.inline_asm(
                            "global_load_dwordx4 $0, $1, $2", "=&v,v,s",
                            [b.mul(k, b.const_i32(16)), a], _V4,
                        ))  # fmt: skip
                        if len(pending) == PREFETCH_BATCH:
                            _drain(b, pending)
                            pending = []
            if pending:
                _drain(b, pending)

    return prologue


# prefetch loads in flight per idle-CTA thread before one wait (x 4 VGPRs)
PREFETCH_BATCH = 16


def _drain(b: IRBuilder, outs: list[Value]) -> None:
    """``s_waitcnt vmcnt(0)`` that keeps ``outs`` (asm load destinations)
    allocated until it retires."""
    b.inline_asm("s_waitcnt vmcnt(0)", ",".join(["v"] * len(outs)), list(outs))


from rocke.core.ir import VectorType  # noqa: E402

_V4 = VectorType(I32, 4)


def build_toy_chain(spec: ToyChainSpec) -> KernelDef:
    """The toy chain kernel (params: ``step_ptr``, ``layer``, ATOM args)."""
    table = toy_table(spec)
    names = table.order
    # exchange words, two copies each (``_pb``): rp, la at [8 pb], dot / fan
    # at [8 pb + w]
    lds = (LdsEntry("rp", 16), LdsEntry("dot", 16), LdsEntry("fan", 16),
           LdsEntry("la", 16))  # fmt: skip
    specs = []
    guards = _parity_guards(spec, table)
    for n in names:
        st = table.stage(n)
        if any(g == n for _, g in guards):
            specs.append(None)  # filled below (guarded emitter)
            continue
        if n.startswith("g"):
            i = int(n[1:])
            pro = _prefetch_prologue(spec) if (spec.prefetch != "off" and i == 0) else None
            specs.append(StageSpec(st, _gemv_emit(spec, i), lds=lds, prologue=pro))
        elif n == "out":
            specs.append(StageSpec(st, _reduce_emit(spec, spec.depth - 1, out=True), lds=lds))
        else:
            i = int(n[1:])
            specs.append(
                StageSpec(st, _reduce_emit(spec, i, out=i == spec.depth - 1), lds=lds)
            )
    for k, n in enumerate(names):
        if specs[k] is None:
            specs[k] = _guarded(spec, table.stage(n), n, guards, lds)

    return build_layer_kernel(
        spec.name(), specs, table, toy_layout(spec), kind="toy",
        knobs=spec.knobs, threads=spec.threads, layers=spec.layers,
        parity=spec.parity, entry=_early_entry(spec) if spec.early == "entry" else None,
    )  # fmt: skip


def _guarded(spec: ToyChainSpec, st, n: str, guards: frozenset, lds) -> StageSpec:
    """Stage ``n`` with a CTA barrier at its entry in the layers listed in
    ``guards`` (:func:`_parity_guards`)."""
    if n.startswith("g"):
        i = int(n[1:])
        inner = _gemv_emit(spec, i)
        pro = _prefetch_prologue(spec) if (spec.prefetch != "off" and i == 0) else None
    else:
        i = spec.depth - 1 if n == "out" else int(n[1:])
        inner = _reduce_emit(spec, i, out=(n == "out" or i == spec.depth - 1))
        pro = None

    def emit(b: IRBuilder, ctx: LayerCtx, p: Params, t: Value) -> None:
        if (ctx.li, n) in guards:
            b.sync_lds_only(fence="compiler")
        inner(b, ctx, p, t)

    return StageSpec(st, emit, lds=lds, prologue=pro)


def _early_entry(spec: ToyChainSpec):
    """Framework ``entry`` hook: the first layer's g0 W and X loads, issued
    by every CTA before the step load's wait (CTAs without a g0 task load X
    instead of a W row: always in bounds, an L2 hit after the first)."""
    from rocke.helpers.task_schedule import has_task, task_index

    def entry(b: IRBuilder, ctx: LayerCtx, p: Params) -> None:
        if not spec.work:
            ctx.extra["g0_prods"] = []
            return
        T_ = spec.threads
        t = task_index(b, ctx.table, "g0", ctx.bid, 0)
        has = has_task(b, ctx.table, "g0", t)
        tc = b.select(has, t, b.const_i32(0))
        wbase = b.add(p.addr("w_down"), b.mul(b.zext(tc, I64), b.const_i64(spec.row_words * 4)))
        base = b.select(has, wbase, p.addr("w_up"))
        wp = dops.ptr_from_i64(b, b.readfirstlane(base))
        xp = p.ptr(b, "w_up")
        prods = []
        for j in range(spec.work):
            k = b.add(ctx.tid, b.const_i32(j * T_))
            prods.append((b.global_load_i32(wp, k), b.global_load_i32(xp, k)))
        ctx.extra["g0_prods"] = prods

    return entry


def best_spec(depth: int, work: int, *, layers: int = 1) -> ToyChainSpec:
    """The toy chain at the T9 gate's current best (hill climb with lookback,
    ledger ``reports/qwen38-rocke/ledger/layerkernel.md``), per shape:

    * always: DPP wave sums, calibrated dual-copy NUMA mailboxes (flip table
      in ``s_down``, host calibration
      ``persistent_mailbox.harness.calibrate_iod_home``), kernarg preload of
      the first 8 parameters (Python engine) and the ``ir`` mailbox
      (``mailbox_ir``: compiler-scheduled ``tagged_poll`` loops, Python
      engine; ~1 % on hop-only chains, neutral with weights);
    * with weights (``work > 0``, ``work % 4 == 0``) in deep chains: idle-CTA
      L2 prefetch of the later levels' weights and the first level's weights
      issued at the kernel entry -- in 2-layer launches from depth 4 on, plus
      the ``s_sleep`` poll pause (review-fix round 5: about 8 % at depth 4
      and 12 % at depth 8 of a 2-layer launch with weights; it loses at
      depths 1-2), in single-layer launches from depth 8 on (about 5 %; it
      loses at depths 2-4).

    ``layers=2`` adds the 2-layer stack. The knob defaults stay the shipped
    behaviour: the NUMA lever needs the host calibration, kernarg preload and
    the ``ir`` mailbox the Python engine."""
    from rocke.helpers import mailbox as mbx

    from . import mailbox_ir  # noqa: F401  (registers MailboxKnobs(impl="ir"))

    weights = work > 0 and work % 4 == 0
    deep = (layers >= 2 and depth >= 4) or depth >= 8
    pf = weights and deep
    sleep = pf and layers >= 2
    knobs = FrameworkKnobs(
        mailbox=mbx.MailboxKnobs(numa_copies=2, impl="ir",
                                 **({"pause": "s_sleep1"} if sleep else {})),
        numa_flip_arg="s_down", kernarg_preload=8,
    )  # fmt: skip
    kw = {"prefetch": "l2_touch", "early": "entry"} if pf else {}
    return ToyChainSpec(depth=depth, work=work, reduce="dpp", layers=layers, knobs=knobs, **kw)


def build_step_kernel() -> KernelDef:
    """``step[0] += 1`` (one thread): advances the launch tags between graph
    replays, as the serving step's meta kernel does."""
    b = IRBuilder("rk_toy_step_inc")
    b.kernel.attrs["max_workgroup_size"] = 64
    step = b.param("step", PtrType(I32, "global"))
    with b.scf_if(b.cmp_eq(b.thread_id_x(), b.const_i32(0))):
        v = b.global_load_i32(step, b.const_i32(0))
        b.global_store(step, b.const_i32(0), b.add(v, b.const_i32(1)), align=4)
    b.ret()
    return b.kernel


# --------------------------------------------------------------- host model


def toy_reference(
    spec: ToyChainSpec,
    seed: int,
    x: np.ndarray,
    w: np.ndarray,
    tags: list[int],
) -> dict:
    """Exact int32 model of one launch: ``tags[li]`` per layer of the launch;
    ``w`` shaped ``[layers][depth][producers][row_words]`` (int32), ``x``
    ``[row_words]``. Returns ``{"y": [li][i] -> int32[producers], "r":
    [li][i] -> int, "out": [li] -> int}``."""
    P = spec.producers
    i32 = np.int64(1 << 32)

    def wrap(v):
        v = int(v) % int(i32)
        return v - (1 << 32) if v >= (1 << 31) else v

    out = {"y": [], "r": [], "out": []}
    r_prev = wrap(seed)
    for li in range(spec.layers):
        ys, rs = [], []
        for i in range(spec.depth):
            if spec.work:
                dots = (w[li, i].astype(np.int64) * x.astype(np.int64)).sum(axis=1)
            else:
                dots = np.zeros(P, np.int64)
            y = np.array(
                [wrap(int(dots[t]) + r_prev * (t + 1) + tags[li]) for t in range(P)],
                dtype=np.int64,
            )
            ys.append(y.astype(np.int32))
            r_prev = wrap(int(y.sum()))
            rs.append(r_prev)
        out["y"].append(ys)
        out["r"].append(rs)
        out["out"].append(r_prev)
    return out
