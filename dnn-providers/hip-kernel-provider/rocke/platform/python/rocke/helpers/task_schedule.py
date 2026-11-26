# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static task tables for persistent (resident-grid) kernels: the host model.

A persistent kernel runs a fixed sequence of *stages* on a resident grid of
``grid`` workgroups (CTAs). Stage ``s`` has ``tasks`` tasks; task ``t`` runs on
one CTA chosen by a placement map, and every CTA walks the stages in the
table's order, running the tasks it holds. Stages hand data to each other
through global scratch regions, ordered by flags or counters that a consumer
polls.

This module is host-only (it emits no IR). It models such a schedule and
checks it before a kernel is built:

* :class:`Stage`, :class:`Dep`, :class:`TaskTable`: the data model.
* :meth:`TaskTable.validate`: dependencies point strictly earlier in the order
  (deadlock freedom on a resident grid), every placement map is one task per
  CTA a round, and every region a stage reads is written by a stage it depends
  on (or is a declared input). For a multi-layer launch it adds the
  cross-layer write-after-read check (:func:`validate_stack`).
* :meth:`TaskTable.place`: placement policies (``given`` = the table's own
  bases, ``sequential``, ``xcd_blocked``), searchable as a knob.
* :meth:`TaskTable.lds_barriers`: the stage boundaries at which a CTA barrier
  is needed because LDS phases overlay each other.

Users: the persistent hipTensor contraction monokernel
(``library/kernels/gfx950/hiptensor_mono``) validates its pack / GEMM / reduce
stages with it before building.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

__all__ = [
    "MAPS",
    "PLACEMENTS",
    "Dep",
    "Stage",
    "TaskTable",
    "ValidationError",
    "ordered_after_all",
    "validate_stack",
]

# Task -> CTA maps. ``affine``: task t of round k on CTA (base + t) % grid.
# ``xcd_blocked``: consecutive tasks on the same XCD under round-robin
# dispatch (CTA b on XCD b % xcds): task t on CTA
# ((t % per) * xcds + t // per + base) % grid with per = grid // xcds.
MAPS = ("affine", "xcd_blocked")
PLACEMENTS = ("given", "sequential", "xcd_blocked")


class ValidationError(ValueError):
    """A task table failed :meth:`TaskTable.validate`; ``errors`` lists all."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("task table invalid:\n  " + "\n  ".join(self.errors))


@dataclass(frozen=True)
class Dep:
    """A dependency of a stage on stage ``stage``: every task of the
    dependent stage polls at least one task of ``stage``.

    ``full``: every task of the dependent stage polls every task of
    ``stage`` (an all-to-all or fan-in poll), so all of ``stage`` completes
    before any dependent task proceeds.
    ``covers`` (implied by ``full``): every task of ``stage`` happens before
    at least one dependent task, directly (e.g. each producer tile is read by
    some consumer task) or through the producer stage's own internal
    hand-offs (e.g. chunk tasks feeding a final task that every consumer
    polls). Neither flag (a partial dependency that leaves producer tasks
    unordered) orders only the tasks it polls.
    ``layer``: 0 = same layer, -1 = the previous layer of a multi-layer launch
    (e.g. the first stage polling the previous layer's output tiles).
    """

    stage: str
    full: bool = True
    layer: int = 0
    covers: bool = False

    def __post_init__(self) -> None:
        if self.layer not in (0, -1):
            raise ValueError("Dep.layer must be 0 or -1")
        if self.full:
            object.__setattr__(self, "covers", True)


def _dep(d: Dep | str) -> Dep:
    return d if isinstance(d, Dep) else Dep(d)


@dataclass(frozen=True)
class Stage:
    """One stage of a persistent kernel's schedule.

    ``tasks`` tasks, task ``t`` of round ``k`` (``t + k * grid`` overall) on
    the CTA the placement map gives for ``t`` with base ``base``. ``deps``:
    stage names (full, same layer) or :class:`Dep`. ``reads`` / ``writes``:
    scratch region names, for the dataflow checks. ``lds_phase``: the LDS
    overlay phase the stage's LDS entries live in (``None``: none).
    """

    name: str
    tasks: int
    base: int = 0
    deps: tuple[Dep | str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    lds_phase: str | None = None
    map: str = "affine"

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"stage name must be an identifier, got {self.name!r}")
        if self.tasks < 1:
            raise ValueError(f"{self.name}: tasks must be >= 1")
        if self.map not in MAPS:
            raise ValueError(f"{self.name}: map must be one of {MAPS}")
        object.__setattr__(self, "deps", tuple(_dep(d) for d in self.deps))
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))


@dataclass(frozen=True)
class TaskTable:
    """A persistent kernel's schedule: ``stages`` in program order on ``grid`` CTAs.

    ``inputs``: regions the host (or the previous launch) fills, read without
    a producing stage. ``parity``: regions that a multi-layer launch double
    buffers by layer parity (slot ``layer & 1``); see :meth:`validate`.
    ``kind``: a free-form label of the kernel the table describes.
    """

    stages: tuple[Stage, ...]
    grid: int = 256
    xcds: int = 8
    inputs: tuple[str, ...] = ()
    parity: tuple[str, ...] = ()
    kind: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "parity", tuple(self.parity))
        g = self.grid
        if g < 1 or g & (g - 1):
            raise ValueError(f"grid must be a power of two, got {g}")
        if self.xcds < 1 or g % self.xcds:
            raise ValueError(f"xcds {self.xcds} must divide grid {g}")

    # ------------------------------------------------------------ queries

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages)

    @property
    def tasks(self) -> dict[str, int]:
        return {s.name: s.tasks for s in self.stages}

    @property
    def base(self) -> dict[str, int]:
        return {s.name: s.base % self.grid for s in self.stages}

    def stage(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def index(self, name: str) -> int:
        return self.order.index(name)

    def rounds(self, name: str) -> int:
        """Task rounds of stage ``name``: round ``k`` runs task ``t + k grid``."""
        return -(-self.stage(name).tasks // self.grid)

    def cta_of(self, name: str, task: int) -> int:
        """The CTA running task ``task`` (any round) of stage ``name``."""
        s = self.stage(name)
        if not 0 <= task < s.tasks:
            raise IndexError(f"{name}: task {task} not in 0..{s.tasks - 1}")
        t = task % self.grid
        if s.map == "affine":
            u = t
        else:
            per = self.grid // self.xcds
            u = (t % per) * self.xcds + t // per
        return (u + s.base) % self.grid

    def task_of(self, name: str, cta: int, rnd: int = 0) -> int | None:
        """The task of stage ``name`` that CTA ``cta`` runs in round ``rnd``,
        or ``None``. The inverse of :meth:`cta_of`."""
        s = self.stage(name)
        u = (cta - s.base) % self.grid
        if s.map == "affine":
            t = u
        else:
            per = self.grid // self.xcds
            t = (u % self.xcds) * per + u // self.xcds
        t += rnd * self.grid
        return t if t < s.tasks else None

    def cta_tasks(self, cta: int) -> list[tuple[str, int]]:
        """``(stage, task)`` pairs CTA ``cta`` runs, in program order."""
        out = []
        for s in self.stages:
            for k in range(self.rounds(s.name)):
                t = self.task_of(s.name, cta, k)
                if t is not None:
                    out.append((s.name, t))
        return out

    def histogram(self) -> list[int]:
        """Tasks per CTA (all stages and rounds)."""
        h = [0] * self.grid
        for s in self.stages:
            for t in range(s.tasks):
                h[self.cta_of(s.name, t)] += 1
        return h

    def lds_barriers(self) -> tuple[int, ...]:
        """Indices ``i`` of stages before which every CTA must barrier: the
        LDS phase of stage ``i`` differs from the last phase used before it,
        so its LDS entries overlay ones another wave may still read."""
        out, last = [], None
        for i, s in enumerate(self.stages):
            if s.lds_phase is None:
                continue
            if last is not None and s.lds_phase != last:
                out.append(i)
            last = s.lds_phase
        return tuple(out)

    # ------------------------------------------------------------ checks

    def validate(self, *, layers: int = 1, strict: bool = True) -> list[str]:
        """Check the schedule; raise :class:`ValidationError` listing every
        problem (``strict``), or return the list.

        * names unique, dependencies known;
        * same-layer dependencies point strictly earlier in the order. Every
          CTA runs the stages in order and the grid is resident, so a task
          only ever waits on tasks of earlier stages, which every CTA reaches
          first: no deadlock;
        * every map is one task per CTA a round (checked on the host model);
        * every region read is written by a same-layer dependency, by the
          stage itself (an intra-stage hand-off: tasks of one stage must
          then only wait on tasks of the same or an earlier round), or is an
          input (a single-layer launch gets the previous layer's outputs
          through the grid boundary: list them in :attr:`inputs`);
        * ``layers >= 2``: :func:`validate_stack` on ``layers`` copies of
          this table (one launch running several layers of this kind).
        """
        if layers > 1:
            return validate_stack([self] * layers, strict=strict)
        errs = self._check_single()
        if errs and strict:
            raise ValidationError(errs)
        return errs

    def _check_single(self) -> list[str]:
        errs: list[str] = []
        names = self.order
        if len(set(names)) != len(names):
            errs.append(f"duplicate stage names in {names}")
        pos = {n: i for i, n in enumerate(names)}
        for s in self.stages:
            for d in s.deps:
                if d.layer == -1:
                    continue  # resolved against the previous layer's table
                if d.stage not in pos:
                    errs.append(f"{s.name}: unknown dependency {d.stage!r}")
                elif pos[d.stage] >= pos[s.name]:
                    errs.append(
                        f"{s.name}: depends on {d.stage!r}, which is not earlier "
                        "in the order (a CTA could wait on a task it has yet to run)"
                    )
        for s in self.stages:
            seen = set()
            for t in range(min(s.tasks, self.grid)):
                seen.add(self.cta_of(s.name, t))
            if len(seen) != min(s.tasks, self.grid):
                errs.append(f"{s.name}: placement map is not one task per CTA a round")
        writers0: dict[str, set[str]] = {}
        for s in self.stages:
            for r in s.writes:
                writers0.setdefault(r, set()).add(s.name)
        for s in self.stages:
            for r in s.reads:
                if r in self.inputs or r in s.writes:
                    continue  # an input, or an intra-stage hand-off
                ok = any(
                    r in self.stage(d.stage).writes
                    for d in s.deps
                    if d.layer == 0 and d.stage in pos
                )
                prev = any(d.layer == -1 for d in s.deps)
                if not ok and not prev:
                    errs.append(
                        f"{s.name}: reads region {r!r} that no dependency writes "
                        f"(writers: {sorted(writers0.get(r, ()))}) and it is not an input"
                    )
        return errs

    # --------------------------------------------------------- placement

    def place(self, policy: str = "given", *, start: int = 0) -> TaskTable:
        """A copy with stage bases / maps set by ``policy``:

        * ``given``: unchanged (the bases the table was built with);
        * ``sequential``: stage ``i`` starts where stage ``i - 1``'s tasks
          end (``acc % grid``), from ``start``;
        * ``xcd_blocked``: ``sequential`` bases with the ``xcd_blocked`` map
          (consecutive tasks on one XCD).
        """
        if policy not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        if policy == "given":
            return self
        acc = start
        stages = []
        for s in self.stages:
            m = "xcd_blocked" if policy == "xcd_blocked" else "affine"
            stages.append(replace(s, base=acc % self.grid, map=m))
            acc += s.tasks
        return replace(self, stages=tuple(stages))

    def with_stages(self, stages: Iterable[Stage]) -> TaskTable:
        return replace(self, stages=tuple(stages))


# ---------------------------------------------------------- multi-layer

Node = tuple[int, str]


def _edges(tables: Sequence[TaskTable]) -> dict[Node, list[tuple[Node, bool, bool]]]:
    """Producer node -> [(consumer node, full, covers)]; node = (layer, stage)."""
    succ: dict[Node, list[tuple[Node, bool, bool]]] = {
        (li, s.name): [] for li, t in enumerate(tables) for s in t.stages
    }
    for li, t in enumerate(tables):
        for s in t.stages:
            for d in s.deps:
                src = li + d.layer
                if src >= 0 and (src, d.stage) in succ:
                    succ[(src, d.stage)].append(((li, s.name), d.full, d.covers))
    return succ


def ordered_after_all(tables: Sequence[TaskTable], node: Node) -> set[Node]:
    """Nodes every task of which starts only after EVERY task of ``node``
    completed, through dependencies (stage-level happens-before):

    * ``E``: ``node`` plus the nodes reached over covering edges (each task
      of ``node`` happens before some task of every node in ``E``);
    * a full edge out of a node of ``E`` orders all of ``node`` before every
      task of its consumer; from there any dependency keeps that order
      (each consumer task polls some producer task).
    """
    succ = _edges(tables)
    e, stack = {node}, [node]
    while stack:
        x = stack.pop()
        for y, _full, cov in succ[x]:
            if cov and y not in e:
                e.add(y)
                stack.append(y)
    a: set[Node] = set()
    stack = [y for x in e for y, full, _ in succ[x] if full]
    while stack:
        y = stack.pop()
        if y in a:
            continue
        a.add(y)
        stack += [z for z, _, _ in succ[y]]
    return a


def validate_stack(tables: Sequence[TaskTable], *, strict: bool = True) -> list[str]:
    """Check a multi-layer launch running ``tables[l]`` as layer ``l`` (the
    tables may describe different kinds of layers):

    * every table passes its single-layer checks;
    * ``layer=-1`` dependencies name a stage of the previous layer's table
      (the first layer's are satisfied by the previous launch), and regions
      read through them are written by that stage;
    * cross-layer write-after-read: every write of region ``R`` by layer
      ``L`` is in :func:`ordered_after_all` of every read of ``R`` by an
      earlier layer that uses the same slot (slot ``L & 1`` for regions in
      any table's :attr:`TaskTable.parity`, else the one slot). Same-CTA
      program order does not count: it orders one CTA's tasks only.
    """
    errs: list[str] = []
    for li, t in enumerate(tables):
        errs += [f"layer {li}: {e}" for e in t._check_single()]
        if li == 0:
            continue
        prev = tables[li - 1]
        pnames = set(prev.order)
        tnames = set(t.order)
        for s in t.stages:
            for d in s.deps:
                if d.layer == -1 and d.stage not in pnames:
                    errs.append(
                        f"layer {li} {s.name}: unknown previous-layer stage {d.stage!r}"
                    )
            for r in s.reads:
                if r in t.inputs or r in s.writes:
                    continue
                same = any(
                    d.layer == 0 and d.stage in tnames and r in t.stage(d.stage).writes
                    for d in s.deps
                )
                back = any(
                    d.layer == -1
                    and d.stage in pnames
                    and r in prev.stage(d.stage).writes
                    for d in s.deps
                )
                if not (same or back):
                    errs.append(
                        f"layer {li} {s.name}: reads region {r!r} that no dependency "
                        "writes and it is not an input"
                    )
    if errs:
        if strict:
            raise ValidationError(errs)
        return errs
    parity: set[str] = set()
    for t in tables:
        parity |= set(t.parity)

    def slot(r: str, li: int) -> int:
        return li & 1 if r in parity else 0

    after = {
        (li, s.name): ordered_after_all(tables, (li, s.name))
        for li, t in enumerate(tables)
        for s in t.stages
    }
    for li in range(1, len(tables)):
        for w in tables[li].stages:
            for r in w.writes:
                for lp in range(li):
                    if slot(r, lp) != slot(r, li):
                        continue
                    for rd in tables[lp].stages:
                        if r in rd.reads and (li, w.name) not in after[(lp, rd.name)]:
                            errs.append(
                                f"cross-layer WAR: layer {li} {w.name} writes {r!r} "
                                f"(slot {slot(r, li)}) before layer {lp} {rd.name} is "
                                "known to have read it"
                            )
    if errs and strict:
        raise ValidationError(errs)
    return errs
