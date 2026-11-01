# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Slot-reuse (write-after-read) checker of the CCL exchanges.

A push of global use ``g`` into slot ``g mod S`` of rank Q overwrites the
location that use ``g - S`` (or the last earlier use mapped to that slot)
wrote; it is safe iff every read of that older use on Q happens before the
push. "Happens before" is the transitive closure of:

* program order inside one (rank, wave) of one launch;
* the kernel boundary: every event of launch L on a rank precedes every
  event of launch L + 1 on that rank (stream order);
* the exchange itself: a poll on Q of (use, unit, src P) follows P's push
  of that (use, unit) into Q.

No grid sync, barrier or host step is assumed. The checker works on an
abstract schedule (per (rank, wave) lists of push / poll events), so the
large-batch task graphs can feed it the same way the microkernels do
(:func:`oneshot_schedule`, :func:`twoshot_schedule`, :func:`spec_schedule`).
Mutants the tests use: a consumer moved off the producer's wave
(``consumer_shift``), ``S = 1``, and slots that restart at every launch
(``continuous=False``: a launch boundary without the slot-base advance).

Lamport form (CCL-LM, ``lamport=True``): after every poll the consumer
writes the slot back to empty (a ``reset`` event, a store into its OWN
arena). That store must also be performed before the next push into the
slot: every reset of use ``g - S`` on Q must precede the push of use ``g``.
The builds drain each wave's stores before its next push, so a reset is
ordered before every later event of its wave (program order);
``reset_unordered=True`` models a build without that drain (the reset gets
no outgoing program-order edge) and must fail the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Event", "Schedule", "WarViolation", "check", "oneshot_schedule", "spec_schedule", "twoshot_schedule"]


@dataclass(frozen=True)
class Event:
    kind: str  # "push" | "poll"
    ex: str
    use: int  # global use index (across launches)
    unit: int
    peers: Tuple[int, ...]  # push: destination ranks; poll: source ranks


@dataclass
class Schedule:
    ranks: int
    slots: int
    launches: int
    # programs[launch][rank][wave] = list of events
    programs: List[List[List[List[Event]]]] = field(default_factory=list)
    continuous: bool = True
    uses_per_launch: int = 0
    reset_unordered: bool = False  # lamport mutant: no drain between a wave's reset stores and its later events

    def slot(self, use: int) -> int:
        u = use if self.continuous else use % self.uses_per_launch
        return u % self.slots


class WarViolation(AssertionError):
    pass


def _mapping(units: int, waves: int, shift: int, rank: int, shift_rank: int):
    def owner(u: int) -> int:
        return u % waves

    def consumer(u: int) -> int:
        if shift and rank == shift_rank:
            return (u + shift) % waves
        return u % waves

    return owner, consumer


def oneshot_schedule(ranks: int, waves: int, units: int, uses: int, launches: int, slots: int, *,
                     consumer_shift: int = 0, shift_rank: int = 0, continuous: bool = True, ex: str = "ar",
                     lamport: bool = False, reset_unordered: bool = False) -> Schedule:
    """Unit ``u`` belongs to wave ``u mod waves`` on every rank: per use a
    wave pushes its units to every rank, then polls them from every rank
    (``lamport``: and resets what it polled)."""
    sch = Schedule(ranks, slots, launches, continuous=continuous, uses_per_launch=uses, reset_unordered=reset_unordered)
    allr = tuple(range(ranks))
    for L in range(launches):
        per_rank = []
        for r in range(ranks):
            owner, consumer = _mapping(units, waves, consumer_shift, r, shift_rank)
            progs: List[List[Event]] = [[] for _ in range(waves)]
            for k in range(uses):
                g = L * uses + k
                for u in range(units):
                    progs[owner(u)].append(Event("push", ex, g, u, allr))
                for u in range(units):
                    progs[consumer(u)].append(Event("poll", ex, g, u, allr))
                    if lamport:
                        progs[consumer(u)].append(Event("reset", ex, g, u, allr))
            per_rank.append(progs)
        sch.programs.append(per_rank)
    return sch


def twoshot_schedule(ranks: int, waves: int, units: int, uses: int, launches: int, slots: int, *, continuous: bool = True,
                     lamport: bool = False, reset_unordered: bool = False) -> Schedule:
    """Reduce-scatter to the owner ``u mod ranks``, owner sums and pushes to
    every rank (``ag``), everyone polls ``ag``; unit ``u`` on wave
    ``u mod waves`` everywhere."""
    sch = Schedule(ranks, slots, launches, continuous=continuous, uses_per_launch=uses, reset_unordered=reset_unordered)
    allr = tuple(range(ranks))
    for L in range(launches):
        per_rank = []
        for r in range(ranks):
            progs: List[List[Event]] = [[] for _ in range(waves)]
            for k in range(uses):
                g = L * uses + k
                for u in range(units):
                    progs[u % waves].append(Event("push", "rs", g, u, (u % ranks,)))
                for u in range(units):
                    if u % ranks == r:
                        progs[u % waves].append(Event("poll", "rs", g, u, allr))
                        if lamport:
                            progs[u % waves].append(Event("reset", "rs", g, u, allr))
                        progs[u % waves].append(Event("push", "ag", g, u, allr))
                for u in range(units):
                    progs[u % waves].append(Event("poll", "ag", g, u, (u % ranks,)))
                    if lamport:
                        progs[u % waves].append(Event("reset", "ag", g, u, (u % ranks,)))
            per_rank.append(progs)
        sch.programs.append(per_rank)
    return sch


def spec_schedule(spec, *, waves_total: Optional[int] = None, uses: int = 3, launches: int = 3) -> Schedule:
    """The schedule a :class:`.layout.AllReduceSpec` / ``AllGatherSpec``
    build runs (unit ``u`` on global wave ``u mod (G W)``)."""
    tw = waves_total or spec.grid() * spec.knobs.waves
    lam = getattr(spec.knobs, "form", "") == "lamport"
    if getattr(spec.knobs, "ar", "oneshot") == "twoshot":
        return twoshot_schedule(spec.ranks, tw, spec.units, uses, launches, spec.knobs.slots, lamport=lam)
    return oneshot_schedule(spec.ranks, tw, spec.units, uses, launches, spec.knobs.slots, lamport=lam)


def check(sch: Schedule) -> List[str]:
    """Violations (empty when every slot reuse is ordered after its reads)."""
    # number events; node = (launch, rank, wave, index)
    nodes: List[Tuple[int, int, int, int, Event]] = []
    index: Dict[Tuple[int, int, int, int], int] = {}
    for L, per_rank in enumerate(sch.programs):
        for r, progs in enumerate(per_rank):
            for w, prog in enumerate(progs):
                for i, e in enumerate(prog):
                    index[(L, r, w, i)] = len(nodes)
                    nodes.append((L, r, w, i, e))
    n = len(nodes)
    preds: List[List[int]] = [[] for _ in range(n)]
    # program order and launch boundaries
    last_of: Dict[Tuple[int, int], List[int]] = {}
    for L, per_rank in enumerate(sch.programs):
        for r, progs in enumerate(per_rank):
            ends = []
            for w, prog in enumerate(progs):
                for i in range(1, len(prog)):
                    j = i - 1
                    if sch.reset_unordered:
                        while j > 0 and prog[j].kind == "reset":
                            j -= 1  # an undrained reset store orders nothing after it
                    preds[index[(L, r, w, i)]].append(index[(L, r, w, j)])
                if prog:
                    ends.append(index[(L, r, w, len(prog) - 1)])
            last_of[(L, r)] = ends
            if L > 0:
                for w, prog in enumerate(progs):
                    if prog:
                        preds[index[(L, r, w, 0)]].extend(last_of[(L - 1, r)])
    # RAW edges: a poll of (ex, use, unit) from src P on rank Q follows P's push of it to Q
    push_at: Dict[Tuple[str, int, int, int, int], int] = {}  # (ex, use, unit, src, dst) -> node
    for k, (L, r, w, i, e) in enumerate(nodes):
        if e.kind == "push":
            for d in e.peers:
                push_at[(e.ex, e.use, e.unit, r, d)] = k
    for k, (L, r, w, i, e) in enumerate(nodes):
        if e.kind == "poll":
            for p in e.peers:
                src = push_at.get((e.ex, e.use, e.unit, p, r))
                if src is None:
                    raise WarViolation(f"poll {e} on rank {r} has no matching push from rank {p}")
                preds[k].append(src)
    # ancestors as bitsets, in a topological order
    order = _topo(n, preds)
    anc = [0] * n
    for k in order:
        a = 0
        for p in preds[k]:
            a |= anc[p] | (1 << p)
        anc[k] = a
    # WAR: per location (dst, ex, slot, src, unit), consecutive writers
    writes: Dict[Tuple[int, str, int, int, int], List[Tuple[int, int]]] = {}
    reads: Dict[Tuple[int, str, int, int, int, int], List[int]] = {}  # + use; lamport resets count as reads
    for k, (L, r, w, i, e) in enumerate(nodes):
        s = sch.slot(e.use)
        if e.kind == "push":
            for d in e.peers:
                writes.setdefault((d, e.ex, s, r, e.unit), []).append((e.use, k))
        else:
            for p in e.peers:
                reads.setdefault((r, e.ex, s, p, e.unit, e.use), []).append(k)
    bad = []
    for loc, ws in writes.items():
        ws.sort()
        for (u0, _k0), (u1, k1) in zip(ws, ws[1:]):
            for rk in reads.get((*loc, u0), []):
                if not (anc[k1] >> rk) & 1:
                    d, ex, s, src, unit = loc
                    bad.append(
                        f"rank {src} push of use {u1} ({ex} unit {unit}) into rank {d} slot {s} is not ordered "
                        f"after rank {d}'s read of use {u0}"
                    )
    if sch.slots < 1:
        bad.append("slots < 1")
    return bad


def _topo(n: int, preds: Sequence[Sequence[int]]) -> List[int]:
    succ: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for k, ps in enumerate(preds):
        for p in ps:
            succ[p].append(k)
            indeg[k] += 1
    stack = [k for k in range(n) if indeg[k] == 0]
    out = []
    while stack:
        k = stack.pop()
        out.append(k)
        for s in succ[k]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    if len(out) != n:
        raise WarViolation("cyclic schedule: some poll waits on a push that follows it (deadlock)")
    return out
