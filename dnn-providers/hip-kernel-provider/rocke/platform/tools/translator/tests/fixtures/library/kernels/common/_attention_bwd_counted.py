# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Counted in-loop waits of the attention backward q step (``loop_sync = "counted"``).

Family glue for the backward kernels (Python-only, no C++ twin). Under
``loop_sync = "counted"`` the q step changes its issue order and the wait of its
in-loop barriers; the math, the LDS buffers and every accumulator chain stay
those of the default step, so dK / dV and every dQ partial are bit for bit
the same:

* **Issue order.** The Q / dO tile and the lse2 / Dsum rows of the *next* q
  step are loaded into registers inside the current step, before its dQ
  atomics: right after the staging barrier (``counted_issue = "early"``) or
  right before G4 (``"late"``). They are carried across the loop back-edge and
  written to the LDS tile at the top of the next step (one LDS slot, the
  default buffers). The first step of every q loop is loaded by a prologue
  before the loop, inside a 0 / 1-trip guard loop, so an empty q range issues
  no load, followed by one ``s_waitcnt vmcnt(0)`` before the loop: no load is
  pending at the loop entry, so the backend's wait for the carried values at
  the step top is not a ``vmcnt(0)`` on every trip (the counted barriers
  retire the carried loads before the back-edge). The last trip loads its own
  step again (every address stays in range; the values are not used).
* **Counted waits.** Every in-loop barrier is ``s_waitcnt vmcnt(N)
  lgkmcnt(0)`` followed by ``s_barrier``, where ``N`` is the number of dQ
  atomics the wave issued after its last global load (in program order through
  the back-edge). ``vmcnt`` retires in issue order on CDNA, and no-return
  atomics count in it, so after the wait every global load of the wave has
  retired and only the step's dQ atomics may still be in flight. ``N`` is
  exact: a smaller value drains atomics the barrier does not need, a larger one
  leaves a load in flight.

Per step (``n`` = the step's dQ atomics per wave, ``A`` / ``B`` / ``C`` the
staging, dS (or P / dS) and step-end barriers):

========  ===========  ==================================  =================
issue     ``A``        ``B``                               ``C``
========  ===========  ==================================  =================
early     ``n``        ``0`` (the next step's loads)       ``n``
late      ``n``        ``n``                               ``n``
========  ===========  ==================================  =================

With ``sched = "stage_table_atomics"`` (early issue only) the step's ``n``
atomics are the previous step's deferred batch, issued in stage A after the
next-step loads, so ``B`` is ``n`` as well; the late issue would load after
them and is refused (``_attention_bwd_sched.COUNTED_LATE_DEFERRED``). The
other schedule values leave the atomics in G4 and the table as it is.

On the first trip of a loop no load is in flight at the entry (the prologue's
``vmcnt(0)``), so every count holds there too; on every trip the next-step
loads have retired by the step-end barrier, so nothing loaded is pending
across the back-edge.

Counted waits are CDNA-only: on the RDNA targets no-return atomics count in
``vscnt``, not ``vmcnt``, so the atomics never held the barrier and the
LDS-only flavour already covers them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rocke.core.ir import F32, IRBuilder, Value, VectorType

from ._attention_bwd_addr import chunk_coords, deal_chunks

__all__ = [
    "COUNTED_ISSUES",
    "COUNTED_VMCNT_MAX",
    "CountedWaits",
    "StagedStep",
    "TileChunk",
    "issue_tile_loads",
    "store_tile",
    "tile_chunks",
    "zero_like",
]

# Where the next step's global loads are issued: right after the staging
# barrier (ahead of G0 / G2) or right before G4 (after G1 / G3).
COUNTED_ISSUES = ("early", "late")
# Largest vmcnt a CDNA s_waitcnt encodes (6 bits on gfx942 / gfx950).
COUNTED_VMCNT_MAX = 63


@dataclass(frozen=True)
class TileChunk:
    """One ``vec``-element chunk of a staged row tile, as one thread moves it.

    ``row`` / ``col`` address both the global load and the LDS store (a tail
    chunk past the tile loads and addresses the tile's last chunk); ``guard``
    is the chunk index to test against ``total`` before the store (``None``:
    every thread's chunk is inside the tile).
    """

    row: Value
    col: Value
    guard: Value | None
    total: int


def tile_chunks(
    b: IRBuilder, *, rows: int, cols: int, tid: Value, threads: int, vec: int = 8
) -> list[TileChunk]:
    """The chunks of a ``rows x cols`` tile the thread ``tid`` moves, in the
    order and with the clamps of the ``loads_first`` copy order."""
    per_row = cols // vec
    total = rows * per_row
    out = []
    for chunk, tail in deal_chunks(b, tid, total, threads):
        src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
        r, c = chunk_coords(b, src, per_row, vec)
        out.append(TileChunk(r, c, chunk if tail else None, total))
    return out


def issue_tile_loads(
    b: IRBuilder,
    loads: Sequence[Callable[[Value, Value], Value]],
    chunks: Sequence[TileChunk],
) -> list[Value]:
    """Every global load of the tiles ``loads`` (one loader per tile), tile
    by tile, chunk by chunk; the values in that order."""
    return [load(ch.row, ch.col) for load in loads for ch in chunks]


def store_tile(
    b: IRBuilder,
    views: Sequence,
    chunks: Sequence[TileChunk],
    values: Sequence[Value],
    vec: int = 8,
) -> None:
    """Store the values of :func:`issue_tile_loads` into the LDS ``views``
    (one view per tile, the same order); a tail chunk is stored only when its
    chunk is inside the tile."""
    if len(values) != len(views) * len(chunks):
        raise ValueError("staged values do not match the tile chunks")
    it = iter(values)
    for view in views:
        for ch in chunks:
            val = next(it)
            if ch.guard is None:
                view.store(b, ch.row, ch.col, val, vec)
            else:
                with b.scf_if(b.cmp_lt(ch.guard, b.const_i32(ch.total))):
                    view.store(b, ch.row, ch.col, val, vec)


def zero_like(b: IRBuilder, types: Sequence) -> list[Value]:
    """A zero of each type (f32 scalars, 16-bit vectors): the init values of a
    guard loop that yields loaded values."""
    out = []
    for t in types:
        if t == F32:
            out.append(b.const_f32(0.0))
        elif isinstance(t, VectorType):
            out.append(b.zero_vec(t.elem, t.count))
        else:
            raise ValueError(f"no zero for staged type {t}")
    return out


@dataclass(frozen=True, eq=False)
class StagedStep:
    """A q step's loaded inputs: ``stats`` (lse2, Dsum per stats key, flat)
    and ``qdo`` (the Q then the dO chunk values)."""

    stats: tuple[Value, ...]
    qdo: tuple[Value, ...]

    def flat(self) -> list[Value]:
        return list(self.stats) + list(self.qdo)

    @classmethod
    def from_flat(cls, values: Sequence[Value], n_stats: int) -> "StagedStep":
        vals = list(values)
        return cls(stats=tuple(vals[:n_stats]), qdo=tuple(vals[n_stats:]))


class CountedWaits:
    """The vmcnt of every counted barrier of one q step, from the issue order.

    ``n_atomics`` is the step's dQ atomics per wave (one ``global_atomic_add``
    per dQ slot). The emitter reports the events in program order:
    :meth:`step_entry` (the previous trip issued its next-step loads, then its
    ``n_atomics`` atomics), :meth:`loads` (this step's next-step loads),
    :meth:`atomics` (the dQ sink) and :meth:`barrier`, which emits the wait and
    the barrier. ``waits`` records the emitted counts in order.
    """

    def __init__(self, n_atomics: int) -> None:
        if not 0 <= n_atomics <= COUNTED_VMCNT_MAX:
            raise ValueError(
                f"{n_atomics} dQ atomics per wave and step exceed the vmcnt field "
                f"({COUNTED_VMCNT_MAX})"
            )
        self.n_atomics = n_atomics
        self._since_load = n_atomics
        self._issued = 0
        self.waits: list[int] = []

    def step_entry(self) -> None:
        self._since_load = self.n_atomics
        self._issued = 0
        self.waits = []

    def loads(self) -> None:
        self._since_load = 0

    def atomics(self, n: int) -> None:
        self._since_load += n
        self._issued += n

    def end_step(self) -> None:
        """Check that the step issued exactly ``n_atomics`` atomics."""
        if self._issued != self.n_atomics:
            raise AssertionError(
                f"the q step issued {self._issued} dQ atomics, the count model "
                f"assumes {self.n_atomics}"
            )

    def barrier(self, b: IRBuilder) -> None:
        n = self._since_load
        self.waits.append(n)
        b.s_waitcnt(vmcnt=n, lgkmcnt=0)
        b.s_barrier_bare()
