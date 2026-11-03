# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""dQ sinks of the attention backward tile step.

Family glue for the backward kernels (Python-only, no C++ twin).  The tile
step hands each q step's partial ``dQ = dS K`` fragment to a sink:

    sink.begin_head(b, h)                  # i64 rebase when the head changes
    sink.consume(b, dq_frag, q0, h)        # one q step
    sink.finish(b)                         # after the last kv tile / q step

``dq_frag`` is a sequence of :class:`DqSlot` (row offset inside the q step,
head-dimension column, fp32 value); ``q0`` is the i32 first query row of the
step within the sequence.

Sinks
-----
* :class:`AtomicWorkspaceSink` (default for every KV-parallel path):
  ``global_atomic_add_f32`` (agent scope, monotonic, native) into the fp32 dQ
  workspace.  The workspace pointer is rebased in i64 per (head, sequence base
  row); the remaining i32 in-slab index is ``< S_max * D`` (head-major).  In
  the token-major layout a row is ``H * D`` elements, so the pointer is
  rebased in i64 per (head, q-step target row ``min(q0, len_q - 1)``) and the
  in-slab index is ``< k_m0 * H * D``.  The
  target row is ``min(q, len_q - 1)`` (``clamp=False``, for a step whose rows
  are all inside the sequence: ``q``); a row at or past ``ws_rows`` adds an
  exact 0 at a clamped address.  No branch surrounds any atomic.  An optional
  fp32 ``scale`` multiplies each value before the atomic (scale placement
  ``dq_before_atomic``).
* :class:`PackedAtomicWorkspaceSink`: the atomic sink of a q step whose M
  rows pack several query heads; ``rows`` is a ``PackedRows`` map, the i64
  rebase is per (head, row) of each distinct slot row and a dead packed row
  adds an exact 0.
* :class:`NullSink`: the split-dQ main kernels, where G4 is not emitted.
* :class:`RegisterAccumulateSink`: dQ carried in registers across kv tiles and
  stored once in the output dtype (Q-parallel designs).
* :class:`SplitSliceSink`: per-split fp32 slices reduced in a fixed order
  (deterministic designs); recorded, not built - constructing one raises.

:class:`PendingDq` is a q step's dQ partial held back by the deferred sink
of ``sched = "stage_table_atomics"``: the tile step hands it to the next q
step (loop-carried) and issues its atomics there, through the sink above.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rocke.core.ir import IRBuilder, Value

from ._attention_bwd_addr import (
    RowSlab,
    SeqInfo,
    StridedTensorAddr,
    WorkspaceAddr,
    as_i32,
)

__all__ = [
    "AtomicWorkspaceSink",
    "DqSink",
    "DqSlot",
    "NullSink",
    "PackedAtomicWorkspaceSink",
    "PendingDq",
    "RegisterAccumulateSink",
    "SplitSliceSink",
    "split_slice_workspace_bytes",
]

IntOrValue = int | Value


@dataclass(frozen=True, eq=False)
class DqSlot:
    """One element of a q step's partial dQ fragment."""

    row: IntOrValue  # row offset inside the q step (0 .. k_m0 - 1)
    col: IntOrValue  # head-dimension column
    value: Value  # fp32 partial


@dataclass(frozen=True, eq=False)
class PendingDq:
    """A q step's dQ partial whose atomics are issued one step later.

    ``values``: the fp32 partial per lane, G4 fragment by fragment with the
    slot inner (the order the sink would have received it); ``q0``: the
    step's first q row; ``h``: its query head (the first head of a packed
    step); ``n_heads`` / ``rows_per_head``: the packing of a head-packed
    step (``None`` otherwise; ``rows_per_head`` is loop-invariant and not
    carried). :meth:`flat` / :meth:`from_flat` are the loop-carried values.
    """

    values: tuple[Value, ...]
    q0: Value
    h: Value
    n_heads: Value | None = None
    rows_per_head: Value | None = None

    def flat(self) -> list[Value]:
        extra = [] if self.n_heads is None else [self.n_heads]
        return list(self.values) + [self.q0, self.h] + extra

    def names(self, prefix: str) -> list[str]:
        out = [f"{prefix}dq{i}" for i in range(len(self.values))]
        out += [f"{prefix}dqrow", f"{prefix}dqhead"]
        return out + ([] if self.n_heads is None else [f"{prefix}dqnh"])

    def from_flat(self, vals: Sequence[Value]) -> "PendingDq":
        """A partial of this shape from loop-carried ``vals``."""
        n = len(self.values)
        if len(vals) != len(self.flat()):
            raise ValueError("carried dQ values do not match the partial")
        return PendingDq(
            values=tuple(vals[:n]),
            q0=vals[n],
            h=vals[n + 1],
            n_heads=None if self.n_heads is None else vals[n + 2],
            rows_per_head=self.rows_per_head,
        )


class DqSink:
    """Interface of a dQ sink (see the module docstring)."""

    #: False when the tile step must not emit G4 (split-dQ main kernels).
    emits_g4: bool = True

    def begin_head(self, b: IRBuilder, h: Value) -> None:
        raise NotImplementedError

    def consume(
        self,
        b: IRBuilder,
        dq_frag: Sequence[DqSlot],
        rows: Value,
        h: Value,
        *,
        clamp: bool = True,
    ) -> None:
        raise NotImplementedError

    def finish(self, b: IRBuilder) -> None:
        raise NotImplementedError


def _key(x: IntOrValue):
    """Hashable identity of a static int or an SSA value."""
    return x if isinstance(x, int) else ("v", id(x))


class _RowCache:
    """Per-step cache of per-row values keyed by the row expression."""

    def __init__(self) -> None:
        self._d: dict = {}

    def get(self, key, make):
        k = _key(key)
        if k not in self._d:
            self._d[k] = make()
        return self._d[k]


class AtomicWorkspaceSink(DqSink):
    """fp32 atomics into the dQ workspace (branch-free, i64 head rebase).

    ``ws`` addresses the dQ workspace of the CTA's sequence (``width = D``);
    ``seq`` supplies ``last = max(len_q - 1, 0)``.
    """

    def __init__(
        self,
        b: IRBuilder,
        ws: WorkspaceAddr,
        *,
        seq: SeqInfo,
        scale: Value | None = None,
    ) -> None:
        self.ws = ws
        self.seq = seq
        self.scale = scale
        self._head: Value | None = None
        self._slab = None

    def begin_head(self, b: IRBuilder, h: Value) -> None:
        """Rebase to head ``h`` (head-major; token-major rebases per step).

        A caller that emits several step variants calls this once where it
        dominates all of them, so every variant reuses the same rebase.
        """
        self._head = h
        self._slab = self.ws.slab(h) if self.ws.layout == "head_major" else None

    def consume(
        self,
        b: IRBuilder,
        dq_frag: Sequence[DqSlot],
        rows: Value,
        h: Value,
        *,
        clamp: bool = True,
    ) -> None:
        last = self.seq.last if clamp else None
        if self.ws.layout == "token_major":
            # Rows of one head are H * D apart: rebase in i64 to the step's
            # first target row so the in-slab index stays below k_m0 * H * D.
            base = rows if last is None else b.smin(rows, last)
            slab = self.ws.slab(h, base)
        else:
            if self._slab is None or self._head is not h:
                self.begin_head(b, h)
            base = None
            slab = self._slab
        cache = _RowCache()
        for slot in dq_frag:

            def target(slot=slot):
                q = b.add(rows, as_i32(b, slot.row))
                if last is not None:
                    q = b.smin(q, last)
                return q if base is None else b.sub(q, base)

            q = cache.get(slot.row, target)
            v = slot.value
            if self.scale is not None:
                v = b.fmul(v, self.scale)
            slab.atomic_add(q, slot.col, v)

    def finish(self, b: IRBuilder) -> None:
        return None


class PackedAtomicWorkspaceSink(AtomicWorkspaceSink):
    """:class:`AtomicWorkspaceSink` for q steps that pack query heads.

    ``consume`` takes a :class:`._attention_bwd_addr.PackedRows` as ``rows``
    (``h`` is unused): each distinct slot row is located to its ``(head,
    row)``, rebased in i64 to that head (and, token-major, to the clamped
    row), and the value of a dead packed row is replaced by an exact 0 before
    the atomic.  Rows are clamped to ``max(len_q - 1, 0)``; the ``ws_rows``
    guard and the optional scale are those of the base sink.  Branch-free.
    """

    def begin_head(self, b: IRBuilder, h: Value) -> None:
        self._head = h  # rebases are per packed row (see consume)

    def consume(
        self,
        b: IRBuilder,
        dq_frag: Sequence[DqSlot],
        rows,
        h: Value | None = None,
        *,
        clamp: bool = True,
    ) -> None:
        last = self.seq.last if clamp else None
        zero = b.const_f32(0.0)
        # Group the slots by row so each row's 64-bit rebase is short-lived.
        groups: dict = {}
        for slot in dq_frag:
            groups.setdefault(_key(slot.row), []).append(slot)
        for slots in groups.values():
            head, q, ok = rows.locate(b, slots[0].row)
            if last is not None:
                q = b.smin(q, last)
            if self.ws.layout == "token_major":
                slab, q = self.ws.slab(head, q), b.const_i32(0)
            else:
                slab = self.ws.slab(head)
            for slot in slots:
                v = slot.value
                if self.scale is not None:
                    v = b.fmul(v, self.scale)
                slab.atomic_add(q, slot.col, b.select(ok, v, zero))


class NullSink(DqSink):
    """No dQ from the main kernel (split-dQ mode): G4 is not emitted."""

    emits_g4 = False

    def begin_head(self, b: IRBuilder, h: Value) -> None:
        return None

    def consume(self, b, dq_frag, rows, h, *, clamp: bool = True) -> None:
        raise RuntimeError("NullSink receives no dQ fragments (G4 is not emitted)")

    def finish(self, b: IRBuilder) -> None:
        return None


class RegisterAccumulateSink(DqSink):
    """dQ carried in registers, stored once in the output dtype.

    Every ``consume`` must hand over the same fragment shape (same slot order,
    rows and columns); values are summed slot by slot.  ``state`` /
    ``set_state`` expose the accumulators so a loop can carry them.
    ``finish`` stores ``scale * acc`` to ``out`` at rows ``q0 + row`` that are
    ``< len_q`` (one predicated store per row and column).
    """

    def __init__(
        self,
        b: IRBuilder,
        out: StridedTensorAddr,
        *,
        scale: Value | None = None,
    ) -> None:
        self.out = out
        self.scale = scale
        self._layout: tuple | None = None
        self._acc: list = []
        self._q0: Value | None = None
        self._head: Value | None = None

    def begin_head(self, b: IRBuilder, h: Value) -> None:
        if self._head is not None and self._head is not h:
            raise ValueError("RegisterAccumulateSink holds one head per CTA")
        self._head = h

    def consume(
        self,
        b: IRBuilder,
        dq_frag: Sequence[DqSlot],
        rows: Value,
        h: Value,
        *,
        clamp: bool = True,
    ) -> None:
        self.begin_head(b, h)
        layout = tuple((_key(s.row), _key(s.col)) for s in dq_frag)
        if self._layout is None:
            self._layout = layout
            self._slots = [(s.row, s.col) for s in dq_frag]
            self._acc = [s.value for s in dq_frag]
            self._q0 = rows
            return
        if layout != self._layout:
            raise ValueError("fragment shape changed between consume calls")
        if self._q0 is not rows:
            raise ValueError("RegisterAccumulateSink holds one q block per CTA")
        self._acc = [b.fadd(a, s.value) for a, s in zip(self._acc, dq_frag)]

    def state(self) -> list:
        return list(self._acc)

    def set_state(self, values: Sequence[Value]) -> None:
        if len(values) != len(self._acc):
            raise ValueError("state length mismatch")
        self._acc = list(values)

    def finish(self, b: IRBuilder) -> None:
        if self._layout is None:
            return
        slab: RowSlab = self.out.slab(self._head, clamp=False)
        length = self.out.seq.length
        by_row: dict = {}
        for (row, col), acc in zip(self._slots, self._acc):
            by_row.setdefault(_key(row), (row, []))[1].append((col, acc))
        for row, cols in by_row.values():
            q = b.add(self._q0, as_i32(b, row))
            with b.scf_if(b.cmp_lt(q, length)):
                for col, acc in cols:
                    v = acc if self.scale is None else b.fmul(acc, self.scale)
                    slab.store(q, col, b.cast_f32_to(v, self.out.dtype))


class SplitSliceSink(DqSink):
    """Per-split fp32 dQ slices with a fixed-order reduce (not built).

    Recorded for the deterministic and decode-backward designs; its workspace
    is ``split_slice_workspace_bytes``.  Constructing one raises.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "SplitSliceSink is a recorded design and is not built"
        )


def split_slice_workspace_bytes(
    *, n_split: int, batch: int, s_q: int, h_q: int, head_size: int
) -> int:
    """``A(4 * n_split * B * s_q * h_q * D)`` with 256-byte rounding."""
    for name, v in (
        ("n_split", n_split),
        ("batch", batch),
        ("s_q", s_q),
        ("h_q", h_q),
        ("head_size", head_size),
    ):
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    raw = 4 * n_split * batch * s_q * h_q * head_size
    return (raw + 255) // 256 * 256
