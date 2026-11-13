# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opt-in extension points for the MFMA / WMMA attention forward inner body.

Everything here is default-off: the inner body takes one keyword-only
``ext: Optional[AttnFwdExt] = None`` and emits nothing extra when it is None.
Hooks are Python callables that run at build time and emit IR through the
builder they are handed; they are not part of any kernel name or cache key
(the kernel spec that creates them is).

This module imports only ``core.ir`` so the attention bodies can import it
without creating an import cycle.

Contracts
---------
* ``score_hook`` runs after the ``scale_log2`` multiply and the legacy
  ``extra_score_transform`` hook, before the key-tail select and the body's own
  mask.  Scores are in the log2 domain: an additive natural-log bias must be
  multiplied by ``LOG2E`` inside the hook.  Memory loads inside the hook must be
  guarded by ``q_valid`` / ``k_valid`` (or use clamped addresses).
* A cell masked by a hook must be filled with a true ``-inf`` (or a value
  ``<= -1e30``).  Under ``guard_seqlen_k`` the key-tail fill is ``-inf``.
* Row validity for the output store is ``m_log2 > ROW_VALID_SENTINEL`` (and
  ``q_row < seqlen_q`` when set), independent of the running sum ``l``.  The
  epilogue hook may return an i1 that is ANDed into that validity, so a caller
  can derive validity from its mask/bounds instead.  Invalid rows store exactly
  0.0 (a select, never a multiply).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..core.ir import IRBuilder, Value

__all__ = [
    "AttnFwdExt",
    "LN2",
    "LOG2E",
    "ROW_VALID_SENTINEL",
    "RowEpilogue",
    "ScoreCoord",
    "emit_rows_epilogue",
    "last_index",
    "natural_lse_from_log2_stats",
    "score_stage",
]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
# The running max starts at -1e30 and only rises when a finite kept score is
# seen, so anything above this threshold means "the row has a kept key".
ROW_VALID_SENTINEL = -1e29


@dataclass(frozen=True, eq=False)
class ScoreCoord:
    """Coordinates of one score cell, handed to ``AttnFwdExt.score_hook``."""

    q_row: Value  # i32 query row within the sequence
    k_col: Value  # i32 key column within the sequence
    kt: Value  # raw K-loop induction value
    slot: int  # accumulator slot inside the lane (0..3 MFMA, 0..7 WMMA)
    q_valid: Optional[Value]  # i1: q_row < seqlen_q   (None unless seqlen_q set)
    k_valid: Optional[Value]  # i1: k_col < seqlen_k   (None unless guard_seqlen_k)


@dataclass(frozen=True, eq=False)
class RowEpilogue:
    """One query row slot after the K loop, handed to ``epilogue_hook``."""

    slot: int
    q_row: Value  # i32 query row within the sequence
    row_valid: Value  # i1 base validity, before the hook's returned predicate
    in_range: Optional[Value]  # i1: q_row < seqlen_q (None unless seqlen_q set)
    m_log2: Value  # f32 running max, log2 domain (same on every lane of the row)
    l: Value  # f32 running sum of exp2(s - m)  # noqa: E741
    is_row_leader: Value  # i1: exactly one lane per row is true (store once)


@dataclass(frozen=True, eq=False)
class AttnFwdExt:
    """Default-off extension carrier for the attention forward inner bodies.

    * ``score_hook(b, s, ScoreCoord) -> s'``: per-cell score transform.
    * ``seqlen_q``: i32 runtime query length; clamps Q addresses and predicates
      the O store so rows ``>= seqlen_q`` are never read or written.
    * ``guard_seqlen_k``: ceil K trip count, clamp K/V row addresses and mask
      key columns ``>= seqlen_k`` with ``-inf``.
    * ``epilogue_hook(b, RowEpilogue) -> Optional[i1]``: runs once per row slot
      after the K loop; the returned i1 is ANDed into the O-store validity.
    """

    score_hook: Optional[Callable[[IRBuilder, Value, ScoreCoord], Value]] = None
    seqlen_q: Optional[Value] = None
    guard_seqlen_k: bool = False
    epilogue_hook: Optional[Callable[[IRBuilder, RowEpilogue], Optional[Value]]] = None

    @property
    def wants_epilogue(self) -> bool:
        return self.seqlen_q is not None or self.epilogue_hook is not None


def last_index(b: IRBuilder, length: Value) -> Value:
    """``max(length - 1, 0)``: the clamp bound for an index into ``length`` rows."""
    return b.smax(b.sub(length, b.const_i32(1)), b.const_i32(0))


def natural_lse_from_log2_stats(b: IRBuilder, m_log2: Value, row_sum: Value) -> Value:
    """Natural-log LSE ``LN2 * (m_log2 + log2(row_sum))`` from the log2 stats.

    The result is meaningless for invalid rows; the caller selects ``-inf``.
    """
    return b.fmul(b.fadd(m_log2, b.log2(row_sum)), b.const_f32(LN2))


def score_stage(
    b: IRBuilder,
    ext: AttnFwdExt,
    s: Value,
    *,
    q_row: Value,
    k_col: Value,
    kt: Value,
    slot: int,
    seqlen_k: Value,
) -> Value:
    """Run the score hook and the key-tail mask for one score cell."""
    q_valid = b.cmp_lt(q_row, ext.seqlen_q) if ext.seqlen_q is not None else None
    k_valid = b.cmp_lt(k_col, seqlen_k) if ext.guard_seqlen_k else None
    if ext.score_hook is not None:
        s = ext.score_hook(b, s, ScoreCoord(q_row, k_col, kt, slot, q_valid, k_valid))
    if k_valid is not None:
        s = b.select(k_valid, s, b.const_f32(float("-inf")))
    return s


def emit_rows_epilogue(
    b: IRBuilder,
    ext: AttnFwdExt,
    *,
    n_slots: int,
    n_tiles: int,
    row_rel_fn,
    q_pos_base: Value,
    ms,
    ls,
    accs,
    is_row_leader: Value,
    out_addr_fn,
    O: Value,  # noqa: E741
    dtype_ir,
    v_scale: Optional[Value],
) -> None:
    """Flagged epilogue: hook call, validity, guarded O store per row slot."""
    zero_f = b.const_f32(0.0)
    sentinel_gate = b.const_f32(ROW_VALID_SENTINEL)
    for r in range(n_slots):
        row_rel = row_rel_fn(r)
        q_row = b.add(q_pos_base, row_rel)
        in_range = b.cmp_lt(q_row, ext.seqlen_q) if ext.seqlen_q is not None else None
        valid = b.fcmp("ogt", ms[r], sentinel_gate)
        if in_range is not None:
            valid = b.land(valid, in_range)
        valid_store = valid
        if ext.epilogue_hook is not None:
            ret = ext.epilogue_hook(
                b,
                RowEpilogue(r, q_row, valid, in_range, ms[r], ls[r], is_row_leader),
            )
            if ret is not None:
                valid_store = b.land(valid, ret)
        inv_l = b.rcp(ls[r])

        def store_row(r=r, row_rel=row_rel, inv_l=inv_l, valid_store=valid_store):
            for n in range(n_tiles):
                v = b.fmul(b.vec_extract(accs[n], r), inv_l)
                if v_scale is not None:
                    v = b.fmul(v, v_scale)
                v = b.select(valid_store, v, zero_f)
                b.global_store(
                    O, out_addr_fn(r, n, row_rel), b.cast_f32_to(v, dtype_ir), align=2
                )

        if in_range is not None:
            with b.scf_if(in_range):
                store_row()
        else:
            store_row()
