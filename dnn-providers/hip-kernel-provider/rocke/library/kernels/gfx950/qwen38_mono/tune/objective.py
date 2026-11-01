# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a search minimizes, and how a result is ranked against the headline
targets (PLAN user decision 7).

A search always minimizes a *time* at one level:

- ``stage``: one stage's time in the stage rig (isolated or in situ);
- ``layer``: one layer launch (in-situ graph replays, DESIGN 9.1);
- ``round``: a whole verify round (K=3) or decode step (K=0);
- ``e2e``: served tokens per second per request from the ATOM-truemono serve
  harness, turned into a time per token (``1 / tok/s``) so lower is better
  everywhere.

A result's *ratio* is its time over the baseline's at the same level. To rank
levers across levels, :meth:`Objective.gap_closed` converts a ratio into the
share of the gap to the headline target it closes: a stage that is ``share``
of the cell's round (hybrid units) and runs at ``ratio`` of its baseline
saves ``(1 - ratio) * share`` of the round; the cell's target asks for
``1 - target`` (K=3 CONC 1: round <= 0.83 of the hybrid's; K=0 CONC 1:
per-token time <= 0.78). Shares come from the gap table / budgets of DESIGN
(``shares`` mapping), never from this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["HEADLINE", "LEVELS", "Objective", "rank_by_gap"]

LEVELS = ("stage", "layer", "round", "e2e")

#: headline targets as time ratios vs the pinned hybrid (PLAN decision 7)
HEADLINE = {
    "K3C1": 0.83,  # verify round, K=3 CONC 1 (>= 1000 tok/s per request)
    "K0C1": 0.78,  # per-token time, K=0 CONC 1 (>= 550 tok/s)
}


@dataclass(frozen=True)
class Objective:
    """``level`` (:data:`LEVELS`), ``cell`` (a :data:`HEADLINE` key or any
    cell with an explicit ``target``), ``share``: the fraction of the cell's
    round time the measured unit is (1.0 for ``round`` / ``e2e``)."""

    level: str
    cell: str = "K3C1"
    share: float = 1.0
    target: float | None = None
    name: str = ""

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"Objective level {self.level!r}, expected one of {LEVELS}")
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"Objective share {self.share}, expected 0 < share <= 1")
        if self.level in ("round", "e2e") and self.share != 1.0:
            raise ValueError("a round / e2e objective is the whole round (share 1)")

    @property
    def target_ratio(self) -> float:
        t = self.target if self.target is not None else HEADLINE.get(self.cell)
        if t is None:
            raise KeyError(f"no headline target for cell {self.cell!r}; pass target=")
        return float(t)

    @staticmethod
    def time_of(value: float, *, tokens_per_s: bool = False) -> float:
        """The minimized time of a measured value (``1 / tok/s`` for e2e)."""
        if tokens_per_s:
            return math.inf if value <= 0 else 1.0 / value
        return value

    def round_ratio(self, ratio: float) -> float:
        """The cell's round ratio implied by a unit ratio (other work fixed)."""
        return 1.0 - (1.0 - ratio) * self.share

    def gap_closed(self, ratio: float) -> float:
        """Share of the gap to the headline target closed by ``ratio`` (1.0:
        the target alone; negative: a regression)."""
        gap = 1.0 - self.target_ratio
        return (1.0 - ratio) * self.share / gap

    def describe(self) -> dict:
        return {"level": self.level, "cell": self.cell, "share": self.share,
                "target_ratio": self.target_ratio, "name": self.name}  # fmt: skip


def rank_by_gap(results: Sequence[Mapping], *, objectives: Mapping[str, Objective]) -> list[dict]:
    """Rank lever results (``{"name", "objective" (a key of objectives),
    "ratio", ...}``) by the share of the gap they close, best first."""
    out = []
    for r in results:
        obj = objectives[r["objective"]]
        d = dict(r)
        d["round_ratio"] = obj.round_ratio(r["ratio"])
        d["gap_closed"] = obj.gap_closed(r["ratio"])
        d["cell"] = obj.cell
        out.append(d)
    out.sort(key=lambda d: -d["gap_closed"])
    return out
