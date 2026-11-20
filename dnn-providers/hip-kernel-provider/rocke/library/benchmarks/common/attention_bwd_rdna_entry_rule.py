# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The RDNA tuned-table entry rule (offline selection logic, no runtime use).

On the RDNA targets (gfx1151, gfx1201) few or no keys have a configuration
that beats PyTorch's own backward on every cell (the CDNA confirm rule), so an
RDNA entry is instead the best rocKE configuration of its key. :func:`select_rdna_general_entries` is the one
statement of that rule; ``dispatch/attention_bwd/tuned_tables.py`` ("RDNA
entries") and ``rdna_general_tuning_case_study.md`` point here.

It runs over a *judge document*: per (tuning key, finalist), the outcome of
timing the shipped plan and the sweep's ``runtime`` finalists on the key's
dense census cells with :mod:`.attention_bwd_bench` (``--aa-control all``).
Every field of :class:`CellVerdict` is MEASURED on the device, not modelled:

* ``correct``: the arm completed and passed the bench's pruning-set gate;
* ``aa_clean``: the finalist's own ``#aa`` twin was not judged different from
  it either way (the paired A/A control);
* ``verdict``: :func:`.attention_bwd_bench.paired_verdict` of the finalist
  against the shipped plan after its reruns -- ``"pass"`` or ``"fail"`` (the
  bench resolves an ambiguous comparison by its own ``TIE_EPS`` before this
  rule sees it, so the rule adds no tolerance of its own);
* ``ratio_vs_shipped``: the finalist's time over the shipped plan's on the
  cell (the bench's measurement; this module reads or writes no absolute
  time).

Gate behaviour: the rule's only statistic is the paired verdict, so its
false-positive rate (a finalist that is in truth slower than shipped yet
judged ``pass``) is the bench's: a ratio within ``TIE_EPS`` of 1 after
``RERUN_MAX`` reruns counts as not slower. The rule adds a strict
``geomean < 1`` bar on top, which a tie alone never clears. No tolerance here
tightens as the measurement improves: ``TIE_EPS`` is a fixed per-device
constant of the bench.

The steps that produce a judge document and turn the rule's output into a
shipped table are in ``tuned_tables.py`` ("Regeneration").
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import Iterable, Mapping

__all__ = [
    "CellVerdict",
    "RdnaEntryVerdict",
    "load_cell_verdicts",
    "select_rdna_general_entries",
]

_VERDICTS = ("pass", "fail")


@dataclasses.dataclass(frozen=True)
class CellVerdict:
    """One finalist's measured outcome on one dense census cell of a tuning
    key (or, from :func:`load_cell_verdicts`, on all ``cells_checked`` cells
    of it); see the module docstring for what each field measures."""

    key: tuple
    finalist: str
    correct: bool
    aa_clean: bool
    verdict: str  # "pass" | "fail"
    ratio_vs_shipped: float
    cells_checked: int = 1

    def __post_init__(self):
        if self.verdict not in _VERDICTS:
            raise ValueError(
                f"verdict: {self.verdict!r} is not one of {_VERDICTS} "
                "(the bench resolves ambiguous verdicts before this rule)"
            )


@dataclasses.dataclass(frozen=True)
class RdnaEntryVerdict:
    """The qualifying verdict of the tuning key's shipped entry: the finalist
    that won it, how many cells it was checked on, and its geometric-mean
    time ratio against the shipped plan (the quantity the rule compares to
    1; not written to the manifest)."""

    key: tuple
    finalist: str
    cells_checked: int
    geomean_ratio_vs_shipped: float


def select_rdna_general_entries(
    cells: Iterable[CellVerdict],
) -> dict[tuple, RdnaEntryVerdict]:
    """The RDNA entry rule. A ``runtime`` finalist qualifies for tuning key K
    when, on *every* dense census cell of K, it is

    1. correct (its arm completes and passes the pruning-set gate),
    2. A/A-clean (its ``#aa`` twin is not judged different from it either
       way), and
    3. not slower than rocKE's shipped plan (paired verdict ``pass``).

    Among the qualifying finalists of K, the one with the lowest geometric-
    mean time ratio against the shipped plan is K's entry, and only when that
    geomean is below 1. A key with no qualifying finalist, or whose best
    geomean is 1 or more, is absent from the result and keeps the defaults.
    """
    by_key: dict[tuple, dict[str, list[CellVerdict]]] = {}
    for v in cells:
        by_key.setdefault(v.key, {}).setdefault(v.finalist, []).append(v)

    out: dict[tuple, RdnaEntryVerdict] = {}
    for key, by_finalist in by_key.items():
        qualifying = []
        for finalist, vs in by_finalist.items():
            if not all(v.correct and v.aa_clean and v.verdict == "pass" for v in vs):
                continue
            geomean = math.exp(sum(math.log(v.ratio_vs_shipped) for v in vs) / len(vs))
            cells_checked = sum(v.cells_checked for v in vs)
            qualifying.append((geomean, finalist, cells_checked))
        if not qualifying:
            continue
        geomean, finalist, n = min(qualifying, key=lambda t: (t[0], t[1]))
        if geomean >= 1.0:
            continue
        out[key] = RdnaEntryVerdict(
            key=key,
            finalist=finalist,
            cells_checked=n,
            geomean_ratio_vs_shipped=geomean,
        )
    return out


def load_cell_verdicts(doc: Mapping) -> list[CellVerdict]:
    """``[CellVerdict, ...]`` from one arch's judge document,
    ``{"slices": {"<d>": {"keys": {<key>: {"cells", "finalists":
    {<finalist>: {"ok", "geomean_vs_shipped", "why"}}}}}}}``.

    A judge document already aggregates a finalist's per-cell checks into one
    record per ``(key, finalist)``: ``ok`` is true when every cell was
    correct and ``pass``; ``why`` lists the cells that failed the A/A control
    or a check. Each record becomes one :class:`CellVerdict` carrying the
    key's cell count, with ``aa_clean`` false whenever ``why`` is non-empty
    (whatever ``ok`` says). A finalist with no timed cell (``geomean`` null)
    gets an infinite ratio.
    """
    out = []
    for slice_doc in doc["slices"].values():
        for key_str, rec in slice_doc["keys"].items():
            key = (
                tuple(ast.literal_eval(key_str))
                if isinstance(key_str, str)
                else tuple(key_str)
            )
            for finalist, fin in rec["finalists"].items():
                gm = fin.get("geomean_vs_shipped")
                out.append(
                    CellVerdict(
                        key=key,
                        finalist=finalist,
                        correct=bool(fin["ok"]),
                        aa_clean=not fin.get("why"),
                        verdict="pass" if fin["ok"] else "fail",
                        ratio_vs_shipped=math.inf if gm is None else float(gm),
                        cells_checked=int(rec["cells"]),
                    )
                )
    return out
