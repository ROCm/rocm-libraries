# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The checked, testable RDNA tuned-table entry rule (selection logic only).

The RDNA targets (gfx1151, gfx1201) have no entry that beats PyTorch's own
backward (the CDNA confirm rule), so an RDNA entry is instead the best rocKE
configuration of its key (see "RDNA entries" in :mod:`.tuned_tables`). This
module is the checked implementation of that selection rule, run over a
*judge document*: one record per (tuning key, finalist, dense census cell)
produced by timing the shipped plan and the sweep's finalists on a device
(``attention_bwd_bench.py``, ``--aa-control all``), outside the source tree.
Every field of :class:`CellVerdict` is MEASURED on the device, not modelled:

* ``correct``: the arm completed and passed the bench's pruning-set gate;
* ``aa_clean``: the finalist's own ``#aa`` twin was not judged different from
  it either way (the paired A/A control);
* ``verdict``: the bench's paired verdict of the finalist against the
  shipped plan on this cell (``"pass"``, ``"fail"`` or ``"ambiguous"``);
* ``ratio_vs_shipped``: the finalist's time divided by the shipped plan's
  time on this cell (the bench's own measurement; this module never reads or
  writes an absolute time).

:func:`select_rdna_general_entries` is the only tolerance in the rule:
``tie_eps`` lets an ``"ambiguous"`` verdict count as not-slower when the
measured ratio is within ``tie_eps`` of 1 (the same tolerance the bench's own
gate uses to call two timings indistinguishable from noise on the device).
Shrinking ``tie_eps`` rejects more entries (fewer false positives, more false
negatives against noise); growing it does the reverse. A finalist that is
truly ``tie_eps`` slower than shipped on every cell of a key is therefore a
false positive this rule accepts by design, not a bug: ``tie_eps`` is
calibrated per device from the sweep's own A/A spread (see
:mod:`.tuned_tables` "Regeneration"), not a repo constant, so it is not
shipped here.

Regeneration (outside the source tree, no timing data in the repo):

1. ``attention_bwd_bench.py`` times the shipped plan and the sweep's
   finalists (``--aa-control all``) on the slice's dense census cells and
   writes one judge document per arch (``{"slices": {"<d>": {"keys":
   {<key>: {"cells", "finalists": {<finalist>: {"ok", "geomean_vs_shipped",
   "why"}}}}}}}``);
2. :func:`load_cell_verdicts` turns it into ``CellVerdict`` records;
3. :func:`select_rdna_general_entries` applies the rule per key;
4. the winning finalist's knobs (kept with the sweep records, not here) are
   written as ``{"entries": [{"key", "knobs"}]}`` and checked by
   ``<gfx>_tuning.general_entries`` -- the checked mapping that replaces
   ``GENERAL_TUNED_TABLE``;
5. the same per-key verdicts (finalist id, cells checked, ``correct`` --
   no ratio, the repo carries no absolute or relative number outside a
   qualifying result document) are written to ``rdna_entries_manifest.json``,
   checked against the shipped table by
   ``tests.dispatch.attention_bwd.test_tuned_tables``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Mapping

__all__ = [
    "CellVerdict",
    "RdnaEntryVerdict",
    "load_cell_verdicts",
    "select_rdna_general_entries",
]


@dataclasses.dataclass(frozen=True)
class CellVerdict:
    """One finalist's measured outcome on one dense census cell of a tuning
    key (one record of a judge document); see the module docstring for what
    each field measures."""

    key: tuple
    finalist: str
    correct: bool
    aa_clean: bool
    verdict: str  # "pass" | "fail" | "ambiguous"
    ratio_vs_shipped: float
    cells_checked: int = 1


@dataclasses.dataclass(frozen=True)
class RdnaEntryVerdict:
    """The qualifying verdict of the tuning key's shipped entry: the finalist
    that won it, how many cells it was checked on, and its geometric-mean
    time ratio against the shipped plan (the quantity the rule compares to
    1; not written to the manifest -- see the module docstring)."""

    key: tuple
    finalist: str
    cells_checked: int
    geomean_ratio_vs_shipped: float


def _is_not_slower(v: CellVerdict, tie_eps: float) -> bool:
    if v.verdict == "pass":
        return True
    if v.verdict == "ambiguous":
        return v.ratio_vs_shipped <= 1.0 + tie_eps
    return False


def select_rdna_general_entries(
    cells: Iterable[CellVerdict], *, tie_eps: float
) -> dict[tuple, RdnaEntryVerdict]:
    """The RDNA entry rule: per tuning key, among the finalists that are
    correct, A/A-clean and not slower than the shipped plan (paired verdict
    ``pass``, or ``ambiguous`` within ``tie_eps``) on *every* cell of the key,
    the one with the lowest geometric-mean ratio against shipped wins -- only
    when that geomean is below 1 (a finalist equal to or slower than shipped
    overall changes nothing). A key with no qualifying finalist is absent
    from the result and keeps the defaults."""
    by_key: dict[tuple, dict[str, list[CellVerdict]]] = {}
    for v in cells:
        by_key.setdefault(v.key, {}).setdefault(v.finalist, []).append(v)

    out: dict[tuple, RdnaEntryVerdict] = {}
    for key, by_finalist in by_key.items():
        qualifying = []
        for finalist, vs in by_finalist.items():
            if not vs:
                continue
            if not all(
                v.correct and v.aa_clean and _is_not_slower(v, tie_eps) for v in vs
            ):
                continue
            geomean = math.exp(sum(math.log(v.ratio_vs_shipped) for v in vs) / len(vs))
            cells_checked = sum(v.cells_checked for v in vs)
            qualifying.append((geomean, finalist, cells_checked))
        if not qualifying:
            continue
        geomean, finalist, n = min(qualifying, key=lambda t: t[0])
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
    """``[CellVerdict, ...]`` from one arch's judge document (the module
    docstring's schema): a judge document already aggregates a finalist's
    per-cell checks into one ``{"ok", "geomean_vs_shipped", "why"}`` per
    ``(key, finalist)``, so each becomes one record carrying the key's cell
    count (``why`` empty: both correct and A/A-clean; non-empty: the bench's
    rejection reasons, this loader folds them into ``correct=False``)."""
    import ast

    out = []
    for slice_doc in doc["slices"].values():
        for key_str, rec in slice_doc["keys"].items():
            key = (
                tuple(ast.literal_eval(key_str))
                if isinstance(key_str, str)
                else tuple(key_str)
            )
            for finalist, fin in rec["finalists"].items():
                out.append(
                    CellVerdict(
                        key=key,
                        finalist=finalist,
                        correct=bool(fin["ok"]),
                        aa_clean=not fin.get("why"),
                        verdict="pass" if fin["ok"] else "fail",
                        ratio_vs_shipped=float(fin["geomean_vs_shipped"]),
                        cells_checked=int(rec["cells"]),
                    )
                )
    return out
