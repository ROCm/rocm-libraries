#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Propose a static ``auto`` default per gfx950 GDN/KDA decode case from
``decode_table_sweep.py`` rows. Advisory only: it changes nothing; a person
reviews the proposal and ships ``AUTO_DEFAULTS`` (``dispatch/gdn/axes.py``).

A *case* is one ``AUTO_DEFAULTS`` row: gate kind x fused x state width (an f16
state reads the bf16 row). Its *shapes* are the head geometries x batches the
rows cover. Per case the report prints:

* per shape: the fastest configuration, and ``auto`` and the proposal against
  it (time / the shape's fastest);
* ``auto``'s score (the geomean of those ratios) and worst shape;
* the proposal by the rule ``AUTO_DEFAULTS`` was chosen with. Candidates are
  the configurations the sweep walked and timed at every shape of the case:
  the walk at its ``--sweep-level`` (default ``production``: ``waves_per_eu`` 0
  and 4 only; ``full``: a random sample per tile), not every configuration
  there is. The report prints that walk beside each proposal. A
  configuration's score is the geomean over the shapes of its time / the
  shape's fastest. Take the
  lowest score; among the configurations within ``--tie`` of it, the lower
  worst-shape ratio, worst shapes within ``--tie`` counting as tied; then fewer
  non-default knobs; then the lower score. It names the step that decided, the
  gain against ``auto`` (proposal score / ``auto`` score) and an
  ``AUTO_DEFAULTS``-shaped row;
* the ``--top`` candidates by score; with the pick and every shape's fastest
  configuration, a ``--tuning-ids`` list to re-time, carrying the stage-1 walk
  flags (``--sweep-level`` / ``--tuning-sample`` / ``--seed`` /
  ``--bases-only``) the ids came
  from. Each shape's fastest is on the list so the stage-2 ratios are against
  that shape's real best, not the best of the shortlist.

A shape of the case with no timed row (every configuration failed or errored,
or the dispatcher refused it), or where ``auto`` failed the gate, is listed
with its statuses and the case exits 1: a case is never scored over fewer
shapes than the rows cover without saying so.

The report exits 1 with a ``WITHHELD`` line and no case at all (so no proposal
of any case) when any row of the scored stage was not timed in cold memory
(``rotate_mb`` below the cold target the sweep recorded) or the scored rows mix
``rotate_mb`` values. A case whose rows come from different walks withholds
that case's proposal only: it exits 1 and the other cases still print. Exit 0
means every case was scored.

It prints the rows' toolchain (ROCm, LLVM, torch, lowering backend, LLVM
flavor) beside the sha and device: one line per distinct set, so rows timed
with different toolchains show it. A mismatch withholds nothing.

``--stage 1`` (default) scores the stage-1 screen: every configuration, timed
once each. ``--stage 2`` scores stage-2 rows, where only configurations re-timed
at every shape are candidates: re-time the stage-1 shortlist at every shape
first (the printed ``re-time:`` flags, for ``decode_table_sweep.py``).

    rocke-gdn-decode-defaults-report 's1_*.json'
    rocke-gdn-decode-defaults-report --stage 2 'retime_*.json'
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import math
import statistics
import sys
from typing import Iterable

from benchmarks.gfx950.gdn.decode_table_sweep import (
    OK,
    WALK_FIELDS,
    Shape,
    case_name,
    is_cold,
    load_rows,
    shape_of,
    walk_flags,
    walk_of,
)
from dispatch.gdn.axes import _STREAMING_STATE as _STREAMING
from dispatch.gdn.cli_args import usage_error_text

# The report's default relative tie band (1%), set with ``--tie``. It is a tool
# default, not a measured noise floor, and the report does not prove it
# reproduces the shipped AUTO_DEFAULTS rows (the kda fused f32 row is a
# documented override of the rule; see dispatch/gdn/axes.py).
TIE = 0.01


@dc.dataclass(frozen=True)
class Candidate:
    """One configuration scored over a case's shapes."""

    tuning_id: str
    tile: tuple
    knobs: dict
    score: float
    worst: float
    worst_shape: Shape
    ratios: dict

    @property
    def nknobs(self) -> int:
        """Non-default knobs: a canonical knob set lists only those."""
        return len(self.knobs)


def case_label(case: tuple) -> str:
    return case_name(case).replace("_", " ")


def fastest_id(times: dict) -> str:
    """The ``tuning_id`` with the least time in ``{tuning_id: us}``; a tie goes
    to the smaller ``tuning_id``, whatever the order of the dict."""
    return min(times, key=lambda tid: (times[tid], tid))


def geomean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(statistics.mean(math.log(v) for v in values))


def collect(rows: Iterable[dict], stage: int) -> dict:
    """``{case: {"times": {shape: {tuning_id: us}}, "auto": {shape: tuning_id},
    "configs": {tuning_id: row}, "shapes": {shape}, "bad": {shape: [row]},
    "walks": {(sweep_level, tuning_sample, seed, bases_only)}}`` from the rows of
    ``stage``. ``times`` holds the timed rows; ``shapes`` every shape seen,
    timed or not; ``bad`` the rows that failed, raised or were not timed.
    Repeated measurements of one configuration at one shape merge by their
    median."""
    times: dict = {}
    cases: dict = {}
    for row in rows:
        if row["stage"] != stage:
            continue
        shape = shape_of(row)
        case = cases.setdefault(
            shape.auto_key,
            {
                "times": {},
                "auto": {},
                "configs": {},
                "shapes": set(),
                "bad": {},
                "walks": set(),
            },
        )
        case["shapes"].add(shape)
        case["walks"].add(walk_of(row))
        if row["status"] != OK:
            case["bad"].setdefault(shape, []).append(row)
            continue
        times.setdefault(shape, {}).setdefault(row["tuning_id"], []).append(row["us"])
        case["times"].setdefault(shape, {})
        case["configs"].setdefault(row["tuning_id"], row)
        if row["auto"]:
            case["auto"][shape] = row["tuning_id"]
    for case in cases.values():
        for shape in case["times"]:
            case["times"][shape] = {
                tid: statistics.median(us) for tid, us in times[shape].items()
            }
    return cases


def unscored(case: dict) -> dict:
    """``{shape: [bad row]}`` of the shapes the case cannot be scored on as
    measured: no timed row at all, or an ``auto`` row that failed (``auto``
    then has no time to score there). Empty when every shape is sound."""
    out = {}
    for shape in sorted(case["shapes"]):
        bad = case["bad"].get(shape, [])
        auto_failed = shape not in case["auto"] and any(r["auto"] for r in bad)
        if shape not in case["times"] or auto_failed:
            out[shape] = bad
    return out


def score_candidates(case: dict) -> list:
    """Every configuration timed at every shape, scored (:class:`Candidate`)."""
    times = case["times"]
    best = {shape: min(t.values()) for shape, t in times.items()}
    common = set.intersection(*(set(t) for t in times.values())) if times else set()
    out = []
    for tid in sorted(common):
        ratios = {shape: times[shape][tid] / best[shape] for shape in times}
        worst_shape = max(sorted(ratios), key=ratios.get)
        row = case["configs"][tid]
        out.append(
            Candidate(
                tuning_id=tid,
                tile=tuple(row["tile"]),
                knobs=dict(row["knobs"]),
                score=geomean(ratios.values()),
                worst=ratios[worst_shape],
                worst_shape=worst_shape,
                ratios=ratios,
            )
        )
    return sorted(out, key=lambda c: (c.score, c.tuning_id))


def apply_rule(candidates: list, tie: float = TIE):
    """``(pick, decided_by)`` by the ``AUTO_DEFAULTS`` rule (module docstring);
    ``decided_by`` names the first step that left one configuration."""
    if not candidates:
        raise ValueError("no configuration was timed at every shape of the case")
    best_score = min(c.score for c in candidates)
    tied = [c for c in candidates if c.score <= best_score * (1 + tie)]
    if len(tied) == 1:
        return tied[0], "score"
    best_worst = min(c.worst for c in tied)
    tied_worst = [c for c in tied if c.worst <= best_worst * (1 + tie)]
    if len(tied_worst) == 1:
        return tied_worst[0], "worst shape"
    fewest = min(c.nknobs for c in tied_worst)
    tied_knobs = [c for c in tied_worst if c.nknobs == fewest]
    if len(tied_knobs) == 1:
        return tied_knobs[0], "fewer knobs"
    return min(tied_knobs, key=lambda c: (c.score, c.tuning_id)), "score (last step)"


def auto_score(case: dict):
    """``(score, worst, worst_shape, missing)`` of the ``auto`` rows, or
    ``None`` scores when ``auto`` is missing at some shape."""
    times = case["times"]
    missing = sorted(s for s in times if s not in case["auto"])
    if missing:
        return None, None, None, missing
    ratios = {s: times[s][case["auto"][s]] / min(times[s].values()) for s in times}
    worst_shape = max(sorted(ratios), key=ratios.get)
    return geomean(ratios.values()), ratios[worst_shape], worst_shape, []


def _value(value) -> str:
    return json.dumps(value) if isinstance(value, str) else repr(value)


def auto_defaults_row(case: tuple, pick: Candidate) -> str:
    """``pick`` as an ``AUTO_DEFAULTS`` entry. ``_auto`` adds the streaming
    state hints, so a pick without them is written as a bare ``AutoDefault``."""
    gate, fused, state = case
    key = f'("{gate}", {fused}, "{state}")'
    knobs = dict(pick.knobs)
    if all(knobs.get(k) == v for k, v in _STREAMING.items()):
        rest = {k: v for k, v in knobs.items() if k not in _STREAMING}
        args = "".join(f", {k}={_value(v)}" for k, v in sorted(rest.items()))
        return f"{key}: _auto({pick.tile}{args}),"
    body = ", ".join(f'"{k}": {_value(v)}' for k, v in sorted(knobs.items()))
    return f"{key}: AutoDefault({pick.tile}, MappingProxyType({{{body}}})),"


def report_case(case: tuple, data: dict, args, out) -> int:
    """Print one case; 1 when a shape is unscored (:func:`unscored`), the rows
    mix walks, or no proposal or no ``auto`` score is possible."""
    times = data["times"]
    print(
        f"\n## {case_label(case)}: {len(data['shapes'])} shapes, "
        f"{len(times)} with a timed row",
        file=out,
    )
    status = 0
    for shape, bad in unscored(data).items():
        statuses = ", ".join(
            f"{r['status'].upper()}{' (auto)' if r['auto'] else ''} "
            f"{r['tuning_id'] or '-'}: {r['reason'] or r['status']}"
            for r in bad
        )
        print(f"UNSCORED {shape.label}: {statuses or 'no row'}", file=out)
        status = 1
    if not times:
        print("no timed row at any shape: nothing to score", file=out)
        return 1
    candidates = score_candidates(data)
    a_score, a_worst, a_worst_shape, missing = auto_score(data)
    try:
        pick, decided_by = apply_rule(candidates, args.tie)
    except ValueError as exc:
        print(f"no proposal: {exc}", file=out)
        pick, decided_by = None, ""
    print(
        f"{'shape':<36} {'fastest tuning_id':<44} {'auto/fastest':>12} "
        f"{'proposal/fastest':>16}",
        file=out,
    )
    for shape in sorted(times):
        fastest = fastest_id(times[shape])
        tid = data["auto"].get(shape)
        auto = (
            f"{times[shape][tid] / times[shape][fastest]:12.3f}"
            if tid
            else f"{'-':>12}"
        )
        prop = f"{pick.ratios[shape]:16.3f}" if pick else f"{'-':>16}"
        print(f"{shape.label:<36} {fastest:<44} {auto} {prop}", file=out)
    if missing:
        print(
            f"auto: not timed at {len(missing)} shapes, e.g. {missing[0].label}",
            file=out,
        )
    else:
        auto_ids = sorted(set(data["auto"].values()))
        print(
            f"auto {', '.join(auto_ids)}: score {a_score:.4f}, worst shape "
            f"{a_worst:.4f} at {a_worst_shape.label}",
            file=out,
        )
    print(f"candidates timed at every shape: {len(candidates)}", file=out)
    if pick is None:
        return 1
    if len(data["walks"]) != 1:
        # Rows swept with different walks do not rank one candidate set.
        print(
            f"WITHHELD: rows come from {len(data['walks'])} walks "
            f"({', '.join(WALK_FIELDS)}) {sorted(data['walks'])}: no "
            "proposal; re-run stage 1 of the case with one",
            file=out,
        )
        return 1
    gain = (
        f"{pick.score / a_score:.4f}" if a_score else "n/a (auto not timed everywhere)"
    )
    same = sorted(set(data["auto"].values())) == [pick.tuning_id]
    walks = ", ".join(walk_flags(w) for w in sorted(data["walks"]))
    print(
        f"proposal {pick.tuning_id} (decided by {decided_by}): score {pick.score:.4f}, "
        f"worst shape {pick.worst:.4f} at {pick.worst_shape.label}, "
        f"{pick.nknobs} non-default knobs {json.dumps(pick.knobs, sort_keys=True)}; "
        f"proposal / auto score = {gain}"
        + ("; the proposal IS the current auto" if same else ""),
        file=out,
    )
    print(
        f"    best of the walk the rows were swept with ({walks}), not of every "
        "configuration",
        file=out,
    )
    print(f"    {auto_defaults_row(case, pick)}", file=out)
    print(
        f"{'top':>4} {'score':>7} {'worst':>7} {'knobs':>5}  tuning_id  (knobs)",
        file=out,
    )
    top = candidates[: args.top]
    for i, c in enumerate(top, start=1):
        mark = " <- proposal" if c is pick else ""
        print(
            f"{i:>4} {c.score:7.4f} {c.worst:7.4f} {c.nknobs:>5}  {c.tuning_id}  "
            f"{json.dumps(c.knobs, sort_keys=True)}{mark}",
            file=out,
        )
    shortlist = [c.tuning_id for c in top] + [pick.tuning_id]
    shortlist += [fastest_id(times[s]) for s in sorted(times)]
    shortlist = list(dict.fromkeys(shortlist))
    (walk,) = data["walks"]
    print(
        f"re-time: {walk_flags(walk)} "
        f"--tuning-ids {','.join(shortlist)} --top-k {len(shortlist)}",
        file=out,
    )
    return status if a_score else 1


_TOOLCHAIN = (
    ("rocm", "rocm_version"),
    ("llvm", "llvm_version"),
    ("torch", "torch_version"),
    ("backend", "backend"),
    ("flavor", "llvm_flavor"),
)


def toolchain_lines(rows: Iterable[dict]) -> list:
    """One ``toolchain: rocm .. llvm .. torch .. backend .. flavor ..`` line
    per distinct set the rows recorded (a row from an older file lacks a field:
    ``unknown``). Rows that disagree print every set: a mismatch withholds
    nothing but is never hidden."""
    sets = sorted(
        {tuple(str(r.get(key, "unknown")) for _, key in _TOOLCHAIN) for r in rows}
    )
    return [
        "toolchain: "
        + " ".join(f"{name} {value}" for (name, _), value in zip(_TOOLCHAIN, one))
        for one in sets
    ]


def report(rows: list, args, out=None) -> int:
    out = out or sys.stdout
    print(
        "ADVISORY: this report changes nothing. A person reviews it and ships "
        "AUTO_DEFAULTS (dispatch/gdn/axes.py), the literal table in "
        "test_gfx950_wiring.py and ALGORITHM.md section 4.6 together.",
        file=out,
    )
    print(f"stage {args.stage}; tie band {args.tie:g}", file=out)
    staged = [r for r in rows if r["stage"] == args.stage]
    warm = [r for r in staged if not is_cold(r)]
    rotates = sorted({r["rotate_mb"] for r in staged})
    if warm:
        print(
            f"WITHHELD: {len(warm)} of {len(staged)} stage-{args.stage} rows were "
            f"not timed in cold memory (rotate_mb {sorted({r['rotate_mb'] for r in warm})} "
            "MiB below the cold target): such times may come from caches and are "
            "not comparable with cold ones, so no proposal is made; re-time with "
            "the default --rotate-mb",
            file=out,
        )
    if len(rotates) > 1:
        print(
            f"WITHHELD: the stage-{args.stage} rows mix rotate_mb values {rotates}: "
            "their times are not comparable; re-time them with one",
            file=out,
        )
    if warm or len(rotates) > 1:
        return 1
    shas = sorted({r["sha"] for r in rows})
    devices = sorted({r["device"] for r in rows})
    print(f"rows from sha {', '.join(shas)} on {', '.join(devices)}", file=out)
    for line in toolchain_lines(rows):
        print(line, file=out)
    cases = collect(rows, args.stage)
    if not any(case["times"] for case in cases.values()):
        print(f"no timed stage-{args.stage} rows", file=out)
        return 1
    worst = 0
    for case in sorted(cases):
        worst = max(worst, report_case(case, cases[case], args, out))
    return worst


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json", nargs="+", help="decode_table_sweep.py --output-json files")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--tie", type=float, default=TIE, help="relative tie band")
    ap.add_argument("--top", type=int, default=8, help="candidates listed per case")
    args = ap.parse_args(argv)
    if not (math.isfinite(args.tie) and args.tie >= 0):
        ap.error("--tie must be a finite number >= 0")
    if args.top < 1:
        ap.error("--top must be >= 1")
    try:
        rows = load_rows(args.json)
    except (OSError, ValueError) as exc:
        ap.error(usage_error_text(exc))
    return report(rows, args)


if __name__ == "__main__":
    raise SystemExit(main())
