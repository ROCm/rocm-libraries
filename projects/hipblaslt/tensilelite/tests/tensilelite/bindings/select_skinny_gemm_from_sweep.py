# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pick a small kernel library from a one-shot skinny GEMM sweep (msgpack or YAML).

Reads sweep_skinny_gemm.py results where every screened configuration is listed
in each case's top entry (one-shot / full screening). No GPU work: coverage uses
the recorded timeUs values only.

For each case the best time is top[0].timeUs. A configuration covers a case when
its timeUs is within --tolerance of that best.

Only problem records with useBeta true are kept (the USE_BETA1 kernel path). Top
entries with config useBeta false are skipped (no USE_BETA0 symbols).

Config identity is one emitted kernel symbol: bm, tn, lanesK, waves, prefetch,
upIters, partialSteps, group, numXcds, grid3d, blocks, mfma, lds, use_beta true.

Selection (greedy, up to --max-configs rounds):
  1. Among cases not yet assigned, count how often each config identity appears
     within tolerance of that case's sweep best (same per-case filter as below).
  2. Add the identity with the highest count to the library (ties: identity tuple),
     then assign every case disjointly to its fastest within-tolerance config in
     the library so far.
  3. Repeat until the library is full, no identity helps remaining cases, or every
     case is assigned. Uncovered cases have no library config within tolerance.

Example:

  python tests/tensilelite/bindings/select_skinny_gemm_from_sweep.py \\
      results.msgpack -o selection.yaml --tolerance 0.10 --max-configs 14
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

import yaml

from sweep_skinny_gemm_results import read_results

DEFAULT_TOLERANCE = 0.10
LABEL_KEYS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")


class ConfigIdentity(NamedTuple):
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_iters: int
    partial_steps: int
    group: int
    xcds: int
    grid3d: bool
    blocks: int | None
    mfma: bool
    lds: bool
    use_beta: bool


@dataclass(frozen=True)
class SweepCase:
    index: int
    problem: dict
    best_us: float
    top: list

    @property
    def label(self) -> dict:
        return {key: self.problem[key] for key in LABEL_KEYS}

    def text(self) -> str:
        p = self.problem
        return f"{p['name']} m={p['m']} n={p['n']} k={p['k']} spk={p['spk']}"


@dataclass(frozen=True)
class QualifiedEntry:
    time_us: float
    cfg: dict
    kernel: str
    rank: int


@dataclass(frozen=True)
class Cover:
    identity: ConfigIdentity
    entry: QualifiedEntry


@dataclass(frozen=True)
class Selection:
    library: list[ConfigIdentity]
    tolerance_hits: Counter[ConfigIdentity]
    covers: dict[int, Cover]
    representatives: dict[ConfigIdentity, dict]


def identity_of(cfg: dict) -> ConfigIdentity:
    grid3d = bool(cfg["grid3d"])
    return ConfigIdentity(
        cfg["bm"],
        cfg["tn"],
        cfg["lanesK"],
        cfg["waves"],
        cfg["prefetch"],
        int(cfg["upIters"]),
        int(cfg["partialSteps"]),
        cfg["group"],
        cfg["numXcds"],
        grid3d,
        None if grid3d else cfg["blocks"],
        bool(cfg.get("mfma", False)),
        bool(cfg.get("lds", False)),
        True,
    )


def keep_problem(problem: dict) -> bool:
    return bool(problem.get("useBeta"))


def top_entry_allowed(item: dict) -> bool:
    return bool(item["config"].get("useBeta"))


def is_use_beta_kernel_name(name: str) -> bool:
    return "_USE_BETA1" in name and "_USE_BETA0" not in name


def tie_break_key(identity: ConfigIdentity) -> tuple[int, ...]:
    return tuple(-1 if value is None else int(value) for value in identity)


def rounded(value: float) -> float:
    return round(float(value), 4)


def config_record(identity: ConfigIdentity) -> dict:
    record = {name: getattr(identity, name) for name in identity._fields}
    record["grid3d"] = bool(identity.grid3d)
    return record


def load_cases(path: str, only: str | None, limit: int | None) -> list[SweepCase]:
    document = read_results(path)
    entries = [
        entry
        for entry in document["results"]
        if entry.get("top") and keep_problem(entry["problem"])
    ]
    if only:
        entries = [entry for entry in entries if only in entry["problem"]["name"]]
    if limit is not None:
        entries = entries[:limit]
    cases = []
    for index, entry in enumerate(entries):
        top = entry["top"]
        cases.append(SweepCase(index, entry["problem"], top[0]["timeUs"], top))
    return cases


def qualified_per_case(
    case: SweepCase, tolerance: float
) -> dict[ConfigIdentity, QualifiedEntry]:
    limit = case.best_us * (1 + tolerance)
    per_id: dict[ConfigIdentity, QualifiedEntry] = {}
    for item in case.top:
        if not top_entry_allowed(item):
            continue
        time_us = item["timeUs"]
        if time_us > limit:
            break
        identity = identity_of(item["config"])
        entry = QualifiedEntry(time_us, item["config"], item["kernel"], item["rank"])
        known = per_id.get(identity)
        if known is None or time_us < known.time_us:
            per_id[identity] = entry
    return per_id


def build_qualified(
    cases: list[SweepCase], tolerance: float
) -> dict[int, dict[ConfigIdentity, QualifiedEntry]]:
    return {case.index: qualified_per_case(case, tolerance) for case in cases}


def tolerance_hits(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
) -> Counter:
    counts: Counter[ConfigIdentity] = Counter()
    for per_id in qualified.values():
        for identity in per_id:
            counts[identity] += 1
    return counts


def hits_on_cases(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    case_indices: list[int],
) -> Counter[ConfigIdentity]:
    counts: Counter[ConfigIdentity] = Counter()
    for index in case_indices:
        for identity in qualified.get(index, {}):
            counts[identity] += 1
    return counts


def pick_next_identity(
    hits: Counter[ConfigIdentity],
    library: set[ConfigIdentity],
    min_config_cases: int,
) -> ConfigIdentity | None:
    candidates = [
        (identity, count)
        for identity, count in hits.items()
        if identity not in library and count >= min_config_cases
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (-pair[1], tie_break_key(pair[0])))[0]


def fastest_winner(
    options: dict[ConfigIdentity, QualifiedEntry],
) -> tuple[ConfigIdentity, QualifiedEntry]:
    return min(
        options.items(),
        key=lambda pair: (pair[1].time_us, tie_break_key(pair[0])),
    )


def assign_disjoint(
    cases: list[SweepCase],
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    library: set[ConfigIdentity],
) -> dict[int, Cover]:
    """Disjoint assignment: prefer the case's fastest qualifier; else fastest in library."""
    winners: dict[int, ConfigIdentity] = {}
    for case in cases:
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        identity, _ = fastest_winner(per_id)
        winners[case.index] = identity

    covers: dict[int, Cover] = {}
    for case in cases:
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        identity = winners[case.index]
        if identity in library:
            covers[case.index] = Cover(identity, per_id[identity])
    for case in cases:
        if case.index in covers:
            continue
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        options = {i: e for i, e in per_id.items() if i in library}
        if not options:
            continue
        identity, entry = fastest_winner(options)
        covers[case.index] = Cover(identity, entry)
    return covers


def representatives(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    library: set[ConfigIdentity],
) -> dict[ConfigIdentity, dict]:
    reps: dict[ConfigIdentity, dict] = {}
    for per_id in qualified.values():
        for identity, entry in per_id.items():
            if identity not in library:
                continue
            record = entry.cfg
            known = reps.get(identity)
            if known is None or entry.rank < known["_rank"]:
                reps[identity] = {**record, "_rank": entry.rank}
    for identity in list(reps):
        reps[identity] = {k: v for k, v in reps[identity].items() if k != "_rank"}
    return reps


def select_library(
    cases: list[SweepCase],
    tolerance: float,
    max_configs: int,
    min_config_cases: int,
) -> Selection:
    qualified = build_qualified(cases, tolerance)
    global_hits = tolerance_hits(qualified)
    library_set: set[ConfigIdentity] = set()
    covers: dict[int, Cover] = {}
    for _ in range(max_configs):
        uncovered = [case.index for case in cases if case.index not in covers]
        if not uncovered:
            break
        round_hits = hits_on_cases(qualified, uncovered)
        identity = pick_next_identity(round_hits, library_set, min_config_cases)
        if identity is None:
            break
        library_set.add(identity)
        covers = assign_disjoint(cases, qualified, library_set)
    library = sorted(
        library_set,
        key=lambda identity: (-global_hits[identity], tie_break_key(identity)),
    )
    reps = representatives(qualified, library_set)
    return Selection(library, global_hits, covers, reps)


def case_assignment_record(case: SweepCase, cover: Cover) -> dict:
    record = dict(case.label)
    record.update(
        kernel=cover.entry.kernel,
        timeUs=rounded(cover.entry.time_us),
        bestUs=rounded(case.best_us),
        ratio=rounded(cover.entry.time_us / case.best_us),
        rank=cover.entry.rank,
    )
    return record


def selected_record(
    identity: ConfigIdentity,
    hits: Counter[ConfigIdentity],
    cases: list[SweepCase],
    covers: dict[int, Cover],
    representatives: dict[ConfigIdentity, dict],
) -> dict:
    assigned = [i for i, cover in covers.items() if cover.identity == identity]
    kernels = sorted({covers[i].entry.kernel for i in assigned})
    return {
        "config": config_record(identity),
        "sweepConfig": representatives.get(identity),
        "toleranceHits": hits[identity],
        "num_cases": len(assigned),
        "kernels": kernels,
        "cases": [
            case_assignment_record(cases[i], covers[i]) for i in sorted(assigned)
        ],
    }


def validate_selected_rows(selected: list[dict]) -> None:
    for index, row in enumerate(selected):
        kernels = row["kernels"]
        if len(kernels) != 1:
            raise ValueError(
                f"selected[{index}] has {len(kernels)} kernels, expected 1: {kernels}"
            )
        kernel = kernels[0]
        if not is_use_beta_kernel_name(kernel):
            raise ValueError(f"selected[{index}] kernel is not USE_BETA1: {kernel}")
        for case in row["cases"]:
            if case.get("kernel") != kernel:
                raise ValueError(
                    f"selected[{index}] case kernel {case.get('kernel')!r} "
                    f"!= row kernel {kernel!r}"
                )


def uncovered_record(
    case: SweepCase,
    qualified: dict[ConfigIdentity, QualifiedEntry],
    library: set[ConfigIdentity],
    tolerance: float,
) -> dict:
    record = dict(case.label)
    record["bestUs"] = rounded(case.best_us)
    if not qualified:
        record["reason"] = "no configurations in sweep top"
        record["sweepBest"] = None
        record["bestInLibrary"] = None
        return record
    identity, entry = fastest_winner(qualified)
    record["sweepBest"] = {
        "config": config_record(identity),
        "timeUs": rounded(entry.time_us),
        "ratio": rounded(entry.time_us / case.best_us),
        "withinTolerance": entry.time_us <= case.best_us * (1 + tolerance),
        "inLibrary": identity in library,
    }
    in_lib = {i: e for i, e in qualified.items() if i in library}
    if in_lib:
        lib_id, lib_entry = fastest_winner(in_lib)
        record["bestInLibrary"] = {
            "config": config_record(lib_id),
            "timeUs": rounded(lib_entry.time_us),
            "ratio": rounded(lib_entry.time_us / case.best_us),
            "withinTolerance": lib_entry.time_us <= case.best_us * (1 + tolerance),
        }
    else:
        record["bestInLibrary"] = None
    if in_lib:
        record["reason"] = "within tolerance but lost disjoint assignment"
    elif record["sweepBest"]["withinTolerance"]:
        record["reason"] = (
            "sweep best not in library; no library config within tolerance"
        )
    else:
        record["reason"] = "no configuration within tolerance"
    return record


def write_selection_yaml(
    path: str,
    results_path: str,
    cases: list[SweepCase],
    selection: Selection,
    tolerance: float,
    max_configs: int,
    min_config_cases: int,
    only: str | None,
    limit_cases: int | None,
) -> None:
    qualified = build_qualified(cases, tolerance)
    library_set = set(selection.library)
    uncovered = [
        uncovered_record(case, qualified.get(case.index, {}), library_set, tolerance)
        for case in cases
        if case.index not in selection.covers
    ]
    selected_rows = [
        selected_record(
            identity,
            selection.tolerance_hits,
            cases,
            selection.covers,
            selection.representatives,
        )
        for identity in selection.library
    ]
    validate_selected_rows(selected_rows)
    document = {
        "meta": {
            "tool": "select_skinny_gemm_from_sweep.py",
            "results": os.path.abspath(results_path),
            "tolerance": tolerance,
            "max_configs": max_configs,
            "min_config_cases": min_config_cases,
            "use_beta_only": True,
            "only": only,
            "limit_cases": limit_cases,
            "num_cases": len(cases),
            "num_selected": len(selection.library),
            "num_covered": len(selection.covers),
            "num_uncovered": len(uncovered),
        },
        "selected": selected_rows,
        "uncovered": uncovered,
    }
    with open(path, "w") as stream:
        yaml.safe_dump(
            document, stream, sort_keys=False, width=1000, default_flow_style=None
        )


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select skinny GEMM kernels from a one-shot sweep results file."
    )
    parser.add_argument(
        "results",
        help="msgpack or YAML results from sweep_skinny_gemm.py (--one-shot)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="selection.yaml",
        help="output YAML path (default: selection.yaml)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="allowed relative slowdown over each case's best time (default 0.10)",
    )
    parser.add_argument(
        "--max-configs",
        type=positive_int,
        required=True,
        help="maximum number of kernel configurations in the library",
    )
    parser.add_argument(
        "--min-config-cases",
        type=non_negative_int,
        default=0,
        help="drop configs with fewer tolerance hits before applying --max-configs",
    )
    parser.add_argument("--only", help="keep cases whose name contains this substring")
    parser.add_argument(
        "--limit-cases", type=int, help="keep the first N cases after --only"
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.tolerance < 0:
        print("error: --tolerance must be non-negative", file=sys.stderr)
        return 2
    cases = load_cases(args.results, args.only, args.limit_cases)
    if not cases:
        print("error: no cases with top timings in results", file=sys.stderr)
        return 2
    selection = select_library(
        cases, args.tolerance, args.max_configs, args.min_config_cases
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_selection_yaml(
        args.output,
        args.results,
        cases,
        selection,
        args.tolerance,
        args.max_configs,
        args.min_config_cases,
        args.only,
        args.limit_cases,
    )
    print(
        f"selected {len(selection.library)} configurations covering "
        f"{len(selection.covers)} of {len(cases)} cases -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
