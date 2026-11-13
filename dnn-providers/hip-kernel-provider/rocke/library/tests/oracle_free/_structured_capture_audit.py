# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Replayable audit of saved structured-logit GPU captures, coordinate by coordinate.

The GPU lane writes the raw output bits of each quick case as
``<case_id>.npz`` (one array per exposure) when
``ROCKE_ORACLE_FREE_CALIBRATION_OUT`` is set. Tier B membership alone cannot
explain such a capture: the Tier B of an affected FP16 coordinate already held
both epilogue roundings before assumption A-F named them. This audit asks the
sharper question. At every coordinate the stored bits must equal one member of
the exact union {IR-exact, single-rounded epilogue} of the IR-exact path, and
the coordinates where the two differ must follow the named artifact's
per-site rule. Membership in Tier B is checked separately. Unexplained
coordinates are reported, never absorbed into an enclosure.

Usage, from ``rocke/library`` with ``rocke`` importable::

    PYTHONPATH=.:../platform/python python -m \\
        tests.oracle_free._structured_capture_audit \\
        --artifact rocm-7.13.0 <capture dir> [<capture dir> ...]

The first directory is audited; every further one must be bit-identical to it.
The exit status is 0 only when every coordinate is explained.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ._structured_analytical import (
    STRUCTURED_EPILOGUE_ARTIFACTS,
    STRUCTURED_EXPOSURES,
    STRUCTURED_QUICK_CASES,
    StructuredCase,
    epilogue_alternatives,
    tier_b_bounds,
    tier_b_exposure,
)

Coordinate = tuple[str, int, int, int]  # exposure, query, head, dim


@dataclass(frozen=True)
class CaseAudit:
    """Every coordinate of one case judged against the exact epilogue union.

    ``ir_mismatches`` differ from the IR-exact emulation and
    ``artifact_mismatches`` from the named artifact's. ``lane_rule`` counts,
    per ``d mod 4``, the coordinates whose single and double roundings differ
    as ``(stored single, stored double)``: the only coordinates that can
    confirm or refute a position rule."""

    case_id: str
    coordinates: int
    ir_mismatches: tuple[Coordinate, ...]
    artifact_mismatches: tuple[Coordinate, ...]
    unexplained: tuple[tuple[Coordinate, int], ...]
    outside_tier_b: tuple[Coordinate, ...]
    lane_rule: Mapping[int, tuple[int, int]]


@dataclass(frozen=True)
class CaptureAudit:
    artifact: str
    directories: tuple[Path, ...]
    problems: tuple[str, ...]  # missing or malformed captures, cross-capture drift
    cases: Mapping[str, CaseAudit]

    @property
    def explained(self) -> bool:
        return not self.problems and all(
            not (item.unexplained or item.outside_tier_b or item.artifact_mismatches)
            for item in self.cases.values()
        )


def load_capture(
    directory: Path, case: StructuredCase
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """One case's saved exposures, or ``None`` and the reasons it is unusable."""
    path = Path(directory) / f"{case.case_id}.npz"
    if not path.is_file():
        return None, [f"{path}: missing"]
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    problems = []
    if sorted(arrays) != sorted(STRUCTURED_EXPOSURES):
        problems.append(
            f"{path}: exposures {sorted(arrays)} != {sorted(STRUCTURED_EXPOSURES)}"
        )
    for name, array in arrays.items():
        if array.dtype != np.uint16 or array.shape != tuple(case.output_shape):
            problems.append(
                f"{path}: {name} is {array.dtype}{list(array.shape)}, "
                f"expected uint16{list(case.output_shape)}"
            )
    return (None if problems else arrays), problems


def audit_case(
    case: StructuredCase,
    capture: Mapping[str, np.ndarray],
    *,
    artifact: str,
    rows: Iterable[int] | None = None,
) -> CaseAudit:
    """Judge every coordinate of ``rows`` (default: all) of one capture."""
    single_lanes = STRUCTURED_EPILOGUE_ARTIFACTS[artifact].single_lanes
    queries = tuple(range(case.base.seqlen_q)) if rows is None else tuple(rows)
    lane = np.arange(case.base.head_dim) % 4
    single_site = np.isin(lane, sorted(single_lanes))
    ir_mismatches: list[Coordinate] = []
    artifact_mismatches: list[Coordinate] = []
    unexplained: list[tuple[Coordinate, int]] = []
    outside: list[Coordinate] = []
    rule = np.zeros((4, 2), dtype=np.int64)
    coordinates = 0
    for exposure in STRUCTURED_EXPOSURES:
        if rows is None:
            low, high = tier_b_exposure(case, exposure)
        else:
            low, high = tier_b_bounds(case, exposure, rows=queries)
        for query in queries:
            for head in range(case.base.nhead_q):
                observed = capture[exposure][0, query, head]
                double, single = epilogue_alternatives(case, exposure, query, head)
                expected = np.where(single_site, single, double)
                coordinates += observed.size
                differs = single != double
                for value in range(4):
                    at = differs & (lane == value)
                    rule[value, 0] += int(np.count_nonzero(at & (observed == single)))
                    rule[value, 1] += int(np.count_nonzero(at & (observed == double)))
                inside = (observed >= low[0, query, head]) & (
                    observed <= high[0, query, head]
                )
                for found, mask in (
                    (ir_mismatches, observed != double),
                    (artifact_mismatches, observed != expected),
                    (outside, ~inside),
                ):
                    found.extend(
                        (exposure, query, head, int(dim)) for dim in np.nonzero(mask)[0]
                    )
                unexplained.extend(
                    ((exposure, query, head, int(dim)), int(observed[dim]))
                    for dim in np.nonzero((observed != double) & (observed != single))[
                        0
                    ]
                )
    return CaseAudit(
        case.case_id,
        coordinates,
        tuple(ir_mismatches),
        tuple(artifact_mismatches),
        tuple(unexplained),
        tuple(outside),
        MappingProxyType(
            {value: (int(rule[value, 0]), int(rule[value, 1])) for value in range(4)}
        ),
    )


def audit_captures(
    directories: Sequence[Path],
    *,
    artifact: str,
    cases: Sequence[StructuredCase] = STRUCTURED_QUICK_CASES,
    rows: Iterable[int] | None = None,
) -> CaptureAudit:
    """Audit the first capture directory and compare the others bit-for-bit."""
    if artifact not in STRUCTURED_EPILOGUE_ARTIFACTS:
        raise ValueError(f"no epilogue artifact named {artifact!r}")
    if not directories:
        raise ValueError("at least one capture directory is required")
    directories = tuple(Path(directory) for directory in directories)
    rows = None if rows is None else tuple(rows)
    problems: list[str] = []
    audited: dict[str, CaseAudit] = {}
    for case in cases:
        reference, found = load_capture(directories[0], case)
        problems.extend(found)
        for other in directories[1:]:
            capture, found = load_capture(other, case)
            problems.extend(found)
            if reference is None or capture is None:
                continue
            differing = sum(
                int(np.count_nonzero(reference[name] != capture[name]))
                for name in STRUCTURED_EXPOSURES
            )
            if differing:
                problems.append(
                    f"{case.case_id}: {differing} coordinates differ between "
                    f"{directories[0]} and {other}"
                )
        if reference is not None:
            audited[case.case_id] = audit_case(
                case, reference, artifact=artifact, rows=rows
            )
    return CaptureAudit(
        artifact, directories, tuple(problems), MappingProxyType(audited)
    )


def format_audit(audit: CaptureAudit, *, limit: int = 8) -> str:
    compiler = STRUCTURED_EPILOGUE_ARTIFACTS[audit.artifact].compiler
    captures = [str(path) for path in audit.directories]
    lines = [f"artifact={audit.artifact} ({compiler}); captures={captures}"]
    lines.extend(f"PROBLEM {problem}" for problem in audit.problems)
    for case_id, item in audit.cases.items():
        rule = " ".join(
            f"d%4={value}:single={single}/double={double}"
            for value, (single, double) in item.lane_rule.items()
        )
        lines.append(
            f"{case_id}: coordinates={item.coordinates} "
            f"ir_exact_mismatches={len(item.ir_mismatches)} "
            f"artifact_exact_mismatches={len(item.artifact_mismatches)} "
            f"unexplained={len(item.unexplained)} "
            f"outside_tier_b={len(item.outside_tier_b)}; {rule}"
        )
        for label, found in (
            ("ir_exact mismatch", item.ir_mismatches),
            ("artifact_exact mismatch", item.artifact_mismatches),
            ("unexplained", item.unexplained),
            ("outside Tier B", item.outside_tier_b),
        ):
            for entry in found[:limit]:
                lines.append(f"  {label}: {entry}")
            if len(found) > limit:
                lines.append(f"  ... {len(found) - limit} more {label}")
    lines.append("explained" if audit.explained else "NOT EXPLAINED")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifact", required=True, choices=sorted(STRUCTURED_EPILOGUE_ARTIFACTS)
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("directories", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    audit = audit_captures(arguments.directories, artifact=arguments.artifact)
    print(format_audit(audit, limit=arguments.limit))
    return 0 if audit.explained else 1


if __name__ == "__main__":
    sys.exit(main())
