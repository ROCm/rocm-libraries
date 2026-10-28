# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turns benchmark results into routing cells, rows, held-out validation and report tables.

Per point, the time of a solution is the minimum over the successful repeats: contention
from other tenants only adds time, so the minimum of short runs approaches the
uncontended time (the median and the lower quartile were much noisier).

The lattice of a grid cuts every axis (skinny extent, wide extent, K) into disjoint
integer intervals: one point interval per lattice value and one gap interval between
neighbours. An elementary box is a product of three intervals; its corners are lattice
points measured by the campaign. A pool entry is a candidate of a box when it supports
the K interval, is a clear win at every corner and does not regress any other measured
point inside the box. Boxes are merged along K, then wide, then skinny while their
candidate sets keep a common entry, and each merged cell gets the candidate closest to
the best generated kernel at its corners. Cells are disjoint, so the order of the
resulting Range rows does not matter.
"""

import bisect
import itertools
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from . import SkinnyGEMM, SkinnyGEMMLogic
from .SkinnyGEMMBench import (SKINNY_MAX, Grid, Knobs, Point, RunRecord, SolutionTable, lattice_points, make_point,
                              orientation_of)

# Points whose old-default to generated time ratio lies in this band get more repeats: the
# split-half noise of the minimum of 6 runs on the shared GPUs has a p90 of 1.04-1.27 and a
# p99 of about 2, so ratios around the default margin of 1.25 are not yet decided.
REFINE_LOW = 0.8
REFINE_HIGH = 1.6

MAX_HELD_OUT_ATTEMPTS = 50
# Share of held-out K values that are multiples of 32 (the MX block); the rest probe the tails.
BLOCK_ALIGNED_SHARE = 0.8
BLOCK = 32
LD_MULTIPLE = SkinnyGEMM.LD_MULTIPLE

Measurements = Dict[Tuple[str, Point], "Measurement"]
Key = Tuple[int, ...]


@dataclass(frozen=True)
class Thresholds:
    margin: float = 1.25
    tolerance: float = 1.10
    regression: float = 1.05


@dataclass
class Measurement:
    phase: str
    point: Point
    old_index: Optional[int]
    old_time: float
    gen: Dict[Knobs, float]

    @property
    def best_gen(self) -> float:
        return min(self.gen.values(), default=math.inf)


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int
    corners: Tuple[int, ...]


@dataclass(frozen=True)
class Cell:
    orientation: str
    s: Tuple[int, int]
    w: Tuple[int, int]
    k: Tuple[int, int]
    candidates: FrozenSet[Knobs]
    corners: FrozenSet[Point]
    entry: Optional[Knobs] = None


@dataclass(frozen=True)
class Route:
    cell_id: str
    key: Key
    knobs: Knobs
    corners: int = 0


# ---------------------------------------------------------------------------
# Aggregation and acceptance.


def _old_default(runs: Sequence[RunRecord], table: SolutionTable, point: Point) -> Optional[int]:
    """The index Origami picks; None when its pick is a generated kernel (Origami supported nothing)."""
    picks = [None if table.knobs(run.first_index) is not None else run.first_index for run in runs]
    if len(set(picks)) > 1:
        print(f"warning: the repeats of {point.key} disagree on the old default: {dict(Counter(picks))}",
              file=sys.stderr)
    return vote(picks)


def vote(values: Sequence[Optional[int]]) -> Optional[int]:
    """The most frequent value, the larger one on ties."""
    counts = Counter(values)
    return max(counts, key=lambda value: (counts[value], -1 if value is None else value))


def _measure(phase: str, point: Point, runs: Sequence[RunRecord], later_runs: Sequence[RunRecord],
             table: SolutionTable) -> Measurement:
    old = _old_default(runs, table, point)
    all_runs = [*runs, *later_runs]
    old_times = [run.times[old] for run in all_runs if old in run.times]
    gen: Dict[Knobs, float] = {}
    for run in all_runs:
        for index, microseconds in run.times.items():
            knobs = table.knobs(index)
            # Routing only targets the useBeta=false variants, and the bench times both with beta 0.
            if knobs is not None and not SkinnyGEMMLogic.usesBeta(knobs):
                gen[knobs] = min(microseconds, gen.get(knobs, microseconds))
    return Measurement(phase, point, old, min(old_times, default=math.inf), gen)


def aggregate(records: Iterable[RunRecord], table: SolutionTable) -> Measurements:
    """Per (phase, point) measurement of the campaign runs, which ran with TENSILE_PREDICTION_LIB.

    Validation runs of the same point add repeats of the same solutions (the old default comes from the
    campaign runs only): contention on shared GPUs varies over hours and the minimum only gains from more repeats.
    """
    groups: Dict[Tuple[str, Point], List[RunRecord]] = {}
    later: Dict[Point, List[RunRecord]] = {}
    for run in records:
        if run.return_code != 0:
            continue
        if run.prediction_lib:
            groups.setdefault((run.phase, run.point), []).append(run)
        else:
            later.setdefault(run.point, []).append(run)
    return {key: _measure(key[0], key[1], runs, later.get(key[1], []), table) for key, runs in groups.items()}


def accepts(measurement: Measurement, knobs: Knobs, limits: Thresholds) -> bool:
    time = measurement.gen.get(knobs)
    return (time is not None and time * limits.margin <= measurement.old_time
            and time <= limits.tolerance * measurement.best_gen)


def no_regression(measurement: Measurement, knobs: Knobs, limits: Thresholds) -> bool:
    time = measurement.gen.get(knobs)
    return time is not None and time <= limits.regression * measurement.old_time


def k_supports(knobs: Knobs, lo: int, hi: int) -> bool:
    """Whether the entry's K predicates hold for every K in [lo, hi]; mirrors SkinnyGEMM._asserts."""
    asserts = SkinnyGEMM._asserts(SkinnyGEMMLogic.asParams(knobs))
    if lo <= asserts["AssertSizeGreaterThan"][3] or hi >= asserts["AssertSizeLessThan"][3]:
        return False
    step = asserts.get("AssertSummationElementMultiple")
    return step is None or (lo == hi and lo % step == 0)


# ---------------------------------------------------------------------------
# Elementary boxes.


def partition(values: Sequence[int]) -> List[Interval]:
    """Point intervals of the values and the non-empty gaps between neighbours; disjoint, covering [min, max]."""
    ordered = sorted(set(values))
    intervals = []
    for index, value in enumerate(ordered):
        intervals.append(Interval(value, value, (value,)))
        if index + 1 < len(ordered) and ordered[index + 1] - value > 1:
            intervals.append(Interval(value + 1, ordered[index + 1] - 1, (value, ordered[index + 1])))
    return intervals


def to_point(orientation: str, skinny: int, wide: int, k: int) -> Point:
    return make_point(wide, skinny, k) if orientation == "n" else make_point(skinny, wide, k)


def split_point(point: Point) -> Optional[Tuple[str, int, int, int]]:
    orientation = orientation_of(point)
    if orientation is None:
        return None
    skinny, wide = (point.n, point.m) if orientation == "n" else (point.m, point.n)
    return orientation, skinny, wide, point.k


def _find(intervals: Sequence[Interval], value: int) -> Optional[int]:
    index = bisect.bisect_right([interval.lo for interval in intervals], value) - 1
    return index if index >= 0 and value <= intervals[index].hi else None


class CellBuilder:
    """Builds the merged cells of every orientation of a grid from its measurements."""

    def __init__(self, grid: Grid, measurements: Measurements, limits: Thresholds, pool: SkinnyGEMMLogic.Pool,
                 include_held_out: bool):
        self.grid = grid
        self.limits = limits
        self.order = {knobs: index for index, knobs in enumerate(pool)}
        self.axes = {"s": partition(grid.skinny), "w": partition(grid.wide), "k": partition(grid.k)}
        self.corners = {point: m for (phase, point), m in measurements.items() if phase == "grid"}
        lattice = set(lattice_points(grid))
        phases = ("grid", "held_out") if include_held_out else ("grid",)
        self.interior: Dict[Tuple[str, int, int, int], List[Measurement]] = {}
        for (phase, point), measurement in sorted(measurements.items(), key=lambda item: item[0][1]):
            located = self._locate(point)
            if phase in phases and point not in lattice and located is not None:
                self.interior.setdefault(located, []).append(measurement)
        self._accepted: Dict[Point, FrozenSet[Knobs]] = {}
        self._k_supported = [frozenset(k for k in self.order if k_supports(k, i.lo, i.hi)) for i in self.axes["k"]]

    def _locate(self, point: Point) -> Optional[Tuple[str, int, int, int]]:
        split = split_point(point)
        if split is None or split[0] not in self.grid.orientations:
            return None
        found = [_find(self.axes[axis], value) for axis, value in zip("swk", split[1:])]
        return None if None in found else (split[0], *found)

    def _accepted_at(self, point: Point) -> FrozenSet[Knobs]:
        if point not in self._accepted:
            measurement = self.corners.get(point)
            self._accepted[point] = frozenset() if measurement is None else frozenset(
                knobs for knobs in measurement.gen if accepts(measurement, knobs, self.limits))
        return self._accepted[point]

    def _box_corners(self, orientation: str, s: Interval, w: Interval, k: Interval) -> FrozenSet[Point]:
        return frozenset(to_point(orientation, *corner) for corner in itertools.product(s.corners, w.corners, k.corners))

    def _box(self, orientation: str, si: int, wi: int, ki: int) -> Optional[Cell]:
        s, w, k = self.axes["s"][si], self.axes["w"][wi], self.axes["k"][ki]
        corners = self._box_corners(orientation, s, w, k)
        candidates = set(self._k_supported[ki])
        for corner in corners:
            candidates &= self._accepted_at(corner)
        for measurement in self.interior.get((orientation, si, wi, ki), ()):
            candidates = {knobs for knobs in candidates if no_regression(measurement, knobs, self.limits)}
        if not candidates:
            return None
        return Cell(orientation, (s.lo, s.hi), (w.lo, w.hi), (k.lo, k.hi), frozenset(candidates), corners)

    def boxes(self) -> List[Cell]:
        found = (self._box(o, si, wi, ki) for o in self.grid.orientations
                 for si, wi, ki in itertools.product(*(range(len(self.axes[a])) for a in "swk")))
        return [box for box in found if box is not None]

    def _score(self, knobs: Knobs, corners: Iterable[Point]) -> float:
        """Geometric mean over the corners of the time relative to the best generated kernel."""
        logs = [math.log(self.corners[c].gen[knobs] / self.corners[c].best_gen) for c in corners]
        return round(math.exp(sum(logs) / len(logs)), 12)

    def choose_entry(self, cell: Cell) -> Cell:
        entry = min(cell.candidates, key=lambda knobs: (self._score(knobs, cell.corners), self.order[knobs]))
        return replace(cell, entry=entry)

    def build(self) -> List[Cell]:
        cells = self.boxes()
        for axis in "kws":
            cells = merge_along(cells, axis)
        return [self.choose_entry(cell) for cell in sorted(cells, key=cell_key)]


def merge_along(cells: Sequence[Cell], axis: str) -> List[Cell]:
    """Joins neighbours along one axis that agree on the other two ranges while a common candidate remains."""
    others = [a for a in "swk" if a != axis]
    groups: Dict[Tuple, List[Cell]] = {}
    for cell in cells:
        groups.setdefault((cell.orientation, *(getattr(cell, a) for a in others)), []).append(cell)
    merged: List[Cell] = []
    for key in sorted(groups):
        run: Optional[Cell] = None
        for cell in sorted(groups[key], key=lambda c: getattr(c, axis)):
            joined = None if run is None else _join(run, cell, axis)
            if joined is not None:
                run = joined
                continue
            merged += [run] if run is not None else []
            run = cell
        merged += [run] if run is not None else []
    return merged


def _join(run: Cell, cell: Cell, axis: str) -> Optional[Cell]:
    if getattr(run, axis)[1] + 1 != getattr(cell, axis)[0]:
        return None
    common = run.candidates & cell.candidates
    if not common:
        return None
    span = (getattr(run, axis)[0], getattr(cell, axis)[1])
    return replace(run, candidates=common, corners=run.corners | cell.corners, **{axis: span})


# ---------------------------------------------------------------------------
# Routes (the Range rows) and their invariants.


def cell_key(cell: Cell) -> Key:
    """The Range key [m_lo, m_hi, n_lo, n_hi, b_lo, b_hi, k_lo, k_hi] of a cell."""
    ranges = (*cell.w, *cell.s) if cell.orientation == "n" else (*cell.s, *cell.w)
    return (*ranges, 1, 1, *cell.k)


def routes_from(entries: Iterable[Tuple[Key, Knobs, int]]) -> List[Route]:
    """Numbers the (key, knobs, corner count) entries in key order and checks that they are disjoint."""
    routes = []
    for index, (key, knobs, corners) in enumerate(sorted(entries, key=lambda entry: entry[0])):
        orientation = "n" if key[3] <= SKINNY_MAX else "m"
        routes.append(Route(f"{orientation}{index:03d}", key, knobs, corners))
    assert_disjoint(routes)
    return routes


def make_routes(cells: Sequence[Cell]) -> List[Route]:
    assert all(cell.entry is not None for cell in cells), "a cell without an entry"
    return routes_from((cell_key(cell), cell.entry, len(cell.corners)) for cell in cells)


def _overlaps(first: Key, second: Key) -> bool:
    return all(a_lo <= b_hi and b_lo <= a_hi
               for a_lo, a_hi, b_lo, b_hi in zip(first[0::2], first[1::2], second[0::2], second[1::2]))


def assert_disjoint(routes: Sequence[Route]) -> None:
    for first, second in itertools.combinations(routes, 2):
        assert not _overlaps(first.key, second.key), f"cells {first.cell_id} and {second.cell_id} overlap"


def matches_key(key: Key, point: Point) -> bool:
    return (key[0] <= point.m <= key[1] and key[2] <= point.n <= key[3] and key[6] <= point.k <= key[7])


def find_route(routes: Sequence[Route], point: Point) -> Optional[Route]:
    return next((route for route in routes if matches_key(route.key, point)), None)


def assert_no_regression(routes: Sequence[Route], measurements: Measurements, limits: Thresholds) -> None:
    """Every measured point inside a cell must satisfy the regression bound for the cell's entry."""
    for (phase, point), measurement in measurements.items():
        route = find_route(routes, point)
        assert route is None or no_regression(measurement, route.knobs, limits), (
            f"{phase} point {point.key} regresses in cell {route.cell_id}")


# ---------------------------------------------------------------------------
# Held-out sampling and offline validation.


def _multiple_in(rng: random.Random, lo: int, hi: int, step: int) -> Optional[int]:
    first, last = -(-lo // step), hi // step
    return None if first > last else step * rng.randint(first, last)


def _draw_k(rng: random.Random, lo: int, hi: int) -> Optional[int]:
    if lo == hi:
        return lo
    if rng.random() < BLOCK_ALIGNED_SHARE:
        return _multiple_in(rng, lo, hi, BLOCK)
    for _ in range(MAX_HELD_OUT_ATTEMPTS):
        k = _multiple_in(rng, lo, hi, LD_MULTIPLE)
        if k is not None and k % BLOCK != 0:
            return k
    return None


def _draw_point(rng: random.Random, cell: Cell) -> Optional[Point]:
    skinny = rng.randint(*cell.s)
    wide = _multiple_in(rng, *cell.w, LD_MULTIPLE)
    k = _draw_k(rng, *cell.k)
    return None if wide is None or k is None else to_point(cell.orientation, skinny, wide, k)


def sample_held_out(cells: Sequence[Cell], grid: Grid) -> List[Point]:
    """Deterministic points spread over the cells that are neither lattice nor extra points."""
    if not cells or grid.held_out_count <= 0:
        return []
    rng = random.Random(grid.held_out_seed)
    excluded = set(lattice_points(grid)) | set(grid.extra_points)
    sample: Dict[Point, None] = {}
    for _ in range(grid.held_out_count * MAX_HELD_OUT_ATTEMPTS):
        if len(sample) >= grid.held_out_count:
            break
        point = _draw_point(rng, rng.choice(cells))
        if point is not None and point not in excluded:
            sample[point] = None
    return list(sample)


@dataclass(frozen=True)
class HeldOutResult:
    point: Point
    cell_id: str
    knobs: Knobs
    old_time: float
    gen_time: Optional[float]
    regression: bool

    @property
    def speedup(self) -> Optional[float]:
        return None if self.gen_time is None else self.old_time / self.gen_time


def validate_held_out(routes: Sequence[Route], measurements: Measurements, limits: Thresholds,
                    points: Iterable[Point]) -> List[HeldOutResult]:
    """Routes each measured held-out point through the cells and compares the entry with the old default."""
    results = []
    for point in points:
        measurement, route = measurements.get(("held_out", point)), find_route(routes, point)
        if measurement is None or route is None:
            continue
        time = measurement.gen.get(route.knobs)
        results.append(HeldOutResult(point, route.cell_id, route.knobs, measurement.old_time, time,
                                     not no_regression(measurement, route.knobs, limits)))
    return results


# ---------------------------------------------------------------------------
# Refinement of the points that the noise leaves undecided.


def in_noise_band(ratio: float) -> bool:
    return REFINE_LOW <= ratio <= REFINE_HIGH


def _decision_time(measurement: Measurement, phase: str, routes: Sequence[Route]) -> float:
    if phase == "grid":
        return measurement.best_gen
    route = find_route(routes, measurement.point)
    return math.inf if route is None else measurement.gen.get(route.knobs, math.inf)


def refinement_points(measurements: Measurements, phase: str, routes: Sequence[Route] = ()) -> List[Point]:
    """Grid: old default over the best generated kernel. Held-out: old default over the routed entry."""
    points = []
    for (measured_phase, point), measurement in sorted(measurements.items(), key=lambda item: item[0][1]):
        time = _decision_time(measurement, phase, routes)
        if measured_phase != phase or math.isinf(measurement.old_time) or math.isinf(time):
            continue
        if in_noise_band(measurement.old_time / time):
            points.append(point)
    return points
