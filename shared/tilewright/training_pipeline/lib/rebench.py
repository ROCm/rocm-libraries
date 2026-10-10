# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Near-winner re-reads: time again the candidates of each GEMM that came
within a tolerance of its winner, in bench processes of their own, and
label them by the median of their readings.

One bench pass cannot order kernels whose latencies differ by less than its
noise, and the training targets are built from that order. For every
complete problem of a stage03 log, the candidates whose latency (a tested
row's timing or a skip row's warm-up) is within `tolerance` of the winner
(the fastest tested row), or within `tolerance_us` of it when that is wider
(`tolerance_limit`: a reading's error that does not scale with the kernel's
latency, such as a bench process's own offset, outweighs a relative
tolerance on a short winner), form its near-tie set; a problem whose set
holds only the winner needs nothing. Each near-tie set is read again in
runs, as many as the band of the GEMM's winner latency asks (`Bands`), every
run a bench of its own (`reread_line`: an explicit `solution_index_list`,
optionally led by a sacrificial copy of its first kernel, no skip screen,
the line's adaptive timing, at least `slow_min_iters` iterations per window
in the slow band, rotating buffers), its list in a seeded shuffle rotating
from run to run (`run_order`). With `inline_copies` k, a run's line lists
the set k times (`reread_list`), each copy rotated one place further, so a
run reads every candidate k times in one bench process: one data set-up
for the line instead of k, and its readings share that process's offset.
Every re-read candidate is replaced in the log by one row, its rank kept: a
skip row becomes a tested row, and the `Winner:` block names the new
fastest row (`merge_log`). Its latency is the median of its original timing
and its re-reads (`with_original`), a copy of the row of the lower middle
reading with that latency. With `align_to_original`, each run's re-reads of
a GEMM are first shifted by the median difference between them and the
original timings of the same kernels (`Samples.alignment`): a process
reads every kernel of a line late or early by about the same time, so
without the shift a candidate's label moves by part of the run's offset
against the kernels of its GEMM that were not read again. A run is shifted
only when its offset rests on at least `MIN_ALIGN_DIFFS` differences and
leaves every reading of the run above 0; any other run merges as it was
read (`Alignment.unaligned` says why). A skip row's warm-up is no timing
and takes no part, so in a set that holds a skip row every candidate is
labelled from its re-reads alone, unshifted, which keeps the labels of one
GEMM from one kind of process. Without `with_original`, the row of the
re-read with the lower median latency, the original timing left out
because it put the candidate in the set. Candidates no run timed keep their
original rows. Kernels a blocklist names (lib/correctness.py,
`blocked_solutions`) take no part in a set, as candidates or as the winner
the tolerance counts from.
"""
from __future__ import annotations

import dataclasses
import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import procs
from .bench_log import HEADER_RE, Problem, iter_problems
from .bench_yaml import drop_line_knob, format_solution_list, set_line_knob

Shape = Tuple[int, int, int, int]

DEFAULT_TOLERANCE = 0.05
DEFAULT_MAX_KERNELS = 16
# Runs of the pick re-bench (`rebench_picks`).
DEFAULT_RUNS = 3
# Lines of a GPU's share per stage03 block of a re-bench, and the most
# blocks a re-bench is cut into per GPU (stage03's own default).
REBENCH_LINES_PER_BLOCK = 1024
MAX_BLOCKS_PER_GPU = 20

_RANK_RE = re.compile(r"^(\s*)\[(\d+)\]:")
_ROTATING_RE = re.compile(r"(?<![\w])rotating:\s*(\d+)")
_MIN_ITERS_RE = re.compile(r"(?<![\w])min_iters:\s*(\d+)")
GEMM_ORDERS = ("shuffled", "listed")
# The fewest differences from the original timings a run's offset may rest
# on before `align_to_original` shifts the run: the median of two is their
# mean, which one slow re-read moves by half its excess, so a set of two
# candidates is shifted only with two inline copies or more.
MIN_ALIGN_DIFFS = 3
# Why `align_to_original` left a run's re-reads of a GEMM as they were read
# (`Alignment.unaligned`): the GEMM's set holds a skip row (see the module
# docstring), the run's offset would rest on fewer than MIN_ALIGN_DIFFS
# differences, or the shift would take a reading of the run to 0 or below.
UNALIGNED_SKIP_ROW = "skip_row"
UNALIGNED_FEW = "few_differences"
UNALIGNED_NONPOSITIVE = "nonpositive"
UNALIGNED_REASONS = (UNALIGNED_SKIP_ROW, UNALIGNED_FEW, UNALIGNED_NONPOSITIVE)


@dataclass(frozen=True)
class Bands:
    """How often stage03b reads a GEMM's near-winners again, by the latency
    of its winner (the fastest tested row of its original bench):
    `fast_runs` times below `fast_below_us`, where single readings of
    launch-bound kernels flip the winner most; `slow_runs` times from
    `slow_from_us` on, each window at least `slow_min_iters` iterations
    long (0: the line's own, which ends after a few iterations of such a
    kernel); `runs` times in between. A band of 0 runs is not read
    again."""

    runs: int = 1
    fast_below_us: float = 10.0
    fast_runs: int = 3
    slow_from_us: float = 1000.0
    slow_runs: int = 3
    slow_min_iters: int = 0

    def band(self, winner_us: float) -> str:
        if winner_us < self.fast_below_us:
            return "fast"
        if winner_us >= self.slow_from_us:
            return "slow"
        return "mid"

    def runs_of(self, winner_us: float) -> int:
        return {"fast": self.fast_runs, "mid": self.runs, "slow": self.slow_runs}[
            self.band(winner_us)
        ]

    def min_iters_of(self, winner_us: float) -> int:
        return self.slow_min_iters if self.band(winner_us) == "slow" else 0


# The values at which the keys for readings that carry an offset per bench
# process change nothing: no absolute floor, one copy, no shift.
NEUTRAL_REREADS: Dict[str, Any] = {
    "tolerance_us": 0.0,
    "inline_copies": 1,
    "align_to_original": False,
}
# stage03b's `bench.rebench` keys and their defaults; `enabled` is the
# orchestrator's.
REREAD_DEFAULTS: Dict[str, Any] = {
    "tolerance": DEFAULT_TOLERANCE,
    "tolerance_us": 0.0,
    "max_kernels": DEFAULT_MAX_KERNELS,
    **{f.name: f.default for f in dataclasses.fields(Bands)},
    "inline_copies": 1,
    "with_original": True,
    "align_to_original": False,
    "sacrificial_kernel": True,
    "gemm_order": "shuffled",
}
# What stage03b did before the bands, the merge with the original reading,
# the sacrificial copy and the GEMM shuffle (README: Near-winner re-reads);
# every key is pinned, so later default changes leave it as it was.
PREDATING_REREADS: Dict[str, Any] = {
    "tolerance": 0.03,
    "runs": 3,
    "fast_runs": 3,
    "slow_runs": 3,
    "slow_min_iters": 0,
    "with_original": False,
    "sacrificial_kernel": False,
    "gemm_order": "listed",
    **NEUTRAL_REREADS,
}
# The slow band's settings before they became three re-reads in the line's
# own windows (README: Near-winner re-reads).
PREDATING_SLOW_BAND: Dict[str, Any] = {"slow_runs": 1, "slow_min_iters": 10}


def bands_of(options: Mapping[str, Any]) -> Bands:
    """The Bands of stage03b's resolved options (REREAD_DEFAULTS keys)."""
    return Bands(**{f.name: options[f.name] for f in dataclasses.fields(Bands)})


def align_copies_warning(align: bool, copies: int) -> Optional[str]:
    """What `align_to_original` with `copies` inline copies leaves as read,
    when a set of two candidates gives a run fewer than MIN_ALIGN_DIFFS
    differences; None otherwise."""
    if not align or 2 * int(copies) >= MIN_ALIGN_DIFFS:
        return None
    return (
        f"align_to_original with inline_copies {int(copies)} leaves the re-reads "
        f"of a GEMM with two near-winners as read: a run's offset needs at least "
        f"{MIN_ALIGN_DIFFS} differences from the original timings, which "
        f"inline_copies {-(-MIN_ALIGN_DIFFS // 2)} gives such a GEMM"
    )


@dataclass(frozen=True)
class Candidate:
    rank: int
    sol_index: int
    us: float
    skipped: bool


@dataclass
class TieSet:
    """The near-tie set of one GEMM: its candidates in rank order."""

    shape: Shape
    winner_us: float
    winner_sol: int
    candidates: List[Candidate]

    @property
    def sol_indices(self) -> List[int]:
        return [c.sol_index for c in self.candidates]

    @property
    def has_skip_row(self) -> bool:
        return any(c.skipped for c in self.candidates)


def _row_us(fields: Mapping[str, str]) -> Optional[float]:
    try:
        v = float(fields["us"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def tested_latencies(
    prob: Problem, blocked: AbstractSet[int] = frozenset()
) -> Dict[int, float]:
    """{solution index: latency} of a problem's tested rows (the fastest
    row of a repeated index), but those of the `blocked` solutions."""
    out: Dict[int, float] = {}
    for t in prob.tested:
        us = _row_us(t.fields)
        if t.sol_index is None or us is None or t.sol_index in blocked:
            continue
        if t.sol_index not in out or us < out[t.sol_index]:
            out[t.sol_index] = us
    return out


def blocked_solutions(
    logs: Iterable[Sequence[str]],
    blocklist: Any,
    ids: Optional[Mapping[int, str]] = None,
) -> Set[int]:
    """The solution indices of the logs' kernels that `blocklist`
    (`lib.correctness.Blocklist`, or None) names: through `ids` ({solution
    index: kernel id} of the library the logs were benched with) when
    given, else by the solution names their tested rows print."""
    if not blocklist:
        return set()
    if ids is not None:
        return blocklist.blocked_indices(ids)
    out: Set[int] = set()
    for lines in logs:
        for prob in iter_problems(lines):
            for t in prob.tested:
                if t.sol_index is not None and blocklist.blocks_name(t.sol_name):
                    out.add(t.sol_index)
    return out


def tolerance_limit(
    winner_us: float, tolerance: float, tolerance_us: float = 0.0
) -> float:
    """The latency up to which a kernel is within the tolerance of a winner
    of `winner_us`: `tolerance` (relative) above it, or `tolerance_us` above
    it when that is wider. With `tolerance_us` 0 the relative limit alone,
    to the bit."""
    winner_us = float(winner_us)
    return max(winner_us * (1.0 + float(tolerance)), winner_us + float(tolerance_us))


def near_ties(
    prob: Problem,
    tolerance: float = DEFAULT_TOLERANCE,
    max_kernels: int = DEFAULT_MAX_KERNELS,
    blocked: AbstractSet[int] = frozenset(),
    tolerance_us: float = 0.0,
) -> Optional[TieSet]:
    """The near-tie set of a complete problem: tested and skip rows within
    the tolerance of the fastest tested row (`tolerance_limit`: `tolerance`
    relative, or `tolerance_us` when wider), at most `max_kernels` of them
    (the fastest), in rank order, rows of the `blocked` solutions left out;
    None for an incomplete problem, a problem without a shape or a timed
    row, or a set of one."""
    if not prob.complete:
        return None
    shape = prob.shape()
    tested = tested_latencies(prob, blocked)
    if shape is None or not tested:
        return None
    winner_sol = min(tested, key=lambda s: (tested[s], s))
    winner_us = tested[winner_sol]
    limit = tolerance_limit(winner_us, tolerance, tolerance_us)
    best: Dict[int, Candidate] = {}
    for t in prob.tested:
        us = _row_us(t.fields)
        if t.sol_index is None or t.sol_index in blocked:
            continue
        if us is not None and us <= limit:
            if t.sol_index not in best or us < best[t.sol_index].us:
                best[t.sol_index] = Candidate(t.rank, t.sol_index, us, False)
    for s in prob.skipped:
        us = s.warm_up_us
        if s.sol_index is None or s.sol_index in blocked:
            continue
        if us is None or us <= 0 or us > limit:
            continue
        if s.sol_index not in best:
            best[s.sol_index] = Candidate(s.rank, s.sol_index, us, True)
    if len(best) < 2:
        return None
    kept = sorted(best.values(), key=lambda c: (c.us, c.rank))[: max(2, max_kernels)]
    return TieSet(
        shape=shape,
        winner_us=winner_us,
        winner_sol=winner_sol,
        candidates=sorted(kept, key=lambda c: c.rank),
    )


def rebench_line(
    line: str,
    sol_indices: Sequence[int],
    *,
    rotating_mib: Optional[int] = None,
) -> str:
    """`line` restamped to time exactly `sol_indices`, in that order: the
    explicit list (which overrides the algo method), no skip screen, and a
    rotating buffer (`rotating_mib` when the line has none). The line keeps
    its adaptive timing knobs."""
    out = set_line_knob(line, "solution_index_list", format_solution_list(sol_indices))
    out = set_line_knob(out, "skip_slow_solution_ratio", 0)
    out = set_line_knob(out, "requested_solution_num", -1)
    for key in ("algo_method", "skip_screen_prepass"):
        out = drop_line_knob(out, key)
    m = _ROTATING_RE.search(out)
    if (m is None or int(m.group(1)) == 0) and rotating_mib:
        out = set_line_knob(out, "rotating", int(rotating_mib))
    return out


def reread_line(
    line: str,
    sol_indices: Sequence[int],
    *,
    rotating_mib: Optional[int] = None,
    sacrificial: bool = False,
    min_iters: int = 0,
) -> str:
    """`rebench_line` of `sol_indices`, with `sacrificial` led by a copy of
    the first, whose row the log readers set aside (lib/bench_log.py: the
    first timing of a line reads slow) as long as the bench takes one more,
    and its windows at least `min_iters` iterations long when the line asks
    fewer."""
    from .planner import MAX_BENCH_LIST

    kernels = list(sol_indices)
    if sacrificial and kernels and len(kernels) < MAX_BENCH_LIST:
        kernels = [kernels[0], *kernels]
    out = rebench_line(line, kernels, rotating_mib=rotating_mib)
    m = _MIN_ITERS_RE.search(out)
    if min_iters > 0 and (m is None or int(m.group(1)) < min_iters):
        out = set_line_knob(out, "min_iters", int(min_iters))
    return out


def rotate(items: Sequence[int], run: int) -> List[int]:
    """`items` rotated left by `run` places."""
    if not items:
        return []
    k = run % len(items)
    return list(items[k:]) + list(items[:k])


def run_order(
    items: Sequence[int], run: int, order_seed: Optional[int], key: Sequence[Any]
) -> List[int]:
    """The order run `run` benches `items` in: shuffled with `order_seed`
    and `key` (`lib.planner.shuffled`; as listed when it is None), then
    rotated by `run` places, so that over the runs every item takes another
    place."""
    if order_seed is None:
        return rotate(items, run)
    from .planner import shuffled

    return rotate(shuffled(items, order_seed, *key), run)


def reread_list(
    items: Sequence[int],
    run: int,
    copies: int,
    order_seed: Optional[int],
    key: Sequence[Any],
) -> List[int]:
    """The kernels run `run` lists to read `items` `copies` times in one
    line: copy j in the order of `run_order` at place `run * copies + j`,
    the copies one after another, so that each reading of a kernel takes
    another place, in the line and over the runs. One copy is `run_order`
    itself. A set of two or more never lists one kernel first and second,
    which the log readers would take for a sacrificial copy."""
    n = max(1, int(copies))
    return [k for j in range(n) for k in run_order(items, run * n + j, order_seed, key)]


# ── log rows ─────────────────────────────────────────────────────────────────


def read_log_lines(text: str) -> List[str]:
    """The lines of a log, each ending with a newline, so that row blocks
    can move within and between logs."""
    return [ln if ln.endswith("\n") else ln + "\n" for ln in text.splitlines(True)]


def _values_index(lines: Sequence[str], header_idx: int) -> int:
    """Index of the values line the parser reads after a row header: the
    next non-empty line."""
    i = header_idx + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def row_block(lines: Sequence[str], header_line_no: int) -> Tuple[int, int]:
    """[start, end) indices of the row whose header is at 1-based
    `header_line_no`: the header, its values and the `--` lines after
    them."""
    start = header_line_no - 1
    end = _values_index(lines, start) + 1
    while end < len(lines) and lines[end].strip().startswith("--"):
        end += 1
    return start, end


def with_rank(block: Sequence[str], rank: int) -> List[str]:
    """A row block whose header carries `rank`."""
    out = list(block)
    if out:
        out[0] = _RANK_RE.sub(lambda m: f"{m.group(1)}[{rank}]:", out[0], count=1)
    return out


def with_latency(block: Sequence[str], us: float) -> List[str]:
    """A row block whose values give `us` in the header's `us` column; the
    block as it is when its header has none or its values are short."""
    out = list(block)
    head = HEADER_RE.match(out[0].strip()) if out else None
    if head is None:
        return out
    columns = [c.strip() for c in head.group(2).split(",")]
    if "us" not in columns:
        return out
    i = _values_index(out, 0)
    if i >= len(out):
        return out
    text = out[i].rstrip("\n")
    indent = text[: len(text) - len(text.lstrip())]
    values = text.strip().split(",")
    col = columns.index("us")
    if col >= len(values):
        return out
    values[col] = f"{us:.6g}"
    out[i] = indent + ",".join(values) + "\n"
    return out


@dataclass
class Sample:
    """One re-bench timing of one candidate: its latency and row block."""

    us: float
    block: List[str]
    run: int


@dataclass(frozen=True)
class Alignment:
    """How `merge_log` shifts one GEMM's re-reads to its original timings
    (`Samples.alignment`): `offsets` ({run: offset}) of the runs it
    shifts, `unaligned` ({run: reason}, UNALIGNED_REASONS) of the other
    runs that read the GEMM, which merge as they were read."""

    offsets: Dict[int, float] = field(default_factory=dict)
    unaligned: Dict[int, str] = field(default_factory=dict)


@dataclass
class Samples:
    """Re-bench timings by shape and solution index."""

    by_shape: Dict[Shape, Dict[int, List[Sample]]] = field(default_factory=dict)

    def add_log(self, lines: Sequence[str], run: int) -> int:
        """Add every timed row of the complete problems of one log; returns
        the number of rows added."""
        n = 0
        for prob in iter_problems(lines):
            shape = prob.shape()
            if not prob.complete or shape is None:
                continue
            per_sol = self.by_shape.setdefault(shape, {})
            for t in prob.tested:
                us = _row_us(t.fields)
                if t.sol_index is None or us is None:
                    continue
                start, end = row_block(lines, t.line_no)
                per_sol.setdefault(t.sol_index, []).append(
                    Sample(us, list(lines[start:end]), run)
                )
                n += 1
        return n

    def median(self, shape: Shape, sol_index: int) -> Optional[Sample]:
        """The sample with the (lower) median latency, or None."""
        samples = self.by_shape.get(shape, {}).get(sol_index) or []
        if not samples:
            return None
        target = statistics.median_low([s.us for s in samples])
        return next(s for s in samples if s.us == target)

    def runs(self, shape: Shape) -> List[int]:
        """The runs that timed a kernel of `shape`, in order."""
        per_sol = self.by_shape.get(shape, {})
        return sorted({s.run for samples in per_sol.values() for s in samples})

    def _differences(
        self, shape: Shape, original: Mapping[int, float]
    ) -> Dict[int, List[float]]:
        """{run: [reading - original latency]} over the run's readings of
        the kernels of `original` ({solution index: latency})."""
        diffs: Dict[int, List[float]] = {}
        for sol, samples in self.by_shape.get(shape, {}).items():
            if sol not in original:
                continue
            for s in samples:
                diffs.setdefault(s.run, []).append(s.us - float(original[sol]))
        return dict(sorted(diffs.items()))

    def run_offsets(
        self, shape: Shape, original: Mapping[int, float]
    ) -> Dict[int, float]:
        """{run: offset} of the runs that read a kernel of `original`
        ({solution index: latency} of one original reading of `shape`): the
        median of the differences between the run's readings of those
        kernels and their `original` latencies, which is how much later the
        run's bench process read them than the original one did (negative:
        earlier). `alignment` says which of them `merge_log` applies."""
        return {
            r: statistics.median(d)
            for r, d in self._differences(shape, original).items()
        }

    def alignment(
        self,
        shape: Shape,
        original: Mapping[int, float],
        min_diffs: int = MIN_ALIGN_DIFFS,
    ) -> Alignment:
        """The shift of the re-reads of `shape` to its `original` timings
        ({solution index: latency} of the set's kernels in one original
        reading): the `run_offsets` of the runs whose offset rests on at
        least `min_diffs` differences and leaves every reading of the run,
        of any kernel, above 0. Every other run that timed a kernel of
        `shape` stays as it was read, with the reason (UNALIGNED_FEW,
        UNALIGNED_NONPOSITIVE): an offset of two differences is their mean,
        which one slow re-read raises by half its excess, and the shift
        then takes as much off the other candidate's reading, down to 0 or
        below."""
        diffs = self._differences(shape, original)
        lowest: Dict[int, float] = {}
        for samples in self.by_shape.get(shape, {}).values():
            for s in samples:
                lowest[s.run] = min(s.us, lowest.get(s.run, math.inf))
        offsets: Dict[int, float] = {}
        unaligned: Dict[int, str] = {}
        for run in sorted(lowest):
            d = diffs.get(run, [])
            if len(d) < max(1, int(min_diffs)):
                unaligned[run] = UNALIGNED_FEW
                continue
            offset = statistics.median(d)
            if lowest[run] - offset <= 0:
                unaligned[run] = UNALIGNED_NONPOSITIVE
                continue
            offsets[run] = offset
        return Alignment(offsets, unaligned)

    def label(
        self,
        shape: Shape,
        sol_index: int,
        original: Optional[Tuple[float, List[str]]] = None,
        offsets: Optional[Mapping[int, float]] = None,
    ) -> Optional[Tuple[float, List[str]]]:
        """(latency, row block) of a candidate's re-reads merged with its
        `original` (latency, row block) timing: the median of all of them,
        each re-read less the offset of its run in `offsets`
        (`Alignment.offsets`), and the row of the lower middle one with that
        latency; None without a re-read."""
        samples = self.by_shape.get(shape, {}).get(sol_index) or []
        if not samples:
            return None
        shift = offsets or {}
        readings = [(s.us - shift.get(s.run, 0.0), s.block, s.us) for s in samples]
        if original is not None:
            readings.append((original[0], original[1], original[0]))
        readings.sort(key=lambda r: r[0])
        us = statistics.median(r[0] for r in readings)
        _, rep_block, rep_us = readings[(len(readings) - 1) // 2]
        return us, (list(rep_block) if rep_us == us else with_latency(rep_block, us))


# ── merge ────────────────────────────────────────────────────────────────────


@dataclass
class ProblemChange:
    """What `merge_log` did to one problem: {solution: (latency before,
    label)} of its replaced rows, its winners before and after, the offset
    (`Samples.alignment`) each run's re-reads were shifted by before they
    merged with the original timings, when they were, and with `align` why
    each other run that read it was left as read (`unaligned`, {run:
    reason})."""

    shape: Shape
    replaced: Dict[int, Tuple[float, float]]
    winner_before: int
    winner_after: int
    offsets: Dict[int, float] = field(default_factory=dict)
    unaligned: Dict[int, str] = field(default_factory=dict)

    @property
    def winner_changed(self) -> bool:
        return self.winner_before != self.winner_after


def merge_log(
    lines: Sequence[str],
    tie_sets: Mapping[Shape, TieSet],
    samples: Samples,
    with_original: bool = False,
    blocked: AbstractSet[int] = frozenset(),
    align: bool = False,
) -> Tuple[List[str], List[ProblemChange]]:
    """The log `lines` with every near-tie candidate of `tie_sets` replaced
    by the row of its readings' median (see the module docstring; the
    original timing among them `with_original`, unless the set holds a skip
    row, and then with `align` each run's re-reads shifted by that run's
    offset against the problem's original timings of the set, where
    `Samples.alignment` allows it), and one ProblemChange per problem that
    changed; the winners before and after, and the `Winner:` block, are the
    fastest rows of solutions not `blocked`."""

    def merged(
        shape: Shape,
        sol: int,
        original: Optional[Tuple[float, List[str]]],
        offsets: Mapping[int, float],
    ) -> Optional[Tuple[float, List[str]]]:
        if with_original:
            return samples.label(shape, sol, original, offsets)
        med = samples.median(shape, sol)
        return None if med is None else (med.us, med.block)

    edits: Dict[int, Tuple[int, List[str]]] = {}
    changes: List[ProblemChange] = []
    for prob in iter_problems(lines):
        shape = prob.shape()
        ties = tie_sets.get(shape) if shape is not None else None
        if ties is None or not prob.complete:
            continue
        wanted = set(ties.sol_indices)
        alone = ties.has_skip_row
        tested = tested_latencies(prob, blocked)
        latency = dict(tested)
        alignment = Alignment()
        if with_original and align and alone:
            alignment = Alignment(
                unaligned={r: UNALIGNED_SKIP_ROW for r in samples.runs(shape)}
            )
        elif with_original and align:
            alignment = samples.alignment(
                shape, {s: us for s, us in tested.items() if s in wanted}
            )
        offsets = alignment.offsets
        replaced: Dict[int, Tuple[float, float]] = {}
        blocks: Dict[int, List[str]] = {}
        ranks: Dict[int, int] = {}
        for t in prob.tested:
            if t.sol_index is None or t.sol_index in ranks:
                continue
            start, end = row_block(lines, t.line_no)
            ranks[t.sol_index] = t.rank
            blocks[t.sol_index] = list(lines[start:end])
            if t.sol_index not in wanted:
                continue
            us = _row_us(t.fields)
            got = merged(
                shape,
                t.sol_index,
                None if us is None or alone else (us, blocks[t.sol_index]),
                offsets,
            )
            if got is None:
                continue
            new_block = with_rank(got[1], t.rank)
            edits[start] = (end, new_block)
            blocks[t.sol_index] = new_block
            replaced[t.sol_index] = (latency[t.sol_index], got[0])
            latency[t.sol_index] = got[0]
        for s in prob.skipped:
            if s.sol_index is None or s.sol_index not in wanted:
                continue
            if s.sol_index in ranks:
                continue
            got = merged(shape, s.sol_index, None, offsets)
            if got is None:
                continue
            new_block = with_rank(got[1], s.rank)
            edits[s.line_no - 1] = (s.line_no, new_block)
            ranks[s.sol_index] = s.rank
            blocks[s.sol_index] = new_block
            replaced[s.sol_index] = (float(s.warm_up_us or math.nan), got[0])
            latency[s.sol_index] = got[0]
        if not replaced:
            continue
        after = min(latency, key=lambda k: (latency[k], k))
        before = min(tested, key=lambda k: (tested[k], k)) if tested else after
        if prob.winner is not None:
            start, end = row_block(lines, prob.winner.line_no)
            edits[start] = (end, with_rank(blocks[after], ranks[after]))
        changes.append(
            ProblemChange(
                shape,
                replaced,
                before,
                after,
                dict(offsets),
                dict(alignment.unaligned),
            )
        )
    out: List[str] = []
    i = 0
    while i < len(lines):
        if i in edits:
            end, block = edits[i]
            out.extend(block)
            i = end
        else:
            out.append(lines[i])
            i += 1
    return out, changes


# ── pick re-bench ────────────────────────────────────────────────────────────
#
# A deployed pick without a timed row of its own (a skip-screened kernel, or
# one off a planned list) has no latency the evaluation can trust. The pick
# re-bench times it in one line with the GEMM's timed winner, so the ratio
# of their medians cancels device and day drift, and lib/evaluate charges
# the pick by that ratio (`pick_measurement="rebench"`).

DEFAULT_EXTRA_KERNELS = 2


@dataclass(frozen=True)
class PickRatio:
    """Re-bench of one GEMM's pick: the median latencies of the pick and of
    the GEMM's winner over the runs that timed both."""

    shape: Shape
    pick_sol: int
    winner_sol: int
    pick_us: float
    winner_us: float
    n_runs: int

    @property
    def ratio(self) -> float:
        """winner / pick latency; above 1 when the pick beat the winner."""
        return self.winner_us / self.pick_us


def _shape_of_gemm(g: Mapping[str, Any]) -> Shape:
    return (int(g["m"]), int(g["n"]), int(g["k"]), int(g.get("batch_count", 1)))


def _timed_of_gemm(g: Mapping[str, Any]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in g.get("candidates", ()):
        sid, us = int(c.get("sol_idx_global", -1)), float(c.get("us", 0) or 0)
        if sid < 0 or c.get("is_skip") or not us > 0 or not math.isfinite(us):
            continue
        if sid not in out or us < out[sid]:
            out[sid] = us
    return out


def pick_kernel_list(
    pick: int, timed: Mapping[int, float], extra: int = DEFAULT_EXTRA_KERNELS
) -> List[int]:
    """[pick, the fastest timed kernel, then up to `extra` other timed
    kernels, fastest first]; an empty list without a timed kernel."""
    if not timed:
        return []
    order = sorted(timed, key=lambda s: (timed[s], s))
    others = [s for s in order if s != int(pick)]
    return [int(pick)] + others[: 1 + max(0, int(extra))]


def rebench_picks(
    gemms: Sequence[Mapping[str, Any]],
    picks: Mapping[Shape, int],
    lines: Mapping[Shape, str],
    run_bench: Callable[[int, List[str]], Iterable[Sequence[str]]],
    *,
    runs: int = DEFAULT_RUNS,
    extra: int = DEFAULT_EXTRA_KERNELS,
    rotating_mib: Optional[Callable[[Shape], Optional[int]]] = None,
    order_seed: Optional[int] = None,
) -> Tuple[Dict[Shape, PickRatio], Samples]:
    """Bench the pick of every GEMM of `gemms` in `picks` ({shape: solution
    index}) with the GEMM's timed winner and up to `extra` other timed
    kernels (`pick_kernel_list`), `runs` times, one `rebench_line` of the
    GEMM's bench line (`lines`) per GEMM, each list in the order of
    `run_order` (shuffled with `order_seed` and the shape, then rotated by
    one place per run). `run_bench(run, yaml_lines)` benches the lines and
    returns the logs. Returns ({shape: PickRatio} of the GEMMs whose pick
    and winner some run timed together, the samples). The medians are
    `median_low` over those runs, so each ratio pairs timings of the same
    runs."""
    jobs: Dict[Shape, List[int]] = {}
    winners: Dict[Shape, int] = {}
    for g in gemms:
        shape = _shape_of_gemm(g)
        if shape not in picks or shape not in lines or shape in jobs:
            continue
        timed = _timed_of_gemm(g)
        sols = pick_kernel_list(picks[shape], timed, extra)
        if len(sols) < 2:
            continue
        jobs[shape] = sols
        winners[shape] = min(timed, key=lambda s: (timed[s], s))
    samples = Samples()
    for r in range(int(runs) if jobs else 0):
        yaml_lines = [
            rebench_line(
                lines[shape],
                run_order(sols, r, order_seed, ("rebench", *shape)),
                rotating_mib=rotating_mib(shape) if rotating_mib else None,
            )
            for shape, sols in jobs.items()
        ]
        for log in run_bench(r, yaml_lines):
            samples.add_log(list(log), r)
    out: Dict[Shape, PickRatio] = {}
    for shape in jobs:
        per_sol = samples.by_shape.get(shape, {})
        pick, win = picks[shape], winners[shape]
        both = {s.run for s in per_sol.get(pick, [])} & {
            s.run for s in per_sol.get(win, [])
        }
        if not both:
            continue

        def med(sol: int) -> float:
            return statistics.median_low(s.us for s in per_sol[sol] if s.run in both)

        out[shape] = PickRatio(shape, pick, win, med(pick), med(win), len(both))
    return out, samples


def attach_rebench(
    gemms: Iterable[Dict[str, Any]], ratios: Mapping[Shape, PickRatio]
) -> int:
    """Set `rebench` ({solution index: ratio}) on every GEMM with a
    re-benched pick, the input of `pick_measurement="rebench"` in
    lib/evaluate; returns the number of GEMMs."""
    n = 0
    for g in gemms:
        r = ratios.get(_shape_of_gemm(g))
        if r is None:
            continue
        g.setdefault("rebench", {})[int(r.pick_sol)] = float(r.ratio)
        n += 1
    return n


def blocks_per_gpu(
    n_lines: int,
    n_gpus: int,
    lines_per_block: int = REBENCH_LINES_PER_BLOCK,
    cap: int = MAX_BLOCKS_PER_GPU,
) -> int:
    """stage03's --blocks-per-gpu for `n_lines` bench lines on `n_gpus`:
    one block per `lines_per_block` lines of a GPU's share, at least one,
    at most `cap`. Every block is a bench process of its own, whose start-up
    costs seconds, so a small call runs one block per GPU."""
    share = math.ceil(max(0, int(n_lines)) / max(1, int(n_gpus)))
    return max(1, min(int(cap), math.ceil(share / max(1, int(lines_per_block)))))


def stage03_runner(
    bench_binary: Path,
    work_dir: Path,
    *,
    devices: str = "0",
    extra_args: Sequence[str] = (),
    returncodes: Optional[List[int]] = None,
) -> Callable[[int, List[str]], List[List[str]]]:
    """`run_bench` for `rebench_picks` that benches each run through stage03
    (crash recovery, stall watchdog) in `<work_dir>/run_<r>/`, without a
    warm-up shape (its processes still lead with the warm-up line the
    config of `extra_args` gives), its lines cut into `blocks_per_gpu`
    blocks per device unless `extra_args` sets --blocks-per-gpu, and returns
    its block logs; each run's stage03 exit code is appended to
    `returncodes`."""
    stage03 = (
        Path(__file__).resolve().parent.parent
        / "stages"
        / "stage03_load_balance_offline_tuning.py"
    )

    def run(r: int, yaml_lines: List[str]) -> List[List[str]]:
        run_dir = Path(work_dir) / f"run_{r}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        text = "".join(ln.rstrip("\n") + "\n" for ln in yaml_lines)
        (run_dir / "shapes.yaml").write_text(text)
        cmd = [sys.executable, str(stage03), "--in-dir", str(run_dir)]
        cmd += ["--out-dir", str(run_dir / "stage03")]
        cmd += ["--bench-binary", str(bench_binary), "--devices", str(devices)]
        if "--blocks-per-gpu" not in extra_args:
            n_gpus = len([d for d in str(devices).split(",") if d.strip()])
            blocks = blocks_per_gpu(len(yaml_lines), n_gpus)
            cmd += ["--blocks-per-gpu", str(blocks)]
        cmd += ["--no-warmup-pass", "--quiet", *extra_args]
        with (run_dir / "stage03.log").open("w") as log:
            rc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=procs.die_with_parent(),
            ).returncode
        if returncodes is not None:
            returncodes.append(rc)
        return [
            read_log_lines(p.read_text(errors="replace"))
            for p in sorted((run_dir / "stage03" / "logs").glob("block_*.log"))
        ]

    return run


def tie_sets_of_logs(
    logs: Iterable[Sequence[str]],
    tolerance: float = DEFAULT_TOLERANCE,
    max_kernels: int = DEFAULT_MAX_KERNELS,
    blocked: AbstractSet[int] = frozenset(),
    tolerance_us: float = 0.0,
) -> Tuple[Dict[Shape, TieSet], int]:
    """{shape: TieSet} over the complete problems of several logs (the
    union of the sets of a shape benched more than once; `near_ties`, the
    `blocked` solutions left out), and the number of complete problems."""
    out: Dict[Shape, TieSet] = {}
    n_complete = 0
    for lines in logs:
        for prob in iter_problems(lines):
            if prob.complete:
                n_complete += 1
            ties = near_ties(prob, tolerance, max_kernels, blocked, tolerance_us)
            if ties is None:
                continue
            prev = out.get(ties.shape)
            if prev is None:
                out[ties.shape] = ties
                continue
            have = set(prev.sol_indices)
            extra = [c for c in ties.candidates if c.sol_index not in have]
            prev.candidates = sorted(prev.candidates + extra, key=lambda c: c.rank)
    return out, n_complete
