# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parser for hipblaslt-bench output as captured in stage02 / stage03 logs.

For each problem (one yaml entry) the bench prints a preamble, then one record
per candidate kernel in rank order (the library's ranking, so rank 0 is the
library's own pick), then, when there was more than one candidate, the
fastest one again after `Winner:`:

    Rotating buffer <N> MiB. Needed Size: ...       (only when rotating > 0)
    Adaptive timing: warmup ...                     (only with adaptive)
    Is supported <N> / Total solutions: <T>
    [<rank>]:transA,transB,...,us[,...]             tested candidate: header,
        T,N,...                                     values,
        --Solution index: <index>                   and identity
        --Solution name:  <name>
        --kernel name:    <name>
    Skip solution: <rank> (best warm-up = <x> us , warm-up = <y> us, skip ratio = <r>, solution index = <index>)
    Winner:
    [<rank>]:...

A problem without candidates prints `NO solution found` instead of the
`Is supported` line. Every process starts with a `hipBLASLt version:` banner,
and stage03 writes `*** ...` annotations between sub-runs; both end any open
problem. Values of a tested row are mapped by the header printed with it:
optional columns (splitK, wgm, efficiency counters, adaptive statistics) move
the others.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

PREAMBLE_MARKERS = ("Rotating buffer", "Adaptive timing:", "Is supported")
PROCESS_BANNER = "hipBLASLt version"
ANNOTATION_PREFIX = "***"
NO_SOLUTION_MARK = "NO solution found"
WINNER_MARK = "Winner:"
SKIP_MARK = "Skip solution"

# Warm-up times come in fixed notation or in default floating-point notation
# with two significant digits (`1.2e+02`), and may be non-finite.
_NUM = r"[-+]?(?:inf|nan|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
SKIP_RE = re.compile(
    r"^Skip solution:\s*(?P<rank>\d+)\s*\(\s*best warm-up\s*=\s*(?P<best>"
    + _NUM
    + r")\s*us\s*,\s*warm-up\s*=\s*(?P<warm>"
    + _NUM
    + r")\s*us\s*,\s*skip ratio\s*=\s*(?P<ratio>"
    + _NUM
    + r")(?:\s*,\s*solution index\s*=\s*(?P<index>-?\d+))?\s*\)\s*$",
    re.IGNORECASE,
)
HEADER_RE = re.compile(r"^\[(\d+)\]:(.+)$")
SUPPORTED_RE = re.compile(r"^Is supported\s+(\d+)")
SOL_INDEX_RE = re.compile(r"^--Solution index:\s*(-?\d+)")
SOL_NAME_RE = re.compile(r"^--Solution name:\s*(\S+)")
KERNEL_NAME_RE = re.compile(r"^--kernel name:\s*(.*?)\s*$")


@dataclass
class TestedRow:
    rank: int
    fields: Dict[str, str]
    sol_index: Optional[int] = None
    sol_name: str = ""
    kernel_name: str = ""
    line_no: int = 0


@dataclass
class SkipRow:
    rank: int
    warm_up_us: Optional[float]
    best_warm_up_us: Optional[float]
    sol_index: Optional[int]
    line_no: int = 0


@dataclass
class Problem:
    line_no: int
    n_supported: Optional[int] = None
    tested: List[TestedRow] = field(default_factory=list)
    skipped: List[SkipRow] = field(default_factory=list)
    winner_seen: bool = False
    winner: Optional[TestedRow] = None
    no_solution: bool = False
    unparsed_skip_lines: int = 0

    @property
    def has_body(self) -> bool:
        return bool(
            self.tested
            or self.skipped
            or self.winner_seen
            or self.no_solution
            or self.unparsed_skip_lines
        )

    @property
    def ranks(self) -> Set[int]:
        return {r.rank for r in self.tested} | {s.rank for s in self.skipped}

    @property
    def complete(self) -> bool:
        """Every candidate was reported: `Winner:` was printed, or every rank
        0..N-1 of `Is supported N` appeared (a single candidate gets no
        `Winner:` line)."""
        if self.winner_seen:
            return True
        if not self.n_supported:
            return False
        return self.ranks.issuperset(range(self.n_supported))

    @property
    def finished(self) -> bool:
        """The bench got past this problem: it is complete or had no
        solution."""
        return self.complete or self.no_solution

    def shape(self) -> Optional[Tuple[int, int, int, int]]:
        """(m, n, k, batch_count) from the first tested row that has them."""
        rows = self.tested + ([self.winner] if self.winner is not None else [])
        for row in rows:
            try:
                return (
                    int(row.fields["m"]),
                    int(row.fields["n"]),
                    int(row.fields["k"]),
                    int(row.fields.get("batch_count", 1)),
                )
            except (KeyError, ValueError):
                continue
        return None


@dataclass
class LogStats:
    problems: int = 0
    skip_lines: int = 0
    unparsed_skip_lines: int = 0
    unparsed_skip_examples: List[str] = field(default_factory=list)
    malformed_rows: int = 0

    _MAX_EXAMPLES = 5

    def note_unparsed_skip(self, line: str) -> None:
        self.unparsed_skip_lines += 1
        if len(self.unparsed_skip_examples) < self._MAX_EXAMPLES:
            self.unparsed_skip_examples.append(line)


def _is_reset_line(s: str) -> bool:
    return s.startswith(PROCESS_BANNER) or s.startswith(ANNOTATION_PREFIX)


def _is_preamble_line(s: str) -> bool:
    return s.startswith(PREAMBLE_MARKERS)


def _is_body_line(s: str) -> bool:
    return (
        s.startswith(SKIP_MARK)
        or s.startswith(WINNER_MARK)
        or HEADER_RE.match(s) is not None
        or NO_SOLUTION_MARK in s
    )


class ProblemBoundaryTracker:
    """Incremental form of the problem-boundary rule of `iter_problems`:
    `feed(line)` is True when `line` opens a new problem."""

    def __init__(self) -> None:
        self._open = False
        self._has_body = False

    def feed(self, line: str) -> bool:
        s = line.strip()
        if _is_reset_line(s):
            self._open = False
            self._has_body = False
            return False
        if _is_preamble_line(s):
            if not self._open or self._has_body:
                self._open = True
                self._has_body = False
                return True
            return False
        if _is_body_line(s):
            opened = not self._open
            self._open = True
            self._has_body = True
            return opened
        return False


def _finite(text: str) -> Optional[float]:
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class _Parser:
    def __init__(self, stats: LogStats) -> None:
        self.stats = stats
        self.cur: Optional[Problem] = None
        self.pending: Optional[Tuple[int, List[str], int]] = None
        self.last_row: Optional[TestedRow] = None
        self.in_winner = False
        self.done: List[Problem] = []

    def close(self) -> None:
        if self.pending is not None:
            self.stats.malformed_rows += 1
            self.pending = None
        if self.cur is not None:
            self.done.append(self.cur)
            self.stats.problems += 1
        self.cur = None
        self.last_row = None
        self.in_winner = False

    def _current(self, line_no: int) -> Problem:
        if self.cur is None:
            self.cur = Problem(line_no=line_no)
        return self.cur

    def feed(self, line_no: int, raw: str) -> None:
        s = raw.strip()
        if not s:
            return
        if self.pending is not None:
            rank, cols, hdr_line = self.pending
            self.pending = None
            values = [v.strip() for v in s.split(",")]
            if len(values) == len(cols) and not _is_body_line(s):
                row = TestedRow(
                    rank=rank, fields=dict(zip(cols, values)), line_no=hdr_line
                )
                prob = self._current(hdr_line)
                if self.in_winner:
                    prob.winner = row
                else:
                    prob.tested.append(row)
                self.last_row = row
                return
            self.stats.malformed_rows += 1
        if _is_reset_line(s):
            self.close()
            return
        if _is_preamble_line(s):
            if self.cur is not None and self.cur.has_body:
                self.close()
            prob = self._current(line_no)
            m = SUPPORTED_RE.match(s)
            if m:
                prob.n_supported = int(m.group(1))
            return
        if NO_SOLUTION_MARK in s:
            self._current(line_no).no_solution = True
            self.last_row = None
            return
        if s.startswith(WINNER_MARK):
            self._current(line_no).winner_seen = True
            self.in_winner = True
            self.last_row = None
            return
        if s.startswith(SKIP_MARK):
            prob = self._current(line_no)
            self.last_row = None
            self.stats.skip_lines += 1
            m = SKIP_RE.match(s)
            if m is None:
                prob.unparsed_skip_lines += 1
                self.stats.note_unparsed_skip(s)
                return
            prob.skipped.append(
                SkipRow(
                    rank=int(m.group("rank")),
                    warm_up_us=_finite(m.group("warm")),
                    best_warm_up_us=_finite(m.group("best")),
                    sol_index=(
                        None if m.group("index") is None else int(m.group("index"))
                    ),
                    line_no=line_no,
                )
            )
            return
        hm = HEADER_RE.match(s)
        if hm:
            self._current(line_no)
            cols = [c.strip() for c in hm.group(2).split(",")]
            self.pending = (int(hm.group(1)), cols, line_no)
            self.last_row = None
            return
        if self.last_row is None:
            return
        m = SOL_INDEX_RE.match(s)
        if m:
            self.last_row.sol_index = int(m.group(1))
            return
        m = SOL_NAME_RE.match(s)
        if m:
            self.last_row.sol_name = m.group(1)
            return
        m = KERNEL_NAME_RE.match(s)
        if m:
            self.last_row.kernel_name = m.group(1)


def iter_problems(
    lines: Iterable[str], stats: Optional[LogStats] = None
) -> Iterator[Problem]:
    """Yield every problem in `lines` in order, complete or not; callers decide
    what to do with incomplete ones (`Problem.complete`)."""
    parser = _Parser(stats if stats is not None else LogStats())
    for line_no, line in enumerate(lines, 1):
        parser.feed(line_no, line)
        if parser.done:
            yield from parser.done
            parser.done.clear()
    parser.close()
    yield from parser.done


def parse_log_file(path, stats: Optional[LogStats] = None) -> List[Problem]:
    with open(path, errors="replace") as f:
        return list(iter_problems(f, stats))


def count_finished_problems(lines: Iterable[str]) -> int:
    """How many problems of one bench process the bench got past: every
    problem followed by another one, plus the last one if it is finished."""
    problems = list(iter_problems(lines))
    if not problems:
        return 0
    return len(problems) - 1 + (1 if problems[-1].finished else 0)


# Lines that mean the GPU queue is dead and the process will make no further
# progress, so it is killed at once instead of waiting out the stall timeout.
# Not listed: errors printed during teardown after every shape was logged
# (e.g. a failing hipModuleUnload); stage03's recovery treats those runs as
# complete.
FATAL_GPU_MARKERS = (
    "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
    "aborting with error",
)


class LogProbe:
    """Incremental reader of the output one process appends to a log,
    starting at byte offset `start` (the log holds earlier sub-runs too)."""

    def __init__(self, path, start: int) -> None:
        self.path = path
        self.offset = start
        self.partial = ""
        self.tracker = ProblemBoundaryTracker()
        self.problem_starts = 0
        self.winner_seen = False
        self.fatal = False

    def poll(self) -> None:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return
        self.offset += len(chunk)
        lines = (self.partial + chunk.decode("utf-8", "replace")).split("\n")
        self.partial = lines.pop()
        for line in lines:
            if self.tracker.feed(line):
                self.problem_starts += 1
            if line.strip().startswith(WINNER_MARK):
                self.winner_seen = True
        if any(m in "\n".join(lines + [self.partial]) for m in FATAL_GPU_MARKERS):
            self.fatal = True

    def progressed(self) -> bool:
        """True once a problem has finished: a winner was printed or a second
        problem started."""
        self.poll()
        return self.winner_seen or self.problem_starts >= 2

    def fatal_seen(self) -> bool:
        self.poll()
        return self.fatal
