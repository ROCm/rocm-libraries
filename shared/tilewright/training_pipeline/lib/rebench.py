# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Near-tie re-bench: time again the candidates of each GEMM that came
within a tolerance of its winner, and take the median over the runs.

One bench pass cannot order kernels whose latencies differ by less than its
noise, and the training targets are built from that order. For every
complete problem of a stage03 log, the candidates whose latency (a tested
row's timing or a skip row's warm-up) is within `tolerance` of the winner
(the fastest tested row) form its near-tie set; a problem whose set holds
only the winner needs nothing. Each near-tie set is benched again
(`rebench_line`: an explicit `solution_index_list`, no skip screen, more
iterations, rotating buffers) in several runs, the list order rotating from
run to run, and every re-timed candidate is replaced in the log by the row
of its run with the median latency (`median_low`, always an actual
measurement), its rank kept: a skip row becomes a tested row, and the
`Winner:` block names the new fastest row (`merge_log`). The original
timings take no part in the median: they are the ones that put the
candidates in the set. Candidates a run did not time keep their original
rows.
"""
from __future__ import annotations

import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .bench_log import Problem, iter_problems
from .bench_yaml import drop_line_knob, format_solution_list, set_line_knob

Shape = Tuple[int, int, int, int]

DEFAULT_TOLERANCE = 0.03
DEFAULT_RUNS = 3
DEFAULT_ITERS_FACTOR = 2.0
DEFAULT_MAX_KERNELS = 16

_RANK_RE = re.compile(r"^(\s*)\[(\d+)\]:")
_ITER_KNOB_RE = {
    key: re.compile(rf"(?<![\w]){key}:\s*(\d+)") for key in ("iters", "cold_iters")
}
_ROTATING_RE = re.compile(r"(?<![\w])rotating:\s*(\d+)")


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


def _row_us(fields: Mapping[str, str]) -> Optional[float]:
    try:
        v = float(fields["us"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def tested_latencies(prob: Problem) -> Dict[int, float]:
    """{solution index: latency} of a problem's tested rows (the fastest
    row of a repeated index)."""
    out: Dict[int, float] = {}
    for t in prob.tested:
        us = _row_us(t.fields)
        if t.sol_index is None or us is None:
            continue
        if t.sol_index not in out or us < out[t.sol_index]:
            out[t.sol_index] = us
    return out


def near_ties(
    prob: Problem,
    tolerance: float = DEFAULT_TOLERANCE,
    max_kernels: int = DEFAULT_MAX_KERNELS,
) -> Optional[TieSet]:
    """The near-tie set of a complete problem: tested and skip rows within
    `tolerance` (relative) of the fastest tested row, at most `max_kernels`
    of them (the fastest), in rank order; None for an incomplete problem, a
    problem without a shape or a timed row, or a set of one."""
    if not prob.complete:
        return None
    shape = prob.shape()
    tested = tested_latencies(prob)
    if shape is None or not tested:
        return None
    winner_sol = min(tested, key=lambda s: (tested[s], s))
    winner_us = tested[winner_sol]
    limit = winner_us * (1.0 + float(tolerance))
    best: Dict[int, Candidate] = {}
    for t in prob.tested:
        us = _row_us(t.fields)
        if t.sol_index is not None and us is not None and us <= limit:
            if t.sol_index not in best or us < best[t.sol_index].us:
                best[t.sol_index] = Candidate(t.rank, t.sol_index, us, False)
    for s in prob.skipped:
        us = s.warm_up_us
        if s.sol_index is None or us is None or us <= 0 or us > limit:
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
    iters_factor: float = DEFAULT_ITERS_FACTOR,
    rotating_mib: Optional[int] = None,
) -> str:
    """`line` restamped to time exactly `sol_indices`, in that order: the
    explicit list (which overrides the algo method), no skip screen, the
    line's iteration counts scaled by `iters_factor` (at least one cold
    call), and a rotating buffer (`rotating_mib` when the line has none).
    Adaptive lines keep their own timing knobs."""
    out = set_line_knob(line, "solution_index_list", format_solution_list(sol_indices))
    out = set_line_knob(out, "skip_slow_solution_ratio", 0)
    out = set_line_knob(out, "requested_solution_num", -1)
    for key in ("algo_method", "skip_screen_prepass"):
        out = drop_line_knob(out, key)
    if not re.search(r"(?<![\w])adaptive:\s*true", out):
        for key, pat in _ITER_KNOB_RE.items():
            m = pat.search(out)
            if m is None:
                continue
            scaled = int(math.ceil(int(m.group(1)) * max(1.0, float(iters_factor))))
            out = set_line_knob(out, key, max(1, scaled))
    m = _ROTATING_RE.search(out)
    if (m is None or int(m.group(1)) == 0) and rotating_mib:
        out = set_line_knob(out, "rotating", int(rotating_mib))
    return out


def rotate(items: Sequence[int], run: int) -> List[int]:
    """`items` rotated left by `run` places."""
    if not items:
        return []
    k = run % len(items)
    return list(items[k:]) + list(items[:k])


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


@dataclass
class Sample:
    """One re-bench timing of one candidate: its latency and row block."""

    us: float
    block: List[str]
    run: int


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


# ── merge ────────────────────────────────────────────────────────────────────


@dataclass
class ProblemChange:
    """What `merge_log` did to one problem."""

    shape: Shape
    replaced: Dict[int, Tuple[float, float]]
    winner_before: int
    winner_after: int

    @property
    def winner_changed(self) -> bool:
        return self.winner_before != self.winner_after


def merge_log(
    lines: Sequence[str],
    tie_sets: Mapping[Shape, TieSet],
    samples: Samples,
) -> Tuple[List[str], List[ProblemChange]]:
    """The log `lines` with every near-tie candidate of `tie_sets` replaced
    by its median re-bench row (see the module docstring), and one
    ProblemChange per problem that changed."""
    edits: Dict[int, Tuple[int, List[str]]] = {}
    changes: List[ProblemChange] = []
    for prob in iter_problems(lines):
        shape = prob.shape()
        ties = tie_sets.get(shape) if shape is not None else None
        if ties is None or not prob.complete:
            continue
        wanted = set(ties.sol_indices)
        tested = tested_latencies(prob)
        latency = dict(tested)
        replaced: Dict[int, Tuple[float, float]] = {}
        blocks: Dict[int, List[str]] = {}
        ranks: Dict[int, int] = {}
        for t in prob.tested:
            if t.sol_index is None or t.sol_index in ranks:
                continue
            start, end = row_block(lines, t.line_no)
            ranks[t.sol_index] = t.rank
            blocks[t.sol_index] = list(lines[start:end])
            med = samples.median(shape, t.sol_index) if t.sol_index in wanted else None
            if med is None:
                continue
            new_block = with_rank(med.block, t.rank)
            edits[start] = (end, new_block)
            blocks[t.sol_index] = new_block
            replaced[t.sol_index] = (latency[t.sol_index], med.us)
            latency[t.sol_index] = med.us
        for s in prob.skipped:
            if s.sol_index is None or s.sol_index not in wanted:
                continue
            if s.sol_index in ranks:
                continue
            med = samples.median(shape, s.sol_index)
            if med is None:
                continue
            new_block = with_rank(med.block, s.rank)
            edits[s.line_no - 1] = (s.line_no, new_block)
            ranks[s.sol_index] = s.rank
            blocks[s.sol_index] = new_block
            replaced[s.sol_index] = (float(s.warm_up_us or math.nan), med.us)
            latency[s.sol_index] = med.us
        if not replaced:
            continue
        before = min(tested, key=lambda k: (tested[k], k))
        after = min(latency, key=lambda k: (latency[k], k))
        if prob.winner is not None:
            start, end = row_block(lines, prob.winner.line_no)
            edits[start] = (end, with_rank(blocks[after], ranks[after]))
        changes.append(ProblemChange(shape, replaced, before, after))
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
    iters_factor: float = DEFAULT_ITERS_FACTOR,
    rotating_mib: Optional[Callable[[Shape], Optional[int]]] = None,
) -> Tuple[Dict[Shape, PickRatio], Samples]:
    """Bench the pick of every GEMM of `gemms` in `picks` ({shape: solution
    index}) with the GEMM's timed winner and up to `extra` other timed
    kernels (`pick_kernel_list`), `runs` times, one `rebench_line` of the
    GEMM's bench line (`lines`) per GEMM, each list rotated by one place per
    run. `run_bench(run, yaml_lines)` benches the lines and returns the
    logs. Returns ({shape: PickRatio} of the GEMMs whose pick and winner some
    run timed together, the samples). The medians are `median_low` over
    those runs, so each ratio pairs timings of the same runs."""
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
                rotate(sols, r),
                iters_factor=iters_factor,
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
    warm-up pass, and returns its block logs; each run's stage03 exit code
    is appended to `returncodes`."""
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
        cmd += ["--no-warmup-pass", "--quiet", *extra_args]
        with (run_dir / "stage03.log").open("w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
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
) -> Tuple[Dict[Shape, TieSet], int]:
    """{shape: TieSet} over the complete problems of several logs (the
    union of the sets of a shape benched more than once), and the number of
    complete problems."""
    out: Dict[Shape, TieSet] = {}
    n_complete = 0
    for lines in logs:
        for prob in iter_problems(lines):
            if prob.complete:
                n_complete += 1
            ties = near_ties(prob, tolerance, max_kernels)
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
