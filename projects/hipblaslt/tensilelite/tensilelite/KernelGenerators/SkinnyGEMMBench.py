# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmarks the skinny MXFP8 GEMM solutions with hipblaslt-bench on a grid of shapes.

One bench process times every supported solution of a shape, so a run yields the
time of Origami's picks and of every generated kernel at once. This module holds
the parts that talk to the bench: the grid file, the command line, the parser of
its output, the resumable multi-GPU runner and the results directory.

Results directory layout:

    results.jsonl             one JSON object per run, appended and flushed per run
    solutions.json            solution index -> kernel name (checked for consistency)
    raw/<phase>/<key>_r<n>.txt.gz   the raw bench output of each run
"""

import gzip
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import SkinnyGEMMLogic

# Largest skinny extent: the decode batch sizes up to 16 are the target of the kernels.
SKINNY_MAX = 16
LD_PAD = 4

PHASES = ("grid", "held_out", "validate")
# The validate phase runs without the env var so that the library's routing rows are active.
PREDICTION_PHASES = ("grid", "held_out")

TIMEOUT_SECONDS = 900
PROGRESS_EVERY = 25
PARSE_ERROR = -2
TIMED_OUT = -3

_GRID_KEYS = {"orientations", "skinny", "wide", "k", "extra_points", "held_out", "bench"}
_BENCH_KEYS = {"warmup", "iterations", "rotating", "repeats", "refine_repeats"}

_HEADER = re.compile(r"^\[\d+\]:transA,")
_INDEX = re.compile(r"^\s*--Solution index:\s*(\d+)\s*$")
_NAME = re.compile(r"^\s*--Solution name:\s*(\S+)\s*$")
_KERNEL = re.compile(r"^\s*--kernel name:\s*(\S+)\s*$")
_GENERATED = re.compile(r"gemm_BM(\d+)_TN(\d+)_LANES_K(\d+)_WAVES(\d+)_UP_ITERS(\d+)_PARTIAL_STEPS(\d+)"
                        r"_PF(\d+)_GROUP(\d+)_NUM_XCDS(\d+)_GRID_3D([01])(?:_USE_BETA([01]))?")

Knobs = SkinnyGEMMLogic.Knobs
SolutionBlock = Tuple[int, float, str]


@dataclass(frozen=True, order=True)
class Point:
    """A problem shape; ld is the padded ldc and ldd of an m-skinny point, None for the bench defaults."""
    m: int
    n: int
    k: int
    ld: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.m}x{self.n}x{self.k}" + (f"_ld{self.ld}" if self.ld is not None else "")


@dataclass(frozen=True)
class BenchParams:
    warmup: int
    iterations: int
    rotating: int
    repeats: int
    refine_repeats: int


@dataclass(frozen=True)
class Grid:
    orientations: Tuple[str, ...]
    skinny: Tuple[int, ...]
    wide: Tuple[int, ...]
    k: Tuple[int, ...]
    extra_points: Tuple[Point, ...]
    held_out_count: int
    held_out_seed: int
    bench: BenchParams


@dataclass(frozen=True)
class Job:
    phase: str
    point: Point
    repeat: int


@dataclass
class RunRecord:
    phase: str
    point: Point
    repeat: int
    gpu: int
    wall_seconds: float
    return_code: int
    prediction_lib: bool
    first_index: Optional[int]
    times: Dict[int, float]

    def to_json(self) -> str:
        point = self.point
        return json.dumps({"phase": self.phase, "m": point.m, "n": point.n, "k": point.k, "ld": point.ld,
                           "repeat": self.repeat, "gpu": self.gpu, "wall_seconds": round(self.wall_seconds, 3),
                           "return_code": self.return_code, "prediction_lib": self.prediction_lib,
                           "first_index": self.first_index, "solutions": {str(i): us for i, us in self.times.items()}},
                          separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> "RunRecord":
        data = json.loads(text)
        return RunRecord(data["phase"], Point(data["m"], data["n"], data["k"], data["ld"]), data["repeat"],
                         data["gpu"], data["wall_seconds"], data["return_code"], data["prediction_lib"],
                         data["first_index"], {int(i): us for i, us in data["solutions"].items()})

    @property
    def job_key(self) -> Tuple[str, Point, int]:
        return (self.phase, self.point, self.repeat)


def padded_ld(skinny: int) -> int:
    return -(-skinny // LD_PAD) * LD_PAD


def make_point(m: int, n: int, k: int) -> Point:
    """A point whose ldc and ldd pad the m-skinny extent to a multiple of 4, as benched (not required)."""
    return Point(m, n, k, padded_ld(m) if m <= SKINNY_MAX < n else None)


def orientation_of(point: Point) -> Optional[str]:
    """n when n is the skinny extent, m when m is, None for any other shape."""
    if point.n <= SKINNY_MAX < point.m:
        return "n"
    if point.m <= SKINNY_MAX < point.n and point.ld == padded_ld(point.m):
        return "m"
    return None


def _axis_values(data: Dict[str, Any], key: str) -> Tuple[int, ...]:
    values = data[key]
    if not isinstance(values, list) or not values or not all(isinstance(v, int) and v >= 1 for v in values):
        raise ValueError(f"grid {key} must be a non-empty list of positive ints, got {values!r}")
    return tuple(sorted(set(values)))


def _load_extra_points(raw: Any) -> Tuple[Point, ...]:
    points = []
    for entry in raw or []:
        if not isinstance(entry, list) or len(entry) != 3 or not all(isinstance(v, int) for v in entry):
            raise ValueError(f"grid extra_points entries must be [m, n, k], got {entry!r}")
        point = make_point(*entry)
        if orientation_of(point) is None:
            raise ValueError(f"grid extra point {entry} has no skinny extent of at most {SKINNY_MAX}")
        points.append(point)
    return tuple(dict.fromkeys(points))


def _load_bench(raw: Dict[str, Any]) -> BenchParams:
    if set(raw) != _BENCH_KEYS:
        raise ValueError(f"grid bench must have exactly the keys {sorted(_BENCH_KEYS)}")
    bench = BenchParams(**raw)
    if bench.repeats < 1 or bench.refine_repeats < bench.repeats:
        raise ValueError("grid bench needs 1 <= repeats <= refine_repeats")
    return bench


def parse_grid(data: Dict[str, Any]) -> Grid:
    if set(data) != _GRID_KEYS:
        raise ValueError(f"the grid must have exactly the keys {sorted(_GRID_KEYS)}")
    orientations = tuple(data["orientations"])
    if not orientations or not set(orientations) <= {"n", "m"}:
        raise ValueError(f"grid orientations must be a non-empty subset of [n, m], got {orientations}")
    skinny, wide = _axis_values(data, "skinny"), _axis_values(data, "wide")
    if skinny[-1] > SKINNY_MAX or wide[0] <= SKINNY_MAX:
        raise ValueError(f"grid skinny values must be at most {SKINNY_MAX} and wide values above it")
    held_out = data["held_out"]
    return Grid(orientations, skinny, wide, _axis_values(data, "k"), _load_extra_points(data["extra_points"]),
                int(held_out["count"]), int(held_out["seed"]), _load_bench(data["bench"]))


def load_grid(path: Path) -> Grid:
    import yaml

    with open(path, encoding="utf-8") as stream:
        return parse_grid(yaml.safe_load(stream))


def lattice_points(grid: Grid) -> List[Point]:
    """The lattice points in a fixed order: orientation, skinny, wide, k."""
    points = []
    for orientation in grid.orientations:
        for skinny in grid.skinny:
            for wide in grid.wide:
                for k in grid.k:
                    points.append(make_point(wide, skinny, k) if orientation == "n" else make_point(skinny, wide, k))
    return points


def bench_command(bench: Sequence[str], point: Point, params: BenchParams) -> List[str]:
    command = list(bench) + ["-m", str(point.m), "-n", str(point.n), "-k", str(point.k),
                             "--transA", "T", "--transB", "N", "--a_type", "f8_r", "--b_type", "f8_r",
                             "--c_type", "bf16_r", "--d_type", "bf16_r", "--compute_type", "f32_r",
                             "--scaleA", "1001", "--scaleB", "1001", "--print_kernel_info",
                             "--requested_solution", "-1", "--rotating", str(params.rotating),
                             "--use_gpu_timer", "-j", str(params.warmup), "-i", str(params.iterations)]
    if point.ld is not None:
        command += ["--ldc", str(point.ld), "--ldd", str(point.ld)]
    return command


def _parse_block(lines: Sequence[str], start: int) -> SolutionBlock:
    where = f"solution block at line {start + 1}"
    tail = lines[start + 1:start + 5]
    if len(tail) < 4:
        raise ValueError(f"truncated {where}")
    index, name, kernel = _INDEX.match(tail[1]), _NAME.match(tail[2]), _KERNEL.match(tail[3])
    if not (index and name and kernel):
        raise ValueError(f"malformed {where}")
    try:
        microseconds = float(tail[0].split(",")[-1])
    except ValueError as error:
        raise ValueError(f"malformed time in {where}") from error
    return int(index.group(1)), microseconds, kernel.group(1)


def parse_output(text: str) -> Tuple[int, List[SolutionBlock]]:
    """Returns the index of the first listed solution and every block; an index may appear twice."""
    lines = text.splitlines()
    blocks = [_parse_block(lines, number) for number, line in enumerate(lines) if _HEADER.match(line)]
    if not blocks:
        raise ValueError("no solution block in the bench output")
    return blocks[0][0], blocks


def generated_knobs(kernel_name: str, pool: SkinnyGEMMLogic.Pool) -> Optional[Knobs]:
    """The knobs of a generated kernel name, None for any other kernel; unknown generated kernels are errors."""
    if not kernel_name.startswith("gemm_"):
        return None
    match = _GENERATED.fullmatch(kernel_name)
    if match is None:
        raise ValueError(f"unparsable generated kernel name {kernel_name}")
    # A name without the suffix predates the useBeta variants.
    values = [int(group or 0) for group in match.groups()]
    # The name order is bm, tn, lanesK, waves, upIters, partialSteps, prefetch, group, numXcds, grid3d, useBeta.
    bm, tn, lanesK, waves, upIters, partialSteps, prefetch, group, numXcds, grid3d, useBeta = values
    knobs = (bm, tn, lanesK, waves, prefetch, upIters, partialSteps, group, numXcds, bool(grid3d), bool(useBeta))
    if knobs not in pool:
        raise ValueError(f"generated kernel {kernel_name} is not in the pool")
    return knobs


def fastest_per_index(blocks: Iterable[SolutionBlock]) -> Dict[int, float]:
    times: Dict[int, float] = {}
    for index, microseconds, _ in blocks:
        times[index] = min(microseconds, times.get(index, microseconds))
    return times


class SolutionTable:
    """Solution index -> kernel name of the library under test, persisted in solutions.json."""

    def __init__(self, path: Path, pool: SkinnyGEMMLogic.Pool):
        self.path = path
        self.pool = pool
        self.names: Dict[int, str] = {}
        if path.exists():
            self.names = {int(index): name for index, name in json.loads(path.read_text(encoding="utf-8")).items()}

    def knobs(self, index: int) -> Optional[Knobs]:
        return generated_knobs(self.names[index], self.pool)

    def register(self, blocks: Iterable[SolutionBlock]) -> bool:
        """Adds the solutions of one run; returns whether the table grew."""
        grew = False
        for index, _, name in blocks:
            generated_knobs(name, self.pool)
            known = self.names.get(index)
            if known is None:
                self.names[index] = name
                grew = True
            elif known != name:
                raise ValueError(f"solution {index} is {name} but {known} in {self.path}: "
                                 "the library changed, use a new results directory")
        return grew

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({str(i): name for i, name in sorted(self.names.items())}), encoding="utf-8")
        os.replace(temporary, self.path)


class ResultsDir:
    """The results of a campaign: an append-only run log, the raw outputs and the solution table."""

    def __init__(self, path: Path, pool: SkinnyGEMMLogic.Pool):
        self.path = path
        self.log = path / "results.jsonl"
        path.mkdir(parents=True, exist_ok=True)
        self.table = SolutionTable(path / "solutions.json", pool)
        self.lock = threading.Lock()

    def records(self) -> List[RunRecord]:
        if not self.log.exists():
            return []
        with open(self.log, encoding="utf-8") as stream:
            return [RunRecord.from_json(line) for line in stream if line.strip()]

    def done_keys(self) -> Set[Tuple[str, Point, int]]:
        return {record.job_key for record in self.records() if record.return_code == 0}

    def raw_path(self, job: Job) -> Path:
        return self.path / "raw" / job.phase / f"{job.point.key}_r{job.repeat}.txt.gz"

    def append(self, job: Job, record: RunRecord, output: str, blocks: Sequence[SolutionBlock]) -> None:
        raw_path = self.raw_path(job)
        with self.lock:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
                stream.write(output)
            if self.table.register(blocks):
                self.table.save()
            with open(self.log, "a", encoding="utf-8") as stream:
                stream.write(record.to_json() + "\n")
                stream.flush()


def parse_gpus(text: str) -> List[int]:
    """Parses 0-7, 0,2,5 or a mix such as 0-3,6."""
    gpus: List[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        gpus += range(int(low), int(high or low) + 1)
    return list(dict.fromkeys(gpus))


def order_jobs(phase: str, points: Sequence[Point], repeats: Iterable[int], seed: int) -> List[Job]:
    """Repeat-major order with the points of each repeat shuffled, so the repeats of a point land far apart in time."""
    jobs = []
    for repeat in repeats:
        shuffled = list(points)
        random.Random(seed + repeat).shuffle(shuffled)
        jobs += [Job(phase, point, repeat) for point in shuffled]
    return jobs


@dataclass(frozen=True)
class Runner:
    bench: Sequence[str]
    params: BenchParams
    gpus: Sequence[int]
    results: ResultsDir

    def _environment(self, gpu: int, phase: str) -> Dict[str, str]:
        env = dict(os.environ)
        env.pop("TENSILE_PREDICTION_LIB", None)
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        if phase in PREDICTION_PHASES:
            env["TENSILE_PREDICTION_LIB"] = "1"
        return env

    def execute(self, job: Job, gpu: int) -> Tuple[RunRecord, str, List[SolutionBlock]]:
        command = bench_command(self.bench, job.point, self.params)
        start = time.monotonic()
        try:
            completed = subprocess.run(command, env=self._environment(gpu, job.phase), capture_output=True,
                                       text=True, timeout=TIMEOUT_SECONDS, errors="replace")
            code, output = completed.returncode, completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as error:
            code, output = TIMED_OUT, f"timeout after {TIMEOUT_SECONDS} s\n{error.stdout or ''}"
        record = RunRecord(job.phase, job.point, job.repeat, gpu, time.monotonic() - start, code,
                           job.phase in PREDICTION_PHASES, None, {})
        blocks: List[SolutionBlock] = []
        if code != 0:
            return record, output, blocks
        try:
            record.first_index, blocks = parse_output(output)
            record.times = fastest_per_index(blocks)
        except ValueError as error:
            record.return_code, output = PARSE_ERROR, f"{output}\nparse error: {error}\n"
        return record, output, blocks

    def run(self, jobs: Sequence[Job], progress: Callable[[str], None] = print) -> int:
        """Runs the jobs not yet done on one worker per GPU; returns the number of failed jobs."""
        done = self.results.done_keys()
        pending = [job for job in jobs if (job.phase, job.point, job.repeat) not in done]
        return _Session(self, pending, progress).execute()


class _Session:
    """The queue, the workers and the counters of one Runner.run call."""

    def __init__(self, runner: Runner, jobs: Sequence[Job], progress: Callable[[str], None]):
        self.runner = runner
        self.progress = progress
        self.pending: "queue.Queue[Tuple[Job, int]]" = queue.Queue()
        for job in jobs:
            self.pending.put((job, 0))
        self.total = len(jobs)
        self.finished = 0
        self.failed = 0
        self.errors: List[BaseException] = []
        self.stop = threading.Event()
        self.start = time.monotonic()

    def _complete(self, job: Job, record: RunRecord, output: str, blocks: Sequence[SolutionBlock]) -> None:
        results = self.runner.results
        results.append(job, record, output, blocks)
        with results.lock:
            self.finished += 1
            self.failed += record.return_code != 0
            finished = self.finished
        if finished % PROGRESS_EVERY == 0 or finished == self.total:
            eta = (time.monotonic() - self.start) / finished * (self.total - finished)
            self.progress(f"{finished}/{self.total} jobs, eta {eta:.0f} s")

    def _drain(self, gpu: int) -> None:
        while not self.stop.is_set():
            try:
                job, attempt = self.pending.get_nowait()
            except queue.Empty:
                return
            record, output, blocks = self.runner.execute(job, gpu)
            if record.return_code != 0 and attempt == 0:
                # A second attempt, possibly on another GPU, before the failure is recorded.
                self.pending.put((job, 1))
                continue
            self._complete(job, record, output, blocks)

    def _work(self, gpu: int) -> None:
        try:
            self._drain(gpu)
        except BaseException as error:
            self.errors.append(error)
            self.stop.set()

    def execute(self) -> int:
        threads = [threading.Thread(target=self._work, args=(gpu,), daemon=True) for gpu in self.runner.gpus]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.stop.set()
            for thread in threads:
                thread.join()
            raise
        if self.errors:
            raise self.errors[0]
        return self.failed
