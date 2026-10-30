# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Greedy selection of a small set of skinny MXFP8 GEMM kernels covering a sweep.

The input is the results YAML of sweep_skinny_gemm.py. A case (one entry of
results:) is covered by a configuration when the configuration, re-targeted to
the case and timed here, is within --tolerance of the case's best time of the
sweep. The tool picks configurations greedily so that few kernels cover many
cases, and writes the selection, the case to kernel assignment and the assembly
of every kernel used.

Run from the tensilelite directory:

  source ~/.tensile/bin/activate
  export PYTHONPATH=$PWD/build_tmp/tensilelite/src/tensilelite/bindings
  export PYTHONPATH=$PYTHONPATH:$PWD/tests/tensilelite/bindings
  export LD_LIBRARY_PATH=/opt/rocm/lib
  python tests/tensilelite/bindings/select_skinny_gemm_kernels.py results.yml \\
      -o OUT_DIR [--max-configs N] [--tolerance 0.10] [--iters 30] [--warmup 5] \\
      [--verify] [--only SUBSTRING] [--limit-cases N]

Config identity matches select_skinny_gemm_from_sweep.py: (bm, tn, lanes_k,
waves, pf, up_iters, partial_steps, group, xcds, grid3d, blocks, mfma, lds,
use_beta). blocks is None for grid3d (derived) and literal otherwise.
useBeta and spk are not part of the identity: useBeta always comes from the
target case, and spk is re-derived per target (see build_problem).

Algorithm. Every case starts with the identities of its sweep top list as
candidates (the best time per identity). Rounds repeat until every case is
covered or no untried candidate is left. --max-configs caps the library size,
not the search: rounds continue until candidates are exhausted.

  1. Tally the untried candidates of the uncovered cases. The best identity C
     maximizes, in order, frequency weighted by sweep closeness (frequency divided
     by mean ratio to each case's best), mean rank, mean ratio, and finally the
     identity tuple (None as -1), which makes the choice deterministic.
  2. C is re-targeted to every uncovered case, using the case's own candidate
     entry when C is in its candidates and otherwise the representative of C:
     its best-ranked candidate over all cases (then lowest ratio, then lowest
     case index). C is marked tried for the case, so every round progresses.
  3. A measured case within tolerance is recorded per configuration. After each
     round, cases are assigned disjointly to the fastest qualifying configuration
     for that case (ties broken by config identity). The final library keeps only
     configurations with at least --min-config-cases assigned shapes, then applies
     --max-configs by dropping the smallest remaining configs; unassigned cases
     pick their next-fastest library configuration if any.
  4. For every case measured this round, the unprotected candidates D with
     measured time of C <= time of D * (1 + tolerance) are dropped: C is within
     tolerance of D, so D cannot add coverage that C does not. Candidates that
     are clearly faster than C stay.
  5. After the search, rebuild_assignment runs once more for output.

Timing mirrors bench_skinny_gemm.benchProblem but flushes the LLC with
sweep_skinny_gemm_flush.FastLlcFlush, the flush of the sweep (same cold-LLC
contract, one flush before every GEMM launch, warm reduce). Like the sweep,
--warmup untimed launches (flushed the same way, the reduce included) precede
the --iters timed ones, followed by launches until --warmup-ms of wall time
(default 30) have passed, because the count alone lasts under a millisecond,
before the GPU settles after the uploads, and biased the times up by about 3.5
percent. A time is the minimum over the timed launches, the first included
(for split-K, the minimum over the iterations of the GEMM plus the reduce time
of that iteration). The sweep's refinement of its top
candidates (50 iterations in results.yml) is intentionally not implemented.

Execution is pipelined. Kernels are assembled by a pool of --jobs spawned
processes (BuildStage of sweep_skinny_gemm.py) into <work-dir>/hsaco, which is
also a cache across rounds and runs (use a fresh work dir after changing the
emitter). One persistent worker process per GPU (--gpus, default every GPU below
5 percent use) loads the hsaco files (once per worker, they stay loaded) and
times the problems; this process never initializes HIP. A job is sent to a GPU
as soon as its kernels are built, and the build submissions run a window of
max(2 * jobs, 4 * gpus) kernels ahead of the dispatch, so everything starts at
once without queueing a whole round. Timing is serial within a GPU, and
separate GPUs do not interfere, but the clocks are not locked and all the GPUs
must be the same model; --tolerance absorbs the remaining noise. The results
are applied in case order after the round, so the selection is deterministic
up to timing noise. A kernel that fails to emit or assemble fails its case like
any other error, a dead worker aborts the run, and every case prints a
progress line (in completion order) with elapsed time and ETA.

Inputs are the sweep's (sweep_skinny_gemm_inputs.py): operands generated in
parallel by table lookup, a hundred times faster than support.makeInputs, which
spent seconds to tens of seconds per case and left the GPUs idle. A worker keeps
its operands (host bytes, bounded by --cache-gb) and the device buffers of its
recent cases (device bytes, bounded by --cache-gb too). Per worker the host
memory is therefore the operand cache plus the host copies of D and the
workspace of each cached case plus up to three inputs not in the cache (the one
timed, the one prepared and the one in preparation), and 8 GPUs hold 8 times
that. A thread prepares the host operands of the next job while the GPU
times the current one (uploads stay on the timing thread, so they cannot land
inside a timed launch). A case always goes to the GPU case_index % gpus, so its
operands are reused by later rounds; an idle GPU takes a case when its own is
full, which costs one cache miss. Up to three jobs are queued per GPU. The
round summary shows the GPU busy share: the job seconds of the workers over
wall time times GPUs.

The output directory gets selection.yaml (meta, the selected configurations
with their cases, the uncovered cases) and asm/<kernel>.s, the assembly of one
representative problem per kernel name used.
"""

import argparse
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import traceback
from collections import Counter, OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from statistics import mean
from typing import NamedTuple

import yaml

import bench_skinny_gemm as bench
import skinny_gemm_support as support
from skinny_gemm_support import DeviceBuffer, Problem
from sweep_skinny_gemm import BuildResult, BuildStage
from sweep_skinny_gemm_config import SETTING_DEFAULTS
from sweep_skinny_gemm_flush import FastLlcFlush
from sweep_skinny_gemm_inputs import (
    CaseInputs,
    OperandCache,
    Verifier,
    make_case_inputs,
)
from sweep_skinny_gemm_smi import SweepAbort, select_gpus
from sweep_skinny_gemm_worker import hsaco_file

DEFAULT_TOLERANCE = 0.10
DEFAULT_ITERS = 30
DEFAULT_WARMUP = 5
MAX_DEFAULT_JOBS = 32
MAX_BUSY_PERCENT = 5
JOBS_PER_GPU = 3
DEFAULT_CACHE_GB = 4.0
DEFAULT_WARMUP_MS = 30.0
WARMUP_BATCH = 5
BUILDS_PER_GPU = 4
POLL_S = 0.05
STOP_TIMEOUT_S = 10
LABEL_KEYS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")
REASON_CAP = "max_configs reached"
REASON_EXHAUSTED = "no untried candidates left"
REASON_UNMEASURED = "no candidate measured successfully"


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


class Candidate(NamedTuple):
    identity: ConfigIdentity
    cfg: dict
    source_k: int
    source_case: int
    rank: int
    time_us: float


@dataclass(frozen=True)
class Case:
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
class CoverRecord:
    identity: ConfigIdentity
    time_us: float
    problem: Problem
    kernel_name: str
    spk_used: int
    spk_degraded: bool


@dataclass(frozen=True)
class WithinMeasurement:
    time_us: float
    plan: "Plan"


@dataclass(frozen=True)
class SelectionResult:
    selected: list[ConfigIdentity]
    covers: dict[int, CoverRecord]
    measured_best: dict[int, tuple[ConfigIdentity, float]]
    cap_reached: bool
    search_exhausted: bool


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
        bool(cfg.get("useBeta", False)),
    )


def tie_break_key(identity: ConfigIdentity) -> tuple[int, ...]:
    return tuple(-1 if v is None else int(v) for v in identity)


def load_cases(path: str, only: str | None, limit: int | None) -> list[Case]:
    from sweep_skinny_gemm_results import read_results

    document = read_results(path)
    entries = [e for e in document["results"] if e["top"]]
    if only:
        entries = [e for e in entries if only in e["problem"]["name"]]
    if limit is not None:
        entries = entries[:limit]
    return [
        Case(i, e["problem"], e["top"][0]["timeUs"], e["top"])
        for i, e in enumerate(entries)
    ]


def seed_candidates(case: Case) -> dict[ConfigIdentity, Candidate]:
    found: dict[ConfigIdentity, Candidate] = {}
    for entry in case.top:
        cfg = entry["config"]
        identity = identity_of(cfg)
        candidate = Candidate(
            identity,
            cfg,
            case.problem["k"],
            case.index,
            entry["rank"],
            entry["timeUs"],
        )
        known = found.get(identity)
        if known is None or candidate.time_us < known.time_us:
            found[identity] = candidate
    return found


def representative_key(candidate: Candidate, cases: list[Case]) -> tuple:
    ratio = candidate.time_us / cases[candidate.source_case].best_us
    return (candidate.rank, ratio, candidate.source_case)


def find_representatives(
    cases: list[Case], candidates: list[dict[ConfigIdentity, Candidate]]
) -> dict[ConfigIdentity, Candidate]:
    reps: dict[ConfigIdentity, Candidate] = {}
    for per_case in candidates:
        for identity, candidate in per_case.items():
            known = reps.get(identity)
            if known is None or representative_key(candidate, cases) < (
                representative_key(known, cases)
            ):
                reps[identity] = candidate
    return reps


def literal_spk_allowed(tp: dict, spk: int) -> bool:
    return spk >= 1 and (tp["k"] <= spk or spk % 256 == 0)


def make_target_problem(tp: dict, cfg: dict, spk: int) -> Problem:
    align = support.alignUp(tp["k"], 4)
    # Explicit, so that an old VALU config never gets the MFMA defaults.
    mfma = bool(cfg.get("mfma", False))
    return support.makeProblem(
        tp["m"],
        tp["n"],
        tp["k"],
        spk=spk,
        bm=cfg["bm"],
        tn=cfg["tn"],
        lanes_k=cfg["lanesK"],
        waves=cfg["waves"],
        pf=cfg["prefetch"],
        group=cfg["group"],
        xcds=cfg["numXcds"],
        up_scales=0 if mfma else int(cfg["upIters"] > 0),
        blocks=None if cfg["grid3d"] else cfg["blocks"],
        padding=(tp["ldA"] - align, tp["ldB"] - align, 0),
        ldD=tp["ldD"] // 2,
        ldC=tp["ldC"] // 2,
        alpha=tp["alpha"],
        beta=tp["beta"],
        use_beta=tp["useBeta"],
        mfma=mfma,
        lds=bool(cfg.get("lds", False)),
    )


def build_problem(tp: dict, cfg: dict, source_k: int) -> tuple[Problem, bool]:
    """Re-targets cfg to the case tp; the flag is true when the split size of
    cfg could not be kept and the case runs unsplit instead."""
    if cfg["spk"] >= source_k or not literal_spk_allowed(tp, cfg["spk"]):
        return make_target_problem(tp, cfg, tp["k"]), cfg["spk"] < source_k
    try:
        return make_target_problem(tp, cfg, cfg["spk"]), False
    except AssertionError:
        return make_target_problem(tp, cfg, tp["k"]), True


def case_key(p: Problem) -> tuple:
    """What the device buffers of a case depend on; the kernel knobs do not."""
    return (
        p.m,
        p.n,
        p.k,
        p.ldA,
        p.ldB,
        p.ldC,
        p.ldD,
        p.splits,
        p.use_beta,
        p.beta == 0,
    )


class DeviceCase:
    """The device buffers of a case: A, B, the scales and C are read only, D and
    the workspace are rewritten by every launch. A context manager that frees
    them."""

    def __init__(self, p: Problem, inputs: CaseInputs):
        self.d_host, self.ws_host = support.makeOutputs(p.m, p.n, p.ldD, p.splits)
        self.stack = ExitStack()
        arrays = (
            self.d_host,
            inputs.a.raw,
            inputs.b.raw,
            inputs.a.swizzled,
            inputs.b.swizzled,
            self.ws_host,
            inputs.c_bits,
        )
        self.nbytes = sum(array.nbytes for array in arrays)
        try:
            self.d, self.a, self.b, self.scl_a, self.scl_b, self.ws, self.c = (
                self.stack.enter_context(DeviceBuffer(array)) for array in arrays
            )
        except BaseException:
            self.stack.close()
            raise

    def reset_outputs(self) -> None:
        self.d.upload(self.d_host)
        self.ws.upload(self.ws_host)

    def close(self) -> None:
        self.stack.close()


class DeviceCaseCache:
    """The device buffers of the recent cases of a worker, bounded in bytes, so
    that a case timed again (another round) is not uploaded again."""

    def __init__(self, limit: int):
        self.limit = limit
        self.cases: OrderedDict[tuple, DeviceCase] = OrderedDict()

    def get(self, p: Problem, inputs: CaseInputs) -> DeviceCase:
        key = case_key(p)
        if key in self.cases:
            self.cases.move_to_end(key)
            return self.cases[key]
        case = DeviceCase(p, inputs)
        self.cases[key] = case
        total = sum(c.nbytes for c in self.cases.values())
        while total > self.limit and len(self.cases) > 1:
            _, evicted = self.cases.popitem(last=False)
            total -= evicted.nbytes
            evicted.close()
        return case

    def close(self) -> None:
        for case in self.cases.values():
            case.close()
        self.cases.clear()


def time_reduce_ns(p: Problem, device: DeviceCase, launches: int) -> list[int]:
    """The warm launches of the split-K reduce, in ns."""
    reducer = support.reduceKernel(support.DEFAULT_CODE_OBJECT_VERSION, 0)
    grid, block = support.reduceLaunchDims(p.m, p.n)
    args = support.reduceArgs(
        device.d, device.ws, device.c, p.ldD, p.m, p.n, p.splits, p.ldC, p.alpha, p.beta
    )
    return bench.timeLaunches(reducer, grid, block, args, launches, None)


def make_verifier(p: Problem, inputs: CaseInputs) -> Verifier:
    full = (p.m + p.n) * p.k <= SETTING_DEFAULTS["verify_full_max_elems"]
    return Verifier(inputs, full)


def warm_up(launch: Callable[[int], object], count: int, warmup_ms: float) -> None:
    """Runs count launches, then more in small batches until warmup_ms of wall
    time have passed. A count alone ends in under a millisecond, before the clocks
    and the pipeline have settled after the uploads, which biased the times up by
    a few percent."""
    if count > 0:
        launch(count)
    start = time.monotonic()
    while (time.monotonic() - start) * 1e3 < warmup_ms:
        launch(WARMUP_BATCH)


def measure_us(
    job: "Job", inputs: CaseInputs, device: DeviceCase, flush: FastLlcFlush
) -> float:
    """The best time in microseconds, reduce included for split-K, to be
    comparable with the timeUs of the sweep: the minimum over the iterations of
    the GEMM time plus the reduce time of the same iteration, after warmup
    untimed launches. Mirrors bench.benchProblem, but flushes the LLC with the
    flush of the sweep and, with verify, checks the result like the sweep."""
    p, iters, warmup, warmup_ms = job.problem, job.iters, job.warmup, job.warmup_ms
    kernel = support.gemmKernel(p, support.DEFAULT_CODE_OBJECT_VERSION, 0)
    args = support.gemmArgs(
        p, device.d, device.a, device.b, device.scl_a, device.scl_b, device.ws, device.c
    )
    block = support.gemmBlock(p)
    verifier = make_verifier(p, inputs) if job.verify else None
    if verifier:
        device.reset_outputs()
    warm_up(
        lambda n: bench.timeLaunches(kernel, p.grid, block, args, n, flush),
        warmup,
        warmup_ms,
    )
    gemm_ns = bench.timeLaunches(kernel, p.grid, block, args, iters, flush)
    if verifier:
        d_bits, ws = device.d_host.copy(), device.ws_host.copy()
        device.d.download(d_bits)
        device.ws.download(ws)
        verifier.check_gemm(p, d_bits, ws)
    if p.splits == 1:
        return min(gemm_ns) * 1e-3
    warm_up(lambda n: time_reduce_ns(p, device, n), warmup, warmup_ms)
    reduce_ns = time_reduce_ns(p, device, iters)
    if verifier:
        device.d.download(d_bits)
        verifier.check_reduced(p, d_bits)
    return min(g + r for g, r in zip(gemm_ns, reduce_ns)) * 1e-3


class TallyEntry(NamedTuple):
    identity: ConfigIdentity
    frequency: int
    mean_rank: float
    mean_ratio: float


def tally(
    cases: list[Case],
    uncovered: list[int],
    candidates: list[dict[ConfigIdentity, Candidate]],
    tried: list[set],
) -> list[TallyEntry]:
    """The untried identities of the uncovered cases, best first."""
    ranks: dict[ConfigIdentity, list[int]] = {}
    ratios: dict[ConfigIdentity, list[float]] = {}
    for i in uncovered:
        for identity, candidate in candidates[i].items():
            if identity in tried[i]:
                continue
            ranks.setdefault(identity, []).append(candidate.rank)
            ratios.setdefault(identity, []).append(candidate.time_us / cases[i].best_us)
    entries = [
        TallyEntry(identity, len(r), mean(r), mean(ratios[identity]))
        for identity, r in ranks.items()
    ]
    entries.sort(
        key=lambda e: (
            -e.frequency / max(e.mean_ratio, 1e-9),
            e.mean_rank,
            e.mean_ratio,
            tie_break_key(e.identity),
        )
    )
    return entries


def log_error(case: Case, identity: ConfigIdentity, reason: str) -> None:
    print(f"case {case.text()}: {identity} does not cover: {reason}", file=sys.stderr)


def first_line(err: Exception) -> str:
    lines = str(err).strip().splitlines()
    return f"{type(err).__name__}: {lines[0] if lines else ''}"


class KernelPipeline:
    """Assembles kernels in a process pool, a bounded window ahead of use."""

    def __init__(self, work_dir: str, jobs: int, window: int):
        self.hsaco_dir = os.path.join(work_dir, "hsaco")
        os.makedirs(self.hsaco_dir, exist_ok=True)
        self.stage = BuildStage(self.hsaco_dir, jobs)
        self.window = window
        self.known: set[str] = set()
        self.pending: list[tuple[str, Problem | None]] = []
        self.unrequested: set[str] = set()

    def path(self, name: str) -> str:
        return hsaco_file(self.hsaco_dir, name)

    def enqueue(self, name: str, problem: Problem | None) -> None:
        if name in self.known:
            return
        self.known.add(name)
        if os.path.exists(self.path(name)):
            self.stage.results[name] = BuildResult(name, True, "", "cached")
            return
        self.pending.append((name, problem))

    def top_up(self) -> None:
        while self.pending and len(self.unrequested) < self.window:
            name, problem = self.pending.pop(0)
            self.stage.submit(name, problem)
            self.unrequested.add(name)

    def result(self, name: str) -> BuildResult | None:
        """The build result of a kernel if it is done; asking for a kernel
        frees its slot of the window."""
        self.unrequested.discard(name)
        self.top_up()
        self.stage.poll(0)
        return self.stage.results.get(name)

    def close(self) -> None:
        self.stage.shutdown()


class Job(NamedTuple):
    case_index: int
    problem: Problem
    kernels: tuple[tuple[str, str], ...]
    iters: int
    warmup: int
    warmup_ms: float
    verify: bool


def load_job_kernels(job: Job) -> None:
    """Puts the kernels of the job in the cache of the support module, where the
    benchmark finds them and so does not assemble. They stay loaded, so a kernel
    is read from disk once per worker."""
    for name, path in job.kernels:
        key = (name, support.DEFAULT_CODE_OBJECT_VERSION, 0)
        if key in support._kernelCache:
            continue
        with open(path, "rb") as stream:
            support._kernelCache[key] = support.loadKernel(stream.read(), name)


class PrepareFailure(NamedTuple):
    text: str


def prepare_jobs(jobs, prepared: queue.Queue, operands: OperandCache) -> None:
    """The thread of a worker that takes the jobs and generates their host
    operands while the GPU times the previous job. Only the host is touched here:
    an upload from this thread could land inside a timed launch."""
    try:
        while True:
            job = jobs.get()
            if job is None:
                prepared.put(None)
                return
            try:
                inputs = make_case_inputs(job.problem, operands)
            except Exception as err:
                prepared.put((job, None, first_line(err)))
                continue
            prepared.put((job, inputs, None))
    except Exception:
        # Anything else would leave the timing loop blocked on the queue.
        prepared.put(PrepareFailure(traceback.format_exc()))


def run_job(
    job: Job,
    inputs: CaseInputs,
    devices: DeviceCaseCache,
    flush: FastLlcFlush,
) -> tuple[float | None, str | None]:
    try:
        load_job_kernels(job)
        device = devices.get(job.problem, inputs)
        return measure_us(job, inputs, device, flush), None
    except Exception as err:
        return None, first_line(err)


def gpu_worker(gpu: int, jobs, results, cache_bytes: int) -> None:
    """The process of one GPU: runs the jobs of its queue, until the None
    sentinel, and puts ("done", case_index, time_us, error, busy_s) or
    ("fatal", gpu, text) on the result queue. A thread prepares the host operands
    of the next job meanwhile."""
    # HIP is first used below, so the device the worker sees is always 0.
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    devices = DeviceCaseCache(cache_bytes)
    prepared: queue.Queue = queue.Queue(maxsize=1)
    operands = OperandCache(cache_bytes)
    threading.Thread(
        target=prepare_jobs, args=(jobs, prepared, operands), daemon=True
    ).start()
    try:
        with FastLlcFlush() as flush:
            while True:
                item = prepared.get()
                if item is None:
                    return
                if isinstance(item, PrepareFailure):
                    raise RuntimeError(f"prepare thread failed:\n{item.text}")
                job, inputs, error = item
                start = time.monotonic()
                time_us = None
                if error is None:
                    time_us, error = run_job(job, inputs, devices, flush)
                busy = time.monotonic() - start
                results.put(("done", job.case_index, time_us, error, busy))
    except Exception:
        results.put(("fatal", gpu, traceback.format_exc()))
        raise
    finally:
        devices.close()


class GpuPool:
    """One worker process per GPU, each with its own job queue."""

    def __init__(self, gpus: list[int], cache_bytes: int):
        context = multiprocessing.get_context("spawn")
        self.jobs = [context.Queue() for _ in gpus]
        self.results = context.Queue()
        self.workers = [
            context.Process(
                target=gpu_worker,
                args=(gpu, jobs, self.results, cache_bytes),
                daemon=True,
            )
            for gpu, jobs in zip(gpus, self.jobs)
        ]
        for worker in self.workers:
            worker.start()

    def submit(self, position: int, job: Job) -> None:
        self.jobs[position].put(job)

    def poll(self, timeout: float) -> tuple | None:
        """The next result message; raises when a worker died."""
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            pass
        dead = [w.name for w in self.workers if not w.is_alive()]
        if dead:
            raise RuntimeError(f"gpu worker(s) {dead} died")
        return None

    def stop(self) -> None:
        for jobs in self.jobs:
            jobs.put(None)
        for worker in self.workers:
            worker.join(timeout=STOP_TIMEOUT_S)
            if worker.is_alive():
                worker.terminate()


class Plan(NamedTuple):
    case_index: int
    problem: Problem | None
    degraded: bool
    reason: str | None


class Outcome(NamedTuple):
    time_us: float | None
    error: str | None


def default_jobs() -> int:
    return max(1, min(MAX_DEFAULT_JOBS, (os.cpu_count() or 2) // 2))


class Selector:
    """The state of the greedy selection."""

    def __init__(
        self,
        cases: list[Case],
        tolerance: float,
        iters: int,
        warmup: int,
        warmup_ms: float,
        verify: bool,
        pipeline: KernelPipeline,
        gpus: GpuPool,
        num_gpus: int,
        max_configs: int | None = None,
        min_config_cases: int = 0,
    ):
        self.cases = cases
        self.max_configs = max_configs
        self.min_config_cases = min_config_cases
        self.tolerance = tolerance
        self.iters = iters
        self.warmup = warmup
        self.warmup_ms = warmup_ms
        self.verify = verify
        self.pipeline = pipeline
        self.gpus = gpus
        self.num_gpus = num_gpus
        self.busy_s = 0.0
        self.candidates = [seed_candidates(c) for c in cases]
        self.representatives = find_representatives(cases, self.candidates)
        self.protected: list[set] = [set() for _ in cases]
        self.tried: list[set] = [set() for _ in cases]
        self.measured_best: dict[int, tuple[ConfigIdentity, float]] = {}
        self.qualified: dict[int, dict[ConfigIdentity, WithinMeasurement]] = {}
        self.covers: dict[int, CoverRecord] = {}
        self.selected: list[ConfigIdentity] = []

    def uncovered(self) -> list[int]:
        return [c.index for c in self.cases if c.index not in self.covers]

    def count_cases_for(self, identity: ConfigIdentity) -> int:
        return sum(1 for c in self.covers.values() if c.identity == identity)

    def record_within(
        self, i: int, identity: ConfigIdentity, plan: Plan, time_us: float
    ) -> None:
        if plan.problem is None or not self.covers_within(i, time_us):
            return
        per = self.qualified.setdefault(i, {})
        meas = WithinMeasurement(time_us, plan)
        known = per.get(identity)
        if known is None or time_us < known.time_us:
            per[identity] = meas
        self.protected[i].add(identity)
        if identity not in self.candidates[i]:
            source = self.source_for(i, identity)
            self.candidates[i][identity] = source._replace(time_us=time_us)

    def fastest_winner(
        self, per_id: dict[ConfigIdentity, WithinMeasurement]
    ) -> tuple[ConfigIdentity, WithinMeasurement]:
        return min(
            per_id.items(),
            key=lambda kv: (kv[1].time_us, tie_break_key(kv[0])),
        )

    def library_from_counts(
        self, counts: Counter[ConfigIdentity]
    ) -> set[ConfigIdentity]:
        eligible = {
            identity for identity, num in counts.items() if num >= self.min_config_cases
        }
        if not eligible:
            return set()
        ranked = sorted(
            eligible,
            key=lambda identity: (-counts[identity], tie_break_key(identity)),
        )
        if self.max_configs is not None:
            ranked = ranked[: self.max_configs]
        return set(ranked)

    def write_cover(
        self, i: int, identity: ConfigIdentity, meas: WithinMeasurement
    ) -> None:
        p = meas.plan.problem
        self.covers[i] = CoverRecord(
            identity,
            meas.time_us,
            p,
            support.gemmKernelName(p),
            p.spk,
            meas.plan.degraded,
        )

    def rebuild_assignment(self) -> None:
        """Disjoint assignment: each case uses its fastest within-tolerance config."""
        winners: dict[int, ConfigIdentity] = {}
        for i, per_id in self.qualified.items():
            if not per_id:
                continue
            identity, _ = self.fastest_winner(per_id)
            winners[i] = identity
        library = self.library_from_counts(Counter(winners.values()))
        self.covers.clear()
        for i, identity in winners.items():
            if identity not in library:
                continue
            self.write_cover(i, identity, self.qualified[i][identity])
        for i, per_id in self.qualified.items():
            if i in self.covers:
                continue
            options = {id: m for id, m in per_id.items() if id in library}
            if not options:
                continue
            identity, meas = self.fastest_winner(options)
            self.write_cover(i, identity, meas)
        counts = Counter(c.identity for c in self.covers.values())
        self.selected = sorted(
            counts.keys(),
            key=lambda identity: (-counts[identity], tie_break_key(identity)),
        )

    def round_cases(self, identity: ConfigIdentity, uncovered: list[int]) -> list[int]:
        """Cases to time this round; after the library cap, skip pairs already tried."""
        at_cap = self.max_configs is not None and len(self.selected) >= self.max_configs
        if not at_cap:
            return uncovered
        return [
            i
            for i in uncovered
            if identity not in self.tried[i] and identity in self.candidates[i]
        ]

    def source_for(self, i: int, identity: ConfigIdentity) -> Candidate:
        own = self.candidates[i].get(identity)
        return own if own is not None else self.representatives[identity]

    def plan_case(self, i: int, identity: ConfigIdentity) -> Plan:
        case = self.cases[i]
        source = self.source_for(i, identity)
        try:
            p, degraded = build_problem(case.problem, source.cfg, source.source_k)
        except Exception as err:
            log_error(case, identity, first_line(err))
            return Plan(i, None, False, first_line(err))
        return Plan(i, p, degraded, None)

    def enqueue_builds(self, plans: list[Plan]) -> None:
        for plan in plans:
            p = plan.problem
            if p is None:
                continue
            if p.splits > 1:
                self.pipeline.enqueue(support.REDUCE_KERNEL_NAME, None)
            self.pipeline.enqueue(support.gemmKernelName(p), p)
        self.pipeline.top_up()

    def kernel_names(self, p: Problem) -> list[str]:
        names = [support.gemmKernelName(p)]
        if p.splits > 1:
            names.insert(0, support.REDUCE_KERNEL_NAME)
        return names

    def make_job(self, plan: Plan) -> Job | Outcome | None:
        """The job of a plan whose kernels are built; None while they are
        building; an outcome when a build failed."""
        p = plan.problem
        kernels = []
        for name in self.kernel_names(p):
            built = self.pipeline.result(name)
            if built is None:
                return None
            if not built.ok:
                return Outcome(None, f"{built.stage}: {built.message}")
            kernels.append((name, self.pipeline.path(name)))
        kernels = tuple(kernels)
        return Job(
            plan.case_index,
            p,
            kernels,
            self.iters,
            self.warmup,
            self.warmup_ms,
            self.verify,
        )

    def covers_within(self, i: int, time_us: float) -> bool:
        return time_us <= self.cases[i].best_us * (1 + self.tolerance)

    def pick_gpu(self, case_index: int, in_flight: list[int]) -> int | None:
        """The GPU of a case: its fixed one, so that its operands stay cached
        there across rounds, or, when that one is full, an idle one."""
        home = case_index % self.num_gpus
        if in_flight[home] < JOBS_PER_GPU:
            return home
        idle = [g for g, count in enumerate(in_flight) if count == 0]
        return idle[0] if idle else None

    def dispatch(
        self,
        pending: list[Plan],
        in_flight: list[int],
        gpu_of: dict[int, int],
        resolve,
    ) -> None:
        """Sends the pending plans whose kernels are built to their GPUs, looking
        only at the first few so that the build window is not drained."""
        scan = 2 * JOBS_PER_GPU * self.num_gpus
        for plan in pending[:scan]:
            gpu = self.pick_gpu(plan.case_index, in_flight)
            if gpu is None:
                continue
            job = self.make_job(plan)
            if job is None:
                continue
            pending.remove(plan)
            if isinstance(job, Outcome):
                resolve(plan, job)
                continue
            gpu_of[plan.case_index] = gpu
            in_flight[gpu] += 1
            self.gpus.submit(gpu, job)

    def execute_plans(
        self, plans: list[Plan], identity: ConfigIdentity
    ) -> dict[int, Outcome]:
        """Times the plans on the GPUs as their kernels get built; returns the
        outcome of every case, which is also logged as it arrives."""
        outcomes: dict[int, Outcome] = {}
        pending = [p for p in plans if p.problem is not None]
        todo = len(plans)
        in_flight = [0] * self.num_gpus
        gpu_of: dict[int, int] = {}
        start = time.monotonic()

        def resolve(plan: Plan, outcome: Outcome) -> None:
            outcomes[plan.case_index] = outcome
            if outcome.error:
                log_error(self.cases[plan.case_index], identity, outcome.error)
            covered = self.count_covered(outcomes)
            line = self.progress_line(
                len(outcomes), todo, plan, outcome, start, covered
            )
            print(line, flush=True)

        by_case = {p.case_index: p for p in plans}
        for plan in plans:
            if plan.problem is None:
                resolve(plan, Outcome(None, plan.reason))
        while len(outcomes) < todo:
            self.dispatch(pending, in_flight, gpu_of, resolve)
            message = self.gpus.poll(POLL_S)
            if message is None:
                continue
            if message[0] == "fatal":
                raise RuntimeError(f"gpu worker {message[1]} failed:\n{message[2]}")
            _, case_index, time_us, error, busy = message
            in_flight[gpu_of[case_index]] -= 1
            self.busy_s += busy
            resolve(by_case[case_index], Outcome(time_us, error))
        return outcomes

    def drop_dominated(self, i: int, identity: ConfigIdentity, time_us: float) -> None:
        dominated = [
            d
            for d, c in self.candidates[i].items()
            if d != identity
            and d not in self.protected[i]
            and time_us <= c.time_us * (1 + self.tolerance)
        ]
        for d in dominated:
            del self.candidates[i][d]

    def count_covered(self, outcomes: dict[int, Outcome]) -> int:
        """The resolved cases within tolerance, the criterion of the cover."""
        return sum(
            1
            for i, o in outcomes.items()
            if o.time_us is not None and self.covers_within(i, o.time_us)
        )

    def progress_line(
        self,
        j: int,
        total: int,
        plan: Plan,
        outcome: Outcome,
        start: float,
        covered: int,
    ) -> str:
        case = self.cases[plan.case_index]
        elapsed = time.monotonic() - start
        eta = elapsed / j * (total - j)
        label = f"{case.text()} useBeta={int(case.problem['useBeta'])}"
        label += f" beta={case.problem['betaMode']}"
        time_us = outcome.time_us
        if time_us is None:
            status, timing = "failed", "time=n/a"
            timing += f" best={case.best_us:.2f}us ratio=n/a"
        else:
            within = self.covers_within(plan.case_index, time_us)
            status = "covered" if within else "not covered"
            timing = f"time={time_us:.2f}us best={case.best_us:.2f}us"
            timing += f" ratio={time_us / case.best_us:.3f}"
        return (
            f"  [{j}/{total}] {label} {status} {timing} "
            f"covered so far={covered} elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )

    def run_round(self, identity: ConfigIdentity, case_indices: list[int]) -> int:
        """Re-targets identity to case_indices; returns the new covers recorded."""
        if not case_indices:
            return 0
        for i in case_indices:
            self.tried[i].add(identity)
        plans = [self.plan_case(i, identity) for i in case_indices]
        self.enqueue_builds(plans)
        outcomes = self.execute_plans(plans, identity)
        measured: dict[int, float] = {}
        before = len(self.covers)
        for plan in plans:
            i = plan.case_index
            time_us = outcomes[i].time_us
            if time_us is None:
                continue
            measured[i] = time_us
            best = self.measured_best.get(i)
            if best is None or time_us < best[1]:
                self.measured_best[i] = (identity, time_us)
            self.record_within(i, identity, plan, time_us)
        for i, time_us in measured.items():
            self.drop_dominated(i, identity, time_us)
        self.rebuild_assignment()
        return len(self.covers) - before

    def round_limit(self) -> int:
        """Upper bound on greedy rounds: one timing pass per case-candidate pair."""
        return (
            sum(len(per) for per in self.candidates)
            + len(self.cases)
            + (self.max_configs or 0)
            + 1
        )

    def run(self) -> SelectionResult:
        round_number = 0
        limit = self.round_limit()
        while True:
            uncovered = self.uncovered()
            if not uncovered:
                break
            entries = tally(self.cases, uncovered, self.candidates, self.tried)
            if not entries:
                self.rebuild_assignment()
                return self.result(bool(self.uncovered()), search_exhausted=True)
            if round_number >= limit:
                print(
                    f"warning: round limit ({limit}) reached with "
                    f"{len(uncovered)} uncovered, stopping search",
                    flush=True,
                )
                break
            round_number += 1
            identity = entries[0].identity
            case_indices = self.round_cases(identity, uncovered)
            at_cap = (
                self.max_configs is not None and len(self.selected) >= self.max_configs
            )
            cap_note = " (library full, refine)" if at_cap else ""
            print(
                f"round {round_number}{cap_note}: testing {identity} frequency "
                f"{entries[0].frequency} on {len(case_indices)} cases "
                f"({len(uncovered)} uncovered)",
                flush=True,
            )
            if not case_indices:
                for i in uncovered:
                    self.tried[i].add(identity)
                continue
            start = time.monotonic()
            self.busy_s = 0.0
            new_covers = self.run_round(identity, case_indices)
            wall = time.monotonic() - start
            busy = 100 * self.busy_s / (wall * self.num_gpus)
            in_library = identity in self.selected
            print(
                f"round {round_number}: {identity} frequency "
                f"{entries[0].frequency}, {len(uncovered)} uncovered, covered "
                f"{new_covers}, "
                f"{'in library' if in_library and new_covers else 'wasted' if not new_covers else 'added'}, "
                f"{wall:.1f}s, gpu busy {busy:.0f}%",
                flush=True,
            )
        self.rebuild_assignment()
        return self.result(bool(self.uncovered()), search_exhausted=False)

    def result(self, cap_reached: bool, search_exhausted: bool) -> SelectionResult:
        return SelectionResult(
            self.selected,
            self.covers,
            self.measured_best,
            cap_reached,
            search_exhausted,
        )


def run_selection(
    cases: list[Case], args: argparse.Namespace, gpus: list[int]
) -> SelectionResult:
    work_dir = args.work_dir or os.path.join(args.output, "work")
    window = max(2 * args.jobs, BUILDS_PER_GPU * len(gpus))
    pipeline = KernelPipeline(work_dir, args.jobs, window)
    pool = GpuPool(gpus, int(args.cache_gb * 2**30))
    try:
        selector = Selector(
            cases,
            args.tolerance,
            args.iters,
            args.warmup,
            args.warmup_ms,
            args.verify,
            pipeline,
            pool,
            len(gpus),
            args.max_configs,
            args.min_config_cases,
        )
        return selector.run()
    finally:
        pool.stop()
        pipeline.close()


def rounded(value: float) -> float:
    return round(float(value), 4)


def config_record(identity: ConfigIdentity) -> dict:
    record = {name: getattr(identity, name) for name in identity._fields}
    record["grid3d"] = bool(identity.grid3d)
    return record


def cover_record(case: Case, cover: CoverRecord) -> dict:
    record = dict(case.label)
    record.update(
        kernel=cover.kernel_name,
        spk_used=int(cover.spk_used),
        spk_degraded=cover.spk_degraded,
        timeUs=rounded(cover.time_us),
        bestUs=rounded(case.best_us),
        ratio=rounded(cover.time_us / case.best_us),
    )
    return record


def selected_record(
    identity: ConfigIdentity, cases: list[Case], covers: dict[int, CoverRecord]
) -> dict:
    assigned = [i for i, c in sorted(covers.items()) if c.identity == identity]
    return {
        "config": config_record(identity),
        "kernels": sorted({covers[i].kernel_name for i in assigned}),
        "num_cases": len(assigned),
        "cases": [cover_record(cases[i], covers[i]) for i in assigned],
    }


def uncovered_record(
    case: Case, result: SelectionResult, reason: str, tolerance: float
) -> dict:
    record = dict(case.label)
    record["bestUs"] = rounded(case.best_us)
    best = result.measured_best.get(case.index)
    if best is None:
        record["reason"] = REASON_UNMEASURED
        record["measuredBest"] = None
        return record
    record["reason"] = reason
    identity, time_us = best
    record["measuredBest"] = {
        "config": config_record(identity),
        "timeUs": rounded(time_us),
        "ratio": rounded(time_us / case.best_us),
        "withinTolerance": time_us <= case.best_us * (1 + tolerance),
        "inLibrary": identity in result.selected,
    }
    return record


def write_asm(out_dir: str, covers: dict[int, CoverRecord]) -> None:
    asm_dir = os.path.join(out_dir, "asm")
    os.makedirs(asm_dir, exist_ok=True)
    written: set[str] = set()
    for _, cover in sorted(covers.items()):
        if cover.kernel_name in written:
            continue
        written.add(cover.kernel_name)
        path = os.path.join(asm_dir, f"{cover.kernel_name}.s")
        with open(path, "w") as stream:
            stream.write(support.gemmAsm(cover.problem))


def write_outputs(
    out_dir: str,
    cases: list[Case],
    result: SelectionResult,
    args: argparse.Namespace,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if result.search_exhausted:
        reason = REASON_EXHAUSTED
    elif result.cap_reached:
        reason = REASON_CAP
    else:
        reason = REASON_EXHAUSTED
    uncovered = [
        uncovered_record(c, result, reason, args.tolerance)
        for c in cases
        if c.index not in result.covers
    ]
    meta = {
        "results": os.path.abspath(args.results),
        "tolerance": args.tolerance,
        "iters": args.iters,
        "warmup": args.warmup,
        "warmup_ms": args.warmup_ms,
        "max_configs": args.max_configs,
        "min_config_cases": args.min_config_cases,
        "verify": args.verify,
        "only": args.only,
        "limit_cases": args.limit_cases,
        "num_cases": len(cases),
        "num_selected": len(result.selected),
        "num_covered": len(result.covers),
        "num_uncovered": len(uncovered),
    }
    document = {
        "meta": meta,
        "selected": [selected_record(s, cases, result.covers) for s in result.selected],
        "uncovered": uncovered,
    }
    with open(os.path.join(out_dir, "selection.yaml"), "w") as stream:
        yaml.safe_dump(
            document, stream, sort_keys=False, width=1000, default_flow_style=None
        )
    write_asm(out_dir, result.covers)


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


def non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select few skinny GEMM kernels that cover a sweep."
    )
    parser.add_argument("results", help="results YAML of sweep_skinny_gemm.py")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="library size cap (default: unlimited)",
    )
    parser.add_argument(
        "--min-config-cases",
        type=non_negative_int,
        default=0,
        help="drop configurations assigned fewer than this many shapes from the "
        "final library (default 0)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="allowed relative slowdown over the best time of a case",
    )
    parser.add_argument(
        "--iters",
        type=positive_int,
        default=DEFAULT_ITERS,
        help="timed launches per case (default 30)",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=DEFAULT_WARMUP,
        help="untimed launches before the timed ones (default 5, the value of "
        "the sweep that produced results.yml)",
    )
    parser.add_argument(
        "--warmup-ms",
        type=non_negative_float,
        default=DEFAULT_WARMUP_MS,
        help="after --warmup, keep launching until this many ms have passed, "
        "so the GPU clocks and pipeline settle (default 30, 0 disables)",
    )
    parser.add_argument(
        "--work-dir",
        help="kernel cache, hsaco/<kernel>.hsaco (default: OUT_DIR/work)",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help=f"auto (the GPUs below {MAX_BUSY_PERCENT}%% use) or a list like 0,1,3",
    )
    parser.add_argument(
        "--allow-busy", action="store_true", help="time on listed GPUs even if busy"
    )
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=DEFAULT_CACHE_GB,
        help="host bytes of operands, and device bytes of buffers, cached per "
        f"gpu worker, in GiB (default {DEFAULT_CACHE_GB})",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="assembler processes (default: min(32, half of the cpus))",
    )
    parser.add_argument(
        "--verify", action="store_true", help="check every timed result on the host"
    )
    parser.add_argument("--only", help="keep the cases whose name contains this")
    parser.add_argument(
        "--limit-cases", type=int, help="keep the first N cases after --only"
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    cases = load_cases(args.results, args.only, args.limit_cases)
    try:
        gpus = select_gpus(args.gpus, MAX_BUSY_PERCENT, args.allow_busy)
    except SweepAbort as err:
        sys.exit(f"error: {err}")
    print(f"{len(cases)} cases, gpus {gpus}, {args.jobs} assembler jobs", flush=True)
    result = run_selection(cases, args, gpus)
    write_outputs(args.output, cases, result, args)
    print(
        f"selected {len(result.selected)} configurations covering "
        f"{len(result.covers)} of {len(cases)} cases",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
