# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knob sweep benchmark of the emitted skinny MXFP8 GEMM.

For every problem of an input YAML (schema in sweep_skinny_gemm_config.py) the
tool emits, assembles and times every valid configuration of the compile-time
knobs of SkinnyGEMMConfig, and reports the fastest ones per problem, verified
against the host reference, in an output YAML.

Run from the tensilelite directory:

  source ~/.tensile/bin/activate
  export PYTHONPATH=$PWD/build_tmp/tensilelite/src/tensilelite/bindings
  export PYTHONPATH=$PYTHONPATH:$PWD/tests/tensilelite/bindings
  export LD_LIBRARY_PATH=/opt/rocm/lib
  python tests/tensilelite/bindings/sweep_skinny_gemm.py [input.yaml] \\
      [--gpus auto|1,3,6,7] [-o results.yaml] [--one-shot] [--output-format msgpack] \\
      [--only SUBSTRING] [--hang-timeout SECONDS] [--resume] [--list|--dry-run]

The default input is a one-shot input (refine_top, refine_iters and top are 0):
a timed or --dry-run sweep of it needs --one-shot, else it stops with an error
before building anything; --list works without it.

Stages, all driven by the main process, which never initializes HIP (so a
spawned child can safely own a GPU):

  1. Expand the input into cases (shape, split size, beta mode) and their valid
     configurations; configurations that violate the support constraints are
     recorded as failed to build (stage constraint).
  2. BUILD, CPU-parallel: a process pool (spawn) emits and assembles every
     unique kernel once, keyed by its kernel name, which encodes every
     compile-time knob including the derived UP_ITERS and USE_BETA, so kernels
     are shared across shapes and split sizes. The hsaco files are written to
     <work-dir>/hsaco/<name>.hsaco and an existing file is reused as a cache
     (use a fresh work dir after changing the emitter). A failure at emit or
     assemble is recorded and never stops the sweep.
  3. TIMED, one long-lived worker process per GPU, started with
     HIP_VISIBLE_DEVICES set so that it sees its device as 0 (a ROCR or CUDA
     mask of the parent is dropped, as it would renumber the devices). Each
     worker has its own pipe. A job is a whole case: once all its kernels are
     built the main process hands it to an idle worker, which times ALL its
     configurations strictly one after another. Kernels
     that share a GPU contend for CUs, HBM and the LLC and would corrupt each
     other's timings, whereas separate GPUs do not; handing out whole cases
     instead of single configurations keeps all candidates of a case on the
     same physical device, so silicon and clock differences between devices
     cannot reorder them. With one GPU the cases simply run serially while the
     CPU builds of the later ones overlap the timing.
     Per configuration: POST_LOAD_LAUNCHES (2) plus warmup untimed
     flush+launch pairs, then iters timed launches, each preceded by its own
     512 MiB LLC flush (twice the 256 MiB MALL), so that both operands, the
     scales and C start cold in HBM: N timed iterations are N flush launches
     plus N GEMM launches. The HIP events bracket only the kernel under test,
     never the flush, and are attached to the dispatch with
     hipExtModuleLaunchKernel (the fallback, named in meta.timing, records them
     around hipModuleLaunchKernel and reads 3 to 4 us more, with a floor of about
     6 us instead of 4 us); the score is the MEAN of the timed launches only
     (the warm launches never enter it). A 50 ms flush ramp brings the clocks up
     before the screening of every case, and the refinement rotates the order
     of the candidates every round. The flush kernel is about 8x faster than
     bench_skinny_gemm's. For split-K the reduce is timed WARM, without a flush,
     right after the GEMM launches of the case, as in the real pipeline where
     it follows its GEMM, and the score is the mean over the iterations of the
     flushed GEMM time plus the warm reduce time. The operands are generated
     once per worker and shape in parallel blocks and cached
     (sweep_skinny_gemm_inputs.py). Unless --one-shot is set, the refine_top
     best are re-timed in interleaved rounds so that slow drift hits them alike,
     and the winners are ranked by that refined score and verified in rank order
     against the host reference; a failing configuration is dropped and the next
     refined one promoted. With --one-shot every screened configuration is
     written to top (ranked by screening time), refinement is skipped, and each
     timed configuration gets one verification launch (full or sampled by size);
     failures are recorded in the failed list with verified=failed. Every ranked
     entry also lists its per-launch times in ns (gemmNs and, for split-K,
     reduceNs, the launches timeUs is the mean of; in refine mode also
     screeningGemmNs and screeningReduceNs), so that other statistics can be
     computed offline; refine entries add the informational minUs and p90Us.
     --output-format msgpack (or a .msgpack path) stores the document in binary.
     A worker that dies, or sends nothing (not even the progress heartbeat it
     sends at least every 5 s while it works) for --hang-timeout seconds
     (default 600), is killed and restarted, and its case is recorded with an
     error; a GPU whose workers were lost 3 times aborts the sweep. A kernel that
     leaves the device in an error state ends its case with an error and the
     worker is replaced by a fresh process. If a process of the build pool
     crashes, the pool is recreated, the kernels that were being built by the
     dead process are retried alone and fail on their own if they crash again;
     the sweep aborts after 3 pool crashes.
     In refine+top mode every reported winner is verified (never skipped), but only
     the winners: the other screened candidates are not. The reference is dense ("full")
     when (m + n) * k <= verify_full_max_elems, else sampled (see
     sweep_skinny_gemm_inputs.py); --no-verify turns the verification off and
     marks the winners "no".
  4. The results stream back; the output is rewritten atomically after finished
     cases, at most every 30 s (or 10 times the duration of the last rewrite, if
     longer) and always at the end, on abort and on Ctrl-C.
     The failed configurations are appended after every case to
     <output stem>.failed.yaml, ONE YAML list with a group per case: problem
     (written once) and failures, a list of {config, status, stage, reason}.
     Stale <name>.<pid>.tmp and <kernel>.hsaco.<pid>.building files of dead
     processes are removed at start.

--resume continues a sweep: the finished cases of the existing output (-o, or
the default path in --work-dir) are reused as they are and only the others are
built and timed. The previous results must have been made with the same
--one-shot and the same settings (including the verification flag), else the
sweep stops before building anything. A previous record is reused when its
problem record equals that of a current case and it has no error, so errored,
lost and faulted cases run again and the records of cases that are not in the
input are dropped. The failed file keeps only the groups of the reused cases.

useBeta defaults to [false, true] and beta to the modes zero (beta 0.0, the
kernel's exactly-zero path, C poisoned) and random (a nonzero draw), whenever the
beta matters (useBeta, or split-K, whose reduce applies beta * C): each mode is
its own result record (problem.betaMode) with its own ranking and verification,
and the kernels are built once. alpha is random in [0.25, 2] and a random beta in
[-2, -0.25] and [0.25, 2]; draws come from a generator seeded by --seed /
settings.seed and the problem's name, m, n, k and useBeta, so a run is
reproducible whatever the order and the GPU assignment. The resolved values are
in the problem records of the output (and of failed.yaml). A number in the input
pins alpha or beta.

Parallelism: builds are CPU-parallel across all unique kernels; timing is
serialized per GPU and parallel across GPUs by case; builds overlap the timing.

The clocks of the GPUs are locked for the whole run, builds included, only with
--require-stable-clocks (see GpuStabilizer), and the sweep aborts if a lock is
refused. Without the flag nothing is set and no elevated rights are needed;
amd-smi is still read for the GPU use and process list at selection, for the
PCI address of every GPU (the GPU ids are HIP ordinals, as in
HIP_VISIBLE_DEVICES, and are mapped to the amd-smi devices by PCI address, as
the numbering of amd-smi itself can differ), and for the sclk, mclk and throttle status sampled
around and during every case. A case with a throttler active is marked
throttled in its record and counted in casesThrottled.
"""

import argparse
import collections
import concurrent.futures
import contextlib
import datetime
import math
import multiprocessing
import multiprocessing.connection
import os
import queue
import re
import signal
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any

import yaml

import skinny_gemm_support as support
from bench_skinny_gemm import FLUSH_MB
from skinny_gemm_support import Problem
from sweep_skinny_gemm_config import (
    ConfigFailure,
    ExpandedCase,
    SweepConfigError,
    check_settings,
    config_record,
    expand_case,
    first_line,
    load_input,
)
from sweep_skinny_gemm_smi import (
    GpuStabilizer,
    SweepAbort,
    check_same_device,
    git_head,
    log,
    map_hip_ordinals,
    script_dir,
    select_gpus,
)
from sweep_skinny_gemm_results import (
    FORMAT_MSGPACK,
    FORMAT_YAML,
    dead_process_files,
    dump_block,
    failed_path,
    output_format,
    remove_stale_temporaries,
    write_results,
)
from sweep_skinny_gemm_resume import Resumed, load_resume
from sweep_skinny_gemm_worker import Job, gpu_worker, hsaco_file

TOOL = "sweep_skinny_gemm.py"
DEFAULT_INPUT = os.path.join(script_dir(), "sweep_skinny_gemm_shapes.yaml")
DEFAULT_BUILD_JOBS = 64
# How long the main loop waits for an event before it looks at the other source.
POLL_S = 0.1
BUILD_STATUS = "failed to build"
# The output is rewritten at most this often: a rewrite costs O(cases so far).
WRITE_INTERVAL_S = 30.0
# Rewrites take at most about 1 / (1 + WRITE_DUTY_FACTOR) of the time.
WRITE_DUTY_FACTOR = 10
# Errored case labels listed in the final summary.
MAX_ERROR_LABELS = 20
# A worker silent for this long (no message, not even a progress heartbeat) is
# considered hung.
HANG_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 10
# A GPU whose workers died this many times aborts the sweep.
MAX_WORKER_DEATHS = 3
# Crashes of the build pool after which the sweep aborts.
MAX_POOL_RESTARTS = 3
BUILDING_SUFFIX = ".building"
# The generation of the executor that retries the suspects alone.
ISOLATED_GENERATION = -1
CRASH_MESSAGE = "build process crashed (retried alone)"


@dataclass
class BuildResult:
    name: str
    ok: bool
    stage: str
    message: str


def ignore_sigint() -> None:
    """The main process handles Ctrl-C and shuts the pool down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextlib.contextmanager
def building_marker(path: str):
    """Names the kernel a pool process is building: a process that dies in the
    assembler leaves the marker behind, which names the suspect."""
    marker = f"{path}.{os.getpid()}{BUILDING_SUFFIX}"
    with open(marker, "w"):
        pass
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(marker)


def compile_kernel(name: str, problem: Problem | None, path: str) -> BuildResult:
    # The emitter reports its errors as ValueError, but a failure of any kind
    # must only fail this kernel.
    try:
        asm = support.reduceAsm() if problem is None else support.gemmAsm(problem)
    except Exception as err:
        return BuildResult(name, False, "emit", first_line(err))
    try:
        hsaco = support.assemble(asm)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as stream:
            stream.write(hsaco)
        os.replace(temporary, path)
    except Exception as err:
        return BuildResult(name, False, "assemble", first_line(err))
    return BuildResult(name, True, "", "built")


def build_kernel(name: str, problem: Problem | None, hsaco_dir: str) -> BuildResult:
    """Emits and assembles one kernel into its hsaco file; the reduce kernel
    when problem is None. Runs in a pool process."""
    path = hsaco_file(hsaco_dir, name)
    if os.path.exists(path):
        return BuildResult(name, True, "", "cached")
    with building_marker(path):
        return compile_kernel(name, problem, path)


def make_build_executor(jobs: int) -> concurrent.futures.Executor:
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=ignore_sigint,
    )


def crashed_kernels(hsaco_dir: str) -> list[str]:
    """The kernels whose build marker belongs to a dead process; removes the
    markers."""
    names = []
    for marker in dead_process_files(hsaco_dir, suffix=BUILDING_SUFFIX):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(hsaco_dir, marker))
        stem = re.sub(rf"\.\d+{re.escape(BUILDING_SUFFIX)}$", "", marker)
        names.append(stem.removesuffix(".hsaco"))
    return names


class BuildStage:
    """The CPU-parallel build of the unique kernels.

    A crash of a pool process (the assembler segfaulting) breaks the whole
    executor. The stage then starts a new one, retries the innocent kernels
    there and retries the suspects, the kernels with a marker of a dead process,
    alone and one at a time, so that a crashing kernel fails only itself."""

    def __init__(
        self,
        hsaco_dir: str,
        jobs: int,
        executor_factory: Callable[[int], concurrent.futures.Executor] | None = None,
    ):
        self.hsaco_dir = hsaco_dir
        self.jobs = jobs
        self.factory = executor_factory or make_build_executor
        self.executor = self.factory(jobs)
        self.generation = 0
        self.restarts = 0
        self.isolated = None
        self.isolated_running = False
        # Name, problem and executor generation of every future in flight.
        self.pending: dict[
            concurrent.futures.Future, tuple[str, Problem | None, int]
        ] = {}
        self.done: queue.SimpleQueue = queue.SimpleQueue()
        self.suspects: collections.deque[tuple[str, Problem | None]] = (
            collections.deque()
        )
        self.fresh: list[BuildResult] = []
        self.results: dict[str, BuildResult] = {}
        self.submitted: set[str] = set()

    def submit(self, name: str, problem: Problem | None) -> None:
        if name in self.submitted:
            return
        self.submitted.add(name)
        try:
            self.launch(self.executor, name, problem, self.generation)
        except BrokenProcessPool:
            self.recover()
            self.launch(self.executor, name, problem, self.generation)

    def launch(self, executor, name: str, problem, generation: int) -> None:
        future = executor.submit(build_kernel, name, problem, self.hsaco_dir)
        self.pending[future] = (name, problem, generation)
        future.add_done_callback(self.done.put)

    def poll(self, timeout: float) -> list[BuildResult]:
        """The builds finished within timeout."""
        if self.pending:
            for future in self.completed(timeout):
                self.collect(future)
        self.start_next_suspect()
        finished, self.fresh = self.fresh, []
        return finished

    def completed(self, timeout: float) -> list[concurrent.futures.Future]:
        """Blocks for the first finished future, then takes the others that are
        ready."""
        try:
            if timeout > 0:
                futures = [self.done.get(timeout=timeout)]
            else:
                futures = [self.done.get_nowait()]
        except queue.Empty:
            return []
        while True:
            try:
                futures.append(self.done.get_nowait())
            except queue.Empty:
                return futures

    def record(self, result: BuildResult) -> None:
        self.results[result.name] = result
        self.fresh.append(result)

    def record_future(self, name: str, future: concurrent.futures.Future) -> None:
        try:
            result = future.result()
        except Exception as err:
            message = f"build process failed: {first_line(err)}"
            result = BuildResult(name, False, "emit", message)
        self.record(result)

    def collect(self, future: concurrent.futures.Future) -> None:
        # A future of a replaced executor was already handled by recover.
        entry = self.pending.get(future)
        if entry is None:
            return
        name, _, generation = entry
        broken = not future.cancelled() and isinstance(
            future.exception(), BrokenProcessPool
        )
        if broken and generation == ISOLATED_GENERATION:
            self.fail_isolated(future, name)
            return
        if broken:
            if generation == self.generation:
                self.recover()
            return
        if generation == ISOLATED_GENERATION:
            self.isolated_running = False
        del self.pending[future]
        self.record_future(name, future)

    def fail_isolated(self, future: concurrent.futures.Future, name: str) -> None:
        """The suspect crashed the pool process again: the kernel fails."""
        del self.pending[future]
        self.isolated_running = False
        crashed_kernels(self.hsaco_dir)
        self.isolated.shutdown(wait=False, cancel_futures=True)
        self.isolated = None
        self.record(BuildResult(name, False, "emit", CRASH_MESSAGE))

    def recover(self) -> None:
        """Replaces the broken executor: suspects wait for their solo retry, the
        other kernels in flight are submitted again."""
        self.restarts += 1
        suspects = crashed_kernels(self.hsaco_dir)
        if self.restarts > MAX_POOL_RESTARTS:
            names = ", ".join(suspects) or "unknown"
            raise SweepAbort(
                f"the build pool crashed {self.restarts} times; last suspects: {names}"
            )
        old, generation = self.executor, self.generation
        stale = {f: e for f, e in self.pending.items() if e[2] == generation}
        for future in stale:
            del self.pending[future]
        self.generation += 1
        self.executor = self.factory(self.jobs)
        old.shutdown(wait=False, cancel_futures=True)
        for future, (name, problem, _) in stale.items():
            self.settle(future, name, problem, name in suspects)

    def settle(self, future, name: str, problem, suspect: bool) -> None:
        """Keeps what a future of a broken executor finished with, else retries
        the kernel."""
        finished = future.done() and not future.cancelled()
        if finished and not isinstance(future.exception(), BrokenProcessPool):
            self.record_future(name, future)
        elif suspect:
            self.suspects.append((name, problem))
        else:
            self.launch(self.executor, name, problem, self.generation)

    def start_next_suspect(self) -> None:
        if self.isolated_running or not self.suspects:
            return
        name, problem = self.suspects.popleft()
        if self.isolated is None:
            self.isolated = self.factory(1)
        self.isolated_running = True
        self.launch(self.isolated, name, problem, ISOLATED_GENERATION)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.isolated is not None:
            self.isolated.shutdown(wait=False, cancel_futures=True)


@dataclass
class GpuWorker:
    """The process of one GPU, the pipe to it and what it is doing."""

    gpu: int
    process: Any
    conn: Any
    ready: bool = False
    job: Job | None = None
    last_seen: float = field(default_factory=time.monotonic)
    dispatched: float = 0.0
    deaths: int = 0
    broken: bool = False


class GpuPool:
    """One worker process per GPU, each with its own pipe: a killed worker
    cannot corrupt what the others read, and the main process always knows which
    case each worker holds."""

    def __init__(self, gpus: list[int], hang_timeout_s: float):
        self.context = multiprocessing.get_context("spawn")
        self.hang_timeout_s = hang_timeout_s
        self.workers = [GpuWorker(gpu, *self.spawn(gpu)) for gpu in gpus]

    def spawn(self, gpu: int) -> tuple:
        parent, child = self.context.Pipe()
        process = self.context.Process(
            target=gpu_worker, args=(gpu, child), daemon=True
        )
        process.start()
        child.close()
        return process, parent

    def idle(self) -> list[GpuWorker]:
        return [w for w in self.workers if w.ready and w.job is None and not w.broken]

    def send(self, worker: GpuWorker, job: Job) -> None:
        worker.job = job
        worker.last_seen = worker.dispatched = time.monotonic()
        try:
            worker.conn.send(job)
        except (OSError, ValueError):
            # lost() reports the worker and the case it holds.
            worker.broken = True

    def poll(self, timeout: float) -> list[tuple[GpuWorker, tuple]]:
        """The messages that arrived within timeout."""
        live = {w.conn: w for w in self.workers if not w.broken}
        messages = []
        for conn in multiprocessing.connection.wait(list(live), timeout):
            worker = live[conn]
            self.drain(worker, messages)
        return messages

    @staticmethod
    def drain(worker: GpuWorker, messages: list) -> None:
        """Receives all the messages that are available, so that the last words
        of a worker that then died are not lost."""
        while True:
            try:
                message = worker.conn.recv()
            except (EOFError, OSError):
                worker.broken = True
                return
            worker.last_seen = time.monotonic()
            messages.append((worker, message))
            if not worker.conn.poll(0):
                return

    def lost(self) -> list[tuple[GpuWorker, str]]:
        """The workers that died or hang, with the reason."""
        now = time.monotonic()
        lost = []
        for worker in self.workers:
            if not worker.process.is_alive():
                reason = f"worker died (exit code {worker.process.exitcode})"
                lost.append((worker, reason))
                continue
            if worker.broken:
                lost.append((worker, "worker connection lost"))
                continue
            busy = worker.job is not None or not worker.ready
            if busy and now - worker.last_seen > self.hang_timeout_s:
                reason = f"timeout: no progress for {self.hang_timeout_s:.0f} s"
                lost.append((worker, reason))
        return lost

    @staticmethod
    def reap(worker: GpuWorker) -> None:
        process = worker.process
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        worker.conn.close()

    def restart(self, worker: GpuWorker) -> None:
        """A fresh process for the GPU, with a clean HIP context."""
        self.reap(worker)
        worker.process, worker.conn = self.spawn(worker.gpu)
        worker.ready = False
        worker.job = None
        worker.broken = False
        worker.last_seen = time.monotonic()

    def stop(self) -> None:
        # The pipes have no feeder threads, so there is nothing to drain or to
        # cancel_join_thread on, unlike with queues.
        for worker in self.workers:
            with contextlib.suppress(OSError, ValueError):
                worker.conn.send(None)
        for worker in self.workers:
            worker.process.join(STOP_TIMEOUT_S)
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.terminate()
        for worker in self.workers:
            worker.process.join(5)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
            worker.conn.close()


@dataclass
class CaseState:
    """A case on its way through the stages."""

    index: int
    expanded: ExpandedCase
    names: list[str]
    kernels: dict[str, Problem | None]
    waiting: int = 0
    entries: list[tuple[str, Problem]] = field(default_factory=list)
    build_failures: list[ConfigFailure] = field(default_factory=list)


def make_state(index: int, expanded: ExpandedCase) -> CaseState:
    names = [support.gemmKernelName(p) for p in expanded.problems]
    kernels: dict[str, Problem | None] = {}
    for name, p in zip(names, expanded.problems):
        kernels.setdefault(name, p)
    if any(p.splits > 1 for p in expanded.problems):
        kernels[support.REDUCE_KERNEL_NAME] = None
    return CaseState(index, expanded, names, kernels)


class ResultWriter:
    """The output document, rewritten atomically (throttled), and the failed
    configurations, appended per case to their own YAML list."""

    def __init__(
        self,
        path: str,
        meta: dict,
        output_format: str,
        interval_s: float = WRITE_INTERVAL_S,
        resumed: Resumed | None = None,
    ):
        self.path = path
        self.output_format = output_format
        self.failed_path = failed_path(path)
        self.meta = meta
        self.interval_s = interval_s
        self.last_write = -math.inf
        self.rewrite_s = 0.0
        self.records: dict[int, dict] = dict(resumed.records) if resumed else {}
        self.reset_failed(resumed.failures if resumed else [])

    def reset_failed(self, groups: list[dict]) -> None:
        """Starts the failed file again, atomically, with the kept groups."""
        temporary = f"{self.failed_path}.{os.getpid()}.tmp"
        with open(temporary, "w") as stream:
            if groups:
                dump_block(groups, stream)
        os.replace(temporary, self.failed_path)

    def add(self, index: int, record: dict, failures: list[dict]) -> None:
        self.records[index] = record
        if failures:
            group = [{"problem": record["problem"], "failures": failures}]
            with open(self.failed_path, "a") as stream:
                dump_block(group, stream)
        self.write(force=False)

    def write(self, force: bool) -> None:
        """Rewrites the document, unless the last rewrite is more recent than
        the interval, which grows with the document so that the rewrites never
        take more than a tenth of the time; force always writes."""
        wait = max(self.interval_s, WRITE_DUTY_FACTOR * self.rewrite_s)
        started = time.monotonic()
        if not force and started - self.last_write < wait:
            return
        results = [self.records[index] for index in sorted(self.records)]
        document = {"meta": self.meta, "results": results}
        write_results(self.path, document, self.output_format)
        self.last_write = time.monotonic()
        self.rewrite_s = self.last_write - started


def timing_meta(method: str) -> dict:
    """How the times of the document were taken."""
    return {
        "method": method,
        "flushMiB": FLUSH_MB,
        "statistic": "mean of the timed launches",
        "flushPerLaunch": True,
        "reduce": "warm: not flushed, its launches follow the GEMM launches",
    }


def summarize(labels: list[str]) -> str:
    """The labels, at most MAX_ERROR_LABELS of them."""
    shown = labels[:MAX_ERROR_LABELS]
    hidden = len(labels) - len(shown)
    return "; ".join(shown) + (f"; ... and {hidden} more" if hidden else "")


def failure_entry(state: CaseState, failure: ConfigFailure, status: str) -> dict:
    return {
        "config": failure.config,
        "status": status,
        "stage": failure.stage,
        "reason": failure.reason,
    }


def top_entries(state: CaseState, outcome: dict) -> list[dict]:
    top = []
    for rank, winner in enumerate(outcome["ranked"], start=1):
        name, p = state.entries[winner["index"]]
        entry = {"rank": rank, "config": config_record(p), "kernel": name}
        entry.update({key: v for key, v in winner.items() if key != "index"})
        top.append(entry)
    return top


def run_failures(state: CaseState, outcome: dict) -> list[tuple[str, ConfigFailure]]:
    """The (status, failure) pairs of the configurations that failed on the GPU."""
    failures = []
    for item in outcome["failures"]:
        _, p = state.entries[item["index"]]
        failure = ConfigFailure(item["stage"], item["reason"], config_record(p))
        failures.append((item["status"], failure))
    return failures


def is_throttled(clocks: dict) -> bool:
    """True if a sample during the case, or at its start or end, saw a nonzero
    throttle status (some throttler was active; the bits are ASIC specific)."""
    if clocks.get("during", {}).get("throttledSamples", 0) > 0:
        return True
    return any(clocks.get(key, {}).get("throttleStatus") for key in ("start", "end"))


def case_record(state: CaseState, outcome: dict | None) -> dict:
    """The result of a case; outcome is None when nothing was timed."""
    constraint = state.expanded.failures
    reasons = Counter(
        f"{f.stage}: {f.reason}" for f in constraint + state.build_failures
    )
    outcome = outcome or {"ranked": [], "failures": [], "timed": 0}
    statuses = Counter(item["status"] for item in outcome["failures"])
    record = {
        "problem": state.expanded.case.record(),
        "counts": {
            "swept": state.expanded.swept,
            "builtOk": len(state.entries),
            "failedToBuild": len(constraint) + len(state.build_failures),
            "failedToRun": statuses["failed to run"],
            "wrongResult": statuses["wrong result"],
            "timed": outcome["timed"],
        },
        "failedToBuildReasons": dict(reasons.most_common()),
        "top": top_entries(state, outcome),
    }
    if outcome.get("phases"):
        record["phaseSeconds"] = outcome["phases"]
    if "error" in outcome:
        record["error"] = outcome["error"]
    if is_throttled(outcome.get("clocks", {})):
        record["throttled"] = True
    return record


def all_failure_entries(state: CaseState, outcome: dict | None) -> list[dict]:
    built = state.expanded.failures + state.build_failures
    entries = [failure_entry(state, f, BUILD_STATUS) for f in built]
    if outcome is not None:
        entries += [
            failure_entry(state, failure, status)
            for status, failure in run_failures(state, outcome)
        ]
    return entries


class Sweep:
    """Drives the build and timed stages of the expanded cases."""

    def __init__(
        self,
        options,
        settings: dict,
        expanded: list[ExpandedCase],
        resumed: Resumed | None = None,
    ):
        self.options = options
        self.settings = settings
        self.resumed = resumed
        self.states = [make_state(i, e) for i, e in enumerate(expanded)]
        self.hsaco_dir = os.path.join(options.work_dir, "hsaco")
        self.builds = None
        self.pool = None
        reused = resumed.records if resumed else {}
        self.remaining = len(self.states) - len(reused)
        self.waiters: dict[str, list[CaseState]] = {}
        self.meta = {}
        self.writer = None
        self.gpu_meta: dict[int, dict] = {}
        self.errored_labels: list[str] = []
        self.ready_jobs: collections.deque[Job] = collections.deque()
        self.finished: set[int] = set(reused)
        self.lost_cases: dict[int, list[str]] = collections.defaultdict(list)

    def start(self, gpus: list[int], stabilizer: GpuStabilizer) -> None:
        os.makedirs(self.hsaco_dir, exist_ok=True)
        self.remove_stale_files()
        self.meta = self.make_meta(gpus, stabilizer)
        self.writer = ResultWriter(
            self.options.output,
            self.meta,
            self.options.output_format,
            resumed=self.resumed,
        )
        self.builds = BuildStage(self.hsaco_dir, self.options.build_jobs)
        # Nothing left to time: no worker needs to own a GPU.
        timed = bool(gpus) and self.remaining > 0
        self.pool = GpuPool(gpus, self.options.hang_timeout) if timed else None
        self.submit_builds()

    def remove_stale_files(self) -> None:
        output = os.path.abspath(self.options.output)
        removed = remove_stale_temporaries(self.hsaco_dir)
        removed += remove_stale_temporaries(self.hsaco_dir, suffix=BUILDING_SUFFIX)
        for path in (output, failed_path(output)):
            removed += remove_stale_temporaries(
                os.path.dirname(path), os.path.basename(path) + "."
            )
        if removed:
            log(f"removed {removed} stale temporary file(s)")

    def make_meta(self, gpus: list[int], stabilizer: GpuStabilizer) -> dict:
        locks = stabilizer.summary()
        per_gpu = locks.get("perGpu", {})
        self.gpu_meta = {
            gpu: {"id": gpu, "name": None, "timing": None, "clockLock": per_gpu.get(gpu), "clocks": []}
            for gpu in gpus
        }
        meta = {
            "tool": TOOL,
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "gitHead": git_head(script_dir()),
            "arch": support.ARCH,
            "gpuName": None,
            "timing": None,
            "inputFile": os.path.abspath(self.options.input),
            "settings": meta_settings(self.settings, self.options),
            "oneShot": self.options.one_shot,
            "outputFormat": self.options.output_format,
            "verification": self.verification_note(),
            "gpus": list(self.gpu_meta.values()),
            "clockLock": {key: v for key, v in locks.items() if key != "perGpu"},
            "buildJobs": self.options.build_jobs,
            "workDir": os.path.abspath(self.options.work_dir),
            "kernelsBuilt": 0,
            "kernelsFailed": 0,
            "resumedCases": 0,
            "casesDone": 0,
            "casesErrored": 0,
            "casesThrottled": 0,
            "interrupted": False,
            "aborted": None,
            "casesTotal": len(self.states),
            "elapsedSeconds": 0,
        }
        if self.resumed:
            self.adopt_resumed(meta)
        return meta

    def adopt_resumed(self, meta: dict) -> None:
        """Counts the reused cases; when nothing is timed, the device and the
        timing method are those of the previous results."""
        records = self.resumed.records.values()
        meta["resumedCases"] = meta["casesDone"] = len(self.resumed.records)
        meta["casesThrottled"] = sum(1 for r in records if r.get("throttled"))
        if self.remaining:
            return
        for key in ("gpuName", "timing", "gpus"):
            meta[key] = self.resumed.meta.get(key)

    def verification_note(self) -> str:
        if self.options.no_verify:
            return "disabled by --no-verify: configurations are marked no"
        if self.options.one_shot:
            limit = self.settings["verify_full_max_elems"]
            return (
                "one-shot: every timed configuration is verified once (full when "
                f"(m + n) * k <= {limit}, else sampled); failures stay listed with "
                "verified=failed"
            )
        limit = self.settings["verify_full_max_elems"]
        return (
            f"every winner is verified: full when (m + n) * k <= {limit}, "
            "else sampled; the screened candidates are not verified"
        )

    def submit_builds(self) -> None:
        """In case order, so that the early cases finish first."""
        states = [s for s in self.states if s.index not in self.finished]
        for state in states:
            for name, problem in state.kernels.items():
                self.builds.submit(name, problem)
                self.waiters.setdefault(name, []).append(state)
                state.waiting += 1
        for state in states:
            if state.waiting == 0:
                self.case_built(state)

    def on_build(self, result: BuildResult) -> None:
        key = "kernelsBuilt" if result.ok else "kernelsFailed"
        self.meta[key] += 1
        for state in self.waiters.pop(result.name, []):
            state.waiting -= 1
            if state.waiting == 0:
                self.case_built(state)

    def case_built(self, state: CaseState) -> None:
        """Splits the configurations into built and failed and queues the case
        for timing."""
        results = self.builds.results
        reduce = results.get(support.REDUCE_KERNEL_NAME)
        for name, p in zip(state.names, state.expanded.problems):
            reduce_failed = reduce is not None and not reduce.ok and p.splits > 1
            result = reduce if reduce_failed else results[name]
            if result.ok:
                state.entries.append((name, p))
                continue
            reason = result.message
            if result is reduce:
                reason = f"{support.REDUCE_KERNEL_NAME}: {reason}"
            state.build_failures.append(
                ConfigFailure(result.stage, reason, config_record(p))
            )
        if self.pool is None or not state.entries:
            self.finish_case(state, None)
            return
        job = Job(
            state.index,
            state.entries,
            self.hsaco_dir,
            self.settings,
            not self.options.no_verify,
            self.options.one_shot,
        )
        self.ready_jobs.append(job)

    def dispatch(self) -> None:
        """Hands the ready cases to the idle workers."""
        for worker in self.pool.idle():
            if not self.ready_jobs:
                return
            self.pool.send(worker, self.ready_jobs.popleft())

    def finish_case(self, state: CaseState, outcome: dict | None) -> None:
        self.finished.add(state.index)
        record = case_record(state, outcome)
        if record.get("throttled"):
            self.meta["casesThrottled"] += 1
            log(
                f"warning: {state.expanded.case.label}: a throttler was active "
                "while timing; the times may be pessimistic"
            )
        self.writer.add(state.index, record, all_failure_entries(state, outcome))
        self.remaining -= 1
        self.meta["casesDone"] += 1
        if "error" in record:
            self.meta["casesErrored"] += 1
            self.errored_labels.append(state.expanded.case.label)
        self.report(state, record, outcome)

    def report(self, state: CaseState, record: dict, outcome: dict | None) -> None:
        counts = record["counts"]
        done = f"[{self.meta['casesDone']}/{self.meta['casesTotal']}]"
        best = "not timed"
        if record["top"]:
            top = record["top"][0]
            best = f"best {top['timeUs']} us {top['kernel']}"
        where = f" on GPU {outcome['gpu']} in {outcome['seconds']}s" if outcome else ""
        log(
            f"{done} {state.expanded.case.label}: {best}{where} (built "
            f"{counts['builtOk']}, failed to build {counts['failedToBuild']}, "
            f"failed to run {counts['failedToRun']}, wrong {counts['wrongResult']})"
        )
        if "error" in record:
            log(f"warning: {state.expanded.case.label}: {record['error']}")

    def on_message(self, worker: GpuWorker, message: tuple) -> None:
        kind, gpu, payload = message
        if kind == "ready":
            check_same_device(gpu, payload.get("pci"))
            worker.ready = True
            self.gpu_meta[gpu]["name"] = payload["name"]
            self.gpu_meta[gpu]["timing"] = payload["timing"]
            self.meta["gpuName"] = payload["name"]
            if self.meta["timing"] is None:
                self.meta["timing"] = timing_meta(payload["timing"])
            return
        if kind == "progress":
            return
        if kind == "fatal":
            log(f"error: the worker of GPU {gpu} failed:\n{payload}")
            lines = [line for line in payload.splitlines() if line.strip()]
            self.handle_lost(worker, f"worker failed: {lines[-1] if lines else ''}")
            return
        self.on_case(worker, payload)

    def on_case(self, worker: GpuWorker, outcome: dict) -> None:
        worker.job = None
        state = self.states[outcome["index"]]
        samples = {"case": state.expanded.case.label, **outcome["clocks"]}
        self.gpu_meta[outcome["gpu"]]["clocks"].append(samples)
        self.finish_case(state, outcome)
        if not outcome.get("fault"):
            return
        # A faulting kernel is not a GPU death: the new process starts clean.
        log(
            f"warning: {state.expanded.case.label}: {outcome['error']}; "
            f"restarting the worker of GPU {worker.gpu}"
        )
        self.pool.restart(worker)

    def handle_lost(self, worker: GpuWorker, reason: str) -> None:
        if worker.job is not None:
            self.lose_case(worker, reason)
        worker.deaths += 1
        if worker.deaths >= MAX_WORKER_DEATHS:
            raise SweepAbort(self.abort_message(worker, reason))
        log(f"warning: the worker of GPU {worker.gpu} is lost ({reason}); restarting")
        self.pool.restart(worker)

    def lose_case(self, worker: GpuWorker, reason: str) -> None:
        state = self.states[worker.job.index]
        outcome = {
            "index": state.index,
            "gpu": worker.gpu,
            "ranked": [],
            "failures": [],
            "timed": 0,
            "clocks": {},
            "phases": {},
            "seconds": round(time.monotonic() - worker.dispatched, 2),
            "error": reason,
        }
        worker.job = None
        self.lost_cases[worker.gpu].append(state.expanded.case.label)
        self.finish_case(state, outcome)

    def abort_message(self, worker: GpuWorker, reason: str) -> str:
        return (
            f"the worker of GPU {worker.gpu} was lost {worker.deaths} times (last: "
            f"{reason}); cases lost on it: {summarize(self.lost_cases[worker.gpu])}; "
            f"cases not completed: {summarize(self.unfinished_labels())}"
        )

    def unfinished_labels(self) -> list[str]:
        return [
            state.expanded.case.label
            for state in self.states
            if state.index not in self.finished
        ]

    def loop(self) -> None:
        start = time.time()
        while self.remaining:
            for result in self.builds.poll(0 if self.pool else POLL_S):
                self.on_build(result)
            if self.pool is not None:
                self.dispatch()
                for worker, message in self.pool.poll(POLL_S):
                    self.on_message(worker, message)
                for worker, reason in self.pool.lost():
                    self.handle_lost(worker, reason)
            self.meta["elapsedSeconds"] = round(time.time() - start, 1)

    def run(self) -> int:
        status = 0
        try:
            self.loop()
        except KeyboardInterrupt:
            log("interrupted: the results so far are written")
            self.meta["interrupted"] = True
            self.log_not_completed()
            status = 130
        except SweepAbort as err:
            log(f"error: {err}")
            self.meta["aborted"] = str(err)
            self.log_not_completed()
            status = 1
        finally:
            self.shutdown()
        self.log_errored()
        if status == 0 and self.meta["casesErrored"]:
            return 1
        return status

    def log_errored(self) -> None:
        if not self.errored_labels:
            return
        labels = self.errored_labels
        log(f"{len(labels)} case(s) errored: {summarize(labels)}")

    def log_not_completed(self) -> None:
        labels = self.unfinished_labels()
        self.meta["casesNotCompleted"] = len(labels)
        if labels:
            log(f"{len(labels)} case(s) not completed: {summarize(labels)}")

    def shutdown(self) -> None:
        """Stops what start got to create; the final write happens whatever
        fails on the way."""
        try:
            try:
                if self.pool is not None:
                    self.pool.stop()
            finally:
                if self.builds is not None:
                    self.builds.shutdown()
        finally:
            if self.writer is not None:
                self.writer.write(force=True)


def select_cases(cases: list, only: str | None, limit: int | None) -> list:
    if only:
        cases = [case for case in cases if only in case.name]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise SweepConfigError("no problem selected (--only/--limit-shapes)")
    return cases


def print_listing(expanded: list[ExpandedCase]) -> None:
    kernels = set()
    total = 0
    for item in expanded:
        valid = len(item.problems)
        total += valid
        kernels.update(support.gemmKernelName(p) for p in item.problems)
        print(
            f"{item.case.label}: swept={item.swept} valid={valid} "
            f"constraintFailures={len(item.failures)}"
        )
    print(
        f"total: shapes={len(expanded)} configs={total} "
        f"uniqueGemmKernels<={len(kernels)}"
    )


def print_dry_run(sweep: Sweep) -> None:
    reasons = Counter()
    for state in sweep.states:
        reasons.update(f"{f.stage}: {f.reason}" for f in state.expanded.failures)
        reasons.update(f"{f.stage}: {f.reason}" for f in state.build_failures)
    print(
        f"kernels built {sweep.meta['kernelsBuilt']}, "
        f"failed {sweep.meta['kernelsFailed']}; configs failed to build "
        f"{sum(reasons.values())}"
    )
    for reason, count in reasons.most_common(10):
        print(f"  {count:6d}  {reason}")


def meta_settings(settings: dict, options) -> dict:
    """The settings as the output records them."""
    return {**settings, "verify": not options.no_verify}


def resolve_settings(settings: dict, options) -> dict:
    resolved = dict(settings)
    for key in settings:
        value = getattr(options, key)
        if value is not None:
            resolved[key] = value
    check_settings(resolved, options.one_shot)
    if options.build_jobs < 1:
        raise SweepConfigError(
            f"--build-jobs: must be at least 1, got {options.build_jobs}"
        )
    if not 0 <= options.max_busy <= 100:
        raise SweepConfigError(
            f"--max-busy: must be in [0, 100], got {options.max_busy}"
        )
    if not math.isfinite(options.hang_timeout) or options.hang_timeout <= 0:
        raise SweepConfigError(
            f"--hang-timeout: must be finite and positive, got {options.hang_timeout}"
        )
    return resolved


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the knobs of the skinny MXFP8 GEMM over a list of shapes"
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="input YAML")
    parser.add_argument("-o", "--output", default=None, help="output YAML or msgpack")
    parser.add_argument(
        "--output-format",
        choices=(FORMAT_YAML, FORMAT_MSGPACK),
        default=None,
        help="yaml or msgpack (default: yaml, or msgpack when -o ends with .msgpack)",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="screen every configuration once, save all to top, skip refinement",
    )
    parser.add_argument("--work-dir", default=None, help="hsaco cache directory")
    parser.add_argument(
        "--gpus",
        default="auto",
        help="auto or a list such as 1,3 of HIP ordinals (the numbering of "
        "HIP_VISIBLE_DEVICES, mapped to amd-smi by PCI address)",
    )
    parser.add_argument("--max-busy", type=float, default=5.0, help="idle GPU use, %%")
    parser.add_argument(
        "--allow-busy", action="store_true", help="time on listed busy GPUs"
    )
    parser.add_argument(
        "--build-jobs", type=int, default=min(os.cpu_count() or 1, DEFAULT_BUILD_JOBS)
    )
    parser.add_argument(
        "--hang-timeout",
        type=float,
        default=HANG_TIMEOUT_S,
        help="seconds without any message from a GPU worker (it sends a progress "
        "heartbeat at least every 5 s while it works) after which it is considered "
        "hung, killed and restarted (default: %(default)s)",
    )
    parser.add_argument("--only", default=None, help="problems whose name has this")
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="count, do not build")
    parser.add_argument("--dry-run", action="store_true", help="build, no GPU")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the finished cases of the existing output (same --one-shot and "
        "settings required); errored cases and new problems run again, and records "
        "of problems not in the current selection (for example with --only) are "
        "dropped from the rewritten output",
    )
    parser.add_argument(
        "--require-stable-clocks",
        action="store_true",
        help="lock the GPU clocks for the run; abort if any GPU cannot be locked",
    )
    for name in ("iters", "warmup", "refine_top", "refine_iters", "top"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=int)
    parser.add_argument(
        "--verify-full-max-elems", dest="verify_full_max_elems", type=int
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed of the random alpha and beta of the problems (settings.seed): "
        "a draw depends only on it and on the name, m, n, k and useBeta of the "
        "problem; random alpha is in [0.25, 2], random beta in [-2, -0.25] and "
        "[0.25, 2]; every beta problem also runs with beta 0 (betaModes)",
    )
    return parser


def prepare_directories(options) -> None:
    if options.work_dir is None:
        options.work_dir = tempfile.mkdtemp(prefix="skinny-sweep-")
    if options.output is None:
        name = "results.msgpack" if options.one_shot else "results.yaml"
        options.output = os.path.join(options.work_dir, name)
    options.output_format = output_format(options.output, options.output_format)
    os.makedirs(os.path.dirname(os.path.abspath(options.output)), exist_ok=True)
    os.makedirs(options.work_dir, exist_ok=True)


def load_resumed(options, settings: dict, expanded: list[ExpandedCase]):
    """The reusable part of the previous output with --resume, else None."""
    if not options.resume:
        return None
    problems = [e.case.record() for e in expanded]
    resumed = load_resume(
        options.output,
        options.one_shot,
        meta_settings(settings, options),
        problems,
    )
    log(
        f"resuming from {options.output}: reusing {len(resumed.records)} case(s), "
        f"running {resumed.rerun}, dropping {resumed.dropped} record(s) of cases "
        "that are not in the input"
    )
    return resumed


def run(options) -> int:
    sweep_input = load_input(options.input, options.seed)
    cases = select_cases(sweep_input.cases, options.only, options.limit_shapes)
    expanded = [expand_case(case) for case in cases]
    # Listing only counts configurations: the refine settings do not matter.
    if options.list:
        print_listing(expanded)
        return 0
    settings = resolve_settings(sweep_input.settings, options)
    prepare_directories(options)
    # Before amd-smi and HIP are touched, so that a bad --resume fails fast.
    resumed = load_resumed(options, settings, expanded)
    gpus = []
    if not options.dry_run:
        map_hip_ordinals()
        gpus = select_gpus(options.gpus, options.max_busy, options.allow_busy)
    mode = "one-shot" if options.one_shot else "refine+top"
    log(
        f"work dir {options.work_dir}, output {options.output} "
        f"({options.output_format}, {mode}), GPUs {gpus}"
    )
    with GpuStabilizer(gpus, options.require_stable_clocks) as stabilizer:
        sweep = Sweep(options, settings, expanded, resumed)
        try:
            sweep.start(gpus, stabilizer)
        except BaseException:
            sweep.shutdown()
            raise
        status = sweep.run()
    if options.dry_run:
        print_dry_run(sweep)
    log(f"results in {options.output}")
    return status


def main() -> int:
    options = make_parser().parse_args()
    try:
        return run(options)
    except (SweepConfigError, SweepAbort, OSError, yaml.YAMLError) as err:
        log(f"error: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
