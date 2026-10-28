# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GPU side of the skinny GEMM sweep: one long-lived worker process per GPU
that times whole cases (see sweep_skinny_gemm.py for the design).

HIP is first imported inside the worker, after HIP_VISIBLE_DEVICES is set, so
the device the worker sees is always 0. This module does not import HIP at
import time, which keeps it safe to import from the main process.
"""

import os
import signal
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass

import bench_skinny_gemm as bench
import skinny_gemm_support as support
from skinny_gemm_support import DeviceBuffer, LoadedKernel, Problem
from sweep_skinny_gemm_config import first_line
from sweep_skinny_gemm_flush import FastLlcFlush
from sweep_skinny_gemm_inputs import (
    CaseInputs,
    OperandCache,
    Verifier,
    make_case_inputs,
)
from sweep_skinny_gemm_smi import sample_clocks

# Launches of one candidate per round of the interleaved refinement: few enough
# that a slow drift hits every candidate alike, many enough to amortise the
# event creation.
ROUND_LAUNCHES = 5
NS_PER_US = 1000.0
RUN_STATUS = "failed to run"
VERIFY_STATUS = "wrong result"


@dataclass
class Job:
    """A case whose kernels are built: the problems with their kernel names, in
    the order the outcome refers to them."""

    index: int
    entries: list[tuple[str, Problem]]
    hsaco_dir: str
    settings: dict
    verify: bool


@dataclass
class Timing:
    """Launch times in ns; reduce_ns is empty without split-K."""

    gemm_ns: list[int]
    reduce_ns: list[int]

    def extend(self, other: "Timing") -> None:
        self.gemm_ns += other.gemm_ns
        self.reduce_ns += other.reduce_ns

    @property
    def total_ns(self) -> int:
        """The minimum of the per-iteration gemm (flushed) + reduce (warm) sums:
        contention and clock dips only ever add time, so the minimum is the
        least disturbed launch."""
        if not self.reduce_ns:
            return min(self.gemm_ns)
        return min(g + r for g, r in zip(self.gemm_ns, self.reduce_ns))


def hsaco_file(hsaco_dir: str, name: str) -> str:
    return os.path.join(hsaco_dir, name + ".hsaco")


def load_kernel(hsaco_dir: str, name: str) -> LoadedKernel:
    with open(hsaco_file(hsaco_dir, name), "rb") as stream:
        return LoadedKernel(stream.read(), name)


class ReduceStep:
    """The split-K reduce launch of a case, which does not depend on the GEMM
    configuration."""

    def __init__(self, kernel: LoadedKernel, p: Problem, buffers):
        self.kernel = kernel
        self.grid, self.block = support.reduceLaunchDims(p.m, p.n)
        self.args = support.reduceArgs(
            buffers.d,
            buffers.ws,
            buffers.c,
            p.ldD,
            p.m,
            p.n,
            p.splits,
            p.ldC,
            p.alpha,
            p.beta,
        )


class CaseBuffers:
    """The device buffers shared by all configurations of a case: A, B, their
    scales and C are read only, D and the workspace are reused. A context
    manager that frees them."""

    def __init__(self, job: Job, inputs: CaseInputs):
        first = job.entries[0][1]
        splits = max(p.splits for _, p in job.entries)
        self.inputs = inputs
        # Pristine copies: the verification restarts from them.
        self.d_host, self.ws_host = support.makeOutputs(
            first.m, first.n, first.ldD, splits
        )
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
        try:
            self.d, self.a, self.b, self.scl_a, self.scl_b, self.ws, self.c = (
                self.stack.enter_context(DeviceBuffer(array)) for array in arrays
            )
            self.reduce = None
            if splits > 1:
                kernel = load_kernel(job.hsaco_dir, support.REDUCE_KERNEL_NAME)
                self.stack.enter_context(kernel)
                self.reduce = ReduceStep(kernel, first, self)
        except BaseException:
            self.stack.close()
            raise

    def gemm_args(self, p: Problem) -> tuple:
        return support.gemmArgs(
            p, self.d, self.a, self.b, self.scl_a, self.scl_b, self.ws, self.c
        )

    def reset_outputs(self) -> None:
        self.d.upload(self.d_host)
        self.ws.upload(self.ws_host)

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.stack.close()


class Candidate:
    """A loaded kernel with its arguments, timed with a cold LLC."""

    def __init__(self, p: Problem, kernel: LoadedKernel, buffers: CaseBuffers):
        self.problem = p
        self.kernel = kernel
        self.args = buffers.gemm_args(p)
        self.reduce = buffers.reduce

    def time(self, count: int, flush: FastLlcFlush) -> Timing:
        """The GEMM launches, flushed, then the reduce launches, warm, as
        bench_skinny_gemm.benchProblem times them."""
        p = self.problem
        gemm = bench.timeLaunches(
            self.kernel, p.grid, support.gemmBlock(p), self.args, count, flush
        )
        if self.reduce is None:
            return Timing(gemm, [])
        step = self.reduce
        reduce = bench.timeLaunches(
            step.kernel, step.grid, step.block, step.args, count, None
        )
        return Timing(gemm, reduce)


def run_failure(index: int, status: str, stage: str, error: BaseException) -> dict:
    return {
        "index": index,
        "status": status,
        "stage": stage,
        "reason": first_line(error),
    }


def screen(job: Job, buffers: CaseBuffers, flush, outcome: dict) -> dict[int, Timing]:
    """One timing per configuration, one after another so that only one kernel
    runs on the GPU at a time."""
    warmup, iters = job.settings["warmup"], job.settings["iters"]
    screening = {}
    for index, (name, p) in enumerate(job.entries):
        try:
            with load_kernel(job.hsaco_dir, name) as kernel:
                candidate = Candidate(p, kernel, buffers)
                candidate.time(warmup, flush)
                screening[index] = candidate.time(iters, flush)
        except (RuntimeError, OSError) as err:
            outcome["failures"].append(run_failure(index, RUN_STATUS, "run", err))
    return screening


def refine(
    job: Job,
    buffers: CaseBuffers,
    flush,
    screening: dict[int, Timing],
    outcome: dict,
) -> dict[int, Timing]:
    """Re-times the best screened configurations in round-robin rounds, so that
    slow drift (thermals, clocks) hits all of them alike."""
    best = sorted(screening, key=lambda index: screening[index].total_ns)
    best = best[: job.settings["refine_top"]]
    refined = {index: Timing([], []) for index in best}
    with ExitStack() as stack:
        candidates = {}
        for index in best:
            name, p = job.entries[index]
            try:
                kernel = stack.enter_context(load_kernel(job.hsaco_dir, name))
            except (RuntimeError, OSError) as err:
                outcome["failures"].append(run_failure(index, RUN_STATUS, "run", err))
                del refined[index]
                continue
            candidates[index] = Candidate(p, kernel, buffers)
        left = job.settings["refine_iters"]
        while left > 0 and candidates:
            count = min(ROUND_LAUNCHES, left)
            for index in list(candidates):
                try:
                    refined[index].extend(candidates[index].time(count, flush))
                except RuntimeError as err:
                    failure = run_failure(index, RUN_STATUS, "run", err)
                    outcome["failures"].append(failure)
                    del candidates[index], refined[index]
            left -= count
    return refined


def check_result(
    job: Job, buffers: CaseBuffers, p: Problem, name: str, verifier: Verifier
) -> None:
    """Runs the configuration once from fresh outputs and checks D (or the
    workspace and then D for split-K) against the host reference; raises
    AssertionError on a mismatch."""
    buffers.reset_outputs()
    d_out, ws_out = buffers.d_host.copy(), buffers.ws_host.copy()
    with load_kernel(job.hsaco_dir, name) as kernel:
        support.launchKernel(kernel, p.grid, support.gemmBlock(p), buffers.gemm_args(p))
    buffers.d.download(d_out)
    buffers.ws.download(ws_out)
    verifier.check_gemm(p, d_out, ws_out)
    step = buffers.reduce
    if step is None:
        return
    support.launchKernel(step.kernel, step.grid, step.block, step.args)
    buffers.d.download(d_out)
    verifier.check_reduced(p, d_out)


def winner_entry(index: int, refined: Timing, screened: Timing, verified: str) -> dict:
    entry = {
        "index": index,
        "timeUs": round(refined.total_ns / NS_PER_US, 3),
        "gemmUs": round(min(refined.gemm_ns) / NS_PER_US, 3),
        "screeningUs": round(screened.total_ns / NS_PER_US, 3),
        "verified": verified,
    }
    if refined.reduce_ns:
        entry["reduceUs"] = round(min(refined.reduce_ns) / NS_PER_US, 3)
    return entry


def pick_winners(
    job: Job,
    buffers: CaseBuffers,
    screening: dict[int, Timing],
    refined: dict[int, Timing],
    outcome: dict,
) -> None:
    """The top configurations in refined order, every one verified (full or
    sampled, by the size of the case) unless the verification is off. A
    configuration that fails is dropped and the next one is promoted, among the
    refined ones."""
    first = job.entries[0][1]
    ranking = sorted(refined, key=lambda index: refined[index].total_ns)
    verifier = None
    if job.verify and ranking:
        elements = (first.m + first.n) * first.k
        full = elements <= job.settings["verify_full_max_elems"]
        verifier = Verifier(buffers.inputs, full)
    for index in ranking:
        if len(outcome["ranked"]) >= job.settings["top"]:
            break
        name, p = job.entries[index]
        verified = "no"
        if verifier is not None:
            try:
                check_result(job, buffers, p, name, verifier)
            except AssertionError as err:
                outcome["failures"].append(
                    run_failure(index, VERIFY_STATUS, "verify", err)
                )
                continue
            except (RuntimeError, OSError) as err:
                failure = run_failure(index, RUN_STATUS, "verify", err)
                outcome["failures"].append(failure)
                continue
            verified = verifier.label(p)
        outcome["ranked"].append(
            winner_entry(index, refined[index], screening[index], verified)
        )


def time_case(job: Job, gpu: int, flush, cache: OperandCache, outcome: dict) -> None:
    phases = outcome["phases"]
    start = time.perf_counter()
    inputs = make_case_inputs(job.entries[0][1], cache)
    with CaseBuffers(job, inputs) as buffers:
        phases["inputsAndUploadS"] = time.perf_counter() - start
        outcome["clocks"] = {"start": sample_clocks(gpu)}
        start = time.perf_counter()
        screening = screen(job, buffers, flush, outcome)
        outcome["timed"] = len(screening)
        phases["screeningS"] = time.perf_counter() - start
        start = time.perf_counter()
        refined = refine(job, buffers, flush, screening, outcome)
        outcome["clocks"]["end"] = sample_clocks(gpu)
        phases["refinementS"] = time.perf_counter() - start
        start = time.perf_counter()
        pick_winners(job, buffers, screening, refined, outcome)
        phases["verificationS"] = time.perf_counter() - start


def run_job(job: Job, gpu: int, flush, cache: OperandCache) -> dict:
    """The outcome of a case: the ranked winners (indices into the job's
    entries), the per-configuration failures and the clocks around the timing.
    A failure of the case itself is reported, not raised, so that the worker
    survives it."""
    started = time.time()
    outcome = {
        "index": job.index,
        "gpu": gpu,
        "ranked": [],
        "failures": [],
        "timed": 0,
        "clocks": {},
        "phases": {},
    }
    try:
        time_case(job, gpu, flush, cache, outcome)
    except Exception as err:
        outcome["error"] = f"{type(err).__name__}: {first_line(err)}"
    outcome["phases"] = {k: round(v, 2) for k, v in outcome["phases"].items()}
    outcome["seconds"] = round(time.time() - started, 2)
    return outcome


def device_name() -> str:
    props = support.getDeviceProperties()
    name = props.name
    return name.decode() if isinstance(name, bytes) else str(name)


def gpu_worker(gpu: int, jobs, results) -> None:
    """The process of one GPU: takes one case at a time from the job queue,
    until the None sentinel, and puts the messages ("ready", gpu, name),
    ("case", gpu, outcome) and ("fatal", gpu, text) on the result queue."""
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    # The main process handles Ctrl-C and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cache = OperandCache()
        with FastLlcFlush() as flush:
            results.put(("ready", gpu, device_name()))
            while True:
                job = jobs.get()
                if job is None:
                    return
                results.put(("case", gpu, run_job(job, gpu, flush, cache)))
    except Exception:
        results.put(("fatal", gpu, traceback.format_exc()))
        raise
