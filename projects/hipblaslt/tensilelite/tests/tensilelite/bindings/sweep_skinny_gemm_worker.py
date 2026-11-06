# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GPU side of the skinny GEMM sweep: one long-lived worker process per GPU
that times whole cases (see sweep_skinny_gemm.py for the design).

HIP is first imported inside the worker, after HIP_VISIBLE_DEVICES is set, so
the device the worker sees is always 0. This module does not import HIP at
import time, which keeps it safe to import from the main process.

The worker talks to the main process over one duplex connection. It receives
Job objects (None stops it) and sends ("ready", gpu, {"name": device name, "timing": method,
"pci": PCI address of the device or None}),
("progress", gpu, case index) heartbeats while it works, ("case", gpu, outcome)
and ("fatal", gpu, traceback text). A kernel that leaves the device in a
sticky error fault ends the case with outcome["fault"] and the worker exits
right after sending it, so that its replacement gets a clean HIP context.
"""

import itertools
import os
import signal
import time
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing.connection import Connection

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
from sweep_skinny_gemm_smi import (
    ClockSampler,
    format_pci_address,
    sample_metrics,
    use_hip_ordinals,
)

# Launches of one candidate per round of the interleaved refinement: few enough
# that a slow drift hits every candidate alike, many enough to amortise the
# event creation.
ROUND_LAUNCHES = 5
NS_PER_US = 1000.0
# Wall time of flushes before the screening of a case, which brings the clocks
# up after the uploads.
RAMP_MS = 50
# The variables that renumber or hide devices.
VISIBILITY_VARIABLES = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
RUN_STATUS = "failed to run"
VERIFY_STATUS = "wrong result"
# The longest a worker stays silent while it works.
PROGRESS_INTERVAL_S = 5.0


class DeviceFault(Exception):
    """The device is in a sticky error state after a failed configuration."""


def device_healthy() -> bool:
    """True if the device answers a synchronize without error; never raises."""
    try:
        hip = support._hip()
        result = hip.hipDeviceSynchronize()
        error = result[0] if isinstance(result, tuple) else result
        return error == hip.hipError_t.hipSuccess
    except Exception:
        return False


def no_heartbeat() -> None:
    """The heartbeat of a caller that does not report progress."""


class Heartbeat:
    """Tells the main process, at most every PROGRESS_INTERVAL_S, that the worker
    is alive and which case it works on."""

    def __init__(self, conn: Connection, gpu: int, index: int):
        self.conn = conn
        self.message = ("progress", gpu, index)
        self.last = time.monotonic()

    def __call__(self) -> None:
        now = time.monotonic()
        if now - self.last < PROGRESS_INTERVAL_S:
            return
        self.last = now
        try:
            self.conn.send(self.message)
        except (OSError, ValueError):
            # The main process is gone; the job loop notices at its next recv.
            pass


@dataclass
class Job:
    """A case whose kernels are built: the problems with their kernel names, in
    the order the outcome refers to them."""

    index: int
    entries: list[tuple[str, Problem]]
    hsaco_dir: str
    settings: dict
    verify: bool
    one_shot: bool = False


@dataclass
class Timing:
    """Launch times in ns, one per timed iteration; reduce_ns is empty without
    split-K. The untimed post-load and warmup launches are not in it."""

    gemm_ns: list[int]
    reduce_ns: list[int]

    def extend(self, other: "Timing") -> None:
        self.gemm_ns += other.gemm_ns
        self.reduce_ns += other.reduce_ns

    @property
    def totals_ns(self) -> list[int]:
        """The per-iteration gemm (flushed) + reduce (warm) sums."""
        if not self.reduce_ns:
            return self.gemm_ns
        return [g + r for g, r in zip(self.gemm_ns, self.reduce_ns)]

    @property
    def total_ns(self) -> float:
        """The mean of the per-iteration totals, the ranking metric."""
        return bench.mean(self.totals_ns)

    @property
    def min_ns(self) -> int:
        """Informational: the fastest iteration."""
        return min(self.totals_ns)

    @property
    def p90_ns(self) -> int:
        """Informational: the 90th percentile of the iterations."""
        return bench.percentile(self.totals_ns, 0.9)

    @property
    def gemm_mean_ns(self) -> float:
        return bench.mean(self.gemm_ns)

    @property
    def reduce_mean_ns(self) -> float:
        return bench.mean(self.reduce_ns)


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

    def close(self) -> None:
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback) -> None:
        support.closeOnExit(self, value)


class Candidate:
    """A loaded kernel with its arguments, timed with a cold LLC."""

    def __init__(self, p: Problem, kernel: LoadedKernel, buffers: CaseBuffers):
        self.problem = p
        self.kernel = kernel
        self.args = buffers.gemm_args(p)
        self.reduce = buffers.reduce

    def prime(self, flush: FastLlcFlush) -> None:
        """The untimed launches after the load, whatever the warmup setting."""
        self.warm(bench.POST_LOAD_LAUNCHES, flush)

    def warm(self, count: int, flush: FastLlcFlush) -> None:
        """Untimed launches in the pattern of time: the GEMM after its flush,
        then the reduce, warm."""
        p = self.problem
        bench.warmLaunches(
            self.kernel, p.grid, support.gemmBlock(p), self.args, count, flush
        )
        if self.reduce is None:
            return
        step = self.reduce
        bench.warmLaunches(step.kernel, step.grid, step.block, step.args, count, None)

    def time(self, count: int, flush: FastLlcFlush) -> Timing:
        """The GEMM launches, each after its own flush, then the reduce launches,
        warm and unflushed, as bench_skinny_gemm.benchProblem times them."""
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


def fail_configuration(outcome: dict, failure: dict, err: BaseException) -> None:
    """Records the failure of one configuration; raises DeviceFault if it left
    the device unusable, since the other configurations would fail alike."""
    outcome["failures"].append(failure)
    if not device_healthy():
        raise DeviceFault(failure["reason"]) from err


def screen(
    job: Job,
    buffers: CaseBuffers,
    flush,
    outcome: dict,
    heartbeat: Callable[[], None] = no_heartbeat,
) -> dict[int, Timing]:
    """One timing per configuration, one after another so that only one kernel
    runs on the GPU at a time."""
    warmup, iters = job.settings["warmup"], job.settings["iters"]
    screening = {}
    for index, (name, p) in enumerate(job.entries):
        try:
            with load_kernel(job.hsaco_dir, name) as kernel:
                candidate = Candidate(p, kernel, buffers)
                candidate.prime(flush)
                candidate.warm(warmup, flush)
                screening[index] = candidate.time(iters, flush)
        except Exception as err:
            fail_configuration(outcome, run_failure(index, RUN_STATUS, "run", err), err)
        heartbeat()
    return screening


def refine(
    job: Job,
    buffers: CaseBuffers,
    flush,
    screening: dict[int, Timing],
    outcome: dict,
    heartbeat: Callable[[], None] = no_heartbeat,
) -> dict[int, Timing]:
    """Re-times the best screened configurations in round-robin rounds, so that
    slow drift (thermals, clocks) hits all of them alike. Round r starts at
    candidate r modulo their number, so that none is always the one that runs
    right after the previous round's last candidate."""
    best = sorted(screening, key=lambda index: screening[index].total_ns)
    best = best[: job.settings["refine_top"]]
    refined = {index: Timing([], []) for index in best}
    with ExitStack() as stack:
        candidates = {}
        for index in best:
            name, p = job.entries[index]
            try:
                kernel = stack.enter_context(load_kernel(job.hsaco_dir, name))
            except Exception as err:
                del refined[index]
                failure = run_failure(index, RUN_STATUS, "run", err)
                fail_configuration(outcome, failure, err)
                heartbeat()
                continue
            candidate = Candidate(p, kernel, buffers)
            try:
                candidate.prime(flush)
            except Exception as err:
                del refined[index]
                failure = run_failure(index, RUN_STATUS, "run", err)
                fail_configuration(outcome, failure, err)
                heartbeat()
                continue
            candidates[index] = candidate
        left = job.settings["refine_iters"]
        for round_index in itertools.count():
            if left <= 0 or not candidates:
                break
            count = min(ROUND_LAUNCHES, left)
            order = list(candidates)
            first = round_index % len(order)
            for index in order[first:] + order[:first]:
                try:
                    refined[index].extend(candidates[index].time(count, flush))
                except Exception as err:
                    del candidates[index], refined[index]
                    failure = run_failure(index, RUN_STATUS, "run", err)
                    fail_configuration(outcome, failure, err)
                heartbeat()
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


def launch_lists(timing: Timing, gemm_key: str, reduce_key: str) -> dict:
    """The per-launch times in ns of the timing, for offline analysis."""
    lists = {gemm_key: list(timing.gemm_ns)}
    if timing.reduce_ns:
        lists[reduce_key] = list(timing.reduce_ns)
    return lists


def screening_only_entry(index: int, timing: Timing, verified: str) -> dict:
    """A ranked configuration from the screening pass only (--one-shot)."""
    entry = {
        "index": index,
        "timeUs": round(timing.total_ns / NS_PER_US, 3),
        "gemmUs": round(timing.gemm_mean_ns / NS_PER_US, 3),
        "screeningUs": round(timing.total_ns / NS_PER_US, 3),
        "verified": verified,
    }
    if timing.reduce_ns:
        entry["reduceUs"] = round(timing.reduce_mean_ns / NS_PER_US, 3)
    return entry | launch_lists(timing, "gemmNs", "reduceNs")


def verify_screened(
    job: Job,
    buffers: CaseBuffers,
    screening: dict[int, Timing],
    outcome: dict,
    heartbeat: Callable[[], None] = no_heartbeat,
) -> dict[int, str]:
    """One host check per screened configuration (--one-shot)."""
    if not job.verify:
        return {index: "no" for index in screening}
    first = job.entries[0][1]
    elements = (first.m + first.n) * first.k
    full = elements <= job.settings["verify_full_max_elems"]
    verifier = Verifier(buffers.inputs, full)
    labels: dict[int, str] = {}
    for index in screening:
        name, p = job.entries[index]
        try:
            check_result(job, buffers, p, name, verifier)
        except AssertionError as err:
            failure = run_failure(index, VERIFY_STATUS, "verify", err)
            outcome["failures"].append(failure)
            labels[index] = "failed"
            continue
        except Exception as err:
            labels[index] = "failed"
            failure = run_failure(index, RUN_STATUS, "verify", err)
            fail_configuration(outcome, failure, err)
            continue
        finally:
            heartbeat()
        labels[index] = verifier.label(p)
    return labels


def rank_all_screened(
    screening: dict[int, Timing], outcome: dict, verified: dict[int, str]
) -> None:
    """Fills outcome['ranked'] with every screened configuration, fastest first."""
    for index in sorted(screening, key=lambda i: screening[i].total_ns):
        outcome["ranked"].append(
            screening_only_entry(index, screening[index], verified[index])
        )


def winner_entry(index: int, refined: Timing, screened: Timing, verified: str) -> dict:
    """A refined configuration: timeUs is the mean of the refined launches;
    minUs and p90Us are informational."""
    entry = {
        "index": index,
        "timeUs": round(refined.total_ns / NS_PER_US, 3),
        "gemmUs": round(refined.gemm_mean_ns / NS_PER_US, 3),
        "screeningUs": round(screened.total_ns / NS_PER_US, 3),
        "minUs": round(refined.min_ns / NS_PER_US, 3),
        "p90Us": round(refined.p90_ns / NS_PER_US, 3),
        "verified": verified,
    }
    if refined.reduce_ns:
        entry["reduceUs"] = round(refined.reduce_mean_ns / NS_PER_US, 3)
    entry |= launch_lists(refined, "gemmNs", "reduceNs")
    return entry | launch_lists(screened, "screeningGemmNs", "screeningReduceNs")


def pick_winners(
    job: Job,
    buffers: CaseBuffers,
    screening: dict[int, Timing],
    refined: dict[int, Timing],
    outcome: dict,
    heartbeat: Callable[[], None] = no_heartbeat,
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
                failure = run_failure(index, VERIFY_STATUS, "verify", err)
                outcome["failures"].append(failure)
                continue
            except Exception as err:
                failure = run_failure(index, RUN_STATUS, "verify", err)
                fail_configuration(outcome, failure, err)
                continue
            finally:
                heartbeat()
            verified = verifier.label(p)
        outcome["ranked"].append(
            winner_entry(index, refined[index], screening[index], verified)
        )


def time_stage(
    job: Job,
    gpu: int,
    buffers: CaseBuffers,
    flush,
    outcome: dict,
    heartbeat: Callable[[], None],
) -> tuple[dict, dict | None]:
    """The screening and, unless one-shot, the refinement of a case, sampling
    the clocks and the throttle status meanwhile; returns the screening and the
    refinement (None in one-shot mode)."""
    phases = outcome["phases"]
    outcome["clocks"] = {"start": sample_metrics(gpu)}
    sampler = ClockSampler(gpu)
    refined = None
    with sampler:
        start = time.perf_counter()
        flush.ramp(RAMP_MS)
        screening = screen(job, buffers, flush, outcome, heartbeat)
        outcome["timed"] = len(screening)
        phases["screeningS"] = time.perf_counter() - start
        start = time.perf_counter()
        if not job.one_shot:
            refined = refine(job, buffers, flush, screening, outcome, heartbeat)
        phases["refinementS"] = 0.0 if job.one_shot else time.perf_counter() - start
    outcome["clocks"]["during"] = sampler.summary()
    outcome["clocks"]["end"] = sample_metrics(gpu)
    return screening, refined


def time_case(
    job: Job,
    gpu: int,
    flush,
    cache: OperandCache,
    outcome: dict,
    heartbeat: Callable[[], None] = no_heartbeat,
) -> None:
    phases = outcome["phases"]
    start = time.perf_counter()
    inputs = make_case_inputs(job.entries[0][1], cache)
    with CaseBuffers(job, inputs) as buffers:
        phases["inputsAndUploadS"] = time.perf_counter() - start
        screening, refined = time_stage(job, gpu, buffers, flush, outcome, heartbeat)
        start = time.perf_counter()
        if refined is None:
            verified = verify_screened(job, buffers, screening, outcome, heartbeat)
            phases["verificationS"] = time.perf_counter() - start
            rank_all_screened(screening, outcome, verified)
            return
        pick_winners(job, buffers, screening, refined, outcome, heartbeat)
        phases["verificationS"] = time.perf_counter() - start


def run_job(
    job: Job,
    gpu: int,
    flush,
    cache: OperandCache,
    heartbeat: Callable[[], None] = no_heartbeat,
) -> dict:
    """The outcome of a case: the ranked winners (indices into the job's
    entries), the per-configuration failures and the clocks around the timing.
    A failure of the case itself is reported, not raised, so that the worker
    survives it; outcome['fault'] tells that the device is unusable and the
    worker must be replaced."""
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
        time_case(job, gpu, flush, cache, outcome, heartbeat)
    except DeviceFault as err:
        outcome["error"] = f"device fault: {err}"
        outcome["fault"] = True
    except Exception as err:
        outcome["error"] = f"{type(err).__name__}: {first_line(err)}"
        outcome["fault"] = not device_healthy()
    outcome["phases"] = {k: round(v, 2) for k, v in outcome["phases"].items()}
    outcome["seconds"] = round(time.time() - started, 2)
    return outcome


def device_name() -> str:
    props = support.getDeviceProperties()
    name = props.name
    return name.decode() if isinstance(name, bytes) else str(name)


def pci_address_of(device: int) -> str:
    props = support.getDeviceProperties(device)
    return format_pci_address(props.pciDomainID, props.pciBusID, props.pciDeviceID)


def device_pci_address() -> str | None:
    """The PCI address of the HIP device 0 of this process in the format of
    amd-smi, None if it cannot be read."""
    try:
        return pci_address_of(0)
    except Exception:
        return None


def hip_pci_addresses() -> dict[int, str]:
    """The PCI address of every HIP ordinal. The visibility masks are dropped
    like in gpu_worker, so that the ordinals are the ones the workers index
    with HIP_VISIBLE_DEVICES. Runs in a short-lived child of the main process,
    which never initializes HIP."""
    for name in VISIBILITY_VARIABLES:
        os.environ.pop(name, None)
    return {i: pci_address_of(i) for i in range(support.getDeviceCount())}


def serve_jobs(gpu: int, conn: Connection, flush, cache: OperandCache) -> None:
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        outcome = run_job(job, gpu, flush, cache, Heartbeat(conn, gpu, job.index))
        conn.send(("case", gpu, outcome))
        if outcome.get("fault"):
            return


def gpu_worker(gpu: int, conn: Connection) -> None:
    """The process of one GPU: runs the cases it receives on conn, until None,
    and reports with the messages described in the module docstring."""
    # A parent-level mask would renumber the devices under HIP_VISIBLE_DEVICES.
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    # The main process handles Ctrl-C and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cache = OperandCache()
        with FastLlcFlush() as flush:
            pci = device_pci_address()
            if pci is not None:
                # The amd-smi calls of this worker name the device by its id.
                use_hip_ordinals({gpu: pci})
            ready = {
                "name": device_name(),
                "timing": bench.timingMethod(),
                "pci": pci,
            }
            conn.send(("ready", gpu, ready))
            serve_jobs(gpu, conn, flush, cache)
    except Exception:
        try:
            conn.send(("fatal", gpu, traceback.format_exc()))
        except (OSError, ValueError):
            pass
        raise
