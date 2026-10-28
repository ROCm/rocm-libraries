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
      [--gpus auto|1,3,6,7] [-o results.yaml] [--only SUBSTRING] [--list|--dry-run]

Stages, all driven by the main process, which never initializes HIP (so a
spawned child can safely own a GPU):

  1. Expand the input into cases (shape, split size, beta mode) and their valid
     configurations; configurations that violate the support constraints are
     recorded as failed to build (stage constraint).
  2. BUILD, CPU-parallel: a process pool (spawn) emits and assembles every
     unique kernel once, keyed by its kernel name, which encodes every
     compile-time knob including the derived UP_ITERS and PARTIAL_STEPS and
     USE_BETA, so kernels are shared across shapes and split sizes. The hsaco
     files are written to <work-dir>/hsaco/<name>.hsaco and an existing file is
     reused as a cache (use a fresh work dir after changing the emitter). A
     failure at emit or assemble is recorded and never stops the sweep.
  3. TIMED, one long-lived worker process per GPU, started with
     HIP_VISIBLE_DEVICES set so that it sees its device as 0. A job is a whole
     case: once all its kernels are built the main process queues it and a
     worker times ALL its configurations strictly one after another. Kernels
     that share a GPU contend for CUs, HBM and the LLC and would corrupt each
     other's timings, whereas separate GPUs do not; handing out whole cases
     instead of single configurations keeps all candidates of a case on the
     same physical device, so silicon and clock differences between devices
     cannot reorder them. With one GPU the cases simply run serially while the
     CPU builds of the later ones overlap the timing.
     Per configuration: warmup untimed launches, then iters timed launches
     (256 MiB LLC flush before each, HIP events, as bench_skinny_gemm, with an
     8x faster flush kernel); the score is the minimum, because contention and
     clock dips only add time. The operands are generated once per worker and
     shape in parallel blocks and cached (sweep_skinny_gemm_inputs.py). For
     split-K the score is the minimum over iterations of the flushed GEMM time
     plus the warm reduce time. The refine_top best are then re-timed in
     interleaved rounds so that slow drift hits them alike, and the winners are
     ranked by that refined score and verified in rank order against the host
     reference; a failing configuration is dropped and the next refined one
     promoted. Every reported winner is verified (never skipped), but only the
     winners: the screened candidates are not. The reference is dense ("full")
     when (m + n) * k <= verify_full_max_elems, else sampled (see
     sweep_skinny_gemm_inputs.py); --no-verify turns the verification off and
     marks the winners "no".
  4. The results stream back; the output is rewritten atomically after every
     finished case, and the failed configurations are appended to
     <output>.failed.yaml.

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

The clocks of the GPUs are locked for the timed stage when rocm-smi allows it
(see GpuStabilizer); otherwise the sweep goes on without, unless
--require-stable-clocks. The sclk and mclk are sampled around every case.
"""

import argparse
import concurrent.futures
import datetime
import multiprocessing
import os
import queue
import signal
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field

import yaml

import skinny_gemm_support as support
from skinny_gemm_support import Problem
from sweep_skinny_gemm_config import (
    ConfigFailure,
    ExpandedCase,
    SweepConfigError,
    config_record,
    expand_case,
    first_line,
    load_input,
)
from sweep_skinny_gemm_smi import (
    GpuStabilizer,
    SweepAbort,
    git_head,
    log,
    script_dir,
    select_gpus,
)
from sweep_skinny_gemm_worker import Job, gpu_worker, hsaco_file

TOOL = "sweep_skinny_gemm.py"
DEFAULT_INPUT = os.path.join(script_dir(), "sweep_skinny_gemm_shapes.yaml")
DEFAULT_BUILD_JOBS = 64
# How long the main loop waits for an event before it looks at the other queue.
POLL_S = 0.1
BUILD_STATUS = "failed to build"
HEADER = """\
# Skinny GEMM sweep results written by sweep_skinny_gemm.py.
# Times are microseconds: timeUs is the refined minimum of gemm (+ reduce for
# split-K), screeningUs the minimum of the screening pass. verified is full (the
# whole result was compared with a dense host reference), sampled(N) (N output
# elements were compared, each over all its splits) or no (--no-verify); a
# configuration that fails verification is dropped, not listed. phaseSeconds
# splits the wall time of a case on its GPU.
"""


@dataclass
class BuildResult:
    name: str
    ok: bool
    stage: str
    message: str


def ignore_sigint() -> None:
    """The main process handles Ctrl-C and shuts the pool down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_kernel(name: str, problem: Problem | None, hsaco_dir: str) -> BuildResult:
    """Emits and assembles one kernel into its hsaco file; the reduce kernel
    when problem is None. Runs in a pool process."""
    path = hsaco_file(hsaco_dir, name)
    if os.path.exists(path):
        return BuildResult(name, True, "", "cached")
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


class BuildStage:
    """The CPU-parallel build of the unique kernels."""

    def __init__(self, hsaco_dir: str, jobs: int):
        self.hsaco_dir = hsaco_dir
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=ignore_sigint,
        )
        self.pending: dict[concurrent.futures.Future, str] = {}
        self.results: dict[str, BuildResult] = {}
        self.submitted: set[str] = set()

    def submit(self, name: str, problem: Problem | None) -> None:
        if name in self.submitted:
            return
        self.submitted.add(name)
        future = self.executor.submit(build_kernel, name, problem, self.hsaco_dir)
        self.pending[future] = name

    def poll(self, timeout: float) -> list[BuildResult]:
        """The builds finished within timeout."""
        if not self.pending:
            return []
        done, _ = concurrent.futures.wait(
            self.pending, timeout, concurrent.futures.FIRST_COMPLETED
        )
        finished = []
        for future in done:
            name = self.pending.pop(future)
            try:
                result = future.result()
            except Exception as err:
                message = f"build process failed: {first_line(err)}"
                result = BuildResult(name, False, "emit", message)
            self.results[name] = result
            finished.append(result)
        return finished

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class GpuPool:
    """One worker process per GPU sharing a job queue."""

    def __init__(self, gpus: list[int]):
        context = multiprocessing.get_context("spawn")
        self.jobs = context.Queue()
        self.results = context.Queue()
        self.workers = [
            context.Process(
                target=gpu_worker, args=(gpu, self.jobs, self.results), daemon=True
            )
            for gpu in gpus
        ]
        for worker in self.workers:
            worker.start()

    def submit(self, job: Job) -> None:
        self.jobs.put(job)

    def poll(self, timeout: float) -> tuple | None:
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            return None

    def dead_workers(self) -> list[str]:
        return [w.name for w in self.workers if not w.is_alive()]

    def stop(self) -> None:
        for _ in self.workers:
            self.jobs.put(None)
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()


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


def microseconds(value: float) -> float:
    return round(value, 3)


def dump(data, stream) -> None:
    """Block YAML whose leaf mappings are on one line, to keep it scannable."""
    yaml.safe_dump(data, stream, sort_keys=False, width=1000, default_flow_style=None)


class ResultWriter:
    """The output YAML, rewritten atomically after every case, and the failed
    configurations, appended to their own file."""

    def __init__(self, path: str, meta: dict):
        self.path = path
        self.failed_path = os.path.splitext(path)[0] + ".failed.yaml"
        self.meta = meta
        self.records: dict[int, dict] = {}
        with open(self.failed_path, "w"):
            pass

    def add(self, index: int, record: dict, failures: list[dict]) -> None:
        self.records[index] = record
        if failures:
            with open(self.failed_path, "a") as stream:
                dump(failures, stream)
        self.write()

    def write(self) -> None:
        results = [self.records[index] for index in sorted(self.records)]
        document = {"meta": self.meta, "results": results}
        temporary = f"{self.path}.{os.getpid()}.tmp"
        with open(temporary, "w") as stream:
            stream.write(HEADER)
            dump(document, stream)
        os.replace(temporary, self.path)


def failure_entry(state: CaseState, failure: ConfigFailure, status: str) -> dict:
    return {
        "problem": state.expanded.case.record(),
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

    def __init__(self, options, settings: dict, expanded: list[ExpandedCase]):
        self.options = options
        self.settings = settings
        self.states = [make_state(i, e) for i, e in enumerate(expanded)]
        self.hsaco_dir = os.path.join(options.work_dir, "hsaco")
        self.builds = None
        self.pool = None
        self.remaining = len(self.states)
        self.waiters: dict[str, list[CaseState]] = {}
        self.meta = {}
        self.writer = None
        self.gpu_meta: dict[int, dict] = {}

    def start(self, gpus: list[int], stabilizer: GpuStabilizer) -> None:
        os.makedirs(self.hsaco_dir, exist_ok=True)
        self.meta = self.make_meta(gpus, stabilizer)
        self.writer = ResultWriter(self.options.output, self.meta)
        self.builds = BuildStage(self.hsaco_dir, self.options.build_jobs)
        self.pool = GpuPool(gpus) if gpus else None
        self.submit_builds()

    def make_meta(self, gpus: list[int], stabilizer: GpuStabilizer) -> dict:
        locks = stabilizer.summary()
        per_gpu = locks.get("perGpu", {})
        self.gpu_meta = {
            gpu: {"id": gpu, "name": None, "clockLock": per_gpu.get(gpu), "clocks": []}
            for gpu in gpus
        }
        return {
            "tool": TOOL,
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "gitHead": git_head(script_dir()),
            "arch": support.ARCH,
            "gpuName": None,
            "inputFile": os.path.abspath(self.options.input),
            "settings": {**self.settings, "verify": not self.options.no_verify},
            "verification": self.verification_note(),
            "gpus": list(self.gpu_meta.values()),
            "clockLock": {key: v for key, v in locks.items() if key != "perGpu"},
            "buildJobs": self.options.build_jobs,
            "workDir": os.path.abspath(self.options.work_dir),
            "kernelsBuilt": 0,
            "kernelsFailed": 0,
            "casesDone": 0,
            "casesTotal": len(self.states),
            "elapsedSeconds": 0,
        }

    def verification_note(self) -> str:
        if self.options.no_verify:
            return "disabled by --no-verify: winners are marked no"
        limit = self.settings["verify_full_max_elems"]
        return (
            f"every winner is verified: full when (m + n) * k <= {limit}, "
            "else sampled; the screened candidates are not verified"
        )

    def submit_builds(self) -> None:
        """In case order, so that the early cases finish first."""
        for state in self.states:
            for name, problem in state.kernels.items():
                self.builds.submit(name, problem)
                self.waiters.setdefault(name, []).append(state)
                state.waiting += 1
        for state in self.states:
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
        self.pool.submit(
            Job(
                state.index,
                state.entries,
                self.hsaco_dir,
                self.settings,
                not self.options.no_verify,
            )
        )

    def finish_case(self, state: CaseState, outcome: dict | None) -> None:
        record = case_record(state, outcome)
        self.writer.add(state.index, record, all_failure_entries(state, outcome))
        self.remaining -= 1
        self.meta["casesDone"] += 1
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

    def on_message(self, message: tuple) -> None:
        kind, gpu, payload = message
        if kind == "ready":
            self.gpu_meta[gpu]["name"] = payload
            self.meta["gpuName"] = payload
            return
        if kind == "fatal":
            raise SweepAbort(f"the worker of GPU {gpu} failed:\n{payload}")
        state = self.states[payload["index"]]
        samples = {"case": state.expanded.case.label, **payload["clocks"]}
        self.gpu_meta[gpu]["clocks"].append(samples)
        self.finish_case(state, payload)

    def loop(self) -> None:
        start = time.time()
        while self.remaining:
            for result in self.builds.poll(POLL_S):
                self.on_build(result)
            if self.pool is not None:
                message = self.pool.poll(POLL_S)
                if message is not None:
                    self.on_message(message)
                elif self.pool.dead_workers():
                    raise SweepAbort(f"GPU worker(s) died: {self.pool.dead_workers()}")
            self.meta["elapsedSeconds"] = round(time.time() - start, 1)

    def run(self) -> int:
        status = 0
        try:
            self.loop()
        except KeyboardInterrupt:
            log("interrupted: the results so far are written")
            self.meta["interrupted"] = True
            status = 130
        except SweepAbort as err:
            log(f"error: {err}")
            self.meta["aborted"] = str(err)
            status = 1
        finally:
            self.shutdown()
        return status

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.stop()
        self.builds.shutdown()
        self.writer.write()


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


def resolve_settings(settings: dict, options) -> dict:
    resolved = dict(settings)
    for key in settings:
        value = getattr(options, key)
        if value is not None:
            resolved[key] = value
    if resolved["refine_top"] < resolved["top"]:
        raise SweepConfigError("settings: refine_top must be at least top")
    return resolved


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the knobs of the skinny MXFP8 GEMM over a list of shapes"
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="input YAML")
    parser.add_argument("-o", "--output", default=None, help="output YAML")
    parser.add_argument("--work-dir", default=None, help="hsaco cache directory")
    parser.add_argument("--gpus", default="auto", help="auto or a list such as 1,3")
    parser.add_argument("--max-busy", type=float, default=5.0, help="idle GPU use, %%")
    parser.add_argument(
        "--allow-busy", action="store_true", help="time on listed busy GPUs"
    )
    parser.add_argument(
        "--build-jobs", type=int, default=min(os.cpu_count() or 1, DEFAULT_BUILD_JOBS)
    )
    parser.add_argument("--only", default=None, help="problems whose name has this")
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="count, do not build")
    parser.add_argument("--dry-run", action="store_true", help="build, no GPU")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--require-stable-clocks", action="store_true")
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
        options.output = os.path.join(options.work_dir, "results.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(options.output)), exist_ok=True)
    os.makedirs(options.work_dir, exist_ok=True)


def run(options) -> int:
    sweep_input = load_input(options.input, options.seed)
    settings = resolve_settings(sweep_input.settings, options)
    cases = select_cases(sweep_input.cases, options.only, options.limit_shapes)
    expanded = [expand_case(case) for case in cases]
    if options.list:
        print_listing(expanded)
        return 0
    gpus = (
        []
        if options.dry_run
        else select_gpus(options.gpus, options.max_busy, options.allow_busy)
    )
    prepare_directories(options)
    log(f"work dir {options.work_dir}, output {options.output}, GPUs {gpus}")
    with GpuStabilizer(gpus, options.require_stable_clocks) as stabilizer:
        sweep = Sweep(options, settings, expanded)
        sweep.start(gpus, stabilizer)
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
