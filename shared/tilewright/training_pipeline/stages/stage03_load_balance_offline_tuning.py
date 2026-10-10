# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03_load_balance_offline_tuning -- bench every shape with
hipblaslt-bench on all GPUs: a dynamic work queue, per-shape crash recovery,
and a retry pass.

Reads `<in-dir>/shapes.yaml` (stage01 or stage02) and cuts it into blocks.
One worker thread per GPU pulls the next block whenever it finishes one, so
faster GPUs take more blocks.

Every shape is timed adaptively (lib.bench_yaml): a line with fixed
iteration counts, from a bench yaml written before adaptive timing became
the only mode, gets `adaptive: true` and the config's `bench.adaptive`
fields in their place.

Crash recovery within a block: when a bench process exits non-zero or is
killed by the watchdog, the problems it got past are counted from its output
(`lib.bench_log.count_finished_problems`); the next shape is taken to be the
one that crashed it and is recorded as skipped, and a new process resumes
after it. A block whose processes keep dying without finishing a shape is
abandoned, and its remaining shapes are recorded as unattempted.

Retry pass: skipped and unattempted shapes, up to --max-retry-gemms of them,
are benched once more in one process on the first device that still takes
blocks (crashes under heavy multi-GPU load often do not repeat in
isolation). --no-retry-skipped turns it off.

Entry names: the bench converts its --yaml with clients/hipblaslt_gentest.py,
which runs only the first of entries equal in every argument, while the
stage maps a process's output to the entries of its yaml by position (the
warm-up's output, the crashed shape). So every entry gets a name of its own
(lib/bench_yaml.py `name_entry`: `warmup`, `canary_start<i>` and
`canary_end<i>`, and `line<N>` for line N of the input yaml). A process
whose output does not line up with its entries (a problem of another shape
than its entry, more problems than entries, or fewer from a process that
exited cleanly) is noted in its block log and fails the stage; one with the
canary is benched again instead (see Canary).

GPUs: the stage leases the GPUs it benches on (lib/gpu_lease.py; under the
orchestrator it works under the orchestrator's lease) and refuses to start,
naming the holder, when another process tree holds one; it records the
GPUs' settings in platform.json and refuses to start when they disagree on
the performance level or the power cap (lib/fingerprint.py).

Every bench sees its GPU alone: it runs with ROCR_VISIBLE_DEVICES=<the
device's GPU> and `--device 0`, in log names still as gpuD. It also gets
the config's runtime.env (lib/bench_env.py), which platform.json records,
and no launch override the config's runtime.env does not set; platform.json
records the launch overrides it has and those the stage's environment had,
the launch heuristic of the bench build and the workspace the lines pass
(lib/launch_settings.py, under `launch`).

Quarantine: while a block's bench runs, the processes with memory on its GPU
and the card's VRAM in use are sampled with the stall watchdog's polls and
checked once more when the process exits (lib/kfd.py `GpuWatch`). A process
that is not the stage's own (one of its benches or a process they started),
or VRAM no listed process holds (unattributed) beyond its settle window,
which grows with the memory a bench held, stops the block: its logs move to
`quarantine/` with a record of the other use, and the block goes back on the
work queue, at most --max-quarantines times. A GPU that still has the other
use on it after a quarantine, or before a block starts, is retired: it is
re-checked every few seconds and takes blocks again after a few minutes
without other use (`kfd.RECHECK_S`, `kfd.READMIT_CLEAN_S`).
While every GPU is retired the stage waits for one, at most
`kfd.ALL_RETIRED_WAIT_S`, then gives the rest up. A bench holding memory on
a GPU other than its own means the GPU numbering is wrong; the block is
quarantined and the GPU retired for good.

Canary (--canary, `bench.canary.enabled`; off by default): other load on the
host slows the benches, a launch-bound GEMM the most. The stage's
least-work bench line, restamped to time one kernel alone without a screen
(lib/bench_yaml.py `canary_line`: the first kernel the line lists, else the
first the library holds for the problem; a block-scaled line without a list
takes the first kernel of the first --identity-stem file, and without
--identity-dir gets no canary), is benched --canary-timings times in every
bench process right before the block's shapes (after the warm-up shape)
and again after them, into logs/canary_NNNN_gpuD.log; the median of each
group is a reading. A GPU's reference is the median of every reading taken
on it. An attempt with a reading more than --canary-tolerance above its
GPU's reference drifted: its logs move to `drift/` with a record, and the
block goes back on the work queue once, to be benched no sooner than
--canary-backoff-s later; when its next attempt drifts too, that attempt is
kept and the block flagged. At the stage's end every kept attempt is judged
once more against its GPU's final reference (while the stage ran, the
first blocks on a GPU were judged against little more than their own
readings), and one over it is flagged too. A flagged block keeps its data
and its log a `*** CANARY FLAGGED` line, by which stage04 marks its rows.
The canary is blamed for a bench process that fails only once its own
output has started: failing in it before the block's shapes, the process
leaves the shapes to a new one, and none of them is charged for it; there,
after the shapes, or getting no timing, it goes off on that GPU. A process
with the canary whose output does not line up with its entries may have
put shapes into the canary log: what it wrote moves to `misaligned/` with a
record, the canary goes off on every GPU, and a new process takes the
shapes.

Outputs in `<out-dir>/`, removed and rewritten by every run:
  logs/block_NNNN_linesX-Y_gpuD.log  one log per block (block id, 1-based
                                     source-yaml line range, GPU)
  logs/block_retry_gpuD.log          the retry pass; stage04 reads it like any
                                     other block log
  logs/warmup_NNNN_gpuD.log          discarded warm-up output (the prefix keeps
                                     it out of stage04's `block_*` glob)
  logs/canary_NNNN_gpuD.log          the canary's output, per block (the retry
                                     pass's is canary_retry_gpuD.log)
  drift/<log stem>_dN/               the logs of a block's attempt whose canary
                                     drifted and drift.json: its readings, its
                                     GPU's reference, the tolerance, the GPU
                                     and the canary kernel's signature
  misaligned/<log stem>_sN/          what the N-th bench process of a block
                                     wrote to its block and canary logs when
                                     its output did not line up with its
                                     entries, and misaligned.json: why
  quarantine/<log stem>_qN/          the logs of a block's N-th quarantined
                                     attempt and quarantine.json: the rule, the
                                     other processes (pid, command, user, when
                                     seen, start time), the unattributed VRAM
                                     (bytes, card and processes' use, settle
                                     window), the GPU
  platform.json                      the GPUs' settings (lib/fingerprint.py)
  skipped_gemms.txt                  shapes still without data after the retry
                                     pass: yaml_line,M,N,K,B,rc,reason with
                                     reason `crashed`, `unattempted` or
                                     `quarantined`
  stage03_summary.json               counts, crash blocks, tolerance, rc, the
                                     quarantines, retired GPUs and the ones
                                     that returned, the processes whose
                                     output did not line up with their
                                     entries (`misaligned`), per block its
                                     GPU, that GPU's settings and its canary
                                     readings, the canary (its line and
                                     shape, per GPU its reference and
                                     kernel by signature, where it was
                                     turned off, the blocks benched again
                                     and flagged, the output set aside), and
                                     with --identity-stem the kernel
                                     identity of those library files when
                                     the bench started, naming the kernel
                                     the canary line lists
                                     (lib/kernel_identity.py), which stage04
                                     checks its .dat files against
  tmp/<pid>/                         bench yamls of the running processes of
                                     the stage process <pid>, removed when it
                                     ends; a stage never removes another
                                     live process's directory
  libdir_pool/                       with --pool-only-library: the library
                                     directory the benches load
                                     (lib/pool_libdir.py)

A `requested_solution_num: -1` line under TENSILE_PREDICTION_LIB=1 benches
the pool kernels the Prediction library ranks plus whatever
getAllSolutions() appends: the rest of the pool and every kernel of the
problem type's other libraries (such as a _CU256_ exact-match library).
Those take most of the bench time, stage04 drops their rows, and their
warm-ups enter the skip screen, so they can screen pool kernels out.
--pool-only-library STEM (with --library-dir) benches against a copy of
the library directory in which the problem type's other libraries are
empty. Planned bench lines (stage02) list their kernels and need none of
this.

A crash block is a block that was abandoned and whose unattempted shapes the
retry pass did not all recover. The stage exits 0 when the number of crash
blocks is within --max-crash-blocks (default: one block in twenty, rounded
down), else 2. Individually skipped shapes are reported but never fail it.
It also exits 2 when a block or the retry pass is quarantined more than
--max-quarantines times, when the GPU numbering is wrong, when a bench
process's output does not line up with its entries, or when blocks are left
because every GPU stopped taking them; 2 too for a refused lease or platform
check.

Every bench process leads its own process group. SIGINT, SIGTERM or SIGHUP
stops all of them (`lib.procs`) and the stage exits with 128 + the signal
number.
"""
from __future__ import annotations

import argparse
import contextlib
import heapq
import itertools
import math
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import fingerprint  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import kfd  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import procs  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_env import (  # noqa: E402
    LIBPATH_VAR,
    bench_env_with,
    recorded_env,
    runtime_env,
)
from lib.bench_log import (  # noqa: E402
    CANARY_FLAG_MARK,
    LogProbe,
    Problem,
    ProblemBoundaryTracker,
    TestedRow,
    count_finished_problems,
    iter_problems,
)
from lib.bench_yaml import (  # noqa: E402
    ADAPTIVE_KEYS,
    CANARY_DEFAULTS,
    adaptive_line,
    canary_line,
    canary_needs_kernel,
    least_work_line,
    line_has_prescreen,
    line_solution_list,
    name_entry,
    read_bench_lines,
    shape_of_line,
)
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.procs import file_size  # noqa: E402

# ── work blocks ──────────────────────────────────────────────────────────────


@dataclass
class TaskBlock:
    block_id: int
    name: str  # log file name without the `_gpu<D>.log` suffix
    lines: List[str]
    line_numbers: List[int]  # 1-based source-yaml line of each entry
    device: Optional[int] = None
    wall_s: float = 0.0
    rc: int = 0
    # Entries before `next_offset` were benched or skipped.
    next_offset: int = 0
    # offset -> return code of the process it crashed
    skipped: Dict[int, int] = field(default_factory=dict)
    abandoned: bool = False
    # offsets (skipped or unattempted) that the retry pass benched
    recovered: Set[int] = field(default_factory=set)
    # the GpuWatch record of an attempt that found another process
    contaminated: Optional[Dict[str, Any]] = None
    # quarantine.json of every quarantined attempt, oldest first
    quarantines: List[Dict[str, Any]] = field(default_factory=list)
    # quarantined more than --max-quarantines times: no data
    gave_up: bool = False
    # the GPU and its settings when the last attempt started
    gpu_fields: Dict[str, Any] = field(default_factory=dict)
    # why processes of the last attempt printed output that does not line up
    # with their entries
    misaligned: List[str] = field(default_factory=list)
    # the last attempt's canary readings, one entry per bench process
    canary_readings: List[Dict[str, Any]] = field(default_factory=list)
    # the verdict on them: the GPU's reference and the readings over it
    canary: Dict[str, Any] = field(default_factory=dict)
    # drift.json of every attempt benched again for its canary
    drifts: List[Dict[str, Any]] = field(default_factory=list)
    # data kept although its canary drifted: again after a re-bench, or
    # against the GPU's reference at the stage's end (`canary_late`)
    canary_flagged: bool = False
    canary_late: bool = False

    def reset(self) -> None:
        """Forget the last attempt, so that the block is benched anew."""
        self.device = None
        self.wall_s = 0.0
        self.rc = 0
        self.next_offset = 0
        self.skipped = {}
        self.abandoned = False
        self.recovered = set()
        self.contaminated = None
        self.misaligned = []
        self.canary_readings = []
        self.canary = {}

    def canary_record(self) -> Dict[str, Any]:
        return {
            **self.canary,
            "rebenched": len(self.drifts),
            "flagged": self.canary_flagged,
            "flagged_at_end": self.canary_late,
        }

    @property
    def unattempted(self) -> List[int]:
        if not self.abandoned:
            return []
        return list(range(self.next_offset, len(self.lines)))

    @property
    def n_benched(self) -> int:
        return self.next_offset - len(self.skipped)


def _build_blocks(lines: List[str], block_size: int) -> List[TaskBlock]:
    blocks: List[TaskBlock] = []
    for bi, start in enumerate(range(0, len(lines), block_size)):
        chunk = lines[start : start + block_size]
        numbers = list(range(start + 1, start + len(chunk) + 1))
        blocks.append(
            TaskBlock(
                block_id=bi,
                name=f"block_{bi:04d}_lines{numbers[0]}-{numbers[-1]}",
                lines=chunk,
                line_numbers=numbers,
            )
        )
    return blocks


_TRANS_RE = {
    key: re.compile(rf"(?<![\w]){key}:\s*(\w)") for key in ("transA", "transB")
}


def _line_trans(line: str, key: str) -> str:
    m = _TRANS_RE[key].search(line)
    return m.group(1) if m else "?"


# ── log helpers ──────────────────────────────────────────────────────────────


def _read_from(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _annotate(log_path: Path, msg: str) -> None:
    """Append a pipeline annotation (sub-run boundary, skipped GEMM, kill,
    abort) to the block log."""
    try:
        with log_path.open("a") as f:
            f.write(msg)
    except OSError:
        pass


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(v) for v in shape)


def _misalignment(
    problems: Sequence[Problem],
    shapes: Sequence[Optional[Tuple[int, int, int, int]]],
    ran_through: bool,
    single: bool = False,
) -> Optional[str]:
    """Why the problems one bench process wrote to one log do not line up
    with `shapes`, the shapes of the entries whose output the log takes, in
    yaml order: a problem of another shape than its entry (with `single`,
    or one with more than one kernel), more problems than entries, or fewer
    from a process that ran through (`ran_through`). None when they line
    up."""
    for i, (prob, want) in enumerate(zip(problems, shapes)):
        got = prob.shape()
        if got is not None and want is not None and tuple(got) != tuple(want):
            return (
                f"problem {i + 1} of the process is {_shape_text(got)}, its "
                f"entry {_shape_text(want)}"
            )
        if single and max(len(prob.tested), prob.n_supported or 0) > 1:
            return f"problem {i + 1} of the process times more than one kernel"
    if len(problems) > len(shapes) or (ran_through and len(problems) < len(shapes)):
        return (
            f"the process printed {len(problems)} problem(s) for {len(shapes)} "
            f"entr(ies)"
        )
    return None


# ── warm-up pass ─────────────────────────────────────────────────────────────
#
# Every block runs in a fresh hipblaslt-bench process, and Tensile loads kernel
# modules lazily on a candidate's first launch. On the first shape of a
# process the skip screen (`skip_slow_solution_ratio`, which compares one cold
# call per candidate against the best seen so far) therefore compares module
# load times, not kernel times, and prunes fast kernels that were merely slow
# to load. So one throwaway copy of the block's first shape is prepended to
# every process's yaml: it touches every candidate (rsn -1, no pruning, the
# cheapest timing), its output goes to the warm-up log, and the real shapes
# are screened against resident modules. It has to run in the same process,
# because module residency does not outlive the process.
#
# Lines with `skip_screen_prepass` get no warm-up: the bench then launches
# every candidate once, untimed, before its screening warm-up, which loads
# the modules without the library heuristic.
_WARMUP_DROP_KEYS = frozenset(("adaptive",) + ADAPTIVE_KEYS)
_WARMUP_FORCE = (
    ("requested_solution_num", "-1"),
    ("skip_slow_solution_ratio", "0"),
    ("iters", "1"),
    ("cold_iters", "1"),
    ("rotating", "0"),
    ("print_kernel_info", "0"),
)


def render_warmup_line(line: str) -> Optional[str]:
    """`line` restamped as the throwaway warm-up shape, or None when it is not
    a flat flow-mapping bench line (`- {k: v, ...}`, no nested braces and no
    commas inside values) or needs no warm-up (`skip_screen_prepass`). Key
    order is kept; forced keys not present are appended."""
    if line_has_prescreen(line):
        return None
    s = line.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    forced = dict(_WARMUP_FORCE)
    kept: List[Tuple[str, str]] = []
    seen = set()
    for part in s[i + 1 : j].split(","):
        k, sep, v = part.partition(":")
        if not sep:
            return None
        k = k.strip()
        if k in _WARMUP_DROP_KEYS:
            continue
        if k in forced:
            v = forced[k]
            seen.add(k)
        kept.append((k, v.strip()))
    for k, v in _WARMUP_FORCE:
        if k not in seen:
            kept.append((k, v))
    return s[:i] + "{" + ", ".join(f"{k}: {v}" for k, v in kept) + "}"


def _stream_split(
    proc_stdout,
    warmup_f,
    block_f,
    *,
    warmups: int = 1,
    canary_f=None,
    canaries: int = 0,
    body: int = 0,
) -> float:
    """Copy the bench's output line by line, problem by problem (one per
    entry of the process's yaml, `_entries`): the first `warmups` (the
    warm-up) to `warmup_f`; with `canary_f`, the next `canaries` to it, then
    `body` (the block's shapes) to `block_f` and the rest to `canary_f`
    again; without, every later problem to `block_f`. The process banner
    before the first problem goes to `warmup_f` and `block_f`. Returns the
    seconds spent before the first problem after the warm-up started, i.e.
    the warm-up's cost."""
    t0 = time.time()
    tracker = ProblemBoundaryTracker()
    problems = 0
    warmup_s: Optional[float] = None
    head = warmups + canaries
    while True:
        line = proc_stdout.readline()
        if not line:
            break
        if tracker.feed(line):
            problems += 1
            if problems == warmups + 1:
                warmup_s = time.time() - t0
        if problems == 0:
            for f in (warmup_f, block_f):
                if f is not None:
                    f.write(line)
        elif problems <= warmups:
            warmup_f.write(line)
        elif canary_f is not None and (problems <= head or problems > head + body):
            canary_f.write(line)
        else:
            block_f.write(line)
    return time.time() - t0 if warmup_s is None else warmup_s


# ── crash-recovery runner (one block on one device) ──────────────────────────

# Consecutive processes that finished no shape before the block is abandoned.
# Each such process was most likely killed while still loading kernel
# modules, and a new process pays that load again.
_MAX_ZERO_PROGRESS_SUBRUNS = 3


def _entries(
    warmup_line: Optional[str],
    canary_lines: Sequence[str],
    remaining: Sequence[str],
    numbers: Sequence[int],
) -> List[str]:
    """The yaml entries of one bench process, each named apart (see the
    module docstring): the warm-up shape, the canary lines, the block's
    `remaining` lines (source-yaml lines `numbers`) and the canary lines
    again."""
    out = [name_entry(warmup_line, "warmup")] if warmup_line else []
    out += [name_entry(ln, f"canary_start{i}") for i, ln in enumerate(canary_lines, 1)]
    out += [name_entry(ln, f"line{n}") for ln, n in zip(remaining, numbers)]
    out += [name_entry(ln, f"canary_end{i}") for i, ln in enumerate(canary_lines, 1)]
    return out


def _run_block_with_recovery(
    block: TaskBlock,
    device: int,
    bench_binary: Path,
    log_path: Path,
    warmup: bool = False,
    warmup_log_path: Optional[Path] = None,
    on_warmup: Optional[Callable[[int, float], None]] = None,
    stall_s: float = 900.0,
    startup_grace_s: float = 900.0,
    children: Optional[procs.ChildGroups] = None,
    work_dir: Optional[Path] = None,
    watch: Optional[kfd.GpuWatch] = None,
    canary: Optional["Canary"] = None,
    canary_log_path: Optional[Path] = None,
    aside_dir: Optional[Path] = None,
) -> None:
    """Bench `block` on `device`, appending all output to `log_path`. After a
    crash the shape after the last finished one is skipped and a new process
    resumes behind it (see the module docstring). With `warmup`, every
    process gets the throwaway warm-up shape prepended and its output split
    into `warmup_log_path`; `on_warmup(subrun, seconds)` reports its cost.
    With `canary`, every process benches the canary lines before the shapes
    (after the warm-up) and after them, into `canary_log_path`, and its
    readings join `block.canary_readings`; the output of a process with the
    canary that does not line up with its entries moves to `aside_dir`
    (default: `misaligned/` beside the log) and the canary goes off.
    Processes start through `children` with their yaml in `work_dir`
    (default: the log's directory). With `watch`, a process that finds
    another process on the GPU ends the attempt with `block.contaminated`
    set."""
    children = children if children is not None else procs.ChildGroups()
    work_dir = work_dir if work_dir is not None else log_path.parent
    poll_s = procs.poll_interval(stall_s)
    if watch is not None:
        poll_s = min(poll_s, kfd.WATCH_POLL_S)
    block.device = device
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(block.lines)
    if warmup and (warmup_log_path is None or n_total < 2):
        warmup = False
    subrun = 0
    zero_progress = 0
    last_rc = 0
    t0 = time.time()
    while block.next_offset < n_total:
        remaining = block.lines[block.next_offset :]
        numbers = block.line_numbers[block.next_offset :]
        warmup_line = (
            render_warmup_line(remaining[0])
            if (warmup and len(remaining) >= 2)
            else None
        )
        canary_lines = (
            canary.lines(device)
            if canary is not None and canary_log_path is not None
            else []
        )
        if subrun > 0:
            _annotate(
                log_path,
                f"\n*** SUB-RUN #{subrun + 1}: resume at block-offset "
                f"{block.next_offset} with {len(remaining)} GEMMs remaining ***\n",
            )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix=f"{block.name}_", dir=work_dir, delete=False
        ) as f:
            tmp_yaml = Path(f.name)
            f.writelines(_entries(warmup_line, canary_lines, remaining, numbers))
        sub_start = file_size(log_path)
        warmup_start = file_size(warmup_log_path)
        canary_start = file_size(canary_log_path)
        progress = LogProbe(log_path, sub_start)
        bench_device, env = kfd.one_gpu_launch(device, bench_env_with(BENCH_ENV_EXTRA))
        cmd = [str(bench_binary), "--device", bench_device, "--yaml", str(tmp_yaml)]
        reason: Optional[str] = None
        ended: Optional[float] = None
        try:
            if warmup_line or canary_lines:
                probes = [progress]
                with contextlib.ExitStack() as files:
                    lf = files.enter_context(log_path.open("a", buffering=1))
                    wf = cf = None
                    if warmup_line and warmup_log_path is not None:
                        probes.append(LogProbe(warmup_log_path, warmup_start))
                        wf = files.enter_context(warmup_log_path.open("a", buffering=1))
                        wf.write(
                            f"*** WARM-UP (discarded) sub-run #{subrun + 1}, "
                            f"{block.name}, gpu {device} ***\n"
                        )
                    if canary_lines and canary_log_path is not None:
                        probes.append(LogProbe(canary_log_path, canary_start))
                        cf = files.enter_context(canary_log_path.open("a", buffering=1))
                        cf.write(
                            f"*** CANARY sub-run #{subrun + 1}, {block.name}, "
                            f"gpu {device} ***\n"
                        )
                    proc = children.popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=env,
                    )
                    if watch is not None:
                        watch.add_bench(proc.pid)
                    # The split has to run on its own thread: a hung process
                    # keeps its pipe open, and the watchdog below must still
                    # get to kill it.
                    warm_s = {"s": 0.0}

                    def _drain() -> None:
                        try:
                            warm_s["s"] = _stream_split(
                                proc.stdout,
                                wf,
                                lf,
                                warmups=1 if wf is not None else 0,
                                canary_f=cf,
                                canaries=len(canary_lines),
                                body=len(remaining),
                            )
                        except Exception:
                            pass

                    drain = threading.Thread(target=_drain, daemon=True)
                    drain.start()
                    last_rc, reason = procs.wait_with_stall_watchdog(
                        proc,
                        lambda: sum(
                            file_size(p)
                            for p in (log_path, warmup_log_path, canary_log_path)
                        ),
                        stall_s,
                        poll_s=poll_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=progress.progressed,
                        probe_fatal=lambda: any(p.fatal_seen() for p in probes),
                        probe_contaminated=watch.sample if watch else None,
                    )
                    ended = kfd.boot_clock()
                    drain.join(timeout=30)
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                if wf is not None and on_warmup is not None:
                    on_warmup(subrun + 1, warm_s["s"])
            else:
                with log_path.open("a", buffering=1) as lf:
                    proc = children.popen(
                        cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    if watch is not None:
                        watch.add_bench(proc.pid)
                    last_rc, reason = procs.wait_with_stall_watchdog(
                        proc,
                        lambda: file_size(log_path),
                        stall_s,
                        poll_s=poll_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=progress.progressed,
                        probe_fatal=progress.fatal_seen,
                        probe_contaminated=watch.sample if watch else None,
                    )
                    ended = kfd.boot_clock()
        finally:
            try:
                tmp_yaml.unlink()
            except OSError:
                pass
        if watch is not None and (reason == "contaminated" or watch.check_end(ended)):
            block.contaminated = watch.record()
            _annotate(
                log_path,
                f"\n*** QUARANTINED [{block.contaminated['rule']}]: "
                f"{block.contaminated['reason']} ***\n",
            )
            break
        text = _read_from(log_path, sub_start)
        shape_problems = list(iter_problems(text.splitlines()))
        ran_through = last_rc == 0 and reason is None
        off = _misalignment(
            shape_problems, [shape_of_line(ln) for ln in remaining], ran_through
        )
        if canary is not None and canary_lines and canary_log_path is not None:
            canary_problems = list(
                iter_problems(_read_from(canary_log_path, canary_start).splitlines())
            )
            off = off or _misalignment(
                canary_problems,
                [canary.shape] * (2 * len(canary_lines)),
                ran_through,
                single=True,
            )
            if off is not None:
                _canary_misaligned(
                    block,
                    device,
                    canary,
                    subrun + 1,
                    off,
                    [(log_path, sub_start), (canary_log_path, canary_start)],
                    aside_dir or log_path.parent / MISALIGNED_DIR,
                )
                subrun += 1
                continue
            failed = _read_canary(
                block,
                device,
                canary,
                subrun + 1,
                canary_problems,
                shape_problems,
                len(canary_lines),
                len(remaining),
                not ran_through,
            )
            if failed is not None:
                _annotate(
                    log_path,
                    f"\n*** CANARY FAILED (rc={last_rc}"
                    + (f", {reason}" if reason else "")
                    + f"): {failed}; a new process takes the shapes ***\n",
                )
                subrun += 1
                continue
        elif off is not None:
            block.misaligned.append(off)
            _annotate(log_path, f"\n*** MISALIGNED: {off} ***\n")
            ui.err("bench", f"{block.name} on gpu{device}: {off}")
        if reason is not None:
            what = (
                "unrecoverable GPU queue abort in the log"
                if reason == "fatal"
                else f"no output for {stall_s:.0f}s"
            )
            _annotate(log_path, f"\n*** KILLED [{reason}]: {what} ***\n")
        subrun += 1
        if last_rc == 0:
            block.next_offset = n_total
            break
        finished = min(count_finished_problems(text.splitlines()), len(remaining))
        zero_progress = zero_progress + 1 if finished == 0 else 0
        crash_idx = block.next_offset + finished
        if crash_idx >= n_total:
            # Every shape finished before the non-zero exit, so the fault came
            # from process teardown and no data was lost.
            last_rc = 0
            block.next_offset = n_total
            break
        line = block.lines[crash_idx]
        shape = shape_of_line(line)
        m, n, k, b = shape if shape is not None else ("?", "?", "?", "?")
        _annotate(
            log_path,
            f"\n*** SKIPPED CRASHED GEMM at block-offset {crash_idx} (yaml line "
            f"{block.line_numbers[crash_idx]}): M={m} N={n} K={k} B={b} "
            f"transA={_line_trans(line, 'transA')} "
            f"transB={_line_trans(line, 'transB')} (rc={last_rc}) ***\n",
        )
        block.skipped[crash_idx] = last_rc
        block.next_offset = crash_idx + 1
        abandon = ""
        if block.next_offset < n_total:
            if zero_progress >= _MAX_ZERO_PROGRESS_SUBRUNS:
                abandon = f"{zero_progress} consecutive sub-runs finished no GEMM"
            elif subrun > n_total + 1:
                abandon = f"too many sub-runs ({subrun})"
        if abandon:
            block.abandoned = True
            _annotate(
                log_path,
                f"\n*** ABORT: {abandon}; {n_total - block.next_offset} GEMMs "
                f"left unattempted ***\n",
            )
            break
    block.wall_s = time.time() - t0
    block.rc = last_rc


# ── canary ───────────────────────────────────────────────────────────────────

CANARY_LOG_PREFIX = "canary_"
DRIFT_DIR = "drift"
DRIFT_RECORD = "drift.json"
MISALIGNED_DIR = "misaligned"
MISALIGNED_RECORD = "misaligned.json"


def _latency(fields: Mapping[str, str]) -> Optional[float]:
    try:
        us = float(fields["us"])
    except (KeyError, TypeError, ValueError):
        return None
    return us if math.isfinite(us) and us > 0 else None


def kernel_label(row: TestedRow, identity: Any = None) -> str:
    """How diagnostics name the kernel of a tested row: its signature
    (lib/kernel_identity.py `signature_label`) when `identity`, the
    LibraryIdentity of the library files the bench loaded, holds the row's
    solution index under the row's solution name; else the macro tile and
    name hash its solution name gives (`label_of_name`)."""
    from lib import kernel_identity as ki

    name = row.sol_name or row.kernel_name
    if identity is not None and row.sol_index is not None:
        kernel = identity.kernels.get(int(row.sol_index))
        if kernel is not None and kernel.get("kernel_name") == name:
            return ki.signature_label(kernel)
    return ki.label_of_name(name)


class Canary:
    """The stage's canary line and its readings. Every bench process of a
    block benches `timings` canary problems before the block's shapes and
    again after them; the median latency of each group is a reading. A
    GPU's reference is the median of every reading taken on it; a reading
    more than `tolerance` above it drifts, and its block waits `backoff_s`
    before it is benched again. The canary is off on a GPU once it failed a
    bench process there or got no timing, and on every GPU once a process's
    output did not line up with its entries (`disable`). `identity` (a
    LibraryIdentity of the library files the bench loads, or None) names
    the canary kernel in diagnostics."""

    def __init__(
        self,
        line: str,
        timings: int,
        tolerance: float,
        backoff_s: float = 0.0,
        identity: Any = None,
    ) -> None:
        self.line = line.rstrip("\n") + "\n"
        self.shape = shape_of_line(self.line)
        self.timings = int(timings)
        self.tolerance = float(tolerance)
        self.backoff_s = float(backoff_s)
        self.identity = identity
        self.readings: Dict[int, List[float]] = {}
        # per GPU, the solution name of the kernel its canary timed and how
        # diagnostics name that kernel
        self.kernels: Dict[int, str] = {}
        self.labels: Dict[int, str] = {}
        self.off: Dict[int, str] = {}
        self.disabled: Optional[str] = None
        self.misaligned: List[Dict[str, Any]] = []
        self.rebenched: List[str] = []
        self.flagged: List[str] = []
        self.flagged_at_end: List[str] = []
        self._lock = threading.Lock()

    def describe(self) -> str:
        m, n, k, b = self.shape or ("?", "?", "?", "?")
        return (
            f"M={m} N={n} K={k} B={b}, {self.timings} timings before and after "
            f"each block's shapes, tolerance {self.tolerance:g}, back-off "
            f"{ui.fmt_dur(self.backoff_s)}"
        )

    def lines(self, device: int) -> List[str]:
        """The lines a bench process on `device` benches before the block's
        shapes and again after them."""
        with self._lock:
            if self.disabled is not None or device in self.off:
                return []
            return [self.line] * self.timings

    def turn_off(self, device: int, why: str) -> None:
        with self._lock:
            if self.disabled is not None or device in self.off:
                return
            self.off[device] = why
        ui.warn("canary", f"gpu{device} benches without the canary from now: {why}")

    def disable(self, why: str) -> None:
        with self._lock:
            if self.disabled is not None:
                return
            self.disabled = why
        ui.err("canary", f"every GPU benches without the canary from now: {why}")

    def timings_of(self, device: int, problems: Sequence[Problem]) -> List[float]:
        """The canary kernel's latency in each complete problem of
        `problems`: the kernel, by solution name, of the first timing on
        `device`; a problem that timed another, or names none, gives none."""
        out: List[float] = []
        for prob in problems:
            if not prob.complete or not prob.tested:
                continue
            row = prob.tested[0]
            us = _latency(row.fields)
            name = row.sol_name or row.kernel_name
            if us is None or not name:
                continue
            with self._lock:
                if self.kernels.setdefault(device, name) != name:
                    continue
                named = device in self.labels
            if not named:
                label = kernel_label(row, self.identity)
                with self._lock:
                    self.labels.setdefault(device, label)
            out.append(us)
        return out

    def judge(self, device: int, values: Sequence[float]) -> Tuple[float, List[float]]:
        """Add `values`, the readings of one attempt on `device`, to the
        GPU's; (its reference with them, the values over it by more than the
        tolerance)."""
        with self._lock:
            taken = self.readings.setdefault(device, [])
            taken.extend(values)
            reference = statistics.median(taken)
        return reference, self.over(reference, values)

    def over(self, reference: float, values: Sequence[float]) -> List[float]:
        limit = reference * (1.0 + self.tolerance)
        return [v for v in values if v > limit]

    def reference(self, device: int) -> Optional[float]:
        """The median of every reading taken on `device` so far."""
        with self._lock:
            taken = self.readings.get(device)
            return statistics.median(taken) if taken else None

    def note(self, name: str, flagged: bool, at_end: bool = False) -> None:
        with self._lock:
            if at_end:
                self.flagged_at_end.append(name)
            else:
                (self.flagged if flagged else self.rebenched).append(name)

    def set_aside(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.misaligned.append(dict(record))

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "line": self.line.rstrip("\n"),
                "shape": list(self.shape) if self.shape else None,
                "timings": self.timings,
                "tolerance": self.tolerance,
                "backoff_s": self.backoff_s,
                "references": {
                    str(d): statistics.median(v)
                    for d, v in sorted(self.readings.items())
                },
                "n_readings": {
                    str(d): len(v) for d, v in sorted(self.readings.items())
                },
                "kernels": {str(d): k for d, k in sorted(self.labels.items())},
                "off": {str(d): w for d, w in sorted(self.off.items())},
                "disabled": self.disabled,
                "misaligned": list(self.misaligned),
                "rebenched": list(self.rebenched),
                "flagged": list(self.flagged),
                "flagged_at_end": list(self.flagged_at_end),
            }


def _first_kernel(identity: Any) -> Optional[int]:
    """The lowest solution index of the first library file of `identity`
    (the orchestrator names the pool's first), None without one."""
    if identity is None or not identity.stems:
        return None
    ids = identity.file_ids.get(identity.stems[0]) or {}
    return min(ids) if ids else None


def _resolve_canary(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    lines: Sequence[str],
    identity: Any = None,
) -> Tuple[Optional[Canary], str]:
    """The stage's canary (CLI flags > `bench.canary` of the config >
    CANARY_DEFAULTS) on its least-work line, or None and why there is none.
    A block-scaled line that lists no kernel times the first kernel of the
    first library file of `identity` (`_first_kernel`). ValueError for a
    tolerance, number of timings or back-off out of range."""
    blk = (cfg.get("bench") or {}).get("canary")
    blk = blk if isinstance(blk, dict) else {}

    def pick(cli: Any, key: str) -> Any:
        return cli if cli is not None else blk.get(key, CANARY_DEFAULTS[key])

    if not pick(args.canary, "enabled"):
        return None, "turned off"
    tolerance = float(pick(args.canary_tolerance, "tolerance"))
    timings = int(pick(args.canary_timings, "timings"))
    backoff_s = float(pick(args.canary_backoff_s, "backoff_s"))
    if not tolerance > 0 or timings < 1 or not backoff_s >= 0:
        raise ValueError(
            f"the canary needs a tolerance above 0, at least one timing and a "
            f"back-off of at least 0 s (got {tolerance:g}, {timings} and "
            f"{backoff_s:g})"
        )
    source = least_work_line(lines)
    if source is None:
        return None, "no bench line names its shape"
    kernel = _first_kernel(identity) if canary_needs_kernel(source) else None
    line = canary_line(source, kernel)
    if line is None:
        return None, (
            "the least-work line is block-scaled and lists no kernel, and "
            "without --identity-dir there is no library file to take the "
            "canary's kernel from"
        )
    return Canary(line, timings, tolerance, backoff_s, identity), ""


def _n_finished(problems: Sequence[Problem]) -> int:
    """How many of one process's `problems` it got past (as
    `count_finished_problems`)."""
    if not problems:
        return 0
    return len(problems) - 1 + (1 if problems[-1].finished else 0)


def _read_canary(
    block: TaskBlock,
    device: int,
    canary: Canary,
    subrun: int,
    canary_problems: Sequence[Problem],
    shape_problems: Sequence[Problem],
    n_canary: int,
    n_shapes: int,
    failed: bool,
) -> Optional[str]:
    """Add the readings of one bench process of `block` to
    `block.canary_readings`: its `canary_problems` are the `n_canary` it
    benched before the `n_shapes` shapes whose problems are
    `shape_problems`, and those after them. A process that failed
    (`failed`) is blamed on the canary only once the canary's own output
    had started: in the canary before the shapes (returned: why; a new
    process takes the shapes) or after them; either way the canary goes off
    on `device`. So does a canary that got no timing in a process that ran
    through. None unless the process failed in the canary before the
    shapes."""
    head, tail = canary_problems[:n_canary], canary_problems[n_canary:]
    start = canary.timings_of(device, head)
    end = canary.timings_of(device, tail)
    block.canary_readings.append(
        {
            "subrun": subrun,
            "start": statistics.median(start) if start else None,
            "end": statistics.median(end) if end else None,
            "start_timings": start,
            "end_timings": end,
        }
    )
    if not failed:
        if not start:
            canary.turn_off(device, "the canary got no timing")
        return None
    if not shape_problems and head:
        if len(head) < n_canary or not head[-1].finished:
            why = "a bench process failed in the canary before the block's shapes"
            canary.turn_off(device, why)
            return why
    if tail and _n_finished(shape_problems) >= n_shapes:
        if len(tail) < n_canary or not tail[-1].finished:
            canary.turn_off(
                device, "a bench process failed in the canary after the block's shapes"
            )
    return None


def _move_aside(
    dest: Path, files: Sequence[Optional[Path]], name: str, record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Move the existing `files` into `dest` and write `record`, with the
    moved names and the time, as `dest/<name>`; returns it."""
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in files:
        if p is not None and Path(p).exists():
            shutil.move(str(p), str(dest / Path(p).name))
            moved.append(Path(p).name)
    rec = {**record, "moved": kfd.utc_iso(), "files": moved}
    write_json(dest / name, rec)
    return rec


def _cut_aside(
    dest: Path, parts: Sequence[Tuple[Optional[Path], int]], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Move what each log of `parts`, (log, byte offset) pairs, holds from
    its offset on into a file of its name in `dest`, cutting the log back to
    the offset, and write `record`, with the moved names and the time, as
    `dest/misaligned.json`; returns it."""
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for path, start in parts:
        if path is None:
            continue
        try:
            with open(path, "rb+") as f:
                f.seek(start)
                data = f.read()
                f.seek(start)
                f.truncate()
        except OSError:
            continue
        if data:
            (dest / Path(path).name).write_bytes(data)
            moved.append(Path(path).name)
    rec = {**record, "moved": kfd.utc_iso(), "files": moved}
    write_json(dest / MISALIGNED_RECORD, rec)
    return rec


def _canary_misaligned(
    block: TaskBlock,
    device: int,
    canary: Canary,
    subrun: int,
    why: str,
    parts: Sequence[Tuple[Optional[Path], int]],
    aside_dir: Path,
) -> None:
    """A bench process of `block` with the canary printed output that does
    not line up with its entries (`why`), so some of it may sit in the wrong
    log: what it wrote to its logs (`parts`, the block log first) moves to
    `aside_dir/<block log stem>_s<subrun>/` with misaligned.json, and the
    canary goes off on every GPU. A new process takes the shapes."""
    log_path = Path(str(parts[0][0]))
    dest = aside_dir / f"{log_path.stem}_s{subrun}"
    rec = _cut_aside(
        dest,
        parts,
        {
            "block": block.name,
            "yaml_lines": [block.line_numbers[0], block.line_numbers[-1]],
            "device": device,
            "subrun": subrun,
            "reason": why,
        },
    )
    canary.set_aside({**rec, "dir": str(dest)})
    canary.disable(
        f"a bench process of {block.name} on gpu{device} printed output that "
        f"does not line up with its entries ({why})"
    )
    _annotate(
        log_path,
        f"\n*** CANARY MISALIGNED: {why}; the process's output moved to {dest}, "
        f"a new process takes the shapes without the canary ***\n",
    )


def _reading_values(block: TaskBlock) -> List[float]:
    return [
        r[k]
        for r in block.canary_readings
        for k in ("start", "end")
        if r[k] is not None
    ]


def _judge_canary(
    block: TaskBlock,
    device: int,
    canary: Canary,
    logs: Sequence[Optional[Path]],
    drift_dir: Path,
) -> bool:
    """Judge the attempt of `block` that just ended on `device` by its canary
    readings (`Canary.judge`). On its first drift the attempt's `logs` move
    to `drift_dir/<log stem>_d1/` with drift.json and the block is reset:
    True, bench it again. A block that drifts again keeps its data and is
    flagged (`CANARY_FLAG_MARK` in its log, which stage04 marks its rows
    by)."""
    values = _reading_values(block)
    if not values:
        return False
    reference, high = canary.judge(device, values)
    block.canary = {
        "reference": reference,
        "readings": [dict(r) for r in block.canary_readings],
        "drifted": high,
    }
    if not high:
        return False
    log_path = Path(str(logs[0]))
    why = (
        f"canary {', '.join(f'{v:.4g}' for v in high)} us on gpu{device}, more "
        f"than {canary.tolerance:g} over its reference {reference:.4g} us"
    )
    if block.drifts:
        block.canary_flagged = True
        canary.note(block.name, flagged=True)
        _annotate(log_path, f"\n{CANARY_FLAG_MARK}: {why}, again; kept ***\n")
        ui.warn("canary", f"{block.name}: {why}, again; kept and flagged")
        return False
    _annotate(log_path, f"\n*** CANARY DRIFT: {why}; benched again ***\n")
    dest = drift_dir / f"{log_path.stem}_d1"
    rec = _move_aside(
        dest,
        logs,
        DRIFT_RECORD,
        {
            "block": block.name,
            "yaml_lines": [block.line_numbers[0], block.line_numbers[-1]],
            **block.gpu_fields,
            "device": device,
            "reason": why,
            "tolerance": canary.tolerance,
            **({"kernel": canary.labels[device]} if device in canary.labels else {}),
            **block.canary,
        },
    )
    block.drifts.append({**rec, "dir": str(dest)})
    canary.note(block.name, flagged=False)
    ui.warn(
        "canary",
        f"{block.name}: {why} -> {dest}; benched again in "
        f"{ui.fmt_dur(canary.backoff_s)}",
    )
    block.reset()
    return True


def _flag_at_end(blocks: Sequence[TaskBlock], canary: Canary, logs_dir: Path) -> None:
    """Judge the kept attempt of every block once more, against its GPU's
    reference at the stage's end: while the stage ran, an attempt was judged
    against the readings taken so far, its own among them, so the first
    blocks on a GPU were hardly judged. An attempt with a reading over the
    final reference by more than the tolerance keeps its data and is
    flagged (`CANARY_FLAG_MARK` in its log)."""
    for b in blocks:
        if b.gave_up or b.canary_flagged or b.device is None:
            continue
        values = _reading_values(b)
        reference = canary.reference(b.device)
        if not values or reference is None:
            continue
        high = canary.over(reference, values)
        if not high:
            continue
        why = (
            f"canary {', '.join(f'{v:.4g}' for v in high)} us on gpu{b.device}, "
            f"more than {canary.tolerance:g} over its reference at the stage's "
            f"end {reference:.4g} us"
        )
        b.canary_flagged = b.canary_late = True
        b.canary.update(reference_at_end=reference, drifted_at_end=high)
        canary.note(b.name, flagged=True, at_end=True)
        _annotate(
            logs_dir / f"{b.name}_gpu{b.device}.log",
            f"\n{CANARY_FLAG_MARK}: {why}; kept ***\n",
        )
        ui.warn("canary", f"{b.name}: {why}; kept and flagged")


# ── GPUs and quarantine ──────────────────────────────────────────────────────

QUARANTINE_DIR = kfd.QUARANTINE_DIR
DEFAULT_MAX_QUARANTINES = 2


def _retire(gpus: kfd.StageGpus, device: int, why: str, temporary: bool) -> None:
    if gpus.retire(device, why, temporary):
        if temporary:
            ui.warn(
                "quiet",
                f"gpu{device} takes no blocks until it has had no other use for "
                f"{ui.fmt_dur(kfd.READMIT_CLEAN_S)}: {why}",
            )
        else:
            ui.warn("quiet", f"gpu{device} takes no more blocks: {why}")


def _await_gpu(
    gpus: kfd.StageGpus, devices: Sequence[int], done: Callable[[], bool]
) -> Optional[int]:
    """The first of `devices` in service, re-checking the retired ones that
    may return every `kfd.RECHECK_S` until one does or `done()`. While every
    GPU of the stage is retired the wait lasts at most
    `kfd.ALL_RETIRED_WAIT_S`. None when no device is in service."""
    noted: Optional[float] = None
    while True:
        for d in devices:
            was = gpus.active(d)
            if gpus.recheck(d):
                if not was:
                    ui.ok(
                        "quiet",
                        f"gpu{d} takes blocks again: no other use for "
                        f"{ui.fmt_dur(kfd.READMIT_CLEAN_S)}",
                    )
                return d
        if done() or not any(gpus.may_return(d) for d in devices):
            return None
        waited = gpus.all_retired_for()
        if waited is not None:
            limit = kfd.ALL_RETIRED_WAIT_S
            reporter = min(d for d in gpus.gpu if gpus.may_return(d))
            if waited >= limit:
                if reporter in devices:
                    ui.err(
                        "quiet",
                        f"every GPU has been retired for {ui.fmt_dur(waited)}; "
                        f"giving up",
                    )
                return None
            if reporter in devices and (noted is None or waited - noted >= 60.0):
                ui.warn(
                    "quiet",
                    f"every GPU is retired; waiting up to "
                    f"{ui.fmt_dur(limit - waited)} more for one to return",
                )
                noted = waited
        deadline = time.monotonic() + kfd.RECHECK_S
        while time.monotonic() < deadline and not done():
            time.sleep(min(0.2, kfd.RECHECK_S))


def _block_fields(gpus: kfd.StageGpus, device: int) -> Dict[str, Any]:
    return fingerprint.block_fields(
        device, gpus.physical.get(device), gpus.gpu.get(device)
    )


def _quarantine(
    block: TaskBlock,
    device: int,
    log_path: Path,
    warmup_log_path: Optional[Path],
    quarantine_dir: Path,
    gpus: kfd.StageGpus,
    canary_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Move the logs of `block`'s contaminated attempt to
    `quarantine_dir/<log stem>_q<n>/` with quarantine.json, reset the block
    for a re-bench, and retire `device` when the GPU numbering is wrong
    (for good) or the other use is still on it (until it is gone). Returns
    the record."""
    n = len(block.quarantines) + 1
    dest = quarantine_dir / f"{log_path.stem}_q{n}"
    rec = kfd.quarantine(
        dest,
        (log_path, warmup_log_path, canary_log_path),
        {
            "block": block.name,
            "yaml_lines": [block.line_numbers[0], block.line_numbers[-1]],
            "device": device,
            "attempt": n,
            **(block.contaminated or {}),
        },
    )
    block.quarantines.append({**rec, "dir": str(dest)})
    ui.warn("quarantine", f"{block.name} on gpu{device}: {rec['reason']} -> {dest}")
    if rec.get("rule") == "gpu-numbering":
        _retire(gpus, device, str(rec["reason"]), temporary=False)
    else:
        still = gpus.busy(device)
        if still:
            _retire(
                gpus,
                device,
                "; ".join(f.describe() for f in still) + " still on it",
                temporary=True,
            )
    block.reset()
    return rec


class _Work:
    """The work queue: blocks to bench, which a quarantined block re-enters
    up to `max_quarantines` times, a block whose canary drifted after a
    delay, and the blocks that are finished or given up. A worker waits for
    work while blocks are still out."""

    def __init__(
        self,
        blocks: Sequence[TaskBlock],
        max_quarantines: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.q: queue.Queue = queue.Queue()
        for b in blocks:
            self.q.put(b)
        self.max_quarantines = int(max_quarantines)
        self.clock = clock
        self.lock = threading.Lock()
        self.done: List[TaskBlock] = []
        self._open = len(blocks)
        self._later: List[Tuple[float, int, TaskBlock]] = []
        self._seq = itertools.count()

    def _due(self) -> Optional[TaskBlock]:
        with self.lock:
            if self._later and self._later[0][0] <= self.clock():
                return heapq.heappop(self._later)[2]
        return None

    def take(self, stop: Callable[[], bool]) -> Optional[TaskBlock]:
        while not stop():
            block = self._due()
            if block is not None:
                return block
            try:
                return self.q.get(timeout=0.2)
            except queue.Empty:
                with self.lock:
                    if self._open == 0:
                        return None
        return None

    def closed(self) -> bool:
        with self.lock:
            return self._open == 0

    def finish(self, block: TaskBlock) -> None:
        with self.lock:
            self.done.append(block)
            self._open -= 1

    def requeue(self, block: TaskBlock, delay_s: float = 0.0) -> None:
        """Put `block` back, to be taken no sooner than `delay_s` from now;
        a block quarantined more than `max_quarantines` times is given up."""
        if len(block.quarantines) > self.max_quarantines:
            block.gave_up = True
            self.finish(block)
        elif delay_s > 0:
            with self.lock:
                due = self.clock() + float(delay_s)
                heapq.heappush(self._later, (due, next(self._seq), block))
        else:
            self.q.put(block)


# ── device worker ────────────────────────────────────────────────────────────


class _DeviceWorker(threading.Thread):
    """Take blocks off `work`, run each on `device`, finish or requeue it;
    while `device` is retired, wait for it to return (`_await_gpu`)."""

    def __init__(
        self,
        device: int,
        work: _Work,
        bench_binary: Path,
        logs_dir: Path,
        *,
        warmup: bool = True,
        stall_s: float = 300.0,
        startup_grace_s: float = 900.0,
        quiet: bool = False,
        children: Optional[procs.ChildGroups] = None,
        work_dir: Optional[Path] = None,
        gpus: Optional[kfd.StageGpus] = None,
        quarantine_dir: Optional[Path] = None,
        canary: Optional[Canary] = None,
    ) -> None:
        super().__init__(name=f"bench-d{device}", daemon=True)
        self.device = device
        self.work = work
        self.bench_binary = bench_binary
        self.logs_dir = logs_dir
        self.warmup = warmup
        self.stall_s = stall_s
        self.startup_grace_s = startup_grace_s
        self.quiet = quiet
        self.children = children
        self.work_dir = work_dir
        self.gpus = gpus if gpus is not None else kfd.StageGpus([])
        self.quarantine_dir = quarantine_dir or logs_dir.parent / QUARANTINE_DIR
        self.canary = canary
        self.n_blocks_completed = 0
        self.wall_s = 0.0
        self.warmup_s = 0.0

    def run(self) -> None:
        t0 = time.time()
        while True:
            if not self.gpus.active(self.device):
                if _await_gpu(self.gpus, [self.device], self.work.closed) is None:
                    break
            block = self.work.take(lambda: not self.gpus.active(self.device))
            if block is None:
                if self.gpus.active(self.device):
                    break
                continue
            busy = self.gpus.busy(self.device)
            if busy:
                _retire(
                    self.gpus,
                    self.device,
                    "; ".join(f.describe() for f in busy) + " on it",
                    temporary=True,
                )
                self.work.requeue(block)
                continue
            settled = False
            try:
                self._bench(block)
                settled = True
            finally:
                if not settled:
                    block.abandoned = True
                    block.gave_up = block.contaminated is not None
                    self.work.finish(block)
        self.wall_s = time.time() - t0

    def _bench(self, block: TaskBlock) -> None:
        """Bench `block`, then finish it, or quarantine and requeue it."""
        log_path = self.logs_dir / f"{block.name}_gpu{self.device}.log"
        warmup_log_path = (
            self.logs_dir / f"warmup_{block.block_id:04d}_gpu{self.device}.log"
        )
        canary_log_path = self.logs_dir / (
            f"{CANARY_LOG_PREFIX}{block.block_id:04d}_gpu{self.device}.log"
        )

        def _on_warmup(subrun: int, secs: float) -> None:
            self.warmup_s += secs
            if not self.quiet:
                ui.grey(
                    "warm",
                    f"gpu{self.device} {block.name} sub-run #{subrun}: warm-up "
                    f"took {ui.fmt_dur(secs)} (discarded -> {warmup_log_path.name})",
                )

        block.gpu_fields = _block_fields(self.gpus, self.device)
        try:
            _run_block_with_recovery(
                block,
                self.device,
                self.bench_binary,
                log_path,
                warmup=self.warmup,
                warmup_log_path=warmup_log_path,
                stall_s=self.stall_s,
                startup_grace_s=self.startup_grace_s,
                on_warmup=_on_warmup,
                children=self.children,
                work_dir=self.work_dir,
                watch=self.gpus.watch(self.device),
                canary=self.canary,
                canary_log_path=canary_log_path,
                aside_dir=self.logs_dir.parent / MISALIGNED_DIR,
            )
        except Exception as exc:
            block.rc = -1
            block.abandoned = True
            _annotate(log_path, f"\n*** ERROR: {exc} ***\n")
        if block.contaminated is not None:
            _quarantine(
                block,
                self.device,
                log_path,
                warmup_log_path,
                self.quarantine_dir,
                self.gpus,
                canary_log_path,
            )
            self.work.requeue(block)
            return
        logs = (log_path, warmup_log_path, canary_log_path)
        drift_dir = self.logs_dir.parent / DRIFT_DIR
        if self.canary is not None and _judge_canary(
            block, self.device, self.canary, logs, drift_dir
        ):
            self.work.requeue(block, self.canary.backoff_s)
            return
        self.work.finish(block)
        self.n_blocks_completed += 1


# ── main ─────────────────────────────────────────────────────────────────────


def _read_config(path: Optional[Path]) -> Dict[str, Any]:
    """The run config of --config-yaml; {} (every default) without one or
    when it cannot be read."""
    if path is None or not path.exists():
        return {}
    try:
        import yaml

        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("the top level is not a mapping")
        return cfg
    except Exception as exc:
        ui.warn(
            "cfg",
            f"could not read {path}: {exc}; using the defaults of "
            f"bench.warmup_pass (on) and bench.adaptive",
        )
        return {}


def _resolve_warmup(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """CLI flag > `bench.warmup_pass` of the config > on."""
    if args.warmup_pass is not None:
        return bool(args.warmup_pass), "cli"
    v = (cfg.get("bench") or {}).get("warmup_pass")
    if v is not None:
        return bool(v), "config"
    return True, "default"


def _clean_previous_outputs(out_dir: Path, logs_dir: Path) -> int:
    removed = 0
    for pattern in ("block_*.log", "warmup_*.log", f"{CANARY_LOG_PREFIX}*.log"):
        for p in logs_dir.glob(pattern):
            p.unlink()
            removed += 1
    for name in (
        "skipped_gemms.txt",
        "stage03_summary.json",
        fingerprint.PLATFORM_FILE,
    ):
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    for name in (QUARANTINE_DIR, DRIFT_DIR, MISALIGNED_DIR):
        if (out_dir / name).is_dir():
            shutil.rmtree(out_dir / name)
            removed += 1
    return removed


def _retry_pass(
    blocks: List[TaskBlock],
    devices: Sequence[int],
    bench_binary: Path,
    logs_dir: Path,
    stall_s: float,
    startup_grace_s: float,
    children: Optional[procs.ChildGroups] = None,
    work_dir: Optional[Path] = None,
    gpus: Optional[kfd.StageGpus] = None,
    max_quarantines: int = DEFAULT_MAX_QUARANTINES,
    canary: Optional[Canary] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[int, int, Optional[TaskBlock]]:
    """Re-bench every skipped and unattempted shape of `blocks` in one process
    on the first of `devices` that takes blocks (`_await_gpu`; another one
    after a quarantine, up to `max_quarantines` times, or once more, after
    the canary's back-off, when its canary drifted), marking the ones that
    finish in `TaskBlock.recovered`. Returns (retried, recovered, the retry
    block)."""
    gpus = gpus if gpus is not None else kfd.StageGpus([])
    pending = [
        (b, off)
        for b in blocks
        if not b.gave_up
        for off in sorted(set(b.skipped) | set(b.unattempted))
    ]
    if not pending:
        return 0, 0, None
    retry = TaskBlock(
        block_id=-1,
        name="block_retry",
        lines=[b.lines[off] for b, off in pending],
        line_numbers=[b.line_numbers[off] for b, off in pending],
    )
    while True:
        device = _await_gpu(gpus, devices, lambda: False)
        if device is None:
            ui.warn("retry", "no GPU takes blocks; the retry pass does not run")
            return len(pending), 0, retry
        log_path = logs_dir / f"{retry.name}_gpu{device}.log"
        canary_log_path = logs_dir / f"{CANARY_LOG_PREFIX}retry_gpu{device}.log"
        retry.gpu_fields = _block_fields(gpus, device)
        _run_block_with_recovery(
            retry,
            device,
            bench_binary,
            log_path,
            warmup=False,
            stall_s=stall_s,
            startup_grace_s=startup_grace_s,
            children=children,
            work_dir=work_dir,
            watch=gpus.watch(device),
            canary=canary,
            canary_log_path=canary_log_path,
            aside_dir=logs_dir.parent / MISALIGNED_DIR,
        )
        if retry.contaminated is None:
            logs = (log_path, None, canary_log_path)
            drift_dir = logs_dir.parent / DRIFT_DIR
            if canary is None or not _judge_canary(
                retry, device, canary, logs, drift_dir
            ):
                break
            sleep(canary.backoff_s)
            continue
        _quarantine(
            retry,
            device,
            log_path,
            None,
            logs_dir.parent / QUARANTINE_DIR,
            gpus,
            canary_log_path,
        )
        if len(retry.quarantines) > max_quarantines:
            retry.gave_up = True
            return len(pending), 0, retry
    failed = set(retry.skipped) | set(retry.unattempted)
    recovered = 0
    for i, (b, off) in enumerate(pending):
        if i not in failed:
            b.recovered.add(off)
            recovered += 1
    return len(pending), recovered, retry


# runtime.env and, with --pool-only-library, the library directory: every
# bench process of the stage gets them.
BENCH_ENV_EXTRA: Dict[str, str] = {}
TMP_DIR = "tmp"


def _devices(args: argparse.Namespace) -> List[int]:
    return [int(d) for d in args.devices.split(",") if d.strip()]


def _make_work_dir(work_dir: Path) -> None:
    """Create `work_dir` (`<out-dir>/tmp/<pid>`) empty. The directories
    beside it of processes that are gone are removed; a live process's never
    is, so a stale attempt still running into the output directory and a new
    one never remove each other's bench yamls."""
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    for d in work_dir.parent.iterdir():
        if d != work_dir and d.name.isdigit() and kfd.proc_info(int(d.name)).gone:
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir()


def main() -> int:
    args = _parse_args()
    work_dir = args.out_dir / TMP_DIR / str(os.getpid())
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        try:
            with gpu_lease.hold(_devices(args), "stage03 bench", args.out_dir):
                return _run(args, children, work_dir)
        except gpu_lease.LeaseError as e:
            ui.err("lease", str(e))
            return 2
        finally:
            children.stop_all()
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                work_dir.parent.rmdir()
            except OSError:
                pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03 -- per-GPU hipblaslt-bench launcher (work queue, "
        "crash recovery, retry pass)."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="directory of the bench yaml (shapes.yaml), e.g. stage01's.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    ap.add_argument(
        "--startup-grace-s",
        type=float,
        default=900.0,
        help="minimum lifetime of a fresh bench process before the stall rule "
        "may kill it, until it finishes its first shape (kernel-module "
        "loading prints nothing). Must exceed the module-load time with every "
        "GPU loading at once.",
    )
    ap.add_argument(
        "--stall-timeout-s",
        type=float,
        default=300.0,
        help="kill a bench process whose log has not grown for this many "
        "seconds and treat it as a crash; 0 disables the watchdog.",
    )
    ap.add_argument(
        "--blocks-per-gpu",
        type=int,
        default=20,
        help="target blocks per GPU; block_size = ceil(n_shapes / (n_gpus * "
        "blocks_per_gpu)). Smaller blocks balance better.",
    )
    ap.add_argument("--progress-every", type=float, default=30.0)
    ap.add_argument(
        "--warmup-pass",
        dest="warmup_pass",
        action="store_true",
        default=None,
        help="prepend the discarded warm-up shape to every bench process "
        "(default: bench.warmup_pass from --config-yaml, else on).",
    )
    ap.add_argument(
        "--no-warmup-pass",
        dest="warmup_pass",
        action="store_false",
        help="disable the warm-up shape.",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config; bench.adaptive is read, and bench.warmup_pass when "
        "neither --warmup-pass nor --no-warmup-pass is given, and bench.canary "
        "for the canary options not given.",
    )
    ap.add_argument(
        "--canary",
        dest="canary",
        action="store_true",
        default=None,
        help="bench the canary before and after the shapes of every bench "
        "process (default: bench.canary.enabled from --config-yaml, else "
        f"{'on' if CANARY_DEFAULTS['enabled'] else 'off'}).",
    )
    ap.add_argument(
        "--no-canary", dest="canary", action="store_false", help="no canary."
    )
    ap.add_argument(
        "--canary-tolerance",
        type=float,
        default=None,
        help="how far above its GPU's reference a canary reading may be before "
        "its block is benched again (default: bench.canary.tolerance, else "
        f"{CANARY_DEFAULTS['tolerance']:g}).",
    )
    ap.add_argument(
        "--canary-timings",
        type=int,
        default=None,
        help="canary problems per reading (default: bench.canary.timings, "
        f"else {CANARY_DEFAULTS['timings']}).",
    )
    ap.add_argument(
        "--canary-backoff-s",
        type=float,
        default=None,
        help="seconds a block whose canary drifted waits before it is benched "
        "again (default: bench.canary.backoff_s, else "
        f"{CANARY_DEFAULTS['backoff_s']:g}).",
    )
    ap.add_argument(
        "--no-retry-skipped",
        action="store_true",
        help="skip the retry pass over skipped and unattempted shapes.",
    )
    ap.add_argument(
        "--max-retry-gemms",
        type=int,
        default=200,
        help="skip the retry pass when more shapes than this need it (a "
        "failure that large is systematic, not transient).",
    )
    ap.add_argument(
        "--max-crash-blocks",
        type=int,
        default=None,
        help="crash blocks tolerated before the stage fails (default: one "
        "block in twenty, rounded down).",
    )
    ap.add_argument(
        "--max-quarantines",
        type=int,
        default=DEFAULT_MAX_QUARANTINES,
        help="times a block (or the retry pass) whose GPU another process "
        "used is benched again before the stage fails (default "
        f"{DEFAULT_MAX_QUARANTINES}).",
    )
    ap.add_argument(
        "--pool-only-library",
        action="append",
        default=[],
        metavar="STEM",
        help="bench against a copy of --library-dir in which the other "
        "libraries of STEM's problem type are empty (repeat to keep several, "
        "e.g. the pool and a bank library)",
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library directory of the device (<build>/Tensile/library/"
        "<arch>), for --pool-only-library",
    )
    ap.add_argument(
        "--identity-dir",
        type=Path,
        default=None,
        help="Tensile library directory of the bench's build, with " "--identity-stem",
    )
    ap.add_argument(
        "--identity-stem",
        action="append",
        default=[],
        metavar="STEM",
        help="library file whose solution indices the bench lines name "
        "(repeatable); its kernel identity is recorded in the summary",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.pool_only_library and args.library_dir is None:
        ap.error("--pool-only-library needs --library-dir")
    if bool(args.identity_stem) != (args.identity_dir is not None):
        ap.error("--identity-dir and --identity-stem go together")
    return args


def library_identity(identity_dir: Optional[Path], stems: Sequence[str]) -> Any:
    """The LibraryIdentity (lib/kernel_identity.py) of the library files
    `stems` in `identity_dir`, which the bench lines name kernels of; None
    without them. FileNotFoundError / ValueError when one cannot be read."""
    if identity_dir is None or not stems:
        return None
    from lib import kernel_identity as ki

    return ki.library_identity(identity_dir, list(stems))


def bench_identity(identity: Any, indices: Sequence[int] = ()) -> Dict[str, Any]:
    """The summary entry holding the kernel identity record of `identity`
    (`library_identity`) with the kernels of `indices` listed; empty
    without an identity."""
    if identity is None:
        return {}
    from lib import kernel_identity as ki

    return {ki.RECORD_KEY: ki.record(identity, indices)}


def _pool_libdir(args: argparse.Namespace) -> Path:
    from lib.pool_libdir import build_pool_libdir

    dst = args.out_dir / "libdir_pool"
    shutil.rmtree(dst, ignore_errors=True)
    summary = build_pool_libdir(args.library_dir, args.pool_only_library, dst)
    if not args.quiet:
        ui.info(
            "libdir",
            f"pool-only library dir {dst}: {len(summary['emptied'])} other "
            f"librar(ies) of the problem type emptied",
        )
    return dst


def _run(args: argparse.Namespace, children: procs.ChildGroups, work_dir: Path) -> int:
    cfg = _read_config(args.config_yaml)
    BENCH_ENV_EXTRA.clear()
    BENCH_ENV_EXTRA.update(runtime_env(cfg))
    warmup, warmup_src = _resolve_warmup(args, cfg)
    devices = _devices(args)
    if not devices:
        ui.err("err", "--devices is empty")
        return 1
    in_yaml = args.in_dir / "shapes.yaml"
    if not in_yaml.exists():
        ui.err("err", f"{in_yaml} does not exist")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"hipblaslt-bench not found: {args.bench_binary}")
        return 1
    read = read_bench_lines(in_yaml)
    lines = [adaptive_line(line, cfg) for line in read]
    n_shapes = len(lines)
    if n_shapes == 0:
        ui.err("err", f"no shape lines in {in_yaml}")
        return 1
    try:
        library = library_identity(args.identity_dir, args.identity_stem)
    except (OSError, ValueError) as e:
        ui.err("identity", f"cannot read the library files the bench names: {e}")
        return 1
    try:
        canary, no_canary = _resolve_canary(args, cfg, lines, library)
    except ValueError as e:
        ui.err("canary", str(e))
        return 1
    identity = bench_identity(
        library, (line_solution_list(canary.line) or []) if canary else []
    )
    n_converted = sum(a != b for a, b in zip(read, lines))
    if n_converted:
        ui.warn(
            "yaml",
            f"{n_converted} line(s) of {in_yaml} have fixed iteration counts; "
            f"they are timed adaptively",
        )

    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    n_removed = _clean_previous_outputs(args.out_dir, logs_dir)
    _make_work_dir(work_dir)
    env = bench_env_with(BENCH_ENV_EXTRA)
    launch = launch_settings.bench_record(
        args.bench_binary, cfg, env, os.environ, lines
    )
    if launch["stripped"]:
        ui.warn(
            "launch",
            f"not passed to the benches: {', '.join(launch['stripped'])} (launch "
            f"overrides reach a bench only through the config's runtime.env)",
        )
    if launch["overrides"]:
        ui.warn(
            "launch",
            "the benches run with the launch overrides "
            + ", ".join(f"{k}={v}" for k, v in launch["overrides"].items()),
        )
    try:
        fingerprint.start_session(
            devices,
            args.out_dir,
            environment=recorded_env(cfg, env),
            launch=launch,
        )
    except fingerprint.PlatformError as e:
        ui.err("platform", f"refused: {e}")
        return 2
    gpus = kfd.StageGpus(devices)
    note = gpus.unwatched_note()
    if note and not args.quiet:
        ui.warn("quiet", note)
    if args.pool_only_library:
        try:
            libdir = _pool_libdir(args)
        except (OSError, ValueError) as e:
            ui.err("libdir", str(e))
            return 1
        BENCH_ENV_EXTRA[LIBPATH_VAR] = str(libdir)

    n_gpus = len(devices)
    target_blocks = max(n_gpus, n_gpus * int(args.blocks_per_gpu))
    block_size = max(1, math.ceil(n_shapes / target_blocks))
    blocks = _build_blocks(lines, block_size)
    tolerance = (
        int(args.max_crash_blocks)
        if args.max_crash_blocks is not None
        else int(0.05 * len(blocks))
    )

    if not args.quiet:
        ui.banner(f"stage03 bench launcher  ({n_gpus} devices)")
        ui.info("cfg", f"yaml         : {in_yaml}")
        ui.info("cfg", f"devices      : {devices}")
        ui.info("cfg", f"bench_binary : {args.bench_binary}")
        ui.info("cfg", f"out_dir      : {args.out_dir}")
        ui.info("cfg", f"n_shapes     : {n_shapes:,}")
        ui.info("cfg", f"block_size   : {block_size}  (n_blocks {len(blocks):,})")
        ui.info("cfg", f"crash blocks : tolerated {tolerance}")
        ui.info(
            "cfg",
            f"warmup_pass  : {warmup}  ({warmup_src})"
            + (
                ""
                if warmup
                else "  -- the first shape of every process is skip-screened "
                "on module-load times"
            ),
        )
        ui.info(
            "cfg",
            "canary       : "
            + (canary.describe() if canary is not None else f"none ({no_canary})"),
        )
        if n_removed:
            ui.info("cfg", f"removed {n_removed} output file(s) of a previous run")

    work = _Work(blocks, args.max_quarantines)
    quarantine_dir = args.out_dir / QUARANTINE_DIR
    if not args.quiet:
        ui.banner("Bench launch (work queue + crash recovery)", ui.C.BLUE)
    workers = [
        _DeviceWorker(
            dev,
            work,
            args.bench_binary,
            logs_dir,
            warmup=warmup,
            stall_s=float(args.stall_timeout_s),
            startup_grace_s=float(args.startup_grace_s),
            quiet=args.quiet,
            children=children,
            work_dir=work_dir,
            gpus=gpus,
            quarantine_dir=quarantine_dir,
            canary=canary,
        )
        for dev in devices
    ]
    t0 = time.time()
    for w in workers:
        w.start()
    last_print = 0.0
    while any(w.is_alive() for w in workers):
        time.sleep(0.5)
        if not args.quiet and time.time() - last_print >= args.progress_every:
            with work.lock:
                n_done = len(work.done)
                n_skip = sum(len(b.skipped) for b in work.done)
            alive = [w.device for w in workers if w.is_alive()]
            ui.grey(
                "bench",
                f"blocks_done={n_done:>4d}/{len(blocks):<4d}  alive_gpus={alive}  "
                f"skipped_gemms={n_skip}  elapsed={ui.fmt_dur(time.time() - t0)}",
            )
            last_print = time.time()
    for w in workers:
        w.join()
    done = list(work.done)
    done_ids = {id(b) for b in done}
    unbenched = [b for b in blocks if id(b) not in done_ids]
    for b in unbenched:
        b.abandoned = True
        done.append(b)
    done.sort(key=lambda b: b.block_id)
    elapsed = time.time() - t0

    n_skipped = sum(len(b.skipped) for b in done)
    n_unattempted = sum(len(b.unattempted) for b in done)
    n_pending = n_skipped + n_unattempted
    retried = recovered = 0
    retry_block: Optional[TaskBlock] = None
    if n_pending and args.no_retry_skipped:
        ui.warn("retry", f"retry pass disabled; {n_pending} GEMM(s) stay without data")
    elif n_pending > args.max_retry_gemms:
        ui.warn(
            "retry",
            f"{n_pending} GEMM(s) need a retry (> --max-retry-gemms "
            f"{args.max_retry_gemms}); skipping the retry pass",
        )
    elif n_pending:
        if not args.quiet:
            ui.banner(
                f"Retry pass: re-benching {n_pending} GEMM(s) on device "
                f"{gpus.first_active(devices)}",
                ui.C.BLUE,
            )
        retried, recovered, retry_block = _retry_pass(
            done,
            devices,
            args.bench_binary,
            logs_dir,
            float(args.stall_timeout_s),
            float(args.startup_grace_s),
            children=children,
            work_dir=work_dir,
            gpus=gpus,
            max_quarantines=args.max_quarantines,
            canary=canary,
        )
        if not args.quiet:
            ui.ok("retry", f"recovered {recovered}/{retried} GEMM(s)")
    if canary is not None:
        _flag_at_end(
            done + ([retry_block] if retry_block is not None else []),
            canary,
            logs_dir,
        )

    unbenched_ids = {id(b) for b in unbenched}
    lost: List[Tuple[TaskBlock, int, str]] = []
    for b in done:
        if b.gave_up:
            lost.extend((b, off, "quarantined") for off in range(len(b.lines)))
            continue
        for off in sorted(b.skipped):
            if off not in b.recovered:
                lost.append((b, off, "crashed"))
        left = (
            "quarantined" if id(b) in unbenched_ids and b.quarantines else "unattempted"
        )
        for off in b.unattempted:
            if off not in b.recovered:
                lost.append((b, off, left))
    crash_blocks = [
        b
        for b in done
        if not b.gave_up and any(off not in b.recovered for off in b.unattempted)
    ]
    rc = 0 if len(crash_blocks) <= tolerance else 2
    attempted = done + ([retry_block] if retry_block is not None else [])
    quarantined = [q for b in attempted for q in b.quarantines]
    problems: List[str] = []
    gave_up = [b.name for b in attempted if b.gave_up]
    if gave_up:
        problems.append(
            f"{len(gave_up)} block(s) quarantined more than --max-quarantines "
            f"{args.max_quarantines} time(s): {', '.join(gave_up)}"
        )
    numbering = [q for q in quarantined if q.get("rule") == "gpu-numbering"]
    if numbering:
        problems.append(numbering[0]["reason"])
    if unbenched and gpus.retired:
        why = "; ".join(f"gpu{d}: {w}" for d, w in sorted(gpus.retired.items()))
        problems.append(
            f"{len(unbenched)} block(s) left unbenched: every GPU stopped "
            f"taking blocks ({why})"
        )
    misaligned = [f"{b.name}: {m}" for b in attempted for m in b.misaligned]
    if misaligned:
        problems.append(
            f"{len(misaligned)} bench process(es) printed output that does not "
            f"line up with their entries, e.g. {misaligned[0]}"
        )
    if problems:
        rc = 2

    if lost:
        skip_path = args.out_dir / "skipped_gemms.txt"
        rows = ["yaml_line,M,N,K,B,rc,reason\n"]
        for b, off, reason in sorted(lost, key=lambda t: t[0].line_numbers[t[1]]):
            shape = shape_of_line(b.lines[off]) or ("", "", "", "")
            rows.append(
                f"{b.line_numbers[off]},{shape[0]},{shape[1]},{shape[2]},"
                f"{shape[3]},{b.skipped.get(off, '')},{reason}\n"
            )
        atomic_write_text(skip_path, "".join(rows))
        ui.warn("skip", f"{len(lost)} GEMM(s) without data -- see {skip_path}")
    write_json(
        args.out_dir / "stage03_summary.json",
        {
            "in_yaml": str(in_yaml),
            "devices": devices,
            "n_shapes": n_shapes,
            "n_blocks": len(blocks),
            "block_size": block_size,
            "warmup_pass": warmup,
            "benched": sum(b.n_benched for b in done),
            "skipped": n_skipped,
            "unattempted": n_unattempted,
            "retried": retried,
            "recovered": recovered,
            "lost": len(lost),
            "crash_blocks": [b.name for b in crash_blocks],
            "crash_block_tolerance": tolerance,
            "max_quarantines": args.max_quarantines,
            "quarantined": quarantined,
            "quarantine_failed": gave_up,
            "unbenched_blocks": [b.name for b in unbenched],
            "retired_devices": [
                {"device": d, "why": w} for d, w in sorted(gpus.retired.items())
            ],
            "returned_devices": gpus.returned,
            "unwatched_devices": gpus.unwatched(),
            "misaligned": misaligned,
            "canary": (
                canary.summary()
                if canary is not None
                else {"enabled": False, "why": no_canary}
            ),
            "problems": problems,
            "rc": rc,
            **identity,
            "blocks": [
                {
                    "name": b.name,
                    "yaml_lines": [b.line_numbers[0], b.line_numbers[-1]],
                    **b.gpu_fields,
                    "wall_s": round(b.wall_s, 3),
                    "rc": b.rc,
                    "quarantines": len(b.quarantines),
                    **({"canary": b.canary_record()} if canary is not None else {}),
                }
                for b in attempted
            ],
        },
    )
    for p in problems:
        ui.err("problem", p)

    if not args.quiet:
        ui.banner(
            "Done",
            ui.C.GREEN if rc == 0 and not lost and not quarantined else ui.C.YELLOW,
        )
        for w in workers:
            ui.ok(
                "dev ",
                f"  device {w.device}  blocks={w.n_blocks_completed:>4d}  "
                f"wall={ui.fmt_dur(w.wall_s)}  warmup={ui.fmt_dur(w.warmup_s)}",
            )
        ui.ok("done", f"total wall       : {ui.fmt_dur(elapsed)}")
        ui.ok("done", f"blocks           : {len(done)}/{len(blocks)}")
        ui.ok(
            "done",
            f"GEMMs benched    : {sum(b.n_benched for b in done):,}/{n_shapes:,} "
            f"(+{recovered} recovered by the retry pass)",
        )
        ui.ok(
            "done",
            f"GEMMs skipped    : {n_skipped} crashed, {n_unattempted} unattempted; "
            f"{len(lost)} still without data",
        )
        if quarantined:
            ui.ok(
                "done",
                f"quarantined      : {len(quarantined)} attempt(s) in {quarantine_dir}",
            )
        if canary is not None:
            c = canary.summary()
            ui.ok(
                "done",
                f"canary           : {len(c['rebenched'])} block(s) benched again "
                f"for their canary, {len(c['flagged'])} flagged, "
                f"{len(c['flagged_at_end'])} flagged at the end",
            )
    if canary is not None:
        c = canary.summary()
        if c["disabled"]:
            ui.warn("canary", f"off on every GPU: {c['disabled']}")
        for d, why in c["off"].items():
            ui.warn("canary", f"off on gpu{d}: {why}")
    if not args.quiet:
        ui.ok("done", f"logs dir         : {logs_dir}")
    if crash_blocks:
        (ui.warn if rc == 0 else ui.err)(
            "tol",
            f"{len(crash_blocks)} crash block(s) "
            f"({', '.join(b.name for b in crash_blocks)}); tolerated {tolerance}"
            + ("" if rc == 0 else " -- failing (raise --max-crash-blocks to accept)"),
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
