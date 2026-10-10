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
after it. A shape the stall watchdog killed (no output for
--stall-timeout-s) hung instead: it leaves its block for good and is
quarantined in the run under the measurement protocol --hung-scope names
(see Guard; `none` quarantines nothing); with --retry-hung (audit protocol
v2) the retry pass benches it once more first, in its own process, and it
is quarantined only when that does not bench it to the end. A block whose
processes keep dying without finishing a shape is abandoned, and its
remaining shapes are recorded as unattempted.

Retry pass: skipped and unattempted shapes, up to --max-retry-gemms of them,
and with --retry-hung the hung ones, are benched once more in one process
on the first device that still takes blocks and is ready (see Guard; a GPU
the run holds, or whose health canary failed, is passed over) (crashes
under heavy multi-GPU load often do not repeat in isolation).
--no-retry-skipped turns it off, and with it --retry-hung.

Warm-up line (--warmup-line, `bench.warmup_line`; on by default): every
bench process, the retry pass's too, starts with the stage's throwaway
warm-up line (lib/bench_yaml.py `warmup_line`): a small fixed problem of
the stage's problem type, screened and timed as its lines are, over a few of
the kernels they list, one of each library file of --identity-stem in turn.
Its output goes to the warm-up log, and platform.json and the summary
record it. A process that fails before it got past the warm-up line (the
block's first shape never started) turns it off for the rest of the stage,
and a new process takes the shapes without charging any of them.

Entry names: the bench converts its --yaml with clients/hipblaslt_gentest.py,
which runs only the first of entries equal in every argument, while the
stage maps a process's output to the entries of its yaml by position (the
warm-up's output, the crashed shape). So every entry gets a name of its own
(lib/bench_yaml.py `name_entry`: `warmup_line`, `warmup`, `canary_start<i>`
and `canary_end<i>`, and `line<N>` for line N of the input yaml). A process
whose output does not line up with its entries (a problem of another shape
than its entry, more problems than entries, or fewer from a process that
exited cleanly) is noted in its block log and fails the stage; one with the
warm-up line or the canary is benched again instead, without them (see
Canary).

GPUs: the stage leaves out the devices whose GPU the config's
runtime.exclude_gpus names, leases the GPUs it benches on
(lib/gpu_lease.py `hold_usable`; under the orchestrator it works under the
orchestrator's lease), leaves a GPU another process tree holds to it,
naming the holder, and refuses to start when every one is held; it records
the GPUs' settings in platform.json and refuses to start when they disagree
on the performance level, the power cap, the CU count, the partition mode
or the XCC count (lib/fingerprint.py). A GPU the run holds for a departure
of its own (see Guard; the stage checks the GPUs' health first) is recorded
as held and left out of that comparison: it benches once the run stops
holding it and its settings agree with the stage's other GPUs.

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

Guard (lib/host_load.py, lib/gpu_health.py; the run's record in the
directory the orchestrator names, else `<out-dir>/guard/`): a block starts
on a device only when the run does not hold the device's GPU, the stage's
cores are quiet (`bench.quiet_host`, --quiet-host), the GPU's settings
agree with the stage's other GPUs (the platform check above; a GPU is read
again once the run stops holding it, since its state may have been
accepted as the run's: scripts/run_guard.py accept) and the GPU read its
health canary in this session since the run last held it
(`bench.health.canary`, --health-canary, which follows --health); a device
that waited long for a block checks this again once it has one, and puts
the block back when it cannot start it yet. A thread checks the GPUs
against the run's baseline every `bench.health.poll_s` (--health) and
settles what it finds into the run's hold: a GPU that departs is held
alone, and the stage's other GPUs go on, while the host's problems (ttm
workers) or departures on every GPU the check reads, two or more, hold
every GPU of the run; it settles only the hold's problems of the GPUs it
reads (its own and the ones the hold names) and the host's. A block that
sees the cores turn hot, or the run hold its GPU, is stopped: its logs
move to `aborted/` with a record, and it goes back on the work queue, at
most MAX_ABORTS times. With `bench.health.max_wait_s`, a device whose GPU
stays held, or unlike the others, or whose health canary keeps failing,
for longer takes no more blocks in the stage (it is retired, as
`retired_devices` records), while a hold of every GPU or busy cores for
longer stop the stage; with --leave-held-gpus (audit protocol v2, whose
other workers take the part) a device whose GPU the run holds alone is
retired at once. Lines the run quarantined because they hung under
the stage's --hung-scope are left out of the stage (when that is every
line, the stage benches nothing and exits 1, its summary and
skipped_gemms.txt naming them), and the hold of a GPU runtime.exclude_gpus
names is dropped.

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
  logs/warmup_NNNN_gpuD.log          discarded output of the warm-up line and
                                     the warm-up shape, per block (the retry
                                     pass's is warmup_retry_gpuD.log; the
                                     prefix keeps it out of stage04's
                                     `block_*` glob)
  logs/canary_NNNN_gpuD.log          the canary's output, per block (the retry
                                     pass's is canary_retry_gpuD.log)
  drift/<log stem>_dN/               the logs of a block's attempt whose canary
                                     drifted and drift.json: its readings, its
                                     GPU's reference, the tolerance, the GPU
                                     and the canary kernel's signature
  misaligned/<log stem>_sN/          what the N-th bench process of a block
                                     wrote to its block, canary and warm-up
                                     logs when its output did not line up
                                     with its entries, and misaligned.json:
                                     why
  aborted/<log stem>_aN/             the logs of a block's N-th attempt the
                                     guard stopped and aborted.json: why
                                     (`contended` or `health`), the load on
                                     the stage's cores while it ran, the
                                     run's hold
  host_load.jsonl                    the load on the stage's cores, one
                                     window per line
  quarantine/<log stem>_qN/          the logs of a block's N-th quarantined
                                     attempt and quarantine.json: the rule, the
                                     other processes (pid, command, user, when
                                     seen, start time), the unattributed VRAM
                                     (bytes, card and processes' use, settle
                                     window), the GPU
  platform.json                      the GPUs' settings (lib/fingerprint.py;
                                     `held` on a GPU the run held when the
                                     stage started), the warm-up line (its
                                     line, shape and kernels, by library
                                     file and signature when --identity-stem
                                     names them) or why there is none, and
                                     at the end what the guard saw of the
                                     cores (`host_load`) and the GPUs
                                     (`health`; under `platform` the GPUs
                                     held at the start, the GPUs that joined
                                     the others after the start and those
                                     found unlike them)
  skipped_gemms.txt                  shapes still without data after the retry
                                     pass: yaml_line,M,N,K,B,rc,reason with
                                     reason `crashed`, `unattempted`,
                                     `quarantined`, `aborted` (stopped by the
                                     guard more than MAX_ABORTS times),
                                     `hung` (with --retry-hung: and not
                                     benched to the end by the retry pass)
                                     or `quarantined_line` (hung in an
                                     earlier stage of the run under the same
                                     --hung-scope)
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
                                     and flagged, the output set aside), the
                                     warm-up line (as platform.json records
                                     it, why it went off, the output set
                                     aside), and with --identity-stem the
                                     kernel
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
--max-quarantines times or stopped by the guard more than MAX_ABORTS times,
when the guard stopped waiting (`bench.quiet_host.max_wait_s`, or
`bench.health.max_wait_s` while the run held every GPU), when the GPU
numbering is wrong, when a bench process's output does not line up with
its entries, or when blocks are left because every GPU stopped taking
them; 2 too for a refused lease or platform check. It exits 1 without a
bench when the run quarantined every line under its --hung-scope.

Every bench process leads its own process group. SIGINT, SIGTERM or SIGHUP
stops all of them (`lib.procs`) and the stage exits with 128 + the signal
number.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
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
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import bench_yaml  # noqa: E402
from lib import fingerprint  # noqa: E402
from lib import gpu_health  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import host_load  # noqa: E402
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
    format_solution_list,
    least_work_line,
    line_has_prescreen,
    line_solution_list,
    name_entry,
    read_bench_lines,
    set_line_knob,
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
    # lines the stall watchdog killed: out of `lines` for good, quarantined
    # in the run, or first benched once more in the retry pass with
    # --retry-hung (their records, oldest first)
    hung: List[Dict[str, Any]] = field(default_factory=list)
    # yaml line -> text of the hung lines the retry pass benches once more,
    # and those of them it benched to the end
    hung_lines: Dict[int, str] = field(default_factory=dict)
    hung_recovered: Set[int] = field(default_factory=set)
    # why the stage's guard stopped the last attempt ("contended", "health")
    aborted: Optional[Dict[str, Any]] = None
    # aborted.json of every attempt the guard stopped, oldest first
    aborts: List[Dict[str, Any]] = field(default_factory=list)
    # the load on the stage's cores while the last attempt ran
    host_load: Dict[str, Any] = field(default_factory=dict)

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
        self.aborted = None
        self.misaligned = []
        self.canary_readings = []
        self.canary = {}
        self.host_load = {}

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


def _build_blocks(
    lines: List[str], block_size: int, numbers: Optional[Sequence[int]] = None
) -> List[TaskBlock]:
    """Blocks of `block_size` of `lines`, whose 1-based source-yaml line
    numbers are `numbers` (default: their positions)."""
    numbering = list(numbers) if numbers is not None else list(range(1, len(lines) + 1))
    blocks: List[TaskBlock] = []
    for bi, start in enumerate(range(0, len(lines), block_size)):
        chunk = lines[start : start + block_size]
        numbers = numbering[start : start + len(chunk)]
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
# Lines with `skip_screen_prepass` get no warm-up shape: the bench then
# launches every candidate once, untimed, before its screening warm-up,
# which loads the modules without the library heuristic. The warm-up line
# (below) leads every process whatever its lines.
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


# ── warm-up line ─────────────────────────────────────────────────────────────
#
# Every bench process starts with the stage's warm-up line (`bench.warmup_line`,
# lib/bench_yaml.py `warmup_line`), before the warm-up shape when there is
# one: one fixed line for the whole stage, built from its lines and, with
# --identity-stem, one kernel of each library file in turn, its output
# discarded to the warm-up log. A process that fails before it got past the
# line, or whose output does not line up with its entries, turns it off for
# the rest of the stage; a new process takes the shapes without it, and none
# of them is charged for it.
WARMUP_LINE_ENTRY = "warmup_line"


class WarmupLine:
    """The stage's warm-up line (`line`) over `kernels` of `identity`'s
    library (None: unknown), as `settings` (`kernels`, `size`) made it, and
    whether it is still on."""

    def __init__(
        self,
        line: str,
        kernels: Sequence[int],
        settings: Mapping[str, Any],
        identity: Any = None,
    ) -> None:
        self.line = line.rstrip("\n") + "\n"
        self.shape = shape_of_line(self.line)
        self.kernels = [int(k) for k in kernels]
        self.settings = dict(settings)
        self.identity = identity
        self.off: Optional[str] = None
        self.misaligned: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def current(self) -> Optional[str]:
        """The line a new bench process starts with; None once it is off."""
        with self._lock:
            return None if self.off is not None else self.line

    def turn_off(self, why: str) -> None:
        with self._lock:
            if self.off is not None:
                return
            self.off = why
        ui.warn(
            "warm-up",
            f"every bench process goes without the warm-up line from now: {why}",
        )

    def set_aside(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.misaligned.append(dict(record))

    def describe(self) -> str:
        m, n, k, b = self.shape or ("?", "?", "?", "?")
        what = (
            f"{len(self.kernels)} listed kernel(s)"
            if self.kernels
            else f"the first {self.settings['kernels']} kernel(s) the bench takes"
        )
        return f"M={m} N={n} K={k} B={b}, {what}, first in every bench process"

    def record(self) -> Dict[str, Any]:
        """What platform.json records of the line: its settings (`kernels`,
        `size`), the line and its shape, and per listed kernel its solution
        index, library file and kernel identity (null where the stage has no
        library identity)."""
        from lib import kernel_identity as ki

        owner: Dict[int, str] = {}
        if self.identity is not None:
            for stem in self.identity.stems:
                for sid in self.identity.file_ids.get(stem, {}):
                    owner.setdefault(int(sid), stem)
        return {
            "enabled": True,
            "settings": dict(self.settings),
            "line": self.line.rstrip("\n"),
            "shape": list(self.shape) if self.shape else None,
            "kernels": [
                {"sol_index": k, "file": owner.get(k), **ki.naming(self.identity, k)}
                for k in self.kernels
            ],
        }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.record(),
                "off": self.off,
                "misaligned": list(self.misaligned),
            }


def _resolve_warmup_line(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    lines: Sequence[str],
    identity: Any = None,
) -> Tuple[Optional[WarmupLine], str]:
    """The stage's warm-up line (CLI flag > `bench.warmup_line` of the config
    > WARMUP_LINE_DEFAULTS of lib/bench_yaml.py) from its `lines`, the first
    that lists kernels its template, its kernels one of each library file of
    `identity` in turn; or None and why there is none. ValueError for a
    kernel count or a size below 1."""
    st = bench_yaml.warmup_line_settings(cfg)
    on = args.warmup_line if args.warmup_line is not None else st["enabled"]
    if not on:
        return None, "turned off"
    count, size = int(st["kernels"]), int(st["size"])
    if count < 1 or size < 1:
        raise ValueError(
            f"the warm-up line needs at least one kernel and a size of at least 1 "
            f"(got {count} and {size})"
        )
    files = None
    if identity is not None:
        files = [identity.file_ids.get(stem, {}) for stem in identity.stems]
    kernels = bench_yaml.warmup_kernels(lines, count, files)
    template = next((ln for ln in lines if line_solution_list(ln)), lines[0])
    line = bench_yaml.warmup_line(template, kernels, count=count, size=size, config=cfg)
    return WarmupLine(line, kernels, {"kernels": count, "size": size}, identity), ""


def _warmup_record(
    warm: Optional[WarmupLine], why: str, outcome: bool = False
) -> Dict[str, Any]:
    """platform.json's record of the stage's warm-up line, or why there is
    none; with `outcome`, the summary's, which adds why it went off and the
    output set aside."""
    if warm is None:
        return {"enabled": False, "why": why}
    return warm.summary() if outcome else warm.record()


def _died_in_warmup_line(
    warm_problems: Sequence[Problem], later: Sequence[Sequence[Problem]]
) -> bool:
    """A failed process with the warm-up line first never got past it: its
    warm-up log holds no problem, or one that did not finish, and the logs of
    the entries after it (`later`) hold none. The block's first shape never
    started, so it is not to blame."""
    if any(later):
        return False
    return not warm_problems or (
        len(warm_problems) == 1 and not warm_problems[0].finished
    )


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
    warmup_shape: Optional[str],
    canary_lines: Sequence[str],
    remaining: Sequence[str],
    numbers: Sequence[int],
    lead: Optional[str] = None,
) -> List[str]:
    """The yaml entries of one bench process, each named apart (see the
    module docstring): the warm-up line `lead`, the warm-up shape, the canary
    lines, the block's `remaining` lines (source-yaml lines `numbers`) and
    the canary lines again."""
    out = [name_entry(lead, WARMUP_LINE_ENTRY)] if lead else []
    out += [name_entry(warmup_shape, "warmup")] if warmup_shape else []
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
    guard: Optional["_Guard"] = None,
    warmup_line: Optional[WarmupLine] = None,
) -> None:
    """Bench `block` on `device`, appending all output to `log_path`. After a
    crash the shape after the last finished one is skipped and a new process
    resumes behind it (see the module docstring); a shape the stall
    watchdog killed leaves the block and, with `guard`, is quarantined in
    the run (`_Guard.quarantine_line`). With `guard`, a process its abort
    probe stops ends the attempt with `block.aborted` set. With `warmup_line`
    (and `warmup_log_path`), every process starts with the stage's warm-up
    line, whose output goes to `warmup_log_path`; a process that fails before
    it got past it turns it off and leaves the shapes to a new process. With
    `warmup`, every process gets the throwaway warm-up shape after it, its
    output split into `warmup_log_path` too; `on_warmup(subrun, seconds)`
    reports what both cost. With `canary`, every process benches the canary
    lines before the shapes (after the warm-ups) and after them, into
    `canary_log_path`, and its readings join `block.canary_readings`. The
    output of a process with the warm-up line or the canary that does not
    line up with its entries moves to `aside_dir` (default: `misaligned/`
    beside the log) and they go off.
    Processes start through `children` with their yaml in `work_dir`
    (default: the log's directory). With `watch`, a process that finds
    another process on the GPU ends the attempt with `block.contaminated`
    set."""
    children = children if children is not None else procs.ChildGroups()
    work_dir = work_dir if work_dir is not None else log_path.parent
    poll_s = procs.poll_interval(stall_s)
    if watch is not None or guard is not None:
        poll_s = min(poll_s, kfd.WATCH_POLL_S)
    attempt = guard.attempt(device) if guard is not None else None
    probe_abort = attempt.probe if attempt is not None else None
    block.device = device
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(block.lines)
    if warmup and (warmup_log_path is None or n_total < 2):
        warmup = False
    if warmup_log_path is None:
        warmup_line = None
    subrun = 0
    zero_progress = 0
    last_rc = 0
    t0 = time.time()
    while block.next_offset < n_total:
        remaining = block.lines[block.next_offset :]
        numbers = block.line_numbers[block.next_offset :]
        lead = warmup_line.current() if warmup_line is not None else None
        warmup_shape = (
            render_warmup_line(remaining[0])
            if (warmup and len(remaining) >= 2)
            else None
        )
        n_warm = (1 if lead else 0) + (1 if warmup_shape else 0)
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
            f.writelines(_entries(warmup_shape, canary_lines, remaining, numbers, lead))
        sub_start = file_size(log_path)
        warmup_start = file_size(warmup_log_path)
        canary_start = file_size(canary_log_path)
        progress = LogProbe(log_path, sub_start)
        bench_device, env = kfd.one_gpu_launch(device, bench_env_with(BENCH_ENV_EXTRA))
        cmd = [str(bench_binary), "--device", bench_device, "--yaml", str(tmp_yaml)]
        reason: Optional[str] = None
        ended: Optional[float] = None
        try:
            if n_warm or canary_lines:
                probes = [progress]
                with contextlib.ExitStack() as files:
                    lf = files.enter_context(log_path.open("a", buffering=1))
                    wf = cf = None
                    if n_warm and warmup_log_path is not None:
                        probes.append(LogProbe(warmup_log_path, warmup_start))
                        wf = files.enter_context(warmup_log_path.open("a", buffering=1))
                        what = [
                            name
                            for name, on in (
                                ("the warm-up line", lead),
                                ("the warm-up shape", warmup_shape),
                            )
                            if on
                        ]
                        wf.write(
                            f"*** WARM-UP (discarded) sub-run #{subrun + 1}, "
                            f"{block.name}, gpu {device}: {' and '.join(what)} ***\n"
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
                                warmups=n_warm if wf is not None else 0,
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
                        probe_abort=probe_abort,
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
                        probe_abort=probe_abort,
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
        if reason in GUARD_ABORTS:
            block.aborted = {"reason": reason, "subrun": subrun + 1}
            _annotate(
                log_path, f"\n*** ABORTED [{reason}]: {GUARD_ABORTS[reason]} ***\n"
            )
            break
        text = _read_from(log_path, sub_start)
        shape_problems = list(iter_problems(text.splitlines()))
        ran_through = last_rc == 0 and reason is None
        off: Optional[str] = None
        warm_problems: List[Problem] = []
        if lead and warmup_line is not None and warmup_log_path is not None:
            warm_problems = list(
                iter_problems(_read_from(warmup_log_path, warmup_start).splitlines())
            )
            warm_shapes = [warmup_line.shape]
            if warmup_shape:
                warm_shapes.append(shape_of_line(warmup_shape))
            why = _misalignment(warm_problems, warm_shapes, ran_through)
            off = f"in the warm-up log, {why}" if why else None
        off = off or _misalignment(
            shape_problems, [shape_of_line(ln) for ln in remaining], ran_through
        )
        has_canary = canary is not None and bool(canary_lines)
        canary_problems: List[Problem] = []
        if canary is not None and has_canary and canary_log_path is not None:
            canary_problems = list(
                iter_problems(_read_from(canary_log_path, canary_start).splitlines())
            )
            off = off or _misalignment(
                canary_problems,
                [canary.shape] * (2 * len(canary_lines)),
                ran_through,
                single=True,
            )
        if off is not None and (has_canary or lead):
            parts: List[Tuple[Optional[Path], int]] = [(log_path, sub_start)]
            if has_canary:
                parts.append((canary_log_path, canary_start))
            if lead:
                parts.append((warmup_log_path, warmup_start))
            _misaligned_aside(
                block,
                device,
                subrun + 1,
                off,
                parts,
                aside_dir or log_path.parent / MISALIGNED_DIR,
                canary=canary if has_canary else None,
                warmup_line=warmup_line if lead else None,
            )
            subrun += 1
            continue
        if (
            lead
            and warmup_line is not None
            and not ran_through
            and _died_in_warmup_line(warm_problems, [shape_problems, canary_problems])
        ):
            how = f"rc={last_rc}" + (f", {reason}" if reason else "")
            warmup_line.turn_off(
                f"a bench process of {block.name} on gpu{device} failed in it ({how})"
            )
            _annotate(
                log_path,
                f"\n*** WARM-UP LINE FAILED ({how}): a new process takes the shapes "
                f"without it ***\n",
            )
            subrun += 1
            continue
        if canary is not None and has_canary and canary_log_path is not None:
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
        if reason == "stall":
            open_problem = (
                shape_problems[finished] if finished < len(shape_problems) else None
            )
            rec = _hung_line(block, crash_idx, open_problem, stall_s, guard)
            text = block.lines[crash_idx]
            del block.lines[crash_idx], block.line_numbers[crash_idx]
            n_total = len(block.lines)
            block.hung.append(rec)
            again = guard is not None and guard.retry_hung and block.block_id >= 0
            if again:
                block.hung_lines[int(rec["yaml_line"])] = text
                head, fate = "HUNG LINE", "benched once more in the retry pass"
            elif guard is not None and not guard.quarantines:
                head, fate = "HUNG LINE", "not benched again in this stage"
            else:
                head, fate = "QUARANTINED HUNG LINE", "never benched again in this run"
            _annotate(
                log_path,
                f"\n*** {head} at block-offset {crash_idx} (yaml line "
                f"{rec['yaml_line']}): M={m} N={n} K={k} B={b}, no output for "
                f"{stall_s:.0f}s"
                + (f" in {rec['in_flight']}" if rec.get("in_flight") else "")
                + f"; {fate} ***\n",
            )
            if guard is not None:
                (guard.defer_hung if again else guard.quarantine_line)(rec, device)
            block.next_offset = crash_idx
        else:
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
    if attempt is not None:
        block.host_load = attempt.load()


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


def _restamped(line: str, index: int) -> str:
    """Canary line `line` timing solution index `index` instead."""
    return set_line_knob(line, "solution_index_list", format_solution_list([index]))


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


def _misaligned_aside(
    block: TaskBlock,
    device: int,
    subrun: int,
    why: str,
    parts: Sequence[Tuple[Optional[Path], int]],
    aside_dir: Path,
    canary: Optional[Canary] = None,
    warmup_line: Optional[WarmupLine] = None,
) -> None:
    """A bench process of `block` with the canary or the warm-up line
    printed output that does not line up with its entries (`why`), so some
    of it may sit in the wrong log: what it wrote to its logs (`parts`, the
    block log first) moves to `aside_dir/<block log stem>_s<subrun>/` with
    misaligned.json, and `canary` goes off on every GPU and `warmup_line`
    for the rest of the stage. A new process takes the shapes."""
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
    culprit = (
        f"a bench process of {block.name} on gpu{device} printed output that "
        f"does not line up with its entries ({why})"
    )
    without = []
    if canary is not None:
        canary.set_aside({**rec, "dir": str(dest)})
        canary.disable(culprit)
        without.append("the canary")
    if warmup_line is not None:
        warmup_line.set_aside({**rec, "dir": str(dest)})
        warmup_line.turn_off(culprit)
        without.append("the warm-up line")
    tag = "CANARY" if canary is not None else "WARM-UP LINE"
    _annotate(
        log_path,
        f"\n*** {tag} MISALIGNED: {why}; the process's output moved to {dest}, "
        f"a new process takes the shapes without {' and '.join(without)} ***\n",
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


# ── the run's guard: quiet cores, GPU health, hung lines ─────────────────────

# Why the guard stops a running attempt, as its block log says.
GUARD_ABORTS = {
    "contended": (
        "other load on the stage's cores; the block is benched again once they "
        "are quiet"
    ),
    "health": (
        "the run holds the work of the block's GPU; the block is benched again "
        "on a GPU the run does not hold, once it reads its canary"
    ),
}
ABORTED_DIR = "aborted"
ABORTED_RECORD = "aborted.json"
MAX_ABORTS = 10
HOLD_READ_S = 2.0
STOPPING = "the stage is stopping"


def _next_listed(listed: Sequence[int], problem: Optional[Problem]) -> int:
    """The position in `listed` after the last kernel the open `problem`
    reported, following the solution indices its rows (rank order) give;
    the rank after its last row where they give none. A listed kernel the
    library does not support has no rank, so ranks alone can point too
    early in the list."""
    rows = []
    if problem is not None:
        rows = [*problem.tested, *problem.skipped]
        if problem.sacrificial is not None:
            rows.append(problem.sacrificial)
    rows.sort(key=lambda r: r.rank)
    if not rows:
        return 0
    if any(r.sol_index is None for r in rows):
        return rows[-1].rank + 1
    at = 0
    for r in rows:
        try:
            at = listed.index(int(r.sol_index), at) + 1
        except ValueError:
            return rows[-1].rank + 1
    return at


def _in_flight(
    line: str, problem: Optional[Problem], guard: Optional["_Guard"]
) -> Dict[str, Any]:
    """Where a line hung: the kernel after the last one its open `problem`
    reported, named by signature when the stage knows the library."""
    ranks = sorted(problem.ranks) if problem is not None else []
    rank = ranks[-1] + 1 if ranks else 0
    if not ranks and line_has_prescreen(line):
        return {
            "in_flight": "the line's untimed launch of its kernels or its first "
            "timing",
            "kernel": None,
        }
    listed = line_solution_list(line) or []
    at = _next_listed(listed, problem)
    label = None
    if at < len(listed) and guard is not None:
        label = guard.kernel_label(listed[at])
    return {
        "in_flight": f"kernel {label}" if label else f"its kernel of rank {rank}",
        "kernel": label,
        "rank": rank,
    }


def _hung_line(
    block: TaskBlock,
    offset: int,
    problem: Optional[Problem],
    stall_s: float,
    guard: Optional["_Guard"],
) -> Dict[str, Any]:
    """The record of the line at `offset` of `block` that the stall
    watchdog killed."""
    line = block.lines[offset]
    shape = shape_of_line(line)
    return {
        "shape": list(shape) if shape is not None else None,
        "yaml_line": block.line_numbers[offset],
        "block": block.name,
        "stall_s": stall_s,
        "line_sha256": hashlib.sha256(line.strip().encode()).hexdigest()[:16],
        **_in_flight(line, problem, guard),
    }


class _Attempt:
    """One attempt of a block on `device` under the guard: its abort probe
    and, when it ends, the load on the stage's cores while it ran."""

    def __init__(self, guard: "_Guard", device: int) -> None:
        self.guard = guard
        self.device = device
        self.mark = guard.monitor.mark() if guard.monitor is not None else 0

    def probe(self) -> Optional[str]:
        g = self.guard
        if g.hold_on(self.device) is not None:
            return "health"
        if (
            g.monitor is not None
            and g.monitor.hot_since(self.mark) >= g.quiet.abort_samples
        ):
            return "contended"
        return None

    def load(self) -> Dict[str, Any]:
        m = self.guard.monitor
        return m.window(self.mark) if m is not None else {}


class _Guard:
    """The run's safeguards around this stage's blocks (lib/host_load.py,
    lib/gpu_health.py; the run's record in `state`). A block starts on a
    device when the run does not hold the device's GPU (`hold_on`: a hold
    of every GPU, or of that one), the stage's cores are quiet, the GPU's
    settings agree with the stage's other GPUs (`_unlike`: the session's
    platform check leaves out the GPUs the run holds, `held_devices`, and
    a GPU is read again once the run stops holding it) and the GPU passed
    its canary in this session since the run last held it: `ready` waits
    for that, `blocked` says without waiting what stands in the way, which
    a worker asks again once it holds a block. A running block is aborted
    when the run holds its GPU or its cores turn hot (`_Attempt.probe`); a
    line the stall watchdog kills is quarantined in the run
    (`quarantine_line`). A thread checks the GPUs' health every
    `health.poll_s` and settles what it finds into the run's hold. With
    `health.max_wait_s`, a device whose own GPU stays held, unlike the
    others, or whose canary keeps failing, longer than that takes no more
    blocks in this stage (`_leave_out`) while the others go on (with
    `leave_held`, one whose GPU the run holds alone at once); a hold of
    every GPU, or cores that stay busy, that long stop the stage (`gave_up`
    says why)."""

    def __init__(
        self,
        *,
        quiet: host_load.Settings,
        health: gpu_health.Settings,
        canary: bool,
        state: gpu_health.Guard,
        gpus: kfd.StageGpus,
        out_dir: Path,
        bench_binary: Path,
        children: procs.ChildGroups,
        work_dir: Path,
        stall_s: float,
        startup_grace_s: float,
        lines: Sequence[str],
        identity: Any = None,
        cores: Optional[Sequence[int]] = None,
        clock: Callable[[], float] = time.monotonic,
        hung_scope: str = gpu_health.BENCH_SCOPE,
        retry_hung: bool = False,
        leave_held: bool = False,
    ) -> None:
        from lib.contexts import wrapped_bench

        self.quiet = quiet
        self.leave_held = bool(leave_held)
        self.health = health
        self.canary_on = bool(canary)
        self.state = state
        self.gpus = gpus
        self.out_dir = out_dir
        self.bench = wrapped_bench(bench_binary)
        self.children = children
        self.work_dir = work_dir
        self.stall_s = float(stall_s)
        self.startup_grace_s = float(startup_grace_s)
        self.lines = list(lines)
        self.identity = identity
        self.clock = clock
        self.hung_scope = str(hung_scope)
        self.retry_hung = bool(retry_hung)
        self.monitor: Optional[host_load.Monitor] = None
        if quiet.enabled:
            self.monitor = host_load.Monitor(
                cores if cores is not None else host_load.stage_cores(),
                os.getpid(),
                quiet,
                run_root=host_load.run_root(),
                marker=Path(__file__).name,
                series_path=out_dir / host_load.SERIES_FILE,
                report=lambda m: ui.warn("quiet", m),
            )
        self.health_gpus = [g for g in gpus.gpu.values() if g is not None]
        self.gave_up: Optional[str] = None
        self.canary_off: Optional[str] = None
        self.canaries: List[Dict[str, Any]] = []
        self.hung: List[Dict[str, Any]] = []
        self.holds: List[Dict[str, Any]] = []
        self.skipped_lines: List[Dict[str, Any]] = []
        # The session's platform (`admit`, `_unlike`): the devices whose
        # settings agree with the session's, each with the last time the run
        # stopped holding its GPU before they were read (`released_at`);
        # those the platform check left out, held; their records.
        self.held_at_start: Dict[int, str] = {}
        self.joined: List[Dict[str, Any]] = []
        self.unlike: List[Dict[str, Any]] = []
        self._admitted: Dict[int, Tuple[str, Dict[str, Any]]] = {}
        self._released_seen: Dict[int, str] = {}
        self._unlike_noted: Dict[int, Tuple[str, float]] = {}
        self._checked = False
        self._next_canary: Dict[int, float] = {}
        self._held: Tuple[float, Optional[Dict[str, Any]]] = (-HOLD_READ_S, None)
        self._noted = -gpu_health.REPORT_S
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._watch_health, name="gpu-health", daemon=True
        )

    # ── life ──

    def start(self) -> "_Guard":
        if self.monitor is not None:
            self.monitor.start()
        if self.health.enabled:
            self.check_now()
            self._thread.start()
        return self

    def check_now(self) -> None:
        """The stage's first health check (while the checks are on), made
        before its platform check, so that a GPU departing now is held, and
        left out of that check (`held_devices`), rather than refusing the
        session."""
        if self.health.enabled and not self._checked:
            self._checked = True
            self._check_health()

    def stop(self) -> None:
        self._stop.set()
        if self.monitor is not None:
            self.monitor.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=60.0)

    def attempt(self, device: int) -> _Attempt:
        return _Attempt(self, device)

    # ── health and hold ──

    def _check_health(self) -> None:
        try:
            what, hold = self.state.settle(
                gpu_health.check(self.state, self.health_gpus, host=self.health.ttm)
            )
        except Exception as e:
            ui.warn("health", f"could not check the GPUs: {e!r}")
            return
        now = self.clock()
        if what == "cleared":
            assert hold is not None
            with self._lock:
                self.holds.append(
                    {"cleared": hold["cleared"], "raised": hold["raised"]}
                )
            ui.ok(
                "health",
                f"the GPUs pass their checks again (held since {hold['raised']}); "
                f"each reads its canary before it benches again",
            )
        elif hold is not None:
            with self._lock:
                new = what in ("raised", "added")
                due = new or now - self._noted >= gpu_health.REPORT_S
                if due:
                    self._noted = now
                if new:
                    self.holds.append(
                        {
                            what: hold["raised" if what == "raised" else "seen"],
                            "problems": list(hold.get("problems") or []),
                            "every_gpu": gpu_health.run_wide(hold),
                            "gpus": gpu_health.held_keys(hold),
                        }
                    )
            if due:
                say = ui.err if gpu_health.run_wide(hold) else ui.warn
                say("health", gpu_health.hold_text(hold, self.state.path))
        with self._lock:
            self._held = (-HOLD_READ_S, None)

    def _watch_health(self) -> None:
        while not self._stop.wait(max(1.0, self.health.poll_s)):
            self._check_health()

    def held(self) -> Optional[Dict[str, Any]]:
        """The run's hold, read at most every HOLD_READ_S."""
        now = self.clock()
        with self._lock:
            t, h = self._held
            if now - t < HOLD_READ_S:
                return h
        h = self.state.hold()
        with self._lock:
            self._held = (now, h)
        return h

    def hold_on(self, device: int) -> Optional[Dict[str, Any]]:
        """The run's hold when it holds `device`'s GPU (`gpu_health.holds`:
        a hold of every GPU, or one of that GPU; any hold holds a device
        whose GPU the stage cannot name), while the health checks are on."""
        if not self.health.enabled:
            return None
        hold = self.held()
        gpu = self.gpus.gpu.get(device)
        key = gpu_health.gpu_key(gpu) if gpu is not None else None
        return hold if gpu_health.holds(hold, key) else None

    # ── the session's platform ──

    def held_devices(self) -> Dict[int, str]:
        """{device: its GPU's problems} of the devices whose GPU the run holds
        for a departure of its own (an entry of that GPU in the hold), while
        the health checks are on: the session's platform check leaves them
        out (fingerprint.start_session `held`), since their settings are
        not the ones they bench under."""
        if not self.health.enabled:
            return {}
        doc = self.state.read()
        hold = doc.get("hold") if isinstance(doc.get("hold"), dict) else None
        entries = gpu_health.hold_entries(hold)
        out: Dict[int, str] = {}
        for d, g in self.gpus.gpu.items():
            if g is None:
                continue
            key = gpu_health.gpu_key(g)
            with self._lock:
                self._released_seen[d] = self.state.released_at(key, doc)
            e = entries.get(key)
            if e is not None:
                out[d] = "; ".join(str(p) for p in e.get("problems") or []) or "held"
        return out

    def admit(self, session: Mapping[str, Any]) -> None:
        """Take the session's platform record (fingerprint.start_session): the
        devices it compared agree on fingerprint.MUST_MATCH and may bench; one
        it left out as held is compared before it does (`_unlike`)."""
        with self._lock:
            for row in session.get("gpus") or []:
                if row.get("device") is None:
                    continue
                d = int(row["device"])
                if "held" in row:
                    self.held_at_start[d] = str(row["held"])
                else:
                    self._admitted[d] = (self._released_seen.get(d, ""), dict(row))

    def _unlike(self, device: int) -> Optional[str]:
        """What keeps `device` from benching beside the stage's other GPUs: the
        settings of fingerprint.MUST_MATCH it does not share with those that
        may bench and the run does not hold (None: it shares them, or the
        health checks are off). A device the session's platform check left
        out, or whose GPU the run stopped holding since it last agreed
        (`gpu_health.Guard.released_at`: the GPU came back, or its state was
        accepted as the run's), is read again, and benches once it agrees."""
        if not self.health.enabled:
            return None
        gpu = self.gpus.gpu.get(device)
        key = gpu_health.gpu_key(gpu) if gpu is not None else None
        with self._lock:
            got = self._admitted.get(device)
        seen = self.state.released_at(key) if key is not None else ""
        if got is not None and seen <= got[0]:
            return None
        row = fingerprint.gpu_settings(device, self.gpus.physical.get(device), gpu)
        with self._lock:
            admitted = dict(self._admitted)
        others = [
            r for d, (_, r) in admitted.items() if d != device and not self.hold_on(d)
        ]
        bad = fingerprint.mismatches(
            [*others, row], who=f"gpu{device} and the stage's other GPUs"
        )
        if bad:
            why = "; ".join(bad)
            with self._lock:
                self._admitted.pop(device, None)
            self._note_unlike(device, key, why)
            return why
        with self._lock:
            first = device in self.held_at_start and not any(
                j["device"] == device for j in self.joined
            )
            was = self._unlike_noted.pop(device, None)
            self._admitted[device] = (seen, row)
            self.joined.append({"device": device, "gpu": key, "at": kfd.utc_iso()})
        if first or was is not None:
            ui.ok(
                "health",
                f"gpu{device} ({key}) agrees with the stage's other GPUs on "
                f"{', '.join(label for _, label in fingerprint.MUST_MATCH)}: it "
                f"benches",
            )
        return None

    def _note_unlike(self, device: int, key: Optional[str], why: str) -> None:
        now = self.clock()
        with self._lock:
            last = self._unlike_noted.get(device)
            new = last is None or last[0] != why
            due = new or now - last[1] >= gpu_health.REPORT_S
            if due:
                self._unlike_noted[device] = (why, now)
            if new:
                self.unlike.append(
                    {"device": device, "gpu": key, "at": kfd.utc_iso(), "why": why}
                )
        if due:
            ui.warn(
                "health",
                f"gpu{device} ({key}) takes no block: {why}. It benches once it "
                f"agrees with them again; restore its settings, or leave it out "
                f"with runtime.exclude_gpus",
            )

    # ── before a block ──

    def ready(
        self,
        device: int,
        stop: Callable[[], bool],
        *,
        wait: bool = True,
        since: Optional[float] = None,
    ) -> str:
        """Wait until `device` may bench a block: the run does not hold its
        GPU (`hold_on`), the stage's cores are quiet, its settings agree with
        the stage's other GPUs (`_unlike`) and the GPU passed its canary.
        "go"; "again" when the device was retired meanwhile, or left out
        because its GPU stayed held, unlike the others, or its canary kept
        failing for longer than bench.health.max_wait_s (`_leave_out`; with
        `leave_held`, at once while the run holds its GPU alone);
        without `wait`, "later" while the run holds that GPU alone, it is
        unlike the others, or its canary failed and is not due again; "stop"
        when `stop()` turns true or the stage stops waiting (`gave_up`: a
        hold of every GPU, or busy cores, for longer than their limits). The
        waits count from `since`, default now."""
        t0 = self.clock() if since is None else since
        limit = self.health.max_wait_s

        def overdue() -> bool:
            return limit > 0 and self.clock() - t0 >= limit

        while True:
            if stop() or self.gave_up is not None:
                return "stop"
            hold = self.hold_on(device)
            if hold is not None:
                if gpu_health.run_wide(hold):
                    if overdue():
                        self._give_up(
                            f"the run held the work of every GPU for longer than "
                            f"bench.health.max_wait_s {limit:g} s"
                        )
                        return "stop"
                else:
                    problems = "; ".join(hold.get("problems") or [])
                    if self.leave_held:
                        self._leave_out(
                            device,
                            f"the run holds its GPU ({problems}), and "
                            f"--leave-held-gpus leaves it out at once",
                        )
                        return "again"
                    if overdue():
                        self._leave_out(
                            device,
                            f"the run held its GPU for longer than "
                            f"bench.health.max_wait_s {limit:g} s ({problems})",
                        )
                        return "again"
                    if not wait:
                        return "later"
                time.sleep(0.5)
                continue
            if self.monitor is not None and not self.monitor.wait_quiet(
                lambda: stop() or self.hold_on(device) is not None
            ):
                if self.monitor.gave_up:
                    self._give_up(
                        f"the stage's cores were not quiet for "
                        f"bench.quiet_host.max_wait_s {self.quiet.max_wait_s:g} s"
                    )
                    return "stop"
                continue
            unlike = self._unlike(device)
            if unlike is not None:
                if overdue():
                    self._leave_out(
                        device,
                        f"{unlike}, for longer than bench.health.max_wait_s "
                        f"{limit:g} s",
                    )
                    return "again"
                if not wait:
                    return "later"
                time.sleep(HOLD_READ_S)
                continue
            due = self._canary_due(device)
            if due is None:
                return "go"
            if due > self.clock():
                if overdue():
                    self._leave_out(
                        device,
                        f"failed its canary for longer than "
                        f"bench.health.max_wait_s {limit:g} s",
                    )
                    return "again"
                if not wait:
                    return "later"
                time.sleep(0.5)
                continue
            busy = self.gpus.busy(device)
            if busy:
                _retire(
                    self.gpus,
                    device,
                    "; ".join(f.describe() for f in busy) + " on it",
                    temporary=True,
                )
                return "again"
            if self._run_canary(device) is None:
                return "stop"

    def blocked(self, device: int) -> Optional[str]:
        """What keeps `device` from starting a block now, without waiting
        (None: nothing); the conditions `ready` waits for."""
        if self.gave_up is not None:
            return self.gave_up
        if not self.gpus.active(device):
            return f"gpu{device} takes no more blocks"
        if self.hold_on(device) is not None:
            return f"the run holds the work of gpu{device}'s GPU"
        if self.monitor is not None and not self.monitor.quiet_now():
            return "the stage's cores are not quiet"
        unlike = self._unlike(device)
        if unlike is not None:
            return unlike
        if self._canary_due(device) is not None:
            return f"gpu{device} has its canary to read"
        return None

    def first_ready(
        self, devices: Sequence[int], active: Callable[[int], bool]
    ) -> Tuple[str, Optional[int]]:
        """The first of `devices` in service (`active`) that may bench a
        block (`ready`), passing over a device whose GPU the run holds
        alone, whose settings are unlike the others', or whose canary
        failed, until it may: ("go", device); ("again", None) when none of
        them is in service; ("stop", None) when the stage stops waiting."""
        since = self.clock()
        while True:
            live = [d for d in devices if active(d)]
            if not live:
                return "again", None
            for d in live:
                go = self.ready(d, lambda: False, wait=False, since=since)
                if go == "go":
                    return "go", d
                if go == "stop":
                    return "stop", None
            time.sleep(0.5)

    def _give_up(self, why: str) -> None:
        with self._lock:
            if self.gave_up is None:
                self.gave_up = why
                ui.err("guard", f"{why}; the stage benches no more blocks")

    def _leave_out(self, device: int, why: str) -> None:
        """Retire `device` for the rest of the stage (its GPU stayed held,
        or its canary kept failing, for longer than bench.health.max_wait_s);
        the stage's other devices go on."""
        if self.gpus.retire(device, why, temporary=False):
            ui.err(
                "health",
                f"gpu{device} takes no more blocks in this stage: {why}; the "
                f"stage's other GPUs go on",
            )

    # ── canary ──

    def _canary_due(self, device: int) -> Optional[float]:
        """None when `device` needs no canary now; else the clock time from
        which it may read one."""
        gpu = self.gpus.gpu.get(device)
        if not self.canary_on or self.canary_off is not None or gpu is None:
            return None
        if not self.state.canary_due(gpu_health.gpu_key(gpu)):
            return None
        with self._lock:
            return self._next_canary.get(device, self.clock())

    def _restamp(self, line: str) -> Optional[str]:
        kernel = _first_kernel(self.identity) if canary_needs_kernel(line) else None
        return canary_line(line, kernel)

    def _setup(
        self, replaces: Optional[Mapping[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """The run's canary setup: chosen from this stage's lines when the
        run has none, or when the one it has is `replaces`; None (and the
        canary off) when no line of the stage can be a canary."""
        setup = self.state.canary_setup()
        if setup is not None and replaces is None:
            return setup
        lines = gpu_health.choose_canary_lines(
            self.lines, self.health.canary.lines, self._restamp, shape_of_line
        )
        if not lines:
            self.canary_off = "no bench line of the stage can be a canary"
            ui.warn("health", f"no canary: {self.canary_off}")
            return None
        return self.state.set_canary_setup(
            lines,
            self.health.canary,
            source=str(self.out_dir),
            kernels=gpu_health.canary_kernels(lines, line_solution_list, self.identity),
            replaces=gpu_health.record_id(replaces) if replaces else None,
        )

    def _lines_now(
        self,
    ) -> Optional[Tuple[Dict[str, Any], List[Optional[str]], List[str]]]:
        """(the run's canary setup, its lines as this stage benches them, what
        changed: `gpu_health.canary_lines_now`); a setup this stage's library
        can time fewer than half the lines of is chosen anew. None when the
        canary is off."""
        setup = self._setup()
        if setup is None:
            return None
        lines, notes = gpu_health.canary_lines_now(setup, self.identity, _restamped)
        if 2 * sum(ln is not None for ln in lines) >= len(lines):
            return setup, lines, notes
        ui.warn(
            "health",
            f"the run's canary lines name kernels the library holds no more ("
            f"{'; '.join(notes)}): the canary is chosen anew from this stage's "
            f"lines, and every GPU's next canary becomes its reference",
        )
        setup = self._setup(replaces=setup)
        if setup is None:
            return None
        lines, notes = gpu_health.canary_lines_now(setup, self.identity, _restamped)
        return setup, lines, notes

    def _run_canary(self, device: int) -> Optional[bool]:
        """Read `device`'s canary; True when it passes, became the GPU's
        reference, or could not be read (its bench failed, its output did
        not line up, or it read fewer than half its lines): the GPU then
        benches without a reference or a verdict, and its next canary takes
        or judges it, as for a first canary; False when it fails (it is read
        again `canary.retry_s` later); None when the stage stops."""
        gpu = self.gpus.gpu[device]
        assert gpu is not None
        key = gpu_health.gpu_key(gpu)
        got = self._lines_now()
        if got is None:
            return True
        setup, lines, changed = got
        kept = [i for i, ln in enumerate(lines) if ln is not None]
        timings = int(setup["timings"])
        entries = gpu_health.canary_entries(
            [str(lines[i]) for i in kept], timings, name_entry
        )
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with self._lock:
            n = len(self.canaries)
        name = f"{key.replace(':', '-')}_{stamp}_{os.getpid()}_{n}.log"
        log = self.state.dir / "canary" / name
        latencies, names, run = self._bench_canary(device, entries, log)
        if run.get("reason") == STOPPING:
            return None
        readings: List[Optional[float]] = [None] * len(lines)
        kernels: List[Optional[str]] = [None] * len(lines)
        values = gpu_health.canary_readings(latencies, len(kept), timings)
        timed = gpu_health.timed_kernels(names, len(kept), timings)
        for j, i in enumerate(kept):
            readings[i], kernels[i] = values[j], timed[j]
        ref = self.state.reference(key)
        rec: Dict[str, Any] = {
            "gpu": gpu.index,
            "key": key,
            "device": device,
            "taken": kfd.utc_iso(),
            "session": self.state.session,
            "by": self.state.by,
            "log": str(log),
            **run,
            "readings_us": readings,
            "kernels": kernels,
            **({"lines_changed": changed} if changed else {}),
        }
        n_read = sum(r is not None for r in readings)
        clean = run.get("rc") == 0 and not run.get("reason")
        clean = clean and not run.get("misaligned")
        how = (
            f"{n_read} of {len(lines)} canary lines read, bench rc {run.get('rc')}"
            + (f", {run['reason']}" if run.get("reason") else "")
            + (
                ", its output not lined up with its entries"
                if run.get("misaligned")
                else ""
            )
        )
        retake = unread = ""
        if ref is not None:
            verdict = gpu_health.judge(
                ref.get("readings_us") or [],
                readings,
                float(setup["tolerance"]),
                ref.get("kernels"),
                kernels,
            )
            need = max(1, math.ceil(len(ref.get("readings_us") or []) / 2))
            if (
                verdict["shift"] is None
                and verdict["other_kernels"]
                and verdict["lines_read"] + verdict["other_kernels"] >= need
            ):
                retake = (
                    f"{verdict['other_kernels']} of {len(lines)} canary lines "
                    f"timed another kernel than the GPU's reference of "
                    f"{ref.get('taken')}"
                )
                rec["replaces"] = gpu_health.record_id(ref)
            elif clean and verdict["shift"] is not None:
                rec.update(
                    verdict,
                    reference=False,
                    judged=True,
                    reference_taken=ref.get("taken"),
                )
            else:
                unread = how
        if ref is None or retake:
            enough = clean and 2 * n_read >= len(lines)
            if enough:
                why = f"{retake}: taken as its reference anew" if retake else ""
            else:
                why = (
                    f"no reference taken ({how}"
                    + (f"; {retake}" if retake else "")
                    + "); the GPU benches without one and takes it at its next "
                    "canary"
                )
            rec.update(reference=enough, judged=False, passed=True, why=why)
            if not enough:
                rec["unread"] = True
        elif unread:
            assert ref is not None
            rec.update(
                reference=False,
                judged=False,
                passed=True,
                unread=True,
                reference_taken=ref.get("taken"),
                why=f"no verdict ({unread}); the GPU benches without one, as after "
                f"a first canary that cannot be read, and its next canary judges "
                f"it",
            )
        rec = self.state.record_canary(key, rec)
        with self._lock:
            self.canaries.append(rec)
        if rec.get("reference"):
            ui.ok(
                "health",
                f"gpu{device} ({key}): canary taken as its reference"
                + (f" ({rec['why']})" if rec.get("why") else ""),
            )
        elif not rec.get("judged"):
            ui.warn("health", f"gpu{device} ({key}): {rec['why']}; log {log}")
        elif rec.get("passed"):
            ratio = rec.get("ratio")
            ui.ok(
                "health",
                f"gpu{device} ({key}): canary passes"
                + (f", {ratio:.3f}x its reference" if ratio else ""),
            )
        else:
            with self._lock:
                self._next_canary[device] = self.clock() + self.health.canary.retry_s
            ui.err(
                "health",
                f"gpu{device} ({key}): canary fails: {rec.get('why')} (tolerance "
                f"{setup['tolerance']:g}, recorded with the run's canary on "
                f"{setup['created']}). The GPU benches no block until a canary "
                f"passes; the next is read in "
                f"{ui.fmt_dur(self.health.canary.retry_s)}. Log: {log}; to take "
                f"its current state as the reference: scripts/run_guard.py accept "
                f"{self.state.dir} --gpu {key}",
            )
        return bool(rec.get("passed"))

    def _bench_canary(
        self, device: int, entries: Sequence[str], log: Path
    ) -> Tuple[List[Optional[float]], List[Optional[str]], Dict[str, Any]]:
        """Bench the canary `entries` in one process on `device`, its output
        in `log`; (the latency of each entry and the name hash of the kernel
        it timed, None for one without, and the process's rc and kill
        reason)."""
        from lib import kernel_identity as ki

        none: List[Any] = [None] * len(entries)
        log.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            prefix="health_canary_",
            dir=self.work_dir,
            delete=False,
        ) as f:
            tmp = Path(f.name)
            f.writelines(entries)
        bench_device, env = kfd.one_gpu_launch(device, bench_env_with(BENCH_ENV_EXTRA))
        cmd = [str(self.bench), "--device", bench_device, "--yaml", str(tmp)]
        rc: Optional[int] = None
        reason: Optional[str] = None
        try:
            with log.open("w") as lf:
                proc = self.children.popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env
                )
                probe = LogProbe(log, 0)
                rc, reason = procs.wait_with_stall_watchdog(
                    proc,
                    lambda: file_size(log),
                    self.stall_s,
                    poll_s=procs.poll_interval(self.stall_s),
                    startup_grace_s=self.startup_grace_s,
                    probe_progress=probe.progressed,
                    probe_fatal=probe.fatal_seen,
                )
        except procs.ChildrenStopped:
            return none, list(none), {"rc": None, "reason": STOPPING}
        finally:
            tmp.unlink(missing_ok=True)
        problems = list(iter_problems(log.read_text(errors="replace").splitlines()))
        off = _misalignment(
            problems,
            [shape_of_line(e) for e in entries],
            rc == 0 and reason is None,
            single=True,
        )
        if off is not None:
            return none, list(none), {"rc": rc, "reason": reason, "misaligned": off}
        rows = [p.tested[0] if p.complete and p.tested else None for p in problems]
        out = [_latency(r.fields) if r is not None else None for r in rows]
        names = [
            ki.name_hash(r.sol_name) if r is not None and r.sol_name else None
            for r in rows
        ]
        pad = [None] * (len(entries) - len(out))
        return out + pad, names + list(pad), {"rc": rc, "reason": reason}

    # ── lines ──

    def kernel_label(self, sid: int) -> Optional[str]:
        if self.identity is None:
            return None
        from lib import kernel_identity as ki

        kernel = self.identity.kernels.get(int(sid))
        return ki.signature_label(kernel) if kernel is not None else None

    @property
    def quarantines(self) -> bool:
        """Whether a line that hangs is quarantined in the run (under
        `hung_scope`); with NO_SCOPE it only leaves its block."""
        return self.hung_scope != gpu_health.NO_SCOPE

    def _note_hung(self, rec: Dict[str, Any], device: int) -> None:
        """Add where `rec` hung to it, and keep it for the stage's record."""
        gpu = self.gpus.gpu.get(device)
        rec.update(
            device=device,
            gpu=gpu.index if gpu is not None else self.gpus.physical.get(device),
            stage=str(self.out_dir),
            scope=self.hung_scope,
        )
        with self._lock:
            if not any(h is rec for h in self.hung):
                self.hung.append(rec)

    def quarantine_line(self, rec: Dict[str, Any], device: int) -> None:
        """Quarantine the hung line of `rec` in the run under `hung_scope`
        (unless that is NO_SCOPE); `rec` says so (`quarantined`)."""
        self._note_hung(rec, device)
        rec["quarantined"] = self.quarantines
        if self.quarantines:
            try:
                self.state.quarantine_lines([rec])
            except OSError as e:
                ui.warn("hung", f"could not record the hung line in the run: {e}")
        m, n, k, b = rec.get("shape") or ("?",) * 4
        fate = (
            f"quarantined in the run ({self.hung_scope}), not benched again"
            if self.quarantines
            else "not benched again in this stage"
        )
        if rec.get("retried"):
            fate = f"{rec['retried']} in the retry pass, then {fate}"
        ui.warn(
            "hung",
            f"{rec['block']} on gpu{device}: line {rec['yaml_line']} (M={m} N={n} "
            f"K={k} B={b}) gave no output for {rec['stall_s']:.0f}s in "
            f"{rec['in_flight']}: killed, {fate}",
        )

    def defer_hung(self, rec: Dict[str, Any], device: int) -> None:
        """Keep the hung line of `rec` for the retry pass (`retry_hung`):
        it is quarantined only when that pass does not bench it to the end
        (`settle_hung`)."""
        self._note_hung(rec, device)
        rec.update(retry=True, quarantined=False)
        m, n, k, b = rec.get("shape") or ("?",) * 4
        ui.warn(
            "hung",
            f"{rec['block']} on gpu{device}: line {rec['yaml_line']} (M={m} N={n} "
            f"K={k} B={b}) gave no output for {rec['stall_s']:.0f}s in "
            f"{rec['in_flight']}: killed; the retry pass benches it once more in "
            f"a process of its own before it is quarantined",
        )

    def settle_hung(
        self, blocks: Sequence[TaskBlock], hung_again: AbstractSet[int]
    ) -> None:
        """After the retry pass: a line of `blocks` it took for hanging
        (`defer_hung`) is quarantined unless the pass benched it to the end
        (`TaskBlock.hung_recovered`); one that hung there again (its yaml
        line in `hung_again`) the pass quarantined itself."""
        for b in blocks:
            for rec in b.hung:
                if not rec.get("retry") or rec.get("retried"):
                    continue
                n = int(rec["yaml_line"])
                if n in b.hung_recovered:
                    rec["retried"] = "benched"
                elif n in hung_again:
                    rec["retried"] = "hung again"
                else:
                    rec["retried"] = "not benched to the end"
                    self.quarantine_line(rec, int(rec["device"]))

    def drop_quarantined(
        self, lines: Sequence[str]
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """(the lines the run has not quarantined under `hung_scope`, their
        1-based line numbers, a record per line left out)."""
        banned = (
            self.state.quarantined_shapes(self.hung_scope)
            if self.quarantines
            else set()
        )
        kept: List[str] = []
        numbers: List[int] = []
        dropped: List[Dict[str, Any]] = []
        for i, ln in enumerate(lines, 1):
            shape = shape_of_line(ln)
            if shape is not None and tuple(shape) in banned:
                dropped.append({"yaml_line": i, "shape": list(shape)})
            else:
                kept.append(ln)
                numbers.append(i)
        self.skipped_lines = dropped
        return kept, numbers, dropped

    # ── records ──

    def record(self) -> Dict[str, Any]:
        with self._lock:
            canaries = [
                {k: v for k, v in c.items() if k != "log"} for c in self.canaries
            ]
            holds = list(self.holds)
            hung = list(self.hung)
            platform = {
                "held_at_start": [
                    {"device": d, "why": w}
                    for d, w in sorted(self.held_at_start.items())
                ],
                "joined": list(self.joined),
                "unlike": list(self.unlike),
            }
        return {
            "guard_dir": str(self.state.dir),
            "session": self.state.session,
            "quiet_host": (
                self.monitor.summary()
                if self.monitor is not None
                else {"enabled": False}
            ),
            "health": {
                "enabled": self.health.enabled,
                "settings": self.health.to_json(),
                "gpus": [gpu_health.gpu_key(g) for g in self.health_gpus],
                "holds": holds,
                "platform": platform,
                "canary": {
                    "enabled": self.canary_on,
                    "off": self.canary_off,
                    "read": canaries,
                },
            },
            "hung_lines": hung,
            "quarantined_lines_left_out": list(self.skipped_lines),
            "gave_up": self.gave_up,
        }


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


def _set_aborted_aside(
    block: TaskBlock,
    device: int,
    logs: Sequence[Optional[Path]],
    aborted_dir: Path,
    guard: Optional[_Guard],
) -> Dict[str, Any]:
    """Move the logs of `block`'s attempt the guard stopped to
    `aborted_dir/<log stem>_a<n>/` with aborted.json (why, the load on the
    stage's cores while it ran, the run's hold), and reset the block for a
    re-bench. Returns the record."""
    n = len(block.aborts) + 1
    log_path = Path(str(logs[0]))
    dest = aborted_dir / f"{log_path.stem}_a{n}"
    held = guard.state.hold() if guard is not None else None
    rec = _move_aside(
        dest,
        logs,
        ABORTED_RECORD,
        {
            "block": block.name,
            "yaml_lines": (
                [block.line_numbers[0], block.line_numbers[-1]]
                if block.line_numbers
                else []
            ),
            **block.gpu_fields,
            "device": device,
            "attempt": n,
            **(block.aborted or {}),
            "host_load": dict(block.host_load),
            **({"hold": held} if held else {}),
        },
    )
    block.aborts.append({**rec, "dir": str(dest)})
    why = (block.aborted or {}).get("reason")
    ui.warn(
        "guard",
        f"{block.name} on gpu{device}: attempt {n} aborted ({why}) -> {dest}; "
        + (
            "benched again later"
            if n <= MAX_ABORTS
            else f"aborted more than {MAX_ABORTS} times, given up"
        ),
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
    while `device` is retired, wait for it to return (`_await_gpu`). With a
    guard, the worker waits until `device` is ready before it takes a block
    and asks again once it holds one, since the wait for a block can be
    long (`_Work.take` waits while blocks are out): a block that cannot
    start yet goes back on the queue."""

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
        guard: Optional[_Guard] = None,
        warmup_line: Optional[WarmupLine] = None,
    ) -> None:
        super().__init__(name=f"bench-d{device}", daemon=True)
        self.device = device
        self.work = work
        self.bench_binary = bench_binary
        self.logs_dir = logs_dir
        self.warmup = warmup
        self.warmup_line = warmup_line
        self.stall_s = stall_s
        self.startup_grace_s = startup_grace_s
        self.quiet = quiet
        self.children = children
        self.work_dir = work_dir
        self.gpus = gpus if gpus is not None else kfd.StageGpus([])
        self.quarantine_dir = quarantine_dir or logs_dir.parent / QUARANTINE_DIR
        self.canary = canary
        self.guard = guard
        self.n_blocks_completed = 0
        self.wall_s = 0.0
        self.warmup_s = 0.0

    def run(self) -> None:
        t0 = time.time()
        while True:
            if not self.gpus.active(self.device):
                if _await_gpu(self.gpus, [self.device], self.work.closed) is None:
                    break
            if self.guard is not None:
                go = self.guard.ready(self.device, self.work.closed)
                if go == "stop":
                    break
                if go == "again":
                    continue
            block = self.work.take(lambda: not self.gpus.active(self.device))
            if block is None:
                if self.gpus.active(self.device):
                    break
                continue
            if self.guard is not None and self.guard.blocked(self.device):
                self.work.requeue(block)
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
                guard=self.guard,
                warmup_line=self.warmup_line,
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
        if block.aborted is not None:
            _set_aborted_aside(
                block, self.device, logs, self.logs_dir.parent / ABORTED_DIR, self.guard
            )
            if len(block.aborts) > MAX_ABORTS:
                block.gave_up = True
                self.work.finish(block)
            else:
                self.work.requeue(block)
            return
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
            f"bench.warmup_pass (on), bench.warmup_line and bench.adaptive",
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
        host_load.SERIES_FILE,
    ):
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    for name in (QUARANTINE_DIR, DRIFT_DIR, MISALIGNED_DIR, ABORTED_DIR):
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
    guard: Optional[_Guard] = None,
    warmup_line: Optional[WarmupLine] = None,
) -> Tuple[int, int, Optional[TaskBlock]]:
    """Re-bench every skipped and unattempted shape of `blocks`, and every
    hung line they keep for it (`TaskBlock.hung_lines`, --retry-hung), in
    one process (led by `warmup_line`, as every process) on the first of
    `devices` that takes blocks (`_await_gpu`; another one after a
    quarantine, up to `max_quarantines` times, or once more, after the
    canary's back-off, when its canary drifted; with `guard`, the first one
    that is ready, `_Guard.first_ready`, which passes over a GPU the run
    holds or whose health canary failed, and again after an abort, up to
    MAX_ABORTS times), marking the ones that finish in `TaskBlock.recovered`
    (`hung_recovered`). Returns (retried, recovered, the retry block)."""
    gpus = gpus if gpus is not None else kfd.StageGpus([])
    pending: List[Tuple[TaskBlock, Optional[int], int, str]] = [
        (b, off, b.line_numbers[off], b.lines[off])
        for b in blocks
        if not b.gave_up
        for off in sorted(set(b.skipped) | set(b.unattempted))
    ]
    pending += [
        (b, None, n, text) for b in blocks for n, text in sorted(b.hung_lines.items())
    ]
    if not pending:
        return 0, 0, None
    retry = TaskBlock(
        block_id=-1,
        name="block_retry",
        lines=[text for _, _, _, text in pending],
        line_numbers=[n for _, _, n, _ in pending],
    )
    while True:
        device = _await_gpu(gpus, devices, lambda: False)
        if device is None:
            ui.warn("retry", "no GPU takes blocks; the retry pass does not run")
            return len(pending), 0, retry
        if guard is not None:
            go, ready = guard.first_ready(devices, gpus.active)
            if go == "again":
                continue
            if go == "stop" or ready is None:
                why = guard.gave_up or STOPPING
                ui.warn("retry", f"the retry pass does not run: {why}")
                return len(pending), 0, retry
            device = ready
        log_path = logs_dir / f"{retry.name}_gpu{device}.log"
        canary_log_path = logs_dir / f"{CANARY_LOG_PREFIX}retry_gpu{device}.log"
        warmup_log_path = (
            logs_dir / f"warmup_retry_gpu{device}.log"
            if warmup_line is not None
            else None
        )
        retry.gpu_fields = _block_fields(gpus, device)
        _run_block_with_recovery(
            retry,
            device,
            bench_binary,
            log_path,
            warmup=False,
            warmup_log_path=warmup_log_path,
            stall_s=stall_s,
            startup_grace_s=startup_grace_s,
            children=children,
            work_dir=work_dir,
            watch=gpus.watch(device),
            canary=canary,
            canary_log_path=canary_log_path,
            aside_dir=logs_dir.parent / MISALIGNED_DIR,
            guard=guard,
            warmup_line=warmup_line,
        )
        if retry.aborted is not None:
            _set_aborted_aside(
                retry,
                device,
                (log_path, warmup_log_path, canary_log_path),
                logs_dir.parent / ABORTED_DIR,
                guard,
            )
            if len(retry.aborts) > MAX_ABORTS:
                retry.gave_up = True
                return len(pending), 0, retry
            continue
        if retry.contaminated is None:
            logs = (log_path, warmup_log_path, canary_log_path)
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
            warmup_log_path,
            logs_dir.parent / QUARANTINE_DIR,
            gpus,
            canary_log_path,
        )
        if len(retry.quarantines) > max_quarantines:
            retry.gave_up = True
            return len(pending), 0, retry
    failed = {
        retry.line_numbers[i] for i in set(retry.skipped) | set(retry.unattempted)
    }
    benched = set(retry.line_numbers)
    recovered = 0
    for b, off, number, _ in pending:
        if number in benched and number not in failed:
            if off is None:
                b.hung_recovered.add(number)
            else:
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


def stage_devices(
    devices: Sequence[int], cfg: Mapping[str, Any]
) -> Tuple[List[int], List[int], List[str]]:
    """(`devices` without those whose GPU the config's runtime.exclude_gpus
    names, the devices left out, the entries that name no GPU here)."""
    refs = ((cfg.get("runtime") or {}).get("exclude_gpus") or []) if cfg else []
    if not refs:
        return list(devices), [], []
    excluded, unknown = kfd.gpu_refs(refs)
    kept, dropped = kfd.without_gpus(devices, excluded)
    return kept, dropped, unknown


def release_excluded(state: gpu_health.Guard, cfg: Mapping[str, Any]) -> List[str]:
    """Drop the run's hold of the GPUs the config's runtime.exclude_gpus
    names (`gpu_health.release_excluded`); their keys."""
    refs = ((cfg.get("runtime") or {}).get("exclude_gpus") or []) if cfg else []
    gone = gpu_health.release_excluded(state, refs)
    if gone:
        ui.warn(
            "health",
            f"the run's hold of {', '.join(gone)} is dropped: runtime.exclude_gpus "
            f"leaves them out",
        )
    return gone


def main() -> int:
    args = _parse_args()
    work_dir = args.out_dir / TMP_DIR / str(os.getpid())
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        try:
            cfg = _read_config(args.config_yaml)
            devices, dropped, unknown = stage_devices(_devices(args), cfg)
            for ref in unknown:
                ui.warn("gpus", f"runtime.exclude_gpus: {ref} names no GPU here")
            if dropped:
                ui.warn(
                    "gpus",
                    f"devices {dropped} left out: runtime.exclude_gpus names their GPUs",
                )
            if not devices:
                ui.err("err", "every device of --devices is excluded")
                return 1
            with gpu_lease.hold_usable(
                devices,
                "stage03 bench",
                args.out_dir,
                report=lambda m: ui.warn("lease", m),
            ) as leases:
                return _run(args, children, work_dir, cfg, leases.devices)
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
        default=600.0,
        help="kill a bench process whose log has not grown for this many "
        "seconds and quarantine the line it hung in, in the run; 0 disables "
        "the watchdog.",
    )
    ap.add_argument(
        "--hung-scope",
        default=gpu_health.BENCH_SCOPE,
        metavar="NAME",
        help="the measurement protocol of the stage's lines: a line that hangs "
        "is quarantined in the run under it, and only the lines the run "
        f"quarantined under it are left out (default {gpu_health.BENCH_SCOPE!r}, "
        f"the run's own benches; {gpu_health.NO_SCOPE!r} quarantines and leaves "
        "out nothing).",
    )
    ap.add_argument(
        "--retry-hung",
        action="store_true",
        help="bench a line that hangs once more in the retry pass's own process "
        "before it is quarantined (not with --no-retry-skipped).",
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
        "--warmup-line",
        dest="warmup_line",
        action="store_true",
        default=None,
        help="start every bench process with the stage's discarded warm-up "
        "line (default: bench.warmup_line.enabled from --config-yaml, else "
        f"{'on' if bench_yaml.WARMUP_LINE_DEFAULTS['enabled'] else 'off'}).",
    )
    ap.add_argument(
        "--no-warmup-line",
        dest="warmup_line",
        action="store_false",
        help="no warm-up line.",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config; bench.adaptive is read, bench.warmup_pass and "
        "bench.warmup_line.enabled when their command-line flags are not "
        "given, the rest of bench.warmup_line, and bench.canary for the "
        "canary options not given.",
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
    ap.add_argument(
        "--quiet-host",
        dest="quiet_host",
        action="store_true",
        default=None,
        help="start a block only when the stage's cores are quiet and abort one "
        "they turn hot under (default: bench.quiet_host.enabled, else on)",
    )
    ap.add_argument("--no-quiet-host", dest="quiet_host", action="store_false")
    ap.add_argument(
        "--health",
        dest="health",
        action="store_true",
        default=None,
        help="check the GPUs against the run's baseline and hold the run's GPU "
        "work while they fail (default: bench.health.enabled, else on)",
    )
    ap.add_argument("--no-health", dest="health", action="store_false")
    ap.add_argument(
        "--health-canary",
        dest="health_canary",
        action="store_true",
        default=None,
        help="read every GPU's reference canary before its first block of the "
        "session (default: bench.health.canary.enabled while the health checks "
        "are on, else on)",
    )
    ap.add_argument("--no-health-canary", dest="health_canary", action="store_false")
    ap.add_argument(
        "--leave-held-gpus",
        action="store_true",
        help="leave a device whose GPU the run holds alone out of the stage at "
        "once rather than wait for the hold (for a stage whose work another "
        "process can take, such as a part of audit protocol v2)",
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


def _guard_settings(
    args: argparse.Namespace, cfg: Mapping[str, Any]
) -> Tuple[host_load.Settings, gpu_health.Settings, bool]:
    """The quiet-host and health settings of the stage (the command line's
    switches over the config's `bench.quiet_host` and `bench.health`) and
    whether it reads the health canary: as the health checks say
    (`gpu_health.Settings.canary_on`) unless the command line says."""
    quiet = host_load.Settings.of(cfg)
    health = gpu_health.Settings.of(cfg)
    if args.quiet_host is not None:
        quiet = dataclasses.replace(quiet, enabled=bool(args.quiet_host))
    if args.health is not None:
        health = dataclasses.replace(health, enabled=bool(args.health))
    canary = health.canary_on
    if args.health_canary is not None:
        canary = bool(args.health_canary)
    return quiet, health, canary


def _write_skipped(
    out_dir: Path, no_data: Sequence[Tuple[int, Sequence[Any], Any, str]]
) -> None:
    """`<out_dir>/skipped_gemms.txt` of the GEMMs without data, (yaml line,
    shape, rc, reason) each; nothing when there is none."""
    if not no_data:
        return
    skip_path = out_dir / "skipped_gemms.txt"
    rows = ["yaml_line,M,N,K,B,rc,reason\n"]
    for number, shape, code, reason in sorted(no_data, key=lambda t: t[0]):
        rows.append(
            f"{number},{shape[0]},{shape[1]},{shape[2]},{shape[3]},{code},{reason}\n"
        )
    atomic_write_text(skip_path, "".join(rows))
    ui.warn("skip", f"{len(no_data)} GEMM(s) without data -- see {skip_path}")


def _all_quarantined(
    args: argparse.Namespace,
    in_yaml: Path,
    devices: Sequence[int],
    n_shapes: int,
    guard: _Guard,
    identity: Mapping[str, Any],
) -> int:
    """Exit 1 without a bench when the run quarantined every line of the
    stage under its --hung-scope, recording each line as left out
    (skipped_gemms.txt, and stage03_summary.json with the guard's record,
    from which audit protocol v2 drops their GEMMs)."""
    problem = f"every line of {in_yaml} is quarantined in the run ({guard.hung_scope})"
    ui.err("err", problem)
    left = guard.skipped_lines
    _write_skipped(
        args.out_dir,
        [(int(d["yaml_line"]), d["shape"], "", "quarantined_line") for d in left],
    )
    write_json(
        args.out_dir / "stage03_summary.json",
        {
            "in_yaml": str(in_yaml),
            "devices": list(devices),
            "n_shapes": n_shapes,
            "n_blocks": 0,
            "benched": 0,
            "lost": len(left),
            "quarantined_lines_left_out": len(left),
            "quarantined": [],
            "aborted": [],
            "guard": guard.record(),
            "problems": [problem],
            "rc": 1,
            **identity,
            "blocks": [],
        },
    )
    return 1


def _run(
    args: argparse.Namespace,
    children: procs.ChildGroups,
    work_dir: Path,
    cfg: Optional[Dict[str, Any]] = None,
    devices: Optional[List[int]] = None,
) -> int:
    cfg = _read_config(args.config_yaml) if cfg is None else cfg
    BENCH_ENV_EXTRA.clear()
    BENCH_ENV_EXTRA.update(runtime_env(cfg))
    warmup, warmup_src = _resolve_warmup(args, cfg)
    devices = _devices(args) if devices is None else list(devices)
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
    try:
        warm, no_warm = _resolve_warmup_line(args, cfg, lines, library)
    except ValueError as e:
        ui.err("warm-up", str(e))
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
    gpus = kfd.StageGpus(devices)
    quiet, health, health_canary = _guard_settings(args, cfg)
    guard = _Guard(
        quiet=quiet,
        health=health,
        canary=health_canary,
        state=gpu_health.Guard(
            gpu_health.guard_dir(args.out_dir / gpu_health.GUARD_DIR),
            gpu_health.session(),
            by=f"stage03 {args.out_dir} pid {os.getpid()}",
        ),
        gpus=gpus,
        out_dir=args.out_dir,
        bench_binary=args.bench_binary,
        children=children,
        work_dir=work_dir,
        stall_s=float(args.stall_timeout_s),
        startup_grace_s=float(args.startup_grace_s),
        lines=lines,
        identity=library,
        hung_scope=args.hung_scope,
        retry_hung=bool(args.retry_hung) and not args.no_retry_skipped,
        leave_held=bool(args.leave_held_gpus),
    )
    release_excluded(guard.state, cfg)
    kept, numbers, left_out = guard.drop_quarantined(lines)
    if left_out:
        ui.warn(
            "hung",
            f"{len(left_out)} line(s) left out: the run quarantined them when they "
            f"hung before ({guard.state.path})",
        )
    if not kept:
        return _all_quarantined(args, in_yaml, devices, n_shapes, guard, identity)
    guard.check_now()
    held = guard.held_devices()
    try:
        session = fingerprint.start_session(
            devices,
            args.out_dir,
            environment=recorded_env(cfg, env),
            launch=launch,
            held=held,
            warmup_line=_warmup_record(warm, no_warm),
        )
    except fingerprint.PlatformError as e:
        ui.err("platform", f"refused: {e}")
        return 2
    guard.admit(session)
    for d, why in sorted(held.items()):
        ui.warn(
            "platform",
            f"gpu{d} is left out of the session's platform check: the run holds "
            f"its GPU ({why}); it benches once the run stops holding it and its "
            f"settings agree with the stage's other GPUs",
        )
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
    guard.start()

    n_gpus = len(devices)
    target_blocks = max(n_gpus, n_gpus * int(args.blocks_per_gpu))
    block_size = max(1, math.ceil(len(kept) / target_blocks))
    blocks = _build_blocks(kept, block_size, numbers)
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
            "warmup_line  : "
            + (warm.describe() if warm is not None else f"none ({no_warm})"),
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
            guard=guard,
            warmup_line=warm,
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
    n_pending = n_skipped + n_unattempted + sum(len(b.hung_lines) for b in done)
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
            guard=guard,
            warmup_line=warm,
        )
        if not args.quiet:
            ui.ok("retry", f"recovered {recovered}/{retried} GEMM(s)")
    hung_retry = {
        int(h["yaml_line"]) for h in (retry_block.hung if retry_block else [])
    }
    guard.settle_hung(done, hung_retry)
    if canary is not None:
        _flag_at_end(
            done + ([retry_block] if retry_block is not None else []),
            canary,
            logs_dir,
        )
    guard.stop()
    guard_record = guard.record()
    fingerprint.add_to_record(
        args.out_dir,
        {"host_load": guard_record["quiet_host"], "health": guard_record["health"]},
    )

    unbenched_ids = {id(b) for b in unbenched}
    lost: List[Tuple[TaskBlock, int, str]] = []
    for b in done:
        if b.gave_up:
            why = "aborted" if len(b.aborts) > MAX_ABORTS else "quarantined"
            lost.extend((b, off, why) for off in range(len(b.lines)))
            continue
        for off in sorted(b.skipped):
            if off not in b.recovered:
                hung = b.line_numbers[off] in hung_retry
                lost.append((b, off, "hung" if hung else "crashed"))
        left = (
            "quarantined" if id(b) in unbenched_ids and b.quarantines else "unattempted"
        )
        for off in b.unattempted:
            if off not in b.recovered:
                hung = b.line_numbers[off] in hung_retry
                lost.append((b, off, "hung" if hung else left))
    hung_lines = [h for b in done for h in b.hung if h.get("retried") != "benched"]
    crash_blocks = [
        b
        for b in done
        if not b.gave_up and any(off not in b.recovered for off in b.unattempted)
    ]
    rc = 0 if len(crash_blocks) <= tolerance else 2
    attempted = done + ([retry_block] if retry_block is not None else [])
    quarantined = [q for b in attempted for q in b.quarantines]
    problems: List[str] = []
    gave_up = [b.name for b in attempted if b.gave_up and len(b.aborts) <= MAX_ABORTS]
    if gave_up:
        problems.append(
            f"{len(gave_up)} block(s) quarantined more than --max-quarantines "
            f"{args.max_quarantines} time(s): {', '.join(gave_up)}"
        )
    aborted_out = [b.name for b in attempted if len(b.aborts) > MAX_ABORTS]
    if aborted_out:
        problems.append(
            f"{len(aborted_out)} block(s) aborted by the guard more than "
            f"{MAX_ABORTS} times: {', '.join(aborted_out)}"
        )
    if guard.gave_up:
        problems.append(f"the guard stopped the stage: {guard.gave_up}")
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

    no_data: List[Tuple[int, Sequence[Any], Any, str]] = [
        (
            b.line_numbers[off],
            shape_of_line(b.lines[off]) or ("", "", "", ""),
            b.skipped.get(off, ""),
            reason,
        )
        for b, off, reason in lost
    ]
    no_data += [
        (int(h["yaml_line"]), h.get("shape") or ("", "", "", ""), "", "hung")
        for h in hung_lines
    ]
    no_data += [
        (int(d["yaml_line"]), d["shape"], "", "quarantined_line")
        for d in guard.skipped_lines
    ]
    _write_skipped(args.out_dir, no_data)
    write_json(
        args.out_dir / "stage03_summary.json",
        {
            "in_yaml": str(in_yaml),
            "devices": devices,
            "n_shapes": n_shapes,
            "n_blocks": len(blocks),
            "block_size": block_size,
            "warmup_pass": warmup,
            "warmup_line": _warmup_record(warm, no_warm, outcome=True),
            "benched": sum(b.n_benched for b in done),
            "skipped": n_skipped,
            "unattempted": n_unattempted,
            "retried": retried,
            "recovered": recovered,
            "lost": len(no_data),
            "hung": len({int(h["yaml_line"]) for h in hung_lines} | hung_retry),
            "hung_recovered": sum(len(b.hung_recovered) for b in done),
            "quarantined_lines_left_out": len(guard.skipped_lines),
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
            "aborted": [a for b in attempted for a in b.aborts],
            "guard": guard_record,
            "problems": problems,
            "rc": rc,
            **identity,
            "blocks": [
                {
                    "name": b.name,
                    "yaml_lines": (
                        [b.line_numbers[0], b.line_numbers[-1]]
                        if b.line_numbers
                        else []
                    ),
                    **b.gpu_fields,
                    "wall_s": round(b.wall_s, 3),
                    "rc": b.rc,
                    "quarantines": len(b.quarantines),
                    "aborts": len(b.aborts),
                    "hung": [h["yaml_line"] for h in b.hung],
                    "host_load": b.host_load,
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
    if warm is not None and warm.off:
        ui.warn("warm-up", f"the warm-up line went off during the stage: {warm.off}")
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
