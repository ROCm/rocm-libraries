# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03_load_balance_offline_tuning -- bench every shape with
hipblaslt-bench on all GPUs: a dynamic work queue, per-shape crash recovery,
and a retry pass.

Reads `<in-dir>/shapes.yaml` (stage01 or stage02) and cuts it into blocks,
about --blocks-per-gpu per GPU, each one bench process, whose start-up
costs the same whatever the block holds. --min-block-lines N (1 by default)
raises the block size to N lines, so that a stage makes at most
ceil(lines / N) blocks, but never above one block per GPU's size
(`_min_block_size`). One worker thread per GPU pulls the next block whenever
it finishes one, so faster GPUs take more blocks. With
`bench.heavy_operand_gib` above 0 (0, no size, by default), lines with an
operand (A, B, C or D over the batch) of that many GiB or more
(lib/bench_yaml.py `heavy_line`) go into blocks of their own at the end of
the work queue, at least one per GPU where there are enough of them, so that
a GPU they wedge has benched the other lines first; a stage of one block per
GPU keeps them with the others. The heavy blocks are cut by --blocks-per-gpu
alone: --min-block-lines sizes the other blocks only.

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

Wedged GPUs: a bench process stuck in the GPU driver (uninterruptible
sleep) when the watchdog or the guard kills it gets no SIGKILL until it
leaves the driver call, and is left running when it stays stuck for
procs.STUCK_WAIT_S (lib/procs.py). After any kill, and after a process
that failed without finishing a shape, the stage waits for the process to
end (REAP_S) and looks for processes the GPU driver lists stuck in
uninterruptible sleep (lib/hazards.py `stuck_processes`, where KFD names
processes by the stage's pids; those stuck since before the stage started,
and those an earlier check found, count only when they are its own); a
process other than the bench counts only when it stays stuck through
another REAP_S, as a slow driver call or the teardown of a process that
is ending does not. Wedged are: the bench's GPU, when the bench does not
end; the stage's GPUs the stuck processes hold memory on; and the host,
when processes the stage started are stuck holding memory on no GPU. A
process stuck only on other GPUs, or on none, that the stage did not
start is recorded and counts for nothing: another run's wedge is not this
stage's. Nothing is judged once the stage is stopping. The run holds each
wedged GPU until the host reboots (lib/gpu_health.py WEDGED;
scripts/run_guard.py accept lifts it too) and the stage starts no process
on it again. Only a wedge of the bench's own GPU is the bench's: a check
that finds only other GPUs, or the host alone, wedged holds them, and the
bench counts as a crash like any other (its line skipped and benched
again in the retry pass), its GPU benching on unless the stage stops. A
block whose own GPU wedged goes back on the work queue and is benched
anew on another GPU, without the line its process crashed in, or was
stuck in when the guard stopped it (the warm-up shape is the shape of
the block's next line, so a process that wedged its GPU there did so
in that line), which no GPU benches again in the stage
(skipped_gemms.txt: `wedged`, in the retry pass too) and which the run
quarantines under the stage's --hung-scope, as a hung line (`why`
wedged; `scripts/run_guard.py release` lets it back in); a process
that wedged its GPU before it started a line of the block (in its
start-up, such as the library load, or in the warm-up line or the
canary) leaves every line in the block, and a line it hung in leaves
the block as any hung line does; a block whose every shape finished
keeps its data. A wedged host, two or more wedged GPUs of the stage
(found in one check or in several, or held so by the run), or a wedge
of every GPU of the stage stops the stage: it aborts the running
blocks, starts no bench process, benches no more and exits 3
(`procs.WEDGED_EXIT`: device wedged, reboot). A GPU the run holds
as wedged takes no block from the start, while the health checks are on
(with bench.health.enabled false the stage reads no hold of the run, and
leaves out only the GPUs it finds wedged itself). A process that finished
every shape but had to be killed, or reported a fatal GPU error, and a
block abandoned after consecutive processes that finished no shape, are
GPU health events: the GPU benches its device probe (the stage's
least-work line, restamped to time one kernel alone, in a process of its
own under the watchdog) before its next block. A probe that is stuck,
reports a fatal GPU error, or hangs or prints nothing until the watchdog
kills it holds the GPU as wedged. One that crashes, or exits non-zero
before any problem, without a fatal GPU error takes the GPU out of this
stage only, since a bench binary, yaml or library that fails does that on
any GPU: the run does not hold it, and a stage that loses every GPU so
exits 2, which another attempt may cure.

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
(lib/launch_settings.py, under `launch`). It runs on its GPU's share of the
stage's CPUs (lib/host_load.py `device_cpus`): whole cores of those local to
the GPU, split among the GPUs that share them, else an even split of the
stage's CPU mask. The split is over the run's GPUs when the orchestrator
names them (lib/host_load.py BENCH_GPUS_VAR) and they hold the stage's own,
so that concurrent stages of one run split alike, else over the stage's
GPUs. OMP_NUM_THREADS is the share's CPU count unless the environment sets
it: the bench generates its host data with OpenMP, whose default team is
every CPU the process may run on, so benches sharing a CPU mask would
oversubscribe it together and preempt each other's launching threads. The
summary records the shares and how they were made (`bench_cpus`).

Quarantine: while a block's bench runs, the processes with memory on its GPU
and on the other partitions of its card, and the card's VRAM in use, are
sampled with the stall watchdog's polls and checked once more when the
process exits (lib/kfd.py `GpuWatch`). A process that is not the stage's
own (one of its benches or a process they started) on the GPU, one on
another partition of its card, which shares the card's power budget, that
belongs to none of the run's bench stages either (the trees of the run's
stage 3 processes, which the quiet cores count as the run's own load:
lib/host_load.py `bench_trees`), or VRAM no listed process holds
(unattributed) beyond its settle window,
which grows with the memory a bench held, stops the block: its logs move to
`quarantine/` with a record of the other use, and the block goes back on the
work queue, at most --max-quarantines times. A GPU that still has the other
use after a quarantine, or before a block starts, is retired: it is
re-checked every few seconds and takes blocks again after a few minutes
without other use (`kfd.RECHECK_S`, `kfd.READMIT_CLEAN_S`).
While every GPU is retired the stage waits for one, at most
`kfd.ALL_RETIRED_WAIT_S`, then gives the rest up. A bench holding memory on
a GPU other than its own means the GPU numbering is wrong; the block is
quarantined and the GPU retired for good.

Guard (lib/host_load.py, lib/gpu_health.py; the run's record in the
directory the orchestrator names, else `<out-dir>/guard/`): a block starts
on a device only when the run does not hold the device's GPU, the stage's
cores are quiet (`bench.quiet_host`, --quiet-host; the threads the run's
benches run beyond their CPUs count as load there), the GPU's settings
agree with the stage's other GPUs (the platform check above; a GPU is read
again once the run stops holding it, since its state may have been
accepted as the run's: scripts/run_guard.py accept) and the GPU read its
health canary in this session since the run last held it
(`bench.health.canary`, --health-canary, which follows --health; a canary
whose bench is stuck, reports a fatal GPU error, or hangs or prints
nothing until the watchdog kills it fails, as one beyond its tolerance
does, and the device leaves the stage after CANARY_FAULT_LIMIT such
canaries in a row, since the run's canary is fixed and one that faults on
its kernel faults every time; one whose bench crashes or exits non-zero
before any problem gives no verdict, as a bench that fails on any GPU
does); a device
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

A run that an interruption or an AC cycle cut off is continued by the next
invocation over the same lines and settings (lib/bench_resume.py): a done
block keeps its data, a partial one goes on after its last finished shape,
in its log (under the name of the GPU it goes on on), and the shape cut off
is benched again. So is a run the stage's own stop cut off (SIGINT, SIGTERM
or SIGHUP, as the orchestrator stops a stage): a block whose bench process
the stop ends is left unfinished, not crashed, and its lines go to that run
rather than to the retry pass; a line whose process crashed on its own
before the stop stays a crash. Every bench process starts with a PROCESS header line
naming its GPU, the host's boot, the run and the segment (stage04 stamps
them on the rows behind it). Each GPU's first attempt in the new run holds
its canary to the GPU's baseline from before (its own readings, or for a
GPU new to the stage those of the last run's GPUs), on either side: within
the tolerance the segment and its baseline go on; slower or faster beyond
it the GPU starts a new segment and that attempt, with the earlier part of
its block, moves to `drift/<log stem>_bR/`, the block benched again from its
start. A partial block of another segment than its GPU's moves to
`drift/<log stem>_sR/` and is benched from its start. The GPUs of a resume
must be of the data's model (arch, chip id, CU count), or the stage exits 2.
A run that failed (exit 2) is continued too, benching again only what
failed (`_reopen_after_failure`): its crash blocks after their last finished
shape, the crash-block count starting anew; the blocks it gave up, and on
each GPU the blocks finished after its last block whose canary passed, from
their start (`drift/<block>_fR/`). A run that passed stands with
--keep-passed (nothing to bench). The inputs a run must share with the one
it continues are its lines, the options that shape the benches, the bench
binary, the library files' identity and what of the config the benches
read (not the config's path, which the orchestrator snapshots anew). A
change of this stage's code starts anew unless --resume-across-code-change
(with --note) lets the run go on, recording the change; every PROCESS
header names the code. Outputs a run does not continue move to
`superseded/<utc>/`, never deleted.

Outputs in `<out-dir>/`, rewritten by every run that does not continue one:
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
                                     (`contended`, `health` or `wedged`),
                                     the load on the stage's cores while it
                                     ran, the run's hold
  probe/probe_NNNN_gpuD.log          the output of a GPU's device probe
  host_load.jsonl                    the load on the stage's cores, one
                                     window per line
  quarantine/<log stem>_qN/          the logs of a block's N-th quarantined
                                     attempt and quarantine.json: the rule, the
                                     other processes (pid, command, user, when
                                     seen, start time, the partitions of the
                                     card they held memory on when they held
                                     none on the GPU), the unattributed VRAM
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
                                     benched to the end by the retry pass),
                                     `wedged` (its process left its GPU
                                     wedged, in a block or the retry pass)
                                     or `quarantined_line` (hung, or left
                                     its GPU wedged, in an earlier stage of
                                     the run under the same --hung-scope)
  stage03_summary.json               counts (heavy lines among them),
                                     crash blocks, tolerance, rc, the
                                     quarantines, retired GPUs and the ones
                                     that returned, the processes whose
                                     output did not line up with their
                                     entries (`misaligned`), the guard's
                                     record (`guard`: wedged GPUs, the
                                     lines that left their GPU wedged,
                                     device probes), per block its GPU, that
                                     GPU's settings, its health events and
                                     its canary readings, the canary (its
                                     line and
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
  resume/journal.json, resume/units/ the resume journal: the inputs' digest,
                                     the GPU model, the block layout, each
                                     run (boot, GPUs, segments, canary
                                     verdicts) and how the stage ended; each
                                     block's state (lib/bench_resume.py)
  superseded/<utc>/                  the outputs of an earlier invocation the
                                     stage did not continue

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
blocks is within --max-crash-blocks (default: one in twenty, rounded down,
of the blocks --blocks-per-gpu alone would make, whatever --min-block-lines
says), else 2. Individually skipped shapes are reported but never fail it.
It also exits 2 when a block or the retry pass is quarantined more than
--max-quarantines times or stopped by the guard more than MAX_ABORTS times,
when the guard stopped waiting (`bench.quiet_host.max_wait_s`, or
`bench.health.max_wait_s` while the run held every GPU), when the GPU
numbering is wrong, when a bench process's output does not line up with
its entries, or when blocks are left because every GPU stopped taking
them; 2 too for a refused lease or platform check. It exits 3 when it
stopped for a wedge (see Wedged GPUs) and 1 without a bench when the run
quarantined every line under its --hung-scope.

Every bench process leads its own process group. SIGINT, SIGTERM or SIGHUP
stops all of them (`lib.procs`) and the stage exits with 128 + the signal
number. The device workers start no process once the stage is stopping, and
the stage waits for them to end before it removes its bench yamls'
directory (`tmp/<pid>` of --out-dir).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import heapq
import itertools
import json
import math
import os
import queue
import re
import shutil
import signal
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
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import bench_resume  # noqa: E402
from lib import bench_yaml  # noqa: E402
from lib import fingerprint  # noqa: E402
from lib import gpu_health  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import hazards  # noqa: E402
from lib import host_load  # noqa: E402
from lib import kfd  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import node_faults  # noqa: E402
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
    fatal_marker,
    iter_problems,
)
from lib.bench_yaml import (  # noqa: E402
    ADAPTIVE_KEYS,
    CANARY_DEFAULTS,
    adaptive_line,
    canary_line,
    canary_needs_kernel,
    format_solution_list,
    heavy_line,
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
    # the stage's own stop cut the last attempt off: the lines from
    # `next_offset` on stay for the run that resumes the stage
    stopped: bool = False
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
    # lines whose memory access fault followed them (lib/node_faults.py):
    # out of `lines` for good, quarantined in the run (their records)
    faulted: List[Dict[str, Any]] = field(default_factory=list)
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
    # lines with an operand of bench.heavy_operand_gib GiB or more alone
    heavy: bool = False
    # what wedged the last attempt's GPU (`_wedge_of`), and why its GPU
    # benches its device probe before its next block (`_Guard.probe_due`)
    wedged: Optional[Dict[str, Any]] = None
    health_events: List[str] = field(default_factory=list)
    # lines a process crashed in and left its GPU wedged (`_check_wedge`): out
    # of `lines` for good, without data in this stage (their records)
    wedge_lines: List[Dict[str, Any]] = field(default_factory=list)
    # the segment of the attempt's data (lib/bench_resume.py): its device's
    # when the attempt starts, which a later run continues
    segment: Optional[int] = None
    # when the stage finished the block (epoch seconds), which orders the
    # blocks of a GPU after a run that failed (`_reopen_after_failure`)
    finished: Optional[float] = None

    def reset(self) -> None:
        """Forget the last attempt, so that the block is benched anew."""
        self.segment = None
        self.finished = None
        self.device = None
        self.wall_s = 0.0
        self.rc = 0
        self.next_offset = 0
        self.skipped = {}
        self.abandoned = False
        self.stopped = False
        self.recovered = set()
        self.contaminated = None
        self.aborted = None
        self.misaligned = []
        self.canary_readings = []
        self.canary = {}
        self.host_load = {}
        self.wedged = None

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
    lines: List[str],
    block_size: int,
    numbers: Optional[Sequence[int]] = None,
    first_id: int = 0,
    heavy: bool = False,
) -> List[TaskBlock]:
    """Blocks of `block_size` of `lines`, whose 1-based source-yaml line
    numbers are `numbers` (default: their positions), numbered from
    `first_id`, marked `heavy` (lines with a large operand alone)."""
    numbering = list(numbers) if numbers is not None else list(range(1, len(lines) + 1))
    blocks: List[TaskBlock] = []
    for i, start in enumerate(range(0, len(lines), block_size)):
        bi = first_id + i
        chunk = lines[start : start + block_size]
        numbers = numbering[start : start + len(chunk)]
        blocks.append(
            TaskBlock(
                block_id=bi,
                name=f"block_{bi:04d}_lines{numbers[0]}-{numbers[-1]}",
                lines=chunk,
                line_numbers=numbers,
                heavy=heavy,
            )
        )
    return blocks


def _min_block_size(
    block_size: int, n_lines: int, n_gpus: int, min_block_lines: int
) -> int:
    """`block_size` raised to `min_block_lines`, so that `n_lines` lines
    make at most ceil(n_lines / min_block_lines) blocks, but to no more than
    ceil(n_lines / n_gpus): a minimum never makes blocks larger than one
    block per GPU would. A minimum of 1 keeps `block_size`."""
    per_gpu = math.ceil(n_lines / max(1, n_gpus))
    return max(block_size, min(int(min_block_lines), per_gpu))


def _blocks_heavy_last(
    lines: List[str],
    numbers: Sequence[int],
    n_gpus: int,
    blocks_per_gpu: int,
    heavy_gib: float,
    min_block_lines: int = 1,
) -> Tuple[List[TaskBlock], int, int]:
    """The blocks of a stage: `lines` (source-yaml line numbers `numbers`)
    in about `blocks_per_gpu` blocks per GPU, those with an operand of
    `heavy_gib` GiB or more (`heavy_line`) in blocks of their own after the
    others, at least one per GPU where there are enough of them. With one
    block per GPU, or `heavy_gib` 0, heavy lines stay among the others.
    The blocks of the other lines hold at least `min_block_lines` lines
    (`_min_block_size`); the heavy blocks are cut as without it.
    (blocks, block size of the other lines, heavy lines)."""
    target = max(n_gpus, n_gpus * int(blocks_per_gpu))
    block_size = max(1, math.ceil(len(lines) / target))
    split = heavy_gib > 0 and target > n_gpus
    heavy = [i for i, ln in enumerate(lines) if split and heavy_line(ln, heavy_gib)]
    if not heavy:
        size = _min_block_size(block_size, len(lines), n_gpus, min_block_lines)
        return _build_blocks(lines, size, numbers), size, 0
    marked = set(heavy)
    light = [i for i in range(len(lines)) if i not in marked]
    size = _min_block_size(block_size, len(light), n_gpus, min_block_lines)
    blocks = _build_blocks([lines[i] for i in light], size, [numbers[i] for i in light])
    heavy_size = max(1, min(block_size, math.ceil(len(heavy) / max(1, n_gpus))))
    blocks += _build_blocks(
        [lines[i] for i in heavy],
        heavy_size,
        [numbers[i] for i in heavy],
        first_id=len(blocks),
        heavy=True,
    )
    return blocks, size, len(heavy)


def default_crash_blocks(
    lines: List[str],
    numbers: Sequence[int],
    n_gpus: int,
    blocks_per_gpu: int,
    heavy_gib: float,
) -> int:
    """The crash blocks a stage tolerates by default: one in twenty, rounded
    down, of the blocks `blocks_per_gpu` alone cuts its lines into, so that
    `--min-block-lines`, which makes fewer and larger blocks, never lowers
    it."""
    blocks, _, _ = _blocks_heavy_last(lines, numbers, n_gpus, blocks_per_gpu, heavy_gib)
    return int(0.05 * len(blocks))


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


def _fatal_since(*parts: Tuple[Optional[Path], int]) -> Optional[str]:
    """The first fatal GPU error (lib/bench_log.py FATAL_GPU_MARKERS) the
    logs of `parts`, (log, byte offset) pairs, hold from their offsets on."""
    for path, start in parts:
        if path is not None:
            marker = fatal_marker(_read_from(path, start))
            if marker is not None:
                return marker
    return None


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


# ── wedged GPUs ──────────────────────────────────────────────────────────────

# Seconds a bench process that was killed, or failed without finishing a
# shape, gets to end before the stage looks for processes stuck in the GPU
# driver.
REAP_S = hazards.REAP_S
# Readings, hazards.STUCK_INTERVAL_S apart, that must all find a process in
# uninterruptible sleep before it counts as stuck: a GPU it wedges is held
# until the host reboots, so a driver call that is only slow must not.
WEDGE_SAMPLES = 10
PROBE_DIR = "probe"
# Health canaries in a row whose bench faults on a GPU (`bench_fault`
# against it) before the GPU leaves the stage: the run's canary lines and
# kernels are fixed, so a canary kernel that faults on its shape does so
# every time, and the stage must not wait for it for good.
CANARY_FAULT_LIMIT = 3
CANARY_FAULT_REMEDY = (
    "a canary that faults on every GPU points at its own kernel rather than "
    "the GPUs, and only turning the health canary off "
    "(bench.health.canary.enabled: false) lets the stage bench past it: "
    "bench.health.max_wait_s only makes a GPU leave sooner, and "
    "scripts/run_guard.py accept and a reboot keep the same canary"
)


def _reaped(proc: Any, rc: int, wait_s: float) -> bool:
    """Whether `proc` ended, waiting up to `wait_s` for it; False at once for
    one left running because it stayed stuck in a driver call through the
    kill's own wait (procs.STUCK_RC, procs.STUCK_WAIT_S)."""
    if rc == procs.STUCK_RC:
        return False
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return True
    return True


def _gpu_of(device: int) -> Optional[kfd.Gpu]:
    """The KFD GPU of `device` in this process's environment, None when it
    cannot be named."""
    try:
        physical = kfd.physical_gpu(device)
    except kfd.GpuMapError:
        return None
    return next((g for g in kfd.gpus() if g.index == physical), None)


def _stuck_processes() -> List[Tuple[int, str]]:
    """lib/hazards.py `stuck_processes`, where KFD names processes by this
    process's pids (kfd.pids_comparable), over WEDGE_SAMPLES readings; none
    elsewhere, where a pid it lists may be another process here."""
    if not kfd.pids_comparable():
        return []
    return hazards.stuck_processes(samples=WEDGE_SAMPLES)


def _lasting(pids: Sequence[int], window_s: float) -> Set[int]:
    """The `pids` in uninterruptible sleep in every reading over `window_s`,
    hazards.STUCK_INTERVAL_S apart (lib/procs.py `stuck_in_driver`): a
    process that is only slow in a driver call, such as one whose teardown
    is still ending, leaves it within REAP_S."""
    interval = hazards.STUCK_INTERVAL_S
    samples = 1 + max(0, math.ceil(float(window_s) / interval))
    return set(procs.stuck_in_driver(pids, samples=samples, interval_s=interval))


def _wedge_of(
    device: int,
    proc: Any,
    rc: int,
    gpus: Mapping[int, Optional[kfd.Gpu]],
    baseline: AbstractSet[int] = frozenset(),
    reap_s: Optional[float] = None,
    elsewhere: Optional[List[Dict[str, Any]]] = None,
    stopping: Callable[[], bool] = lambda: False,
) -> Optional[Dict[str, Any]]:
    """What a bench process on `device` that was killed (or left running,
    stuck: procs.STUCK_RC), or failed without finishing a shape, leaves
    wedged, once it had `reap_s` (default REAP_S) to end: the processes the
    GPU driver lists stuck in uninterruptible sleep (lib/hazards.py
    `stuck_processes`, those of `baseline`, stuck since before the stage or
    found by an earlier check, left out unless they are the bench's) and the
    bench itself when it did not end. The bench's count against its
    device's GPU, any other against the stage's GPUs it holds memory on
    (`gpus` names the stage's devices' GPUs), and one the stage started
    (`procs.descendants`) that holds memory on no GPU against the host;
    such another process counts only when it stays stuck through another
    `reap_s` (`_lasting`), since a slow driver call, or the teardown of a
    process that is ending, does not. A process
    stuck only on other GPUs, or on none, that the stage did not start
    counts for nothing (another run's wedge is not this stage's) and is
    added to `elsewhere`, with the GPUs it holds memory on. The record:
    `gpus` {key: {gpu, processes}}, `host` [processes], whether the bench
    ended (`reaped`); None when nothing is wedged, and when `stopping()`
    says the stage is stopping, whose own kills prove nothing."""
    if stopping():
        return None
    wait_s = REAP_S if reap_s is None else float(reap_s)
    reaped = _reaped(proc, rc, wait_s)
    pid = getattr(proc, "pid", None)
    own: Set[int] = set()
    if not reaped and isinstance(pid, int):
        own = set(procs.session_members(pid)) | {pid}
    elif isinstance(pid, int):
        own = {pid}
    stuck = [
        (p, comm) for p, comm in _stuck_processes() if p in own or p not in baseline
    ]
    if (reaped and not stuck) or stopping():
        return None
    by_id = {g.gpu_id: g for g in kfd.gpus()}
    users = kfd.memory_users() if any(p not in own for p, _ in stuck) else {}
    stage_keys = {gpu_health.gpu_key(g) for g in gpus.values() if g is not None}
    mine = gpus.get(device)
    mine_key = gpu_health.gpu_key(mine) if mine is not None else f"device {device}"
    found: Dict[str, Dict[str, Any]] = {}
    host: List[Dict[str, Any]] = []
    started: Optional[Set[int]] = None
    # the other processes that count, if they stay stuck: each with the
    # stage's GPUs it holds memory on, [] for the host
    counted: List[Tuple[int, str, List[kfd.Gpu]]] = []

    def add(key: str, gpu: Optional[int], who: Dict[str, Any]) -> None:
        entry = found.setdefault(key, {"gpu": gpu, "processes": []})
        if all(w["pid"] != who["pid"] for w in entry["processes"]):
            entry["processes"].append(who)

    if not reaped and isinstance(pid, int):
        add(
            mine_key,
            mine.index if mine is not None else None,
            {"pid": pid, "comm": "bench", "ended": False},
        )
    for p, comm in stuck:
        if p in own:
            add(
                mine_key,
                mine.index if mine is not None else None,
                {"pid": p, "comm": comm},
            )
            continue
        held = sorted(
            (
                by_id[i]
                for i, used in users.get(p, {}).items()
                if used > 0 and i in by_id
            ),
            key=lambda g: g.index,
        )
        here = [g for g in held if gpu_health.gpu_key(g) in stage_keys]
        if here:
            counted.append((p, comm, here))
            continue
        if not held:
            if started is None:
                started = set(procs.descendants(os.getpid()))
            if p in started:
                counted.append((p, comm, []))
                continue
        if elsewhere is not None:
            elsewhere.append({"pid": p, "comm": comm, "gpus": [g.index for g in held]})
    lasting = _lasting([p for p, _, _ in counted], wait_s) if counted else set()
    if stopping():
        return None
    for p, comm, here in counted:
        if p not in lasting:
            continue
        if not here:
            host.append({"pid": p, "comm": comm})
        for g in here:
            add(gpu_health.gpu_key(g), g.index, {"pid": p, "comm": comm})
    if not found and not host:
        return None
    return {
        "device": device,
        "gpu": mine_key,
        "reaped": reaped,
        "gpus": found,
        "host": host,
        "at": kfd.utc_iso(),
    }


def _who(processes: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(
        f"pid {p['pid']} ({p['comm']}"
        + (", did not end" if p.get("ended") is False else "")
        + ")"
        for p in processes
    )


def _wedge_why(info: Mapping[str, Any]) -> str:
    return str(
        info.get("why")
        or f"{_who(info.get('processes') or [])} stuck in the GPU driver"
    )


def describe_wedge(rec: Mapping[str, Any]) -> str:
    """One line of what a wedge record (`_wedge_of`) found."""
    parts = [
        f"GPU {info.get('gpu')} ({key}): {_wedge_why(info)}"
        for key, info in sorted((rec.get("gpus") or {}).items())
    ]
    if rec.get("host"):
        parts.append(
            f"the host: {_who(rec['host'])} stuck in the GPU driver, holding "
            f"memory on no GPU"
        )
    return "wedged " + "; ".join(parts)


def own_wedge(rec: Optional[Mapping[str, Any]]) -> bool:
    """Whether a wedge record (`_wedge_of`) holds the GPU of the process it
    was taken after (its `gpu` among its `gpus`): only then are that process
    and the line it ran part of the wedge. A record of other GPUs, or of the
    host alone, is held in the run all the same (`_Guard.wedged`), but says
    nothing against that process's GPU or line."""
    return rec is not None and str(rec.get("gpu")) in (rec.get("gpus") or {})


def _spread_text(keys: Sequence[str], what: str) -> str:
    """Why a stage stops when two or more of its GPUs are wedged (`keys`),
    the last one by `what`."""
    return f"{len(keys)} GPUs of the stage are wedged ({', '.join(keys)}): {what}"


def _check_wedge(
    block: TaskBlock,
    device: int,
    proc: Any,
    rc: int,
    guard: Optional["_Guard"],
    log_path: Path,
    children: procs.ChildGroups,
) -> Optional[Dict[str, Any]]:
    """After a bench process of `block` on `device` was killed, or failed
    without finishing a shape: what it left wedged (`_wedge_of`), with
    `guard` held in the run (`_Guard.wedged`), when that holds `device`'s
    GPU (`own_wedge`), noted in the block and its log. A wedge of other
    GPUs, or of the host alone, is held all the same and noted in the log,
    but the process is a crash like any other: None. None too when nothing
    is wedged, or once `children` are being stopped (the stage is
    stopping)."""
    if guard is not None:
        rec = guard.check_wedge(device, proc, rc)
    else:
        rec = _wedge_of(
            device,
            proc,
            rc,
            {device: _gpu_of(device)},
            stopping=lambda: children.stopped,
        )
        if rec is not None:
            ui.err("wedge", f"{block.name} on gpu{device}: {describe_wedge(rec)}")
    if rec is None:
        return None
    if not own_wedge(rec):
        _annotate(
            log_path,
            f"\n*** WEDGE ELSEWHERE: {describe_wedge(rec)}; the run holds what it "
            f"names, and gpu{device}'s process counts as a crash ***\n",
        )
        return None
    block.wedged = rec
    _annotate(
        log_path,
        f"\n*** WEDGED: {describe_wedge(rec)}; no new bench process on gpu{device} "
        f"***\n",
    )
    return rec


def _leave_out_wedging_line(
    block: TaskBlock,
    offset: int,
    rc: int,
    device: int,
    problem: Optional[Problem],
    guard: Optional["_Guard"],
    log_path: Path,
    warmup: bool = False,
) -> None:
    """Take the line at `offset` of `block` out of it: its bench process on
    `device` (exit status `rc`) left its own GPU wedged in it (`problem`,
    its open problem; with `warmup`, in its warm-up shape). No GPU benches
    it again in the stage, and `guard` quarantines it in the run, as a hung
    line (`_Guard.wedging_line`), so that no later stage of its protocol
    benches it either: a line that wedges every GPU it runs on would wedge
    them one by one."""
    line = block.lines[offset]
    shape = shape_of_line(line)
    m, n, k, b = shape if shape is not None else ("?", "?", "?", "?")
    gone = {
        "yaml_line": block.line_numbers[offset],
        "shape": list(shape) if shape is not None else None,
        "rc": rc,
        "device": device,
    }
    del block.lines[offset], block.line_numbers[offset]
    block.wedge_lines.append(gone)
    fate = "no GPU benches it again in this stage"
    if guard is not None:
        guard.wedging_line(gone, block.name, line, problem, device, warmup=warmup)
        if gone.get("quarantined"):
            fate = f"quarantined in the run ({guard.hung_scope})"
    where = " in its warm-up shape" if warmup else ""
    _annotate(
        log_path,
        f"\n*** LEFT OUT WEDGING GEMM at block-offset {offset} (yaml line "
        f"{gone['yaml_line']}): M={m} N={n} K={k} B={b} (rc={rc}); its process "
        f"left the GPU wedged{where}; {fate} ***\n",
    )
    ui.warn(
        "wedge",
        f"{block.name} on gpu{device}: line {gone['yaml_line']} (M={m} N={n} "
        f"K={k} B={b}) left its GPU wedged{where}: {fate}",
    )


def _health_event(
    block: TaskBlock,
    device: int,
    guard: Optional["_Guard"],
    log_path: Path,
    why: str,
) -> None:
    """`device` benches its device probe before its next block (`why`:
    what its last bench process of `block` did)."""
    block.health_events.append(why)
    _annotate(
        log_path,
        f"\n*** GPU HEALTH EVENT: {why}; gpu{device} benches its device probe "
        f"before its next block ***\n",
    )
    if guard is not None:
        guard.probe_due(device, why)


def _signal_of(rc: Optional[int]) -> Optional[signal.Signals]:
    """The signal a return code reports a death by (-N from subprocess,
    128 + N from a shell), None for any other."""
    if rc is None or rc == procs.STUCK_RC:
        return None
    n = -rc if rc < 0 else rc - 128
    try:
        return signal.Signals(n) if n > 0 else None
    except ValueError:
        return None


class BenchFault(NamedTuple):
    """How a bench process of a GPU's health canary or device probe failed
    (`what` it did), and whether that tells against the GPU (`gpu`)."""

    what: str
    gpu: bool


def bench_fault(
    rc: Optional[int],
    reason: Optional[str],
    problems: Sequence[Problem],
    text: str,
) -> Optional[BenchFault]:
    """How a bench process of a GPU's health canary or device probe (exit
    status `rc`, the watchdog's `reason`, the `problems` and `text` of its
    log) failed. Against the GPU (`gpu`): it is stuck in the GPU driver,
    reported a fatal GPU error, or hung or printed nothing until the
    watchdog killed it. Not against it: it crashed by a signal other than
    the kill signals, or exited non-zero before any problem, without a fatal
    GPU error, as a bench binary, yaml or library that fails does on any
    GPU. None when it did none of these (it ran through, was stopped from
    outside, or failed otherwise once its output began)."""
    if rc == procs.STUCK_RC:
        return BenchFault("is stuck in the GPU driver", True)
    marker = fatal_marker(text)
    if reason == "fatal" or marker is not None:
        return BenchFault(
            f"reported a fatal GPU error ({marker or 'a fatal error'})", True
        )
    if reason == "stall":
        return BenchFault("hung" if problems else "printed nothing", True)
    if rc in (None, 0):
        return None
    sig = _signal_of(rc)
    if sig is not None:
        if sig in procs.KILL_SIGNALS:
            return None
        return BenchFault(f"crashed ({sig.name})", False)
    return None if problems else BenchFault(f"printed no problem (rc={rc})", False)


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
    on_subrun: Optional[Callable[[TaskBlock, Path, bool], None]] = None,
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
    (default: the log's directory); once `children` are being stopped (the
    stage is stopping) no process starts and the block ends with its lines
    left to the run that resumes the stage (`TaskBlock.stopped`), and a
    process the stop ends leaves the line it was in to that run too, where
    a process that crashes on its own has its line skipped. With
    `watch`, a process that finds another process on the GPU ends the
    attempt with `block.contaminated` set."""
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
        if subrun > 0 and guard is not None and guard.wedged_stop is not None:
            # A wedge stops the stage (one this block's last process saw
            # elsewhere, or another device's): no new process starts.
            block.aborted = {
                "reason": "wedged",
                "subrun": subrun + 1,
                "why": guard.wedged_stop,
            }
            _annotate(
                log_path, f"\n*** ABORTED [wedged]: {GUARD_ABORTS['wedged']} ***\n"
            )
            break
        if children.stopped:
            # No sub-run starts, and none writes its bench yaml into
            # `work_dir`, which the stage removes as it ends.
            block.stopped = True
            _annotate(
                log_path,
                f"\n*** STOPPED: {STOPPING}; no new bench process; the lines from "
                f"block-offset {block.next_offset} on are benched when the run "
                f"resumes ***\n",
            )
            break
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
        if on_subrun is not None:
            # The resume journal's checkpoint and the process's header line.
            on_subrun(block, log_path, bool(canary_lines))
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix=f"{block.name}_", dir=work_dir, delete=False
        ) as f:
            tmp_yaml = Path(f.name)
            f.writelines(_entries(warmup_shape, canary_lines, remaining, numbers, lead))
        sub_start = file_size(log_path)
        warmup_start = file_size(warmup_log_path)
        canary_start = file_size(canary_log_path)
        progress = LogProbe(log_path, sub_start)
        bench_device, env, pin = _bench_launch(device)
        cmd = [str(bench_binary), "--device", bench_device, "--yaml", str(tmp_yaml)]
        reason: Optional[str] = None
        ended: Optional[float] = None
        proc: Optional[subprocess.Popen] = None
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
                        preexec_fn=pin,
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
                    if last_rc != procs.STUCK_RC:
                        drain.join(timeout=30)
                    # A process that has not ended may hold its pipe open
                    # for good, and closing it would wait for the thread
                    # reading it.
                    if not drain.is_alive():
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
                        preexec_fn=pin,
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
        text = _read_from(log_path, sub_start)
        shape_problems = list(iter_problems(text.splitlines()))
        if proc is not None and last_rc != 0 and children.ended_by_stop(proc):
            # The stage's own stop ended the process, as an interruption
            # would: the line it was in is no crash, and it and the lines
            # after it stay for the run that resumes the stage.
            finished = min(count_finished_problems(text.splitlines()), len(remaining))
            block.next_offset += finished
            block.stopped = True
            _annotate(
                log_path,
                f"\n*** STOPPED at block-offset {block.next_offset}: {STOPPING}; "
                f"its bench process ended (rc={last_rc}) and the lines from here "
                f"on are benched when the run resumes ***\n",
            )
            break
        wedge: Optional[Dict[str, Any]] = None
        if proc is not None and (
            reason is not None or (last_rc != 0 and _n_finished(shape_problems) == 0)
        ):
            wedge = _check_wedge(
                block, device, proc, last_rc, guard, log_path, children
            )

        def end_wedged(n: int) -> None:
            block.aborted = {"reason": "wedged", "subrun": n, "wedge": wedge}
            _annotate(
                log_path, f"\n*** ABORTED [wedged]: {GUARD_ABORTS['wedged']} ***\n"
            )

        def wedged_in() -> Optional[Tuple[int, Problem, bool]]:
            """Where this process, which left its own GPU wedged, was: the
            block-offset of the line it was in, that line's open problem,
            and whether it was in the warm-up shape, the shape of the
            block's next line over all its kernels (`render_warmup_line`);
            None when it was in no line: in its start-up, the warm-up line
            or the canary, or between two lines."""
            done = _n_finished(shape_problems)
            if done < len(shape_problems) and done < len(remaining):
                return block.next_offset + done, shape_problems[done], False
            if shape_problems or not warmup_shape or warmup_log_path is None:
                return None
            if canary_lines and canary_log_path is not None:
                later = _read_from(canary_log_path, canary_start).splitlines()
                if next(iter_problems(later), None) is not None:
                    return None
            warm = list(
                iter_problems(_read_from(warmup_log_path, warmup_start).splitlines())
            )
            if len(warm) != (2 if lead else 1) or warm[-1].finished:
                return None
            return block.next_offset, warm[-1], True

        if watch is not None and (reason == "contaminated" or watch.check_end(ended)):
            block.contaminated = watch.record()
            _annotate(
                log_path,
                f"\n*** QUARANTINED [{block.contaminated['rule']}]: "
                f"{block.contaminated['reason']} ***\n",
            )
            break
        if reason in GUARD_ABORTS:
            at = wedged_in() if wedge is not None else None
            if at is not None:
                # The process the guard stopped is stuck in this line on its
                # own GPU: the GPU that takes the block next would wedge on
                # it too.
                _leave_out_wedging_line(
                    block, at[0], last_rc, device, at[1], guard, log_path, at[2]
                )
            block.aborted = {"reason": reason, "subrun": subrun + 1}
            if wedge is not None:
                block.aborted["wedge"] = wedge
            _annotate(
                log_path, f"\n*** ABORTED [{reason}]: {GUARD_ABORTS[reason]} ***\n"
            )
            break
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
            if wedge is not None:
                end_wedged(subrun + 1)
                break
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
            if wedge is not None:
                end_wedged(subrun + 1)
                break
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
                if wedge is not None:
                    end_wedged(subrun + 1)
                    break
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
            head = "KILLED" if last_rc != procs.STUCK_RC else "LEFT RUNNING"
            if last_rc == procs.STUCK_RC:
                what += ", but it is stuck in the GPU driver: no signal sent"
            _annotate(log_path, f"\n*** {head} [{reason}]: {what} ***\n")
        subrun += 1
        if last_rc == 0:
            block.next_offset = n_total
            break
        finished = min(count_finished_problems(text.splitlines()), len(remaining))
        zero_progress = zero_progress + 1 if finished == 0 else 0
        crash_idx = block.next_offset + finished
        if crash_idx >= n_total:
            # Every shape finished before the non-zero exit, so the fault came
            # from process teardown and no data was lost. A process the
            # watchdog had to kill, or one that reported a fatal GPU error,
            # may have left its GPU unable to bench.
            last_rc = 0
            block.next_offset = n_total
            fatal = _fatal_since(
                (log_path, sub_start),
                (canary_log_path, canary_start),
                (warmup_log_path, warmup_start),
            )
            if wedge is None and (reason is not None or fatal is not None):
                why = (
                    f"a bench process that finished every shape reported a fatal "
                    f"GPU error ({fatal})"
                    if fatal is not None
                    else "a bench process that finished every shape did not exit "
                    f"(killed: {reason})"
                )
                _health_event(block, device, guard, log_path, why)
            break
        open_problem = (
            shape_problems[finished] if finished < len(shape_problems) else None
        )
        at = wedged_in() if wedge is not None else None
        if wedge is not None and at is None:
            # The process wedged its GPU before it started the line after the
            # ones it finished (in its start-up, such as the library load, in
            # the warm-up line or the canary, or between two lines): no line
            # is to blame for it.
            _annotate(
                log_path,
                f"\n*** WEDGED BEFORE BLOCK-OFFSET {crash_idx}: its process started "
                f"no line there (rc={last_rc}); no line leaves the block ***\n",
            )
            end_wedged(subrun)
            break
        line = block.lines[crash_idx]
        shape = shape_of_line(line)
        m, n, k, b = shape if shape is not None else ("?", "?", "?", "?")
        fault = None
        if guard is not None and reason != "stall":
            fault = guard.node_fault(
                block, crash_idx, device, text, log_path, wedge, bench_binary
            )
        if fault is not None and fault["decision"] == FAULT_STOP:
            block.aborted = {"reason": "node_fault", "subrun": subrun, "fault": fault}
            block.next_offset = crash_idx
            _annotate(
                log_path,
                f"\n*** ABORTED [node fault] at block-offset {crash_idx} (yaml line "
                f"{block.line_numbers[crash_idx]}): M={m} N={n} K={k} B={b}, "
                f"{fault['reason']}; the node is retired ***\n",
            )
            break
        if fault is not None:
            del block.lines[crash_idx], block.line_numbers[crash_idx]
            n_total = len(block.lines)
            block.faulted.append(fault)
            block.next_offset = crash_idx
            _annotate(
                log_path,
                f"\n*** QUARANTINED FAULTING LINE at block-offset {crash_idx} (yaml "
                f"line {fault['yaml_line']}): M={m} N={n} K={k} B={b}, "
                f"{fault['reason']}; never benched again in this run ***\n",
            )
        elif reason == "stall":
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
        elif at is not None:
            _leave_out_wedging_line(
                block, crash_idx, last_rc, device, at[1], guard, log_path, at[2]
            )
            n_total = len(block.lines)
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
        if wedge is not None:
            if block.next_offset < n_total:
                end_wedged(subrun)
            break
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
            if zero_progress >= _MAX_ZERO_PROGRESS_SUBRUNS:
                _health_event(
                    block,
                    device,
                    guard,
                    log_path,
                    f"{zero_progress} bench processes in a row finished no GEMM",
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
        tolerance_us: float = 0.0,
    ) -> None:
        self.line = line.rstrip("\n") + "\n"
        self.shape = shape_of_line(self.line)
        self.timings = int(timings)
        self.tolerance = float(tolerance)
        self.tolerance_us = float(tolerance_us)
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
        # The stage's run and each device's GPU (lib/bench_resume.py), which
        # tag every reading; per device, the baseline of an earlier run its
        # first judged attempt is held against (`boundary`), and the
        # readings of every earlier run, per (GPU, segment), that judge the
        # attempts of those runs at the stage's end (`reference_of`).
        self.run = 1
        self.gpus: Dict[int, str] = {}
        self.baselines: Dict[int, Dict[str, Any]] = {}
        self.earlier: Dict[Tuple[str, int], List[float]] = {}
        self.boundaries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def describe(self) -> str:
        m, n, k, b = self.shape or ("?", "?", "?", "?")
        return (
            f"M={m} N={n} K={k} B={b}, {self.timings} timings before and after "
            f"each block's shapes, tolerance {self.limit_text()}, back-off "
            f"{ui.fmt_dur(self.backoff_s)}"
        )

    def limit_text(self) -> str:
        """The tolerance, with its absolute floor when it has one."""
        return f"{self.tolerance:g}" + (
            f" (at least {self.tolerance_us:g} us)" if self.tolerance_us > 0 else ""
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

    def judge(
        self,
        device: int,
        values: Sequence[float],
        add: Optional[Sequence[float]] = None,
    ) -> Tuple[Optional[float], List[float]]:
        """Add `values`, the readings of one attempt on `device`, to the
        GPU's (`add` instead: those of them this run took); (its reference
        with them, the values over it by more than the tolerance), (None,
        []) while the GPU has none."""
        with self._lock:
            taken = self.readings.setdefault(device, [])
            taken.extend(values if add is None else add)
            if not taken:
                return None, []
            reference = statistics.median(taken)
        return reference, self.over(reference, values)

    def tags(self, device: int, block: "TaskBlock") -> Dict[str, Any]:
        """What a reading of `block` on `device` names: the stage's run, the
        GPU and the segment of the block's data."""
        return {
            "run": self.run,
            "gpu": self.gpus.get(device, f"device{device}"),
            "segment": block.segment,
        }

    def restore(
        self, device: int, segment: int, values: Sequence[float], gpus: Sequence[str]
    ) -> None:
        """The baseline of an earlier run `device` benches on: `values`, the
        readings of `segment` on `gpus` (its own GPU, or those of the last
        run for a GPU new to the stage)."""
        with self._lock:
            self.readings[device] = list(values)
            self.baselines[device] = {
                "segment": segment,
                "reference": statistics.median(values),
                "n_readings": len(values),
                "gpus": list(gpus),
            }

    def boundary(self, device: int, fresh: Sequence[float]) -> Optional[Dict[str, Any]]:
        """The verdict of `fresh`, the readings this run's first judged
        attempt on `device` took, on the baseline it resumed (`restore`):
        held, within the canary's tolerance of the baseline's reference on
        either side (`outside`), the GPU benches on with that baseline;
        moved, slower or faster, its baseline starts anew, from the readings
        of the attempts it keeps from then on. None without a baseline still
        to judge."""
        with self._lock:
            base = self.baselines.get(device)
            if base is None or not fresh:
                return None
            del self.baselines[device]
        reference = base["reference"]
        out = self.outside(reference, fresh)
        direction = None
        if out:
            direction = "slower" if any(v > reference for v in out) else "faster"
        verdict = {
            **base,
            "device": device,
            "gpu": self.gpus.get(device),
            "readings": list(fresh),
            "drifted": out,
            "direction": direction,
            "held": not out,
        }
        with self._lock:
            if out:
                self.readings[device] = []
            self.boundaries.append(verdict)
        return verdict

    def reference_of(self, block: "TaskBlock") -> Optional[float]:
        """The reference the kept attempt of `block` is judged against at the
        stage's end: its GPU's in this run, or for an attempt of an earlier
        run, the readings of its baseline then."""
        last = next(
            (
                r
                for r in reversed(block.canary_readings)
                if r.get("start") is not None or r.get("end") is not None
            ),
            None,
        )
        if last is None or int(last.get("run", self.run)) == self.run:
            return self.reference(block.device) if block.device is not None else None
        with self._lock:
            taken = self.earlier.get(
                (str(last.get("gpu")), int(last.get("segment") or 0))
            )
            return statistics.median(taken) if taken else None

    def over(self, reference: float, values: Sequence[float]) -> List[float]:
        """The `values` above `reference` by more than the tolerance and by
        more than its absolute floor (`tolerance_us`)."""
        limit = max(reference * (1.0 + self.tolerance), reference + self.tolerance_us)
        return [v for v in values if v > limit]

    def outside(self, reference: float, values: Sequence[float]) -> List[float]:
        """The `values` farther from `reference` than the tolerance and its
        absolute floor, above it (`over`) or below it: the test across an
        interruption, after which a GPU may read either way (an AC cycle
        resets its clocks, thermal state and firmware)."""
        low = min(reference * (1.0 - self.tolerance), reference - self.tolerance_us)
        high = set(self.over(reference, values))
        return [v for v in values if v in high or v < low]

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
                "tolerance_us": self.tolerance_us,
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
                "boundaries": [dict(b) for b in self.boundaries],
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
    floor = float(pick(getattr(args, "canary_tolerance_us", None), "tolerance_us"))
    timings = int(pick(args.canary_timings, "timings"))
    backoff_s = float(pick(args.canary_backoff_s, "backoff_s"))
    if not tolerance > 0 or timings < 1 or not backoff_s >= 0:
        raise ValueError(
            f"the canary needs a tolerance above 0, at least one timing and a "
            f"back-off of at least 0 s (got {tolerance:g}, {timings} and "
            f"{backoff_s:g})"
        )
    if not floor >= 0:
        raise ValueError(
            f"the canary's tolerance_us must be at least 0 (got {floor:g})"
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
    return Canary(line, timings, tolerance, backoff_s, identity, floor), ""


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
            **canary.tags(device, block),
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


def _reading_values(block: TaskBlock, run: Optional[int] = None) -> List[float]:
    """The canary readings of `block`'s attempt, those of run `run` alone
    when given."""
    return [
        r[k]
        for r in block.canary_readings
        if run is None or int(r.get("run", run)) == run
        for k in ("start", "end")
        if r[k] is not None
    ]


def _judge_canary(
    block: TaskBlock,
    device: int,
    canary: Canary,
    logs: Sequence[Optional[Path]],
    drift_dir: Path,
    journal: Optional[bench_resume.Journal] = None,
) -> bool:
    """Judge the attempt of `block` that just ended on `device` by its canary
    readings (`Canary.judge`). On its first drift the attempt's `logs` move
    to `drift_dir/<log stem>_d1/` with drift.json and the block is reset:
    True, bench it again. A block that drifts again keeps its data and is
    flagged (`CANARY_FLAG_MARK` in its log, which stage04 marks its rows
    by). With `journal`, the first attempt a resumed run judges on a GPU
    first holds this run's readings against the GPU's baseline before the
    interruption (`Canary.boundary`): held, the GPU's segment goes on;
    moved, the GPU starts a new segment and the attempt, with any earlier
    part of the block, moves to `drift_dir/<log stem>_b<run>/` and the
    block is benched again from its start (`_rebench_at_boundary`)."""
    values = _reading_values(block)
    if not values:
        return False
    fresh = _reading_values(block, canary.run)
    verdict = canary.boundary(device, fresh) if journal is not None else None
    if verdict is not None and verdict["held"]:
        journal.note(device, verdict)
    elif verdict is not None:
        segment = journal.new_segment(device, verdict)
        return _rebench_at_boundary(
            block, device, canary, logs, drift_dir, verdict, segment
        )
    reference, high = canary.judge(device, values, add=fresh)
    if reference is None:
        return False
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
        f"than {canary.limit_text()} over its reference {reference:.4g} us"
    )
    if any(not d.get("boundary") for d in block.drifts):
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
            "tolerance_us": canary.tolerance_us,
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


def _rebench_at_boundary(
    block: TaskBlock,
    device: int,
    canary: Canary,
    logs: Sequence[Optional[Path]],
    drift_dir: Path,
    verdict: Mapping[str, Any],
    segment: int,
) -> bool:
    """The canary of a resumed run moved on `device` (`verdict`): the
    attempt's `logs`, with any part of the block an earlier run benched,
    move to `drift_dir/<log stem>_b<run>/` with drift.json naming the
    disagreement, and the block is benched again from its start in the
    GPU's new `segment`. True: bench it again."""
    log_path = Path(str(logs[0]))
    why = (
        f"canary {', '.join(f'{v:.4g}' for v in verdict['drifted'])} us on "
        f"gpu{device} after the stage's interruption, {verdict['direction']} by "
        f"more than {canary.limit_text()} than the reference "
        f"{verdict['reference']:.4g} us of its baseline before it (segment "
        f"{verdict['segment']}); segment {segment} starts with a baseline of its own"
    )
    _annotate(log_path, f"\n*** CANARY BOUNDARY: {why}; benched again ***\n")
    dest = drift_dir / f"{log_path.stem}_b{canary.run}"
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
            "tolerance_us": canary.tolerance_us,
            **({"kernel": canary.labels[device]} if device in canary.labels else {}),
            "boundary": dict(verdict),
            "segment": segment,
            "readings": [dict(r) for r in block.canary_readings],
        },
    )
    block.drifts.append({**rec, "dir": str(dest), "boundary": True})
    canary.note(block.name, flagged=False)
    ui.warn("canary", f"{block.name}: {why} -> {dest}; benched again")
    block.reset()
    return True


def _segment_aside(
    block: TaskBlock,
    device: int,
    logs: Sequence[Optional[Path]],
    drift_dir: Path,
    segment: Optional[int],
    run: int,
) -> None:
    """Before an attempt on `device` of a block an earlier run left partial
    in another segment than the one `device` benches (`segment`): its
    `logs` move to `drift_dir/<log stem>_s<run>/` with drift.json, and the
    block is benched from its start."""
    log_path = Path(str(logs[0]))
    why = (
        f"the part an earlier run benched is of segment {block.segment}, and "
        f"gpu{device} benches segment {segment}"
    )
    _annotate(log_path, f"\n*** SEGMENT BOUNDARY: {why}; benched again ***\n")
    dest = drift_dir / f"{log_path.stem}_s{run}"
    rec = _move_aside(
        dest,
        logs,
        DRIFT_RECORD,
        {
            "block": block.name,
            "yaml_lines": [block.line_numbers[0], block.line_numbers[-1]],
            "device": device,
            "reason": why,
            "boundary": {"segment": block.segment},
            "segment": segment,
            "readings": [dict(r) for r in block.canary_readings],
        },
    )
    block.drifts.append({**rec, "dir": str(dest), "boundary": True})
    ui.warn("resume", f"{block.name}: {why} -> {dest}; benched again from its start")
    block.reset()


def _carry_logs(paths: Sequence[Optional[Path]], device: int) -> List[str]:
    """Before the attempt on `device` of a block an earlier run left partial
    on another GPU: its logs there take the names of `paths`, its logs on
    `device`, so that the block's data stays in one log, each process named
    by its PROCESS header. The moves, in words."""
    moved: List[str] = []
    suffix = f"_gpu{device}.log"
    for p in paths:
        if p is None or p.exists() or not p.name.endswith(suffix):
            continue
        stem = p.name[: -len(suffix)]
        pattern = re.compile(re.escape(stem) + r"_gpu\d+\.log")
        found = [
            q for q in p.parent.glob(f"{stem}_gpu*.log") if pattern.fullmatch(q.name)
        ]
        if len(found) == 1:
            found[0].rename(p)
            moved.append(f"{found[0].name} -> {p.name}")
    return moved


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
        reference = canary.reference_of(b)
        if not values or reference is None:
            continue
        high = canary.over(reference, values)
        if not high:
            continue
        why = (
            f"canary {', '.join(f'{v:.4g}' for v in high)} us on gpu{b.device}, "
            f"more than {canary.limit_text()} over its reference at the stage's "
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
        "other load on the stage's cores, or the run's benches running more "
        "threads than their CPUs; the block is benched again once they are quiet"
    ),
    "health": (
        "the run holds the work of the block's GPU; the block is benched again "
        "on a GPU the run does not hold, once it reads its canary"
    ),
    "wedged": (
        "the block's GPU is wedged, or a wedge stops the stage (processes stuck "
        "in the GPU driver); the block goes back on the work queue for a GPU "
        "that is not wedged"
    ),
}
ABORTED_DIR = "aborted"
ABORTED_RECORD = "aborted.json"
MAX_ABORTS = 10
HOLD_READ_S = 2.0
STOPPING = "the stage is stopping"
# What the fault policy (lib/node_faults.py, `_Guard.node_fault`) decided:
# the node is retired and the stage stops, or the line is quarantined in
# the run (under FAULT_WHY); and where the stage keeps the faults' records.
FAULT_STOP = "stop"
FAULT_LINE = "line"
FAULT_WHY = "fault"
FAULTS_DIR = "faults"


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
        if g.wedged_stop is not None:
            return "wedged"
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
        faults: Optional[node_faults.NodeStore] = None,
    ) -> None:
        from lib.contexts import wrapped_bench

        self.quiet = quiet
        self.leave_held = bool(leave_held)
        # The store of node faults the fault policy marks (None: the policy
        # is off), and the faults the stage met.
        self.faults = faults
        self.fault_records: List[Dict[str, Any]] = []
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
        # Wedged GPUs: the processes stuck in the GPU driver when the stage
        # started (left out of what a later check blames), what each check
        # found wedged, the stuck processes it found that count for nothing
        # here (`_wedge_of` `elsewhere`), why the stage stopped for a wedge,
        # the keys of the stage's GPUs it knows wedged (found so, or held so
        # by the run) and the devices it took out for that, the devices that
        # bench their device probe before their next block and its records.
        self.stuck_at_start: Set[int] = set()
        self.wedges: List[Dict[str, Any]] = []
        self.elsewhere: List[Dict[str, Any]] = []
        self.wedged_stop: Optional[str] = None
        self._wedged_keys: Set[str] = set()
        self._wedged_devices: Set[int] = set()
        self.probes: List[Dict[str, Any]] = []
        self._probe_due: Dict[int, str] = {}
        # the lines whose process left its GPU wedged (`wedging_line`), and
        # per device the health canaries in a row whose bench faulted
        self.wedge_lines: List[Dict[str, Any]] = []
        self._canary_faults: Dict[int, int] = {}
        self._probe_line: Optional[Tuple[Optional[str], str]] = None
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
        stuck = _stuck_processes()
        self.stuck_at_start = {p for p, _ in stuck}
        if stuck:
            ui.warn(
                "wedge",
                f"processes stuck in the GPU driver since before the stage: "
                f"{', '.join(f'pid {p} ({c})' for p, c in stuck[:6])}; a GPU they "
                f"hold memory on may not bench",
            )
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
        the stage's other GPUs (`_unlike`), the GPU passed its device probe
        when one is due (`probe_due`) and its canary. "go"; "again" when the
        device was retired meanwhile, left out because its GPU stayed held,
        unlike the others, or its canary kept failing for longer than
        bench.health.max_wait_s (`_leave_out`; with `leave_held`, at once
        while the run holds its GPU alone), or at once because its GPU is
        wedged (the run's hold says so, or it failed its probe); without
        `wait`, "later" while the run holds that GPU alone, it is unlike the
        others, or its canary failed and is not due again; "stop" when
        `stop()` turns true or the stage stops waiting (`gave_up`: a hold of
        every GPU, or busy cores, for longer than their limits, or a wedge,
        `wedged_stop`). The waits count from `since`, default now."""
        t0 = self.clock() if since is None else since
        limit = self.health.max_wait_s

        def overdue() -> bool:
            return limit > 0 and self.clock() - t0 >= limit

        while True:
            if stop() or self.gave_up is not None:
                return "stop"
            if not self.gpus.active(device):
                return "again"
            hold = self.hold_on(device)
            if hold is not None:
                wedged = self._held_wedged(device, hold)
                if wedged is not None:
                    return wedged
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
            if self._probe_pending(device):
                passed = self._probe(device)
                if passed is None:
                    return "stop"
                if not passed:
                    return "stop" if self.gave_up is not None else "again"
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
                _retire(self.gpus, device, _in_use(busy), temporary=True)
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
        if self._probe_pending(device):
            return f"gpu{device} has its device probe to bench"
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

    # ── wedged GPUs ──

    def check_wedge(self, device: int, proc: Any, rc: int) -> Optional[Dict[str, Any]]:
        """What a bench process on `device` that was killed, or failed
        without finishing a shape, left wedged (`_wedge_of`), held in the
        run (`wedged`), whatever GPUs it names (`own_wedge` tells whether
        `device`'s is one); None when nothing is, and once the stage is
        stopping (its bench processes are being stopped:
        `ChildGroups.stopped`). The processes an earlier check found stuck
        are left out with those stuck since before the stage: the run holds
        what they wedged already, and each later check would wait REAP_S
        more to find them again."""
        seen: List[Dict[str, Any]] = []
        with self._lock:
            found = {
                int(p["pid"])
                for w in self.wedges
                for p in [
                    *(
                        q
                        for info in (w.get("gpus") or {}).values()
                        for q in info.get("processes") or []
                    ),
                    *(w.get("host") or []),
                ]
                if isinstance(p.get("pid"), int)
            }
        rec = _wedge_of(
            device,
            proc,
            rc,
            self.gpus.gpu,
            self.stuck_at_start | found,
            elsewhere=seen,
            stopping=lambda: self.children.stopped,
        )
        with self._lock:
            known = {e["pid"] for e in self.elsewhere}
            new = [e for e in seen if e["pid"] not in known]
            self.elsewhere.extend(new)
        if new:
            ui.warn(
                "wedge",
                f"{_who(new)}: stuck in the GPU driver, on no GPU of the stage and "
                f"not started by it; recorded, but this holds or stops nothing here",
            )
        if rec is not None:
            self.wedged(device, rec)
        return rec

    def _key(self, device: int) -> str:
        gpu = self.gpus.gpu.get(device)
        return gpu_health.gpu_key(gpu) if gpu is not None else f"device {device}"

    def wedged(self, device: int, rec: Mapping[str, Any]) -> None:
        """Hold the GPUs of `rec` (`_wedge_of`) as wedged in the run, the
        host when processes the stage started are stuck holding memory on no
        GPU (gpu_health.Guard.hold_wedged), take the stage's devices on them
        out of the stage and stop the stage when the wedges it knows warrant
        it (`_stop_if_spread`)."""
        what = describe_wedge(rec)
        texts = {
            key: f"GPU {info.get('gpu')} ({key}) is wedged: {_wedge_why(info)}"
            for key, info in (rec.get("gpus") or {}).items()
        }
        if rec.get("host"):
            texts[gpu_health.HOST] = (
                f"the host is wedged: {_who(rec['host'])} stuck in the GPU driver, "
                f"holding memory on no GPU"
            )
        try:
            self.state.hold_wedged(texts)
        except OSError as e:
            ui.warn("wedge", f"could not record the wedge in the run: {e}")
        hit = sorted(d for d in self.gpus.gpu if self._key(d) in texts)
        with self._lock:
            self.wedges.append({**rec, "devices": hit})
        every, keys = self._note_wedged([k for k in texts if k != gpu_health.HOST], hit)
        for d in hit:
            self._wedged_out(d, what)
        if rec.get("host"):
            self._stop_wedged(what)
        self._stop_if_spread(every, keys, what)

    def _note_wedged(
        self, keys: Sequence[str], devices: Sequence[int]
    ) -> Tuple[bool, List[str]]:
        """Note the stage's GPUs `keys`, on its `devices`, as wedged:
        (whether every device of the stage is wedged now, the keys of the
        stage's GPUs known wedged), both read where they were noted, so
        that two devices noted at once agree on what they see."""
        with self._lock:
            self._wedged_keys.update(keys)
            self._wedged_devices.update(devices)
            every = set(self.gpus.gpu) <= self._wedged_devices
            return every, sorted(self._wedged_keys)

    def _stop_if_spread(self, every: bool, keys: Sequence[str], what: str) -> None:
        """Stop the stage when every device of it is wedged (`every`) or two
        or more of its GPUs are (`keys`), by records of their own too or held
        so by the run. A stage whose other devices left it for other reasons
        (a probe or canary that failed, another process on the GPU) runs out
        of devices without a wedge stop: it ends as any stage that did, and
        another attempt benches on the GPUs the run does not hold."""
        if every:
            self._stop_wedged(f"every GPU of the stage is wedged ({what})")
        elif len(keys) >= 2:
            self._stop_wedged(_spread_text(keys, what))

    def _wedged_out(self, device: int, why: str) -> None:
        """Take `device`, whose GPU is wedged, out of the stage for good."""
        if self.gpus.retire(device, f"wedged: {why}", temporary=False):
            ui.err(
                "wedge",
                f"gpu{device} takes no more blocks: its GPU is wedged ({why}); the "
                f"run holds it until the host reboots",
            )

    def _stop_wedged(self, why: str) -> None:
        with self._lock:
            if self.wedged_stop is None:
                self.wedged_stop = why
        self._give_up(
            f"{procs.WEDGED_TEXT}: {why}; reboot the host, then run the same "
            f"command to go on"
        )

    # ── node faults ──

    def node_fault(
        self,
        block: TaskBlock,
        offset: int,
        device: int,
        text: str,
        log_path: Path,
        wedge: Optional[Mapping[str, Any]],
        bench_binary: Path,
    ) -> Optional[Dict[str, Any]]:
        """Apply the fault policy (lib/node_faults.py) to the bench process
        of `block` on `device` that failed in the line at `offset` with its
        output since it started `text`: None when the policy is off or the
        output reports no fault it knows; else the fault's record, whose
        `decision` is FAULT_STOP when the node is retired (error 719; error
        700 that left the GPU wedged, or whose re-run of the line does not
        fault: the node is marked in the store and the stage stops) and
        FAULT_LINE when an error 700 follows the line (the line is
        quarantined in the run). The record and the re-run's log go to
        `<out dir>/faults/`."""
        code = node_faults.fault_of(text)
        if self.faults is None or code is None:
            return None
        line = block.lines[offset]
        shape = shape_of_line(line)
        gpu = self.gpus.gpu.get(device)
        rec: Dict[str, Any] = {
            "code": code,
            "block": block.name,
            "device": device,
            "gpu": gpu.index if gpu is not None else self.gpus.physical.get(device),
            "yaml_line": block.line_numbers[offset],
            "shape": list(shape) if shape is not None else None,
            "line_sha256": hashlib.sha256(line.strip().encode()).hexdigest()[:16],
            "stage": str(self.out_dir),
            "log": str(log_path),
            "record": node_faults.runtime_record(text),
            "why": FAULT_WHY,
        }
        stem = f"{block.name}_line{rec['yaml_line']}"
        if code == node_faults.LAUNCH_FAILURE:
            rec["decision"], rec["reason"] = FAULT_STOP, "error 719 (launch failure)"
        elif wedge is not None:
            rec["decision"] = FAULT_STOP
            rec["reason"] = "error 700 (memory access fault) that left its GPU wedged"
        else:
            rerun = self._rerun_line(device, line, bench_binary, stem)
            rec["rerun"] = rerun
            if rerun["code"] == node_faults.LAUNCH_FAILURE:
                rec["decision"] = FAULT_STOP
                rec["reason"] = (
                    "the re-run of a line with error 700 failed with error 719"
                )
            elif rerun["code"] == node_faults.ILLEGAL_ADDRESS:
                rec["decision"] = FAULT_LINE
                rec["reason"] = (
                    "error 700 that follows the line (its re-run faulted again)"
                )
            else:
                rec["decision"] = FAULT_STOP
                rec["reason"] = (
                    "error 700 that does not follow the line (its re-run "
                    f"{'ran through' if rerun['rc'] == 0 else 'failed otherwise'})"
                )
        with self._lock:
            self.fault_records.append(rec)
        out = self.out_dir / FAULTS_DIR
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{stem}.json").write_text(json.dumps(rec, indent=2) + "\n")
        except OSError as e:
            ui.warn("fault", f"could not keep the fault record: {e}")
        where = f"{block.name} on gpu{device}, yaml line {rec['yaml_line']}"
        if rec["decision"] == FAULT_LINE:
            self._note_hung(rec, device)
            try:
                self.state.quarantine_lines([rec])
            except OSError as e:
                ui.warn("fault", f"could not record the faulting line in the run: {e}")
            ui.warn("fault", f"{where}: {rec['reason']}; the line is quarantined")
            return rec
        node = node_faults.node_name()
        kept = ("code", "block", "device", "gpu", "yaml_line", "shape", "stage")
        try:
            self.faults.mark(
                node,
                {k: rec.get(k) for k in kept}
                | {"why": rec["reason"], "record": rec["record"]},
            )
        except OSError as e:
            ui.err("fault", f"could not mark node {node} in {self.faults.root}: {e}")
        ui.err("fault", f"{where}: {rec['reason']}; node {node} is retired")
        self._stop_node(node, rec["reason"])
        return rec

    def _rerun_line(
        self, device: int, line: str, bench_binary: Path, stem: str
    ) -> Dict[str, Any]:
        """Bench `line` once more in a fresh process on `device` through the
        block's bench, with node_faults.RERUN_ENV: its rc, kill reason, the
        fault it reports (None for none), the runtime's record of it and
        its log."""
        log = self.out_dir / FAULTS_DIR / f"{stem}_rerun.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix="fault_rerun_", dir=self.work_dir, delete=False
        ) as f:
            tmp = Path(f.name)
            f.write(name_entry(line, "fault_rerun"))
        bench_device, env, pin = _bench_launch(device)
        env = {**env, **node_faults.RERUN_ENV}
        cmd = [str(bench_binary), "--device", bench_device, "--yaml", str(tmp)]
        rc: Optional[int] = None
        reason: Optional[str] = None
        try:
            with log.open("w") as lf:
                proc = self.children.popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    preexec_fn=pin,
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
            reason = STOPPING
        finally:
            tmp.unlink(missing_ok=True)
        text = log.read_text(errors="replace") if log.is_file() else ""
        code = node_faults.fault_of(text)
        return {
            "rc": rc,
            "reason": reason,
            "code": code,
            "record": node_faults.runtime_record(text) if code else None,
            "log": str(log),
            "env": dict(node_faults.RERUN_ENV),
        }

    def _stop_node(self, node: str, why: str) -> None:
        """Stop the stage: node `node` is retired, its faults marked."""
        with self._lock:
            if self.wedged_stop is None:
                self.wedged_stop = f"node {node} retired: {why}"
        root = self.faults.root if self.faults is not None else "the store"
        self._give_up(
            f"{procs.WEDGED_TEXT}: node {node} is retired ({why}); once it was "
            f"rebooted and checked: scripts/node_faults.py reset {node} --store "
            f"{root} --note '...', then run the same command to go on"
        )

    def _held_wedged(self, device: int, hold: Mapping[str, Any]) -> Optional[str]:
        """What `ready` returns for a hold that holds `device` as wedged:
        "stop" for a wedged host, or when the stage stops for the wedges it
        knows (`_stop_if_spread`), "again" for its own GPU's wedge otherwise
        (the device leaves the stage); None for any other hold."""
        entries = gpu_health.hold_entries(hold)
        host = entries.get(gpu_health.HOST)
        if gpu_health.is_wedge(host):
            self._stop_wedged("; ".join(host.get("problems") or []) or "the host")
            return "stop"
        own = entries.get(self._key(device))
        if not gpu_health.is_wedge(own):
            return None
        why = "; ".join(own.get("problems") or []) or "wedged"
        every, keys = self._note_wedged([self._key(device)], [device])
        self._wedged_out(device, why)
        self._stop_if_spread(every, keys, why)
        return "stop" if self.wedged_stop is not None else "again"

    def probe_due(self, device: int, why: str) -> None:
        """`device` benches its device probe before its next block (`why`:
        what its last bench process did)."""
        with self._lock:
            self._probe_due.setdefault(device, why)
        ui.warn("probe", f"gpu{device}: {why}; it benches its device probe first")

    def _probe_pending(self, device: int) -> bool:
        with self._lock:
            return device in self._probe_due

    def probe_line(self) -> Tuple[Optional[str], str]:
        """The line of the device probe: the stage's least-work line,
        restamped to time one kernel alone (`_restamp`); or None and why
        there is none."""
        with self._lock:
            if self._probe_line is not None:
                return self._probe_line
        least = least_work_line(self.lines)
        line = self._restamp(least) if least is not None else None
        why = ""
        if line is None:
            why = (
                "no bench line names its shape"
                if least is None
                else "the least-work line is block-scaled and lists no kernel, "
                "and there is no library file to take one from"
            )
        with self._lock:
            self._probe_line = (line, why)
        return line, why

    def _probe(self, device: int) -> Optional[bool]:
        """Bench the device probe on `device` in a process of its own: True
        when the GPU passes (the process ran through; without a probe line,
        at once), False when it does not: a probe that tells against the GPU
        (`bench_fault`: stuck in the GPU driver, a fatal GPU error, a hang or
        silence until the watchdog killed it; or processes left stuck in the
        driver on this GPU) holds it as wedged in the run, and one that
        crashed, or exited non-zero before any problem, takes the device out
        of this stage only (`_leave_out`), since a bench binary, yaml or
        library that fails does that on any GPU; either way it leaves the
        stage. What the probe's wedge check finds on other GPUs, or on the
        host alone, the run holds, but it tells nothing about this one.
        None when the stage stops."""
        with self._lock:
            why = self._probe_due.pop(device, None)
            n = len(self.probes)
        if why is None:
            return True
        gpu = self.gpus.gpu.get(device)
        line, none = self.probe_line()
        rec: Dict[str, Any] = {
            "device": device,
            "gpu": gpu.index if gpu is not None else self.gpus.physical.get(device),
            "key": self._key(device),
            "why": why,
            "taken": kfd.utc_iso(),
        }
        if line is None:
            rec.update(passed=None, note=f"no device probe: {none}")
            with self._lock:
                self.probes.append(rec)
            ui.warn("probe", f"gpu{device}: no device probe ({none}); it benches on")
            return True
        log = self.out_dir / PROBE_DIR / f"probe_{n:04d}_gpu{device}.log"
        _, _, run = self._bench_canary(
            device, [name_entry(line, "device_probe")], log, what="device_probe"
        )
        if run.get("reason") == STOPPING:
            return None
        fault = run.get("fault") or (
            "left processes stuck in the GPU driver" if run.get("wedged") else None
        )
        error = None if fault is not None else run.get("bench_error")
        passed = fault is None and error is None
        rec.update(passed=passed, log=str(log), rc=run.get("rc"))
        if passed:
            with self._lock:
                self.probes.append(rec)
            ui.ok("probe", f"gpu{device}: passes its device probe")
            return True
        if fault is None:
            rec["bench_error"] = error
            with self._lock:
                self.probes.append(rec)
            self._leave_out(
                device,
                f"its device probe {error}, after {why}; a bench, yaml or "
                f"library that fails does that on any GPU, so the run does not "
                f"hold it as wedged (log: {log})",
            )
            return False
        rec["fault"] = fault
        with self._lock:
            self.probes.append(rec)
        if not run.get("wedged"):
            self.wedged(
                device,
                {
                    "device": device,
                    "gpu": self._key(device),
                    "reaped": True,
                    "gpus": {
                        self._key(device): {
                            "gpu": rec["gpu"],
                            "processes": [],
                            "why": f"its device probe {fault}, after {why}",
                        }
                    },
                    "host": [],
                    "at": rec["taken"],
                    "probe": str(log),
                },
            )
        return False

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
        reference, or could not be read (its bench failed in a way that tells
        nothing against the GPU, `bench_fault`: it crashed, or exited
        non-zero before any problem or once its output began; its output did
        not line up; or it read fewer than half its lines): the GPU then
        benches without a reference or a verdict, and its next canary takes
        or judges it, as for a first canary; False when it fails (it is read
        again `canary.retry_s` later): beyond its tolerance, or with a bench
        that is stuck in the GPU driver, reports a fatal GPU error, or hangs
        or prints nothing until the watchdog kills it (`bench_fault`; one
        that leaves processes stuck in the GPU driver on this GPU also holds
        it as wedged; a wedge its check finds on other GPUs, or on the host
        alone, tells nothing against this one). A device whose canary bench
        faults so in CANARY_FAULT_LIMIT canaries in a row leaves the stage
        (`_leave_out`): the canary's lines and kernels are the run's for
        good, so one that faults on its kernel faults every time, whatever
        bench.health.max_wait_s says. None when the stage stops."""
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
        fault = run.get("fault") or (
            "left processes stuck in the GPU driver" if run.get("wedged") else None
        )
        if fault is not None:
            rec.update(
                reference=False,
                judged=False,
                passed=False,
                faulted=True,
                why=f"its bench {fault}",
            )
            rec = self.state.record_canary(key, rec)
            with self._lock:
                self.canaries.append(rec)
                self._next_canary[device] = self.clock() + self.health.canary.retry_s
                faults = self._canary_faults.get(device, 0) + 1
                self._canary_faults[device] = faults
            if run.get("wedged"):
                return False
            if faults >= CANARY_FAULT_LIMIT:
                self._leave_out(
                    device,
                    f"its health canary's bench {fault}, in {faults} canaries in "
                    f"a row (log: {log}); {CANARY_FAULT_REMEDY}",
                )
                return False
            ui.err(
                "health",
                f"gpu{device} ({key}): canary fails: its bench {fault}. The GPU "
                f"benches no block until a canary passes; the next is read in "
                f"{ui.fmt_dur(self.health.canary.retry_s)}, and after "
                f"{CANARY_FAULT_LIMIT} such canaries in a row the GPU leaves the "
                f"stage. Log: {log}; {CANARY_FAULT_REMEDY}",
            )
            return False
        with self._lock:
            self._canary_faults.pop(device, None)
        n_read = sum(r is not None for r in readings)
        clean = run.get("rc") == 0 and not run.get("reason")
        clean = clean and not run.get("misaligned")
        how = (
            f"{n_read} of {len(lines)} canary lines read, "
            + (
                f"its bench {run['bench_error']}"
                if run.get("bench_error")
                else f"bench rc {run.get('rc')}"
            )
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
                tolerance_us=float(setup.get("tolerance_us") or 0.0),
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
            floor = float(setup.get("tolerance_us") or 0.0)
            ui.err(
                "health",
                f"gpu{device} ({key}): canary fails: {rec.get('why')} (tolerance "
                f"{setup['tolerance']:g}"
                + (f" and {floor:g} us" if floor > 0 else "")
                + f", recorded with the run's canary on "
                f"{setup['created']}). The GPU benches no block until a canary "
                f"passes; the next is read in "
                f"{ui.fmt_dur(self.health.canary.retry_s)}. Log: {log}; to take "
                f"its current state as the reference: scripts/run_guard.py accept "
                f"{self.state.dir} --gpu {key}",
            )
        return bool(rec.get("passed"))

    def _bench_canary(
        self,
        device: int,
        entries: Sequence[str],
        log: Path,
        what: str = "health_canary",
    ) -> Tuple[List[Optional[float]], List[Optional[str]], Dict[str, Any]]:
        """Bench the canary `entries` (or the device probe's, `what`) in one
        process on `device`, its output in `log`; (the latency of each entry
        and the name hash of the kernel it timed, None for one without, and
        the process's rc and kill reason, with how it failed, `bench_fault`:
        `fault` when that tells against the GPU, else `bench_error`, and
        `wedged` when it left its own GPU wedged, `check_wedge`: a wedge of
        other GPUs, or of the host alone, the run holds, but it tells
        nothing against this one). The reason is STOPPING when the stage is
        stopping."""
        from lib import kernel_identity as ki

        none: List[Any] = [None] * len(entries)
        if self.children.stopped:
            return none, list(none), {"rc": None, "reason": STOPPING}
        log.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            prefix=f"{what}_",
            dir=self.work_dir,
            delete=False,
        ) as f:
            tmp = Path(f.name)
            f.writelines(entries)
        bench_device, env, pin = _bench_launch(device)
        cmd = [str(self.bench), "--device", bench_device, "--yaml", str(tmp)]
        rc: Optional[int] = None
        reason: Optional[str] = None
        proc: Optional[subprocess.Popen] = None
        try:
            with log.open("w") as lf:
                proc = self.children.popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    preexec_fn=pin,
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
        if self.children.stopped:
            return none, list(none), {"rc": rc, "reason": STOPPING}
        text = log.read_text(errors="replace")
        problems = list(iter_problems(text.splitlines()))
        run: Dict[str, Any] = {"rc": rc, "reason": reason}
        fault = bench_fault(rc, reason, problems, text)
        if fault is not None:
            run["fault" if fault.gpu else "bench_error"] = fault.what
        if proc is not None and (
            reason is not None or (rc != 0 and _n_finished(problems) == 0)
        ):
            wedge = self.check_wedge(device, proc, int(rc if rc is not None else 0))
            if own_wedge(wedge):
                run["wedged"] = wedge
        off = _misalignment(
            problems,
            [shape_of_line(e) for e in entries],
            rc == 0 and reason is None,
            single=True,
        )
        if off is not None:
            return none, list(none), {**run, "misaligned": off}
        rows = [p.tested[0] if p.complete and p.tested else None for p in problems]
        out = [_latency(r.fields) if r is not None else None for r in rows]
        names = [
            ki.name_hash(r.sol_name) if r is not None and r.sol_name else None
            for r in rows
        ]
        pad = [None] * (len(entries) - len(out))
        return out + pad, names + list(pad), run

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

    def wedging_line(
        self,
        rec: Dict[str, Any],
        block: str,
        line: str,
        problem: Optional[Problem],
        device: int,
        warmup: bool = False,
    ) -> None:
        """Record the line of `rec` whose process on `device` left its GPU
        wedged (`problem`, the line's open problem, says in which kernel;
        with `warmup`, that of its warm-up shape), and quarantine it in the
        run under `hung_scope`, as a hung line (`why` wedged; unless that is
        NO_SCOPE): one that does so every time would wedge every GPU that
        benches it, one reboot at a time; `rec` says whether it is
        (`quarantined`)."""
        gpu = self.gpus.gpu.get(device)
        flight = _in_flight(line, problem, self)
        if warmup:
            flight["in_flight"] = f"its warm-up shape ({flight['in_flight']})"
        rec.update(
            block=block,
            why=gpu_health.WEDGED_LINE,
            line_sha256=hashlib.sha256(line.strip().encode()).hexdigest()[:16],
            **flight,
            gpu=gpu.index if gpu is not None else self.gpus.physical.get(device),
            stage=str(self.out_dir),
            scope=self.hung_scope,
            quarantined=self.quarantines,
        )
        with self._lock:
            self.wedge_lines.append(rec)
        if self.quarantines:
            try:
                self.state.quarantine_lines([rec])
            except OSError as e:
                ui.warn("wedge", f"could not record the wedging line in the run: {e}")

    def settle_hung(
        self,
        blocks: Sequence[TaskBlock],
        hung_again: AbstractSet[int],
        wedged: AbstractSet[int] = frozenset(),
    ) -> None:
        """After the retry pass: a line of `blocks` it took for hanging
        (`defer_hung`) is quarantined unless the pass benched it to the end
        (`TaskBlock.hung_recovered`); one that hung there again (its yaml
        line in `hung_again`) the pass quarantined itself, and so it did one
        whose process left its GPU wedged there (`wedged`, `wedging_line`)."""
        for b in blocks:
            for rec in b.hung:
                if not rec.get("retry") or rec.get("retried"):
                    continue
                n = int(rec["yaml_line"])
                if n in b.hung_recovered:
                    rec["retried"] = "benched"
                elif n in hung_again:
                    rec["retried"] = "hung again"
                elif n in wedged:
                    rec["retried"] = "left its GPU wedged"
                else:
                    rec["retried"] = "not benched to the end"
                    self.quarantine_line(rec, int(rec["device"]))

    def drop_quarantined(
        self, lines: Sequence[str]
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """(the lines the run has not quarantined under `hung_scope`, their
        1-based line numbers, a record per line left out: its line, shape
        and why the run quarantined it, hung or wedged)."""
        banned = (
            self.state.quarantined_lines(self.hung_scope) if self.quarantines else {}
        )
        kept: List[str] = []
        numbers: List[int] = []
        dropped: List[Dict[str, Any]] = []
        for i, ln in enumerate(lines, 1):
            shape = shape_of_line(ln)
            if shape is not None and tuple(shape) in banned:
                why = str(banned[tuple(shape)].get("why") or "hung")
                dropped.append({"yaml_line": i, "shape": list(shape), "why": why})
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
            wedge_lines = [dict(w) for w in self.wedge_lines]
            wedges = list(self.wedges)
            elsewhere = [dict(e) for e in self.elsewhere]
            probes = [dict(p) for p in self.probes]
            faults = [dict(f) for f in self.fault_records]
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
            "node_faults": {
                "store": str(self.faults.root) if self.faults is not None else None,
                "faults": faults,
            },
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
            "wedge_lines": wedge_lines,
            "quarantined_lines_left_out": list(self.skipped_lines),
            "gave_up": self.gave_up,
            "wedged": {
                "stopped": self.wedged_stop,
                "found": [dict(w) for w in wedges],
                "elsewhere": elsewhere,
                "stuck_at_start": sorted(self.stuck_at_start),
            },
            "probes": probes,
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


def _in_use(busy: Sequence[kfd.OtherUse], still: bool = False) -> str:
    """Why a device whose GPU has the other use `busy` (`StageGpus.busy`) is
    retired: who is on the GPU, or beside it when processes on other
    partitions of its card alone make up that use, as the watch's reason
    says (`kfd.other_use_rule`)."""
    where = "beside" if kfd.other_use_rule(busy) == "sibling-partition" else "on"
    who = "; ".join(f.describe() for f in busy)
    return f"{who}{' still' if still else ''} {where} it"


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
    (for good) or the other use is still there (until it is gone). Returns
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
        busy = gpus.busy(device)
        if busy:
            _retire(gpus, device, _in_use(busy, still=True), temporary=True)
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


def _journal_subrun(
    journal: Optional[bench_resume.Journal],
    block: TaskBlock,
    log_path: Path,
    canary: bool,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Before a bench process of `block` starts: record where the block
    stands and how far its log runs, so that a stage started again continues
    after the problems that process finishes, then name the process's GPU,
    boot, run and segment, and whether the `canary` times it, in the log. The
    attempt's data takes its device's segment as it starts."""
    if journal is None:
        return
    if block.segment is None:
        block.segment = journal.segment_of(block.device)
    size = log_path.stat().st_size if log_path.exists() else 0
    state = {
        "done": False,
        **_block_state(block),
        **(extra or {}),
        "log": {"name": log_path.name, "size": size},
    }
    journal.record(block.name, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(journal.header(block.device, canary))


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
        journal: Optional[bench_resume.Journal] = None,
    ) -> None:
        self.journal = journal
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
        block.finished = time.time()
        with self.lock:
            self.done.append(block)
            self._open -= 1
        if self.journal is not None:
            self.journal.record(block.name, {"done": True, **_block_state(block)})

    def journal_subrun(self, block: TaskBlock, log_path: Path, canary: bool) -> None:
        _journal_subrun(self.journal, block, log_path, canary)

    def leave(self, block: TaskBlock, log_path: Path) -> None:
        """Leave `block`, which the stage's stop cut off, unfinished: its
        record, with the end of its log, has the run that resumes the stage
        go on from its `next_offset` (`_continue_partial`)."""
        if self.journal is not None:
            log = {"name": log_path.name, "size": file_size(log_path)}
            self.journal.record(
                block.name, {"done": False, **_block_state(block), "log": log}
            )

    def requeue(self, block: TaskBlock, delay_s: float = 0.0) -> None:
        """Put `block` back, to be taken no sooner than `delay_s` from now;
        a block quarantined more than `max_quarantines` times is given up."""
        if self.journal is not None:
            self.journal.record(block.name, {"done": False, **_block_state(block)})
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

# How long a stage that leaves early waits for each device worker to end
# once its benches are stopped: as long as procs.kill_group waits for a
# process stuck in a driver call, which a worker may still be waiting on.
WORKER_STOP_S = procs.STUCK_WAIT_S


class _DeviceWorker(threading.Thread):
    """Take blocks off `work`, run each on `device`, finish or requeue it;
    while `device` is retired, wait for it to return (`_await_gpu`). With a
    guard, the worker waits until `device` is ready before it takes a block
    and asks again once it holds one, since the wait for a block can be
    long (`_Work.take` waits while blocks are out): a block that cannot
    start yet goes back on the queue. Once `children` are being stopped
    (the stage is stopping) the worker takes no block, waits for nothing
    and ends."""

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

    def _stopping(self) -> bool:
        return self.children is not None and self.children.stopped

    def _done(self) -> bool:
        return self.work.closed() or self._stopping()

    def run(self) -> None:
        t0 = time.time()
        while not self._stopping():
            if not self.gpus.active(self.device):
                if _await_gpu(self.gpus, [self.device], self._done) is None:
                    break
            if self.guard is not None:
                go = self.guard.ready(self.device, self._done)
                if go == "stop":
                    break
                if go == "again":
                    continue
            block = self.work.take(
                lambda: not self.gpus.active(self.device) or self._stopping()
            )
            if block is None:
                if self.gpus.active(self.device):
                    break
                continue
            if self.guard is not None and self.guard.blocked(self.device):
                self.work.requeue(block)
                continue
            busy = self.gpus.busy(self.device)
            if busy:
                _retire(self.gpus, self.device, _in_use(busy), temporary=True)
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
        journal = self.work.journal
        if journal is not None and block.segment is not None:
            # A block an earlier run left partial: its data stays in one log
            # under this device's names, and goes on only in its segment.
            paths = (log_path, warmup_log_path, canary_log_path)
            for moved in _carry_logs(paths, self.device):
                ui.info("resume", f"{block.name}: {moved}")
            segment = journal.segment_of(self.device)
            if segment != block.segment:
                drift_dir = self.logs_dir.parent / DRIFT_DIR
                _segment_aside(
                    block, self.device, paths, drift_dir, segment, journal.run
                )
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
                on_subrun=self.work.journal_subrun,
            )
        except procs.ChildrenStopped:
            block.stopped = True
        except Exception as exc:
            block.rc = -1
            block.abandoned = True
            _annotate(log_path, f"\n*** ERROR: {exc} ***\n")
        if block.stopped:
            self.work.leave(block, log_path)
            return
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
            block, self.device, self.canary, logs, drift_dir, self.work.journal
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


# TaskBlock fields that are sets, and those keyed by an int, as the resume
# journal (lib/bench_resume.py) writes them in JSON.
_BLOCK_SETS = ("recovered", "hung_recovered")
_BLOCK_INT_KEYS = ("skipped", "hung_lines")
# The fields of the block layout, which the journal records once.
_BLOCK_LAYOUT = ("block_id", "name", "lines", "line_numbers", "heavy")


def _block_state(block: TaskBlock) -> Dict[str, Any]:
    """`block`'s state as the resume journal records it, its layout aside."""
    out = {
        f.name: getattr(block, f.name)
        for f in dataclasses.fields(TaskBlock)
        if f.name not in _BLOCK_LAYOUT
    }
    out = json.loads(json.dumps(out, default=sorted))
    for name in _BLOCK_INT_KEYS:
        out[name] = {str(k): v for k, v in getattr(block, name).items()}
    return out


def _block_of(layout: Mapping[str, Any], state: Mapping[str, Any]) -> TaskBlock:
    """The TaskBlock of a journal's layout entry and unit state."""
    names = {f.name for f in dataclasses.fields(TaskBlock)}
    kw: Dict[str, Any] = {k: v for k, v in state.items() if k in names}
    kw.update({k: layout[k] for k in _BLOCK_LAYOUT})
    for name in _BLOCK_SETS:
        kw[name] = set(kw.get(name) or ())
    for name in _BLOCK_INT_KEYS:
        kw[name] = {int(k): v for k, v in (kw.get(name) or {}).items()}
    return TaskBlock(**kw)


def _layout_of(blocks: Sequence[TaskBlock]) -> List[Dict[str, Any]]:
    return [{k: getattr(b, k) for k in _BLOCK_LAYOUT} for b in blocks]


def _lines_of(layout: Sequence[Mapping[str, Any]]) -> List[Tuple[int, str]]:
    return sorted(
        (int(n), str(t)) for b in layout for n, t in zip(b["line_numbers"], b["lines"])
    )


def _finished_since(path: Path, size: int) -> int:
    """The problems a block log finished past byte `size`: what a process
    the stage's interruption killed got through after its journal entry."""
    if not path.is_file():
        return 0
    with path.open("rb") as f:
        f.seek(max(0, int(size)))
        text = f.read().decode(errors="replace")
    return count_finished_problems(text.splitlines(keepends=True))


def _resume_blocks(
    journal: Optional[bench_resume.Journal],
    blocks: List[TaskBlock],
    out_dir: Path,
    logs_dir: Path,
) -> Tuple[List[TaskBlock], Optional[Dict[str, Any]]]:
    """The blocks of the earlier invocation the stage continues from: the
    resume journal's layout when it holds the lines `blocks` do (whatever
    the GPU count that cut them), each block with its recorded state. A
    done block keeps its record and is not benched again. A partial one
    continues after the problems its last process finished
    (`_continue_partial`), so the shape an interruption cut off is benched
    again from scratch; it keeps its skipped shapes, its quarantines and
    drifts, its segment and its earlier canary readings, which its first
    attempt's end judges with the new ones (`_judge_canary`). Without such
    a journal the earlier block logs move aside. Returns the blocks and
    {done: names, partial: {name: offset}}, or None without a resume."""
    if journal is None:
        return blocks, None
    layout = journal.layout
    if not layout or _lines_of(layout) != _lines_of(_layout_of(blocks)):
        bench_resume.set_aside(out_dir, _previous_outputs(out_dir, logs_dir))
        return blocks, None
    units = journal.units()
    out: List[TaskBlock] = []
    done: List[str] = []
    partial: Dict[str, int] = {}
    for entry in sorted(layout, key=lambda b: int(b["block_id"])):
        state = units.get(str(entry["name"]))
        if state is None:
            out.append(_block_of(entry, {}))
            partial[str(entry["name"])] = 0
            continue
        b = _block_of(entry, state)
        if state.get("done"):
            done.append(b.name)
            out.append(b)
            continue
        _continue_partial(b, state, logs_dir)
        partial[b.name] = b.next_offset
        out.append(b)
    return out, {"done": done, "partial": partial}


def _continue_partial(
    block: TaskBlock, state: Mapping[str, Any], logs_dir: Path
) -> None:
    """A unit an interruption cut off (`state`, its journal record): it goes
    on after the problems its last process finished, counted from its log
    past the size its unit recorded, and forgets that attempt's process."""
    log = state.get("log") or {}
    if log.get("name"):
        block.next_offset = min(
            len(block.lines),
            block.next_offset
            + _finished_since(logs_dir / str(log["name"]), log.get("size", 0)),
        )
    block.device, block.contaminated, block.aborted, block.wedged = (
        None,
        None,
        None,
        None,
    )
    block.rc, block.wall_s, block.abandoned, block.misaligned = 0, 0.0, False, []
    block.stopped = False
    block.host_load = {}


def _resume_retry(
    retry: TaskBlock, journal: Optional[bench_resume.Journal], logs_dir: Path
) -> Tuple[TaskBlock, bool]:
    """The retry pass `retry` as a run of the stage over the same lines left
    it: ended (True: nothing to bench), or cut off (`_continue_partial`).
    Logs of a retry pass over other lines move aside."""
    if journal is None:
        return retry, False
    state = journal.units().get(retry.name)
    if state is None:
        return retry, False
    if state.get("lines_hash") != bench_resume.lines_hash(retry.lines):
        names = (
            "block_retry_gpu*.log",
            "warmup_retry_gpu*.log",
            "canary_retry_gpu*.log",
        )
        old = [p for name in names for p in sorted(logs_dir.glob(name))]
        bench_resume.set_aside(logs_dir.parent, old)
        return retry, False
    b = _block_of({k: getattr(retry, k) for k in _BLOCK_LAYOUT}, state)
    if state.get("done"):
        return b, True
    _continue_partial(b, state, logs_dir)
    return b, False


def _retry_pending(
    blocks: Sequence[TaskBlock],
) -> List[Tuple[TaskBlock, Optional[int], int, str]]:
    """What the retry pass benches: every skipped and unattempted line of
    `blocks` it has not recovered, and every hung line they keep for it
    (`TaskBlock.hung_lines`) not benched to the end: (block, offset, or None
    for a hung line, yaml line, text)."""
    pending: List[Tuple[TaskBlock, Optional[int], int, str]] = [
        (b, off, b.line_numbers[off], b.lines[off])
        for b in blocks
        if not b.gave_up
        for off in sorted(set(b.skipped) | set(b.unattempted))
        if off not in b.recovered
    ]
    pending += [
        (b, None, n, text)
        for b in blocks
        for n, text in sorted(b.hung_lines.items())
        if n not in b.hung_recovered
    ]
    return pending


def _retry_recovered(
    blocks: Sequence[TaskBlock], state: Optional[Mapping[str, Any]]
) -> Set[int]:
    """The yaml lines an earlier run's retry pass, its journal unit `state`,
    benched to the end: its lines are what `_retry_pending` gives the blocks
    as that run left them, and those before its last offset it did not skip
    finished."""
    if not state:
        return set()
    pending = _retry_pending(blocks)
    if state.get("lines_hash") != bench_resume.lines_hash(
        [t for _, _, _, t in pending]
    ):
        return set()
    numbers = [n for _, _, n, _ in pending]
    skipped = {int(k) for k in (state.get("skipped") or {})}
    upto = min(int(state.get("next_offset") or 0), len(numbers))
    return {numbers[i] for i in range(upto) if i not in skipped}


def _canary_passed(block: TaskBlock) -> bool:
    """Whether the canary judged the kept attempt of `block` and it neither
    drifted nor was flagged."""
    return (
        bool(_reading_values(block))
        and bool(block.canary)
        and not block.canary.get("drifted")
        and not block.canary_flagged
    )


def _failure_aside(
    block: TaskBlock, logs_dir: Path, drift_dir: Path, run: int, why: str
) -> None:
    """Before a run that failed is continued: the logs `block` has in
    `logs_dir`, on any GPU, move to `drift_dir/<block>_f<run>/` with
    drift.json saying `why`, and the block is benched again from its start."""
    tag = f"{block.block_id:04d}"
    patterns = (
        f"{block.name}_gpu*.log",
        f"warmup_{tag}_gpu*.log",
        f"{CANARY_LOG_PREFIX}{tag}_gpu*.log",
    )
    files = [p for pat in patterns for p in sorted(logs_dir.glob(pat))]
    dest = drift_dir / f"{block.name}_f{run}"
    rec = _move_aside(
        dest,
        files,
        DRIFT_RECORD,
        {
            "block": block.name,
            "yaml_lines": [block.line_numbers[0], block.line_numbers[-1]],
            "device": block.device,
            "reason": why,
            "readings": [dict(r) for r in block.canary_readings],
        },
    )
    block.drifts.append({**rec, "dir": str(dest), "boundary": True})
    block.reset()


def _reopen_after_failure(
    blocks: List[TaskBlock],
    journal: bench_resume.Journal,
    out_dir: Path,
    logs_dir: Path,
    rc: Optional[int],
) -> Dict[str, Any]:
    """Turn a run that failed (exit `rc`) into one to continue, benching
    again only what failed, never a shape whose data stands:
    - the lines its retry pass benched to the end count as recovered, and
      the pass starts anew over the rest;
    - on each GPU it benched on with the canary, the blocks it finished
      after the last block whose canary passed move to
      `drift/<block>_f<run>/` and are benched again from their start;
    - a block it gave up after its quarantines or the guard's aborts is
      benched again from its start, its counts reset;
    - a block it left with lines unattempted after its crashes, a crash
      block, goes on after its last finished shape, its crashed lines left
      to the retry pass: the run's crash-block count starts anew, and the
      same crashes beyond the tolerance fail it as before.
    Blocks it cut off go on as after an interruption. Returns the record of
    what was reopened, which the journal's run and the summary keep."""
    units = journal.units()
    recovered = _retry_recovered(blocks, units.get("block_retry"))
    journal.archive_unit("block_retry")
    for b in blocks:
        for off in set(b.skipped) | set(b.unattempted):
            if b.line_numbers[off] in recovered:
                b.recovered.add(off)
        b.hung_recovered |= {n for n in b.hung_lines if n in recovered}
    done = {name for name, state in units.items() if state.get("done")}
    record: Dict[str, Any] = {
        "rc": rc,
        "crash_count_reset": True,
        "recovered_by_earlier_retry": len(recovered),
        "after_last_passed_canary": {},
        "given_up": {},
        "crash_blocks": [],
        "reopened": [],
    }
    by_device: Dict[int, List[TaskBlock]] = {}
    for b in blocks:
        if b.name in done and b.device is not None and b.finished is not None:
            by_device.setdefault(int(b.device), []).append(b)
    late: Set[str] = set()
    for d, group in sorted(by_device.items()):
        if not any(_reading_values(b) for b in group):
            continue
        passed = [b.finished for b in group if _canary_passed(b)]
        last = max(passed) if passed else float("-inf")
        after = sorted(
            (b for b in group if b.finished > last), key=lambda b: b.finished
        )
        if after:
            record["after_last_passed_canary"][f"gpu{d}"] = [b.name for b in after]
            late.update(b.name for b in after)
    drift_dir = out_dir / DRIFT_DIR
    for b in blocks:
        if b.name not in done:
            continue
        if b.name in late:
            why = (
                "finished after the last block of its GPU whose canary passed, in "
                "a run that failed"
            )
            _failure_aside(b, logs_dir, drift_dir, journal.run, why)
        elif b.gave_up:
            record["given_up"][b.name] = {
                "quarantines": len(b.quarantines),
                "aborts": len(b.aborts),
            }
            why = "given up in a run that failed; benched again, its counts reset"
            _failure_aside(b, logs_dir, drift_dir, journal.run, why)
            b.gave_up, b.quarantines, b.aborts = False, [], []
        elif any(off not in b.recovered for off in b.unattempted):
            record["crash_blocks"].append(b.name)
            b.abandoned = False
            _continue_partial(b, {}, logs_dir)
        else:
            continue
        record["reopened"].append(b.name)
    return record


def _restore_canary(
    canary: Canary,
    journal: bench_resume.Journal,
    devices: Sequence[int],
    skip: AbstractSet[str] = frozenset(),
) -> None:
    """The canary's baselines of the earlier runs a resumed stage benches
    on: every reading their kept attempts took, per (GPU, segment)
    (`Canary.earlier`), but those of the units `skip` (benched again from
    their start after a failure), and per device the readings of the
    segment it benches, its own GPU's, else (a GPU new to the stage) those
    of the last run's GPUs in that segment, which its first judged attempt
    is held against (`Canary.boundary`)."""
    earlier: Dict[Tuple[str, int], List[float]] = {}
    for name, state in journal.units().items():
        if name in skip:
            continue
        for r in state.get("canary_readings") or []:
            if r.get("run") is None or r.get("segment") is None:
                continue
            if int(r["run"]) >= journal.run:
                continue
            key = (str(r.get("gpu")), int(r["segment"]))
            earlier.setdefault(key, []).extend(
                r[k] for k in ("start", "end") if r.get(k) is not None
            )
    canary.earlier = earlier
    for d in devices:
        segment = journal.segment_of(d)
        if segment is None:
            continue
        own = journal.gpu_of(d)
        gpus = [own] if earlier.get((own, segment)) else journal.earlier_gpus(segment)
        values = [v for g in gpus for v in earlier.get((g, segment), [])]
        if values:
            canary.restore(
                d, segment, values, [g for g in gpus if earlier.get((g, segment))]
            )


# Arguments a run may change and still continue an earlier one: how it
# reports, its tolerances, timeouts and the guard's policies, the block
# layout (the journal's stands), and the config's path, a snapshot per
# orchestrator invocation (its content counts: `_config_digest`). Those that
# choose the devices change too (one GPU model alone: `Journal.begin`).
_RESUME_FREE_ARGS = frozenset(
    {
        "quiet",
        "keep_passed",
        "resume_across_code_change",
        "note",
        "progress_every",
        "config_yaml",
        "max_crash_blocks",
        "max_quarantines",
        "max_retry_gemms",
        "no_retry_skipped",
        "retry_hung",
        "hung_scope",
        "stall_timeout_s",
        "startup_grace_s",
        "node_faults",
        "health",
        "health_canary",
        "quiet_host",
        "min_block_lines",
    }
)


def _config_digest(cfg: Mapping[str, Any]) -> str:
    """The digest of what of the config shapes this stage's benches: its
    bench section, the runtime environment the benches get, the hipBLASLt
    section and the arch."""
    shaping = {
        "bench": cfg.get("bench"),
        "runtime_env": runtime_env(cfg),
        "hipblaslt": cfg.get("hipblaslt"),
        "arch": cfg.get("arch"),
    }
    return bench_resume.lines_hash([json.dumps(shaping, sort_keys=True, default=str)])


def _resume_inputs(
    args: argparse.Namespace,
    lines: Sequence[str],
    library: Any = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> str:
    """The digest of what an invocation benches and how: its lines, its
    arguments but those of _RESUME_FREE_ARGS and those that choose its
    devices, what of the config shapes the benches (`_config_digest`), the
    bench binary and the library files the bench names (their identity).
    This stage's code is the journal's apart (`_journal_fit`)."""
    shaping = {
        k: v
        for k, v in sorted(vars(args).items())
        if k not in _RESUME_FREE_ARGS and "device" not in k and "gpu" not in k
    }
    return bench_resume.lines_hash(
        [
            *lines,
            json.dumps(shaping, sort_keys=True, default=str),
            _config_digest(cfg or {}),
            bench_resume.file_digest(args.bench_binary),
            getattr(library, "digest", None),
        ]
    )


def _layout_fits(
    layout: Optional[Sequence[Mapping[str, Any]]], lines: Sequence[str]
) -> bool:
    """Whether every line of a journal's block layout is one of `lines`, at
    its 1-based number."""
    if not layout:
        return False
    return set(_lines_of(layout)) <= set(enumerate(lines, 1))


def _journal_fit(
    journal: Optional[bench_resume.Journal],
    inputs: str,
    code: Optional[str],
    args: argparse.Namespace,
    lines: Sequence[str],
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """(how, change, why) the stage starts over the journal of an earlier
    invocation in its --out-dir. how: "keep" (a run that passed over the same
    inputs and code stands, with --keep-passed), "continue" (a run an
    interruption cut off), "after_failure" (a run that failed, exit 2:
    `_reopen_after_failure`) or "fresh". A run of other code is continued only
    with --resume-across-code-change, and then `change` ({from, to, note})
    says so; a journal from before the code was recorded apart (`from` None)
    is then held to its block layout alone."""
    if journal is None:
        return "fresh", None, ""
    across = bool(args.resume_across_code_change)
    change: Optional[Dict[str, Any]] = None
    if journal.code is None:
        if not across:
            return "fresh", None, "its journal is of other inputs"
        if not _layout_fits(journal.layout, lines):
            return "fresh", None, "its journal holds other lines"
        change = {"from": None, "to": code, "note": args.note}
    elif journal.inputs != inputs:
        return "fresh", None, "its journal is of other inputs"
    elif journal.code != code:
        if not across:
            return (
                "fresh",
                None,
                (
                    "this stage's code changed since it ran "
                    "(--resume-across-code-change continues it)"
                ),
            )
        change = {"from": journal.code, "to": code, "note": args.note}
    if journal.passed:
        if args.keep_passed:
            return "keep", change, "it passed"
        return "fresh", None, "it passed (--keep-passed keeps it)"
    if journal.failed:
        return (
            "after_failure",
            change,
            f"it failed with exit {journal.data['end'].get('rc')}",
        )
    return "continue", change, "an interruption cut it off"


def _device_identity(
    devices: Sequence[int], gpus: kfd.StageGpus
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """({device: GPU model}, {device: {uuid, bdf}}) of the stage's devices,
    from the GPUs the stage resolved them to."""
    models: Dict[int, Dict[str, Any]] = {}
    ids: Dict[int, Dict[str, Any]] = {}
    for d in devices:
        g = gpus.gpu.get(int(d))
        models[int(d)] = bench_resume.gpu_model(g)
        ids[int(d)] = {
            "uuid": g.uuid if g is not None else None,
            "bdf": g.bdf if g is not None else None,
        }
    return models, ids


def _previous_outputs(out_dir: Path, logs_dir: Path) -> List[Path]:
    """The outputs an earlier invocation of the stage left in `out_dir`."""
    out: List[Path] = []
    for pattern in ("block_*.log", "warmup_*.log", f"{CANARY_LOG_PREFIX}*.log"):
        out.extend(sorted(logs_dir.glob(pattern)))
    for name in (
        "skipped_gemms.txt",
        "stage03_summary.json",
        fingerprint.PLATFORM_FILE,
        host_load.SERIES_FILE,
        str(bench_resume.JOURNAL.parent),
    ):
        if (out_dir / name).exists():
            out.append(out_dir / name)
    for name in (QUARANTINE_DIR, DRIFT_DIR, MISALIGNED_DIR, ABORTED_DIR, PROBE_DIR):
        if (out_dir / name).is_dir():
            out.append(out_dir / name)
    return out


def _clean_previous_outputs(out_dir: Path, logs_dir: Path) -> int:
    """Move the outputs of an earlier invocation the stage does not continue
    from to `superseded/<utc>/` (lib/bench_resume.py `set_aside`): measured
    data is never deleted."""
    found = _previous_outputs(out_dir, logs_dir)
    bench_resume.set_aside(out_dir, found)
    return len(found)


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
    journal: Optional[bench_resume.Journal] = None,
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
    (`hung_recovered`). With `journal`, the pass is a unit of the resume
    journal, which a later run over the same lines continues
    (`_resume_retry`). Returns (retried, recovered, the retry block)."""
    gpus = gpus if gpus is not None else kfd.StageGpus([])
    pending = _retry_pending(blocks)
    if not pending:
        return 0, 0, None
    retry = TaskBlock(
        block_id=-1,
        name="block_retry",
        lines=[text for _, _, _, text in pending],
        line_numbers=[n for _, _, n, _ in pending],
    )
    retry, finished = _resume_retry(retry, journal, logs_dir)
    extra = {"lines_hash": bench_resume.lines_hash(retry.lines)}

    def on_subrun(b: TaskBlock, p: Path, timed: bool) -> None:
        _journal_subrun(journal, b, p, timed, extra)

    while not finished:
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
        if journal is not None and retry.segment is not None:
            paths = (log_path, warmup_log_path, canary_log_path)
            for moved in _carry_logs(paths, device):
                ui.info("resume", f"{retry.name}: {moved}")
            segment = journal.segment_of(device)
            if segment != retry.segment:
                drift_dir = logs_dir.parent / DRIFT_DIR
                _segment_aside(retry, device, paths, drift_dir, segment, journal.run)
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
            on_subrun=on_subrun if journal is not None else None,
        )
        if retry.stopped:
            # Its journal record from before its last process stands, so a
            # later run goes on after the lines that process finished.
            return len(pending), 0, retry
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
                retry, device, canary, logs, drift_dir, journal
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
    if journal is not None:
        journal.record(retry.name, {"done": True, **_block_state(retry), **extra})
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
# {device: the CPUs its bench processes run on} (`_bench_placement`).
BENCH_CPUS: Dict[int, Tuple[int, ...]] = {}
TMP_DIR = "tmp"


def _bench_placement(
    devices: Sequence[int],
    physical: Mapping[int, Optional[int]],
    cpus: Iterable[int],
    run: Optional[Sequence[int]] = None,
    local_of: Callable[[int], Optional[Set[int]]] = lambda gpu: None,
) -> Tuple[Dict[int, Tuple[int, ...]], str]:
    """({device: its CPUs}, how they were split) of the stage's benches
    (lib/host_load.py `device_cpus`): `cpus` split over the run's physical
    GPUs `run` when they hold the GPU of every device (`physical`), else
    over the devices' own GPUs, `local_of(gpu)` giving a GPU's local CPUs;
    each device takes its GPU's share. Devices without a GPU of their own
    split `cpus` evenly by device."""
    mine = {int(d): physical.get(int(d)) for d in devices}
    own = {p for p in mine.values() if p is not None}
    if len(own) < len(mine):
        return host_load.device_cpus(sorted(mine), cpus)
    keys = sorted(set(run)) if run and own <= set(run) else sorted(own)
    shares, how = host_load.device_cpus(keys, cpus, {k: local_of(k) for k in keys})
    return {d: shares[p] for d, p in mine.items() if p in shares}, how


def _bench_launch(
    device: int,
) -> Tuple[str, Dict[str, str], Optional[Callable[[], None]]]:
    """(`--device` value, environment, what the process runs before its
    program) of a bench process on `device`: `kfd.one_gpu_launch`'s, and with
    the device's share of the stage's CPUs (BENCH_CPUS) the process runs on
    those alone, with an OpenMP team of their count unless the environment
    sets one."""
    bench_device, env = kfd.one_gpu_launch(device, bench_env_with(BENCH_ENV_EXTRA))
    cpus = BENCH_CPUS.get(int(device))
    if not cpus:
        return bench_device, env, None
    env.setdefault("OMP_NUM_THREADS", str(len(cpus)))
    return bench_device, env, lambda: os.sched_setaffinity(0, cpus)


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
    if args.node_faults is not None:
        store = node_faults.NodeStore(args.node_faults)
        node = node_faults.node_name()
        held = store.held(node)
        if held:
            ui.err("fault", node_faults.describe_hold(node, held, store.root))
            return procs.WEDGED_EXIT
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
    ap.add_argument(
        "--keep-passed",
        action="store_true",
        help="when an invocation over the same lines and settings passed in "
        "--out-dir (lib/bench_resume.py), keep its outputs and bench nothing; "
        "without it they move aside and the lines are benched again. One that "
        "an interruption cut off is continued either way.",
    )
    ap.add_argument(
        "--resume-across-code-change",
        action="store_true",
        help="continue the run in --out-dir although this stage's code changed "
        "since it ran (a bench fix that does not change what is measured, "
        "--note says why); the run records the change and the note, and each "
        "bench process its code. Without it a code change starts anew.",
    )
    ap.add_argument(
        "--note",
        default=None,
        help="with --resume-across-code-change: why the code change does not "
        "change what the stage measures",
    )
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
    ap.add_argument(
        "--min-block-lines",
        type=int,
        default=1,
        metavar="N",
        help="fewest lines of a block, at least 1 (bench.min_block_lines): "
        "block_size is raised to N, but to no more than ceil(n_shapes / "
        "n_gpus), one block per GPU, so that a stage makes at most "
        "ceil(n_shapes / N) blocks; the heavy lines' blocks are cut by "
        "--blocks-per-gpu alone. 1 (the default): the blocks of "
        "--blocks-per-gpu.",
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
        "--canary-tolerance-us",
        type=float,
        default=None,
        help="absolute floor of the canary's tolerance: a reading is over its "
        "GPU's reference only when it is above it by more than this many "
        "microseconds too (default: bench.canary.tolerance_us, else "
        f"{CANARY_DEFAULTS['tolerance_us']:g}, no floor).",
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
    ap.add_argument(
        "--node-faults",
        type=Path,
        default=None,
        metavar="DIR",
        help="apply the fault policy (lib/node_faults.py) with the node fault "
        "store DIR: bench nothing on a node it holds, stop and retire the node "
        "on error 719, and on error 700 re-run the line once to tell whether "
        "the fault follows it (default: no policy)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.pool_only_library and args.library_dir is None:
        ap.error("--pool-only-library needs --library-dir")
    if bool(args.identity_stem) != (args.identity_dir is not None):
        ap.error("--identity-dir and --identity-stem go together")
    if args.min_block_lines < 1:
        ap.error(f"--min-block-lines must be at least 1, got {args.min_block_lines}")
    if args.resume_across_code_change and not (args.note or "").strip():
        ap.error(
            "--resume-across-code-change needs --note: why the change does not "
            "change what the stage measures"
        )
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


def _heavy_operand_gib(cfg: Mapping[str, Any]) -> float:
    """`bench.heavy_operand_gib` of the config, else HEAVY_OPERAND_GIB of
    lib/bench_yaml.py (0); 0 keeps heavy lines among the others."""
    value = (cfg.get("bench") or {}).get("heavy_operand_gib") if cfg else None
    return float(bench_yaml.HEAVY_OPERAND_GIB if value is None else value)


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
    # A run of an earlier invocation over the same inputs and code that an
    # interruption cut off, or that failed: the stage continues it
    # (`_resume_blocks`, `_reopen_after_failure`) and keeps its block logs and
    # records; what it rewrites moves aside. One that passed stands with
    # --keep-passed: nothing to bench. Otherwise every earlier output moves
    # aside (`_journal_fit`).
    inputs = _resume_inputs(args, lines, library, cfg)
    code = bench_resume.file_digest(Path(__file__))
    journal = bench_resume.Journal.load(args.out_dir)
    how, change, why = _journal_fit(journal, inputs, code, args, lines)
    if how == "keep":
        ui.ok(
            "resume",
            f"nothing to bench: an invocation over the same lines and settings "
            f"ended in {args.out_dir} ({journal.last_run()}); its outputs stand "
            f"(move them aside to bench again)",
        )
        return 0
    if how == "fresh":
        journal = None
    if journal is None:
        n_removed = _clean_previous_outputs(args.out_dir, logs_dir)
        if n_removed and why:
            ui.info("resume", f"an earlier invocation's outputs move aside: {why}")
    elif change is not None:
        ui.warn(
            "resume",
            f"continuing the run in {args.out_dir} across a change of this stage's "
            f"code ({str(change['from'] or 'unrecorded')[:12]} -> "
            f"{str(change['to'])[:12]}): {change['note']}",
        )
    if journal is not None:
        n_removed = 0
        bench_resume.set_aside(
            args.out_dir,
            [
                p
                for p in _previous_outputs(args.out_dir, logs_dir)
                if p.parent == args.out_dir and p.is_file()
            ],
        )
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
    gpus = kfd.StageGpus(
        devices, run_root=host_load.run_root(), marker=Path(__file__).name
    )
    topology = {g.index: g for g in kfd.gpus()}
    shares, split = _bench_placement(
        devices,
        gpus.physical,
        os.sched_getaffinity(0),
        host_load.run_gpus(),
        lambda gpu: host_load.gpu_local_cpus(topology.get(gpu)),
    )
    BENCH_CPUS.clear()
    BENCH_CPUS.update(shares)
    if BENCH_CPUS and not args.quiet:
        ui.info(
            "cpus",
            f"benches run on their GPU's CPUs ({split}): "
            + "; ".join(
                f"gpu{d} {fingerprint.format_cpulist(c)}"
                for d, c in sorted(BENCH_CPUS.items())
            ),
        )
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
        faults=(
            node_faults.NodeStore(args.node_faults)
            if args.node_faults is not None
            else None
        ),
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
    heavy_gib = _heavy_operand_gib(cfg)
    min_lines = int(args.min_block_lines)
    blocks, block_size, n_heavy = _blocks_heavy_last(
        kept, numbers, n_gpus, int(args.blocks_per_gpu), heavy_gib, min_lines
    )
    blocks, resumed = _resume_blocks(journal, blocks, args.out_dir, logs_dir)
    failed_rc = (
        (journal.data.get("end") or {}).get("rc") if journal is not None else None
    )
    if resumed is None:
        journal = bench_resume.Journal.new(args.out_dir, inputs, code)
        change = None
    try:
        journal.begin(
            *_device_identity(devices, gpus),
            _layout_of(blocks),
            code=code,
            change=change,
        )
    except ValueError as exc:
        ui.warn("resume", str(exc))
        return 2
    after_failure: Optional[Dict[str, Any]] = None
    if resumed is not None and how == "after_failure":
        after_failure = _reopen_after_failure(
            blocks, journal, args.out_dir, logs_dir, failed_rc
        )
        journal.note_run("after_failure", after_failure)
        reopened = set(after_failure["reopened"])
        resumed["done"] = [n for n in resumed["done"] if n not in reopened]
        for b in blocks:
            if b.name in reopened:
                resumed["partial"][b.name] = b.next_offset
    if canary is not None:
        canary.run = journal.run
        canary.gpus = {int(d): journal.gpu_of(d) for d in devices}
        if resumed is not None:
            skip: Set[str] = set()
            if after_failure is not None:
                skip = set(after_failure["reopened"]) - set(
                    after_failure["crash_blocks"]
                )
            _restore_canary(canary, journal, devices, skip)
    tolerance = (
        int(args.max_crash_blocks)
        if args.max_crash_blocks is not None
        else default_crash_blocks(
            kept, numbers, n_gpus, int(args.blocks_per_gpu), heavy_gib
        )
    )

    if not args.quiet:
        ui.banner(f"stage03 bench launcher  ({n_gpus} devices)")
        ui.info("cfg", f"yaml         : {in_yaml}")
        ui.info("cfg", f"devices      : {devices}")
        ui.info("cfg", f"bench_binary : {args.bench_binary}")
        ui.info("cfg", f"out_dir      : {args.out_dir}")
        ui.info("cfg", f"n_shapes     : {n_shapes:,}")
        ui.info(
            "cfg",
            f"block_size   : {block_size}  (n_blocks {len(blocks):,}"
            + (f"; min_block_lines {min_lines}" if min_lines > 1 else "")
            + ")",
        )
        if n_heavy:
            ui.info(
                "cfg",
                f"heavy lines  : {n_heavy:,} with an operand of {heavy_gib:g} GiB "
                f"or more, in {sum(b.heavy for b in blocks)} block(s) after the "
                f"others",
            )
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
            ui.info(
                "cfg",
                f"moved {n_removed} output(s) of a previous run to "
                f"{bench_resume.SUPERSEDED}/",
            )
        if resumed is not None:
            ui.info(
                "cfg",
                f"resumed      : {len(resumed['done'])} block(s) done, "
                f"{len(resumed['partial'])} to continue (run {journal.run})",
            )
        if after_failure is not None:
            ui.info(
                "cfg",
                f"after failure: run {journal.run - 1} failed (exit "
                f"{after_failure['rc']}); the crash-block count starts anew; "
                f"benched again: {', '.join(after_failure['reopened']) or 'none'}",
            )

    done_before = set(resumed["done"]) if resumed is not None else set()
    work = _Work(
        [b for b in blocks if b.name not in done_before],
        args.max_quarantines,
        journal=journal,
    )
    work.done.extend(b for b in blocks if b.name in done_before)
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
    try:
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
    except BaseException:
        # Leaving early (an interrupt, or an error): the benches stop, and
        # the device workers end before the stage removes `work_dir`, so
        # that none writes a bench yaml into it meanwhile.
        children.stop_all()
        for w in workers:
            if w.is_alive():
                w.join(timeout=WORKER_STOP_S)
        left = [w.device for w in workers if w.is_alive()]
        if left:
            ui.warn(
                "interrupt",
                f"the workers of gpus {left} did not end within "
                f"{WORKER_STOP_S:.0f}s of the benches' stop",
            )
        raise
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
            journal=journal,
        )
        if not args.quiet:
            ui.ok("retry", f"recovered {recovered}/{retried} GEMM(s)")
    hung_retry = {
        int(h["yaml_line"]) for h in (retry_block.hung if retry_block else [])
    }
    # lines whose process left its GPU wedged in the retry pass: without data
    # as `wedged`, not as their source block's crash or unattempted line
    wedged_retry = {
        int(w["yaml_line"]) for w in (retry_block.wedge_lines if retry_block else [])
    }
    guard.settle_hung(done, hung_retry, wedged_retry)
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
            if off not in b.recovered and b.line_numbers[off] not in wedged_retry:
                hung = b.line_numbers[off] in hung_retry
                lost.append((b, off, "hung" if hung else "crashed"))
        left = (
            "quarantined" if id(b) in unbenched_ids and b.quarantines else "unattempted"
        )
        for off in b.unattempted:
            if off not in b.recovered and b.line_numbers[off] not in wedged_retry:
                hung = b.line_numbers[off] in hung_retry
                lost.append((b, off, "hung" if hung else left))
    hung_lines = [
        h
        for b in done
        for h in b.hung
        if h.get("retried") != "benched" and int(h["yaml_line"]) not in wedged_retry
    ]
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
    if guard.wedged_stop is not None:
        rc = procs.WEDGED_EXIT

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
        (int(f["yaml_line"]), f.get("shape") or ("", "", "", ""), f["code"], "fault")
        for b in attempted
        for f in b.faulted
    ]
    no_data += [
        (int(w["yaml_line"]), w.get("shape") or ("", "", "", ""), w["rc"], "wedged")
        for b in attempted
        for w in b.wedge_lines
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
            "bench_cpus": {
                "how": split,
                "cpus": {
                    str(d): fingerprint.format_cpulist(c)
                    for d, c in sorted(BENCH_CPUS.items())
                },
            },
            "n_shapes": n_shapes,
            "n_blocks": len(blocks),
            "block_size": block_size,
            "min_block_lines": min_lines,
            "heavy_operand_gib": heavy_gib,
            "heavy_lines": n_heavy,
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
            "after_failure": after_failure,
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
                    "heavy": b.heavy,
                    "health_events": list(b.health_events),
                    "hung": [h["yaml_line"] for h in b.hung],
                    "wedge_lines": [w["yaml_line"] for w in b.wedge_lines],
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
    if guard.wedged_stop is not None:
        ui.err(
            "wedge",
            f"{procs.WEDGED_TEXT}: {guard.wedged_stop}; reboot the host, then run "
            f"the same command to go on (the run's hold of a wedged GPU lifts by "
            f"itself after the reboot; when no process stays stuck in the GPU "
            f"driver, scripts/run_guard.py accept {guard.state.dir} lifts it "
            f"without one); exit {procs.WEDGED_EXIT}",
        )
    if not args.quiet:
        ui.ok("done", f"logs dir         : {logs_dir}")
    if crash_blocks:
        (ui.warn if rc == 0 else ui.err)(
            "tol",
            f"{len(crash_blocks)} crash block(s) "
            f"({', '.join(b.name for b in crash_blocks)}); tolerated {tolerance}"
            + ("" if rc == 0 else " -- failing (raise --max-crash-blocks to accept)"),
        )
    if guard.wedged_stop is None:
        # An invocation that ran to its end is not continued: passed, the
        # same one later benches nothing; failed, a later one starts anew. A
        # wedge's is, after the host's reboot.
        journal.end(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
