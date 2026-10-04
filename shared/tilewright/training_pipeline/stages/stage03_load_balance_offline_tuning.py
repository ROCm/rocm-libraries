# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03_load_balance_offline_tuning -- bench every shape with
hipblaslt-bench on all GPUs: a dynamic work queue, per-shape crash recovery,
and a retry pass.

Reads `<in-dir>/shapes_probed.yaml` (stage02) or else `<in-dir>/shapes.yaml`
(stage01) and cuts it into blocks. One worker thread per GPU pulls the next
block whenever it finishes one, so faster GPUs take more blocks.

Crash recovery within a block: when a bench process exits non-zero or is
killed by the watchdog, the problems it got past are counted from its output
(`lib.bench_log.count_finished_problems`); the next shape is taken to be the
one that crashed it and is recorded as skipped, and a new process resumes
after it. A block whose processes keep dying without finishing a shape is
abandoned, and its remaining shapes are recorded as unattempted.

Retry pass: skipped and unattempted shapes, up to --max-retry-gemms of them,
are benched once more in one process on the first device (crashes under heavy
multi-GPU load often do not repeat in isolation). --no-retry-skipped turns it
off.

Outputs in `<out-dir>/`, removed and rewritten by every run:
  logs/block_NNNN_linesX-Y_gpuD.log  one log per block (block id, 1-based
                                     source-yaml line range, GPU)
  logs/block_retry_gpuD.log          the retry pass; stage04 reads it like any
                                     other block log
  logs/warmup_NNNN_gpuD.log          discarded warm-up output (the prefix keeps
                                     it out of stage04's `block_*` glob)
  skipped_gemms.txt                  shapes still without data after the retry
                                     pass: yaml_line,M,N,K,B,rc,reason with
                                     reason `crashed` or `unattempted`
  stage03_summary.json               counts, crash blocks, tolerance, rc
  tmp/                               bench yamls of the running processes,
                                     removed when the stage ends

A crash block is a block that was abandoned and whose unattempted shapes the
retry pass did not all recover. The stage exits 0 when the number of crash
blocks is within --max-crash-blocks (default: one block in twenty, rounded
down), else 2. Individually skipped shapes are reported but never fail it.

Every bench process leads its own process group. SIGINT, SIGTERM or SIGHUP
stops all of them (`lib.procs`) and the stage exits with 128 + the signal
number.
"""
from __future__ import annotations

import argparse
import math
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import procs  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_env import bench_env_with  # noqa: E402
from lib.bench_log import (  # noqa: E402
    LogProbe,
    ProblemBoundaryTracker,
    count_finished_problems,
)
from lib.bench_yaml import ADAPTIVE_KEYS, read_bench_lines, shape_of_line  # noqa: E402
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
    commas inside values). Key order is kept; forced keys not present are
    appended."""
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


def _stream_split(proc_stdout, warmup_f, block_f) -> float:
    """Copy the bench's output line by line: the first problem (the warm-up)
    to `warmup_f`, every later problem to `block_f`, and the process banner
    before the first problem to both. Returns the seconds spent before the
    second problem started, i.e. the warm-up's cost."""
    t0 = time.time()
    tracker = ProblemBoundaryTracker()
    problems = 0
    warmup_s = 0.0
    while True:
        line = proc_stdout.readline()
        if not line:
            break
        if tracker.feed(line):
            problems += 1
            if problems == 2:
                warmup_s = time.time() - t0
        if problems == 0:
            warmup_f.write(line)
            block_f.write(line)
        elif problems == 1:
            warmup_f.write(line)
        else:
            block_f.write(line)
    if problems < 2:
        warmup_s = time.time() - t0
    return warmup_s


# ── crash-recovery runner (one block on one device) ──────────────────────────

# Consecutive processes that finished no shape before the block is abandoned.
# Each such process was most likely killed while still loading kernel
# modules, and a new process pays that load again.
_MAX_ZERO_PROGRESS_SUBRUNS = 3


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
) -> None:
    """Bench `block` on `device`, appending all output to `log_path`. After a
    crash the shape after the last finished one is skipped and a new process
    resumes behind it (see the module docstring). With `warmup`, every
    process gets the throwaway warm-up shape prepended and its output split
    into `warmup_log_path`; `on_warmup(subrun, seconds)` reports its cost.
    Processes start through `children` with their yaml in `work_dir`
    (default: the log's directory)."""
    children = children if children is not None else procs.ChildGroups()
    work_dir = work_dir if work_dir is not None else log_path.parent
    poll_s = procs.poll_interval(stall_s)
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
        warmup_line = (
            render_warmup_line(remaining[0])
            if (warmup and len(remaining) >= 2)
            else None
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
            if warmup_line:
                f.write(warmup_line + "\n")
            f.writelines(remaining)
        sub_start = file_size(log_path)
        progress = LogProbe(log_path, sub_start)
        cmd = [str(bench_binary), "--device", str(device), "--yaml", str(tmp_yaml)]
        reason: Optional[str] = None
        try:
            if warmup_line and warmup_log_path is not None:
                warm = LogProbe(warmup_log_path, file_size(warmup_log_path))
                with warmup_log_path.open("a", buffering=1) as wf, log_path.open(
                    "a", buffering=1
                ) as lf:
                    wf.write(
                        f"*** WARM-UP (discarded) sub-run #{subrun + 1}, "
                        f"{block.name}, gpu {device} ***\n"
                    )
                    proc = children.popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=bench_env_with(),
                    )
                    # The split has to run on its own thread: a hung process
                    # keeps its pipe open, and the watchdog below must still
                    # get to kill it.
                    warm_s = {"s": 0.0}

                    def _drain() -> None:
                        try:
                            warm_s["s"] = _stream_split(proc.stdout, wf, lf)
                        except Exception:
                            pass

                    drain = threading.Thread(target=_drain, daemon=True)
                    drain.start()
                    last_rc, reason = procs.wait_with_stall_watchdog(
                        proc,
                        lambda: file_size(log_path) + file_size(warmup_log_path),
                        stall_s,
                        poll_s=poll_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=progress.progressed,
                        probe_fatal=lambda: warm.fatal_seen() or progress.fatal_seen(),
                    )
                    drain.join(timeout=30)
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                if on_warmup is not None:
                    on_warmup(subrun + 1, warm_s["s"])
            else:
                with log_path.open("a", buffering=1) as lf:
                    proc = children.popen(
                        cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=bench_env_with(),
                    )
                    last_rc, reason = procs.wait_with_stall_watchdog(
                        proc,
                        lambda: file_size(log_path),
                        stall_s,
                        poll_s=poll_s,
                        startup_grace_s=startup_grace_s,
                        probe_progress=progress.progressed,
                        probe_fatal=progress.fatal_seen,
                    )
        finally:
            try:
                tmp_yaml.unlink()
            except OSError:
                pass
        text = _read_from(log_path, sub_start)
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


# ── device worker ────────────────────────────────────────────────────────────


class _DeviceWorker(threading.Thread):
    """Pull blocks off `q`, run each on `device`, append it to `done`."""

    def __init__(
        self,
        device: int,
        q: queue.Queue,
        done: List[TaskBlock],
        done_lock: threading.Lock,
        bench_binary: Path,
        logs_dir: Path,
        *,
        warmup: bool = True,
        stall_s: float = 300.0,
        startup_grace_s: float = 900.0,
        quiet: bool = False,
        children: Optional[procs.ChildGroups] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(name=f"bench-d{device}", daemon=True)
        self.device = device
        self.q = q
        self.done = done
        self.done_lock = done_lock
        self.bench_binary = bench_binary
        self.logs_dir = logs_dir
        self.warmup = warmup
        self.stall_s = stall_s
        self.startup_grace_s = startup_grace_s
        self.quiet = quiet
        self.children = children
        self.work_dir = work_dir
        self.n_blocks_completed = 0
        self.wall_s = 0.0
        self.warmup_s = 0.0

    def run(self) -> None:
        t0 = time.time()
        while True:
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                break
            log_path = self.logs_dir / f"{block.name}_gpu{self.device}.log"
            warmup_log_path = (
                self.logs_dir / f"warmup_{block.block_id:04d}_gpu{self.device}.log"
            )

            def _on_warmup(
                subrun: int, secs: float, _b=block, _w=warmup_log_path
            ) -> None:
                self.warmup_s += secs
                if not self.quiet:
                    ui.grey(
                        "warm",
                        f"gpu{self.device} {_b.name} sub-run #{subrun}: warm-up "
                        f"took {ui.fmt_dur(secs)} (discarded -> {_w.name})",
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
                )
            except Exception as exc:
                block.rc = -1
                block.abandoned = True
                _annotate(log_path, f"\n*** ERROR: {exc} ***\n")
            with self.done_lock:
                self.done.append(block)
            self.q.task_done()
            self.n_blocks_completed += 1
        self.wall_s = time.time() - t0


# ── main ─────────────────────────────────────────────────────────────────────


def _resolve_warmup(args: argparse.Namespace) -> Tuple[bool, str]:
    """CLI flag > `bench.warmup_pass` in --config-yaml > on."""
    if args.warmup_pass is not None:
        return bool(args.warmup_pass), "cli"
    if args.config_yaml is not None and args.config_yaml.exists():
        try:
            import yaml

            with args.config_yaml.open() as f:
                cfg = yaml.safe_load(f) or {}
            v = (cfg.get("bench") or {}).get("warmup_pass")
            if v is not None:
                return bool(v), "config"
        except Exception as exc:
            ui.warn(
                "cfg",
                f"could not read bench.warmup_pass from {args.config_yaml}: "
                f"{exc}; using the default (on)",
            )
    return True, "default"


def _clean_previous_outputs(out_dir: Path, logs_dir: Path) -> int:
    removed = 0
    for pattern in ("block_*.log", "warmup_*.log"):
        for p in logs_dir.glob(pattern):
            p.unlink()
            removed += 1
    for name in ("skipped_gemms.txt", "stage03_summary.json"):
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    return removed


def _retry_pass(
    blocks: List[TaskBlock],
    device: int,
    bench_binary: Path,
    logs_dir: Path,
    stall_s: float,
    startup_grace_s: float,
    children: Optional[procs.ChildGroups] = None,
    work_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Re-bench every skipped and unattempted shape of `blocks` in one process
    on `device`, marking the ones that finish in `TaskBlock.recovered`.
    Returns (retried, recovered)."""
    pending = [
        (b, off) for b in blocks for off in sorted(set(b.skipped) | set(b.unattempted))
    ]
    if not pending:
        return 0, 0
    retry = TaskBlock(
        block_id=-1,
        name="block_retry",
        lines=[b.lines[off] for b, off in pending],
        line_numbers=[b.line_numbers[off] for b, off in pending],
    )
    _run_block_with_recovery(
        retry,
        device,
        bench_binary,
        logs_dir / f"{retry.name}_gpu{device}.log",
        warmup=False,
        stall_s=stall_s,
        startup_grace_s=startup_grace_s,
        children=children,
        work_dir=work_dir,
    )
    failed = set(retry.skipped) | set(retry.unattempted)
    recovered = 0
    for i, (b, off) in enumerate(pending):
        if i not in failed:
            b.recovered.add(off)
            recovered += 1
    return len(pending), recovered


def main() -> int:
    args = _parse_args()
    work_dir = args.out_dir / "tmp"
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        try:
            return _run(args, children, work_dir)
        finally:
            children.stop_all()
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03 -- per-GPU hipblaslt-bench launcher (work queue, "
        "crash recovery, retry pass)."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="stage02 dir (shapes_probed.yaml) or stage01 dir (shapes.yaml).",
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
        help="run config; only bench.warmup_pass is read, when neither "
        "--warmup-pass nor --no-warmup-pass is given.",
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
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _run(args: argparse.Namespace, children: procs.ChildGroups, work_dir: Path) -> int:
    warmup, warmup_src = _resolve_warmup(args)
    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    if not devices:
        ui.err("err", "--devices is empty")
        return 1
    probed = args.in_dir / "shapes_probed.yaml"
    raw = args.in_dir / "shapes.yaml"
    if probed.exists():
        in_yaml = probed
    elif raw.exists():
        in_yaml = raw
        if not args.quiet:
            ui.warn(
                "yaml",
                f"shapes_probed.yaml not found; benching the uncalibrated "
                f"{raw.name}",
            )
    else:
        ui.err("err", f"neither {probed} nor {raw} exists")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"hipblaslt-bench not found: {args.bench_binary}")
        return 1
    lines = read_bench_lines(in_yaml)
    n_shapes = len(lines)
    if n_shapes == 0:
        ui.err("err", f"no shape lines in {in_yaml}")
        return 1

    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    n_removed = _clean_previous_outputs(args.out_dir, logs_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir()

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
        if n_removed:
            ui.info("cfg", f"removed {n_removed} output file(s) of a previous run")

    q: queue.Queue = queue.Queue()
    for b in blocks:
        q.put(b)
    done: List[TaskBlock] = []
    done_lock = threading.Lock()
    if not args.quiet:
        ui.banner("Bench launch (work queue + crash recovery)", ui.C.BLUE)
    workers = [
        _DeviceWorker(
            dev,
            q,
            done,
            done_lock,
            args.bench_binary,
            logs_dir,
            warmup=warmup,
            stall_s=float(args.stall_timeout_s),
            startup_grace_s=float(args.startup_grace_s),
            quiet=args.quiet,
            children=children,
            work_dir=work_dir,
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
            with done_lock:
                n_done = len(done)
                n_skip = sum(len(b.skipped) for b in done)
            alive = [w.device for w in workers if w.is_alive()]
            ui.grey(
                "bench",
                f"blocks_done={n_done:>4d}/{len(blocks):<4d}  alive_gpus={alive}  "
                f"skipped_gemms={n_skip}  elapsed={ui.fmt_dur(time.time() - t0)}",
            )
            last_print = time.time()
    for w in workers:
        w.join()
    done_ids = {id(b) for b in done}
    for b in blocks:
        if id(b) not in done_ids:
            b.abandoned = True
            done.append(b)
    done.sort(key=lambda b: b.block_id)
    elapsed = time.time() - t0

    n_skipped = sum(len(b.skipped) for b in done)
    n_unattempted = sum(len(b.unattempted) for b in done)
    n_pending = n_skipped + n_unattempted
    retried = recovered = 0
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
                f"Retry pass: re-benching {n_pending} GEMM(s) on device {devices[0]}",
                ui.C.BLUE,
            )
        retried, recovered = _retry_pass(
            done,
            devices[0],
            args.bench_binary,
            logs_dir,
            float(args.stall_timeout_s),
            float(args.startup_grace_s),
            children=children,
            work_dir=work_dir,
        )
        if not args.quiet:
            ui.ok("retry", f"recovered {recovered}/{retried} GEMM(s)")

    lost: List[Tuple[TaskBlock, int, str]] = []
    for b in done:
        for off in sorted(b.skipped):
            if off not in b.recovered:
                lost.append((b, off, "crashed"))
        for off in b.unattempted:
            if off not in b.recovered:
                lost.append((b, off, "unattempted"))
    crash_blocks = [
        b for b in done if any(off not in b.recovered for off in b.unattempted)
    ]
    rc = 0 if len(crash_blocks) <= tolerance else 2

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
            "rc": rc,
        },
    )

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN if rc == 0 and not lost else ui.C.YELLOW)
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
