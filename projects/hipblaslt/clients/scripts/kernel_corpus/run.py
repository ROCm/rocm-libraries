#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run every kernel in an installed hipBLASLt library through fast_check, one kernel at a time.

Indexes the library (corpus_index.py), plans a small profile of problems for each solution
(plan.py), and runs them with the installed hipblaslt-test, each case pinned to its solution
index. Cases run in chunks, one hipblaslt-test process per chunk. A crash, GPU memory fault or
hang ends only the case that caused it: the runner records it, checks the GPU still works, and
continues with the rest of the chunk in a new process. Results are appended to
<out>/shard-<i>-of-<n>/results.jsonl as cases finish, so a rerun resumes where the last one
stopped.

    run.py --rocm-dir build/dist/rocm --out corpus-gfx90a

Sharding follows TheRock's test jobs: SHARD_INDEX (1-based) and TOTAL_SHARDS select a
contiguous slice of the corpus. Failures that match the known-bugs list (known_bugs.py) are
reported as known_fail. The exit code is 1 when a case failed, crashed or timed out outside that
list, when the list needs attention (an entry's cases all passed, or its review date has
passed), or when hipblaslt-test could not run a case; 2 for a usage error or a malformed list;
3 when the GPU stopped responding; and 0 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_index  # noqa: E402
import plan  # noqa: E402
import report  # noqa: E402

EXIT_FAILURES = 1
EXIT_USAGE = 2
EXIT_GPU_UNHEALTHY = 3

# Outcomes. Only the first group makes the run fail.
FAILING = ("fail", "crash", "timeout")
NOT_RUN = ("deferred", "unmapped", "unreachable", "unsupported", "harness_error")

_RUN = re.compile(r"^\[ RUN      \] (\S+)")
_OK = re.compile(r"^\[       OK \] (\S+) \((\d+) ms\)")
_FAILED = re.compile(r"^\[  FAILED  \] ([^,\s]+)(?:, where .*)?\((\d+) ms\)\s*$")
_SKIPPED = re.compile(r"^\[  SKIPPED \] (\S+) \((\d+) ms\)")
_BENCH = re.compile(r"^hipblaslt-bench .*")
_FAULT = re.compile(r"MEMORY_FAULT|illegal memory access|Memory access fault|page fault", re.I)
_DEVICE = re.compile(r"Device ID \d+ : (.*?) (gfx[0-9a-z]+)")
_CHIP_ID = re.compile(r"PCI Chip ID: 0x([0-9a-fA-F]+)")
_VERSION = re.compile(r"hipBLASLt git version: (\S+)")
_MAX_MESSAGE = 6000


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Tool:
    """The installed hipblaslt-test and the files it needs."""

    def __init__(self, rocm_dir: Path):
        self.rocm_dir = rocm_dir
        self.test_bin = rocm_dir / "bin" / "hipblaslt-test"
        self.common_yaml = rocm_dir / "bin" / "hipblaslt_common.yaml"
        for path in (self.test_bin, self.common_yaml):
            if not path.exists():
                raise FileNotFoundError(path)

    def env(self) -> dict:
        env = dict(os.environ)
        # Prints the hipblaslt-bench command for each launch, which becomes the reproduction.
        env.setdefault("HIPBLASLT_LOG_MASK", "32")
        # The host side of a smoke case is tiny; a full OpenMP team spins on it and halves the
        # case rate.
        env.setdefault("OMP_NUM_THREADS", "8")
        return env

    def device(self, attempts: int = 3) -> dict:
        """The device line hipblaslt-test prints at startup: name, arch and library version.
        A GPU just released by an earlier job can fail the first query, so it retries."""
        info = {"device": None, "arch": None, "hipblaslt_git_version": None, "output": ""}
        for attempt in range(attempts):
            proc = subprocess.run(
                [str(self.test_bin), "--gtest_list_tests", "--gtest_filter=-*"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=300,
                env=self.env(),
            )
            info["output"] = proc.stdout
            match = _DEVICE.search(proc.stdout)
            if match:
                info["device"], info["arch"] = match.group(1).strip(), match.group(2)
                break
            if attempt + 1 < attempts:
                time.sleep(15)
        match = _VERSION.search(info["output"])
        if match:
            info["hipblaslt_git_version"] = match.group(1)
        match = _CHIP_ID.search(info["output"])
        info["pci_chip_id"] = int(match.group(1), 16) if match else None
        return info

    def gpu_agent(self) -> dict:
        """The first GPU agent's compute-unit count and PCI chip ID from rocminfo, if available.
        Some solutions require one: gfx950 solutions name PCI chip IDs, and gfx942 solutions
        name CU counts. They are reachable only on a matching device."""
        for rocminfo in (self.rocm_dir / "bin" / "rocminfo", Path("/opt/rocm/bin/rocminfo")):
            if not rocminfo.exists():
                continue
            try:
                out = subprocess.run(
                    [str(rocminfo)], capture_output=True, text=True, timeout=120
                ).stdout
            except (OSError, subprocess.TimeoutExpired):
                continue
            return parse_rocminfo_gpu(out)
        return {"cu_count": None, "pci_chip_id": None}


def parse_rocminfo_gpu(text: str) -> dict:
    for agent in text.split("Agent ")[1:]:
        if re.search(r"Device Type:\s+GPU", agent):
            cu = re.search(r"Compute Unit:\s+(\d+)", agent)
            chip = re.search(r"Chip ID:\s+(\d+)", agent)
            return {
                "cu_count": int(cu.group(1)) if cu else None,
                "pci_chip_id": int(chip.group(1)) if chip else None,
            }
    return {"cu_count": None, "pci_chip_id": None}


# TensileLite's ChipIdRegistry (tensilelite/include/Tensile/AMDGPUPredicates.hpp): a device
# can run solutions for its own PCI chip ID and for the IDs it falls back to. Only gfx950
# libraries carry chip-ID predicates.
CHIP_ID_FALLBACKS = {
    0x75B0: (0x75A0,),
    0x75A2: (0x75A0,),
    0x75B2: (0x75A0,),
    0x75A3: (0x75A0,),
    0x75B3: (0x75A0,),
    0x75A8: (0x75A0,),
    0x75B8: (0x75A0,),
}


def unreachable_reason(record: dict, cu_count: Optional[int], chip_id: Optional[int]) -> str:
    """Why this device cannot run the solution, or an empty string. Unknown device facts never
    rule a solution out; the library's own check then refuses it."""
    hardware = record["hardware"]
    required_cu = hardware.get("cu_count")
    if required_cu and cu_count and required_cu != cu_count:
        return f"unreachable: needs {required_cu} CUs, device has {cu_count}"
    chip_ids = hardware.get("pci_chip_ids") or []
    if chip_ids and chip_id:
        usable = {chip_id, *CHIP_ID_FALLBACKS.get(chip_id, ())}
        if not usable.intersection(chip_ids):
            wanted = ", ".join(f"0x{c:x}" for c in chip_ids)
            return f"unreachable: needs PCI chip ID {wanted}, device is 0x{chip_id:x}"
    return ""


class ChunkRun:
    """Streams one hipblaslt-test process and turns its gtest output into case results."""

    def __init__(self, cases: dict[str, plan.Case], case_timeout: float, log):
        self.cases = cases
        self.case_timeout = case_timeout
        self.log = log
        self.results: list[dict] = []
        self.current: Optional[str] = None
        self.message: list[str] = []
        self.bench: Optional[str] = None
        self.fault = False
        self.version: Optional[str] = None

    def _finish(self, status: str, ms: Optional[int], **extra) -> None:
        case_id = self.current
        self.current = None
        if case_id is None or case_id not in self.cases:
            return
        message = "\n".join(self.message)[-_MAX_MESSAGE:]
        self.results.append(
            {
                "case_id": case_id,
                "status": status,
                "ms": ms,
                "message": message,
                "bench_cmd": self.bench,
                **extra,
            }
        )

    def feed(self, line: str) -> None:
        line = line.rstrip("\n")
        if self.version is None:
            match = _VERSION.search(line)
            if match:
                self.version = match.group(1)
        match = _RUN.match(line)
        if match:
            self.current = plan.case_id_from_test_name(match.group(1))
            self.message, self.bench, self.fault = [], None, False
            return
        if self.current is None:
            return
        match = _OK.match(line)
        if match:
            self._finish("pass", int(match.group(2)))
            return
        match = _SKIPPED.match(line)
        if match:
            self._finish("skip", int(match.group(2)))
            return
        match = _FAILED.match(line)
        if match:
            status = report.classify_failure("\n".join(self.message), self.fault)
            self._finish(status, int(match.group(2)), fault=self.fault)
            return
        if _BENCH.match(line):
            self.bench = line.strip()
            return
        if _FAULT.search(line):
            self.fault = True
        if len(self.message) < 200:
            self.message.append(line)

    def _line(self, line: str) -> None:
        self.log.write(line + "\n")
        self.feed(line)

    def run(self, cmd: list[str], env: dict, cwd: Path) -> str:
        """Runs the process to completion. Returns how it ended: exit, crash, timeout or fault."""
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Raw reads: a buffered readline can hold a case's RUN line where select cannot see it,
        # and a case that then hangs would never time out.
        fd = proc.stdout.fileno()
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)
        last_output = time.monotonic()
        pending = b""
        ending = "exit"
        try:
            while ending == "exit":
                if selector.select(timeout=5.0):
                    data = os.read(fd, 65536)
                    if not data:
                        if pending:
                            self._line(pending.decode(errors="replace"))
                        break
                    last_output = time.monotonic()
                    *lines, pending = (pending + data).split(b"\n")
                    for raw in lines:
                        self._line(raw.decode(errors="replace"))
                        if self.fault and self.current is not None:
                            # A memory fault poisons the process's GPU context; every later
                            # case in it would fail for that reason, so stop here.
                            ending = "fault"
                            break
                elif (
                    self.current is not None and time.monotonic() - last_output > self.case_timeout
                ):
                    ending = "timeout"
        finally:
            if proc.poll() is None:
                _kill_group(proc)
            proc.wait()
            selector.close()
            proc.stdout.close()
        if ending == "exit" and proc.returncode < 0:
            ending = "crash"
        if self.current is not None:
            if ending == "exit":
                ending = "crash"
            status = "timeout" if ending == "timeout" else "crash"
            self.message.append(
                f"[runner] process ended by {ending}, return code {proc.returncode}"
            )
            self._finish(status, None, fault=self.fault, returncode=proc.returncode)
        return ending


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


class Shard:
    def __init__(self, args: argparse.Namespace, tool: Tool, out: Path):
        self.args = args
        self.tool = tool
        self.out = out
        self.dir = out / f"shard-{args.shard_index}-of-{args.total_shards}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.jsonl"
        self.log_path = self.dir / "hipblaslt-test.log"
        self.records: dict[int, dict] = {}
        self.base: dict = {}
        self.canary_output = ""

    def done_case_ids(self) -> set[str]:
        """Cases with a final result. A harness error is retried on the next run, and a
        solution unreachable on one device is planned again on the next, which may reach it."""
        done = set()
        if self.results_path.exists():
            for row in corpus_index.read_jsonl(self.results_path):
                if row["status"] in ("harness_error", "unreachable"):
                    done.discard(row["case_id"])
                else:
                    done.add(row["case_id"])
        return done

    def append(self, rows: list[dict]) -> None:
        with self.results_path.open("a") as f:
            for row in rows:
                f.write(json.dumps({**self.base, **row}, sort_keys=True) + "\n")

    def describe(self, case: plan.Case) -> dict:
        record = self.records[case.index]
        s = case.shape
        return {
            "index": case.index,
            "shape": s.tag,
            "m": s.m,
            "n": s.n,
            "k": s.k,
            "batch": s.batch,
            "name": record["name"],
            "problem_type": corpus_index.problem_type_key(record),
            "types": case.types,
            "trans": case.trans_a + case.trans_b,
        }

    def healthy(self) -> bool:
        """Runs one tiny heuristic GEMM; a GPU that cannot do that is not worth testing on."""
        canary = plan.Case(
            -1,
            "canary",
            plan.Shape("exact", 64, 64, 64),
            {
                "a_type": "f32_r",
                "b_type": "f32_r",
                "c_type": "f32_r",
                "d_type": "f32_r",
                "compute_type": "c_f32_r",
                "scale_type": "f32_r",
            },
            "N",
            "N",
        )
        path = self.dir / "canary.yaml"
        plan.write_cases_yaml([canary], path, self.tool.common_yaml)
        text = (
            path.read_text()
            .replace("algo_method: 2", "algo_method: 0")
            .replace("solution_index: -1", "solution_index: 0")
        )
        path.write_text(text)
        for attempt in range(2):
            try:
                out = subprocess.run(
                    [
                        str(self.tool.test_bin),
                        "--yaml",
                        str(path),
                        f"--gtest_filter={plan.gtest_filter()}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    timeout=300,
                    cwd=self.dir,
                    env=self.tool.env(),
                ).stdout
                self.canary_output = out
                if re.search(r"\[  PASSED  \] 1 test", out):
                    return True
            except subprocess.TimeoutExpired:
                self.canary_output = "[runner] the one-GEMM check timed out after 300 s"
            time.sleep(30)
        return False

    def run_cases(self, cases: list[plan.Case]) -> int:
        """Runs the pending cases chunk by chunk. Returns an exit code."""
        size = max(self.args.chunk_cases, 1)
        total = len(cases)
        finished = 0
        started = time.monotonic()
        for start in range(0, total, size):
            pending = {c.case_id: c for c in cases[start : start + size]}
            stalls = 0
            while pending:
                chunk_path = self.dir / "chunk.yaml"
                plan.write_cases_yaml(list(pending.values()), chunk_path, self.tool.common_yaml)
                cmd = [
                    str(self.tool.test_bin),
                    "--yaml",
                    str(chunk_path),
                    f"--gtest_filter={plan.gtest_filter()}",
                ]
                with self.log_path.open("a") as log:
                    log.write(f"\n### {utc_now()} {' '.join(cmd)} ({len(pending)} cases)\n")
                    chunk = ChunkRun(pending, self.args.case_timeout, log)
                    ending = chunk.run(cmd, self.tool.env(), self.dir)
                if chunk.version:
                    self.base["hipblaslt_git_version"] = chunk.version
                rows = []
                for result in chunk.results:
                    case = pending.pop(result["case_id"], None)
                    if case is not None:
                        rows.append({**self.describe(case), **result, "time": utc_now()})
                self.append(rows)
                finished += len(rows)
                for row in rows:
                    if row["status"] in FAILING:
                        print(
                            f"  {row['status'].upper():8s} {row['case_id']:>14s} {row['problem_type']} "
                            f"M{row['m']} N{row['n']} K{row['k']} B{row['batch']}",
                            flush=True,
                        )
                if ending in ("crash", "timeout", "fault") and not self.healthy():
                    # Usually the GPU, but a hipblaslt-test that cannot run any case at all
                    # (mismatched test-data files, say) fails the check the same way.
                    print(
                        "error: hipblaslt-test no longer runs a basic GEMM; stopping this shard. "
                        "Its output:",
                        file=sys.stderr,
                    )
                    print(self.canary_output[-3000:], file=sys.stderr)
                    return EXIT_GPU_UNHEALTHY
                if not rows:
                    stalls += 1
                    if stalls >= 2:
                        # The process ran none of the cases, for example a YAML the generator
                        # rejects. Record them rather than loop.
                        tail = self.log_path.read_text()[-_MAX_MESSAGE:]
                        self.append(
                            [
                                {
                                    **self.describe(c),
                                    "case_id": c.case_id,
                                    "status": "harness_error",
                                    "message": tail,
                                    "time": utc_now(),
                                }
                                for c in pending.values()
                            ]
                        )
                        finished += len(pending)
                        pending.clear()
                else:
                    stalls = 0
            elapsed = time.monotonic() - started
            rate = finished / elapsed if elapsed > 0 else 0
            print(f"[{utc_now()}] {finished}/{total} cases, {rate:.1f} cases/s", flush=True)
        return 0


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = records
    if args.index_range:
        lo, _, hi = args.index_range.partition(":")
        lo_i = int(lo) if lo else 0
        hi_i = int(hi) if hi else 1 << 62
        selected = [r for r in selected if lo_i <= r["index"] < hi_i]
    if args.problem_filter:
        pattern = re.compile(args.problem_filter)
        selected = [r for r in selected if pattern.search(corpus_index.problem_type_key(r))]
    if args.dedupe != "none":
        field = "name" if args.dedupe == "name" else "kernel_name"
        seen, unique = set(), []
        for r in selected:
            key = (r[field], corpus_index.problem_type_key(r))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        selected = unique
    if args.per_type:
        taken: dict[str, int] = {}
        sample = []
        for r in selected:
            key = corpus_index.problem_type_key(r)
            if taken.get(key, 0) < args.per_type:
                taken[key] = taken.get(key, 0) + 1
                sample.append(r)
        selected = sample
    n, i = args.total_shards, args.shard_index
    lo, hi = (len(selected) * (i - 1)) // n, (len(selected) * i) // n
    selected = selected[lo:hi]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        default=corpus_index.default_rocm_dir(),
        help="Install tree with bin/hipblaslt-test and lib/hipblaslt/library "
        "(default: THEROCK_BIN_DIR/.. or ROCM_PATH)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Results directory")
    parser.add_argument("--arch", help="Library architecture to test (default: the device's)")
    parser.add_argument("--profile", default="smoke", choices=plan.PROFILES)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=env_int("SHARD_INDEX", 1),
        help="1-based shard (default: SHARD_INDEX or 1)",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=env_int("TOTAL_SHARDS", 1),
        help="Number of shards (default: TOTAL_SHARDS or 1)",
    )
    parser.add_argument(
        "--chunk-cases",
        type=int,
        default=400,
        help="Cases per hipblaslt-test process (default: 400)",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=300.0,
        help="Seconds without output before a running case counts as hung",
    )
    parser.add_argument("--index-range", help="Solution index range lo:hi (hi exclusive)")
    parser.add_argument("--problem-filter", help="Regex on the problem-type key, e.g. 'Half/Half'")
    parser.add_argument(
        "--dedupe",
        default="none",
        choices=("none", "name", "kernel"),
        help="Test one solution per solution name or per code-object kernel "
        "within a problem type (default: test every solution)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        help="Test only the first N solutions of each problem type, a quick stratified sample",
    )
    parser.add_argument("--limit", type=int, help="Test at most this many solutions")
    parser.add_argument("--cu-count", type=int, help="Device CU count (default: from rocminfo)")
    parser.add_argument(
        "--pci-chip-id",
        type=lambda s: int(s, 0),
        help="Device PCI chip ID, e.g. 0x75a0 (default: from rocminfo)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and report; run nothing")
    report.add_known_bugs_argument(parser)
    args = parser.parse_args(argv)
    if not args.rocm_dir:
        parser.error("--rocm-dir is required when THEROCK_BIN_DIR and ROCM_PATH are unset")
    if not 1 <= args.shard_index <= args.total_shards:
        parser.error("need 1 <= shard index <= total shards")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        known = report.load_known_bugs(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    try:
        tool = Tool(args.rocm_dir.resolve())
    except FileNotFoundError as e:
        print(f"error: missing {e}", file=sys.stderr)
        return EXIT_USAGE
    device = tool.device()
    arch = args.arch or device["arch"]
    if not arch:
        print(
            "error: could not detect the GPU architecture; pass --arch. hipblaslt-test said:",
            file=sys.stderr,
        )
        print(device["output"][-3000:], file=sys.stderr)
        return EXIT_GPU_UNHEALTHY
    lib_dir = corpus_index.library_dir(tool.rocm_dir, arch)
    if not lib_dir.is_dir():
        print(f"error: no hipBLASLt library for {arch} at {lib_dir}", file=sys.stderr)
        return EXIT_USAGE
    agent = tool.gpu_agent() if not (args.cu_count and args.pci_chip_id) else {}
    cu_count = args.cu_count or agent.get("cu_count")
    # The chip ID hipblaslt-test reports is the one the library's predicates see.
    chip_id = args.pci_chip_id or device.get("pci_chip_id") or agent.get("pci_chip_id")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.jsonl"
    if not corpus_path.exists():
        corpus_index.write_jsonl(corpus_index.build_corpus(lib_dir, arch), corpus_path)
    records = corpus_index.read_jsonl(corpus_path)
    selected = select_records(records, args)

    shard = Shard(args, tool, out)
    shard.records = {r["index"]: r for r in selected}
    shard.base = {
        "arch": arch,
        "device": device["device"],
        "host": socket.gethostname(),
        "hipblaslt_git_version": device["hipblaslt_git_version"],
        "profile": args.profile,
    }
    meta = {
        **shard.base,
        "rocm_dir": str(tool.rocm_dir),
        "cu_count": cu_count,
        "pci_chip_id": chip_id,
        "shard_index": args.shard_index,
        "total_shards": args.total_shards,
        "solutions_in_corpus": len(records),
        "solutions_in_shard": len(selected),
        "started": utc_now(),
        "argv": sys.argv[1:],
    }
    (shard.dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"{device['device']} {arch} ({cu_count} CUs, chip ID "
        f"{hex(chip_id) if chip_id else '?'}), hipBLASLt {device['hipblaslt_git_version']}: "
        f"shard {args.shard_index}/{args.total_shards} has {len(selected)} of {len(records)} "
        f"solutions",
        flush=True,
    )

    done = shard.done_case_ids()
    cases: list[plan.Case] = []
    not_planned: list[dict] = []
    for record in selected:
        reason = unreachable_reason(record, cu_count, chip_id)
        kernel_cases = []
        if not reason:
            kernel_cases, reason = plan.plan_kernel(record, args.profile)
        if reason:
            case_id = f"{record['index']}:-"
            if case_id not in done:
                status = reason.split(":")[0] if reason.split(":")[0] in NOT_RUN else "unmapped"
                not_planned.append(
                    {
                        "case_id": case_id,
                        "index": record["index"],
                        "shape": None,
                        "name": record["name"],
                        "problem_type": corpus_index.problem_type_key(record),
                        "status": status,
                        "message": reason,
                        "time": utc_now(),
                    }
                )
            continue
        cases.extend(c for c in kernel_cases if c.case_id not in done)
    shard.append(not_planned)
    print(
        f"{len(cases)} cases to run ({len(done)} already done, {len(not_planned)} solutions "
        f"not planned this time)",
        flush=True,
    )

    code = 0
    if not args.dry_run and cases:
        code = shard.run_cases(cases)
    summary = report.write_reports(out, known=known)
    if code == 0 and (report.run_failed(summary) or summary["case_status"].get("harness_error")):
        code = EXIT_FAILURES
    meta["finished"] = utc_now()
    meta["exit_code"] = code
    (shard.dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
