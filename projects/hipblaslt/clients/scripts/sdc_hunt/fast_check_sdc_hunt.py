#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hunt intermittent silent data corruption with fast_check, one environment axis at a time.

Runs hipblaslt-test's fast_check cases (by default the on-demand sdc_hunt category) under each
combination of the requested environment axes, and appends one JSON line per run to a results
file. Every line records the environment the run saw, so a failure can be matched to it:

  - concurrency: a second GPU workload running alongside (another GEMM, a CU-occupying cotenant
    kernel, or any command, such as a copy loop or the EDP helper);
  - XNACK: HSA_XNACK unset, 0 or 1;
  - preemption: the amdgpu cwsr_enable module parameter, which needs a reload to change, so it
    is recorded rather than toggled;
  - placement: every buffer a failing case reports crossing a 4 GiB boundary. If failures line up
    with such buffers they are carry-drop defects (AIHPBLAS-4994), not races.

    fast_check_sdc_hunt.py --test-bin build/release/clients/hipblaslt-test \\
        --xnack unset 0 --load none gemm --runs 3 --results sdc.jsonl

This is a manual recipe for dedicated hardware, not a CI job: a concurrent workload and repeated
runs take the GPU for a long time, and on gfx1250 a fault can wedge it (ROCM-32049).
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def command_output(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_description() -> dict:
    """GPU names and architectures from rocminfo, in device order."""
    out = command_output(["rocminfo"]) or ""
    names = re.findall(r"Marketing Name:\s+(.+)", out)
    archs = sorted(set(re.findall(r"Name:\s+(gfx\w+)", out)))
    gpus = [n.strip() for n in names if "CPU" not in n and n.strip()]
    return {"gpus": gpus, "archs": archs}


def environment_record(xnack: str, load: str) -> dict:
    """Everything about the machine and driver that could change a result."""
    params = Path("/sys/module/amdgpu/parameters")
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    record = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host": socket.gethostname(),
        "kernel": os.uname().release,
        "amdgpu_version": read_text(Path("/sys/module/amdgpu/version")),
        "cwsr_enable": read_text(params / "cwsr_enable"),
        "noretry": read_text(params / "noretry"),
        "rocm_version": read_text(Path(rocm) / ".info" / "version"),
        "HSA_XNACK": None if xnack == "unset" else xnack,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "TENSILE_SOLUTION_SELECTION_METHOD": os.environ.get(
            "TENSILE_SOLUTION_SELECTION_METHOD"
        ),
        "load": load,
    }
    record.update(gpu_description())
    return record


def load_command(load: str, test_bin: Path) -> Optional[list[str]]:
    """The background workload for one load setting, or None for no load."""
    if load == "none":
        return None
    if load == "gemm":
        bench = test_bin.parent / "hipblaslt-bench"
        return [
            str(bench),
            "-m", "8192", "-n", "8192", "-k", "8192",
            "--precision", "bf16_r", "--compute_type", "f32_r",
            "-i", "1000000", "-j", "0",
        ]  # fmt: skip
    if load.startswith("cotenant:"):
        cotenant = SCRIPT_DIR.parent / "cotenant" / "hipblaslt-cotenant"
        return [
            str(cotenant),
            "--cus",
            load.split(":", 1)[1],
            "--",
            "sleep",
            "infinity",
        ]
    if load.startswith("command:"):
        return shlex.split(load.split(":", 1)[1])
    raise ValueError(f"unknown load '{load}'")


def parse_run(output: str) -> dict:
    """Failing tests, failing solutions and 4 GiB crossings from hipblaslt-test output."""
    failed = sorted(
        set(re.findall(r"^\[  FAILED  \] (\S+?)(?:,|$)", output, re.MULTILINE))
    )
    solutions = sorted(
        set(
            re.findall(
                r"^  solution \d+ \(library index \d+, kernel [^)]*\)(?::.*)?$",
                output,
                re.MULTILINE,
            )
        )
    )
    crossings = sorted(
        set(
            re.findall(
                r"^\s+(\S+): (0x[0-9a-f]+ to 0x[0-9a-f]+) \(\d+ bytes\), crosses a 4 GiB boundary",
                output,
                re.MULTILINE,
            )
        )
    )
    passed = re.search(r"^\[  PASSED  \] (\d+) tests?", output, re.MULTILINE)
    return {
        "tests_passed": int(passed.group(1)) if passed else 0,
        "tests_failed": failed,
        "failing_solutions": solutions,
        "buffers_crossing_4gib": [f"{name}: {span}" for name, span in crossings],
    }


def run_once(args: argparse.Namespace, xnack: str, load: str, index: int) -> dict:
    env = dict(os.environ)
    if xnack == "unset":
        env.pop("HSA_XNACK", None)
    else:
        env["HSA_XNACK"] = xnack

    cmd = [str(args.test_bin), f"--gtest_filter={args.filter}"]
    record = environment_record(xnack, load)
    record.update({"run": index, "command": shlex.join(cmd)})
    stem = (
        f"{args.invocation}.run{index}_{xnack}_{re.sub(r'[^A-Za-z0-9._-]+', '-', load)}"
    )

    background, load_log = None, None
    bg_cmd = load_command(load, args.test_bin)
    # The load is stopped however this run ends, including on Ctrl-C while it settles, so it
    # never carries over into a later combination.
    try:
        if bg_cmd:
            load_log = args.results.with_suffix(f".{stem}.load.log")
            with load_log.open("w") as bg_out:
                background = subprocess.Popen(
                    bg_cmd, env=env, stdout=bg_out, stderr=subprocess.STDOUT
                )
            problem = wait_for_load(background, load, load_log, args)
            if problem:
                record.update(
                    {
                        "exit_code": "load_failed",
                        "load_error": problem,
                        "load_log": str(load_log),
                        "seconds": 0.0,
                    }
                )
                record.update(parse_run(""))
                return record

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            output, code = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            output = as_text(e.stdout) + as_text(e.stderr)
            code = "timeout"
        if background:
            # A load that stopped early left part of the run uncontended.
            record["load_ran_throughout"] = background.poll() is None
    finally:
        if background:
            stop(background)

    record.update({"exit_code": code, "seconds": round(time.monotonic() - start, 1)})
    record.update(parse_run(output))
    if code == 0 and not record["tests_passed"] and not record["tests_failed"]:
        record["error"] = "no tests ran; check --filter and --test-bin"
    if is_failure(record):
        log = args.results.with_suffix(f".{stem}.log")
        log.write_text(output)
        record["log"] = str(log)
    if load_log and is_failure(record):
        record["load_log"] = str(load_log)
    elif load_log:
        load_log.unlink(missing_ok=True)
    return record


def as_text(data) -> str:
    """Partial output from subprocess.TimeoutExpired, which may be bytes, text or None."""
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_load(
    background: subprocess.Popen, load: str, log: Path, args: argparse.Namespace
) -> Optional[str]:
    """Waits until the background load is running. Returns why it is not, or None.

    The cotenant logs READY once its kernels are resident; other loads get a fixed settle time.
    """
    if load.startswith("cotenant:"):
        deadline = time.monotonic() + args.load_ready_seconds
        while time.monotonic() < deadline:
            if background.poll() is not None:
                return f"the cotenant exited with code {background.returncode} before READY"
            if "READY" in (read_text(log) or ""):
                return None
            time.sleep(0.5)
        return f"the cotenant did not report READY within {args.load_ready_seconds} s"
    time.sleep(args.load_settle_seconds)
    if background.poll() is not None:
        return (
            f"the load exited with code {background.returncode} before the run started"
        )
    return None


def is_failure(record: dict) -> bool:
    return bool(
        record["tests_failed"] or record["exit_code"] != 0 or record.get("error")
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-bin", type=Path, required=True, help="path to hipblaslt-test"
    )
    parser.add_argument(
        "--filter",
        default="*sdc_hunt*",
        help="gtest filter selecting the fast_check cases to run (default: %(default)s)",
    )
    parser.add_argument(
        "--xnack",
        nargs="+",
        default=["unset"],
        choices=["unset", "0", "1"],
        help="HSA_XNACK settings to run under (default: unset)",
    )
    parser.add_argument(
        "--load",
        nargs="+",
        default=["none"],
        help="background workloads: none, gemm (hipblaslt-bench), cotenant:<CUs>, or "
        "'command:<cmd>' (default: none)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="repeats of every combination (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="seconds per run (default: %(default)s)",
    )
    parser.add_argument(
        "--load-settle-seconds",
        type=float,
        default=5.0,
        help="seconds to let the background load start (default: %(default)s)",
    )
    parser.add_argument(
        "--load-ready-seconds",
        type=float,
        default=120.0,
        help="seconds to wait for the cotenant's READY (default: %(default)s)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("sdc_hunt_results.jsonl"),
        help="JSON Lines file to append one record per run to (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    for name in ("load_settle_seconds", "load_ready_seconds"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must not be negative")
    if args.timeout <= 0 or args.runs < 1:
        parser.error("--timeout must be positive and --runs at least 1")
    args.results = args.results.resolve()
    # Names this invocation's logs, so a later invocation appending to the same results file
    # does not overwrite them.
    args.invocation = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    any_failed = False
    combos = list(itertools.product(args.xnack, args.load))
    for index, (xnack, load) in enumerate(
        (c for _ in range(args.runs) for c in combos), start=1
    ):
        record = run_once(args, xnack, load, index)
        with args.results.open("a") as f:
            f.write(json.dumps(record) + "\n")
        verdict = "FAIL" if is_failure(record) else "pass"
        any_failed |= verdict == "FAIL"
        print(
            f"run {index}: xnack={xnack} load={load} cwsr={record['cwsr_enable']} "
            f"-> {verdict} ({record['tests_passed']} passed, "
            f"{len(record['tests_failed'])} failed, {record['seconds']} s)"
        )
        for problem in (record.get("load_error"), record.get("error")):
            if problem:
                print(f"  {problem}")
        if record.get("load_ran_throughout") is False:
            print("  the load stopped during the run, so part of it was uncontended")
        for crossing in record["buffers_crossing_4gib"]:
            print(f"  crosses 4 GiB: {crossing}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
