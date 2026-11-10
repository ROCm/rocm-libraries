#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run hipblaslt-test's solution-index cases under rocjitsu, surviving crashes.

hipblaslt-test builds, filters and checks each case itself; gtest reports
the outcomes. When a case crashes or hangs, gtest writes no report, so this
records the cases that finished, marks the running one, and resumes with
the indices that never started.

Some solutions run only on a particular CU count or PCI chip ID. Give one
--config per emulated device, most general first: each later device runs
only the indices that every earlier one skipped because the solution's
hardware predicate rejected it.

Example::

    run_solution_indices.py --rocjitsu rocjitsu --client hipblaslt-test \\
        --library lib/hipblaslt/library/gfx950 --range 0:4096 \\
        --config gfx950-75a3.json --config gfx950-75a8.json \\
        --out results/gfx950
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

TEST = r"Library/SolutionIndexTest\.matmul/idx(\d+)"
RUN = re.compile(rf"^\[ RUN      \] {TEST}$", re.M)
END = re.compile(rf"^\[ +(OK|FAILED|SKIPPED) +\] {TEST}.*\(\d+ ms\)$", re.M)


def indices_from(args):
    if args.range:
        begin, end = map(int, args.range.split(":"))
        return list(range(begin, end))
    return [int(line) for line in Path(args.indices).read_text().split()]


def run_once(args, config_path, out, indices):
    """Run the given indices; return {index: outcome} for every case that ended."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "indices.txt").write_text("".join(f"{i}\n" for i in indices))
    config = json.loads(Path(config_path).read_text())
    env = dict(
        os.environ,
        HIPBLASLT_SOLUTION_INDEX_LIST=str(out / "indices.txt"),
        HIPBLASLT_TENSILE_LIBPATH=args.library,
        OMP_NUM_THREADS="1",
    )
    if args.race:
        # The cases read the race plugin's log to check races and dispatches.
        config["plugins"] = {"race": {}}
        config["sinks"] = {"types": ["file"], "dir": str(out)}
        env["RJ_SINK_DIR"] = str(out)
    (out / "config.json").write_text(json.dumps(config))
    command = [
        args.rocjitsu,
        "--config",
        str(out / "config.json"),
        "--",
        args.client,
        "--host_side_fill_kernel",
        "--gtest_filter=Library/SolutionIndexTest.*",
        f"--gtest_output=json:{out / 'results.json'}",
    ]
    with open(out / "stdout.log", "w") as log:
        try:
            code = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout * len(indices),
            ).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    text = (out / "stdout.log").read_text(errors="replace")
    report = out / "results.json"
    if report.exists():
        outcomes = {}
        for suite in json.loads(report.read_text())["testsuites"]:
            for test in suite["testsuite"]:
                index = int(re.search(r"idx(\d+)", test["name"])[1])
                if test["result"] == "SKIPPED":
                    outcomes[index] = {"outcome": "SKIPPED", "detail": test.get("value_param", "")}
                elif test.get("failures"):
                    detail = "\n".join(f["failure"] for f in test["failures"])
                    outcomes[index] = {"outcome": "FAILED", "detail": detail}
                else:
                    outcomes[index] = {"outcome": "PASSED", "detail": ""}
        return outcomes, None
    # No report: keep the verdicts gtest printed and blame the case left running.
    outcomes = {
        int(index): {"outcome": {"OK": "PASSED"}.get(status, status), "detail": "from log"}
        for status, index in END.findall(text)
    }
    started = [int(i) for i in RUN.findall(text)]
    running = next((i for i in reversed(started) if i not in outcomes), None)
    if running is not None:
        kind = "TIMEOUT" if code == "timeout" else "CRASHED"
        outcomes[running] = {"outcome": kind, "detail": f"process ended ({code}) in this case"}
    return outcomes, code


def run_device(args, config_path, out, indices):
    """Run indices on one emulated device, resuming after crashes and hangs."""
    remaining, results, attempt = list(indices), {}, 0
    while remaining:
        outcomes, failure = run_once(args, config_path, out / f"attempt-{attempt}", remaining)
        results.update(outcomes)
        before = len(remaining)
        remaining = [i for i in remaining if i not in results]
        # A process that ends before starting any case cannot make progress.
        if failure is not None and len(remaining) == before:
            for index in remaining:
                results[index] = {"outcome": "NOT_RUN", "detail": f"process ended ({failure})"}
            break
        if failure is None:
            # Indices absent from the library generate no case.
            break
        attempt += 1
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rocjitsu", "client", "library", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="rocjitsu config of one emulated device; repeat, most general first",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range", help="Half-open BEGIN:END")
    group.add_argument("--indices", help="File with one index per line")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds allowed per index")
    parser.add_argument(
        "--no-race",
        dest="race",
        action="store_false",
        help="Run without the race plugin, for targets it does not support",
    )
    args = parser.parse_args()

    results, pending = {}, indices_from(args)
    for number, config_path in enumerate(args.config):
        if not pending:
            break
        outcomes = run_device(args, config_path, Path(args.out) / f"device-{number}", pending)
        for index, result in outcomes.items():
            result["device"] = number
        rejected = {
            i
            for i, r in outcomes.items()
            if r["outcome"] == "SKIPPED" and "hardware predicate" in r["detail"]
        }
        results.update(outcomes)
        pending = [i for i in pending if i in rejected]
    Path(args.out, "results.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    counts = {}
    for result in results.values():
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
    print(f"{len(results)} cases: {counts}")
    return 0 if all(r["outcome"] in ("PASSED", "SKIPPED") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
