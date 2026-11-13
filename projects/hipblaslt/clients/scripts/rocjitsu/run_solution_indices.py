#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run hipblaslt-test's solution-index cases under rocjitsu, surviving crashes.

hipblaslt-test builds, filters and checks each case itself; gtest reports
the outcomes. When a case crashes or hangs, gtest writes no report, so this
records the cases that finished, marks the running one, and resumes with
the indices that never started.

Some solutions run only on a particular CU count or PCI chip ID, so a run
uses several emulated devices, most general first: each later device runs
only the indices that every earlier one skipped because the solution's
hardware predicate rejected it. --target picks the devices hipBLASLt's
libraries need from rocjitsu's installed configs; --config names config
files instead.

Example, the third of eight shards of every gfx950 index::

    run_solution_indices.py --rocjitsu bin/rocjitsu --client bin/hipblaslt-test \\
        --library lib/hipblaslt/library/gfx950 --target gfx950 \\
        --shard 2/8 --workers 8 --out results/gfx950-2
"""

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The emulated devices each target's libraries need, from rocjitsu's stock
# config for the target: PCI chip ID and CU count, most general first.
TARGET_DEVICES = {
    "gfx942": (
        "gfx942_cdna3.json",
        [{"cus": cus} for cus in (304, 152, 20, 228, 38, 64, 80)],
    ),
    "gfx950": (
        "gfx950_mi350x.json",
        [{"pci": 0x75A3}, {"pci": 0x75A8}, {"pci": 0x75A3, "cus": 128}],
    ),
}

TEST = r"Library/SolutionIndexTest\.matmul/idx(\d+)"
RUN = re.compile(rf"^\[ RUN      \] {TEST}$", re.M)
END = re.compile(rf"^\[ +(OK|FAILED|SKIPPED) +\] {TEST}.*\(\d+ ms\)$", re.M)


def device_configs(args):
    """Write one rocjitsu config per emulated device; return their paths."""
    if args.config:
        return args.config
    name, devices = TARGET_DEVICES[args.target]
    configs = (
        Path(args.configs)
        if args.configs
        else Path(args.rocjitsu).parent.parent / "share/rocjitsu/configs"
    )
    base = json.loads((configs / name).read_text())
    paths = []
    for number, device in enumerate(devices):
        config = json.loads(json.dumps(base))
        # One emulator thread per process; parallelism comes from --workers.
        config.update(num_threads=1, cpu_dispatch_threads=1, async_helper_threads=0)
        config.update(cpu_thread_budget=1, max_ticks=0)
        gpu = config["vm"]["gpu"]["device"]
        if "pci" in device:
            gpu["device_id"] = device["pci"]
        if "cus" in device:
            gpu["simd_count"] = device["cus"] * gpu["simd_per_cu"]
        cus = gpu["simd_count"] // gpu["simd_per_cu"]
        gpu["marketing_name"] = f"{args.target} emulated PCI {gpu['device_id']:#x}, {cus} CUs"
        path = Path(args.out) / f"device-{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config))
        paths.append(str(path))
    return paths


def library_indices(args, config_path):
    """Every solution index of the library, from the test's case list."""
    env = dict(
        os.environ,
        HIPBLASLT_SOLUTION_INDEX_RANGE=f"0:{2**31 - 1}",
        HIPBLASLT_TENSILE_LIBPATH=args.library,
    )
    command = [args.rocjitsu, "--config", config_path, "--", args.client]
    command += ["--gtest_filter=Library/SolutionIndexTest.*", "--gtest_list_tests"]
    listing = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    return sorted({int(i) for i in re.findall(r"idx(\d+)", listing.stdout)})


def indices_from(args, config_path):
    if args.range:
        begin, end = map(int, args.range.split(":"))
        indices = list(range(begin, end))
    elif args.indices:
        indices = [int(line) for line in Path(args.indices).read_text().split()]
    else:
        indices = library_indices(args, config_path)
    if args.shard:
        # Interleaved, so each shard gets a share of every logic file.
        shard, shards = map(int, args.shard.split("/"))
        indices = indices[shard::shards]
    return indices


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
                param = test.get("value_param", "")
                if test["result"] == "SKIPPED":
                    outcomes[index] = {"outcome": "SKIPPED", "detail": param}
                elif test.get("failures"):
                    detail = "\n".join(f["failure"] for f in test["failures"])
                    outcomes[index] = {"outcome": "FAILED", "detail": detail}
                else:
                    outcomes[index] = {"outcome": "PASSED", "detail": ""}
                kernel = re.search(r"kernel (\S+?)(?: \(|$)", param)
                if kernel:
                    outcomes[index]["kernel"] = kernel[1]
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


def reason(result):
    """A short label for why a case did not pass, for summaries."""
    detail, outcome = result["detail"], result["outcome"]
    if outcome != "FAILED":
        return outcome.lower()
    for needle, label in (
        ("reported races", "race"),
        ("NO solution found", "no solution"),
        ("must run exactly once", "kernel not dispatched once"),
        ("Special value mismatch", "numerical"),
        ("norm_check", "numerical"),
    ):
        if needle in detail:
            return label
    return "other"


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


def run_workers(args, config_path, out, indices):
    """Split indices across --workers concurrent runs of one device."""
    step = args.workers
    chunks = [indices[w::step] for w in range(step)]
    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) == 1:
        return run_device(args, config_path, out, chunks[0])
    results = {}
    with ThreadPoolExecutor(len(chunks)) as pool:
        runs = [
            pool.submit(run_device, args, config_path, out / f"worker-{w}", chunk)
            for w, chunk in enumerate(chunks)
        ]
        for run in runs:
            results.update(run.result())
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("rocjitsu", "client", "library", "out"):
        parser.add_argument(f"--{name}", required=True)
    devices = parser.add_mutually_exclusive_group(required=True)
    devices.add_argument(
        "--target",
        choices=sorted(TARGET_DEVICES),
        help="Emulate the devices this target's libraries need",
    )
    devices.add_argument(
        "--config",
        action="append",
        help="rocjitsu config of one emulated device; repeat, most general first",
    )
    parser.add_argument(
        "--configs", help="rocjitsu's config directory (default: next to --rocjitsu)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--range", help="Half-open BEGIN:END (default: every index)")
    group.add_argument("--indices", help="File with one index per line")
    parser.add_argument("--shard", help="I/N: run every N-th index starting at the I-th (0-based)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent emulator processes")
    parser.add_argument(
        "--known-failures",
        help="File of kernel names, one per line, whose failures do not fail the run",
    )
    parser.add_argument("--timeout", type=float, default=120, help="Seconds allowed per index")
    parser.add_argument(
        "--no-race",
        dest="race",
        action="store_false",
        help="Run without the race plugin, for targets it does not support",
    )
    args = parser.parse_args()

    configs = device_configs(args)
    results, pending = {}, indices_from(args, configs[0])
    for number, config_path in enumerate(configs):
        if not pending:
            break
        outcomes = run_workers(args, config_path, Path(args.out) / f"device-{number}", pending)
        for index, result in outcomes.items():
            result["device"] = number
        rejected = {
            i
            for i, r in outcomes.items()
            if r["outcome"] == "SKIPPED" and "hardware predicate" in r["detail"]
        }
        results.update(outcomes)
        pending = [i for i in pending if i in rejected]
    known = set()
    if args.known_failures:
        for line in Path(args.known_failures).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                known.add(line)
    counts, reasons, new = {}, {}, 0
    for result in results.values():
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
        if result["outcome"] not in ("PASSED", "SKIPPED"):
            result["reason"] = reason(result)
            reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
            result["known"] = result.get("kernel") in known
            new += not result["known"]
    Path(args.out, "results.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    summary = f"{len(results)} cases: {counts}"
    if reasons:
        summary += f"; not passed: {reasons}; not in the known failures: {new}"
    print(summary)
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
