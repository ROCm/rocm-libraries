# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paired, bounded measurements of the YAML-only change in PR #14597.

The native workflow measures separate run steps through GitHub's timestamps;
this script does not publish hardware timings. It also acts as a pytest plugin
to select copied, byte-identical fixtures without changing the test wrappers.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "projects/hipblaslt/tensilelite"
COMMON = PACKAGE / "tensilelite/Tests/common"
PREFIX = COMMON.relative_to(REPO).as_posix() + "/"
BASE = "4a113302d3e673472871ad7b1be75e2a350f3bd2"
SPLIT = "7bdf24c878e065200086133ebb9ab1fed2bde949"
COHORTS = {
    "native": [
        "gemm/gfx12/bf16_gfx1250-strict.yaml",
        "gemm/gfx12/cgemm_gfx1250-strict.yaml",
        "gemm/gfx12/f64_gfx1250-strict.yaml",
        "gemm/gfx12/i8ii_gfx1250-strict.yaml",
    ],
    "emulator": ["gemm/gfx12/f64_gfx1250.yaml"],
}
_reports = []


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(root, cohort):
    sys.path.insert(0, str(PACKAGE / "scripts"))
    import split_benchmark_groups as splitter

    if (root / "inputs").exists():
        raise ValueError("use a fresh benchmark root; inputs already exist")
    plan = json.loads(
        git(
            "show",
            f"{SPLIT}:projects/hipblaslt/tensilelite/scripts/benchmark_group_splits.json",
        )
    )
    records = [
        r for r in plan["splits"] if r["path"] in {PREFIX + p for p in COHORTS[cohort]}
    ]
    if len(records) != len(COHORTS[cohort]):
        raise ValueError("every cohort source must have a reviewed split")
    outputs = splitter.planned_outputs(REPO, records, plan["source_ref"])
    manifest = {
        "base": BASE,
        "split": SPLIT,
        "cohort": cohort,
        "files": {"base": [], "split": []},
    }
    for record in records:
        original = git("show", f"{BASE}:{record['path']}")
        parts = [(p, git("show", f"{SPLIT}:{p}")) for p in outputs[record["path"]]]
        doc = splitter.load_config(original)
        expected = Counter(splitter.canonical(g) for g in splitter.groups(doc))
        actual = Counter(
            splitter.canonical(g)
            for _, data in parts
            for g in splitter.groups(splitter.load_config(data))
        )
        if expected != actual or any(
            splitter.settings_digest(splitter.load_config(data))
            != splitter.settings_digest(doc)
            for _, data in parts
        ):
            raise ValueError("group or settings mismatch")
        for strategy, files in (
            ("base", [(record["path"], original)]),
            ("split", parts),
        ):
            for path, data in files:
                relative = "common/" + path.removeprefix(PREFIX)
                target = root / "inputs" / strategy / "Tests" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                manifest["files"][strategy].append(
                    {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}
                )
    write_json(root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "cohort": cohort,
                "files": {k: len(v) for k, v in manifest["files"].items()},
            }
        )
    )


def pytest_configure(config):
    if os.environ.get("SPLIT_BENCH_INPUTS"):
        import config_helpers

        config_helpers._TESTS_ROOT_DIR = os.environ["SPLIT_BENCH_INPUTS"]


def pytest_collection_modifyitems(session, config, items):
    expected = int(os.environ["SPLIT_BENCH_EXPECTED"])
    if len(items) != expected:
        import pytest

        raise pytest.UsageError(f"expected {expected} configs, collected {len(items)}")


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed or report.skipped:
        record = {
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
        }
        if report.when == "call" and report.outcome == "passed":
            record["numerical"] = numerical_verdicts(report.capstdout)
        _reports.append(record)


def numerical_verdicts(output):
    """Keep case identities and verdicts, never hardware metrics or timings."""
    fields = (
        "operation",
        "problem-sizes",
        "bias-type",
        "factor-dim",
        "activation-type",
        "solution",
        "validation",
    )
    header = None
    cases = Counter()
    outcomes = Counter()
    for line in output.splitlines():
        if line.startswith("run,problem-progress,"):
            header = next(csv.reader([line]))
        elif header and line[:1].isdigit():
            row = next(csv.reader([line]))
            if len(row) != len(header) or not row[0].isdigit():
                continue
            values = dict(zip(header, row))
            identity = json.dumps([values[key] for key in fields])
            cases[hashlib.sha256(identity.encode()).hexdigest()] += 1
            outcomes[values["validation"]] += 1
    return {"cases": dict(cases), "outcomes": dict(outcomes)}


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return
    expected = int(os.environ["SPLIT_BENCH_EXPECTED"])
    passed = [r for r in _reports if r["phase"] == "call" and r["outcome"] == "passed"]
    complete = (
        len(passed) == expected
        and len({r["nodeid"] for r in passed}) == expected
        and len(_reports) == expected
        and exitstatus == 0
        and all(
            r.get("numerical", {}).get("outcomes", {}).get("PASSED", 0) > 0
            for r in passed
        )
    )
    write_json(
        Path(os.environ["SPLIT_BENCH_VERDICTS"]),
        {"complete": complete, "tests": _reports},
    )
    if not complete:
        session.exitstatus = 1


def run(args):
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    inputs = root / "inputs" / args.strategy / "Tests"
    for record in manifest["files"][args.strategy]:
        if (
            hashlib.sha256((inputs / record["path"]).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise ValueError("benchmark input changed after preparation")
    trial = root / args.trial
    trial.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(
        SPLIT_BENCH_INPUTS=str(inputs),
        SPLIT_BENCH_EXPECTED=str(len(manifest["files"][args.strategy])),
        SPLIT_BENCH_VERDICTS=str(trial / "verdicts.json"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OMP_THREAD_LIMIT="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent),
            str(COMMON),
            str(PACKAGE),
            env.get("PYTHONPATH", ""),
        ]
    )
    options = (
        f"--cxx-compiler,{args.compiler},--global-parameters,CpuThreads={args.cpu_threads},"
        "DataInitSeed=14597,CheckASMCodeSize=True,PrintWinnersOnly=False"
    )
    if args.msgpack:
        options += ",LibraryFormat='msgpack'"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(COMMON / "test_config.py"),
        "-p",
        "benchmark_yaml_split",
        "-p",
        "no:cacheprovider",
        "-v",
        "--capture=fd",
        "-n",
        str(args.workers),
        "--basetemp",
        str(trial / "pytest"),
        "--timeout",
        str(args.timeout),
        "--gpu-targets",
        args.target,
        "--prebuilt-client",
        str(args.client),
        f"--tensile-options={options}",
    ]
    if manifest["cohort"] == "emulator":
        command += ["--client-lock-scope", "worker"]
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, cwd=PACKAGE, start_new_session=True)
    try:
        code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    result = {
        "strategy": args.strategy,
        "workers": args.workers,
        "exit_code": code,
        "trial": args.trial,
    }
    if args.record_timings:
        result["seconds"] = time.monotonic() - started
    write_json(trial / "result.json", result)
    print(json.dumps(result), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cohort", choices=COHORTS, required=True)
    trial = commands.add_parser("run")
    trial.add_argument("--strategy", choices=("base", "split"), required=True)
    trial.add_argument("--trial", required=True)
    trial.add_argument("--workers", type=int, choices=(1, 4), required=True)
    trial.add_argument("--cpu-threads", type=int, default=4)
    trial.add_argument("--target", choices=("gfx1250", "gfx1250-strict"), required=True)
    trial.add_argument("--client", type=Path, required=True)
    trial.add_argument("--compiler", type=Path, required=True)
    trial.add_argument("--timeout", type=int, default=600)
    trial.add_argument("--msgpack", action="store_true")
    trial.add_argument("--record-timings", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root.resolve(), args.cohort)
        return 0
    if Path(args.trial).name != args.trial or args.trial in (".", ".."):
        parser.error("trial must be a directory basename")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
