#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Stand-in for ``hipdnn_integration_tests`` used by the gauntlet tests. No GPU needed.

It honors the command line the fuzz driver uses:

    stub_binary.py --golden-data-dir DIR --outcome-report FILE
                   [--gtest_filter=PATTERNS] [--gtest_list_tests]
                   [--test-article X] [--test-engine Y] [--test-config Z]

It discovers template-sweep cases under ``--golden-data-dir`` the way
``BundleDiscovery.hpp`` does (suite = relative sweep directory sanitized and joined by
``_``, test = sanitized case id), runs the ones the filter selects, and appends the
``started`` / ``finished`` JSON Lines of RFC 0023 section 9.3 to the outcome report,
flushing per record.

What each case does comes from ``$HIPDNN_STUB_CONTROL``, a JSON file::

    {
      "default": "pass",
      "cases": {"s123": "crash"},                  # by case id
      "seed_mod": {"m": 5, "map": {"0": "declined"}},  # by shape seed modulo m
      "engine": "STUB_ENGINE", "arch": "gfx950",
      "hang_seconds": 3600, "delay": 0.0,          # seconds each case takes
      "probe_path": "some/file", "log": "invocations.jsonl"
    }

Behaviours: ``pass``, ``unclaimed_pass``, ``declined``, ``wrong``, ``unexercised``,
``crash_engine`` (a ``finished`` CRASH record, as a build error gives), ``crash``
(``os._exit`` after ``started``), ``hang`` (sleep after ``started``), ``probe_file`` (a
``PASS`` whose reason says whether ``probe_path`` exists when the case runs),
``load_skip`` (the case is not registered, so it is absent from ``--gtest_list_tests``),
``reg_crash`` (the process dies at registration, before any case starts) and
``run_start_crash`` (listing works; a run that selects the case dies before any case starts).

Top-level control keys, not per-case behaviours: ``startup_fail`` (every invocation exits 1 at
startup, as a bad ``--test-engine`` does) and ``no_builtin`` (the binary has no compiled-in test,
so a listing with no filter selects nothing and prints "Error: zero tests ran."), and
``quiet_load`` (a case skipped at load logs nothing, as the real binary without HIPDNN_LOG_LEVEL).
"""

import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path

BUILTIN_TEST = "StubBuiltin.Smoke"

BEHAVIOURS = (
    "pass", "unclaimed_pass", "declined", "wrong", "unexercised",
    "crash_engine", "crash", "hang", "load_skip", "reg_crash", "probe_file", "run_start_crash",
)  # fmt: skip


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def parse_args(argv):
    opts = {
        "filter": "*",
        "list": False,
        "report": None,
        "golden": None,
        "passthrough": {},
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--gtest_filter="):
            opts["filter"] = a.split("=", 1)[1]
        elif a == "--gtest_list_tests":
            opts["list"] = True
        elif a in ("--golden-data-dir", "--outcome-report"):
            i += 1
            opts["golden" if a == "--golden-data-dir" else "report"] = argv[i]
        elif a in ("--test-article", "--test-engine", "--test-config"):
            i += 1
            opts["passthrough"][a] = argv[i]
        i += 1
    return opts


def gtest_matches(name, pattern_string):
    positive, _, negative = pattern_string.partition("-")
    pos = [p for p in positive.split(":") if p] or ["*"]
    neg = [p for p in negative.split(":") if p]
    return any(fnmatch.fnmatchcase(name, p) for p in pos) and not any(
        fnmatch.fnmatchcase(name, p) for p in neg
    )


def load_control():
    path = os.environ.get("HIPDNN_STUB_CONTROL")
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def behaviour_for(control, case_id):
    cases = control.get("cases", {})
    if case_id in cases:
        return cases[case_id]
    seed_mod = control.get("seed_mod")
    match = re.fullmatch(r"s(\d+)", case_id)
    if seed_mod and match:
        mapped = seed_mod.get("map", {}).get(str(int(match.group(1)) % seed_mod["m"]))
        if mapped:
            return mapped
    return control.get("default", "pass")


def discover(golden):
    """``[(suite, test, case_id, sweep_path, purpose)]`` for every sweep case."""
    root = Path(golden)
    found = []
    for sweep in sorted(root.rglob("sweep.json")):
        if not (sweep.parent / "graph.template.json").exists():
            continue
        suite = "_".join(sanitize(p) for p in sweep.parent.relative_to(root).parts)
        with open(sweep, encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        for case in cases:
            purpose = case.get("metadata", {}).get("purpose")
            found.append((suite, sanitize(case["id"]), case["id"], sweep, purpose))
    return found


class Report:
    def __init__(self, path):
        self._file = open(path, "a", encoding="utf-8", newline="\n") if path else None

    def write(self, record):
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


FINISHED = {
    "pass": ("PASS", "compare", "full", "none", "gpu-reference", ""),
    "unclaimed_pass": ("UNCLAIMED_PASS", "compare", "full", "none", "gpu-reference", ""),
    "declined": (
        "DECLINED", "query", "not-reached", "none", "none",
        "Engine {engine} does not support this graph (1 output tensor(s), 0 ranked engine(s))",
    ),
    "wrong": (
        "WRONG", "compare", "full", "COMPARISON", "gpu-reference",
        "output 0 mismatch: max abs diff 0.5 > tolerance 0.001",
    ),
    "unexercised": ("UNEXERCISED", "setup", "not-reached", "HARNESS", "none", "no oracle available"),
    "crash_engine": (
        "CRASH", "build", "not-reached", "ENGINE", "none", "engine plan build failed: stub error",
    ),
}  # fmt: skip


def main(argv):
    opts = parse_args(argv)
    control = load_control()
    if control.get("startup_fail"):
        sys.stderr.write(
            "Error: Engine 'STUB_ENGINE' is not loaded. Check the plugin path.\n"
        )
        return 1
    engine = opts["passthrough"].get("--test-engine") or control.get(
        "engine", "STUB_ENGINE"
    )
    arch = control.get("arch", "gfx950")
    log = control.get("log")
    cases = discover(opts["golden"]) if opts["golden"] else []

    registered = []
    for suite, test, case_id, sweep, purpose in cases:
        kind = behaviour_for(control, case_id)
        if kind not in BEHAVIOURS:
            raise SystemExit(f"stub: unknown behaviour {kind!r} for case {case_id}")
        if kind == "reg_crash":
            if log:
                _log(log, argv, "registration-crash", [])
            sys.stderr.write(f"stub: fatal error while registering {suite}.{test}\n")
            sys.stderr.flush()
            os._exit(134)
        if kind == "load_skip":
            if not control.get(
                "quiet_load"
            ):  # the real one logs only with HIPDNN_LOG_LEVEL
                sys.stderr.write(
                    f"WARNING: skipping bundle {sweep}#{case_id}: stub load failure"
                    f" for case {case_id}\n"
                )
            continue
        registered.append((suite, test, case_id, sweep, purpose, kind))

    selected = [
        r for r in registered if gtest_matches(f"{r[0]}.{r[1]}", opts["filter"])
    ]

    # Like the real binary, which has C++ tests compiled in: with a filter that selects
    # nothing at all it prints "Error: zero tests ran." and exits 1, listing or running.
    builtin = not control.get("no_builtin") and gtest_matches(
        BUILTIN_TEST, opts["filter"]
    )
    if not selected and not builtin:
        if log:
            _log(log, argv, "list" if opts["list"] else "run", [])
        sys.stderr.write(
            "Error: zero tests ran.\n" f"  gtest_filter:    {opts['filter']}\n"
        )
        return 1

    if opts["list"]:
        if log:
            _log(log, argv, "list", [f"{r[0]}.{r[1]}" for r in selected])
        if builtin:
            print("StubBuiltin.\n  Smoke")
        last = None
        for suite, test, *_ in selected:
            if suite != last:
                print(f"{suite}.")
                last = suite
            print(f"  {test}")
        return 0

    if log:
        _log(log, argv, "run", [f"{r[0]}.{r[1]}" for r in selected])
    if any(r[5] == "run_start_crash" for r in selected):
        sys.stderr.write("stub: fatal error while starting the run\n")
        sys.stderr.flush()
        os._exit(7)
    report = Report(opts["report"])
    failed = False
    for suite, test, case_id, sweep, purpose, kind in selected:
        name = f"{suite}.{test}"
        bundle = f"{sweep}#{case_id}"
        print(f"[ RUN      ] {name}", flush=True)
        report.write({"event": "started", "test": name, "bundle": bundle})
        if kind == "crash":
            sys.stderr.write(f"stub: segmentation fault in {name}\n")
            sys.stderr.flush()
            os._exit(139)
        if kind == "hang":
            time.sleep(float(control.get("hang_seconds", 3600)))
        if control.get("delay"):
            time.sleep(float(control["delay"]))
        outcome, stage, depth, origin, verifier, reason = FINISHED.get(
            kind, FINISHED["pass"]
        )
        if kind == "probe_file":
            reason = f"probe:{os.path.exists(control['probe_path'])}"
        record = {
            "event": "finished",
            "test": name,
            "bundle": bundle,
            "engine": engine,
            "arch": arch,
            "platform": "linux",
            "outcome": outcome,
            "stage": stage,
            "depth": depth,
            "origin": origin,
            "verifier": verifier,
            "reason": reason.format(engine=engine),
        }
        if purpose is not None:
            record["purpose"] = purpose
        report.write(record)
        failed = failed or outcome in ("WRONG", "CRASH")
        print(f"[       {outcome} ] {name}", flush=True)
    return 1 if failed else 0


def _log(path, argv, mode, names):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"mode": mode, "argv": argv, "tests": names}) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
